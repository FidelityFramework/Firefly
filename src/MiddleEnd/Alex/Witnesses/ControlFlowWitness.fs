/// ControlFlowWitness - Witness control flow operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY control flow nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// SPECIAL CASE: Control flow needs to witness sub-graphs (then/else/body branches)
/// that can contain ANY category of nodes. Uses subGraphCombinator to fold over
/// all registered witnesses.
module Alex.Witnesses.ControlFlowWitness

/// Set to true for detailed control flow traversal tracing
let mutable private traceEnabled = false
let private trace fmt = Printf.kprintf (fun s -> if traceEnabled then printfn "%s" s) fmt

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ControlFlowPatterns
open Alex.Elements.SCFElements

// ═══════════════════════════════════════════════════════════════════════════
// Y-COMBINATOR PATTERN
// ═══════════════════════════════════════════════════════════════════════════
//
// Scope witnesses need to handle nested scopes (e.g., IfThenElse inside WhileLoop).
// This requires recursive self-reference: the combinator must include itself.
//
// Solution: Y-combinator fixed point via thunk (unit -> Combinator)
// The combinator getter is passed from WitnessRegistry, allowing deferred evaluation
// and creating a proper fixed point where witnesses can recursively invoke themselves.

/// Helper: Witness a branch/scope by marking boundaries and extracting operations
/// Witness a branch scope (if-then, if-else, while-cond, while-body, for-body)
/// Collects operations from the branch while using the SAME accumulator for bindings/errors
/// Combinator passed through from top-level via Y-combinator fixed point
let private witnessBranchScope (rootId: NodeId) (ctx: WitnessContext) (combinator: WitnessContext -> SemanticNode -> WitnessOutput) : MLIROp list =
    trace "[ControlFlowWitness] witnessBranchScope: Starting visitation of branch root node %A" (NodeId.value rootId)

    // SCOPE-AWARE ACCUMULATION: Create child scope for branch operations
    // Operations witness during branch traversal naturally accumulate into this child scope
    // No counting, no subtraction - operations know their scope at creation time
    let branchScope = ref (ScopeContext.createChild !ctx.ScopeContext BlockLevel)

    // Witness branch nodes using child scope
    // Accumulator and GlobalVisited remain shared (errors and bindings are global)
    match SemanticGraph.tryGetNode rootId ctx.Graph with
    | Some branchNode ->
        trace "[ControlFlowWitness] witnessBranchScope: Found branch node %A (kind: %s)"
            (NodeId.value rootId)
            (branchNode.Kind.ToString().Split('\n').[0])

        match focusOn rootId ctx.Zipper with
        | Some branchZipper ->
            trace "[ControlFlowWitness] witnessBranchScope: Successfully focused on node %A, calling visitAllNodes" (NodeId.value rootId)
            // Create context with child scope - operations will accumulate into branchScope
            let branchCtx = { ctx with
                                Zipper = branchZipper
                                ScopeContext = branchScope }
            // Use GLOBAL visited set - nodes visited once, emit into branch scope
            visitAllNodes combinator branchCtx branchNode ctx.GlobalVisited

            let branchOps = ScopeContext.getOps !branchScope
            trace "[ControlFlowWitness] witnessBranchScope: Completed visitation of node %A - extracted %d ops from child scope"
                (NodeId.value rootId)
                (List.length branchOps)
        | None ->
            trace "[ControlFlowWitness] witnessBranchScope: Failed to focus on node %A" (NodeId.value rootId)
    | None ->
        trace "[ControlFlowWitness] witnessBranchScope: Node %A not found in graph" (NodeId.value rootId)

    // Extract operations from child scope (already in correct order)
    ScopeContext.getOps !branchScope

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness control flow operations - category-selective (handles only control flow nodes)
/// Takes combinator getter (Y-combinator thunk) for recursive self-reference
let private witnessControlFlowWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Get the full combinator (including ourselves) via Y-combinator fixed point
    let combinator = getCombinator()

    match tryMatch pIfThenElse ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((condId, thenId, elseIdOpt), _) ->
        trace "[ControlFlowWitness] Handling IfThenElse node %A (cond=%A, then=%A, else=%A)"
            (NodeId.value node.Id)
            (NodeId.value condId)
            (NodeId.value thenId)
            (elseIdOpt |> Option.map NodeId.value)

        // ARCHITECTURAL FIX: Condition must be visited in CURRENT scope, not as child branch
        // IfThenElse condition operations accumulate into parent scope (before scf.if)
        // Only branch BODIES (then/else) are isolated child scopes for scf.if regions.
        // Since IfThenElse is a scope boundary, children aren't auto-visited - we must visit condition explicitly.
        trace "[ControlFlowWitness] IfThenElse: Visiting condition %A in current scope (not branch scope)" (NodeId.value condId)
        match SemanticGraph.tryGetNode condId ctx.Graph with
        | Some condNode ->
            // Visit condition in CURRENT scope - ops accumulate into parent, not isolated child
            visitAllNodes combinator ctx condNode ctx.GlobalVisited
        | None ->
            trace "[ControlFlowWitness] IfThenElse: ERROR - Condition node %A not found" (NodeId.value condId)

        // Recall the condition result (now available from current scope visitation)
        match MLIRAccumulator.recallNode condId ctx.Accumulator with
        | None ->
            trace "[ControlFlowWitness] IfThenElse: ERROR - Condition %A witnessed but no result" (NodeId.value condId)
            WitnessOutput.error "IfThenElse: Condition witnessed but no result"
        | Some (condSSA, _) ->
            trace "[ControlFlowWitness] IfThenElse: Visiting then branch %A" (NodeId.value thenId)
            let thenOps = witnessBranchScope thenId ctx combinator

            let elseOps =
                match elseIdOpt with
                | Some elseId ->
                    trace "[ControlFlowWitness] IfThenElse: Visiting else branch %A" (NodeId.value elseId)
                    Some (witnessBranchScope elseId ctx combinator)
                | None ->
                    trace "[ControlFlowWitness] IfThenElse: No else branch"
                    None

            trace "[ControlFlowWitness] IfThenElse: Building MLIR (then has %d ops, else has %d ops)"
                (List.length thenOps)
                (elseOps |> Option.map List.length |> Option.defaultValue 0)

            // Add scf.yield terminators to branches (like WhileLoop does)
            match tryMatch (pSCFYield []) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some (yieldTerminator, _) ->
                let thenOpsWithYield = thenOps @ [yieldTerminator]
                let elseOpsWithYield = elseOps |> Option.map (fun ops -> ops @ [yieldTerminator])

                match tryMatch (pBuildIfThenElse condSSA thenOpsWithYield elseOpsWithYield) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some (ops, _) ->
                    trace "[ControlFlowWitness] IfThenElse: Successfully built MLIR with %d ops" (List.length ops)
                    { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
                | None ->
                    trace "[ControlFlowWitness] IfThenElse: ERROR - Pattern emission failed"
                    WitnessOutput.error "IfThenElse pattern emission failed"
            | None ->
                WitnessOutput.error "IfThenElse: pSCFYield failed"

    | None ->
        match tryMatch pWhileLoop ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((condId, bodyId), _) ->
            trace "[ControlFlowWitness] WhileLoop: Visiting condition branch %A" (NodeId.value condId)
            let condOps = witnessBranchScope condId ctx combinator
            trace "[ControlFlowWitness] WhileLoop: Visiting body branch %A" (NodeId.value bodyId)
            let bodyOps = witnessBranchScope bodyId ctx combinator

            // Recall condition result to build scf.condition terminator
            match MLIRAccumulator.recallNode condId ctx.Accumulator with
            | None ->
                trace "[ControlFlowWitness] WhileLoop: ERROR - Condition %A witnessed but no result" (NodeId.value condId)
                WitnessOutput.error "WhileLoop: Condition witnessed but no result"
            | Some (condSSA, _) ->
                trace "[ControlFlowWitness] WhileLoop: Building scf.while with condition SSA %A" condSSA
                // Build scf.condition and scf.yield terminators
                match tryMatch (pSCFCondition condSSA []) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some (condTerminator, _) ->
                    let condOpsWithTerminator = condOps @ [condTerminator]

                    match tryMatch (pSCFYield []) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (yieldTerminator, _) ->
                        let bodyOpsWithTerminator = bodyOps @ [yieldTerminator]

                        match tryMatch (pBuildWhileLoop condOpsWithTerminator bodyOpsWithTerminator) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some (ops, _) ->
                            trace "[ControlFlowWitness] WhileLoop: Successfully built scf.while"
                            { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
                        | None -> WitnessOutput.error "WhileLoop pattern emission failed"
                    | None -> WitnessOutput.error "WhileLoop: pSCFYield failed"
                | None -> WitnessOutput.error "WhileLoop: pSCFCondition failed"

        | None ->
            match tryMatch pForLoop ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((_, lowerId, upperId, _, bodyId), _) ->
                match MLIRAccumulator.recallNode lowerId ctx.Accumulator, MLIRAccumulator.recallNode upperId ctx.Accumulator with
                | Some (lowerSSA, _), Some (upperSSA, _) ->
                    let _bodyOps = witnessBranchScope bodyId ctx combinator
                    WitnessOutput.error "ForLoop needs step constant - gap in patterns"

                | _ -> WitnessOutput.error "ForLoop: Loop bounds not yet witnessed"

            | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Create control flow nanopass with Y-combinator thunk for recursive self-reference
/// The combinator getter allows deferred evaluation, creating a fixed point where
/// this witness can handle nested control flow (e.g., IfThenElse inside WhileLoop)
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass = {
    Name = "ControlFlow"
    Witness = witnessControlFlowWith getCombinator
}

/// Placeholder nanopass export - will be replaced by createNanopass call in registry
let nanopass : Nanopass = {
    Name = "ControlFlow"
    Witness = fun _ _ -> WitnessOutput.error "ControlFlow nanopass not properly initialized - use createNanopass with Y-combinator"
}
