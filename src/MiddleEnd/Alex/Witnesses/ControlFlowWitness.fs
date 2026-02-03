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

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ControlFlowPatterns

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
    printfn "[ControlFlowWitness] witnessBranchScope: Starting visitation of branch root node %A" (NodeId.value rootId)

    // Record current operation count to extract only branch operations
    let opsBefore = List.length ctx.Accumulator.AllOps

    // Witness branch nodes using SAME accumulator (errors and bindings are global)
    match SemanticGraph.tryGetNode rootId ctx.Graph with
    | Some branchNode ->
        printfn "[ControlFlowWitness] witnessBranchScope: Found branch node %A (kind: %s)"
            (NodeId.value rootId)
            (branchNode.Kind.ToString().Split('\n').[0])

        match focusOn rootId ctx.Zipper with
        | Some branchZipper ->
            printfn "[ControlFlowWitness] witnessBranchScope: Successfully focused on node %A, calling visitAllNodes" (NodeId.value rootId)
            let branchCtx = { ctx with Zipper = branchZipper }
            // Use GLOBAL visited set to prevent duplicate visitation by top-level traversal
            visitAllNodes combinator branchCtx branchNode ctx.GlobalVisited

            let opsAfter = List.length ctx.Accumulator.AllOps
            printfn "[ControlFlowWitness] witnessBranchScope: Completed visitation of node %A - added %d ops"
                (NodeId.value rootId)
                (opsAfter - opsBefore)
        | None ->
            printfn "[ControlFlowWitness] witnessBranchScope: Failed to focus on node %A" (NodeId.value rootId)
    | None ->
        printfn "[ControlFlowWitness] witnessBranchScope: Node %A not found in graph" (NodeId.value rootId)

    // Extract branch operations (everything added since opsBefore)
    let opsAfter = List.length ctx.Accumulator.AllOps
    let branchOpsReversed = ctx.Accumulator.AllOps |> List.take (opsAfter - opsBefore)

    // Return branch operations in correct SSA order
    // Post-order traversal accumulates in reverse, so reverse to restore program order
    List.rev branchOpsReversed

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
        printfn "[ControlFlowWitness] Handling IfThenElse node %A (cond=%A, then=%A, else=%A)"
            (NodeId.value node.Id)
            (NodeId.value condId)
            (NodeId.value thenId)
            (elseIdOpt |> Option.map NodeId.value)

        // Visit condition as branch scope (like WhileLoop does)
        // Condition may contain complex expressions that need witnessing
        printfn "[ControlFlowWitness] IfThenElse: Visiting condition branch %A" (NodeId.value condId)
        let _condOps = witnessBranchScope condId ctx combinator

        // Now recall the condition result
        match MLIRAccumulator.recallNode condId ctx.Accumulator with
        | None ->
            printfn "[ControlFlowWitness] IfThenElse: ERROR - Condition %A witnessed but no result" (NodeId.value condId)
            WitnessOutput.error "IfThenElse: Condition witnessed but no result"
        | Some (condSSA, _) ->
            printfn "[ControlFlowWitness] IfThenElse: Visiting then branch %A" (NodeId.value thenId)
            let thenOps = witnessBranchScope thenId ctx combinator

            let elseOps =
                match elseIdOpt with
                | Some elseId ->
                    printfn "[ControlFlowWitness] IfThenElse: Visiting else branch %A" (NodeId.value elseId)
                    Some (witnessBranchScope elseId ctx combinator)
                | None ->
                    printfn "[ControlFlowWitness] IfThenElse: No else branch"
                    None

            printfn "[ControlFlowWitness] IfThenElse: Building MLIR (then has %d ops, else has %d ops)"
                (List.length thenOps)
                (elseOps |> Option.map List.length |> Option.defaultValue 0)

            match tryMatch (pBuildIfThenElse condSSA thenOps elseOps) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some (ops, _) ->
                printfn "[ControlFlowWitness] IfThenElse: Successfully built MLIR with %d ops" (List.length ops)
                { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
            | None ->
                printfn "[ControlFlowWitness] IfThenElse: ERROR - Pattern emission failed"
                WitnessOutput.error "IfThenElse pattern emission failed"

    | None ->
        match tryMatch pWhileLoop ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((condId, bodyId), _) ->
            let condOps = witnessBranchScope condId ctx combinator
            let bodyOps = witnessBranchScope bodyId ctx combinator

            match tryMatch (pBuildWhileLoop condOps bodyOps) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
            | None -> WitnessOutput.error "WhileLoop pattern emission failed"

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
