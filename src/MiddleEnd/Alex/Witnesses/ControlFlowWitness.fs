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
open Alex.Patterns.ElisionPatterns

// ═══════════════════════════════════════════════════════════════════════════
// SUB-GRAPH COMBINATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Build sub-graph combinator from nanopass list
/// Folds over all witnesses, returning first non-skip result
let private makeSubGraphCombinator (nanopasses: Nanopass list) : (WitnessContext -> SemanticNode -> WitnessOutput) =
    fun ctx node ->
        let rec tryWitnesses witnesses =
            match witnesses with
            | [] -> WitnessOutput.skip
            | nanopass :: rest ->
                match nanopass.Witness ctx node with
                | output when output = WitnessOutput.skip ->
                    tryWitnesses rest
                | output -> output
        tryWitnesses nanopasses

/// Helper: Witness a branch/scope by marking boundaries and extracting operations
/// Witness a branch scope (if-then, if-else, while-cond, while-body, for-body)
/// Uses nested accumulator to collect branch operations without markers
let private witnessBranchScope (rootId: NodeId) (ctx: WitnessContext) combinator : MLIROp list =
    // Create NESTED accumulator for branch operations
    let branchAcc = MLIRAccumulator.empty()

    // Witness branch nodes into nested accumulator
    match SemanticGraph.tryGetNode rootId ctx.Graph with
    | Some branchNode ->
        match focusOn rootId ctx.Zipper with
        | Some branchZipper ->
            let branchCtx = { ctx with Zipper = branchZipper; Accumulator = branchAcc }
            let branchVisited = ref Set.empty  // Fresh visited set for branch traversal
            visitAllNodes combinator branchCtx branchNode branchAcc branchVisited
            
            // Copy bindings from branch accumulator to parent
            branchAcc.NodeAssoc |> Map.iter (fun nodeId binding ->
                ctx.Accumulator.NodeAssoc <- Map.add nodeId binding ctx.Accumulator.NodeAssoc)
        | None -> ()
    | None -> ()

    // Return branch operations directly from nested accumulator
    branchAcc.AllOps

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness control flow operations - category-selective (handles only control flow nodes)
/// Takes nanopass list to build sub-graph combinator for branch witnessing
let private witnessControlFlowWith (nanopasses: Nanopass list) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let subGraphCombinator = makeSubGraphCombinator nanopasses

    match tryMatch pIfThenElse ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((condId, thenId, elseIdOpt), _) ->
        match MLIRAccumulator.recallNode condId ctx.Accumulator with
        | None -> WitnessOutput.error "IfThenElse: Condition not yet witnessed"
        | Some (condSSA, _) ->
            let thenOps = witnessBranchScope thenId ctx subGraphCombinator
            let elseOps = elseIdOpt |> Option.map (fun elseId -> witnessBranchScope elseId ctx subGraphCombinator)

            match tryMatch (pBuildIfThenElse condSSA thenOps elseOps) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
            | None -> WitnessOutput.error "IfThenElse pattern emission failed"

    | None ->
        match tryMatch pWhileLoop ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((condId, bodyId), _) ->
            let condOps = witnessBranchScope condId ctx subGraphCombinator
            let bodyOps = witnessBranchScope bodyId ctx subGraphCombinator

            match tryMatch (pBuildWhileLoop condOps bodyOps) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
            | None -> WitnessOutput.error "WhileLoop pattern emission failed"

        | None ->
            match tryMatch pForLoop ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((_, lowerId, upperId, _, bodyId), _) ->
                match MLIRAccumulator.recallNode lowerId ctx.Accumulator, MLIRAccumulator.recallNode upperId ctx.Accumulator with
                | Some (lowerSSA, _), Some (upperSSA, _) ->
                    let _bodyOps = witnessBranchScope bodyId ctx subGraphCombinator
                    WitnessOutput.error "ForLoop needs step constant - gap in patterns"

                | _ -> WitnessOutput.error "ForLoop: Loop bounds not yet witnessed"

            | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Create control flow nanopass with nanopass list for sub-graph traversal
/// This must be called AFTER all other nanopasses are registered
let createNanopass (nanopasses: Nanopass list) : Nanopass = {
    Name = "ControlFlow"
    Witness = witnessControlFlowWith nanopasses
}

/// Placeholder nanopass export - will be replaced by createNanopass call in registry
let nanopass : Nanopass = {
    Name = "ControlFlow"
    Witness = fun _ _ -> WitnessOutput.error "ControlFlow nanopass not properly initialized - use createNanopass"
}
