/// Conditional Witness - Witness conditional control flow to MLIR (SCF dialect)
///
/// SCOPE: Handle if-then-else and sequential expressions.
/// DOES NOT: Implement loops, pattern matching (separate witnesses).
///
/// Uses SCF dialect for structured control flow.
module Alex.Witnesses.ControlFlow.ConditionalWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping

module SCF = Alex.Dialects.SCF.Templates
module SSAAssignment = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Get all pre-assigned SSAs for a node
let private requireNodeSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssignment.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Recall the result (SSA, type) for a previously-processed node
let private recallNodeResult (nodeId: NodeId) (ctx: WitnessContext) : (SSA * MLIRType) option =
    MLIRAccumulator.recallNode (NodeId.value nodeId) ctx.Accumulator

/// Map NativeType to MLIRType
let private mapType = mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENTIAL EXPRESSION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a sequential expression (children already processed in post-order)
/// Returns the result of the last expression.
let witnessSequential (ctx: WitnessContext) (nodeIds: NodeId list) : MLIROp list * TransferResult =
    match List.tryLast nodeIds with
    | Some lastId ->
        // Get the actual emitted SSA from NodeBindings
        match recallNodeResult lastId ctx with
        | Some (ssa, ty) ->
            // Pass through the last child's result
            [], TRValue { SSA = ssa; Type = ty }
        | None ->
            // No result bound - might be void or not yet processed
            [], TRVoid
    | None ->
        [], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// IF-THEN-ELSE
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an if-then-else expression using SCF dialect
/// thenOps and elseOps are the pre-witnessed operations from the regions
let witnessIfThenElse
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (condSSA: SSA)
    (thenOps: MLIROp list)
    (thenResultSSA: SSA option)
    (elseOps: MLIROp list option)
    (elseResultSSA: SSA option)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs: result[0], thenZero[1], elseZero[2]
    let ssas = requireNodeSSAs nodeId ctx

    // Build then region with yield
    let thenYieldOps =
        match thenResultSSA, resultType with
        | Some ssa, Some ty ->
            thenOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
        | None, Some ty ->
            // Need to yield zero/default value
            let zeroSSA = ssas.[1]
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
            thenOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
        | _ ->
            thenOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
    let thenRegion = SCF.singleBlockRegion "then" [] thenYieldOps

    // Build else region with yield
    let elseRegion =
        match elseOps, elseResultSSA, resultType with
        | Some ops, Some ssa, Some ty ->
            let elseYieldOps = ops @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | Some ops, None, Some ty ->
            // Need to yield zero/default value
            let zeroSSA = ssas.[2]
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
            let elseYieldOps = ops @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | Some ops, _, _ ->
            let elseYieldOps = ops @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | None, _, Some ty ->
            // No else clause provided - implicit else with zero/default
            let zeroSSA = ssas.[2]
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
            let elseYieldOps = [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | None, _, None ->
            None  // No else clause, no result type - void if

    // Build the if operation
    let resultSSAs = resultType |> Option.map (fun _ -> ssas.[0]) |> Option.toList
    let resultTypes = resultType |> Option.toList

    let ifOp = SCFOp.If (resultSSAs, condSSA, thenRegion, elseRegion, resultTypes)

    match resultType with
    | Some ty ->
        [MLIROp.SCFOp ifOp], TRValue { SSA = ssas.[0]; Type = ty }
    | None ->
        [MLIROp.SCFOp ifOp], TRVoid
