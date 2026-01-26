/// SeqOp Aggregations Witness - Seq.fold, Seq.collect
///
/// PRD-16: Fold and collect operations
///
/// SCOPE: witnessSeqFold, witnessSeqCollect
/// DOES NOT: Map, filter, take (separate witness)
module Alex.Witnesses.SeqOp.Aggregations

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Witnesses.SeqWitness

module SSAAssign = PSGElaboration.SSAAssignment

let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "SeqOpWitness: No SSAs for node %A" nodeId

/// Standard uniform closure type: {code_ptr, env_ptr}
let witnessSeqFold
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (folder: Val)
    (initial: Val)
    (seq: Val)
    (accType: MLIRType)
    (elementType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =
    [], [], TRError (sprintf "Seq.fold requires special handling - node %d" (NodeId.value appNodeId))

/// Seq.collect (flatMap) requires complex nested iteration state
/// Returns error for special handling
/// Returns: (inlineOps, topLevelOps, result)
let witnessSeqCollect
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapper: Val)
    (outerSeq: Val)
    (outputElementType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =
    [], [], TRError (sprintf "Seq.collect requires complex nested iteration - node %d" (NodeId.value appNodeId))
