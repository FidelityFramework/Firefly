/// SeqOp Filter Witness - Seq.filter operation
///
/// PRD-16: Seq.filter creates FilterSeq wrapper with flat closure
///
/// SCOPE: witnessSeqFilter, witnessFilterMoveNext
/// DOES NOT: Map, take, fold, collect (separate witnesses)
module Alex.Witnesses.SeqOp.Filter

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Witnesses.SeqWitness

module SSAAssign = PSGElaboration.SSAAssignment
module CF = Alex.Dialects.CF.Templates

// ═══════════════════════════════════════════════════════════════════════════
// SHARED HELPERS
// ═══════════════════════════════════════════════════════════════════════════

let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "SeqOpWitness: No SSAs for node %A" nodeId

let private closureType = TStruct [TPtr; TPtr]

let private generateMoveNextName (opName: string) (appNodeId: NodeId) : string =
    sprintf "seqop_%s_moveNext_%d" opName (NodeId.value appNodeId)

/// Build FilterSeq struct type
/// Layout: {state: i32, current: A, moveNext_ptr: ptr, inner_seq: InnerSeqType, predicate: Closure}
let private filterSeqStructType (elementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; elementType; TPtr; innerSeqType; closureType]

// ═══════════════════════════════════════════════════════════════════════════
// MOVENEXT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MoveNext function for FilterSeq
/// Algorithm:
///   loop:
///       if inner.MoveNext():
///           if predicate(inner.current):
///               self.current = inner.current
///               return true
///           else:
///               continue loop
///   return false
let private witnessFilterMoveNext
    (moveNextName: string)
    (filterSeqType: MLIRType)
    (elementType: MLIRType)
    (innerSeqType: MLIRType)
    : MLIROp =

    let mutable ssaCounter = 0
    let nextSSA () = let ssa = V ssaCounter in ssaCounter <- ssaCounter + 1; ssa

    let filterSeqPtrSSA = nextSSA ()

    let trueLitSSA = nextSSA ()
    let falseLitSSA = nextSSA ()
    let oneSSA = nextSSA ()
    let neg1SSA = nextSSA ()

    let loopBlockRef = BlockRef "loop"
    let checkPredicateBlockRef = BlockRef "check_predicate"
    let yieldBlockRef = BlockRef "yield"
    let doneBlockRef = BlockRef "done"

    let entryOps = [
        MLIROp.ArithOp (ArithOp.ConstI (trueLitSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.ConstI (neg1SSA, -1L, MLIRTypes.i32))
        MLIROp.CFOp (CF.brSimple loopBlockRef)
    ]

    let entryBlock: Block = { Label = BlockRef "entry"; Args = []; Ops = entryOps }

    // Loop block
    let innerSeqPtrSSA = nextSSA ()
    let innerSeqLoadSSA = nextSSA ()
    let innerMoveNextPtrSSA = nextSSA ()
    let innerMoveNextResultSSA = nextSSA ()

    let loopOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (innerSeqPtrSSA, filterSeqPtrSSA, 3, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoadSSA, innerSeqPtrSSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerMoveNextPtrSSA, innerSeqLoadSSA, [2], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some innerMoveNextResultSSA, innerMoveNextPtrSSA,
            [{ SSA = innerSeqPtrSSA; Type = TPtr }], MLIRTypes.i1))
        MLIROp.CFOp (CF.condBrSimple innerMoveNextResultSSA checkPredicateBlockRef doneBlockRef)
    ]

    let loopBlock: Block = { Label = loopBlockRef; Args = []; Ops = loopOps }

    // Check predicate block
    let innerSeqLoad2SSA = nextSSA ()
    let innerCurrentSSA = nextSSA ()
    let filterSeqLoadSSA = nextSSA ()
    let predicateSSA = nextSSA ()
    let predicateCodePtrSSA = nextSSA ()
    let predicateEnvPtrSSA = nextSSA ()
    let predicateResultSSA = nextSSA ()

    let checkPredicateOps = [
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoad2SSA, innerSeqPtrSSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerCurrentSSA, innerSeqLoad2SSA, [1], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (filterSeqLoadSSA, filterSeqPtrSSA, filterSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (predicateSSA, filterSeqLoadSSA, [4], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (predicateCodePtrSSA, predicateSSA, [0], closureType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (predicateEnvPtrSSA, predicateSSA, [1], closureType))
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some predicateResultSSA, predicateCodePtrSSA,
            [{ SSA = predicateEnvPtrSSA; Type = TPtr }; { SSA = innerCurrentSSA; Type = elementType }], MLIRTypes.i1))
        MLIROp.CFOp (CF.condBrSimple predicateResultSSA yieldBlockRef loopBlockRef)
    ]

    let checkPredicateBlock: Block = { Label = checkPredicateBlockRef; Args = []; Ops = checkPredicateOps }

    // Yield block
    let innerSeqPtr2SSA = nextSSA ()
    let innerSeqLoad3SSA = nextSSA ()
    let innerCurrent2SSA = nextSSA ()
    let currentPtrSSA = nextSSA ()
    let statePtrSSA = nextSSA ()

    let yieldOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (innerSeqPtr2SSA, filterSeqPtrSSA, 3, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoad3SSA, innerSeqPtr2SSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerCurrent2SSA, innerSeqLoad3SSA, [1], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.StructGEP (currentPtrSSA, filterSeqPtrSSA, 1, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (innerCurrent2SSA, currentPtrSSA, elementType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA, filterSeqPtrSSA, 0, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (oneSSA, statePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some trueLitSSA, Some MLIRTypes.i1))
    ]

    let yieldBlock: Block = { Label = yieldBlockRef; Args = []; Ops = yieldOps }

    // Done block
    let donePtrSSA = nextSSA ()
    let doneOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (donePtrSSA, filterSeqPtrSSA, 0, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (neg1SSA, donePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))
    ]

    let doneBlock: Block = { Label = doneBlockRef; Args = []; Ops = doneOps }

    let bodyRegion: Region = { Blocks = [entryBlock; loopBlock; checkPredicateBlock; yieldBlock; doneBlock] }

    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (moveNextName, [(filterSeqPtrSSA, TPtr)], MLIRTypes.i1, bodyRegion, LLVMPrivate))

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq.filter - create FilterSeq wrapper
/// Returns: (inlineOps, topLevelOps, result)
let witnessSeqFilter
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (predicate: Val)
    (innerSeq: Val)
    (elementType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let filterSeqType = filterSeqStructType elementType innerSeqType
    let moveNextName = generateMoveNextName "filter" appNodeId
    let ssas = requireSSAs appNodeId ssa

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    let moveNextFuncOp = witnessFilterMoveNext moveNextName filterSeqType elementType innerSeqType

    let inlineOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, predicate.SSA, [4], filterSeqType))
    ]

    (inlineOps, [moveNextFuncOp], TRValue { SSA = resultSSA; Type = filterSeqType })
