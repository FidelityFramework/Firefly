/// SeqOp Take Witness - Seq.take operation
///
/// PRD-16: Seq.take creates TakeSeq wrapper with counter
///
/// SCOPE: witnessSeqTake, witnessTakeMoveNext
/// DOES NOT: Map, filter, fold, collect (separate witnesses)
module Alex.Witnesses.SeqOp.Take

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

let private generateMoveNextName (opName: string) (appNodeId: NodeId) : string =
    sprintf "seqop_%s_moveNext_%d" opName (NodeId.value appNodeId)

/// Build TakeSeq struct type
/// Layout: {state: i32, current: A, moveNext_ptr: ptr, inner_seq: InnerSeqType, remaining: i32}
let private takeSeqStructType (elementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; elementType; TPtr; innerSeqType; TInt I32]

// ═══════════════════════════════════════════════════════════════════════════
// MOVENEXT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MoveNext function for TakeSeq
/// Algorithm:
///   if remaining > 0:
///       if inner.MoveNext():
///           self.current = inner.current
///           remaining--
///           return true
///   return false
let private witnessTakeMoveNext
    (moveNextName: string)
    (takeSeqType: MLIRType)
    (elementType: MLIRType)
    (innerSeqType: MLIRType)
    : MLIROp =

    let mutable ssaCounter = 0
    let nextSSA () = let ssa = V ssaCounter in ssaCounter <- ssaCounter + 1; ssa

    let takeSeqPtrSSA = nextSSA ()

    let trueLitSSA = nextSSA ()
    let falseLitSSA = nextSSA ()
    let zeroI32SSA = nextSSA ()
    let oneI32SSA = nextSSA ()
    let neg1SSA = nextSSA ()

    let checkInnerBlockRef = BlockRef "check_inner"
    let yieldBlockRef = BlockRef "yield"
    let doneBlockRef = BlockRef "done"

    let remainingPtrSSA = nextSSA ()
    let remainingValSSA = nextSSA ()
    let remainingGtZeroSSA = nextSSA ()

    let entryOps = [
        MLIROp.ArithOp (ArithOp.ConstI (trueLitSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (zeroI32SSA, 0L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.ConstI (oneI32SSA, 1L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.ConstI (neg1SSA, -1L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.StructGEP (remainingPtrSSA, takeSeqPtrSSA, 4, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (remainingValSSA, remainingPtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (remainingGtZeroSSA, ICmpPred.Sgt, remainingValSSA, zeroI32SSA, TInt I32))
        MLIROp.CFOp (CF.condBrSimple remainingGtZeroSSA checkInnerBlockRef doneBlockRef)
    ]

    let entryBlock: Block = { Label = BlockRef "entry"; Args = []; Ops = entryOps }

    // Check inner block
    let innerSeqPtrSSA = nextSSA ()
    let innerSeqLoadSSA = nextSSA ()
    let innerMoveNextPtrSSA = nextSSA ()
    let innerMoveNextResultSSA = nextSSA ()

    let checkInnerOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (innerSeqPtrSSA, takeSeqPtrSSA, 3, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoadSSA, innerSeqPtrSSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerMoveNextPtrSSA, innerSeqLoadSSA, [2], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some innerMoveNextResultSSA, innerMoveNextPtrSSA,
            [{ SSA = innerSeqPtrSSA; Type = TPtr }], MLIRTypes.i1))
        MLIROp.CFOp (CF.condBrSimple innerMoveNextResultSSA yieldBlockRef doneBlockRef)
    ]

    let checkInnerBlock: Block = { Label = checkInnerBlockRef; Args = []; Ops = checkInnerOps }

    // Yield block
    let innerSeqPtr2SSA = nextSSA ()
    let innerSeqLoad2SSA = nextSSA ()
    let innerCurrentSSA = nextSSA ()
    let currentPtrSSA = nextSSA ()
    let remainingPtr2SSA = nextSSA ()
    let remainingVal2SSA = nextSSA ()
    let newRemainingSSA = nextSSA ()
    let statePtrSSA = nextSSA ()

    let yieldOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (innerSeqPtr2SSA, takeSeqPtrSSA, 3, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoad2SSA, innerSeqPtr2SSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerCurrentSSA, innerSeqLoad2SSA, [1], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.StructGEP (currentPtrSSA, takeSeqPtrSSA, 1, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (innerCurrentSSA, currentPtrSSA, elementType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.StructGEP (remainingPtr2SSA, takeSeqPtrSSA, 4, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (remainingVal2SSA, remainingPtr2SSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.SubI (newRemainingSSA, remainingVal2SSA, oneI32SSA, TInt I32))
        MLIROp.LLVMOp (LLVMOp.Store (newRemainingSSA, remainingPtr2SSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA, takeSeqPtrSSA, 0, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (oneI32SSA, statePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some trueLitSSA, Some MLIRTypes.i1))
    ]

    let yieldBlock: Block = { Label = yieldBlockRef; Args = []; Ops = yieldOps }

    // Done block
    let donePtrSSA = nextSSA ()
    let doneOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (donePtrSSA, takeSeqPtrSSA, 0, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (neg1SSA, donePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))
    ]

    let doneBlock: Block = { Label = doneBlockRef; Args = []; Ops = doneOps }

    let bodyRegion: Region = { Blocks = [entryBlock; checkInnerBlock; yieldBlock; doneBlock] }

    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (moveNextName, [(takeSeqPtrSSA, TPtr)], MLIRTypes.i1, bodyRegion, LLVMPrivate))

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq.take - create TakeSeq wrapper
/// Returns: (inlineOps, topLevelOps, result)
let witnessSeqTake
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (count: Val)
    (innerSeq: Val)
    (elementType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let takeSeqType = takeSeqStructType elementType innerSeqType
    let moveNextName = generateMoveNextName "take" appNodeId
    let ssas = requireSSAs appNodeId ssa

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    let moveNextFuncOp = witnessTakeMoveNext moveNextName takeSeqType elementType innerSeqType

    let inlineOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, count.SSA, [4], takeSeqType))
    ]

    (inlineOps, [moveNextFuncOp], TRValue { SSA = resultSSA; Type = takeSeqType })
