/// SeqOpWitness - Witness Seq operations to MLIR
///
/// PRD-16: Seq.map, Seq.filter, Seq.take, Seq.fold, Seq.collect
///
/// ARCHITECTURAL PRINCIPLES:
/// - Seq operations create COMPOSED FLAT CLOSURES
/// - Each wrapper contains: {state, current, moveNext_ptr, inner_seq, closure}
/// - Both inner_seq and closure are COPIED BY VALUE (inlined)
/// - Follows the Photographer Principle: PSG is already complete, we just witness it
/// - MoveNext calls use INDIRECT calls through the inner seq's code_ptr field
///
/// WRAPPER STRUCT LAYOUTS:
/// - MapSeq<A,B>:    {state: i32, current: B, moveNext_ptr: ptr, inner_seq: Seq<A>, mapper: Closure}
/// - FilterSeq<A>:   {state: i32, current: A, moveNext_ptr: ptr, inner_seq: Seq<A>, predicate: Closure}
/// - TakeSeq<A>:     {state: i32, current: A, moveNext_ptr: ptr, inner_seq: Seq<A>, remaining: i32}
/// - CollectSeq<A,B>: {state: i32, current: B, moveNext_ptr: ptr, outer_seq, mapper, inner_seq}
///
/// CLOSURE INVOCATION (flat closure model - NO env_ptr indirection):
/// 1. Extract code_ptr from closure (index 0)
/// 2. Extract env_ptr from closure (index 1) - this is placeholder for flat captures
/// 3. Call code_ptr(env_ptr, value)
///
/// INNER SEQUENCE MOVENEXT:
/// 1. Get pointer to inner_seq field via GEP
/// 2. Extract code_ptr from inner_seq (index 2)
/// 3. Call code_ptr(inner_seq_ptr) - indirect call
module Alex.Witnesses.SeqOpWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Witnesses.SeqWitness

// Module aliases for dialect templates
module CF = Alex.Dialects.CF.Templates

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSAs for a node from the SSAAssignment coeffect
let private requireNodeSSAs nodeId (z: PSGZipper) =
    match PSGElaboration.SSAAssignment.lookupSSAs nodeId z.State.SSAAssignment with
    | Some ssas -> ssas
    | None -> failwithf "SeqOpWitness: No SSAs pre-assigned for node %d" (NodeId.value nodeId)

/// Standard uniform closure type: {code_ptr, env_ptr}
let closureType = TStruct [TPtr; TPtr]

/// Generate a unique MoveNext function name for a Seq operation
let private generateMoveNextName (opName: string) (appNodeId: NodeId) : string =
    sprintf "seqop_%s_moveNext_%d" opName (NodeId.value appNodeId)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE BUILDERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build MapSeq struct type
/// Layout: {state: i32, current: B, moveNext_ptr: ptr, inner_seq: InnerSeqType, mapper: Closure}
let mapSeqStructType (outputElementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; outputElementType; TPtr; innerSeqType; closureType]

/// Build FilterSeq struct type
/// Layout: {state: i32, current: A, moveNext_ptr: ptr, inner_seq: InnerSeqType, predicate: Closure}
let filterSeqStructType (elementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; elementType; TPtr; innerSeqType; closureType]

/// Build TakeSeq struct type
/// Layout: {state: i32, current: A, moveNext_ptr: ptr, inner_seq: InnerSeqType, remaining: i32}
let takeSeqStructType (elementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; elementType; TPtr; innerSeqType; TInt I32]

/// Build CollectSeq struct type
/// Layout: {state: i32, current: B, moveNext_ptr: ptr, outer_seq, mapper, inner_seq}
let collectSeqStructType
    (outputElementType: MLIRType)
    (outerSeqType: MLIRType)
    (innerSeqType: MLIRType)
    : MLIRType =
    TStruct [TInt I32; outputElementType; TPtr; outerSeqType; closureType; innerSeqType]

// ═══════════════════════════════════════════════════════════════════════════
// MOVENEXT GENERATORS (private - defined before witness functions that use them)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MoveNext function for MapSeq
/// Algorithm:
///   if inner.MoveNext():
///       self.current = mapper(inner.current)
///       return true
///   return false
let private witnessMapMoveNext
    (moveNextName: string)
    (mapSeqType: MLIRType)
    (inputElementType: MLIRType)
    (outputElementType: MLIRType)
    (innerSeqType: MLIRType)
    : MLIROp =

    // SSA counter for function-local SSAs (these are INSIDE the generated function)
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    let mapSeqPtrSSA = nextSSA ()  // %0 - function parameter

    // Constants
    let trueLitSSA = nextSSA ()
    let falseLitSSA = nextSSA ()
    let oneSSA = nextSSA ()
    let neg1SSA = nextSSA ()

    // Inner seq access
    let innerSeqPtrSSA = nextSSA ()
    let innerSeqLoadSSA = nextSSA ()
    let innerMoveNextPtrSSA = nextSSA ()
    let innerMoveNextResultSSA = nextSSA ()

    let callMapperBlockRef = BlockRef "call_mapper"
    let doneBlockRef = BlockRef "done"

    let entryOps = [
        MLIROp.ArithOp (ArithOp.ConstI (trueLitSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.ConstI (neg1SSA, -1L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.StructGEP (innerSeqPtrSSA, mapSeqPtrSSA, 3, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoadSSA, innerSeqPtrSSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerMoveNextPtrSSA, innerSeqLoadSSA, [2], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some innerMoveNextResultSSA, innerMoveNextPtrSSA,
            [{ SSA = innerSeqPtrSSA; Type = TPtr }], MLIRTypes.i1))
        MLIROp.CFOp (CF.condBrSimple innerMoveNextResultSSA callMapperBlockRef doneBlockRef)
    ]

    let entryBlock: Block = { Label = BlockRef "entry"; Args = []; Ops = entryOps }

    // Call mapper block
    let innerSeqLoad2SSA = nextSSA ()
    let innerCurrentSSA = nextSSA ()
    let loadMapSeqSSA = nextSSA ()
    let mapperSSA = nextSSA ()
    let mapperCodePtrSSA = nextSSA ()
    let mapperEnvPtrSSA = nextSSA ()
    let mapperResultSSA = nextSSA ()
    let currentPtrSSA = nextSSA ()
    let statePtrSSA = nextSSA ()

    let callMapperOps = [
        MLIROp.LLVMOp (LLVMOp.Load (innerSeqLoad2SSA, innerSeqPtrSSA, innerSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (innerCurrentSSA, innerSeqLoad2SSA, [1], innerSeqType))
        MLIROp.LLVMOp (LLVMOp.Load (loadMapSeqSSA, mapSeqPtrSSA, mapSeqType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (mapperSSA, loadMapSeqSSA, [4], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (mapperCodePtrSSA, mapperSSA, [0], closureType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (mapperEnvPtrSSA, mapperSSA, [1], closureType))
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some mapperResultSSA, mapperCodePtrSSA,
            [{ SSA = mapperEnvPtrSSA; Type = TPtr }; { SSA = innerCurrentSSA; Type = inputElementType }], outputElementType))
        MLIROp.LLVMOp (LLVMOp.StructGEP (currentPtrSSA, mapSeqPtrSSA, 1, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (mapperResultSSA, currentPtrSSA, outputElementType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA, mapSeqPtrSSA, 0, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (oneSSA, statePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some trueLitSSA, Some MLIRTypes.i1))
    ]

    let callMapperBlock: Block = { Label = callMapperBlockRef; Args = []; Ops = callMapperOps }

    // Done block
    let donePtrSSA = nextSSA ()
    let doneOps = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (donePtrSSA, mapSeqPtrSSA, 0, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.Store (neg1SSA, donePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))
    ]

    let doneBlock: Block = { Label = doneBlockRef; Args = []; Ops = doneOps }

    let bodyRegion: Region = { Blocks = [entryBlock; callMapperBlock; doneBlock] }

    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (moveNextName, [(mapSeqPtrSSA, TPtr)], MLIRTypes.i1, bodyRegion, LLVMPrivate))

/// Generate MoveNext function for FilterSeq
/// Algorithm:
///   while inner.MoveNext():
///       if predicate(inner.current):
///           self.current = inner.current
///           return true
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
// PUBLIC WITNESS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a MapSeq wrapper for Seq.map and emit the MoveNext function
/// Returns: (ops, result Val)
let witnessSeqMap
    (appNodeId: NodeId)
    (z: PSGZipper)
    (mapper: Val)
    (innerSeq: Val)
    (inputElementType: MLIRType)
    (outputElementType: MLIRType)
    : MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let mapSeqType = mapSeqStructType outputElementType innerSeqType
    let moveNextName = generateMoveNextName "map" appNodeId
    let ssas = requireNodeSSAs appNodeId z

    // SSAs pre-assigned by SSAAssignment nanopass
    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    // Emit MoveNext function at top level
    let moveNextFuncOp = witnessMapMoveNext moveNextName mapSeqType inputElementType outputElementType innerSeqType
    emitTopLevel moveNextFuncOp z

    let ops = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, mapper.SSA, [4], mapSeqType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = mapSeqType })

/// Create a FilterSeq wrapper for Seq.filter and emit the MoveNext function
let witnessSeqFilter
    (appNodeId: NodeId)
    (z: PSGZipper)
    (predicate: Val)
    (innerSeq: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let filterSeqType = filterSeqStructType elementType innerSeqType
    let moveNextName = generateMoveNextName "filter" appNodeId
    let ssas = requireNodeSSAs appNodeId z

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    let moveNextFuncOp = witnessFilterMoveNext moveNextName filterSeqType elementType innerSeqType
    emitTopLevel moveNextFuncOp z

    let ops = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], filterSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, predicate.SSA, [4], filterSeqType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = filterSeqType })

/// Create a TakeSeq wrapper for Seq.take and emit the MoveNext function
let witnessSeqTake
    (appNodeId: NodeId)
    (z: PSGZipper)
    (count: Val)
    (innerSeq: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let takeSeqType = takeSeqStructType elementType innerSeqType
    let moveNextName = generateMoveNextName "take" appNodeId
    let ssas = requireNodeSSAs appNodeId z

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    let moveNextFuncOp = witnessTakeMoveNext moveNextName takeSeqType elementType innerSeqType
    emitTopLevel moveNextFuncOp z

    let ops = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], takeSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, count.SSA, [4], takeSeqType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = takeSeqType })

/// Witness Seq.fold - an eager consumer that returns an accumulated value
/// This requires inline loop generation - returns error for special handling in FNCSTransfer
let witnessSeqFold
    (appNodeId: NodeId)
    (z: PSGZipper)
    (folder: Val)
    (initial: Val)
    (seq: Val)
    (accType: MLIRType)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    [], TRError (sprintf "Seq.fold requires special handling in FNCSTransfer - node %d" (NodeId.value appNodeId))

/// Seq.collect (flatMap) requires complex nested iteration state
/// Returns error for special handling
let witnessSeqCollect
    (appNodeId: NodeId)
    (z: PSGZipper)
    (mapper: Val)
    (outerSeq: Val)
    (outputElementType: MLIRType)
    : MLIROp list * TransferResult =
    [], TRError (sprintf "Seq.collect requires complex nested iteration - node %d" (NodeId.value appNodeId))
