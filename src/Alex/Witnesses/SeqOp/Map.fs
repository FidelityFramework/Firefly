/// SeqOp Map Witness - Seq.map operation
///
/// PRD-16: Seq.map creates MapSeq wrapper with flat closure
///
/// SCOPE: witnessSeqMap, witnessMapMoveNext
/// DOES NOT: Filter, take, fold, collect (separate witnesses)
module Alex.Witnesses.SeqOp.Map

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

/// Build MapSeq struct type
/// Layout: {state: i32, current: B, moveNext_ptr: ptr, inner_seq: InnerSeqType, mapper: Closure}
let private mapSeqStructType (outputElementType: MLIRType) (innerSeqType: MLIRType) : MLIRType =
    TStruct [TInt I32; outputElementType; TPtr; innerSeqType; closureType]

// ═══════════════════════════════════════════════════════════════════════════
// MOVENEXT GENERATOR
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

    // SSA layout (pre-assigned by SSAAssignment nanopass):
    // v0 = arg ptr to MapSeq struct
    // v1 = extract moveNext_ptr from inner_seq (field 2)
    // v2 = GEP inner_seq field (field 3)
    // v3 = call inner.MoveNext() -> i1
    // v4 = extract inner.current (field 1)
    // v5 = extract code_ptr from mapper (field 0 of field 4)
    // v6 = extract env_ptr from mapper (field 1 of field 4)
    // v7 = call mapper(env_ptr, inner.current) -> B
    // v8 = GEP current field (field 1)
    // v9 = store mapped_value to current
    // v10 = constant true (i1 1)
    // v11 = constant false (i1 0)

    let selfPtr = Arg 0

    // Extract inner_seq's moveNext_ptr (field 2 of inner_seq at field 3)
    let innerSeqGEP_SSA = V 2
    let moveNextPtr_SSA = V 1

    // Call inner.MoveNext()
    let innerMovedSSA = V 3

    // Extract inner.current (field 1 of inner_seq)
    let innerCurrentSSA = V 4

    // Extract mapper closure (field 4) and call it
    let mapperCodePtrSSA = V 5
    let mapperEnvPtrSSA = V 6
    let mappedValueSSA = V 7

    // Store result to self.current (field 1)
    let currentGEP_SSA = V 8

    let trueSSA = V 10
    let falseSSA = V 11

    let bodyOps = [
        // Get pointer to inner_seq field (field 3)
        MLIROp.LLVMOp (LLVMOp.GEP (innerSeqGEP_SSA, selfPtr, [(V (-1), MLIRTypes.i32)], mapSeqType, Some 3))

        // Extract moveNext_ptr from inner_seq (field 2 of inner_seq)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (moveNextPtr_SSA, innerSeqGEP_SSA, [2], innerSeqType))

        // Call inner.MoveNext(inner_seq_ptr) -> i1
        MLIROp.LLVMOp (LLVMOp.IndirectCall (innerMovedSSA, moveNextPtr_SSA, [(innerSeqGEP_SSA, MLIRTypes.ptr)], MLIRTypes.i1))

        // Branch on result
        CF.condBr innerMovedSSA (BlockRef "map_inner_moved") (BlockRef "map_done")
    ]

    let mapInnerMovedBlock = {
        Label = BlockRef "map_inner_moved"
        Args = []
        Ops = [
            // Extract inner.current (field 1 of inner_seq)
            MLIROp.LLVMOp (LLVMOp.ExtractValue (innerCurrentSSA, innerSeqGEP_SSA, [1], innerSeqType))

            // Extract mapper closure fields (field 4 of MapSeq)
            MLIROp.LLVMOp (LLVMOp.GEP (V 12, selfPtr, [(V (-1), MLIRTypes.i32)], mapSeqType, Some 4))
            MLIROp.LLVMOp (LLVMOp.ExtractValue (mapperCodePtrSSA, V 12, [0], closureType))
            MLIROp.LLVMOp (LLVMOp.ExtractValue (mapperEnvPtrSSA, V 12, [1], closureType))

            // Call mapper(env_ptr, value) -> B
            MLIROp.LLVMOp (LLVMOp.IndirectCall (mappedValueSSA, mapperCodePtrSSA, [(mapperEnvPtrSSA, MLIRTypes.ptr); (innerCurrentSSA, inputElementType)], outputElementType))

            // Store result to self.current (field 1)
            MLIROp.LLVMOp (LLVMOp.GEP (currentGEP_SSA, selfPtr, [(V (-1), MLIRTypes.i32)], mapSeqType, Some 1))
            MLIROp.LLVMOp (LLVMOp.Store (mappedValueSSA, currentGEP_SSA, outputElementType, NotAtomic))

            // Return true
            MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, MLIRTypes.i1))
            MLIROp.LLVMOp (LLVMOp.Return (Some trueSSA, Some MLIRTypes.i1))
        ]
    }

    let mapDoneBlock = {
        Label = BlockRef "map_done"
        Args = []
        Ops = [
            // Return false
            MLIROp.ArithOp (ArithOp.ConstI (falseSSA, 0L, MLIRTypes.i1))
            MLIROp.LLVMOp (LLVMOp.Return (Some falseSSA, Some MLIRTypes.i1))
        ]
    }

    let entryBlock = {
        Label = BlockRef "entry"
        Args = []
        Ops = bodyOps
    }

    let bodyRegion = {
        Blocks = [entryBlock; mapInnerMovedBlock; mapDoneBlock]
    }

    let funcOp = LLVMOp.LLVMFuncDef (
        moveNextName,
        [(Arg 0, MLIRTypes.ptr)],
        MLIRTypes.i1,
        bodyRegion,
        LLVMPrivate
    )

    MLIROp.LLVMOp funcOp

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq.map - create MapSeq wrapper
/// Returns: (inlineOps, topLevelOps, result)
/// - inlineOps: Construction of MapSeq struct
/// - topLevelOps: MoveNext function definition (emitted at module level)
let witnessSeqMap
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapper: Val)
    (innerSeq: Val)
    (inputElementType: MLIRType)
    (outputElementType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =

    let innerSeqType = innerSeq.Type
    let mapSeqType = mapSeqStructType outputElementType innerSeqType
    let moveNextName = generateMoveNextName "map" appNodeId
    let ssas = requireSSAs appNodeId ssa

    // SSAs pre-assigned by SSAAssignment nanopass
    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    let withInnerSeqSSA = ssas.[5]
    let resultSSA = ssas.[6]

    // MoveNext function - returned as top-level op, caller accumulates
    let moveNextFuncOp = witnessMapMoveNext moveNextName mapSeqType inputElementType outputElementType innerSeqType

    let inlineOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withInnerSeqSSA, withCodePtrSSA, innerSeq.SSA, [3], mapSeqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withInnerSeqSSA, mapper.SSA, [4], mapSeqType))
    ]

    (inlineOps, [moveNextFuncOp], TRValue { SSA = resultSSA; Type = mapSeqType })
