/// Seq Creation Witness - Witness Seq<'T> structure creation to MLIR
///
/// PRD-15: Lazy sequence expressions with flat closure architecture
///
/// SCOPE: Seq struct creation with captures and internal state
/// DOES NOT: MoveNext state machine, iteration loops (separate witnesses)
///
/// FLAT CLOSURE ARCHITECTURE (January 2026):
/// Sequences are "extended flat closures" - captures are inlined directly.
/// NO env_ptr, NO nulls - following MLKit-style flat closure principles.
///
/// SEQ STRUCT LAYOUT:
/// !seq_T = !llvm.struct<(i32, T, ptr, cap₀, cap₁, ...)>
///   - Field 0: State (i32) - 0=initial, N=after yield N, -1=done
///   - Field 1: Current value (T) - valid when MoveNext returns true
///   - Field 2: MoveNext function pointer
///   - Field 3+: Inlined captured values
module Alex.Witnesses.Seq.Creation

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSAs for a node, fail if not found
let requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// SEQ STRUCT TYPE (FLAT CLOSURE MODEL)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the MLIR type for Seq<T> with captures and internal state
/// Layout: { state: i32, current: T, moveNext_ptr: ptr, cap₀, ..., capₙ, state₀, ..., stateₘ }
let seqStructTypeFull
    (elementType: MLIRType)
    (captureTypes: MLIRType list)
    (internalStateTypes: MLIRType list)
    : MLIRType =
    TStruct ([TInt I32; elementType; TPtr] @ captureTypes @ internalStateTypes)

/// Build the MLIR type for Seq<T> with N captures (no internal state)
/// Layout: { state: i32, current: T, moveNext_ptr: ptr, cap₀, cap₁, ... }
let seqStructType (elementType: MLIRType) (captureTypes: MLIRType list) : MLIRType =
    seqStructTypeFull elementType captureTypes []

/// Seq struct type with no captures (simplest case)
let seqStructTypeNoCaptures (elementType: MLIRType) : MLIRType =
    TStruct [TInt I32; elementType; TPtr]

// ═══════════════════════════════════════════════════════════════════════════
// SEQ.CREATE - Build sequence expression (FLAT CLOSURE)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness seq expression creation
///
/// Input:
///   - moveNextName: Name of the MoveNext function
///   - elementType: The type T in Seq<T>
///   - captureVals: Values captured by the sequence expression
///
/// Output:
///   - Seq struct: {state=0, current=undef, moveNext_ptr, cap₀, cap₁, ...}
///
/// SSA cost: 5 + numCaptures
let witnessSeqCreate
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (moveNextName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let seqType = seqStructType elementType captureTypes
    let ssas = requireSSAs appNodeId ssa

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]

    let baseOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, seqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], seqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], seqType))
    ]

    let captureOps, finalSSA =
        if captureVals.IsEmpty then
            [], withCodePtrSSA
        else
            let ops, lastSSA =
                captureVals
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                    let nextSSA = ssas.[5 + i]
                    let captureIndex = 3 + i
                    let op = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [captureIndex], seqType))
                    (accOps @ [op], nextSSA)
                ) ([], withCodePtrSSA)
            ops, lastSSA

    (baseOps @ captureOps, TRValue { SSA = finalSSA; Type = seqType })

/// Witness seq expression creation with internal state fields
///
/// PRD-15: For while-based sequences with internal mutable state,
/// the seq struct must include slots for internal state fields.
///
/// SSA cost: 5 + numCaptures + numInternalState
let witnessSeqCreateFull
    (appNodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (moveNextName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    (internalStateTypes: MLIRType list)
    : (MLIROp list * TransferResult) =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let seqType = seqStructTypeFull elementType captureTypes internalStateTypes
    let ssas = requireSSAs appNodeId ssa
    let numCaptures = List.length captureVals

    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]

    let baseOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, seqType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], seqType))
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], seqType))
    ]

    let captureOps, afterCapturesSSA =
        if captureVals.IsEmpty then
            [], withCodePtrSSA
        else
            let ops, lastSSA =
                captureVals
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                    let nextSSA = ssas.[5 + i]
                    let captureIndex = 3 + i
                    let op = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [captureIndex], seqType))
                    (accOps @ [op], nextSSA)
                ) ([], withCodePtrSSA)
            ops, lastSSA

    let internalStateOps, finalSSA =
        if internalStateTypes.IsEmpty then
            [], afterCapturesSSA
        else
            let ops, lastSSA =
                internalStateTypes
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, stateType) ->
                    let zeroConstSSA = ssas.[5 + numCaptures + i * 2]
                    let nextSSA = ssas.[5 + numCaptures + i * 2 + 1]
                    let stateIndex = 3 + numCaptures + i
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroConstSSA, 0L, stateType))
                    let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, zeroConstSSA, [stateIndex], seqType))
                    (accOps @ [zeroOp; insertOp], nextSSA)
                ) ([], afterCapturesSSA)
            ops, lastSSA

    (baseOps @ captureOps @ internalStateOps, TRValue { SSA = finalSSA; Type = seqType })
