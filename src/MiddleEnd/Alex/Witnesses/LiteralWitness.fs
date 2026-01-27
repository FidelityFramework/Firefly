/// Literal Witness - Witness literal values to MLIR
///
/// ARCHITECTURAL PRINCIPLE: Witnesses OBSERVE and RETURN structured MLIROp.
/// They do NOT emit. The FOLD accumulates via withOps.
/// ZERO SPRINTF - all operations through structured types.
/// SSAs come from pre-computed SSAAssignment coeffect, NOT freshSynthSSA.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Bindings.PlatformTypes

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Fat string type: { ptr, i64 }
let private fatStringType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Derive byte length of string in UTF-8 encoding
let private deriveStringByteLength (s: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(s)

/// Get pre-assigned SSAs for a node, fail if not found
let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get result SSA for a node (the final SSA from its allocation)
let private getSingleSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "No result SSA for node %A" nodeId

/// Get 5 SSAs for string literal expansion
let private getStringSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA * SSA * SSA * SSA * SSA =
    match requireSSAs nodeId ssa with
    | [ptrSSA; lenSSA; undefSSA; withPtrSSA; fatPtrSSA] ->
        (ptrSSA, lenSSA, undefSSA, withPtrSSA, fatPtrSSA)
    | ssas -> failwithf "Expected 5 SSAs for string literal, got %d" (List.length ssas)

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a literal value and generate corresponding MLIR
/// ssa: Pre-computed SSA assignments
/// arch: Target architecture
/// nodeId: The ID of the literal node being witnessed
/// lit: The literal value to witness
/// Returns: (operations generated, result info)
let witness
    (ssa: SSAAssign.SSAAssignment)
    (arch: Architecture)
    (nodeId: NodeId)
    (lit: NativeLiteral)
    : MLIROp list * TransferResult =

    match lit with
    | NativeLiteral.Unit ->
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUunit
        let op = MLIROp.ArithOp (ConstI (ssaName, 0L, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Bool b ->
        let value = if b then 1L else 0L
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool
        let op = MLIROp.ArithOp (ConstI (ssaName, value, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Int (n, kind) ->
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstI (ssaName, n, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.UInt (n, kind) ->
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Char c ->
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUchar
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 c, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Float (f, kind) ->
        let ssaName = getSingleSSA nodeId ssa
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstF (ssaName, f, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.String s ->
        let (ptrSSA, lenSSA, undefSSA, withPtrSSA, fatPtrSSA) = getStringSSAs nodeId ssa
        let hash = uint32 (s.GetHashCode())
        let byteLen = deriveStringByteLength s

        let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GString hash))
        let lenOp = MLIROp.ArithOp (ConstI (lenSSA, int64 byteLen, MLIRTypes.i64))
        let undefOp = MLIROp.LLVMOp (Undef (undefSSA, fatStringType))
        let insertPtrOp = MLIROp.LLVMOp (InsertValue (withPtrSSA, undefSSA, ptrSSA, [0], fatStringType))
        let insertLenOp = MLIROp.LLVMOp (InsertValue (fatPtrSSA, withPtrSSA, lenSSA, [1], fatStringType))

        let ops = [addrOp; lenOp; undefOp; insertPtrOp; insertLenOp]
        ops, TRValue { SSA = fatPtrSSA; Type = fatStringType }

    | _ ->
        [], TRError $"Unsupported literal: {lit}"
