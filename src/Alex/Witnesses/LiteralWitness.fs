/// Literal Witness - Witness literal values to MLIR
///
/// ARCHITECTURAL PRINCIPLE: Witnesses OBSERVE and RETURN structured MLIROp.
/// They do NOT emit. The FOLD accumulates via withOps.
/// ZERO SPRINTF - all operations through structured types.
/// SSAs come from pre-computed SSAAssignment coeffect, NOT freshSynthSSA.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Fat string type: { ptr, i64 }
let private fatStringType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

// ═══════════════════════════════════════════════════════════════════════════
// SSA ALLOCATION ACCESSORS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSAs for a specific node ID, fail if not found
let private getSSAsForNode (nodeId: NodeId) (z: PSGZipper) : SSA list =
    requireNodeSSAs nodeId z

/// Get single SSA (for simple literals that expand to 1 op)
let private getSingleSSA (nodeId: NodeId) (z: PSGZipper) : SSA =
    match getSSAsForNode nodeId z with
    | [ssa] -> ssa
    | ssas -> failwithf "Expected 1 SSA for simple literal, got %d" (List.length ssas)

/// Get 5 SSAs for string literal expansion
let private getStringSSAs (nodeId: NodeId) (z: PSGZipper) : SSA * SSA * SSA * SSA * SSA =
    match getSSAsForNode nodeId z with
    | [ptrSSA; lenSSA; undefSSA; withPtrSSA; fatPtrSSA] ->
        (ptrSSA, lenSSA, undefSSA, withPtrSSA, fatPtrSSA)
    | ssas -> failwithf "Expected 5 SSAs for string literal, got %d" (List.length ssas)

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a literal value and generate corresponding MLIR
/// SSAs are looked up from pre-computed SSAAssignment coeffect.
/// nodeId: The ID of the literal node being witnessed (passed from traversal)
/// Returns: (operations generated, result info)
let witness (z: PSGZipper) (nodeId: NodeId) (lit: LiteralValue) : MLIROp list * TransferResult =
    match lit with
    | LiteralValue.Unit ->
        // Unit is represented as i32 0 (consistent with C ABI)
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, 0L, MLIRTypes.i32))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i32 }

    | LiteralValue.Bool b ->
        let value = if b then 1L else 0L
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, value, MLIRTypes.i1))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i1 }

    | LiteralValue.Int8 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i8))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i8 }

    | LiteralValue.Int16 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i16))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i16 }

    | LiteralValue.Int32 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i32))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i32 }

    | LiteralValue.Int64 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, n, MLIRTypes.i64))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i64 }

    | LiteralValue.UInt8 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i8))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i8 }

    | LiteralValue.UInt16 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i16))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i16 }

    | LiteralValue.UInt32 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i32))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i32 }

    | LiteralValue.UInt64 n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i64))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i64 }

    | LiteralValue.NativeInt n ->
        // Platform word size - assume 64-bit for now
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i64))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i64 }

    | LiteralValue.UNativeInt n ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, MLIRTypes.i64))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i64 }

    | LiteralValue.Char c ->
        // Char is i32 (Unicode codepoint)
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 c, MLIRTypes.i32))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.i32 }

    | LiteralValue.Float32 f ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstF (ssaName, float f, MLIRTypes.f32))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.f32 }

    | LiteralValue.Float64 f ->
        let ssaName = getSingleSSA nodeId z
        let op = MLIROp.ArithOp (ConstF (ssaName, f, MLIRTypes.f64))
        [op], TRValue { SSA = ssaName; Type = MLIRTypes.f64 }

    | LiteralValue.String s ->
        // Native string: fat pointer struct { ptr: !llvm.ptr, len: i64 }
        // Get pre-assigned SSAs for all 5 operations
        let (ptrSSA, lenSSA, undefSSA, withPtrSSA, fatPtrSSA) = getStringSSAs nodeId z

        // 1. Register string and get hash for global reference
        let _ = registerString s z  // Side effect: adds to z.State.Strings
        let hash = uint32 (s.GetHashCode())  // Must match registerString's hash calculation
        let byteLen = System.Text.Encoding.UTF8.GetByteCount(s)

        // 2. Get address of global string
        let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GString hash))

        // 3. Create length constant
        let lenOp = MLIROp.ArithOp (ConstI (lenSSA, int64 byteLen, MLIRTypes.i64))

        // 4. Build fat pointer struct: undef -> insert ptr -> insert len
        let undefOp = MLIROp.LLVMOp (Undef (undefSSA, fatStringType))
        let insertPtrOp = MLIROp.LLVMOp (InsertValue (withPtrSSA, undefSSA, ptrSSA, [0], fatStringType))
        let insertLenOp = MLIROp.LLVMOp (InsertValue (fatPtrSSA, withPtrSSA, lenSSA, [1], fatStringType))

        // Return all ops and the result (fatPtrSSA is the final result)
        let ops = [addrOp; lenOp; undefOp; insertPtrOp; insertLenOp]
        ops, TRValue { SSA = fatPtrSSA; Type = fatStringType }

    | _ ->
        [], TRError $"Unsupported literal: {lit}"
