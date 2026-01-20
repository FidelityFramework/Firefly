/// Literal Witness - Witness literal values to MLIR
///
/// ARCHITECTURAL PRINCIPLE: Witnesses OBSERVE and RETURN structured MLIROp.
/// They do NOT emit. The FOLD accumulates via withOps.
/// ZERO SPRINTF - all operations through structured types.
/// SSAs come from pre-computed SSAAssignment coeffect, NOT freshSynthSSA.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
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
///
/// PRINCIPLED DESIGN (January 2026):
/// The NTUKind in the literal IS the type. We use mapNTUKindToMLIRType to get
/// the correct MLIR type with architecture awareness (e.g., nativeint → i32 or i64).
let witness (z: PSGZipper) (nodeId: NodeId) (lit: NativeLiteral) : MLIROp list * TransferResult =
    let arch = z.State.Platform.TargetArch

    match lit with
    | NativeLiteral.Unit ->
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUunit
        let op = MLIROp.ArithOp (ConstI (ssaName, 0L, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Bool b ->
        let value = if b then 1L else 0L
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool
        let op = MLIROp.ArithOp (ConstI (ssaName, value, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    // Integer literals: NTUKind IS the type, resolved per architecture
    | NativeLiteral.Int (n, kind) ->
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstI (ssaName, n, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    // Unsigned integers where value > int64.MaxValue
    | NativeLiteral.UInt (n, kind) ->
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 n, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.Char c ->
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUchar
        let op = MLIROp.ArithOp (ConstI (ssaName, int64 c, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    // Float literals: NTUKind determines f32 vs f64
    | NativeLiteral.Float (f, kind) ->
        let ssaName = getSingleSSA nodeId z
        let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
        let op = MLIROp.ArithOp (ConstF (ssaName, f, ty))
        [op], TRValue { SSA = ssaName; Type = ty }

    | NativeLiteral.String s ->
        // Native string: fat pointer struct { ptr: !llvm.ptr, len: i64 }
        // Get pre-assigned SSAs for all 5 operations
        let (ptrSSA, lenSSA, undefSSA, withPtrSSA, fatPtrSSA) = getStringSSAs nodeId z

        // 1. Derive hash and byte length via pure functions (no side effects)
        // StringTable coeffect already knows about this string from preprocessing
        let hash = uint32 (s.GetHashCode())
        let byteLen = deriveStringByteLength s

        // 2. Get address of global string (GString hash references the global)
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

// ═══════════════════════════════════════════════════════════════════════════
// INTERPOLATED STRING WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Build a fat string from a string literal using pre-assigned SSAs
/// Takes SSAs at given index offset, returns (ops, Val) and number of SSAs consumed
let private buildStringPart (ssas: SSA list) (ssaOffset: int) (s: string) : MLIROp list * Val * int =
    // Derive hash and byte length via pure functions
    let hash = uint32 (s.GetHashCode())
    let byteLen = deriveStringByteLength s

    // Use 5 pre-assigned SSAs
    let ptrSSA = ssas.[ssaOffset]
    let lenSSA = ssas.[ssaOffset + 1]
    let undefSSA = ssas.[ssaOffset + 2]
    let withPtrSSA = ssas.[ssaOffset + 3]
    let fatPtrSSA = ssas.[ssaOffset + 4]

    // Build ops
    let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GString hash))
    let lenOp = MLIROp.ArithOp (ConstI (lenSSA, int64 byteLen, MLIRTypes.i64))
    let undefOp = MLIROp.LLVMOp (Undef (undefSSA, fatStringType))
    let insertPtrOp = MLIROp.LLVMOp (InsertValue (withPtrSSA, undefSSA, ptrSSA, [0], fatStringType))
    let insertLenOp = MLIROp.LLVMOp (InsertValue (fatPtrSSA, withPtrSSA, lenSSA, [1], fatStringType))

    let ops = [addrOp; lenOp; undefOp; insertPtrOp; insertLenOp]
    ops, { SSA = fatPtrSSA; Type = fatStringType }, 5

/// Concatenate two string Vals using pre-assigned SSAs
/// Takes SSAs at given index offset, returns (ops, Val) and number of SSAs consumed
let private concatStringsAt (ssas: SSA list) (ssaOffset: int) (str1: Val) (str2: Val) : MLIROp list * Val * int =
    // Use 10 pre-assigned SSAs
    let ptr1SSA = ssas.[ssaOffset]
    let len1SSA = ssas.[ssaOffset + 1]
    let ptr2SSA = ssas.[ssaOffset + 2]
    let len2SSA = ssas.[ssaOffset + 3]
    let totalLenSSA = ssas.[ssaOffset + 4]
    let bufSSA = ssas.[ssaOffset + 5]
    let offsetSSA = ssas.[ssaOffset + 6]
    let undefSSA = ssas.[ssaOffset + 7]
    let withPtrSSA = ssas.[ssaOffset + 8]
    let resultSSA = ssas.[ssaOffset + 9]

    let extractOps = [
        MLIROp.LLVMOp (ExtractValue (ptr1SSA, str1.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (ExtractValue (len1SSA, str1.SSA, [1], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (ExtractValue (ptr2SSA, str2.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (ExtractValue (len2SSA, str2.SSA, [1], MLIRTypes.nativeStr))
    ]

    // Total length
    let addLenOp = MLIROp.ArithOp (ArithOp.AddI (totalLenSSA, len1SSA, len2SSA, MLIRTypes.i64))

    // Allocate buffer
    let allocOp = MLIROp.LLVMOp (Alloca (bufSSA, totalLenSSA, MLIRTypes.i8, None))

    // Copy first string
    let bufVal = { SSA = bufSSA; Type = MLIRTypes.ptr }
    let ptr1Val = { SSA = ptr1SSA; Type = MLIRTypes.ptr }
    let len1Val = { SSA = len1SSA; Type = MLIRTypes.i64 }
    let memcpy1Op = intrMemcpy bufVal ptr1Val len1Val false

    // GEP to offset for second string
    let gepOp = MLIROp.LLVMOp (GEP (offsetSSA, bufSSA, [(len1SSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Copy second string
    let offsetVal = { SSA = offsetSSA; Type = MLIRTypes.ptr }
    let ptr2Val = { SSA = ptr2SSA; Type = MLIRTypes.ptr }
    let len2Val = { SSA = len2SSA; Type = MLIRTypes.i64 }
    let memcpy2Op = intrMemcpy offsetVal ptr2Val len2Val false

    // Build result fat string
    let buildStrOps = [
        MLIROp.LLVMOp (Undef (undefSSA, MLIRTypes.nativeStr))
        MLIROp.LLVMOp (InsertValue (withPtrSSA, undefSSA, bufSSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (InsertValue (resultSSA, withPtrSSA, totalLenSSA, [1], MLIRTypes.nativeStr))
    ]

    let allOps = extractOps @ [addLenOp; allocOp; memcpy1Op; gepOp; memcpy2Op] @ buildStrOps
    allOps, { SSA = resultSSA; Type = MLIRTypes.nativeStr }, 10

/// Witness an interpolated string
/// nodeId: The InterpolatedString node ID for SSA lookup
/// parts: list of InterpolatedPart (StringPart or ExprPart)
/// resolveExpr: function to resolve ExprPart NodeId to Val
let witnessInterpolated
    (nodeId: NodeId)
    (z: PSGZipper)
    (parts: InterpolatedPart list)
    (resolveExpr: NodeId -> Val option)
    : MLIROp list * TransferResult =

    // Get all pre-assigned SSAs for this InterpolatedString node
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaOffset = 0

    // Convert each part to (ops, Val)
    // String parts consume SSAs, expr parts don't (already computed)
    let partResults =
        parts
        |> List.choose (fun part ->
            match part with
            | InterpolatedPart.StringPart s ->
                let ops, v, consumed = buildStringPart ssas ssaOffset s
                ssaOffset <- ssaOffset + consumed
                Some (ops, v)
            | InterpolatedPart.ExprPart exprNodeId ->
                match resolveExpr exprNodeId with
                | Some v -> Some ([], v)  // No ops needed, expr already witnessed
                | None -> None)

    match partResults with
    | [] ->
        // Empty interpolated string - use pre-assigned SSAs
        let ops, v, _ = buildStringPart ssas ssaOffset ""
        ops, TRValue v
    | [(ops, v)] ->
        // Single part - just return it
        ops, TRValue v
    | (firstOps, firstVal) :: rest ->
        // Multiple parts - fold with concatenation
        let allOps, finalVal =
            rest
            |> List.fold (fun (accOps, accVal) (partOps, partVal) ->
                let concatOps, resultVal, consumed = concatStringsAt ssas ssaOffset accVal partVal
                ssaOffset <- ssaOffset + consumed
                (accOps @ partOps @ concatOps, resultVal)
            ) (firstOps, firstVal)

        allOps, TRValue finalVal
