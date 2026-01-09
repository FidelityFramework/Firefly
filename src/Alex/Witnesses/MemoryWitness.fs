/// Memory Witness - Witness memory and data structure operations to MLIR
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Witnesses OBSERVE and RETURN structured MLIROp lists.
/// They do NOT emit strings. ZERO SPRINTF for MLIR generation.
///
/// Handles:
/// - Array/collection indexing (IndexGet, IndexSet)
/// - Address-of operator (AddressOf)
/// - Tuple/Record/Array/List construction
/// - Field access (FieldGet, FieldSet)
/// - SRTP trait calls
module Alex.Witnesses.MemoryWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Map NativeType to MLIRType (delegated to TypeMapping)
let private mapType = mapNativeType

/// Check if a type is the native string type
let private isNativeStrType (ty: MLIRType) : bool =
    ty = MLIRTypes.nativeStr

// ═══════════════════════════════════════════════════════════════════════════
// INDEX OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array/collection index get
/// Generates GEP + Load sequence
let witnessIndexGet
    (z: PSGZipper)
    (collSSA: SSA)
    (collType: MLIRType)
    (indexSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // GEP to compute element address
    let ptrSSA = freshSynthSSA z
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, collSSA, [(indexSSA, MLIRTypes.i64)], elemType))

    // Load the element
    let loadSSA = freshSynthSSA z
    let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, ptrSSA, elemType, NotAtomic))

    [gepOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }

/// Witness array/collection index set
/// Generates GEP + Store sequence
let witnessIndexSet
    (z: PSGZipper)
    (collSSA: SSA)
    (indexSSA: SSA)
    (valueSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // GEP to compute element address
    let ptrSSA = freshSynthSSA z
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, collSSA, [(indexSSA, MLIRTypes.i64)], elemType))

    // Store the value
    let storeOp = MLIROp.LLVMOp (LLVMOp.Store (valueSSA, ptrSSA, elemType, NotAtomic))

    [gepOp; storeOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// ADDRESS-OF OPERATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Witness address-of operator (&expr)
/// For mutable bindings, the SSA already represents the alloca address
/// For immutable values, we need to allocate and store
let witnessAddressOf
    (z: PSGZipper)
    (exprSSA: SSA)
    (exprType: MLIRType)
    (isMutable: bool)
    : MLIROp list * TransferResult =

    if isMutable then
        // Mutable binding already has an address (alloca)
        // The SSA is the pointer to the mutable slot
        [], TRValue { SSA = exprSSA; Type = TPtr }
    else
        // Need to allocate stack space and store the value
        let oneSSA = freshSynthSSA z
        let allocaSSA = freshSynthSSA z

        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, oneSSA, exprType, None))
            MLIROp.LLVMOp (LLVMOp.Store (exprSSA, allocaSSA, exprType, NotAtomic))
        ]

        ops, TRValue { SSA = allocaSSA; Type = TPtr }

// ═══════════════════════════════════════════════════════════════════════════
// FIELD ACCESS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness field get (struct.field or record.field)
/// Uses extractvalue for value types
let witnessFieldGet
    (z: PSGZipper)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldSSA = freshSynthSSA z
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, [fieldIndex], structType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness field set (struct.field <- value)
/// Uses insertvalue for value types (returns new struct)
let witnessFieldSet
    (z: PSGZipper)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (valueSSA: SSA)
    : MLIROp list * TransferResult =

    let newStructSSA = freshSynthSSA z
    let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newStructSSA, structSSA, valueSSA, [fieldIndex], structType))

    [insertOp], TRValue { SSA = newStructSSA; Type = structType }

// ═══════════════════════════════════════════════════════════════════════════
// TUPLE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness tuple construction
/// Builds a struct by inserting each element
let witnessTupleExpr
    (z: PSGZipper)
    (elements: Val list)
    : MLIROp list * TransferResult =

    let elemTypes = elements |> List.map (fun v -> v.Type)
    let tupleType = TStruct elemTypes

    // Start with undef
    let undefSSA = freshSynthSSA z
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, tupleType))

    // Insert each element
    let insertOps, finalSSA =
        elements
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, elem) ->
            let newSSA = freshSynthSSA z
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, elem.SSA, [idx], tupleType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = tupleType }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness record construction
/// Records are represented as structs with fields in declaration order
let witnessRecordExpr
    (z: PSGZipper)
    (fields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    // Start with undef
    let undefSSA = freshSynthSSA z
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, recordType))

    // Insert each field
    let insertOps, finalSSA =
        fields
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, (_, fieldVal)) ->
            let newSSA = freshSynthSSA z
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, fieldVal.SSA, [idx], recordType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = recordType }

// ═══════════════════════════════════════════════════════════════════════════
// ARRAY CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array expression [| e1; e2; ... |]
/// Allocates array and stores each element
let witnessArrayExpr
    (z: PSGZipper)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let count = List.length elements
    let countSSA = freshSynthSSA z
    let allocaSSA = freshSynthSSA z

    // Allocate array on stack
    let allocOps = [
        MLIROp.ArithOp (ArithOp.ConstI (countSSA, int64 count, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, countSSA, elemType, None))
    ]

    // Store each element
    let storeOps =
        elements
        |> List.indexed
        |> List.collect (fun (idx, elem) ->
            let idxSSA = freshSynthSSA z
            let ptrSSA = freshSynthSSA z
            [
                MLIROp.ArithOp (ArithOp.ConstI (idxSSA, int64 idx, MLIRTypes.i64))
                MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, allocaSSA, [(idxSSA, MLIRTypes.i64)], elemType))
                MLIROp.LLVMOp (LLVMOp.Store (elem.SSA, ptrSSA, elemType, NotAtomic))
            ]
        )

    // Build fat array struct { ptr, count }
    let undefSSA = freshSynthSSA z
    let withPtrSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z

    let arrayType = TStruct [TPtr; MLIRTypes.i64]

    let buildArrayOps = [
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, allocaSSA, [0], arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, countSSA, [1], arrayType))
    ]

    allocOps @ storeOps @ buildArrayOps, TRValue { SSA = resultSSA; Type = arrayType }

// ═══════════════════════════════════════════════════════════════════════════
// LIST CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness list expression [e1; e2; ...]
/// Lists are represented as linked cons cells or array-backed
/// For now, using array representation for simplicity
let witnessListExpr
    (z: PSGZipper)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // Use same representation as arrays for now
    witnessArrayExpr z elements elemType

// ═══════════════════════════════════════════════════════════════════════════
// UNION CASE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness discriminated union case construction
/// DU layout: { i32 discriminator, payload }
let witnessUnionCase
    (z: PSGZipper)
    (tag: int)
    (payload: Val option)
    (unionType: MLIRType)
    : MLIROp list * TransferResult =

    // Create discriminator constant
    let tagSSA = freshSynthSSA z
    let tagOp = MLIROp.ArithOp (ArithOp.ConstI (tagSSA, int64 tag, MLIRTypes.i32))

    // Start with undef
    let undefSSA = freshSynthSSA z
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, unionType))

    // Insert tag at field 0
    let withTagSSA = freshSynthSSA z
    let insertTagOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withTagSSA, undefSSA, tagSSA, [0], unionType))

    match payload with
    | Some payloadVal ->
        // Insert payload at field 1
        let resultSSA = freshSynthSSA z
        let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withTagSSA, payloadVal.SSA, [1], unionType))
        [tagOp; undefOp; insertTagOp; insertPayloadOp], TRValue { SSA = resultSSA; Type = unionType }
    | None ->
        // No payload, just return with tag
        [tagOp; undefOp; insertTagOp], TRValue { SSA = withTagSSA; Type = unionType }

// ═══════════════════════════════════════════════════════════════════════════
// SRTP TRAIT CALLS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness SRTP trait call
/// The trait has been resolved by FNCS to a specific member
/// We generate the appropriate member access/call
let witnessTraitCall
    (z: PSGZipper)
    (receiverSSA: SSA)
    (receiverType: MLIRType)
    (memberName: string)
    (memberType: MLIRType)
    : MLIROp list * TransferResult =

    // For property-like traits (e.g., Length), generate field access
    // For method-like traits, this would need to generate a call
    // The specific implementation depends on the resolved member

    match memberName with
    | "Length" when isNativeStrType receiverType ->
        // String.Length - extract length field (index 1) from fat pointer
        let lenSSA = freshSynthSSA z
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = lenSSA; Type = MLIRTypes.i64 }

    | "Length" ->
        // Array/collection Length - extract count field (index 1) from fat pointer
        let lenSSA = freshSynthSSA z
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = lenSSA; Type = MLIRTypes.i64 }

    | "Pointer" when isNativeStrType receiverType ->
        // String.Pointer - extract pointer field (index 0) from fat pointer
        let ptrSSA = freshSynthSSA z
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (ptrSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = ptrSSA; Type = TPtr }

    | _ ->
        // Generic field access - assume field index 0 for unknown traits
        // This is a fallback and should be refined based on type information
        let resultSSA = freshSynthSSA z
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = memberType }

// ═══════════════════════════════════════════════════════════════════════════
// POINTER OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativePtr.read (pointer dereference)
let witnessNativePtrRead
    (z: PSGZipper)
    (ptrSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let loadSSA = freshSynthSSA z
    let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, ptrSSA, elemType, NotAtomic))

    [loadOp], TRValue { SSA = loadSSA; Type = elemType }

/// Witness NativePtr.write (pointer assignment)
let witnessNativePtrWrite
    (z: PSGZipper)
    (ptrSSA: SSA)
    (valueSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let storeOp = MLIROp.LLVMOp (LLVMOp.Store (valueSSA, ptrSSA, elemType, NotAtomic))

    [storeOp], TRVoid

/// Witness NativePtr.add (pointer arithmetic)
let witnessNativePtrAdd
    (z: PSGZipper)
    (ptrSSA: SSA)
    (offsetSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let newPtrSSA = freshSynthSSA z
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (newPtrSSA, ptrSSA, [(offsetSSA, MLIRTypes.i64)], elemType))

    [gepOp], TRValue { SSA = newPtrSSA; Type = TPtr }

/// Witness NativePtr.toNativeInt (pointer to integer)
let witnessNativePtrToInt
    (z: PSGZipper)
    (ptrSSA: SSA)
    : MLIROp list * TransferResult =

    let intSSA = freshSynthSSA z
    let castOp = MLIROp.LLVMOp (LLVMOp.PtrToInt (intSSA, ptrSSA, MLIRTypes.i64))

    [castOp], TRValue { SSA = intSSA; Type = MLIRTypes.i64 }

/// Witness NativePtr.ofNativeInt (integer to pointer)
let witnessNativePtrOfInt
    (z: PSGZipper)
    (intSSA: SSA)
    : MLIROp list * TransferResult =

    let ptrSSA = freshSynthSSA z
    let castOp = MLIROp.LLVMOp (LLVMOp.IntToPtr (ptrSSA, intSSA, TPtr))

    [castOp], TRValue { SSA = ptrSSA; Type = TPtr }

// ═══════════════════════════════════════════════════════════════════════════
// STACK ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativePtr.stackalloc<'T> n
let witnessStackAlloc
    (z: PSGZipper)
    (countSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ptrSSA = freshSynthSSA z
    let allocaOp = MLIROp.LLVMOp (LLVMOp.Alloca (ptrSSA, countSSA, elemType, None))

    [allocaOp], TRValue { SSA = ptrSSA; Type = TPtr }
