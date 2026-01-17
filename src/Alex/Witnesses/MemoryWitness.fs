/// Memory Witness - Witness memory and data structure operations to MLIR
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Witnesses OBSERVE and RETURN structured MLIROp lists.
/// They do NOT emit strings. ZERO SPRINTF for MLIR generation.
/// All SSAs come from pre-computed SSAAssignment coeffect.
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



/// Check if a type is the native string type
let private isNativeStrType (ty: MLIRType) : bool =
    ty = MLIRTypes.nativeStr

// ═══════════════════════════════════════════════════════════════════════════
// INDEX OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array/collection index get
/// Generates GEP + Load sequence
/// Uses 2 pre-assigned SSAs: gep[0], load[1]
let witnessIndexGet
    (nodeId: NodeId)
    (z: PSGZipper)
    (collSSA: SSA)
    (collType: MLIRType)
    (indexSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z

    // GEP to compute element address
    let ptrSSA = ssas.[0]
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, collSSA, [(indexSSA, MLIRTypes.i64)], elemType))

    // Load the element
    let loadSSA = ssas.[1]
    let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, ptrSSA, elemType, NotAtomic))

    [gepOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }

/// Witness array/collection index set
/// Generates GEP + Store sequence
/// Uses 1 pre-assigned SSA: gep[0]
let witnessIndexSet
    (nodeId: NodeId)
    (z: PSGZipper)
    (collSSA: SSA)
    (indexSSA: SSA)
    (valueSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z

    // GEP to compute element address
    let ptrSSA = ssas.[0]
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
/// Uses up to 2 pre-assigned SSAs: const[0], alloca[1]
let witnessAddressOf
    (nodeId: NodeId)
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
        let ssas = requireNodeSSAs nodeId z
        // Need to allocate stack space and store the value
        let oneSSA = ssas.[0]
        let allocaSSA = ssas.[1]

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
/// Uses 1 pre-assigned SSA: extract[0]
let witnessFieldGet
    (nodeId: NodeId)
    (z: PSGZipper)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldSSA = requireNodeSSA nodeId z
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, [fieldIndex], structType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness nested field get (struct.field1.field2.etc)
/// Uses extractvalue with multiple indices for nested access
/// Uses 1 pre-assigned SSA: extract[0]
let witnessNestedFieldGet
    (nodeId: NodeId)
    (z: PSGZipper)
    (structSSA: SSA)
    (structType: MLIRType)
    (indices: int list)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldSSA = requireNodeSSA nodeId z
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, indices, structType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness field set (struct.field <- value)
/// Uses insertvalue for value types (returns new struct)
/// Uses 1 pre-assigned SSA: insert[0]
let witnessFieldSet
    (nodeId: NodeId)
    (z: PSGZipper)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (valueSSA: SSA)
    : MLIROp list * TransferResult =

    let newStructSSA = requireNodeSSA nodeId z
    let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newStructSSA, structSSA, valueSSA, [fieldIndex], structType))

    [insertOp], TRValue { SSA = newStructSSA; Type = structType }

// ═══════════════════════════════════════════════════════════════════════════
// TUPLE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness tuple construction
/// Builds a struct by inserting each element
/// Uses N+1 pre-assigned SSAs: undef[0], insert[1..N]
let witnessTupleExpr
    (nodeId: NodeId)
    (z: PSGZipper)
    (elements: Val list)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z
    let elemTypes = elements |> List.map (fun v -> v.Type)
    let tupleType = TStruct elemTypes

    // Start with undef
    let undefSSA = ssas.[0]
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, tupleType))

    // Insert each element
    let insertOps, finalSSA =
        elements
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, elem) ->
            let newSSA = ssas.[idx + 1]  // SSAs 1..N are for inserts
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, elem.SSA, [idx], tupleType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = tupleType }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness record construction
/// Records are represented as structs with fields in declaration order
/// Uses N+1 pre-assigned SSAs: undef[0], insert[1..N]
let witnessRecordExpr
    (nodeId: NodeId)
    (z: PSGZipper)
    (fields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z

    // Start with undef
    let undefSSA = ssas.[0]
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, recordType))

    // Insert each field
    let insertOps, finalSSA =
        fields
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, (_, fieldVal)) ->
            let newSSA = ssas.[idx + 1]  // SSAs 1..N are for inserts
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, fieldVal.SSA, [idx], recordType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = recordType }

/// Witness record copy-and-update expression { orig with field1 = val1 }
/// Uses original record as base and updates only specified fields
/// Uses N pre-assigned SSAs for insertvalue operations (N = number of updated fields)
let witnessRecordCopyUpdate
    (nodeId: NodeId)
    (z: PSGZipper)
    (origVal: Val)
    (fieldDefs: (string * FSharp.Native.Compiler.Checking.Native.NativeTypes.NativeType) list)
    (updatedFields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    if List.isEmpty updatedFields then
        // No updates - just return the original
        [], TRValue origVal
    else
        let ssas = requireNodeSSAs nodeId z

        // Build map of field names to their indices
        let fieldIndexMap =
            fieldDefs
            |> List.mapi (fun idx (name, _) -> (name, idx))
            |> Map.ofList

        // Start with original record, insert each updated field at its correct index
        let insertOps, finalSSA =
            updatedFields
            |> List.indexed
            |> List.fold (fun (ops, prevSSA) (i, (fieldName, fieldVal)) ->
                let newSSA = ssas.[i]
                let fieldIdx = Map.find fieldName fieldIndexMap  // Look up correct index
                let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, fieldVal.SSA, [fieldIdx], recordType))
                (ops @ [insertOp], newSSA)
            ) ([], origVal.SSA)

        insertOps, TRValue { SSA = finalSSA; Type = recordType }

// ═══════════════════════════════════════════════════════════════════════════
// ARRAY CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array expression [| e1; e2; ... |]
/// Allocates array and stores each element
/// Uses 5 + 2*N pre-assigned SSAs: count[0], alloca[1], (idx,gep)*N, undef, withPtr, result
let witnessArrayExpr
    (nodeId: NodeId)
    (z: PSGZipper)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    let count = List.length elements
    let countSSA = nextSSA ()
    let allocaSSA = nextSSA ()

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
            let idxSSA = nextSSA ()
            let ptrSSA = nextSSA ()
            [
                MLIROp.ArithOp (ArithOp.ConstI (idxSSA, int64 idx, MLIRTypes.i64))
                MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, allocaSSA, [(idxSSA, MLIRTypes.i64)], elemType))
                MLIROp.LLVMOp (LLVMOp.Store (elem.SSA, ptrSSA, elemType, NotAtomic))
            ]
        )

    // Build fat array struct { ptr, count }
    let undefSSA = nextSSA ()
    let withPtrSSA = nextSSA ()
    let resultSSA = nextSSA ()

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
/// Uses same SSAs as witnessArrayExpr (delegates)
let witnessListExpr
    (nodeId: NodeId)
    (z: PSGZipper)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // Use same representation as arrays for now
    witnessArrayExpr nodeId z elements elemType

// ═══════════════════════════════════════════════════════════════════════════
// UNION CASE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness discriminated union case construction
/// DU layout: { tag, payload... } where types come from unionType struct
/// For 2-element DUs (option), insert payload at [1]
/// For multi-payload DUs (result), insert payload at [tag+1]
/// Uses 3-5 pre-assigned SSAs depending on payload conversion needs
let witnessUnionCase
    (nodeId: NodeId)
    (z: PSGZipper)
    (tag: int)
    (payload: Val option)
    (unionType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Extract tag type and determine payload info based on struct layout
    let tagType, payloadSlotType, payloadIndex =
        match unionType with
        | TStruct [t; p] ->
            // 2-element struct (like option): payload at [1]
            t, Some p, 1
        | TStruct (t :: rest) when List.length rest > 1 ->
            // Multi-payload struct (like result): payload at [tag+1]
            let idx = tag + 1
            if idx <= List.length rest then t, Some (List.item (idx - 1) rest), idx
            else t, None, 1
        | TStruct [t] -> t, None, 1
        | _ -> TInt I8, None, 1  // Fallback

    // Create discriminator constant with correct type
    let tagSSA = nextSSA ()
    let tagOp = MLIROp.ArithOp (ArithOp.ConstI (tagSSA, int64 tag, tagType))

    // Start with undef
    let undefSSA = nextSSA ()
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, unionType))

    // Insert tag at field 0
    let withTagSSA = nextSSA ()
    let insertTagOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withTagSSA, undefSSA, tagSSA, [0], unionType))

    match payload, payloadSlotType with
    | Some payloadVal, Some slotType ->
        // Check if payload needs conversion to match slot type
        let payloadSSA, conversionOps =
            if payloadVal.Type = slotType then
                payloadVal.SSA, []
            else
                // Need to convert payload to slot type
                let convertedSSA = nextSSA ()
                let convOp =
                    match payloadVal.Type, slotType with
                    // Integer widening (sign-extend)
                    | TInt _, TInt I64 ->
                        MLIROp.ArithOp (ArithOp.ExtSI (convertedSSA, payloadVal.SSA, payloadVal.Type, slotType))
                    // Float to i64 (bitcast for storage)
                    | TFloat F64, TInt I64 ->
                        MLIROp.LLVMOp (LLVMOp.Bitcast (convertedSSA, payloadVal.SSA, payloadVal.Type, slotType))
                    | TFloat F32, TInt I64 ->
                        // f32 → i32 bitcast, then i32 → i64 extend
                        let tmpSSA = nextSSA ()
                        // For now, just use the SSA directly - MLIR will handle
                        MLIROp.ArithOp (ArithOp.ExtSI (convertedSSA, payloadVal.SSA, TInt I32, slotType))
                    | _ ->
                        // Default: try sign-extend for integers
                        MLIROp.ArithOp (ArithOp.ExtSI (convertedSSA, payloadVal.SSA, payloadVal.Type, slotType))
                convertedSSA, [convOp]

        // Insert (possibly converted) payload at the computed index
        let resultSSA = nextSSA ()
        let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withTagSSA, payloadSSA, [payloadIndex], unionType))
        [tagOp; undefOp; insertTagOp] @ conversionOps @ [insertPayloadOp], TRValue { SSA = resultSSA; Type = unionType }

    | None, _ ->
        // No payload, just return with tag
        [tagOp; undefOp; insertTagOp], TRValue { SSA = withTagSSA; Type = unionType }

    | Some _, None ->
        // Payload provided but struct has no payload slot - error
        [tagOp; undefOp; insertTagOp], TRValue { SSA = withTagSSA; Type = unionType }

// ═══════════════════════════════════════════════════════════════════════════
// PAYLOAD EXTRACTION (Pattern Matching)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness payload extraction from a DU for pattern matching
/// DU layout: { tag, payload... } - extracts payload at caseIndex+1 and converts to pattern type
/// For 2-element DUs (option), caseIndex is ignored (always extract from [1])
/// For 3-element DUs (result), extract from [caseIndex+1]: Ok=>[1], Error=>[2]
/// Uses 1-2 pre-assigned SSAs: extract[0], convert[1] (if conversion needed)
///
/// FOUR PILLARS: This is the witness for pattern payload extraction.
/// Transfer calls this witness; Transfer does NOT emit ops directly.
let witnessPayloadExtract
    (ssas: SSA list)  // Pre-assigned SSAs for this extraction
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (patternType: MLIRType)
    (caseIndex: int)  // 0-based union case index (e.g., Ok=0, Error=1)
    : MLIROp list * Val =

    // Determine extraction index and slot type based on struct layout
    let extractIndex, slotType =
        match scrutineeType with
        | TStruct [_; p] ->
            // 2-element struct (like option): always extract from [1]
            1, p
        | TStruct fields when List.length fields > 2 ->
            // Multi-payload struct (like result): extract from [caseIndex+1]
            let idx = caseIndex + 1
            if idx < List.length fields then idx, fields.[idx]
            else 1, patternType  // Fallback
        | _ -> 1, patternType  // Fallback

    // Extract payload at the computed index
    let extractSSA = ssas.[0]
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (extractSSA, scrutineeSSA, [extractIndex], scrutineeType))

    // Convert extracted value to pattern type if different
    if slotType = patternType then
        [extractOp], { SSA = extractSSA; Type = patternType }
    else
        // Need conversion - use second pre-assigned SSA
        let convertSSA = ssas.[1]
        let convOp =
            match slotType, patternType with
            // i64 → i32 truncation (narrowing)
            | TInt I64, TInt I32 ->
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, extractSSA, slotType, patternType))
            // i64 → f64 bitcast (reinterpret stored bits as float)
            | TInt I64, TFloat F64 ->
                MLIROp.LLVMOp (LLVMOp.Bitcast (convertSSA, extractSSA, slotType, patternType))
            // i64 → f32 via bitcast (lower 32 bits)
            | TInt I64, TFloat F32 ->
                MLIROp.LLVMOp (LLVMOp.Bitcast (convertSSA, extractSSA, TInt I32, patternType))
            // Other narrowing cases
            | TInt _, TInt _ ->
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, extractSSA, slotType, patternType))
            | _ ->
                // Default: truncate (may be wrong for some type combos)
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, extractSSA, slotType, patternType))
        [extractOp; convOp], { SSA = convertSSA; Type = patternType }

/// Extract pattern binding from a TUPLE scrutinee
/// For `match a, b with | IntVal x, IntVal y -> ...`
/// Scrutinee is tuple of DUs: { DU1, DU2 }
/// Need to: 1) Extract tuple element at tupleIdx, 2) Extract DU payload, 3) Convert
/// Uses 2-3 pre-assigned SSAs: elemExtract[0], payloadExtract[1], convert[2] (if needed)
let witnessTuplePatternExtract
    (ssas: SSA list)  // Pre-assigned SSAs
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)  // The tuple type: struct<(DU1, DU2)>
    (tupleIdx: int)
    (patternType: MLIRType)  // The final pattern binding type (e.g., i32)
    : MLIROp list * Val =

    // Step 1: Extract tuple element (the DU) at tupleIdx
    let elemExtractSSA = ssas.[0]
    let duType =
        match scrutineeType with
        | TStruct fields when tupleIdx < List.length fields -> fields.[tupleIdx]
        | _ -> patternType  // Fallback
    let elemExtractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (elemExtractSSA, scrutineeSSA, [tupleIdx], scrutineeType))

    // Step 2: Extract payload from the DU (field 1)
    let payloadExtractSSA = ssas.[1]
    let payloadType =
        match duType with
        | TStruct [_; p] -> p  // DU layout: { tag, payload }
        | _ -> patternType  // Fallback
    let payloadExtractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (payloadExtractSSA, elemExtractSSA, [1], duType))

    // Step 3: Convert if needed
    if payloadType = patternType then
        [elemExtractOp; payloadExtractOp], { SSA = payloadExtractSSA; Type = patternType }
    else
        let convertSSA = ssas.[2]
        let convOp =
            match payloadType, patternType with
            | TInt I64, TInt I32 ->
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, payloadExtractSSA, payloadType, patternType))
            | TInt I64, TFloat F64 ->
                MLIROp.LLVMOp (LLVMOp.Bitcast (convertSSA, payloadExtractSSA, payloadType, patternType))
            | TInt _, TInt _ ->
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, payloadExtractSSA, payloadType, patternType))
            | _ ->
                MLIROp.ArithOp (ArithOp.TruncI (convertSSA, payloadExtractSSA, payloadType, patternType))
        [elemExtractOp; payloadExtractOp; convOp], { SSA = convertSSA; Type = patternType }

// ═══════════════════════════════════════════════════════════════════════════
// SRTP TRAIT CALLS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness SRTP trait call
/// The trait has been resolved by FNCS to a specific member
/// We generate the appropriate member access/call
/// Uses 1 pre-assigned SSA: result[0]
let witnessTraitCall
    (nodeId: NodeId)
    (z: PSGZipper)
    (receiverSSA: SSA)
    (receiverType: MLIRType)
    (memberName: string)
    (memberType: MLIRType)
    : MLIROp list * TransferResult =

    // For property-like traits (e.g., Length), generate field access
    // For method-like traits, this would need to generate a call
    // The specific implementation depends on the resolved member

    let resultSSA = requireNodeSSA nodeId z

    match memberName with
    | "Length" when isNativeStrType receiverType ->
        // String.Length - extract length field (index 1) from fat pointer
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 }

    | "Length" ->
        // Array/collection Length - extract count field (index 1) from fat pointer
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 }

    | "Pointer" when isNativeStrType receiverType ->
        // String.Pointer - extract pointer field (index 0) from fat pointer
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TPtr }

    | _ ->
        // Generic field access - assume field index 0 for unknown traits
        // This is a fallback and should be refined based on type information
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = memberType }

