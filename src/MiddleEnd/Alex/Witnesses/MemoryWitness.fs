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

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.TypeSizing

module SSAAssignment = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS (WitnessContext accessors)
// ═══════════════════════════════════════════════════════════════════════════

/// Get all pre-assigned SSAs for a node
let private requireNodeSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssignment.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get result SSA for a node (the final SSA from its allocation)
let private requireNodeSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssignment.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No result SSA for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════



/// Check if a type is the native string type (fat pointer with ptr + integer length)
/// Matches any word width (I32 on 32-bit, I64 on 64-bit platforms)
let private isNativeStrType (ty: MLIRType) : bool =
    match ty with
    | TStruct [TPtr; TInt _] -> true  // Fat pointer: {ptr, len} with any integer width
    | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// INDEX OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array/collection index get
/// Generates GEP + Load sequence
/// Uses 2 pre-assigned SSAs: gep[0], load[1]
let witnessIndexGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (collSSA: SSA)
    (collType: MLIRType)
    (indexSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

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
    (ctx: WitnessContext)
    (collSSA: SSA)
    (indexSSA: SSA)
    (valueSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

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
    (ctx: WitnessContext)
    (exprSSA: SSA)
    (exprType: MLIRType)
    (isMutable: bool)
    : MLIROp list * TransferResult =

    if isMutable then
        // Mutable binding already has an address (alloca)
        // The SSA is the pointer to the mutable slot
        [], TRValue { SSA = exprSSA; Type = TPtr }
    else
        let ssas = requireNodeSSAs nodeId ctx
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

/// Resolve field name to struct index based on type
/// For records, uses SemanticGraph.tryGetRecordFields to look up field order
let private resolveFieldIndex (structNativeType: NativeType) (fieldName: string) (ctx: WitnessContext) : int =
    match structNativeType with
    | NativeType.TApp (tc, _) ->
        match tc.Name with
        | "string" ->
            // String layout: (Pointer, Length)
            match fieldName with
            | "Pointer" -> 0
            | "Length" -> 1
            | _ -> failwithf "Unknown string field: %s" fieldName
        | _ ->
            // DU layout: (Tag, Item1, Item2, ...)
            // Record layout: fields in definition order
            match fieldName with
            | "Tag" -> 0
            | name when name.StartsWith("Item") ->
                match System.Int32.TryParse(name.Substring(4)) with
                | true, n -> n  // Item1 -> 1, Item2 -> 2
                | false, _ -> failwithf "Invalid DU field name: %s" name
            | _ ->
                // Record field - look up from TypeDef in SemanticGraph
                // TypeDefs use simple names (not module-qualified) per FNCS architecture
                match SemanticGraph.tryGetRecordFields tc.Name ctx.Graph with
                | Some fields ->
                    // Find field index by name
                    match fields |> List.tryFindIndex (fun (name, _) -> name = fieldName) with
                    | Some idx -> idx
                    | None -> failwithf "Record type '%s' has no field '%s'. Available fields: %A" tc.Name fieldName (fields |> List.map fst)
                | None ->
                    failwithf "Type '%s' not found in SemanticGraph or is not a record type" tc.Name
    | _ -> failwithf "FieldGet on non-TApp type: %A" structNativeType

/// Witness field get (struct.field or record.field)
/// Resolves field name to index based on type, then extracts
///
/// ARCHITECTURAL NOTE: This is TRANSLITERATION.
/// For DU payload extraction (Item1, Item2, etc.), we construct a case-specific
/// struct type using the KNOWN fieldType. LLVM doesn't see "a DU" as a single
/// construct - each match branch interprets memory with its own type.
/// In FloatVal branch: struct is (i8, f64). In IntVal branch: struct is (i8, i32).
/// The fieldType comes from PSG (baked in at Baker level) and is authoritative.
///
/// Uses 1 pre-assigned SSA: extract[0]
let witnessFieldGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structNativeType: NativeType)
    (fieldName: string)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldIndex = resolveFieldIndex structNativeType fieldName ctx

    // For DU payload extraction, construct case-specific struct type
    // using the KNOWN fieldType from PSG. This is transliteration - we state
    // the correct type for this specific branch.
    let structMlirType =
        match fieldName with
        | name when name.StartsWith("Item") ->
            // DU layout: (tag, payload) with case-specific payload type
            TStruct [TInt I8; fieldType]
        | _ ->
            // Tag extraction, string fields, records - use graph-aware mapping
            // that correctly handles nested record types
            mapType structNativeType ctx

    let fieldSSA = requireNodeSSA nodeId ctx
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, [fieldIndex], structMlirType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness nested field get (struct.field1.field2.etc)
/// Uses extractvalue with multiple indices for nested access
/// Uses 1 pre-assigned SSA: extract[0]
let witnessNestedFieldGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structType: MLIRType)
    (indices: int list)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldSSA = requireNodeSSA nodeId ctx
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, indices, structType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness field set (struct.field <- value)
/// Uses insertvalue for value types (returns new struct)
/// Uses 1 pre-assigned SSA: insert[0]
let witnessFieldSet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (valueSSA: SSA)
    : MLIROp list * TransferResult =

    let newStructSSA = requireNodeSSA nodeId ctx
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
    (ctx: WitnessContext)
    (elements: Val list)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
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
    (ctx: WitnessContext)
    (fields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

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
    (ctx: WitnessContext)
    (origVal: Val)
    (fieldDefs: (string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType) list)
    (updatedFields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    if List.isEmpty updatedFields then
        // No updates - just return the original
        [], TRValue origVal
    else
        let ssas = requireNodeSSAs nodeId ctx

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
    (ctx: WitnessContext)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
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
    (ctx: WitnessContext)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // Use same representation as arrays for now
    witnessArrayExpr nodeId ctx elements elemType

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
    (ctx: WitnessContext)
    (tag: int)
    (payload: Val option)
    (unionType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
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
    (ctx: WitnessContext)
    (receiverSSA: SSA)
    (receiverType: MLIRType)
    (memberName: string)
    (memberType: MLIRType)
    : MLIROp list * TransferResult =

    // For property-like traits (e.g., Length), generate field access
    // For method-like traits, this would need to generate a call
    // The specific implementation depends on the resolved member

    let resultSSA = requireNodeSSA nodeId ctx

    match memberName with
    | "Length" when isNativeStrType receiverType ->
        // String.Length - extract length field (index 1) from fat pointer
        // Length is platform word sized (i64 on 64-bit, i32 on 32-bit)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TInt (wordWidth ctx) }

    | "Length" ->
        // Array/collection Length - extract count field (index 1) from fat pointer
        // Length is platform word sized (i64 on 64-bit, i32 on 32-bit)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TInt (wordWidth ctx) }

    | "Pointer" when isNativeStrType receiverType ->
        // String.Pointer - extract pointer field (index 0) from fat pointer
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TPtr }

    | _ ->
        // Generic field access - assume field index 0 for unknown traits
        // This is a fallback and should be refined based on type information
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = memberType }

// ═══════════════════════════════════════════════════════════════════════════
// DISCRIMINATED UNION OPERATIONS (January 2026)
//
// DU Architecture: Pointer-based with case eliminators
// - DUs are represented as pointers to region-allocated storage
// - Each case has its own typed eliminator for payload extraction
// - Pointer bitcast allows case-specific struct types (transliteration)
//
// For now (inline struct phase), DUs use (tag, payload) inline structs.
// The case-specific typing ensures extractvalue uses correct payload type.
// ═══════════════════════════════════════════════════════════════════════════

/// Witness DU tag extraction
/// Extracts the tag (discriminator) from a DU value
/// Tag is at field index 0, stored as i8 (or i16 for >256 cases)
/// Uses 1 pre-assigned SSA: extract[0]
let witnessDUGetTag
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (duSSA: SSA)
    (duType: NativeType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
    let tagType = MLIRTypes.i8

    // Map the DU type to get the MLIR type (graph-aware for consistency)
    let duMlirType = mapType duType ctx

    // Check if this is a pointer-based DU (heterogeneous like Result)
    match duMlirType with
    | TPtr ->
        // Pointer-based DU: need to load the struct first, then extract tag
        // We need the case struct type for the load - use minimal {i8} since we only need tag
        // Actually, we need to load the full struct to get proper alignment
        // For tag extraction, we can load as {i8, ...} but simplest is to load i8 directly
        // Since tag is always at offset 0, we can load just the tag byte
        let loadSSA = ssas.[0]
        let tagSSA = ssas.[1]
        
        // Load the tag byte directly from the pointer (tag is at offset 0)
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, duSSA, tagType, NotAtomic))
        
        [loadOp], TRValue { SSA = loadSSA; Type = tagType }

    | _ ->
        // Inline struct DU: extract tag directly from index 0
        let tagSSA = ssas.[0]
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, duSSA, [0], duMlirType))
        [extractOp], TRValue { SSA = tagSSA; Type = tagType }

/// Witness DU payload elimination (type-safe extraction)
/// This is the CASE ELIMINATOR pattern:
/// 1. Use the ACTUAL DU struct type for extractvalue (all cases share same layout)
/// 2. If the extracted slot type differs from desired payload type, bitcast
///
/// For example, DU Number = IntVal of int | FloatVal of float
/// - Runtime struct: (i8 tag, i64 payload) - i64 holds both int and float bits
/// - IntVal extraction: extractvalue[1] gives i64, use directly
/// - FloatVal extraction: extractvalue[1] gives i64, bitcast to f64
///
/// Uses 1-2 pre-assigned SSAs: extract[0], optional bitcast[1]
let witnessDUEliminate
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (duSSA: SSA)
    (duType: MLIRType)
    (caseIndex: int)
    (caseName: string)
    (payloadType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

    // Check if this is a pointer-based DU (heterogeneous like Result)
    match duType with
    | TPtr ->
        // Pointer-based DU: load the case-specific struct, then extract payload
        // Case struct type is {i8 tag, PayloadType}
        let caseStructType = TStruct [TInt I8; payloadType]
        
        let loadSSA = ssas.[0]
        let extractSSA = ssas.[1]
        
        // Load the case-specific struct from the pointer
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, duSSA, caseStructType, NotAtomic))
        
        // Extract payload from index 1 (index 0 is tag)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (extractSSA, loadSSA, [1], caseStructType))
        
        [loadOp; extractOp], TRValue { SSA = extractSSA; Type = payloadType }

    | _ ->
        // Inline struct DU: extract directly
        let extractSSA = ssas.[0]

        // Get the actual slot type from the DU struct
        let slotType =
            match duType with
            | TStruct [_tagTy; payloadSlotTy] -> payloadSlotTy
            | _ -> payloadType

        // Extract using the ACTUAL DU type
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (extractSSA, duSSA, [1], duType))

        // Check if we need to bitcast (e.g., i64 -> f64 for FloatVal)
        if slotType = payloadType then
            [extractOp], TRValue { SSA = extractSSA; Type = payloadType }
        else
            let bitcastSSA = ssas.[1]
            let bitcastOp = MLIROp.LLVMOp (LLVMOp.Bitcast (bitcastSSA, extractSSA, slotType, payloadType))
            [extractOp; bitcastOp], TRValue { SSA = bitcastSSA; Type = payloadType }

/// Witness DU construction - creates a DU value with the given tag and optional payload
/// SSA layout: Nullary = 3, With payload = 4
let witnessDUConstruct
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (caseName: string)
    (caseIndex: int)
    (payloadOpt: Val option)
    (duType: MLIRType)
    : MLIROp list * TransferResult =

    // Check for DULayout coeffect (heterogeneous DUs needing arena allocation)
    match SSAAssignment.lookupDULayout nodeId ctx.Coeffects.SSA with
    | Some layout ->
        // Arena allocation path for heterogeneous DUs (e.g., Result<'T, 'E>)
        // Uses pre-computed SSAs from DULayout (coeffect pattern)

        // 1. Build case-specific struct: {i8 tag, PayloadType}
        let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (layout.StructUndefSSA, layout.CaseStructType))
        let tagOp = MLIROp.ArithOp (ArithOp.ConstI (layout.TagConstSSA, int64 caseIndex, MLIRTypes.i8))
        let withTagOp = MLIROp.LLVMOp (LLVMOp.InsertValue (layout.WithTagSSA, layout.StructUndefSSA, layout.TagConstSSA, [0], layout.CaseStructType))

        let structOps, finalStructSSA =
            match payloadOpt, layout.WithPayloadSSA with
            | Some payload, Some withPayloadSSA ->
                let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withPayloadSSA, layout.WithTagSSA, payload.SSA, [1], layout.CaseStructType))
                [undefOp; tagOp; withTagOp; insertPayloadOp], withPayloadSSA
            | None, None ->
                [undefOp; tagOp; withTagOp], layout.WithTagSSA
            | _ ->
                failwithf "DULayout payload mismatch for case %s" caseName

        // 2. Compute size from layout.CaseStructType (already determined by FNCS/PSGElaboration)
        let typeString = Serialize.typeToString layout.CaseStructType
        let sizeBytes = TypeSizing.computeSize typeString
        let sizeOps = [
            MLIROp.ArithOp (ArithOp.ConstI (layout.SizeSSA, sizeBytes, MLIRTypes.i64))
        ]

        // 3. Allocate from closure_heap arena (bump allocation)
        let allocOps = [
            MLIROp.LLVMOp (LLVMOp.AddressOf (layout.HeapPosPtrSSA, GFunc "closure_pos"))
            MLIROp.LLVMOp (LLVMOp.Load (layout.HeapPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
            MLIROp.LLVMOp (LLVMOp.AddressOf (layout.HeapBaseSSA, GFunc "closure_heap"))
            MLIROp.LLVMOp (LLVMOp.GEP (layout.HeapResultPtrSSA, layout.HeapBaseSSA, [(layout.HeapPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
            MLIROp.ArithOp (ArithOp.AddI (layout.HeapNewPosSSA, layout.HeapPosSSA, layout.SizeSSA, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Store (layout.HeapNewPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
        ]

        // 4. Store struct to arena
        let storeOp = MLIROp.LLVMOp (LLVMOp.Store (finalStructSSA, layout.HeapResultPtrSSA, layout.CaseStructType, NotAtomic))

        // Result is the pointer to arena-allocated case struct
        let allOps = structOps @ sizeOps @ allocOps @ [storeOp]
        allOps, TRValue { SSA = layout.HeapResultPtrSSA; Type = TPtr }

    | None ->
        // Inline path for homogeneous DUs (e.g., Option<'T>)
        // Direct struct representation - no arena allocation needed
        let ssas = requireNodeSSAs nodeId ctx
        let mutable ssaIdx = 0
        let nextSSA () =
            let ssa = ssas.[ssaIdx]
            ssaIdx <- ssaIdx + 1
            ssa

        // Start with undef
        let undefSSA = nextSSA ()
        let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, duType))

        // Create tag constant and insert at index 0
        let tagConstSSA = nextSSA ()
        let withTagSSA = nextSSA ()
        let tagOps = [
            MLIROp.ArithOp (ArithOp.ConstI (tagConstSSA, int64 caseIndex, MLIRTypes.i8))
            MLIROp.LLVMOp (LLVMOp.InsertValue (withTagSSA, undefSSA, tagConstSSA, [0], duType))
        ]

        // Insert payload if present
        match payloadOpt with
        | Some payload ->
            let resultSSA = nextSSA ()
            let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withTagSSA, payload.SSA, [1], duType))
            [undefOp] @ tagOps @ [insertPayloadOp], TRValue { SSA = resultSSA; Type = duType }
        | None ->
            // Nullary case - just tag, no payload
            [undefOp] @ tagOps, TRValue { SSA = withTagSSA; Type = duType }

