/// MemoryPatterns - Memory operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns to elide memory operations to MLIR.
/// Patterns compose Elements (internal) into semantic memory operations.
module Alex.Patterns.MemoryPatterns

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId - MUST be before TransferTypes
open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.IndexElements
open Alex.Elements.FuncElements
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════
// FIELD EXTRACTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract single field from struct
/// SSA layout (2 total):
///   [0] = offsetConstSSA - index constant for memref.load
///   [1] = resultSSA - result of the load
let pExtractField (ssas: SSA list) (structSSA: SSA) (fieldIndex: int) (structTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2) $"pExtractField: Expected 2 SSAs, got {ssas.Length}"
        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]
        return! pExtractValue resultSSA structSSA fieldIndex offsetSSA structTy
    }

// ═══════════════════════════════════════════════════════════
// FIELD ACCESS PATTERNS (Byte-Offset)
// ═══════════════════════════════════════════════════════════

/// Field access via byte-offset memref operations
/// structType: The NativeType of the struct (for calculating field offset)
let pFieldAccess (structPtr: SSA) (structType: NativeType) (fieldIndex: int) (gepSSA: SSA) (loadSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch

        // Calculate byte offset for the field using FNCS-provided type structure
        let fieldOffset = calculateFieldOffsetForArch arch structType fieldIndex

        // Emit offset constant using SSA observed from coeffects via witness
        let! offsetOp = pConstI gepSSA (int64 fieldOffset) TIndex

        // Memref.load with byte offset
        // Note: This assumes structPtr is memref<Nxi8> and we load at byte offset
        let! loadOp = Alex.Elements.MemRefElements.pLoad loadSSA structPtr [gepSSA]

        return ([offsetOp; loadOp])
    }

/// Field set via byte-offset memref operations
/// structType: The NativeType of the struct (for calculating field offset)
let pFieldSet (structPtr: SSA) (structType: NativeType) (fieldIndex: int) (value: SSA) (gepSSA: SSA) (_indexSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let elemType = mapNativeTypeForArch arch state.Current.Type

        // Calculate byte offset for the field using FNCS-provided type structure
        let fieldOffset = calculateFieldOffsetForArch arch structType fieldIndex

        // Emit offset constant using SSA observed from coeffects via witness
        let! offsetOp = pConstI gepSSA (int64 fieldOffset) TIndex

        // Memref.store with byte offset
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore value structPtr [gepSSA] elemType memrefType

        return ([offsetOp; storeOp])
    }

// ═══════════════════════════════════════════════════════════
// ALLOCATION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Address-of for immutable values: const 1, allocate, store, return pointer
/// SSAs: [0] = const 1, [1] = alloca result
let pAllocaImmutable (valueSSA: SSA) (valueType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 3) $"pAllocaImmutable: Expected 3 SSAs, got {ssas.Length}"

        let constOneSSA = ssas.[0]
        let allocaSSA = ssas.[1]
        let indexSSA = ssas.[2]

        let constOneTy = TInt I64
        let! constOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca allocaSSA 1 valueType None
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, valueType)
        let! storeOp = pStore valueSSA allocaSSA [indexSSA] valueType memrefType

        return ([constOp; allocaOp; indexOp; storeOp])
    }

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Type conversion dispatcher - chooses appropriate conversion Element
let pConvertType (srcSSA: SSA) (srcType: MLIRType) (dstType: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        if srcType = dstType then
            // No conversion needed
            return []
        else
            let! convOp =
                match srcType, dstType with
                // Integer widening (sign-extend)
                | TInt srcWidth, TInt dstWidth when srcWidth < dstWidth ->
                    pExtSI resultSSA srcSSA srcType dstType
                // Integer narrowing (truncate)
                | TInt _, TInt _ ->
                    pTruncI resultSSA srcSSA srcType dstType
                // Float to int
                | TFloat _, TInt _ ->
                    pFPToSI resultSSA srcSSA srcType dstType
                // Int to float
                | TInt _, TFloat _ ->
                    pSIToFP resultSSA srcSSA srcType dstType
                // Unsupported conversion (bitcast removed - no portable memref equivalent)
                | _, _ ->
                    fail (Message $"Unsupported type conversion: {srcType} -> {dstType}")
            return ([convOp])
    }

// ═══════════════════════════════════════════════════════════
// DU PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract DU tag (handles both inline and pointer-based DUs)
/// Pointer-based: Load tag byte from offset 0
/// Inline: ExtractValue at index 0
/// SSAs extracted from coeffects via nodeId: [0] = indexZeroSSA, [1] = tagSSA (result)
let pExtractDUTag (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pExtractDUTag: Expected 2 SSAs, got {ssas.Length}"
        let indexZeroSSA = ssas.[0]
        let tagSSA = ssas.[1]
        let tagTy = TInt I8  // DU tags are always i8

        match duType with
        | TIndex ->
            // Pointer-based DU: load tag byte from offset 0
            let! indexZeroOp = pConstI indexZeroSSA 0L TIndex
            let! loadOp = pLoad tagSSA duSSA [indexZeroSSA]
            return ([indexZeroOp; loadOp], TRValue { SSA = tagSSA; Type = tagTy })
        | _ ->
            // Inline struct DU: extract tag from field 0 (pExtractValue creates constant internally)
            let! ops = pExtractValue tagSSA duSSA 0 indexZeroSSA tagTy
            return (ops, TRValue { SSA = tagSSA; Type = tagTy })
    }

/// Extract DU payload with optional type conversion
/// SSAs extracted from coeffects via nodeId: [0] = offsetSSA, [1] = extractSSA, [2] = convertSSA (if needed)
let pExtractDUPayload (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (payloadIndex: int) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pExtractDUPayload: Expected at least 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let extractSSA = ssas.[1]

        // Determine slot type from DU struct
        // Note: With TMemRefStatic, we use payloadType directly
        let slotType = payloadType

        // Extract payload (pExtractValue creates constant internally)
        let! extractOps = pExtractValue extractSSA duSSA payloadIndex offsetSSA slotType

        // Convert if needed
        if slotType = payloadType then
            return (extractOps, TRValue { SSA = extractSSA; Type = payloadType })
        else
            do! ensure (ssas.Length >= 3) $"pExtractDUPayload: Need 3 SSAs for conversion, got {ssas.Length}"
            let convertSSA = ssas.[2]
            let! convOps = pConvertType extractSSA slotType payloadType convertSSA
            return (extractOps @ convOps, TRValue { SSA = convertSSA; Type = payloadType })
    }

// ═══════════════════════════════════════════════════════════
// RECORD PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record copy-and-update: start with original, insert updated fields
/// SSAs: one per updated field
/// Updates: (fieldIndex, valueSSA) pairs
let pRecordCopyWith (origSSA: SSA) (recordType: MLIRType) (updates: (int * SSA) list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        // Each update needs 2 SSAs: offsetSSA and targetSSA
        do! ensure (ssas.Length = 2 * updates.Length) $"pRecordCopyWith: Expected {2 * updates.Length} SSAs (2 per update), got {ssas.Length}"

        // Fold over updates, threading prevSSA through
        let! result =
            updates
            |> List.mapi (fun i (fieldIdx, valueSSA) ->
                let offsetSSA = ssas.[2*i]
                let targetSSA = ssas.[2*i + 1]
                (offsetSSA, targetSSA, fieldIdx, valueSSA))
            |> List.fold (fun accParser (offsetSSA, targetSSA, fieldIdx, valueSSA) ->
                parser {
                    let! (prevOps, prevSSA) = accParser
                    let! insertOps = pInsertValue targetSSA prevSSA valueSSA fieldIdx offsetSSA recordType
                    return (prevOps @ insertOps, targetSSA)
                }
            ) (preturn ([], origSSA))

        let (ops, _) = result
        return ops
    }

// ═══════════════════════════════════════════════════════════
// ARRAY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build array: allocate, initialize elements, construct fat pointer
/// Array element access via SubView + Load
/// SSAs: gepSSA for subview, loadSSA for result, indexZeroSSA for memref index
let pArrayAccess (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (gepSSA: SSA) (loadSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // MLIR memrefs require indices
        let! loadOp = pLoad loadSSA gepSSA [indexZeroSSA]
        return ([subViewOp; indexZeroOp; loadOp])
    }

/// Array element set via SubView + Store
let pArraySet (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (value: SSA) (gepSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let elemType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore value gepSSA [indexZeroSSA] elemType memrefType
        return ([subViewOp; indexZeroOp; storeOp])
    }

// ═══════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS (FNCS Intrinsics)
// ═══════════════════════════════════════════════════════════

/// Build NativePtr.stackalloc pattern
/// Allocates memory on the stack and returns a pointer
///
/// NativePtr.stackalloc<'T>(count) : nativeptr<'T>
/// SSA extracted from coeffects via nodeId: [0] = result
let pNativePtrStackAlloc (nodeId: NodeId) (count: int) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pNativePtrStackAlloc: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        let! state = getUserState

        // Extract element type from nativeptr<'T> in state.Current.Type
        let elemType =
            match state.Current.Type with
            | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "nativeptr" ->
                // Map inner type to concrete MLIR type via platform
                mapNativeTypeForArch state.Platform.TargetArch innerTy
            | _ ->
                // Fallback to byte storage if type extraction fails
                TInt I8

        // Use the provided count for memref allocation
        let! allocaOp = pAlloca resultSSA count elemType None
        let memrefTy = TMemRefStatic (count, elemType)

        // Return memref type (not TIndex) - conversion to pointer happens at FFI boundary
        return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
    }

/// Build NativePtr.write pattern
/// Writes a value to a pointer location
///
/// NativePtr.write (ptr: nativeptr<'T>) (value: 'T) : unit
let pNativePtrWrite (valueSSA: SSA) (ptrSSA: SSA) (elemType: MLIRType) (indexSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit memref.store operation with index
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore valueSSA ptrSSA [indexSSA] elemType memrefType

        return ([indexOp; storeOp], TRVoid)
    }

/// Build NativePtr.read pattern
/// Reads a value from a pointer location
///
/// NativePtr.read (ptr: nativeptr<'T>) : 'T
/// SSA extracted from coeffects via nodeId: [0] = indexZeroSSA, [1] = resultSSA
let pNativePtrRead (nodeId: NodeId) (ptrSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pNativePtrRead: Expected 2 SSAs, got {ssas.Length}"
        let indexZeroSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // Get result type from XParsec state (type of value being loaded)
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let resultType = mapNativeTypeForArch arch state.Current.Type

        // Emit index constant for memref.load (always load from index 0 for scalar pointer read)
        let! constOp = pConstI indexZeroSSA 0L TIndex

        // Emit memref.load operation with index
        let! loadOp = pLoad resultSSA ptrSSA [indexZeroSSA]

        return ([constOp; loadOp], TRValue { SSA = resultSSA; Type = resultType })
    }

// ═══════════════════════════════════════════════════════════
// ARENA OPERATIONS (F-02: Arena Allocation)
// ═══════════════════════════════════════════════════════════

/// Build Arena.create pattern
/// Allocates an arena buffer on the stack
///
/// Arena.create<'lifetime>(sizeBytes: int) : Arena<'lifetime>
/// Returns: memref<sizeBytes x i8> (stack-allocated byte buffer)
/// SSA extracted from coeffects via nodeId: [0] = result
let pArenaCreate (nodeId: NodeId) (sizeBytes: int) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pArenaCreate: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Allocate arena memory block on stack as byte array
        // Arena IS the memref - no separate control struct in memref semantics
        let elemType = TInt I8
        let! allocaOp = pAlloca resultSSA sizeBytes elemType None
        let memrefTy = TMemRefStatic (sizeBytes, elemType)

        // Return the arena memref (byte buffer)
        return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
    }

/// Build Arena.alloc pattern
/// Allocates memory from an arena
///
/// Arena.alloc(arena: Arena<'lifetime> byref, sizeBytes: int) : nativeint
/// For now: returns the arena memref itself (simplified - proper bump allocation later)
/// TODO: Implement proper bump-pointer allocation with memref.subview and offset tracking
/// SSA extracted from coeffects via nodeId: [0] = result
let pArenaAlloc (nodeId: NodeId) (arenaSSA: SSA) (sizeSSA: SSA) (arenaType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pArenaAlloc: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Simplified implementation: return arena memref as the allocated pointer
        // The memref IS the allocation - caller can use memref.store directly
        // Future: Add offset tracking and memref.subview for true bump allocation

        // For now, just return the arena memref unchanged
        // This works for single allocation per arena (like String.concat2)
        return ([], TRValue { SSA = resultSSA; Type = arenaType })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT FIELD ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract field from struct (e.g., string.Pointer, string.Length)
/// SSA layout (max 3): [0] = intermediate (index or dim const), [1] = intermediate2 (dim result), [2] = result
let pStructFieldGet (nodeId: NodeId) (structSSA: SSA) (fieldName: string) (structTy: MLIRType) (fieldTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 3) $"pStructFieldGet: Expected 3 SSAs, got {ssas.Length}"
        let resultSSA = List.last ssas

        // Check if structTy is a memref (strings are now memref<?xi8>)
        match structTy with
        | TMemRef _ | TMemRefScalar _ ->
            // String as memref - use memref operations
            match fieldName with
            | "Pointer" | "ptr" ->  // Accept both capitalized (old) and lowercase (FNCS)
                // Extract base pointer from memref descriptor as index, then cast to target type
                let! state = getUserState
                let targetTy =
                    match fieldTy with
                    | TIndex -> state.Platform.PlatformWordType  // TIndex → i64/i32 (portable!)
                    | ty -> ty

                match targetTy with
                | TIndex ->
                    // No cast needed - result is index
                    let! extractOp = pExtractBasePtr resultSSA structSSA structTy
                    return ([extractOp], TRValue { SSA = resultSSA; Type = targetTy })
                | _ ->
                    // Cast index → targetTy (e.g., index → i64 for x86-64, index → i32 for ARM32)
                    let indexSSA = ssas.[0]  // Intermediate index from coeffects
                    let! extractOp = pExtractBasePtr indexSSA structSSA structTy
                    let! castOp = pIndexCastS resultSSA indexSSA targetTy
                    return ([extractOp; castOp], TRValue { SSA = resultSSA; Type = targetTy })
            | "Length" | "len" ->  // Accept both capitalized (old) and lowercase (FNCS)
                // Extract length using memref.dim (returns index type)
                let dimIndexSSA = ssas.[0]  // Dim constant (0) from coeffects
                let! constOp = pConstI dimIndexSSA 0L TIndex

                // Check if we need to cast index → fieldTy (for FFI boundaries)
                match fieldTy with
                | TIndex ->
                    // No cast needed - result is index
                    let! dimOp = pMemRefDim resultSSA structSSA dimIndexSSA structTy
                    return ([constOp; dimOp], TRValue { SSA = resultSSA; Type = fieldTy })
                | _ ->
                    // Cast index → fieldTy (e.g., index → i64 for x86-64 syscall, index → i32 for ARM32)
                    let dimResultSSA = ssas.[1]  // Dim result from coeffects
                    let! dimOp = pMemRefDim dimResultSSA structSSA dimIndexSSA structTy
                    let! castOp = pIndexCastS resultSSA dimResultSSA fieldTy
                    return ([constOp; dimOp; castOp], TRValue { SSA = resultSSA; Type = fieldTy })
            | _ ->
                return failwith $"Unknown memref field name: {fieldName}"
        | _ ->
            // LLVM struct - use extractvalue (for closures, option, etc.)
            let fieldIndex =
                match fieldName with
                | "Pointer" | "ptr" -> 0  // Accept both capitalized (old) and lowercase (FNCS)
                | "Length" | "len" -> 1  // Accept both capitalized (old) and lowercase (FNCS)
                | _ -> failwith $"Unknown field name: {fieldName}"

            // Extract field value - pExtractField needs [offsetSSA, resultSSA]
            let extractFieldSSAs = [ssas.[0]; resultSSA]
            let! ops = pExtractField extractFieldSSAs structSSA fieldIndex structTy
            return (ops, TRValue { SSA = resultSSA; Type = fieldTy })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT CONSTRUCTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record struct via Undef + InsertValue chain
/// SSA layout: [0] = undefSSA, then for each field: [2*i+1] = offsetSSA, [2*i+2] = resultSSA
let pRecordStruct (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = 1 + 2 * fields.Length) $"pRecordStruct: Expected {1 + 2 * fields.Length} SSAs, got {ssas.Length}"

        // Compute struct type from field types
        let fieldTypes = fields |> List.map (fun f -> f.Type)
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let structTy = TMemRefStatic(totalBytes, TInt I8)
        let! undefOp = pUndef ssas.[0] structTy

        let! insertOpLists =
            fields
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = ssas.[2*i + 1]
                    let targetSSA = ssas.[2*i + 2]
                    let sourceSSA = if i = 0 then ssas.[0] else ssas.[2*(i-1) + 2]
                    return! pInsertValue targetSSA sourceSSA field.SSA i offsetSSA structTy
                })
            |> sequence

        let insertOps = List.concat insertOpLists
        return undefOp :: insertOps
    }

/// Tuple struct via Undef + InsertValue chain (same as record, but semantically different)
let pTupleStruct (elements: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    pRecordStruct elements ssas  // Same implementation, different semantic context

// ═══════════════════════════════════════════════════════════
// DU CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// DU case construction: tag field (index 0) + payload fields
/// CRITICAL: This is the foundation for all collection patterns (Option, List, Map, Set, Result)
/// SSA layout: [0] = undefSSA, [1] = tagSSA, [2] = tagOffsetSSA, [3] = tagResultSSA,
///             then for each payload: [4+2*i] = offsetSSA, [5+2*i] = resultSSA
let pDUCase (nodeId: NodeId) (tag: int64) (payload: Val list) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4 + 2 * payload.Length) $"pDUCase: Expected at least {4 + 2 * payload.Length} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0] ty

        // Insert tag at index 0 (pInsertValue creates constant internally)
        let tagTy = TInt I8  // DU tags are always i8
        let! tagConstOp = pConstI ssas.[1] tag tagTy
        let! insertTagOps = pInsertValue ssas.[3] ssas.[0] ssas.[1] 0 ssas.[2] ty

        // Insert payload fields starting at index 1
        let! payloadOpLists =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = ssas.[4 + 2*i]
                    let targetSSA = ssas.[5 + 2*i]
                    let sourceSSA = if i = 0 then ssas.[3] else ssas.[3 + 2*i]
                    return! pInsertValue targetSSA sourceSSA field.SSA (i + 1) offsetSSA ty
                })
            |> sequence

        let payloadOps = List.concat payloadOpLists
        let resultSSA = List.last ssas
        return (undefOp :: tagConstOp :: (insertTagOps @ payloadOps), TRValue { SSA = resultSSA; Type = ty })
    }

// ═══════════════════════════════════════════════════════════
// SIMPLE MEMORY STORE
// ═══════════════════════════════════════════════════════════

/// Simple memref.store with no indices (scalar store)
/// Store value to scalar memref (requires index even for 1-element memrefs)
/// Allocates 1 SSA for the index constant
let pMemRefStore (indexSSA: SSA) (value: SSA) (memref: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for scalar/1-element memref
        let memrefType = TMemRefStatic (1, elemType)  // 1-element memref for scalar stores
        let! storeOp = pStore value memref [indexSSA] elemType memrefType
        return ([indexOp; storeOp], TRVoid)
    }

/// Indexed memref.store (for NativePtr.write with NativePtr.add)
/// Store value to memref at computed index
/// Handles: NativePtr.write (NativePtr.add base offset) value -> memref.store value, base[offset]
/// Note: offsetSSA must be index type (nativeint in F# source)
let pMemRefStoreIndexed (memref: SSA) (value: SSA) (offsetSSA: SSA) (elemType: MLIRType) (memrefType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Store value at computed index (offsetSSA is already index type from nativeint)
        let! storeOp = pStore value memref [offsetSSA] elemType memrefType
        return ([storeOp], TRVoid)
    }

/// MemRef copy operation (NativePtr.copy)
/// Emits a call to memcpy library function
/// This is used for bulk memory copying between memrefs
/// Returns unit (no result SSA needed)
let pMemCopy (destSSA: SSA) (srcSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Call memcpy(dest, src, count) - standard C library function
        // memcpy signature: void* memcpy(void* dest, const void* src, size_t count)
        // All arguments are platform word sized (i64/i32 depending on arch)
        let args = [
            { SSA = destSSA; Type = platformWordTy }
            { SSA = srcSSA; Type = platformWordTy }
            { SSA = countSSA; Type = platformWordTy }
        ]

        // memcpy returns void* but we don't use the result (just for side effect)
        // Pass platformWordTy as return type (void* is a pointer)
        let! memcpyCall = pFuncCall None "memcpy" args platformWordTy

        return ([memcpyCall], TRVoid)
    }

/// MemRef load operation - MLIR memref.load (NOT LLVM pointer load)
/// Baker has already transformed NativePtr.read → MemRef.load
/// This emits: %result = memref.load %memref[%index] : memref<?xT>
///
/// SSA Coeffects (1 SSA allocated by SSAAssignment):
///   [0] = result (loaded value)
///
/// Parameters:
/// - nodeId: NodeId for extracting result SSA from coeffects
/// - memrefSSA: The memref to load from
/// - indexSSA: The index to load at (already computed by witness from MemRef.add marker)
let pMemRefLoad (nodeId: NodeId) (memrefSSA: SSA) (indexSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pMemRefLoad: Expected 1 SSA, got {ssas.Length}"

        let resultSSA = ssas.[0]
        let resultType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        // Emit memref.load %memref[%index]
        let! loadOp = pLoad resultSSA memrefSSA [indexSSA]

        return ([loadOp], TRValue { SSA = resultSSA; Type = resultType })
    }
