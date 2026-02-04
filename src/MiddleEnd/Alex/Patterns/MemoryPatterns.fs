/// MemoryPatterns - Memory operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns to elide memory operations to MLIR.
/// Patterns compose Elements (internal) into semantic memory operations.
module Alex.Patterns.MemoryPatterns

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
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════
// FIELD EXTRACTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract single field from struct
let pExtractField (structSSA: SSA) (fieldIndex: int) (resultSSA: SSA) (structTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let! extractOp = pExtractValue resultSSA structSSA [fieldIndex] structTy
        return [extractOp]
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

        // Allocate SSA for offset constant
        // TODO BACKFILL: offsetSSA should come from coeffects or PSG node
        let offsetSSA = failwith "MemoryPatterns.pFieldAccess: offsetSSA must come from coeffects (removed ad-hoc calculation)"
        let! offsetOp = pConstI offsetSSA (int64 fieldOffset) TIndex

        // Memref.load with byte offset
        // Note: This assumes structPtr is memref<Nxi8> and we load at byte offset
        let! loadOp = Alex.Elements.MemRefElements.pLoad loadSSA structPtr [offsetSSA]

        return [offsetOp; loadOp]
    }

/// Field set via byte-offset memref operations
/// structType: The NativeType of the struct (for calculating field offset)
let pFieldSet (structPtr: SSA) (structType: NativeType) (fieldIndex: int) (value: SSA) (_gepSSA: SSA) (_indexSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let elemType = mapNativeTypeForArch arch state.Current.Type

        // Calculate byte offset for the field using FNCS-provided type structure
        let fieldOffset = calculateFieldOffsetForArch arch structType fieldIndex

        // Allocate SSA for offset constant
        // TODO BACKFILL: offsetSSA should come from coeffects or PSG node
        let offsetSSA = failwith "MemoryPatterns.pFieldSet: offsetSSA must come from coeffects (removed ad-hoc calculation)"
        let! offsetOp = pConstI offsetSSA (int64 fieldOffset) TIndex

        // Memref.store with byte offset
        let! storeOp = pStore value structPtr [offsetSSA] elemType

        return [offsetOp; storeOp]
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
        let! allocaOp = pAlloca allocaSSA valueType None
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let! storeOp = pStore valueSSA allocaSSA [indexSSA] valueType

        return [constOp; allocaOp; indexOp; storeOp]
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
            return [convOp]
    }

// ═══════════════════════════════════════════════════════════
// DU PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract DU tag (handles both inline and pointer-based DUs)
/// Pointer-based: Load tag byte from offset 0
/// Inline: ExtractValue at index 0
/// SSAs: tagSSA for result, indexZeroSSA for memref index
let pExtractDUTag (duSSA: SSA) (duType: MLIRType) (tagSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        match duType with
        | TIndex ->
            // Pointer-based DU: load tag byte from offset 0
            let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // MLIR memrefs require indices
            let! loadOp = pLoad tagSSA duSSA [indexZeroSSA]
            return [indexZeroOp; loadOp]
        | _ ->
            // Inline struct DU: extract tag from field 0
            let tagTy = TInt I8  // DU tags are always i8
            let! extractOp = pExtractValue tagSSA duSSA [0] tagTy
            return [extractOp]
    }

/// Extract DU payload with optional type conversion
/// SSAs: [0] = extract, [1] = convert (if needed)
let pExtractDUPayload (duSSA: SSA) (duType: MLIRType) (payloadIndex: int) (payloadType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 1) $"pExtractDUPayload: Expected at least 1 SSA, got {ssas.Length}"

        let extractSSA = ssas.[0]

        // Determine slot type from DU struct
        // Note: With TMemRefStatic, we use payloadType directly
        let slotType = payloadType

        // Extract payload
        let! extractOp = pExtractValue extractSSA duSSA [payloadIndex] slotType

        // Convert if needed
        if slotType = payloadType then
            return [extractOp]
        else
            do! ensure (ssas.Length >= 2) $"pExtractDUPayload: Need 2 SSAs for conversion, got {ssas.Length}"
            let convertSSA = ssas.[1]
            let! convOps = pConvertType extractSSA slotType payloadType convertSSA
            return extractOp :: convOps
    }

// ═══════════════════════════════════════════════════════════
// RECORD PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record copy-and-update: start with original, insert updated fields
/// SSAs: one per updated field
/// Updates: (fieldIndex, valueSSA) pairs
let pRecordCopyWith (origSSA: SSA) (recordType: MLIRType) (updates: (int * SSA) list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = updates.Length) $"pRecordCopyWith: Expected {updates.Length} SSAs, got {ssas.Length}"

        // Fold over updates, threading prevSSA through
        let! result =
            List.zip ssas updates
            |> List.fold (fun accParser (targetSSA, (fieldIdx, valueSSA)) ->
                parser {
                    let! (prevOps, prevSSA) = accParser
                    let! insertOp = pInsertValue targetSSA prevSSA valueSSA [fieldIdx] recordType
                    return (prevOps @ [insertOp], targetSSA)
                }
            ) (preturn ([], origSSA))

        let (ops, _) = result
        return ops
    }

// ═══════════════════════════════════════════════════════════
// ARRAY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build array: allocate, initialize elements, construct fat pointer
/// SSAs: [0] = count const, [1] = alloca, [2..2+3*N] = (idx, gep, indexZero) triples, [last-2] = undef, [last-1] = withPtr, [last] = result
let pBuildArray (elements: Val list) (elemType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let count = List.length elements
        let expectedSSAs = 2 + (3 * count) + 3  // count, alloca, (idx,gep,indexZero)*N, undef, withPtr, result

        do! ensure (ssas.Length >= expectedSSAs) $"pBuildArray: Expected {expectedSSAs} SSAs, got {ssas.Length}"

        let countSSA = ssas.[0]
        let allocaSSA = ssas.[1]

        // Allocate array
        let countTy = TInt I64
        let! countOp = pConstI countSSA (int64 count) countTy
        let! allocaOp = pAlloca allocaSSA elemType None

        // Store each element
        let indexTy = TInt I64
        let! storeOpsList =
            elements
            |> List.indexed
            |> List.map (fun (i, elem) ->
                parser {
                    let idxSSA = ssas.[2 + i * 3]
                    let gepSSA = ssas.[2 + i * 3 + 1]
                    let indexZeroSSA = ssas.[2 + i * 3 + 2]
                    let! idxOp = pConstI idxSSA (int64 i) indexTy
                    let! gepOp = pSubView gepSSA allocaSSA [idxSSA]
                    let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // Index 0 for 1-element memref
                    let! storeOp = pStore elem.SSA gepSSA [indexZeroSSA] elemType
                    return [idxOp; gepOp; indexZeroOp; storeOp]
                })
            |> sequence

        let storeOps = List.concat storeOpsList

        // Build fat pointer {ptr, count}
        let undefSSA = ssas.[ssas.Length - 3]
        let withPtrSSA = ssas.[ssas.Length - 2]
        let resultSSA = ssas.[ssas.Length - 1]
        let totalBytes = mlirTypeSize TIndex + mlirTypeSize (TInt I64)
        let arrayType = TMemRefStatic(totalBytes, TInt I8)

        let! undefOp = pUndef undefSSA arrayType
        let! insertPtrOp = pInsertValue withPtrSSA undefSSA allocaSSA [0] arrayType
        let! insertCountOp = pInsertValue resultSSA withPtrSSA countSSA [1] arrayType

        return [countOp; allocaOp] @ storeOps @ [undefOp; insertPtrOp; insertCountOp]
    }

/// Array element access via SubView + Load
/// SSAs: gepSSA for subview, loadSSA for result, indexZeroSSA for memref index
let pArrayAccess (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (gepSSA: SSA) (loadSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // MLIR memrefs require indices
        let! loadOp = pLoad loadSSA gepSSA [indexZeroSSA]
        return [subViewOp; indexZeroOp; loadOp]
    }

/// Array element set via SubView + Store
let pArraySet (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (value: SSA) (gepSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let elemType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // Index 0 for 1-element memref
        let! storeOp = pStore value gepSSA [indexZeroSSA] elemType
        return [subViewOp; indexZeroOp; storeOp]
    }

// ═══════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS (FNCS Intrinsics)
// ═══════════════════════════════════════════════════════════

/// Build NativePtr.stackalloc pattern
/// Allocates memory on the stack and returns a pointer
///
/// NativePtr.stackalloc<'T>() : nativeptr<'T>
let pNativePtrStackAlloc (resultSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
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

        let! allocaOp = pAlloca resultSSA elemType None
        let memrefTy = TMemRefScalar elemType

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
        let! storeOp = pStore valueSSA ptrSSA [indexSSA] elemType

        return ([indexOp; storeOp], TRVoid)
    }

/// Convert memref to pointer for FFI boundaries
/// Extracts pointer as index, then casts to platform word
let pMemRefToPtr (resultSSA: SSA) (memrefSSA: SSA) (memrefTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Extract pointer as index (portable)
        // TODO BACKFILL: indexSSA should come from coeffects or PSG node
        let indexSSA = failwith "MemoryPatterns.pMemRefToPtr: indexSSA must come from coeffects (removed SSA.V 999996)"
        let! extractOp = pExtractBasePtr indexSSA memrefSSA memrefTy

        // Get platform word size from XParsec state
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Cast index → platform word (i64 on x64, i32 on ARM32)
        let! castOp = pIndexCastS resultSSA indexSSA platformWordTy

        return ([extractOp; castOp], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build NativePtr.read pattern
/// Reads a value from a pointer location
///
/// NativePtr.read (ptr: nativeptr<'T>) : 'T
/// SSAs: resultSSA for loaded value, indexZeroSSA for memref index
let pNativePtrRead (resultSSA: SSA) (ptrSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Get result type from XParsec state (type of value being loaded)
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let resultType = mapNativeTypeForArch arch state.Current.Type

        // Emit memref.load operation with index
        // indexZeroSSA should be pre-assigned by coeffects and already witnessed (constant 0)
        let! loadOp = pLoad resultSSA ptrSSA [indexZeroSSA]

        return ([loadOp], TRValue { SSA = resultSSA; Type = resultType })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT FIELD ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract field from struct (e.g., string.Pointer, string.Length)
/// SSA layout (max 3): [0] = intermediate (index or dim const), [1] = intermediate2 (dim result), [2] = result
let pStructFieldGet (ssas: SSA list) (structSSA: SSA) (fieldName: string) (structTy: MLIRType) (fieldTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
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

            // Extract field value - pass struct type for MLIR type annotation
            let! ops = pExtractField structSSA fieldIndex resultSSA structTy
            return (ops, TRValue { SSA = resultSSA; Type = fieldTy })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT CONSTRUCTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record struct via Undef + InsertValue chain
let pRecordStruct (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = fields.Length + 1) $"pRecordStruct: Expected {fields.Length + 1} SSAs, got {ssas.Length}"

        // Compute struct type from field types
        let fieldTypes = fields |> List.map (fun f -> f.Type)
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let structTy = TMemRefStatic(totalBytes, TInt I8)
        let! undefOp = pUndef ssas.[0] structTy

        let! insertOps =
            fields
            |> List.mapi (fun i field ->
                parser {
                    let targetSSA = ssas.[i+1]
                    let sourceSSA = if i = 0 then ssas.[0] else ssas.[i]
                    return! pInsertValue targetSSA sourceSSA field.SSA [i] structTy
                })
            |> sequence

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
let pDUCase (tag: int64) (payload: Val list) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2 + payload.Length) $"pDUCase: Expected at least {2 + payload.Length} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0] ty

        // Insert tag at index 0
        let tagTy = TInt I8  // DU tags are always i8
        let! tagConstOp = pConstI ssas.[1] tag tagTy
        let! insertTagOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0] ty

        // Insert payload fields starting at index 1
        let! payloadOps =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let targetSSA = ssas.[3 + i]
                    let sourceSSA = if i = 0 then ssas.[2] else ssas.[2 + i]
                    return! pInsertValue targetSSA sourceSSA field.SSA [i + 1] ty
                })
            |> sequence

        return undefOp :: tagConstOp :: insertTagOp :: payloadOps
    }

// ═══════════════════════════════════════════════════════════
// SIMPLE MEMORY STORE
// ═══════════════════════════════════════════════════════════

/// Simple memref.store with no indices (scalar store)
let pMemRefStore (value: SSA) (memref: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! storeOp = pStore value memref [] elemType
        return ([storeOp], TRVoid)
    }
