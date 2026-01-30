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
open Alex.Elements.MLIRElements
open Alex.Elements.MemRefElements
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements
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
// ALLOCATION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Address-of for immutable values: const 1, allocate, store, return pointer
/// SSAs: [0] = const 1, [1] = alloca result
let pAllocaImmutable (valueSSA: SSA) (valueType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2) $"pAllocaImmutable: Expected 2 SSAs, got {ssas.Length}"

        let constOneSSA = ssas.[0]
        let allocaSSA = ssas.[1]

        let constOneTy = TInt I64
        let! constOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca allocaSSA valueType None
        let! storeOp = pStore valueSSA allocaSSA [] valueType

        return [constOp; allocaOp; storeOp]
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
                // Bitcast (reinterpret bits)
                | _, _ ->
                    pBitCast resultSSA srcSSA dstType
            return [convOp]
    }

// ═══════════════════════════════════════════════════════════
// DU PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract DU tag (handles both inline and pointer-based DUs)
/// Pointer-based: Load tag byte from offset 0
/// Inline: ExtractValue at index 0
let pExtractDUTag (duSSA: SSA) (duType: MLIRType) (tagSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        match duType with
        | TPtr ->
            // Pointer-based DU: load tag byte from offset 0
            let! loadOp = pLoad tagSSA duSSA
            return [loadOp]
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
        let slotType =
            match duType with
            | TStruct fields when payloadIndex < List.length fields -> fields.[payloadIndex]
            | _ -> payloadType

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
/// SSAs: [0] = count const, [1] = alloca, [2..2+2*N] = (idx, gep) pairs, [last-2] = undef, [last-1] = withPtr, [last] = result
let pBuildArray (elements: Val list) (elemType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let count = List.length elements
        let expectedSSAs = 2 + (2 * count) + 3  // count, alloca, (idx,gep)*N, undef, withPtr, result

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
                    let idxSSA = ssas.[2 + i * 2]
                    let gepSSA = ssas.[2 + i * 2 + 1]
                    let! idxOp = pConstI idxSSA (int64 i) indexTy
                    let! gepOp = pGEP gepSSA allocaSSA [(idxSSA, TInt I64)]
                    let! storeOp = pStore elem.SSA gepSSA [] elemType
                    return [idxOp; gepOp; storeOp]
                })
            |> sequence

        let storeOps = List.concat storeOpsList

        // Build fat pointer {ptr, count}
        let undefSSA = ssas.[ssas.Length - 3]
        let withPtrSSA = ssas.[ssas.Length - 2]
        let resultSSA = ssas.[ssas.Length - 1]
        let arrayType = TStruct [TPtr; TInt I64]

        let! undefOp = pUndef undefSSA arrayType
        let! insertPtrOp = pInsertValue withPtrSSA undefSSA allocaSSA [0] arrayType
        let! insertCountOp = pInsertValue resultSSA withPtrSSA countSSA [1] arrayType

        return [countOp; allocaOp] @ storeOps @ [undefOp; insertPtrOp; insertCountOp]
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

        // Return memref type (not TPtr) - conversion to pointer happens at FFI boundary
        return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
    }

/// Build NativePtr.write pattern
/// Writes a value to a pointer location
///
/// NativePtr.write (ptr: nativeptr<'T>) (value: 'T) : unit
let pNativePtrWrite (valueSSA: SSA) (ptrSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit memref.store operation
        let! storeOp = pStore valueSSA ptrSSA [] elemType

        return ([storeOp], TRVoid)
    }

/// Convert memref to pointer for FFI boundaries
/// Uses builtin.unrealized_conversion_cast for boundary crossing
let pMemRefToPtr (resultSSA: SSA) (memrefSSA: SSA) (memrefTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! extractOp = pExtractBasePtr resultSSA memrefSSA memrefTy
        return ([extractOp], TRValue { SSA = resultSSA; Type = TPtr })
    }

/// Build NativePtr.read pattern
/// Reads a value from a pointer location
///
/// NativePtr.read (ptr: nativeptr<'T>) : 'T
let pNativePtrRead (resultSSA: SSA) (ptrSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit memref.load operation
        let! loadOp = pLoad resultSSA ptrSSA

        return ([loadOp], TRValue { SSA = resultSSA; Type = TPtr })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT FIELD ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract field from struct (e.g., string.Pointer, string.Length)
/// Maps field name to index for known struct layouts
let pStructFieldGet (resultSSA: SSA) (structSSA: SSA) (fieldName: string) (structTy: MLIRType) (fieldTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Map field name to index for known struct types
        // String fat pointer: [Pointer=0, Length=1]
        let fieldIndex =
            match fieldName with
            | "Pointer" -> 0
            | "Length" -> 1
            | _ -> failwith $"Unknown field name: {fieldName}"

        // Extract field value - pass struct type for MLIR type annotation
        let! ops = pExtractField structSSA fieldIndex resultSSA structTy

        return (ops, TRValue { SSA = resultSSA; Type = fieldTy })
    }
