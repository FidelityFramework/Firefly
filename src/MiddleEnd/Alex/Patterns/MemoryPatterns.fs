/// MemoryPatterns - Memory operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns to elide memory operations to MLIR.
/// Patterns compose Elements (internal) into semantic memory operations.
module Alex.Patterns.MemoryPatterns

open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRElements
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

/// Create parser failure with error message
let private pfail msg : PSGParser<'a> = fail (Message msg)

/// Sequence a list of parsers into a parser of a list
let rec private sequence (parsers: PSGParser<'a> list) : PSGParser<'a list> =
    match parsers with
    | [] -> preturn []
    | p :: ps ->
        parser {
            let! x = p
            let! xs = sequence ps
            return x :: xs
        }

// ═══════════════════════════════════════════════════════════
// FIELD EXTRACTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract single field from struct
let pExtractField (structSSA: SSA) (fieldIndex: int) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractOp = pExtractValue resultSSA structSSA [fieldIndex]
        return [extractOp]
    }

// ═══════════════════════════════════════════════════════════
// ALLOCATION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Address-of for immutable values: const 1, allocate, store, return pointer
/// SSAs: [0] = const 1, [1] = alloca result
let pAllocaImmutable (valueSSA: SSA) (valueType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 2 then
            return! pfail $"pAllocaImmutable: Expected 2 SSAs, got {ssas.Length}"

        let constOneSSA = ssas.[0]
        let allocaSSA = ssas.[1]

        let! constOp = pConstI constOneSSA 1L
        let! allocaOp = pAlloca allocaSSA (Some 1)
        let! storeOp = pStore valueSSA allocaSSA

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
            let! extractOp = pExtractValue tagSSA duSSA [0]
            return [extractOp]
    }

/// Extract DU payload with optional type conversion
/// SSAs: [0] = extract, [1] = convert (if needed)
let pExtractDUPayload (duSSA: SSA) (duType: MLIRType) (payloadIndex: int) (payloadType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 1 then
            return! pfail $"pExtractDUPayload: Expected at least 1 SSA, got {ssas.Length}"

        let extractSSA = ssas.[0]

        // Determine slot type from DU struct
        let slotType =
            match duType with
            | TStruct fields when payloadIndex < List.length fields -> fields.[payloadIndex]
            | _ -> payloadType

        // Extract payload
        let! extractOp = pExtractValue extractSSA duSSA [payloadIndex]

        // Convert if needed
        if slotType = payloadType then
            return [extractOp]
        else
            if ssas.Length < 2 then
                return! pfail $"pExtractDUPayload: Need 2 SSAs for conversion, got {ssas.Length}"
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
        if ssas.Length <> updates.Length then
            return! pfail $"pRecordCopyWith: Expected {updates.Length} SSAs, got {ssas.Length}"

        // Fold over updates, threading prevSSA through
        let! result =
            List.zip ssas updates
            |> List.fold (fun accParser (targetSSA, (fieldIdx, valueSSA)) ->
                parser {
                    let! (prevOps, prevSSA) = accParser
                    let! insertOp = pInsertValue targetSSA prevSSA valueSSA [fieldIdx]
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

        if ssas.Length < expectedSSAs then
            return! pfail $"pBuildArray: Expected {expectedSSAs} SSAs, got {ssas.Length}"

        let countSSA = ssas.[0]
        let allocaSSA = ssas.[1]

        // Allocate array
        let! countOp = pConstI countSSA (int64 count)
        let! allocaOp = pAlloca allocaSSA (Some count)

        // Store each element
        let! storeOpsList =
            elements
            |> List.indexed
            |> List.map (fun (i, elem) ->
                parser {
                    let idxSSA = ssas.[2 + i * 2]
                    let gepSSA = ssas.[2 + i * 2 + 1]
                    let! idxOp = pConstI idxSSA (int64 i)
                    let! gepOp = pGEP gepSSA allocaSSA [(idxSSA, TInt I64)]
                    let! storeOp = pStore elem.SSA gepSSA
                    return [idxOp; gepOp; storeOp]
                })
            |> sequence

        let storeOps = List.concat storeOpsList

        // Build fat pointer {ptr, count}
        let undefSSA = ssas.[ssas.Length - 3]
        let withPtrSSA = ssas.[ssas.Length - 2]
        let resultSSA = ssas.[ssas.Length - 1]
        let arrayType = TStruct [TPtr; TInt I64]

        let! undefOp = pUndef undefSSA
        let! insertPtrOp = pInsertValue withPtrSSA undefSSA allocaSSA [0]
        let! insertCountOp = pInsertValue resultSSA withPtrSSA countSSA [1]

        return [countOp; allocaOp] @ storeOps @ [undefOp; insertPtrOp; insertCountOp]
    }
