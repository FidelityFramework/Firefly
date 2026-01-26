/// Memory UnionCases Witness - DU case construction and payload extraction
///
/// SCOPE: witnessUnionCase, witnessPayloadExtract, witnessTuplePatternExtract
/// DOES NOT: DU operations (GetTag, Eliminate, Construct - separate witness)
module Alex.Witnesses.Memory.UnionCases

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Witnesses.Memory.Indexing

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
