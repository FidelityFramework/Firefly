/// ElisionPatterns - Composable MLIR elision templates via XParsec
///
/// PUBLIC: Witnesses call these patterns to elide PSG structure to MLIR.
/// Patterns compose Elements (internal) into semantic operations.
module Alex.Patterns.ElisionPatterns

open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }, >>=, <|>
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRElements
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements
open Alex.Elements.SCFElements
open Alex.Elements.CFElements
open Alex.Elements.FuncElements
open Alex.Elements.IndexElements
open Alex.Elements.VectorElements
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// ═══════════════════════════════════════════════════════════
// XParsec HELPERS
// ═══════════════════════════════════════════════════════════

/// Create parser failure with error message
let pfail msg : PSGParser<'a> = fail (Message msg)

/// Sequence a list of parsers into a parser of a list
let rec sequence (parsers: PSGParser<'a> list) : PSGParser<'a list> =
    match parsers with
    | [] -> preturn []
    | p :: ps ->
        parser {
            let! x = p
            let! xs = sequence ps
            return x :: xs
        }

// ═══════════════════════════════════════════════════════════
// MEMORY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Field access via StructGEP + Load
let pFieldAccess (structPtr: SSA) (fieldIndex: int) (gepSSA: SSA) (loadSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! gepOp = pStructGEP gepSSA structPtr fieldIndex
        let! loadOp = pLoad loadSSA gepSSA
        return [gepOp; loadOp]
    }

/// Field set via StructGEP + Store
let pFieldSet (structPtr: SSA) (fieldIndex: int) (value: SSA) (gepSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! gepOp = pStructGEP gepSSA structPtr fieldIndex
        let! storeOp = pStore value gepSSA
        return [gepOp; storeOp]
    }

/// Array element access via GEP + Load
let pArrayAccess (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (gepSSA: SSA) (loadSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! gepOp = pGEP gepSSA arrayPtr [(index, indexTy)]
        let! loadOp = pLoad loadSSA gepSSA
        return [gepOp; loadOp]
    }

/// Array element set via GEP + Store
let pArraySet (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (value: SSA) (gepSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! gepOp = pGEP gepSSA arrayPtr [(index, indexTy)]
        let! storeOp = pStore value gepSSA
        return [gepOp; storeOp]
    }

// ═══════════════════════════════════════════════════════════
// STRUCT CONSTRUCTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record struct via Undef + InsertValue chain
let pRecordStruct (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length <> fields.Length + 1 then
            return! pfail $"pRecordStruct: Expected {fields.Length + 1} SSAs, got {ssas.Length}"

        let! undefOp = pUndef ssas.[0]

        let! insertOps =
            fields
            |> List.mapi (fun i field ->
                parser {
                    let targetSSA = ssas.[i+1]
                    let sourceSSA = if i = 0 then ssas.[0] else ssas.[i]
                    return! pInsertValue targetSSA sourceSSA field.SSA [i]
                })
            |> sequence

        return undefOp :: insertOps
    }

/// Tuple struct via Undef + InsertValue chain (same as record, but semantically different)
let pTupleStruct (elements: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    pRecordStruct elements ssas  // Same implementation, different semantic context

/// DU case construction: tag field (index 0) + payload fields
let pDUCase (tag: int64) (payload: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 2 + payload.Length then
            return! pfail $"pDUCase: Expected at least {2 + payload.Length} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0]

        // Insert tag at index 0
        let! tagConstOp = pConstI ssas.[1] tag
        let! insertTagOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0]

        // Insert payload fields starting at index 1
        let! payloadOps =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let targetSSA = ssas.[3 + i]
                    let sourceSSA = if i = 0 then ssas.[2] else ssas.[2 + i]
                    return! pInsertValue targetSSA sourceSSA field.SSA [i + 1]
                })
            |> sequence

        return undefOp :: tagConstOp :: insertTagOp :: payloadOps
    }

// ═══════════════════════════════════════════════════════════
// CLOSURE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Flat closure struct: code_ptr field + capture fields
let pFlatClosure (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 2 + captures.Length then
            return! pfail $"pFlatClosure: Expected at least {2 + captures.Length} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0]

        // Insert code_ptr at index 0
        let! insertCodeOp = pInsertValue ssas.[1] ssas.[0] codePtr [0]

        // Insert captures starting at index 1
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[2 + i]
                    let sourceSSA = if i = 0 then ssas.[1] else ssas.[1 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [i + 1]
                })
            |> sequence

        return undefOp :: insertCodeOp :: captureOps
    }

/// Closure call: extract code_ptr, extract captures, call
let pClosureCall (closureSSA: SSA) (captureCount: int) (args: Val list)
                 (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        if extractSSAs.Length <> captureCount + 1 then
            return! pfail $"pClosureCall: Expected {captureCount + 1} extract SSAs, got {extractSSAs.Length}"

        // Extract code_ptr from index 0
        let codePtrSSA = extractSSAs.[0]
        let! extractCodeOp = pExtractValue codePtrSSA closureSSA [0]

        // Extract captures from indices 1..captureCount
        let! extractCaptureOps =
            [0 .. captureCount - 1]
            |> List.map (fun i ->
                parser {
                    let capSSA = extractSSAs.[i + 1]
                    return! pExtractValue capSSA closureSSA [i + 1]
                })
            |> sequence

        // Call with captures prepended to args
        // Note: Capture types come from ClosureLayout coeffect (PSGElaboration provides this)
        // Gap: Need to thread ClosureLayout through pattern parameters to get actual capture types
        // For now using placeholder types - witnesses should pass correct types from coeffects
        let captureVals = extractSSAs.[1..] |> List.map (fun ssa -> { SSA = ssa; Type = TInt I64 })
        let allArgs = captureVals @ args
        let! callOp = pIndirectCall resultSSA codePtrSSA (allArgs |> List.map (fun v -> v.SSA))

        return extractCodeOp :: extractCaptureOps @ [callOp]
    }

// ═══════════════════════════════════════════════════════════
// LAZY PATTERNS (PRD-14)
// ═══════════════════════════════════════════════════════════

/// Lazy struct: {computed: i1, value: T, code_ptr, captures...}
let pLazyStruct (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 4 + captures.Length then
            return! pfail $"pLazyStruct: Expected at least {4 + captures.Length} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0]

        // Insert computed = false at index 0
        let! falseConstOp = pConstI ssas.[1] 0L  // i1 false = 0
        let! insertComputedOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0]

        // Insert code_ptr at index 2
        let! insertCodeOp = pInsertValue ssas.[3] ssas.[2] codePtr [2]

        // Insert captures starting at index 3
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[4 + i]
                    let sourceSSA = ssas.[3 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [i + 3]
                })
            |> sequence

        return undefOp :: falseConstOp :: insertComputedOp :: insertCodeOp :: captureOps
    }

/// Build lazy struct: High-level pattern for witnesses
/// Combines pLazyStruct with proper result construction
let pBuildLazyStruct (codePtr: SSA) (captures: Val list) (ssas: SSA list) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        let! ops = pLazyStruct codePtr captures ssas

        // Final SSA is the last one (after all insertions)
        let finalSSA = ssas.[3 + captures.Length]

        // Get the lazy type - it's a struct with {computed: i1, value: T, code_ptr, captures...}
        let! state = getUserState
        let lazyType = state.Current.Type
        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch lazyType

        return (ops, TRValue { SSA = finalSSA; Type = mlirType })
    }

/// Build lazy force: Call lazy thunk via struct pointer passing
///
/// LazyForce is a SIMPLE operation (not elaborated by FNCS).
/// SSA cost: Fixed 4 (extract code_ptr, const 1, alloca, call)
///
/// Calling convention: Thunk receives pointer to lazy struct
/// 1. Extract code_ptr from lazy struct [2]
/// 2. Alloca space for lazy struct on stack (const 1 for size)
/// 3. Store lazy struct to get pointer
/// 4. Call thunk with pointer -> result
///
/// The thunk extracts captures internally using LazyLayout coeffect.
///
/// Lazy struct: {computed: i1, value: T, code_ptr: ptr, capture0, capture1, ...}
let pBuildLazyForce (lazySSA: SSA) (resultSSA: SSA) (ssas: SSA list) (arch: Architecture)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        // SSAs: [0] = code_ptr, [1] = const 1, [2] = alloca'd ptr
        if ssas.Length < 3 then
            return! pfail $"pBuildLazyForce: Expected at least 3 SSAs, got {ssas.Length}"

        let codePtrSSA = ssas.[0]
        let constOneSSA = ssas.[1]
        let ptrSSA = ssas.[2]

        // 1. Extract code_ptr from lazy struct [2]
        let! extractCodePtrOp = pExtractValue codePtrSSA lazySSA [2]

        // 2. Alloca space for lazy struct (1 element)
        let! constOneOp = pConstI constOneSSA 1L
        let! allocaOp = pAlloca ptrSSA (Some 1)

        // 3. Store lazy struct to alloca'd space
        let! storeOp = pStore lazySSA ptrSSA

        // 4. Call thunk with pointer -> result
        let! callOp = pIndirectCall resultSSA codePtrSSA [ptrSSA]

        let! state = getUserState
        let resultType = state.Current.Type
        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch resultType

        return ([extractCodePtrOp; constOneOp; allocaOp; storeOp; callOp], TRValue { SSA = resultSSA; Type = mlirType })
    }

// ═══════════════════════════════════════════════════════════
// SEQ PATTERNS (PRD-15)
// ═══════════════════════════════════════════════════════════

/// Seq struct: {state: i32, current: T, code_ptr, captures..., internal_state...}
let pSeqStruct (stateInit: int64) (codePtr: SSA) (captures: Val list)
               (internalState: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let minSSAs = 4 + captures.Length + internalState.Length
        if ssas.Length < minSSAs then
            return! pfail $"pSeqStruct: Expected at least {minSSAs} SSAs, got {ssas.Length}"

        // Create undef struct
        let! undefOp = pUndef ssas.[0]

        // Insert state = stateInit at index 0
        let! stateConstOp = pConstI ssas.[1] stateInit
        let! insertStateOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0]

        // Insert code_ptr at index 2
        let! insertCodeOp = pInsertValue ssas.[3] ssas.[2] codePtr [2]

        // Insert captures starting at index 3
        let captureBaseIdx = 3
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[4 + i]
                    let sourceSSA = ssas.[3 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [captureBaseIdx + i]
                })
            |> sequence

        // Insert internal state starting after captures
        let internalBaseIdx = captureBaseIdx + captures.Length
        let! internalOps =
            internalState
            |> List.mapi (fun i st ->
                parser {
                    let targetSSA = ssas.[4 + captures.Length + i]
                    let sourceSSA = ssas.[3 + captures.Length + i]
                    return! pInsertValue targetSSA sourceSSA st.SSA [internalBaseIdx + i]
                })
            |> sequence

        return undefOp :: stateConstOp :: insertStateOp :: insertCodeOp
               :: (captureOps @ internalOps)
    }

/// Seq MoveNext: extract state, load captures/internal, call code_ptr, update state/current
let pSeqMoveNext (seqSSA: SSA) (captureCount: int) (internalCount: int)
                 (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let expectedExtracts = 2 + captureCount + internalCount  // state, code_ptr, captures, internal
        if extractSSAs.Length < expectedExtracts then
            return! pfail $"pSeqMoveNext: Expected at least {expectedExtracts} extract SSAs, got {extractSSAs.Length}"

        // Extract state from index 0
        let stateSSA = extractSSAs.[0]
        let! extractStateOp = pExtractValue stateSSA seqSSA [0]

        // Extract code_ptr from index 2
        let codePtrSSA = extractSSAs.[1]
        let! extractCodeOp = pExtractValue codePtrSSA seqSSA [2]

        // Extract captures from indices 3..3+captureCount
        let captureBaseIdx = 3
        let! extractCaptureOps =
            [0 .. captureCount - 1]
            |> List.map (fun i ->
                parser {
                    let capSSA = extractSSAs.[2 + i]
                    return! pExtractValue capSSA seqSSA [captureBaseIdx + i]
                })
            |> sequence

        // Extract internal state
        let internalBaseIdx = captureBaseIdx + captureCount
        let! extractInternalOps =
            [0 .. internalCount - 1]
            |> List.map (fun i ->
                parser {
                    let intSSA = extractSSAs.[2 + captureCount + i]
                    return! pExtractValue intSSA seqSSA [internalBaseIdx + i]
                })
            |> sequence

        // Call code_ptr with state + captures + internal
        let allArgs = extractSSAs.[0..] |> List.take (2 + captureCount + internalCount)
        let! callOp = pIndirectCall resultSSA codePtrSSA allArgs

        return extractStateOp :: extractCodeOp :: extractCaptureOps
               @ extractInternalOps @ [callOp]
    }

/// Build seq struct: High-level pattern for SeqExpr witnesses
/// Combines pSeqStruct with proper result construction
///
/// SeqExpr structure from FNCS: `SeqExpr of body: NodeId * captures: CaptureInfo list`
/// The body is the MoveNext lambda that was elaborated by FNCS saturation.
/// Captures are the closed-over variables.
/// Internal state fields come from the body lambda's mutable locals.
let pBuildSeqStruct (codePtr: SSA) (captures: Val list) (internalState: Val list)
                    (ssas: SSA list) (arch: Architecture)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        // Initial state is 0 (unstarted)
        let! ops = pSeqStruct 0L codePtr captures internalState ssas

        // Final SSA is the last one (after all insertions)
        let finalSSA = ssas.[3 + captures.Length + internalState.Length]

        // Get the seq type - it's a struct with {state: i32, current: T, code_ptr, captures..., internal...}
        let! state = getUserState
        let seqType = state.Current.Type
        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch seqType

        return (ops, TRValue { SSA = finalSSA; Type = mlirType })
    }

/// Build ForEach loop: Iterate over sequence with MoveNext calls
///
/// ForEach structure from FNCS: `ForEach of var: string * collection: NodeId * body: NodeId`
/// This generates a while loop calling MoveNext until exhausted.
///
/// Gap: MoveNext calling convention implementation needed
/// Seq struct layout is: {state: i32, current: T, code_ptr: ptr, captures..., internal...}
/// MoveNext should: extract code_ptr[2], alloca seq, store seq, call code_ptr(seq_ptr) -> i1
let pBuildForEachLoop (collectionSSA: SSA) (bodyOps: MLIROp list)
                      (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        // ForEach is a while loop structure:
        // 1. Extract code_ptr from seq struct at [2]
        // 2. Alloca space for seq, store seq to get pointer
        // 3. Call code_ptr(seq_ptr) -> i1 (returns true if has next)
        // 4. If true, extract current element, execute body, loop
        // 5. If false, exit loop

        // Implementation: Use SCF.While with:
        //   - Condition region: call MoveNext, extract result
        //   - Body region: extract current, execute bodyOps, yield
        return! pfail "ForEach MoveNext implementation gap - needs: extract code_ptr[2], alloca/store seq, indirect call"
    }

// ═══════════════════════════════════════════════════════════
// CONTROL FLOW PATTERNS
// ═══════════════════════════════════════════════════════════

/// If/then/else via SCF.If
let pBuildIfThenElse (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list option) : PSGParser<MLIROp list> =
    parser {
        let! ifOp = pSCFIf cond thenOps elseOps
        return [ifOp]
    }

/// While loop via SCF.While
let pBuildWhileLoop (condOps: MLIROp list) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! whileOp = pSCFWhile condOps bodyOps
        return [whileOp]
    }

/// For loop via SCF.For
let pBuildForLoop (lower: SSA) (upper: SSA) (step: SSA) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! forOp = pSCFFor lower upper step bodyOps
        return [forOp]
    }

/// Switch statement via CF.Switch
let pSwitch (flag: SSA) (flagTy: MLIRType) (defaultOps: MLIROp list)
            (cases: (int64 * MLIROp list) list) : PSGParser<MLIROp list> =
    parser {
        // Convert case ops to block refs (would need actual block construction)
        // For now, placeholder structure
        let defaultBlock = BlockRef "default"
        let caseBlocks = cases |> List.map (fun (value, _) ->
            (value, BlockRef $"case_{value}", []))

        let! switchOp = Alex.Elements.CFElements.pSwitch flag flagTy defaultBlock [] caseBlocks
        return [switchOp]
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC PATTERNS
// ═══════════════════════════════════════════════════════════

/// Integer addition
let pAddInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! addOp = pAddI resultSSA lhs rhs
        return [addOp]
    }

/// Integer subtraction
let pSubInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! subOp = pSubI resultSSA lhs rhs
        return [subOp]
    }

/// Integer multiplication
let pMulInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! mulOp = pMulI resultSSA lhs rhs
        return [mulOp]
    }

/// Signed integer division
let pDivSInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! divOp = pDivSI resultSSA lhs rhs
        return [divOp]
    }

/// Unsigned integer division
let pDivUInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! divOp = pDivUI resultSSA lhs rhs
        return [divOp]
    }

/// Signed integer remainder
let pRemSInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! remOp = pRemSI resultSSA lhs rhs
        return [remOp]
    }

/// Unsigned integer remainder
let pRemUInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! remOp = pRemUI resultSSA lhs rhs
        return [remOp]
    }

/// Bitwise AND
let pAndInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! andOp = pAndI resultSSA lhs rhs
        return [andOp]
    }

/// Bitwise OR
let pOrInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! orOp = pOrI resultSSA lhs rhs
        return [orOp]
    }

/// Bitwise XOR
let pXorInt (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! xorOp = pXorI resultSSA lhs rhs
        return [xorOp]
    }

/// Left shift
let pShiftLeft (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! shlOp = pShLI resultSSA lhs rhs
        return [shlOp]
    }

/// Logical right shift (unsigned)
let pShiftRightLogical (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! shrOp = pShRUI resultSSA lhs rhs
        return [shrOp]
    }

/// Arithmetic right shift (signed)
let pShiftRightArith (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! shrOp = pShRSI resultSSA lhs rhs
        return [shrOp]
    }

/// Float addition
let pAddFloat (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! addOp = pAddF resultSSA lhs rhs
        return [addOp]
    }

/// Float subtraction
let pSubFloat (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! subOp = pSubF resultSSA lhs rhs
        return [subOp]
    }

/// Float multiplication
let pMulFloat (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! mulOp = pMulF resultSSA lhs rhs
        return [mulOp]
    }

/// Float division
let pDivFloat (resultSSA: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! divOp = pDivF resultSSA lhs rhs
        return [divOp]
    }

/// Integer comparison
let pCmpI (resultSSA: SSA) (pred: ICmpPred) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp list> =
    parser {
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA pred lhs rhs
        return [cmpOp]
    }

/// Conditional select (ternary operator)
let pSelect (resultSSA: SSA) (cond: SSA) (trueVal: SSA) (falseVal: SSA) : PSGParser<MLIROp list> =
    parser {
        let! selectOp = Alex.Elements.ArithElements.pSelect resultSSA cond trueVal falseVal
        return [selectOp]
    }

// ═══════════════════════════════════════════════════════════
// OPTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Option.Some: tag=1 + value
let pOptionSome (value: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 1L [value] ssas

/// Option.None: tag=0
let pOptionNone (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas

/// Option.IsSome: extract tag, compare with 1
let pOptionIsSome (optionSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractTagOp = pExtractValue tagSSA optionSSA [0]
        let! oneConstOp = pConstI oneSSA 1L
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA oneSSA
        return [extractTagOp; oneConstOp; cmpOp]
    }

/// Option.IsNone: extract tag, compare with 0
let pOptionIsNone (optionSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractTagOp = pExtractValue tagSSA optionSSA [0]
        let! zeroConstOp = pConstI zeroSSA 0L
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Option.Get: extract value field (index 1)
let pOptionGet (optionSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractOp = pExtractValue resultSSA optionSSA [1]
        return [extractOp]
    }

// ═══════════════════════════════════════════════════════════
// LIST PATTERNS
// ═══════════════════════════════════════════════════════════

/// List.Cons: tag=1 + head + tail
let pListCons (head: Val) (tail: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 1L [head; tail] ssas

/// List.Empty: tag=0
let pListEmpty (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas

/// List.IsEmpty: extract tag, compare with 0
let pListIsEmpty (listSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractTagOp = pExtractValue tagSSA listSSA [0]
        let! zeroConstOp = pConstI zeroSSA 0L
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

// ═══════════════════════════════════════════════════════════
// MAP PATTERNS
// ═══════════════════════════════════════════════════════════

/// Map.Empty: empty tree structure (tag=0 for leaf node)
let pMapEmpty (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas

/// Map.Add: create new tree node with key, value, left, right subtrees
/// Structure: tag=1, key, value, left_tree, right_tree
let pMapAdd (key: Val) (value: Val) (left: Val) (right: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 1L [key; value; left; right] ssas

/// Map.ContainsKey: tree traversal comparing keys
/// This is a composite operation that would be implemented as a recursive function
/// The pattern here just shows the key comparison at each node
let pMapKeyCompare (mapNodeSSA: SSA) (searchKey: SSA) (keySSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract key from node (index 1, after tag at index 0)
        let! extractKeyOp = pExtractValue keySSA mapNodeSSA [1]
        // Compare search key with node key
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq searchKey keySSA
        return [extractKeyOp; cmpOp]
    }

/// Map.TryFind: similar to ContainsKey but returns Option<value>
let pMapExtractValue (mapNodeSSA: SSA) (valueSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract value from node (index 2, after tag and key)
        let! extractValueOp = pExtractValue valueSSA mapNodeSSA [2]
        return [extractValueOp]
    }

// ═══════════════════════════════════════════════════════════
// SET PATTERNS
// ═══════════════════════════════════════════════════════════

/// Set.Empty: empty tree structure (tag=0 for leaf node)
let pSetEmpty (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas

/// Set.Add: create new tree node with element, left, right subtrees
/// Structure: tag=1, element, left_tree, right_tree
let pSetAdd (element: Val) (left: Val) (right: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 1L [element; left; right] ssas

/// Set.Contains: tree traversal comparing elements
let pSetElementCompare (setNodeSSA: SSA) (searchElem: SSA) (elemSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract element from node (index 1, after tag at index 0)
        let! extractElemOp = pExtractValue elemSSA setNodeSSA [1]
        // Compare search element with node element
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq searchElem elemSSA
        return [extractElemOp; cmpOp]
    }

/// Set.Union: combines two sets (implemented as tree merge operation)
/// Pattern shows extraction of subtrees for recursive union
let pSetExtractSubtrees (setNodeSSA: SSA) (leftSSA: SSA) (rightSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract left subtree (index 2)
        let! extractLeftOp = pExtractValue leftSSA setNodeSSA [2]
        // Extract right subtree (index 3)
        let! extractRightOp = pExtractValue rightSSA setNodeSSA [3]
        return [extractLeftOp; extractRightOp]
    }

// ═══════════════════════════════════════════════════════════
// RESULT PATTERNS
// ═══════════════════════════════════════════════════════════

/// Result.Ok: tag=0 + value
let pResultOk (value: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 0L [value] ssas

/// Result.Error: tag=1 + error
let pResultError (error: Val) (ssas: SSA list) : PSGParser<MLIROp list> =
    pDUCase 1L [error] ssas

/// Result.IsOk: extract tag, compare with 0
let pResultIsOk (resultSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractTagOp = pExtractValue tagSSA resultSSA [0]
        let! zeroConstOp = pConstI zeroSSA 0L
        let! cmpOp = Alex.Elements.ArithElements.pCmpI cmpSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Result.IsError: extract tag, compare with 1
let pResultIsError (resultSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractTagOp = pExtractValue tagSSA resultSSA [0]
        let! oneConstOp = pConstI oneSSA 1L
        let! cmpOp = Alex.Elements.ArithElements.pCmpI cmpSSA ICmpPred.Eq tagSSA oneSSA
        return [extractTagOp; oneConstOp; cmpOp]
    }

// ═══════════════════════════════════════════════════════════
// LITERAL PATTERNS
// ═══════════════════════════════════════════════════════════

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

/// Build literal: Match literal from PSG and emit constant MLIR
let pBuildLiteral (lit: NativeLiteral) (ssa: SSA) (arch: Architecture) : PSGParser<MLIROp list * TransferResult> =
    parser {
        match lit with
        | NativeLiteral.Unit ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUunit
            let! op = pConstI ssa 0L
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Bool b ->
            let value = if b then 1L else 0L
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool
            let! op = pConstI ssa value
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Int (n, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa n
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.UInt (n, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa (int64 n)
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Char c ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUchar
            let! op = pConstI ssa (int64 c)
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Float (f, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstF ssa f
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.String _ ->
            // String literals need global string table + AddressOf operation
            // StringCollection coeffect (from PSGElaboration) provides the string table
            // Gap: Need to emit global string data section and AddressOf operations
            return! pfail "String literals need global string table + AddressOf implementation"

        | _ ->
            return! pfail $"Unsupported literal: {lit}"
    }

// ═══════════════════════════════════════════════════════════
// STRING PATTERNS
// ═══════════════════════════════════════════════════════════

/// String as fat pointer: {ptr: nativeptr<byte>, length: int}
/// Extract pointer field (index 0)
let pStringGetPtr (stringSSA: SSA) (ptrSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractPtrOp = pExtractValue ptrSSA stringSSA [0]
        return [extractPtrOp]
    }

/// Extract length field (index 1)
let pStringGetLength (stringSSA: SSA) (lengthSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! extractLenOp = pExtractValue lengthSSA stringSSA [1]
        return [extractLenOp]
    }

/// Construct string from pointer and length
let pStringConstruct (ptr: SSA) (length: SSA) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        if ssas.Length < 3 then
            return! pfail $"pStringConstruct: Expected at least 3 SSAs, got {ssas.Length}"

        let! undefOp = pUndef ssas.[0]
        let! insertPtrOp = pInsertValue ssas.[1] ssas.[0] ptr [0]
        let! insertLenOp = pInsertValue ssas.[2] ssas.[1] length [1]
        return [undefOp; insertPtrOp; insertLenOp]
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC PATTERNS (stub implementations for ArithWitness)
// ═══════════════════════════════════════════════════════════

/// Binary arithmetic operations (+, -, *, /, %)
/// Takes: result SSA, LHS SSA, RHS SSA, architecture
/// Matches intrinsic classification and emits appropriate operation
let pBuildBinaryArith (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, category) = pClassifiedIntrinsic

        let! op =
            parser {
                match category with
                | BinaryArith "addi" -> return! pAddI resultSSA lhsSSA rhsSSA
                | BinaryArith "subi" -> return! pSubI resultSSA lhsSSA rhsSSA
                | BinaryArith "muli" -> return! pMulI resultSSA lhsSSA rhsSSA
                | BinaryArith "divsi" -> return! pDivSI resultSSA lhsSSA rhsSSA
                | BinaryArith "divui" -> return! pDivUI resultSSA lhsSSA rhsSSA
                | BinaryArith "remsi" -> return! pRemSI resultSSA lhsSSA rhsSSA
                | BinaryArith "remui" -> return! pRemUI resultSSA lhsSSA rhsSSA
                | BinaryArith "addf" -> return! pAddF resultSSA lhsSSA rhsSSA
                | BinaryArith "subf" -> return! pSubF resultSSA lhsSSA rhsSSA
                | BinaryArith "mulf" -> return! pMulF resultSSA lhsSSA rhsSSA
                | BinaryArith "divf" -> return! pDivF resultSSA lhsSSA rhsSSA
                | _ -> return! pfail $"Unsupported binary arithmetic intrinsic: {info.FullName}"
            }

        let! state = getUserState
        let ty = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch state.Current.Type

        return ([op], TRValue { SSA = resultSSA; Type = ty })
    }

/// Comparison operations (<, <=, >, >=, ==, !=)
/// Takes: result SSA, LHS SSA, RHS SSA, architecture
/// Emits CmpI or CmpF based on operand types
let pBuildComparison (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, category) = pClassifiedIntrinsic

        let! predicate =
            parser {
                match category with
                | Comparison "slt" -> return ICmpPred.Slt
                | Comparison "sle" -> return ICmpPred.Sle
                | Comparison "sgt" -> return ICmpPred.Sgt
                | Comparison "sge" -> return ICmpPred.Sge
                | Comparison "eq" -> return ICmpPred.Eq
                | Comparison "ne" -> return ICmpPred.Ne
                | Comparison "ult" -> return ICmpPred.Ult
                | Comparison "ule" -> return ICmpPred.Ule
                | Comparison "ugt" -> return ICmpPred.Ugt
                | Comparison "uge" -> return ICmpPred.Uge
                | _ -> return! pfail $"Unsupported comparison intrinsic: {info.FullName}"
            }

        let! ops = pCmpI resultSSA predicate lhsSSA rhsSSA

        // Comparison always returns i1 (boolean)
        let resultType = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool

        return (ops, TRValue { SSA = resultSSA; Type = resultType })
    }

/// Bitwise operations (&, |, ^, <<, >>)
/// Note: These are NOT currently in FNCS intrinsics - need to add them
/// For now, return error indicating FNCS elaboration needed
let pBuildBitwise (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        return! pfail "Bitwise operations not yet in FNCS intrinsics - need elaboration (AND, OR, XOR, SHL, SHR)"
    }

/// Unary operations (-, not, ~)
/// Note: Unary minus is typically represented as 0 - x
/// Logical not is typically xor with true (all 1s)
/// For now, return error indicating these need special handling
let pBuildUnary (resultSSA: SSA) (operandSSA: SSA) (arch: Architecture)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        return! pfail "Unary operations need special handling - negation as (0 - x), not as (xor x, -1)"
    }
