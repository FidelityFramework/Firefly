/// ElisionPatterns - Composable MLIR elision templates via XParsec
///
/// PUBLIC: Witnesses call these patterns to elide PSG structure to MLIR.
/// Patterns compose Elements (internal) into semantic operations.
module Alex.Patterns.ElisionPatterns

open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }, >>=, <|>
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRElements
open Alex.Elements.MemRefElements
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements
open Alex.Elements.SCFElements
open Alex.Elements.CFElements
open Alex.Elements.FuncElements
open Alex.Elements.IndexElements
open Alex.Elements.VectorElements
open PSGElaboration.StringCollection  // For deriveGlobalRef, deriveByteLength
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════
// XParsec HELPERS
// ═══════════════════════════════════════════════════════════

/// Create parser failure with error message
let pfail msg : PSGParser<'a> = fail (Message msg)

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
        let! state = getUserState
        let elemType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        let! gepOp = pStructGEP gepSSA structPtr fieldIndex
        let! storeOp = pStore value gepSSA [] elemType
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
        let! state = getUserState
        let elemType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        let! gepOp = pGEP gepSSA arrayPtr [(index, indexTy)]
        let! storeOp = pStore value gepSSA [] elemType
        return [gepOp; storeOp]
    }

// ═══════════════════════════════════════════════════════════
// STRUCT CONSTRUCTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record struct via Undef + InsertValue chain
let pRecordStruct (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = fields.Length + 1) $"pRecordStruct: Expected {fields.Length + 1} SSAs, got {ssas.Length}"

        // Compute struct type from field types
        let structTy = TStruct (fields |> List.map (fun f -> f.Type))
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

/// DU case construction: tag field (index 0) + payload fields
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
// CLOSURE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Flat closure struct: code_ptr field + capture fields
let pFlatClosure (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2 + captures.Length) $"pFlatClosure: Expected at least {2 + captures.Length} SSAs, got {ssas.Length}"

        // Compute closure type: {code_ptr: ptr, capture0, capture1, ...}
        let closureTy = TStruct (codePtrTy :: (captures |> List.map (fun cap -> cap.Type)))

        // Create undef struct
        let! undefOp = pUndef ssas.[0] closureTy

        // Insert code_ptr at index 0
        let! insertCodeOp = pInsertValue ssas.[1] ssas.[0] codePtr [0] closureTy

        // Insert captures starting at index 1
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[2 + i]
                    let sourceSSA = if i = 0 then ssas.[1] else ssas.[1 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [i + 1] closureTy
                })
            |> sequence

        return undefOp :: insertCodeOp :: captureOps
    }

/// Closure call: extract code_ptr, extract captures, call
let pClosureCall (closureSSA: SSA) (closureTy: MLIRType) (captureTypes: MLIRType list)
                 (args: Val list) (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        do! ensure (extractSSAs.Length = captureCount + 1) $"pClosureCall: Expected {captureCount + 1} extract SSAs, got {extractSSAs.Length}"

        // Extract code_ptr from index 0 (first field is always ptr type)
        let codePtrSSA = extractSSAs.[0]
        let codePtrTy = match closureTy with TStruct (ty :: _) -> ty | _ -> TPtr
        let! extractCodeOp = pExtractValue codePtrSSA closureSSA [0] codePtrTy

        // Extract captures from indices 1..captureCount
        let! extractCaptureOps =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let capSSA = extractSSAs.[i + 1]
                    return! pExtractValue capSSA closureSSA [i + 1] capTy
                })
            |> sequence

        // Call with captures prepended to args
        let captureVals = List.zip (extractSSAs.[1..]) captureTypes |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let allArgs = captureVals @ args
        let! callOp = pIndirectCall resultSSA codePtrSSA (allArgs |> List.map (fun v -> (v.SSA, v.Type)))

        return extractCodeOp :: extractCaptureOps @ [callOp]
    }

// ═══════════════════════════════════════════════════════════
// LAZY PATTERNS (PRD-14)
// ═══════════════════════════════════════════════════════════

/// Lazy struct: {computed: i1, value: T, code_ptr, captures...}
let pLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 4 + captures.Length) $"pLazyStruct: Expected at least {4 + captures.Length} SSAs, got {ssas.Length}"

        // Compute lazy type: {computed: i1, value: T, code_ptr: ptr, captures...}
        let lazyTy = TStruct ([TInt I1; valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type)))

        // Create undef struct
        let! undefOp = pUndef ssas.[0] lazyTy

        // Insert computed = false at index 0
        let computedTy = TInt I1
        let! falseConstOp = pConstI ssas.[1] 0L computedTy
        let! insertComputedOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0] lazyTy

        // Insert code_ptr at index 2
        let! insertCodeOp = pInsertValue ssas.[3] ssas.[2] codePtr [2] lazyTy

        // Insert captures starting at index 3
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[4 + i]
                    let sourceSSA = ssas.[3 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [i + 3] lazyTy
                })
            |> sequence

        return undefOp :: falseConstOp :: insertComputedOp :: insertCodeOp :: captureOps
    }

/// Build lazy struct: High-level pattern for witnesses
/// Combines pLazyStruct with proper result construction
let pBuildLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list) 
                     (ssas: SSA list) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        let! ops = pLazyStruct valueTy codePtrTy codePtr captures ssas

        // Final SSA is the last one (after all insertions)
        let finalSSA = ssas.[3 + captures.Length]

        // Lazy type is {computed: i1, value: T, code_ptr, captures...}
        let mlirType = TStruct ([TInt I1; valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type)))

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
let pBuildLazyForce (lazySSA: SSA) (lazyTy: MLIRType) (resultSSA: SSA) (resultTy: MLIRType)
                    (ssas: SSA list) (arch: Architecture)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        // SSAs: [0] = code_ptr, [1] = const 1, [2] = alloca'd ptr
        do! ensure (ssas.Length >= 3) $"pBuildLazyForce: Expected at least 3 SSAs, got {ssas.Length}"

        let codePtrSSA = ssas.[0]
        let constOneSSA = ssas.[1]
        let ptrSSA = ssas.[2]

        // Extract code_ptr from lazy struct [2] - it's always a ptr type
        let codePtrTy = TPtr
        let! extractCodePtrOp = pExtractValue codePtrSSA lazySSA [2] codePtrTy

        // Alloca space for lazy struct
        let constOneTy = TInt I64
        let! constOneOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca ptrSSA lazyTy None

        // Store lazy struct to alloca'd space
        let! storeOp = pStore lazySSA ptrSSA [] lazyTy

        // Call thunk with pointer -> result
        let! callOp = pIndirectCall resultSSA codePtrSSA [(ptrSSA, TPtr)]

        return ([extractCodePtrOp; constOneOp; allocaOp; storeOp; callOp], TRValue { SSA = resultSSA; Type = resultTy })
    }

// ═══════════════════════════════════════════════════════════
// SEQ PATTERNS (PRD-15)
// ═══════════════════════════════════════════════════════════

/// Seq struct: {state: i32, current: T, code_ptr, captures..., internal_state...}
let pSeqStruct (stateInit: int64) (currentTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA)
               (captures: Val list) (internalState: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let minSSAs = 4 + captures.Length + internalState.Length
        do! ensure (ssas.Length >= minSSAs) $"pSeqStruct: Expected at least {minSSAs} SSAs, got {ssas.Length}"

        // Compute seq type: {state: i32, current: T, code_ptr: ptr, captures..., internal...}
        let seqTy = TStruct ([TInt I32; currentTy; codePtrTy] 
                             @ (captures |> List.map (fun cap -> cap.Type)) 
                             @ (internalState |> List.map (fun st -> st.Type)))

        // Create undef struct
        let! undefOp = pUndef ssas.[0] seqTy

        // Insert state = stateInit at index 0
        let stateTy = TInt I32
        let! stateConstOp = pConstI ssas.[1] stateInit stateTy
        let! insertStateOp = pInsertValue ssas.[2] ssas.[0] ssas.[1] [0] seqTy

        // Insert code_ptr at index 2
        let! insertCodeOp = pInsertValue ssas.[3] ssas.[2] codePtr [2] seqTy

        // Insert captures starting at index 3
        let captureBaseIdx = 3
        let! captureOps =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let targetSSA = ssas.[4 + i]
                    let sourceSSA = ssas.[3 + i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA [captureBaseIdx + i] seqTy
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
                    return! pInsertValue targetSSA sourceSSA st.SSA [internalBaseIdx + i] seqTy
                })
            |> sequence

        return undefOp :: stateConstOp :: insertStateOp :: insertCodeOp
               :: (captureOps @ internalOps)
    }

/// Seq MoveNext: extract state, load captures/internal, call code_ptr, update state/current
let pSeqMoveNext (seqSSA: SSA) (seqTy: MLIRType) (captureTypes: MLIRType list)
                 (internalTypes: MLIRType list) (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        let internalCount = internalTypes.Length
        let expectedExtracts = 2 + captureCount + internalCount  // state, code_ptr, captures, internal
        do! ensure (extractSSAs.Length >= expectedExtracts) $"pSeqMoveNext: Expected at least {expectedExtracts} extract SSAs, got {extractSSAs.Length}"

        // Extract state from index 0
        let stateSSA = extractSSAs.[0]
        let stateTy = TInt I32
        let! extractStateOp = pExtractValue stateSSA seqSSA [0] stateTy

        // Extract code_ptr from index 2
        let codePtrSSA = extractSSAs.[1]
        let codePtrTy = TPtr
        let! extractCodeOp = pExtractValue codePtrSSA seqSSA [2] codePtrTy

        // Extract captures from indices 3..3+captureCount
        let captureBaseIdx = 3
        let! extractCaptureOps =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let capSSA = extractSSAs.[2 + i]
                    return! pExtractValue capSSA seqSSA [captureBaseIdx + i] capTy
                })
            |> sequence

        // Extract internal state
        let internalBaseIdx = captureBaseIdx + captureCount
        let! extractInternalOps =
            internalTypes
            |> List.mapi (fun i intTy ->
                parser {
                    let intSSA = extractSSAs.[2 + captureCount + i]
                    return! pExtractValue intSSA seqSSA [internalBaseIdx + i] intTy
                })
            |> sequence

        // Call code_ptr with state + captures + internal
        let stateArg = (stateSSA, stateTy)
        let codePtrArg = (codePtrSSA, codePtrTy)
        let captureArgs = List.zip (extractSSAs.[2..2+captureCount-1] |> List.ofSeq) captureTypes
        let internalArgs = List.zip (extractSSAs.[2+captureCount..2+captureCount+internalCount-1] |> List.ofSeq) internalTypes
        let allArgs = stateArg :: codePtrArg :: (captureArgs @ internalArgs)
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
let pBuildSeqStruct (currentTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) 
                    (captures: Val list) (internalState: Val list)
                    (ssas: SSA list) (arch: Architecture)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        // Initial state is 0 (unstarted)
        let! ops = pSeqStruct 0L currentTy codePtrTy codePtr captures internalState ssas

        // Final SSA is the last one (after all insertions)
        let finalSSA = ssas.[3 + captures.Length + internalState.Length]

        // Seq type is {state: i32, current: T, code_ptr: ptr, captures..., internal...}
        let mlirType = TStruct ([TInt I32; currentTy; codePtrTy] 
                                @ (captures |> List.map (fun cap -> cap.Type)) 
                                @ (internalState |> List.map (fun st -> st.Type)))

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
let pOptionSome (value: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [value] ssas ty

/// Option.None: tag=0
let pOptionNone (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Option.IsSome: extract tag, compare with 1
let pOptionIsSome (optionSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA optionSSA [0] tagTy
        let! oneConstOp = pConstI oneSSA 1L tagTy
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA oneSSA
        return [extractTagOp; oneConstOp; cmpOp]
    }

/// Option.IsNone: extract tag, compare with 0
let pOptionIsNone (optionSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA optionSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Option.Get: extract value field (index 1)
let pOptionGet (optionSSA: SSA) (resultSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let! extractOp = pExtractValue resultSSA optionSSA [1] valueTy
        return [extractOp]
    }

// ═══════════════════════════════════════════════════════════
// LIST PATTERNS
// ═══════════════════════════════════════════════════════════

/// List.Cons: tag=1 + head + tail
let pListCons (head: Val) (tail: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [head; tail] ssas ty

/// List.Empty: tag=0
let pListEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// List.IsEmpty: extract tag, compare with 0
let pListIsEmpty (listSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA listSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

// ═══════════════════════════════════════════════════════════
// MAP PATTERNS
// ═══════════════════════════════════════════════════════════

/// Map.Empty: empty tree structure (tag=0 for leaf node)
let pMapEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Map.Add: create new tree node with key, value, left, right subtrees
/// Structure: tag=1, key, value, left_tree, right_tree
let pMapAdd (key: Val) (value: Val) (left: Val) (right: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [key; value; left; right] ssas ty

/// Map.ContainsKey: tree traversal comparing keys
/// This is a composite operation that would be implemented as a recursive function
/// The pattern here just shows the key comparison at each node
let pMapKeyCompare (mapNodeSSA: SSA) (searchKey: SSA) (keySSA: SSA) (keyTy: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract key from node (index 1, after tag at index 0)
        let! extractKeyOp = pExtractValue keySSA mapNodeSSA [1] keyTy
        // Compare search key with node key
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq searchKey keySSA
        return [extractKeyOp; cmpOp]
    }

/// Map.TryFind: similar to ContainsKey but returns Option<value>
let pMapExtractValue (mapNodeSSA: SSA) (valueSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        // Extract value from node (index 2, after tag and key)
        let! extractValueOp = pExtractValue valueSSA mapNodeSSA [2] valueTy
        return [extractValueOp]
    }

// ═══════════════════════════════════════════════════════════
// SET PATTERNS
// ═══════════════════════════════════════════════════════════

/// Set.Empty: empty tree structure (tag=0 for leaf node)
let pSetEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Set.Add: create new tree node with element, left, right subtrees
/// Structure: tag=1, element, left_tree, right_tree
let pSetAdd (element: Val) (left: Val) (right: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [element; left; right] ssas ty

/// Set.Contains: tree traversal comparing elements
let pSetElementCompare (setNodeSSA: SSA) (searchElem: SSA) (elemSSA: SSA) (elemTy: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract element from node (index 1, after tag at index 0)
        let! extractElemOp = pExtractValue elemSSA setNodeSSA [1] elemTy
        // Compare search element with node element
        let! cmpOp = Alex.Elements.ArithElements.pCmpI resultSSA ICmpPred.Eq searchElem elemSSA
        return [extractElemOp; cmpOp]
    }

/// Set.Union: combines two sets (implemented as tree merge operation)
/// Pattern shows extraction of subtrees for recursive union
let pSetExtractSubtrees (setNodeSSA: SSA) (leftSSA: SSA) (rightSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        // Extract left subtree (index 2)
        let! extractLeftOp = pExtractValue leftSSA setNodeSSA [2] subtreeTy
        // Extract right subtree (index 3)
        let! extractRightOp = pExtractValue rightSSA setNodeSSA [3] subtreeTy
        return [extractLeftOp; extractRightOp]
    }

// ═══════════════════════════════════════════════════════════
// RESULT PATTERNS
// ═══════════════════════════════════════════════════════════

/// Result.Ok: tag=0 + value
let pResultOk (value: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [value] ssas ty

/// Result.Error: tag=1 + error
let pResultError (error: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [error] ssas ty

/// Result.IsOk: extract tag, compare with 0
let pResultIsOk (resultSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA resultSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = Alex.Elements.ArithElements.pCmpI cmpSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Result.IsError: extract tag, compare with 1
let pResultIsError (resultSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA resultSSA [0] tagTy
        let! oneConstOp = pConstI oneSSA 1L tagTy
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
            let! op = pConstI ssa 0L ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Bool b ->
            let value = if b then 1L else 0L
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool
            let! op = pConstI ssa value ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Int (n, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa n ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.UInt (n, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa (int64 n) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Char c ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUchar
            let! op = pConstI ssa (int64 c) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Float (f, kind) ->
            let ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch kind
            let! op = pConstF ssa f ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.String _ ->
            // String literals require witness-level handling with multiple SSAs
            // Use pBuildStringLiteral pattern instead
            return! pfail "String literals require pBuildStringLiteral pattern with SSA list"

        | _ ->
            return! pfail $"Unsupported literal: {lit}"
    }

// ═══════════════════════════════════════════════════════════
// STRING PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build string literal: addressof global + construct fat pointer struct
/// SSAs: [0] = addressof ptr, [1] = length const, [2] = undef, [3] = insert ptr, [4] = insert length (result)
/// Returns: ((ops, globalName, content, byteLength), result)
/// NOTE: Witness must emit GlobalString to TopLevelOps separately
let pBuildStringLiteral (content: string) (ssas: SSA list) (arch: Architecture)
                         : PSGParser<(MLIROp list * string * string * int) * TransferResult> =
    parser {
        do! emitTrace "pBuildStringLiteral.entry" (sprintf "content='%s', ssas=%A, arch=%A" content ssas arch)

        // Declarative guard - no imperative if statements
        do! ensure (ssas.Length >= 5) $"pBuildStringLiteral: Expected 5 SSAs, got {ssas.Length}"

        do! emitTrace "pBuildStringLiteral.ssa_validated" (sprintf "SSA count OK: %d" ssas.Length)

        // Use StringCollection pure derivation (coeffect model)
        let globalName = deriveGlobalRef content
        let byteLength = deriveByteLength content

        do! emitTrace "pBuildStringLiteral.derived" (sprintf "globalName=%s, byteLength=%d" globalName byteLength)

        // String type: {ptr: nativeptr<byte>, length: int}
        let ptrTy = TPtr
        let lengthTy = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUint64
        let stringTy = TStruct [ptrTy; lengthTy]

        do! emitTrace "pBuildStringLiteral.types" (sprintf "ptrTy=%A, lengthTy=%A, stringTy=%A" ptrTy lengthTy stringTy)

        // InlineOps: Build string struct {ptr, length}
        let ptrSSA = ssas.[0]
        let lengthSSA = ssas.[1]
        let undefSSA = ssas.[2]
        let withPtrSSA = ssas.[3]
        let resultSSA = ssas.[4]

        do! emitTrace "pBuildStringLiteral.ssas_extracted" (sprintf "ptr=%A, len=%A, undef=%A, withPtr=%A, result=%A" ptrSSA lengthSSA undefSSA withPtrSSA resultSSA)

        do! emitTrace "pBuildStringLiteral.calling_pAddressOf" (sprintf "ptrSSA=%A, globalName=%s, ptrTy=%A" ptrSSA globalName ptrTy)
        let! addressOfOp = pAddressOf ptrSSA globalName ptrTy
        
        do! emitTrace "pBuildStringLiteral.calling_pConstI" (sprintf "lengthSSA=%A, byteLength=%d, lengthTy=%A" lengthSSA byteLength lengthTy)
        let! lengthConstOp = pConstI lengthSSA (int64 byteLength) lengthTy
        
        do! emitTrace "pBuildStringLiteral.calling_pUndef" (sprintf "undefSSA=%A, stringTy=%A" undefSSA stringTy)
        let! undefOp = pUndef undefSSA stringTy
        
        do! emitTrace "pBuildStringLiteral.calling_pInsertValue_ptr" (sprintf "withPtrSSA=%A, undefSSA=%A, ptrSSA=%A" withPtrSSA undefSSA ptrSSA)
        let! insertPtrOp = pInsertValue withPtrSSA undefSSA ptrSSA [0] stringTy
        
        do! emitTrace "pBuildStringLiteral.calling_pInsertValue_len" (sprintf "resultSSA=%A, withPtrSSA=%A, lengthSSA=%A" resultSSA withPtrSSA lengthSSA)
        let! insertLenOp = pInsertValue resultSSA withPtrSSA lengthSSA [1] stringTy

        do! emitTrace "pBuildStringLiteral.elements_complete" "All Elements succeeded"

        let inlineOps = [addressOfOp; lengthConstOp; undefOp; insertPtrOp; insertLenOp]
        let result = TRValue { SSA = resultSSA; Type = stringTy }

        do! emitTrace "pBuildStringLiteral.returning" (sprintf "Returning %d ops" (List.length inlineOps))

        // Return ops + (globalName, content, byteLength) for witness to emit GlobalString
        return ((inlineOps, globalName, content, byteLength), result)
    }

/// String as fat pointer: {ptr: nativeptr<byte>, length: int}
/// Extract pointer field (index 0)
let pStringGetPtr (stringSSA: SSA) (ptrSSA: SSA) (ptrTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let! extractPtrOp = pExtractValue ptrSSA stringSSA [0] ptrTy
        return [extractPtrOp]
    }

/// Extract length field (index 1)
let pStringGetLength (stringSSA: SSA) (lengthSSA: SSA) (lengthTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let! extractLenOp = pExtractValue lengthSSA stringSSA [1] lengthTy
        return [extractLenOp]
    }

/// Construct string from pointer and length
let pStringConstruct (ptrTy: MLIRType) (lengthTy: MLIRType) (ptr: SSA) (length: SSA) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 3) $"pStringConstruct: Expected at least 3 SSAs, got {ssas.Length}"

        // String type is {ptr: nativeptr<byte>, length: int}
        let stringTy = TStruct [ptrTy; lengthTy]
        let! undefOp = pUndef ssas.[0] stringTy
        let! insertPtrOp = pInsertValue ssas.[1] ssas.[0] ptr [0] stringTy
        let! insertLenOp = pInsertValue ssas.[2] ssas.[1] length [1] stringTy
        return [undefOp; insertPtrOp; insertLenOp]
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC PATTERNS (stub implementations for ArithWitness)
// ═══════════════════════════════════════════════════════════

/// Binary arithmetic operations (+, -, *, /, %)
/// Takes: result SSA, LHS SSA, RHS SSA, architecture
/// Matches atomic operation classification and emits appropriate operation
let pBuildBinaryArith (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, category) = pClassifiedAtomicOp

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
                | _ -> return! pfail $"Unsupported binary arithmetic atomic operation: {info.FullName}"
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
        let! (info, category) = pClassifiedAtomicOp

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
                | _ -> return! pfail $"Unsupported comparison atomic operation: {info.FullName}"
            }

        let! ops = pCmpI resultSSA predicate lhsSSA rhsSSA

        // Comparison always returns i1 (boolean)
        let resultType = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType arch NTUKind.NTUbool

        return (ops, TRValue { SSA = resultSSA; Type = resultType })
    }

/// Bitwise operations (&, |, ^, <<, >>)
/// Note: These are NOT currently in FNCS as atomic operations - need to add them
/// For now, return error indicating FNCS elaboration needed
let pBuildBitwise (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        return! pfail "Bitwise operations not yet in FNCS as atomic operations - need elaboration (AND, OR, XOR, SHL, SHR)"
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

// ═══════════════════════════════════════════════════════════
// APPLICATION PATTERNS (Function Calls)
// ═══════════════════════════════════════════════════════════

/// Build function application (indirect call via function pointer)
/// For known function names, use pDirectCall instead (future optimization)
let pApplicationCall (resultSSA: SSA) (funcSSA: SSA) (args: (SSA * MLIRType) list) (retType: MLIRType)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit indirect call via function pointer
        let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (resultSSA, funcSSA, args, retType))
        return ([callOp], TRValue { SSA = resultSSA; Type = retType })
    }

/// Build direct function call (for known function names - portable)
/// Uses func.call (portable) instead of llvm.call (backend-specific)
let pDirectCall (resultSSA: SSA) (funcName: string) (args: (SSA * MLIRType) list) (retType: MLIRType)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Convert (SSA * MLIRType) list to Val list for pFuncCall
        let vals = args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let! callOp = pFuncCall (Some resultSSA) funcName vals retType
        return ([callOp], TRValue { SSA = resultSSA; Type = retType })
    }
