/// ClosurePatterns - Closure and lambda operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for lambda and closure operations.
/// Patterns compose Elements into semantic closure/lambda operations.
module Alex.Patterns.ClosurePatterns

open XParsec
open XParsec.Parsers     // preturn, fail
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.FuncElements
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// ARENA ALLOCATION
// ═══════════════════════════════════════════════════════════

/// Allocate in global closure heap arena (bump allocator)
/// SSAs: [0] = heap_pos_ptr, [1] = heap_pos, [2] = heap_base, [3] = result_ptr, [4] = new_pos, [5] = index
/// Returns ops and the result pointer SSA
let pAllocateInArena (sizeSSA: SSA) (ssas: SSA list) : PSGParser<MLIROp list * SSA> =
    parser {
        do! ensure (ssas.Length >= 6) $"pAllocateInArena: Expected 6 SSAs, got {ssas.Length}"

        let heapPosPtrSSA = ssas.[0]
        let heapPosSSA = ssas.[1]
        let heapBaseSSA = ssas.[2]
        let resultPtrSSA = ssas.[3]
        let newPosSSA = ssas.[4]
        let indexSSA = ssas.[5]

        // Generate index constant for memref operations (MLIR requires indices)
        let! indexOp = pConstI indexSSA 0L TIndex

        // Load current position
        let! loadPosOp = pLoad heapPosSSA heapPosPtrSSA [indexSSA]

        // Compute result pointer: heap_base + pos
        let! subViewOp = pSubView resultPtrSSA heapBaseSSA [heapPosSSA]

        // Update position: pos + size
        let! addOp = pAddI newPosSSA heapPosSSA sizeSSA (TInt I64)
        let memrefType = TMemRefStatic (1, TInt I64)  // 1-element heap position storage
        let! storePosOp = pStore newPosSSA heapPosPtrSSA [indexSSA] (TInt I64) memrefType

        return ([indexOp; loadPosOp; subViewOp; addOp; storePosOp], resultPtrSSA)
    }

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

/// Create function definition (func.func for named calls, llvm.func for closures)
/// isClosure: true for llvm.func (address taken), false for func.func (named calls)
let pFunctionDef (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                 (body: MLIROp list) (isClosure: bool) : PSGParser<MLIROp> =
    parser {
        // For now, always use func.func
        // The distinction between func.func and llvm.func will be handled by the witness
        let visibility = if name = "main" then FuncVisibility.Public else FuncVisibility.Private
        return! pFuncDef name params' retTy body visibility
    }

// ═══════════════════════════════════════════════════════════
// CAPTURE EXTRACTION
// ═══════════════════════════════════════════════════════════

/// Extract captures from closure struct at function entry
/// SSA layout: [0] = indexZero, [1] = structLoad, then for each capture: [2*i+2] = offsetSSA, [2*i+3] = resultSSA
let pExtractCaptures (baseIndex: int) (captureTypes: MLIRType list) (structType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        do! ensure (ssas.Length >= 2 + 2 * captureCount) $"pExtractCaptures: Expected {2 + 2 * captureCount} SSAs, got {ssas.Length}"

        let indexZeroSSA = ssas.[0]
        let structLoadSSA = ssas.[1]
        let envPtrSSA = Arg 0  // First argument is always env_ptr for closures

        // Load struct from env_ptr (MLIR memrefs require indices)
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex
        let! loadOp = pLoad structLoadSSA envPtrSSA [indexZeroSSA]

        // Extract each capture
        let! extractOpLists =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let offsetSSA = ssas.[2 + 2*i]
                    let extractSSA = ssas.[3 + 2*i]
                    let extractIndex = baseIndex + i
                    return! pExtractValue extractSSA structLoadSSA extractIndex offsetSSA capTy
                })
            |> sequence

        return [indexZeroOp; loadOp] @ List.concat extractOpLists
    }

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// CLOSURE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Flat closure struct: code_ptr field + capture fields
/// SSA layout: [0] = undef, [1-2] = insert code_ptr (offset, result), then for each capture: [3+2*i] = offsetSSA, [4+2*i] = resultSSA
let pFlatClosure (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 3 + 2 * captures.Length) $"pFlatClosure: Expected at least {3 + 2 * captures.Length} SSAs, got {ssas.Length}"

        // Compute closure type: {code_ptr: ptr, capture0, capture1, ...}
        let fieldTypes = codePtrTy :: (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let closureTy = TMemRefStatic(totalBytes, TInt I8)

        // Create undef struct
        let! undefOp = pUndef ssas.[0] closureTy

        // Insert code_ptr at index 0
        let codeOffsetSSA = ssas.[1]
        let codeResultSSA = ssas.[2]
        let! insertCodeOps = pInsertValue codeResultSSA ssas.[0] codePtr 0 codeOffsetSSA closureTy

        // Insert captures starting at index 1
        let! captureOpLists =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let offsetSSA = ssas.[3 + 2*i]
                    let targetSSA = ssas.[4 + 2*i]
                    let sourceSSA = if i = 0 then codeResultSSA else ssas.[2 + 2*i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA (i + 1) offsetSSA closureTy
                })
            |> sequence

        return [undefOp] @ insertCodeOps @ List.concat captureOpLists
    }

/// Closure call: extract code_ptr, extract captures, call
/// SSA layout: [0-1] = code_ptr extract (offset, result), then for each capture: [2+2*i] = offsetSSA, [3+2*i] = resultSSA
let pClosureCall (closureSSA: SSA) (closureTy: MLIRType) (captureTypes: MLIRType list)
                 (args: Val list) (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        do! ensure (extractSSAs.Length >= 2 + 2 * captureCount) $"pClosureCall: Expected {2 + 2 * captureCount} extract SSAs, got {extractSSAs.Length}"

        // Extract code_ptr from index 0 (first field is always ptr type)
        let codeOffsetSSA = extractSSAs.[0]
        let codePtrSSA = extractSSAs.[1]
        let codePtrTy = TIndex  // Code pointer type
        let! extractCodeOps = pExtractValue codePtrSSA closureSSA 0 codeOffsetSSA codePtrTy

        // Extract captures from indices 1..captureCount
        let! extractCaptureOpLists =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let offsetSSA = extractSSAs.[2 + 2*i]
                    let capSSA = extractSSAs.[3 + 2*i]
                    return! pExtractValue capSSA closureSSA (i + 1) offsetSSA capTy
                })
            |> sequence

        // Call with captures prepended to args
        let captureSSAs = List.init captureCount (fun i -> extractSSAs.[3 + 2*i])
        let captureVals = List.zip captureSSAs captureTypes |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let allArgs = captureVals @ args
        let! state = getUserState
        let retType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA allArgs retType

        return extractCodeOps @ List.concat extractCaptureOpLists @ [callOp]
    }

// ═══════════════════════════════════════════════════════════
// LAMBDA CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// Build simple lambda (no captures) - just creates function definition
/// Returns: (topLevelOps, inlineOps, resultSSA)
/// - topLevelOps: function definition
/// - inlineOps: empty (no struct to construct)
/// - resultSSA: function reference
let pBuildSimpleLambda (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                       (bodyOps: MLIROp list) (funcRefSSA: SSA) : PSGParser<MLIROp list * MLIROp list * SSA> =
    parser {
        let! funcDefOp = pFunctionDef name params' retTy bodyOps false
        return [funcDefOp], [], funcRefSSA
    }

/// Build closure lambda (with captures) - creates function definition + closure struct
/// Returns: (topLevelOps, inlineOps, resultSSA)
/// - topLevelOps: function definition
/// - inlineOps: closure struct construction (via pFlatClosure)
/// - resultSSA: closure struct value
let pBuildClosureLambda (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                        (bodyOps: MLIROp list) (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list)
                        (ssas: SSA list) : PSGParser<MLIROp list * MLIROp list * SSA> =
    parser {
        // Create function definition (with env_ptr as first parameter for captures)
        let envPtrParam = Arg 0, TIndex
        let allParams = envPtrParam :: params'
        let! funcDefOp = pFunctionDef name allParams retTy bodyOps true

        // Create closure struct (delegates to pFlatClosure)
        let! structOps = pFlatClosure codePtr codePtrTy captures ssas

        return [funcDefOp], structOps, List.last ssas
    }

// ═══════════════════════════════════════════════════════════
// LAZY PATTERNS (PRD-14)
// ═══════════════════════════════════════════════════════════

/// Lazy struct: {computed: i1, value: T, code_ptr, captures...}
/// SSA layout: [0] = undef, [1] = falseConst, [2-3] = insert computed (offset, result), [4-5] = insert code_ptr (offset, result), then for each capture: [6+2*i] = offsetSSA, [7+2*i] = resultSSA
let pLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 6 + 2 * captures.Length) $"pLazyStruct: Expected at least {6 + 2 * captures.Length} SSAs, got {ssas.Length}"

        // Compute lazy type: {computed: i1, value: T, code_ptr: ptr, captures...}
        let fieldTypes = [TInt I1; valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let lazyTy = TMemRefStatic(totalBytes, TInt I8)

        // Create undef struct
        let! undefOp = pUndef ssas.[0] lazyTy

        // Insert computed = false at index 0
        let computedTy = TInt I1
        let! falseConstOp = pConstI ssas.[1] 0L computedTy
        let! insertComputedOps = pInsertValue ssas.[3] ssas.[0] ssas.[1] 0 ssas.[2] lazyTy

        // Insert code_ptr at index 2
        let! insertCodeOps = pInsertValue ssas.[5] ssas.[3] codePtr 2 ssas.[4] lazyTy

        // Insert captures starting at index 3
        let! captureOpLists =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let offsetSSA = ssas.[6 + 2*i]
                    let targetSSA = ssas.[7 + 2*i]
                    let sourceSSA = if i = 0 then ssas.[5] else ssas.[5 + 2*i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA (i + 3) offsetSSA lazyTy
                })
            |> sequence

        return [undefOp; falseConstOp] @ insertComputedOps @ insertCodeOps @ List.concat captureOpLists
    }

/// Build lazy struct: High-level pattern for witnesses
/// Combines pLazyStruct with proper result construction
let pBuildLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list)
                     (ssas: SSA list) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        let! ops = pLazyStruct valueTy codePtrTy codePtr captures ssas

        // Final SSA is the last one (after all insertions: undef + falseConst + 2*insertComputed + 2*insertCode + 2*captures)
        let finalSSA = ssas.[5 + 2 * captures.Length]

        // Lazy type is {computed: i1, value: T, code_ptr, captures...}
        let fieldTypes = [TInt I1; valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let mlirType = TMemRefStatic(totalBytes, TInt I8)

        return (ops, TRValue { SSA = finalSSA; Type = mlirType })
    }

/// Build lazy force: Call lazy thunk via struct pointer passing
///
/// LazyForce is a SIMPLE operation (not elaborated by FNCS).
/// SSA cost: Fixed 5 (extract code_ptr, const 1, alloca, index, store, call)
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
        // SSAs: [0-1] = code_ptr extract (offset, result), [2] = const 1, [3] = alloca'd ptr, [4] = index
        do! ensure (ssas.Length >= 5) $"pBuildLazyForce: Expected at least 5 SSAs, got {ssas.Length}"

        let codeOffsetSSA = ssas.[0]
        let codePtrSSA = ssas.[1]
        let constOneSSA = ssas.[2]
        let ptrSSA = ssas.[3]
        let indexSSA = ssas.[4]

        // Extract code_ptr from lazy struct [2] - it's always a ptr type
        let codePtrTy = TIndex
        let! extractCodePtrOps = pExtractValue codePtrSSA lazySSA 2 codeOffsetSSA codePtrTy

        // Alloca space for lazy struct
        let constOneTy = TInt I64
        let! constOneOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca ptrSSA 1 lazyTy None

        // Store lazy struct to alloca'd space
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, lazyTy)  // 1-element lazy value storage
        let! storeOp = pStore lazySSA ptrSSA [indexSSA] lazyTy memrefType

        // Call thunk with pointer -> result
        let argVals = [{ SSA = ptrSSA; Type = TIndex }]
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA argVals resultTy

        return (extractCodePtrOps @ [constOneOp; allocaOp; indexOp; storeOp; callOp], TRValue { SSA = resultSSA; Type = resultTy })
    }

// ═══════════════════════════════════════════════════════════
// SEQ PATTERNS (PRD-15)
// ═══════════════════════════════════════════════════════════

/// Seq struct: {state: i32, current: T, code_ptr, captures..., internal_state...}
/// SSA layout: [0] = undef, [1] = stateConst, [2-3] = insert state (offset, result), [4-5] = insert code_ptr (offset, result),
///             then for each capture: [6+2*i] = offsetSSA, [7+2*i] = resultSSA,
///             then for each internal: [6+2*nCaptures+2*i] = offsetSSA, [7+2*nCaptures+2*i] = resultSSA
let pSeqStruct (stateInit: int64) (currentTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA)
               (captures: Val list) (internalState: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let minSSAs = 6 + 2 * captures.Length + 2 * internalState.Length
        do! ensure (ssas.Length >= minSSAs) $"pSeqStruct: Expected at least {minSSAs} SSAs, got {ssas.Length}"

        // Compute seq type: {state: i32, current: T, code_ptr: ptr, captures..., internal...}
        let fieldTypes = [TInt I32; currentTy; codePtrTy]
                         @ (captures |> List.map (fun cap -> cap.Type))
                         @ (internalState |> List.map (fun st -> st.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let seqTy = TMemRefStatic(totalBytes, TInt I8)

        // Create undef struct
        let! undefOp = pUndef ssas.[0] seqTy

        // Insert state = stateInit at index 0
        let stateTy = TInt I32
        let! stateConstOp = pConstI ssas.[1] stateInit stateTy
        let! insertStateOps = pInsertValue ssas.[3] ssas.[0] ssas.[1] 0 ssas.[2] seqTy

        // Insert code_ptr at index 2
        let! insertCodeOps = pInsertValue ssas.[5] ssas.[3] codePtr 2 ssas.[4] seqTy

        // Insert captures starting at index 3
        let captureBaseIdx = 3
        let! captureOpLists =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let offsetSSA = ssas.[6 + 2*i]
                    let targetSSA = ssas.[7 + 2*i]
                    let sourceSSA = if i = 0 then ssas.[5] else ssas.[5 + 2*i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA (captureBaseIdx + i) offsetSSA seqTy
                })
            |> sequence

        // Insert internal state starting after captures
        let internalBaseIdx = captureBaseIdx + captures.Length
        let internalStartIdx = 6 + 2 * captures.Length
        let! internalOpLists =
            internalState
            |> List.mapi (fun i st ->
                parser {
                    let offsetSSA = ssas.[internalStartIdx + 2*i]
                    let targetSSA = ssas.[internalStartIdx + 1 + 2*i]
                    let sourceSSA = if i = 0 then ssas.[5 + 2 * captures.Length] else ssas.[internalStartIdx - 1 + 2*i]
                    return! pInsertValue targetSSA sourceSSA st.SSA (internalBaseIdx + i) offsetSSA seqTy
                })
            |> sequence

        return [undefOp; stateConstOp] @ insertStateOps @ insertCodeOps @ List.concat captureOpLists @ List.concat internalOpLists
    }

/// Seq MoveNext: extract state, load captures/internal, call code_ptr, update state/current
/// SSA layout: [0-1] = state extract (offset, result), [2-3] = code_ptr extract (offset, result),
///             then for each capture: [4+2*i] = offsetSSA, [5+2*i] = resultSSA,
///             then for each internal: [4+2*nCaptures+2*i] = offsetSSA, [5+2*nCaptures+2*i] = resultSSA
let pSeqMoveNext (seqSSA: SSA) (seqTy: MLIRType) (captureTypes: MLIRType list)
                 (internalTypes: MLIRType list) (extractSSAs: SSA list) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        let internalCount = internalTypes.Length
        let expectedExtracts = 4 + 2 * captureCount + 2 * internalCount
        do! ensure (extractSSAs.Length >= expectedExtracts) $"pSeqMoveNext: Expected at least {expectedExtracts} extract SSAs, got {extractSSAs.Length}"

        // Extract state from index 0
        let stateOffsetSSA = extractSSAs.[0]
        let stateSSA = extractSSAs.[1]
        let stateTy = TInt I32
        let! extractStateOps = pExtractValue stateSSA seqSSA 0 stateOffsetSSA stateTy

        // Extract code_ptr from index 2
        let codeOffsetSSA = extractSSAs.[2]
        let codePtrSSA = extractSSAs.[3]
        let codePtrTy = TIndex
        let! extractCodeOps = pExtractValue codePtrSSA seqSSA 2 codeOffsetSSA codePtrTy

        // Extract captures from indices 3..3+captureCount
        let captureBaseIdx = 3
        let! extractCaptureOpLists =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let offsetSSA = extractSSAs.[4 + 2*i]
                    let capSSA = extractSSAs.[5 + 2*i]
                    return! pExtractValue capSSA seqSSA (captureBaseIdx + i) offsetSSA capTy
                })
            |> sequence

        // Extract internal state
        let internalBaseIdx = captureBaseIdx + captureCount
        let internalStartIdx = 4 + 2 * captureCount
        let! extractInternalOpLists =
            internalTypes
            |> List.mapi (fun i intTy ->
                parser {
                    let offsetSSA = extractSSAs.[internalStartIdx + 2*i]
                    let intSSA = extractSSAs.[internalStartIdx + 1 + 2*i]
                    return! pExtractValue intSSA seqSSA (internalBaseIdx + i) offsetSSA intTy
                })
            |> sequence

        // Call code_ptr with state + captures + internal
        let captureSSAs = List.init captureCount (fun i -> extractSSAs.[5 + 2*i])
        let internalSSAs = List.init internalCount (fun i -> extractSSAs.[internalStartIdx + 1 + 2*i])
        let stateArg = { SSA = stateSSA; Type = stateTy }
        let captureArgs = List.zip captureSSAs captureTypes |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let internalArgs = List.zip internalSSAs internalTypes |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let allArgVals = stateArg :: (captureArgs @ internalArgs)
        let! state = getUserState
        let retType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA allArgVals retType

        return extractStateOps @ extractCodeOps @ List.concat extractCaptureOpLists @ List.concat extractInternalOpLists @ [callOp]
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

        // Final SSA is the last one (after all insertions: undef + stateConst + 2*insertState + 2*insertCode + 2*captures + 2*internals)
        let finalSSA = ssas.[5 + 2 * captures.Length + 2 * internalState.Length]

        // Seq type is {state: i32, current: T, code_ptr: ptr, captures..., internal...}
        let fieldTypes = [TInt I32; currentTy; codePtrTy]
                         @ (captures |> List.map (fun cap -> cap.Type))
                         @ (internalState |> List.map (fun st -> st.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let mlirType = TMemRefStatic(totalBytes, TInt I8)

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
        return! fail (Message "ForEach MoveNext implementation gap - needs: extract code_ptr[2], alloca/store seq, indirect call")
    }
