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
        let! addressOfPosOp = pLoad heapPosPtrSSA heapPosPtrSSA [indexSSA]  // TODO: Placeholder - need AddressOf
        let! loadPosOp = pLoad heapPosSSA heapPosPtrSSA [indexSSA]

        // Compute result pointer: heap_base + pos
        let! addressOfBaseOp = pLoad heapBaseSSA heapBaseSSA [indexSSA]  // TODO: Placeholder - need AddressOf
        let! subViewOp = pSubView resultPtrSSA heapBaseSSA [heapPosSSA]

        // Update position: pos + size
        let! addOp = pAddI newPosSSA heapPosSSA sizeSSA
        let! storePosOp = pStore newPosSSA heapPosPtrSSA [indexSSA] (TInt I64)

        return ([addressOfPosOp; loadPosOp; addressOfBaseOp; subViewOp; addOp; indexOp; storePosOp], resultPtrSSA)
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
/// Arg 0 is env_ptr, load struct, extract captures at baseIndex + slotIndex
/// SSAs: [0] = index zero, [1] = struct load, [2..N+1] = extracted captures
let pExtractCaptures (baseIndex: int) (captureTypes: MLIRType list) (structType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        do! ensure (ssas.Length >= captureCount + 2) $"pExtractCaptures: Expected {captureCount + 2} SSAs, got {ssas.Length}"

        let indexZeroSSA = ssas.[0]
        let structLoadSSA = ssas.[1]
        let envPtrSSA = Arg 0  // First argument is always env_ptr for closures

        // Load struct from env_ptr (MLIR memrefs require indices)
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex
        let! loadOp = pLoad structLoadSSA envPtrSSA [indexZeroSSA]

        // Extract each capture
        let! extractOps =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let extractSSA = ssas.[i + 2]  // +2 because [0]=indexZero, [1]=structLoad
                    let extractIndex = baseIndex + i
                    let! extractOp = pExtractValue extractSSA structLoadSSA [extractIndex] capTy
                    return extractOp
                })
            |> sequence

        return indexZeroOp :: loadOp :: extractOps
    }

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// CLOSURE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Flat closure struct: code_ptr field + capture fields
let pFlatClosure (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2 + captures.Length) $"pFlatClosure: Expected at least {2 + captures.Length} SSAs, got {ssas.Length}"

        // Compute closure type: {code_ptr: ptr, capture0, capture1, ...}
        let fieldTypes = codePtrTy :: (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let closureTy = TMemRefStatic(totalBytes, TInt I8)

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
        let codePtrTy = TIndex  // Code pointer type
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
        let! state = getUserState
        let retType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA allArgs retType

        return extractCodeOp :: extractCaptureOps @ [callOp]
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
let pLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 4 + captures.Length) $"pLazyStruct: Expected at least {4 + captures.Length} SSAs, got {ssas.Length}"

        // Compute lazy type: {computed: i1, value: T, code_ptr: ptr, captures...}
        let fieldTypes = [TInt I1; valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let lazyTy = TMemRefStatic(totalBytes, TInt I8)

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
        // SSAs: [0] = code_ptr, [1] = const 1, [2] = alloca'd ptr, [3] = index
        do! ensure (ssas.Length >= 4) $"pBuildLazyForce: Expected at least 4 SSAs, got {ssas.Length}"

        let codePtrSSA = ssas.[0]
        let constOneSSA = ssas.[1]
        let ptrSSA = ssas.[2]
        let indexSSA = ssas.[3]

        // Extract code_ptr from lazy struct [2] - it's always a ptr type
        let codePtrTy = TIndex
        let! extractCodePtrOp = pExtractValue codePtrSSA lazySSA [2] codePtrTy

        // Alloca space for lazy struct
        let constOneTy = TInt I64
        let! constOneOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca ptrSSA lazyTy None

        // Store lazy struct to alloca'd space
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let! storeOp = pStore lazySSA ptrSSA [indexSSA] lazyTy

        // Call thunk with pointer -> result
        let argVals = [{ SSA = ptrSSA; Type = TIndex }]
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA argVals resultTy

        return ([extractCodePtrOp; constOneOp; allocaOp; indexOp; storeOp; callOp], TRValue { SSA = resultSSA; Type = resultTy })
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
        let codePtrTy = TIndex
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
        let allArgVals = allArgs |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let! state = getUserState
        let retType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA allArgVals retType

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
