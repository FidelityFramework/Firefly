/// ElisionPatterns - Composable MLIR construction templates
///
/// VISIBILITY: public module - Witnesses call these
///
/// Patterns compose Elements (atomic MLIR emission) into reusable templates.
/// Witnesses observe PSG structure and call Patterns to elide MLIR.
module Alex.Patterns.ElisionPatterns

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRElements  // CAN import internal Elements

// ═══════════════════════════════════════════════════════════════════════════
// STRUCT PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Build a struct from fields using InsertValue chain
/// Returns: list of MLIR ops (Undef + InsertValue chain)
let buildStruct
    (fields: (SSA * MLIRType) list)
    (structTy: MLIRType)
    (ssas: SSA list)
    : MLIROp list =

    let undefSSA = ssas.[0]
    let undefOp = emitUndef undefSSA structTy

    let insertOps =
        fields
        |> List.mapi (fun i (fieldSSA, _) ->
            let targetSSA = ssas.[i + 1]  // ssas.[0] is undef, rest are inserts
            let sourceSSA = if i = 0 then undefSSA else ssas.[i]
            emitInsertValue targetSSA sourceSSA fieldSSA [i] structTy)

    undefOp :: insertOps

/// Extract fields from a struct
let extractFields
    (structSSA: SSA)
    (structTy: MLIRType)
    (fieldCount: int)
    (ssas: SSA list)
    : MLIROp list =

    List.init fieldCount (fun i ->
        emitExtractValue ssas.[i] structSSA [i] structTy)

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Build flat closure: {code_ptr: ptr, cap0: T0, cap1: T1, ...}
let buildFlatClosure
    (codePtrSSA: SSA)
    (captures: (SSA * MLIRType) list)
    (closureTy: MLIRType)
    (ssas: SSA list)
    : MLIROp list =

    let ptrTy = MLIRType.TPtr
    let allFields = (codePtrSSA, ptrTy) :: captures
    buildStruct allFields closureTy ssas

/// Extract code_ptr from closure (field 0)
let extractClosureCodePtr
    (closureSSA: SSA)
    (closureTy: MLIRType)
    (resultSSA: SSA)
    : MLIROp =
    emitExtractValue resultSSA closureSSA [0] closureTy

/// Extract capture from closure (field index)
let extractClosureCapture
    (closureSSA: SSA)
    (closureTy: MLIRType)
    (captureIndex: int)
    (resultSSA: SSA)
    : MLIROp =
    emitExtractValue resultSSA closureSSA [captureIndex + 1] closureTy  // +1 for code_ptr

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit indirect closure call
let callClosure
    (closureSSA: SSA)
    (closureTy: MLIRType)
    (args: Val list)
    (retTy: MLIRType)
    (ssas: SSA list)
    : MLIROp list =

    let codePtrSSA = ssas.[0]
    let resultSSA = if ssas.Length > 1 then Some ssas.[1] else None

    let extractOp = extractClosureCodePtr closureSSA closureTy codePtrSSA
    let callOp = emitIndirectCall resultSSA codePtrSSA args retTy

    [extractOp; callOp]

// ═══════════════════════════════════════════════════════════════════════════
// LAZY PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Build Lazy<T> struct: {computed: i1, value: T, code_ptr: ptr, cap0, cap1, ...}
let buildLazyStruct
    (thunkName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    (ssas: SSA list)
    : MLIROp list =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let lazyType = MLIRType.TStruct ([MLIRType.TInt IntBitWidth.I1; elementType; MLIRType.TPtr] @ captureTypes)

    // Pre-assigned SSAs
    let falseSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withComputedSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]

    let baseOps = [
        // false constant for computed flag
        emitConstI falseSSA 0L (MLIRType.TInt IntBitWidth.I1)
        // undef lazy struct
        emitUndef undefSSA lazyType
        // insert computed=false at index 0
        emitInsertValue withComputedSSA undefSSA falseSSA [0] lazyType
        // get thunk function address
        emitAddressOf codePtrSSA (GlobalRef.GFunc thunkName)
        // insert code_ptr at index 2
        emitInsertValue withCodePtrSSA withComputedSSA codePtrSSA [2] lazyType
    ]

    // Insert each capture at indices 3, 4, 5, ...
    if captureVals.IsEmpty then
        baseOps
    else
        let captureOps, _ =
            captureVals
            |> List.indexed
            |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                let nextSSA = ssas.[5 + i]
                let captureIndex = 3 + i
                let op = emitInsertValue nextSSA prevSSA capVal.SSA [captureIndex] lazyType
                (accOps @ [op], nextSSA)
            ) ([], withCodePtrSSA)
        baseOps @ captureOps

/// Force Lazy<T>: extract code_ptr, alloca, store, call thunk
let forceLazy
    (lazyVal: Val)
    (elementType: MLIRType)
    (ssas: SSA list)
    : MLIROp list =

    let codePtrSSA = ssas.[0]
    let oneSSA = ssas.[1]
    let allocaPtrSSA = ssas.[2]
    let resultSSA = ssas.[3]

    [
        // Extract code_ptr from lazy struct (index 2)
        emitExtractValue codePtrSSA lazyVal.SSA [2] lazyVal.Type
        // Constant 1 for alloca count
        emitConstI oneSSA 1L (MLIRType.TInt IntBitWidth.I64)
        // Alloca space for lazy struct on stack
        emitAlloca allocaPtrSSA oneSSA lazyVal.Type None
        // Store lazy struct to alloca
        emitStore lazyVal.SSA allocaPtrSSA lazyVal.Type AtomicOrdering.NotAtomic
        // Call thunk with pointer to lazy struct
        emitIndirectCall (Some resultSSA) codePtrSSA [{ SSA = allocaPtrSSA; Type = MLIRType.TPtr }] elementType
    ]

/// Check if Lazy<T> is computed: extract computed flag
let isLazyComputed
    (lazyVal: Val)
    (resultSSA: SSA)
    : MLIROp =
    emitExtractValue resultSSA lazyVal.SSA [0] lazyVal.Type
