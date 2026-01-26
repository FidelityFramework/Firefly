/// Lambda Witness - Witness Lambda nodes to MLIR functions
///
/// ARCHITECTURAL PRINCIPLES (Four Pillars):
/// 1. Coeffects: SSA assignment is pre-computed, lookup via context
/// 2. Active Patterns: Match on semantic meaning, not strings
/// 3. Zipper: Navigate and accumulate structured ops
/// 4. Templates: Return structured MLIROp types, no sprintf
///
/// Functions are built as structured data:
/// - Accumulate body ops during child traversal
/// - Create FuncDef with body region
/// - Emit to top-level
///
/// CLOSURE ARCHITECTURE (MLKit-style flat closures):
/// - Simple Lambda (no captures): Emit func.func, call via func.call @name
/// - Closing Lambda (with captures):
///   - Emit func.func with env_ptr as first parameter
///   - In parent scope, emit closure construction:
///     1. Alloca env struct
///     2. GEP + Store for each captured value
///     3. Build closure struct {code_ptr, env_ptr}
///
/// WITNESS PATTERN: Returns (MLIROp list * TransferResult)
/// Witnesses OBSERVE and RETURN. They do NOT emit directly.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.Coeffects
open PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL ARENA ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Allocate memory in the global 'closure_heap' bump allocator
/// Uses pre-computed SSAs from ClosureLayout (coeffect pattern)
/// Returns (ops, resultPtrSSA)
let private allocateInClosureArena (layout: ClosureLayout) : MLIROp list * SSA =
    let ops = [
        // Load current position
        MLIROp.LLVMOp (AddressOf (layout.HeapPosPtrSSA, GFunc "closure_pos"))
        MLIROp.LLVMOp (Load (layout.HeapPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))

        // Compute result pointer: heap_base + pos
        MLIROp.LLVMOp (AddressOf (layout.HeapBaseSSA, GFunc "closure_heap"))
        MLIROp.LLVMOp (GEP (layout.HeapResultPtrSSA, layout.HeapBaseSSA, [(layout.HeapPosSSA, MLIRTypes.i64)], MLIRTypes.i8))

        // Update position: pos + size (sizeSSA from layout)
        MLIROp.ArithOp (ArithOp.AddI (layout.HeapNewPosSSA, layout.HeapPosSSA, layout.SizeSSA, MLIRTypes.i64))
        MLIROp.LLVMOp (Store (layout.HeapNewPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
    ]
    ops, layout.HeapResultPtrSSA

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE CONSTRUCTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build closure construction ops for a Lambda with captures
/// Uses pre-computed ClosureLayout from SSAAssignment (coeffect pattern)
///
/// TRUE FLAT CLOSURE (HEAP ALLOCATED):
/// 1. Construct the environment struct (captures) by value.
/// 2. Allocate space in the global closure arena.
/// 3. Store the environment struct to the arena.
/// 4. Return Uniform Pair: {code_ptr, env_ptr}
let private buildClosureConstruction
    (layout: ClosureLayout)
    (lambdaName: string)
    (ctx: WitnessContext)
    : MLIROp list =

    // 1. Get address of the lambda function (SSA from ClosureLayout)
    let codeAddrOp = MLIROp.LLVMOp (AddressOf (layout.CodeAddrSSA, GFunc lambdaName))

    // 2. Create the Flat Environment struct (contains captures)
    // All SSAs from ClosureLayout - no synthesis during emission
    let flatUndefOp = MLIROp.LLVMOp (Undef (layout.ClosureUndefSSA, layout.ClosureStructType))

    // Insert code_ptr at index 0
    let flatWithCodeOp = MLIROp.LLVMOp (InsertValue (layout.ClosureWithCodeSSA, layout.ClosureUndefSSA,
        layout.CodeAddrSSA, [0], layout.ClosureStructType))

    // Insert each captured value using pre-computed CaptureInsertSSAs
    let mutable currentFlatSSA = layout.ClosureWithCodeSSA
    let flatOps =
        layout.Captures
        |> List.mapi (fun i slot ->
            let capturedSSA =
                // Look up variable in PARENT scope (current scope is inner lambda)
                let recallParentVarSSA name =
                    match ctx.Accumulator.ScopeStack with
                    | parentScope :: _ -> Map.tryFind name parentScope.VarAssoc
                    | [] -> None

                match recallParentVarSSA slot.Name with
                | Some (ssa, _) -> ssa
                | None ->
                    match slot.SourceNodeId with
                    | Some srcId ->
                        match MLIRAccumulator.recallNode (NodeId.value srcId) ctx.Accumulator with
                        | Some (ssa, _ty) -> ssa
                        | None ->
                            match lookupSSA srcId ctx.Coeffects.SSA with
                            | Some s -> s
                            | None ->
                                failwithf "LambdaWitness: No SSA for captured variable '%s' (sourceNodeId %d)"
                                    slot.Name (NodeId.value srcId)
                    | None ->
                        failwithf "LambdaWitness: No source node for captured variable '%s'" slot.Name

            // Use pre-computed SSA from CaptureInsertSSAs
            let nextSSA = layout.CaptureInsertSSAs.[i]
            let insertIndex = 1 + slot.SlotIndex
            let op = MLIROp.LLVMOp (InsertValue (nextSSA, currentFlatSSA, capturedSSA, [insertIndex], layout.ClosureStructType))
            currentFlatSSA <- nextSSA
            [op]
        )
        |> List.concat

    // 3. Allocate in Global Arena (avoiding stack lifetime issues)
    // Calculate size using GEP null trick - all SSAs from ClosureLayout
    let nullOp = MLIROp.LLVMOp (NullPtr layout.SizeNullPtrSSA)
    // Generate constant 1 for GEP index
    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (layout.SizeOneSSA, 1L, MLIRTypes.i32))
    // GEP null[1] gives pointer to address == sizeof(type)
    let sizeGepOp = MLIROp.LLVMOp (GEP (layout.SizeGepSSA, layout.SizeNullPtrSSA, [(layout.SizeOneSSA, MLIRTypes.i32)], layout.ClosureStructType))
    let ptrToIntOp = MLIROp.LLVMOp (PtrToInt (layout.SizeSSA, layout.SizeGepSSA, MLIRTypes.i64))

    // Allocate using ClosureLayout SSAs
    let allocOps, envPtrSSA = allocateInClosureArena layout

    // Store the environment struct to the arena
    let storeOp = MLIROp.LLVMOp (Store (currentFlatSSA, envPtrSSA, layout.ClosureStructType, NotAtomic))

    // 4. Build the uniform {ptr, ptr} Function Value pair
    // Use layout SSAs for pair construction
    let pairTy = TStruct [TPtr; TPtr]

    let buildPairOps = [
        MLIROp.LLVMOp (Undef (layout.PairUndefSSA, pairTy))
        MLIROp.LLVMOp (InsertValue (layout.PairWithCodeSSA, layout.PairUndefSSA, layout.CodeAddrSSA, [0], pairTy))
        MLIROp.LLVMOp (InsertValue (layout.ClosureResultSSA, layout.PairWithCodeSSA, envPtrSSA, [1], pairTy))
    ]

    [codeAddrOp; flatUndefOp; flatWithCodeOp] @ flatOps @
    [nullOp; oneOp; sizeGepOp; ptrToIntOp] @ allocOps @ [storeOp] @ buildPairOps

// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE EXTRACTION (Inside Closure Function)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate ops to extract captured values from closure struct at function entry
/// These ops are prepended to the function body
///
/// CLOSURE CONVENTION: Closing Lambdas take env_ptr (Arg 0) which points to the struct.
/// The struct type and extraction indices depend on context:
/// - Regular closure: Load {code_ptr, cap0, cap1, ...}, extract at indices 1, 2, ...
/// - Lazy thunk: Load {computed, value, code_ptr, cap0, cap1, ...}, extract at indices 3, 4, ...
///
/// January 2026: Compositional layout via ClosureLayout coeffect
/// SSA layout: v0..v(N-1) = extraction, vN = struct load, v(N+1)+ = body
let private buildCaptureExtractionOps
    (layout: ClosureLayout)
    (_ctx: WitnessContext)
    : MLIROp list =

    // Arg 0 is the environment pointer (passed as TPtr)
    let envPtrSSA = Arg 0

    // Get struct type and extraction base from context (compositional layout)
    let loadStructType = closureLoadStructType layout
    let extractionBase = closureExtractionBaseIndex layout

    // Load the struct from the environment pointer
    // Use pre-computed StructLoadSSA from ClosureLayout (coeffect pattern)
    let loadOp = MLIROp.LLVMOp (Load (layout.StructLoadSSA, envPtrSSA, loadStructType, NotAtomic))

    // Extract each captured value from the loaded struct
    // Extraction index = baseIndex + slotIndex (context-dependent)
    // Extraction SSAs are derived from SlotIndex: v0, v1, ..., v(N-1)
    // This is the child function's SSA namespace - body SSAs start at N+1
    let extractOps =
        layout.Captures
        |> List.map (fun slot ->
            let extractSSA = V slot.SlotIndex  // Derived from PSG structure
            let extractIndex = extractionBase + slot.SlotIndex
            MLIROp.LLVMOp (ExtractValue (extractSSA, layout.StructLoadSSA, [extractIndex], loadStructType))
        )

    loadOp :: extractOps

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Argv Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Bind entry point parameters for C-style main
/// At OS entry point: %arg0: i32 = argc, %arg1: !llvm.ptr = argv
/// For F# string[] parameter, pattern matching handles conversion at use site
let private bindArgvParameters (paramName: string) (ctx: WitnessContext) : unit =
    // Bind C-style argc/argv under well-known names using STRUCTURED types
    MLIRAccumulator.bindVar "__argc" (Arg 0) MLIRTypes.i32 ctx.Accumulator
    MLIRAccumulator.bindVar "__argv" (Arg 1) MLIRTypes.ptr ctx.Accumulator
    // Bind F# parameter name to argv pointer
    MLIRAccumulator.bindVar paramName (Arg 1) MLIRTypes.ptr ctx.Accumulator

// ═══════════════════════════════════════════════════════════════════════════
// Function Body Building
// ═══════════════════════════════════════════════════════════════════════════

let private createReturnOp (valueSSA: SSA) (valueTy: MLIRType) (isClosure: bool) : MLIROp =
    if isClosure then
        MLIROp.LLVMOp (LLVMOp.Return (Some valueSSA, Some valueTy))
    else
        MLIROp.FuncOp (FuncOp.FuncReturn [(valueSSA, valueTy)])

/// Extract final return type by peeling TFun layers
let private extractFinalReturnType (ty: NativeType) (paramCount: int) : NativeType =
    let rec peel ty count =
        match ty with
        | NativeType.TFun(_, resultTy) when count > 0 ->
            peel resultTy (count - 1)
        | NativeType.TFun(_, resultTy) when paramCount = 0 ->
            // Unit lambda: peel the unit->result layer
            resultTy
        | _ -> ty
    peel ty paramCount

// ... (Lambda Witnessing) ...

/// Check if function is internal (private visibility)
/// Main is public, all others are private
let private isFuncInternal (name: string) : bool =
    name <> "main"

