/// MLIRNanopass - Structural MLIR-to-MLIR transformations
///
/// This module establishes the foundation for dual witness infrastructure:
/// - Current: PSG → MLIR (via witnesses)
/// - Future: MLIR → MLIR (via nanopasses) → DCont/Inet dialects
///
/// ARCHITECTURAL PRINCIPLES:
/// 1. Platform-agnostic - no hardcoded backend assumptions
/// 2. Structural transformations - like PSG nanopasses
/// 3. Composable - each pass handles one concern
/// 4. SSA isolation - fresh SSA generation contained to this layer
///
/// LONG-TERM VISION:
/// This is the FIRST component of dual witness infrastructure. Future passes will include:
/// - DCont lowering (sequential/effectful patterns → stack-based async)
/// - Inet lowering (parallel/pure patterns → graph reduction)
/// - Backend targeting (portable dialects → LLVM/SPIR-V/WebAssembly/custom)
/// - Hybrid optimization (mix DCont/Inet based on purity analysis)
module Alex.Pipeline.MLIRNanopass

open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// MLIR TRANSFORM CONTEXT (Isolated SSA Generation)
// ═══════════════════════════════════════════════════════════════════════════

/// Context for MLIR transformations - isolated SSA generation
type MLIRTransformContext = {
    Operations: MLIROp list
    Platform: PlatformResolutionResult
    FreshSSACounter: int ref  // Isolated to MLIR transform layer
}

module MLIRTransformContext =
    /// Create fresh SSA for MLIR-level operations (not PSG nodes)
    /// Uses high numbers (10000+) to avoid collision with PSG-assigned SSAs
    ///
    /// ARCHITECTURAL RATIONALE:
    /// - PSG-assigned SSAs: Represent program semantics (source-level values)
    /// - MLIR-temp SSAs: Implementation details (FFI conversion, spills, etc.)
    /// - Separation maintains clean boundary: semantics vs. implementation
    let freshSSA (ctx: MLIRTransformContext) : SSA =
        let n = !ctx.FreshSSACounter
        ctx.FreshSSACounter := n + 1
        SSA.V (10000 + n)

// ═══════════════════════════════════════════════════════════════════════════
// FFI BOUNDARY CONVERSION PASS (First MLIR Nanopass)
// ═══════════════════════════════════════════════════════════════════════════

/// FFI Boundary Conversion Pass
///
/// Detects memref arguments at FFI boundaries and inserts conversion operations.
/// Platform-agnostic: Works for ANY backend with FFI (LLVM, SPIR-V, WebAssembly, etc.)
///
/// ARCHITECTURAL CONTEXT:
/// F# Native NativePtr behaves like OCaml CTypes "Views" (automatic marshaling):
/// - No explicit conversion in source code (compare OCaml's `to_voidp`)
/// - NativePtr is opaque handle at semantic level
/// - Implementation choice (memref) necessitates FFI boundary conversion
/// - Other backends (SPIR-V) might implement NativePtr differently
///
/// EXAMPLE TRANSFORMATION:
/// Before:
///   %v21 = memref.alloca() : memref<i8>
///   %v54 = func.call @write(%v35, %v21, %v38) : (i64, memref<i8>, i64) -> i64
///
/// After:
///   %v21 = memref.alloca() : memref<i8>
///   %v10000 = builtin.unrealized_conversion_cast %v21 : memref<i8> to !llvm.ptr
///   %v54 = func.call @write(%v35, %v10000, %v38) : (i64, !llvm.ptr, i64) -> i64
let ffiConversionPass (ctx: MLIRTransformContext) : MLIROp list =

    /// Check if function is an FFI boundary (external function, syscall, etc.)
    ///
    /// PLATFORM-AGNOSTIC HEURISTICS:
    /// 1. Known syscalls (write, read, open, close, mmap, etc.)
    /// 2. External function declarations (func.func private @name)
    /// 3. Functions with platform-specific calling conventions
    /// 4. Platform syscalls often prefixed with _ (e.g., _write on Darwin)
    ///
    /// FUTURE ENHANCEMENT: Could be driven by FFI annotations in FNCS intrinsics
    let isFFIBoundary (funcName: string) : bool =
        match funcName with
        // POSIX syscalls (platform-independent interface)
        | "write" | "read" | "open" | "close" | "lseek" | "mmap" | "munmap" -> true
        | "socket" | "bind" | "listen" | "accept" | "connect" | "send" | "recv" -> true
        | "fork" | "exec" | "exit" | "wait" | "kill" | "getpid" | "signal" -> true
        // Platform-prefixed variants (Darwin, Windows)
        | name when name.StartsWith("_") -> true
        // Future: Check function visibility, calling convention, etc.
        | _ -> false

    /// Convert memref argument to pointer for FFI
    ///
    /// Only converts memref types - all other types pass through unchanged.
    /// This preserves platform-agnostic design: no assumptions about scalar types.
    let convertMemRefArg (arg: Val) (ctx: MLIRTransformContext) : Val * MLIROp option =
        match arg.Type with
        | TMemRefScalar _ | TMemRef _ ->
            // Insert conversion: memref → pointer
            let ptrSSA = MLIRTransformContext.freshSSA ctx
            let convOp = MLIROp.MemRefOp (MemRefOp.ExtractBasePtr (ptrSSA, arg.SSA, arg.Type))
            ({ SSA = ptrSSA; Type = TPtr }, Some convOp)
        | _ -> (arg, None)  // Non-memref args pass through unchanged

    /// Transform a single operation (recursively)
    ///
    /// STRUCTURAL TRANSFORMATION: Like PSG nanopasses, this is a structural
    /// tree rewrite. Each operation is transformed, and nested operations
    /// (function bodies, control flow) are recursively transformed.
    let rec transformOp (op: MLIROp) : MLIROp list =
        match op with
        // ── FFI BOUNDARY DETECTION ────────────────────────────────────────
        | MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, args, retTy)) ->
            // DEBUG: Log all FuncCalls
            printfn "[DEBUG] FuncCall: @%s, isFFI=%b, args=%d" funcName (isFFIBoundary funcName) (List.length args)

            if isFFIBoundary funcName then
                // DEBUG: Show argument types
                args |> List.iteri (fun i arg ->
                    printfn "[DEBUG]   arg[%d]: %A" i arg.Type)

                // Convert all memref arguments to pointers
                let convertedArgs, convOps =
                    args
                    |> List.map (fun arg -> convertMemRefArg arg ctx)
                    |> List.unzip
                let convOps = convOps |> List.choose id

                // DEBUG: Log FFI conversion
                if not (List.isEmpty convOps) then
                    printfn "[DEBUG] FFI conversion: @%s - inserted %d conversion ops" funcName (List.length convOps)

                // Emit conversions followed by call with converted args
                let callOp = MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, convertedArgs, retTy))
                convOps @ [callOp]
            else
                // Non-FFI call - pass through unchanged
                [op]

        // ── RECURSIVE TRANSFORMATION (Nested Operations) ──────────────────
        | MLIROp.FuncOp (FuncOp.FuncDef (name, parameters, retTy, body, vis)) ->
            let transformedBody = body |> List.collect transformOp
            [MLIROp.FuncOp (FuncOp.FuncDef (name, parameters, retTy, transformedBody, vis))]

        | MLIROp.SCFOp (SCFOp.If (cond, thenOps, elseOps)) ->
            let transformedThen = thenOps |> List.collect transformOp
            let transformedElse = elseOps |> Option.map (List.collect transformOp)
            [MLIROp.SCFOp (SCFOp.If (cond, transformedThen, transformedElse))]

        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            let transformedCond = condOps |> List.collect transformOp
            let transformedBody = bodyOps |> List.collect transformOp
            [MLIROp.SCFOp (SCFOp.While (transformedCond, transformedBody))]

        | MLIROp.SCFOp (SCFOp.For (lb, ub, step, bodyOps)) ->
            let transformedBody = bodyOps |> List.collect transformOp
            [MLIROp.SCFOp (SCFOp.For (lb, ub, step, transformedBody))]

        | MLIROp.Block (label, blockOps) ->
            let transformedOps = blockOps |> List.collect transformOp
            [MLIROp.Block (label, transformedOps)]

        | MLIROp.Region ops ->
            let transformedOps = ops |> List.collect transformOp
            [MLIROp.Region transformedOps]

        // ── PASS-THROUGH (No transformation needed) ───────────────────────
        | _ -> [op]  // Pass through unchanged

    // Transform all operations (including top-level flat operations)
    ctx.Operations |> List.collect transformOp

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURAL FOLDING NANOPASS (Deduplicate FuncDef bodies from flat stream)
// ═══════════════════════════════════════════════════════════════════════════

/// Structural Folding Nanopass - Removes duplicate operations from flat stream
///
/// PROBLEM: After witnessing, FuncDef nodes have populated bodies, but those same
/// operations also appear in the flat stream. This creates duplicate definitions.
///
/// INPUT: FuncDef nodes with bodies + flat stream with duplicate operations
/// OUTPUT: FuncDef nodes + cleaned flat stream (no duplicates)
///
/// APPROACH: Collect all SSAs defined in FuncDef bodies, filter flat stream to
/// remove operations that define those SSAs.
let structuralFoldingPass (operations: MLIROp list) : MLIROp list =

    /// Extract all SSAs defined by an operation
    let rec getDefinedSSAs (op: MLIROp) : SSA list =
        match op with
        // ArithOp - all operations that define SSAs
        | MLIROp.ArithOp (ArithOp.ConstI (ssa, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.ConstF (ssa, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.AddI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.SubI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.MulI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.DivSI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.DivUI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.RemSI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.RemUI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.CmpI (ssa, _, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.AddF (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.SubF (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.MulF (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.DivF (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.CmpF (ssa, _, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.AndI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.OrI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.XorI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.ShLI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.ShRSI (ssa, _, _, _)) -> [ssa]
        | MLIROp.ArithOp (ArithOp.ShRUI (ssa, _, _, _)) -> [ssa]

        // MemRefOp - operations that define SSAs
        | MLIROp.MemRefOp (MemRefOp.Alloca (ssa, _, _)) -> [ssa]
        | MLIROp.MemRefOp (MemRefOp.Load (ssa, _, _, _)) -> [ssa]
        | MLIROp.MemRefOp (MemRefOp.ExtractBasePtr (ssa, _, _)) -> [ssa]

        // LLVMOp - operations that define SSAs
        | MLIROp.LLVMOp (LLVMOp.ExtractValue (ssa, _, _, _)) -> [ssa]
        | MLIROp.LLVMOp (LLVMOp.InsertValue (ssa, _, _, _, _)) -> [ssa]
        | MLIROp.LLVMOp (LLVMOp.Undef (ssa, _)) -> [ssa]

        // Other operations that define SSAs
        | MLIROp.AddressOf (ssa, _, _) -> [ssa]
        | MLIROp.FuncOp (FuncOp.FuncCall (Some ssa, _, _, _)) -> [ssa]

        // FuncDef - recursively collect SSAs from body
        | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
            body |> List.collect getDefinedSSAs

        // All other operations don't define SSAs
        | _ -> []

    /// Collect all SSAs defined in FuncDef bodies
    let funcDefSSAs =
        operations
        |> List.choose (function
            | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
                Some (body |> List.collect getDefinedSSAs)
            | _ -> None)
        |> List.concat
        |> Set.ofList

    printfn "[DEBUG] Structural folding: Found %d SSAs in FuncDef bodies" (Set.count funcDefSSAs)

    /// Check if operation defines an SSA that's in a FuncDef body
    let isDuplicate (op: MLIROp) : bool =
        let defined = getDefinedSSAs op
        defined |> List.exists (fun ssa -> Set.contains ssa funcDefSSAs)

    /// Deduplicate GlobalStrings by symbol name
    let deduplicateGlobals (ops: MLIROp list) : MLIROp list =
        let mutable seenGlobals = Set.empty
        ops |> List.filter (fun op ->
            match op with
            | MLIROp.GlobalString (name, _, _) ->
                if Set.contains name seenGlobals then
                    false  // Skip duplicate
                else
                    seenGlobals <- Set.add name seenGlobals
                    true  // Keep first occurrence
            | _ -> true)

    /// Filter operations: keep FuncDef nodes and non-duplicate operations
    let rec filterOps ops =
        ops |> List.filter (fun op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef _) -> true  // Keep FuncDef nodes
            | MLIROp.ScopeMarker _ -> false  // Remove scope markers
            | MLIROp.GlobalString _ -> true  // Keep globals (will deduplicate separately)
            | _ -> not (isDuplicate op)  // Keep if not duplicate
        )

    let cleaned = filterOps operations |> deduplicateGlobals
    printfn "[DEBUG] Structural folding: %d ops before, %d ops after" (List.length operations) (List.length cleaned)
    cleaned

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS ORCHESTRATION (Apply All Passes)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply all MLIR nanopasses in sequence
///
/// CURRENT PASSES:
/// 1. Structural Folding (deduplicate FuncDef bodies from flat stream)
/// 2. FFI Boundary Conversion (memref → pointer at syscall boundaries)
///
/// FUTURE PASSES (aligned with DCont/Inet Duality vision):
/// 3. DCont Lowering (async {} → dcont dialect, stack-based continuations)
/// 4. Inet Lowering (query {} → inet dialect, parallel graph reduction)
/// 5. Hybrid Optimization (mix DCont/Inet based on purity analysis)
/// 6. Backend Targeting (func/arith/memref → LLVM/SPIR-V/WebAssembly)
///
/// ARCHITECTURAL NOTE:
/// This is the integration point for eventual TableGen-based transformations.
/// Future vision: Generate TableGen in MiddleEnd, use it to transform dialects.
let applyPasses (operations: MLIROp list) (platform: PlatformResolutionResult) (intermediatesDir: string option) : MLIROp list =
    // Apply passes in sequence (nanopass pipeline)

    // Phase 1: Structural Folding (deduplicate FuncDef bodies from flat stream)
    let afterFolding = structuralFoldingPass operations

    // Serialize Phase 1 intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterFolding
        let filePath = System.IO.Path.Combine(dir, "08_after_structural_folding.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 08_after_structural_folding.mlir"
    | None -> ()

    // Phase 2: FFI Boundary Conversion (memref → pointer at syscall boundaries)
    let ctx = {
        Operations = afterFolding
        Platform = platform
        FreshSSACounter = ref 0
    }
    let afterFFI = ffiConversionPass ctx

    // Serialize Phase 2 intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterFFI
        let filePath = System.IO.Path.Combine(dir, "09_after_ffi_conversion.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 09_after_ffi_conversion.mlir"
    | None -> ()

    // Future passes will be composed here:
    // let afterDCont = dcontLoweringPass afterFFI
    // let afterInet = inetLoweringPass afterDCont
    // let afterBackend = backendTargetingPass afterInet

    afterFFI
