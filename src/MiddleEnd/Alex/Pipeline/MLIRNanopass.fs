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
        | MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, args, retTy))
            when isFFIBoundary funcName ->
            // Convert all memref arguments to pointers
            let convertedArgs, convOps =
                args
                |> List.map (fun arg -> convertMemRefArg arg ctx)
                |> List.unzip
            let convOps = convOps |> List.choose id

            // Emit conversions followed by call with converted args
            let callOp = MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, convertedArgs, retTy))
            convOps @ [callOp]

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

    ctx.Operations |> List.collect transformOp

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS ORCHESTRATION (Apply All Passes)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply all MLIR nanopasses in sequence
///
/// CURRENT PASSES:
/// 1. FFI Boundary Conversion (memref → pointer at syscall boundaries)
///
/// FUTURE PASSES (aligned with DCont/Inet Duality vision):
/// 2. DCont Lowering (async {} → dcont dialect, stack-based continuations)
/// 3. Inet Lowering (query {} → inet dialect, parallel graph reduction)
/// 4. Hybrid Optimization (mix DCont/Inet based on purity analysis)
/// 5. Backend Targeting (func/arith/memref → LLVM/SPIR-V/WebAssembly)
///
/// ARCHITECTURAL NOTE:
/// This is the integration point for eventual TableGen-based transformations.
/// Future vision: Generate TableGen in MiddleEnd, use it to transform dialects.
let applyPasses (operations: MLIROp list) (platform: PlatformResolutionResult) : MLIROp list =
    let ctx = {
        Operations = operations
        Platform = platform
        FreshSSACounter = ref 0
    }

    // Apply passes in sequence (future: configurable pass pipeline)
    ffiConversionPass ctx
    // Future passes will be composed here:
    // |> dcontLoweringPass    // DCont dialect lowering
    // |> inetLoweringPass     // Inet dialect lowering
    // |> backendTargetingPass // Backend-specific lowering
