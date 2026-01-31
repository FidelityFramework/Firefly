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

    /// Extract all SSAs used by an operation
    let rec getUsedSSAs (op: MLIROp) : SSA list =
        match op with
        | MLIROp.MemRefOp (MemRefOp.Store (valueSSA, memrefSSA, indexSSAs, _)) ->
            valueSSA :: memrefSSA :: indexSSAs
        | MLIROp.MemRefOp (MemRefOp.Load (_, memrefSSA, indexSSAs, _)) ->
            memrefSSA :: indexSSAs
        | MLIROp.ArithOp (ArithOp.AddI (_, lhs, rhs, _))
        | MLIROp.ArithOp (ArithOp.SubI (_, lhs, rhs, _))
        | MLIROp.ArithOp (ArithOp.MulI (_, lhs, rhs, _))
        | MLIROp.ArithOp (ArithOp.DivSI (_, lhs, rhs, _))
        | MLIROp.ArithOp (ArithOp.DivUI (_, lhs, rhs, _)) -> [lhs; rhs]
        // Add more cases as needed
        | _ -> []

    /// Check if operation defines OR uses SSAs that are only in FuncDef bodies
    let isDuplicate (op: MLIROp) : bool =
        let defined = getDefinedSSAs op
        let used = getUsedSSAs op
        let definesInternal = defined |> List.exists (fun ssa -> Set.contains ssa funcDefSSAs)
        let usesInternal = used |> List.exists (fun ssa -> Set.contains ssa funcDefSSAs)
        definesInternal || usesInternal

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

    /// Deduplicate function declarations by function name
    let deduplicateFuncDecls (ops: MLIROp list) : MLIROp list =
        let mutable seenFuncs = Set.empty
        ops |> List.filter (fun op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDecl (name, _, _, _)) ->
                if Set.contains name seenFuncs then
                    false  // Skip duplicate
                else
                    seenFuncs <- Set.add name seenFuncs
                    true  // Keep first occurrence
            | _ -> true)

    /// Filter operations: keep FuncDef nodes and non-duplicate operations
    let rec filterOps ops =
        ops |> List.filter (fun op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef _) -> true  // Keep FuncDef nodes
            | MLIROp.GlobalString _ -> true  // Keep globals (will deduplicate separately)
            | _ -> not (isDuplicate op)  // Keep if not duplicate
        )

    let cleaned = filterOps operations |> deduplicateGlobals
    printfn "[DEBUG] Structural folding: %d ops before, %d ops after" (List.length operations) (List.length cleaned)
    cleaned

// ═══════════════════════════════════════════════════════════════════════════
// DECLARATION COLLECTION PASS (Emit FuncDecl from FuncCall analysis)
// ═══════════════════════════════════════════════════════════════════════════

/// Declaration Collection Pass
///
/// Scans all FuncCall operations and emits unified FuncDecl operations.
/// This eliminates the "first witness wins" coordination during witnessing.
///
/// ARCHITECTURAL RATIONALE:
/// - Function declarations are MLIR-level structure (not PSG semantics)
/// - Witnesses emit calls with actual types (no coordination needed)
/// - This pass analyzes ALL calls before emitting declarations (deterministic)
/// - Signature unification happens here (one place, not scattered across witnesses)
///
/// ALGORITHM:
/// 1. Recursively collect all FuncCall operations
/// 2. Group calls by function name
/// 3. Unify signatures (currently: first signature wins, future: type coercion)
/// 4. Emit FuncDecl operations
/// 5. Remove any existing FuncDecl operations from witnesses (duplicates)
let declarationCollectionPass (operations: MLIROp list) : MLIROp list =

    /// Collect all function DEFINITIONS in the module
    /// These are functions implemented in this module - they need NO declarations
    let definedFunctions =
        operations
        |> List.choose (function
            | MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> Some name
            | _ -> None)
        |> Set.ofList

    /// Recursively collect all function calls from operations
    let rec collectCalls (op: MLIROp) : (string * MLIRType list * MLIRType) list =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncCall (_, name, args, retTy)) ->
            // Collect this call's signature
            [(name, args |> List.map (fun v -> v.Type), retTy)]

        // Recurse into nested operations
        | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
            body |> List.collect collectCalls

        | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps)) ->
            let thenCalls = thenOps |> List.collect collectCalls
            let elseCalls = elseOps |> Option.map (List.collect collectCalls) |> Option.defaultValue []
            thenCalls @ elseCalls

        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            let condCalls = condOps |> List.collect collectCalls
            let bodyCalls = bodyOps |> List.collect collectCalls
            condCalls @ bodyCalls

        | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) ->
            bodyOps |> List.collect collectCalls

        | MLIROp.Block (_, blockOps) ->
            blockOps |> List.collect collectCalls

        | MLIROp.Region ops ->
            ops |> List.collect collectCalls

        | _ -> []

    /// Collect all function calls from all operations
    let allCalls = operations |> List.collect collectCalls

    /// Generate declarations ONLY for EXTERNAL functions (not defined in this module)
    /// Internal functions already have FuncDef - declarations would cause redefinition errors
    let declarations =
        allCalls
        |> List.groupBy (fun (name, _, _) -> name)
        |> List.filter (fun (name, _) -> not (Set.contains name definedFunctions))  // ONLY external functions
        |> List.map (fun (name, signatures) ->
            // Pick first signature (deterministic due to post-order)
            let (_, paramTypes, retType) = signatures.Head
            MLIROp.FuncOp (FuncOp.FuncDecl (name, paramTypes, retType, FuncVisibility.Private)))
    
    /// Remove any FuncDecl operations emitted by witnesses (now duplicates)
    let withoutWitnessDecls =
        operations
        |> List.filter (function
            | MLIROp.FuncOp (FuncOp.FuncDecl _) -> false  // Remove witness-emitted declarations
            | _ -> true)

    // Emit only EXTERNAL function declarations (functions called but not defined in this module)
    // Internal functions (with FuncDef) need NO declarations - they're already defined
    if List.isEmpty declarations then
        withoutWitnessDecls
    else
        printfn "[Alex] Declaration collection: Emitted %d external function declarations" (List.length declarations)
        declarations @ withoutWitnessDecls

// ═══════════════════════════════════════════════════════════════════════════
// TYPE NORMALIZATION PASS (Insert casts at call sites)
// ═══════════════════════════════════════════════════════════════════════════

/// Insert memref.cast operations at call sites where argument types don't match parameter types
/// This handles cases like memref<13xi8> → memref<?xi8> (static → dynamic dimension cast)
let typeNormalizationPass (operations: MLIROp list) : MLIROp list =
    // Create transform context for fresh SSA generation (platform unused in this pass)
    let ctx = {
        Operations = operations
        Platform = {
            RuntimeMode = RuntimeMode.Console
            TargetOS = OSFamily.Linux
            TargetArch = Architecture.X86_64
            PlatformWordType = TInt I64
            Bindings = Map.empty
            NeedsStartWrapper = false
        }
        FreshSSACounter = ref 0
    }

    // Collect function signatures (both declarations AND definitions) to get canonical parameter types
    let functionSignatures =
        operations
        |> List.choose (function
            | MLIROp.FuncOp (FuncOp.FuncDecl (name, paramTypes, _, _)) -> Some (name, paramTypes)
            | MLIROp.FuncOp (FuncOp.FuncDef (name, params, _, _, _)) ->
                // Extract types from (SSA * MLIRType) list
                let paramTypes = params |> List.map snd
                Some (name, paramTypes)
            | _ -> None)
        |> Map.ofList

    /// Check if two memref types are compatible but require a cast
    let needsMemRefCast (argTy: MLIRType) (paramTy: MLIRType) : bool =
        match argTy, paramTy with
        // Static → Dynamic memref cast (e.g., memref<13xi8> → memref<?xi8>)
        | TMemRefStatic (_, elemTy1), TMemRef elemTy2 when elemTy1 = elemTy2 -> true
        // Zero-sized → Dynamic memref cast (e.g., memref<0xi8> → memref<?xi8>)
        | TMemRefStatic (0, elemTy1), TMemRef elemTy2 when elemTy1 = elemTy2 -> true
        | _ -> false

    /// Transform a single operation, inserting casts where needed
    let rec transformOp (op: MLIROp) : MLIROp list =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, args, retTy)) ->
            // Check if we have a signature (declaration or definition) for this function
            match Map.tryFind funcName functionSignatures with
            | Some paramTypes when paramTypes.Length = args.Length ->
                // Check each argument against parameter type
                let castsAndArgs =
                    List.zip args paramTypes
                    |> List.map (fun (arg, paramTy) ->
                        if needsMemRefCast arg.Type paramTy then
                            // Insert cast: fresh SSA = memref.cast arg.SSA : arg.Type to paramTy
                            let castSSA = MLIRTransformContext.freshSSA ctx
                            let castOp = MLIROp.MemRefOp (MemRefOp.Cast (castSSA, arg.SSA, arg.Type, paramTy))
                            Some castOp, { SSA = castSSA; Type = paramTy }
                        else
                            None, arg)

                let casts = castsAndArgs |> List.choose fst
                let newArgs = castsAndArgs |> List.map snd
                let newCall = MLIROp.FuncOp (FuncOp.FuncCall (resultOpt, funcName, newArgs, retTy))
                casts @ [newCall]
            | _ ->
                // No declaration or argument count mismatch - leave unchanged
                [op]

        | MLIROp.FuncOp (FuncOp.FuncDef (name, parameters, retTy, body, vis)) ->
            // Recursively transform body operations
            let newBody = body |> List.collect transformOp
            [MLIROp.FuncOp (FuncOp.FuncDef (name, parameters, retTy, newBody, vis))]

        | _ -> [op]

    operations |> List.collect transformOp

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

    // Phase 2: FFI Boundary Conversion Pass (DISABLED)
    //
    // ARCHITECTURAL DECISION: Let mlir-opt handle ALL dialect conversions.
    // The --finalize-memref-to-llvm pass knows how to convert memref → pointer at FFI boundaries.
    // Inserting our own conversion casts causes ordering issues (cast references memref,
    // but then mlir-opt lowers the memref to struct, invalidating the cast).
    //
    // LESSON LEARNED: Don't insert conversion casts before dialect lowering.
    // Let the standard MLIR conversion infrastructure handle type mismatches.
    let afterFFI = afterFolding  // No custom conversion - mlir-opt handles it

    // Serialize Phase 2 intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterFFI
        let filePath = System.IO.Path.Combine(dir, "09_after_ffi_conversion.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 09_after_ffi_conversion.mlir"
    | None -> ()

    // Phase 3: Declaration Collection Pass
    //
    // ARCHITECTURAL RATIONALE:
    // Function declarations are MLIR-level structure, not PSG-level semantics.
    // During witnessing, ApplicationWitness emits FuncCall operations with actual types.
    // This pass scans ALL calls, collects unique function signatures, and emits FuncDecl operations.
    //
    // BENEFITS:
    // 1. Deterministic - same calls always produce same declarations (no "first witness wins")
    // 2. Separates concerns - witnessing (codata) vs declaration emission (structural)
    // 3. Codata principle - witnesses return calls, post-pass handles declarations
    // 4. Signature unification - can analyze ALL calls before deciding signature
    let afterDecls = declarationCollectionPass afterFFI

    // Serialize Phase 3 intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterDecls
        let filePath = System.IO.Path.Combine(dir, "10_after_declaration_collection.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 10_after_declaration_collection.mlir"
    | None -> ()

    // Phase 4: Call Site Type Normalization Pass
    //
    // Insert memref.cast operations at call sites where argument types don't exactly match
    // declared parameter types. This is required for MLIR validity - memref<13xi8> and memref<?xi8>
    // are incompatible without an explicit cast.
    let afterTypeNorm = typeNormalizationPass afterDecls

    // Serialize Phase 4 intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterTypeNorm
        let filePath = System.IO.Path.Combine(dir, "11_after_type_normalization.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 11_after_type_normalization.mlir"
    | None -> ()

    // Future passes will be composed here:
    // let afterDCont = dcontLoweringPass afterTypeNorm
    // let afterInet = inetLoweringPass afterDCont
    // let afterBackend = backendTargetingPass afterInet

    afterTypeNorm
