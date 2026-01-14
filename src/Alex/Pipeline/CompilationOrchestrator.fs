/// CompilationOrchestrator - THE single orchestrator for Firefly compilation
///
/// FNCS-based pipeline (January 2026 - unified project loading):
///   1. Load project (.fidproj) via FNCS ProjectChecker
///   2. Parse and type-check with FNCS (done by ProjectChecker)
///   3. Generate MLIR via witness-based emission
///   4. Lower MLIR → LLVM (Toolchain)
///   5. Link LLVM → Native (Toolchain)
module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open Core.CompilerConfig
open Core.Toolchain
open Alex.Traversal.PSGZipper
open Alex.Bindings.BindingTypes
open Alex.Bindings.PlatformTypes
open Core.FNCS.Integration
open Alex.Traversal.FNCSTransfer

// FNCS Project module - unified project loading
open FSharp.Native.Compiler.Project

// FNCS Phase emission configuration
module FNCSPhaseConfig = FSharp.Native.Compiler.Checking.Native.Infrastructure.PhaseConfig

// Import specific types to avoid shadowing Result.Error
type FNCSDiagnosticSeverity = FSharp.Native.Compiler.Checking.Native.SemanticGraph.NativeDiagnosticSeverity
module FNCSSemanticGraph = FSharp.Native.Compiler.Checking.Native.SemanticGraph

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Compilation options passed from CLI
type CompilationOptions = {
    ProjectPath: string
    OutputPath: string option
    TargetTriple: string option
    KeepIntermediates: bool
    EmitMLIROnly: bool
    EmitLLVMOnly: bool
    Verbose: bool
    ShowTiming: bool
}

/// Result of MLIR generation
type MLIRGenerationResult = {
    Content: string
    HasErrors: bool
    Errors: string list
    CollectedFunctions: string list
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Convert FNCS OutputKind to Firefly OutputKind
let private toFireflyOutputKind (kind: OutputKind) : Core.Types.Dialects.OutputKind =
    match kind with
    | OutputKind.Freestanding -> Core.Types.Dialects.OutputKind.Freestanding
    | OutputKind.Console -> Core.Types.Dialects.OutputKind.Console
    | OutputKind.Library -> Core.Types.Dialects.OutputKind.Console // Map to console for now
    | OutputKind.Embedded -> Core.Types.Dialects.OutputKind.Freestanding // Map to freestanding

/// Determine target platform from triple
let parsePlatform (triple: string) : TargetPlatform =
    match TargetPlatform.parseTriple triple with
    | Some platform -> platform
    | None ->
        // Fallback heuristics
        if triple.Contains("linux") then
            if triple.Contains("aarch64") || triple.Contains("arm64") then
                { OS = Linux; Arch = ARM64; Triple = triple; Features = Set.empty }
            else
                TargetPlatform.linux_x86_64
        elif triple.Contains("darwin") || triple.Contains("macos") then
            if triple.Contains("x86_64") then
                TargetPlatform.macos_x86_64
            else
                TargetPlatform.macos_arm64
        elif triple.Contains("windows") then
            TargetPlatform.windows_x86_64
        else
            TargetPlatform.linux_x86_64

// ═══════════════════════════════════════════════════════════════════════════
// MLIR Generation via FNCS ProjectChecker
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MLIR from a project using FNCS ProjectChecker
let generateMLIRFromFNCS (projectResult: ProjectCheckResult) (targetTriple: string) (isFreestanding: bool) (intermediatesDir: string option) : MLIRGenerationResult =
    // Check for parse errors first
    if not (Map.isEmpty projectResult.ParseErrors) then
        let errors =
            projectResult.ParseErrors
            |> Map.toList
            |> List.collect (fun (file, errs) ->
                errs |> List.map (fun e -> sprintf "%s: %s" file e))
        printfn "[FNCS] Parse errors found:"
        for err in errors do
            printfn "[FNCS]   %s" err
        {
            Content = ""
            HasErrors = true
            Errors = errors
            CollectedFunctions = []
        }
    else
        // Check for type checking errors
        let checkErrors =
            projectResult.CheckResult.Diagnostics
            |> List.filter (fun d -> d.Severity = FNCSDiagnosticSeverity.Error)
            |> List.map (fun d -> d.Message)

        if not (List.isEmpty checkErrors) then
            printfn "[FNCS] Type checking found %d error(s)" checkErrors.Length
            for err in checkErrors do
                printfn "[FNCS] ERROR: %s" err
            {
                Content = ""
                HasErrors = true
                Errors = checkErrors
                CollectedFunctions = []
            }
        else
            let graph = projectResult.CheckResult.Graph
            let nodeCount = Map.count graph.Nodes
            let entryCount = List.length graph.EntryPoints

            // Generate MLIR via witness-based transfer (codata architecture)
            let mlirContent, transferErrors = transferGraphWithDiagnostics graph isFreestanding intermediatesDir

            {
                Content = mlirContent
                HasErrors = not (List.isEmpty transferErrors)
                Errors = transferErrors
                CollectedFunctions = []
            }

/// Generate MLIR from a project - fallback to placeholder if FNCS fails
let generateMLIR (projectResult: ProjectCheckResult) (targetTriple: string) (isFreestanding: bool) (intermediatesDir: string option) : MLIRGenerationResult =
    try
        generateMLIRFromFNCS projectResult targetTriple isFreestanding intermediatesDir
    with ex ->
        let mainContent = """module {
  // Fallback - FNCS processing failed
  llvm.func @main() -> i32 attributes {sym_visibility = "public"} {
    %exit_code = arith.constant 0 : i64
    %syscall_num = arith.constant 60 : i64
    %result = llvm.inline_asm has_side_effects "syscall", "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}" %syscall_num, %exit_code : (i64, i64) -> i64
    llvm.unreachable
  }
}
"""
        {
            Content = mainContent
            HasErrors = true
            Errors = [sprintf "FNCS exception: %s\nStack trace:\n%s" ex.Message ex.StackTrace]
            CollectedFunctions = ["main"]
        }

// ═══════════════════════════════════════════════════════════════════════════
// Main Compilation Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a project - THE single entry point for all compilation
let compileProject (options: CompilationOptions) : int =
    // Enable timing if requested
    Core.Timing.setEnabled options.ShowTiming

    // Enable verbose mode if requested
    if options.Verbose then
        enableVerboseMode()

    printfn "Firefly Compiler v0.5.0 (FNCS)"
    printfn "=============================="
    printfn ""

    // Enable FNCS phase emission if keeping intermediates
    // Must be done BEFORE type checking to capture phases
    if options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly then
        let projectDir = Path.GetDirectoryName(options.ProjectPath)
        let buildDir = Path.Combine(projectDir, "target")
        let intermediatesDir = Path.Combine(buildDir, "intermediates")
        Directory.CreateDirectory(intermediatesDir) |> ignore
        FNCSPhaseConfig.enableAllPhases intermediatesDir

    // Step 1: Load and check project via FNCS ProjectChecker
    let projectResult =
        Core.Timing.timePhase "FNCS" "Project Loading & Type Checking" (fun () ->
            ProjectChecker.checkProject options.ProjectPath)

    match projectResult with
    | Error msg ->
        printfn "Error: %s" msg
        1

    | Ok result ->
        let config = result.Options
        let projectDir = config.ProjectDirectory

        let targetTriple =
            options.TargetTriple
            |> Option.defaultValue (getDefaultTarget())

        // Build directory (default to "target" if not specified)
        let buildDir = Path.Combine(projectDir, "target")
        Directory.CreateDirectory(buildDir) |> ignore

        let outputName = config.OutputName |> Option.defaultValue config.Name
        let outputPath =
            options.OutputPath
            |> Option.defaultValue (Path.Combine(buildDir, outputName))

        let outputKind = toFireflyOutputKind config.OutputKind

        printfn "Project: %s" config.Name
        printfn "Sources: %d files" (List.length result.SourceFiles)
        printfn "Target:  %s" targetTriple
        printfn "Output:  %s" outputPath
        printfn ""

        // Create intermediates directory if needed
        let intermediatesDir =
            if options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly then
                let dir = Path.Combine(buildDir, "intermediates")
                Directory.CreateDirectory(dir) |> ignore
                Some dir
            else
                None

        // Step 2: Generate MLIR via FNCS
        let isFreestanding = (outputKind = Core.Types.Dialects.OutputKind.Freestanding)
        let mlirResult =
            Core.Timing.timePhase "MLIR" "MLIR Generation" (fun () ->
                generateMLIR result targetTriple isFreestanding intermediatesDir)

        if mlirResult.HasErrors then
            printfn ""
            printfn "[MLIR] Emission errors detected:"
            for error in mlirResult.Errors do
                printfn "  ERROR: %s" error
            printfn ""

        // Write MLIR if keeping intermediates
        match intermediatesDir with
        | Some dir ->
            let mlirPath = Path.Combine(dir, config.Name + ".mlir")
            File.WriteAllText(mlirPath, mlirResult.Content)

            if options.EmitMLIROnly then
                printfn ""
                if mlirResult.HasErrors then
                    printfn "MLIR generation completed with errors (--emit-mlir)"
                else
                    printfn "Stopped after MLIR generation (--emit-mlir)"
                Core.Timing.printSummary()
                if mlirResult.HasErrors then 1 else 0
            elif mlirResult.HasErrors then
                printfn ""
                printfn "Compilation failed due to emission errors."
                Core.Timing.printSummary()
                1
            else
                // Step 3: Lower MLIR → LLVM (via Toolchain)
                let llPath = Path.Combine(dir, config.Name + ".ll")

                let llvmResult =
                    Core.Timing.timePhase "MLIR-LOWER" "Lowering MLIR to LLVM IR" (fun () ->
                        lowerMLIRToLLVM mlirPath llPath)

                match llvmResult with
                | Error msg ->
                    printfn "[LLVM] Error: %s" msg
                    Core.Timing.printSummary()
                    1
                | Ok () ->
                    if options.EmitLLVMOnly then
                        printfn ""
                        printfn "Stopped after LLVM IR generation (--emit-llvm)"
                        Core.Timing.printSummary()
                        0
                    else
                        // Step 4: Link LLVM → Native (via Toolchain)
                        let linkResult =
                            Core.Timing.timePhase "LINK" "Compiling to native binary" (fun () ->
                                compileLLVMToNative llPath outputPath targetTriple outputKind)

                        match linkResult with
                        | Error msg ->
                            printfn "[LINK] Error: %s" msg
                            Core.Timing.printSummary()
                            1
                        | Ok () ->
                            printfn ""
                            printfn "Compilation successful: %s" outputPath
                            Core.Timing.printSummary()
                            0

        | None ->
            // No intermediates - use temp directory
            if mlirResult.HasErrors then
                printfn ""
                printfn "Compilation failed due to emission errors."
                printfn "Use -k to generate intermediate files for debugging."
                Core.Timing.printSummary()
                1
            else
                let tempDir = Path.Combine(Path.GetTempPath(), "firefly-" + Guid.NewGuid().ToString("N").[..7])
                Directory.CreateDirectory(tempDir) |> ignore

                let mlirPath = Path.Combine(tempDir, config.Name + ".mlir")
                let llPath = Path.Combine(tempDir, config.Name + ".ll")

                File.WriteAllText(mlirPath, mlirResult.Content)

                let llvmResult =
                    Core.Timing.timePhase "MLIR-LOWER" "Lowering MLIR to LLVM IR" (fun () ->
                        lowerMLIRToLLVM mlirPath llPath)

                match llvmResult with
                | Error msg ->
                    printfn "[LLVM] Error: %s" msg
                    try Directory.Delete(tempDir, true) with _ -> ()
                    Core.Timing.printSummary()
                    1
                | Ok () ->
                    let linkResult =
                        Core.Timing.timePhase "LINK" "Compiling to native binary" (fun () ->
                            compileLLVMToNative llPath outputPath targetTriple outputKind)

                    try Directory.Delete(tempDir, true) with _ -> ()

                    match linkResult with
                    | Error msg ->
                        printfn "[LINK] Error: %s" msg
                        Core.Timing.printSummary()
                        1
                    | Ok () ->
                        printfn ""
                        printfn "Compilation successful: %s" outputPath
                        Core.Timing.printSummary()
                        0
