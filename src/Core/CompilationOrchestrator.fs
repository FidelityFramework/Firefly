/// CompilationOrchestrator - Top-level compiler pipeline coordination
///
/// Orchestrates the full compilation pipeline:
/// FrontEnd (FNCS) → MiddleEnd (Alex + MLIROpt) → BackEnd (LLVM)
///
/// This is the "adult supervision" layer that coordinates all phases.
module Core.CompilationOrchestrator

open System.IO
open System.Reflection
open FSharp.Native.Compiler.Project

open MiddleEnd.MLIROpt.Lowering
open BackEnd.LLVM.Codegen
open Core.Timing
open Core.CompilerConfig
open FSharp.Native.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

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

type CompilationContext = {
    ProjectName: string
    BuildDir: string
    IntermediatesDir: string option
    OutputPath: string
    TargetTriple: string
    OutputKind: Core.Types.Dialects.OutputKind
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: FrontEnd - Compile F# → PSG
// ═══════════════════════════════════════════════════════════════════════════

let private runFrontEnd (projectPath: string) : Result<ProjectCheckResult, string> =
    timePhase "FrontEnd" "F# → PSG (Type Checking & Semantic Graph)" (fun () ->
        FrontEnd.ProjectLoader.load projectPath)

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: MiddleEnd (Alex + PSGElaboration) 
// ═══════════════════════════════════════════════════════════════════════════

let private runMiddleEnd (project: ProjectCheckResult) (ctx: CompilationContext) : Result<string, string> =
    timePhase "MiddleEnd" "MLIR Generation" (fun () ->
        // Get platform context from FNCS
        match Core.FNCS.Integration.platformContext project.CheckResult with
        | None -> Error "No platform context available from FNCS"
        | Some platformCtx ->
            // Delegate to MiddleEnd - it orchestrates PSGElaboration + Alex
            MiddleEnd.MLIRGeneration.generate
                project.CheckResult.Graph
                platformCtx
                ctx.OutputKind
                ctx.IntermediatesDir)

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: BackEnd (MLIR Lowering)
// ═══════════════════════════════════════════════════════════════════════════

let private runMLIRLowering (mlirPath: string) (llPath: string) : Result<unit, string> =
    timePhase "BackEnd.MLIRLower" "Lowering MLIR to LLVM IR" (fun () ->
        lowerToLLVM mlirPath llPath)

// ═══════════════════════════════════════════════════════════════════════════
// Phase 4: BackEnd (Linking)
// ═══════════════════════════════════════════════════════════════════════════

let private runLinking (llPath: string) (outputPath: string) (triple: string) (kind: Core.Types.Dialects.OutputKind) : Result<unit, string> =
    timePhase "BackEnd.Link" "Linking to native binary" (fun () ->
        compileToNative llPath outputPath triple kind)

// ═══════════════════════════════════════════════════════════════════════════
// Context Setup
// ═══════════════════════════════════════════════════════════════════════════

let private setupContext (options: CompilationOptions) (project: ProjectCheckResult) : CompilationContext =
    let config = project.Options
    let buildDir = Path.Combine(config.ProjectDirectory, "target")
    Directory.CreateDirectory(buildDir) |> ignore

    let intermediatesDir =
        if options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly then
            let dir = Path.Combine(buildDir, "intermediates")
            Directory.CreateDirectory(dir) |> ignore
            enableAllPhases dir
            Some dir
        else
            None

    let outputKind =
        match config.OutputKind with
        | OutputKind.Freestanding -> Core.Types.Dialects.OutputKind.Freestanding
        | OutputKind.Console -> Core.Types.Dialects.OutputKind.Console
        | OutputKind.Library -> Core.Types.Dialects.OutputKind.Console
        | OutputKind.Embedded -> Core.Types.Dialects.OutputKind.Freestanding

    {
        ProjectName = config.Name
        BuildDir = buildDir
        IntermediatesDir = intermediatesDir
        OutputPath = options.OutputPath |> Option.defaultValue (Path.Combine(buildDir, config.OutputName |> Option.defaultValue config.Name))
        TargetTriple = options.TargetTriple |> Option.defaultValue (getDefaultTarget())
        OutputKind = outputKind
    }

// ═══════════════════════════════════════════════════════════════════════════
// Main Pipeline
// ═══════════════════════════════════════════════════════════════════════════

let compileProject (options: CompilationOptions) : int =
    // Setup
    setEnabled options.ShowTiming
    if options.Verbose then enableVerboseMode()

    let version = Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                  |> Option.ofObj
                  |> Option.map (fun a -> a.InformationalVersion)
                  |> Option.defaultValue "dev"

    printfn "Firefly Compiler v%s" version
    printfn "======================"
    printfn ""

    // Setup intermediates directory BEFORE loading project (enables FNCS phase emission)
    let needsIntermediates = options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly
    if needsIntermediates then
        let projectDir = Path.GetDirectoryName(options.ProjectPath)
        let intermediatesDir = Path.Combine(projectDir, "target", "intermediates")
        Directory.CreateDirectory(intermediatesDir) |> ignore
        enableAllPhases intermediatesDir

    // Run pipeline: FrontEnd → MiddleEnd → BackEnd
    let result =
        // Phase 1: FrontEnd - Compile F# to PSG
        runFrontEnd options.ProjectPath
        |> Result.bind (fun project ->
            let ctx = setupContext options project

            printfn "Project: %s" ctx.ProjectName
            printfn "Target:  %s" ctx.TargetTriple
            printfn "Output:  %s" ctx.OutputPath
            printfn ""

            // Phase 2: MiddleEnd - Generate MLIR from PSG
            runMiddleEnd project ctx
            |> Result.bind (fun mlirText ->
                // Write MLIR to file
                let mlirPath =
                    match ctx.IntermediatesDir with
                    | Some dir -> Path.Combine(dir, artifactFilename FSharp.Native.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.ArtifactId.Mlir)
                    | None -> Path.Combine(Path.GetTempPath(), ctx.ProjectName + ".mlir")
                File.WriteAllText(mlirPath, mlirText)

                if options.EmitMLIROnly then
                    printfn "Stopped after MLIR generation (--emit-mlir)"
                    Ok ()
                else
                    // Phase 3: BackEnd - Lower MLIR to LLVM IR
                    let llPath =
                        match ctx.IntermediatesDir with
                        | Some dir -> Path.Combine(dir, artifactFilename FSharp.Native.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.ArtifactId.Llvm)
                        | None -> Path.Combine(Path.GetTempPath(), ctx.ProjectName + ".ll")

                    runMLIRLowering mlirPath llPath
                    |> Result.bind (fun () ->
                        if options.EmitLLVMOnly then
                            printfn "Stopped after LLVM IR generation (--emit-llvm)"
                            Ok ()
                        else
                            // Phase 4: BackEnd - Link to native binary
                            runLinking llPath ctx.OutputPath ctx.TargetTriple ctx.OutputKind
                            |> Result.map (fun () ->
                                printfn ""
                                printfn "Compilation successful: %s" ctx.OutputPath))))

    printSummary()
    match result with
    | Ok () -> 0
    | Error msg ->
        printfn "Error: %s" msg
        1
