/// Firefly CLI - Thin wrapper around CompilationOrchestrator
///
/// This is intentionally minimal. All compilation logic lives in the orchestrator.
/// The CLI only handles:
///   - Argument parsing
///   - Calling the orchestrator
///   - Returning exit codes
module CLI.Program

open System
open System.IO
open System.Reflection
open Argu
open Alex.Pipeline.CompilationOrchestrator
open CLI.Commands.VerifyCommand
open CLI.Commands.DoctorCommand

// ═══════════════════════════════════════════════════════════════════════════
// Command Line Arguments
// ═══════════════════════════════════════════════════════════════════════════

/// Compile command arguments
type CompileArgs =
    | [<MainCommand; Unique>] Project of path: string
    | [<AltCommandLine("-o")>] Output of path: string
    | [<AltCommandLine("-t")>] Target of target: string
    | [<AltCommandLine("-k")>] Keep_Intermediates
    | [<AltCommandLine("-v")>] Verbose
    | [<AltCommandLine("-T")>] Timing
    | Emit_MLIR
    | Emit_LLVM

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Project _ -> ".fidproj file or F# source file to compile"
            | Output _ -> "Output executable path"
            | Target _ -> "Target triple (default: host platform)"
            | Keep_Intermediates -> "Keep intermediate files (.mlir, .ll) for debugging"
            | Verbose -> "Enable verbose output"
            | Timing -> "Show timing for each compilation phase"
            | Emit_MLIR -> "Emit MLIR and stop (don't generate executable)"
            | Emit_LLVM -> "Emit LLVM IR and stop (don't generate executable)"

// ═══════════════════════════════════════════════════════════════════════════
// Compile Command Handler
// ═══════════════════════════════════════════════════════════════════════════

/// Execute compile command - delegates to orchestrator
let private executeCompile (args: ParseResults<CompileArgs>) : int =
    // Find project path
    let projectPath =
        match args.TryGetResult(Project) with
        | Some p -> p
        | None ->
            let fidprojs = Directory.GetFiles(".", "*.fidproj")
            if fidprojs.Length = 1 then
                fidprojs.[0]
            elif fidprojs.Length > 1 then
                printfn "Error: Multiple .fidproj files found. Please specify which one to compile."
                exit 1
            else
                printfn "Error: No .fidproj file found and no project specified."
                printfn "Usage: firefly compile <project.fidproj>"
                exit 1

    // Build options and delegate to orchestrator
    let options : CompilationOptions = {
        ProjectPath = projectPath
        OutputPath = args.TryGetResult(Output)
        TargetTriple = args.TryGetResult(Target)
        KeepIntermediates = args.Contains(Keep_Intermediates)
        EmitMLIROnly = args.Contains(Emit_MLIR)
        EmitLLVMOnly = args.Contains(Emit_LLVM)
        Verbose = args.Contains(Verbose)
        ShowTiming = args.Contains(Timing)
    }

    // THE single entry point for compilation
    compileProject options

// ═══════════════════════════════════════════════════════════════════════════
// CLI Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Display version information
/// Version is read from assembly (set in Firefly.fsproj <Version>)
let private showVersion() =
    let assembly = Assembly.GetExecutingAssembly()
    let version =
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
        |> Option.ofObj
        |> Option.map (fun a -> a.InformationalVersion)
        |> Option.defaultValue (assembly.GetName().Version.ToString())
    printfn "Firefly %s - F# to Native Compiler with Deterministic Memory Management" version
    printfn "Copyright (c) 2025-2026 SpeakEZ Technologies, Inc."
    0

/// Display usage information
let private showUsage() =
    printfn "Firefly - F# to Native Compiler"
    printfn ""
    printfn "Usage:"
    printfn "  firefly compile [options]    Compile F# to native code"
    printfn "  firefly verify [options]     Verify binary meets constraints"
    printfn "  firefly doctor [options]     Diagnose toolchain issues"
    printfn "  firefly --version            Display version information"
    printfn ""
    printfn "Use 'firefly <subcommand> --help' for more information about a subcommand."

[<EntryPoint>]
let main argv =
    let errorHandler = ProcessExiter(colorizer = function ErrorCode.HelpText -> None | _ -> Some ConsoleColor.Red)

    try
        if argv.Length = 0 then
            showUsage()
            0
        elif argv.[0] = "--version" then
            showVersion()
        elif argv.[0] = "compile" then
            let compileParser = ArgumentParser.Create<CompileArgs>(programName = "firefly compile", errorHandler = errorHandler)
            let compileArgs = Array.skip 1 argv
            let compileResults = compileParser.ParseCommandLine(compileArgs)
            executeCompile compileResults
        elif argv.[0] = "verify" then
            let verifyParser = ArgumentParser.Create<VerifyArgs>(programName = "firefly verify", errorHandler = errorHandler)
            let verifyArgs = Array.skip 1 argv
            let verifyResults = verifyParser.ParseCommandLine(verifyArgs)
            verify verifyResults
        elif argv.[0] = "doctor" then
            let doctorParser = ArgumentParser.Create<DoctorArgs>(programName = "firefly doctor", errorHandler = errorHandler)
            let doctorArgs = Array.skip 1 argv
            let doctorResults = doctorParser.ParseCommandLine(doctorArgs)
            doctor doctorResults
        elif argv.[0] = "--help" || argv.[0] = "-h" then
            showUsage()
            0
        else
            printfn "Error: Unknown subcommand '%s'" argv.[0]
            printfn ""
            showUsage()
            1
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        1
