module CLI.Commands.DoctorCommand

open System
open System.Runtime.InteropServices
open Argu
open CLI.Diagnostics.Types
open CLI.Diagnostics

/// Command line arguments for the doctor command
type DoctorArgs =
    | Verbose
    | Quick
    | Fix_Hints
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Verbose -> "Show detailed information about each component"
            | Quick -> "Perform only essential checks"
            | Fix_Hints -> "Show installation commands for missing components"

/// Runs the doctor command to diagnose toolchain issues
let doctor (args: ParseResults<DoctorArgs>) =
    let verbose = args.Contains Verbose
    let quick = args.Contains Quick
    let showHints = args.Contains Fix_Hints || not quick
    
    printfn "Firefly Doctor - Checking System Health"
    printfn "======================================"
    printfn ""
    
    // First, show basic system information
    printfn "System Information:"
    printfn "  OS: %s" RuntimeInformation.OSDescription
    printfn "  Architecture: %s" (RuntimeInformation.OSArchitecture.ToString())
    printfn "  .NET Runtime: %s" RuntimeInformation.FrameworkDescription
    
    // Detect MSYS2 environment if on Windows
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        match Environment.GetEnvironmentVariable("MSYSTEM") with
        | null | "" -> 
            printfn "  Environment: Windows (native)"
        | msysEnv -> 
            printfn "  Environment: MSYS2 %s" msysEnv
            let msysRoot = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
            if not (String.IsNullOrEmpty(msysRoot)) then
                printfn "  MSYS2 Root: %s" msysRoot
    
    printfn ""
    
    // Run toolchain verification - fixed casing
    match ToolchainVerification.verifyToolchain verbose with
    | Success () ->
        printfn ""
        printfn "✓ All checks passed! Firefly is ready to compile."
        
        if verbose then
            printfn ""
            printfn "You can compile F# programs with:"
            printfn "  firefly compile --input Program.fs --output Program.exe"
        
        0 // Success exit code
    
    | Failure errors ->
        printfn ""
        printfn "✗ Some issues were found that may prevent compilation."
        
        if showHints then
            printfn ""
            printfn "Suggested fixes:"
            errors |> List.iter (fun error ->
                match error with
                | ConversionError(phase, _, _, message) ->
                    printfn "  - %s: %s" phase message
                | SyntaxError(_, message, context) ->
                    printfn "  - %s: %s" (String.concat " > " context) message
                | TypeCheckError(construct, message, _) ->
                    printfn "  - %s: %s" construct message
                | InternalError(phase, message, details) ->
                    printfn "  - %s: %s" phase message
                    match details with
                    | Some d -> printfn "    %s" d
                    | None -> ()
                | ParseError(_, message) ->
                    printfn "  - Parse: %s" message
                | DependencyResolutionError(symbol, message) ->
                    printfn "  - Dependency '%s': %s" symbol message)
        
        // If on Windows with MSYS2, give specific guidance
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            match Environment.GetEnvironmentVariable("MSYSTEM") with
            | "MSYS" ->
                printfn ""
                printfn "IMPORTANT: You're in the MSYS environment, which is not suitable"
                printfn "for building native Windows executables. Please use MINGW64 instead:"
                printfn "  1. Close this terminal"
                printfn "  2. Open 'MSYS2 MINGW64' from the Start Menu"
                printfn "  3. Run 'firefly doctor' again"
            | _ -> ()
        
        1 // Error exit code

/// Provides a quick check that can be called from other commands
let quickCheck() : bool =
    match ToolchainVerification.quickVerifyToolchain() with
    | true -> true
    | false ->
        printfn ""
        printfn "Toolchain issues detected. Run 'firefly doctor' for details."
        false