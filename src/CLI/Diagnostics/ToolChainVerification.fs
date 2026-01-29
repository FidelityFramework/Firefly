module CLI.Diagnostics.ToolchainVerification

open System
open System.IO
open System.Runtime.InteropServices
open CLI.Diagnostics.Types

/// Represents the status of a toolchain element
type ComponentStatus =
    | Found of version: string
    | Missing of hint: string
    | Error of message: string

/// Represents a required toolchain element
type ToolchainComponent = {
    Name: string
    Description: string
    CheckCommand: string option
    CheckFiles: string list
    InstallHint: string
    Required: bool
}

/// Enhanced MSYS2 environment detection
module MSYS2Detection =
    
    /// Gets detailed MSYS2 environment information
    let getMSYS2EnvironmentInfo() : (string * string * string list) option =
        try
            let msystem = Environment.GetEnvironmentVariable("MSYSTEM")
            let msysRoot = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
            let mingwPrefix = Environment.GetEnvironmentVariable("MINGW_PREFIX")
            let path = Environment.GetEnvironmentVariable("PATH")
            
            match msystem with
            | null | "" -> None
            | env ->
                let pathDirectories = 
                    if String.IsNullOrEmpty(path) then []
                    else path.Split(';') |> Array.toList |> List.filter (not << String.IsNullOrWhiteSpace)
                
                let rootPath = 
                    if not (String.IsNullOrEmpty(msysRoot)) then msysRoot
                    elif not (String.IsNullOrEmpty(mingwPrefix)) then mingwPrefix
                    else "/mingw64"
                
                Some (env, rootPath, pathDirectories)
        with _ -> None
    
    /// Validates MSYS2 environment for compilation
    let validateMSYS2Environment() : ComponentStatus =
        match getMSYS2EnvironmentInfo() with
        | Some ("MSYS", _, _) ->
            Missing "MSYS environment detected - use MINGW64 for native compilation"
        | Some ("MINGW64", rootPath, _) ->
            Found (sprintf "MINGW64 environment at %s" rootPath)
        | Some ("MINGW32", rootPath, _) ->
            Found (sprintf "MINGW32 environment at %s (consider MINGW64 for 64-bit targets)" rootPath)
        | Some (env, rootPath, _) ->
            Found (sprintf "%s environment at %s" env rootPath)
        | None ->
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                Missing "Not in MSYS2 environment - install and use MSYS2 MINGW64"
            else
                Found "Non-Windows environment"

/// Enhanced command availability checking with MSYS2 support
module CommandDetection =
    
    /// Comprehensive command search with multiple execution strategies
    let checkCommandAvailability (command: string) (args: string) : ComponentStatus =
        let commandVariants = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                [command; command + ".exe"]
            else
                [command]
        
        let tryExecuteCommand (cmd: string) (arguments: string) : bool * string =
            try
                let processInfo = System.Diagnostics.ProcessStartInfo()
                processInfo.FileName <- cmd
                processInfo.Arguments <- arguments
                processInfo.UseShellExecute <- false
                processInfo.RedirectStandardOutput <- true
                processInfo.RedirectStandardError <- true
                processInfo.CreateNoWindow <- true
                processInfo.WindowStyle <- System.Diagnostics.ProcessWindowStyle.Hidden
                
                use proc = System.Diagnostics.Process.Start(processInfo)
                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()
                proc.WaitForExit(5000) |> ignore
                
                let combinedOutput = 
                    [output; error] 
                    |> List.filter (not << String.IsNullOrWhiteSpace)
                    |> String.concat " "
                
                (proc.ExitCode = 0, combinedOutput)
            with
            | ex -> (false, ex.Message)
        
        let tryFindInPath (cmd: string) : bool =
            try
                let pathDirs = 
                    Environment.GetEnvironmentVariable("PATH").Split(';')
                    |> Array.filter (not << String.IsNullOrWhiteSpace)
                
                pathDirs |> Array.exists (fun dir ->
                    let fullPath = Path.Combine(dir, cmd)
                    File.Exists(fullPath))
            with _ -> false
        
        let rec tryCommands variants =
            match variants with
            | [] -> Error "Command not found"
            | cmd :: rest ->
                let (success, output) = tryExecuteCommand cmd args
                if success then
                    let version = 
                        output.Split('\n')
                        |> Array.tryHead
                        |> Option.defaultValue output
                        |> fun s -> s.Trim()
                    Found version
                else
                    if tryFindInPath cmd then
                        Error (sprintf "Command found but failed to execute: %s" output)
                    else
                        tryCommands rest
        
        tryCommands commandVariants
    
    /// Specialized LLVM tool detection
    let checkLLVMTools() : (string * ComponentStatus) list =
        let llvmTools = [
            ("llc", "LLVM static compiler")
            ("opt", "LLVM optimizer")
            ("llvm-config", "LLVM configuration tool")
        ]
        
        llvmTools |> List.map (fun (tool, description) ->
            let status = checkCommandAvailability tool "--version"
            (sprintf "%s (%s)" tool description, status))
    
    /// Specialized compiler detection with fallback options
    let checkCompilers() : (string * ComponentStatus) list =
        let compilers = [
            ("gcc", "GNU Compiler Collection")
            ("clang", "Clang C/C++ Compiler")
            ("g++", "GNU C++ Compiler")
        ]
        
        compilers |> List.map (fun (compiler, description) ->
            let status = checkCommandAvailability compiler "--version"
            (sprintf "%s (%s)" compiler description, status))

/// Platform-specific toolchain requirements with enhanced detection
module PlatformRequirements =
    
    /// Enhanced Windows toolchain requirements with MSYS2 integration
    let getWindowsToolchainRequirements() : ToolchainComponent list =
        let msysInfo = MSYS2Detection.getMSYS2EnvironmentInfo()
        let mingwRoot = 
            match msysInfo with
            | Some (_, rootPath, _) -> rootPath
            | None -> "/mingw64"
        
        [
            {
                Name = "MSYS2 Environment"
                Description = "MSYS2 MINGW64 environment for native Windows compilation"
                CheckCommand = None
                CheckFiles = []
                InstallHint = "Install MSYS2 from https://www.msys2.org/ and use MINGW64 terminal"
                Required = true
            }
            {
                Name = "GCC Compiler"
                Description = "GNU Compiler Collection for compiling object files"
                CheckCommand = Some "gcc --version"
                CheckFiles = []
                InstallHint = "pacman -S mingw-w64-x86_64-gcc"
                Required = true
            }
            {
                Name = "MinGW-w64 CRT"
                Description = "C Runtime startup files (critical for linking)"
                CheckCommand = None
                CheckFiles = [
                    Path.Combine(mingwRoot, "lib", "crt1.o")
                    Path.Combine(mingwRoot, "lib", "crt2.o")
                    Path.Combine(mingwRoot, "lib", "libmsvcrt.a")
                ]
                InstallHint = "pacman -S mingw-w64-x86_64-crt-git"
                Required = true
            }
            {
                Name = "LLVM Tools"
                Description = "LLVM compiler infrastructure (llc, opt)"
                CheckCommand = Some "llc --version"
                CheckFiles = []
                InstallHint = "pacman -S mingw-w64-x86_64-llvm"
                Required = true
            }
            {
                Name = "GNU Binutils"
                Description = "Binary utilities including linker"
                CheckCommand = Some "ld --version"
                CheckFiles = []
                InstallHint = "pacman -S mingw-w64-x86_64-binutils"
                Required = true
            }
            {
                Name = "LLD Linker"
                Description = "LLVM's native linker (alternative to GNU ld)"
                CheckCommand = Some "lld --version"
                CheckFiles = []
                InstallHint = "pacman -S mingw-w64-x86_64-lld"
                Required = false
            }
        ]
    
    /// Linux toolchain requirements
    let getLinuxToolchainRequirements() : ToolchainComponent list =
        [
            {
                Name = "GCC Compiler"
                Description = "GNU Compiler Collection"
                CheckCommand = Some "gcc --version"
                CheckFiles = []
                InstallHint = "apt-get install gcc (Ubuntu/Debian) or yum install gcc (RHEL/CentOS)"
                Required = true
            }
            {
                Name = "LLVM Tools"
                Description = "LLVM compiler infrastructure"
                CheckCommand = Some "llc --version"
                CheckFiles = []
                InstallHint = "apt-get install llvm (Ubuntu/Debian) or yum install llvm (RHEL/CentOS)"
                Required = true
            }
        ]
    
    /// macOS toolchain requirements
    let getMacOSToolchainRequirements() : ToolchainComponent list =
        [
            {
                Name = "Clang Compiler"
                Description = "Apple Clang compiler"
                CheckCommand = Some "clang --version"
                CheckFiles = []
                InstallHint = "Install Xcode Command Line Tools: xcode-select --install"
                Required = true
            }
            {
                Name = "LLVM Tools"
                Description = "LLVM compiler infrastructure"
                CheckCommand = Some "llc --version"
                CheckFiles = []
                InstallHint = "Install via Homebrew: brew install llvm"
                Required = true
            }
        ]

/// element checking with enhanced error reporting
module ComponentChecking =
    
    /// Checks if required files exist with detailed reporting
    let checkFiles (files: string list) : ComponentStatus =
        if files.IsEmpty then
            Found "No files to check"
        else
            let missing = files |> List.filter (not << File.Exists)
            let existing = files |> List.filter File.Exists
            
            if missing.IsEmpty then
                Found (sprintf "All %d files present" files.Length)
            else
                let missingList = String.concat ", " missing
                let existingList = String.concat ", " existing
                Missing (sprintf "Missing files: %s (found: %s)" missingList existingList)
    
    /// Enhanced element checking with fallback strategies
    let checkComponent (element: ToolchainComponent) : ComponentStatus =
        match element.CheckCommand with
        | Some cmd ->
            let parts = cmd.Split(' ', 2)
            let executable = parts.[0]
            let args = if parts.Length > 1 then parts.[1] else ""
            CommandDetection.checkCommandAvailability executable args
        | None ->
            if element.CheckFiles.IsEmpty then
                Error "No check method specified"
            else
                checkFiles element.CheckFiles
    
    /// Special check for MSYS2 environment
    let checkMSYS2Environment() : ComponentStatus =
        MSYS2Detection.validateMSYS2Environment()

/// Main toolchain verification with comprehensive reporting
let verifyToolchain (verbose: bool) : DiagnosticResult<unit> =
    printfn "Verifying Firefly toolchain requirements..."
    printfn "=========================================="
    
    // Enhanced platform detection and environment analysis
    let platform = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            match MSYS2Detection.getMSYS2EnvironmentInfo() with
            | Some (env, rootPath, pathDirs) -> 
                sprintf "Windows (MSYS2 %s at %s)" env rootPath
            | None -> 
                "Windows (native - consider using MSYS2)"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            "Linux"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            "macOS"
        else
            "Unknown platform"
    
    printfn "Platform: %s" platform
    printfn "Runtime: %s" (RuntimeInformation.FrameworkDescription)
    printfn ""
    
    // Check MSYS2 environment first on Windows
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        match ComponentChecking.checkMSYS2Environment() with
        | Found msg ->
            printfn "✓ MSYS2 Environment: %s" msg
        | Missing hint ->
            printfn "⚠ MSYS2 Environment: %s" hint
            printfn "  For optimal compatibility, use MSYS2 MINGW64 terminal"
        | Error msg ->
            printfn "✗ MSYS2 Environment: %s" msg
        printfn ""
    
    // Get platform-specific requirements
    let requirements = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            PlatformRequirements.getWindowsToolchainRequirements()
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            PlatformRequirements.getLinuxToolchainRequirements()
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            PlatformRequirements.getMacOSToolchainRequirements()
        else
            []
    
    // Check each element with enhanced reporting
    let results = 
        requirements 
        |> List.map (fun element ->
            let status = ComponentChecking.checkComponent element
            (element, status))
    
    // Display results with detailed information
    let mutable hasErrors = false
    let mutable hasWarnings = false
    
    results |> List.iter (fun (element, status) ->
        let statusSymbol, statusText, isError, isWarning = 
            match status with
            | Found version -> 
                ("✓", version, false, false)
            | Missing hint -> 
                ("✗", sprintf "Missing - %s" hint, element.Required, not element.Required)
            | Error msg -> 
                ("!", sprintf "Error - %s" msg, element.Required, not element.Required)
        
        if isError then hasErrors <- true
        if isWarning then hasWarnings <- true
        
        printfn "%s %s: %s" statusSymbol element.Name statusText
        
        if verbose || isError || isWarning then
            printfn "  %s" element.Description
            
        if isError then
            printfn "  Installation: %s" element.InstallHint
            printfn ""
        elif isWarning then
            printfn "  Optional: %s" element.InstallHint
            printfn ""
    )
    
    // Additional LLVM and compiler analysis
    if verbose then
        printfn ""
        printfn "Additional Tool Analysis:"
        printfn "========================="
        
        let llvmTools = CommandDetection.checkLLVMTools()
        llvmTools |> List.iter (fun (name, status) ->
            match status with
            | Found version -> printfn "✓ %s: %s" name version
            | Missing hint -> printfn "✗ %s: %s" name hint
            | Error msg -> printfn "! %s: %s" name msg)
        
        printfn ""
        let compilers = CommandDetection.checkCompilers()
        compilers |> List.iter (fun (name, status) ->
            match status with
            | Found version -> printfn "✓ %s: %s" name version
            | Missing hint -> printfn "✗ %s: %s" name hint
            | Error msg -> printfn "! %s: %s" name msg)
    
    printfn ""
    printfn "=========================================="
    
    if hasErrors then
        printfn "ERROR: Missing required components!"
        printfn "Please install the missing components before using Firefly."
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            printfn ""
            printfn "Quick setup for Windows:"
            printfn "1. Install MSYS2 from https://www.msys2.org/"
            printfn "2. Open 'MSYS2 MINGW64' terminal"
            printfn "3. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-llvm"
            printfn "4. Run: firefly doctor"
        Failure [InternalError("toolchain", "Missing required toolchain components", None)]
    elif hasWarnings then
        printfn "All required components found!"
        printfn "Some optional components are missing but compilation should work."
        Success ()
    else
        printfn "All components found!"
        Success ()

/// Quick check for critical components (used during compilation)
let quickVerifyToolchain() : bool =
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        // Quick check for the most common issues in MSYS2
        let msysCheck = MSYS2Detection.validateMSYS2Environment()
        let gccCheck = CommandDetection.checkCommandAvailability "gcc" "--version"
        let llcCheck = CommandDetection.checkCommandAvailability "llc" "--version"
        
        match msysCheck, gccCheck, llcCheck with
        | Found _, Found _, Found _ -> true
        | Missing _, _, _ ->
            printfn "WARNING: MSYS2 environment issue detected!"
            false
        | _, Missing _, _ ->
            printfn "WARNING: GCC compiler not found!"
            printfn "Install with: pacman -S mingw-w64-x86_64-gcc"
            false
        | _, _, Missing _ ->
            printfn "WARNING: LLVM tools not found!"
            printfn "Install with: pacman -S mingw-w64-x86_64-llvm"
            false
        | _ -> false
    else
        true // Assume OK on other platforms for now

/// Provides specific suggestions for common toolchain issues
let suggestToolchainFixes (error: string) : unit =
    if error.Contains("WinMain") || error.Contains("entry point") then
        printfn ""
        printfn "This error typically indicates missing MinGW-w64 runtime files."
        printfn "Try running: pacman -S mingw-w64-x86_64-crt-git"
        printfn ""
        printfn "If you're not in MINGW64 environment, switch to it:"
        printfn "Close this terminal and open 'MSYS2 MINGW64' from Start Menu"
    elif error.Contains("gcc: command not found") then
        printfn ""
        printfn "GCC compiler not found. Install with:"
        printfn "pacman -S mingw-w64-x86_64-gcc"
    elif error.Contains("llc: command not found") then
        printfn ""
        printfn "LLVM tools not found. Install with:"
        printfn "pacman -S mingw-w64-x86_64-llvm"
    elif error.Contains("ld: cannot find") then
        printfn ""
        printfn "Linker cannot find required libraries. This usually means:"
        printfn "1. Missing MinGW-w64 libraries: pacman -S mingw-w64-x86_64-crt-git"
        printfn "2. Incorrect environment: ensure you're in MINGW64 terminal"
    else
        printfn ""
        printfn "For general toolchain issues:"
        printfn "1. Verify you're in MSYS2 MINGW64 terminal"
        printfn "2. Update package database: pacman -Sy"
        printfn "3. Install core tools: pacman -S mingw-w64-x86_64-toolchain"
        printfn "4. Run: firefly doctor --verbose"