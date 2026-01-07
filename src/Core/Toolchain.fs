/// Toolchain - External tool invocations for MLIR/LLVM compilation
///
/// This module isolates ALL external process calls to LLVM toolchain binaries.
/// When Firefly becomes self-hosted, this is the ONLY module that needs replacement.
///
/// Current implementation: shells out to mlir-opt, mlir-translate, llc, clang
/// Future implementation: native MLIR/LLVM bindings via Alloy
module Core.Toolchain

open System
open System.IO

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
let lowerMLIRToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    try
        // Step 1: mlir-opt to convert to LLVM dialect
        // Conversion order: scf -> cf -> func -> arith -> llvm, then cleanup casts
        let mlirOptArgs = sprintf "%s --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts" mlirPath
        let mlirOptProcess = new System.Diagnostics.Process()
        mlirOptProcess.StartInfo.FileName <- "mlir-opt"
        mlirOptProcess.StartInfo.Arguments <- mlirOptArgs
        mlirOptProcess.StartInfo.UseShellExecute <- false
        mlirOptProcess.StartInfo.RedirectStandardOutput <- true
        mlirOptProcess.StartInfo.RedirectStandardError <- true
        mlirOptProcess.Start() |> ignore
        let mlirOptOutput = mlirOptProcess.StandardOutput.ReadToEnd()
        let mlirOptError = mlirOptProcess.StandardError.ReadToEnd()
        mlirOptProcess.WaitForExit()

        if mlirOptProcess.ExitCode <> 0 then
            Error (sprintf "mlir-opt failed: %s" mlirOptError)
        else
            // Step 2: mlir-translate to convert LLVM dialect to LLVM IR
            let mlirTranslateProcess = new System.Diagnostics.Process()
            mlirTranslateProcess.StartInfo.FileName <- "mlir-translate"
            mlirTranslateProcess.StartInfo.Arguments <- "--mlir-to-llvmir"
            mlirTranslateProcess.StartInfo.UseShellExecute <- false
            mlirTranslateProcess.StartInfo.RedirectStandardInput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardOutput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardError <- true
            mlirTranslateProcess.Start() |> ignore
            mlirTranslateProcess.StandardInput.Write(mlirOptOutput)
            mlirTranslateProcess.StandardInput.Close()
            let llvmOutput = mlirTranslateProcess.StandardOutput.ReadToEnd()
            let translateError = mlirTranslateProcess.StandardError.ReadToEnd()
            mlirTranslateProcess.WaitForExit()

            if mlirTranslateProcess.ExitCode <> 0 then
                Error (sprintf "mlir-translate failed: %s" translateError)
            else
                File.WriteAllText(llvmPath, llvmOutput)
                Ok ()
    with ex ->
        Error (sprintf "MLIR lowering failed: %s" ex.Message)

/// Compile LLVM IR to native binary using llc and clang
let compileLLVMToNative
    (llvmPath: string)
    (outputPath: string)
    (targetTriple: string)
    (outputKind: Core.Types.MLIRTypes.OutputKind) : Result<unit, string> =
    try
        let objPath = Path.ChangeExtension(llvmPath, ".o")

        // Step 1: llc to compile LLVM IR to object file
        let llcArgs = sprintf "-O0 -filetype=obj %s -o %s" llvmPath objPath
        let llcProcess = new System.Diagnostics.Process()
        llcProcess.StartInfo.FileName <- "llc"
        llcProcess.StartInfo.Arguments <- llcArgs
        llcProcess.StartInfo.UseShellExecute <- false
        llcProcess.StartInfo.RedirectStandardError <- true
        llcProcess.Start() |> ignore
        let llcError = llcProcess.StandardError.ReadToEnd()
        llcProcess.WaitForExit()

        if llcProcess.ExitCode <> 0 then
            Error (sprintf "llc failed: %s" llcError)
        else
            // Step 2: clang to link into executable
            let clangArgs =
                match outputKind with
                | Core.Types.MLIRTypes.Console ->
                    // Use -no-pie to avoid relocation issues with LLVM-generated code
                    // Include webview dependencies: webkit2gtk and gtk3
                    sprintf "-O0 -no-pie %s -o %s -lc -lwebkit2gtk-4.0 -lgtk-3 -lgobject-2.0 -lglib-2.0" objPath outputPath
                | Core.Types.MLIRTypes.Freestanding | Core.Types.MLIRTypes.Embedded ->
                    // Use _start as entry point - it handles argc/argv and calls exit syscall
                    sprintf "-O0 %s -o %s -nostdlib -static -ffreestanding -Wl,-e,_start" objPath outputPath
                | Core.Types.MLIRTypes.Library ->
                    sprintf "-O0 -shared %s -o %s" objPath outputPath

            let clangProcess = new System.Diagnostics.Process()
            clangProcess.StartInfo.FileName <- "clang"
            clangProcess.StartInfo.Arguments <- clangArgs
            clangProcess.StartInfo.UseShellExecute <- false
            clangProcess.StartInfo.RedirectStandardError <- true
            clangProcess.Start() |> ignore
            let clangError = clangProcess.StandardError.ReadToEnd()
            clangProcess.WaitForExit()

            if clangProcess.ExitCode <> 0 then
                Error (sprintf "clang failed: %s" clangError)
            else
                // Clean up object file
                if File.Exists(objPath) then
                    File.Delete(objPath)
                Ok ()
    with ex ->
        Error (sprintf "Native compilation failed: %s" ex.Message)

/// Get default target triple based on host platform
let getDefaultTarget() =
    if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows) then
        "x86_64-pc-windows-gnu"
    elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then
        "x86_64-unknown-linux-gnu"
    elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX) then
        "x86_64-apple-darwin"
    else
        "x86_64-unknown-linux-gnu"
