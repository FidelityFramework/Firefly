/// ProcessBindings - Platform-specific process bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.Process.ProcessBindings

open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.CF.Templates
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data: Syscall numbers
// ===================================================================

module SyscallData =
    /// Linux x86-64 syscall numbers
    let linuxExit = 60L
    let linuxExitGroup = 231L

    /// macOS syscall numbers (with BSD 0x2000000 offset for x86-64)
    let macosExit = 0x2000001L

// ===================================================================
// Helpers
// ===================================================================

/// Helper: wrap SSA as i64 typed arg (syscall convention)
let inline private i64Arg (ssa: SSA) = (ssa, MLIRTypes.i64)

// ===================================================================
// Linux Process Implementation
// ===================================================================

/// Generate exit syscall for Linux x86-64
let bindLinuxExit (z: PSGZipper) (codeSSA: SSA) (codeType: MLIRType) : MLIROp list =
    // Extend exit code to i64 if needed
    let codeExt, extOps =
        match codeType with
        | TInt I64 -> codeSSA, []
        | TInt _ ->
            let ext = freshSynthSSA z
            ext, [MLIROp.ArithOp (extSI ext codeSSA codeType MLIRTypes.i64)]
        | _ -> codeSSA, []

    // Syscall number
    let sysNumSSA = freshSynthSSA z
    let sysNumOp = MLIROp.ArithOp (constI sysNumSSA SyscallData.linuxExit MLIRTypes.i64)

    // Execute syscall: exit(code)
    let syscallResultSSA = freshSynthSSA z
    let syscallOp = MLIROp.LLVMOp (inlineAsmWithSideEffects (Some syscallResultSSA) "syscall"
        "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"
        [i64Arg sysNumSSA; i64Arg codeExt] (Some MLIRTypes.i64))

    // Unreachable - exit never returns
    let unreachableOp = MLIROp.LLVMOp Unreachable

    extOps @ [sysNumOp; syscallOp; unreachableOp]

// ===================================================================
// macOS Process Implementation
// ===================================================================

/// Generate exit syscall for macOS x86-64
let bindMacOSExit (z: PSGZipper) (codeSSA: SSA) (codeType: MLIRType) : MLIROp list =
    // Extend exit code to i64 if needed
    let codeExt, extOps =
        match codeType with
        | TInt I64 -> codeSSA, []
        | TInt _ ->
            let ext = freshSynthSSA z
            ext, [MLIROp.ArithOp (extSI ext codeSSA codeType MLIRTypes.i64)]
        | _ -> codeSSA, []

    // macOS syscall number (with BSD offset)
    let sysNumSSA = freshSynthSSA z
    let sysNumOp = MLIROp.ArithOp (constI sysNumSSA SyscallData.macosExit MLIRTypes.i64)

    // Execute syscall: exit(code)
    let syscallResultSSA = freshSynthSSA z
    let syscallOp = MLIROp.LLVMOp (inlineAsmWithSideEffects (Some syscallResultSSA) "syscall"
        "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"
        [i64Arg sysNumSSA; i64Arg codeExt] (Some MLIRTypes.i64))

    // Unreachable - exit never returns
    let unreachableOp = MLIROp.LLVMOp Unreachable

    extOps @ [sysNumOp; syscallOp; unreachableOp]

// ===================================================================
// Platform Primitive Bindings
// ===================================================================

/// exit - terminate process with exit code
let bindExit (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [code] ->
        match os with
        | Linux ->
            let ops = bindLinuxExit z code.SSA code.Type
            BoundOps (ops, None)  // exit doesn't return
        | MacOS ->
            let ops = bindMacOSExit z code.SSA code.Type
            BoundOps (ops, None)  // exit doesn't return
        | _ ->
            NotSupported $"Exit not supported on {os}"
    | _ ->
        NotSupported "exit requires (exitCode) argument"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Linux x86_64
    PlatformDispatch.register Linux X86_64 "exit" (bindExit Linux)
    PlatformDispatch.register Linux X86_64 "Sys.exit" (bindExit Linux)

    // Linux ARM64
    PlatformDispatch.register Linux ARM64 "exit" (bindExit Linux)
    PlatformDispatch.register Linux ARM64 "Sys.exit" (bindExit Linux)

    // macOS x86_64
    PlatformDispatch.register MacOS X86_64 "exit" (bindExit MacOS)
    PlatformDispatch.register MacOS X86_64 "Sys.exit" (bindExit MacOS)

    // macOS ARM64
    PlatformDispatch.register MacOS ARM64 "exit" (bindExit MacOS)
    PlatformDispatch.register MacOS ARM64 "Sys.exit" (bindExit MacOS)
