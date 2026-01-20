/// Platform Binding Resolution Nanopass
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This nanopass resolves ALL platform decisions BEFORE witnessing.
/// Witnesses just lookup pre-resolved bindings - they don't decide.
///
/// PSG should be complete before witnessing. Platform decisions
/// (freestanding vs console, syscall vs libc) are resolved here.
module PSGElaboration.PlatformBindingResolution

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open PSGElaboration.PlatformConfig
open Alex.Bindings.PlatformTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Check if intrinsic is a platform intrinsic that needs resolution
let private isPlatformIntrinsic (info: IntrinsicInfo) : bool =
    match info.Module with
    | IntrinsicModule.Sys -> true
    | IntrinsicModule.DateTime ->
        // DateTime.now and utcNow need platform resolution (syscall)
        match info.Operation with
        | "now" | "utcNow" -> true
        | _ -> false  // Component extraction is pure arithmetic
    // NOTE: Console is NOT an intrinsic - see fsnative-spec/spec/platform-bindings.md
    | _ -> false

/// Resolve a single intrinsic to a concrete binding based on runtime mode
let private resolveIntrinsic
    (mode: RuntimeMode)
    (os: OSFamily)
    (info: IntrinsicInfo)
    : ResolvedBinding option =

    match mode with
    | Freestanding ->
        // Direct syscall - look up syscall number for OS
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "write" ->
            let syscallNum = SyscallNumbers.getWriteSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"))
        | IntrinsicModule.Sys, "read" ->
            let syscallNum = SyscallNumbers.getReadSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"))
        | IntrinsicModule.Sys, "exit" ->
            let syscallNum = SyscallNumbers.getExitSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"))
        // NOTE: Console.* is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
        // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md
        | _ -> None  // Not a platform-resolvable intrinsic

    | Console ->
        // libc call - use standard function names
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "write" -> Some (LibcCall "write")
        | IntrinsicModule.Sys, "read" -> Some (LibcCall "read")
        | IntrinsicModule.Sys, "exit" -> Some (LibcCall "exit")
        // NOTE: Console.* is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
        // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md
        | _ -> None  // Not a platform-resolvable intrinsic

// ═══════════════════════════════════════════════════════════════════════════
// _start WRAPPER GENERATION (Freestanding Mode)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the _start wrapper for freestanding mode
/// _start is the true entry point - it sets up argc/argv and calls main
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// The mainReturnType is the resolved platform word type from PlatformConfig.
/// On 64-bit platforms, F# `int` resolves to i64 (platform word = 64 bits).
/// This function observes the coeffect; it does not decide the type.
let private buildStartWrapper (os: OSFamily) (arch: Architecture) (mainReturnType: MLIRType) : MLIROp list =
    match os, arch with
    | Linux, X86_64 ->
        // Linux x86_64 ABI: on entry, stack has:
        // [rsp] = argc (i64)
        // [rsp+8] = argv[0]
        // [rsp+16] = argv[1], etc.
        //
        // We need to:
        // 1. Read argc from stack
        // 2. Compute argv pointer (rsp + 8)
        // 3. Call main(argc, argv)
        // 4. Call exit syscall with main's return value

        // Determine if we need to extend the result to i64 for syscall
        let needsExtension = mainReturnType <> MLIRTypes.i64

        // Build the entry block operations
        let entryOps = [
            // Read argc from [rsp] via inline asm
            // %argc = inline_asm "mov (%rsp), $0", "=r" : () -> i64
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V 0),
                "mov (%rsp), $0",
                "=r",
                [],
                Some MLIRTypes.i64,
                false,
                false
            ))

            // Truncate argc to i32 for main signature (argc is always i32 per C ABI)
            // %argc32 = arith.trunci %argc : i64 to i32
            MLIROp.ArithOp (ArithOp.TruncI (V 1, V 0, MLIRTypes.i64, MLIRTypes.i32))

            // Compute argv pointer: rsp + 8
            // %argv = inline_asm "lea 8(%rsp), $0", "=r" : () -> !llvm.ptr
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V 2),
                "lea 8(%rsp), $0",
                "=r",
                [],
                Some MLIRTypes.ptr,
                false,
                false
            ))

            // Call main(argc, argv) -> mainReturnType (platform word)
            // Use func.call since main is func.func (not llvm.func)
            // MLIR allows mixing dialects within function bodies
            MLIROp.FuncOp (FuncOp.FuncCall (
                Some (V 3),
                "main",
                [{ SSA = V 1; Type = MLIRTypes.i32 }; { SSA = V 2; Type = MLIRTypes.ptr }],
                mainReturnType
            ))
        ]

        // If main returns i64, use V 3 directly; otherwise extend
        let (exitValueSSA, extensionOps) =
            if needsExtension then
                // Extend result to i64 for exit syscall
                let extOp = MLIROp.ArithOp (ArithOp.ExtSI (V 4, V 3, mainReturnType, MLIRTypes.i64))
                (V 4, [extOp])
            else
                // main already returns i64, use directly
                (V 3, [])

        // SSA numbering continues after optional extension
        let nextSSA = if needsExtension then 5 else 4

        let syscallOps = [
            // Exit syscall number
            MLIROp.ArithOp (ArithOp.ConstI (V nextSSA, SyscallNumbers.getExitSyscall Linux, MLIRTypes.i64))

            // Call exit syscall: syscall(60, result) -> i64
            // Syscalls ALWAYS return in rax - model hardware reality consistently
            // The unreachable after means LLVM will optimize away the unused result
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V (nextSSA + 1)),
                "syscall",
                "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}",
                [(V nextSSA, MLIRTypes.i64); (exitValueSSA, MLIRTypes.i64)],
                Some MLIRTypes.i64,
                true,
                false
            ))

            // Unreachable - exit never returns, LLVM will optimize away the result
            MLIROp.LLVMOp LLVMOp.Unreachable
        ]

        let allOps = entryOps @ extensionOps @ syscallOps

        // Build _start function
        let entryBlock: Block = {
            Label = BlockRef "entry"
            Args = []
            Ops = allOps
        }

        let bodyRegion: Region = { Blocks = [entryBlock] }

        [
            MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (
                "_start",
                [],  // No parameters - we read from stack
                MLIRTypes.i32,  // Dummy return type (never returns)
                bodyRegion,
                LLVMLinkage.LLVMExternal
            ))
        ]

    | MacOS, X86_64 ->
        // macOS x86_64 has similar entry convention to Linux
        let needsExtension = mainReturnType <> MLIRTypes.i64

        let entryOps = [
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V 0),
                "mov (%rsp), $0",
                "=r",
                [],
                Some MLIRTypes.i64,
                false,
                false
            ))
            MLIROp.ArithOp (ArithOp.TruncI (V 1, V 0, MLIRTypes.i64, MLIRTypes.i32))
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V 2),
                "lea 8(%rsp), $0",
                "=r",
                [],
                Some MLIRTypes.ptr,
                false,
                false
            ))
            // Use func.call since main is func.func (not llvm.func)
            MLIROp.FuncOp (FuncOp.FuncCall (
                Some (V 3),
                "main",
                [{ SSA = V 1; Type = MLIRTypes.i32 }; { SSA = V 2; Type = MLIRTypes.ptr }],
                mainReturnType
            ))
        ]

        let (exitValueSSA, extensionOps) =
            if needsExtension then
                let extOp = MLIROp.ArithOp (ArithOp.ExtSI (V 4, V 3, mainReturnType, MLIRTypes.i64))
                (V 4, [extOp])
            else
                (V 3, [])

        let nextSSA = if needsExtension then 5 else 4

        let syscallOps = [
            MLIROp.ArithOp (ArithOp.ConstI (V nextSSA, SyscallNumbers.getExitSyscall MacOS, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some (V (nextSSA + 1)),
                "syscall",
                "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}",
                [(V nextSSA, MLIRTypes.i64); (exitValueSSA, MLIRTypes.i64)],
                Some MLIRTypes.i64,
                true,
                false
            ))
            MLIROp.LLVMOp LLVMOp.Unreachable
        ]

        let allOps = entryOps @ extensionOps @ syscallOps

        let entryBlock: Block = {
            Label = BlockRef "entry"
            Args = []
            Ops = allOps
        }

        [
            MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (
                "_start",
                [],
                MLIRTypes.i32,
                { Blocks = [entryBlock] },
                LLVMLinkage.LLVMExternal
            ))
        ]

    | _, _ ->
        // Other platforms - generate stub that fails at runtime
        failwithf "_start wrapper not implemented for %A/%A" os arch

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ANALYSIS ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze the semantic graph and resolve all platform bindings
/// This is the main nanopass entry point
let analyze
    (graph: SemanticGraph)
    (mode: RuntimeMode)
    (os: OSFamily)
    (arch: Architecture)
    : PlatformResolutionResult =

    // Resolve platform word type for this architecture
    // This is the authoritative source for what PlatformWord means on this target
    let wordType = platformWordType arch

    // Walk graph, find all Intrinsic nodes that need platform resolution
    let bindings =
        graph.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            // Extract int value from NodeId wrapper
            let (NodeId nodeIdInt) = nodeId
            match node.Kind with
            | SemanticKind.Intrinsic intrinsicInfo when isPlatformIntrinsic intrinsicInfo ->
                match resolveIntrinsic mode os intrinsicInfo with
                | Some resolved ->
                    let entryPoint = sprintf "%s.%s" (string intrinsicInfo.Module) intrinsicInfo.Operation
                    Some (nodeIdInt, {
                        NodeId = nodeIdInt
                        EntryPoint = entryPoint
                        Mode = mode
                        Resolved = resolved
                    })
                | None -> None
            | _ -> None)
        |> Map.ofSeq

    // Build _start wrapper if freestanding
    // Pass the platform word type so _start knows main's return type
    let needsStart = (mode = Freestanding)
    let startOps =
        if needsStart then
            try Some (buildStartWrapper os arch wordType)
            with _ -> None  // If we can't build _start, continue without it
        else None

    {
        RuntimeMode = mode
        TargetOS = os
        TargetArch = arch
        PlatformWordType = wordType
        Bindings = bindings
        NeedsStartWrapper = needsStart
        StartWrapperOps = startOps
    }

// ═══════════════════════════════════════════════════════════════════════════
// LOOKUP HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Lookup a resolved binding by node ID
let lookupBinding (nodeId: int) (result: PlatformResolutionResult) : BindingResolution option =
    Map.tryFind nodeId result.Bindings

/// Check if a node has a platform binding resolution
let hasBinding (nodeId: int) (result: PlatformResolutionResult) : bool =
    Map.containsKey nodeId result.Bindings

/// Get all resolved bindings
let allBindings (result: PlatformResolutionResult) : BindingResolution list =
    result.Bindings |> Map.toList |> List.map snd
