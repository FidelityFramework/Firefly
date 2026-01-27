/// Platform operation witnesses (Sys.write, Sys.read, etc.)
/// Following Farscape quotation-based pattern: recognize + decompose
module Alex.Witnesses.PlatformWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM OPERATION WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Sys.write operation
let witnessSysWrite (ctx: WitnessContext) (node: SemanticNode) (platformModel: PlatformModel) : WitnessOutput =
    match platformModel.Recognize ctx.Graph node with
    | Some (PlatformOperation.SysWrite (fd, bufferSSA, countSSA)) ->
        // Generate MLIR based on runtime mode
        match platformModel.RuntimeMode with
        | RuntimeMode.Freestanding ->
            // Direct syscall via inline assembly
            let syscallNum = SyscallNumbers.getWriteSyscall platformModel.TargetOS
            let resultSSA = ctx.Coeffects.SSA.freshSSA()

            // Inline assembly for syscall
            let inlineAsmOp = MLIROp.LLVMInlineAsm {
                Template = "syscall"
                Constraints = "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                SideEffects = true
                Result = resultSSA
                Args = [
                    (SSA (sprintf "syscall_num_%d" syscallNum), TInt I64)
                    (SSA (sprintf "fd_%d" fd), TInt I32)
                    (bufferSSA, TNativePtr (TInt I8))
                    (countSSA, TInt I64)
                ]
            }

            { InlineOps = [inlineAsmOp]; TopLevelOps = []; Result = Val (resultSSA, TInt I64) }

        | RuntimeMode.Console ->
            // Call to libc write function
            let resultSSA = ctx.Coeffects.SSA.freshSSA()
            let writeCall = MLIROp.LLVMCall {
                Callee = "write"
                Args = [
                    (SSA (sprintf "fd_%d" fd), TInt I32)
                    (bufferSSA, TNativePtr (TInt I8))
                    (countSSA, TInt I64)
                ]
                Result = resultSSA
                ResultType = TInt I64
            }

            { InlineOps = [writeCall]; TopLevelOps = []; Result = Val (resultSSA, TInt I64) }

    | _ ->
        WitnessOutput.error "Not a Sys.write operation"

/// Witness Sys.read operation
let witnessSysRead (ctx: WitnessContext) (node: SemanticNode) (platformModel: PlatformModel) : WitnessOutput =
    match platformModel.Recognize ctx.Graph node with
    | Some (PlatformOperation.SysRead (fd, bufferSSA, countSSA)) ->
        match platformModel.RuntimeMode with
        | RuntimeMode.Freestanding ->
            let syscallNum = SyscallNumbers.getReadSyscall platformModel.TargetOS
            let resultSSA = ctx.Coeffects.SSA.freshSSA()

            let inlineAsmOp = MLIROp.LLVMInlineAsm {
                Template = "syscall"
                Constraints = "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                SideEffects = true
                Result = resultSSA
                Args = [
                    (SSA (sprintf "syscall_num_%d" syscallNum), TInt I64)
                    (SSA (sprintf "fd_%d" fd), TInt I32)
                    (bufferSSA, TNativePtr (TInt I8))
                    (countSSA, TInt I64)
                ]
            }

            { InlineOps = [inlineAsmOp]; TopLevelOps = []; Result = Val (resultSSA, TInt I64) }

        | RuntimeMode.Console ->
            let resultSSA = ctx.Coeffects.SSA.freshSSA()
            let readCall = MLIROp.LLVMCall {
                Callee = "read"
                Args = [
                    (SSA (sprintf "fd_%d" fd), TInt I32)
                    (bufferSSA, TNativePtr (TInt I8))
                    (countSSA, TInt I64)
                ]
                Result = resultSSA
                ResultType = TInt I64
            }

            { InlineOps = [readCall]; TopLevelOps = []; Result = Val (resultSSA, TInt I64) }

    | _ ->
        WitnessOutput.error "Not a Sys.read operation"

/// Witness Sys.exit operation
let witnessSysExit (ctx: WitnessContext) (node: SemanticNode) (platformModel: PlatformModel) : WitnessOutput =
    match platformModel.Recognize ctx.Graph node with
    | Some (PlatformOperation.SysExit codeSSA) ->
        match platformModel.RuntimeMode with
        | RuntimeMode.Freestanding ->
            let syscallNum = SyscallNumbers.getExitSyscall platformModel.TargetOS

            let inlineAsmOp = MLIROp.LLVMInlineAsm {
                Template = "syscall"
                Constraints = "={rax},{rax},{rdi}"
                SideEffects = true
                Result = SSA "exit_noreturn"
                Args = [
                    (SSA (sprintf "syscall_num_%d" syscallNum), TInt I64)
                    (codeSSA, TInt I32)
                ]
            }

            { InlineOps = [inlineAsmOp]; TopLevelOps = []; Result = UnitVal }

        | RuntimeMode.Console ->
            let exitCall = MLIROp.LLVMCall {
                Callee = "exit"
                Args = [(codeSSA, TInt I32)]
                Result = SSA "exit_noreturn"
                ResultType = TVoid
            }

            { InlineOps = [exitCall]; TopLevelOps = []; Result = UnitVal }

    | _ ->
        WitnessOutput.error "Not a Sys.exit operation"

/// General platform operation witness (dispatches to specific witness)
let witnessPlatformOperation (ctx: WitnessContext) (node: SemanticNode) (platformModel: PlatformModel) : WitnessOutput =
    match platformModel.Recognize ctx.Graph node with
    | Some (PlatformOperation.SysWrite _) -> witnessSysWrite ctx node platformModel
    | Some (PlatformOperation.SysRead _) -> witnessSysRead ctx node platformModel
    | Some (PlatformOperation.SysExit _) -> witnessSysExit ctx node platformModel
    | Some (PlatformOperation.SysNanosleep _) ->
        WitnessOutput.error "Sys.nanosleep not yet implemented"
    | Some PlatformOperation.SysClockGetTime ->
        WitnessOutput.error "Sys.clock_gettime not yet implemented"
    | None ->
        WitnessOutput.error "Not a recognized platform operation"
