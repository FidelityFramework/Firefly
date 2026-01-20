/// SyscallBindings - Linux x86_64 syscall bindings via PlatformDispatch
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Platform-specific syscall logic lives HERE, not in witnesses.
/// Witnesses dispatch through PlatformDispatch; bindings generate MLIR.
module Alex.Bindings.SyscallBindings

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════════════
// LINUX X86_64 SYSCALL NUMBERS
// ═══════════════════════════════════════════════════════════════════════════

// These are Linux x86_64 syscall numbers
// See: /usr/include/asm/unistd_64.h
let private SYSCALL_READ = 0L
let private SYSCALL_WRITE = 1L
let private SYSCALL_NANOSLEEP = 35L
let private SYSCALL_CLOCK_GETTIME = 228L

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Build syscall inline asm
// ═══════════════════════════════════════════════════════════════════════════

/// Linux x86_64 syscall constraint string for 3-arg syscalls (rdi, rsi, rdx)
let private syscall3ArgConstraints = "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"

/// Linux x86_64 syscall constraint string for 2-arg syscalls (rdi, rsi)
let private syscall2ArgConstraints = "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"

// ═══════════════════════════════════════════════════════════════════════════
// SYSCALL BINDINGS
// ═══════════════════════════════════════════════════════════════════════════

/// Sys.write: fd:int -> ptr:nativeptr<byte> -> len:int -> int
/// Linux syscall 1 = write(fd, buf, count)
/// ARCHITECTURAL NOTE: The syscall returns i64 (ssize_t on LP64).
/// prim.ReturnType MUST match - if it doesn't, that's a type system bug to surface.
let private bindSysWrite (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    // Type check: syscall returns i64, FNCS should resolve int → i64 on 64-bit
    if prim.ReturnType <> MLIRTypes.i64 then
        NotSupported $"Sys.write return type mismatch: expected i64 (ssize_t), got {prim.ReturnType}. Check FNCS type resolution for 'int' on this platform."
    else
        match prim.Args with
        | [fdVal; ptrVal; lenVal] ->
            let ssas = requireNodeSSAs appNodeId z
            // SSAs: syscallNum[0], resultSSA[1], fdExt[2], lenExt[3]
            let syscallNum = ssas.[0]
            let resultSSA = ssas.[1]

            // Build fd extension ops (if needed)
            let fdOps, fdFinal =
                if fdVal.Type = MLIRTypes.i64 then
                    [], fdVal.SSA
                else
                    let fdExt = ssas.[2]
                    [MLIROp.ArithOp (ArithOp.ExtSI (fdExt, fdVal.SSA, fdVal.Type, MLIRTypes.i64))], fdExt

            // Build len extension ops (if needed)
            let lenOps, lenFinal =
                if lenVal.Type = MLIRTypes.i64 then
                    [], lenVal.SSA
                else
                    let lenExt = ssas.[3]
                    [MLIROp.ArithOp (ArithOp.ExtSI (lenExt, lenVal.SSA, lenVal.Type, MLIRTypes.i64))], lenExt

            let ops =
                [MLIROp.ArithOp (ArithOp.ConstI (syscallNum, SYSCALL_WRITE, MLIRTypes.i64))]
                @ fdOps
                @ lenOps
                @ [
                    MLIROp.LLVMOp (LLVMOp.InlineAsm (
                        Some resultSSA,
                        "syscall",
                        syscall3ArgConstraints,
                        [(syscallNum, MLIRTypes.i64)
                         (fdFinal, MLIRTypes.i64)
                         (ptrVal.SSA, MLIRTypes.ptr)
                         (lenFinal, MLIRTypes.i64)],
                        Some MLIRTypes.i64,
                        true,
                        false))
                ]
            BoundOps (ops, Some { SSA = resultSSA; Type = MLIRTypes.i64 })
        | _ ->
            NotSupported "Sys.write requires 3 arguments: fd, ptr, len"

/// Sys.read: fd:int -> ptr:nativeptr<byte> -> len:int -> int
/// Linux syscall 0 = read(fd, buf, count)
/// ARCHITECTURAL NOTE: The syscall returns i64 (ssize_t on LP64).
/// prim.ReturnType MUST match - if it doesn't, that's a type system bug to surface.
let private bindSysRead (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    // Type check: syscall returns i64, FNCS should resolve int → i64 on 64-bit
    if prim.ReturnType <> MLIRTypes.i64 then
        NotSupported $"Sys.read return type mismatch: expected i64 (ssize_t), got {prim.ReturnType}. Check FNCS type resolution for 'int' on this platform."
    else
        match prim.Args with
        | [fdVal; ptrVal; lenVal] ->
            let ssas = requireNodeSSAs appNodeId z
            // SSAs: syscallNum[0], resultSSA[1], fdExt[2], lenExt[3]
            let syscallNum = ssas.[0]
            let resultSSA = ssas.[1]

            // Build fd extension ops (if needed)
            let fdOps, fdFinal =
                if fdVal.Type = MLIRTypes.i64 then
                    [], fdVal.SSA
                else
                    let fdExt = ssas.[2]
                    [MLIROp.ArithOp (ArithOp.ExtSI (fdExt, fdVal.SSA, fdVal.Type, MLIRTypes.i64))], fdExt

            // Build len extension ops (if needed)
            let lenOps, lenFinal =
                if lenVal.Type = MLIRTypes.i64 then
                    [], lenVal.SSA
                else
                    let lenExt = ssas.[3]
                    [MLIROp.ArithOp (ArithOp.ExtSI (lenExt, lenVal.SSA, lenVal.Type, MLIRTypes.i64))], lenExt

            let ops =
                [MLIROp.ArithOp (ArithOp.ConstI (syscallNum, SYSCALL_READ, MLIRTypes.i64))]
                @ fdOps
                @ lenOps
                @ [
                    MLIROp.LLVMOp (LLVMOp.InlineAsm (
                        Some resultSSA,
                        "syscall",
                        syscall3ArgConstraints,
                        [(syscallNum, MLIRTypes.i64)
                         (fdFinal, MLIRTypes.i64)
                         (ptrVal.SSA, MLIRTypes.ptr)
                         (lenFinal, MLIRTypes.i64)],
                        Some MLIRTypes.i64,
                        true,
                        false))
                ]
            BoundOps (ops, Some { SSA = resultSSA; Type = MLIRTypes.i64 })
        | _ ->
            NotSupported "Sys.read requires 3 arguments: fd, ptr, len"

/// Sys.clock_gettime: unit -> int64
/// Linux syscall 228 = clock_gettime(clockid, timespec*)
/// Returns milliseconds since epoch as int64: (tv_sec * 1000) + (tv_nsec / 1_000_000)
let private bindSysClockGettime (appNodeId: NodeId) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let ssas = requireNodeSSAs appNodeId z
    // SSAs: clockId[0], one[1], allocaSSA[2], syscallNum[3], resultSSA[4],
    //       structVal[5], secVal[6], nsecVal[7],
    //       thousand[8], million[9], secMs[10], nsecMs[11], totalMs[12]
    let clockId = ssas.[0]
    let one = ssas.[1]
    let allocaSSA = ssas.[2]
    let syscallNum = ssas.[3]
    let resultSSA = ssas.[4]
    let structVal = ssas.[5]
    let secVal = ssas.[6]
    let nsecVal = ssas.[7]
    let thousand = ssas.[8]
    let million = ssas.[9]
    let secMs = ssas.[10]
    let nsecMs = ssas.[11]
    let totalMs = ssas.[12]

    // timespec struct type: { i64, i64 } = { tv_sec, tv_nsec }
    let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

    let ops = [
        // Allocate timespec on stack
        MLIROp.ArithOp (ArithOp.ConstI (one, 1L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, one, timespecType, None))
        // CLOCK_REALTIME = 0
        MLIROp.ArithOp (ArithOp.ConstI (clockId, 0L, MLIRTypes.i64))
        // Syscall 228
        MLIROp.ArithOp (ArithOp.ConstI (syscallNum, SYSCALL_CLOCK_GETTIME, MLIRTypes.i64))
        // Call clock_gettime(CLOCK_REALTIME, &timespec)
        MLIROp.LLVMOp (LLVMOp.InlineAsm (
            Some resultSSA,
            "syscall",
            syscall2ArgConstraints,
            [(syscallNum, MLIRTypes.i64)
             (clockId, MLIRTypes.i64)
             (allocaSSA, MLIRTypes.ptr)],
            Some MLIRTypes.i64,
            true,
            false))
        // Load the entire struct, then extract fields
        MLIROp.LLVMOp (LLVMOp.Load (structVal, allocaSSA, timespecType, AtomicOrdering.NotAtomic))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (secVal, structVal, [0], timespecType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (nsecVal, structVal, [1], timespecType))
        // Compute milliseconds: (tv_sec * 1000) + (tv_nsec / 1_000_000)
        MLIROp.ArithOp (ArithOp.ConstI (thousand, 1000L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (million, 1000000L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.MulI (secMs, secVal, thousand, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.DivSI (nsecMs, nsecVal, million, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AddI (totalMs, secMs, nsecMs, MLIRTypes.i64))
    ]
    // Return total milliseconds since epoch
    BoundOps (ops, Some { SSA = totalMs; Type = MLIRTypes.i64 })

/// Sys.nanosleep: int64 -> unit
/// Linux syscall 35 = nanosleep(timespec*, timespec*)
/// Takes nanoseconds to sleep, converts to timespec internally
let private bindSysNanosleep (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [nsVal] ->
        let ssas = requireNodeSSAs appNodeId z
        // SSAs: nsExt[0], one[1], allocaSSA[2], billion[3], secCalc[4], nsecCalc[5],
        //       idx0[6], secPtr[7], idx1[8], nsecPtr[9], nullPtr[10], syscallNum[11], resultSSA[12]
        let nsExt = ssas.[0]
        let one = ssas.[1]
        let allocaSSA = ssas.[2]
        let billion = ssas.[3]
        let secCalc = ssas.[4]
        let nsecCalc = ssas.[5]
        let idx0 = ssas.[6]
        let secPtr = ssas.[7]
        let idx1 = ssas.[8]
        let nsecPtr = ssas.[9]
        let nullPtr = ssas.[10]
        let syscallNum = ssas.[11]
        let resultSSA = ssas.[12]

        // timespec struct type: { i64, i64 }
        let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

        // Extend input to i64 if needed
        let nsSSA, extOps =
            if nsVal.Type = MLIRTypes.i64 then
                nsVal.SSA, []
            else
                // Need sign extension from i32 to i64
                nsExt, [MLIROp.ArithOp (ArithOp.ExtSI (nsExt, nsVal.SSA, nsVal.Type, MLIRTypes.i64))]

        let ops = extOps @ [
            // Allocate timespec on stack
            MLIROp.ArithOp (ArithOp.ConstI (one, 1L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, one, timespecType, None))
            // 1_000_000_000
            MLIROp.ArithOp (ArithOp.ConstI (billion, 1000000000L, MLIRTypes.i64))
            // sec = ns / 1_000_000_000
            MLIROp.ArithOp (ArithOp.DivSI (secCalc, nsSSA, billion, MLIRTypes.i64))
            // nsec = ns % 1_000_000_000
            MLIROp.ArithOp (ArithOp.RemSI (nsecCalc, nsSSA, billion, MLIRTypes.i64))
            // Create index 0 for tv_sec field
            MLIROp.ArithOp (ArithOp.ConstI (idx0, 0L, MLIRTypes.i64))
            // GEP to tv_sec field
            MLIROp.LLVMOp (LLVMOp.GEP (secPtr, allocaSSA, [(idx0, MLIRTypes.i64)], timespecType))
            MLIROp.LLVMOp (LLVMOp.Store (secCalc, secPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))
            // Create index 1 for tv_nsec field
            MLIROp.ArithOp (ArithOp.ConstI (idx1, 1L, MLIRTypes.i64))
            // GEP to tv_nsec field
            MLIROp.LLVMOp (LLVMOp.GEP (nsecPtr, allocaSSA, [(idx1, MLIRTypes.i64)], timespecType))
            MLIROp.LLVMOp (LLVMOp.Store (nsecCalc, nsecPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))
            // Null pointer for remaining time
            MLIROp.LLVMOp (LLVMOp.NullPtr nullPtr)
            // Syscall 35
            MLIROp.ArithOp (ArithOp.ConstI (syscallNum, SYSCALL_NANOSLEEP, MLIRTypes.i64))
            // Call nanosleep(&timespec, NULL)
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some resultSSA,
                "syscall",
                syscall2ArgConstraints,
                [(syscallNum, MLIRTypes.i64)
                 (allocaSSA, MLIRTypes.ptr)
                 (nullPtr, MLIRTypes.ptr)],
                Some MLIRTypes.i64,
                true,
                false))
        ]
        BoundOps (ops, None)  // Returns unit
    | _ ->
        NotSupported "Sys.nanosleep requires 1 argument: nanoseconds"

// ═══════════════════════════════════════════════════════════════════════════
// REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Register all syscall bindings for Linux x86_64
let registerBindings () =
    // Linux x86_64 specific syscalls
    PlatformDispatch.register Linux X86_64 "Sys.write" bindSysWrite
    PlatformDispatch.register Linux X86_64 "Sys.read" bindSysRead
    PlatformDispatch.register Linux X86_64 "Sys.clock_gettime" bindSysClockGettime
    PlatformDispatch.register Linux X86_64 "Sys.nanosleep" bindSysNanosleep

    // ARM64 stubs (different syscall numbers but same semantics)
    // TODO: Add ARM64 syscall numbers when needed
    PlatformDispatch.register Linux ARM64 "Sys.write" bindSysWrite
    PlatformDispatch.register Linux ARM64 "Sys.read" bindSysRead
    PlatformDispatch.register Linux ARM64 "Sys.clock_gettime" bindSysClockGettime
    PlatformDispatch.register Linux ARM64 "Sys.nanosleep" bindSysNanosleep
