/// TimeBindings - Platform-specific time bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.Time.TimeBindings

open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data
// ===================================================================

module SyscallData =
    let linuxClockGettime = 228L
    let linuxNanosleep = 35L
    let CLOCK_REALTIME = 0L
    let CLOCK_MONOTONIC = 1L

// ===================================================================
// Common Types
// ===================================================================

/// timespec struct: { tv_sec: i64, tv_nsec: i64 }
let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

// ===================================================================
// Helpers
// ===================================================================

/// Helper: wrap SSA as i64 typed arg (syscall convention)
let inline private i64Arg (ssa: SSA) = (ssa, MLIRTypes.i64)

/// Helper: wrap SSA as ptr typed arg
let inline private ptrArg (ssa: SSA) = (ssa, MLIRTypes.ptr)

// ===================================================================
// Linux Time Implementation
// ===================================================================

/// Generate clock_gettime syscall, returns ticks (100-ns intervals)
let bindLinuxClockGettime (clockId: int64) (z: PSGZipper) : MLIROp list * SSA =
    // Allocate timespec on stack
    let oneSSA = freshSynthSSA z
    let timespecSSA = freshSynthSSA z
    let allocOps = [
        MLIROp.ArithOp (constI oneSSA 1L MLIRTypes.i64)
        MLIROp.LLVMOp (alloca timespecSSA oneSSA timespecType None)
    ]

    // Syscall args
    let clockIdSSA = freshSynthSSA z
    let sysNumSSA = freshSynthSSA z
    let syscallResultSSA = freshSynthSSA z
    let syscallOps = [
        MLIROp.ArithOp (constI clockIdSSA clockId MLIRTypes.i64)
        MLIROp.ArithOp (constI sysNumSSA SyscallData.linuxClockGettime MLIRTypes.i64)
        MLIROp.LLVMOp (inlineAsmWithSideEffects (Some syscallResultSSA) "syscall"
            "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
            [i64Arg sysNumSSA; i64Arg clockIdSSA; ptrArg timespecSSA] (Some MLIRTypes.i64))
    ]

    // Extract seconds and nanoseconds via GEP + load
    let zeroSSA = freshSynthSSA z
    let secIdxSSA = freshSynthSSA z
    let nsecIdxSSA = freshSynthSSA z
    let secPtrSSA = freshSynthSSA z
    let nsecPtrSSA = freshSynthSSA z
    let secSSA = freshSynthSSA z
    let nsecSSA = freshSynthSSA z
    let extractOps = [
        MLIROp.ArithOp (constI zeroSSA 0L MLIRTypes.i64)
        MLIROp.ArithOp (constI secIdxSSA 0L MLIRTypes.i64)
        MLIROp.ArithOp (constI nsecIdxSSA 1L MLIRTypes.i64)
        MLIROp.LLVMOp (gep secPtrSSA timespecSSA [(zeroSSA, MLIRTypes.i64); (secIdxSSA, MLIRTypes.i64)] timespecType)
        MLIROp.LLVMOp (gep nsecPtrSSA timespecSSA [(zeroSSA, MLIRTypes.i64); (nsecIdxSSA, MLIRTypes.i64)] timespecType)
        MLIROp.LLVMOp (loadNonAtomic secSSA secPtrSSA MLIRTypes.i64)
        MLIROp.LLVMOp (loadNonAtomic nsecSSA nsecPtrSSA MLIRTypes.i64)
    ]

    // Convert to 100-nanosecond ticks: sec * 10_000_000 + nsec / 100
    let ticksPerSecSSA = freshSynthSSA z
    let secTicksSSA = freshSynthSSA z
    let nsecDivSSA = freshSynthSSA z
    let nsecTicksSSA = freshSynthSSA z
    let totalTicksSSA = freshSynthSSA z
    let convertOps = [
        MLIROp.ArithOp (constI ticksPerSecSSA 10000000L MLIRTypes.i64)
        MLIROp.ArithOp (mulI secTicksSSA secSSA ticksPerSecSSA MLIRTypes.i64)
        MLIROp.ArithOp (constI nsecDivSSA 100L MLIRTypes.i64)
        MLIROp.ArithOp (divSI nsecTicksSSA nsecSSA nsecDivSSA MLIRTypes.i64)
        MLIROp.ArithOp (addI totalTicksSSA secTicksSSA nsecTicksSSA MLIRTypes.i64)
    ]

    let allOps = allocOps @ syscallOps @ extractOps @ convertOps
    allOps, totalTicksSSA

/// Generate nanosleep syscall
let bindLinuxNanosleep (z: PSGZipper) (msSSA: SSA) (msType: MLIRType) : MLIROp list =
    // Extend ms to i64 if needed
    let msExt, extOps =
        match msType with
        | TInt I64 -> msSSA, []
        | TInt _ ->
            let ext = freshSynthSSA z
            ext, [MLIROp.ArithOp (extSI ext msSSA msType MLIRTypes.i64)]
        | _ -> msSSA, []

    // Convert ms to seconds and nanoseconds
    let thousandSSA = freshSynthSSA z
    let millionSSA = freshSynthSSA z
    let secondsSSA = freshSynthSSA z
    let remainderSSA = freshSynthSSA z
    let nanosecondsSSA = freshSynthSSA z
    let convertOps = [
        MLIROp.ArithOp (constI thousandSSA 1000L MLIRTypes.i64)
        MLIROp.ArithOp (constI millionSSA 1000000L MLIRTypes.i64)
        MLIROp.ArithOp (divSI secondsSSA msExt thousandSSA MLIRTypes.i64)
        MLIROp.ArithOp (remSI remainderSSA msExt thousandSSA MLIRTypes.i64)
        MLIROp.ArithOp (mulI nanosecondsSSA remainderSSA millionSSA MLIRTypes.i64)
    ]

    // Allocate timespec structs
    let oneSSA = freshSynthSSA z
    let reqTimespecSSA = freshSynthSSA z
    let remTimespecSSA = freshSynthSSA z
    let allocOps = [
        MLIROp.ArithOp (constI oneSSA 1L MLIRTypes.i64)
        MLIROp.LLVMOp (alloca reqTimespecSSA oneSSA timespecType None)
        MLIROp.LLVMOp (alloca remTimespecSSA oneSSA timespecType None)
    ]

    // Store seconds and nanoseconds
    let zeroSSA = freshSynthSSA z
    let secIdxSSA = freshSynthSSA z
    let nsecIdxSSA = freshSynthSSA z
    let secPtrSSA = freshSynthSSA z
    let nsecPtrSSA = freshSynthSSA z
    let storeOps = [
        MLIROp.ArithOp (constI zeroSSA 0L MLIRTypes.i64)
        MLIROp.ArithOp (constI secIdxSSA 0L MLIRTypes.i64)
        MLIROp.ArithOp (constI nsecIdxSSA 1L MLIRTypes.i64)
        MLIROp.LLVMOp (gep secPtrSSA reqTimespecSSA [(zeroSSA, MLIRTypes.i64); (secIdxSSA, MLIRTypes.i64)] timespecType)
        MLIROp.LLVMOp (gep nsecPtrSSA reqTimespecSSA [(zeroSSA, MLIRTypes.i64); (nsecIdxSSA, MLIRTypes.i64)] timespecType)
        MLIROp.LLVMOp (storeNonAtomic secondsSSA secPtrSSA MLIRTypes.i64)
        MLIROp.LLVMOp (storeNonAtomic nanosecondsSSA nsecPtrSSA MLIRTypes.i64)
    ]

    // Call nanosleep syscall
    let sysNumSSA = freshSynthSSA z
    let syscallResultSSA = freshSynthSSA z
    let syscallOps = [
        MLIROp.ArithOp (constI sysNumSSA SyscallData.linuxNanosleep MLIRTypes.i64)
        MLIROp.LLVMOp (inlineAsmWithSideEffects (Some syscallResultSSA) "syscall"
            "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
            [i64Arg sysNumSSA; ptrArg reqTimespecSSA; ptrArg remTimespecSSA] (Some MLIRTypes.i64))
    ]

    extOps @ convertOps @ allocOps @ storeOps @ syscallOps

// ===================================================================
// Platform Primitive Bindings
// ===================================================================

/// getCurrentTicks - get current time in .NET ticks format
let bindGetCurrentTicks (os: OSFamily) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    match os with
    | Linux ->
        let ticksOps, ticksSSA = bindLinuxClockGettime SyscallData.CLOCK_REALTIME z
        // Add Unix epoch offset to convert to .NET ticks (since 0001-01-01)
        let epochOffsetSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let epochOps = [
            MLIROp.ArithOp (constI epochOffsetSSA 621355968000000000L MLIRTypes.i64)
            MLIROp.ArithOp (addI resultSSA ticksSSA epochOffsetSSA MLIRTypes.i64)
        ]
        BoundOps (ticksOps @ epochOps, Some { SSA = resultSSA; Type = MLIRTypes.i64 })
    | _ ->
        NotSupported $"Time not supported on {os}"

/// getMonotonicTicks - get high-resolution monotonic ticks
let bindGetMonotonicTicks (os: OSFamily) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    match os with
    | Linux ->
        let ticksOps, ticksSSA = bindLinuxClockGettime SyscallData.CLOCK_MONOTONIC z
        BoundOps (ticksOps, Some { SSA = ticksSSA; Type = MLIRTypes.i64 })
    | _ ->
        NotSupported $"Monotonic time not supported on {os}"

/// getTickFrequency - get ticks per second (10,000,000 for 100-ns ticks)
let bindGetTickFrequency (_os: OSFamily) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let freqSSA = freshSynthSSA z
    let ops = [MLIROp.ArithOp (constI freqSSA 10000000L MLIRTypes.i64)]
    BoundOps (ops, Some { SSA = freqSSA; Type = MLIRTypes.i64 })

/// sleep - sleep for specified milliseconds
let bindSleep (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [ms] ->
        match os with
        | Linux ->
            let ops = bindLinuxNanosleep z ms.SSA ms.Type
            BoundOps (ops, None)
        | _ ->
            NotSupported $"Sleep not supported on {os}"
    | _ ->
        NotSupported "sleep requires milliseconds argument"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Linux x86_64
    PlatformDispatch.register Linux X86_64 "getCurrentTicks" (bindGetCurrentTicks Linux)
    PlatformDispatch.register Linux X86_64 "getMonotonicTicks" (bindGetMonotonicTicks Linux)
    PlatformDispatch.register Linux X86_64 "getTickFrequency" (bindGetTickFrequency Linux)
    PlatformDispatch.register Linux X86_64 "sleep" (bindSleep Linux)

    // Linux ARM64
    PlatformDispatch.register Linux ARM64 "getCurrentTicks" (bindGetCurrentTicks Linux)
    PlatformDispatch.register Linux ARM64 "getMonotonicTicks" (bindGetMonotonicTicks Linux)
    PlatformDispatch.register Linux ARM64 "getTickFrequency" (bindGetTickFrequency Linux)
    PlatformDispatch.register Linux ARM64 "sleep" (bindSleep Linux)

    // macOS (placeholder - returns NotSupported)
    PlatformDispatch.register MacOS X86_64 "getCurrentTicks" (bindGetCurrentTicks MacOS)
    PlatformDispatch.register MacOS X86_64 "sleep" (bindSleep MacOS)
    PlatformDispatch.register MacOS ARM64 "getCurrentTicks" (bindGetCurrentTicks MacOS)
    PlatformDispatch.register MacOS ARM64 "sleep" (bindSleep MacOS)
