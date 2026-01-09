/// Platform Configuration Types - Runtime mode and binding resolution
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Platform binding resolution is a NANOPASS, not a witness concern.
/// These types define the coeffects computed by PlatformBindingResolution
/// that witnesses lookup during traversal.
///
/// PSG should be complete before witnessing. Platform decisions
/// (freestanding vs console, syscall vs libc) are resolved here.
module Alex.Preprocessing.PlatformConfig

open Alex.Bindings.PlatformTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// RUNTIME MODE
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime environment mode from fidproj output_kind
type RuntimeMode =
    /// No libc, direct syscalls, we generate _start wrapper
    | Freestanding
    /// Links with libc, libc provides _start, uses standard C library calls
    | Console

// ═══════════════════════════════════════════════════════════════════════════
// RESOLVED BINDINGS
// ═══════════════════════════════════════════════════════════════════════════

/// How a platform operation is resolved to concrete implementation
type ResolvedBinding =
    /// Direct syscall via inline assembly
    /// syscallNum: the syscall number for the target OS
    /// constraints: inline asm constraints string
    | Syscall of syscallNum: int64 * constraints: string
    /// Call to libc function (generates llvm.call)
    | LibcCall of funcName: string
    /// Inline assembly (for special operations)
    | InlineAsm of asm: string * constraints: string

/// Resolution result for a single intrinsic node
type BindingResolution = {
    /// PSG node ID this resolution applies to
    NodeId: int
    /// Entry point name (e.g., "Sys.write", "Console.writeln")
    EntryPoint: string
    /// Runtime mode at time of resolution
    Mode: RuntimeMode
    /// How the binding is resolved
    Resolved: ResolvedBinding
}

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM RESOLUTION RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Complete platform binding resolution result (coeffect for witnesses)
type PlatformResolutionResult = {
    /// Runtime mode (freestanding or console)
    RuntimeMode: RuntimeMode
    /// Target operating system
    TargetOS: OSFamily
    /// Target architecture
    TargetArch: Architecture
    /// All resolved bindings keyed by PSG node ID
    Bindings: Map<int, BindingResolution>
    /// Whether _start wrapper is needed (freestanding mode)
    NeedsStartWrapper: bool
    /// Pre-built _start wrapper MLIR ops (if needed)
    StartWrapperOps: MLIROp list option
}

// ═══════════════════════════════════════════════════════════════════════════
// SYSCALL DATA (Linux x86_64)
// ═══════════════════════════════════════════════════════════════════════════

/// Syscall numbers per OS (Linux x86_64 for now)
module SyscallNumbers =

    /// Get write syscall number for OS
    let getWriteSyscall (os: OSFamily) : int64 =
        match os with
        | Linux -> 1L
        | MacOS -> 0x2000004L  // Darwin syscalls have high bits set
        | FreeBSD -> 4L
        | _ -> failwithf "write syscall not implemented for %A" os

    /// Get read syscall number for OS
    let getReadSyscall (os: OSFamily) : int64 =
        match os with
        | Linux -> 0L
        | MacOS -> 0x2000003L
        | FreeBSD -> 3L
        | _ -> failwithf "read syscall not implemented for %A" os

    /// Get exit syscall number for OS
    let getExitSyscall (os: OSFamily) : int64 =
        match os with
        | Linux -> 60L
        | MacOS -> 0x2000001L
        | FreeBSD -> 1L
        | _ -> failwithf "exit syscall not implemented for %A" os

    /// Get nanosleep syscall number for OS
    let getNanosleepSyscall (os: OSFamily) : int64 =
        match os with
        | Linux -> 35L
        | MacOS -> 0x200005CL  // __NR_nanosleep on Darwin
        | FreeBSD -> 240L
        | _ -> failwithf "nanosleep syscall not implemented for %A" os

    /// Get clock_gettime syscall number for OS
    let getClockGetTimeSyscall (os: OSFamily) : int64 =
        match os with
        | Linux -> 228L
        | MacOS -> 0x20000D3L
        | FreeBSD -> 232L
        | _ -> failwithf "clock_gettime syscall not implemented for %A" os

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert fidproj output_kind to RuntimeMode
let parseRuntimeMode (isFreestanding: bool) : RuntimeMode =
    if isFreestanding then Freestanding else Console

/// Lookup a resolved binding by node ID
let lookupBinding (nodeId: int) (result: PlatformResolutionResult) : BindingResolution option =
    Map.tryFind nodeId result.Bindings

/// Check if a node has a platform binding resolution
let hasBinding (nodeId: int) (result: PlatformResolutionResult) : bool =
    Map.containsKey nodeId result.Bindings

/// Create an empty resolution result
let empty (mode: RuntimeMode) (os: OSFamily) (arch: Architecture) : PlatformResolutionResult =
    {
        RuntimeMode = mode
        TargetOS = os
        TargetArch = arch
        Bindings = Map.empty
        NeedsStartWrapper = (mode = Freestanding)
        StartWrapperOps = None
    }
