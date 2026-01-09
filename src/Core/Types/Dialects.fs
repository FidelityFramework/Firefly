module Core.Types.Dialects

type MLIRDialect =
    | Standard | LLVM | Func | Arith | SCF
    | MemRef | Index | Affine | Builtin

/// Output kind for compilation targets
/// This determines linker flags and runtime dependencies
type OutputKind =
    | Console       // Linked with libc, can use stdio
    | Freestanding  // No libc, syscalls only
    | Embedded      // No OS, bare metal
    | Library       // Shared library (.so/.dll)