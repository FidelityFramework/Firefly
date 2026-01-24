# Sophisticated Platform Binding Model

## Overview

The Fidelity platform binding model has evolved from a simple "one-page binding" to a comprehensive capability system supporting memory management by choice across the full hardware spectrum.

## Repository Structure

```
~/repos/Fidelity.Platform/
├── README.md
├── Linux_x86_64/               # Linux on x86-64
│   ├── Types.fs                # NTU type definitions with quotations
│   ├── Platform.fs             # PlatformDescriptor quotation
│   ├── Capabilities.fs         # Platform predicates
│   ├── MemoryRegions.fs        # Stack, heap, arena definitions
│   ├── CacheCharacteristics.fs # Cache info for optimization
│   ├── Syscalls.fs             # Syscall conventions
│   └── Fidelity.Platform.Linux_x86_64.fsproj
├── Linux_ARM64/                # Linux on ARM64 (future)
├── Windows_x86_64/             # Windows on x86-64 (future)
├── MacOS_ARM64/                # macOS on Apple Silicon (future)
└── BareMetal_ARM32/            # Bare-metal ARM Cortex-M (future)
```

## Platform Descriptor

The core platform quotation:

```fsharp
// Platform.fs
let platform: Expr<PlatformDescriptor> = <@
    { Architecture = X86_64
      OperatingSystem = Linux
      WordSize = 64
      Endianness = LittleEndian
      TypeLayouts = Map.ofList [
          "int", { Size = 8; Alignment = 8 }
          "int32", { Size = 4; Alignment = 4 }
          "nativeint", { Size = 8; Alignment = 8 }
          "nativeptr", { Size = 8; Alignment = 8 }
      ]
      SyscallConvention = sysV_AMD64_syscall }
@>
```

## Platform Predicates (F*-Inspired)

Abstract propositions for conditional compilation:

```fsharp
// Capabilities.fs
module Capabilities =
    /// Platform supports 32-bit word operations
    let fits_u32: Expr<bool> = <@ true @>
    
    /// Platform supports 64-bit word operations
    let fits_u64: Expr<bool> = <@ true @>
    
    /// 64-bit implies 32-bit
    let fits_u64_implies_u32: Expr<unit> = <@ () @>
    
    /// Platform has AVX-512 vector support
    let has_avx512: Expr<bool> = <@ false @>  // CPU-dependent
    
    /// Platform has AVX2 vector support
    let has_avx2: Expr<bool> = <@ true @>
    
    /// Platform has NEON vector support (ARM)
    let has_neon: Expr<bool> = <@ false @>
    
    /// Platform has 64-bit atomic operations
    let has_atomics_64: Expr<bool> = <@ true @>
    
    /// Maximum supported vector width in bits
    let vector_width_max: Expr<int> = <@ 256 @>  // AVX2
```

### Using Predicates for Conditional Compilation

```fsharp
// In application code using Fidelity.Platform
let vectorAdd (a: array<float>) (b: array<float>) =
    if Platform.has_avx512 then
        vectorAdd_avx512 a b
    elif Platform.has_avx2 then
        vectorAdd_avx2 a b
    else
        vectorAdd_scalar a b
```

FNCS sees the predicates as abstract. Alex witnesses them to eliminate dead branches at compile time.

## Memory Regions

For memory management by choice:

```fsharp
// MemoryRegions.fs
module MemoryRegions =
    /// Stack characteristics
    let stackRegion: Expr<MemoryRegion> = <@
        { Name = "Stack"
          MaxSize = 8388608      // 8 MB typical
          Alignment = 16
          GrowthDirection = Down
          ThreadLocal = true }
    @>
    
    /// Heap region
    let heapRegion: Expr<MemoryRegion> = <@
        { Name = "Heap"
          Strategy = ArenaOrMalloc
          Alignment = 16
          ThreadLocal = false }
    @>
    
    /// Arena for scratch allocations
    let arenaRegion: Expr<MemoryRegion> = <@
        { Name = "Arena"
          Strategy = BumpAllocator
          DefaultSize = 1048576  // 1 MB
          Alignment = 16 }
    @>
```

## Cache Characteristics

For optimization hints:

```fsharp
// CacheCharacteristics.fs
module CacheInfo =
    /// L1 data cache line size
    let l1_line_size: Expr<int> = <@ 64 @>
    
    /// L1 data cache size
    let l1_size: Expr<int> = <@ 32768 @>  // 32 KB
    
    /// L2 cache size
    let l2_size: Expr<int> = <@ 262144 @>  // 256 KB
    
    /// L3 cache size (shared)
    let l3_size: Expr<int> = <@ 8388608 @>  // 8 MB
    
    /// Prefetch distance hint
    let prefetch_distance: Expr<int> = <@ 256 @>
```

## Syscall Conventions

```fsharp
// Syscalls.fs
module Syscalls =
    /// Linux x86-64 syscall convention
    let convention: Expr<SyscallConvention> = <@
        { CallingConvention = SysV_AMD64
          ArgRegisters = [| RDI; RSI; RDX; R10; R8; R9 |]
          ReturnRegister = RAX
          ErrorReturn = NegativeErrno
          SyscallInstruction = Syscall }
    @>
    
    /// Syscall numbers
    let sys_read: Expr<int> = <@ 0 @>
    let sys_write: Expr<int> = <@ 1 @>
    let sys_open: Expr<int> = <@ 2 @>
    let sys_close: Expr<int> = <@ 3 @>
    let sys_mmap: Expr<int> = <@ 9 @>
    let sys_munmap: Expr<int> = <@ 11 @>
    let sys_nanosleep: Expr<int> = <@ 35 @>
    let sys_exit: Expr<int> = <@ 60 @>
    let sys_exit_group: Expr<int> = <@ 231 @>
```

## Integration with fidproj

Projects reference platform bindings:

```toml
# HelloWorld.fidproj
[package]
name = "HelloWorld"

[dependencies]
platform = { path = "/home/hhh/repos/Fidelity.Platform/Linux_x86_64" }

[build]
sources = ["HelloWorld.fs"]
output = "helloworld"
output_kind = "freestanding"
```

## Pipeline Integration

```
┌─────────────────────────────────────────────────────────┐
│  Firefly CLI                                             │
│  1. Parse fidproj with Fidelity.Toml                    │
│  2. Load Fidelity.Platform library                      │
│  3. Extract quotations (platform, capabilities, etc.)   │
│  4. Pass to FNCS as PlatformContext                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FNCS                                                    │
│  1. Receive PlatformContext as parameter                │
│  2. Attach platform metadata to PSG nodes               │
│  3. Validate types (NTUint identity, not width)         │
│  4. Return SemanticGraph with quotations attached       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex                                                    │
│  1. Read platform quotations from PSG                   │
│  2. Witness NTU types → concrete MLIR types             │
│  3. Eliminate dead branches via predicates              │
│  4. Generate platform-optimized code                    │
└─────────────────────────────────────────────────────────┘
```

## NTU Type Resolution

Platform bindings define how NTU types resolve:

| NTU Type | Linux_x86_64 | Linux_ARM64 | Linux_ARM32 | BareMetal_ARM32 |
|----------|--------------|-------------|-------------|-----------------|
| NTUint | i64 | i64 | i32 | i32 |
| NTUuint | i64 | i64 | i32 | i32 |
| NTUptr<'T> | ptr (8B) | ptr (8B) | ptr (4B) | ptr (4B) |
| NTUsize | u64 | u64 | u32 | u32 |
| NTUdiff | i64 | i64 | i32 | i32 |

## Adding New Platforms

1. **Create platform directory:**
   ```bash
   mkdir ~/repos/Fidelity.Platform/Windows_x86_64
   ```

2. **Create Types.fs** with shared type definitions

3. **Create Platform.fs** with `Expr<PlatformDescriptor>` quotation

4. **Create Capabilities.fs** with platform predicates

5. **Create MemoryRegions.fs** for memory management

6. **Create Syscalls.fs** (or equivalent for bare-metal)

7. **Create .fsproj** and build

8. **Reference in fidproj:**
   ```toml
   [dependencies]
   platform = { path = "/home/hhh/repos/Fidelity.Platform/Windows_x86_64" }
   ```

## Related Documentation

- `NTU_Architecture.md` - NTU type system design
- `Architecture_Canonical.md` - Overall Fidelity architecture
- `FNCS_Architecture.md` - FNCS native type checking
