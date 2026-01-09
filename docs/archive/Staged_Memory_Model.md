# Staged Memory Model: A Pragmatic Path to Deterministic Memory Management

## Document Purpose

This document articulates Firefly's deliberate, staged approach to memory management. The Fidelity framework envisions sophisticated memory semantics including actors, arenas, and sentinel-based lifetime management. These features go substantially beyond what OCaml or Rust contemplate. However, we cannot wait for the complete actor/arena/sentinel model before delivering a working memory system.

This document defines how core memory mapping should progress through defined stages, ensuring each stage delivers immediate value while preserving architectural openness for future capabilities. The goal is a layered approach where:

1. **Stage 1**: Core primitives work now, supporting the January demo scenario (STM32L5 unikernel with CMSIS HAL)
2. **Stage 2**: Memory regions and access kinds provide compile-time safety for embedded development
3. **Stage 3**: Cache-aware and CPU-aware compilation optimizes for specific targets
4. **Stage 4**: Actor/arena integration delivers the full Fidelity memory model
5. **Future**: GPU, NPU, neuromorphic, CGRA processor descriptors extend beyond conventional architectures

## The Three-Layer Memory Surface

Memory concerns enter the Fidelity compilation pipeline through three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Memory Concern Entry Points                          │
│                                                                         │
│  Layer 1: fsnative Primitives                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ string, nativeptr, Span, array, option (all with native semantics) │ │
│  │ Memory region measures: peripheral, sram, flash, stack            │ │
│  │ Access kind measures: readOnly, writeOnly, readWrite              │ │
│  │ Lifetime expressions: Compile-time verified scopes                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  Layer 2: Farscape Libraries (External Bindings)                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ CMSIS HAL register descriptors                                    │ │
│  │ Peripheral base addresses and memory-mapped regions               │ │
│  │ Volatile semantics for hardware access                            │ │
│  │ C/C++ library bindings with memory annotations                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  Layer 3: PSG Pass-through to Alex/MLIR                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Memory region annotations preserved in PSG nodes                  │ │
│  │ Access kind constraints flow to MLIR generation                   │ │
│  │ Alex Bindings generate platform-specific memory operations        │ │
│  │ MLIR dialects express memory semantics (memref, llvm.volatile)    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: fsnative Primitives

FNCS provides the type foundation for memory-safe native compilation. The key insight is that memory semantics begin at the type level, not as annotations applied later.

**Native Type Resolution:**

When FNCS encounters a string literal, it provides native semantics: UTF-8 fat pointer representation (pointer + length) instead of BCL's System.String. Users write `string`, FNCS provides native semantics transparently:

```fsharp
// Source code
let greeting = "Hello, World!"

// FNCS provides native semantics for string
// Type: string (with native semantics - NOT System.String)

// PSG captures
// Node: Const:String
// Type: string (UTF-8 fat pointer)
// Layout: { ptr: nativeptr<byte>, len: int }

// MLIR generated
// %str = llvm.mlir.addressof @__str_0 : !llvm.ptr
// %len = llvm.mlir.constant(13 : i64) : i64
// -- fat pointer pair (ptr, len) for safe string handling
```

**Memory Region Measures:**

FNCS absorbs UMX patterns to provide phantom type parameters for memory regions:

```fsharp
[<Measure>] type peripheral    // Volatile, no cache, memory-mapped I/O
[<Measure>] type sram          // Normal RAM, cacheable
[<Measure>] type flash         // Read-only storage
[<Measure>] type stack         // Thread-local, auto-managed
[<Measure>] type arena         // Region-allocated, batch cleanup

// Pointer types carry region information
type NativePtr<'T, [<Measure>] 'region, [<Measure>] 'access>
```

These measures exist only at compile time. They impose zero runtime overhead while enabling the compiler to verify memory access safety.

**Access Kind Measures:**

Complementing regions, access kinds express read/write permissions:

```fsharp
[<Measure>] type readOnly      // Can read, cannot write (CMSIS __I)
[<Measure>] type writeOnly     // Can write, cannot read (CMSIS __O)
[<Measure>] type readWrite     // Both operations allowed (CMSIS __IO)

// Combined in pointer types
let gpio_idr : NativePtr<uint32, peripheral, readOnly>
let gpio_odr : NativePtr<uint32, peripheral, readWrite>
```

Attempting to write through a `readOnly` pointer generates a compile-time error (FS8002: Cannot write read-only pointer).

### Layer 2: Farscape Libraries

External libraries, particularly hardware abstraction layers, bring their own memory semantics. Farscape generates F# bindings that preserve these semantics through the fsnative type system.

**CMSIS HAL Pattern:**

The January demo targets STM32L5 with CMSIS HAL. Farscape parses C headers and generates type-safe F# bindings:

```fsharp
// Generated by Farscape from stm32l5xx_hal_gpio.h
[<PeripheralDescriptor("GPIOA", 0x48000000UL)>]
type GPIO_TypeDef = {
    [<Register("MODER", 0x00u, "rw")>]
    MODER: NativePtr<uint32, peripheral, readWrite>

    [<Register("IDR", 0x10u, "r")>]
    IDR: NativePtr<uint32, peripheral, readOnly>

    [<Register("ODR", 0x14u, "rw")>]
    ODR: NativePtr<uint32, peripheral, readWrite>

    [<Register("BSRR", 0x18u, "w")>]
    BSRR: NativePtr<uint32, peripheral, writeOnly>
}

// Macro constants become F# values
let GPIO_PIN_5 = 0x0020us
let GPIO_MODE_OUTPUT_PP = 0x01u

// HAL functions use Platform.Bindings pattern (BCL-free)
// Farscape generates these; Alex provides MLIR emission
module Platform.Bindings.HAL.GPIO =
    /// Initialize GPIO peripheral
    let init (gpio: NativePtr<GPIO_TypeDef, peripheral, readWrite>)
             (initStruct: NativePtr<GPIO_InitTypeDef, stack, readOnly>) : unit =
        ()  // Alex emits register writes or HAL call
```

**Memory Semantics Preservation:**

Farscape must preserve memory semantics from C headers:
- `volatile` qualifiers map to `peripheral` region
- `const` qualifiers contribute to `readOnly` access
- Register attributes carry offset and access information

These annotations flow through FNCS type checking to Firefly's PSG.

### Layer 3: PSG Pass-through to Alex/MLIR

The PSG captures memory annotations and Alex generates appropriate MLIR:

```
PSG Node: App:FunctionCall [halGpioInit]
├── Arg1: NativePtr<GPIO_TypeDef, peripheral, readWrite>
├── Arg2: NativePtr<GPIO_InitTypeDef, stack, readOnly>
└── PlatformBinding: { module: CMSIS, entry: HAL_GPIO_Init }

Alex generates (ARM bare-metal):
// Direct memory access, no function call
// %gpio_base = llvm.mlir.constant(0x48000000 : i64) : i64
// %moder_offset = llvm.mlir.constant(0x00 : i32) : i32
// ... register manipulation code ...
```

For peripheral access, Alex may inline the HAL function entirely, replacing it with direct register manipulation. The `peripheral` region ensures volatile semantics.

## Stage 1: Immediate Working Model

The first stage focuses on getting a working memory model that supports:

1. Stack allocation as the default
2. Fat pointers for strings and arrays
3. Platform bindings for I/O operations
4. Basic peripheral access for embedded targets

### Stack-Only Memory Model

For the January demo, we use a conservative stack-only model:

```fsharp
// Memory model configuration in .fidproj
[compilation]
memory_model = "stack_only"
target = "native"

[build]
output_kind = "freestanding"  // No libc
```

In this model:
- All allocations happen on the stack
- No heap, no GC
- String literals are embedded in the binary's data section
- Arrays are stack-allocated with known bounds

### Fat Pointer Representation

All pointer types use fat pointer representation for safety:

```
┌─────────────────────────────────────────────────────────────┐
│  string (Fat Pointer - Native Semantics)                    │
│  ┌─────────────────────┬─────────────────────────────────┐ │
│  │ ptr: nativeptr<byte>│ len: int                        │ │
│  │ (8 bytes on 64-bit) │ (8 bytes)                       │ │
│  └─────────────────────┴─────────────────────────────────┘ │
│                                                             │
│  Guarantees:                                                │
│  - No null-terminated string vulnerabilities                │
│  - Bounds checking can be elided when statically known      │
│  - Slicing without copying (ptr offset + length reduction) │
└─────────────────────────────────────────────────────────────┘
```

### Platform Bindings

Platform bindings use the BCL-free module convention:

```fsharp
// Alloy/Platform.fs
module Platform.Bindings =
    let writeBytes fd buffer count : int = Unchecked.defaultof<int>
    let readBytes fd buffer maxCount : int = Unchecked.defaultof<int>
    let sleep milliseconds : unit = ()
```

Alex recognizes `Platform.Bindings.*` calls and generates platform-specific MLIR:

```fsharp
// For Linux x86_64:
// writeBytes generates: syscall SYS_write (1) with fd, buffer, count

// For ARM bare-metal:
// writeBytes generates: UART register manipulation
```

## Stage 2: Memory Regions and Access Kinds

The second stage adds compile-time verification of memory access safety.

### Region Hierarchy

Memory regions form a hierarchy based on volatility and cacheability:

```
                    Memory Region Hierarchy

        volatile=no                   volatile=yes
        cacheable=yes                 cacheable=no
              │                              │
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │      sram       │            │   peripheral    │
    │  (normal RAM)   │            │ (memory-mapped) │
    └─────────────────┘            └─────────────────┘
              │
              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │      stack      │            │      flash      │
    │ (thread-local)  │            │  (read-only)    │
    └─────────────────┘            └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │      arena      │
    │ (batch cleanup) │
    └─────────────────┘
```

### Constraint Checking

FNCS enforces region constraints at type checking:

```fsharp
// Valid: reading from peripheral
let value = NativePtr.read gpio.IDR  // IDR is readOnly

// Error FS8001: Cannot read write-only pointer
let invalid = NativePtr.read gpio.BSRR  // BSRR is writeOnly

// Error FS8002: Cannot write read-only pointer
NativePtr.write gpio.IDR 0u  // IDR is readOnly

// Valid: writing to read-write
NativePtr.write gpio.ODR value
```

### Region Assignment

The compiler assigns regions based on:

1. **Literal values**: Embedded in flash (read-only data section)
2. **Let bindings**: Stack by default
3. **Mutable bindings**: Stack, with lifetime tracking
4. **Farscape descriptors**: Peripheral region from annotations
5. **Explicit annotation**: Developer can specify region

```fsharp
// Implicit region assignment
let x = 42                  // stack: int
let s = "Hello"             // flash: string (data section, UTF-8)
let mutable counter = 0     // stack: int (mutable)

// Explicit region annotation (future)
let buffer : array<byte, sram, readWrite> = ...
```

## Stage 3: Cache-Aware and CPU-Aware Compilation

The third stage introduces target-specific memory optimizations.

### CPU Cache Hierarchy Awareness

Alex maintains knowledge of target cache hierarchies:

```fsharp
type CacheHierarchy = {
    L1Data: CacheLevel
    L1Instruction: CacheLevel
    L2: CacheLevel option
    L3: CacheLevel option
}

type CacheLevel = {
    Size: int
    LineSize: int
    Associativity: int
    Latency: int  // cycles
}

// Example: ARM Cortex-M33 (STM32L5)
let cortexM33Cache = {
    L1Data = { Size = 16384; LineSize = 32; Associativity = 4; Latency = 1 }
    L1Instruction = { Size = 16384; LineSize = 32; Associativity = 4; Latency = 1 }
    L2 = None
    L3 = None
}
```

### Data Layout Transformation

The compiler can transform data layouts for cache efficiency:

```fsharp
// Array of Structures (AoS) - original
type Particle = { X: float32; Y: float32; Z: float32; Mass: float32 }
let particles: Particle array = ...

// Structure of Arrays (SoA) - cache-optimized
// When processing all X coordinates, they're contiguous in memory
type ParticlesSoA = {
    X: float32 array
    Y: float32 array
    Z: float32 array
    Mass: float32 array
}
```

Alex can perform this transformation automatically when it detects patterns that would benefit.

### Memory Prefetching

For predictable access patterns, Alex can insert prefetch instructions:

```fsharp
// Loop over array elements
for i in 0 .. array.Length - 1 do
    process array.[i]

// Alex might generate:
// prefetch array[i + stride] at top of loop
// to hide memory latency
```

### NUMA Awareness

On multi-socket systems, memory placement matters:

```fsharp
// BAREWire pattern for NUMA-aware allocation
let allocateNuma<'T> (size: int) (node: int) =
    // Alex generates numa_alloc_onnode or equivalent
    ...

// Actor placement can consider NUMA topology
// Actors accessing shared data placed on same NUMA node
```

## Stage 4: Actor/Arena Integration

The fourth stage introduces the full Fidelity memory model with actors and arenas.

### Arena Memory Model

Arenas provide region-based memory allocation:

```fsharp
// Arena creation
use arena = Arena.create(initialSize = 4096)

// All allocations within scope use arena
let data = arena.alloc<MyType>()
let buffer = arena.allocArray<byte>(1024)

// When arena goes out of scope, all memory released at once
// No individual deallocation, no fragmentation
```

This maps to the HyperStack pattern from F*: regions form a stack, allocations happen in the current region, and regions are popped as scopes exit.

### Actor-Arena Binding

The key innovation is binding arena lifetimes to actor lifetimes:

```fsharp
// Each actor owns an arena
type Actor<'Msg, 'State> = {
    State: 'State
    Arena: Arena
    Mailbox: Mailbox<'Msg>
}

// Messages to actor are allocated in actor's arena
// When actor terminates, entire arena is released

// No GC pauses
// No reference counting
// Deterministic cleanup at actor termination
```

### Sentinel-Based Lifetime Management

Sentinels mark lifetime boundaries within arenas:

```fsharp
// Create sentinel in current arena
let sentinel = arena.mark()

// Allocations after mark
let temp1 = arena.alloc<TempData>()
let temp2 = arena.alloc<TempData>()

// Release everything after sentinel
arena.release(sentinel)
// temp1 and temp2 memory reclaimed, but earlier allocations preserved
```

This enables sub-arena lifetimes without heap fragmentation.

### Inter-Actor Communication

BAREWire enables zero-copy message passing between actors:

```fsharp
// Message sent between actors
type TradeMessage = {
    Symbol: string  // native semantics: UTF-8 fat pointer
    Quantity: int
    Price: float
}

// Sender serializes into shared buffer
// Receiver deserializes (zero-copy view)
// No heap allocation for message passing
```

## Future: Heterogeneous Accelerators

The staged approach preserves architectural openness for accelerators.

### GPU Memory Regions

GPU architectures introduce additional memory spaces:

```fsharp
[<Measure>] type gpu_global    // GPU global memory
[<Measure>] type gpu_shared    // Per-block shared memory
[<Measure>] type gpu_constant  // Constant memory (cached)
[<Measure>] type gpu_texture   // Texture memory (spatial caching)
```

Alex generates appropriate MLIR dialect operations:

```mlir
// GPU global memory allocation
%buffer = gpu.alloc() : memref<1024xf32, #gpu.address_space<global>>

// Data transfer
gpu.memcpy %host_data, %buffer : memref<1024xf32>, memref<1024xf32, #gpu.address_space<global>>
```

### NPU/CGRA Descriptors

Neural processing units and coarse-grained reconfigurable arrays have different memory models:

```fsharp
// NPU weight buffer (on-chip SRAM)
[<Measure>] type npu_weights

// CGRA scratchpad
[<Measure>] type cgra_local

// Dataflow buffer (no address, just connection)
[<Measure>] type dataflow
```

These will be defined as new accelerator targets are added to Alex.

### Processing-in-Memory

Future architectures may perform computation near memory:

```fsharp
// HBM with compute capability
[<Measure>] type pim_capable

// Operations on PIM memory execute in-place
// No data movement to CPU/GPU
let result = pimReduce buffer (+)
```

## Relationship to OCaml and Rust

### Where We Learn from OCaml

OCaml's influence on fsnative:
- Native types (not wrapped BCL types)
- Module system patterns
- Efficient unboxed representation
- Garbage collection avoidance through careful design

But OCaml's memory model is simpler:
- Single heap with GC
- No explicit regions
- No ownership tracking

### Where We Learn from Rust

Rust's influence on Fidelity:
- Ownership prevents use-after-free
- Borrowing enables safe references
- Lifetime annotations explicit

But we deliberately avoid Rust's complexity:
- No borrow checker interference at design time
- Regions are more ergonomic than explicit lifetimes
- Actors provide natural ownership boundaries

### Where Fidelity Goes Beyond

Fidelity's actor/arena/sentinel model is substantially beyond either:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Memory Management Comparison                                           │
│                                                                         │
│  OCaml                    Rust                     Fidelity             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ Single heap     │     │ Stack + Heap    │     │ Regions + Arenas│   │
│  │ with GC         │     │ with ownership  │     │ per Actor       │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│         │                        │                       │              │
│         ▼                        ▼                       ▼              │
│  GC pauses possible      No GC, but borrow        No GC, natural      │
│                          checker complexity       lifetime boundaries  │
│                                                                         │
│  Developer doesn't       Developer must prove     Developer designs    │
│  think about memory      lifetimes to compiler    actors, memory       │
│                                                   follows naturally    │
└─────────────────────────────────────────────────────────────────────────┘
```

The insight: **actors provide natural ownership boundaries**. An actor owns its state, its arena owns its allocations, and actor termination provides deterministic cleanup. This is both simpler than Rust's borrow checker and more powerful than OCaml's GC.

## Implementation Roadmap

### Stage 1 Milestones (January Demo)

| Milestone | Status | Description |
|-----------|--------|-------------|
| Stack-only allocation | In Progress | All values on stack, no heap |
| Fat pointer strings | In Progress | string with (ptr, len) native semantics |
| Platform bindings | Working | Console I/O via syscalls |
| Peripheral access | Needed | Farscape + Alex ARM bindings |
| STM32L5 blink demo | Goal | GPIO toggle via CMSIS HAL |

### Stage 2 Milestones (Q1 2025)

| Milestone | Description |
|-----------|-------------|
| Region measures | phantom types for memory regions |
| Access kinds | readOnly/writeOnly/readWrite |
| Constraint errors | FS8001-FS8003 diagnostics |
| Farscape integration | Auto-generated peripheral descriptors |

### Stage 3 Milestones (Q2 2025)

| Milestone | Description |
|-----------|-------------|
| Cache hierarchy model | Alex knows target cache structure |
| Data layout transforms | AoS/SoA conversion where beneficial |
| Prefetch insertion | Automatic prefetch for predictable access |
| NUMA awareness | Multi-socket memory placement |

### Stage 4 Milestones (Q3-Q4 2025)

| Milestone | Description |
|-----------|-------------|
| Arena allocation | Region-based memory with batch cleanup |
| Actor memory binding | Arena per actor pattern |
| Sentinel lifetimes | Sub-arena lifetime management |
| BAREWire zero-copy | Inter-actor message passing |

## Proof Integration

The staged memory model aligns with F* integration for formal verification.

### Design-Time Correspondence

F* specifications serve as design-time verification:

```fstar
// F* specification of safe array access
val get: #a:Type -> #len:nat -> v:vec a len -> i:nat{i < len} -> a
```

This pattern corresponds to fsnative:

```fsharp
// fsnative with compile-time bounds
let inline get (v: array<'a>) (i: int) =
    // Compiler proves i < v.Length from context
    Array.get v i
```

### Proof Hyperedges in PSG

Memory safety proofs can be represented as PSG hyperedges:

```
PSG with Proof Hyperedges:
├── Node: ArrayAccess [arr, idx]
│   └── Proof: BoundsCheck { idx < arr.Length }
├── Node: PointerWrite [ptr, value]
│   └── Proof: AccessKind { ptr.access = readWrite }
└── Node: RegionEscape [ptr]
    └── Proof: LifetimeContained { ptr.lifetime <= scope.lifetime }
```

When proofs are present, Alex can elide runtime checks. When proofs are absent, runtime checks are inserted.

## Conclusion

The staged memory model provides a pragmatic path from working primitives to sophisticated actor/arena integration. Each stage delivers immediate value:

1. **Stage 1**: Working code for the January demo
2. **Stage 2**: Compile-time memory safety for embedded development
3. **Stage 3**: Target-specific optimizations for performance
4. **Stage 4**: Full Fidelity memory model with actors and arenas

The three-layer architecture (fsnative primitives, Farscape libraries, PSG pass-through) ensures that memory concerns from any source integrate cleanly into the compilation pipeline.

By starting with grounded primitives and building incrementally, we avoid the trap of waiting for a perfect model that may never arrive. Each stage is valuable on its own, and each stage provides the foundation for the next.

The future of Fidelity memory management lies in the actor/arena/sentinel model, but the present is stack allocation and fat pointers. This document provides the roadmap from here to there.
