# Semantic Primitives and Loop Abstraction in the Firefly Compiler

## Overview

This document captures a critical architectural insight discovered during the development of the Fidelity HelloWorld sample progression: the distinction between **semantic primitives** that express developer intent and **implementation details** that should be handled by the compiler infrastructure. This distinction is fundamental to Firefly's promise of native compilation without a managed runtime while maintaining the elegance of F# idioms.

The central thesis is that **loops are not a monolithic construct** in compiler design. They represent at least two fundamentally different categories of computation that require different treatment in the compilation pipeline:

1. **Bounded Memory Operations** - Deterministic, pure operations with statically-known iteration counts (copying bytes, zeroing memory, comparing buffers)
2. **Control Flow Loops** - User-experience loops with resource dependencies, temporal concerns, or genuinely unbounded iteration

The Firefly compiler, through its Alloy library and Alex transformation layer, must recognize this distinction and handle each category appropriately for target-aware optimization.

## The Problem: Hidden Complexity in "Simple" Code

Consider the following F# code from the `03_HelloWorldHalfCurried` sample:

```fsharp
// Copy "Hello, " to output buffer
let helloBytes = "Hello, "B
for i = 0 to bytesLen helloBytes - 1 do
    NativePtr.set outputBuffer pos helloBytes.[i]
    pos <- pos + 1

// Copy name to output buffer
for i = 0 to name.Length - 1 do
    NativePtr.set outputBuffer pos (NativePtr.get name.Pointer i)
    pos <- pos + 1
```

This code appears straightforward, but it exposes several layers of complexity:

1. **It's a bounded memory operation** - The iteration count is known at loop entry
2. **It's deterministically pure** - No I/O, no resource access, just memory manipulation
3. **It's target-dependent** - Optimal implementation varies dramatically by hardware:
   - On **microcontrollers** (STM32, RISC-V MCUs): Stack-based tight loop, size-optimized
   - On **application CPUs** (x86_64, Apple M-series, Qualcomm ARM, Graviton, RISC-V datacenter): SIMD vectorization, arena allocation viable
   - On **GPU**: Parallel lane execution
   - With sufficient data: `memcpy` intrinsics or DMA operations

The fundamental issue is that **this code expresses implementation, not intent**. The developer's intent is "concatenate these strings into the output buffer." The byte-by-byte loop is merely one possible implementation strategy.

## The Solution: Semantic Primitives in Alloy

### Design Principle

> **Alloy primitives should express WHAT, not HOW. Alex chooses HOW based on target.**

This principle aligns with the broader Fidelity framework philosophy articulated in the [Context-Aware Compilation](../../../SpeakEZ/hugo/content/blog/Context%20Aware%20Compilation.md) blog post, where coeffect analysis guides the compiler to select optimal execution strategies based on computation characteristics.

### Proposed Primitives

The Alloy library should provide semantic primitives for common memory operations:

```fsharp
module Alloy.Memory

/// Copy bytes from source to destination
/// Semantic: "Transfer N bytes from src to dest"
/// Implementation: Target-dependent (loop, memcpy, SIMD, DMA)
val copy : src:nativeptr<byte> -> dest:nativeptr<byte> -> length:int -> unit

/// Zero a memory region
/// Semantic: "Set N bytes to zero"
/// Implementation: Target-dependent (loop, memset, SIMD)
val zero : dest:nativeptr<byte> -> length:int -> unit

/// Compare two memory regions
/// Semantic: "Are these N bytes identical?"
/// Implementation: Target-dependent (loop, memcmp, SIMD)
val compare : a:nativeptr<byte> -> b:nativeptr<byte> -> length:int -> bool
```

For string operations specifically:

```fsharp
module Alloy.Text.String

/// Concatenate multiple strings into a destination buffer
/// Semantic: "Combine these strings sequentially"
/// Implementation: Target-dependent series of copy operations
val concat : dest:nativeptr<byte> -> parts:string list -> string

/// Copy a string to a destination buffer
/// Semantic: "Duplicate this string's content"
/// Implementation: Single Memory.copy operation
val copyTo : src:string -> dest:nativeptr<byte> -> string
```

### Simplified Sample Code

With these primitives, `03_HelloWorldHalfCurried` becomes:

```fsharp
let hello() =
    let inputBuffer = NativePtr.stackalloc<byte> 256

    Console.writeB "Enter your name: "B

    let nameLen = Console.readLine inputBuffer 256
    let name = String.fromBuffer(inputBuffer, nameLen)

    let outputBuffer = NativePtr.stackalloc<byte> 512

    // Express INTENT, not implementation
    let greeting = String.concat outputBuffer [
        "Hello, "
        name
        "!"
    ]

    greeting |> Console.writeln
```

This version:
- **Tests the intended patterns**: Pipe operator, curried functions, string construction
- **Hides implementation complexity**: No explicit loops in user code
- **Enables target-aware optimization**: Alex can select optimal copy strategy
- **Maintains F# idioms**: Clean, functional style without exposing native pointer arithmetic

## The Compilation Pipeline

### Stage 1: PSG Construction

The Program Semantic Graph captures the semantic structure. For `String.concat`, the PSG would represent:

```
App:FunctionCall [concat]
├── Ident:outputBuffer
└── List
    ├── App:FunctionCall [fromBytes]
    │   └── Const:Bytes "Hello, "
    ├── Ident:name
    └── App:FunctionCall [fromBytes]
        └── Const:Bytes "!"
```

Critically, the PSG captures **what** is happening (concatenation) without prescribing **how** (loops, memcpy, etc.).

### Stage 2: Coeffect Analysis

The existing `ContextRequirement` type in `PSG/Types.fs` already supports the necessary distinctions:

```fsharp
type ContextRequirement =
    | Pure              // No external dependencies - candidate for optimization
    | AsyncBoundary     // Suspension point - requires continuation handling
    | ResourceAccess    // File/network access - requires sequencing
```

Memory operations like `Memory.copy` would be classified as `Pure` with a `MemoryAccess` pattern, enabling aggressive optimization. In contrast, `Console.readLine` would be classified as `ResourceAccess`, requiring proper sequencing.

This aligns with the coeffect-driven compilation strategy described in [Context-Aware Compilation](../../../SpeakEZ/hugo/content/blog/Context%20Aware%20Compilation.md):

> "By tracking what code needs from its environment, coeffects provide exactly the information needed to choose between parallel execution strategies."

### Stage 3: Alex Target-Aware Lowering

Alex, the "Library of Alexandria" transformation layer, maintains target-specific knowledge and selects appropriate lowering strategies. For `Memory.copy`:

```fsharp
// Conceptual Alex pattern matching
match (operation, target) with
| (MemoryCopy(src, dest, len), ARM) ->
    // ARM: Emit tight stack-based loop
    emitARMCopyLoop src dest len

| (MemoryCopy(src, dest, len), x86_64) when len >= 32 ->
    // x86_64 with sufficient data: Use SIMD
    emitAVXCopy src dest len

| (MemoryCopy(src, dest, len), x86_64) when len < 32 ->
    // x86_64 small copy: Inline unrolled
    emitUnrolledCopy src dest len

| (MemoryCopy(src, dest, len), GPU) ->
    // GPU: Parallel lane copy
    emitParallelCopy src dest len
```

This architecture enables the optimization strategies described in [Cache-Conscious Memory Management: CPU Edition](../../../SpeakEZ/hugo/content/blog/Cache%20Aware%20Compilation%20CPU.md):

> "The Firefly compiler's sophisticated cache analysis will become possible through BAREWire's deterministic memory layouts. Where traditional compilers must make conservative assumptions about memory access patterns, Firefly is designed to operate with complete knowledge of data placement."

### Stage 4: MLIR MLIR Generation

The final MLIR output reflects the target-aware decision. For x86_64 with a small copy:

```mlir
// Unrolled copy for 7 bytes ("Hello, ")
%0 = llvm.load %src : !llvm.ptr -> i64  // Load 8 bytes
llvm.store %0, %dest : i64, !llvm.ptr   // Store 8 bytes (overlapping is fine)
```

For larger copies:

```mlir
// LLVM intrinsic for optimized memcpy
call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %src, i64 %len, i1 false)
```

## The Two Categories of Loops

### Category 1: Bounded Memory Operations (Semantic Primitives)

**Characteristics:**
- Iteration count known at loop entry
- Pure computation (no I/O, no resource access)
- Operations on contiguous memory
- Deterministic behavior

**Examples:**
- `Memory.copy`, `Memory.zero`, `Memory.compare`
- String concatenation, buffer initialization
- Array transformations with known bounds

**Compilation Strategy:**
- Capture as semantic primitive in PSG
- Coeffect: `Pure` with `MemoryAccess` pattern
- Alex selects target-optimal implementation
- May not emit a "loop" at all (SIMD, intrinsics, unrolling)

**PSG Representation:**
```
SemanticOperation:MemoryCopy
├── Source: nativeptr<byte>
├── Destination: nativeptr<byte>
└── Length: int (statically known or runtime value)
```

### Category 2: Control Flow Loops (User-Experience Loops)

**Characteristics:**
- Iteration count potentially unbounded or data-dependent
- Resource dependencies (I/O, network, user interaction)
- Temporal concerns (timing, state changes over time)
- Observable side effects

**Examples:**
- Event loops, REPL loops
- Polling for input (`while true do ... readLine ...`)
- Retry logic, timeout loops
- The `TimeLoop` sample: `while counter < iterations do ... sleep ...`

**Compilation Strategy:**
- Emit actual control flow (branches, blocks)
- Coeffect: `ResourceAccess` or `Temporal`
- Cannot be "flattened" or vectorized
- Must preserve sequential semantics

**PSG Representation:**
```
WhileLoop
├── Condition: boolean expression
└── Body: Sequential with ResourceAccess operations
```

### The TimeLoop Example

The `TimeLoop` sample demonstrates a legitimate control flow loop:

```fsharp
let displayTimeLoop (iterations: int) =
    let mutable counter = 0

    while counter < iterations do
        let now = currentDateTimeString()  // ResourceAccess: system time
        WriteLine now                       // ResourceAccess: console I/O
        sleep 1000                          // Temporal: timing dependency
        counter <- counter + 1

    WriteLine "Done."
```

This loop **cannot** be optimized into a memory operation because:
1. It has `ResourceAccess` (console I/O, system time)
2. It has `Temporal` dependencies (sleep)
3. The iterations are semantically meaningful (user sees 5 separate outputs)

The compiler must emit this as actual control flow, not a flattened operation.

## The Runtime Without a Runtime

A key challenge in Firefly's design is providing "runtime" functionality without a managed runtime. This is where the Alloy library becomes essential.

### Traditional Managed Runtimes Provide:
- Garbage collection
- String interning and manipulation
- Buffer management
- I/O abstraction

### Firefly/Alloy Provides:
- **Deterministic memory**: Stack allocation, arena allocation, explicit lifetime
- **Semantic primitives**: Operations that express intent, compiled to target-optimal code
- **Zero-copy patterns**: BAREWire protocol for data transfer without copying
- **RAII-style resource management**: `use` bindings for automatic cleanup

The semantic primitives like `Memory.copy` ARE the "runtime machinery" - but they're resolved at compile time to target-specific implementations rather than runtime library calls.

From [Cache-Conscious Memory Management: CPU Edition](../../../SpeakEZ/hugo/content/blog/Cache%20Aware%20Compilation%20CPU.md):

> "BAREWire eliminates this uncertainty through deterministic, compile-time memory layouts. Every field offset, structure size, and alignment requirement becomes statically known and guaranteed."

This determinism enables the compiler to make optimization decisions that would be impossible with a traditional runtime's dynamic allocation.

## Integration with Existing Infrastructure

### PSG Types (Already Exists)

The `ContextRequirement` and `ComputationPattern` types in `PSG/Types.fs` already support this model:

```fsharp
type ContextRequirement =
    | Pure              // Candidate for aggressive optimization
    | AsyncBoundary     // Requires continuation handling
    | ResourceAccess    // Requires sequencing

type ComputationPattern =
    | DataDriven        // Push-based, eager - may parallelize
    | DemandDriven      // Pull-based, lazy - sequential by nature
```

### Alex Architecture (Already Exists)

Alex's multi-target design already supports the pattern:
- `Bindings/` contains target-aware MLIR generation patterns
- `Patterns/` contains semantic pattern recognition
- `CodeGeneration/TypeMapping.fs` maps F# types to target representations

### Required Additions

1. **Alloy primitives**: `Memory.copy`, `Memory.zero`, `String.concat`
2. **PSG patterns**: Recognition of semantic memory operations
3. **Alex bindings**: Target-specific lowering for memory primitives
4. **Coeffect integration**: Classification of memory operations as `Pure`

## Sample Progression Implications

The Fidelity HelloWorld samples should progress as follows:

| Sample | Tests | Loop Strategy |
|--------|-------|---------------|
| 01_HelloWorldDirect | Direct module calls, static strings | No loops |
| 02_HelloWorldSaturated | Saturated function calls | No loops |
| 03_HelloWorldHalfCurried | Pipe operator, curried functions, **Alloy primitives** | Loops hidden in primitives |
| 04_HelloWorldFullCurried | Full currying, Result.map, lambdas | Loops hidden in primitives |
| TimeLoop | **Explicit control flow loops**, timing, I/O | Emitted as control flow |

This progression ensures:
1. Simple samples don't require loop MLIR generation infrastructure
2. Loop MLIR generation is tested where loops ARE the semantic intent
3. Developer experience remains clean and F#-idiomatic
4. Target-aware optimization is possible for memory operations

## Conclusion

The distinction between **semantic primitives** (bounded memory operations) and **control flow loops** (user-experience loops with resource dependencies) is fundamental to Firefly's architecture. By expressing memory operations as semantic intent rather than implementation, we enable:

1. **Target-aware optimization**: ARM loops, x86 SIMD, GPU parallelism
2. **Clean developer experience**: F# idioms without exposing low-level details
3. **Proper coeffect analysis**: Pure operations can be optimized aggressively
4. **Maintainable compiler**: Loop MLIR generation deferred to samples where it's the point

This aligns with the broader Fidelity vision articulated in the SpeakEZ blog corpus:

> "The beauty of coeffect-driven compilation is that developers write clean, intent-focused code while the compiler handles the messy details of determining strategies for parallel execution."

The Alloy library is not merely a BCL replacement - it's the semantic layer that enables Firefly to deliver native performance with functional elegance. The primitives we add today become the foundation for tomorrow's hardware-aware optimizations across CPU, GPU, and emerging accelerator architectures.

## References

- [Context-Aware Compilation](../../../SpeakEZ/hugo/content/blog/Context%20Aware%20Compilation.md) - Coeffect analysis and compilation strategies
- [Cache-Conscious Memory Management: CPU Edition](../../../SpeakEZ/hugo/content/blog/Cache%20Aware%20Compilation%20CPU.md) - BAREWire and deterministic memory layouts
- [GPU Cache-Aware Compilation](../../../SpeakEZ/hugo/content/blog/Cache%20Aware%20Compilation%20GPU.md) - Target-aware optimization for parallel architectures
- `PSG/Types.fs` - ContextRequirement and ComputationPattern types
- `Alex/` - Target-aware transformation layer
- `samples/console/TimeLoop/` - Control flow loop example
- `samples/console/FidelityHelloWorld/` - Progressive sample suite
