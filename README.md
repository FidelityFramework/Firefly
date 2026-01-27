# Firefly: F# to Native Compiler

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](Commercial.md)

<p align="center">
🚧 <strong>Under Active Development</strong> 🚧<br>
<em>This project is in early development and not intended for production use.</em>
</p>

Firefly is an ahead-of-time F# compiler that produces native executables without runtime dependencies or garbage collection. Built as a .NET CLI tool, Firefly leverages [F# Native Compiler Services (FNCS)](https://github.com/FidelityFramework/fsnative) for type checking and semantic analysis, then generates MLIR through the Alex multi-targeting layer, and finally produces native binaries via LLVM.

## 🎯 Vision

Firefly transforms F# from a managed runtime language into a true systems programming language with deterministic memory guarantees. By orchestrating compilation through MLIR, Firefly provides flexible memory management strategies - from zero-allocation stack-based code to arena-managed bulk operations and structured concurrency through actors. This enables developers to write everything from embedded firmware to high-performance services while preserving F#'s elegant syntax and type safety.

Central to Firefly's approach is the Program Semantic Graph (PSG) - a representation that combines syntactic structure with rich type information and optimization metadata. This enables comprehensive static analysis and allows the compiler to choose optimal memory strategies based on usage patterns.

**Key Innovations:** 
- **Flexible memory strategies** from zero-allocation to arena-based management
- **Deterministic resource management** through RAII principles and compile-time tracking
- **Type-preserving compilation** maintaining F#'s rich type system throughout the pipeline
- **Progressive lowering** through MLIR dialects with continuous verification
- **Platform-aware optimization** adapting to target hardware characteristics

## 🏗️ Architecture

Firefly is organized into five major layers:

### Pipeline Overview

```
F# Source Code
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FrontEnd: FNCS Integration                                  │
│ - Type checking & semantic analysis (FNCS)                  │
│ - Native type universe (NTUKind)                            │
│ - Intrinsic operations (Sys.*, NativePtr.*, etc.)          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Program Semantic Graph (PSG) - Nanopass Pipeline           │
│ Phase 1: Structural Construction (SynExpr → PSG nodes)     │
│ Phase 2: Symbol Correlation (attach FSharpSymbol)          │
│ Phase 3: Soft-Delete Reachability (mark unreachable)       │
│ Phase 4: Typed Tree Overlay (type resolution via zipper)   │
│ Phase 5+: Enrichment (def-use, operations, saturation)     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ MiddleEnd: PSGElaboration + Alex                           │
│ - PSGElaboration: Coeffect analysis (SSA, Platform, etc.)  │
│ - Alex/Zipper: Traverse PSG with XParsec patterns          │
│ - Alex/Elements: Atomic MLIR ops (internal)                │
│ - Alex/Patterns: Composable elision templates              │
│ - Alex/Witnesses: Category-based code generation           │
└─────────────────────────────────────────────────────────────┘
    ↓
MLIR (multiple dialects)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ MiddleEnd: MLIR Optimization                               │
│ - Dialect lowering (scf→cf, arith→llvm, etc.)             │
│ - MLIR-to-LLVM translation                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
LLVM IR
    ↓
┌─────────────────────────────────────────────────────────────┐
│ BackEnd: Native Code Generation                            │
│ - LLVM compilation to object files                         │
│ - Linking (freestanding, console, embedded)                │
└─────────────────────────────────────────────────────────────┘
    ↓
Native Binary (no runtime dependencies)
```

### Directory Structure

- **`CLI/`** - Command-line interface (commands, diagnostics)
- **`Core/`** - Core types, configuration, timing utilities
- **`FrontEnd/`** - FNCS integration, platform templates
- **`MiddleEnd/`**
  - `PSGElaboration/` - Coeffect analysis (SSA, platform, mutability, etc.)
  - `Alex/` - Multi-targeting MLIR generation layer
    - `Dialects/Core/` - MLIR type system
    - `CodeGeneration/` - Type mapping and sizing
    - `Traversal/` - PSGZipper and XParsec combinators
    - `Elements/` - Atomic MLIR operations (internal)
    - `Patterns/` - Composable elision templates
    - `Witnesses/` - Category-based code generators
    - `Pipeline/` - Compilation orchestration
  - `MLIROpt/` - MLIR dialect lowering
- **`BackEnd/`** - LLVM code generation and linking

### Key Principles

1. **True Nanopass Architecture** - PSG construction through distinct phases, each inspectable
2. **Coeffects Over Runtime** - Pre-computed analysis (SSA, platform, lifetimes) guides generation
3. **Zipper + XParsec** - Bidirectional PSG traversal with composable pattern matching
4. **Element/Pattern/Witness** - Three-layer MLIR generation (atomic ops → compositions → observers)
5. **Type Fidelity** - F# types map precisely to native representations, never erased

## Native Type System

Firefly uses **F# Native Compiler Services (FNCS)** to provide a native type universe at compile time. Unlike traditional F# where types are .NET runtime types, FNCS types (`NTUKind`) map directly to native representations:

- **Primitive types** - `i8`, `i16`, `i32`, `i64`, `f32`, `f64` map to MLIR integer/float types
- **Pointers** - `nativeptr<'T>` becomes opaque LLVM pointers (`!llvm.ptr`)
- **Structures** - Records and unions become LLVM structs with precise layouts
- **Functions** - Function types preserve calling conventions

### Intrinsic Operations

Platform operations are compiler intrinsics defined in FNCS, not library code:

**System operations (`Sys` module):**
- `Sys.write` - Direct syscall for file descriptor writes
- `Sys.read` - Direct syscall for file descriptor reads
- `Sys.exit` - Process termination
- Platform-specific bindings (Linux x86_64, ARM64, etc.)

**Memory operations (`NativePtr` module):**
- `NativePtr.read` - Load from pointer
- `NativePtr.write` - Store to pointer
- `NativePtr.add` - Pointer arithmetic

**String operations (`NativeStr` module):**
- Native UTF-8 strings (fat pointer: `{ptr: !llvm.ptr, len: i64}`)
- Zero-copy string operations

All intrinsics are resolved at compile time to platform-specific implementations.

## Hello World Example

A minimal Firefly program uses FNCS intrinsics for I/O:

```fsharp
module HelloWorld

// String literal - becomes UTF-8 fat pointer {ptr, len}
let greeting = "Hello, World!\n"

[<EntryPoint>]
let main (args: string[]) : int =
    // Sys.write is a compiler intrinsic
    // fd=1 (stdout), NativePtr to string data, length
    let written = Sys.write 1 (NativeStr.ptr greeting) (NativeStr.length greeting)
    0
```

Compile and run:
```bash
cd samples/console/FidelityHelloWorld/01_HelloWorldDirect
firefly compile HelloWorld.fidproj
./HelloWorld
```

See `/samples/console/FidelityHelloWorld/` for progressive complexity examples:
- `01_HelloWorldDirect` - Direct syscall writes
- `02_HelloWorldSaturated` - Let bindings and function calls
- `03_HelloWorldHalfCurried` - Pipe operators and partial application
- `04_HelloWorldFullCurried` - Full currying, Result.map, lambdas

## 🎛️ Project Configuration

Firefly projects use `.fidproj` files (TOML format):

```toml
[package]
name = "HelloWorld"

[compilation]
memory_model = "stack_only"    # Current: stack-based allocation
target = "native"               # Native binary output

[build]
sources = ["HelloWorld.fs"]
output = "HelloWorld"
output_kind = "console"         # "console" or "freestanding"
```

### Output Kinds

- **`console`** - Links with libc, suitable for desktop/server applications
- **`freestanding`** - No libc dependency, for embedded or OS-level code

### Build Process

1. FNCS type-checks F# source and produces PSG
2. PSGElaboration computes coeffects (SSA, platform, etc.)
3. Alex traverses PSG and generates MLIR
4. MLIR optimizations and dialect lowering
5. LLVM compilation to object files
6. Linking to final binary

## 🔬 Development Workflow

```bash
# Build the Firefly compiler
cd src
dotnet build

# Compile a project
firefly compile MyProject.fidproj

# Keep intermediate files (-k flag) for debugging
firefly compile MyProject.fidproj -k

# Inspect intermediates (in target/intermediates/)
# 01_psg0.json          - Initial PSG with reachability
# 02_intrinsic_recipes.json - Intrinsic elaboration
# 03_psg1.json          - PSG after intrinsic fold-in
# 04_saturation_recipes.json - Baker saturation
# 05_psg2.json          - Final PSG to Alex
# 06_coeffects.json     - Coeffect analysis
# 07_output.mlir        - Generated MLIR
# 08_output.ll          - LLVM IR

# Run regression tests
cd tests/regression
dotnet fsi Runner.fsx           # Run all samples
dotnet fsi Runner.fsx -- --parallel  # Parallel execution
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect  # Specific sample
```

## 🎯 Current Status & Design Philosophy

### ✅ What Firefly Provides Today

- **No runtime dependencies** - Freestanding binaries or minimal libc linkage
- **Type-preserving compilation** - F# types map precisely to native representations
- **FNCS native type universe** - Compile-time type resolution (NTUKind)
- **MLIR-based codegen** - Progressive lowering through verified dialects
- **Platform-specific intrinsics** - Syscalls and memory operations adapted to target

### 🏗️ Design Principles

- **Upstream fixes** - Issues belong at their source in the pipeline, not patched downstream
- **Coeffects over runtime** - Pre-computed analysis guides code generation
- **Nanopass architecture** - Small, composable transformations
- **Type-level firewalls** - Internal modules enforce architectural boundaries
- **No semantic gaps** - Alex fills zero gaps; missing semantics belong in FNCS

## 📁 Samples

Located in `/samples/console/FidelityHelloWorld/`, these samples demonstrate progressive F# language features:

### Basic I/O and Functions
- `01_HelloWorldDirect` - Direct syscall writes
- `02_HelloWorldSaturated` - Let bindings and saturated calls
- `03_HelloWorldHalfCurried` - Pipe operators and partial application
- `04_HelloWorldFullCurried` - Full currying, Result.map, lambdas
- `05_AddNumbers` - Simple arithmetic operations
- `06_AddNumbersInteractive` - User input and string parsing
- `07_BitsTest` - Bitwise operations

### Data Structures
- `08_Option` - Option<'T> type (Some/None)
- `09_Result` - Result<'T, 'E> type (Ok/Error)
- `10_Records` - Record types with field access
- `13a_SimpleCollections` - List, Map, Set operations

### Functions and Control Flow
- `11_Closures` - Closure capture and flat closures
- `12_HigherOrderFunctions` - Functions as values, map, filter
- `13_Recursion` - Recursive functions and pattern matching

### Advanced Features
- `14_Lazy` - Lazy<'T> evaluation (flat closures + memoization)
- `15_SimpleSeq` - Seq<'T> basic iteration
- `16_SeqOperations` - Seq.map, Seq.filter, etc.

### Other Samples
- `TimeLoop/` - Mutable state, while loops, platform time operations
- `SignalTest/` - Signal handling (POSIX signals)

Run the test suite: `cd tests/regression && dotnet fsi Runner.fsx`

## 📋 Roadmap

Firefly development follows category-prefixed PRDs (Product Requirement Documents) organized by functional area. See [docs/PRDs/INDEX.md](docs/PRDs/INDEX.md) for complete details.

### ✅ Completed PRDs

**Foundation (F-xx) - Core Compilation** - Complete
- F-01 through F-10: Samples 01-10 (HelloWorld, Pipes, Currying, DUs, Option, Result, Records)
- Basic types, arithmetic, control flow, pattern matching

### 🚧 In Progress

**Computation (C-xx) - Functional Abstractions**
- ✅ **C-01: Flat Closures** (Sample 11) - MLKit-style closure capture with SSA cost model
- ✅ **C-02: Higher-Order Functions** (Sample 12) - Function composition, map, filter, fold
- ✅ **C-03: Recursion** (Sample 13) - Tail-call optimization
- ✅ **C-04: Core Collections** (Sample 13a) - List, Map, Set with native implementations
- ✅ **C-05: Lazy Evaluation** (Sample 14) - Lazy<'T> via flat closures + memoization
- ✅ **C-06: Simple Sequences** (Sample 15) - Seq<'T> state machines with MoveNext protocol
- ✅ **C-07: Sequence Operations** (Sample 16) - Seq.map, Seq.filter, Seq.fold, etc.

### 📋 Planned PRDs

**Async (A-xx) - Asynchronous Programming**
- A-01: Basic Async (Sample 17) - Delimited continuations
- A-02: Async/Await (Sample 18) - Async state machines
- A-03: Async Parallel (Sample 19) - Parallel async execution
- A-04 through A-06: Region-based memory management

**IO (I-xx) - Network & File I/O**
- I-01: Socket Basics (Sample 23)
- I-02: WebSocket Echo (Sample 24)

**Desktop (D-xx) - Desktop Applications**
- D-01: GTK Window (Sample 25)
- D-02: WebView Basic (Sample 26)

**Threading (T-xx) - Concurrency**
- T-01 through T-05: Threads, Mutex, Actors (Samples 27-31)

**Reactive (R-xx) - Reactive Extensions**
- R-01 through R-03: Observable patterns (Samples 32-34)

**Embedded (E-xx) - MCU & Unikernel**
- E-01: USB Device Stack
- E-02: RTOS Integration
- E-03: LVGL Basics

### Active Feature Areas

**WREN Stack (Alpha)**
- **W**eb Assembly Runtime Environment **N**ative
- Freestanding WebAssembly compilation target
- WASM-specific intrinsics and memory model
- Browser and WASI runtime support
- Zero JavaScript interop overhead

**Micro-Controller Support**
- ARM Cortex-M targets (M0, M0+, M3, M4, M7)
- Freestanding embedded execution
- Bare-metal hardware access
- Interrupt-driven programming model
- Target platforms:
  - STM32F7 and STM32L5 (STMicroelectronics)
  - Renesas RA6M5 (Renesas)
  - RP2040 (Raspberry Pi Pico)

### Future Directions

**Advanced Type System:**
- Linear types for zero-copy operations
- Affine types for resource management
- Region-based memory safety

**Concurrency & Distribution:**
- **Olivier**: Actor-based memory isolation with per-actor arenas
- **Prospero**: Cross-process memory coordination
- **BAREWire**: Zero-copy serialization for distributed systems

**Platform Expansion:**
- WebAssembly via WAMI dialect
- Embedded ARM Cortex-M (STM32, nRF)
- GPU compute kernels
- RISC-V support

## 🤝 Contributing

We will welcome contributions after establishing a solid baseline. Areas of particular interest:

- **Memory optimization patterns** - Novel approaches to deterministic memory management
- **MLIR dialect design** - Preserving F# semantics through compilation
- **Platform targets** - Backend support for new architectures
- **Verification** - Formal proofs of memory safety properties

## License

Firefly is dual-licensed under both the Apache License 2.0 and a Commercial License.

### Open Source License

For open source projects, academic use, non-commercial applications, and internal tools, use Firefly under the **Apache License 2.0**.

### Commercial License

A Commercial License is required for incorporating Firefly into commercial products or services. See [Commercial.md](Commercial.md) for details.

### Patent Notice

Firefly is part of the Fidelity Framework, which includes technology covered by U.S. Patent Application No. 63/786,247 "System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol". See [PATENTS.md](PATENTS.md) for licensing details.

## 🙏 Acknowledgments

- **Don Syme and F# Contributors**: For creating an elegant functional language
- **MLIR Community**: For the multi-level IR infrastructure
- **LLVM Community**: For robust code generation
- **Rust Community**: For demonstrating zero-cost abstractions in systems programming
- **Fable Project**: For showing F# can target alternative environments