# Firefly: F# Native Compiler

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](Commercial.md)
[![Pipeline](https://img.shields.io/badge/Pipeline-25%20nanopasses-blue)]()
[![Samples](https://img.shields.io/badge/Samples-3/16%20working-yellow)]()

<p align="center">
🚧 <strong>Under Active Development</strong> 🚧<br>
<em>Early development (Feb 2026: 3/16 samples working). Not production-ready.</em>
</p>

Ahead-of-time F# compiler producing native executables without managed runtime or garbage collection. Leverages [F# Native Compiler Services (FNCS)](https://github.com/FidelityFramework/fsnative) for type checking and semantic analysis, generates MLIR through Alex multi-targeting layer, produces native binaries via LLVM.

## Current Status (February 2026)

**Working Samples**: 3 of 16 console samples compile and execute correctly:
- ✅ 01_HelloWorldDirect (static strings, basic Console)
- ✅ 02_HelloWorldSaturated (mutable variables in loops, string interpolation)
- ✅ 03_HelloWorldHalfCurried (pipe operators, function values)

**Recent Achievements**:
- **VarRef SSA Auto-Loading**: Mutable variables used as memref indices now auto-load values compositionally
- **FNCS Contract Compliance**: NativeStr.fromPointer honors substring extraction via allocate + memcpy
- **Compositional Patterns**: Element/Pattern/Witness stratification validated with cross-discipline composition

**Known Limitations**:
- 13 of 16 samples fail compilation (closure capture, higher-order functions, complex control flow)
- Managed mutability limited to local variables in simple loops
- No escape analysis (mutables that outlive scope unsupported)
- Generic instantiation and SRTP resolution issues remain

See: `docs/PRDs/README.md` for full feature roadmap and status.

## Architecture

Firefly implements a true nanopass compiler architecture with ~25 distinct passes from F# source to native binary. Each pass performs a single, well-defined transformation on an intermediate representation.

### Nanopass Pipeline

```
F# Source
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FNCS (6 phases)                                             │
│ Phase 0: FCS parse and type check                           │
│ Phase 1: Structural construction (SynExpr → PSG)            │
│ Phase 2: Symbol correlation (attach FSharpSymbol)           │
│ Phase 3: Soft-delete reachability (mark unreachable)        │
│ Phase 4: Typed tree overlay (type resolution via zipper)    │
│ Phase 5+: Enrichment (def-use, operations, saturation)      │
└─────────────────────────────────────────────────────────────┘
    ↓ PSG (Program Semantic Graph)
┌─────────────────────────────────────────────────────────────┐
│ Alex: Element/Pattern/Witness Architecture                  │
│ • Elements (module internal): Atomic MLIR ops with XParsec  │
│ • Patterns (public): Composable templates from Elements     │
│ • Witnesses (public): Thin observers (~20 lines each)       │
│                                                             │
│ 16 category-selective witnesses:                            │
│ - ApplicationWitness: function calls, intrinsics            │
│ - ControlFlowWitness: if/while/for with MLIR SCF dialect    │
│ - BindingWitness: let bindings, mutable variables          │
│ - LambdaWitness: function definitions                        │
│ - OptionWitness, LazyWitness, SeqWitness: type constructors │
│ - 9 additional witnesses for F# coverage                    │
└─────────────────────────────────────────────────────────────┘
    ↓ Portable MLIR (memref, arith, func, index, scf)
┌─────────────────────────────────────────────────────────────┐
│ MLIR Structural Passes (4 passes)                           │
│ 1. Structural folding (deduplicate function bodies)          │
│ 2. Declaration collection (external function declarations)   │
│ 3. Type normalization (insert memref.cast at call sites)     │
│ 4. FFI conversion (delegated to mlir-opt)                    │
└─────────────────────────────────────────────────────────────┘
    ↓ MLIR (portable dialects)
┌─────────────────────────────────────────────────────────────┐
│ mlir-opt Dialect Lowering                                    │
│ - memref → LLVM struct                                       │
│ - arith → LLVM arithmetic                                    │
│ - scf → cf → LLVM control flow                               │
│ - index → platform word size                                 │
└─────────────────────────────────────────────────────────────┘
    ↓ LLVM IR
┌─────────────────────────────────────────────────────────────┐
│ LLVM + Clang                                                 │
│ - Optimization passes                                        │
│ - Code generation                                            │
│ - Linking                                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Native Binary (zero runtime dependencies)
```

## Architectural Principles

**1. Element/Pattern/Witness Stratification (Feb 2026)**
- **Elements** (module internal): Atomic MLIR operations with XParsec state threading
- **Patterns** (public): Composable templates that compose Elements across disciplines (memref + arith + func)
- **Witnesses** (public): Thin observers (~20 lines) that delegate to Patterns via `tryMatch`

Witnesses physically cannot import Elements - they must use Patterns. This enforces compositional architecture.

**2. XParsec Throughout**
Composable parser combinators at every level: Elements use `parser { }` CE for state threading, Patterns pull data from Coeffects monadically, Witnesses use `tryMatch` for PSG structure matching. No central dispatch hub, no mutable accumulator state.

**3. Coeffects Over Runtime**
Pre-computed analysis (SSA assignment, platform resolution, mutability tracking, DU layouts) guides code generation. No runtime discovery. Coeffects are computed once before Alex witnessing begins.

**4. Codata Witnesses**
Witnesses observe PSG structure and return MLIR operations. They do not build or transform—observation only. This preserves PSG immutability and enables nanopass composition.

**5. Zipper + XParsec**
Bidirectional PSG traversal with composable pattern matching. Enables local reasoning without global context threading.

**6. Portable Until Proven Backend-Specific**
MiddleEnd emits only portable MLIR dialects (memref, arith, func, index, scf). Target-specific lowering delegated to mlir-opt and LLVM.

## Native Type System

FNCS provides native type universe (`NTUKind`) at compile time. Types are compiler intrinsics, not runtime constructs:

- Primitives: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `nativeint` → MLIR integer/float/index types
- Pointers: `nativeptr<'T>` → opaque pointers
- **Strings**: `memref<?xi8>` directly (NO fat pointers, NO structs) → memref operations
- Structures: Records/unions → MLIR struct types with precise layout
- **Mutable Variables**: `let mutable x = ...` → `memref<1x'T>` with alloca + load/store (limited to local scope)

### String Representation (MLIR Memref Semantics)

**ARCHITECTURAL PRINCIPLE**: Strings ARE memrefs, not fat pointer structs.

```mlir
Static literal:    memref<13xi8>    // "Hello, World!"
Dynamic (readln):  memref<?xi8>     // Runtime-sized, dimension intrinsic
Concatenation:     memref<?xi8>     // Allocated with actual combined length
```

String operations use memref.dim to get length, memref.alloc for runtime-sized allocation, memcpy for substring extraction. This is MLIR-native, not LLVM-specific.

### Intrinsic Operations

Platform operations defined in FNCS as compiler intrinsics:

**System (`Sys` module):**
- `Sys.write(fd: i64, buf: memref<?xi8>): i64` — syscall (extracts ptr + length from memref)
- `Sys.read(fd: i64, buf: memref<?xi8>): i64` — syscall
- `Sys.exit(code: i32): unit` — process termination

**Memory (`NativePtr` module):**
- `NativePtr.read(ptr: nativeptr<'T>): 'T` — load
- `NativePtr.write(ptr: nativeptr<'T>, value: 'T): unit` — store
- `NativePtr.stackalloc(count: nativeint): nativeptr<'T>` — stack allocation (memref.alloca)

**String (`String` + `NativeStr` modules):**
- `String.length(s: memref<?xi8>): int` — memref.dim extraction
- `String.concat2(s1: memref<?xi8>, s2: memref<?xi8>): memref<?xi8>` — allocate + memcpy
- `NativeStr.fromPointer(buf: memref<Nxi8>, len: nativeint): memref<?xi8>` — substring extraction

All intrinsics resolve to platform-specific MLIR during Alex traversal.

## Minimal Example

```fsharp
module HelloWorld

[<EntryPoint>]
let main argv =
    Console.write "Enter your name: "
    let name = Console.readln()
    Console.writeln $"Hello, {name}!"
    0
```

Compiles to native binary with:
- Zero .NET runtime dependencies
- Direct syscalls for I/O
- Stack allocation for locals (memref.alloca)
- Mutable variables via TMemRef auto-loading
- MLIR → LLVM optimization

```bash
firefly compile HelloWorld.fidproj
echo "Alice" | ./target/helloworld
# Output: "Enter your name: Hello, Alice!"
```

See `/samples/console/FidelityHelloWorld/` for progressive examples (3 of 16 currently working).

## Project Configuration

`.fidproj` files use TOML:

```toml
[package]
name = "HelloWorld"

[compilation]
memory_model = "stack_only"
target = "native"

[build]
sources = ["HelloWorld.fs"]
output = "helloworld"
output_kind = "console"  # or "freestanding"
```

## Build Workflow

```bash
# Build compiler
cd src && dotnet build

# Compile project
firefly compile MyProject.fidproj

# Keep intermediates for inspection
firefly compile MyProject.fidproj -k
```

### Intermediate Artifacts

With `-k` flag, inspect each nanopass output in `target/intermediates/`:

| File | Nanopass Output |
|------|----------------|
| `01_psg0.json` | Initial PSG with reachability |
| `02_intrinsic_recipes.json` | Intrinsic elaboration recipes |
| `03_psg1.json` | PSG after intrinsic fold-in |
| `04_saturation_recipes.json` | Baker saturation recipes |
| `05_psg2.json` | Final saturated PSG to Alex |
| `06_coeffects.json` | SSA, platform, mutability analysis |
| `07_output.mlir` | Alex-generated portable MLIR |
| `08_after_structural_folding.mlir` | Deduplicated function bodies |
| `09_after_ffi_conversion.mlir` | FFI boundary preparation |
| `10_after_declaration_collection.mlir` | External function declarations |
| `11_after_type_normalization.mlir` | Call site type casts |
| `12_output.ll` | LLVM IR after mlir-opt lowering |

### Regression Testing

```bash
cd tests/regression
dotnet fsi Runner.fsx                    # All samples
dotnet fsi Runner.fsx -- --parallel      # Parallel execution
dotnet fsi Runner.fsx -- --sample 02_HelloWorldSaturated
```

**Current Status** (Feb 2026):
- 3 of 16 samples pass (01, 02, 03)
- 13 samples fail compilation (04-16)

## Directory Structure

```
src/
├── CLI/                    Command-line interface
├── Core/                   Configuration, timing, diagnostics
├── FrontEnd/               FNCS integration
├── MiddleEnd/
│   ├── PSGElaboration/     Coeffect analysis (SSA, platform, DU layouts)
│   └── Alex/               MLIR generation layer
│       ├── Dialects/       MLIR type system
│       ├── CodeGeneration/ Type mapping, sizing
│       ├── Traversal/      PSGZipper, XParsec combinators
│       ├── Elements/       Atomic MLIR ops (module internal)
│       ├── Patterns/       Composable templates (public)
│       ├── Witnesses/      16 category-selective observers (public)
│       └── Pipeline/       Orchestration, MLIR passes
└── BackEnd/                LLVM compilation, linking
```

## Multi-Stack Targeting

Portable MLIR enables diverse hardware targets:

| Target | Status | Lowering Path |
|--------|--------|---------------|
| x86-64 CPU | ✅ Working (limited) | memref → LLVM struct |
| ARM Cortex-M | 🚧 Planned | memref → custom embedded lowering |
| CUDA GPU | 🚧 Planned | memref → SPIR-V/PTX lowering |
| AMD ROCm | 🚧 Planned | memref → SPIR-V lowering |
| Xilinx FPGA | 🚧 Planned | memref → HDL stream buffer |
| CGRA | 🚧 Planned | memref → dataflow lowering |
| NPU | 🚧 Planned | memref → tensor descriptor |
| WebAssembly | 🚧 Planned | memref → WASM linear memory |

Previously blocked by hard-coded LLVM types. Now possible via target-specific mlir-opt lowering.

## Documentation

| Document | Content |
|----------|---------|
| `docs/Architecture_Canonical.md` | FNCS-first architecture, intrinsic modules |
| `docs/PSG_Nanopass_Architecture.md` | Phase 0-5+ detailed design |
| `docs/Alex_Architecture_Overview.md` | Element/Pattern/Witness stratification |
| `docs/XParsec_PSG_Architecture.md` | Pattern combinators, codata witnesses |
| `docs/Coeffect_Analysis_Architecture.md` | SSA assignment, DU layouts, platform resolution |
| `docs/PRDs/README.md` | Product requirement documents by category |

## Roadmap

Development organized by category-prefixed PRDs. See [docs/PRDs/README.md](docs/PRDs/README.md).

**Foundation (F-01 to F-10)**: Core compilation
- ✅ F-01 HelloWorldDirect (static strings)
- ✅ F-02 ArenaAllocation (now: memref.alloc for strings)
- ✅ F-03 PipeOperators (|> reduction)
- ⏳ F-04 to F-10 (in progress, partial support)

**Computation (C-01 to C-07)**: Closures, HOFs, Sequences
- 🚧 C-01 Closures (active development - closure capture unimplemented)
- 📋 C-02 to C-07 (planned - depends on C-01)

**Async (A-01 to A-06)**: Async workflows, region-based memory
- 📋 A-01 to A-06 (planned - depends on C-01)

**Other Categories**: I/O (I-xx), Desktop (D-xx), Threading (T-xx), Reactive (R-xx), Embedded (E-xx) - all planned for future work.

## Recent Changes (February 2026)

### Managed Mutability Milestone

**Achievement**: Local mutable variables in simple loops now work via TMemRef auto-loading.

**What Works**:
- `let mutable pos = 0` → `memref.alloca() : memref<1xindex>`
- Mutable variables as memref indices (auto-load value before use)
- Mutable variables in loop conditions (while, for)
- String operations honoring FNCS contracts (substring extraction)

**What Doesn't Work**:
- Mutable variables captured in closures (no escape analysis)
- Mutable variables passed across function boundaries
- Higher-order functions with mutable state
- Complex control flow with escaping mutables

**Architectural Pattern Established**: Compositional auto-loading via type-driven discrimination (Rule 9 in managed mutability architecture principles).

See: Serena memory `managed_mutability_feb2026_milestone` for complete details.

## Contributing

Areas of interest:
- MLIR dialect design for novel hardware targets
- Memory optimization patterns (escape analysis, loop unrolling)
- Nanopass transformations for advanced F# features
- Closure capture and higher-order function support
- F* integration for proof-carrying code

## License

Dual-licensed under Apache License 2.0 and Commercial License. See [Commercial.md](Commercial.md) for commercial use. Patent notice: U.S. Patent Application No. 63/786,247 "System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol". See [PATENTS.md](PATENTS.md).

## Acknowledgments

- **Don Syme and F# Contributors**: Quotations, active patterns, computation expressions enable self-hosting
- **MLIR Community**: Multi-level IR infrastructure
- **LLVM Project**: Robust code generation
- **Nanopass Framework**: Compiler architecture principles
- **Triton-CPU**: MLIR-based compilation patterns
- **MLKit**: Flat closure representation patterns
