# WRENStack Roadmap

## Overview

WRENStack is the paved path for building native desktop applications with F#. The goal: a user installs MLIR/clang tooling, .NET 10 SDK, and the Firefly dotnet tool, then uses a WRENStack template to build native desktop applications with WebView frontends and bidirectional IPC.

**WREN** = **W**ebview + **R**eactive + **E**mbedded + **N**ative

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        F# Source Code                                │
│  ┌──────────────────┐         ┌──────────────────┐                  │
│  │  Frontend (Fable) │         │  Backend (Firefly)│                 │
│  │  - Partas.Solid   │         │  - Native binary  │                 │
│  │  - SolidJS UI     │         │  - GTK/WebKitGTK  │                 │
│  └────────┬─────────┘         └────────┬─────────┘                  │
│           │                            │                             │
│           │    ┌────────────────┐      │                             │
│           └───►│  BAREWire IPC  │◄─────┘                             │
│                │  (Shared Types)│                                    │
│                └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Compilation Pipeline                             │
│                                                                      │
│  F# Source → FNCS → PSG → Alex → MLIR → LLVM → Native Binary        │
│                                                                      │
│  FNCS: F# Native Compiler Services (type resolution, intrinsics)    │
│  PSG:  Program Semantic Graph (nanopass-enriched coeffects)         │
│  Alex: Zipper + XParsec + Bindings (witness and emit via Templates) │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Philosophy

Every feature follows the **Photographer Principle**:

1. **Nanopasses build the scene** - Enrich the PSG with coeffects (metadata the Zipper will observe)
2. **The Zipper moves attention** - Navigate the graph structure, never dispatch
3. **Active Patterns focus the lens** - Semantic classification, not string matching
4. **Transfer snaps the picture** - Emit MLIR via parameterized Templates

If you find yourself computing metadata during code generation, stop - that belongs in a nanopass. The mise-en-place is complete before Transfer begins.

## Key Architectural Decisions

### 1. LLVM Coroutines for Async (NOT MLIR Async Dialect)

The MLIR async dialect requires linking against `mlir-async-runtime`, which brings in pthreads and malloc dependencies. This conflicts with Fidelity's freestanding/minimal-deps goal.

**Decision**: Use LLVM coroutine intrinsics (`llvm.coro.*`) which compile to state machines at compile time - no runtime library needed.

Async is codata - defined by observation (resume/suspend), not construction. A nanopass tags suspension points with state indices (coeffect). Alex folds over the async body; at each `AwaitPoint`, the `CoroSuspend` template emits the LLVM intrinsic. The coroutine frame is the "frozen computation."

See: [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md)

### 2. Desktop API Stratification

**Decision**: Fidelity.Desktop provides a unified API with platform backends as separate FFI packages (not intrinsics).

```
Fidelity.Desktop          # Unified API (platform-agnostic)
├── Fidelity.Desktop.GTK  # GTK backend (FFI to libgtk/WebKitGTK)
└── Fidelity.Desktop.Qt   # Qt backend (future, FFI to Qt)
```

GTK bindings are Layer 2 (Farscape-generated FFI). The `(|FFICall|_|)` Active Pattern matches external function calls. Alex witnesses and emits via the `ExternCall` template. No special GTK logic in the compiler - the bindings are data, not routing.

### 3. seq/lazy as MoveNext Structs

**Decision**: Implement `seq` and `lazy` as struct-based state machines (MoveNext pattern). This is the interim approach; DCont-style (delimited continuations) is the future path.

Seq and Lazy are codata - defined by observation (MoveNext/Current, Force), not construction. A nanopass tags each `yield` with its state index (coeffect). Alex folds over the seq body; at each `YieldPoint`, the `SeqStateMachine` template emits the state transition. No central seq dispatcher.

### 4. Scoped Regions for Dynamic Memory

**Decision**: Implement compiler-inferred deterministic memory regions that sit between stack-only allocation and full Olivier/Prospero actor-managed arenas.

**Key Properties**:
- **Compiler-inferred disposal**: No `IDisposable`, no `use` keyword. Region is a coeffect - the compiler knows Region-typed values need cleanup and inserts disposal at scope exits.
- **Scope-bound lifetime**: Disposal determined by lexical scope, not reachability. A nanopass performs scope analysis and tags exit points.
- **Passable to functions**: Region parameters carry `BorrowedRegion` coeffect - can allocate but doesn't own lifetime.
- **Growable by default**: `Region.create` allows growth; `Region.createFixed` for embedded/MCU targets.

Alex witnesses `RegionCreate` and emits via platform Bindings (`PageAlloc` template: mmap on Linux, VirtualAlloc on Windows). This is NOT a runtime - the compiler manages allocation, scope determines disposal.

See: Serena memory `scoped_regions_architecture`

### 5. MailboxProcessor as Capstone Feature

**Decision**: MailboxProcessor (F#'s actor model primitive) is the **capstone feature** of WRENStack. It synthesizes:
- Async (for message loop) via LLVM coroutines
- Closures (for behavior functions) - capture set is coeffect-tagged
- Threading (for true parallelism) via OS thread primitives
- **Scoped Regions (for dynamic memory in worker threads)**
- Records/DUs (for message types)

**Foundational Implementation**: OS thread per actor + mutex-protected queue + LLVM coroutine for async loop. No DCont, no Olivier/Prospero supervision - just the core actor semantics compiled to native.

The `(|ActorStart|_|)` Active Pattern matches `MailboxProcessor.Start`. Alex witnesses and emits via composition of existing templates - `ThreadCreate` for the actor thread, `CoroFrame` for the async loop, `MutexQueue` for the message buffer. The actor struct: `{ Thread; Queue; Behavior }`.

This foundation works for desktop AND embedded/MCU/unikernel targets. True parallel worker threads can process compute-heavy tasks while the main thread handles WebView communication.

See: Serena memory `mailboxprocessor_first_stage`

## Feature Inventory by PRD Category

Features are organized by PRD category (see [PRDs/INDEX.md](./PRDs/INDEX.md) for complete mapping):

| PRD Category | Features Proven |
|--------------|-----------------|
| **F-01 to F-04** | Basic I/O, let bindings, pipes, currying, lambdas |
| **F-05 to F-06** | Discriminated unions, interactive parsing |
| **F-07** | Bits intrinsics (htons, ntohl, bit casting) |
| **F-08** | Option type (homogeneous DU) |
| **F-09** | Result type (heterogeneous DU, arena affinity) |
| **F-10** | Record types and patterns |
| **C-01, C-02** | Closures, higher-order functions |
| **C-03** | Recursion and tail calls |
| **C-04** | Core collections (List, Map, Set) |
| **C-05** | Lazy evaluation - thunk observation |
| **C-06, C-07** | Sequences - codata state machines |
| **A-01 to A-03** | Async via LLVM coroutines - suspension coeffects |
| **A-04 to A-06** | **Scoped Regions** - compiler-inferred disposal |
| **I-01, I-02** | Sockets, WebSocket - Region-backed I/O buffers |
| **D-01, D-02** | GTK window, WebView - FFI bindings |
| **T-01, T-02** | Threading primitives - Thread/Mutex coeffects |
| **T-03 to T-05** | **MailboxProcessor - CAPSTONE** |
| **R-01 to R-03** | Observable/Reactive - event streams |

See: [PRDs/INDEX.md](./PRDs/INDEX.md) for detailed PRD specifications

## Dependency Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WRENStack Application                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Fidelity.Desktop│  │ Fidelity.Signal │  │ Fidelity.WebView│     │
│  │ (Unified API)   │  │ (Reactive)      │  │ (WebView API)   │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                │                                    │
│                    ┌───────────▼───────────┐                       │
│                    │  Fidelity.Platform    │                       │
│                    │  (OS Abstractions)    │                       │
│                    │  - Sys.* intrinsics   │                       │
│                    │  - GTK/Qt bindings    │                       │
│                    │  - WebSocket          │                       │
│                    └───────────┬───────────┘                       │
│                                │                                    │
│                    ┌───────────▼───────────┐                       │
│                    │      BAREWire         │                       │
│                    │  (Binary Protocol)    │                       │
│                    │  - Encoding/Decoding  │                       │
│                    │  - IPC Messages       │                       │
│                    └───────────────────────┘                       │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        Compiler Infrastructure                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │    Firefly      │  │    fsnative     │  │   FCS (F#)      │     │
│  │  (Compiler)     │  │    (FNCS)       │  │  (Frontend)     │     │
│  │  - Alex/Zipper  │  │  - Intrinsics   │  │  - Parsing      │     │
│  │  - PSG/Nanopass │  │  - NTU Types    │  │  - Type Check   │     │
│  │  - Templates    │  │  - Coeffects    │  │  - Symbols      │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Milestone Checklist (by PRD)

### Phase A: Foundation Completion (F-xx)
- [ ] **F-10**: Record pattern matching - `RecordPattern` Active Pattern
- [ ] **C-01**: Closure source verification - capture coeffects

### Phase B: Computation (C-xx)
- [ ] **C-03**: Recursive let bindings - `RecursiveBinding` coeffect
- [ ] **C-04**: Core collections (List, Map, Set)
- [ ] **C-05**: Lazy evaluation - thunk observation
- [ ] **C-06**: Simple seq { for/yield } - yield state indices
- [ ] **C-07**: Seq.map, Seq.filter, Seq.take - composed iterators

### Phase C: Async & Regions (A-xx)
- [ ] **A-01**: Basic async { return value } - trivial coroutine
- [ ] **A-02**: async { let! x = ... } - suspension coeffects
- [ ] **A-03**: Async.Parallel - sequential composition
- [ ] **A-04**: Basic region allocation and disposal - `NeedsCleanup` coeffect
- [ ] **A-05**: Region parameter passing - `BorrowedRegion` coeffect
- [ ] **A-06**: Region escape prevention and copyOut - escape analysis nanopass

### Phase D: Networking (I-xx)
- [ ] **I-01**: TCP socket basics - Sys.* intrinsics with Region buffers
- [ ] **I-02**: WebSocket echo server - Layer 3 library on Sys intrinsics

### Phase E: Desktop (D-xx)
- [ ] **D-01**: GTK window - `FFICall` Active Pattern, `ExternCall` template
- [ ] **D-02**: WebView with HTML content - WebKitGTK FFI bindings

### Phase F: Threading & Actors (T-xx)
- [ ] **T-01**: Thread.create / Thread.join - Thread coeffect, capture analysis
- [ ] **T-02**: Mutex synchronization - `SyncPrimitive` coeffect
- [ ] **T-03**: Basic actor with Post/Receive - `ActorStart` Active Pattern
- [ ] **T-04**: PostAndReply (request-response) - `SyncChannel` template
- [ ] **T-05**: Parallel actors with region-based worker memory - composed templates

### Phase G: Reactive (R-xx)
- [ ] **R-01**: Observable foundations - basic event streams
- [ ] **R-02**: Observable operators - map, filter, merge
- [ ] **R-03**: Observable integration with Async/Actors

### Phase H: WRENStack Template
- [ ] Working WrenHello demonstration with actor backend
- [ ] dotnet template package
- [ ] Installation documentation

## Library Repositories

| Library | Location | Purpose |
|---------|----------|---------|
| Fidelity.Platform | `~/repos/Fidelity.Platform/` | OS abstractions, syscalls, GTK bindings |
| Fidelity.Signal | `~/repos/Fidelity.Signal/` | Reactive primitives (SolidJS-inspired) |
| Fidelity.WebView | `~/repos/Fidelity.WebView/` | High-level WebView API |
| Fidelity.Desktop | `~/repos/Fidelity.Desktop/` | Unified desktop API |
| BAREWire | `~/repos/BAREWire/` | Binary protocol, IPC |
| Partas.Solid | `~/repos/Partas.Solid/` | F# DSL for SolidJS (Fable) |

## Reference Application

**WrenHello** (`~/repos/WrenHello/`) is the reference WREN application demonstrating:
- Dual-track build (Fable frontend + Firefly backend)
- Shared Protocol.fs with BAREWire codecs
- WebView IPC communication
- GTK window management

## Related Documentation

**PRDs** (authoritative feature specifications):
- [PRDs/INDEX.md](./PRDs/INDEX.md) - Master PRD index with dependency graph

**Architecture**:
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Two-layer FNCS/Alex model
- [FNCS_Architecture.md](./FNCS_Architecture.md) - F# Native Compiler Services
- [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md) - Technical approach for async
- [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) - Desktop UI architecture

**Serena Memories** (architectural guidance):
- `architecture_principles` - Layer separation, non-dispatch model
- `four_pillars_of_transfer` - Coeffects, Active Patterns, Zipper, Templates
- `codata_photographer_principle` - Witness, don't construct
- `computation_strategy_architecture` - Strategy overview
- `async_llvm_coroutines` - LLVM coro strategy for async/seq/lazy
- `mailboxprocessor_first_stage` - MailboxProcessor implementation strategy
- `scoped_regions_architecture` - Region design and coeffects
