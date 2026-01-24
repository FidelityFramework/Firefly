# Firefly PRDs - Master Index

> **Purpose**: Category-prefixed PRD organization enabling modular growth across platform targets

---

## Category Overview

| Prefix | Category | Description | PRD Range |
|--------|----------|-------------|-----------|
| **F-xx** | Foundation | Core compilation, Samples 01-10 | F-00 to F-10 |
| **C-xx** | Computation | Closures, HOFs, Lazy, Seq | C-01 to C-07 |
| **A-xx** | Async | Async, Await, Regions | A-01 to A-06 |
| **I-xx** | IO | Sockets, WebSocket | I-01 to I-02 |
| **D-xx** | Desktop | GTK, WebView | D-01 to D-02 |
| **T-xx** | Threading | Threads, Mutex, Actors | T-01 to T-05 |
| **R-xx** | Reactive | Observable, Rx operators | R-01 to R-03 |
| **E-xx** | Embedded | USB, RTOS, LVGL | Future |

---

## Target Requirements Matrix

Not all PRDs apply to all targets. This matrix clarifies which features are needed for which platform configurations.

| PRD Category | WREN Stack | QuantumCredential | LVGL MCU | Unikernel |
|--------------|------------|-------------------|----------|-----------|
| **Foundation (F)** | Required | Required | Required | Required |
| **Computation (C)** | Required | Required | Required | Required |
| **Async (A)** | Required | Partial (A-01,02) | Partial | Required |
| **IO (I)** | Required | Partial | N/A | Required |
| **Desktop (D)** | Required | N/A | N/A | N/A |
| **Threading (T)** | Required | Partial | N/A | Required |
| **Reactive (R)** | Required | Partial | N/A | Optional |
| **Embedded (E)** | N/A | Required | Required | N/A |

---

## Complete PRD List

### Foundation (F-xx) - Core Compilation

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [F-00](F-00-Synopsis.md) | Foundation Series Synopsis | 01-10 | Complete |
| [F-01](F-01-HelloWorldDirect.md) | HelloWorldDirect | 01 | Retrospective |
| [F-02](F-02-ArenaAllocation.md) | Arena Allocation | 02 | Retrospective |
| [F-03](F-03-PipeOperators.md) | Pipe Operators | 03 | Retrospective |
| [F-04](F-04-CurryingLambdas.md) | Currying & Lambdas | 04 | Retrospective |
| [F-05](F-05-DiscriminatedUnions.md) | Discriminated Unions | 05 | Retrospective |
| [F-06](F-06-InteractiveParsing.md) | Interactive Parsing | 06 | Retrospective |
| [F-07](F-07-BitsIntrinsics.md) | Bits Intrinsics | 07 | Retrospective |
| [F-08](F-08-OptionType.md) | Option Type | 08 | Retrospective |
| [F-09](F-09-ResultType.md) | Result Type | 09 | Retrospective |
| [F-10](F-10-RecordTypes.md) | Record Types | 10 | Retrospective |

### Computation (C-xx) - Functional Abstractions

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [C-01](C-01-Closures.md) | MLKit-Style Flat Closures | 11 | In Progress |
| [C-02](C-02-HigherOrderFunctions.md) | Higher-Order Functions | 12 | Planned |
| [C-03](C-03-Recursion.md) | Recursion & Tail Calls | 13 | Planned |
| [C-04](C-04-CoreCollections.md) | Core Collections | 13a | Planned |
| [C-05](C-05-Lazy.md) | Lazy Evaluation | 14 | Planned |
| [C-06](C-06-SimpleSeq.md) | Simple Sequences | 15 | Planned |
| [C-07](C-07-SeqOperations.md) | Sequence Operations | 16 | Planned |

### Async (A-xx) - Asynchronous Programming

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [A-01](A-01-BasicAsync.md) | Basic Async | 17 | Planned |
| [A-02](A-02-AsyncAwait.md) | Async/Await | 18 | Planned |
| [A-03](A-03-AsyncParallel.md) | Async Parallel | 19 | Planned |
| [A-04](A-04-BasicRegion.md) | Basic Regions | 20 | Planned |
| [A-05](A-05-RegionPassing.md) | Region Passing | 21 | Planned |
| [A-06](A-06-RegionEscape.md) | Region Escape Analysis | 22 | Planned |

### IO (I-xx) - Network & File I/O

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [I-01](I-01-SocketBasics.md) | Socket Basics | 23 | Planned |
| [I-02](I-02-WebSocketEcho.md) | WebSocket Echo | 24 | Planned |

### Desktop (D-xx) - Desktop Applications

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [D-01](D-01-GTKWindow.md) | GTK Window | 25 | Planned |
| [D-02](D-02-WebViewBasic.md) | WebView Basic | 26 | Planned |

### Threading (T-xx) - Concurrency

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [T-01](T-01-BasicThread.md) | Basic Threading | 27 | Planned |
| [T-02](T-02-MutexSync.md) | Mutex Synchronization | 28 | Planned |
| [T-03](T-03-BasicActor.md) | Basic Actor | 29 | Planned |
| [T-04](T-04-ActorReply.md) | Actor Reply | 30 | Planned |
| [T-05](T-05-ParallelActors.md) | Parallel Actors | 31 | Planned |

### Reactive (R-xx) - Reactive Extensions

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| [R-01](R-01-ObservableFoundations.md) | Observable Foundations | 32 | Planned |
| [R-02](R-02-ObservableOperators.md) | Observable Operators | 33 | Planned |
| [R-03](R-03-ObservableIntegration.md) | Observable Integration | 34 | Planned |

### Embedded (E-xx) - MCU & Unikernel

| PRD | Title | Sample | Status |
|-----|-------|--------|--------|
| E-01 | USB Device Stack | Future | Planned |
| E-02 | RTOS Integration | Future | Planned |
| E-03 | LVGL Basics | Future | Planned |

---

## Dependency Graph

```
Foundation (F-01 to F-10)
    └── Complete: Core compilation infrastructure
            │
            ├── Computation (C-01 to C-07)
            │       │
            │       ├── C-01 Closures ← F-04 Lambdas, F-10 Records
            │       ├── C-02 HOFs ← C-01 Closures
            │       ├── C-03 Recursion ← C-01 Closures
            │       ├── C-04 Collections ← C-02, C-03
            │       ├── C-05 Lazy ← C-01 Closures
            │       ├── C-06 SimpleSeq ← C-05 Lazy
            │       └── C-07 SeqOps ← C-06 SimpleSeq
            │
            ├── Async (A-01 to A-06)
            │       │
            │       ├── A-01 BasicAsync ← C-01, C-05
            │       ├── A-02 AsyncAwait ← A-01
            │       ├── A-03 AsyncParallel ← A-02
            │       ├── A-04 BasicRegion ← A-01
            │       ├── A-05 RegionPassing ← A-04
            │       └── A-06 RegionEscape ← A-05
            │
            ├── IO (I-01 to I-02)
            │       │
            │       ├── I-01 SocketBasics ← A-02
            │       └── I-02 WebSocketEcho ← I-01
            │
            ├── Desktop (D-01 to D-02)
            │       │
            │       ├── D-01 GTKWindow ← A-02
            │       └── D-02 WebViewBasic ← D-01
            │
            ├── Threading (T-01 to T-05)
            │       │
            │       ├── T-01 BasicThread ← A-02
            │       ├── T-02 MutexSync ← T-01
            │       ├── T-03 BasicActor ← T-02, C-07
            │       ├── T-04 ActorReply ← T-03
            │       └── T-05 ParallelActors ← T-04
            │
            └── Reactive (R-01 to R-03)
                    │
                    ├── R-01 ObservableFoundations ← C-01, A-02
                    ├── R-02 ObservableOperators ← R-01, C-07
                    └── R-03 ObservableIntegration ← R-02, T-03
```

---

## Status Legend

| Status | Meaning |
|--------|---------|
| Complete | Sample passes regression tests, PRD closed |
| In Progress | Active development |
| Planned | PRD written, not started |
| Retrospective | Foundation sample, PRD backfilled |
| Future | Not yet specified |

---

## Migration Notes

This index replaces the previous sequential `WREN_Stack_PRDs/00_Index.md (now PRDs/INDEX.md)` with category-prefixed organization.

**Mapping from old to new:**
- PRD-00 → F-00 (Foundation Synopsis)
- PRD-11 to PRD-16 → C-01 to C-07 (Computation)
- PRD-17 to PRD-22 → A-01 to A-06 (Async)
- PRD-23 to PRD-24 → I-01 to I-02 (IO)
- PRD-25 to PRD-26 → D-01 to D-02 (Desktop)
- PRD-27 to PRD-31 → T-01 to T-05 (Threading)

Cross-references in existing PRDs have been updated to use the new naming scheme.
