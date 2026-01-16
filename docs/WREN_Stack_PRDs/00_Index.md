# WREN Stack PRDs - Feature Implementation Guides

> **Product Requirement Documents for F# Native Language Features**

## Purpose

This folder contains detailed PRDs (Product Requirement Documents) for each language feature needed to complete the WREN Stack soft launch. Each PRD serves as a standalone implementation guide covering:

1. **FNCS Layer** - Type checking, semantic graph construction, intrinsic definitions
2. **Firefly/Alex Layer** - SSA assignment, witness emission, MLIR generation
3. **Specification Updates** - fsnative-spec changes required
4. **Validation** - Which FidelityHelloWorld samples must pass

## Architectural Foundation

All PRDs assume familiarity with:
- [Architecture_Canonical.md](../Architecture_Canonical.md) - Two-layer FNCS/Alex model
- [FNCS_Architecture.md](../FNCS_Architecture.md) - F# Native Compiler Services
- [PSG_Nanopass_Architecture.md](../PSG_Nanopass_Architecture.md) - Nanopass pipeline

## Reference Implementations

| Domain | Reference | Location |
|--------|-----------|----------|
| ML Type Systems | MLKit, FStar, OCaml | Academic papers, ~/repos/FStar |
| MLIR Generation | Triton-CPU, mlir-hs | ~/repos/triton-cpu, ~/repos/mlir-hs |
| F# Semantics | fsnative-spec | ~/repos/fsnative-spec/spec/ |

## PRD Index

### Phase A: Foundation (Samples 11-13)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-11](./PRD-11-Closures.md) | MLKit-Style Flat Closures | 11_Closures | In Progress |
| [PRD-12](./PRD-12-HigherOrderFunctions.md) | Higher-Order Functions | 12_HigherOrderFunctions | Planned |
| [PRD-13](./PRD-13-Recursion.md) | Recursive Bindings | 13_Recursion | Planned |

### Phase B: Sequences (Samples 14-15)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-14](./PRD-14-SimpleSeq.md) | Sequence Expressions | 14_SimpleSeq | Planned |
| [PRD-15](./PRD-15-SeqOperations.md) | Seq.map/filter/take | 15_SeqOperations | Planned |

### Phase C: Lazy Evaluation (Sample 16)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-16](./PRD-16-LazyValues.md) | Lazy Thunks | 16_LazyValues | Planned |

### Phase D: Async (Samples 17-19)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-17](./PRD-17-BasicAsync.md) | LLVM Coroutines Foundation | 17_BasicAsync | Planned |
| [PRD-18](./PRD-18-AsyncAwait.md) | let! and Suspension | 18_AsyncAwait | Planned |
| [PRD-19](./PRD-19-AsyncParallel.md) | Async.Parallel | 19_AsyncParallel | Planned |

### Phase E: Scoped Regions (Samples 20-22)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-20](./PRD-20-BasicRegion.md) | Region Allocation | 20_BasicRegion | Planned |
| [PRD-21](./PRD-21-RegionPassing.md) | Region Parameters | 21_RegionPassing | Planned |
| [PRD-22](./PRD-22-RegionEscape.md) | Escape Analysis | 22_RegionEscape | Planned |

### Phase F: Networking (Samples 23-24)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-23](./PRD-23-SocketBasics.md) | Socket I/O | 23_SocketBasics | Planned |
| [PRD-24](./PRD-24-WebSocketEcho.md) | WebSocket Protocol | 24_WebSocketEcho | Planned |

### Phase G: Desktop (Samples 25-26)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-25](./PRD-25-GTKWindow.md) | GTK FFI | 25_GTKWindow | Planned |
| [PRD-26](./PRD-26-WebViewBasic.md) | WebKitGTK | 26_WebViewBasic | Planned |

### Phase H: Threading (Samples 27-28)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-27](./PRD-27-BasicThread.md) | Thread Primitives | 27_BasicThread | Planned |
| [PRD-28](./PRD-28-MutexSync.md) | Mutex/Lock | 28_MutexSync | Planned |

### Phase I: MailboxProcessor Capstone (Samples 29-31)

| PRD | Feature | Sample | Status |
|-----|---------|--------|--------|
| [PRD-29](./PRD-29-BasicActor.md) | Actor Foundation | 29_BasicActor | Planned |
| [PRD-30](./PRD-30-ActorReply.md) | PostAndReply | 30_ActorReply | Planned |
| [PRD-31](./PRD-31-ParallelActors.md) | Multi-Actor System | 31_ParallelActors | Planned |

## PRD Template Structure

Each PRD follows this structure:

```
1. Executive Summary
2. Language Feature Specification
3. FNCS Layer Implementation
   - Type Definitions
   - SemanticKind Changes
   - Type Checking Logic
   - Traversal Updates
4. Firefly/Alex Layer Implementation
   - SSA Assignment
   - Coeffect/Analysis Passes
   - Witness Emission
   - MLIR Output Specification
5. fsnative-spec Updates
6. Reference Patterns (MLKit/FStar/Triton)
7. Validation
   - Sample Code
   - Expected Output
   - Regression Tests
8. Files to Create/Modify
9. Implementation Checklist
```

## Dependency Graph

```
Sample 11 (Closures)
    │
    ├──► Sample 12 (HOF) ──► Sample 29 (BasicActor)
    │                              │
    │                              ▼
    │                        Sample 30 (ActorReply)
    │                              │
    │                              ▼
    │                        Sample 31 (ParallelActors)
    │                              ▲
    │                              │
Sample 13 (Recursion)              │
                                   │
Sample 14-15 (Seq) ────────────────┤
                                   │
Sample 16 (Lazy) ──────────────────┤
                                   │
Sample 17-19 (Async) ──────────────┤
                                   │
Sample 20-22 (Regions) ────────────┤
                                   │
Sample 27-28 (Threading) ──────────┘
```

## Serena Memory References

Key architectural memories to consult:
- `architecture_principles` - Core constraints
- `fncs_architecture` - FNCS design patterns
- `codata_photographer_principle` - Witness observation model
- `four_pillars_of_transfer` - Transfer architecture
- `negative_examples` - Mistakes to avoid

## Contributing

When creating a new PRD:
1. Copy the template structure from an existing PRD
2. Fill in all sections completely
3. Ensure FNCS and Firefly changes are both addressed
4. Include specific file paths and function names
5. Add to this index with correct status
