# Parallel Nanopass Architecture - Final Design

**Date**: January 27, 2026
**Status**: Implemented - Ready for Witness Integration
**Pattern**: One Witness = One Nanopass + IcedTasks Parallel Execution

---

## Core Concept

**Each witness file = one nanopass**

- **LiteralWitness.fs** → LiteralNanopass
- **ArithWitness.fs** → ArithNanopass
- **ControlFlowWitness.fs** → ControlFlowNanopass
- etc.

**Each nanopass**:
- Complete PSG zipper traversal
- Selectively witnesses its category of nodes
- Returns `WitnessOutput.skip` for nodes it doesn't handle
- Runs in parallel with other nanopasses via IcedTasks

**Results overlay/fold** into single cohesive MLIR graph.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PSG from FNCS                          │
└─────────────────────────────────────────────────────────┘
                         ↓
          ┌──────────────┴──────────────┐
          │   Nanopass Registry         │
          │   (all witnesses register)  │
          └──────────────┬──────────────┘
                         ↓
          ┌──────────────────────────────┐
          │  IcedTasks.ColdTask.Parallel │
          └──────────────┬───────────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
  LiteralNanopass  ArithNanopass  ControlFlowNanopass
  (complete PSG)   (complete PSG)  (complete PSG)
        ↓                ↓                ↓
   Partial MLIR    Partial MLIR    Partial MLIR
   {ops, bindings} {ops, bindings} {ops, bindings}
        │                │                │
        └────────────────┼────────────────┘
                         ↓
          ┌──────────────────────────────┐
          │   Envelope Pass (Collect)    │
          │   Overlay/Fold Accumulators  │
          └──────────────┬───────────────┘
                         ↓
          ┌──────────────────────────────┐
          │   Cohesive MLIR Graph        │
          └──────────────────────────────┘
```

---

## Implementation

### 1. Core Types (`NanopassArchitecture.fs`)

```fsharp
/// A nanopass is a complete PSG traversal
type Nanopass = {
    Name: string
    Witness: WitnessContext -> SemanticNode -> WitnessOutput
}

/// Registry of all nanopasses
type NanopassRegistry = {
    Nanopasses: Nanopass list
}

/// Run a single nanopass over entire PSG
let runNanopass (nanopass: Nanopass) (graph: SemanticGraph) (coeffects: TransferCoeffects)
    : MLIRAccumulator =
    // Traverse entire PSG from entry points
    // Call nanopass.Witness on each node
    // Collect results in accumulator
    ...

/// Overlay/merge two accumulators (associative)
let overlayAccumulators (acc1: MLIRAccumulator) (acc2: MLIRAccumulator) : MLIRAccumulator =
    // Merge ops, bindings, visited sets
    ...
```

### 2. Parallel Execution (`ParallelNanopass.fs`)

```fsharp
/// Run multiple nanopasses in parallel via IcedTasks
let runNanopassesParallel (nanopasses: Nanopass list) (graph: SemanticGraph) (coeffects: TransferCoeffects)
    : MLIRAccumulator list =

    // Create cold tasks
    let nanopassTasks =
        nanopasses
        |> List.map (fun nanopass ->
            coldTask {
                return runNanopass nanopass graph coeffects
            })

    // Execute in parallel
    nanopassTasks
    |> List.toArray
    |> Array.Parallel.map (fun task -> task.Result)
    |> Array.toList

/// Envelope pass: Collect and merge results
let collectEnvelope (nanopassResults: MLIRAccumulator list) : MLIRAccumulator =
    nanopassResults
    |> List.reduce overlayAccumulators

/// Main orchestration
let executeNanopasses (config: ParallelConfig) (registry: NanopassRegistry) (graph: SemanticGraph) (coeffects: TransferCoeffects)
    : MLIRAccumulator =
    // Run nanopasses (parallel or sequential)
    let results =
        if config.EnableParallel then
            runNanopassesParallel registry.Nanopasses graph coeffects
        else
            runNanopassesSequential registry.Nanopasses graph coeffects

    // Envelope: Overlay results
    collectEnvelope results
```

### 3. Witness Integration Pattern

**Each witness file registers a nanopass:**

```fsharp
// LiteralWitness.fs
module Alex.Witnesses.LiteralWitness

open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.TransferTypes

/// Witness literal nodes ONLY
let private witnessLiteral (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Literal lit ->
        // Handle literal
        witnessLiteralImpl ctx node lit

    | _ ->
        // Not a literal - skip
        WitnessOutput.skip

/// Register as nanopass
let nanopass : Nanopass = {
    Name = "Literal"
    Witness = witnessLiteral
}
```

**ArithWitness.fs:**

```fsharp
module Alex.Witnesses.ArithWitness

/// Witness arithmetic nodes ONLY
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.BinaryOp _ -> witnessBinaryArith ctx node
    | SemanticKind.UnaryOp _ -> witnessUnary ctx node
    | SemanticKind.Comparison _ -> witnessComparison ctx node
    | _ -> WitnessOutput.skip

/// Register as nanopass
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
```

### 4. Registry Population

**In MLIRTransfer.fs or new `WitnessRegistry.fs`:**

```fsharp
module Alex.Traversal.WitnessRegistry

open Alex.Traversal.NanopassArchitecture

// Import all witnesses
module LiteralWitness = Alex.Witnesses.LiteralWitness
module ArithWitness = Alex.Witnesses.ArithWitness
module ControlFlowWitness = Alex.Witnesses.ControlFlowWitness
// ... etc

/// Global registry of all witness nanopasses
let globalRegistry =
    NanopassRegistry.empty
    |> NanopassRegistry.register LiteralWitness.nanopass
    |> NanopassRegistry.register ArithWitness.nanopass
    |> NanopassRegistry.register ControlFlowWitness.nanopass
    // ... register all witnesses
```

---

## Selective Witnessing

**Key mechanism**: Each nanopass traverses ALL nodes but only witnesses relevant ones.

```fsharp
// LiteralNanopass sees ALL nodes
for node in allNodesInPSG do
    match node.Kind with
    | SemanticKind.Literal _ ->
        // This nanopass handles it
        emitMLIR ...

    | _ ->
        // Skip (some other nanopass handles it)
        WitnessOutput.skip
```

**Result**: Each node witnessed by exactly ONE nanopass.

---

## Overlay/Fold Properties

### Associativity

`overlayAccumulators` is associative:

```fsharp
overlay (overlay a b) c = overlay a (overlay b c)
```

**Why**:
- `TopLevelOps`: List append is associative
- `Visited`: Set union is associative
- `Bindings`: Map merge is associative (disjoint keys - different nanopasses)

### Disjoint Nodes

**Critical property**: Each node witnessed by exactly ONE nanopass.

- LiteralNanopass witnesses `SemanticKind.Literal`
- ArithNanopass witnesses `SemanticKind.BinaryOp`, `SemanticKind.UnaryOp`
- ControlFlowNanopass witnesses `SemanticKind.IfThenElse`, `SemanticKind.While`

**No conflicts**: Bindings from different nanopasses have disjoint keys.

---

## IcedTasks Integration

**Why IcedTasks**:
- Cold tasks = proper F# async/task integration
- `Array.Parallel.map` for parallel execution
- Clean API for collecting results

**Pattern**:
```fsharp
coldTask {
    return runNanopass nanopass graph coeffects
}
```

**Execution**:
```fsharp
tasks
|> Array.Parallel.map (fun task -> task.Result)
```

---

## Future: Tiered Nanopasses

**Current**: All nanopasses run in parallel (no dependencies).

**Future** (if needed): Tier nanopasses by dependencies.

```fsharp
type NanopassTier = {
    Tier: int  // 0 = no dependencies, 1 = depends on tier 0, etc.
    Nanopasses: Nanopass list
}

// Execute tiers sequentially, nanopasses within tier in parallel
for tier in tiers do
    let results = runNanopassesParallel tier.Nanopasses graph coeffects
    // ... merge
```

**For now**: Simple flat parallelism (all nanopasses independent).

---

## Design Decision: Full-Fat Parallel Execution

**Current Strategy**: ALL registered nanopasses run in parallel, regardless of whether nodes exist for them to witness.

**Rationale**:
- Empty nanopasses are cheap (fast skip-only traversals)
- Simplicity: no discovery coordination, easy to reason about
- Correctness first: no risk of missing node types
- For small programs, empty pass overhead is negligible (microseconds)

**Future Optimization**: Discovery pass for large projects (10,000+ nodes) to filter nanopasses before execution. **Deferred until profiling shows need.**

See `Parallel_Nanopass_Final_Summary.md` for complete trade-off analysis.

---

## Migration Path for Witnesses

### Current Witnesses (OLD pattern)

```fsharp
// ArithWitness.fs - OLD
let witnessBinaryArith (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Binary arithmetic pattern match failed"
```

Called from MLIRTransfer.fs dispatch:
```fsharp
match node.Kind with
| SemanticKind.BinaryOp _ -> ArithWitness.witnessBinaryArith ctx node
| SemanticKind.Literal _ -> LiteralWitness.witnessLiteral ctx node
// ...
```

### New Pattern (Nanopass registration)

```fsharp
// ArithWitness.fs - NEW
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.BinaryOp _ ->
        match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Binary arithmetic pattern match failed"

    | SemanticKind.UnaryOp _ ->
        match tryMatch pUnary ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Unary operation pattern match failed"

    | _ ->
        // Not arithmetic - skip
        WitnessOutput.skip

/// Nanopass registration
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
```

No dispatch in MLIRTransfer.fs - all handled by parallel orchestration.

---

## Benefits

### 1. Natural Parallelism

Each witness runs independently on entire PSG. No coordination needed.

### 2. Simple Addition

New witness? Just:
1. Create witness file
2. Export `nanopass` value
3. Register in `WitnessRegistry.fs`

No changes to orchestration logic.

### 3. Clean Separation

Each witness only knows its category. No global dispatch logic.

### 4. Scalability

**Dozens of nanopasses** scale naturally:
- Each runs in parallel
- Results overlay associatively
- No coordination overhead

### 5. Testability

Each nanopass testable in isolation:
```fsharp
[<Test>]
let ``LiteralNanopass handles int literals`` () =
    let result = runNanopass LiteralWitness.nanopass graph coeffects
    // Assert MLIR contains constant ops
```

---

## Next Steps

### Immediate

1. ✅ **Architecture implemented** (NanopassArchitecture.fs, ParallelNanopass.fs)
2. ✅ **WitnessOutput.skip added** (for selective witnessing)
3. ⬜ **Create WitnessRegistry.fs** (populate global registry)
4. ⬜ **Update witnesses** to export `nanopass` value
5. ⬜ **Update MLIRTransfer.fs** to use `executeNanopasses`

### Short-term

1. ⬜ **Migrate all witnesses** to nanopass pattern
2. ⬜ **Test parallel execution** (compare to sequential)
3. ⬜ **Measure performance** (actual speedup)

### Medium-term

1. ⬜ **Add tiering** (if dependencies emerge between nanopasses)
2. ⬜ **Optimize overlay** (if merge is bottleneck)
3. ⬜ **Add instrumentation** (timing per nanopass)

---

## Summary

**Simple starting point**: One witness = one nanopass

**Parallel execution**: IcedTasks cold tasks

**Result collection**: Envelope pass overlays accumulators

**Scalability**: Dozens of nanopasses run naturally in parallel

**Next**: Create WitnessRegistry and migrate witnesses.

---

**The standing art composes up. Parallel nanopasses scale naturally.**
