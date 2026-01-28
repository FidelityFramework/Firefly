# Parallel Zipper Architecture for Alex MLIR Transfer

**Date**: January 27, 2026
**Status**: Design Phase - Key Architectural Finding
**Context**: Alex XParsec Remediation

---

## The Insight

**Multiple zippers CAN operate on the same immutable graph simultaneously**, enabling parallel compilation passes with associative accumulation. This is proven in production compilers (Chez Scheme, Triton-CPU) and theorized in the nanopass framework.

## The Pattern: Fan-Out → Fold-In

```
        Entry Point
             ↓
    Create Initial Zipper
             ↓
         Fan-Out
      ╱      |      ╲
   Z₁        Z₂       Z₃   (Independent zippers)
     ↓        ↓        ↓
   Acc₁     Acc₂     Acc₃  (Independent accumulators)
      ╲      |      ╱
         Fold-In
             ↓
    Merge Accumulators (associative)
             ↓
      Unified MLIR Output
```

## Key Architectural Properties

### 1. Immutability Enables Parallelism

**The Graph is immutable** - all zippers read the same SemanticGraph structure:
```fsharp
type PSGZipper = {
    Focus: SemanticNode        // Current position (varies per zipper)
    Path: ZipperPath          // Breadcrumbs (varies per zipper)
    Graph: SemanticGraph      // SHARED read-only reference
}
```

**Multiple zippers = Multiple lenses on ONE graph**. No data races, no coordination needed during traversal.

### 2. Zippers Navigate Structurally, Query Semantically

**Critical distinction** from research:

- **Structural edges**: Parent-child (what zippers follow via up/down/left/right)
- **Semantic edges**: Def-use, call graph, control flow (accessed via graph lookups)

**Huet zippers are STRUCTURAL navigators** - they cannot natively follow semantic edges. The hybrid pattern:

```fsharp
// Structural navigation (zipper operations)
match PSGZipper.down 0 zipper with
| Some childZipper -> ... // Navigate to child

// Semantic navigation (graph lookup + re-focus)
let targetNode = followDefUseEdge currentNode graph
match PSGZipper.focusOn targetNode.Id zipper with
| Some refocused -> ... // Jump to semantic target
```

**`focusOn` clears the path** - you lose structural breadcrumbs when jumping semantically. This is by design: semantic jumps are not structural moves.

### 3. Witnesses Return Codata (Pull Model)

**No push-based emission**. Witnesses observe and return:

```fsharp
type WitnessOutput = {
    InlineOps: MLIROp list      // Operations for current scope
    TopLevelOps: MLIROp list    // Operations for module level
    Result: TransferResult      // Value/void/error
}
```

**The accumulator is WRITE-ONLY during traversal**. Witnesses don't query it to make decisions (that's a coeffect). They just append to it.

### 4. Associative Merge Enables Order Independence

**Critical property**: If accumulation is associative, order doesn't matter:

```fsharp
// These produce same result:
merge (merge acc1 acc2) acc3
merge acc1 (merge acc2 acc3)

// Enables parallel execution:
[acc1; acc2; acc3] |> List.reduce merge  // Sequential
async { ... } |> Async.Parallel         // Parallel
```

**For MLIR generation**:
- Function definitions: Order independent (can emit in any order)
- Block ordering: Must respect dominance (dependencies tracked via coeffects)
- SSA assignments: Pre-computed (coeffect), not generated during traversal

## The Question: How to Maximize Parallelism?

### Current Architecture (Sequential)

```fsharp
let rec visitNode (ctx: WitnessContext) (nodeId: NodeId) (acc: MLIRAccumulator) =
    // Focus on node
    // Call witness
    // Accumulate result
    // Recursively visit children/dependencies (sequential)
```

**Single zipper** threads through the entire graph. Works, but sequential.

### Potential Parallel Architecture (Fan-Out/Fold-In)

**Key insight**: If subtrees are independent (no data dependencies), we can fan out:

```fsharp
// Phase 1: Identify independent subtrees (coeffect analysis)
type CompilationUnit = {
    RootNode: NodeId
    Dependencies: Set<NodeId>  // What this unit needs
    Provides: Set<NodeId>      // What this unit produces
}

// Phase 2: Build dependency graph of units
let unitGraph = buildUnitDependencyGraph graph coeffects

// Phase 3: Topological sort to find parallelizable batches
let batches = toposort unitGraph

// Phase 4: Process each batch in parallel
for batch in batches do
    let results =
        batch
        |> Array.map (fun unit ->
            async {
                let zipper = PSGZipper.create graph unit.RootNode
                let acc = MLIRAccumulator.empty()
                let ctx = { Coeffects = coeffects; Accumulator = acc; Graph = graph; Zipper = zipper }
                return visitNode ctx unit.RootNode acc
            })
        |> Async.Parallel
        |> Async.RunSynchronously

    // Fold-in: Merge accumulators associatively
    let batchAcc = results |> Array.fold mergeAccumulators (MLIRAccumulator.empty())
    globalAcc <- mergeAccumulators globalAcc batchAcc
```

## Constraints and Design Decisions Needed

### 1. What Constitutes a "Unit"?

**Options:**
- **Function-level**: Each function is a unit (natural for F#)
- **Module-level**: Each module is a unit (coarser granularity)
- **Expression-level**: Each independent expression tree (finest granularity)

**Trade-off**: Finer = more parallelism, but more overhead in merge.

### 2. How to Handle Dependencies?

**The SSA coeffect already tracks dependencies** - use it:

```fsharp
// Node 42 depends on nodes that define its argument SSAs
let deps = SSAAssignment.getDependencies nodeId coeffects.SSA
```

**Build dependency DAG** → Toposort → Find independent batches.

### 3. What Operations Are Associative?

**Safe for parallel merge:**
- ✅ Function definitions (order independent in MLIR module)
- ✅ Global constants (order independent)
- ✅ Type definitions (order independent)

**Require ordering:**
- ❌ Basic blocks within function (dominance)
- ❌ Operations within block (SSA def before use)

**Solution**: Parallelize at function/module level, keep intra-function sequential.

### 4. Memory Pressure vs Speed

**Each zipper + accumulator = memory allocation**. For small programs, overhead may exceed benefit.

**Heuristic needed**: Only parallelize if graph has > N independent units (where N is tuned).

## Prior Art to Study

### 1. Nanopass Framework (Scheme)

**Location**: `~/repos/nanopass-framework-scheme/`

**Research questions**:
- How does the framework identify parallelizable passes?
- What's the granularity of parallelism (whole-program vs subtree)?
- How are results merged?
- See `doc/user-guide.pdf` for architecture

### 2. Triton-CPU (MLIR)

**Location**: `~/triton-cpu/`

**Research questions**:
- How does Triton parallelize dialect transformations?
- What's the PassManager coordination strategy?
- How are immutable IR copies managed?
- Look at pass pipeline implementation

### 3. Chez Scheme Compiler

**Research questions** (web search needed):
- How does it parallelize its ~30 nanopass transformations?
- What dependency tracking exists?
- How does it merge transformed IRs?

## Next Context Window Research Plan

### Phase 1: Nanopass Framework Deep Dive (2-3 hours)

**Tasks:**
1. Read `nanopass-framework-scheme/doc/user-guide.pdf` completely
2. Examine `define-pass` macro implementation
3. Understand catamorphism generation (auto-parallelizable?)
4. Document: What parallelization opportunities exist?

### Phase 2: Triton-CPU Pass Manager Study (2-3 hours)

**Tasks:**
1. Locate PassManager implementation in `~/triton-cpu/`
2. Trace how passes are scheduled and coordinated
3. Identify immutability guarantees
4. Document: How does production MLIR compiler parallelize?

### Phase 3: Design Proposal (2-3 hours)

**Deliverable**: Concrete proposal for Alex parallel zipper architecture
- What granularity? (function, module, expression)
- What dependency tracking? (reuse SSA coeffect? new analysis?)
- What merge strategy? (batch-wise? stream-based?)
- What heuristics? (when to parallelize vs stay sequential)
- What testing strategy? (small samples should work with N=1 zipper)

### Phase 4: Prototype (Optional, if time permits)

**Goal**: Implement minimal fan-out/fold-in for HelloWorld samples
- Measure: Does it actually speed up compilation?
- Validate: Do results match sequential version?
- Learn: What problems arise in practice?

## Open Questions

1. **Should `visitNode` even take a `nodeId` parameter?** Or should it only witness `ctx.Zipper.Focus`?
   - Current: `visitNode ctx nodeId` (hybrid - uses focusOn to jump)
   - Alternative: `visitNode ctx` (pure zipper - only witness focus)

2. **Is `focusOn` the right operation for semantic jumps?** It clears the path (loses context).
   - Maybe: Separate `jumpTo` that preserves path metadata?
   - Or: Accept that semantic jumps lose structural context?

3. **Can we avoid re-creating zippers via shadowing?**
   - Current: `let ctx = { ctx with Zipper = newZipper }` (creates new record)
   - Can we mutate zipper in place? (NO - immutability required for correctness)

4. **What's the right merge algorithm?** Simple append? Tree-based reduction? Streaming?

## References

### Research Conducted (January 27, 2026)

Three parallel agents explored:
1. **Huet Zipper and Graph Navigation** (agent a7a4101)
2. **Parallel Compilation with Zippers** (agent a9eb9af)
3. **Semantic Edge Traversal Patterns** (agent a18f1b7)

Key findings documented inline above.

### Key Papers

- Huet, G. (1997). "The Zipper" - Original zipper paper
- Ramsey, N. & Dias, J. (2006). "An Applicative Control-Flow Graph Based on Huet's Zipper"
- Sarkar, D., Waddell, O., Dybvig, R. K. "Nanopass Framework" papers
- McBride, C. (2001). "The Derivative of a Regular Type is its Type of One-Hole Contexts"

### Local Resources

- `/home/hhh/repos/nanopass-framework-scheme/` - Nanopass implementation
- `/home/hhh/repos/triton-cpu/` - Production MLIR compiler with parallelization
- `/home/hhh/repos/Firefly/docs/PSG_Nanopass_Architecture.md` - PSG nanopass design
- `/home/hhh/repos/SpeakEZ/hugo/content/blog/Learning to Walk.md` - Zipper philosophy

---

## Summary

We've discovered that **parallel zipper passes are architecturally viable** for Alex. The immutable PSG + codata witnesses + associative accumulation = natural parallelism opportunity.

**But we need intentional design choices** to maximize benefits:
- What granularity?
- What dependency tracking?
- What merge strategy?
- When to parallelize vs stay sequential?

**Next steps**: Deep study of nanopass framework and Triton-CPU, then design proposal.

**Current state**: MLIRTransfer.fs updated to shadow context (no `ctx'` pollution), but NO further implementation until design is complete.
