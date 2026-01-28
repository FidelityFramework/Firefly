# Context Window Handoff: Parallel Zipper Architecture Research

**Date**: January 27, 2026
**Previous Context**: Alex XParsec remediation - discovered parallel zipper architecture opportunity
**Next Context Goal**: Deep research on nanopass + Triton to design fan-out/fold-in correctly

---

## What Was Discovered

**Multiple zippers can operate on the same immutable graph simultaneously**, enabling parallel compilation. This is proven in:
- Chez Scheme (commercial compiler with ~30 nanopasses)
- Triton-CPU (production MLIR compiler)
- Nanopass framework (systematic compiler construction)

**The pattern**: Fan-out → Independent Zippers → Fold-in via Associative Merge

---

## The Key Insight (User's Hypothesis)

> "I'm imagining the nanopass scheme example repository will show us that each nanopass *is* a zipper traversal with associative hooks to re-accumulate into a final folded graph."

**This reframes the parallelism question:**

Not just: "Can we parallelize WITHIN a pass?" (fan-out subtrees in one traversal)
But also: "Can we parallelize ACROSS passes?" (run independent passes simultaneously)

### The Model

```
Pass 1 (Zipper Z₁)    Pass 2 (Zipper Z₂)    Pass 3 (Zipper Z₃)
      ↓                     ↓                     ↓
   Graph₁ ←───────────→ Graph₂ ←───────────→ Graph₃

If Pass 1 and Pass 2 are independent:

   ┌─→ Pass 1 (Z₁) → Graph₁ ─┐
   │                           ├─→ Merge → Graph'
   └─→ Pass 2 (Z₂) → Graph₂ ─┘
```

**Each nanopass is a catamorphism** (structure-preserving fold). If passes are independent (no data dependencies), they can run in parallel and merge via associative composition.

---

## Research Questions for Next Context

### Primary Question: How Does Nanopass Enable Parallelism?

**Location**: `~/repos/nanopass-framework-scheme/`
**Document**: `doc/user-guide.pdf`

1. **Is each pass a zipper traversal?**
   - How does `define-pass` generate traversal code?
   - Are catamorphisms explicitly zipper-based?
   - Can we see the traversal + accumulation pattern?

2. **How are passes composed?**
   - Sequential by default?
   - Dependency declarations between passes?
   - Any parallelization hooks?

3. **What's the merge/accumulation strategy?**
   - Does each pass produce a new IR?
   - How are transformed IRs combined?
   - Is composition associative?

4. **Can passes run in parallel?**
   - What prevents parallel execution?
   - What enables it?
   - Are there examples of parallel pass execution?

### Secondary Question: How Does Triton-CPU Parallelize?

**Location**: `~/triton-cpu/`

1. **What's the PassManager architecture?**
   - How are passes scheduled?
   - What dependency tracking exists?
   - How are immutable IR copies managed?

2. **When do passes run in parallel?**
   - Across modules? Within module?
   - What granularity?

3. **How are results merged?**
   - Batch-wise after parallel execution?
   - Streaming as passes complete?

### Tertiary Question: Chez Scheme Design

**Research needed** (web search + papers):

1. How does Chez Scheme's nanopass compiler work?
2. ~30 passes - are any parallelized?
3. What's the coordination strategy?

---

## Specific Files to Examine

### Nanopass Framework
- `~/repos/nanopass-framework-scheme/doc/user-guide.pdf` - Complete read (PRIORITY)
- `~/repos/nanopass-framework-scheme/nanopass/*.ss` - Implementation
- Look for: `define-pass`, catamorphism generation, traversal patterns

### Triton-CPU
- Find `PassManager` or equivalent
- Find pass registration/scheduling code
- Look for dependency declarations between passes
- Check for parallel execution infrastructure

---

## Architectural Constraints from Current Firefly

### What We Have (Correct)
- ✅ Immutable SemanticGraph
- ✅ Pure navigational PSGZipper
- ✅ Codata-returning witnesses
- ✅ Pre-computed coeffects (SSA, platform, etc.)
- ✅ Associative accumulation (MLIRAccumulator)

### What We Need to Design
- ❓ Granularity: Function-level? Module-level? Expression-level?
- ❓ Dependency tracking: How to identify independent passes/subtrees?
- ❓ Merge strategy: Batch? Tree-reduction? Streaming?
- ❓ Heuristics: When to parallelize vs stay sequential?

### What We Must Preserve
- **Single context variable** (no `ctx'` pollution) - shadow only
- **Structural vs semantic navigation** - zippers are structural, semantics via graph lookups
- **`focusOn` clears path** - semantic jumps lose structural breadcrumbs (by design)

---

## The Current Code State (DO NOT MODIFY)

### Changed Files (January 27, 2026)
- `TransferTypes.fs` - Added `Zipper: PSGZipper` to `WitnessContext`
- `MLIRTransfer.fs` - Updated to shadow `ctx` with focused zipper (no `ctx'`)
- `LazyWitness.fs` - Uses `ctx.Zipper` from context
- `ArithWitness.fs` - Uses `ctx.Zipper` from context
- `PlatformWitness.fs` - **DELETED** (architectural violation - had `freshSSA`)

### Build Status
**BROKEN** - .NET SDK version mismatch (net10.0 vs available 8.0)
**DO NOT FIX** - We're in design mode, not implementation

### Line Count (Before Remediation)
- MLIRTransfer.fs: 296 lines (under 300 limit ✅)
- Total witnesses: ~5,773 lines (target: ~600)

---

## Success Criteria for Next Context

### Research Phase Outputs

1. **Nanopass Analysis Document** (~2-3 pages)
   - How passes are defined
   - How traversal works (zipper-based?)
   - How composition works
   - Parallelization opportunities

2. **Triton Analysis Document** (~2-3 pages)
   - PassManager architecture
   - Dependency tracking
   - Parallel execution strategy
   - Merge/coordination mechanisms

3. **Design Proposal** (~5-10 pages)
   - Concrete parallel zipper architecture for Alex
   - Answers to all design questions (granularity, dependency, merge, heuristics)
   - Testing strategy
   - Implementation roadmap

### Optional: Prototype

If time permits, implement minimal fan-out/fold-in on HelloWorld samples to validate design.

---

## Key Documentation References

### Already Created (This Context)
- `/home/hhh/repos/Firefly/docs/Parallel_Zipper_Architecture.md` - Full architectural analysis
- Serena memory: `parallel_zipper_fan_out_fold_in` - Core findings
- This handoff document

### Existing (Prior Contexts)
- `/home/hhh/repos/Firefly/docs/PSG_Nanopass_Architecture.md` - PSG nanopass design
- `/home/hhh/repos/SpeakEZ/hugo/content/blog/Learning to Walk.md` - Zipper philosophy
- Serena memory: `mlir_transfer_canonical_architecture` - Transfer layer design
- Serena memory: `alex_xparsec_throughout_architecture` - XParsec usage

### To Read (Next Context)
- `~/repos/nanopass-framework-scheme/doc/user-guide.pdf` - **PRIORITY 1**
- Triton-CPU source code (find PassManager) - **PRIORITY 2**
- Research papers on parallel compilation (web search) - **PRIORITY 3**

---

## Critical Reminders

1. **NO COMPILER CHURN** - Stay in design mode
2. **Research before implementing** - Understand prior art deeply
3. **Intentional design choices** - Don't cargo-cult parallelism
4. **Test with small samples** - HelloWorld should work with N=1 sequential baseline
5. **Associativity is non-negotiable** - Merge must be order-independent for correctness

---

## The Vision

**Each nanopass = A zipper traversal with associative fold**

If this is true (verify in nanopass framework):
- We can run independent passes in parallel
- We can fan-out within a pass for independent subtrees
- We can fold-in results associatively
- We get natural parallelism at multiple levels

**This is the key to maximizing compiler performance while preserving correctness.**

---

**Next Action**: Spawn agents to deeply research nanopass framework and Triton-CPU, then synthesize findings into design proposal.
