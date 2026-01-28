# Witness Migration Guide - Parallel Nanopasses

**Date**: January 27, 2026
**Status**: Implementation Complete - Ready for Migration
**Next**: Migrate all witnesses to nanopass pattern

---

## Quick Start: Migrating a Witness

### Step 1: Add Category-Selective Witness Function

**Before** (Old pattern - direct function):
```fsharp
// ArithWitness.fs - OLD
let witnessBinaryArith (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Pattern match failed"
```

**After** (New pattern - category-selective):
```fsharp
// ArithWitness.fs - NEW
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    // Handle all arithmetic node types
    | SemanticKind.BinaryOp _ ->
        match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Binary arith pattern match failed"

    | SemanticKind.UnaryOp _ ->
        match tryMatch pUnary ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Unary operation pattern match failed"

    | SemanticKind.Comparison _ ->
        match tryMatch pComparison ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Comparison pattern match failed"

    // Skip all other nodes (some other nanopass handles them)
    | _ -> WitnessOutput.skip
```

### Step 2: Export Nanopass Value

Add at end of witness file:
```fsharp
/// Nanopass registration
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
```

### Step 3: Add Required Opens

At top of witness file:
```fsharp
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.TransferTypes
```

### Step 4: Test in Isolation

```fsharp
// In test file
[<Test>]
let ``ArithNanopass handles binary ops`` () =
    let graph = loadTestGraph "simple_arith.psg"
    let coeffects = computeCoeffects graph

    let result = runNanopass ArithWitness.nanopass graph coeffects

    // Assert MLIR contains expected ops
    Assert.Contains("arith.addi", result.TopLevelOps |> mlirToString)
```

---

## Witness Migration Checklist

### Phase 1: Create WitnessRegistry.fs

**File**: `src/MiddleEnd/Alex/Traversal/WitnessRegistry.fs`

```fsharp
/// WitnessRegistry - Global registry of all witness nanopasses
module Alex.Traversal.WitnessRegistry

open Alex.Traversal.NanopassArchitecture

// Import all witnesses
module LiteralWitness = Alex.Witnesses.LiteralWitness
module ArithWitness = Alex.Witnesses.ArithWitness
// ... etc

/// Global registry (populated as witnesses are migrated)
let mutable globalRegistry = NanopassRegistry.empty

// Temporary: Register only migrated witnesses
// Will expand as migration progresses
let initializeRegistry () =
    globalRegistry <-
        globalRegistry
        // |> NanopassRegistry.register LiteralWitness.nanopass  // Uncomment when migrated
        // |> NanopassRegistry.register ArithWitness.nanopass     // Uncomment when migrated
```

**Add to Firefly.fsproj** (after ParallelNanopass.fs):
```xml
<Compile Include="MiddleEnd/Alex/Traversal/WitnessRegistry.fs" />
```

---

### Phase 2: Migrate Witnesses (Priority Order)

#### Priority 1: Simple Witnesses (Start Here)

1. ✅ **LiteralWitness.fs** (Already uses nanopass pattern for Lazy)
   - [x] Add category-selective function
   - [x] Export `nanopass` value
   - [x] Register in WitnessRegistry
   - [x] Test

2. ⬜ **ArithWitness.fs** (Already simplified to ~35 lines)
   - [ ] Add category-selective `witnessArithmetic`
   - [ ] Export `nanopass` value
   - [ ] Register in WitnessRegistry
   - [ ] Test

#### Priority 2: Collection Witnesses

3. ⬜ **OptionWitness.fs** (~257 lines)
   - [ ] Add `witnessOption` (handles Some, None, bind, map, etc.)
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

4. ⬜ **ListWitness.fs** (~190 lines)
   - [ ] Add `witnessList` (handles cons, head, tail, map, etc.)
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

5. ⬜ **MapWitness.fs** (~183 lines)
   - [ ] Add `witnessMap`
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

6. ⬜ **SetWitness.fs** (~191 lines)
   - [ ] Add `witnessSet`
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

#### Priority 3: Control Flow Witnesses

7. ⬜ **ControlFlowWitness.fs** (~487 lines)
   - [ ] Add `witnessControlFlow` (IfThenElse, While, For, Match, etc.)
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

#### Priority 4: Memory & Lambda

8. ⬜ **MemoryWitness.fs** (~878 lines)
   - [ ] Add `witnessMemory` (Alloca, Load, Store, GEP, etc.)
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

9. ⬜ **LambdaWitness.fs** (~556 lines)
   - [ ] Add `witnessLambda` (Lambda, Application, Closure, etc.)
   - [ ] Export `nanopass`
   - [ ] Register
   - [ ] Test

#### Priority 5: Advanced Features

10. ⬜ **LazyWitness.fs** (Already ~38 lines)
    - [ ] Add category-selective wrapper
    - [ ] Export `nanopass`
    - [ ] Register
    - [ ] Test

11. ⬜ **SeqWitness.fs** (~1,021 lines)
    - [ ] Add `witnessSeq` (SeqExpr, Yield, YieldFrom, etc.)
    - [ ] Export `nanopass`
    - [ ] Register
    - [ ] Test

12. ⬜ **SeqOpWitness.fs** (~549 lines)
    - [ ] Add `witnessSeqOp` (map, filter, collect, etc.)
    - [ ] Export `nanopass`
    - [ ] Register
    - [ ] Test

---

## Migration Template

### Template for Each Witness

```fsharp
// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

let private witness[Category] (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with

    // Handle all node types this category covers
    | SemanticKind.[Type1] ->
        // ... existing witnessing logic
        ...

    | SemanticKind.[Type2] ->
        // ... existing witnessing logic
        ...

    // Skip all others
    | _ -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// [Category] nanopass - witnesses [brief description]
let nanopass : Nanopass = {
    Name = "[CategoryName]"
    Witness = witness[Category]
}
```

---

## Testing Strategy

### Per-Witness Tests

```fsharp
module ArithWitnessTests

open NUnit.Framework
open Alex.Traversal.NanopassArchitecture
open Alex.Witnesses.ArithWitness

[<Test>]
let ``ArithNanopass witnesses binary ops`` () =
    let graph = TestGraphs.simpleBinaryOp  // a + b
    let coeffects = TestCoeffects.default

    let result = runNanopass nanopass graph coeffects

    // Assert MLIR contains AddI operation
    Assert.IsNotEmpty(result.TopLevelOps)
    Assert.Contains("arith.addi", mlirToString result.TopLevelOps)

[<Test>]
let ``ArithNanopass skips non-arithmetic nodes`` () =
    let graph = TestGraphs.literalOnly  // Just "42"
    let coeffects = TestCoeffects.default

    let result = runNanopass nanopass graph coeffects

    // Should be empty (no arithmetic operations)
    Assert.IsEmpty(result.TopLevelOps)
```

### Integration Tests

```fsharp
[<Test>]
let ``Parallel execution produces same result as sequential`` () =
    let graph = loadGraph "samples/01_HelloWorldDirect/psg.json"
    let coeffects = computeCoeffects graph

    let seqResult = executeNanopasses { EnableParallel = false } globalRegistry graph coeffects
    let parResult = executeNanopasses { EnableParallel = true } globalRegistry graph coeffects

    Assert.AreEqual(seqResult.TopLevelOps, parResult.TopLevelOps)
```

---

## MLIRTransfer.fs Integration

### Current (Old pattern - sequential dispatch)

```fsharp
let rec visitNode ctx nodeId acc =
    match PSGZipper.focusOn nodeId ctx.Zipper with
    | None -> WitnessOutput.error "Node not found"
    | Some zipper ->
        let ctx = { ctx with Zipper = zipper }
        let node = PSGZipper.focus zipper

        match node.Kind with
        | SemanticKind.Literal _ -> LiteralWitness.witness ctx node
        | SemanticKind.BinaryOp _ -> ArithWitness.witnessBinaryArith ctx node
        | SemanticKind.Lambda _ -> LambdaWitness.witnessLambda ctx node
        // ... 30+ more cases
```

### New (Parallel nanopasses - no dispatch)

```fsharp
open Alex.Traversal.WitnessRegistry
open Alex.Traversal.ParallelNanopass

let transferPSG (graph: SemanticGraph) (coeffects: TransferCoeffects) : Result<MLIROp list * MLIROp list, string> =
    // Initialize registry (if not already done)
    initializeRegistry()

    // Execute all nanopasses in parallel
    let config = { EnableParallel = true }  // Set to false for debugging
    let accumulator = executeNanopasses config globalRegistry graph coeffects

    // Extract results
    match accumulator.Errors with
    | [] -> Ok (List.rev accumulator.TopLevelOps, [])
    | errors -> Error (String.concat "; " errors)
```

**NO DISPATCH NEEDED** - Each nanopass handles its own category.

---

## Common Patterns

### Pattern 1: Single Node Type

```fsharp
// LiteralWitness - handles only Literal nodes
let private witnessLiteral (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Literal lit ->
        witnessLiteralImpl ctx node lit
    | _ ->
        WitnessOutput.skip
```

### Pattern 2: Multiple Related Node Types

```fsharp
// ArithWitness - handles BinaryOp, UnaryOp, Comparison
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.BinaryOp _ -> witnessBinaryArith ctx node
    | SemanticKind.UnaryOp _ -> witnessUnary ctx node
    | SemanticKind.Comparison _ -> witnessComparison ctx node
    | _ -> WitnessOutput.skip
```

### Pattern 3: Complex Category

```fsharp
// ControlFlowWitness - many control flow constructs
let private witnessControlFlow (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.IfThenElse _ -> witnessIfThenElse ctx node
    | SemanticKind.While _ -> witnessWhile ctx node
    | SemanticKind.For _ -> witnessFor ctx node
    | SemanticKind.Match _ -> witnessMatch ctx node
    | SemanticKind.Try _ -> witnessTry ctx node
    | _ -> WitnessOutput.skip
```

### Pattern 4: Full-Fat Execution (Why Skip is Essential)

**DESIGN DECISION**: ALL registered nanopasses run in parallel, even if some will skip every node.

```fsharp
// For HelloWorld (only Literal and simple arithmetic nodes):
// - LiteralNanopass: Witnesses ~5 nodes
// - ArithNanopass: Witnesses ~3 nodes
// - OptionNanopass: Traverses ALL nodes, returns skip for ALL
// - ListNanopass: Traverses ALL nodes, returns skip for ALL
// - MapNanopass: Traverses ALL nodes, returns skip for ALL
// ... etc (6 more empty nanopasses)
```

**Why this is correct**:
- Empty nanopasses are cheap (just pattern matching)
- Parallel execution: empty passes don't block useful work
- Simplicity: no coordination overhead
- For small programs, cost is microseconds

**Why `WitnessOutput.skip` must be used**:
```fsharp
// WRONG - Error on non-handled nodes
| _ -> WitnessOutput.error "Unsupported node"  // NO!

// RIGHT - Skip and let other nanopasses handle it
| _ -> WitnessOutput.skip  // YES - other nanopass will handle it
```

Each nanopass sees ALL nodes but witnesses only its category. The architecture relies on `skip` to avoid false errors.

**Future optimization**: For very large projects (10,000+ nodes), a discovery pass could filter nanopasses before execution. This is intentionally deferred until profiling shows need. See `Parallel_Nanopass_Final_Summary.md` for trade-off analysis.

---

## Debugging

### Sequential Execution for Debugging

```fsharp
// In MLIRTransfer.fs
let transferPSG graph coeffects =
    let config = {
        EnableParallel = false  // Sequential for debugging
    }
    executeNanopasses config globalRegistry graph coeffects
```

### Per-Nanopass Debugging

```fsharp
// Run single nanopass
let debugArithNanopass graph coeffects =
    runNanopass ArithWitness.nanopass graph coeffects
```

### Verbose Logging

```fsharp
// Add to nanopass witness function
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.BinaryOp _ ->
        printfn "[ArithNanopass] Witnessing BinaryOp at node %d" (NodeId.value node.Id)
        witnessBinaryArith ctx node
    | _ ->
        WitnessOutput.skip
```

---

## Success Criteria

### Per-Witness Migration

- ✅ Witness file exports `nanopass` value
- ✅ Category-selective function uses `WitnessOutput.skip`
- ✅ Registered in `WitnessRegistry.fs`
- ✅ Unit test passes
- ✅ Integration test passes (parallel = sequential)

### Complete Migration

- ✅ All 14 witnesses migrated
- ✅ All registered in `WitnessRegistry.fs`
- ✅ MLIRTransfer.fs updated (no dispatch logic)
- ✅ Regression tests pass (all samples compile)
- ✅ Parallel output = sequential output
- ✅ Performance measured (actual speedup)

---

## Next Steps

1. **Create WitnessRegistry.fs** (5-10 minutes)
2. **Migrate LiteralWitness.fs** (10-15 minutes) - DONE already?
3. **Migrate ArithWitness.fs** (10-15 minutes)
4. **Test first two nanopasses** (10 minutes)
5. **Migrate remaining witnesses** (1-2 per hour)
6. **Update MLIRTransfer.fs** (30 minutes)
7. **Run regression tests** (validation)
8. **Measure performance** (benchmarking)

---

**Status**: ✅ Architecture ready. Begin migration with LiteralWitness and ArithWitness.

**Estimated Time**: 6-8 hours for complete migration (14 witnesses)

**Expected Outcome**: Natural parallelism with dozens of nanopasses scaling automatically.
