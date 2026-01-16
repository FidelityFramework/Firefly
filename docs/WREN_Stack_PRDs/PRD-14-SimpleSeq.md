# PRD-14: Sequence Expressions

> **Sample**: `14_SimpleSeq` | **Status**: Planned | **Depends On**: PRD-11-13 (Closures, HOFs, Recursion)

## 1. Executive Summary

Sequence expressions (`seq { }`) produce lazy, on-demand enumeration of values. Unlike lists, sequences don't compute all values upfront - they generate values one at a time as requested. This is the first **codata** feature in the progression.

**Key Insight**: Sequences are state machines with `MoveNext`/`Current` interface. The `yield` points become state indices, and the sequence struct holds captured variables plus current state.

**Reference**: The StateMachine strategy from `async_llvm_coroutines` memory applies here - sequences use the same pattern without LLVM coroutine intrinsics.

## 2. Language Feature Specification

### 2.1 Basic Sequence Expression

```fsharp
let numbers = seq {
    yield 1
    yield 2
    yield 3
}
```

Produces values 1, 2, 3 on demand.

### 2.2 Sequence with Computation

```fsharp
let squares = seq {
    for i in 1..5 do
        yield i * i
}
```

Generates 1, 4, 9, 16, 25.

### 2.3 Infinite Sequences

```fsharp
let naturals = seq {
    let mutable n = 0
    while true do
        n <- n + 1
        yield n
}
```

Infinite stream of natural numbers.

### 2.4 Sequence Consumption

```fsharp
for x in numbers do
    Console.writeln (Format.int x)
```

The `for...in` construct drives the state machine.

## 3. FNCS Layer Implementation

### 3.1 SemanticKind.SeqExpr

```fsharp
type SemanticKind =
    | SeqExpr of body: NodeId * yieldPoints: int list
```

The `yieldPoints` are indices assigned during checking - each `yield` gets a unique state number.

### 3.2 SemanticKind.Yield

```fsharp
type SemanticKind =
    | Yield of value: NodeId * stateIndex: int
```

Each yield has a state index that the state machine returns to after resuming.

### 3.3 Seq Intrinsic Type

```fsharp
// In NativeTypes.fs
| TSeq of elementType: NativeType

// In CheckExpressions.fs - seq is a type constructor
| "seq" -> fun elemTy -> NativeType.TSeq(elemTy)
```

### 3.4 Capture Analysis for Seq

Sequences capture variables just like closures. Reuse the capture analysis from PRD-11:

```fsharp
let checkSeqExpr env builder seqBody =
    // 1. Check body, collecting yield points
    let (bodyNode, yieldPoints) = checkSeqBody env builder seqBody

    // 2. Collect captures (reuse closure logic)
    let captures = collectCaptures env builder bodyNode.Id

    // 3. Create SeqExpr node
    builder.Create(
        SemanticKind.SeqExpr(bodyNode.Id, yieldPoints, captures),
        NativeType.TSeq(yieldElementType),
        range)
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Sequence State Machine Structure

```fsharp
type SeqStateMachine<'T> = {
    State: int              // Current state (-1 = done, 0 = start, N = after yield N)
    Current: 'T             // Current value (valid after MoveNext returns true)
    // ...captured variables...
}
```

### 4.2 State Index Assignment (Nanopass)

**File**: `src/Alex/Preprocessing/YieldStateIndices.fs`

Before SSA assignment, number all yield points:

```fsharp
type YieldStateCoeffect = {
    SeqExprId: NodeId
    YieldIndices: Map<NodeId, int>  // Yield NodeId -> state index
}

let run (graph: SemanticGraph) =
    // Walk each SeqExpr, assign indices to yields
```

### 4.3 MoveNext Function Generation

For each sequence expression, generate a `MoveNext` function:

```fsharp
let emitSeqMoveNext z seqExprId =
    // Emit state machine switch
    emit "switch %state {"
    emit "  0: goto ^init"
    for (yieldId, stateIdx) in yieldIndices do
        emit $"  {stateIdx}: goto ^after_yield_{stateIdx}"
    emit "  default: return false"
    emit "}"

    // Emit state machine body with yield points
    emitSeqBody z seqExprId
```

### 4.4 Yield Emission

At each yield point:
1. Store the yielded value in `Current`
2. Store the next state index
3. Return `true`

```fsharp
let emitYield z yieldNodeId stateIdx valueSSA =
    // Store current value
    emit $"  %%curr_ptr = llvm.getelementptr %%self[0, 1]"
    emit $"  llvm.store %%{valueSSA}, %%curr_ptr"

    // Store next state
    emit $"  %%state_ptr = llvm.getelementptr %%self[0, 0]"
    emit $"  llvm.store {stateIdx + 1}, %%state_ptr"

    // Return true (has value)
    emit "  llvm.return %true"
```

## 5. MLIR Output Specification

### 5.1 Sequence Struct Type

```mlir
// seq { yield 1; yield 2; yield 3 }
!seq_state = !llvm.struct<(
    i32,     // state: current state index
    i32,     // current: current yielded value
    // no captures in this example
)>
```

### 5.2 MoveNext Function

```mlir
llvm.func @seq_moveNext(%self: !llvm.ptr) -> i1 {
    %state_ptr = llvm.getelementptr %self[0, 0]
    %state = llvm.load %state_ptr : i32

    llvm.switch %state : i32 [
        0: ^state0,
        1: ^state1,
        2: ^state2,
        3: ^state3
    ], ^done

^state0:  // Initial state
    %curr_ptr = llvm.getelementptr %self[0, 1]
    llvm.store %c1, %curr_ptr   // yield 1
    llvm.store %c1, %state_ptr  // next state = 1
    llvm.return %true : i1

^state1:
    llvm.store %c2, %curr_ptr   // yield 2
    llvm.store %c2, %state_ptr  // next state = 2
    llvm.return %true : i1

^state2:
    llvm.store %c3, %curr_ptr   // yield 3
    llvm.store %c3, %state_ptr  // next state = 3
    llvm.return %true : i1

^state3:  // After last yield
    llvm.store %-1, %state_ptr  // done
    llvm.return %false : i1

^done:
    llvm.return %false : i1
}
```

### 5.3 Sequence Creation

```mlir
// let numbers = seq { yield 1; yield 2; yield 3 }
%seq = llvm.alloca 1 x !seq_state
%state_ptr = llvm.getelementptr %seq[0, 0]
llvm.store %c0, %state_ptr  // Initial state = 0
```

### 5.4 For Loop Consumption

```mlir
// for x in numbers do Console.writeln (Format.int x)
llvm.br ^loop_check

^loop_check:
    %has_next = llvm.call @seq_moveNext(%seq) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^loop_body, ^loop_end

^loop_body:
    %curr_ptr = llvm.getelementptr %seq[0, 1]
    %x = llvm.load %curr_ptr : i32
    // ... use x ...
    llvm.br ^loop_check

^loop_end:
    // continue after loop
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module SimpleSeqSample

let simpleSeq = seq {
    yield 1
    yield 2
    yield 3
}

let rangeSeq (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Simple Seq Test ==="

    Console.writeln "--- Basic Seq ---"
    for x in simpleSeq do
        Console.writeln (Format.int x)

    Console.writeln "--- Range Seq ---"
    for x in rangeSeq 5 10 do
        Console.writeln (Format.int x)

    0
```

### 6.2 Expected Output

```
=== Simple Seq Test ===
--- Basic Seq ---
1
2
3
--- Range Seq ---
5
6
7
8
9
10
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `SemanticGraph.fs` | MODIFY | Add SeqExpr, Yield SemanticKinds |
| `NativeTypes.fs` | MODIFY | Add TSeq type constructor |
| `Expressions/Coordinator.fs` | MODIFY | Handle seq { } expressions |
| `Expressions/Computations.fs` | CREATE | Sequence expression checking |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/YieldStateIndices.fs` | CREATE | Assign state indices to yields |
| `src/Alex/Witnesses/SeqWitness.fs` | CREATE | Emit state machine MLIR |
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle SeqExpr, Yield, for-in |
| `src/Firefly.fsproj` | MODIFY | Add new files |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TSeq to NativeTypes
- [ ] Add SeqExpr, Yield to SemanticKind
- [ ] Implement seq { } expression checking
- [ ] Implement yield point collection

### Phase 2: Alex State Machine
- [ ] Create YieldStateIndices nanopass
- [ ] Implement SeqWitness for state machine emission
- [ ] Implement yield emission
- [ ] Implement for-in loop consumption

### Phase 3: Validation
- [ ] Sample 14 compiles without errors
- [ ] Sample 14 produces correct output
- [ ] Samples 01-13 still pass

## 9. Academic References

- "Lazy Evaluation" - Hughes (1989)
- "Codata in ML" - Abel, Pientka (2013)
- MLKit sequence implementation

## 10. Related PRDs

- **PRD-11**: Closures - Sequences capture variables
- **PRD-15**: Seq Operations - `Seq.map`, `Seq.filter`, etc.
- **PRD-16**: Lazy - Similar thunk/force pattern
