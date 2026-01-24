# A-02: Async Await (let! and Suspension)

> **Sample**: `18_AsyncAwait` | **Status**: Planned | **Depends On**: A-01 (BasicAsync)

## 1. Executive Summary

This PRD adds `let!` - the ability to await other async computations. This is where LLVM coroutines become essential: each `let!` is a **suspension point** where the coroutine may pause and later resume.

**Key Insight**: `let!` compiles to `llvm.coro.suspend`. The CoroSplit pass transforms the function into a state machine that can pause at each suspension point and resume with the awaited value.

## 2. Language Feature Specification

### 2.1 let! (Async Bind)

```fsharp
let composed = async {
    let! x = async { return 10 }
    let! y = async { return 20 }
    return x + y
}
```

Each `let!` awaits the inner async before continuing.

### 2.2 do! (Async Ignore)

```fsharp
let withSideEffect = async {
    do! async { Console.writeln "Side effect" }
    return 42
}
```

Like `let!` but discards the result.

### 2.3 Suspension Semantics

At each `let!`:
1. Evaluate the inner async
2. If inner async is incomplete, suspend
3. When inner completes, resume with its result
4. Bind result to name and continue

For now (single-threaded), inner asyncs complete immediately, so suspension is technically immediate resumption. True suspension matters when combined with I/O (I-01/I-02) or threading (T-01/T-02).

## 3. FNCS Layer Implementation

### 3.1 SemanticKind.AsyncBind

```fsharp
type SemanticKind =
    | AsyncBind of
        name: string *
        inner: NodeId *        // The async being awaited
        continuation: NodeId * // Code after the let!
        suspensionIndex: int   // Unique index for this suspension point
```

### 3.2 SemanticKind.AsyncDo

```fsharp
| AsyncDo of
    inner: NodeId *
    continuation: NodeId *
    suspensionIndex: int
```

Same as AsyncBind but without binding a name.

### 3.3 Type Checking let!

```fsharp
let checkAsyncBind env builder name innerExpr contExpr =
    // 1. Check inner expression - must be Async<'a>
    let innerNode = checkExpr env builder innerExpr
    match innerNode.Type with
    | TAsync elemTy ->
        // 2. Bind name to 'a in continuation
        let envWithBinding = addBinding name elemTy env

        // 3. Check continuation
        let contNode = checkExpr envWithBinding builder contExpr

        // 4. Assign suspension index
        let suspIdx = nextSuspensionIndex ()

        builder.Create(
            SemanticKind.AsyncBind(name, innerNode.Id, contNode.Id, suspIdx),
            contNode.Type,  // Type is continuation's type
            range)
    | _ ->
        error "let! requires Async<_> on right-hand side"
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Suspension Point Numbering (Nanopass)

**File**: `src/Alex/Preprocessing/SuspensionIndices.fs`

Assign unique indices to all `let!` and `do!` points:

```fsharp
type SuspensionCoeffect = {
    AsyncExprId: NodeId
    SuspensionPoints: Map<NodeId, int>  // AsyncBind/Do NodeId -> index
}
```

### 4.2 Coroutine State Machine

With suspension points, the async becomes a true coroutine:

```fsharp
type AsyncFrame<'T> = {
    State: int          // Current state (0 = start, N = after suspension N)
    Result: 'T          // Final result
    // ... intermediate values from let! bindings ...
    // ... captures ...
}
```

### 4.3 AsyncBind Emission

At each `let!`, emit suspension logic:

```fsharp
let emitAsyncBind z bindNodeId name innerSSA suspIdx =
    // 1. Evaluate inner async (for now, completes immediately)
    let innerResultSSA = emitRunSynchronously z innerSSA

    // 2. Store result for use after potential suspension
    emit $"  %%{name}_ptr = llvm.getelementptr %%frame[0, {suspIdx + 2}]"
    emit $"  llvm.store %%{innerResultSSA}, %%{name}_ptr"

    // 3. Suspension point (for true async in future)
    // emit $"  %susp_{suspIdx} = llvm.call @llvm.coro.suspend(...)"
    // emit $"  llvm.switch %susp_{suspIdx} [0: ^resume_{suspIdx}, 1: ^cleanup]"

    // 4. Continue with bound value
    emit $"^after_let_{suspIdx}:"
    let boundSSA = freshSSA ()
    emit $"  %%{boundSSA} = llvm.load %%{name}_ptr"

    // Add binding to state
    z |> addVarBinding name boundSSA (typeOf innerSSA)
```

### 4.4 Full Coroutine Form

With multiple suspension points:

```mlir
llvm.func @composed_async(%frame: !llvm.ptr) -> !llvm.ptr
    attributes {presplitcoroutine} {
entry:
    %id = llvm.call @llvm.coro.id(...)
    %hdl = llvm.call @llvm.coro.begin(...)

    // Load state
    %state_ptr = llvm.getelementptr %frame[0, 0]
    %state = llvm.load %state_ptr : i32
    llvm.switch %state [
        0: ^start,
        1: ^after_let1,
        2: ^after_let2
    ]

^start:
    // let! x = async { return 10 }
    // ... run inner async ...
    %x = ...  // result is 10
    llvm.store %x, %x_slot
    llvm.store 1, %state_ptr  // next state
    // For true suspension: %susp = llvm.call @llvm.coro.suspend(...)
    llvm.br ^after_let1

^after_let1:
    %x_val = llvm.load %x_slot : i32

    // let! y = async { return 20 }
    // ... run inner async ...
    %y = ...  // result is 20
    llvm.store %y, %y_slot
    llvm.store 2, %state_ptr
    llvm.br ^after_let2

^after_let2:
    %y_val = llvm.load %y_slot : i32

    // return x + y
    %result = arith.addi %x_val, %y_val : i32
    llvm.store %result, %result_slot

    llvm.call @llvm.coro.end(...)
    llvm.return %hdl
}
```

## 5. MLIR Output Specification

### 5.1 Frame with Suspension Points

```mlir
// async { let! x = ...; let! y = ...; return x + y }
!composed_frame = !llvm.struct<(
    i32,    // state
    i32,    // result
    i32,    // x_slot (bound by first let!)
    i32     // y_slot (bound by second let!)
)>
```

### 5.2 State Machine Switch

```mlir
// Entry point dispatches based on state
%state = llvm.load %state_ptr : i32
llvm.switch %state : i32 [
    0: ^state0,
    1: ^state1,
    2: ^state2
], ^done
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module AsyncAwaitSample

let asyncAdd a b = async {
    return a + b
}

let composed = async {
    let! x = asyncAdd 10 20
    let! y = asyncAdd x 5
    return y
}

let withDo = async {
    do! async { Console.writeln "Step 1" }
    do! async { Console.writeln "Step 2" }
    return 42
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Async Await Test ==="

    Console.writeln "--- Composed Async ---"
    let v1 = Async.RunSynchronously composed
    Console.write "Result: "
    Console.writeln (Format.int v1)

    Console.writeln "--- Async with do! ---"
    let v2 = Async.RunSynchronously withDo
    Console.write "Result: "
    Console.writeln (Format.int v2)

    0
```

### 6.2 Expected Output

```
=== Async Await Test ===
--- Composed Async ---
Result: 35
--- Async with do! ---
Step 1
Step 2
Result: 42
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `SemanticGraph.fs` | MODIFY | Add AsyncBind, AsyncDo SemanticKinds |
| `Expressions/Computations.fs` | MODIFY | Handle let! and do! in async |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/SuspensionIndices.fs` | MODIFY | Number suspension points |
| `src/Alex/Witnesses/AsyncWitness.fs` | MODIFY | Emit state machine with suspension |
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle AsyncBind, AsyncDo |

## 8. Implementation Checklist

### Phase 1: FNCS Suspension Points
- [ ] Add AsyncBind, AsyncDo to SemanticKind
- [ ] Implement let!/do! type checking
- [ ] Assign suspension indices during checking

### Phase 2: Alex State Machine
- [ ] Extend frame struct for bound variables
- [ ] Emit state-based dispatch switch
- [ ] Emit suspension point code
- [ ] Emit continuation after each let!

### Phase 3: Validation
- [ ] Sample 18 compiles without errors
- [ ] Sample 18 produces correct output
- [ ] Samples 01-17 still pass

## 9. Future: True Suspension

Currently, inner asyncs complete immediately. True suspension requires:

1. **I/O Operations** (I-01/I-02): Socket read that may block
2. **Thread Integration** (T-01/T-02): Async running on different thread

When these exist, `llvm.coro.suspend` actually suspends, and a scheduler resumes the coroutine when the awaited operation completes.

## 10. Related PRDs

- **A-01**: BasicAsync - Foundation
- **A-03**: AsyncParallel - Running multiple asyncs
- **I-01/I-02**: Networking - True async I/O
- **T-03-31**: MailboxProcessor - Async message loop
