# A-01: Basic Async (LLVM Coroutines Foundation)

> **Sample**: `17_BasicAsync` | **Status**: Planned | **Depends On**: C-01 to C-07 (Closures through Lazy)

## 1. Executive Summary

This PRD establishes the foundation for async computation using **LLVM coroutine intrinsics**. Unlike .NET's Task-based async (which requires a runtime), Fidelity's async compiles directly to LLVM coroutine state machines that the CoroSplit pass transforms at compile time.

**Key Insight**: LLVM coroutines are a compile-time transformation, not a runtime feature. The compiler marks suspension points, and LLVM's passes handle frame allocation, state saving, and resumption code generation.

**Reference**: See `async_llvm_coroutines` memory for the StateMachine strategy.

## 2. Language Feature Specification

### 2.1 Basic Async Expression

```fsharp
let simpleAsync = async {
    return 42
}
```

A trivial async that immediately returns - no suspension points.

### 2.2 Async.RunSynchronously

```fsharp
let result = Async.RunSynchronously simpleAsync
```

Runs the async to completion, blocking the current thread.

### 2.3 Async Return Types

```fsharp
// Async<int> - async returning int
let intAsync = async { return 42 }

// Async<string> - async returning string
let strAsync = async { return "hello" }

// Async<unit> - async with no meaningful return
let unitAsync = async { Console.writeln "done" }
```

## 3. FNCS Layer Implementation

### 3.1 Async Type

```fsharp
// In NativeTypes.fs
| TAsync of resultType: NativeType

// Type constructor
| "Async" -> fun resTy -> NativeType.TAsync(resTy)
```

### 3.2 SemanticKind.AsyncExpr

```fsharp
type SemanticKind =
    | AsyncExpr of
        body: NodeId *
        suspensionPoints: int list *  // Indices for let! points
        captures: CaptureInfo list
```

### 3.3 SemanticKind.AsyncReturn

```fsharp
| AsyncReturn of value: NodeId
```

The `return` keyword in async context.

### 3.4 Async Intrinsics

```fsharp
// In CheckExpressions.fs
| "Async.RunSynchronously" ->
    // Async<'a> -> 'a
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TAsync(aVar), aVar)

| "async.Return" ->  // Internal, for return keyword
    // 'a -> Async<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(aVar, NativeType.TAsync(aVar))
```

## 4. Firefly/Alex Layer Implementation

### 4.1 LLVM Coroutine Intrinsics

| Intrinsic | Purpose |
|-----------|---------|
| `llvm.coro.id` | Get coroutine identity token |
| `llvm.coro.begin` | Start coroutine with frame pointer |
| `llvm.coro.size.i64` | Get required frame size |
| `llvm.coro.alloc` | Check if frame allocation needed |
| `llvm.coro.suspend` | Mark suspension point |
| `llvm.coro.resume` | Resume suspended coroutine |
| `llvm.coro.done` | Check if coroutine complete |
| `llvm.coro.end` | Mark coroutine end |
| `llvm.coro.free` | Free coroutine frame |

### 4.2 Coroutine Frame Structure

For basic async (no suspension points), the frame is minimal:
```
AsyncFrame<T> = {
    Result: T           // The return value
    // ...captures...
}
```

### 4.3 AsyncExpr Witness (No Suspension)

```fsharp
let witnessAsyncExpr z asyncNodeId =
    let coeffect = lookupAsyncCoeffect asyncNodeId z

    // For basic async with no let!, we can inline
    // The "coroutine" immediately completes

    // 1. Allocate frame
    emit $"  %%{coeffect.FrameSSA} = llvm.alloca 1 x {coeffect.FrameType}"

    // 2. Store captures
    for (i, cap) in List.indexed coeffect.Captures do
        emitCaptureStore z coeffect.FrameSSA i cap

    // 3. Execute body (no suspension)
    let bodyResult = witnessAsyncBody z asyncNodeId

    // 4. Store result
    emit $"  %%result_ptr = llvm.getelementptr %%{coeffect.FrameSSA}[0, 0]"
    emit $"  llvm.store %%{bodyResult}, %%result_ptr"

    TRValue { SSA = coeffect.FrameSSA; Type = TAsync }
```

### 4.4 Async.RunSynchronously Witness

For basic async (no suspension), just extract the result:

```fsharp
let witnessRunSynchronously z asyncSSA =
    // For non-suspending async, result is already computed
    emit $"  %%result_ptr = llvm.getelementptr %%{asyncSSA}[0, 0]"
    emit $"  %%result = llvm.load %%result_ptr"

    TRValue { SSA = "result"; Type = resultType }
```

## 5. MLIR Output Specification

### 5.1 Basic Async Frame

```mlir
// async { return 42 }
!async_int_frame = !llvm.struct<(
    i32    // result
)>
```

### 5.2 Async Creation (Trivial)

```mlir
// let simpleAsync = async { return 42 }
%frame = llvm.alloca 1 x !async_int_frame
%result_ptr = llvm.getelementptr %frame[0, 0]
%c42 = arith.constant 42 : i32
llvm.store %c42, %result_ptr
```

### 5.3 RunSynchronously (Trivial)

```mlir
// let result = Async.RunSynchronously simpleAsync
%result_ptr = llvm.getelementptr %simpleAsync[0, 0]
%result = llvm.load %result_ptr : i32
```

### 5.4 Full Coroutine Form (For Reference)

When suspension points exist (A-02), the structure becomes:

```mlir
llvm.func @async_body(%frame: !llvm.ptr) -> !llvm.ptr
    attributes {presplitcoroutine} {
entry:
    %zero1 = llvm.mlir.zero : !llvm.ptr
    %zero2 = llvm.mlir.zero : !llvm.ptr
    %zero3 = llvm.mlir.zero : !llvm.ptr
    %id = llvm.call @llvm.coro.id(i32 0, ptr %zero1, ptr %zero2, ptr %zero3)
    %need_alloc = llvm.call @llvm.coro.alloc(token %id)
    llvm.cond_br %need_alloc, ^alloc, ^begin

^alloc:
    %size = llvm.call @llvm.coro.size.i64()
    %mem = llvm.call @malloc(i64 %size)
    llvm.br ^begin(%mem : !llvm.ptr)

^begin(%frame_mem: !llvm.ptr):
    %hdl = llvm.call @llvm.coro.begin(token %id, ptr %frame_mem)

    // ... async body code ...

    %final = llvm.call @llvm.coro.end(ptr %hdl, i1 false, token none)
    llvm.return %hdl : !llvm.ptr
}
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicAsyncSample

let simpleAsync = async {
    return 42
}

let asyncWithCapture x = async {
    return x * 2
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Async Test ==="

    Console.writeln "--- Simple Async ---"
    let v1 = Async.RunSynchronously simpleAsync
    Console.write "Result: "
    Console.writeln (Format.int v1)

    Console.writeln "--- Async with Capture ---"
    let v2 = Async.RunSynchronously (asyncWithCapture 21)
    Console.write "Result: "
    Console.writeln (Format.int v2)

    0
```

### 6.2 Expected Output

```
=== Basic Async Test ===
--- Simple Async ---
Result: 42
--- Async with Capture ---
Result: 42
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TAsync type constructor |
| `SemanticGraph.fs` | MODIFY | Add AsyncExpr, AsyncReturn SemanticKinds |
| `CheckExpressions.fs` | MODIFY | Add Async.RunSynchronously intrinsic |
| `Expressions/Computations.fs` | MODIFY | Handle async { } expressions |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/AsyncWitness.fs` | CREATE | Emit async frame and basic coroutine |
| `src/Alex/Preprocessing/SuspensionIndices.fs` | CREATE | Number suspension points |
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle AsyncExpr, AsyncReturn |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TAsync to NativeTypes
- [ ] Add AsyncExpr, AsyncReturn to SemanticKind
- [ ] Implement async { } checking
- [ ] Add Async.RunSynchronously intrinsic

### Phase 2: Alex Basic Async
- [ ] Create AsyncWitness for non-suspending async
- [ ] Implement trivial frame allocation
- [ ] Implement RunSynchronously for immediate completion

### Phase 3: Validation
- [ ] Sample 17 compiles without errors
- [ ] Sample 17 produces correct output
- [ ] Samples 01-16 still pass

## 9. Why LLVM Coroutines?

| Alternative | Drawback |
|-------------|----------|
| .NET Task | Requires runtime, GC, thread pool |
| libco/boost.context | External dependency, non-portable |
| Custom state machine | Complex, error-prone |
| **LLVM Coroutines** | **Built-in, cross-platform, no runtime** |

LLVM's coroutine passes are production-quality and handle all the difficult state machine generation automatically.

## 10. Related PRDs

- **A-02**: Async Await - Adds `let!` (suspension points)
- **A-03**: Async Parallel - Adds `Async.Parallel`
- **T-03 to T-05**: MailboxProcessor - Uses async for message loop
