# A-03: Async.Parallel

> **Sample**: `19_AsyncParallel` | **Status**: Planned | **Depends On**: A-01-18 (BasicAsync, AsyncAwait)

## 1. Executive Summary

`Async.Parallel` runs multiple async computations and collects their results. This is a composition operation - it doesn't introduce new suspension semantics, but combines multiple asyncs into one.

**Key Insight**: `Async.Parallel` in single-threaded mode runs asyncs sequentially. True parallelism requires threading (T-01/T-02). But the API and semantics are established here.

## 2. Language Feature Specification

### 2.1 Async.Parallel

```fsharp
let tasks = [|
    async { return 1 }
    async { return 2 }
    async { return 3 }
|]

let results = Async.Parallel tasks |> Async.RunSynchronously
// results = [| 1; 2; 3 |]
```

### 2.2 Async.Sequential (For Comparison)

```fsharp
let results = Async.Sequential tasks |> Async.RunSynchronously
```

Explicitly sequential execution (same as Parallel in single-threaded mode).

### 2.3 Type Signatures

```fsharp
Async.Parallel : Async<'T>[] -> Async<'T[]>
Async.Sequential : Async<'T>[] -> Async<'T[]>
```

## 3. FNCS Layer Implementation

### 3.1 Async.Parallel Intrinsic

```fsharp
// In CheckExpressions.fs
| "Async.Parallel" ->
    // Async<'a>[] -> Async<'a[]>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TArray(NativeType.TAsync(aVar)),
        NativeType.TAsync(NativeType.TArray(aVar)))

| "Async.Sequential" ->
    // Same signature
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TArray(NativeType.TAsync(aVar)),
        NativeType.TAsync(NativeType.TArray(aVar)))
```

### 3.2 No New SemanticKind

Async.Parallel is recognized as an intrinsic call - no special PSG node needed. The witness handles it based on the intrinsic tag.

## 4. Firefly/Alex Layer Implementation

### 4.1 Sequential Implementation (Single-Threaded)

```fsharp
let witnessAsyncParallel z asyncArraySSA =
    // For single-threaded mode, just run each in sequence
    let resultArraySSA = freshSSA ()

    // 1. Get array length
    emit $"  %%len = llvm.call @array_length(%%{asyncArraySSA})"

    // 2. Allocate result array
    emit $"  %%{resultArraySSA} = llvm.call @array_create_uninit(%%len)"

    // 3. Loop over asyncs
    emit "  %i = llvm.alloca 1 x i32"
    emit "  llvm.store 0, %i"
    emit "  llvm.br ^loop"

    emit "^loop:"
    emit "  %idx = llvm.load %i : i32"
    emit "  %done = arith.cmpi uge, %idx, %len : i32"
    emit "  llvm.cond_br %done, ^exit, ^body"

    emit "^body:"
    // Get async at index
    emit $"  %%async_i = llvm.call @array_get(%%{asyncArraySSA}, %%idx)"

    // Run it synchronously
    emit "  %result_i = llvm.call @async_run_sync(%async_i)"

    // Store result
    emit $"  llvm.call @array_set(%%{resultArraySSA}, %%idx, %%result_i)"

    // Increment and loop
    emit "  %next = arith.addi %idx, 1 : i32"
    emit "  llvm.store %next, %i"
    emit "  llvm.br ^loop"

    emit "^exit:"

    TRValue { SSA = resultArraySSA; Type = TArray resultElemType }
```

### 4.2 Parallel Implementation (With Threading)

When threading is available (T-01/T-02):

```fsharp
let witnessAsyncParallelThreaded z asyncArraySSA =
    // 1. Allocate result array and completion counter
    // 2. For each async, spawn a thread that:
    //    a. Runs the async
    //    b. Stores result in array
    //    c. Increments completion counter
    // 3. Wait for all completions
    // 4. Return result array
```

This is deferred to after threading PRDs.

## 5. MLIR Output Specification

### 5.1 Sequential Loop

```mlir
// Async.Parallel [| async1; async2; async3 |]
llvm.func @async_parallel_seq(%asyncs: !llvm.ptr, %len: i32) -> !llvm.ptr {
    // Allocate result array
    %results = llvm.call @array_create_uninit(%len) : (i32) -> !llvm.ptr

    // Initialize loop counter
    %i_ptr = llvm.alloca 1 x i32
    llvm.store %c0, %i_ptr
    llvm.br ^loop

^loop:
    %i = llvm.load %i_ptr : i32
    %done = arith.cmpi uge, %i, %len : i32
    llvm.cond_br %done, ^exit, ^body

^body:
    // Get async at index
    %async_ptr = llvm.call @array_get_ptr(%asyncs, %i)
    %async_val = llvm.load %async_ptr : !llvm.ptr

    // Run synchronously
    %result = llvm.call @async_run_sync(%async_val)

    // Store result
    %result_ptr = llvm.call @array_get_ptr(%results, %i)
    llvm.store %result, %result_ptr

    // Increment
    %next = arith.addi %i, %c1 : i32
    llvm.store %next, %i_ptr
    llvm.br ^loop

^exit:
    llvm.return %results : !llvm.ptr
}
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module AsyncParallelSample

let makeAsync (n: int) = async {
    Console.write "Computing "
    Console.writeln (Format.int n)
    return n * n
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Async Parallel Test ==="

    let tasks = [|
        makeAsync 1
        makeAsync 2
        makeAsync 3
        makeAsync 4
        makeAsync 5
    |]

    Console.writeln "--- Running Parallel ---"
    let results = Async.Parallel tasks |> Async.RunSynchronously

    Console.writeln "--- Results ---"
    for r in results do
        Console.writeln (Format.int r)

    0
```

### 6.2 Expected Output

```
=== Async Parallel Test ===
--- Running Parallel ---
Computing 1
Computing 2
Computing 3
Computing 4
Computing 5
--- Results ---
1
4
9
16
25
```

Note: In single-threaded mode, "Computing N" appears in order. With true threading, the order may vary.

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Async.Parallel, Async.Sequential intrinsics |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/AsyncWitness.fs` | MODIFY | Implement parallel/sequential execution |

## 8. Implementation Checklist

### Phase 1: Sequential Implementation
- [ ] Add Async.Parallel intrinsic to FNCS
- [ ] Implement sequential execution witness
- [ ] Test with array of asyncs

### Phase 2: Validation
- [ ] Sample 19 compiles without errors
- [ ] Sample 19 produces correct output
- [ ] Samples 01-18 still pass

### Phase 3 (Future): True Parallelism
- [ ] After T-01/T-02: Implement threaded version
- [ ] Add thread pool or worker spawn
- [ ] Implement completion synchronization

## 9. Design Decision: Arrays vs Lists

Using arrays (`Async<'T>[]`) rather than lists because:
1. Known length enables result array pre-allocation
2. Index-based access for parallel assignment
3. Cache-friendly iteration

Lists could be supported via `Async.ParallelSeq : seq<Async<'T>> -> Async<'T[]>` that first collects to array.

## 10. Related PRDs

- **A-01-18**: BasicAsync, AsyncAwait - Foundation
- **T-01/T-02**: Threading - True parallelism
- **T-03-31**: MailboxProcessor - Parallel actors
