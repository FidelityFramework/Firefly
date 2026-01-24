# R-02: Observable Operators

> **Sample**: `33_ObservableOperators` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD defines the core Observable operators: map, filter, merge, combine, take, skip. Following the Fidelity principle, operators decompose in Baker to primitives rather than building operator chains at runtime.

**Key Insight**: Observable operators are compile-time composition, not runtime operator objects.

---

## 2. Operator Specification

### 2.1 Transformation Operators

| Operator | Type | Description |
|----------|------|-------------|
| `Observable.map` | `('T -> 'U) -> Observable<'T, 'M> -> Observable<'U, 'M>` | Transform each value |
| `Observable.filter` | `('T -> bool) -> Observable<'T, 'M> -> Observable<'T, 'M>` | Keep values matching predicate |
| `Observable.scan` | `('S -> 'T -> 'S) -> 'S -> Observable<'T, 'M> -> Observable<'S, 'M>` | Accumulate state |

### 2.2 Combination Operators

| Operator | Type | Description |
|----------|------|-------------|
| `Observable.merge` | `Observable<'T, 'M> list -> Observable<'T, 'M>` | Interleave multiple sources |
| `Observable.combineLatest` | `Observable<'T, _> -> Observable<'U, _> -> Observable<'T * 'U, _>` | Latest from each |
| `Observable.zip` | `Observable<'T, _> -> Observable<'U, _> -> Observable<'T * 'U, _>` | Pairwise combine |

### 2.3 Filtering Operators

| Operator | Type | Description |
|----------|------|-------------|
| `Observable.take` | `int -> Observable<'T, 'M> -> Observable<'T, 'M>` | First n values |
| `Observable.skip` | `int -> Observable<'T, 'M> -> Observable<'T, 'M>` | Skip first n values |
| `Observable.distinct` | `Observable<'T, 'M> -> Observable<'T, 'M>` | Remove duplicates |
| `Observable.throttle` | `TimeSpan -> Observable<'T, 'M> -> Observable<'T, 'M>` | Rate limit |

---

## 3. Baker Decomposition

### 3.1 Observable.map Decomposition

```fsharp
Observable.map f source
```

Decomposes to transformed observer subscription:

```fsharp
// Baker output
Observable.subscribe {
    OnNext = fun x -> downstream.OnNext (f x)
    OnError = downstream.OnError
    OnCompleted = downstream.OnCompleted
} source
```

The `f` call is inlined into the OnNext closure - no operator object created.

### 3.2 Observable.filter Decomposition

```fsharp
Observable.filter pred source
```

Decomposes to:

```fsharp
Observable.subscribe {
    OnNext = fun x ->
        if pred x then downstream.OnNext x
    OnError = downstream.OnError
    OnCompleted = downstream.OnCompleted
} source
```

### 3.3 Operator Fusion

Consecutive operators fuse in Baker:

```fsharp
source
|> Observable.map f
|> Observable.filter pred
|> Observable.map g
```

Fuses to single observer:

```fsharp
Observable.subscribe {
    OnNext = fun x ->
        let y = f x
        if pred y then
            let z = g y
            downstream.OnNext z
    // ...
} source
```

One closure, no intermediate operator objects.

---

## 4. Implementation Strategy

### 4.1 ObservableRecipes in Baker

```fsharp
// In Baker
module ObservableRecipes =
    let decomposeMap f source downstream =
        // Generate transformed subscription
        ...

    let decomposeFilter pred source downstream =
        // Generate conditional subscription
        ...

    let detectFusion ops =
        // Identify fusible operator chains
        // Return single fused decomposition
        ...
```

### 4.2 State-Carrying Operators

Some operators require state:

```fsharp
Observable.scan folder initial source
```

Requires mutable accumulator:

```fsharp
let mutable acc = initial
Observable.subscribe {
    OnNext = fun x ->
        acc <- folder acc x
        downstream.OnNext acc
    // ...
} source
```

The `acc` is captured by-ref in the closure (per C-01 patterns).

### 4.3 combineLatest State

```fsharp
Observable.combineLatest source1 source2
```

Requires state for latest values:

```fsharp
type CombineState = {
    mutable Latest1: 'T option
    mutable Latest2: 'U option
    mutable HasBoth: bool
}

let state = { Latest1 = None; Latest2 = None; HasBoth = false }

// Subscribe to source1
Observable.subscribe {
    OnNext = fun x ->
        state.Latest1 <- Some x
        if state.Latest2.IsSome then
            state.HasBoth <- true
            downstream.OnNext (x, state.Latest2.Value)
    // ...
} source1

// Subscribe to source2 (similar)
```

---

## 5. Coeffects

| Coeffect | Purpose |
|----------|---------|
| ClosureLayout | Operator closures with captures |
| OperatorFusion | Track fusible operator chains |
| MutableCapture | State-carrying operators |

---

## 6. MLIR Patterns

### 6.1 Fused map-filter

```mlir
// Fused map(f).filter(pred) observer
%onNext = llvm.func @fused_observer_onNext(%x: !T, %env: !env_t) {
  // Apply f
  %y = llvm.call @f(%x, %f_env)

  // Apply pred
  %keep = llvm.call @pred(%y, %pred_env)

  llvm.cond_br %keep, ^emit, ^skip

^emit:
  // Forward to downstream
  %downstream = llvm.extractvalue %env[0]
  llvm.call @invoke_onNext(%downstream, %y)
  llvm.br ^done

^skip:
  llvm.br ^done

^done:
  llvm.return
}
```

### 6.2 Stateful scan

```mlir
// scan observer with mutable state
%onNext = llvm.func @scan_observer_onNext(%x: !T, %env: !scan_env_t) {
  // Load current accumulator (byref capture)
  %acc_ptr = llvm.extractvalue %env[0]
  %acc = llvm.load %acc_ptr

  // Apply folder
  %new_acc = llvm.call @folder(%acc, %x, %folder_env)

  // Store new accumulator
  llvm.store %new_acc, %acc_ptr

  // Emit accumulated value
  %downstream = llvm.extractvalue %env[1]
  llvm.call @invoke_onNext(%downstream, %new_acc)

  llvm.return
}
```

---

## 7. Sample Code

```fsharp
module ObservableOperators

let source = Observable.fromSeq [1; 2; 3; 4; 5]

// Fused operator chain
let processed =
    source
    |> Observable.map (fun x -> x * 2)
    |> Observable.filter (fun x -> x > 4)
    |> Observable.map (fun x -> $"Value: {x}")

let observer = {
    OnNext = Console.writeln
    OnError = fun _ -> ()
    OnCompleted = fun () -> Console.writeln "Done"
}

Observable.subscribe observer processed
// Output: Value: 6, Value: 8, Value: 10, Done
```

---

## 8. Validation Criteria

- [ ] Observable.map transforms values
- [ ] Observable.filter removes values
- [ ] Consecutive operators fuse in Baker
- [ ] Observable.scan accumulates state correctly
- [ ] combineLatest emits when both sources have values

---

## 9. Dependencies

| Dependency | Purpose |
|------------|---------|
| R-01 ObservableFoundations | Base Observable type |
| C-01 Closures | Operator closures |
| C-07 SeqOperations | Pattern for operator decomposition |

---

## 10. Related Documents

- [R-01-ObservableFoundations](R-01-ObservableFoundations.md) - Base infrastructure
- [R-03-ObservableIntegration](R-03-ObservableIntegration.md) - Cross-pattern integration
- [C-07-SeqOperations](C-07-SeqOperations.md) - Similar decomposition patterns
