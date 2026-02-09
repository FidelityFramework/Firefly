# R-01: Observable Foundations

> **Sample**: `32_BasicObservable` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD introduces `Observable<'T, 'Model>` as the foundational reactive primitive. Unlike Rx.NET which relies on garbage collection and heap allocation, Fidelity observables leverage the existing closure infrastructure (C-01), async patterns (A-01), and arena allocation (F-02) to achieve zero-allocation multicast in common cases.

**Key Insight**: Observables are composition of existing primitives, not a new runtime concept.

---

## 2. Language Feature Specification

### 2.1 Observable Type

```fsharp
type Observable<'T, 'Model> =
    | Multicast of subscribers: Observer<'T> list
    | Unicast of subscriber: Observer<'T> option
    | Cold of factory: (Observer<'T> -> IDisposable)

type Observer<'T> = {
    OnNext: 'T -> unit
    OnError: exn -> unit
    OnCompleted: unit -> unit
}
```

### 2.2 Observable Model

The `'Model` type parameter distinguishes broadcast semantics:

| Model | Description | Allocation |
|-------|-------------|------------|
| `Multicast` | Multiple subscribers, single source | Zero (inline dispatch) |
| `Unicast` | Single subscriber, cold source | Minimal |
| `Hot` | Always emitting, drop if no subscriber | None |
| `Cold` | Factory-created per subscription | Per subscription |

### 2.3 Basic Operations

```fsharp
// Creating observables
Observable.return : 'T -> Observable<'T, Unicast>
Observable.empty : unit -> Observable<'T, _>
Observable.never : unit -> Observable<'T, _>

// Subscribing
Observable.subscribe : Observer<'T> -> Observable<'T, 'M> -> IDisposable

// From existing patterns
Observable.fromSeq : seq<'T> -> Observable<'T, Cold>
Observable.fromAsync : Async<'T> -> Observable<'T, Cold>
```

---

## 3. Architecture

### 3.1 Composition from Standing Art

Observable reuses existing infrastructure:

| Component | Reused From | Purpose |
|-----------|-------------|---------|
| Closures | C-01 | OnNext/OnError/OnCompleted callbacks |
| Flat struct | F-10 Records | Observer record type |
| Arena allocation | F-02 | Subscription state when needed |
| DCont | A-01 Async | Cold observable factory |
| Seq iteration | C-06, C-07 | fromSeq implementation |

### 3.2 Zero-Allocation Multicast

For `MulticastInline` strategy:

```
Observable Inline Layout (Multicast, 2 subscribers)
┌─────────────────────────────────────────────────────┐
│ count: i32                                          │
│ observer0: Observer<'T>                             │
│   ├── onNext: closure                              │
│   ├── onError: closure                             │
│   └── onCompleted: closure                         │
│ observer1: Observer<'T>                             │
│   └── ...                                          │
└─────────────────────────────────────────────────────┘
```

Emission is direct dispatch to inlined closures - no allocation per event.

### 3.3 FNCS Type Definitions

```fsharp
// In NativeTypes.fs
| NTUKind.Observable -> TObservable(elemType, modelType)
| NTUKind.Observer -> TObserver(elemType)

// Intrinsic operations
| "Observable.subscribe" -> intrinsic with DCont integration
| "Observable.emit" -> inline dispatch loop
```

---

## 4. Implementation Strategy

### 4.1 Observer as Closure Triple

An `Observer<'T>` is three closures with optional captures:

```fsharp
type Observer<'T> = {
    OnNext: 'T -> unit       // Closure from C-01
    OnError: exn -> unit     // Closure from C-01
    OnCompleted: unit -> unit // Closure from C-01
}
```

**MLIR Representation**:
```mlir
// Observer is memref<Nxi8> (flat byte buffer containing closures)
// Each closure field accessed via memref.view at computed offsets
// onNext: offset 0, onError: offset K, onCompleted: offset 2K
```

### 4.2 Multicast Emission

```mlir
// emit(observable, value)
// Observable is memref<Nxi8> with count at offset 0, observers at subsequent offsets
func.func @observable_emit(%obs: memref<?xi8>, %value: memref<?xi8>) {
  // Extract observer count
  %count_ref = memref.reinterpret_cast %obs to offset: [0], sizes: [1], strides: [1]
      : memref<?xi8> to memref<1xi32>
  %count = memref.load %count_ref[%c0] : memref<1xi32>
  %zero = arith.constant 0 : i32

  // Loop over observers using scf.while
  scf.while (%i = %zero) : (i32) -> () {
    %done = arith.cmpi sge, %i, %count : i32
    %not_done = arith.xori %done, %true : i1
    scf.condition(%not_done)
  } do {
  ^bb0(%i: i32):
    // Extract observer[i] via computed offset into flat buffer
    // Get onNext closure, invoke per C-01 calling convention
    // func.call @invoke_closure(%onNext, %value)
    %next = arith.addi %i, %c1_i32 : i32
    scf.yield %next : i32
  }
  func.return
}
```

### 4.3 Cold Observable Factory

Cold observables use DCont (from A-01) for lazy subscription:

```fsharp
let cold = Observable.create (fun observer ->
    // Factory runs on subscribe
    observer.OnNext 1
    observer.OnNext 2
    observer.OnCompleted()
    Disposable.empty
)
```

The factory is a closure capturing its environment; subscription invokes it.

---

## 5. Coeffects

| Coeffect | Purpose |
|----------|---------|
| ClosureLayout | Observer closures (from C-01) |
| ObservableModel | Track Multicast vs Unicast vs Cold |
| SSA | Event values through emission |

---

## 6. Sample Code

```fsharp
module BasicObservable

let observer = {
    OnNext = fun x -> Console.writeln $"Received: {x}"
    OnError = fun e -> Console.writeln $"Error: {e}"
    OnCompleted = fun () -> Console.writeln "Done"
}

// Unicast cold observable
let cold = Observable.fromSeq [1; 2; 3]
let subscription = Observable.subscribe observer cold

// Multicast hot observable
let subject = Observable.createSubject<int>()
Observable.subscribe observer subject
subject.OnNext 42
```

---

## 7. Target Applicability

| Target | Applicable | Notes |
|--------|------------|-------|
| WREN Stack | Yes | Full reactive UI patterns |
| QuantumCredential | Partial | Event streams without UI |
| LVGL MCU | No | Memory constraints |
| Unikernel | Optional | Network event handling |

---

## 8. Dependencies

| Dependency | Purpose |
|------------|---------|
| C-01 Closures | Observer callbacks |
| F-10 Records | Observer struct |
| A-01 BasicAsync | Cold observable DCont |
| C-06 SimpleSeq | fromSeq implementation |

---

## 9. Validation Criteria

- [ ] Observable.return emits single value
- [ ] Observable.subscribe receives emissions
- [ ] Multicast dispatches to multiple observers
- [ ] Cold observable factory runs on subscribe
- [ ] Zero allocation for inline multicast case

---

## 10. Related Documents

- [C-01-Closures](C-01-Closures.md) - Closure infrastructure
- [A-01-BasicAsync](A-01-BasicAsync.md) - DCont patterns
- [R-02-ObservableOperators](R-02-ObservableOperators.md) - Operator combinators
