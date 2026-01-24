# R-03: Observable Integration

> **Sample**: `34_ObservableAsync` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD defines integration between Observables and other async patterns: Async, Actors, and Sequences. These bridges enable seamless composition across paradigms without allocation overhead.

**Key Insight**: Integration is compile-time routing between DCont (Async) and Inet (Observable), not runtime adapters.

---

## 2. Integration Patterns

### 2.1 Observable ↔ Async

| Function | Type | Description |
|----------|------|-------------|
| `Observable.toAsync` | `Observable<'T, _> -> Async<'T>` | Await first value |
| `Observable.toAsyncSeq` | `Observable<'T, _> -> AsyncSeq<'T>` | Async enumeration |
| `Async.toObservable` | `Async<'T> -> Observable<'T, Cold>` | Single-value observable |

### 2.2 Observable ↔ Actor

| Function | Type | Description |
|----------|------|-------------|
| `Observable.fromActor` | `MailboxProcessor<'T> -> Observable<'T, Hot>` | Actor output stream |
| `Observable.toActor` | `Observable<'T, _> -> MailboxProcessor<'T>` | Observable-fed actor |
| `Actor.observeReplies` | `MailboxProcessor<'Msg> -> Observable<'Reply, Hot>` | Reply stream |

### 2.3 Observable ↔ Seq

| Function | Type | Description |
|----------|------|-------------|
| `Observable.fromSeq` | `seq<'T> -> Observable<'T, Cold>` | Seq to observable |
| `Observable.toSeq` | `Observable<'T, _> -> seq<'T>` | Blocking collect |
| `Observable.toList` | `Observable<'T, _> -> Async<'T list>` | Async collect |

---

## 3. Architecture

### 3.1 DCont/Inet Substrate

Fidelity uses two continuation patterns:

| Pattern | Domain | Continuation Style |
|---------|--------|--------------------|
| DCont | Async, Lazy | Delimited (suspendable) |
| Inet | Observable | Immediate (non-blocking) |

Integration bridges these patterns:

```
Observable.toAsync: Inet → DCont
  - Create DCont that awaits first Inet emission
  - Inet.OnNext resumes DCont

Async.toObservable: DCont → Inet
  - Wrap DCont execution in Cold observable factory
  - On subscribe, run DCont, emit result, complete
```

### 3.2 Zero-Copy Bridging

Bridges don't allocate intermediate buffers:

```fsharp
// Observable.toAsync implementation sketch
let toAsync (obs: Observable<'T, _>) : Async<'T> =
    Async.create (fun cont ->
        let mutable subscription = Unchecked.defaultof<IDisposable>
        subscription <- Observable.subscribe {
            OnNext = fun x ->
                subscription.Dispose()
                cont.Return x
            OnError = cont.Throw
            OnCompleted = fun () -> cont.Throw (InvalidOperationException "Empty")
        } obs
    )
```

The continuation is the same memory; no boxing or queue allocation.

### 3.3 Actor Integration

Actors naturally produce event streams:

```fsharp
// Actor reply stream
let observeReplies (actor: MailboxProcessor<'Msg * AsyncReplyChannel<'Reply>>) =
    Observable.createSubject<'Reply>() |> fun subject ->
        // Wrap actor to emit replies to subject
        ...
        subject :> Observable<'Reply, Hot>
```

---

## 4. Implementation Strategy

### 4.1 toAsync via DCont

```fsharp
// In Baker decomposition
Observable.toAsync obs
```

Decomposes to DCont construction:

```fsharp
// Generated code
DCont.create (fun k ->
    let sub = Observable.subscribe {
        OnNext = fun x ->
            DCont.resume k x
        OnError = DCont.throw k
        OnCompleted = fun () ->
            DCont.throw k emptyException
    } obs
    DCont.onCancel (fun () -> sub.Dispose())
)
```

### 4.2 fromActor via Subject

```fsharp
Observable.fromActor actor
```

Decomposes to:

```fsharp
let subject = Observable.createSubject()
// Start forwarding task
Async.start (async {
    while true do
        let! msg = actor.Receive()
        subject.OnNext msg
})
subject
```

### 4.3 toSeq Blocking

```fsharp
Observable.toSeq obs
```

Requires blocking collect:

```fsharp
seq {
    let queue = ConcurrentQueue()
    let complete = ManualResetEvent(false)

    Observable.subscribe {
        OnNext = queue.Enqueue
        OnCompleted = fun () -> complete.Set()
        OnError = fun e -> complete.Set()  // + store error
    } obs

    while not (complete.WaitOne(0) && queue.IsEmpty) do
        match queue.TryDequeue() with
        | true, x -> yield x
        | _ -> Thread.Yield()
}
```

**Warning**: toSeq blocks; prefer toAsyncSeq in async contexts.

---

## 5. Coeffects

| Coeffect | Purpose |
|----------|---------|
| ContinuationKind | Track DCont vs Inet |
| BridgeAllocation | Minimal state for bridges |
| ActorIntegration | Actor-Observable coupling |

---

## 6. MLIR Patterns

### 6.1 toAsync DCont Creation

```mlir
// Observable.toAsync creates DCont
%dcont = llvm.call @dcont_create(%handler)

// Handler captures observable subscription
%handler = llvm.func @toasync_handler(%k: !dcont_t, %env: !env_t) {
  // Subscribe to observable
  %obs = llvm.extractvalue %env[0]

  // Create bridge observer
  %bridge = llvm.mlir.undef : !observer_t
  // OnNext resumes DCont
  %onNext = llvm.func @bridge_onnext(%x: !T, %bridge_env: !env_t) {
    %k = llvm.extractvalue %bridge_env[0]
    %sub = llvm.extractvalue %bridge_env[1]
    llvm.call @dispose(%sub)
    llvm.call @dcont_resume(%k, %x)
    llvm.return
  }
  // ...

  %sub = llvm.call @observable_subscribe(%bridge, %obs)
  llvm.return
}
```

### 6.2 fromSeq Cold Factory

```mlir
// Observable.fromSeq creates Cold observable
%cold = llvm.call @observable_cold(%factory)

%factory = llvm.func @fromseq_factory(%observer: !observer_t, %env: !env_t) {
  %seq = llvm.extractvalue %env[0]

  // Iterate seq, emit each value
  llvm.br ^loop

^loop:
  %has_next = llvm.call @seq_movenext(%seq)
  llvm.cond_br %has_next, ^emit, ^done

^emit:
  %x = llvm.call @seq_current(%seq)
  llvm.call @observer_onnext(%observer, %x)
  llvm.br ^loop

^done:
  llvm.call @observer_oncompleted(%observer)
  llvm.return %disposable_empty
}
```

---

## 7. Sample Code

```fsharp
module ObservableIntegration

// Async to Observable
let asyncValue = async { return 42 }
let obs = Async.toObservable asyncValue

// Observable to Async
let firstValue = Observable.toAsync obs

// Actor integration
let actor = MailboxProcessor.Start(fun inbox -> async {
    while true do
        let! msg = inbox.Receive()
        Console.writeln $"Actor received: {msg}"
})

let actorStream = Observable.fromActor actor
Observable.subscribe {
    OnNext = fun x -> Console.writeln $"Stream: {x}"
    OnError = ignore
    OnCompleted = ignore
} actorStream

// Seq integration
let numbers = Observable.fromSeq [1; 2; 3]
let collected = Observable.toList numbers |> Async.RunSynchronously
// collected = [1; 2; 3]
```

---

## 8. Validation Criteria

- [ ] Observable.toAsync awaits first emission
- [ ] Async.toObservable emits async result
- [ ] Observable.fromSeq emits all seq elements
- [ ] Observable.toList collects all emissions
- [ ] Actor integration forwards messages
- [ ] No intermediate allocation for bridges

---

## 9. Dependencies

| Dependency | Purpose |
|------------|---------|
| R-01 ObservableFoundations | Base Observable |
| R-02 ObservableOperators | Operator patterns |
| A-02 AsyncAwait | DCont infrastructure |
| T-03 BasicActor | Actor integration |
| C-06 SimpleSeq | Seq infrastructure |

---

## 10. Related Documents

- [R-01-ObservableFoundations](R-01-ObservableFoundations.md) - Base infrastructure
- [R-02-ObservableOperators](R-02-ObservableOperators.md) - Operators
- [A-02-AsyncAwait](A-02-AsyncAwait.md) - DCont patterns
- [T-03-BasicActor](T-03-BasicActor.md) - Actor model
