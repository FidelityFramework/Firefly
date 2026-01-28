# T-02: Mutex Synchronization

> **Sample**: `28_MutexSync` | **Status**: Planned | **Depends On**: T-01 (BasicThread)

## 1. Executive Summary

This PRD adds mutual exclusion (mutex) primitives for thread synchronization. Mutexes prevent data races when multiple threads access shared state.

**Key Insight**: Mutexes are OS primitives (pthread_mutex on Linux, CRITICAL_SECTION on Windows). They protect critical sections where shared data is accessed.

## 2. Language Feature Specification

### 2.1 Mutex Creation

```fsharp
let mutex = Mutex.create ()
```

### 2.2 Lock/Unlock

```fsharp
Mutex.lock mutex
// Critical section - only one thread at a time
sharedCounter <- sharedCounter + 1
Mutex.unlock mutex
```

### 2.3 Lock Guard Pattern

```fsharp
let withLock (mutex: Mutex) (f: unit -> 'T) : 'T =
    Mutex.lock mutex
    let result = f ()
    Mutex.unlock mutex
    result

// Usage
let value = withLock mutex (fun () ->
    sharedData.[index])
```

### 2.4 Condition Variables (Future)

```fsharp
let cond = CondVar.create ()
CondVar.wait cond mutex    // Atomically unlock and wait
CondVar.signal cond        // Wake one waiter
CondVar.broadcast cond     // Wake all waiters
```

## 3. FNCS Layer Implementation

### 3.1 Mutex Type

```fsharp
// In NativeTypes.fs
| TMutex      // Opaque mutex handle
| TCondVar    // Opaque condition variable handle
```

### 3.2 Mutex Intrinsics

```fsharp
// In CheckExpressions.fs
| "Mutex.create" ->
    // unit -> Mutex
    NativeType.TFun(env.Globals.UnitType, NativeType.TMutex)

| "Mutex.lock" ->
    // Mutex -> unit
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)

| "Mutex.unlock" ->
    // Mutex -> unit
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)

| "Mutex.tryLock" ->
    // Mutex -> bool
    NativeType.TFun(NativeType.TMutex, env.Globals.BoolType)

| "Mutex.destroy" ->
    // Mutex -> unit
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)
```

### 3.3 CondVar Intrinsics

```fsharp
| "CondVar.create" ->
    // unit -> CondVar
    NativeType.TFun(env.Globals.UnitType, NativeType.TCondVar)

| "CondVar.wait" ->
    // CondVar -> Mutex -> unit
    NativeType.TFun(NativeType.TCondVar,
        NativeType.TFun(NativeType.TMutex, env.Globals.UnitType))

| "CondVar.signal" ->
    // CondVar -> unit
    NativeType.TFun(NativeType.TCondVar, env.Globals.UnitType)

| "CondVar.broadcast" ->
    // CondVar -> unit
    NativeType.TFun(NativeType.TCondVar, env.Globals.UnitType)
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Mutex Structure

On Linux, pthread_mutex_t is 40 bytes. We allocate opaque storage:

```fsharp
type Mutex = {
    Storage: byte[40]  // pthread_mutex_t
}
```

### 4.2 Mutex.create Witness

```fsharp
let witnessMutexCreate z =
    let mutexSSA = freshSSA ()

    // Allocate mutex storage
    emit $"  %%{mutexSSA} = llvm.alloca 40 x i8"

    // Initialize (attr=0 means default attributes)
    emit "  %attr_zero = llvm.mlir.zero : !llvm.ptr"
    emit $"  llvm.call @pthread_mutex_init(%%{mutexSSA}, %%attr_zero)"

    TRValue { SSA = mutexSSA; Type = TMutex }
```

### 4.3 Mutex.lock/unlock Witness

```fsharp
let witnessMutexLock z mutexSSA =
    emit $"  llvm.call @pthread_mutex_lock(%%{mutexSSA})"
    TRVoid

let witnessMutexUnlock z mutexSSA =
    emit $"  llvm.call @pthread_mutex_unlock(%%{mutexSSA})"
    TRVoid
```

### 4.4 CondVar.wait Witness

```fsharp
let witnessCondVarWait z condSSA mutexSSA =
    emit $"  llvm.call @pthread_cond_wait(%%{condSSA}, %%{mutexSSA})"
    TRVoid
```

## 5. MLIR Output Specification

### 5.1 Mutex Type

```mlir
// Opaque storage for pthread_mutex_t
!mutex_type = !llvm.array<40 x i8>
```

### 5.2 Mutex Operations

```mlir
// let mutex = Mutex.create ()
%mutex = llvm.alloca 40 x i8
llvm.call @pthread_mutex_init(%mutex, %null) : (!llvm.ptr, !llvm.ptr) -> i32

// Mutex.lock mutex
llvm.call @pthread_mutex_lock(%mutex) : (!llvm.ptr) -> i32

// ... critical section ...

// Mutex.unlock mutex
llvm.call @pthread_mutex_unlock(%mutex) : (!llvm.ptr) -> i32
```

### 5.3 CondVar Operations

```mlir
// CondVar.wait cond mutex
llvm.call @pthread_cond_wait(%cond, %mutex) : (!llvm.ptr, !llvm.ptr) -> i32

// CondVar.signal cond
llvm.call @pthread_cond_signal(%cond) : (!llvm.ptr) -> i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module MutexSyncSample

let mutable sharedCounter = 0

let incrementer (mutex: Mutex) (count: int) () =
    for _ in 1..count do
        Mutex.lock mutex
        sharedCounter <- sharedCounter + 1
        Mutex.unlock mutex

[<EntryPoint>]
let main _ =
    Console.writeln "=== Mutex Sync Test ==="

    let mutex = Mutex.create ()

    // Without mutex (race condition)
    sharedCounter <- 0
    let t1_unsafe = Thread.create (fun () ->
        for _ in 1..100000 do
            sharedCounter <- sharedCounter + 1)
    let t2_unsafe = Thread.create (fun () ->
        for _ in 1..100000 do
            sharedCounter <- sharedCounter + 1)
    Thread.join t1_unsafe
    Thread.join t2_unsafe
    Console.write "Without mutex (expect < 200000): "
    Console.writeln (Format.int sharedCounter)

    // With mutex (correct)
    sharedCounter <- 0
    let t1_safe = Thread.create (incrementer mutex 100000)
    let t2_safe = Thread.create (incrementer mutex 100000)
    Thread.join t1_safe
    Thread.join t2_safe
    Console.write "With mutex (expect 200000): "
    Console.writeln (Format.int sharedCounter)

    Mutex.destroy mutex
    0
```

### 6.2 Expected Output

```
=== Mutex Sync Test ===
Without mutex (expect < 200000): 156789  (varies, usually < 200000)
With mutex (expect 200000): 200000
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TMutex, TCondVar types |
| `CheckExpressions.fs` | MODIFY | Add Mutex/CondVar intrinsics |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/MutexWitness.fs` | CREATE | Mutex/CondVar witnesses |
| `src/Alex/Bindings/Mutex_Linux_x86_64.fs` | CREATE | pthread_mutex bindings |
| `src/Alex/Bindings/Mutex_Windows_x86_64.fs` | CREATE | CRITICAL_SECTION bindings |

## 8. Implementation Checklist

### Phase 1: Basic Mutex
- [ ] Add TMutex type
- [ ] Add Mutex.create/lock/unlock/destroy intrinsics
- [ ] Implement witnesses
- [ ] Add platform bindings

### Phase 2: Condition Variables
- [ ] Add TCondVar type
- [ ] Add CondVar intrinsics
- [ ] Implement witnesses

### Phase 3: Validation
- [ ] Sample 28 compiles
- [ ] Race condition visible without mutex
- [ ] Correct result with mutex
- [ ] Samples 01-27 still pass

## 9. Platform Bindings

| Operation | Linux | Windows |
|-----------|-------|---------|
| Mutex init | pthread_mutex_init | InitializeCriticalSection |
| Mutex lock | pthread_mutex_lock | EnterCriticalSection |
| Mutex unlock | pthread_mutex_unlock | LeaveCriticalSection |
| Mutex destroy | pthread_mutex_destroy | DeleteCriticalSection |
| CondVar init | pthread_cond_init | InitializeConditionVariable |
| CondVar wait | pthread_cond_wait | SleepConditionVariableCS |
| CondVar signal | pthread_cond_signal | WakeConditionVariable |

## 10. Related PRDs

- **T-01**: BasicThread - Threading foundation
- **T-03-31**: MailboxProcessor - Uses mutex for message queue
