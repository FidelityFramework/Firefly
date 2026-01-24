# T-01: Thread Primitives

> **Sample**: `27_BasicThread` | **Status**: Planned | **Depends On**: C-01 (Closures)

## 1. Executive Summary

This PRD introduces OS threading primitives - the foundation for true parallelism. Threads enable concurrent execution on multiple CPU cores.

**Key Insight**: Threads use OS APIs (pthreads on Linux, Windows threads on Windows). The thread entry function receives a closure's environment pointer, enabling captured state.

## 2. Language Feature Specification

### 2.1 Thread Creation

```fsharp
let threadFn () =
    Console.writeln "Hello from thread!"

let thread = Thread.create threadFn
```

### 2.2 Thread Join

```fsharp
Thread.join thread  // Wait for thread to complete
```

### 2.3 Thread with Closure

```fsharp
let counter = ref 0
let increment () =
    counter := !counter + 1

let t1 = Thread.create increment
let t2 = Thread.create increment
Thread.join t1
Thread.join t2
// counter may be 2 (or less due to race - see T-02 for synchronization)
```

### 2.4 Thread Return Value

```fsharp
let compute () = 42

let thread = Thread.create compute
let result = Thread.join thread  // Returns 42
```

## 3. FNCS Layer Implementation

### 3.1 Thread Type

```fsharp
// In NativeTypes.fs
| TThread of resultType: NativeType  // Thread<'T>
```

### 3.2 Thread Intrinsics

```fsharp
// In CheckExpressions.fs
| "Thread.create" ->
    // (unit -> 'T) -> Thread<'T>
    let tVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(env.Globals.UnitType, tVar),
        NativeType.TThread(tVar))

| "Thread.join" ->
    // Thread<'T> -> 'T
    let tVar = freshTypeVar ()
    NativeType.TFun(NativeType.TThread(tVar), tVar)

| "Thread.detach" ->
    // Thread<'T> -> unit
    let tVar = freshTypeVar ()
    NativeType.TFun(NativeType.TThread(tVar), env.Globals.UnitType)
```

### 3.3 Thread State Tracking

FNCS may track thread creation for escape analysis - data passed to threads must remain valid.

## 4. Firefly/Alex Layer Implementation

### 4.1 Thread Handle Structure

```fsharp
type ThreadHandle<'T> = {
    OsHandle: int64          // pthread_t or HANDLE
    ResultSlot: nativeptr<'T>  // Where result will be stored
    Closure: ClosureStruct     // The function to run
}
```

### 4.2 Thread Entry Wrapper

pthreads expects `void* (*)(void*)`. We generate a wrapper:

```fsharp
let threadEntryWrapper (arg: nativeptr<byte>) : nativeptr<byte> =
    // arg points to our ThreadHandle
    let handle = NativePtr.read<ThreadHandle> arg
    // Call the closure
    let result = handle.Closure.CodePtr(handle.Closure.EnvPtr)
    // Store result
    NativePtr.write handle.ResultSlot result
    NativePtr.nullPtr
```

### 4.3 Thread.create Witness

```fsharp
let witnessThreadCreate z closureSSA =
    let handleSSA = freshSSA ()

    // Allocate ThreadHandle struct
    emit $"  %%{handleSSA} = llvm.alloca 1 x !thread_handle"

    // Store closure
    emit $"  %%closure_slot = llvm.getelementptr %%{handleSSA}[0, 2]"
    emit $"  llvm.store %%{closureSSA}, %%closure_slot"

    // Allocate result slot
    emit $"  %%result_slot = llvm.alloca 1 x {resultType}"
    emit $"  %%result_slot_ptr = llvm.getelementptr %%{handleSSA}[0, 1]"
    emit $"  llvm.store %%result_slot, %%result_slot_ptr"

    // pthread_create
    emit $"  %%os_handle_ptr = llvm.getelementptr %%{handleSSA}[0, 0]"
    emit $"  llvm.call @pthread_create(%%os_handle_ptr, ptr null, @thread_entry, %%{handleSSA})"

    TRValue { SSA = handleSSA; Type = TThread resultType }
```

### 4.4 Thread.join Witness

```fsharp
let witnessThreadJoin z handleSSA =
    let resultSSA = freshSSA ()

    // Get OS handle
    emit $"  %%os_handle_ptr = llvm.getelementptr %%{handleSSA}[0, 0]"
    emit "  %os_handle = llvm.load %os_handle_ptr : i64"

    // pthread_join
    emit "  llvm.call @pthread_join(%os_handle, ptr null)"

    // Read result
    emit $"  %%result_slot_ptr = llvm.getelementptr %%{handleSSA}[0, 1]"
    emit "  %result_slot = llvm.load %result_slot_ptr : !llvm.ptr"
    emit $"  %%{resultSSA} = llvm.load %%result_slot"

    TRValue { SSA = resultSSA; Type = resultType }
```

## 5. MLIR Output Specification

### 5.1 Thread Handle Type

```mlir
!thread_handle = !llvm.struct<(
    i64,         // OS handle (pthread_t)
    ptr,         // result slot pointer
    !closure_type // closure to run
)>
```

### 5.2 Thread Entry Wrapper

```mlir
llvm.func @thread_entry(%arg: !llvm.ptr) -> !llvm.ptr {
    // Load closure from handle
    %closure_ptr = llvm.getelementptr %arg[0, 2]
    %closure = llvm.load %closure_ptr : !closure_type

    // Extract and call
    %code = llvm.extractvalue %closure[0]
    %env = llvm.extractvalue %closure[1]
    %result = llvm.call %code(%env) : (!llvm.ptr) -> i32

    // Store result
    %result_slot_ptr = llvm.getelementptr %arg[0, 1]
    %result_slot = llvm.load %result_slot_ptr : !llvm.ptr
    llvm.store %result, %result_slot

    llvm.return %null : !llvm.ptr
}
```

### 5.3 Thread Creation

```mlir
// let thread = Thread.create myFunc
%handle = llvm.alloca 1 x !thread_handle

// Store closure
%closure_slot = llvm.getelementptr %handle[0, 2]
llvm.store %myFunc_closure, %closure_slot

// Allocate result slot
%result_slot = llvm.alloca 1 x i32
%result_ptr = llvm.getelementptr %handle[0, 1]
llvm.store %result_slot, %result_ptr

// Create thread
%os_handle_ptr = llvm.getelementptr %handle[0, 0]
llvm.call @pthread_create(%os_handle_ptr, %null, @thread_entry, %handle)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicThreadSample

let work (id: int) () =
    Console.write "Thread "
    Console.write (Format.int id)
    Console.writeln " starting"
    // Simulate work
    let mutable sum = 0
    for i in 1..1000000 do
        sum <- sum + i
    Console.write "Thread "
    Console.write (Format.int id)
    Console.write " done, sum = "
    Console.writeln (Format.int sum)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Thread Test ==="

    let t1 = Thread.create (work 1)
    let t2 = Thread.create (work 2)
    let t3 = Thread.create (work 3)

    Console.writeln "All threads created, waiting..."

    Thread.join t1
    Thread.join t2
    Thread.join t3

    Console.writeln "All threads completed"
    0
```

### 6.2 Expected Output (Order May Vary)

```
=== Basic Thread Test ===
All threads created, waiting...
Thread 1 starting
Thread 2 starting
Thread 3 starting
Thread 2 done, sum = 500000500000
Thread 1 done, sum = 500000500000
Thread 3 done, sum = 500000500000
All threads completed
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TThread type |
| `CheckExpressions.fs` | MODIFY | Add Thread intrinsics |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/ThreadWitness.fs` | CREATE | Thread operation witnesses |
| `src/Alex/Bindings/Thread_Linux_x86_64.fs` | CREATE | pthread bindings |
| `src/Alex/Bindings/Thread_Windows_x86_64.fs` | CREATE | Windows thread bindings |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TThread type
- [ ] Add Thread.create/join/detach intrinsics

### Phase 2: Alex Implementation
- [ ] Create thread entry wrapper
- [ ] Implement Thread.create witness
- [ ] Implement Thread.join witness
- [ ] Add platform bindings

### Phase 3: Validation
- [ ] Sample 27 compiles
- [ ] Threads run concurrently
- [ ] Join waits correctly
- [ ] Samples 01-26 still pass

## 9. Platform Bindings

| Operation | Linux | Windows |
|-----------|-------|---------|
| Create | pthread_create | CreateThread |
| Join | pthread_join | WaitForSingleObject |
| Detach | pthread_detach | CloseHandle |

## 10. Related PRDs

- **C-01**: Closures - Thread functions are closures
- **T-02**: Mutex - Synchronization for shared state
- **T-03-31**: MailboxProcessor - Threaded message processing
