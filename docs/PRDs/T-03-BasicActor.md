# T-03: MailboxProcessor Basic Actor

> **Sample**: `29_BasicActor` | **Status**: Planned | **Depends On**: C-01-28 (All Prior Features)

## 1. Executive Summary

MailboxProcessor is the **capstone feature** of the WREN Stack - it synthesizes closures, async, threading, mutex synchronization, and regions into a single coherent abstraction. An actor is a concurrent unit with a private mailbox that processes messages sequentially.

**Key Insight**: MailboxProcessor is a composition, not a primitive. It emerges from combining existing capabilities:
- Thread (T-01) for concurrent execution
- Mutex + CondVar (T-02) for message queue synchronization
- Async (A-01-19) for message loop coroutine
- Closures (C-01) for behavior function capture
- Regions (A-04 to A-06) for per-batch memory management

**Reference**: See `mailboxprocessor_first_stage` memory for implementation strategy.

## 2. Language Feature Specification

### 2.1 Actor Creation

```fsharp
let counter = MailboxProcessor.Start(fun inbox ->
    let rec loop count = async {
        let! msg = inbox.Receive()
        match msg with
        | Increment -> return! loop (count + 1)
        | Get reply -> reply.Reply count; return! loop count
    }
    loop 0)
```

### 2.2 Posting Messages

```fsharp
counter.Post(Increment)
counter.Post(Increment)
counter.Post(Increment)
```

`Post` is asynchronous - it enqueues and returns immediately.

### 2.3 Message Type

```fsharp
type CounterMessage =
    | Increment
    | Get of AsyncReplyChannel<int>
```

Messages are discriminated unions (already supported via F-05+).

## 3. FNCS Layer Implementation

### 3.1 MailboxProcessor Type

```fsharp
// In NativeTypes.fs
| TMailboxProcessor of messageType: NativeType
```

### 3.2 Inbox Type

```fsharp
// The inbox passed to behavior function
| TInbox of messageType: NativeType
```

### 3.3 MailboxProcessor Intrinsics

```fsharp
// In CheckExpressions.fs
| "MailboxProcessor.Start" ->
    // (Inbox<'Msg> -> Async<unit>) -> MailboxProcessor<'Msg>
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(
            NativeType.TInbox(msgVar),
            NativeType.TAsync(env.Globals.UnitType)),
        NativeType.TMailboxProcessor(msgVar))

| "MailboxProcessor.Post" ->
    // MailboxProcessor<'Msg> -> 'Msg -> unit
    // Note: 'this' parameter in member syntax
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TMailboxProcessor(msgVar),
        NativeType.TFun(msgVar, env.Globals.UnitType))

| "Inbox.Receive" ->
    // Inbox<'Msg> -> Async<'Msg>
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TInbox(msgVar),
        NativeType.TAsync(msgVar))
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Actor Structure

```fsharp
type MailboxProcessor<'Msg> = {
    Queue: MessageQueue<'Msg>  // Thread-safe queue
    Thread: ThreadHandle       // Worker thread
    Behavior: Inbox<'Msg> -> Async<unit>  // Closure
}

type MessageQueue<'Msg> = {
    Head: nativeptr<MessageNode<'Msg>>
    Tail: nativeptr<MessageNode<'Msg>>
    Mutex: Mutex
    CondVar: CondVar
}

type MessageNode<'Msg> = {
    Next: nativeptr<MessageNode<'Msg>>
    Data: 'Msg
}
```

### 4.2 MailboxProcessor.Start Witness

```fsharp
let witnessMailboxProcessorStart z behaviorClosureSSA =
    let actorSSA = freshSSA ()

    // 1. Allocate MailboxProcessor struct
    emit $"  %%{actorSSA} = llvm.alloca 1 x !mailbox_processor"

    // 2. Initialize message queue
    emit "  %queue_ptr = llvm.getelementptr %actor[0, 0]"
    emit "  llvm.call @queue_init(%queue_ptr)"

    // 3. Store behavior closure
    emit "  %behavior_ptr = llvm.getelementptr %actor[0, 2]"
    emit $"  llvm.store %%{behaviorClosureSSA}, %%behavior_ptr"

    // 4. Create worker thread
    emit "  %thread_ptr = llvm.getelementptr %actor[0, 1]"
    emit $"  llvm.call @pthread_create(%%thread_ptr, ptr null, @actor_loop, %%{actorSSA})"

    TRValue { SSA = actorSSA; Type = TMailboxProcessor msgType }
```

### 4.3 Post Witness

```fsharp
let witnessPost z actorSSA messageSSA =
    // 1. Allocate message node
    emit "  %node = llvm.alloca 1 x !message_node"
    emit "  %data_ptr = llvm.getelementptr %node[0, 1]"
    emit $"  llvm.store %%{messageSSA}, %%data_ptr"

    // 2. Get queue
    emit "  %queue_ptr = llvm.getelementptr %actor[0, 0]"

    // 3. Lock, enqueue, signal, unlock
    emit "  %mutex = llvm.getelementptr %queue_ptr[0, 2]"
    emit "  llvm.call @pthread_mutex_lock(%mutex)"

    // Enqueue at tail
    emit "  %tail_ptr = llvm.getelementptr %queue_ptr[0, 1]"
    emit "  %old_tail = llvm.load %tail_ptr"
    emit "  %next_ptr = llvm.getelementptr %old_tail[0, 0]"
    emit "  llvm.store %node, %next_ptr"
    emit "  llvm.store %node, %tail_ptr"

    // Signal waiter
    emit "  %cond = llvm.getelementptr %queue_ptr[0, 3]"
    emit "  llvm.call @pthread_cond_signal(%cond)"

    emit "  llvm.call @pthread_mutex_unlock(%mutex)"

    TRVoid
```

### 4.4 Actor Loop (Worker Thread)

```fsharp
let emitActorLoop z actorSSA =
    // Entry point for worker thread
    emit "llvm.func @actor_loop(%actor: !llvm.ptr) -> !llvm.ptr {"

    // Get queue and behavior
    emit "  %queue = llvm.getelementptr %actor[0, 0]"
    emit "  %behavior_ptr = llvm.getelementptr %actor[0, 2]"
    emit "  %behavior = llvm.load %behavior_ptr : !closure_type"

    // Create inbox (points to queue)
    emit "  %inbox = %queue"

    // Start behavior coroutine
    emit "  %code = llvm.extractvalue %behavior[0]"
    emit "  %env = llvm.extractvalue %behavior[1]"
    emit "  llvm.call %code(%env, %inbox)"

    emit "  llvm.return %null"
    emit "}"
```

### 4.5 Inbox.Receive Witness

```fsharp
let witnessInboxReceive z inboxSSA =
    let resultSSA = freshSSA ()

    // Lock queue
    emit "  %mutex = llvm.getelementptr %inbox[0, 2]"
    emit "  llvm.call @pthread_mutex_lock(%mutex)"

    // Wait while empty
    emit "  llvm.br ^check"
    emit "^check:"
    emit "  %head_ptr = llvm.getelementptr %inbox[0, 0]"
    emit "  %head = llvm.load %head_ptr"
    emit "  %is_empty = llvm.icmp eq %head, %null"
    emit "  llvm.cond_br %is_empty, ^wait, ^dequeue"

    emit "^wait:"
    emit "  %cond = llvm.getelementptr %inbox[0, 3]"
    emit "  llvm.call @pthread_cond_wait(%cond, %mutex)"
    emit "  llvm.br ^check"

    emit "^dequeue:"
    // Dequeue from head
    emit "  %next_ptr = llvm.getelementptr %head[0, 0]"
    emit "  %next = llvm.load %next_ptr"
    emit "  llvm.store %next, %head_ptr"
    emit "  %data_ptr = llvm.getelementptr %head[0, 1]"
    emit $"  %%{resultSSA} = llvm.load %%data_ptr"

    // Unlock
    emit "  llvm.call @pthread_mutex_unlock(%mutex)"

    TRValue { SSA = resultSSA; Type = msgType }
```

## 5. MLIR Output Specification

### 5.1 Actor Types

```mlir
!message_node = !llvm.struct<(ptr, i32)>  // next, data (example: int message)

!message_queue = !llvm.struct<(
    ptr,     // head
    ptr,     // tail
    !mutex,  // mutex
    !condvar // condvar
)>

!mailbox_processor = !llvm.struct<(
    !message_queue,   // queue
    i64,              // thread handle
    !closure_type     // behavior
)>
```

### 5.2 Start Implementation

```mlir
// MailboxProcessor.Start(behavior)
%actor = llvm.alloca 1 x !mailbox_processor

// Initialize queue
%queue = llvm.getelementptr %actor[0, 0]
llvm.call @queue_init(%queue)

// Store behavior
%behavior_slot = llvm.getelementptr %actor[0, 2]
llvm.store %behavior_closure, %behavior_slot

// Create thread
%thread_slot = llvm.getelementptr %actor[0, 1]
llvm.call @pthread_create(%thread_slot, %null, @actor_loop, %actor)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicActorSample

type Message =
    | Increment
    | Decrement
    | Print

let counter = MailboxProcessor.Start(fun inbox ->
    let rec loop count = async {
        let! msg = inbox.Receive()
        match msg with
        | Increment ->
            Console.writeln "Incrementing"
            return! loop (count + 1)
        | Decrement ->
            Console.writeln "Decrementing"
            return! loop (count - 1)
        | Print ->
            Console.write "Count: "
            Console.writeln (Format.int count)
            return! loop count
    }
    loop 0)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Actor Test ==="

    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Print)
    counter.Post(Decrement)
    counter.Post(Print)
    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Print)

    // Give actor time to process
    Thread.sleep 100

    Console.writeln "Done"
    0
```

### 6.2 Expected Output

```
=== Basic Actor Test ===
Incrementing
Incrementing
Count: 2
Decrementing
Count: 1
Incrementing
Incrementing
Count: 3
Done
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TMailboxProcessor, TInbox |
| `CheckExpressions.fs` | MODIFY | Add MailboxProcessor intrinsics |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/ActorWitness.fs` | CREATE | MailboxProcessor witnesses |
| `src/Alex/CodeGeneration/ActorTypes.fs` | CREATE | Actor struct type generation |

## 8. Implementation Checklist

### Phase 1: Core Types
- [ ] Add TMailboxProcessor, TInbox types
- [ ] Add Start, Post, Receive intrinsics

### Phase 2: Queue Implementation
- [ ] Implement thread-safe message queue
- [ ] Implement enqueue (Post)
- [ ] Implement dequeue (Receive)

### Phase 3: Actor Loop
- [ ] Implement worker thread entry
- [ ] Integrate with async coroutine

### Phase 4: Validation
- [ ] Sample 29 compiles
- [ ] Messages process in order
- [ ] Actor runs on separate thread
- [ ] Samples 01-28 still pass

## 9. Why This Is the Capstone

MailboxProcessor demonstrates mastery of:

| Capability | PRD | Usage in Actor |
|------------|-----|----------------|
| Closures | 11 | Behavior function |
| HOFs | 12 | Recursive loop |
| Recursion | 13 | `let rec loop` |
| Async | 17-19 | `async { }` body |
| Regions | 20-22 | Message batch memory |
| Threading | 27 | Worker thread |
| Mutex | 28 | Queue synchronization |
| DUs | 05 | Message types |

## 10. Related PRDs

- **T-04**: PostAndReply - Two-way communication
- **T-05**: ParallelActors - Multiple interacting actors
