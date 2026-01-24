# T-04: Actor PostAndReply

> **Sample**: `30_ActorReply` | **Status**: Planned | **Depends On**: T-03 (BasicActor)

## 1. Executive Summary

This PRD adds two-way communication to MailboxProcessor via `PostAndReply`. The caller sends a message and blocks until the actor replies. This enables request-response patterns.

**Key Insight**: `PostAndReply` bundles a reply channel with the message. The reply channel contains a condvar that the caller waits on and the actor signals.

## 2. Language Feature Specification

### 2.1 PostAndReply

```fsharp
let result = actor.PostAndReply(fun reply ->
    GetValue reply)
```

The lambda receives an `AsyncReplyChannel<'Reply>` and constructs the message.

### 2.2 AsyncReplyChannel

```fsharp
type AsyncReplyChannel<'Reply> =
    member Reply : 'Reply -> unit
```

The actor calls `reply.Reply(value)` to send the response.

### 2.3 Complete Example

```fsharp
type BankMessage =
    | Deposit of int
    | Withdraw of int * AsyncReplyChannel<Result<int, string>>
    | GetBalance of AsyncReplyChannel<int>

let bank = MailboxProcessor.Start(fun inbox ->
    let rec loop balance = async {
        let! msg = inbox.Receive()
        match msg with
        | Deposit amount ->
            return! loop (balance + amount)
        | Withdraw (amount, reply) ->
            if amount > balance then
                reply.Reply(Error "Insufficient funds")
                return! loop balance
            else
                reply.Reply(Ok (balance - amount))
                return! loop (balance - amount)
        | GetBalance reply ->
            reply.Reply(balance)
            return! loop balance
    }
    loop 0)

// Usage
bank.Post(Deposit 100)
let balance = bank.PostAndReply(fun reply -> GetBalance reply)
```

## 3. FNCS Layer Implementation

### 3.1 AsyncReplyChannel Type

```fsharp
// In NativeTypes.fs
| TAsyncReplyChannel of replyType: NativeType
```

### 3.2 PostAndReply Intrinsic

```fsharp
// In CheckExpressions.fs
| "MailboxProcessor.PostAndReply" ->
    // MailboxProcessor<'Msg> -> (AsyncReplyChannel<'Reply> -> 'Msg) -> 'Reply
    let msgVar = freshTypeVar ()
    let replyVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TMailboxProcessor(msgVar),
        NativeType.TFun(
            NativeType.TFun(NativeType.TAsyncReplyChannel(replyVar), msgVar),
            replyVar))

| "AsyncReplyChannel.Reply" ->
    // AsyncReplyChannel<'Reply> -> 'Reply -> unit
    let replyVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TAsyncReplyChannel(replyVar),
        NativeType.TFun(replyVar, env.Globals.UnitType))
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Reply Channel Structure

```fsharp
type AsyncReplyChannel<'Reply> = {
    ResultSlot: nativeptr<'Reply>  // Where result is stored
    Mutex: Mutex                    // Protects completion state
    CondVar: CondVar                // Signals completion
    Completed: bool                 // Has Reply been called?
}
```

### 4.2 PostAndReply Witness

```fsharp
let witnessPostAndReply z actorSSA msgBuilderClosureSSA =
    let resultSSA = freshSSA ()

    // 1. Allocate reply channel
    emit "  %channel = llvm.alloca 1 x !reply_channel"
    emit "  %result_slot = llvm.alloca 1 x %reply_type"
    emit "  %slot_ptr = llvm.getelementptr %channel[0, 0]"
    emit "  llvm.store %result_slot, %slot_ptr"

    // Initialize mutex and condvar
    emit "  %mutex = llvm.getelementptr %channel[0, 1]"
    emit "  llvm.call @pthread_mutex_init(%mutex, %null)"
    emit "  %cond = llvm.getelementptr %channel[0, 2]"
    emit "  llvm.call @pthread_cond_init(%cond, %null)"
    emit "  %completed = llvm.getelementptr %channel[0, 3]"
    emit "  llvm.store %false, %completed"

    // 2. Call message builder with channel
    emit "  %code = llvm.extractvalue %msgBuilder[0]"
    emit "  %env = llvm.extractvalue %msgBuilder[1]"
    emit "  %message = llvm.call %code(%env, %channel)"

    // 3. Post message
    emit "  llvm.call @actor_post(%actor, %message)"

    // 4. Wait for reply
    emit "  llvm.call @pthread_mutex_lock(%mutex)"
    emit "  llvm.br ^wait_check"

    emit "^wait_check:"
    emit "  %is_done = llvm.load %completed : i1"
    emit "  llvm.cond_br %is_done, ^done, ^wait"

    emit "^wait:"
    emit "  llvm.call @pthread_cond_wait(%cond, %mutex)"
    emit "  llvm.br ^wait_check"

    emit "^done:"
    emit "  llvm.call @pthread_mutex_unlock(%mutex)"

    // 5. Read result
    emit $"  %%{resultSSA} = llvm.load %%result_slot"

    // 6. Cleanup
    emit "  llvm.call @pthread_mutex_destroy(%mutex)"
    emit "  llvm.call @pthread_cond_destroy(%cond)"

    TRValue { SSA = resultSSA; Type = replyType }
```

### 4.3 Reply Witness

```fsharp
let witnessReply z channelSSA valueSSA =
    // Store result
    emit "  %slot_ptr = llvm.getelementptr %channel[0, 0]"
    emit "  %slot = llvm.load %slot_ptr"
    emit $"  llvm.store %%{valueSSA}, %%slot"

    // Mark completed and signal
    emit "  %mutex = llvm.getelementptr %channel[0, 1]"
    emit "  llvm.call @pthread_mutex_lock(%mutex)"
    emit "  %completed = llvm.getelementptr %channel[0, 3]"
    emit "  llvm.store %true, %completed"
    emit "  %cond = llvm.getelementptr %channel[0, 2]"
    emit "  llvm.call @pthread_cond_signal(%cond)"
    emit "  llvm.call @pthread_mutex_unlock(%mutex)"

    TRVoid
```

## 5. MLIR Output Specification

### 5.1 Reply Channel Type

```mlir
!reply_channel = !llvm.struct<(
    ptr,      // result slot
    !mutex,   // mutex
    !condvar, // condvar
    i1        // completed
)>
```

### 5.2 PostAndReply Implementation

```mlir
// result = actor.PostAndReply(fun reply -> GetBalance reply)

// Allocate channel
%channel = llvm.alloca 1 x !reply_channel
%result_slot = llvm.alloca 1 x i32
%slot_ptr = llvm.getelementptr %channel[0, 0]
llvm.store %result_slot, %slot_ptr

// Init sync primitives
%mutex = llvm.getelementptr %channel[0, 1]
llvm.call @pthread_mutex_init(%mutex, %null)
%cond = llvm.getelementptr %channel[0, 2]
llvm.call @pthread_cond_init(%cond, %null)
%completed = llvm.getelementptr %channel[0, 3]
llvm.store %false, %completed

// Build and post message
%msg = llvm.call %msgBuilder_code(%msgBuilder_env, %channel)
llvm.call @actor_post(%actor, %msg)

// Wait for reply
llvm.call @pthread_mutex_lock(%mutex)
llvm.br ^check

^check:
    %done = llvm.load %completed : i1
    llvm.cond_br %done, ^got_reply, ^wait

^wait:
    llvm.call @pthread_cond_wait(%cond, %mutex)
    llvm.br ^check

^got_reply:
    llvm.call @pthread_mutex_unlock(%mutex)
    %result = llvm.load %result_slot : i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module ActorReplySample

type CalcMessage =
    | Add of int * int * AsyncReplyChannel<int>
    | Multiply of int * int * AsyncReplyChannel<int>

let calculator = MailboxProcessor.Start(fun inbox ->
    async {
        while true do
            let! msg = inbox.Receive()
            match msg with
            | Add (a, b, reply) ->
                reply.Reply(a + b)
            | Multiply (a, b, reply) ->
                reply.Reply(a * b)
    })

[<EntryPoint>]
let main _ =
    Console.writeln "=== Actor Reply Test ==="

    let sum = calculator.PostAndReply(fun reply -> Add(10, 20, reply))
    Console.write "10 + 20 = "
    Console.writeln (Format.int sum)

    let product = calculator.PostAndReply(fun reply -> Multiply(6, 7, reply))
    Console.write "6 * 7 = "
    Console.writeln (Format.int product)

    let sum2 = calculator.PostAndReply(fun reply -> Add(sum, product, reply))
    Console.write "(10+20) + (6*7) = "
    Console.writeln (Format.int sum2)

    0
```

### 6.2 Expected Output

```
=== Actor Reply Test ===
10 + 20 = 30
6 * 7 = 42
(10+20) + (6*7) = 72
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TAsyncReplyChannel |
| `CheckExpressions.fs` | MODIFY | Add PostAndReply, Reply intrinsics |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/ActorWitness.fs` | MODIFY | Add PostAndReply, Reply witnesses |

## 8. Implementation Checklist

### Phase 1: Reply Channel
- [ ] Add TAsyncReplyChannel type
- [ ] Implement Reply intrinsic
- [ ] Generate reply channel struct

### Phase 2: PostAndReply
- [ ] Implement PostAndReply intrinsic
- [ ] Implement blocking wait
- [ ] Implement channel cleanup

### Phase 3: Validation
- [ ] Sample 30 compiles
- [ ] Replies return correct values
- [ ] No deadlocks or races
- [ ] Samples 01-29 still pass

## 9. Timeout Support (Future)

```fsharp
// With timeout
let result = actor.TryPostAndReply(
    (fun reply -> GetValue reply),
    timeout = 1000)  // ms
// Returns: Some value | None (timeout)
```

This requires timer integration - deferred to future work.

## 10. Related PRDs

- **T-03**: BasicActor - Foundation
- **T-05**: ParallelActors - Multiple actors with replies
