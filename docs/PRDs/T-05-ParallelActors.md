# T-05: Parallel Actors (Capstone)

> **Sample**: `31_ParallelActors` | **Status**: Planned | **Depends On**: T-03-30 (BasicActor, ActorReply)

## 1. Executive Summary

This PRD demonstrates **multiple interacting actors** - the full actor model in action. Multiple actors run concurrently, communicate via messages, and coordinate to solve problems.

**Key Insight**: This is the ultimate validation of the WREN Stack. If multiple actors can run, communicate, and produce correct results, then closures, async, threading, synchronization, and regions all work correctly together.

## 2. Language Feature Specification

### 2.1 Multiple Actors

```fsharp
let worker1 = MailboxProcessor.Start(...)
let worker2 = MailboxProcessor.Start(...)
let coordinator = MailboxProcessor.Start(...)
```

### 2.2 Inter-Actor Communication

```fsharp
// Coordinator sends work to workers
worker1.Post(ProcessChunk chunk1)
worker2.Post(ProcessChunk chunk2)

// Workers reply to coordinator
coordinator.Post(ChunkComplete workerId result)
```

### 2.3 Actor Hierarchy

```fsharp
// Parent creates children
let children = Array.init 4 (fun i ->
    MailboxProcessor.Start(workerBehavior i))

// Parent distributes work
for i, child in Array.indexed children do
    child.Post(Work workItems.[i])
```

## 3. Sample: Parallel Sum

### 3.1 Architecture

```
          ┌─────────────┐
          │ Coordinator │
          └──────┬──────┘
       ┌─────────┼─────────┐
       │         │         │
   ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
   │Worker1│ │Worker2│ │Worker3│
   └───────┘ └───────┘ └───────┘
```

### 3.2 Messages

```fsharp
type WorkerMessage =
    | ComputeSum of data: int[] * replyTo: MailboxProcessor<CoordinatorMessage>

type CoordinatorMessage =
    | PartialSum of workerId: int * sum: int
    | GetTotal of AsyncReplyChannel<int>
```

### 3.3 Worker Behavior

```fsharp
let workerBehavior (workerId: int) (inbox: Inbox<WorkerMessage>) = async {
    while true do
        let! msg = inbox.Receive()
        match msg with
        | ComputeSum (data, replyTo) ->
            let sum = Array.fold (+) 0 data
            replyTo.Post(PartialSum (workerId, sum))
}
```

### 3.4 Coordinator Behavior

```fsharp
let coordinatorBehavior (inbox: Inbox<CoordinatorMessage>) = async {
    let mutable totalSum = 0
    let mutable completed = 0
    let expectedWorkers = 4

    while true do
        let! msg = inbox.Receive()
        match msg with
        | PartialSum (workerId, sum) ->
            Console.write "Worker "
            Console.write (Format.int workerId)
            Console.write " computed: "
            Console.writeln (Format.int sum)
            totalSum <- totalSum + sum
            completed <- completed + 1
        | GetTotal reply ->
            reply.Reply(totalSum)
}
```

## 4. FNCS Layer Implementation

No new FNCS features needed beyond T-03-30. This sample validates composition of existing features.

## 5. Firefly/Alex Layer Implementation

No new Alex features needed. This sample validates that multiple actors work correctly.

## 6. Validation

### 6.1 Sample Code

```fsharp
module ParallelActorsSample

type WorkerMessage =
    | ComputeSum of data: int[] * replyTo: MailboxProcessor<CoordinatorMessage>
    | Shutdown

and CoordinatorMessage =
    | PartialSum of workerId: int * sum: int
    | GetTotal of AsyncReplyChannel<int>
    | AllComplete

let workerBehavior (workerId: int) (inbox: Inbox<WorkerMessage>) = async {
    let mutable running = true
    while running do
        let! msg = inbox.Receive()
        match msg with
        | ComputeSum (data, replyTo) ->
            // Simulate work
            let mutable sum = 0
            for x in data do
                sum <- sum + x
            replyTo.Post(PartialSum (workerId, sum))
        | Shutdown ->
            running <- false
}

let coordinatorBehavior (numWorkers: int) (inbox: Inbox<CoordinatorMessage>) = async {
    let mutable totalSum = 0
    let mutable completed = 0

    while completed < numWorkers do
        let! msg = inbox.Receive()
        match msg with
        | PartialSum (workerId, sum) ->
            Console.write "Worker "
            Console.write (Format.int workerId)
            Console.write " result: "
            Console.writeln (Format.int sum)
            totalSum <- totalSum + sum
            completed <- completed + 1
        | GetTotal reply ->
            reply.Reply(totalSum)
        | AllComplete -> ()

    // Now handle GetTotal requests
    while true do
        let! msg = inbox.Receive()
        match msg with
        | GetTotal reply -> reply.Reply(totalSum)
        | _ -> ()
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Parallel Actors Test ==="

    let numWorkers = 4
    let dataSize = 1000

    // Create coordinator
    let coordinator = MailboxProcessor.Start(coordinatorBehavior numWorkers)

    // Create workers
    let workers = Array.init numWorkers (fun i ->
        MailboxProcessor.Start(workerBehavior i))

    // Prepare data chunks
    let fullData = Array.init dataSize (fun i -> i + 1)
    let chunkSize = dataSize / numWorkers
    let chunks = Array.init numWorkers (fun i ->
        Array.sub fullData (i * chunkSize) chunkSize)

    Console.writeln "Distributing work..."

    // Distribute work
    for i in 0..(numWorkers - 1) do
        workers.[i].Post(ComputeSum (chunks.[i], coordinator))

    // Wait a bit for processing
    Thread.sleep 500

    // Get total
    let total = coordinator.PostAndReply(fun reply -> GetTotal reply)

    Console.write "Total sum: "
    Console.writeln (Format.int total)

    // Expected: 1+2+3+...+1000 = 500500
    Console.write "Expected: "
    Console.writeln (Format.int 500500)

    // Shutdown workers
    for worker in workers do
        worker.Post(Shutdown)

    Console.writeln "All workers shutdown"
    0
```

### 6.2 Expected Output

```
=== Parallel Actors Test ===
Distributing work...
Worker 0 result: 125250
Worker 1 result: 125250
Worker 2 result: 125250
Worker 3 result: 125250
Total sum: 500500
Expected: 500500
All workers shutdown
```

(Note: Worker completion order may vary)

## 7. Files to Create/Modify

No new files needed. Sample source demonstrates actor composition.

## 8. Implementation Checklist

### Phase 1: Verify Prerequisites
- [ ] Sample 29 (BasicActor) passes
- [ ] Sample 30 (ActorReply) passes
- [ ] Multiple threads work correctly

### Phase 2: Sample Implementation
- [ ] Implement worker actors
- [ ] Implement coordinator actor
- [ ] Implement work distribution
- [ ] Implement result aggregation

### Phase 3: Validation
- [ ] Sample 31 compiles
- [ ] Workers run in parallel
- [ ] Results aggregate correctly
- [ ] No races or deadlocks
- [ ] Samples 01-30 still pass

## 9. What This Proves

If Sample 31 passes, the WREN Stack has proven:

| Capability | Status |
|------------|--------|
| Closures | Actor behaviors are closures |
| Higher-Order Functions | Message handlers use pattern matching |
| Recursion | Async loops are recursive |
| Async | Actor loops use async |
| Threading | Each actor runs on its thread |
| Mutex/CondVar | Message queues are synchronized |
| Regions | Message memory is managed |
| DUs | Messages are discriminated unions |
| **Composition** | All features work together |

## 10. The WREN Stack Vision Realized

```
┌────────────────────────────────────────┐
│              WREN Stack                │
├────────────────────────────────────────┤
│  W - WebView (D-02)                  │
│  R - Regions (A-04 to A-06)               │
│  E - Elmish (via Actors)               │
│  N - Native (all of Fidelity)          │
├────────────────────────────────────────┤
│         MailboxProcessor               │
│   The Capstone Abstraction             │
├────────────────────────────────────────┤
│  Threads │ Async │ Closures │ DUs     │
│  Mutex   │ Cond  │ Regions  │ Types   │
└────────────────────────────────────────┘
```

Sample 31 validates that functional programming + actor model + native compilation work together without a managed runtime.

## 11. Related PRDs

- **T-03**: BasicActor - Foundation
- **T-04**: ActorReply - Two-way communication
- (Future): Supervision trees, fault tolerance, distributed actors
