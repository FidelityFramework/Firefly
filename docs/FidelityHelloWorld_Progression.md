# FidelityHelloWorld Sample Progression

## Overview

The FidelityHelloWorld samples in `/samples/console/FidelityHelloWorld/` form a progressive test suite for Firefly compiler capabilities. Each sample builds on previous ones, proving specific F# features compile correctly to native code.

**Implementation Philosophy**: Each feature follows the Photographer Principle - nanopasses build the scene (enrich the PSG with coeffects), the Zipper moves attention through the graph, Active Patterns focus the semantic lens, and Transfer snaps the picture (emits MLIR via Templates). If you find yourself computing metadata during code generation, stop - that belongs in a nanopass.

## Current Samples (01-13)

### Working Samples

| Sample | Name | Features Proven | Status |
|--------|------|-----------------|--------|
| 01 | HelloWorldDirect | Static strings, direct Console.writeln calls | Working |
| 02 | HelloWorldSaturated | Let bindings, string literals, function calls | Working |
| 03 | HelloWorldHalfCurried | Pipe operators (`\|>`), partial application | Working |
| 04 | HelloWorldFullCurried | Full currying, Result.map, lambdas | Working |
| 05 | AddNumbers | Integer arithmetic, pattern matching on DUs | Working |
| 06 | AddNumbersInteractive | Console.readln, string parsing, interactive I/O | Working |
| 07 | BitsTest | Bit manipulation operators (`&&&`, `\|\|\|`, `^^^`, `<<<`, `>>>`) | Working |
| 08 | Option | Option type (Some/None), 2-element DU | Working |
| 09 | Result | Result type (Ok/Error), multi-payload DU | Working |

### Samples Needing Fixes

| Sample | Name | Issue | Root Cause |
|--------|------|-------|------------|
| 10 | Records | Record pattern matching not implemented | Compiler limitation |
| 11 | HigherOrderFunctions | Source code type errors | Source bugs (not compiler) |
| 12 | Closures | Source code type error | Source bugs (not compiler) |
| 13 | Recursion | Recursive function bindings not resolved | Compiler limitation |

## Planned Development Phases

### Phase A: Foundation Fixes

Fix existing samples that have compiler limitations.

#### A.1: Fix Sample 10 (Records)

**Problem**: Record pattern matching in match expressions not implemented.

**What needs to work**:
```fsharp
type Person = { Name: string; Age: int }

match person with
| { Name = n; Age = a } -> printfn "%s is %d" n a
```

**Implementation**: The `RecordPattern` Active Pattern (semantic lens) identifies record destructuring. The pattern's field bindings are coeffect-tagged during PSG construction. Alex witnesses each bound field and emits struct field extraction via the `FieldAccess` template - no string matching on field names.

#### A.2: Fix Sample 12 (Closures)

**Problem**: Source code has type errors.

**Action**: Review and fix source code, then verify compiler handles closures correctly. Closure capture is coeffect-tagged - the PSG knows which variables are captured. Alex witnesses the capture set and emits environment struct allocation.

#### A.3: Fix Sample 13 (Recursion)

**Problem**: Recursive function bindings not found in VarBindings.

**What needs to work**:
```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
```

**Implementation**: Recursive bindings require pre-binding the function name before processing the body. The `RecursiveBinding` coeffect marks self-references. Alex witnesses the recursion and emits a forward declaration followed by the body - the recursive call resolves to the same function symbol.

---

### Phase B: Sequences

Implement `seq` computation expression using MoveNext struct pattern.

The Seq iterator is codata - defined by observation (MoveNext/Current), not construction. The Zipper witnesses each `yield` point and the `SeqYield` template emits state machine transitions. No central seq dispatcher - each yield is a local state transition.

#### B.1: Sample 14 - SimpleSeq

**Features**:
- Basic `seq { }` builder
- `for ... in ... do yield` pattern
- Iterator state machine generation

**Source**:
```fsharp
let numbers = seq {
    for i in 1..10 do
        yield i
}

for n in numbers do
    Console.writeln (Format.int n)
```

**Implementation**: FNCS recognizes the Seq builder. A nanopass tags each `yield` with its state index (coeffect). Alex folds over the seq body - at each `YieldPoint` node, the `SeqStateMachine` template emits the state transition. The iterator struct holds `{ State: int; Current: 'T }`. For-in desugars to while + MoveNext.

#### B.2: Sample 15 - SeqOperations

**Features**:
- `Seq.map`
- `Seq.filter`
- `Seq.take`

**Source**:
```fsharp
let doubled = Seq.map (fun x -> x * 2) numbers
let evens = Seq.filter (fun x -> x % 2 = 0) numbers
let first5 = Seq.take 5 numbers
```

**Implementation**: Seq module operations are FNCS intrinsics. Each combinator wraps an upstream iterator. The `(|SeqCombinator|_|)` Active Pattern matches map/filter/take nodes. Alex witnesses the combinator and emits a composed iterator struct that delegates MoveNext to the upstream.

---

### Phase C: Lazy Evaluation

Implement `lazy` thunks - frozen computation that executes at most once.

Lazy is the simplest codata: you observe by forcing. The thunk either yields a cached value or runs the computation. No runtime scheduler, no GC - just a struct with a flag and a function pointer.

#### C.1: Sample 16 - LazyValues

**Features**:
- `lazy { }` expression
- `Lazy.force` / `.Value`
- Memoization

**Source**:
```fsharp
let expensive = lazy {
    Console.writeln "Computing..."
    42
}

let value1 = Lazy.force expensive  // Prints "Computing..."
let value2 = Lazy.force expensive  // No print, cached
```

**Implementation**: FNCS recognizes the Lazy builder. The lazy body becomes a thunk closure (coeffect-tagged with captures). Alex witnesses `LazyExpr` nodes and emits `{ Computed: bool; Value: 'T; Thunk: unit -> 'T }`. The `LazyForce` template emits: check flag, branch, call thunk, store result, set flag. Thread-safe version deferred.

---

### Phase D: Async (LLVM Coroutines)

Implement F# async using LLVM coroutine intrinsics - no runtime library.

Async is codata with suspension points. The coroutine frame is the "frozen computation" that resumes on demand. LLVM's `coro.*` intrinsics compile to state machines at compile time - zero runtime overhead.

See: [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md)

#### D.1: Sample 17 - BasicAsync

**Features**:
- `async { return value }`
- `Async.RunSynchronously`

**Source**:
```fsharp
let simple = async {
    return 42
}

let result = Async.RunSynchronously simple
Console.writeln (Format.int result)
```

**Implementation**: FNCS recognizes the Async builder. A trivial async (no suspension) compiles to an immediate return. The `(|AsyncReturn|_|)` Active Pattern matches the simple case. Alex witnesses and emits direct value return - no coroutine frame needed.

#### D.2: Sample 18 - AsyncAwait

**Features**:
- `let!` binding (await)
- Suspension points
- State machine with multiple states

**Source**:
```fsharp
let fetchData = async {
    return "data"
}

let process = async {
    let! data = fetchData
    return String.length data
}
```

**Implementation**: Each `let!` is a suspension point. A nanopass tags suspension points with state indices (coeffect). Alex folds over the async body - at each `AwaitPoint`, the `CoroSuspend` template emits `llvm.coro.suspend`. The frame struct captures variables live across suspension. Resume/cleanup branches handle continuation.

#### D.3: Sample 19 - AsyncParallel

**Features**:
- `Async.Parallel`
- Multiple concurrent asyncs

**Source**:
```fsharp
let task1 = async { return 1 }
let task2 = async { return 2 }
let task3 = async { return 3 }

let results = Async.Parallel [task1; task2; task3]
              |> Async.RunSynchronously
```

**Implementation**: `Async.Parallel` is an FNCS intrinsic. In single-threaded mode, it executes sequentially (parallel semantics, sequential execution). The `(|AsyncParallel|_|)` lens matches the combinator. Alex witnesses and emits a loop that runs each async to completion, collecting results.

---

### Phase E: Scoped Regions

Implement compiler-inferred deterministic memory regions - dynamic allocation without runtime overhead.

Region is a coeffect - the compiler knows Region-typed values need cleanup. Disposal flows from scope analysis, not interface dispatch. Alex witnesses Region nodes and emits platform-specific page allocation via Bindings (mmap/VirtualAlloc). The mise-en-place is complete before code generation - we just plate what the nanopasses prepared.

**Stack-first proof**: Phases B-D prove Seq/Lazy/Async with stack-only allocation. Regions unlock realistic I/O workloads.

#### E.1: Sample 20 - BasicRegion

**Features**:
- `Region.create` / `Region.alloc`
- Compiler-inferred disposal at scope exit
- Bump-pointer allocation

**Source**:
```fsharp
open Fidelity.Memory

let main () =
    let region = Region.create 4  // 4 pages initial

    // Allocate in region (fast bump-pointer)
    let buffer = Region.alloc<int> region 1000

    // Fill buffer
    for i in 0..999 do
        buffer.[i] <- i * 2

    // Sum values
    let mutable sum = 0
    for i in 0..999 do
        sum <- sum + buffer.[i]

    Console.writeln ("Sum: " + Format.int sum)

    // Compiler inserts: Region.release region
```

**Expected Output**:
```
Sum: 999000
```

**Implementation**: `Region` is an FNCS intrinsic type with coeffect `NeedsCleanup`. A nanopass performs scope analysis and tags each Region binding's exit points. Alex witnesses `RegionCreate` and emits via the `PageAlloc` Binding template (platform-aware: mmap on Linux, VirtualAlloc on Windows). At scope exits, the `PageFree` template emits deallocation. No `IDisposable`, no `use` - the compiler knows.

#### E.2: Sample 21 - RegionPassing

**Features**:
- Region passed to functions
- Caller scope determines lifetime
- Multiple allocations in same region

**Source**:
```fsharp
open Fidelity.Memory

let processData (r: Region) (size: int) =
    let temp = Region.alloc<float> r size
    for i in 0..(size-1) do
        temp.[i] <- float i * 1.5
    let sum = Array.fold (+) 0.0 temp
    sum

let main () =
    let region = Region.create 8

    let result1 = processData region 1000
    let result2 = processData region 500   // Same region, more allocation

    Console.writeln ("Result 1: " + Format.float result1)
    Console.writeln ("Result 2: " + Format.float result2)
    Console.writeln ("Used: " + Format.int (Region.usedBytes region) + " bytes")

    // Compiler inserts: Region.release region
```

**Implementation**: Region parameters carry the `BorrowedRegion` coeffect - the function can allocate but doesn't own lifetime. Alex witnesses allocations and emits bump-pointer arithmetic (increment offset, return pointer). The `RegionAlloc` template is parameterized by element type and count.

#### E.3: Sample 22 - RegionEscape

**Features**:
- `Region.copyOut` for escaping data
- Compiler prevents implicit escape

**Source**:
```fsharp
open Fidelity.Memory

let createResult (size: int) =
    let region = Region.create 2
    let data = Region.alloc<int> region size

    for i in 0..(size-1) do
        data.[i] <- i * i

    // Must explicitly copy to escape region
    let result = Region.copyOut data

    // Compiler inserts: Region.release region
    result  // Returns copy, not region-allocated data

let main () =
    let squares = createResult 10

    for i in 0..9 do
        Console.writeln (Format.int squares.[i])
```

**Implementation**: A nanopass performs escape analysis - tracking which region-allocated values flow to return positions. Implicit escape is a compile error. `Region.copyOut` is the explicit escape hatch. Alex witnesses `CopyOut` and emits memcpy to caller's context (stack or caller's region).

---

### Phase F: Networking

Implement socket operations for WebSocket support. With Regions available, we can allocate proper I/O buffers.

#### F.1: Sample 23 - SocketBasics

**Features**:
- `Sys.socket` / `Sys.connect` / `Sys.bind` / `Sys.listen` / `Sys.accept`
- `Sys.read` / `Sys.write` on socket FDs
- `Sys.close`

**Source**:
```fsharp
let server () =
    let region = Region.create 2
    let buffer = Region.alloc<byte> region 1024

    let sock = Sys.socket AF_INET SOCK_STREAM 0
    Sys.bind sock addr port
    Sys.listen sock 5
    let client = Sys.accept sock
    let bytesRead = Sys.read client buffer 1024
    Sys.write client response (String.length response)
    Sys.close client
    Sys.close sock
```

**Implementation**: Socket operations are FNCS Sys module intrinsics. The `(|SysCall|_|)` Active Pattern matches syscall nodes. Alex witnesses and emits via platform Bindings - each syscall number is data in the Binding, not routing logic. The buffer comes from a Region, giving proper I/O workspace.

#### F.2: Sample 24 - WebSocketEcho

**Features**:
- HTTP upgrade handshake
- WebSocket frame encoding/decoding
- Echo server loop

**Source**:
```fsharp
let wsServer () =
    let sock = acceptConnection ()
    WebSocket.handshake sock
    while true do
        let frame = WebSocket.readFrame sock
        WebSocket.writeFrame sock frame
```

**Implementation**: Uses Fidelity.Platform WebSocket module (Layer 3, built on Sys intrinsics). Frame buffers allocated in Regions. The WebSocket protocol logic lives in the library - Alex just witnesses the calls and emits via standard function call templates.

---

### Phase G: Desktop Scaffold

Implement GTK/WebView integration via FFI bindings.

#### G.1: Sample 25 - GTKWindow

**Features**:
- GTK initialization
- Window creation
- Event loop

**Source**:
```fsharp
open Fidelity.Desktop.GTK

let main () =
    GTK.init ()
    let window = GTK.windowNew "Hello GTK"
    GTK.windowShow window
    GTK.main ()
```

**Implementation**: GTK bindings are Layer 2 (Farscape-generated FFI). The `(|FFICall|_|)` Active Pattern matches external function calls. Alex witnesses and emits via the `ExternCall` template with the appropriate calling convention. No special GTK logic in the compiler.

#### G.2: Sample 26 - WebViewBasic

**Features**:
- WebKitGTK WebView widget
- HTML content loading
- JavaScript evaluation (optional)

**Source**:
```fsharp
open Fidelity.Desktop.GTK
open Fidelity.WebView

let main () =
    GTK.init ()
    let window = GTK.windowNew "WebView Demo"
    let webview = WebView.create ()
    WebView.loadHtml webview "<h1>Hello from Fidelity!</h1>"
    GTK.containerAdd window webview
    GTK.windowShow window
    GTK.main ()
```

**Implementation**: WebKitGTK FFI bindings from Fidelity.Platform. HTML string passed as native string (UTF-8 fat pointer). Widget hierarchy managed through GTK container API.

---

### Phase H: Threading Primitives

Implement OS threading for true parallelism.

Threading primitives are FNCS intrinsics that map to platform syscalls. The Thread coeffect marks functions that spawn threads - affecting what can be captured and how resources are managed.

#### H.1: Sample 27 - BasicThread

**Features**:
- `Thread.create` / `Thread.join`
- Parallel execution of computation

**Source**:
```fsharp
open Fidelity.Threading

let main () =
    let compute () =
        Console.writeln "Worker thread running"
        Thread.sleep 100
        Console.writeln "Worker thread done"

    Console.writeln "Main thread starting worker"
    let worker = Thread.create compute
    Console.writeln "Main thread waiting"
    Thread.join worker
    Console.writeln "All done"
```

**Expected Output**:
```
Main thread starting worker
Main thread waiting
Worker thread running
Worker thread done
All done
```

**Implementation**: `Thread.create` is an FNCS intrinsic. The closure passed becomes the thread entry point - its capture set is coeffect-tagged. Alex witnesses `ThreadCreate` and emits via platform Bindings: `pthread_create` on POSIX, `CreateThread` on Windows. The closure's environment struct is passed as the thread argument.

#### H.2: Sample 28 - MutexSync

**Features**:
- `Mutex.create` / `Mutex.lock` / `Mutex.unlock`
- Shared mutable state with synchronization

**Source**:
```fsharp
open Fidelity.Threading

let main () =
    let mutable counter = 0
    let mutex = Mutex.create ()

    let increment () =
        for _ in 1..1000 do
            Mutex.lock mutex
            counter <- counter + 1
            Mutex.unlock mutex

    let t1 = Thread.create increment
    let t2 = Thread.create increment
    Thread.join t1
    Thread.join t2

    Console.writeln (Format.int counter)  // Should be 2000
```

**Implementation**: Mutex operations are FNCS intrinsics with `SyncPrimitive` coeffect. Alex witnesses mutex nodes and emits platform-specific syscalls via Bindings. The mutable variable capture is coeffect-tagged as `SharedMutable` - the compiler knows it crosses thread boundaries.

---

### Phase I: MailboxProcessor (CAPSTONE)

**MailboxProcessor is the capstone feature** - it synthesizes all prior capabilities:
- Async (for message loop) via LLVM coroutines
- Closures (for behavior function capture)
- Threading (for true parallelism via OS threads)
- **Scoped Regions (for dynamic memory in worker threads)**
- Records/DUs (for message type definitions)

This proves the compiler can handle F#'s actor model primitive with full native compilation.

**Foundational Implementation**: OS thread per actor + mutex-protected queue + LLVM coroutine for async loop. No DCont, no Olivier/Prospero supervision - just the core actor semantics. This foundation works for desktop AND embedded/MCU/unikernel targets.

#### I.1: Sample 29 - BasicActor

**Features**:
- `MailboxProcessor.Start`
- `Post` (fire-and-forget)
- `Receive` in async loop

**Source**:
```fsharp
open Fidelity.Actors

type Message =
    | Greet of string
    | Shutdown

let main () =
    let actor = MailboxProcessor.Start(fun inbox ->
        let rec loop () = async {
            let! msg = inbox.Receive()
            match msg with
            | Greet name ->
                Console.writeln ("Hello, " + name + "!")
                return! loop ()
            | Shutdown ->
                Console.writeln "Shutting down"
                return ()
        }
        loop ()
    )

    actor.Post(Greet "Alice")
    actor.Post(Greet "Bob")
    actor.Post(Shutdown)
    Thread.sleep 100  // Give actor time to process
```

**Expected Output**:
```
Hello, Alice!
Hello, Bob!
Shutting down
```

**Implementation**: `MailboxProcessor.Start` is an FNCS intrinsic that synthesizes: Thread (for actor thread), Async (for message loop), Closure (for behavior), Queue (mutex + condvar + buffer). The `(|ActorStart|_|)` Active Pattern matches the Start call. Alex witnesses and emits via composition of existing templates - `ThreadCreate` for the actor thread, `CoroFrame` for the async loop, `MutexQueue` for the message buffer. The actor struct: `{ Thread; Queue; Behavior }`.

#### I.2: Sample 30 - ActorReply

**Features**:
- `PostAndReply` (request-response)
- `AsyncReplyChannel`
- Blocking wait for response

**Source**:
```fsharp
open Fidelity.Actors

type CounterMsg =
    | Increment
    | Decrement
    | GetValue of AsyncReplyChannel<int>

let main () =
    let counter = MailboxProcessor.Start(fun inbox ->
        let rec loop count = async {
            let! msg = inbox.Receive()
            match msg with
            | Increment ->
                return! loop (count + 1)
            | Decrement ->
                return! loop (count - 1)
            | GetValue reply ->
                reply.Reply(count)
                return! loop count
        }
        loop 0
    )

    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Decrement)

    let value = counter.PostAndReply(fun reply -> GetValue reply)
    Console.writeln ("Counter value: " + Format.int value)
```

**Expected Output**:
```
Counter value: 2
```

**Implementation**: `AsyncReplyChannel` is a struct with `{ Mutex; CondVar; ResultSlot }`. `PostAndReply` creates the channel, posts the message, and waits on the condvar. `Reply` fills the slot and signals. Alex witnesses `ReplyChannel` nodes and emits the synchronization struct via `SyncChannel` template.

#### I.3: Sample 31 - ParallelActors

**Features**:
- Multiple actors running in parallel
- Inter-actor communication via Post
- True parallelism (multiple OS threads)
- **Region-based worker memory**

**Source**:
```fsharp
open Fidelity.Actors
open Fidelity.Memory

type WorkerMsg =
    | Compute of int * MailboxProcessor<CoordinatorMsg>

type CoordinatorMsg =
    | Result of int

let main () =
    let coordinator = MailboxProcessor.Start(fun inbox ->
        let rec loop results = async {
            if List.length results >= 3 then
                let sum = List.fold (+) 0 results
                Console.writeln ("Total: " + Format.int sum)
                return ()
            else
                let! msg = inbox.Receive()
                match msg with
                | Result n -> return! loop (n :: results)
        }
        loop []
    )

    let createWorker id =
        MailboxProcessor.Start(fun inbox ->
            async {
                // Each worker has its own region for computation scratch space
                let region = Region.create 4
                let! msg = inbox.Receive()
                match msg with
                | Compute (n, reply) ->
                    let buffer = Region.alloc<int> region 100
                    // ... computation using buffer ...
                    let result = n * n
                    reply.Post(Result result)
                // Region released when actor terminates
            }
        )

    let w1 = createWorker 1
    let w2 = createWorker 2
    let w3 = createWorker 3

    w1.Post(Compute(10, coordinator))  // 100
    w2.Post(Compute(20, coordinator))  // 400
    w3.Post(Compute(30, coordinator))  // 900

    Thread.sleep 100
```

**Expected Output**:
```
Total: 1400
```

**Implementation**: Each actor is an OS thread with its own Region for scratch memory. The Region's lifetime is tied to the actor's lifetime - when the actor terminates, its region is released. This gives each worker isolated, deterministic memory without GC. Alex witnesses the actor + region composition and emits thread creation with region allocation in the entry point.

---

## Sample Directory Structure

```
samples/console/FidelityHelloWorld/
├── 01_HelloWorldDirect/
├── 02_HelloWorldSaturated/
├── 03_HelloWorldHalfCurried/
├── 04_HelloWorldFullCurried/
├── 05_AddNumbers/
├── 06_AddNumbersInteractive/
├── 07_BitsTest/
├── 08_Option/
├── 09_Result/
├── 10_Records/              # Needs fix
├── 11_HigherOrderFunctions/
├── 12_Closures/             # Source fix needed
├── 13_Recursion/            # Needs fix
├── 14_SimpleSeq/            # Planned (Phase B)
├── 15_SeqOperations/        # Planned (Phase B)
├── 16_LazyValues/           # Planned (Phase C)
├── 17_BasicAsync/           # Planned (Phase D)
├── 18_AsyncAwait/           # Planned (Phase D)
├── 19_AsyncParallel/        # Planned (Phase D)
├── 20_BasicRegion/          # Planned (Phase E)
├── 21_RegionPassing/        # Planned (Phase E)
├── 22_RegionEscape/         # Planned (Phase E)
├── 23_SocketBasics/         # Planned (Phase F)
├── 24_WebSocketEcho/        # Planned (Phase F)
├── 25_GTKWindow/            # Planned (Phase G)
├── 26_WebViewBasic/         # Planned (Phase G)
├── 27_BasicThread/          # Planned (Phase H)
├── 28_MutexSync/            # Planned (Phase H)
├── 29_BasicActor/           # Planned (Phase I - CAPSTONE)
├── 30_ActorReply/           # Planned (Phase I - CAPSTONE)
└── 31_ParallelActors/       # Planned (Phase I - CAPSTONE)
```

## Validation Protocol

For each sample:

1. **Build compiler**: `cd /home/hhh/repos/Firefly/src && dotnet build`
2. **Compile sample**: `Firefly compile <Sample>.fidproj`
3. **Execute binary**: `./<output_binary>`
4. **Verify output**: Compare with expected output
5. **Keep intermediates**: Use `-k` flag for debugging if needed

```bash
# Example validation
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/20_BasicRegion
/home/hhh/repos/Firefly/src/bin/Debug/net10.0/Firefly compile BasicRegion.fidproj
./target/basicregion
# Expected: Sum: 999000
```

## Dependencies by Phase

| Phase | FNCS Additions | Nanopass Enrichment | Alex Templates |
|-------|----------------|---------------------|----------------|
| A | None | Pattern binding coeffects | RecordFieldAccess |
| B | Seq module intrinsics | Yield state indices | SeqStateMachine |
| C | Lazy module intrinsics | Thunk capture coeffects | LazyThunk, LazyForce |
| D | Async module intrinsics | Suspension state indices | CoroFrame, CoroSuspend |
| E | Region intrinsics | Scope exit points, escape analysis | PageAlloc, PageFree, RegionAlloc |
| F | Socket syscall intrinsics | None | SysCall (platform Bindings) |
| G | None | None | ExternCall (FFI) |
| H | Thread/Mutex intrinsics | Thread capture coeffects | ThreadCreate, MutexOps |
| **I** | **MailboxProcessor intrinsics** | **Actor composition coeffects** | **Composed from D, E, H** |

### Capstone Feature Dependencies

```
Phase I (MailboxProcessor) - CAPSTONE
    │
    ├── requires Phase D (Async)
    │   └── LLVM coroutine for message loop
    │
    ├── requires Phase E (Scoped Regions)
    │   └── dynamic memory for worker threads
    │
    ├── requires Phase H (Threading)
    │   └── OS thread per actor
    │
    ├── requires Phase A (Closures)
    │   └── behavior function capture
    │
    └── requires Samples 05, 08, 09 (DUs)
        └── message type definitions
```

## Related Documentation

- [WRENStack_Roadmap.md](./WRENStack_Roadmap.md) - Overall architecture
- [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md) - Async implementation
- [FNCS_Architecture.md](./FNCS_Architecture.md) - Adding new intrinsics
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Compiler pipeline

**Serena Memories** (architectural guidance):
- `four_pillars_of_transfer` - Coeffects, Active Patterns, Zipper, Templates
- `codata_photographer_principle` - Witness, don't construct
- `architecture_principles` - Layer separation, non-dispatch model
