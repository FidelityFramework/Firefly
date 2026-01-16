# Gaining Closure: Non-Null Function Values in F# Native

*How MLKit-style flat closures enable null-free, GC-free function capture in Fidelity*

---

## The Problem with Closures

Every functional programmer loves closures. They're the magic that makes this work:

```fsharp
let makeCounter start =
    let mutable count = start
    fun () ->
        count <- count + 1
        count

let counter = makeCounter 0
counter()  // 1
counter()  // 2
counter()  // 3
```

The inner function "closes over" the mutable `count` variable, maintaining state across calls. Beautiful. Elegant. And in most implementations, a source of hidden complexity.

In .NET, that closure becomes a heap-allocated object with a reference to the captured variable. The garbage collector tracks it. If you're not careful, you create memory leaks. And somewhere in your runtime, there's a null check—because that closure reference *might* be null.

What if it couldn't be?

---

## The Fidelity Guarantee

Fidelity—the F# native compilation framework—makes a radical promise: **absolute null-freedom**. Not "we'll check for nulls at runtime." Not "we'll use option types." No nulls exist. Period.

This extends to closures. A function value in Fidelity is never null. It's never a reference that might point to nothing. It's a concrete struct with a known layout, allocated in a known location, with a known lifetime.

But how do you implement closures without references? Without a heap? Without a garbage collector?

The answer comes from 30 years of ML compiler research.

---

## Learning from Standard ML

In 1994, Zhong Shao and Andrew Appel published "Space-Efficient Closure Representations" while working on Standard ML of New Jersey. They identified two fundamental approaches to closure implementation:

### Linked Closures (The Common Approach)

```
┌─────────────────────┬─────────────────────┐
│ code_ptr            │ env_ptr ──────────► │ outer environment
└─────────────────────┴─────────────────────┘
```

Each closure stores a pointer to its enclosing environment. Simple to implement. But there's a problem: that pointer keeps the entire outer environment alive. Even if you only need one variable from a function three levels up, you're keeping everything alive.

Appel called this "not safe for space"—the garbage collector can't reclaim memory that a reference-counting collector would.

### Flat Closures (The MLKit Way)

```
┌─────────────────────┬─────────────────────────────────┐
│ code_ptr            │ captured values (inline)        │
└─────────────────────┴─────────────────────────────────┘
```

Copy every captured variable directly into the closure. No pointers to outer scopes. No chains to traverse. Each closure is self-contained.

This is what the MLKit compiler—developed by Mads Tofte and colleagues for region-based memory management—chose. And it's what Fidelity chooses.

---

## Why Flat Closures Win for Native Compilation

The benefits compound when you're compiling to native code without a runtime:

### 1. Cache-Friendly Access

All captured values are contiguous in memory. Access any capture with a single offset calculation. No pointer chasing through nested environments.

A closure with three int captures fits in 32 bytes—half a cache line. Invoke it without a single cache miss.

### 2. Deterministic Allocation

Flat closures have a known size at compile time. They can be stack-allocated when their lifetime is bounded. No heap fragmentation. No GC pauses.

```fsharp
let processItems items =
    let mutable total = 0
    items |> List.iter (fun x -> total <- total + x)
    total
```

That lambda? It's on the stack. It captures `total` by reference (a pointer to the stack slot). When `processItems` returns, everything is gone. No allocation. No cleanup. No runtime.

### 3. Region Compatibility

MLKit pioneered region-based memory management—allocate values in regions, deallocate entire regions at once. Flat closures fit perfectly:

```fsharp
Region.using (fun region ->
    let data = Region.alloc region 1000
    let processor = fun i -> data.[i] * 2  // Captures 'data'
    // processor lives in the region with data
    // Both deallocated together
)
```

The closure doesn't outlive its captures. The region system guarantees it.

### 4. No Null, Ever

A flat closure is a struct. It has a code pointer and inline data. There's no reference that might be null. The type system knows this. The generated code knows this. No null checks anywhere.

---

## The Two-Pass Architecture

Implementing flat closures in Fidelity required solving a subtle dependency problem.

FNCS (F# Native Compiler Services) builds a Program Semantic Graph (PSG) containing all semantic information. Alex (the code generation layer) needs to:

1. Know which variables are captured (to assign them SSA names)
2. Know the SSA names to build the environment struct

But SSA assignment needs to know about captures to allocate address SSAs for mutable captures. Chicken and egg.

The solution: **two preprocessing passes**.

```
PSG from FNCS
    │
    ▼
Pass 1: CaptureIdentification
    │   (Find lambdas with captures, mark captured variables)
    ▼
SSA Assignment
    │   (Assign SSAs, now knowing which vars need address SSAs)
    ▼
Pass 2: ClosureLayout
    │   (Compute environment struct with actual SSAs)
    ▼
Witness Emission
    (Observe coeffects, emit MLIR)
```

Each pass does one thing. Each pass is inspectable. The final emission phase just observes pre-computed coeffects and fills in templates.

This is the "Photographer Principle" that guides Fidelity development:

> **Stop trying to build the scene while you're taking the picture.**

Preprocessing builds the scene. The zipper moves the camera. Witnesses snap the photo.

---

## Capture Modes: By Value and By Reference

Not all captures are equal:

```fsharp
let makeGreeter name =        // 'name' is immutable
    fun greeting -> 
        $"{greeting}, {name}!"

let makeCounter start =
    let mutable count = start  // 'count' is mutable
    fun () ->
        count <- count + 1
        count
```

For `makeGreeter`, we copy `name` into the closure. The closure owns its copy.

For `makeCounter`, we can't copy `count`—the closure needs to mutate the *original*. So we capture a pointer to the stack slot.

| Variable Kind | Capture Mode | In Environment |
|---------------|--------------|----------------|
| Immutable | By Value | `T` (copy) |
| Mutable | By Reference | `ptr<T>` (pointer) |

This creates a lifetime constraint: the closure can't outlive the mutable variable's stack frame. FNCS enforces this at compile time. Try to return that counter's closure from a function that owns the mutable? Compile error.

---

## The Generated Code

Here's what Fidelity generates for `makeCounter`:

```mlir
func.func @makeCounter(%start: i32) -> !fidelity.closure<() -> i32> {
  // Stack slot for mutable count
  %count_slot = llvm.alloca 1 x i32
  llvm.store %start, %count_slot

  // Allocate environment (just a pointer)
  %env = llvm.alloca 1 x !llvm.struct<(ptr)>
  %slot0 = llvm.getelementptr %env[0, 0]
  llvm.store %count_slot, %slot0  // Store pointer to count

  // Build closure struct
  %code = llvm.addressof @counter_impl
  %closure = llvm.insertvalue { %code, %env }
  
  return %closure
}

func.func private @counter_impl(%env: ptr) -> i32 {
  // Extract count pointer from environment
  %count_ptr_ptr = llvm.getelementptr %env[0, 0]
  %count_ptr = llvm.load %count_ptr_ptr
  
  // Increment
  %old = llvm.load %count_ptr
  %new = arith.addi %old, 1
  llvm.store %new, %count_ptr
  
  return %new
}
```

No heap allocation. No GC interaction. No null checks. Just structs and pointers with compile-time-verified lifetimes.

---

## Standing on Giants' Shoulders

This work doesn't happen in isolation. The academic lineage is clear:

- **Appel & Shao (1994)**: Proved flat closures are space-safe
- **Tofte & Talpin (1997)**: Region-based memory management for ML
- **MLKit (2003+)**: Production implementation of regions with flat closures
- **Perconti & Ahmed (2019)**: Formal proof that closure conversion preserves space bounds

Fidelity takes these ideas and applies them to F#, a language with richer features than Standard ML. The challenge is ensuring F#'s features—computation expressions, active patterns, type providers—compose correctly with null-free, GC-free closures.

---

## What This Enables

With flat closures working, Fidelity unlocks:

### Higher-Order Functions
```fsharp
items |> List.map (fun x -> x * 2) |> List.filter (fun x -> x > 10)
```
Each lambda is a stack-allocated struct. The entire pipeline executes without heap allocation.

### Async Without Runtime
```fsharp
async {
    let! data = fetchData url
    return process data
}
```
The async state machine captures variables in a flat struct. No Task objects. No heap. Just a coroutine frame on the stack.

### Actors Without Overhead
```fsharp
MailboxProcessor.Start(fun inbox ->
    let rec loop state = async {
        let! msg = inbox.Receive()
        return! loop (update state msg)
    }
    loop initialState
)
```
The behavior function is a flat closure. The message loop is a coroutine. No GC pressure from millions of actors.

---

## Gaining Closure

The title is a pun, obviously. But it's also literal.

Fidelity "gains closure" in the sense of completing a capability—function values that capture their environment work correctly, efficiently, and safely.

It gains closure in the mathematical sense—the type system is closed under function formation. You can always create a function that captures values, and the result is always well-typed and non-null.

And it gains closure in the emotional sense. After wrestling with witness pollution, SSA dependencies, and nanopass ordering, seeing `makeCounter` compile and run correctly—incrementing on each call, maintaining state, no GC, no nulls—feels like closing a chapter.

On to the next sample. On to sequences, lazy values, async, regions, threading, actors. Each building on this foundation of null-free, GC-free, flat closures.

---

*This is part of the FidelityHelloWorld sample progression, proving F# native compilation one feature at a time. The full implementation is documented in the [Closure Nanopass Architecture](../Firefly/docs/Closure_Nanopass_Architecture.md) and the [fsnative-spec closure representation](../fsnative-spec/spec/closure-representation.md).*

---

## References

- Shao, Z., & Appel, A. W. (1994). [Space-Efficient Closure Representations](https://www.cs.tufts.edu/comp/150FP/archive/andrew-appel/cpcps.pdf). LFP '94.
- Tofte, M., & Talpin, J.-P. (1997). Region-Based Memory Management. Information and Computation.
- Elsman, M. (2003). Garbage Collection Safety for Region-Based Memory Management. TLDI '03.
- Perconti, J. T., & Ahmed, A. (2019). [Closure Conversion Is Safe for Space](https://zoep.github.io/icfp2019.pdf). ICFP '19.
- [MLKit Documentation](https://elsman.com/mlkit/doc.html)
