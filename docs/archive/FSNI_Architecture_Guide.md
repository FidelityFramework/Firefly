# F# Native Interactive (FSNI): Architecture Guide

## Introduction

If you've used F# Interactive (FSI), you know the power of exploratory programming. Type an expression, see the result. Define a function, test it immediately. Build up a solution piece by piece, with instant feedback at every step. FSI is one of F#'s most beloved features.

But FSI depends entirely on the .NET runtime. It works because .NET provides a managed heap with garbage collection, runtime type information, and the ability to dynamically load and execute code. When you type `let x = 42` in FSI, the .NET JIT compiler generates machine code on the fly, and the runtime manages everything else.

Fidelity doesn't have a runtime. That's the whole point. We compile F# to native binaries that run without .NET, without a garbage collector, without runtime type information. So how could we possibly provide an interactive experience?

This document explores approaches to this problem. We examine how other native-first languages have solved it, consider what infrastructure we might leverage, and propose an architecture that we believe could work for Fidelity. This is a design document; the implementation work lies ahead.

---

## Part 1: Understanding What FSI Actually Does

Before we can design FSNI, we need to understand what FSI does under the hood. Most F# developers use FSI without thinking about the machinery that makes it work.

### The FSI Illusion

When you type this in FSI:

```fsharp
> let greeting = "Hello, World!"
val greeting: string = "Hello, World!"
```

It feels like an interpreter is reading your code and executing it directly. But that's not what happens. FSI is not an interpreter. It's an incremental compiler with dynamic execution.

Here's what actually occurs:

1. **Parsing**: FSI parses your input as F# source code
2. **Type Checking**: The F# compiler type-checks your expression
3. **IL Generation**: The compiler generates .NET Intermediate Language (IL)
4. **JIT Compilation**: The .NET runtime's JIT compiler converts IL to machine code
5. **Execution**: The machine code runs in the context of FSI's process
6. **Display**: The result is captured and pretty-printed

The magic ingredient is the .NET runtime's JIT compiler. "JIT" stands for "Just-In-Time"; the runtime compiles code to native machine instructions right before executing them. This is different from "AOT" (Ahead-Of-Time) compilation, where all code is compiled before the program runs.

### Why FSI Feels Fast

FSI feels instant because the .NET JIT is extremely fast. It's been optimized over decades to compile code in milliseconds. More importantly, the entire .NET type system and garbage collector are available at runtime. When you create a string, .NET allocates memory and tracks it for you. When you define a type, .NET stores metadata about it that can be queried later.

This runtime infrastructure is what Fidelity explicitly doesn't have. We compile everything ahead of time, determine memory layout at compile time, and produce binaries with no runtime dependencies.

So our challenge is: how do we get the interactive experience without the runtime?

---

## Part 2: How Other Native Languages Approach This Problem

We're not the first to face this challenge. Several languages that compile to native code have tackled the interactive programming problem. Each approach teaches us something.

### Rust: The Evcxr Approach

Rust, like Fidelity, compiles to native code without a garbage collector. The Rust community created [Evcxr](https://github.com/evcxr/evcxr) (pronounced "e-vex-er") to provide REPL functionality.

**How Evcxr works:**

When you type an expression in Evcxr, it doesn't interpret your code. Instead, it:

1. Wraps your input in a compilable Rust program
2. Invokes the real Rust compiler (`rustc`) to build a shared library
3. Dynamically loads that library into the Evcxr process
4. Calls the compiled function and captures the result
5. Displays the result

This approach is worth noting: Evcxr uses the real compiler for every evaluation. Your interactive code goes through the exact same compilation pipeline as production code. There's no separate interpreter that might behave differently.

**The tradeoff:**

Every evaluation requires a full compile-and-link cycle. Even though `rustc` is fast, there's noticeable latency, typically hundreds of milliseconds per evaluation. For quick experiments this is acceptable, but it can feel sluggish compared to FSI.

**What we take from this:**

Using the real compiler ensures identical behavior between interactive and compiled code. Shared library loading enables code to persist across evaluations. State management (remembering `let` bindings) requires careful symbol tracking.

### C++: The Clang-Repl Approach

C++ is the original "native compilation" language, and for decades it lacked any interactive capability. The LLVM project changed this with [Clang-Repl](https://clang.llvm.org/docs/ClangRepl.html), an interactive C++ environment.

**How Clang-Repl works:**

Clang-Repl takes a different approach than Evcxr. Instead of compiling to shared libraries, it uses incremental JIT compilation:

1. Parse input to an Abstract Syntax Tree (AST)
2. Generate LLVM Intermediate Representation (IR)
3. JIT-compile the IR to machine code in memory
4. Execute the machine code directly
5. Capture and display results

What makes this possible is LLVM's ORC JIT framework. ORC (On-Request Compilation) is LLVM's infrastructure for compiling code dynamically within a running process. Instead of writing object files to disk and loading them, ORC compiles directly to executable memory.

**Why ORC JIT matters for us:**

Traditional compilers write their output to files:

```
Source → Compiler → Object File → Linker → Executable File → Load → Run
```

ORC JIT eliminates the file I/O:

```
Source → Compiler → LLVM IR → ORC JIT → Executable Memory → Run
```

This should be much faster. There's no disk access, no linker invocation, no process spawning. The compilation happens entirely in memory, and execution begins immediately.

**What we take from this:**

JIT compilation can provide low-latency interactive evaluation. LLVM's ORC framework is mature and battle-tested. Incremental compilation requires maintaining state between evaluations.

### Swift: The Debugger Approach

Apple's Swift language takes yet another approach with its REPL, which is built on top of the LLDB debugger.

**How Swift REPL works:**

The Swift REPL is actually a debugger session. When you type an expression:

1. The embedded Swift compiler compiles your code
2. The code is JIT-compiled for execution
3. The debugger context captures the result
4. Display uses the debugger's introspection capabilities

This seems roundabout, but it's clever. By building on the debugger, Swift gets expression evaluation (the debugger already knows how to compile and execute expressions), variable inspection (the debugger can examine memory and display values), and breakpoints for free (you can set breakpoints in your REPL-defined functions).

**What we take from this:**

Tight integration between compiler and runtime tools enables powerful features. The REPL and debugger can share infrastructure. A matched compiler/runtime pair is essential for consistency.

---

## Part 3: The LLVM Foundation

Before we design FSNI, we need to understand LLVM, since Firefly already uses it. If you've never worked with LLVM, this section provides the context you need.

### What is LLVM?

LLVM (historically "Low Level Virtual Machine," though it's no longer an acronym) is a compiler infrastructure project. Think of it as a "compiler construction kit": a set of reusable components for building compilers.

The central idea of LLVM is separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLVM Architecture                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Frontend (Language-Specific)    Middle End       Backend (Target)   │
│  ┌─────────────────────────┐    ┌──────────┐    ┌───────────────┐   │
│  │  F# → FNCS → PSG → Alex │───►│ LLVM IR  │───►│ x86-64        │   │
│  │  C++ → Clang            │    │          │    │ ARM64         │   │
│  │  Rust → rustc           │    │ Optimize │    │ RISC-V        │   │
│  │  Swift → swiftc         │    │          │    │ WebAssembly   │   │
│  └─────────────────────────┘    └──────────┘    └───────────────┘   │
│                                                                      │
│  Many languages can share the same optimization and code generation │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Different language frontends produce LLVM IR (Intermediate Representation), which is a common format. LLVM then optimizes this IR and generates machine code for the target architecture. By building on LLVM, Firefly gets world-class code optimization and support for many CPU architectures without having to implement them ourselves.

### What is MLIR?

MLIR (Multi-Level Intermediate Representation) is a newer layer that sits above LLVM IR. While LLVM IR is relatively low-level (close to assembly), MLIR allows higher-level abstractions.

Firefly uses MLIR as an intermediate step:

```
F# Source → FNCS → PSG → Alex → MLIR → LLVM IR → Machine Code
```

MLIR's "dialects" allow us to express concepts like memory regions (stack vs. heap vs. peripheral), ownership (who owns a piece of memory), and high-level operations (before lowering to primitive instructions).

This intermediate representation should make it easier to implement F# Native's memory safety guarantees.

### What is ORC JIT?

ORC (On-Request Compilation) is LLVM's framework for JIT compilation. It provides:

1. **ExecutionSession**: A container for JIT'd code and its state
2. **JITDylib**: A "dynamic library" that exists only in memory
3. **MaterializationUnit**: A deferred unit of compilation
4. **Lazy compilation**: Only compile code when it's actually called

ORC is what powers Clang-Repl, the Swift REPL, and many other interactive tools. It's mature, performant, and designed for exactly our use case.

**How ORC maintains state:**

When you define `let x = 42` in an ORC-based REPL, the expected flow would be:

1. The expression is compiled to a function that returns 42
2. The function is JIT'd and executed
3. The result (42) is stored in a location that ORC tracks
4. The symbol `x` is registered in the JITDylib, pointing to that location

Later, when you type `x + 1`:

1. The expression is compiled to code that loads `x` and adds 1
2. The reference to `x` resolves to the previously stored value
3. The result (43) is computed and displayed

The JITDylib acts like a persistent symbol table, allowing later code to reference earlier definitions.

---

## Part 4: Proposed FSNI Architecture

With this understanding, we can now propose an architecture for FSNI. Our approach would combine aspects of Clang-Repl and Evcxr, adapted to Firefly's existing architecture.

### The Core Insight

Firefly already has a complete compilation pipeline:

```
F# Source → FNCS → PSG → Alex → MLIR → LLVM IR → Object File → Executable
```

For FSNI, we would only need to change the last step. Instead of writing to an object file and linking, we would JIT-compile directly to memory:

```
F# Source → FNCS → PSG → Alex → MLIR → LLVM IR → ORC JIT → Execute
```

This is potentially powerful: FSNI would use the exact same frontend and middle-end as batch compilation. There would be no separate interpreter, no simplified type checker, no alternative execution model. The same FNCS that type-checks your `.fs` files would type-check your REPL input. The same Alex that generates MLIR for production code would generate MLIR for interactive evaluation.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          FSNI Process Architecture                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        User Interface Layer                            │ │
│  │  ┌─────────────┐ ┌──────────────────┐ ┌────────────────────────────┐  │ │
│  │  │ FSNI Shell  │ │ Jupyter Kernel   │ │ Editor Integration (future)│  │ │
│  │  │ (terminal)  │ │ (notebooks)      │ │ (VS Code inline eval)      │  │ │
│  │  └──────┬──────┘ └────────┬─────────┘ └─────────────┬──────────────┘  │ │
│  │         │                 │                         │                  │ │
│  │         └─────────────────┼─────────────────────────┘                  │ │
│  │                           ▼                                            │ │
│  │                  ┌────────────────────┐                                │ │
│  │                  │ Input Handler      │                                │ │
│  │                  │ - Parse input      │                                │ │
│  │                  │ - Detect commands  │                                │ │
│  │                  │ - Route to engine  │                                │ │
│  │                  └─────────┬──────────┘                                │ │
│  └────────────────────────────┼───────────────────────────────────────────┘ │
│                               │                                             │
│                               ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      Evaluation Engine                                 │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Evaluation Context                           │   │ │
│  │  │  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐  │   │ │
│  │  │  │ Symbol Table     │ │ Type Registry    │ │ Session Arena  │  │   │ │
│  │  │  │                  │ │                  │ │                │  │   │ │
│  │  │  │ x = 42          │ │ Person = record  │ │ ┌────────────┐ │  │   │ │
│  │  │  │ greet = <func>  │ │ Result<'T> = DU  │ │ │ Allocated  │ │  │   │ │
│  │  │  │ ...             │ │ ...              │ │ │ Values     │ │  │   │ │
│  │  │  └──────────────────┘ └──────────────────┘ │ └────────────┘ │  │   │ │
│  │  │                                            └────────────────┘  │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │              Incremental Compilation Pipeline                    │  │ │
│  │  │                                                                  │  │ │
│  │  │  Input     FNCS         PSG        Alex        MLIR     ORC JIT │  │ │
│  │  │    │    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌─────┐ │  │ │
│  │  │    └───►│ Type   │─►│ Build  │─►│ Generate│─►│ Lower  │─►│ JIT │ │  │ │
│  │  │         │ Check  │  │ Graph  │  │ MLIR   │  │ to IR  │  │     │ │  │ │
│  │  │         └────────┘  └────────┘  └────────┘  └────────┘  └──┬──┘ │  │ │
│  │  │                                                            │    │  │ │
│  │  │              Context is consulted ◄────────────────────────┘    │  │ │
│  │  │              and updated at each step                           │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    ORC JIT Session                               │  │ │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │ ExecutionSession                                           │  │  │ │
│  │  │  │  ┌─────────────────┐ ┌─────────────────┐                  │  │  │ │
│  │  │  │  │ Main JITDylib   │ │ Alloy JITDylib  │                  │  │  │ │
│  │  │  │  │ (user symbols)  │ │ (library funcs) │                  │  │  │ │
│  │  │  │  │                 │ │                 │                  │  │  │ │
│  │  │  │  │ x → addr1       │ │ WriteLine→addr  │                  │  │  │ │
│  │  │  │  │ greet → addr2   │ │ sqrt → addr    │                  │  │  │ │
│  │  │  │  └─────────────────┘ └─────────────────┘                  │  │  │ │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                               │                                             │
│                               ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Output Layer                                    │ │
│  │  ┌──────────────────┐ ┌────────────────────┐ ┌─────────────────────┐  │ │
│  │  │ Value Display    │ │ Error Formatting   │ │ Type Information    │  │ │
│  │  │ (pretty print)   │ │ (diagnostics)      │ │ (signatures)        │  │ │
│  │  └──────────────────┘ └────────────────────┘ └─────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Deep Dive

#### The Evaluation Context

The Evaluation Context would be the persistent state that makes FSNI feel continuous. Unlike a batch compiler that starts fresh each time, FSNI would accumulate state across evaluations.

**Symbol Table:**

When you write `let x = 42`, the symbol `x` would be added to the symbol table. This would be a mapping from names to their locations in memory:

```fsharp
type BoundValue = {
    Name: string
    Type: NativeType
    Address: nativeint        // Where the value lives in memory
    IsMutable: bool
}

type SymbolTable = Map<string, BoundValue>
```

Later, when you reference `x`, the symbol table would tell us where to find the value.

**Type Registry:**

When you define `type Person = { Name: string; Age: int }`, the type would be recorded in the registry. This would allow later code to reference `Person`:

```fsharp
type TypeDefinition =
    | RecordType of fields: (string * NativeType) list * layout: MemoryLayout
    | UnionType of cases: UnionCase list * layout: MemoryLayout
    | AliasType of target: NativeType

type TypeRegistry = Map<string, TypeDefinition>
```

**Session Arena:**

Fidelity uses arena allocation for memory management. The session arena would hold all values created during an FSNI session:

```fsharp
type SessionArena = {
    // Long-lived arena for session values
    SessionMemory: Arena

    // Short-lived arena for temporary computations (optional)
    ScratchMemory: Arena
}
```

When you create a value, it would be allocated in the session arena. Because arenas are contiguous memory regions, allocation is just incrementing a pointer, which should be very fast.

The session arena would not be freed until the FSNI session ends. This differs from Fidelity's normal mode, where arenas might be scoped to functions. In FSNI, we would need values to persist.

#### Incremental FNCS

Standard FNCS processes complete files. For FSNI, we would need incremental type checking:

```fsharp
module FNCSIncremental =

    /// The state accumulated during a session
    type IncrementalState = {
        /// Known bindings from previous evaluations
        Bindings: Map<string, FSharpType>

        /// Known type definitions
        Types: Map<string, FSharpTypeDefinition>

        /// Open namespaces (from `open` declarations)
        OpenedNamespaces: Set<string>

        /// FNCS internal state (caches, etc.)
        InternalState: FNCSCheckState
    }

    /// Check a single fragment (expression or declaration)
    let checkFragment
        (state: IncrementalState)
        (input: string)
        : Result<TypedFragment * IncrementalState, FNCSError list> =

        // 1. Parse the input as a fragment
        let parsed = parseFragment input

        // 2. Augment the environment with known bindings
        let augmentedEnv = augmentEnvironment state.InternalState state.Bindings

        // 3. Type-check in the augmented environment
        let typed = typeCheck augmentedEnv parsed

        // 4. Update state with any new bindings/types
        let newState = updateState state typed

        Ok (typed, newState)
```

What we expect here is that FNCS already handles complex type inference. We would just need to make it incremental, able to add new bindings without forgetting old ones.

#### The JIT Pipeline

Once FNCS produces a typed fragment, we would generate code and execute it:

**Step 1: Build PSG**

The Program Semantic Graph would be constructed from the typed fragment, just as with batch compilation:

```fsharp
let psgFragment = PSGBuilder.buildFromFragment typedFragment context.SymbolTable
```

**Step 2: Generate MLIR**

Alex would generate MLIR for the fragment:

```fsharp
let mlirModule = Alex.generateMLIR psgFragment
```

For an expression like `x + 1`, this might produce:

```mlir
func.func @__repl_eval_7() -> i32 {
    %x = load @x : i32              // Load x from symbol table location
    %one = arith.constant 1 : i32
    %result = arith.addi %x, %one : i32
    return %result : i32
}
```

**Step 3: Lower to LLVM IR**

Standard MLIR lowering:

```fsharp
let llvmIR = MLIR.lowerToLLVM mlirModule
```

**Step 4: JIT Compile**

ORC would compile the IR to executable memory:

```fsharp
let jitSession = context.JitSession
let functionPtr = ORC.addModule jitSession llvmIR
```

**Step 5: Execute and Capture**

We would call the compiled function and capture its result:

```fsharp
let result = invokeFunction functionPtr
let displayValue = ValueDisplay.format result resultType
```

#### Value Display

In FSI, .NET reflection enables sophisticated value display. We can inspect any object, enumerate its fields, and format it nicely.

FSNI wouldn't have reflection. Types are erased at compile time; we would only have raw memory at runtime. But we could still display values because we would know their types at compile time:

```fsharp
module ValueDisplay =

    /// Format a value for display, given its compile-time type
    let rec format (address: nativeint) (typ: NativeType) : string =
        match typ with
        | NativeType.Int32 ->
            let value = NativePtr.read<int32> (NativePtr.ofNativeInt address)
            sprintf "%d" value

        | NativeType.Float64 ->
            let value = NativePtr.read<float> (NativePtr.ofNativeInt address)
            sprintf "%g" value

        | NativeType.String ->  // string with native semantics (UTF-8 fat pointer)
            let strPtr = NativePtr.read<nativeint> (NativePtr.ofNativeInt address)
            let len = NativePtr.read<int> (NativePtr.ofNativeInt (strPtr + 8n))
            let chars = readUtf8Bytes strPtr len
            sprintf "\"%s\"" (System.Text.Encoding.UTF8.GetString chars)

        | NativeType.Record fields ->
            let fieldStrs =
                fields
                |> List.map (fun (name, fieldType, offset) ->
                    let fieldAddr = address + nativeint offset
                    sprintf "%s = %s" name (format fieldAddr fieldType))
            sprintf "{ %s }" (String.concat "; " fieldStrs)

        | NativeType.ValueOption inner ->
            let tag = NativePtr.read<byte> (NativePtr.ofNativeInt address)
            if tag = 0uy then
                "ValueNone"
            else
                let valueAddr = address + 8n  // Adjust for alignment
                sprintf "ValueSome %s" (format valueAddr inner)

        | NativeType.Function _ ->
            "<function>"

        | NativeType.Unit ->
            "()"
```

The type information would flow from FNCS through the entire pipeline. By the time we display a value, we would know exactly what type it is and could format it appropriately.

### The "Warm Compiler" Insight

Compiler startup is expensive. Loading the Alloy library, building symbol tables, initializing MLIR passes: this takes time. If we did this for every evaluation, FSNI would feel sluggish.

The solution we anticipate: keep the compiler warm.

When FSNI starts, it would initialize the full compilation pipeline once:

```fsharp
module FSNIServer =

    type ServerState = {
        FncsState: FNCSIncremental.IncrementalState
        JitSession: ORC.ExecutionSession
        AlloyDylib: ORC.JITDylib
        EvalContext: EvaluationContext
    }

    let initialize () : ServerState =
        // 1. Initialize FNCS with Alloy preloaded
        let fncsState = FNCSIncremental.initialize alloyPath

        // 2. Create ORC JIT session
        let jitSession = ORC.createSession ()

        // 3. Pre-JIT Alloy library functions
        let alloyDylib = preloadAlloy jitSession alloyPath

        // 4. Create empty evaluation context
        let evalContext = EvaluationContext.empty

        { FncsState = fncsState
          JitSession = jitSession
          AlloyDylib = alloyDylib
          EvalContext = evalContext }

    let evaluate (state: ServerState) (input: string) : EvalResult * ServerState =
        // Compilation uses already-warm state
        // ...
```

After initialization, each evaluation would only pay for parsing the input fragment, incremental type checking, generating MLIR for just that fragment, JIT compiling (typically microseconds to milliseconds), and execution.

This is why we expect FSNI could feel responsive despite being native compilation.

**Synergy with FSNAC:**

FSNAC (the LSP server) would also need a warm compiler state. In fact, FSNAC and FSNI could potentially share the same background process:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fidelity Language Service                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Warm Compiler State                            │   │
│  │  - FNCS initialized with Alloy                                   │   │
│  │  - Symbol tables cached                                          │   │
│  │  - MLIR passes ready                                             │   │
│  │  - ORC JIT session active                                        │   │
│  └───────────────────────────┬──────────────────────────────────────┘   │
│                              │                                           │
│              ┌───────────────┼───────────────┐                          │
│              ▼               ▼               ▼                          │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────────┐    │
│  │ LSP Handlers     │ │ FSNI Engine  │ │ Background Analysis      │    │
│  │ (hover, goto-def)│ │ (evaluate)   │ │ (diagnostics, linting)   │    │
│  └──────────────────┘ └──────────────┘ └──────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Memory Management in FSNI

Fidelity's core promise is deterministic memory management. In batch compilation, lifetimes are known at compile time. We can prove when memory is no longer needed and free it immediately.

FSNI complicates this. Values must persist indefinitely because the user might reference them later. How do we reconcile this with Fidelity's memory model?

### Session-Scoped Arena

The answer we propose is a session-scoped arena. All FSNI allocations would come from an arena that lives as long as the session:

```fsharp
type SessionMemory = {
    /// Main arena for session-lived values
    SessionArena: Arena

    /// Statistics for user visibility
    mutable TotalAllocated: int64
    mutable AllocationCount: int64
}

module SessionMemory =

    let create (initialCapacity: int) : SessionMemory =
        { SessionArena = Arena.create initialCapacity
          TotalAllocated = 0L
          AllocationCount = 0L }

    let allocate (mem: SessionMemory) (size: int) : nativeint =
        mem.TotalAllocated <- mem.TotalAllocated + int64 size
        mem.AllocationCount <- mem.AllocationCount + 1L
        Arena.allocate mem.SessionArena size

    let reset (mem: SessionMemory) : unit =
        Arena.reset mem.SessionArena
        mem.TotalAllocated <- 0L
        mem.AllocationCount <- 0L
```

When the session ends, the entire arena would be freed at once. During the session, we wouldn't free individual values because we couldn't know if the user would reference them.

### Memory Pressure

Long FSNI sessions would accumulate memory. We would provide visibility into this:

```
fsni> :memory
Session memory: 42.3 KB allocated in 127 allocations
Session arena capacity: 1 MB

fsni> :reset
Session reset. Memory freed.
```

The `:reset` command would clear all bindings and free the arena, giving users control over memory pressure.

### What About Drop Semantics?

In batch compilation, Fidelity can insert drop calls when values go out of scope. In FSNI, values don't go out of scope; they persist in the session.

However, the evaluation of an expression can have temporary values:

```fsharp
fsni> let result = String.concat ", " [| "a"; "b"; "c" |]
```

The intermediate array `[| "a"; "b"; "c" |]` is temporary. It's only needed during the evaluation. The `result` string needs to persist.

We would handle this with two-tier allocation:

```fsharp
type EvaluationMemory = {
    /// Persistent: values that must outlive the evaluation
    Session: SessionMemory

    /// Temporary: values that can be freed after evaluation
    Scratch: Arena
}

// After each evaluation:
// 1. Persistent values have been allocated in Session arena
// 2. Scratch arena is reset (freeing temporaries)
Arena.reset evalMemory.Scratch
```

The code generator would know which values escape (are bound to names or returned) and which are temporary. Escaping values would go to the session arena; temporaries would go to scratch.

---

## Part 6: Comparison with FSI

Let's concretely compare how FSI and FSNI would differ:

### Type Display

```
FSI:
> let x = "hello"
val x: string = "hello"

FSNI:
> let x = "hello"
val x: string = "hello"
```

In FSI, strings are `System.String` (UTF-16, heap-allocated). In FSNI, they would still be displayed as `string`, but with native semantics: UTF-8 encoded, fat pointer representation, deterministic memory.

### Option Types

```
FSI:
> let y = Some 42
val y: int option = Some 42

FSNI:
> let y = Some 42
val y: int voption = ValueSome 42
```

FSNI would use `voption` (value option), which is a struct, not a heap-allocated reference type.

### Memory Visibility

```
FSI:
(no memory information - GC handles it)

FSNI:
> :memory
Session memory: 156 bytes in 3 allocations
```

FSNI would make memory usage visible because it's managed explicitly.

### Library Access

```
FSI:
> open System.IO
> File.Exists("foo.txt")
val it: bool = false

FSNI:
> open Alloy.FileSystem
> File.exists "foo.txt"
val it: bool = false
```

FSI uses .NET's BCL. FSNI would use Alloy, which provides native implementations of common functionality.

### SRTP Resolution

```
FSI:
> let inline double x = x + x
val inline double: x: ^a -> ^a when ^a: (static member (+): ^a * ^a -> ^a)

> double 21
val it: int = 42

FSNI:
> let inline double x = x + x
val inline double: x: ^a -> ^a when ^a: (static member (+): ^a * ^a -> ^a)

> double 21
val it: int = 42
```

SRTP would work identically at the syntax level. But in FSNI, they would resolve against Alloy's witness hierarchy instead of .NET method tables.

---

## Part 7: Implementation Roadmap

FSNI would be a significant undertaking. Here's how we anticipate building it incrementally.

### Phase 0: Foundation (Prerequisites)

Before FSNI development could begin, we would need:

1. **Stable FNCS incremental API**: Ability to type-check fragments against accumulated context, with efficient state management.

2. **Complete Alex code generation**: All core F# constructs must generate correct MLIR. Platform bindings for output (console, etc.) must work.

3. **ORC JIT integration layer**: F# bindings for LLVM's ORC APIs. JITDylib management. Symbol resolution.

### Phase 1: Minimal Evaluator

**Goal**: Evaluate expressions and display results

**Deliverables**:
- `fsni` CLI executable
- Expression evaluation for literals and arithmetic
- Primitive value display (int, float, string, bool)

**Validation**:
```
fsni> 1 + 1
val it: int = 2

fsni> 3.14159 * 2.0
val it: float = 6.28318

fsni> "Hello"
val it: string = "Hello"
```

### Phase 2: Binding Persistence

**Goal**: `let` bindings persist across evaluations

**Deliverables**:
- Symbol table implementation
- Binding persistence in JITDylib
- Mutable variable support

**Validation**:
```
fsni> let x = 42
val x: int = 42

fsni> x * 2
val it: int = 84

fsni> let mutable count = 0
val mutable count: int = 0

fsni> count <- count + 1; count
val it: int = 1
```

### Phase 3: Functions

**Goal**: Define and call functions

**Deliverables**:
- Function definition support
- Recursive function support
- First-class function values
- Inline functions with SRTP

**Validation**:
```
fsni> let square x = x * x
val square: int -> int

fsni> square 7
val it: int = 49

fsni> let rec factorial n = if n <= 1 then 1 else n * factorial (n - 1)
val factorial: int -> int

fsni> factorial 5
val it: int = 120

fsni> let inline twice f x = f (f x)
val inline twice: f: (^a -> ^a) -> x: ^a -> ^a

fsni> twice square 3
val it: int = 81
```

### Phase 4: Types

**Goal**: Define and use custom types

**Deliverables**:
- Record type definitions
- Discriminated union definitions
- Type display with memory layout info
- Pattern matching on custom types

**Validation**:
```
fsni> type Point = { X: float; Y: float }
type Point = { X: float; Y: float }  // 16 bytes

fsni> let origin = { X = 0.0; Y = 0.0 }
val origin: Point = { X = 0.0; Y = 0.0 }

fsni> type Shape = Circle of radius: float | Rectangle of width: float * height: float
type Shape  // 24 bytes (max variant)

fsni> Circle 5.0
val it: Shape = Circle 5.0
```

### Phase 5: Alloy Integration

**Goal**: Full access to Alloy library

**Deliverables**:
- Auto-import of `Alloy.Core`
- `open` declaration support
- Platform binding execution
- I/O operations

**Validation**:
```
fsni> open Alloy.Console
fsni> WriteLine "Hello, native world!"
Hello, native world!
val it: unit = ()

fsni> open Alloy.Math
fsni> sin (Math.PI / 2.0)
val it: float = 1.0
```

### Phase 6: Jupyter Kernel

**Goal**: Notebook-style interaction

**Deliverables**:
- `fsni-jupyter` kernel executable
- Jupyter messaging protocol implementation
- Rich output rendering
- Markdown cell support

**Validation**:
- Create and execute F# Native notebooks in Jupyter
- Display tables, formatted output
- Share notebooks with reproducible results

### Phase 7: Debugger Integration (Future)

**Goal**: REPL + Debugger hybrid

**Deliverables**:
- LLDB integration for F# Native
- Breakpoint support in REPL-defined functions
- Variable inspection in debug context
- Expression evaluation at breakpoints

This phase is exploratory and would depend on learnings from earlier phases.

---

## Part 8: FAQ

### Why not just use FSI?

FSI requires the .NET runtime. Fidelity's whole purpose is native compilation without runtime dependencies. Using FSI would mean interactive and compiled code behave differently, exactly what we want to avoid.

### Would FSNI be slower than FSI?

For compilation, probably yes. We would be doing full native code generation. But execution of the compiled code might be faster since there's no .NET overhead. The experience should feel similar due to the warm compiler optimization.

### Can I use .NET libraries in FSNI?

Not directly, but some libraries may work with modification. If you're familiar with Fable (the F#-to-JavaScript compiler), you've encountered this pattern: some NuGet packages are .NET-only, some are Fable-only, and some are "cross-platform" libraries that work in both contexts. The library author provides implementations suitable for each target.

We anticipate a similar pattern emerging for Fidelity. A library that depends only on core F# language features and uses abstractions over I/O could potentially offer Fidelity-compatible implementations alongside its .NET implementation. In the early days, this would require library authors to consciously support the Fidelity target, just as they consciously support Fable today.

For FSNI specifically, you would use Alloy (Fidelity's native standard library) and any libraries that have been written for or adapted to the Fidelity ecosystem. We expect tooling to emerge that will help identify which libraries and which parts of libraries are compatible with Fidelity projects. Over time, we anticipate conversion tooling that could assist in adapting existing pure F# libraries for native compilation, though this remains a future goal.

### What about NuGet packages?

NuGet is the .NET package ecosystem. Fidelity will develop its own package management approach, likely based on `.fidproj` dependencies. That said, the Fable precedent suggests a path forward: packages on NuGet can include Fable-compatible implementations, and someday they might include Fidelity-compatible implementations as well. The mechanics of package resolution in FSNI would mirror whatever approach Fidelity adopts for batch compilation.

### Can I save an FSNI session?

This is a planned feature. The idea would be to serialize session state (bindings, types, code) to a `.fsnx` script file that could be replayed or compiled to a standalone binary.

### How would FSNI handle errors?

FNCS would produce typed error messages (FS8xxx codes for native-specific errors). FSNI would display these with source locations. Since we would use the real compiler, error messages would be the same as in batch compilation.

### What platforms would FSNI support?

FSNI would target the same platforms as Firefly: Linux x86_64, Linux ARM64, macOS, and eventually Windows. The ORC JIT would handle platform-specific code generation.

---

## Part 9: Alternative Approaches Considered

In designing FSNI, we considered several alternative architectures before settling on the ORC JIT approach.

### Interpreter Mode

**Idea**: Build a pure interpreter that executes F# without compilation.

**Why we don't favor this**:
- An interpreter would not exercise native semantics; behavior would differ from compiled code
- We would need to maintain two execution models (interpreter + compiler)
- Performance would be poor for compute-intensive explorations
- The whole point of Fidelity is native compilation; an interpreter contradicts this

### Dynamic Library Approach (Evcxr-style)

**Idea**: Compile each evaluation to a shared library (`.so` or `.dylib`), then dynamically load it.

**Trade-offs**:
- Higher latency than JIT due to file I/O and the linker
- Symbol management across libraries is complex
- However, this approach is well-understood and could serve as a fallback if JIT proves difficult

We may revisit this approach if ORC JIT integration proves more challenging than expected.

### Full Recompilation

**Idea**: Recompile the entire session (all accumulated definitions) on each input.

**Why we don't favor this**:
- Latency would grow linearly with session size
- A session with 50 definitions would take 50x longer than the first evaluation
- This would be unacceptable for interactive use

The incremental approach, where we only compile the new fragment and link against previous definitions, should scale much better.

---

## Part 10: Comparison Summary

Here's a direct comparison of FSI and FSNI:

| Aspect | FSI | FSNI |
|--------|-----|------|
| Compilation | .NET JIT (IL → machine code) | LLVM ORC JIT (MLIR → LLVM IR → machine code) |
| Type system | .NET types (BCL semantics) | Same type names, native semantics |
| Memory model | Garbage collected | Arena/Stack with deterministic lifetimes |
| Startup time | ~1 second | ~100ms (warm), longer cold |
| String type | `string` (UTF-16, heap) | `string` (UTF-8, fat pointer) |
| Option types | `option<T>` (reference, heap) | `option<T>` (value, stack) |
| Library access | .NET BCL, NuGet packages | Alloy, Fidelity-compatible libraries |
| Reflection | Full runtime reflection | None (types erased after compilation) |
| Script files | `.fsx` | `.fsnx` (planned) |
| Memory visibility | Hidden (GC managed) | Explicit (`:memory` command) |

---

## Related Resources

These resources informed our design thinking:

| Resource | Purpose |
|----------|---------|
| [Evcxr](https://github.com/evcxr/evcxr) | Rust REPL reference implementation |
| [Clang-Repl](https://clang.llvm.org/docs/ClangRepl.html) | C++ incremental JIT REPL |
| [Swift REPL](https://www.swift.org/documentation/lldb/) | Debugger-integrated REPL |
| [LLVM ORC JIT](https://llvm.org/docs/ORCv2.html) | LLVM's JIT compilation infrastructure |
| [MLIR ExecutionEngine](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) | MLIR JIT compilation tutorial |
| [jank](https://jank-lang.org/) | Clojure dialect using LLVM JIT |
| [LFortran](https://lfortran.org/) | Fortran with REPL via LLVM JIT |

---

## Conclusion

FSNI appears achievable by leveraging LLVM's ORC JIT for low-latency code generation, incremental FNCS for fragment-by-fragment type checking, session-scoped arenas for memory management, and warm compiler state for responsive evaluation.

What we find encouraging is that Firefly already does most of the work. FSNI would add an incremental mode and JIT backend, not a fundamentally new architecture. The same FNCS that would check your files would check your REPL input. The same Alex that would generate production MLIR would generate interactive MLIR.

The implementation work lies ahead. This document represents our current thinking about how FSNI could work. As we build it, we expect to learn and adjust.
