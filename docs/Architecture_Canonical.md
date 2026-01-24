# Firefly Architecture: Canonical Reference

> **Memory Architecture**: For hardware targets (CMSIS, embedded), see [Quotation_Based_Memory_Architecture.md](./Quotation_Based_Memory_Architecture.md)
> which describes the quotation + active pattern infrastructure spanning fsnative, BAREWire, and Farscape.
>
> **Desktop UI Stack**: For WebView-based desktop applications, see [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md)
> which describes Partas.Solid frontend + Firefly native backend with system webview rendering.

## The Pipeline Model

**ARCHITECTURE UPDATE (January 2026)**: Alloy absorbed into FNCS. Types and operations are compiler intrinsics.

```
┌─────────────────────────────────────────────────────────┐
│  F# Application Code                                    │
│  - Uses FNCS intrinsics: Console.writeln, Sys.write    │
│  - Types provided by NTUKind: string, int, Uuid, etc.  │
│  - NO external library dependencies                     │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Compiled by FNCS
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FNCS (F# Native Compiler Services)                     │
│  - Parses F# source (SynExpr, SynModule)                │
│  - Type checking with NTUKind native types              │
│  - SRTP resolution during type checking                 │
│  - Intrinsic modules: Sys.*, NativePtr.*, Console.*    │
│  - PSG CONSTRUCTION with intrinsic markers              │
│                                                         │
│  OUTPUT: PSG with native types, intrinsics marked       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ PSG (correct by construction)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FIREFLY (Consumes PSG from FNCS)                       │
├─────────────────────────────────────────────────────────┤
│  Lowering Nanopasses (if needed)                        │
│  - FlattenApplications, ReducePipeOperators             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex (Compiler Targeting Layer)                        │
│  - Consumes PSG as "correct by construction"            │
│  - NO type checking needed - trusts FNCS                │
│  - Zipper traversal + XParsec pattern matching          │
│  - Intrinsic → MLIR mapping using NTUKind               │
│  - Platform implementations for Sys.* intrinsics        │
└─────────────────────────────────────────────────────────┘
```

**FNCS-First Architecture:** Types and operations ARE the compiler, not library code:
- **FNCS**: Defines NTUKind types, provides intrinsic modules, builds PSG with intrinsics marked
- **Alex**: Traverses PSG → generates MLIR → LLVM → native binary
- **No external library**: Following ML/Rust/Triton-CPU patterns, types ARE the language

**Zipper Coherence:** Alex uses PSGZipper to traverse the PSG from FNCS:
- PSG comes from FNCS with all type information attached
- Intrinsics are marked with `SemanticKind.Intrinsic`
- Alex maps intrinsics to platform-specific MLIR

## Alloy: Historical Archive (Absorbed January 2026)

> **Note**: Alloy has been absorbed into FNCS. The repository is preserved as a historical artifact.
> See blog entry: [Absorbing Alloy](/blog/absorbing-alloy/)

Alloy was a BCL-free F# standard library that proved native compilation was possible. Its functionality is now provided by FNCS intrinsic modules.

### What Alloy Taught Us

Alloy demonstrated:
- **BCL-free F# is possible** - No System.* dependencies needed
- **Fat pointer types work** - NativeStr, NativeArray as (ptr, length) structs
- **SRTP enables zero-cost abstractions** - Compile-time polymorphism

These lessons are now embodied in FNCS as NTUKind types and intrinsic modules.

### FNCS Intrinsics (Replacement)

What was Alloy is now FNCS:

```fsharp
// FNCS intrinsic modules - defined in CheckExpressions.fs
// Sys.* for platform operations
| "Sys.write" -> NativeType.TFun(intType, TFun(ptrType, TFun(intType, intType)))
| "Sys.clock_gettime" -> NativeType.TFun(unitType, int64Type)

// Console.* for I/O (thin wrappers over Sys.*)
| "Console.writeln" -> NativeType.TFun(stringType, unitType)

// Application code uses intrinsics directly
let main() =
    Console.writeln "Hello, World!"  // FNCS intrinsic, not library call
```

**Why intrinsics?** Following ML/Rust/Triton-CPU patterns:
- Types ARE the language (NTUKind)
- Operations ARE the language (intrinsic modules)
- No external library needed

## Alex: The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

Alex generates MLIR through Zipper traversal and platform Bindings. There is **NO central dispatch hub**.

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)     -- provides "attention"
    ↓
Fold over structure (pre-order/post-order)
    ↓
At each node: XParsec matches locally → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates           -- correct centralization
    ↓
Output: Complete MLIR module
```

**Component Roles:**

- **Zipper**: Purely navigational - provides focus with context, carries state (SSA counters)
- **XParsec**: Local pattern matching - composable patterns, NOT a routing table
- **Bindings**: Platform-specific MLIR - looked up by extern entry point, are DATA not routing
- **MLIR Builder**: Where centralization correctly occurs - the single accumulation point

**Bindings are DATA, not routing logic:**

```fsharp
// Syscall numbers as data
module SyscallData =
    let linuxSyscalls = Map [
        "write", 1L
        "read", 0L
        "clock_gettime", 228L
    ]
    let macosSyscalls = Map [
        "write", 0x2000004L  // BSD offset
        "read", 0x2000003L
    ]

// Bindings registered by (OS, Arch, EntryPoint)
ExternDispatch.register Linux X86_64 "fidelity_write_bytes"
    (fun ext -> bindWriteBytes TargetPlatform.linux_x86_64 ext)
```

**NO central dispatch match statement. Bindings are looked up by entry point.**

## The Fidelity Mission

Unlike Fable (AST→AST, delegates memory to target runtime), Fidelity:
- **Preserves type fidelity**: F# types → precise native representations
- **Preserves memory fidelity**: Compiler-verified lifetimes, deterministic allocation
- **PSG carries proofs**: Not just syntax, but semantic guarantees about memory, types, ownership

The generated native binary has the same safety properties as the source F#.

## FNCS Intrinsic Modules

FNCS provides intrinsic modules that Alex maps to platform-specific implementations:

| Intrinsic Module | MLIR Mapping | Purpose |
|-----------------|--------------|---------|
| Sys.write | write syscall | Low-level I/O |
| Sys.read | read syscall | Low-level I/O |
| Sys.clock_gettime | clock_gettime | Wall clock time |
| Sys.clock_monotonic | clock_gettime(MONOTONIC) | High-resolution timing |
| Sys.tick_frequency | constant (platform-specific) | Timer resolution |
| Sys.nanosleep | nanosleep/Sleep | Thread sleep |
| Console.write | Sys.write wrapper | String output |
| Console.writeln | Sys.write + newline | Line output |
| Console.readln | Sys.read wrapper | Line input |

Alex provides implementations for each `(intrinsic, platform)` pair based on target platform.

> **Note**: Webview bindings call library functions (WebKitGTK, WebView2, WKWebView) rather than syscalls. See [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) for the full desktop UI stack architecture.

## File Organization

```
fsnative/src/Compiler/Checking.Native/  # TYPES AND OPERATIONS
├── NativeService.fs        # Public API for FNCS
├── NativeTypes.fs          # NTUKind enum - native type universe
├── NativeGlobals.fs        # Type constructors (string, int, Uuid, etc.)
├── CheckExpressions.fs     # Intrinsic modules (Sys.*, Console.*, etc.)
├── SemanticGraph.fs        # PSG data structures with intrinsic markers
├── SRTPResolution.fs       # SRTP resolution during type checking
└── NameResolution.fs       # Compositional name resolution

Firefly/src/Core/PSG/Nanopass/  # LOWERING PASSES (post-FNCS)
├── FlattenApplications.fs
├── ReducePipeOperators.fs
└── ...

Firefly/src/Alex/
├── Traversal/
│   ├── PSGZipper.fs       # Bidirectional traversal (attention)
│   └── PSGXParsec.fs      # Local pattern matching combinators
│   └── FNCSTransfer.fs    # Intrinsic → MLIR mapping
├── Bindings/
│   ├── BindingTypes.fs    # Platform types
│   ├── SysBindings.fs     # Sys.* platform implementations
│   └── ...
├── CodeGeneration/
│   ├── MLIRBuilder.fs     # MLIR accumulation (correct centralization)
│   └── TypeMapping.fs     # NTUKind → MLIR type mapping
└── Pipeline/
    └── CompilationOrchestrator.fs  # Entry point

Alloy/ (HISTORICAL ARCHIVE - absorbed into FNCS January 2026)
└── README.md              # Explains absorbed status
```

**Note:** Alloy functionality absorbed into FNCS as intrinsic modules.
**Note:** PSGEmitter.fs and PSGScribe.fs were removed - they were antipatterns.

## Anti-Patterns (DO NOT DO)

```fsharp
// WRONG: BCL dependencies anywhere
open System.Runtime.InteropServices
[<DllImport("__fidelity")>]
extern int writeBytes(...)  // NO! BCL pollution

// WRONG: Pattern matching on namespace names
match symbolName with
| "MyApp.Console.Write" -> ...  // NO! Use intrinsic markers

// WRONG: Expecting library to provide what compiler should
// (This was the Alloy anti-pattern - library as BCL equivalent)
open Alloy  // NO! Types are NTUKind, operations are intrinsics

// WRONG: Central dispatch hub (the "emitter" or "scribe" antipattern)
module PSGEmitter =
    let handlers = Dictionary<string, NodeHandler>()
    let emit node =
        match handlers.TryGetValue(getPrefix node) with
        | true, h -> h node
        | _ -> default node
// This was removed TWICE. Centralization belongs at MLIR Builder output,
// not at traversal dispatch.
```

**The correct model:**
- Types defined by NTUKind in FNCS
- Operations defined as intrinsic modules in FNCS
- Alex recognizes `SemanticKind.Intrinsic` markers
- Zipper provides attention (focus + context)
- XParsec provides local pattern matching
- MLIR Builder accumulates (correct centralization point)

## PSG Construction: Handled by FNCS

**ARCHITECTURE UPDATE**: PSG construction has moved to FNCS.

See: `docs/PSG_Nanopass_Architecture.md` for nanopass principles.
See: `docs/FNCS_Architecture.md` for how FNCS builds the PSG.

FNCS builds the PSG with:
- Native types attached during type checking
- SRTP resolved during type checking (not post-hoc)
- Full symbol information preserved for design-time tooling

Firefly receives the completed PSG and applies **lowering nanopasses** if needed:
- FlattenApplications
- ReducePipeOperators  
- DetectPlatformBindings
- etc.

### Why FNCS Builds PSG

With FNCS handling PSG construction:
- SRTP is resolved during type checking, not in a separate pass
- Types are attached to nodes during construction, not overlaid later
- No separate typed tree correlation needed
- Symbol information flows directly from type checker to PSG

## Validation Samples

These samples must compile WITHOUT modification:
- `01_HelloWorldDirect` - Console.write, Console.writeln (FNCS intrinsics)
- `02_HelloWorldSaturated` - Console.readln, interpolated strings
- `03_HelloWorldHalfCurried` - Pipe operators, string formatting

The samples use FNCS intrinsics directly. Compilation flow:
1. FNCS: Parse, type check with NTUKind types, recognize intrinsics
2. FNCS: Build PSG with intrinsics marked, SRTP resolved
3. Firefly: Receive PSG from FNCS ("correct by construction")
4. Firefly: Apply lowering nanopasses (flatten apps, reduce pipes, etc.)
5. Alex/Zipper: Traverse PSG, map intrinsics → MLIR
6. MLIR → LLVM → native binary

---

## Cross-References

### Core Architecture
- [FNCS_Architecture.md](./FNCS_Architecture.md) - FNCS and PSG construction (PRIMARY)
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) - Nanopass principles
- [Quotation_Based_Memory_Architecture.md](./Quotation_Based_Memory_Architecture.md) - Memory model for embedded targets
- Note: Baker_Architecture.md is deprecated - FNCS now handles type correlation

### Desktop UI Stack
- [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) - Partas.Solid + webview architecture
- [WebView_Build_Integration.md](./WebView_Build_Integration.md) - Firefly as unified build orchestrator
- [WebView_Desktop_Design.md](./WebView_Desktop_Design.md) - Implementation details (callbacks, IPC)

### QuantumCredential Demo
- [QC_Demo/](./QC_Demo/) - Demo documentation folder
- [QC_Demo/01_Demo_Strategy_Integrated.md](./QC_Demo/01_Demo_Strategy_Integrated.md) - Integrated demo strategy (desktop + embedded)

### Platform Bindings
- See `~/repos/Farscape/docs/` for native library binding patterns (quotation-based architecture)
