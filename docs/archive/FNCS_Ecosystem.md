# FNCS Ecosystem: The Fidelity Compilation Stack

## Overview

The Fidelity framework compiles F# to native binaries through a coordinated ecosystem of repositories. This document explains how the components work together and which repository to modify for different kinds of changes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FNCS Ecosystem                               │
│                                                                 │
│  fsnative-spec          fsnative           Firefly              │
│  ┌─────────────┐       ┌─────────────┐    ┌─────────────┐      │
│  │ F# Native   │       │ FNCS        │    │ PSG/Alex    │      │
│  │ Language    │──────▶│ Compiler    │───▶│ Native      │      │
│  │ Spec        │ impl  │ Services    │uses│ Pipeline    │      │
│  └─────────────┘       └─────────────┘    └─────────────┘      │
│        │                     │                   │              │
│        │                     │                   │              │
│        ▼                     ▼                   ▼              │
│   Normative rules      Typed tree +        MLIR → LLVM         │
│   for native types     SRTP resolution     → Native            │
│                                                                 │
│  BAREWire              Farscape            Alloy                │
│  ┌─────────────┐       ┌─────────────┐    ┌─────────────┐      │
│  │ Memory      │       │ C/C++       │    │ Native      │      │
│  │ Layouts     │       │ Bindings    │    │ Library     │      │
│  └─────────────┘       └─────────────┘    └─────────────┘      │
│        │                     │                   │              │
│        └──────────┬──────────┴───────────────────┘              │
│                   ▼                                             │
│          Type-safe hardware access                              │
│          Zero-copy IPC                                          │
│          Deterministic memory                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Repository Purposes

### Core Compilation

| Repository | Purpose | Primary Artifacts |
|------------|---------|-------------------|
| **fsnative-spec** | Language specification | Normative rules, type semantics |
| **fsnative** | F# Native compiler services | Typed trees, SRTP resolution |
| **Firefly** | Native compilation pipeline | PSG, Alex, MLIR generation |

### Supporting Libraries

| Repository | Purpose | Primary Artifacts |
|------------|---------|-------------------|
| **Alloy** | Native F# standard library | String/Array ops, Collections, Platform bindings |
| **BAREWire** | Memory-efficient serialization | Zero-copy IPC schemas |
| **Farscape** | C/C++ binding generator | Peripheral descriptors, FFI |
| **XParsec** | Parser combinator library | Used in Alex pattern matching |

## Data Flow

### Compilation Pipeline

```
F# Source Code
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FNCS (fsnative)                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Parse     │───▶│ Type Check  │───▶│   SRTP      │      │
│  │  (SynExpr)  │    │ (FSharpExpr)│    │ Resolution  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                            │                  │              │
│                            └────────┬─────────┘              │
│                                     ▼                        │
│                              Typed Tree +                    │
│                              SRTP Metadata                   │
└──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Firefly                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │    PSG      │───▶│  Nanopass   │───▶│   Alex/     │      │
│  │ Construction│    │  Pipeline   │    │   Zipper    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│        │                  │                   │              │
│        │                  │                   ▼              │
│        │                  │           ┌─────────────┐        │
│        │                  │           │  Bindings   │        │
│        │                  │           │ (Platform)  │        │
│        │                  │           └─────────────┘        │
│        │                  │                   │              │
│        └────────────┬─────┴───────────────────┘              │
│                     ▼                                        │
│               MLIR Module                                    │
└──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                     ┌─────────────────────────────┐
                     │  LLVM → Native Binary       │
                     └─────────────────────────────┘
```

### Type Resolution Flow

```
F# Syntax: let s = "Hello"
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Standard FCS                                                │
│  Type: System.String                                         │
│  (BCL reference type)                                        │
└─────────────────────────────────────────────────────────────┘

F# Syntax: let s = "Hello"
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  FNCS (fsnative)                                             │
│  Type: string (with native semantics)                        │
│  (UTF-8 fat pointer: {Pointer, Length})                     │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Firefly PSG                                                 │
│  Node: Const:String with native string semantics            │
│  + memory region: Stack                                      │
│  + access kind: ReadOnly                                     │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Alex/MLIR                                                   │
│  %str = llvm.mlir.constant("Hello\00") : !llvm.array<6 x i8>│
│  %ptr = llvm.mlir.addressof @str : !llvm.ptr                │
│  %len = llvm.mlir.constant(5 : i64) : i64                   │
│  (fat pointer: {ptr, len})                                   │
└─────────────────────────────────────────────────────────────┘
```

## Which Repository for Which Change?

### Adding a New Native Type

| Step | Repository | File(s) |
|------|------------|---------|
| 1. Specify semantics | fsnative-spec | `FNCS_Specification.md` Part 1 |
| 2. Implement resolution | fsnative | `TcGlobals.fs`, `NativeTypes.fs` |
| 3. Provide implementation | Alloy | Appropriate module |
| 4. Generate code | Firefly | `Alex/Bindings/*.fs` |

### Adding a New Platform Binding

| Step | Repository | File(s) |
|------|------------|---------|
| 1. Define API shape | Alloy | `Platform.fs` |
| 2. Specify semantics | fsnative-spec | `FNCS_Specification.md` Part 6 |
| 3. Implement binding | Firefly | `Alex/Bindings/PlatformBindings.fs` |

### Adding a New Memory Region

| Step | Repository | File(s) |
|------|------------|---------|
| 1. Specify semantics | fsnative-spec | `FNCS_Specification.md` Part 9 |
| 2. Add measure type | fsnative | `NativeTypes.fs` |
| 3. Add constraint checking | fsnative | `ConstraintSolver.fs` |
| 4. Handle in code gen | Firefly | `Alex/CodeGeneration/TypeMapper.fs` |

### Adding a Peripheral Family (via Farscape)

| Step | Repository | File(s) |
|------|------------|---------|
| 1. Parse C headers | Farscape | `CppParser.fs` |
| 2. Generate F# bindings | Farscape | `TypeMapper.fs`, `CodeGen.fs` |
| 3. Recognize attributes | fsnative | `PeripheralAttributes.fs` |
| 4. Generate register access | Firefly | `Alex/Bindings/PeripheralBindings.fs` |

### Changing SRTP Resolution

| Step | Repository | File(s) |
|------|------------|---------|
| 1. Specify resolution order | fsnative-spec | `FNCS_Specification.md` Part 3 |
| 2. Implement witness search | fsnative | `NativeSRTP.fs`, `ConstraintSolver.fs` |
| 3. Expose via API | fsnative | `FNCSPublicAPI.fs` |
| 4. Consume in PSG | Firefly | `Core/PSG/Builder.fs` |

## Integration Points

### FNCS → Firefly

Firefly consumes FNCS output through the public API:

```fsharp
// Firefly/src/Core/FCS/Integration.fs
open FSharp.Compiler.Service
open FNCS.PublicAPI

let processSource (source: string) =
    let checker = FSharpChecker.Create()
    let parseResults, checkResults = checker.ParseAndCheckFileInProject(...)

    // FNCS-specific APIs
    let rangeCorrelation = RangeCorrelationService.create checkResults
    let srtpResolutions = SRTPService.getResolutions checkResults

    // Build PSG with FNCS metadata
    PSG.Builder.create parseResults checkResults rangeCorrelation srtpResolutions
```

### Alloy → FNCS

FNCS resolves types against Alloy definitions:

```fsharp
// fsnative type resolution
// When seeing: let s = "Hello"

// 1. Standard FCS would resolve to System.String
// 2. FNCS provides native semantics for string
// 3. string has UTF-8 fat pointer representation {Pointer, Length}

// When seeing SRTP: a + b where a, b : int
// 1. Standard FCS searches System.Int32.op_Addition
// 2. FNCS searches Alloy.BasicOps.Add<int>
```

### Farscape → FNCS → Firefly

Peripheral bindings flow through all layers:

```fsharp
// 1. Farscape generates from SVD/headers:
[<PeripheralDescriptor("GPIO", 0x48000000UL)>]
type GPIO_TypeDef = { ... }

// 2. FNCS recognizes attributes, preserves in typed tree
// 3. Firefly PSG captures peripheral metadata
// 4. Alex generates volatile memory access patterns
```

### BAREWire → Firefly

Memory layouts inform code generation:

```fsharp
// BAREWire schema defines wire format
// Firefly respects alignment, packing, endianness
// Alex generates zero-copy serialization code
```

## Cross-Repository Dependencies

```
                    fsnative-spec
                         │
            specifies    │    specifies
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      fsnative        Firefly         Alloy
         │               │               │
         │    uses       │    uses       │
         └───────┬───────┴───────────────┘
                 ▼
            Application
                 │
                 │    may use
         ┌───────┴───────┐
         ▼               ▼
     BAREWire        Farscape
```

**Build Order:**
1. XParsec (no deps)
2. Alloy (no runtime deps, only specification deps)
3. fsnative (depends on Alloy types conceptually)
4. Firefly (depends on fsnative, Alloy, XParsec)
5. BAREWire, Farscape (optional, application-level)

## Serena Memory References

For detailed architectural decisions, consult Serena memories:

| Memory | Contents |
|--------|----------|
| `architecture_principles` | Core constraints, layer separation |
| `fncs_architecture` | FNCS-specific design decisions |
| `alex_zipper_architecture` | Zipper + XParsec + Bindings model |
| `negative_examples` | Mistakes to avoid |
| `fncs_ecosystem` | This document's summary |

## Quick Reference

### I want to...

| Goal | Repository |
|------|------------|
| Define what F# Native means | fsnative-spec |
| Implement type resolution | fsnative |
| Add a platform syscall | Alloy + Firefly |
| Generate code for a construct | Firefly |
| Add a native collection type | Alloy |
| Define memory layout for IPC | BAREWire |
| Generate bindings from C headers | Farscape |
| Use parser combinators in Alex | XParsec |

### Something is broken in...

| Symptom | Likely Repository |
|---------|-------------------|
| Wrong type inferred | fsnative |
| SRTP not resolving | fsnative |
| Missing platform function | Alloy |
| Bad code generated | Firefly |
| Peripheral access broken | Farscape + Firefly |
