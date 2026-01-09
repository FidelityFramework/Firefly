# From F# to FsNative: The Evolution of Native F# Compilation

> **Status:** Breadcrumbs for future blog entry
> **Last Updated:** December 2025

## The Journey

This document captures key milestones in the evolution from standard F# (FCS-based) to FsNative (native-first compilation).

---

## Milestone: Units of Measure on Any Type (December 2025)

### The Problem in .NET F#

In standard F#, units of measure only work on numeric types:

```fsharp
// Works in F#
let distance: float<meters> = 100.0<meters>

// Does NOT work - measures are numeric-only
let userId: string<customerId> = "cust-123"  // Error!
```

The FSharp.UMX library provides a workaround via `[<MeasureAnnotatedAbbreviation>]`, but it's a library hack, not native support.

### The FsNative Solution

In fsnative, **measures work on any type**. This isn't an extension - it's the native behavior:

```fsharp
// Type parameters can be either types or measures
type TypeParamKind =
    | Type      // Regular: 'T
    | Measure   // Measure: 'region, 'access

// Type constructors track which parameters are which
type TypeConRef = {
    Name: string
    ParamKinds: TypeParamKind list  // [Type; Measure; Measure] for Ptr<'T, 'region, 'access>
    Layout: TypeLayout
}
```

### Why This Matters

Memory region and access tracking become first-class:

```fsharp
// Pointer to peripheral memory with read-write access
let gpio: Ptr<uint32, peripheral, rw> = ...

// DMA buffer - compiler tracks both region AND access
let buffer: Span<byte, dma, rw> = ...

// Compiler catches violations at compile time:
// - Can't write to read-only memory
// - Can't pass peripheral pointer where stack is expected
// - Access mode mismatches are type errors
```

### Built-in Memory Measures

| Region | Access |
|--------|--------|
| `stack` - scope-bound | `ro` - read-only |
| `arena` - allocator-managed | `wo` - write-only |
| `sram` - on-chip RAM | `rw` - read-write |
| `flash` - persistent storage | |
| `peripheral` - MMIO | |
| `dma` - DMA-accessible | |

### Implementation

- `Checking.Native/NativeTypes.fs` - TypeParamKind, updated TypeConRef
- `Checking.Native/NativeGlobals.fs` - MemoryRegions, AccessModes, Ptr/Span/Ref types
- `Checking.Native/UnionFind.fs` - Kind-aware type parameter generation

---

## Milestone: BCL Independence (December 2025)

### What Was Removed

The cascade deletion revealed that **3.2MB across 59 files** depended on IL import assumptions. These were removed:

- `Driver/CompilerImports.fs` - .NET assembly imports
- `Symbols/*` - FCS public API (10 files)
- `Service/*` - FCS service layer (~57 files)
- `Interactive/*` - FSI (not needed)
- All `InfoReader`, `MethodCalls`, `ConstraintSolver` - IL-based type checking

### What Was Built

A native type checker from scratch:

- `NativeTypes.fs` - Type representation without IL
- `NativeGlobals.fs` - Built-in types with native semantics
- `UnionFind.fs` - Efficient substitution
- `Unify.fs` - Unification algorithm
- `SemanticGraph.fs` - Output structure

### Key Insight

> The type checker must be **rebuilt**, not **pruned**.

The IL import assumption permeates FCS's entire type-checking layer. Selective removal creates more problems than starting fresh.

---

## Milestone: Native Type Semantics (December 2025)

### String → UTF-8 Fat Pointer

```fsharp
// .NET F#: string is System.String (GC-managed, UTF-16)
// FsNative: string is NativeStr (16-byte fat pointer: ptr + length, UTF-8)
let stringTyCon = mkTypeConRef "string" 0 (TypeLayout.Inline(16, 8))
```

### Option → Value Type

```fsharp
// .NET F#: option is reference type (can be null)
// FsNative: option is value type (tag + payload, no null)
let optionTyCon = mkTypeConRef "option" 1 (TypeLayout.Inline(-1, -1))
```

### No `obj`

There is no universal base type in fsnative. Every type has a known layout at compile time.

---

## Future Milestones

### Planned: SRTP During Type Checking

SRTP resolution moves from post-hoc nanopass to intrinsic type checking behavior.

### Planned: Hard Prune Reachability

Unreachable code is physically removed before handoff to Firefly (not soft-deleted).

### Planned: fsil Semantic Absorption

Functions become transparent by default; SRTP resolves through witness hierarchy automatically.

---

## Blog Entry Outline

1. **The Problem**: .NET runtime assumptions in F# compilation
2. **The Insight**: IL import permeates everything - rebuild, don't prune
3. **Native Types**: String as fat pointer, option as value type, no obj
4. **Measures Everywhere**: Memory regions and access modes as types
5. **The Result**: Type-safe hardware access, compile-time memory safety
6. **What's Next**: Self-hosting, XParsec parser

---

## References

- `fsnative/.serena/memories/umx_native_integration.md` - Technical details
- `fsnative/.serena/memories/native_type_checker_architecture.md` - Full architecture
- `Firefly/docs/FNCS_Architecture.md` - Ecosystem overview
