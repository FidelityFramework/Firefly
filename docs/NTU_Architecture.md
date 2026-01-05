# NTU (Native Type Universe) Architecture

## Executive Summary

The NTU (Native Type Universe) architecture provides platform-generic types for Fidelity that resolve via quotation-based platform bindings. This follows the F* pattern where type WIDTH is an **erased assumption**, not part of type identity.

**Core Principle**: Platform awareness flows FROM THE TOP via quotation-based binding libraries, not from FNCS type inference.

## Nomenclature

NTU = **N**ative **T**ype **U**niverse

The "NTU" prefix is used internally by FNCS to mark platform-generic types, similar to how FCS used "IL" prefixes extensively.

## NTU Type System

### Platform-Dependent Types (Resolved via Quotations)

| NTU Type | Meaning | x86_64 | ARM32 |
|----------|---------|--------|-------|
| `NTUint` | Platform word (signed) | i64 | i32 |
| `NTUuint` | Platform word (unsigned) | i64 | i32 |
| `NTUnint` | Native int (pointer-sized) | i64 | i32 |
| `NTUptr<'T>` | Native pointer | 8 bytes | 4 bytes |
| `NTUsize` | Size type (`size_t`) | u64 | u32 |
| `NTUdiff` | Pointer difference (`ptrdiff_t`) | i64 | i32 |

### Fixed-Width Types (Platform-Independent)

| NTU Type | Meaning | Always |
|----------|---------|--------|
| `NTUint8` | 8-bit signed | i8 |
| `NTUint16` | 16-bit signed | i16 |
| `NTUint32` | 32-bit signed | i32 |
| `NTUint64` | 64-bit signed | i64 |
| `NTUuint8` | 8-bit unsigned | u8 |
| `NTUuint16` | 16-bit unsigned | u16 |
| `NTUuint32` | 32-bit unsigned | u32 |
| `NTUuint64` | 64-bit unsigned | u64 |
| `NTUfloat32` | 32-bit float | f32 |
| `NTUfloat64` | 64-bit float | f64 |

## Option B: NTU Internal with Semantic Aliases

The architecture uses a layered approach:

```
┌─────────────────────────────────────────────────────────┐
│  Application Code                                        │
│  Level 1: int, uint (standard F#)                       │
│  Level 2/3: platformint, platformsize (explicit)        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alloy                                                   │
│  Semantic aliases: platformint, platformsize, etc.      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FNCS                                                    │
│  NTU types: NTUint, NTUuint, NTUsize, etc.              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex                                                    │
│  Witnesses quotations → concrete MLIR (i32/i64)         │
└─────────────────────────────────────────────────────────┘
```

### Mapping Table

| Level 1 (Default) | Level 2/3 (Explicit) | FNCS Internal |
|-------------------|----------------------|---------------|
| `int` | `platformint` | `NTUint` |
| `uint` | `platformuint` | `NTUuint` |
| `nativeint` | `nativeint` | `NTUnint` |
| `nativeptr<'T>` | `nativeptr<'T>` | `NTUptr<'T>` |

## Type Identity vs Type Width

**Key Insight from F***: Type WIDTH is an erased assumption, not part of type identity.

### What FNCS Enforces (Type Identity)
- `NTUint ≠ NTUint64` - These are **different types**
- `NTUint + NTUint` ✓ - Same type, valid
- `NTUint + NTUint64` ✗ - Different types, compile error

### What FNCS Does NOT Assume (Type Width)
- Whether `NTUint` is 32 or 64 bits
- Memory layout of platform-dependent types
- Exact byte sizes

### What Alex Resolves (Type Width)
- Platform quotations provide width information
- `NTUint` on x86_64 → `i64`
- `NTUint` on ARM32 → `i32`

## Integration with Memory Management by Choice

NTU supports all three memory management levels:

### Level 1 (Default): Invisible Management
- Developers write standard F#
- FNCS treats `int` as `NTUint` internally
- Platform quotations resolve width
- Compiler manages all memory decisions

### Level 2 (Hints): Memory Pattern Guidance
- Developers use `[<Struct>]`, capacity hints
- Platform bindings honor alignment requirements
- Compiler respects hints while optimizing

### Level 3 (Explicit): Precise Layouts
- Developers use `[<BAREStruct>]`, memory pools
- Platform bindings provide region specifications
- Full control over memory placement

## Platform Predicates (F*-Inspired)

Following F*'s `fits_u32`/`fits_u64` pattern:

```fsharp
module Platform.Predicates =
    /// Platform supports 32-bit word operations
    val fits_u32 : Expr<bool>

    /// Platform supports 64-bit word operations
    val fits_u64 : Expr<bool>

    /// 64-bit implies 32-bit support
    val fits_u64_implies_u32 : Expr<unit>

    /// Platform has AVX-512 vector support
    val has_avx512 : Expr<bool>

    /// Platform has NEON vector support
    val has_neon : Expr<bool>
```

These predicates enable conditional compilation without runtime checks.

## Implementation in FNCS

### NTUKind Discriminated Union

```fsharp
/// NTU (Native Type Universe) type kinds
type NTUKind =
    // Platform-dependent (resolved via quotations)
    | NTUint      // Platform word, signed
    | NTUuint     // Platform word, unsigned
    | NTUnint     // Native int (pointer-sized signed)
    | NTUunint    // Native uint (pointer-sized unsigned)
    | NTUptr of NativeType  // Pointer to type
    | NTUsize     // size_t equivalent
    | NTUdiff     // ptrdiff_t equivalent
    
    // Fixed width (platform-independent)
    | NTUint8 | NTUint16 | NTUint32 | NTUint64
    | NTUuint8 | NTUuint16 | NTUuint32 | NTUuint64
    | NTUfloat32 | NTUfloat64
```

### NTULayout (Erased Assumptions)

```fsharp
/// Platform-resolved type layout (erased at runtime)
type NTULayout = {
    Kind: NTUKind
    /// Erased - only for type checking, resolved by Alex
    AssumedSize: int option
    AssumedAlignment: int option
}
```

## Pipeline Flow

```
F# Source (int, platformint)
    │
    ▼
┌──────────────────────────────────────────────┐
│ FNCS: Maps to NTU types (NTUint)             │
│ - Type identity checking                      │
│ - SRTP resolution                             │
│ - Width is NOT resolved                       │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ SemanticGraph: Carries NTU annotations       │
│ - Platform quotations attached               │
│ - Type identity preserved                     │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Alex: Witnesses quotations                   │
│ - Reads Fidelity.Platform bindings           │
│ - Resolves NTUint → i64 on x86_64           │
│ - Generates platform-specific MLIR           │
└──────────────────────────────────────────────┘
    │
    ▼
Native Binary
```

## How This Resolves Type Errors

The 507 type errors in Alloy exist because it uses explicit `int64` where it should use `platformint`:

**Before (507 errors):**
```fsharp
let write (fd: int) (buf: nativeptr<byte>) (count: int) : int64 = ...
//                                                        ^^^^ explicit 64-bit
```

**After (NTU architecture):**
```fsharp
let write (fd: platformint) (buf: nativeptr<byte>) (count: platformsize) : platformint = ...
//             ^^^^^^^^^^^ resolved via quotations
```

FNCS sees `platformint` as `NTUint`, validates type identity (not width), and Alex resolves to `i64` on x86_64 via platform quotations.

## Related Documentation

- `Platform_Binding_Model.md` - Sophisticated platform binding capabilities
- `docs/fidelity/NTU_Type_System.md` (fsnative) - Implementation details
- `docs/fidelity/Platform_Predicates.md` (fsnative) - F*-style predicates
- `spec/ntu-types.md` (fsnative-spec) - Normative specification
