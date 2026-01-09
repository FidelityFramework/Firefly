# FNCS TypedTree Architecture: From IL to Native

*A technical narrative on transforming F# Compiler Services for native-first compilation*

---

## Preface: Why This Document Exists

This document captures architectural discoveries made during the creation of FNCS (F# Native Compiler Services). The insights here emerged from a careful, methodical investigation into how the F# compiler represents types internally, and what must change for native-first compilation.

The central question was: **How do we remove BCL/IL dependencies from the F# compiler's type system while preserving its semantic richness?**

The answer required understanding not just *what* to change, but *why* the current architecture exists and *where* the boundaries of change actually lie.

---

## Part I: The Layered Architecture of FCS

### The Two Faces of the Typed Tree

FCS (F# Compiler Services) presents two interfaces to its typed representation:

```
                INTERNAL                           PUBLIC API
         (TypedTree.fs/fsi)                  (Symbols/Exprs.fs/fsi)
         ─────────────────                   ─────────────────────
              TType                              FSharpType
              Expr                               FSharpExpr
              Typar                              FSharpGenericParameter
              TyconRef                           FSharpEntity
              ValRef                             FSharpMemberOrFunctionOrValue
```

**The internal representation** (`FSharp.Compiler.TypedTree`) is what the compiler uses for type checking, optimization, and code generation. It uses IL types directly.

**The public API** (`FSharp.Compiler.Symbols`) wraps the internal representation for external consumers. Baker in Firefly consumes this public API exclusively.

This layering is crucial: **Baker never sees IL types directly.** It works with `FSharpType`, `FSharpExpr`, and other wrapper types. The IL types are an internal implementation detail.

### Why IL Types Exist

The internal TypedTree uses IL types because FCS was designed with a specific target in mind: **IL emission**. The typed tree is an intermediate representation on the path to generating .NET bytecode.

When you write F# code that references `System.String`, the compiler needs to:
1. Resolve that reference to a specific type in a specific assembly (mscorlib)
2. Understand the type's structure (methods, fields, interfaces)
3. Generate IL that correctly references that type

The IL types in TypedTree encode exactly this information:

```fsharp
type TILObjectReprData =
    | TILObjectReprData of
        scope: ILScopeRef *      // Which assembly?
        nesting: ILTypeDef list * // Nested type path?
        definition: ILTypeDef     // Full type metadata
```

This representation is perfect for IL emission. It is fundamentally wrong for native compilation.

---

## Part II: Where IL Types Actually Live

### The Focused Scope

A critical discovery: **IL types are NOT pervasive in TypedTree.** They appear in specific, well-defined locations.

#### 1. Type Import Mechanism

```fsharp
type TyconRepresentation =
    | TFSharpTyconRepr of FSharpTyconData   // F# types (records, unions, etc.)
    | TILObjectRepr of TILObjectReprData    // Imported .NET types ← IL HERE
    | TAsmRepr of ILType                     // Inline IL assembly
    | TMeasureableRepr of TType             // Measure-parameterized types
    | TProvidedTypeRepr of ...              // Type providers
    | TNoRepr                               // Not yet known
```

`TILObjectRepr` is used **only for types imported from .NET assemblies**. F# types (records, discriminated unions, classes defined in F#) use `TFSharpTyconRepr`, which does NOT contain IL types.

#### 2. Operations (TOp)

```fsharp
type TOp =
    // F# operations (no IL)
    | UnionCase of UnionCaseRef
    | Tuple of TupInfo
    | TraitCall of TraitConstraintInfo   // SRTP - critical to preserve!
    | ValFieldGet of RecdFieldRef
    // ... many more F# operations

    // IL-specific operations
    | ILAsm of instrs: ILInstr list * retTypes: TTypes  // Inline IL
    | ILCall of ... ilMethRef: ILMethodRef * ...        // .NET method calls
    | Goto of ILCodeLabel                                // State machines
    | Label of ILCodeLabel                               // State machines
```

IL types appear in:
- `ILAsm` - Inline IL assembly (`(# ... #)` syntax)
- `ILCall` - Calls to .NET methods
- `Goto`/`Label` - State machine compilation (but these are just `int` labels)

#### 3. Quotations (Minor)

Quotation expressions carry IL type references for reflection purposes. This is a specialized use case.

### What Does NOT Depend on IL

The core type representation is IL-free:

```fsharp
type TType =
    | TType_forall of typars: Typars * bodyTy: TType
    | TType_app of tyconRef: TyconRef * typeInstantiation: TypeInst * nullness: Nullness
    | TType_tuple of tupInfo: TupInfo * elementTypes: TTypes
    | TType_fun of domainType: TType * rangeType: TType * nullness: Nullness
    | TType_var of typar: Typar * nullness: Nullness
    | TType_measure of measure: Measure
    // ... no IL types here
```

The expression structure is also IL-free in its core:

```fsharp
type Expr =
    | Const of value: Const * range: range * constType: TType
    | Val of valRef: ValRef * flags: ValUseFlag * range: range
    | Sequential of expr1: Expr * expr2: Expr * kind: SequentialOpKind * range: range
    | Lambda of ... * bodyExpr: Expr * range: Text.range * overallType: TType
    | App of funcExpr: Expr * formalType: TType * typeArgs: TypeInst * args: Exprs * range: Text.range
    | Let of binding: Binding * bodyExpr: Expr * range: Text.range * frees: FreeVarsCache
    | Match of ... * decision: DecisionTree * targets: DecisionTreeTarget array * ...
    // ... IL appears only in Op cases
```

Every `Expr` carries:
- **Source range** - the location in source code
- **Type information** - resolved `TType`
- **References** - to declarations (`ValRef`, `TyconRef`)

The IL dependency enters through `Expr.Op` when the operation is `TOp.ILAsm` or `TOp.ILCall`.

---

## Part III: The FNCS Transformation

### What FNCS Must Change

For native-first compilation, we need to replace the IL-centric type import mechanism with a native-centric one.

| FCS Component | Purpose | FNCS Transformation |
|---------------|---------|---------------------|
| `TILObjectReprData` | Imported .NET types | `TNativeReprData` - native type layout |
| `ILScopeRef` | Assembly provenance | `NativeScopeRef` - module provenance |
| `ILTypeDef` | .NET type metadata | `NativeTypeDef` - native layout (size, alignment) |
| `ILMethodRef` | .NET method reference | `NativeFunctionRef` - native function |
| `TOp.ILCall` | .NET method call | `TOp.NativeCall` - native function call |
| `TOp.ILAsm` | Inline IL | **Remove** - no inline IL in native |
| `ILInstr` | IL instructions | Empty scaffolding (unused) |
| `ILCodeLabel` | State machine labels | Keep as-is (`int` alias) |

### What FNCS Preserves

The semantic richness of the typed tree is preserved:

- **Type inference and resolution** - `TType`, `Typar`, `TyparConstraint`
- **SRTP resolution** - `TOp.TraitCall` with `TraitConstraintInfo`
- **Expression structure** - `Expr` with ranges and types
- **F# type representations** - `TFSharpTyconRepr` for records, unions, classes

### The Native Type Layer

Instead of IL types representing .NET metadata, FNCS uses native types representing memory layout:

```fsharp
// FNCS: Native module provenance
type NativeScopeRef =
    | Local                           // Current compilation unit
    | NativeModule of ModulePath      // External native module (Alloy)
    | NativeLibrary of LibraryRef     // External native library (future: C interop)

// FNCS: Native type reference
type NativeTypeRef = {
    Scope: NativeScopeRef
    Path: string list                  // Module path
    Name: string                       // Type name
}

// FNCS: Native type definition (layout-focused)
type NativeTypeDef = {
    Name: string
    Fields: NativeFieldDef list
    Size: int                          // Size in bytes
    Alignment: int                     // Alignment requirement
    Region: MemoryRegion option        // Memory region constraint (UMX)
}
```

This representation carries the information needed for native code generation:
- **Size and alignment** for memory allocation
- **Field layout** for struct access (`llvm.extractvalue`)
- **Memory region** for Fidelity's memory-safe compilation

---

## Part IV: The Baker Opportunity

### The Current Architecture

In the current Firefly architecture, Baker exists as an external component that correlates the typed tree with the PSG (Program Semantic Graph):

```
FCS Output → [PSG Phase 1-3] → [BAKER (Phase 4)] → [PSG Enriched] → [ALEX] → MLIR
```

Baker uses a **dual-zipper traversal** - walking `FSharpExpr` (typed tree) and `PSGNode` (syntax-derived graph) in parallel, correlating by source range.

This is necessary because FCS produces syntax and typed trees as **separate outputs**:
- `SynExpr` - the syntax tree from parsing
- `FSharpExpr` - the typed tree from type checking

These are not explicitly correlated by FCS. Baker reconstructs the correlation externally.

### The Opportunity in FNCS

Since FNCS is a modification of the compiler itself (not an external consumer), we have an opportunity to build this correlation directly into the compiler.

Consider what Baker extracts:
1. **Resolved types** - Already in `Expr` as `TType`
2. **SRTP resolution** - Already in `TOp.TraitCall` as `TraitConstraintInfo`
3. **Source ranges** - Already in every `Expr` case
4. **Member bodies** - Available via `FSharpImplementationFileDeclaration`

All of this information exists in the typed tree. Baker's job is to make it accessible in a structure that parallels the PSG.

**What if FNCS exposed an API that provided this correlation directly?**

```fsharp
// Hypothetical FNCS API
type CorrelatedExpr = {
    TypedExpr: FSharpExpr              // The typed expression
    SyntaxRef: SynExpr option          // Reference to source syntax
    ResolvedSRTP: SRTPResolution list  // Resolved trait calls
    MemberBody: FSharpExpr option      // For function definitions
}
```

If FNCS maintained syntax correlation during type checking, Baker's dual-zipper traversal would become unnecessary. The correlation would already exist.

This is not a required change for Phase A, but it represents a deeper opportunity: **moving Baker's logic back into the compiler where it has direct access to both representations.**

---

## Part V: Implementation Phases

### Phase A: Native Scaffolding

Create `NativeIL.fs` in FNCS that provides types with the same API surface as `IL.fs` but representing native concepts:

**Types to scaffold:**
- `NativeScopeRef` (mirroring `ILScopeRef`)
- `NativeTypeRef` (mirroring `ILTypeRef`)
- `NativeTypeDef` (mirroring `ILTypeDef`)
- `NativeMethodRef` (mirroring `ILMethodRef`)
- Empty `ILInstr` (never used - FNCS has no inline IL)
- `ILCodeLabel = int` (unchanged)
- `NativeMemberAccess` (mirroring `ILMemberAccess`)
- `NativeTypeDefAccess` (mirroring `ILTypeDefAccess`)

**Goal:** TypedTree compiles with native semantics.

### Phase B: Replace IL Operations

Modify TypedTree to use native operations:

- `TILObjectReprData` → `TNativeReprData`
- `TOp.ILCall` → `TOp.NativeCall`
- `TOp.ILAsm` → Remove (emit error if encountered)
- Quotation IL refs → Remove or replace

**Goal:** TypedTree represents native types, not IL types.

### Phase C: Type Resolution

Modify type checking to produce native type semantics:

- String literals → `string` with native semantics (UTF-8 fat pointer)
- Option types → Value semantics (stack-allocated)
- SRTP resolution → Native witness hierarchy

**Goal:** Type inference produces native types, not BCL types.

### Phase D: Baker Integration (Future)

Consider exposing correlation API from FNCS:

- Syntax references in typed tree
- Direct SRTP resolution access
- Member body accessibility

**Goal:** Eliminate need for external Baker correlation.

---

## Part VI: Key Principles

### 1. The IL Types Are Localized

IL types are not scattered throughout the compiler. They appear in specific, well-defined locations related to:
- Type import from .NET assemblies
- IL code emission
- Inline IL assembly

This makes the transformation tractable.

### 2. The Semantic Core is IL-Free

The core type representation (`TType`, `Typar`) and expression structure (`Expr`) do not directly depend on IL types. The semantic richness of F# type checking is preserved.

### 3. Scaffolding, Not Stubbing

We are creating **scaffolding** - structural support for building a native-first compiler. This is not throwaway stub code. The native type representations carry real semantic meaning:
- `NativeScopeRef` represents module provenance
- `NativeTypeDef` represents memory layout
- These are the foundations of native compilation

### 4. Baker is Correlation, Not Transformation

Baker doesn't transform the typed tree. It correlates it with the PSG. The information Baker extracts already exists in the typed tree. The opportunity is to make this information directly accessible, eliminating the need for external correlation.

### 5. SRTP is Preserved

`TOp.TraitCall` with `TraitConstraintInfo` is the mechanism for SRTP resolution. This is critical to Fidelity's compilation model and is preserved unchanged.

---

## Appendix: File Locations

### FCS Internal Types
- `/src/Compiler/TypedTree/TypedTree.fsi` - Type definitions
- `/src/Compiler/TypedTree/TypedTree.fs` - Implementation
- `/src/Compiler/AbstractIL/il.fsi` - IL type definitions
- `/src/Compiler/AbstractIL/il.fs` - IL type implementation

### FCS Public API
- `/src/Compiler/Symbols/Exprs.fsi` - FSharpExpr public API
- `/src/Compiler/Symbols/Symbols.fsi` - FSharpType, FSharpEntity, etc.

### FNCS Scaffolding (To Be Created)
- `/src/Compiler/AbstractIL/NativeIL.fsi` - Native type definitions
- `/src/Compiler/AbstractIL/NativeIL.fs` - Native type implementation

### Firefly Components
- `/src/Baker/` - Type correlation component
- `/src/Alex/` - MLIR generation component
- `/src/Core/PSG/` - Program Semantic Graph

---

## References

- `fncs_architecture` memory - FNCS design principles
- `baker_component` memory - Baker architecture
- `dual_zipper_parallel_traversal` memory - The correlation mechanism
- `architecture_principles` memory - Fidelity architectural constraints
- `/docs/FNCS_Ecosystem.md` - Three-repository architecture

---

*Document created: December 2024*
*Last updated: December 2024*
*Status: Living document - updated as FNCS development progresses*
