# F-00: Foundation Series Synopsis (Samples 01-10)

> **Status**: Retrospective | **Samples**: 01-10 | **Completed**: January 2026

## Overview

Samples 01-10 form the **Foundation Series** - the bootstrapping phase of the Firefly compiler. These samples were developed with ephemeral documentation (conversations, working sessions, iterative refinement) rather than formal PRDs. This document provides a retrospective synopsis of what was accomplished, both at the surface feature level and in the deeper compiler infrastructure.

**Key Insight**: Each sample appears simple on the surface ("hello world", "add two numbers"), but beneath the surface lies significant PSG enrichment, coeffect infrastructure, and MLIR emission patterns that compose upward into the Computation features (C-01+).

---

## The Foundation Principle

> "The standing art composes up."

Each sample in the Foundation Series wasn't just about making a feature work - it was about establishing **principled infrastructure** that subsequent samples build upon. A "hello world" that cheats doesn't help when you need closures. A DU that works by accident breaks when you need Result.map.

The Foundation Series demanded:
- **Correct PSG representation** - Structure in the graph, not synthesized during emission
- **Coeffect-driven emission** - Pre-computed metadata, observed by witnesses
- **Baker decomposition** - HOFs lowered to primitives before Alex sees them
- **Type-faithful MLIR** - Native representations, not managed runtime assumptions

---

## Sample Synopses

### Sample 01: HelloWorldDirect

**Surface**: Print "Hello, World!" to the console.

**Deeper Infrastructure**:
- **String representation**: UTF-8 fat pointer (`{ptr, length}`) as NTU primitive, not managed System.String
- **Console intrinsics**: `Sys.write` syscall binding through platform descriptors
- **Basic MLIR emission**: Entry point generation, function scaffolding, return value handling
- **Static data**: String literals in `.rodata` section with proper null termination

**Architectural Contribution**: Established that even "simple" output requires the full FNCS→PSG→Alex→MLIR pipeline. No shortcuts.

---

### Sample 02: HelloWorldSaturated

**Surface**: Read user input, display personalized greeting.

**Deeper Infrastructure**:
- **Arena allocation**: `Arena<'lifetime>` as FNCS intrinsic with deterministic bump allocation
- **byref parameters**: `byref<Arena<'lifetime>>` for in-place arena modification
- **Console.readlnFrom**: Arena-backed string allocation that survives function returns
- **String interpolation**: Lowered to concatenation intrinsics (no runtime formatting)
- **Let bindings**: SSA assignment for local variables, scope tracking

**Architectural Contribution**: Demonstrated that memory management is explicit and deterministic. Arenas are stack-created, explicitly passed, and enable controlled allocation lifetimes.

---

### Sample 03: HelloWorldHalfCurried

**Surface**: Use F# pipe operator (`|>`) for data flow.

**Deeper Infrastructure**:
- **Pipe operator reduction**: `ReducePipeOperators` nanopass transforms `x |> f` to `f x` in PSG
- **Function application patterns**: Partial application detection and handling
- **Inline expansion**: Stack allocation movement for inlined functions
- **Control flow normalization**: Pipes become direct calls before MLIR emission

**Architectural Contribution**: Established that F# syntactic sugar is resolved in PSG transformations, not in code generation. Alex sees normalized structure.

---

### Sample 04: HelloWorldFullCurried

**Surface**: Curried functions, partial application, `fun` expressions.

**Deeper Infrastructure**:
- **Lambda representation**: `SemanticKind.Lambda` with parameter list and body
- **Partial application**: Curried function binding creates thunks
- **Closure preparation**: Lambda nodes structured for capture analysis (C-01)
- **Function types**: `TFun` chains for multi-parameter curried functions

**Architectural Contribution**: Laid groundwork for closure infrastructure. Lambdas exist in PSG with proper structure; Sample 11 adds capture analysis on top.

---

### Sample 05: AddNumbers

**Surface**: Pattern match on discriminated union (`Number = IntVal | FloatVal`).

**Deeper Infrastructure**:
- **DU representation**: Homogeneous DUs with inline struct layout `{tag, payload}`
- **Pattern matching lowered**: `Match` → `IfThenElse` decision tree via `MatchRecipes`
- **Tag extraction**: `TupleGet` for accessing tag field
- **Payload extraction**: Type-aware access to union payload slot
- **Bits coercion**: `Bits.float64ToInt64Bits` for storing float payloads in int64 slots
- **DUConstruct lowering**: `UnionCase` → `DUConstruct` transformation in Baker saturation

**Architectural Contribution**: Established that DUs are about **representation fidelity**. The compiler computes the memory layout; witnesses emit type-faithful access patterns.

---

### Sample 06: AddNumbersInteractive

**Surface**: Parse user input to detect int vs float, add numbers with type promotion.

**Deeper Infrastructure**:
- **String.contains intrinsic**: Character search for decimal point detection
- **Parse intrinsics**: `Parse.int`, `Parse.float` for string-to-number conversion
- **Multi-way pattern matching**: Tuple patterns `(a, b)` in match expressions
- **Type promotion**: `float x` conversion intrinsic
- **Multiple Console.readln calls**: Proper SSA isolation between reads

**Architectural Contribution**: Demonstrated compound control flow with multiple pattern matches and type conversions, all expressed in PSG and lowered through Baker.

---

### Sample 07: BitsTest

**Surface**: Byte-order conversion (htons/ntohl) and bit-casting (float↔int bits).

**Deeper Infrastructure**:
- **Bits intrinsic module**: Network byte order (`htons`, `ntohs`, `htonl`, `ntohl`)
- **Bit-level reinterpretation**: `float32ToInt32Bits`, `int32BitsToFloat32`, `float64ToInt64Bits`, `int64BitsToFloat64`
- **No-computation casts**: LLVM `bitcast` for reinterpretation without conversion
- **DU slot storage**: Foundation for storing different-typed payloads in uniform DU slots

**Architectural Contribution**: Established type reinterpretation patterns essential for heterogeneous DU implementation and binary protocol handling.

---

### Sample 08: Option

**Surface**: `Some`/`None` construction, pattern matching on Option type.

**Deeper Infrastructure**:
- **Homogeneous DU representation**: Option uses inline struct `{tag: i8, value: T}`
- **None encoding**: Tag = 0, value slot undefined
- **Some encoding**: Tag = 1, value slot populated
- **isSome/isNone intrinsics**: Direct tag comparison
- **Option.get intrinsic**: Payload extraction (unchecked)
- **OptionRecipes in Baker**: `Option.map`, `Option.bind`, `Option.filter` decomposition

**Architectural Contribution**: Option is the canonical homogeneous DU. Its inline representation (no arena allocation) sets the pattern for single-payload-type unions.

---

### Sample 09: Result

**Surface**: `Ok`/`Error` construction, pattern matching on Result type.

**Deeper Infrastructure**:
- **Heterogeneous DU representation**: Result uses arena allocation when `'T ≠ 'E`
- **DULayout coeffect**: Pre-computed layout for arena-allocated DU construction
- **Case-specific structs**: `{tag: i8, payload: T}` built inline, then stored to arena
- **Arena pointer return**: Result value is pointer to arena-allocated case struct
- **Uniform extraction**: Pattern match extracts tag, then type-specific payload

**Architectural Contribution**: Result is the canonical heterogeneous DU. Its arena-allocated representation demonstrates the DULayout coeffect pattern that enables type-safe union representation without boxing.

---

### Sample 10: Records

**Surface**: Record types, field access, copy-and-update (`{ r with field = value }`).

**Deeper Infrastructure**:
- **Record representation**: Struct layout with named fields at computed offsets
- **Field access**: GEP-based field extraction by known offset
- **Copy-and-update**: Struct copy with single field modification
- **Nested records**: Compound struct types with proper alignment
- **Pattern matching on records**: Field extraction patterns (`{ Age = a }`)
- **Guard expressions**: `when` clauses in pattern matches

**Architectural Contribution**: Records establish structured data representation with named field access - essential for capturing closures, state machines, and typed IPC messages.

---

## Cross-Cutting Infrastructure

### Coeffect System Evolution

| Coeffect | Introduced | Purpose |
|----------|------------|---------|
| NodeSSAAllocation | Sample 01 | Pre-computed SSA assignments for all PSG nodes |
| PatternBindings | Sample 05 | SSA assignments for pattern match bindings |
| ClosureLayout | Sample 04 (prep) | Flat closure struct layout (activated in C-01) |
| DULayout | Sample 09 | Arena allocation for heterogeneous DU construction |

### Baker Saturation Growth

| Recipe Module | Introduced | Operations |
|---------------|------------|------------|
| MatchRecipes | Sample 05 | Match → IfThenElse decision trees |
| OptionRecipes | Sample 08 | Option.map, Option.bind, Option.filter |
| ResultRecipes | Sample 09 | Result.map, Result.mapError, Result.bind |
| ListRecipes | (foundation) | List HOFs for downstream samples |
| MapRecipes | (foundation) | Map operations |
| SetRecipes | (foundation) | Set operations |
| SeqRecipes | (foundation) | Sequence HOFs |

### Type Representation Patterns

| Pattern | Example | Representation |
|---------|---------|----------------|
| Primitives | `int`, `float`, `string` | NTUKind direct MLIR types |
| Homogeneous DU | `Option<int>` | Inline struct `{i8, i32}` |
| Heterogeneous DU | `Result<int, string>` | Arena-allocated `ptr` to `{i8, payload}` |
| Records | `{Name: string; Age: int}` | Struct with computed offsets |
| Functions | `int -> string` | Closure struct `{code_ptr, captures...}` |

---

## Why This Matters for C-01+

The Foundation Series isn't just "samples that work" - it's **infrastructure that composes**:

1. **C-01 Closures** builds on Sample 04's lambda structure and Sample 10's record representation
2. **C-02 HOFs** uses the Baker decomposition established in Samples 05-09
3. **C-05 Lazy** reuses the thunk pattern from Sample 04 plus memoization state like Sample 10's records
4. **C-06/C-07 Seq** builds on closures plus state machines using patterns from all foundation samples
5. **A-01+ Async** synthesizes closures + state machines + DU results

Every architectural decision in Samples 01-10 has downstream consequences. The principled approach - correct PSG structure, coeffect-driven emission, Baker decomposition - means C-01+ can compose from standing art rather than fighting accumulated technical debt.

---

## Validation Status

| Sample | Compiles | Executes | Output Verified |
|--------|----------|----------|-----------------|
| 01_HelloWorldDirect | ✅ | ✅ | ✅ |
| 02_HelloWorldSaturated | ✅ | ✅ | ✅ |
| 03_HelloWorldHalfCurried | ✅ | ✅ | ✅ |
| 04_HelloWorldFullCurried | ✅ | ✅ | ✅ |
| 05_AddNumbers | ✅ | ✅ | ✅ |
| 06_AddNumbersInteractive | ✅ | ✅ | ✅ |
| 07_BitsTest | ✅ | ✅ | ✅ |
| 08_Option | ✅ | ✅ | ✅ |
| 09_Result | ✅ | ✅ | ✅ |
| 10_Records | ⏳ | ⏳ | ⏳ |

---

## Lessons Learned

### What Worked

1. **Principled coeffect design** - Pre-computing SSA assignments, closure layouts, and DU layouts enabled clean witness code
2. **Baker decomposition** - Lowering HOFs to primitives before emission kept Alex simple
3. **Ordinal artifact naming** - Pipeline-ordered intermediates (`01_psg0.json` through `07_output.mlir`) made debugging tractable
4. **Regression runner** - Automated validation caught regressions immediately

### What Was Hard

1. **DU representation bifurcation** - Homogeneous vs heterogeneous DUs needed different infrastructure
2. **Bits coercion discovery** - Float payloads in int slots required late-stage fixes
3. **Pattern binding SSAs** - Ensuring pattern match bindings got SSA assignments required traversal updates
4. **Arena affinity tracking** - Knowing which arena to allocate from required explicit propagation

### What We'd Do Differently

1. **Earlier coeffect documentation** - The coeffect pattern emerged organically; documenting it earlier would have accelerated C-01+
2. **Formalize representation rules** - The homogeneous/heterogeneous distinction should be in fsnative-spec
3. **More intermediate sample steps** - Some jumps (05→06, 09→10) were larger than ideal

---

## Related Documents

- **Compiler_Spec_Proposal.md** - Proposal for formal compiler specification
- **C-01-Closures.md** - First formal PRD, builds directly on foundation
- **Architecture_Canonical.md** - Authoritative architecture overview
- **Coeffects.fs** - Coeffect type definitions (NodeSSAAllocation, ClosureLayout, DULayout)
- **BakerSaturation.fs** - Baker decomposition implementation

---

## Acknowledgment

The Foundation Series was built through intensive iterative development with ephemeral documentation - design discussions, debugging sessions, and incremental refinement. This synopsis captures the technical substance; the journey was collaborative and emergent.

> "These samples look simple. They are not. Each one taught us something about what it means to compile F# to native code with fidelity."
