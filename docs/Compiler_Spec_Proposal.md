# Compiler Specification Proposal

> **Status**: Proposal
> **Date**: January 2026
> **Authors**: Fidelity Team

---

## 1. Motivation

### 1.1 The Gap Between Language and Implementation

The **fsnative-spec** documents what F# Native programmers can express: types, expressions, memory regions, and representation guarantees. It answers "what does this program mean?"

But there is no specification for **how the compiler achieves these semantics**. This gap matters because:

1. **Reproducibility**: Another implementer cannot build a conforming compiler without understanding the coeffect system, witness architecture, and elaboration machinery.

2. **Verification**: We cannot formally verify that the compiler preserves semantics without specifying what transformations are performed and what invariants they maintain.

3. **Evolution**: As the compiler evolves, we need a stable specification to ensure changes don't violate architectural principles.

### 1.2 A Functionally Pure Compilation Model

Firefly aspires to be a **functionally pure compiler** in a specific sense: the compilation pipeline is a series of deterministic, composable transformations with explicit data dependencies.

**This aspiration warrants careful qualification.** We have not thoroughly researched whether other commercial-grade compilers achieve similar purity. What we can assert is our design intent:

- **No mutable state in the core pipeline**: PSG construction, elaboration, saturation, and coeffect analysis are pure functions over immutable data structures.

- **Coeffects over effects**: Where traditional compilers might accumulate state during traversal (symbol tables, SSA counters), Firefly pre-computes this information as "coeffects" - metadata that witnesses observe but do not mutate.

- **Recipes as data**: FNCS elaboration produces "recipes" (declarative transformation specifications) rather than imperatively modifying the PSG. Fold-in applies recipes to produce a new PSG.

Whether this makes Firefly unique among production compilers is an open research question. What it provides is **architectural clarity**: each stage has well-defined inputs and outputs, making the compiler amenable to formal reasoning.

### 1.3 The Shared Edge Problem

F# Native exists in a rich ecosystem of related languages and runtimes. A compiler specification must draw clear lines:

| System | Shared Edge | Line of Distinction |
|--------|-------------|---------------------|
| **.NET/CLR** | F# syntax, core library signatures | No managed runtime, no GC, no reflection |
| **Rust** | Ownership intuitions, LLVM backend | No borrow checker, arena-based not RC-based |
| **OCaml** | ML heritage, algebraic types | Native types (NTU), no boxing, explicit regions |
| **F\*** (FStar) | Dependent types for verification | Coeffects vs effects, compilation vs proof |
| **Standard ML** | Module system, value restriction | Platform-aware type resolution (SRTP) |

Each shared edge creates opportunities for knowledge transfer but also risks of false assumptions. The compiler spec must make these boundaries explicit.

---

## 2. Proposed Scope

### 2.1 What the Compiler Spec Would Document

**Part I: Pipeline Architecture**
- FNCS → PSG → Alex → MLIR → LLVM → Native
- Stage responsibilities and interfaces
- Invariants maintained at each stage

**Part II: The Coeffect System**
- What coeffects exist (SSA, ClosureLayout, DULayout, YieldState, etc.)
- When coeffects are computed (post-saturation, pre-emission)
- How witnesses consume coeffects (observation, not mutation)

**Part III: Elaboration and Saturation**
- Recipe data structure
- Fan-out (recipe creation) and fold-in (recipe application)
- Baker decomposition for HOFs
- Intrinsic elaboration patterns

**Part IV: Witness Architecture**
- The Zipper as attention mechanism
- XParsec pattern matching
- Witness functions: PSG node → MLIR ops
- The transliteration principle (witnesses don't synthesize structure)

**Part V: Memory Management Compilation**
- Stack vs arena allocation decisions
- Escape analysis integration
- Closure capture analysis
- DU heterogeneity classification

**Part VI: Platform Targeting**
- Platform descriptors and bindings
- Dialect selection (portable vs backend-specific)
- ABI compliance

### 2.2 Relationship to fsnative-spec

The compiler spec has a **shared edge** with fsnative-spec:

```
fsnative-spec                          compiler-spec
─────────────────────────────────────────────────────────────
"Arena<'lifetime> is a bump           "Arena allocation uses
 allocator with these operations"  →   ArenaLayout coeffect with
                                       5 pre-computed SSAs"

"Closures capture variables           "ClosureLayout coeffect
 by value or by reference"         →   tracks CaptureSlot per
                                       captured variable"

"Result<'T,'E> uses pointer           "DULayout coeffect handles
 representation for heterogeneous  →   arena allocation for
 payloads"                             heterogeneous DUs"
```

The language spec says **what**; the compiler spec says **how**.

---

## 3. Architectural Principles to Document

### 3.1 The Mise-en-Place Principle

> All preparation is done before emission begins.

Coeffects are computed during analysis. Witnesses only observe pre-computed values. This separation enables:
- Parallel coefficient computation
- Deterministic SSA assignment
- Testable, isolated stages

### 3.2 The Transliteration Principle

> Witnesses transliterate structure; they do not synthesize it.

If a witness needs to emit a load-before-extract sequence, that structure should exist in the PSG (via elaboration), not be invented by the witness. This principle:
- Keeps witnesses simple and uniform
- Makes PSG the single source of truth
- Enables PSG-level optimization

### 3.3 The Coeffect-Not-Effect Principle

> Compiler stages produce coeffects (observable metadata), not effects (mutations).

Traditional compilers might increment an SSA counter during emission. Firefly pre-computes all SSA assignments as a coeffect map. This enables:
- Parallel emission (no shared mutable state)
- Reproducible builds (deterministic assignment)
- Easier debugging (inspect coeffects, not trace mutations)

### 3.4 The Recipe Pattern

> Transformations are specified as data, then applied uniformly.

FNCS doesn't imperatively rewrite the PSG. It produces recipes (replacement specifications), and fold-in applies them. This enables:
- Inspectable intermediate artifacts
- Transformation composition
- Rollback capability (keep original PSG)

---

## 4. Research Questions

Before claiming architectural uniqueness, we should investigate:

1. **Functional purity in production compilers**: How do GHC, MLton, rustc, and modern JITs manage state during compilation? Are there precedents for our coeffect model?

2. **Formal verification opportunities**: Can we leverage the pure pipeline for verified compilation? What would Coq/F*/Lean proofs look like?

3. **Performance implications**: Does the coeffect model impose overhead compared to mutable state? Where are the tradeoffs?

4. **Ecosystem alignment**: How do our architectural choices affect IDE integration, incremental compilation, and tooling?

---

## 5. Next Steps

1. **Skeleton creation**: Establish document structure in `/docs/compiler-spec/`
2. **Coeffect catalog**: Document all existing coeffects with their schemas
3. **Witness inventory**: Catalog witness functions and their PSG→MLIR mappings
4. **Cross-reference**: Link compiler-spec sections to fsnative-spec counterparts
5. **Literature review**: Research comparable compiler architectures

---

## 6. Document Location

This proposal recommends creating the compiler specification as a separate document set within Firefly:

```
/docs/
├── Architecture_Canonical.md          # Existing overview
├── Compiler_Spec_Proposal.md          # This document
└── compiler-spec/                     # New directory
    ├── 00-introduction.md
    ├── 01-pipeline-architecture.md
    ├── 02-coeffect-system.md
    ├── 03-elaboration-saturation.md
    ├── 04-witness-architecture.md
    ├── 05-memory-compilation.md
    └── 06-platform-targeting.md
```

The compiler-spec directory would contain normative specification documents, while this proposal serves as the rationale and roadmap.

---

## 7. Conclusion

A compiler specification serves multiple audiences:

- **Implementers** who need to understand the architecture
- **Verifiers** who want to prove correctness properties
- **Educators** who teach compiler construction
- **Contributors** who extend the compiler

By documenting the coeffect model, witness architecture, and elaboration system, we make Firefly's functional purity explicit and inspectable. This transparency is itself a contribution to compiler engineering practice.
