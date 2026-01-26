# Dimensional Types, Refinement Types, and the Formalism Boundary

> **Document Purpose**: Assessment of the relationship between dimensional types (units of measure), refinement types (array bounds), and coeffect analysis—establishing coherence of the Fidelity architecture's approach to type-level safety without mandatory formalism.

> **Related**: [James_Faure_Price_of_Rust.md](./James_Faure_Price_of_Rust.md) - Dialectic assessment of Faure's Rust critique

> **Date**: January 2026

---

## Executive Summary

This document addresses a key architectural question: **Are array bounds (refinement types) "in effect" dimensional types?**

The answer is **yes, with important nuance**. Both are indexed type families—types parameterized by some indexing structure. The Fidelity architecture handles them through a coherent three-level hierarchy that avoids the "slippery slope into formalism" while preserving the benefits of type-level safety.

**Key Insight**: The control-flow ↔ dataflow pivot (enabled by SSA ≡ functional programming equivalence) requires knowing the **shape** of data, not just its type. This is exactly what size-related dimensions and refinements provide.

---

## The Unifying Structure: Indexed Type Families

Both dimensional types and refinement types are **indexed type families**—types parameterized by some indexing structure:

| Type System | Index Domain | Example | Operations |
|-------------|--------------|---------|------------|
| **Units of Measure** | Abstract measures | `float<meters>` | Dimensional algebra (m × s⁻¹ = m/s) |
| **Memory Semantics** | Access modes | `Ptr<T, ReadOnly>` | Capability checking |
| **Region Types** | Lifetime parameters | `Arena<'lifetime>` | Lifetime ordering (⊇) |
| **Refinement Types** | Values (dependent) | `array<T, n>` | Arithmetic constraints (i < n) |

The SpeakEZ blog's key insight—"memory semantics and physical semantics are both dimensions"—extends naturally: **quantitative constraints are also dimensions, just indexed by values rather than abstract measures**.

---

## Why This Matters: The Control-Flow ↔ Dataflow Pivot

The [Dimensional Type Safety](https://speakez.io/blog/dimensional-type-safety/) article identifies the critical insight from Appel (1998)[^1]: **SSA ≡ functional programming**. This equivalence enables the pivot between control-flow (CPU) and dataflow (FPGA/CGRA) interpretations.

The pivot requires knowing the **SHAPE** of data, not just its type:

- Is this a fixed-size array or a stream?
- Is the iteration count statically known or dynamic?
- Can this loop be fully unrolled or must it be a state machine?

These are exactly the questions that refinement types answer:

```fsharp
// Fixed-size: enables full unrolling, spatial dataflow
let processFixed (data: array<float<celsius>, 16>) = ...

// Dynamic-size: requires state machine or streaming
let processStream (data: seq<float<celsius>>) = ...
```

The dimensional type (`float<celsius>`) is invariant across both. The **size dimension** (`16` vs unbounded) determines whether the pivot goes to spatial dataflow or sequential state machine.

---

## The Three-Level Hierarchy

The Fidelity architecture has three mechanisms that form a coherent hierarchy:

| Level | Mechanism | What It Tracks | Source Annotation | Example |
|-------|-----------|----------------|-------------------|---------|
| **1. Coeffects** | Compiler analysis | Quantitative properties | None (inferred) | SSA versions, usage counts, lifetimes |
| **2. Dimensional Types** | Indexed type families | Categorical properties | Measure declarations | `<meters>`, `<ReadOnly>`, `<Peripheral>` |
| **3. Refinement Types** | Dependent types (F*) | Value-indexed properties | F* annotations | `array<T, n>` where `n` is a value |

### Level 1: Coeffects (No Developer Annotation)

The compiler computes quantitative properties through analysis:

- **SSA Assignment**: Variable versions through mutations
- **Mutability Analysis**: Which bindings are mutable, which have address taken
- **Yield State Indices**: State machine structure for seq expressions
- **Pattern Bindings**: How values flow through match expressions
- **Lifetime Inference**: Which allocations can be stack vs arena

See [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md) for details.

### Level 2: Dimensional Types (Lightweight Annotation)

Developer declares measures; compiler preserves through PSG to inform code generation:

```fsharp
[<Measure>] type meters
[<Measure>] type seconds
[<Measure>] type ReadOnly
[<Measure>] type Peripheral

let velocity (d: float<meters>) (t: float<seconds>) : float<meters/seconds> = d / t
let gpioReg : Ptr<uint32, Peripheral, ReadOnly> = ...
```

This is FSharp.UMX as a compiler intrinsic. Zero runtime cost—erased at final lowering after all compilation decisions that benefit from dimensional information have been made.

### Level 3: Refinement Types (Opt-In F* Proofs)

For safety-critical paths requiring value-dependent proofs:

```fsharp
[<F* Requires("n > 0")>]
[<F* Ensures("length(result) = n")>]
let createBuffer (n: int) : array<float, n> = ...

[<F* Requires("i < length(arr)")>]
let safeIndex (arr: array<'T>) (i: int) : 'T = arr.[i]
```

This requires F* and SMT solving. Only used when proofs need to go beyond what coeffects can infer.

---

## Avoiding the "Slippery Slope into Formalism"

The three-level hierarchy addresses the concern about mandatory formalism:

### Most Code: Level 1 + Level 2

```fsharp
// No F* annotations needed
let computeTrajectory
    (velocity: float<meters/seconds>)
    (time: float<seconds>)
    : float<meters> =
    velocity * time  // Dimensional algebra verified at compile time
```

The developer writes standard F# with lightweight measure annotations. The compiler:
1. Verifies dimensional consistency (Level 2)
2. Infers SSA, lifetimes, allocation strategy (Level 1)
3. Generates optimized native code

### Optimization-Critical Code: Add Size Dimensions

```fsharp
[<Measure>] type tiny    // fits in registers
[<Measure>] type small   // fits in L1 cache
[<Measure>] type large   // requires streaming

type SizedSpan<'T, [<Measure>] 'sizeClass> = Span<'T>

// Enables full SIMD unroll
let vectorize (data: SizedSpan<float<meters>, tiny>) = ...

// Must use streaming approach
let stream (data: SizedSpan<float<meters>, large>) = ...
```

This is still Level 2—categorical dimensions, not value-dependent proofs. The compiler knows "tiny fits in registers" without knowing the exact count.

### Safety-Critical Code: Level 3 Refinements

```fsharp
[<F* Requires("writable(output) && length(output) >= length(input)")>]
[<F* Ensures("forall i. i < length(input) ==> output[i] = transform(input[i])")>]
let processData (input: Span<float>) (output: Span<float>) =
    for i in 0 .. input.Length - 1 do
        output.[i] <- transform input.[i]
```

This is opt-in. As stated in [Proof-Aware Compilation](https://speakez.io/blog/proof-aware-compilation/):

> "This makes formalism restrained and approachable... developers don't have to become experts with proofs in order to leverage their advantages."

---

## Array Bounds: Three Options

Array bounds can be handled at different levels depending on requirements:

### Option A: Bounds as Categorical Dimension (Level 2)

```fsharp
[<Measure>] type fixedSize
[<Measure>] type stream

type FixedArray<'T, [<Measure>] 'size> = ...
type StreamSeq<'T> = ...
```

Tells the compiler "fixed-size" vs "streaming" without exact count. Sufficient for control-flow ↔ dataflow pivot.

### Option B: Bounds as Coeffect (Level 1)

The compiler infers array sizes through SSA analysis where possible:

```fsharp
let arr = Array.zeroCreate 16  // Compiler knows: size is 16
for i in 0 .. arr.Length - 1 do  // Compiler proves: i < 16
    arr.[i] <- compute i
```

This is already what the coeffect system does—SSA assignment tracks versions, and analysis can track bounds when statically determinable.

### Option C: Bounds as Refinement (Level 3)

```fsharp
[<F* Requires("i < length(arr)")>]
let safeIndex (arr: array<'T>) (i: int) : 'T = arr.[i]
```

Requires F* and SMT. Only when proofs depend on runtime values.

---

## Alignment with "Memory Management by Choice"

The three-level hierarchy matches the memory management philosophy exactly:

| Memory Level | Type Level | Annotation | Developer Experience |
|--------------|------------|------------|----------------------|
| **Implicit** (compiler infers) | **Coeffects** | None | Write standard F#, compiler handles allocation |
| **Hints** (developer guides) | **Dimensional Types** | `[<Measure>]` | Lightweight annotations, zero runtime cost |
| **Explicit** (full control) | **Refinement Types** | F* annotations | Opt-in proofs for safety-critical paths |

Both systems follow the same principle: **progressive disclosure of complexity**.

---

## The Inline Default and Coeffect Analysis

In fsnative, `inline` is the default positioning (unlike .NET F# where it's opt-in). This enables:

1. **Aggressive inlining** exposes SSA structure
2. **SSA structure** enables coeffect analysis
3. **Coeffect analysis** determines allocation strategy and bounds
4. **Dimensional types** constrain the analysis results

The inline default means more code is analyzable without annotation—exactly the "formalism restrained and approachable" philosophy.

---

## The Mathematical Relationship

### Shared Structure

Both dimensional types and refinement types are:

1. **Indexed type families** - types parameterized by something
2. **Zero-cost abstractions** - erased at final lowering
3. **Constraint systems** - you can't add meters to seconds; you can't index beyond bounds
4. **Optimization guides** - bounds proofs enable unrolling; dimensional proofs enable hardware synthesis

### Key Difference

| Property | Dimensional Types | Refinement Types |
|----------|-------------------|------------------|
| **Index domain** | Abstract measures | Values |
| **Decidability** | Always decidable | May require SMT |
| **Annotation burden** | Low | Higher |
| **Formalism level** | Restrained | Full (opt-in) |

### Unified View

In the Fidelity architecture:

```
Coeffects (quantitative, inferred)
    ↓ provides
Dimensional Types (categorical, lightweight)
    ↓ constrains
Refinement Types (value-dependent, opt-in)
```

Each level builds on the previous. A function with dimensional types benefits from coeffect inference. A function with refinements benefits from both.

---

## Coherence Summary

| Concern | Mechanism | Annotation Burden | Formalism Level |
|---------|-----------|-------------------|-----------------|
| Physical units | Dimensional types | Low (`<meters>`) | Restrained |
| Memory access modes | Dimensional types | Low (`<ReadOnly>`) | Restrained |
| Variable lifetimes | Coeffect analysis | None (inferred) | None |
| Array size class | Dimensional types | Low (`<tiny>`) | Restrained |
| Exact array bounds | Refinement types (F*) | High (F* annotations) | Full (opt-in) |

The position is coherent because:

1. **Each level is independently useful** - Coeffects provide safety without annotation; dimensions add categorical safety; refinements add value-level proofs

2. **Higher levels are opt-in** - Most code uses Level 1 + Level 2; Level 3 is for safety-critical paths

3. **The mechanisms compose** - Refinement proofs can reference dimensional constraints; dimensional types benefit from coeffect inference

4. **The philosophy is consistent** - "Memory Management by Choice" and "Formalism by Choice" follow the same progressive disclosure pattern

---

## Implications for the Control-Flow ↔ Dataflow Pivot

The three-level hierarchy directly supports heterogeneous compilation:

| Target | Requires | Mechanism |
|--------|----------|-----------|
| **CPU (control-flow)** | SSA form, allocation strategy | Coeffects |
| **GPU (parallel)** | Memory hierarchy hints | Dimensional types |
| **FPGA (dataflow)** | Fixed-size vs streaming | Size dimensions |
| **Certified (verified)** | Bounds proofs | Refinement types |

The dimensional type `float<celsius>` is invariant across all targets. The size class determines the pivot strategy. The refinement proofs (if present) verify correctness.

---

## Conclusion

The intuition that array bounds are "in effect" dimensional types is correct. Both are indexed type families that constrain operations and guide compilation. The Fidelity architecture handles them through a coherent three-level hierarchy:

1. **Coeffects** - Quantitative properties, compiler-inferred
2. **Dimensional Types** - Categorical properties, lightweight annotation
3. **Refinement Types** - Value-indexed properties, opt-in proofs

This avoids the "slippery slope into formalism" by making each level independently useful and higher levels opt-in. The same philosophy applies to memory management ("Memory Management by Choice") and type-level verification ("Formalism by Choice").

The position is coherent. The tension between engineering pragmatism and mathematical rigor is navigated by making formalism **layered and optional** rather than **pervasive and mandatory**.

---

## References

[^1]: Appel, A. W. (1998). [SSA is Functional Programming](https://www.cs.princeton.edu/~appel/papers/ssafun.pdf). ACM SIGPLAN Notices, 33(4), 17-20.

### Related Documents

- [James_Faure_Price_of_Rust.md](./James_Faure_Price_of_Rust.md) - Dialectic assessment of Faure's Rust critique
- [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md) - Coeffect analysis details
- [NTU_Architecture.md](./NTU_Architecture.md) - Native Type Universe and erased assumptions
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Overall system architecture

### SpeakEZ Blog References

- [Dimensional Type Safety Across Execution Models](https://speakez.io/blog/dimensional-type-safety/)
- [Doubling Down on DMM and DTS](https://speakez.io/blog/doubling-down/)
- [Proof-Aware Compilation Through Hypergraphs](https://speakez.io/blog/proof-aware-compilation/)
- [Danger Close: Why Types Matter](https://speakez.io/blog/danger-close-why-types-matter/)

### Academic References

- [Functional Ownership through Fractional Uniqueness (OOPSLA 2024)](https://dl.acm.org/doi/10.1145/3649848)
- [A Mixed Linear and Graded Logic (CSL 2025)](https://arxiv.org/abs/2401.17199)
- [Quantitative Type Theory (Atkey)](https://bentnib.org/quantitative-type-theory.html)
- [Kennedy: Relational Parametricity and Units of Measure (POPL 1997)](https://dl.acm.org/doi/10.1145/263699.263761)

---

*Document generated from architectural analysis, January 2026*
