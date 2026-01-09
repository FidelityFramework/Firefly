# Extending Computation: Numerical Stability as a Compiler Concern

## Introduction

Modern numerical computation rests on a foundation that most developers never question: IEEE-754 floating-point arithmetic. This standard, formalized in 1985 and revised in 2008 and 2019, defines how computers represent and manipulate real numbers. It works well enough that entire industries, graphics, finance, scientific computing, machine learning, build upon it without much thought.

But "well enough" hides a quiet crisis. As computations grow more ambitious, deeper neural networks, longer simulations, more extreme parameter ranges, IEEE-754's limitations surface as mysterious failures. Training runs diverge. Simulations accumulate drift. Iterative algorithms produce nonsense after enough iterations. The standard response is to add precision: move from 32-bit to 64-bit floats, or to 128-bit, or to arbitrary precision libraries. This delays failure but doesn't prevent it.

The Fidelity framework takes a different approach. Rather than treating numerical representation as a fixed constraint that developers must work around, Firefly treats numerical stability as a first-class compilation concern. The compiler understands what the developer intends mathematically and chooses how to realize that intent on available hardware, whether through alternative number formats like posits, through algorithmic transformations that preserve stability in IEEE-754, or through intelligent distribution of computation across heterogeneous processors.

This document explains the problem, the solutions, and how Firefly's architecture enables numerical strategies impossible in conventional compilation.

## The IEEE-754 Precision Problem

### What Floating-Point Actually Does

A 64-bit IEEE-754 double stores a number as:

```
(-1)^sign × 1.mantissa × 2^(exponent - 1023)
```

The mantissa provides 52 bits of precision (plus an implicit leading 1), yielding roughly 15-16 decimal digits of accuracy. The exponent spans -1022 to +1023, covering magnitudes from roughly 10^-308 to 10^308.

This seems generous. Fifteen decimal digits should suffice for any reasonable calculation. The dynamic range spans phenomena from subatomic to cosmological scales. What could go wrong?

### Catastrophic Cancellation

Consider subtracting two nearly equal numbers:

```
a = 1.234567890123456
b = 1.234567890123455
a - b = 0.000000000000001
```

Both `a` and `b` are represented with 15-16 significant digits. But their difference has only 1 significant digit. The other 14-15 digits of the result are meaningless, artifacts of the representation, not information about the mathematical value.

This is catastrophic cancellation: when subtracting nearly equal values, most significant digits vanish. The relative error of the result can be arbitrarily large even when the inputs are exact.

### Accumulation Drift

Now consider an iterative computation:

```python
x = 1.0
for i in range(1000000):
    x = x + 0.0000001
    x = x - 0.0000001
```

Mathematically, `x` should remain exactly 1.0. In IEEE-754, it drifts. Each addition rounds. Each subtraction rounds. The rounding errors don't cancel, they accumulate. After enough iterations, `x` bears little relation to 1.0.

This pattern appears everywhere: numerical integration, iterative solvers, gradient descent, physical simulation. Every iteration introduces rounding error. Over enough iterations, the errors dominate the signal.

### The "Add More Bits" Response

The conventional solution is to increase precision. If 64-bit doubles fail, use 128-bit quad precision. Or Double-Double arithmetic (two doubles representing one value with ~31 decimal digits). Or arbitrary precision libraries.

This works, to a point. More bits delay failure. But consider:

1. **Performance cost**: Higher precision means more memory bandwidth and slower operations. Arbitrary precision can be 100-1000x slower than hardware floats.

2. **Hardware mismatch**: Modern GPUs and vector units are optimized for 32-bit and 64-bit floats. Higher precision often means falling back to scalar CPU code.

3. **Fundamental limits**: Catastrophic cancellation doesn't care how many bits you have. Subtracting 1.23456789... from 1.23456788... loses the same fraction of significant digits whether you started with 16 or 160.

More bits delay the problem. They don't solve it.

## A Different Approach: Stability Over Precision

### The Mandelbrot Insight

In 2025, developer Michael Stebel published an explanation of how his Mandelbrot Metal renderer achieves extreme zoom depths on GPU hardware. The insight is instructive.

The Mandelbrot iteration is simple:

```
z(n+1) = z(n)² + c
```

At modest zoom levels, this works fine in standard floats. But at extreme magnification, 10^16× and beyond, two problems emerge:

1. The coordinates `c` become extremely small (we're zoomed into a tiny region)
2. The iterated values `z` can grow large before escaping

Each iteration involves arithmetic between numbers differing by many orders of magnitude. Catastrophic cancellation destroys precision. Even Double-Double arithmetic eventually fails, not because 31 digits is insufficient, but because the arithmetic itself is unstable.

Stebel's solution: **perturbation theory**. Instead of computing each pixel's orbit independently, he computes one high-precision reference orbit, then tracks how each pixel's orbit *deviates* from the reference.

If `z₀(n)` is the reference orbit at iteration n, and a nearby pixel has orbit `z(n) = z₀(n) + δz(n)`, then:

```
z(n+1) = z(n)² + c
       = (z₀(n) + δz(n))² + (c₀ + δc)
       ≈ z₀(n)² + 2·z₀(n)·δz(n) + c₀ + δc
       = z₀(n+1) + 2·z₀(n)·δz(n) + δc
```

So the deviation updates as:

```
δz(n+1) = 2·z₀(n)·δz(n) + δc
```

The key insight: **δz stays small**. Nearby pixels have similar orbits. The deviation from the reference remains in a numerically stable range even as the absolute coordinates grow extreme. The arithmetic stays in the "sweet spot" of the floating-point format.

This is a general principle: **precision delays failure; stability prevents it**.

### The Audio Engineering Parallel

Engineers trained in digital audio recognize this pattern immediately. When mastering for CD (16-bit linear PCM), converting from 32-bit float production formats requires discarding precision. Naive truncation creates quantization distortion, harmonically related to the signal, audible as harshness.

The solution is dithering: adding carefully shaped noise before truncation. This trades coherent distortion for incoherent noise. The ear forgives random hiss far more than harmonic distortion.

But step back: why is any of this necessary? Because 16-bit linear PCM allocates precision uniformly across the amplitude range. A whisper and a scream get the same step size. But human hearing is logarithmic, we're more sensitive to quiet sounds. The representation doesn't match the perceptual domain.

Audio engineers spend enormous effort managing this mismatch. Dithering algorithms, noise shaping, psychoacoustic models, all to hide the fact that the representation is wrong for the domain.

The same pattern appears in numerical computing. IEEE-754 allocates precision uniformly across the exponent range. But most computations cluster around certain magnitudes. We burn cycles managing precision at magnitudes we rarely use while running out of precision where we need it.

## Universal Numbers: A Representation That Fits

### Posit Arithmetic

The Universal Numbers library, developed by Stillwater Supercomputing, offers an alternative representation called posits. Where IEEE-754 has fixed precision across all magnitudes, posits have **tapered precision**: more bits near 1.0, fewer bits at extremes.

A posit is parameterized by total bits and exponent size:

```
posit<32, 2>  // 32 bits total, 2-bit exponent field
posit<64, 3>  // 64 bits total, 3-bit exponent field
```

The variable-length exponent encoding means that numbers near 1.0, where most computation happens, get more mantissa bits than numbers near zero or infinity. This isn't a fixed trade-off; the format adapts to the magnitude being represented.

For iterative computations, this is transformative. The operating point of most algorithms is near 1.0 (or can be normalized to be). Posits give maximum precision exactly where the computation lives.

### The Quire: Exact Accumulation

Posits come with a companion structure: the quire. A quire is a wide fixed-point accumulator (typically 512 or more bits) that can hold the exact sum of many posit products without any rounding.

```cpp
quire<32, 2> q;        // Accumulator for posit<32,2>
q.clear();
for (int i = 0; i < 1000000; i++) {
    q.fma(a[i], b[i]);  // Fused multiply-add, exact
}
posit<32, 2> result = q.to_posit();  // Single rounding at the end
```

One million multiply-accumulates with a single rounding at the end. Compare to IEEE-754, which rounds after every operation, a million roundings, each introducing error that can compound.

The quire makes dot products, sums, and iterative accumulations exact up to the final conversion. For algorithms that suffer from accumulation drift, this is decisive.

### Hardware Availability

As of 2025, posit arithmetic is available in software libraries and FPGA implementations. Native silicon support is emerging:

- **NextSilicon** processors include native posit execution units
- Several research chips have demonstrated posit ALUs
- FPGA implementations show competitive performance with IEEE-754 for many workloads

The hardware landscape is heterogeneous. Some targets have native posit support. Most have only IEEE-754. A practical compilation strategy must handle both.

## The Fidelity Approach: Intent Over Representation

### The Core Insight

Conventional compilers treat the source code as specifying operations on representations. When you write `a + b` where `a` and `b` are `double`, you get an IEEE-754 double-precision addition. The representation is the semantics.

Firefly inverts this. Source code specifies **mathematical intent**. When you write `a + b`, you mean the sum of two real numbers. The compiler's job is to approximate that mathematical ideal as faithfully as the target hardware allows.

This creates a managed space between intent and execution. The compiler can:

1. Choose the best available representation (posit, IEEE-754, fixed-point)
2. Transform algorithms for stability (perturbation, compensated summation)
3. Distribute computation across heterogeneous hardware
4. Prove that the chosen realization meets specified accuracy bounds

### Two-Level Abstraction

Fidelity expresses numerical stability through two layers:

**Application level**: Developers write mathematical intent using computation expressions that hide numerical mechanism:

```fsharp
let integrate f a b steps =
    stable {
        let dx = (b - a) / float steps
        let mutable sum = 0.0
        for i in 0 .. steps - 1 do
            let x = a + float i * dx
            let! term = accumulate (f x * dx)
            sum <- sum + term
        return sum
    }
```

The `stable { }` computation expression signals that this code has numerical stability requirements. The `accumulate` keyword marks values that must be accumulated stably. The developer thinks in mathematics; the compiler thinks in representations.

**Library level**: Alloy.Numerics provides witness types that implement stability strategies:

```fsharp
type IStableArithmetic<'T> =
    abstract Zero: 'T
    abstract Accumulator: unit -> IAccumulator<'T>
    abstract Finalize: IAccumulator<'T> -> 'T

type PositArithmetic<'nbits, 'es>() =
    interface IStableArithmetic<Posit<'nbits, 'es>> with
        member _.Zero = Posit.Zero
        member _.Accumulator() = Quire<'nbits, 1024>() :> IAccumulator<_>
        member _.Finalize(q) = q.ToPosit()

type PerturbedArithmetic<'T when 'T :> IFloatingPoint<'T>>() =
    interface IStableArithmetic<Perturbed<'T>> with
        member _.Zero = Perturbed.Zero
        member _.Accumulator() = PerturbationTracker<'T>() :> IAccumulator<_>
        member _.Finalize(p) = p.Consolidate()
```

Library authors implement the mechanisms. Application developers use the abstractions. The compiler mediates between them.

### Target-Dependent Code Generation

The fidproj file declares target capabilities:

```toml
[package]
name = "ClimateModel"

[compilation]
numerical_strategy = "stable"
preferred_format = "posit<32,2>"
fallback_format = "ieee754_perturbed"

[target.nextsilicon]
triple = "nextsilicon-unknown-fidelity"
numerical_capability = "posit_native"

[target.amd_hsa]
triple = "amdgcn-amd-amdhsa"
numerical_capability = "ieee754"
compute_units = ["cpu", "gpu"]
```

When compiling for NextSilicon, the compiler generates native posit operations. The `stable { }` blocks lower to quire-based accumulation. No transformation needed, the hardware provides stability natively.

When compiling for AMD HSA (Heterogeneous System Architecture), the compiler transforms `stable { }` blocks using perturbation mechanics. The same source code generates IEEE-754 operations restructured for stability:

```fsharp
// Source (what developer writes)
stable {
    for i in 0 .. n do
        let! z = accumulate (z * z + c)
        // ...
}

// Generated for posit target
let q = Quire.Zero
for i in 0 .. n do
    q.Clear()
    q.FusedMultiplyAdd(z, z)
    q.Add(c)
    z <- q.ToPosit()
    // ...

// Generated for IEEE-754 with perturbation
let mutable z_ref = z_reference.[0]
let mutable delta_z = Complex.Zero
for i in 0 .. n do
    // δz' = 2·z₀·δz + δc (perturbation update)
    delta_z <- 2.0 * z_ref * delta_z + delta_c
    z_ref <- z_reference.[i + 1]
    // Consolidate when delta grows too large
    if delta_z.Magnitude > consolidation_threshold then
        z_ref <- z_ref + delta_z
        delta_z <- Complex.Zero
    // ...
```

Same source. Different targets. Both stable. The compiler chooses.

### Heterogeneous Distribution

For targets with multiple compute units (CPU + GPU, or multiple GPU types), the compiler can distribute computation to match numerical requirements:

```fsharp
// High-precision reference computation on CPU
let referenceOrbit =
    cpu {
        stable {
            // Double-Double or quad precision
            // Computed once, used by all GPU threads
        }
    }

// Parallel perturbation tracking on GPU
let pixels =
    gpu {
        parallel_for pixel in image do
            stable {
                let delta_c = pixel.coord - reference.coord
                // IEEE-754 perturbation, millions of threads
            }
    }
```

The Mandelbrot Metal pattern, high-precision reference on CPU, low-precision perturbation on GPU, becomes expressible directly in the language. The compiler understands the numerical contract and can verify that the distribution preserves stability guarantees.

## Verification: Proving Numerical Properties

### The F* Connection

Fidelity integrates with F*, the proof-oriented programming language, to verify numerical properties at compile time. This isn't just type checking; it's proving that the chosen numerical strategy meets specified accuracy bounds.

Consider a stability specification:

```fsharp
[<F* Requires("steps > 0 && steps < 10000000")>]
[<F* Ensures("abs(result - true_integral(f, a, b)) < error_bound(steps)")>]
let integrate f a b steps =
    stable {
        // ...
    }
```

The F* verifier can prove:

1. The posit/quire implementation meets the error bound
2. The perturbation transformation preserves the bound (possibly with a different constant)
3. The two implementations are equivalent within the specified tolerance

This proof flows through the compilation pipeline. When the compiler selects a numerical strategy, it's not guessing, it's choosing from proven-equivalent alternatives.

### SMT-Based Bound Checking

For simpler cases, Z3 (the SMT solver) can verify numerical bounds without full F* proofs:

```fsharp
[<Z3 Assert("no_overflow(accumulation)")>]
[<Z3 Assert("final_error < 1e-10")>]
let sumLargeArray (arr: float[]) =
    stable {
        let mutable sum = 0.0
        for x in arr do
            let! term = accumulate x
            sum <- sum + term
        return sum
    }
```

The compiler generates verification conditions that Z3 checks. If the bounds can't be proven, compilation fails with a clear error, not a runtime surprise.

## Architectural Integration: The Hypergraph View

### From Control Flow to Data Flow

Firefly's Program Semantic Graph (PSG) represents computation as a hypergraph where edges can connect multiple nodes. This structure naturally captures data flow, enabling analysis and transformation that control-flow representations obscure.

Numerical stability analysis benefits directly. Consider:

```fsharp
let result =
    values
    |> Array.map (fun x -> x * x)
    |> Array.sum
```

In a control-flow view, this is a sequence of operations. In the data-flow hypergraph, it's a reduction pattern: many inputs flowing to a single output through a commutative, associative operation.

The hypergraph representation lets the compiler:

1. Recognize the accumulation pattern
2. Determine that order of summation doesn't affect the mathematical result
3. Choose an order that minimizes numerical error (e.g., summing smallest to largest)
4. Or replace the entire pattern with quire-based exact accumulation

This isn't a special case; it's a consequence of having the right representation. The hypergraph exposes structure that imperative code hides.

### Hardware-Aware Lowering

Alex, the Firefly backend, uses the hypergraph structure to make hardware-aware decisions. For numerical operations, this means:

1. **Pattern recognition**: Identify accumulation, reduction, and iteration patterns
2. **Capability matching**: Determine what stability mechanisms the target supports
3. **Strategy selection**: Choose posit, perturbation, compensated summation, or hybrid approaches
4. **MLIR generation**: Lower to appropriate dialect operations

For a target like NextSilicon with native posit support:

```mlir
// Posit accumulation via quire
%q = posit.quire.init : !posit.quire<32, 2, 1024>
scf.for %i = %c0 to %n step %c1 {
    %val = memref.load %arr[%i] : memref<?x!posit.posit<32, 2>>
    posit.quire.add %q, %val : !posit.quire<32, 2, 1024>, !posit.posit<32, 2>
}
%result = posit.quire.finalize %q : !posit.quire<32, 2, 1024> -> !posit.posit<32, 2>
```

For an IEEE-754 target with perturbation transformation:

```mlir
// Perturbation-based accumulation
%ref = memref.load %reference[%c0] : memref<?xf64>
%delta = arith.constant 0.0 : f64
scf.for %i = %c0 to %n step %c1 iter_args(%d = %delta, %r = %ref) -> (f64, f64) {
    %val = memref.load %arr[%i] : memref<?xf64>
    %delta_val = arith.subf %val, %r : f64
    %new_delta = arith.addf %d, %delta_val : f64
    // Consolidation check
    %mag = math.absf %new_delta : f64
    %threshold = arith.constant 1.0e10 : f64
    %needs_consolidate = arith.cmpf "ogt", %mag, %threshold : f64
    %consolidated_r, %consolidated_d = scf.if %needs_consolidate -> (f64, f64) {
        %new_r = arith.addf %r, %new_delta : f64
        scf.yield %new_r, %cst_zero : f64, f64
    } else {
        scf.yield %r, %new_delta : f64, f64
    }
    scf.yield %consolidated_d, %consolidated_r : f64, f64
}
%result = arith.addf %ref_final, %delta_final : f64
```

Same source semantics. Radically different machine code. Both provably correct.

## The Bigger Picture: Why This Matters

### Beyond "Add More Bits"

The industry's default response to numerical problems is brute force: more precision, more memory, more compute. This works until it doesn't, and when it doesn't, debugging is nightmarish. Was the algorithm wrong? The implementation? The accumulation of rounding errors over a million iterations?

Firefly offers a different path. Numerical stability becomes a property the compiler understands and guarantees. When you write `stable { }`, you're not hoping for stability, you're specifying it. The compiler either proves it can deliver or tells you why it can't.

This shifts numerical correctness from runtime mystery to compile-time contract.

### Heterogeneous Hardware Utilization

Modern systems are heterogeneous. A workstation might have:

- CPU cores with AVX-512 (IEEE-754, 32/64-bit)
- Integrated GPU (IEEE-754, optimized for 32-bit)
- Discrete GPU (IEEE-754, tensor cores for reduced precision)
- Eventually, accelerators with native posit support

Today's software stacks treat this heterogeneity as a deployment problem: manually partition work, manually manage precision, hope it all fits together.

Firefly treats it as an optimization opportunity. The compiler understands numerical requirements and hardware capabilities. It can automatically:

- Place high-precision reference computations on CPU
- Distribute perturbation tracking across GPU threads
- Route posit operations to accelerators that support them
- Fall back gracefully when preferred hardware isn't available

The developer specifies intent. The compiler handles placement.

### A Competitive Differentiator

No other technology stack contemplates this integration:

- **Python/NumPy**: IEEE-754 only, no stability analysis, no proof integration
- **Julia**: Flexible representations but no compiler-driven stability strategy
- **Rust**: Strong types but numerical representation is developer's problem
- **C++/CUDA**: Maximum control, zero compiler assistance

Fidelity offers something new: mathematical intent as source code, with compiler-managed numerical strategy selection, proven equivalence across representations, and automatic distribution across heterogeneous hardware.

For domains where numerical stability matters, scientific computing, financial modeling, machine learning, physical simulation, this is transformative.

## Conclusion

IEEE-754 floating-point arithmetic has served computing well for four decades. But its limitations are increasingly apparent as computations grow more ambitious. The conventional response, add more bits, delays problems without solving them.

Fidelity takes a different approach. By treating numerical stability as a compiler concern, Firefly can:

1. Offer developers clean abstractions (`stable { }`, `accumulate`) that express mathematical intent
2. Provide library authors concrete mechanisms (posit witnesses, perturbation trackers) that implement stability strategies
3. Select strategies based on target capabilities, generating native posit operations or IEEE-754 with perturbation transformation
4. Prove that selected strategies meet specified accuracy bounds
5. Distribute computation across heterogeneous hardware to match numerical requirements

This isn't incremental improvement. It's a fundamental rethinking of how numerical software is written, compiled, and executed.

The gap between mathematical intent and machine execution becomes a managed space where the compiler makes provable decisions. Developers stop fighting representations and start expressing mathematics. The representation serves the computation, not the other way around.

This is what it means to extend computation: not more bits, but better alignment between what we mean and how machines execute it.
