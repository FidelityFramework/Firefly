# MLIR Numerical Stability Patterns

## Introduction

This document is a companion to `Fsnative_Extending_Computation.md`, which describes the high-level vision for treating numerical stability as a compiler concern. This document focuses on the concrete implementation: how Firefly leverages existing MLIR infrastructure to generate stability-preserving code across heterogeneous targets.

The key insight is that MLIR already provides the primitives we need—and emerging hardware-specific dialects like AMD's MLIR-AIE extend these capabilities dramatically. Standard dialects offer fused multiply-add, constrained floating-point operations, and type flexibility. AMD's AIE dialects add hardware accumulators, explicit precision transitions, and spatial compute arrays with coherent memory. What we build on top is **pattern libraries**—code generation strategies that emit these primitives in configurations that preserve numerical stability while exploiting heterogeneous hardware.

## Target Hardware: AMD Strix Halo as Reference Platform

The AMD Ryzen AI Max (Strix Halo) processor exemplifies the heterogeneous future we're designing for. It integrates:

- **Zen 5 CPU cores**: 16 cores / 32 threads with AVX-512 for IEEE-754 computation
- **RDNA 3.5 GPU**: 40 Compute Units for parallel floating-point workloads
- **XDNA 2 NPU**: 50 TOPS via AI Engine tiles with hardware accumulators

Most significantly, Strix Halo provides a [unified coherent memory architecture](https://www.tomshardware.com/pc-components/cpus/amds-beastly-strix-halo-ryzen-ai-max-debuts-with-radical-new-memory-tech-to-feed-rdna-3-5-graphics-and-zen-5-cpu-cores) with up to 128GB of LPDDR5X shared across all compute units. The GPU can read from the entire memory pool while the CPU maintains cache coherence. This eliminates the copy overhead that typically dominates heterogeneous computation.

For numerical stability, this architecture enables a powerful pattern: **high-precision reference computation on CPU, massively parallel perturbation tracking on GPU/NPU, with zero-copy data sharing**. The Mandelbrot Metal pattern we discussed in `Fsnative_Extending_Computation.md` maps directly to this hardware.

## References and Resources

This document draws on several key sources:

- [MLIR-AIE GitHub Repository](https://github.com/Xilinx/mlir-aie) - AMD's open-source MLIR toolchain for AI Engine devices
- [IRON API and MLIR-AIE Documentation](https://xilinx.github.io/mlir-aie/) - Official dialect documentation
- [AIEVec Dialect Reference](https://xilinx.github.io/mlir-aie/AIEVecDialect.html) - Vector operations and accumulator semantics
- [Leveraging MLIR to Design for AI Engines (FCCM 2023)](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/leveraging-mlir-to-design-for-aie-fccm-2023.pdf) - AMD's architectural paper
- [Triton-CPU](https://github.com/triton-lang/triton) - Reference for LLVM lowering patterns

## MLIR Primitives for Numerical Stability

### Fused Multiply-Add: The Foundation

The `llvm.intr.fma` intrinsic computes `a * b + c` with a single rounding at the end, not two separate roundings. This seemingly minor distinction is the foundation of error-free arithmetic.

In standard IEEE-754:
```
product = a * b      // rounding happens here
result = product + c // rounding happens again here
```

With FMA:
```
result = fma(a, b, c) // single rounding at the end
```

The difference is the intermediate product `a * b` is computed exactly (to infinite precision internally), then added to `c`, and only the final result is rounded. This enables error-free transforms—techniques that capture the rounding error that IEEE-754 normally discards.

### Error-Free Transforms

Using FMA, we can decompose an operation into its rounded result plus an exact error term:

**Error-Free Multiplication (TwoProd)**

Given `a * b`, we want both the rounded product and the exact error:

```mlir
// Returns (product, error) where product + error = a * b exactly
func.func @two_prod(%a: f64, %b: f64) -> (f64, f64) {
    // Rounded product
    %p = arith.mulf %a, %b : f64

    // Exact error via FMA: err = fma(a, b, -p) = a*b - p exactly
    %neg_p = arith.negf %p : f64
    %err = llvm.intr.fma(%a, %b, %neg_p) : (f64, f64, f64) -> f64

    return %p, %err : f64, f64
}
```

The magic: `fma(a, b, -p)` computes `a * b - p` with `a * b` evaluated exactly. Since `p` is the rounded version of `a * b`, the difference `a * b - p` is precisely the rounding error—and FMA gives it to us exactly.

**Error-Free Addition (TwoSum)**

For addition, we use the Knuth two-sum algorithm:

```mlir
// Returns (sum, error) where sum + error = a + b exactly
func.func @two_sum(%a: f64, %b: f64) -> (f64, f64) {
    %s = arith.addf %a, %b : f64

    // Recover the larger operand
    %a_prime = arith.subf %s, %b : f64

    // Recover the smaller operand
    %b_prime = arith.subf %s, %a_prime : f64

    // Compute individual errors
    %delta_a = arith.subf %a, %a_prime : f64
    %delta_b = arith.subf %b, %b_prime : f64

    // Total error
    %err = arith.addf %delta_a, %delta_b : f64

    return %s, %err : f64, f64
}
```

This works because floating-point addition loses information about the smaller operand. By carefully extracting what was lost, we recover the exact error.

### Constrained Floating-Point Operations

The `llvm.intr.experimental.constrained.*` family provides explicit control over rounding modes and exception behavior:

```mlir
// Truncate with explicit rounding mode
%result = llvm.intr.experimental.constrained.fptrunc %val
    tonearest ignore : f64 to f32
```

The `tonearest` specifies round-to-nearest-ties-to-even (the IEEE default). The `ignore` specifies exception handling. For perturbation-style computation, explicit rounding control prevents optimizer transformations that could violate our stability assumptions.

### Floating-Point Classification

The `llvm.intr.is.fpclass` intrinsic detects special values:

```mlir
// Check if value is a normal number (not denormal, infinity, or NaN)
%is_normal = llvm.intr.is.fpclass %val, 264 : f64 -> i1
// 264 = normal positive (256) + normal negative (8)

// Check for infinity
%is_inf = llvm.intr.is.fpclass %val, 516 : f64 -> i1
// 516 = positive infinity (512) + negative infinity (4)
```

This is essential for consolidation checks in perturbation tracking. When a perturbation term grows too large or becomes denormal, we need to detect and handle it.

## AMD MLIR-AIE: Hardware Accumulators and Spatial Compute

The [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) provides dialects specifically designed for AMD's AI Engine architecture, found in Ryzen AI processors and Versal devices. For numerical stability, the most significant features are **hardware accumulators** and **explicit precision management**.

### The AIE Dialect Family

AMD provides several dialects at different abstraction levels:

| Dialect | Purpose |
|---------|---------|
| `aie` | Device configuration, tile connectivity, DMA programming |
| `aiex` | Experimental extensions for advanced features |
| `aievec` | Vector operations with accumulator semantics |
| `adf` | Adaptive Data Flow graph representation |

For numerical stability patterns, **aievec** is the critical dialect. It exposes hardware accumulators that provide exact intermediate results—similar in spirit to the posit quire, but implemented in silicon.

### AIEVec Accumulator Operations

The AI Engine tiles contain wide accumulator registers (256-bit, 512-bit, or 1024-bit) that hold intermediate results at higher precision than the input operands. The dialect provides explicit operations for moving between vector and accumulator domains:

**`aievec.ups` (Upshift)**: Converts from vector type to accumulator type with precision adjustment:

```mlir
// Upshift vector to accumulator with 8-bit shift
%acc = aievec.ups %vec {shift = 8 : i8} : vector<16xbf16> -> vector<16xf32>
```

**`aievec.srs` (Shift-Round-Saturate)**: Converts accumulator back to vector with controlled rounding:

```mlir
// Downshift accumulator to vector with rounding and saturation
%vec = aievec.srs %acc {shift = 8 : i8} : vector<16xf32> -> vector<16xbf16>
```

The shift parameter controls precision—larger shifts preserve more fractional bits during the transition. This explicit precision management is exactly what numerical stability requires.

### Supported Type Combinations

The aievec dialect supports specific type combinations that reflect the hardware capabilities:

| Input Type | Input Type | Accumulator Type |
|:----------:|:----------:|:----------------:|
| `vector<32xi8>` | `vector<32xi8>` | `vector<32xi32>` |
| `vector<32xi16>` | `vector<32xi16>` | `vector<32xi32>` |
| `vector<16xi32>` | `vector<16xi32>` | `vector<16xi64>` |
| `vector<16xbf16>` | `vector<16xbf16>` | `vector<16xf32>` |
| `vector<16xf32>` | `vector<16xf32>` | `vector<16xf32>` |

Note the **bfloat16 → float32 accumulator** path. This is significant: bfloat16 operations accumulate in full float32 precision, eliminating the precision loss that plagues naive bfloat16 computation. The hardware does what we'd otherwise implement in software with compensated arithmetic.

### Fused Multiply-Accumulate

The `aievec.mac_elem` operation performs element-wise multiply-accumulate with hardware accumulator semantics:

```mlir
// Multiply-accumulate: result = lhs * rhs + acc
%result = aievec.mac_elem %lhs, %rhs, %acc :
    vector<16xbf16>, vector<16xbf16>, vector<16xf32> -> vector<16xf32>
```

The operation also supports `fmsub` (fused multiply-subtract) via an attribute:

```mlir
// Multiply-subtract: result = acc - lhs * rhs
%result = aievec.mac_elem %lhs, %rhs, %acc {fmsub = true} :
    vector<16xbf16>, vector<16xbf16>, vector<16xf32> -> vector<16xf32>
```

### Matrix Operations with Accumulation

For matrix multiplication, `aievec.matmul` provides hardware-accelerated accumulation:

```mlir
// Matrix multiply-accumulate
%result = aievec.matmul %A, %B, %C :
    vector<4x8xi8>, vector<8x4xi8>, vector<4x4xi32> -> vector<4x4xi32>
```

The 8-bit inputs accumulate into 32-bit results—a 4× precision expansion that prevents overflow during large reductions. This is the same principle as the quire, implemented in spatial hardware.

### Precision Transitions as First-Class Operations

What makes aievec valuable for numerical stability is that precision transitions are **explicit, not implicit**. In standard MLIR, a multiplication of two f32 values produces an f32 result—the intermediate precision is hidden. In aievec, you explicitly manage when precision expands (ups), when accumulation happens (mac_elem, matmul), and when precision contracts (srs).

This explicitness maps directly to our stability-aware compilation model. The compiler can:

1. Recognize accumulation patterns in the PSG
2. Emit aievec operations that use hardware accumulators
3. Control precision transitions to maximize stability
4. Prove error bounds based on the explicit precision flow

### Spatial Distribution: Tiles and Data Movement

The AI Engine is a spatial array of tiles connected by stream switches. Each tile contains:

- **AIE Core**: VLIW vector processor with local memory
- **Memory Tile**: Shared memory with DMA engines
- **Interface Tile**: Connection to CPU/memory subsystem

The `aie` dialect configures this spatial structure:

```mlir
// Define a 2x2 tile array
aie.device(xcvc1902) {
    %tile00 = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)

    // Configure stream connections
    aie.flow(%tile00, DMA : 0, %tile01, DMA : 0)
    aie.flow(%tile01, DMA : 0, %tile11, DMA : 0)

    // Cores contain compute kernels
    %core00 = aie.core(%tile00) {
        // Kernel code here
        aie.end
    }
}
```

For numerical stability patterns, this spatial model enables:

- **Pipelining**: Reference orbit computation flows through tiles as perturbation tracking consumes it
- **Parallelism**: Multiple perturbation computations execute simultaneously on different tiles
- **Data locality**: Accumulators stay in tile-local memory, avoiding bandwidth limits

### Mapping Stability Patterns to AIE

The perturbation pattern maps naturally to the AIE spatial model:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   CPU (Zen 5)   │────▶│   Memory Tile   │────▶│   AIE Tile 0    │
│                 │     │                 │     │                 │
│ Compute high-   │     │ Reference orbit │     │ Perturbation    │
│ precision ref   │     │ buffer (shared) │     │ tracking (bf16  │
│ orbit (f64)     │     │                 │     │ with f32 acc)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌──────────────────────────┤
                              ▼                          ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │   AIE Tile 1    │     │   AIE Tile N    │
                    │                 │     │                 │
                    │ Perturbation    │     │ Perturbation    │
                    │ tracking        │ ... │ tracking        │
                    │ (parallel)      │     │ (parallel)      │
                    └─────────────────┘     └─────────────────┘
```

The CPU computes the reference orbit once at high precision (f64). The orbit streams through shared memory to the AIE array. Each AIE tile tracks perturbations for a subset of pixels using bfloat16 arithmetic with float32 accumulators. The coherent memory architecture means no explicit copies—the CPU writes, the AIE tiles read, the hardware maintains consistency.

### AIE-ML Precision Considerations

The AIE-ML (second generation AI Engine) has specific precision characteristics that affect stability strategies:

- **Native bfloat16**: Full hardware support with f32 accumulators
- **Native int8/int16**: With int32 accumulators for ML inference
- **Emulated int32/f32**: Not native, but can be composed from narrower operations

This means stability patterns should prefer bfloat16 on AIE-ML when possible—it has the best hardware support. The f32 accumulator provides the precision headroom; the bfloat16 operands provide the throughput.

For applications requiring f32 precision throughout, the GPU (RDNA 3.5) is the better target. The compiler's job is to map computation to the appropriate unit based on precision requirements.

## Pattern Libraries in Alex

Rather than defining new MLIR dialects, we implement numerical stability through **pattern libraries**—modules that emit standard MLIR operations in stability-preserving configurations.

### Compensated Summation (Kahan)

The Kahan summation algorithm maintains a running compensation term that captures accumulated rounding errors:

```mlir
// Kahan compensated sum
// Achieves O(ε) error instead of O(ε√n) for naive summation
func.func @kahan_sum(%arr: memref<?xf64>, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64

    // Iterate with sum and compensation term
    %result:2 = scf.for %i = %c0 to %n step %c1
        iter_args(%sum = %zero, %comp = %zero) -> (f64, f64) {

        %val = memref.load %arr[%i] : memref<?xf64>

        // Compensated value: what we meant to add last time
        %y = arith.subf %val, %comp : f64

        // New sum (rounded)
        %t = arith.addf %sum, %y : f64

        // New compensation: what we lost in rounding
        // comp = (t - sum) - y = -(y - (t - sum))
        %t_minus_sum = arith.subf %t, %sum : f64
        %new_comp = arith.subf %t_minus_sum, %y : f64

        scf.yield %t, %new_comp : f64, f64
    }

    return %result#0 : f64
}
```

The compensation term `comp` tracks the cumulative rounding error. Each iteration subtracts this error from the next value, effectively carrying forward the lost precision.

### Pairwise Summation

For large arrays, pairwise (cascade) summation offers better cache behavior and parallelism while improving accuracy:

```mlir
// Pairwise summation - O(ε log n) error bound
// Recursively sum pairs, reducing error accumulation
func.func @pairwise_sum(%arr: memref<?xf64>, %start: index, %len: index) -> f64 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %threshold = arith.constant 32 : index  // Base case size

    %is_base = arith.cmpi ult, %len, %threshold : index
    %result = scf.if %is_base -> f64 {
        // Base case: Kahan sum for small arrays
        %base_sum = func.call @kahan_sum_range(%arr, %start, %len) :
            (memref<?xf64>, index, index) -> f64
        scf.yield %base_sum : f64
    } else {
        // Recursive case: split and sum halves
        %half = arith.divui %len, %c2 : index
        %mid = arith.addi %start, %half : index
        %right_len = arith.subi %len, %half : index

        %left_sum = func.call @pairwise_sum(%arr, %start, %half) :
            (memref<?xf64>, index, index) -> f64
        %right_sum = func.call @pairwise_sum(%arr, %mid, %right_len) :
            (memref<?xf64>, index, index) -> f64

        // Combine with error-free addition
        %total, %err = func.call @two_sum(%left_sum, %right_sum) :
            (f64, f64) -> (f64, f64)

        scf.yield %total : f64
    }

    return %result : f64
}
```

### Perturbation-Based Iteration

For iterative algorithms like Mandelbrot or physical simulation, perturbation tracking maintains stability by working with small deviations from a reference trajectory:

```mlir
// Perturbation-based complex iteration
// z(n+1) = z(n)² + c becomes δz(n+1) = 2·z₀(n)·δz(n) + δc
func.func @iterate_perturbed(
    %delta_c_re: f64,                    // Real part of pixel offset
    %delta_c_im: f64,                    // Imaginary part of pixel offset
    %ref_orbit_re: memref<?xf64>,        // Reference orbit real parts
    %ref_orbit_im: memref<?xf64>,        // Reference orbit imaginary parts
    %max_iter: index
) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    %two = arith.constant 2.0 : f64
    %escape_sq = arith.constant 4.0 : f64
    %consolidation_threshold = arith.constant 1.0e10 : f64

    // Initial perturbation is zero (we start at reference point)
    %result:3 = scf.for %i = %c0 to %max_iter step %c1
        iter_args(%d_re = %zero, %d_im = %zero, %iter = %c0)
        -> (f64, f64, index) {

        // Load reference point for this iteration
        %z0_re = memref.load %ref_orbit_re[%i] : memref<?xf64>
        %z0_im = memref.load %ref_orbit_im[%i] : memref<?xf64>

        // Perturbation update: δz' = 2·z₀·δz + δc
        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // So 2·z₀·δz = 2·((z0_re·d_re - z0_im·d_im) + (z0_re·d_im + z0_im·d_re)i)

        // Real part of 2·z₀·δz
        %prod_rr = arith.mulf %z0_re, %d_re : f64
        %prod_ii = arith.mulf %z0_im, %d_im : f64
        %real_prod = arith.subf %prod_rr, %prod_ii : f64
        %real_scaled = arith.mulf %two, %real_prod : f64
        %new_d_re = arith.addf %real_scaled, %delta_c_re : f64

        // Imaginary part of 2·z₀·δz
        %prod_ri = arith.mulf %z0_re, %d_im : f64
        %prod_ir = arith.mulf %z0_im, %d_re : f64
        %imag_prod = arith.addf %prod_ri, %prod_ir : f64
        %imag_scaled = arith.mulf %two, %imag_prod : f64
        %new_d_im = arith.addf %imag_scaled, %delta_c_im : f64

        // Check perturbation magnitude for consolidation
        %d_re_sq = arith.mulf %new_d_re, %new_d_re : f64
        %d_im_sq = arith.mulf %new_d_im, %new_d_im : f64
        %d_mag_sq = arith.addf %d_re_sq, %d_im_sq : f64
        %needs_consolidation = arith.cmpf ogt, %d_mag_sq, %consolidation_threshold : f64

        // Check escape: |z₀ + δz|² > 4
        %z_re = arith.addf %z0_re, %new_d_re : f64
        %z_im = arith.addf %z0_im, %new_d_im : f64
        %z_re_sq = arith.mulf %z_re, %z_re : f64
        %z_im_sq = arith.mulf %z_im, %z_im : f64
        %z_mag_sq = arith.addf %z_re_sq, %z_im_sq : f64
        %escaped = arith.cmpf ogt, %z_mag_sq, %escape_sq : f64

        // Update iteration count
        %next_iter = arith.addi %iter, %c1 : index

        // Early exit on escape
        %continue = arith.xori %escaped, %true : i1
        scf.condition(%continue) %new_d_re, %new_d_im, %next_iter : f64, f64, index
    }

    return %result#2 : index
}
```

### Double-Double Arithmetic

For cases requiring extended precision without hardware support, double-double arithmetic represents a number as the unevaluated sum of two doubles:

```mlir
// Double-double addition
// Adds (a_hi, a_lo) + (b_hi, b_lo) -> (s_hi, s_lo)
// where a = a_hi + a_lo and b = b_hi + b_lo exactly
func.func @dd_add(
    %a_hi: f64, %a_lo: f64,
    %b_hi: f64, %b_lo: f64
) -> (f64, f64) {
    // Add high parts with error recovery
    %s_hi, %e1 = func.call @two_sum(%a_hi, %b_hi) : (f64, f64) -> (f64, f64)

    // Add low parts
    %t = arith.addf %a_lo, %b_lo : f64

    // Combine errors
    %e = arith.addf %e1, %t : f64

    // Final normalization
    %r_hi, %r_lo = func.call @two_sum(%s_hi, %e) : (f64, f64) -> (f64, f64)

    return %r_hi, %r_lo : f64, f64
}

// Double-double multiplication
// Multiplies (a_hi, a_lo) * (b_hi, b_lo) -> (p_hi, p_lo)
func.func @dd_mul(
    %a_hi: f64, %a_lo: f64,
    %b_hi: f64, %b_lo: f64
) -> (f64, f64) {
    // Main product with error
    %p_hi, %e1 = func.call @two_prod(%a_hi, %b_hi) : (f64, f64) -> (f64, f64)

    // Cross terms (approximate, these are small)
    %cross1 = arith.mulf %a_hi, %b_lo : f64
    %cross2 = arith.mulf %a_lo, %b_hi : f64

    // Sum corrections
    %e2 = arith.addf %e1, %cross1 : f64
    %e3 = arith.addf %e2, %cross2 : f64

    // Normalize
    %r_hi, %r_lo = func.call @two_sum(%p_hi, %e3) : (f64, f64) -> (f64, f64)

    return %r_hi, %r_lo : f64, f64
}
```

## Integration with Alex Code Generation

### Pattern Selection

Alex selects numerical patterns based on target capabilities and code analysis:

```fsharp
module Alex.Bindings.NumericalStrategy

type NumericalCapability =
    | PositNative of bits: int * es: int
    | IEEE754 of hasF16: bool * hasBF16: bool * hasFMA: bool
    | AIEngine of generation: AIEGeneration * hasBF16Acc: bool
    | DoubleDouble

and AIEGeneration = AIE1 | AIE_ML  // AIE-ML is Ryzen AI / XDNA 2

type ComputeUnit =
    | CPU
    | GPU
    | NPU
    | Accelerator of name: string

type HeterogeneousTarget = {
    Units: Map<ComputeUnit, NumericalCapability>
    CoherentMemory: bool
    SharedMemoryGB: int
}

type AccumulationPattern =
    | SimpleSum of count: int
    | IterativeUpdate of maxIterations: int option
    | DotProduct of vectorLength: int
    | Reduction of op: BinaryOp
    | MatrixMultiply of m: int * n: int * k: int

type StabilityStrategy =
    | NativePositQuire
    | KahanSummation
    | PairwiseSummation
    | CompensatedDotProduct
    | PerturbationTransform of consolidationThreshold: float
    | DoubleDoubleEmulation
    | AIEAccumulator of inputType: string * accType: string
    | HeterogeneousDistribution of reference: ComputeUnit * parallel: ComputeUnit
    | StandardIEEE754  // No stability enhancement

let selectStrategy
    (target: HeterogeneousTarget)
    (pattern: AccumulationPattern)
    (errorBound: float option)
    : StabilityStrategy =

    // Check for heterogeneous patterns first
    match pattern with
    | IterativeUpdate (Some n) when n > 1000 && target.CoherentMemory ->
        // Perturbation with CPU reference, NPU/GPU parallel tracking
        let refUnit = CPU  // High precision reference
        let parUnit =
            if target.Units.ContainsKey(NPU) then NPU
            elif target.Units.ContainsKey(GPU) then GPU
            else CPU
        if refUnit <> parUnit then
            HeterogeneousDistribution(reference = refUnit, parallel = parUnit)
        else
            PerturbationTransform(consolidationThreshold = 1e10)

    | MatrixMultiply _ when target.Units.ContainsKey(NPU) ->
        // Use AIE hardware accumulators for matrix ops
        match target.Units.[NPU] with
        | AIEngine(AIE_ML, true) ->
            AIEAccumulator(inputType = "bf16", accType = "f32")
        | AIEngine(_, _) ->
            AIEAccumulator(inputType = "i16", accType = "i32")
        | _ ->
            CompensatedDotProduct

    | _ ->
        // Fall back to single-unit strategies
        let primaryCap =
            target.Units
            |> Map.tryFind CPU
            |> Option.defaultValue (IEEE754(false, false, true))

        match primaryCap, pattern, errorBound with
        // Native posit: always use quire
        | PositNative _, _, _ ->
            NativePositQuire

        // AIE with bfloat16 accumulators
        | AIEngine(AIE_ML, true), SimpleSum _, _ ->
            AIEAccumulator(inputType = "bf16", accType = "f32")

        // Large iterative patterns: perturbation
        | IEEE754 _, IterativeUpdate (Some n), _ when n > 1000 ->
            PerturbationTransform(consolidationThreshold = 1e10)

        // Long sums with tight error bounds: double-double
        | IEEE754 _, SimpleSum n, Some bound when n > 10000 && bound < 1e-14 ->
            DoubleDoubleEmulation

        // Long sums: pairwise + Kahan hybrid
        | IEEE754 _, SimpleSum n, _ when n > 1000 ->
            PairwiseSummation

        // Medium sums: Kahan
        | IEEE754 _, SimpleSum n, _ when n > 100 ->
            KahanSummation

        // Dot products with FMA: compensated
        | IEEE754(hasFMA = true), DotProduct _, _ ->
            CompensatedDotProduct

        // Small patterns: not worth overhead
        | IEEE754 _, _, _ ->
            StandardIEEE754

        | _ ->
            StandardIEEE754
```

### MLIR Emission

Each strategy has a corresponding emission function:

```fsharp
module Alex.Bindings.NumericalEmitters

open Alex.MLIR.Builder

let emitKahanSum (builder: MLIRBuilder) (array: Value) (length: Value) : Value =
    // Emit the Kahan summation loop
    let zero = builder.Constant(0.0)
    let one = builder.IndexConstant(1)

    let loopResult = builder.For(
        start = builder.IndexConstant(0),
        stop = length,
        step = one,
        iterArgs = [| ("sum", zero); ("comp", zero) |],
        body = fun i [| sum; comp |] ->
            let value = builder.Load(array, i)
            let y = builder.SubF(value, comp)
            let t = builder.AddF(sum, y)
            let newComp = builder.SubF(builder.SubF(t, sum), y)
            [| t; newComp |]
    )

    loopResult.[0]  // Return final sum

let emitTwoSum (builder: MLIRBuilder) (a: Value) (b: Value) : Value * Value =
    let s = builder.AddF(a, b)
    let aPrime = builder.SubF(s, b)
    let bPrime = builder.SubF(s, aPrime)
    let deltaA = builder.SubF(a, aPrime)
    let deltaB = builder.SubF(b, bPrime)
    let err = builder.AddF(deltaA, deltaB)
    (s, err)

let emitTwoProd (builder: MLIRBuilder) (a: Value) (b: Value) : Value * Value =
    let p = builder.MulF(a, b)
    let negP = builder.NegF(p)
    let err = builder.FMA(a, b, negP)
    (p, err)

let emitPerturbedIteration
    (builder: MLIRBuilder)
    (refOrbitRe: Value) (refOrbitIm: Value)
    (deltaCRe: Value) (deltaCIm: Value)
    (maxIter: Value)
    : Value =
    // Emit the full perturbation-tracking loop
    // ... (as shown in MLIR above)
```

## Verification with F*

The numerical patterns are verified once, not per-use. F* proofs establish error bounds that the compiler can rely on:

### Kahan Summation Error Bound

```fstar
module Kahan.Proof

open FStar.Real
open FStar.Seq

// Machine epsilon for IEEE-754 double
let eps : real = 0x1.0p-53

// Kahan summation error bound: O(nε) instead of O(n²ε) for naive
val kahan_error_bound:
    arr: seq real{length arr > 0} ->
    Lemma (ensures (
        let n = length arr in
        let kahan_result = kahan_sum arr in
        let exact_result = sum arr in
        let max_val = max_abs arr in
        abs(kahan_result - exact_result) <= 2.0 * (real_of_int n) * eps * max_val
    ))
```

### Perturbation Equivalence

```fstar
module Perturbation.Proof

open FStar.Real
open Complex

// Perturbation transformation preserves iteration semantics
val perturbation_equivalence:
    z0: complex ->           // Initial point
    c: complex ->            // Parameter
    delta_c: complex ->      // Perturbation in c
    n: nat{n < max_iter} ->  // Iteration count
    Lemma (requires magnitude delta_c < perturbation_bound)
          (ensures (
              let direct = iterate z0 (c +. delta_c) n in
              let ref_orbit = iterate z0 c n in
              let perturbed = iterate_perturbed ref_orbit delta_c n in
              magnitude(direct -. perturbed) <= error_bound n delta_c
          ))
```

### Error-Free Transform Correctness

```fstar
module ErrorFree.Proof

open FStar.Real
open IEEE754

// TwoProd is exact: product + error = a * b mathematically
val two_prod_exact:
    a: double ->
    b: double ->
    Lemma (ensures (
        let (p, e) = two_prod a b in
        to_real p +. to_real e = to_real a *. to_real b
    ))

// TwoSum is exact: sum + error = a + b mathematically
val two_sum_exact:
    a: double ->
    b: double ->
    Lemma (ensures (
        let (s, e) = two_sum a b in
        to_real s +. to_real e = to_real a +. to_real b
    ))
```

## The Complete Pipeline

Bringing it all together, here's how a stability-sensitive computation flows through Firefly:

### 1. Source Code

```fsharp
let computeSum (values: float array) =
    stable {
        let mutable total = 0.0
        for v in values do
            let! accumulated = accumulate v
            total <- total + accumulated
        return total
    }
```

### 2. PSG Analysis

The nanopass pipeline recognizes the accumulation pattern:

```
Node: ForLoop
  Pattern: SimpleSum
  Count: values.Length (dynamic)
  Body: Addition accumulation
  Annotation: StabilityRequired
```

### 3. Strategy Selection

Based on target capabilities (from fidproj):

```toml
[target.x86_64]
numerical_capability = "ieee754"
has_fma = true
```

Alex selects: `KahanSummation` (or `PairwiseSummation` if array is large)

### 4. MLIR Generation

Alex emits the Kahan pattern:

```mlir
func.func @computeSum(%values: memref<?xf64>) -> f64 {
    %n = memref.dim %values, %c0 : memref<?xf64>
    %result = call @kahan_sum(%values, %n) : (memref<?xf64>, index) -> f64
    return %result : f64
}
```

### 5. LLVM Lowering

Standard MLIR-to-LLVM lowering produces efficient machine code using available vector instructions.

### 6. Verification

The pattern selection inherits the F* proof for Kahan summation. The compiler knows the result satisfies `|error| ≤ 2nε·max|values|`.

## Summary

MLIR provides the primitives: FMA, constrained operations, type flexibility. Hardware-specific dialects like AMD's MLIR-AIE extend these with silicon-level accumulator support. Firefly builds pattern libraries that emit these primitives in stability-preserving configurations across heterogeneous targets:

### Software Stability Patterns (Standard MLIR)

| Pattern | When Used | Error Bound | MLIR Operations |
|---------|-----------|-------------|-----------------|
| Kahan Summation | Medium sums (100-1000 elements) | O(nε) | `arith.addf`, `arith.subf` |
| Pairwise Summation | Large sums (>1000 elements) | O(ε log n) | Recursive `scf.if` + TwoSum |
| Compensated Dot Product | Vector products with FMA | O(nε) | `llvm.intr.fma` |
| Perturbation Transform | Iterative algorithms | Bounded by δ | Full loop with tracking |
| Double-Double | Extreme precision needs | ~31 digits | TwoSum + TwoProd compositions |

### Hardware Accumulator Patterns (MLIR-AIE)

| Pattern | When Used | Precision | MLIR Operations |
|---------|-----------|-----------|-----------------|
| BF16→F32 Accumulator | ML inference, reductions | 8-bit mantissa → 24-bit | `aievec.ups`, `aievec.mac_elem`, `aievec.srs` |
| INT8→INT32 Accumulator | Quantized inference | 8-bit → 32-bit exact | `aievec.matmul`, `aievec.mac_elem` |
| INT16→INT32 Accumulator | Audio, signal processing | 16-bit → 32-bit exact | `aievec.ups`, `aievec.mac_elem` |
| Explicit Precision Control | Any accumulation | Configurable via shift | `aievec.ups {shift=N}`, `aievec.srs {shift=N}` |

### Heterogeneous Distribution Patterns (Strix Halo, HSA)

| Pattern | When Used | Architecture | Strategy |
|---------|-----------|--------------|----------|
| CPU Reference + NPU Parallel | Deep iterative (Mandelbrot) | Coherent memory | f64 reference on CPU, bf16+f32 acc on NPU |
| CPU Reference + GPU Parallel | Graphics, simulation | Coherent memory | f64 reference on CPU, f32 parallel on GPU |
| Spatial Pipelining | Streaming computation | AIE tile array | Orbit flows through tiles, perturbations track in parallel |

### Pattern Selection Hierarchy

1. **Posit Hardware Available?** → Use native quire accumulation
2. **AIE Hardware Available?** → Use hardware accumulators (bf16→f32 or int→int32)
3. **Heterogeneous with Coherent Memory?** → Distribute reference/parallel across units
4. **FMA Available?** → Use compensated arithmetic (TwoSum, TwoProd)
5. **Standard IEEE-754** → Use software patterns (Kahan, pairwise, double-double)

The patterns are verified once with F*. When Alex selects a pattern, the proof travels with it. The result: stability guarantees without per-use verification overhead.

For targets with native posit support, the same source code generates quire-based accumulation instead—same semantics, different mechanism, both proven correct.

For heterogeneous targets like Strix Halo, the compiler distributes computation across units based on precision requirements and throughput characteristics, exploiting coherent memory to avoid copy overhead.
