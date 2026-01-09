# Case Study: Variational Quantum Eigensolver on Heterogeneous Hardware

## Introduction

This document presents a case study in compiling quantum simulation workloads for heterogeneous processor architectures. The specific target is the AMD Ryzen AI Max (Strix Halo) processor, which integrates a conventional CPU, a graphics processor, and a neural processing unit into a single chip with unified memory. The specific workload is the Variational Quantum Eigensolver, an algorithm that combines quantum state evolution with classical optimization. The broader purpose is to demonstrate how the Fidelity compiler framework partitions computation across fundamentally different processing units while preserving numerical guarantees.

The reader should come away understanding three things. First, quantum simulation is not a monolithic computation but rather a composition of distinct computational patterns, each with different precision requirements and parallelism characteristics. Second, modern heterogeneous processors offer specialized capabilities that match these patterns in non-obvious ways. Third, a compiler can analyze source code to automatically partition work across processing units, generating specialized code for each while proving that the combined result maintains required precision bounds.

This document assumes familiarity with IEEE-754 floating point arithmetic and basic PyTorch conventions for numerical computing. It does not assume prior knowledge of quantum computing, heterogeneous processor architectures, or compiler design. Where quantum mechanical concepts are necessary, they will be developed from first principles with emphasis on the computational rather than physical interpretation.

## The Algorithm at a Glance

Before diving into the details, it helps to see the complete picture. The Variational Quantum Eigensolver executes in iterations, where each iteration touches all three processing units. The coherent memory architecture means no data copying between steps; each unit operates directly on the shared state vector.

```
VQE Iteration Structure
═══════════════════════════════════════════════════════════════════════════════

Phase 1: State Preparation ─────────────────────────────▶  GPU (parallel gates)
         |0⟩⊗ⁿ → |+⟩⊗ⁿ                                     f32 sufficient
         Initialize superposition                          embarrassingly parallel

Phase 2: Parameterized Ansatz ──────────────────────────▶  CPU (f64) + GPU (f32)
         Ry(θ₁), Rz(θ₂), CNOT, ...                         rotations on CPU
         Apply quantum gates with                          entangling gates on GPU
         tunable rotation angles

Phase 3: Hamiltonian Expectation ───────────────────────▶  NPU (accumulator)
         ⟨ψ|H|ψ⟩ = Σᵢ cᵢ⟨ψ|Pᵢ|ψ⟩                            extended precision sum
         Compute energy as weighted                        billions of terms
         sum of Pauli expectations

Phase 4: Classical Optimization ────────────────────────▶  CPU
         θ' = θ − η∇E(θ)                                   gradient descent
         Update rotation angles                            BFGS, COBYLA, etc.
         to minimize energy

         ↺ Repeat until energy converges

═══════════════════════════════════════════════════════════════════════════════
```

The key insight is that different phases have fundamentally different computational characteristics. State preparation is embarrassingly parallel with low precision requirements. The parameterized ansatz mixes precision-sensitive rotations with parallel entangling operations. Measurement is a massive reduction that benefits from hardware accumulators. Classical optimization is inherently sequential. No single processing unit excels at all four patterns, but the heterogeneous processor handles each phase with the appropriate hardware.

### Memory Requirements and Practical Limits

The state vector for an n-qubit system contains 2ⁿ complex amplitudes. In single precision (f32), each complex number requires 8 bytes. In double precision (f64), each requires 16 bytes. The following table shows state vector sizes and their feasibility on a system with 64 GB of coherent memory:

| Qubits | Amplitudes | State Vector (f32) | State Vector (f64) | Feasibility |
|:------:|:----------:|:------------------:|:------------------:|:-----------:|
| 20 | 1 million | 8 MB | 16 MB | ✓ Comfortable |
| 25 | 33 million | 256 MB | 512 MB | ✓ Comfortable |
| 28 | 268 million | 2 GB | 4 GB | ✓ Good |
| 30 | 1 billion | 8 GB | 16 GB | ✓ Fine |
| 31 | 2 billion | 16 GB | 32 GB | ✓ Practical ceiling |
| 32 | 4 billion | 32 GB | 64 GB | ⚠ Tight, no headroom |
| 35 | 34 billion | 256 GB | 512 GB | ✗ Exceeds memory |

For practical work, **31 qubits represents the ceiling** on a 64 GB system. This leaves headroom for intermediate buffers, measurement results, and the classical optimizer's working memory. At 31 qubits, the simulation manages 2 billion complex amplitudes, which is sufficient to explore meaningful quantum algorithms and to stress-test the heterogeneous compilation strategy.

### The Heterogeneous Advantage

The following diagram illustrates how data flows between processing units through coherent memory. There are no explicit copy operations; each unit reads from and writes to the same physical addresses.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           64 GB Coherent Memory                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     State Vector (16 GB for 30 qubits)              │   │
│   │              2³⁰ complex amplitudes: α₀, α₁, α₂, ... α₂³⁰₋₁          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         ▲                           ▲                           ▲           │
│         │                           │                           │           │
│    read/write                  read/write                  read/write       │
│         │                           │                           │           │
│   ┌─────┴─────┐             ┌───────┴───────┐           ┌───────┴───────┐   │
│   │           │             │               │           │               │   │
│   │  Zen 5    │             │   RDNA 3.5    │           │   XDNA 2      │   │
│   │  CPU      │◀───────────▶│   GPU         │◀─────────▶│   NPU         │   │
│   │           │  barriers   │               │  barriers │               │   │
│   │  • f64    │             │  • f32        │           │  • bf16→f32   │   │
│   │  • AVX512 │             │  • 2560 cores │           │  • accumulator│   │
│   │  • control│             │  • parallel   │           │  • reduction  │   │
│   │           │             │               │           │               │   │
│   └───────────┘             └───────────────┘           └───────────────┘   │
│                                                                             │
│   Synchronization: memory barriers at phase boundaries ensure consistency   │
│   No copies: all units access the same physical memory addresses            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This architecture fundamentally changes the economics of heterogeneous computing. In traditional discrete GPU systems, the cost of copying a 16 GB state vector between CPU and GPU memory dominates execution time. On Strix Halo, phase transitions are nearly free; only lightweight barrier synchronization is required to ensure consistency.

With this overview in mind, the following sections develop each concept in detail: the computational structure of quantum simulation, the capabilities of each processing unit, the compiler's analysis and partitioning strategy, and a worked example tracing a complete VQE calculation.

## The Problem of Quantum Simulation

### What We Mean by Simulation

A quantum computer manipulates quantum bits, called qubits, which exist in superposition states rather than the definite zero-or-one states of classical bits. A system of n qubits can exist in a superposition of all 2ⁿ possible classical states simultaneously. The state of such a system is described by 2ⁿ complex numbers called amplitudes, one for each possible classical configuration.

When we simulate a quantum computer on classical hardware, we must explicitly represent and manipulate all 2ⁿ amplitudes. This exponential scaling is both the source of quantum computing's potential power and the fundamental limitation of classical simulation. A 30-qubit system requires tracking approximately one billion complex amplitudes. A 50-qubit system would require more amplitudes than atoms in the Earth.

The simulation proceeds by applying quantum gates, which are mathematical transformations of the amplitude vector. A single-qubit gate operating on qubit k transforms pairs of amplitudes according to a 2×2 complex matrix. A two-qubit gate transforms groups of four amplitudes according to a 4×4 matrix. The full computation is a sequence of such gates, followed by a measurement that collapses the superposition to a classical outcome with probability determined by the squared magnitudes of the amplitudes.

### The Precision Challenge

Quantum algorithms rely on interference, the phenomenon where amplitudes can add constructively or destructively depending on their phases. The phase of a complex amplitude is the angle it makes in the complex plane. When two paths through a computation produce amplitudes with matching phases, they reinforce each other. When the phases differ by 180 degrees, they cancel. Quantum algorithms are designed so that correct answers have reinforcing amplitudes while incorrect answers have canceling amplitudes.

This interference is exquisitely sensitive to phase errors. Each floating point operation introduces a small rounding error. In standard IEEE-754 double precision arithmetic, this error is on the order of 10⁻¹⁶ relative to the operand magnitudes. This sounds negligible until we consider that a quantum algorithm might apply thousands of gates, each introducing phase errors that accumulate. After one thousand gates, the accumulated phase error can reach 10⁻¹³ radians, which is still small. After one million gates, or in algorithms with amplifying feedback loops, the accumulated error can grow to the point where interference patterns are destroyed and the simulation produces meaningless results.

The traditional response is to use higher precision arithmetic. Double precision works for shallow circuits. Quadruple precision or beyond is needed for deep circuits. But higher precision arithmetic is slower and consumes more memory. Quadruple precision complex amplitudes for a 30-qubit system require 32 gigabytes of storage, compared to 16 gigabytes for double precision.

### The Parallelism Opportunity

While the precision challenge pushes toward more expensive arithmetic, the structure of quantum simulation offers enormous parallelism that can offset the cost. Consider a single-qubit gate applied to qubit k in an n-qubit system. This gate transforms each pair of amplitudes (α_j, α_{j+2^k}) according to the same 2×2 matrix. There are 2ⁿ⁻¹ such pairs, and they are completely independent of each other. We can transform all pairs simultaneously if we have enough parallel processing capacity.

This is embarrassingly parallel computation of the purest form. A modern GPU with thousands of processing elements can transform all amplitude pairs for a single-qubit gate in a single parallel sweep. Two-qubit gates are similar, with 2ⁿ⁻² independent 4×4 transformations.

The challenge is that not all parts of the simulation benefit equally from parallelism, and different parts have different precision requirements. State preparation might be simple and imprecision-tolerant. Deep rotation gates might require high precision. Measurement involves a massive reduction operation that accumulates billions of squared magnitudes into a single probability value. Each of these patterns maps best to different hardware capabilities.

## The Target Hardware: AMD Strix Halo

### A New Class of Processor

The AMD Ryzen AI Max processor, codenamed Strix Halo, represents a new approach to processor design. Rather than focusing on a single type of computation, it integrates three fundamentally different processing architectures onto a single chip with unified memory access. The three units are a conventional CPU based on the Zen 5 architecture, a graphics processor based on the RDNA 3.5 architecture, and a neural processing unit based on the XDNA 2 architecture.

What makes this integration significant is not merely that all three units exist on one chip, but that they share coherent access to the same physical memory. Traditional systems with discrete GPUs require explicit data copying between CPU and GPU memory spaces. This copying takes time and requires careful programming to manage. On Strix Halo, all three processing units can read and write the same memory addresses directly, with hardware maintaining cache coherence so that updates made by one unit are immediately visible to the others.

For quantum simulation, this means the state vector, which might be 16 gigabytes for a 30-qubit system, can live in one place while all three processing units operate on it. There is no need to copy the state vector to the GPU before applying parallel gates, then copy it back to the CPU for precision-sensitive operations. Each unit simply accesses the memory directly.

### The Zen 5 CPU: Precision and Control

The CPU portion of Strix Halo comprises 16 Zen 5 cores capable of executing 32 simultaneous threads. Each core includes AVX-512 vector units that can process 8 double-precision floating point values in a single instruction. This is not the massive parallelism of a GPU, but it offers something the GPU cannot: full IEEE-754 double precision arithmetic with strict compliance to the standard.

The CPU serves two roles in quantum simulation. First, it handles precision-critical computation where double precision is required to maintain phase accuracy. When the compiler determines that a particular gate sequence has strict precision requirements, perhaps because it involves arbitrary-angle rotations deep in the circuit, it generates AVX-512 code targeting the CPU.

Second, the CPU handles classical control flow. Quantum algorithms often include mid-circuit measurements and conditional operations, where the choice of subsequent gates depends on measurement outcomes. This control flow is inherently sequential and poorly suited to GPU execution. The CPU handles this coordination naturally.

### The RDNA 3.5 GPU: Parallel Throughput

The integrated GPU comprises 40 Compute Units, each containing 64 stream processors capable of single-precision floating point arithmetic. This totals 2560 parallel processing elements that can operate simultaneously on different data. For operations that can be expressed as parallel sweeps over large data structures, the GPU offers an order of magnitude more throughput than the CPU.

The limitation is precision. The GPU's native arithmetic is single precision, with 32-bit floating point values offering roughly 10⁻⁷ relative precision compared to 10⁻¹⁶ for double precision. For many quantum gates, particularly the simple ones applied early in a circuit, this precision is sufficient. The Hadamard gate, which creates an equal superposition, involves only values of ±1/√2. The Pauli gates are simple negations and swaps. The CNOT gate copies values conditionally. None of these operations inherently require high precision for a single application.

The precision concern arises from accumulation across many operations. A single-precision Hadamard introduces negligible error. Ten thousand sequential gates, each introducing small errors that compound, may produce unacceptable total error. The compiler's task is to determine which operations can safely use GPU acceleration and which require CPU precision, based on their position in the circuit and the algorithm's tolerance for phase error.

### The XDNA 2 NPU: Accumulation Precision

The neural processing unit is designed for machine learning inference, where the dominant computation pattern is multiply-accumulate: multiplying many pairs of values and summing the results. The XDNA 2 architecture implements this through AI Engine tiles, each containing vector processors with specialized accumulator registers.

These accumulator registers are the feature most relevant to quantum simulation. When summing a long series of products, each intermediate addition can introduce rounding error. If we sum one million values in single precision, the accumulated error can become significant relative to individual terms. The AI Engine addresses this through extended-precision accumulators that maintain more bits during the accumulation, only rounding to the final precision when the sum is complete.

The typical accumulator path accepts bfloat16 operands (16-bit floating point with 8 mantissa bits) and accumulates into 32-bit floating point registers. This means the individual multiply operations use reduced precision, but the sum maintains full single-precision accuracy. For quantum simulation, this pattern applies directly to measurement, where we must sum the squared magnitudes of billions of amplitudes to compute outcome probabilities.

Consider measuring a 30-qubit system. We must compute:

```
P(outcome) = Σᵢ |αᵢ|²
```

where the sum ranges over approximately one billion terms. In naive single precision, the accumulated rounding error from one billion additions would be substantial. Using the NPU's extended accumulators, the sum maintains precision because intermediate results are stored with extra bits.

## The Variational Quantum Eigensolver

### The Algorithm's Purpose

The Variational Quantum Eigensolver, commonly abbreviated VQE, is a quantum algorithm for finding the lowest energy state of a quantum system. This is a fundamental problem in chemistry and materials science. The behavior of molecules depends on how their electrons arrange themselves, and the stable arrangement is the one with lowest total energy. If we can find this ground state energy, we can predict chemical properties, reaction rates, and material characteristics.

Classical algorithms for this problem scale poorly. The exact solution requires manipulating matrices whose size grows exponentially with the number of electrons. Approximation methods trade accuracy for tractability but often fail for strongly correlated systems where electron interactions are complex.

VQE takes a hybrid approach. It uses a quantum computer (or simulator) to prepare and measure quantum states representing electron configurations. It uses a classical computer to optimize the parameters of these states. The quantum portion handles the exponentially large state space naturally, since a quantum computer with n qubits can represent superpositions of 2ⁿ configurations directly. The classical portion handles the optimization, which is well-suited to conventional processors.

### The Algorithm's Structure

VQE proceeds in iterations. Each iteration has four phases.

In the first phase, we prepare an initial quantum state, typically starting from a simple product state where each qubit is independently set to zero or one. This requires minimal computation and can be done in parallel.

In the second phase, we apply a parameterized quantum circuit called an ansatz. This circuit contains rotation gates whose angles are the parameters we will optimize. Different rotation angles produce different quantum states. The ansatz structure is chosen to be capable of producing states similar to the molecular ground state we seek. This phase contains the quantum gates that transform amplitudes and where precision concerns arise.

In the third phase, we measure the quantum state to estimate its energy. This involves computing expectation values, which are weighted averages of amplitude products. Each expectation value requires summing over all amplitudes, making this phase reduction-heavy and suitable for the NPU's accumulator capabilities.

In the fourth phase, a classical optimizer uses the measured energy to adjust the rotation parameters, seeking angles that minimize energy. This is standard numerical optimization using gradients or gradient-free methods. It runs on the CPU.

The algorithm repeats these four phases until the energy converges to a minimum. A typical VQE run might perform hundreds or thousands of iterations, each involving state preparation, dozens to hundreds of parameterized gates, multiple measurements, and an optimization step.

### Why VQE Suits Heterogeneous Execution

The four phases of VQE map naturally to different processing capabilities.

State preparation is embarrassingly parallel with low precision requirements. Setting a billion amplitudes to initial values is a memory-fill operation that the GPU handles efficiently.

The parameterized circuit contains a mix of operations. Simple gates like CNOTs are highly parallel and precision-tolerant. Rotation gates involve trigonometric functions (sine and cosine of the rotation angle) that may require higher precision, especially when the angles are small or the rotations are deep in the circuit. The compiler can analyze the circuit structure to determine which gates can use GPU acceleration and which require CPU precision.

Measurement is a massive reduction operation. Computing each expectation value requires summing products of amplitudes across the entire state vector. This is exactly the multiply-accumulate pattern that the NPU's extended accumulators handle efficiently.

Classical optimization is inherently sequential and runs on the CPU. Between optimization steps, all parameter updates are immediately visible to other processing units through the coherent memory architecture.

The coherent memory is what makes this partitioning practical. Without it, we would need to copy the 16-gigabyte state vector between processing units at phase boundaries. With it, each unit simply accesses the state vector in place, with hardware maintaining consistency.

## Compilation Strategy

The compiler transforms a high-level circuit description into partitioned, hardware-specific code. The following diagram summarizes the complete compilation flow:

```
Compilation Pipeline
═══════════════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────┐
                    │    Quantum Circuit      │
                    │    Description          │
                    │                         │
                    │  H(0), CNOT(0,1),       │
                    │  Ry(θ₁, 2), ...         │
                    └───────────┬─────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │              Circuit Analysis                  │
        │                                               │
        │  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
        │  │ Dependency  │  │  Precision  │  │Reduction│ │
        │  │   Graph     │  │  Analysis   │  │Detection│ │
        │  └─────────────┘  └─────────────┘  └────────┘ │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │           Precision Classification             │
        │                                               │
        │   Gate        Position    Path        Class   │
        │   ────        ────────    ────        ─────   │
        │   H(0)        early       non-critical  Low   │
        │   CNOT(0,1)   early       non-critical  Exact │
        │   Ry(θ,2)     mid         critical      High  │
        │   Rz(φ,3)     late        critical      High  │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │            Partition Assignment                │
        │                                               │
        │   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
        │   │   CPU   │   │   GPU   │   │   NPU   │     │
        │   │         │   │         │   │         │     │
        │   │ • High  │   │ • Low   │   │ • Reduc-│     │
        │   │ • Control│  │ • Exact │   │   tions │     │
        │   │         │   │ • Medium│   │         │     │
        │   └─────────┘   └─────────┘   └─────────┘     │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │              Code Generation                   │
        │                                               │
        │  ┌───────────┐ ┌───────────┐ ┌───────────┐    │
        │  │  AVX-512  │ │  AMDGPU   │ │  AIEVec   │    │
        │  │   MLIR    │ │   MLIR    │ │   MLIR    │    │
        │  └───────────┘ └───────────┘ └───────────┘    │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Coordinated Binary    │
                    │   with Synchronization  │
                    │   Barriers              │
                    └─────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

### The Compiler's Analysis

The Fidelity compiler receives a quantum circuit description and a target hardware specification. From the circuit, it extracts the sequence of gates, their parameters, and the measurements to be performed. From the target specification, it knows the capabilities of each processing unit: the CPU's AVX-512 double precision, the GPU's parallel single precision, and the NPU's accumulator architecture.

The compiler performs several analyses on the circuit.

First, it builds a dependency graph showing which gates can execute in parallel. Gates on different qubits with no control dependencies between them can potentially execute simultaneously. This identifies opportunities for GPU acceleration.

Second, it computes precision requirements for each gate. This analysis considers the gate's position in the circuit (how many subsequent gates will compound any errors), the gate's inherent precision sensitivity (arbitrary rotations are more sensitive than Paulis), and whether the gate lies on a critical interference path.

Third, it identifies reduction patterns where many values must be combined into fewer. Measurement operations are the primary example, but normalization checks and expectation value computations follow the same pattern.

Based on these analyses, the compiler partitions the circuit into segments and assigns each segment to a processing unit.

### Precision Classification

The precision classifier assigns each gate to one of four categories.

Gates classified as Exact require no floating point computation at all. The Pauli-X gate, for example, simply swaps pairs of amplitudes without arithmetic. The compiler can implement these as memory operations rather than multiplications.

Gates classified as Low precision can tolerate single-precision or even reduced-precision arithmetic. The Hadamard gate falls in this category when applied early in a circuit. Its matrix entries are ±1/√2, which single precision represents accurately. The small rounding error introduced is negligible when the gate is far from measurement.

Gates classified as Medium precision should use single precision but warrant caution. These are gates in the middle of circuits or on moderately important interference paths. The compiler may apply compensated arithmetic techniques to reduce accumulated error.

Gates classified as High precision require double precision arithmetic on the CPU. These include arbitrary-rotation gates deep in the circuit, gates on critical interference paths, and any gate where the requested angle has many significant digits.

The classification algorithm considers context, not just the gate itself. The same Hadamard gate might be classified Low when at circuit position 5 but Medium at position 500, because errors introduced at position 500 have less opportunity to be corrected by subsequent interference before measurement.

### Partition Assignment

With precision requirements determined, the compiler assigns gates to processing units.

High precision gates are assigned to the CPU. The compiler generates AVX-512 intrinsics that process 8 amplitudes per instruction in double precision. For a 30-qubit system with one billion amplitudes, a single-qubit gate requires 500 million pair transformations. With 16 cores each processing 8 pairs per instruction, the CPU can complete this in approximately 4 million cycles, which takes roughly 1 millisecond at 4 GHz.

Low and Medium precision gates are assigned to the GPU when they can be batched together. The compiler groups consecutive GPU-suitable gates into kernels that process the entire state vector in parallel. Each of the 2560 stream processors handles a portion of the amplitude vector, and the full transformation completes in a single parallel sweep.

Measurement reductions are assigned to the NPU. The compiler generates AIE kernel code that loads amplitudes, computes squared magnitudes using the multiply unit, and accumulates results using the extended-precision accumulator registers. The final probability values are stored in coherent memory where the CPU can read them for the optimization step.

### Synchronization and Data Flow

The coherent memory architecture simplifies synchronization but does not eliminate it entirely. When one processing unit modifies amplitudes, other units must wait for those modifications before reading the same amplitudes. The compiler inserts appropriate barriers at segment boundaries.

The typical execution flow for one VQE iteration proceeds as follows.

The CPU initializes the optimization iteration and signals that state preparation should begin. The GPU executes the state preparation kernel, writing initial amplitudes to the state vector in coherent memory. A barrier ensures all writes complete before the next phase.

The CPU and GPU cooperatively execute the parameterized circuit. GPU-assigned gate batches execute in parallel on the GPU. CPU-assigned high-precision gates execute on the CPU. The compiler schedules these to minimize idle time, potentially overlapping CPU work on one circuit segment with GPU work on an independent segment.

After the circuit completes, the NPU executes measurement kernels. For each expectation value needed by the optimizer, the NPU sums amplitude products using its accumulators. Results are written to coherent memory.

The CPU reads the measurement results and executes one step of the classical optimizer, updating the rotation angles for the next iteration.

This entire sequence involves no explicit memory copies. Each unit reads from and writes to the shared state vector directly. The latency savings compared to a discrete GPU system with explicit transfers can be substantial, especially given that quantum simulation often alternates frequently between CPU and GPU work.

## Code Generation

### Representing Quantum Operations

The compiler's internal representation expresses quantum operations in a form suitable for analysis and code generation. Each gate is represented with its target qubits, any control qubits, and its matrix elements.

A single-qubit gate applies a 2×2 unitary matrix to the amplitudes associated with a single qubit position. For qubit k in an n-qubit system, this means locating all amplitude pairs where the k-th bit of the index differs and transforming each pair by matrix multiplication.

```fsharp
type SingleQubitGate = {
    Target: int                    // Which qubit this gate acts on
    Matrix: Complex[,]             // The 2×2 unitary matrix
    PrecisionClass: Precision      // Determined by analysis
    AssignedUnit: ProcessingUnit   // CPU, GPU, or NPU
}

type Precision = Exact | Low | Medium | High
type ProcessingUnit = CPU | GPU | NPU
```

The compiler maintains the full circuit as a sequence of such gate records, along with the dependency graph relating them.

### MLIR Generation for CPU

For gates assigned to the CPU, the compiler generates MLIR operations targeting the AVX-512 instruction set. The generated code processes amplitude pairs in vectors of 8 double-precision complex values, which occupy two 512-bit registers (one for real parts, one for imaginary parts).

The core transformation for a single-qubit gate with matrix [[a, b], [c, d]] applied to qubit k:

```mlir
// Load 8 amplitude pairs at indices [i, i+2^k] for i in batch
%alpha_re = vector.load %state_re[%batch_start] : vector<8xf64>
%alpha_im = vector.load %state_im[%batch_start] : vector<8xf64>
%beta_re = vector.load %state_re[%beta_start] : vector<8xf64>
%beta_im = vector.load %state_im[%beta_start] : vector<8xf64>

// Matrix element constants (broadcast to vectors)
%a_re = arith.constant dense<...> : vector<8xf64>
%a_im = arith.constant dense<...> : vector<8xf64>
// ... similarly for b, c, d

// Compute new_alpha = a * alpha + b * beta (complex multiplication)
// new_alpha_re = a_re * alpha_re - a_im * alpha_im + b_re * beta_re - b_im * beta_im
%t1 = arith.mulf %a_re, %alpha_re : vector<8xf64>
%t2 = arith.mulf %a_im, %alpha_im : vector<8xf64>
%t3 = arith.subf %t1, %t2 : vector<8xf64>
%t4 = arith.mulf %b_re, %beta_re : vector<8xf64>
%t5 = arith.mulf %b_im, %beta_im : vector<8xf64>
%t6 = arith.subf %t4, %t5 : vector<8xf64>
%new_alpha_re = arith.addf %t3, %t6 : vector<8xf64>

// ... similarly for imaginary parts and new_beta

// Store results
vector.store %new_alpha_re, %state_re[%batch_start] : vector<8xf64>
// ... other stores
```

The full loop iterates over all 2ⁿ⁻¹ / 8 batches of amplitude pairs. With AVX-512 fused multiply-add instructions, each complex multiplication can be performed in fewer cycles than the code above suggests, since `a * b + c * d` patterns are recognized and combined.

### MLIR Generation for GPU

For gates assigned to the GPU, the compiler generates compute shader code via the AMDGPU dialect. Each GPU work item processes one or more amplitude pairs independently.

The same single-qubit gate, expressed for GPU execution:

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gbx = %num_blocks, %gby = 1, %gbz = 1)
           threads(%tx, %ty, %tz) in (%gtx = 256, %gty = 1, %gtz = 1) {

    // Global thread ID determines which amplitude pair to process
    %bid = gpu.block_id x
    %tid = gpu.thread_id x
    %gid = arith.addi %bid * 256, %tid  // Global ID

    // Compute amplitude pair indices
    %pair_idx = arith.muli %gid, %c1 : index
    %alpha_idx = call @compute_alpha_index(%pair_idx, %qubit_k)
    %beta_idx = call @compute_beta_index(%pair_idx, %qubit_k)

    // Load amplitudes (single precision)
    %alpha_re = memref.load %state_re[%alpha_idx] : f32
    %alpha_im = memref.load %state_im[%alpha_idx] : f32
    %beta_re = memref.load %state_re[%beta_idx] : f32
    %beta_im = memref.load %state_im[%beta_idx] : f32

    // Matrix multiplication (same structure as CPU, but scalar f32)
    // ... multiplication code ...

    // Store results
    memref.store %new_alpha_re, %state_re[%alpha_idx] : f32
    // ... other stores

    gpu.terminator
}
```

The GPU kernel launches one thread per amplitude pair. With 2ⁿ⁻¹ pairs and 2560 parallel stream processors, a 30-qubit gate completes in approximately 200,000 thread batches, each executing in parallel.

### MLIR Generation for NPU Measurement

Measurement reduction targeting the NPU uses the aievec dialect to exploit hardware accumulators. The measurement kernel computes squared magnitudes and accumulates them:

```mlir
// Tile configuration for measurement reduction
aie.device(xdna2) {
    %tile = aie.tile(0, 0)

    %core = aie.core(%tile) {
        // Accumulator register starts at zero
        %acc = arith.constant dense<0.0> : vector<16xf32>

        // Loop over amplitude chunks
        %result = scf.for %i = %c0 to %num_chunks step %c1
                          iter_args(%running_acc = %acc) -> vector<16xf32> {

            // Load 16 complex amplitudes
            %re = vector.load %state_re[%i * 16] : vector<16xf32>
            %im = vector.load %state_im[%i * 16] : vector<16xf32>

            // Compute squared magnitudes: |α|² = re² + im²
            // Using fused multiply-add for re² + im²
            %re_sq = arith.mulf %re, %re : vector<16xf32>
            %mag_sq = llvm.intr.fma %im, %im, %re_sq : vector<16xf32>

            // Accumulate with extended precision
            %new_acc = aievec.mac_elem %mag_sq, %ones, %running_acc :
                vector<16xf32>, vector<16xf32>, vector<16xf32> -> vector<16xf32>

            scf.yield %new_acc : vector<16xf32>
        }

        // Horizontal reduction of accumulator lanes
        %total = vector.reduction <add>, %result : vector<16xf32> into f32

        // Store probability result
        memref.store %total, %probabilities[%measurement_idx] : f32

        aie.end
    }
}
```

The key insight is that the `aievec.mac_elem` operation uses the hardware accumulator path, which maintains extended precision during the sum. This prevents the loss of small amplitude contributions when summing billions of values.

## Error Analysis and Verification

### The Precision Budget

The compiler maintains a precision budget for each circuit segment, tracking how much error can accumulate before results become unreliable. This budget depends on the algorithm's tolerance for imprecision and the circuit's structure.

For VQE specifically, the tolerance comes from the optimization process. VQE does not require exact expectation values; it only requires that the energy landscape be smooth enough for the optimizer to find the minimum. Errors up to 10⁻⁴ in individual expectation values are typically acceptable, as they produce small noise in the optimization landscape without obscuring the global minimum.

This tolerance is distributed across the circuit. If the total error budget is 10⁻⁴, and the circuit has 100 gates followed by measurement, each gate can introduce approximately 10⁻⁶ error if they compound linearly. In practice, some errors cancel and the actual compounding is sub-linear, so the budget can be allocated more generously.

The compiler tracks precision requirements as intervals. A gate classified as Low precision might have an error tolerance of 10⁻⁵ per operation. A gate classified as High precision might require 10⁻¹⁴ per operation. The sum of allocated error tolerances across all gates must not exceed the total circuit budget.

### Static Verification with F*

The Fidelity framework employs the F* theorem prover to verify that code generation preserves precision properties. Each transformation pattern in the compiler has an associated F* proof that establishes its error characteristics.

For the Kahan summation pattern used in some CPU reductions:

```fstar
module Kahan.ErrorBound

open FStar.Real

// Machine epsilon for IEEE-754 double precision
let eps : real = 0x1.0p-53

// Kahan summation maintains O(n*eps) error instead of O(n²*eps) for naive sum
val kahan_error_bound:
    values: seq real{length values > 0} ->
    Lemma (ensures (
        let n = length values in
        let kahan_result = kahan_sum values in
        let exact_result = exact_sum values in
        let max_val = max_abs values in
        abs(kahan_result - exact_result) <= 2.0 * (real_of_int n) * eps * max_val
    ))
```

For the NPU accumulator pattern:

```fstar
module AIEAccumulator.ErrorBound

open FStar.Real

// Accumulator maintains full f32 precision during summation
val accumulator_sum_bound:
    values: seq float32{length values > 0} ->
    Lemma (ensures (
        let n = length values in
        let acc_result = aie_accumulated_sum values in
        let exact_result = exact_sum values in
        let max_val = max_abs values in
        // Only one rounding at the end, not n roundings
        abs(acc_result - exact_result) <= eps_f32 * abs(exact_result) + eps_f32 * max_val
    ))
```

These proofs are verified once when the compiler pattern is implemented. At compile time, the compiler checks that each generated code segment uses only verified patterns and that the composition of patterns stays within the circuit's precision budget.

### Runtime Monitoring

For development and debugging, the compiler can optionally insert runtime checks that verify precision properties. These checks compare results against reference implementations or monitor accumulator magnitudes for unexpected growth.

```fsharp
// Optional runtime precision monitoring
type MonitoringLevel =
    | None           // No overhead, production mode
    | Statistical    // Sample 0.1% of operations, minimal overhead
    | Full           // Check all operations, significant overhead

let insertMonitoring (level: MonitoringLevel) (kernel: MLIRKernel) : MLIRKernel =
    match level with
    | None -> kernel
    | Statistical ->
        // Insert sampling check: compare random subset against f64 reference
        kernel |> addSampledVerification 0.001
    | Full ->
        // Insert full check: maintain parallel f64 accumulation
        kernel |> addFullVerification
```

Statistical monitoring is useful during algorithm development. It catches precision problems with low overhead, allowing the developer to identify which circuit segments are causing issues and adjust precision classifications accordingly.

## A Complete Example: Hydrogen Molecule Ground State

### The Physical Problem

The simplest non-trivial molecule is molecular hydrogen, H₂, which consists of two hydrogen atoms sharing two electrons. Despite its simplicity, the electronic structure of H₂ already exhibits the correlation effects that make quantum chemistry computationally difficult. The two electrons influence each other's behavior in ways that cannot be captured by treating them independently.

The ground state energy of H₂ as a function of the inter-atomic distance is a standard benchmark for quantum chemistry methods. At the equilibrium bond length of approximately 0.74 Ångströms, the energy is approximately -1.17 Hartree. Classical methods can compute this accurately for H₂, but the same methods fail for larger molecules. VQE on a quantum computer (or accurate simulator) provides a path to treating larger systems.

For our case study, we simulate H₂ using a minimal basis set that requires 4 qubits to represent the electronic configuration. This is small enough to trace through the compilation process in detail while illustrating all the relevant phenomena.

### The Quantum Circuit

The VQE circuit for H₂ begins with preparing an initial state representing two electrons in the lowest-energy orbitals:

```
Initial state: |0011⟩  (qubits 2 and 3 occupied, 0 and 1 empty)
```

The parameterized ansatz applies single-qubit rotations and two-qubit entangling gates. A common choice is the Unitary Coupled Cluster ansatz, which for H₂ simplifies to the following circuit:

```
H₂ VQE Ansatz Circuit (4 qubits, 10 parameters)
═══════════════════════════════════════════════════════════════════════════════

        │ Layer 1  │ Layer 2 │ Layer 3  │ Layer 4 │ Layer 5  │
        │          │         │          │         │          │
q₀: |0⟩─┤ Ry(θ₁)  ├────●────┤ Ry(θ₅)  ├─────────┼──────────┼──────  ───▶
        │          │    │    │          │         │          │
q₁: |0⟩─┤ Ry(θ₂)  ├────⊕────┤ Ry(θ₆)  ├────●────┤ Ry(θ₉)  ├──────  ───▶
        │          │         │          │    │    │          │
q₂: |1⟩─┤ Ry(θ₃)  ├────●────┤ Ry(θ₇)  ├────⊕────┤ Ry(θ₁₀) ├──────  ───▶
        │          │    │    │          │         │          │
q₃: |1⟩─┤ Ry(θ₄)  ├────⊕────┤ Ry(θ₈)  ├─────────┼──────────┼──────  ───▶

        └──────────┴─────────┴──────────┴─────────┴──────────┘

Legend:  ● ─── Control qubit        Ry(θ) ─── Y-rotation by angle θ
         ⊕ ─── Target (CNOT)        |0⟩, |1⟩ ─── Initial qubit states

═══════════════════════════════════════════════════════════════════════════════
```

This circuit has 10 parameters (θ₁ through θ₁₀) that the optimizer will adjust to find the ground state. The circuit depth is 5 layers, and the total gate count is 14: ten single-qubit Ry rotations and four two-qubit CNOT gates. The structure alternates between parallel rotation layers (where all qubits receive independent rotations) and entangling layers (where CNOT gates create correlations between qubits).

The gate types have different computational characteristics:

```
Gate Analysis
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────┬─────────────────────────────────────────────────────────┐
│ Gate Type       │ Computational Characteristics                          │
├─────────────────┼─────────────────────────────────────────────────────────┤
│                 │                                                         │
│ Ry(θ) Rotation  │  Matrix:  ┌                      ┐                      │
│                 │           │  cos(θ/2)  -sin(θ/2) │                      │
│                 │           │  sin(θ/2)   cos(θ/2) │                      │
│                 │           └                      ┘                      │
│                 │                                                         │
│                 │  Operations: 2 trig evaluations + 4 multiplies + 2 adds │
│                 │  Precision: depends on θ magnitude and circuit depth    │
│                 │  Parallelism: all 2ⁿ⁻¹ amplitude pairs independent      │
│                 │                                                         │
├─────────────────┼─────────────────────────────────────────────────────────┤
│                 │                                                         │
│ CNOT            │  Matrix:  ┌            ┐                                │
│ (controlled-X)  │           │ 1  0  0  0 │  Swaps |10⟩ ↔ |11⟩             │
│                 │           │ 0  1  0  0 │  if control qubit is |1⟩       │
│                 │           │ 0  0  0  1 │                                │
│                 │           │ 0  0  1  0 │                                │
│                 │           └            ┘                                │
│                 │                                                         │
│                 │  Operations: conditional swap (no arithmetic)           │
│                 │  Precision: exact (no floating point error)             │
│                 │  Parallelism: all 2ⁿ⁻² amplitude quadruples independent │
│                 │                                                         │
└─────────────────┴─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

The energy measurement requires computing expectation values for multiple Pauli terms in the Hamiltonian. For H₂ in a minimal basis, the Hamiltonian has the form:

```
H = g₀ I + g₁ Z₀ + g₂ Z₁ + g₃ Z₂ + g₄ Z₃
  + g₅ Z₀Z₁ + g₆ Z₀Z₂ + g₇ Z₀Z₃ + g₈ Z₁Z₂ + g₉ Z₁Z₃ + g₁₀ Z₂Z₃
  + g₁₁ X₀X₁Y₂Y₃ + g₁₂ X₀Y₁Y₂X₃ + g₁₃ Y₀X₁X₂Y₃ + g₁₄ Y₀Y₁X₂X₃
```

where the g coefficients depend on the inter-atomic distance and the basis set. Each term requires a separate expectation value computation, and the total energy is the weighted sum.

### Compilation Trace

The compiler receives this circuit and the Strix Halo target specification.

Precision analysis proceeds as follows. The Ry rotation gates involve sine and cosine of the parameter angles. For typical optimization trajectories, these angles range from 0 to 2π with values throughout the interval. Single precision can represent these rotations adequately since the trigonometric functions are smooth and small errors in the angle produce proportionally small errors in the rotation.

The CNOT gates are classified as Exact because they involve no arithmetic, only conditional swapping of amplitude values.

With a 5-layer circuit of depth, errors compound minimally. A single-precision error of 10⁻⁷ compounded over 5 layers yields approximately 5 × 10⁻⁷ total error per amplitude, which is well within the VQE error tolerance of 10⁻⁴.

The compiler's partition decision:
- State preparation: GPU (parallel initialization)
- All Ry gates: GPU (single precision sufficient)
- All CNOT gates: GPU (exact operations, highly parallel)
- Expectation value computation: NPU (reduction with accumulator)
- Classical optimizer: CPU

For this 4-qubit example, the state vector has only 16 complex amplitudes, which is too small to benefit from heterogeneous execution. The overhead of kernel launch and synchronization would dominate actual computation. The compiler recognizes this and generates a CPU-only implementation for the circuit, reserving heterogeneous execution for when the simulation scales to larger qubit counts.

### Scaling to 28 Qubits

To demonstrate the heterogeneous capabilities, consider scaling the same H₂ algorithm structure to 28 qubits, which might represent a more realistic molecular system or a research testbed for algorithmic development. The state vector now has 2²⁸ ≈ 268 million complex amplitudes, requiring approximately 4 gigabytes of storage in single precision.

At this scale, the heterogeneous partition becomes advantageous.

State preparation, setting all amplitudes to zero except the initial configuration, launches as a GPU kernel that fills 4 gigabytes of memory with zeros, then writes the initial amplitude. This completes in a single parallel sweep.

The Ry rotation gates, now classified as Medium precision due to the longer circuit, execute on the GPU in batched kernels. Each single-qubit gate kernel processes 2²⁷ amplitude pairs in parallel. With 10 Ry gates, this amounts to 10 kernel launches, each completing in approximately 100,000 thread batches.

The CNOT gates execute as GPU kernels in the same pattern. The 4 CNOTs add 4 more kernel launches.

Expectation value computation for 15 Hamiltonian terms requires 15 NPU reduction operations. Each reduction sums 2²⁸ squared magnitudes using the extended accumulator. The NPU processes this in chunks of 16 amplitudes per accumulator operation, requiring 2²⁴ accumulator steps per term, or 15 × 2²⁴ ≈ 250 million accumulator operations total.

The classical optimizer runs on the CPU, computing gradients and updating the 10 rotation angles.

The following diagram shows the timing breakdown for one VQE iteration at 28 qubits:

```
VQE Iteration Timing (28 qubits, 268 million amplitudes)
═══════════════════════════════════════════════════════════════════════════════

Phase               Unit      Time        Breakdown
─────               ────      ────        ─────────

State Prep          GPU       0.1 ms      │▌                              │
                                          │                               │
Gate Kernels        GPU       7.0 ms      │███████████████▌               │
  Ry × 10                                 │  (10 rotation kernels)        │
  CNOT × 4                                │  (4 entangling kernels)       │
                                          │                               │
Expectation Values  NPU      30.0 ms      │██████████████████████████████████████████████████████████████│
  15 Pauli terms                          │  (15 reductions, 268M terms each)                            │
                                          │                               │
Optimization        CPU       0.1 ms      │▌                              │
                                          │                               │
───────────────────────────────────────────────────────────────────────────────
Total                        37.2 ms

Time distribution:           │  GPU 19%  │         NPU 81%          │CPU│
                             └───────────┴──────────────────────────┴───┘

═══════════════════════════════════════════════════════════════════════════════
```

The timing reveals that measurement dominates the iteration. This is expected: each of the 15 expectation values requires summing 268 million squared magnitudes, for a total of 4 billion accumulator operations per iteration. The NPU's hardware accumulators make this tractable; on CPU or GPU without extended precision, the accumulated rounding error from 4 billion additions would compromise the energy estimate.

A typical VQE run might require 500 iterations to converge, giving a total simulation time of approximately 18 seconds for a 28-qubit molecular energy calculation. This is competitive with specialized quantum simulation software running on discrete GPUs, while using a laptop processor. The heterogeneous architecture ensures that each phase executes on the hardware best suited to its computational pattern.

## Performance Characteristics

### Scaling Behavior

The three processing units have different scaling characteristics as qubit count increases.

CPU execution with AVX-512 scales linearly with amplitude count. Each amplitude pair requires a fixed number of operations, and the CPU processes 8 pairs per vector instruction. The time for a single-qubit gate is:

```
T_cpu = 2^(n-1) / 8 / (cores × frequency × instructions_per_cycle)
```

For 30 qubits on a 16-core Zen 5 at 4 GHz with 2 fused multiply-adds per cycle:

```
T_cpu = 2^29 / 8 / (16 × 4 × 10^9 × 2) ≈ 2 ms
```

GPU execution benefits from massive parallelism but has higher per-thread overhead. The scaling is:

```
T_gpu = 2^(n-1) / threads_per_batch × batches_per_kernel × kernel_launch_overhead
```

For 30 qubits with 2560 threads and negligible launch overhead:

```
T_gpu = 2^29 / 2560 × time_per_thread ≈ 0.5 ms
```

The GPU is faster per gate, but the CPU wins if many gates must execute sequentially without batching opportunities.

NPU execution for reduction has unique characteristics. The accumulator can process 16 values per operation with extended precision, giving:

```
T_npu = 2^n / 16 / (AIE_frequency × tiles)
```

For 30 qubits with 20 AIE tiles at 1 GHz:

```
T_npu = 2^30 / 16 / (10^9 × 20) ≈ 3 ms per reduction
```

### Memory Bandwidth Considerations

All three units access the same physical memory, which has finite bandwidth. LPDDR5X at 8000 MT/s with a 256-bit interface provides approximately 250 GB/s theoretical bandwidth, with practical bandwidth around 80% of theoretical.

A 30-qubit state vector is 16 GB. A single gate that reads and writes all amplitudes requires 32 GB of memory traffic. At 200 GB/s practical bandwidth, this takes 160 ms if purely memory bound.

In practice, gates do not access all amplitudes uniformly. A single-qubit gate on qubit k accesses pairs separated by 2^k indices. For high k (most significant qubits), these accesses are sequential and cache-friendly. For low k (least significant qubits), accesses are strided and potentially cache-hostile.

The compiler accounts for this by preferring to batch operations on the same qubits and by reordering gates when the dependency graph permits. Batching multiple gates into a single kernel that operates on the same amplitudes reduces memory traffic by reading each amplitude once and writing it once, even if multiple transformations are applied.

### Energy Efficiency

Heterogeneous execution offers energy efficiency advantages beyond raw performance. The NPU, designed for inference workloads, achieves high operations per watt for the multiply-accumulate patterns it handles. The GPU, while power-hungry at full load, can complete parallel work faster and return to idle sooner than the CPU would take to complete the same work. The CPU, fully utilized only for precision-critical work, avoids the energy cost of performing operations that other units handle more efficiently.

For battery-powered development on a laptop, this translates to longer running time per charge when executing quantum simulations compared to a CPU-only approach.

## Conclusion

This case study has traced the compilation of a variational quantum eigensolver from high-level circuit description through analysis, partitioning, code generation, and execution on heterogeneous hardware. The key observations are:

Quantum simulation is not a single computational pattern but a composition of patterns with different requirements. State preparation, gate application, measurement, and classical optimization each map best to different processing capabilities.

Modern heterogeneous processors offer specialized units that match these patterns. The CPU provides precision and control. The GPU provides parallel throughput. The NPU provides efficient reduction with extended accumulation precision.

Coherent memory eliminates the copy overhead that traditionally dominates heterogeneous execution. When all units can access the same state vector directly, the granularity of work partitioning can be much finer.

A compiler can analyze circuit structure to determine precision requirements, identify parallelism opportunities, and generate specialized code for each processing unit. The combination achieves performance competitive with specialized GPU systems while maintaining numerical guarantees verified by formal methods.

The Strix Halo processor, with its 64 GB of coherent memory across CPU, GPU, and NPU, provides a practical platform for exploring these techniques. It supports quantum simulation at meaningful scale (28-31 qubits) while forcing honest confrontation with the challenges of heterogeneous partitioning.

This represents one case study in the broader Fidelity project, which aims to compile high-level code with semantic guarantees preserved through to native execution. The same principles of analyzing precision requirements, partitioning across heterogeneous units, and verifying correctness apply beyond quantum simulation to numerical optimization, differential equations, machine learning training, and other domains where precision and parallelism must be balanced.

The quantum simulation domain is particularly instructive because the consequences of precision loss are dramatic and measurable. An incorrect phase leads to incorrect interference, which leads to wrong measurement outcomes. There is no graceful degradation. This forces the compiler to be honest about precision guarantees and the developer to understand the limits of the hardware. These lessons apply broadly, even to domains where the consequences of imprecision are subtler.
