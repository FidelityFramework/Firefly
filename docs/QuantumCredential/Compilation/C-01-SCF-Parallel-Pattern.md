# MLIR Dialect Strategy: Demo vs Full Vision

> **Demo Path**: Standard MLIR dialects (scf, func, arith, memref, llvm)
>
> **Full Vision**: Custom DCont/Inet dialects with purity-driven selection

---

## Executive Summary

The Fidelity framework's compilation model distinguishes between delimited continuations (DCont) for sequential effects and interaction nets (Inet) for pure parallelism. Implementing these as custom MLIR dialects requires significant infrastructure: TableGen definitions, C++ dialect implementations, lowering passes, and verification tooling.

For the QuantumCredential demo, this investment is unnecessary. Standard MLIR dialects provide the parallel execution semantics needed to validate the concept. The custom dialects become valuable later when formal verification and automatic purity-driven selection are required.

This document captures the pragmatic dialect strategy for the demo and the progression path to full implementation.

---

## The DCont/Inet Duality (Conceptual Model)

The Firefly compiler analyzes referential transparency to determine compilation strategy:

| Pattern | Compilation Target | Characteristics |
|---------|-------------------|-----------------|
| **Pure operations** | Inet dialect | No effects, no dependencies, parallel execution |
| **Effectful operations** | DCont dialect | I/O, state, sequential with suspension points |
| **Hybrid** | Both | Inet wraps DCont at effect boundaries |

For quad-channel ADC sampling:

```
Inet (parallel across 4 cores)
  └── DCont (sequential within each core)
        └── shift at each ADC read (natural preemption point)
```

This model remains valid regardless of implementation approach. The question is whether to implement it via custom dialects or standard MLIR infrastructure.

---

## Custom Dialect Requirements (Full Vision)

Implementing DCont and Inet as first-class MLIR dialects requires:

### TableGen Definitions

```tablegen
// Hypothetical Inet dialect definition
def Inet_Dialect : Dialect {
  let name = "inet";
  let summary = "Interaction net dialect for pure parallel computation";
  let cppNamespace = "::fidelity::inet";
}

def Inet_ParallelOp : Inet_Op<"parallel", [...]> {
  let summary = "Execute independent computations in parallel";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
}

// Hypothetical DCont dialect definition
def DCont_ShiftOp : DCont_Op<"shift", [...]> {
  let summary = "Capture current continuation and suspend";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$continuation);
}
```

### C++ Dialect Implementation

- Operation definitions and verification
- Type system integration
- Canonicalization patterns
- Folding rules

### Lowering Passes

- Inet to scf.parallel or GPU dialect
- DCont to async dialect or coroutine intrinsics
- Hybrid lowering for nested patterns

### Purity Analysis Integration

- Referential transparency detection in Alex
- Automatic dialect selection based on analysis
- Verification that selected dialect matches code properties

### Estimated Effort

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| TableGen definitions | 2-3 weeks | MLIR familiarity |
| C++ dialect code | 4-6 weeks | LLVM/MLIR build infrastructure |
| Lowering passes | 4-6 weeks | Target dialect expertise |
| Purity analysis | 2-4 weeks | Alex architecture |
| Testing/verification | 2-4 weeks | All above |
| **Total** | **14-23 weeks** | Sequential dependencies |

This is post-demo work.

---

## Standard Dialect Approach (Demo Path)

For the demo, standard MLIR dialects provide equivalent runtime behavior:

### Dialect Stack

```
func + scf + arith + memref
        ↓
      llvm
        ↓
   LLVM IR → native
```

### Dialect Roles

| Dialect | Role in Demo | Standard? |
|---------|--------------|-----------|
| **func** | Function definitions, syscall wrappers | Yes |
| **scf** | `scf.parallel` for multi-channel sampling | Yes |
| **arith** | Bit manipulation, interleaving | Yes |
| **memref** | Buffer management, ADC sample storage | Yes |
| **llvm** | Final lowering to LLVM IR | Yes |

All dialects have mature implementations, comprehensive documentation, and battle-tested lowering paths.

### Parallel Sampling via scf.parallel

```mlir
// Quad-channel parallel ADC sampling
func.func @sampleAllChannels(%buffer: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // scf.parallel provides Inet-like "run these simultaneously"
  scf.parallel (%ch) = (%c0) to (%c4) step (%c1) {
    %sample = func.call @readAdcChannel(%ch) : (index) -> i32
    memref.store %sample, %buffer[%ch] : memref<4xi32>
    scf.yield
  }

  return
}

// ADC read via Platform.Bindings (syscall)
func.func private @readAdcChannel(index) -> i32
```

### Continuation Points (Implicit)

The DCont-like suspension points come "for free" from the OS:

1. `@readAdcChannel` calls into Platform.Bindings
2. Platform.Bindings emits a syscall (sysfs read or ioctl)
3. The syscall traps to kernel mode
4. The kernel handles I/O, scheduler can preempt
5. When I/O completes, execution resumes

No explicit continuation capture needed. Linux provides preemptible I/O at the syscall boundary.

### Lowering Path

```
scf.parallel
    ↓ (scf-to-openmp or scf-parallel-loop-tiling)
omp.parallel / omp.wsloop
    ↓ (convert-openmp-to-llvm)
llvm.call @__kmpc_fork_call (OpenMP runtime)
    ↓ (llvm translation)
LLVM IR with pthread/OpenMP calls
    ↓ (llc)
Native ARM64 binary
```

Alternative path without OpenMP:

```
scf.parallel
    ↓ (scf-to-cf)
cf.br / cf.cond_br (unrolled or serialized)
    ↓ (convert-cf-to-llvm)
llvm dialect
    ↓
LLVM IR
```

For true parallelism, the OpenMP path is preferred.

---

## What the Demo Validates

Using standard dialects, the demo still validates:

| Concept | How Validated |
|---------|---------------|
| **Parallel entropy sampling** | scf.parallel executes on 4 cores |
| **Natural suspension at I/O** | Syscalls yield to OS scheduler |
| **Interleaved entropy** | arith operations combine channels |
| **Platform.Bindings pattern** | func.call to Alex-emitted syscalls |
| **Quotation-based constraints** | fsnative nanopasses attach metadata |

What it defers:

| Concept | Deferred Until |
|---------|----------------|
| **Automatic purity analysis** | Custom dialect selection |
| **Zero-allocation continuations** | DCont dialect with stack-based capture |
| **Formal parallelism verification** | Inet dialect with proof obligations |
| **Interaction net reduction** | Inet dialect semantics |

---

## Progression Path

### Phase 1: Demo (January)

- Standard MLIR dialects only
- Manual "this is parallel" decisions in Alex
- scf.parallel for multi-channel sampling
- Syscall-based I/O with OS-provided preemption

### Phase 2: Purity Analysis (Post-Demo)

- Extend Alex with referential transparency detection
- Annotate PSG nodes with purity information
- Generate scf.parallel automatically for pure regions
- Still using standard dialects

### Phase 3: DCont Dialect (Future)

- TableGen definitions for continuation operations
- Stack-based continuation capture (zero allocation)
- Integration with async patterns
- Lowering to coroutine intrinsics

### Phase 4: Inet Dialect (Future)

- TableGen definitions for interaction net primitives
- Formal reduction semantics
- GPU and distributed lowering paths
- Integration with SPIR-V for heterogeneous targets

### Phase 5: Unified Selection (Vision)

- Automatic dialect selection from purity analysis
- Hybrid patterns with clean composition
- Formal verification of parallelism properties
- Full DCont/Inet duality as described in architecture documents

---

## File References

### Demo Implementation

| File | Purpose |
|------|---------|
| `Alex/CodeGeneration/MLIRBuilder.fs` | MLIR emission infrastructure |
| `Alex/Bindings/Linux/` | Platform-specific syscall emission |
| `Fidelity.Platform` | Platform.Bindings signatures |

### Architecture Documents

| Document | Relevance |
|----------|-----------|
| [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md) | Quad-channel hardware design |
| [05_PostQuantum_Architecture.md](./05_PostQuantum_Architecture.md) | Parallel entropy pipeline |
| [README.md](./README.md) | DCont/Inet duality overview |

### SpeakEZ Articles

| Article | Relevance |
|---------|-----------|
| Seeking Referential Transparency | Purity analysis and dialect selection |
| The DCont/Inet Duality | Computation expression decomposition |
| Delimited Continuations: Fidelity's Turning Point | Continuation preservation |

---

## Decision Record

**Decision**: Use standard MLIR dialects for the QuantumCredential demo.

**Rationale**:
1. Custom dialects require 14-23 weeks of infrastructure work
2. Standard dialects provide equivalent runtime behavior for demo scenarios
3. Demo timeline requires pragmatic risk management
4. Conceptual validation does not require formal dialect semantics

**Consequences**:
- Parallel execution validated via scf.parallel
- Continuation points implicit at syscall boundaries
- No formal purity-driven dialect selection
- Architecture documents describe vision; demo validates concept

**Revisit When**:
- Formal verification of parallelism becomes a requirement
- Zero-allocation continuation capture is needed
- Automatic purity analysis drives compilation strategy
- GPU/heterogeneous targets require Inet semantics
