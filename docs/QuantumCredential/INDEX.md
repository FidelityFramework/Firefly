# QuantumCredential Documentation

## Vision

QuantumCredential is a hardware/software co-designed quantum random number generator (QRNG) that produces cryptographically certified entropy for post-quantum security applications. Unlike software-only approaches that treat hardware as a black box, QuantumCredential uses **mathematics to guide both hardware architecture and software implementation**, achieving results that managed runtimes cannot match.

This documentation covers the complete system: from quantum physics (avalanche noise in Zener diodes) through analog circuit design, digital sampling, mathematical combination, native compilation, and cryptographic validation.

---

## The Abstraction Ceiling Problem

Software engineers are trained to work *above* hardware abstractions. We call `RandomNumberGenerator.GetBytes()` and trust that something reasonable happens. This works for most applications, but creates a fundamental problem for security-critical systems:

> **The abstraction ceiling becomes an abstraction floor for adversaries.**

When you cannot see below your abstraction layer, you cannot verify what happens there. An attacker who understands the hardware, the entropy sources, and the timing characteristics has an advantage you cannot counter from above.

QuantumCredential inverts this relationship. By co-designing hardware and software with shared mathematical foundations, we create systems where:

- Security guarantees are **provable**, not assumed
- Entropy quality is **measured**, not trusted
- Attack resistance is **architectural**, not hoped-for

This documentation is written for software engineers who want to understand what exists below their usual abstraction layer, and why it matters.

---

## Document Categories

### M-xx: Mathematics & Theory

The theoretical foundation. XOR's entropy-preserving properties are not heuristics; they are theorems. Understanding the mathematics reveals what's *possible* and guides both hardware and software design.

| Document | Description |
|----------|-------------|
| [M-01-XOR-Entropy-Theorem](Mathematics/M-01-XOR-Entropy-Theorem.md) | The mathematical proof that XOR preserves entropy |
| [M-02-Bias-Reduction-Analysis](Mathematics/M-02-Bias-Reduction-Analysis.md) | Derivation of ε → 8ε⁴ for four-channel combination |
| [M-03-Independence-Requirements](Mathematics/M-03-Independence-Requirements.md) | Why statistical independence matters and how to achieve it |

### H-xx: Hardware Design

Physical implementation guided by mathematical requirements. Each hardware decision traces back to a mathematical necessity.

| Document | Description |
|----------|-------------|
| [H-01-YoshiPi-Platform](Hardware/H-01-YoshiPi-Platform.md) | Raspberry Pi Zero 2 W on YoshiPi carrier board |
| [H-02-Avalanche-Circuit](Hardware/H-02-Avalanche-Circuit.md) | Four-channel Zener avalanche noise generator |
| [H-03-Channel-Independence](Hardware/H-03-Channel-Independence.md) | Electrical isolation for statistical independence |
| [H-04-Per-Channel-Tuning](Hardware/H-04-Per-Channel-Tuning.md) | Bias current, filtering, ADC timing optimization |

### E-xx: Entropy Pipeline

The flow from analog quantum noise to digital bytes. Each stage preserves or improves entropy quality.

| Document | Description |
|----------|-------------|
| [E-01-ADC-Sampling](Entropy/E-01-ADC-Sampling.md) | MCP3004 SPI ADC via Linux IIO subsystem |
| [E-02-Epsilon-Evaluation](Entropy/E-02-Epsilon-Evaluation.md) | Real-time per-channel bias measurement |
| [E-03-XOR-Tree-Implementation](Entropy/E-03-XOR-Tree-Implementation.md) | Parallel tree structure for bias reduction |
| [E-04-Parallel-Architecture](Entropy/E-04-Parallel-Architecture.md) | Logical parallelism in entropy generation |

### V-xx: Validation & Cryptography

Proving that the entropy meets cryptographic requirements. Testing validates; mathematics guarantees.

| Document | Description |
|----------|-------------|
| [V-01-NIST-SP800-22](Validation/V-01-NIST-SP800-22.md) | Statistical test suite for randomness |
| [V-02-TRNG-Certification](Validation/V-02-TRNG-Certification.md) | Cryptographic validity proof methodology |
| [V-03-EM-Attack-Resistance](Validation/V-03-EM-Attack-Resistance.md) | Defense against coherent electromagnetic interference |
| [V-04-Fault-Tolerance](Validation/V-04-Fault-Tolerance.md) | Graceful degradation under channel failure |

### C-xx: Compilation & MLIR

Native compilation that preserves mathematical guarantees through to machine code.

| Document | Description |
|----------|-------------|
| [C-01-SCF-Parallel-Pattern](Compilation/C-01-SCF-Parallel-Pattern.md) | MLIR scf.parallel with XOR reduction |
| [C-02-Zero-Copy-Pipeline](Compilation/C-02-Zero-Copy-Pipeline.md) | Eliminating managed runtime security liabilities |
| [C-03-Platform-Bindings-ADC](Compilation/C-03-Platform-Bindings-ADC.md) | Platform.Bindings.ADC abstraction |

### D-xx: Demo Application

The YoshiPi demonstration that visualizes entropy quality in real-time.

| Document | Description |
|----------|-------------|
| [D-01-Demo-Strategy](Demo/D-01-Demo-Strategy.md) | Overall demonstration approach |
| [D-02-Touch-Display-UI](Demo/D-02-Touch-Display-UI.md) | WREN-stack UI on touchscreen |
| [D-03-Entropy-Visualization](Demo/D-03-Entropy-Visualization.md) | Real-time epsilon and entropy display |

### P-xx: Performance

Quantifying the advantage of native compilation over managed runtimes.

| Document | Description |
|----------|-------------|
| [P-01-Python-Baseline](Performance/P-01-Python-Baseline.md) | Managed runtime performance ceiling |
| [P-02-Native-Projections](Performance/P-02-Native-Projections.md) | Target performance for native compilation |

### L-xx: Legal & Intellectual Property

Patent portfolio and technology foundations.

| Document | Description |
|----------|-------------|
| [L-01-Patent-Portfolio](Legal/L-01-Patent-Portfolio.md) | Pending patent applications |
| [L-02-Technology-Foundations](Legal/L-02-Technology-Foundations.md) | Core technology pillars |

### Phase-Specific Documentation

#### Phase 1: YoshiPi (Linux)

| Document | Description |
|----------|-------------|
| [PH1-01-Linux-Symmetry](Phase1-YoshiPi/PH1-01-Linux-Symmetry.md) | Same compiler targets both YoshiPi and desktop |
| [PH1-02-IIO-ADC-Bindings](Phase1-YoshiPi/PH1-02-IIO-ADC-Bindings.md) | Linux Industrial I/O subsystem integration |

#### Phase 2: Embedded (Bare Metal)

**Primary Target: Renesas RA6M5**

| Document | Description |
|----------|-------------|
| [PH2-00-Embedded-Strategy](Phase2-Embedded/PH2-00-Embedded-Strategy.md) | Platform comparison and strategy |
| [PH2-01-RA6M5-Platform](Phase2-Embedded/PH2-01-RA6M5-Platform.md) | EK-RA6M5 evaluation kit details |
| [PH2-02-RA6M5-Security](Phase2-Embedded/PH2-02-RA6M5-Security.md) | TrustZone and Secure Crypto Engine |

**Secondary Target: STM32L5** (on hold)

| Document | Description |
|----------|-------------|
| [STM32L5 Documentation](Phase2-Embedded/STM32L5/) | Preserved for future reference |

---

## The Mathematics-First Flow

```
Mathematics (M-xx)
    │ proves requirements for
    ▼
Hardware Design (H-xx)
    │ provides physical inputs to
    ▼
Entropy Pipeline (E-xx)
    │ produces bytes validated by
    ▼
Cryptographic Proofs (V-xx)
    │ guarantees implemented via
    ▼
Native Compilation (C-xx)
    │ demonstrated in
    ▼
Demo Application (D-xx)
```

Every downstream decision traces back to mathematical foundations. This is documentation of a derivation, not merely an implementation.

---

## Key Insight: Why Four Channels?

The four-channel architecture isn't arbitrary. It emerges from mathematics:

1. **XOR Theorem**: If either input to XOR is random, the output is random
2. **Bias Reduction**: Two channels reduce bias from ε to 2ε²
3. **Tree Structure**: Four channels in a tree reduce bias to 8ε⁴
4. **Practical Impact**: 5% per-channel bias → 0.005% output bias (1000× improvement)

The hardware (four independent avalanche diodes) implements a mathematical requirement. The software (parallel XOR tree) expresses the same mathematics. They are designed together because they implement the same theorem.

---

## Target Platforms

| Platform | Phase | Purpose | Entropy Source |
|----------|-------|---------|----------------|
| **YoshiPi** | 1 | Development/Demo | 4-channel avalanche via MCP3004 ADC |
| **Desktop Linux** | 1 | Compiler symmetry | Same code, simulated entropy |
| **Renesas RA6M5** | 2 | Production (primary) | Internal 12-bit ADC, TrustZone |
| **STM32L5** | 2 | Alternative (on hold) | Direct ADC, no OS |

The same Firefly-compiled F# code runs on all platforms. Only `Platform.Bindings` differ.

---

## Patent Portfolio

| Application | Title | Coverage |
|-------------|-------|----------|
| US 63/780,027 | Air-Gapped Dual Network Architecture for QRNG... | Entropy harvesting, air-gapped distribution |
| US 63/780,055 | Quantum-Resistant Hardware Security Module... | HSM with decentralized identity |
| US 63/786,247 | [Title TBD] | [Coverage TBD] |

---

## Reading Order for Software Engineers

If you are coming from a pure software background:

1. **Start with M-01** (XOR Entropy Theorem) - understand *why* this works
2. **Read M-02** (Bias Reduction) - see the mathematics that guides hardware
3. **Skim H-02** (Avalanche Circuit) - see how physics becomes electronics
4. **Study E-02** (Epsilon Evaluation) - the core demo capability
5. **Read C-02** (Zero-Copy Pipeline) - why managed runtimes cannot do this
6. **Review V-03** (EM Attack Resistance) - defense through architecture

This path takes you from mathematical foundations through hardware implementation to security guarantees: the journey from abstraction ceiling to abstraction floor.
