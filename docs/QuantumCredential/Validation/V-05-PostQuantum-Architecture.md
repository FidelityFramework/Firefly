# Post-Quantum Credential Architecture: Hardware Entropy and NIST PQC Bindings

This document outlines the architecture for generating post-quantum cryptographic credentials using hardware true random number generators (TRNG) based on zener avalanche noise, with Farscape-generated bindings to NIST-approved PQC reference implementations.

## The Security Mission

QuantumCredential and Keystation are air-gapped devices that communicate exclusively via infrared transceivers. This physical isolation, combined with hardware-sourced entropy and post-quantum cryptography, creates a credential generation and management system resistant to both network-based attacks and future quantum computing threats.

The core insight: **true randomness from physics, processed by mathematically-proven algorithms, transmitted over an air-gapped optical channel**.

## Hardware Entropy: Quad-Channel Zener Avalanche Noise

Both devices incorporate analog circuits based on zener diode avalanche breakdown, a quantum mechanical phenomenon that produces true random noise. The YoshiPi demo uses a quad-channel design with four independent avalanche sources feeding four ADC inputs. This is fundamentally different from:

- **PRNGs**: Deterministic algorithms that expand a seed (predictable given the seed)
- **Clock jitter**: Timing variations that can be influenced by environmental factors
- **Thermal noise**: Can be biased by temperature manipulation

Zener avalanche noise originates from quantum tunneling effects in the semiconductor junction, providing entropy that is:
- Physically unpredictable (quantum mechanical origin)
- Resistant to environmental manipulation
- Continuously available (no "entropy exhaustion")

### Quad-Channel Advantage

The quad avalanche circuit design provides multiple benefits:

| Benefit | Description |
|---------|-------------|
| **Bias Reduction** | Independent sources cancel systematic biases |
| **Autocorrelation Breaking** | Interleaving bits disrupts temporal patterns |
| **Redundancy** | Circuit drift in one channel doesn't compromise quality |
| **Parallel Validation** | Health tests can run concurrently on each channel |

### Data Path: Analog to Digital

| Device | ADC Source | Resolution | Channels | Notes |
|--------|-----------|------------|----------|-------|
| QuantumCredential (YoshiPi) | Carrier ADC | 10-bit | 4 | Quad avalanche to 4 ADC inputs |
| QuantumCredential (STM32L5) | Native ADC | 12-bit | 4+ | Up to 16 external channels |
| Keystation (Sweet Potato) | Piggyback ADC | 12-bit+ | 4 | I2C/SPI (ADS1115 or similar) |

Each device samples four independent noise sources, producing parallel entropy streams that must be validated and combined before use in cryptographic operations.

### Parallel Sampling

The quad-channel architecture maps directly to the quad-core processor. Sampling four independent noise sources is a pure parallel operation ideal for `scf.parallel`:

- **Referentially transparent**: Each channel's sample is independent
- **No cross-channel dependencies**: No operation needs another's result
- **Pure data transformation**: Voltage reading to entropy bits
- **Perfect core mapping**: 4 ADC channels to 4 CPU cores

The Firefly compiler emits `scf.parallel` for this pattern:

```fsharp
// Conceptual: quad-channel parallel sampling
let sampleAllChannels () =
    // scf.parallel: executes simultaneously on 4 cores
    let samples = [| 0..3 |] |> Array.Parallel.map (fun ch ->
        readAdcChannel ch
    )
    interleaveEntropy samples
```

See [03_MLIR_Dialect_Strategy.md](./03_MLIR_Dialect_Strategy.md) for details on the standard MLIR dialect approach used for the demo.

### Entropy Validation Pipeline

Raw ADC samples are not directly usable as cryptographic entropy. The validation pipeline processes all four channels:

1. **Per-Channel Health Tests**: NIST SP 800-90B tests (repetition count, adaptive proportion) run in parallel on each channel
2. **Interleaving**: Combine bits from all four sources to break autocorrelation
   ```
   Interleaved: a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ ...
   ```
3. **Conditioning**: SHAKE-256 extraction to produce uniform output from interleaved input
4. **Accumulation**: Pool conditioned entropy until sufficient for key generation
5. **Rate Limiting**: Ensure adequate entropy per cryptographic operation

The interleaving and conditioning steps are also pure operations suitable for parallel execution via `scf.parallel`, allowing the entire entropy pipeline to benefit from multi-core processing.

This pipeline is implemented in Fidelity F#, compiled by Firefly, with the statistical tests and hash functions either implemented natively or bound via Farscape.

## NIST Post-Quantum Cryptography Standards

On August 13, 2024, NIST released the first three finalized Post-Quantum Cryptography standards:

| Standard | Algorithm | Original Name | Purpose |
|----------|-----------|---------------|---------|
| FIPS 203 | ML-KEM | CRYSTALS-Kyber | Key Encapsulation Mechanism |
| FIPS 204 | ML-DSA | CRYSTALS-Dilithium | Digital Signatures |
| FIPS 205 | SLH-DSA | SPHINCS+ | Stateless Hash-Based Signatures |

For credential generation, the primary algorithms are:
- **ML-KEM** (Kyber): For key agreement / encryption key exchange
- **ML-DSA** (Dilithium): For signing certificates and assertions

## C Reference Implementations

Several high-quality C implementations are available for Farscape binding:

### PQ-CRYSTALS Reference Implementations

The algorithm authors maintain canonical implementations:

- **Kyber**: https://github.com/pq-crystals/kyber
  - `ref/`: Clean reference implementation (prioritizes clarity)
  - `avx2/`: x86-64 optimized with AVX2 instructions

- **Dilithium**: https://github.com/pq-crystals/dilithium
  - `ref/`: Clean reference implementation
  - `avx2/`: x86-64 optimized
  - Supports hedged (randomized) or deterministic signing modes

### liboqs (Open Quantum Safe)

A unified C library providing a common API across multiple PQC algorithms:

- **Repository**: https://github.com/open-quantum-safe/liboqs
- **API Documentation**: https://openquantumsafe.org/liboqs/api/
- **Current Version**: 0.13.0 (April 2025)

liboqs advantages:
- Consistent API across all algorithms
- Formally verified implementations (ML-KEM via mlkem-native)
- Portable C, AVX2, and AArch64 variants
- Active maintenance with NIST standard tracking

### pqm4 (ARM Cortex-M4 Optimized)

For the STM32L5 (Cortex-M33, compatible with M4 code):

- **Repository**: https://github.com/mupq/pqm4
- Implementations optimized for constrained microcontrollers
- Includes M4-specific assembly optimizations
- Benchmarked on STM32 Nucleo boards

Implementation variants in pqm4:
- `clean`: Reference from PQClean
- `ref`: Original NIST submission reference
- `opt`: Optimized portable C
- `m4`: Cortex-M4 assembly optimizations
- `m4f`: Cortex-M4F floating-point register optimizations

## Platform Bindings for PQC Libraries

Fidelity uses the Platform.Bindings pattern (BCL-free) rather than DllImport. FNCS defines function signatures with `Unchecked.defaultof<T>` placeholders, and Alex provides platform-specific MLIR emission that links against the native PQC libraries.

### Layer 1: Platform Bindings (BCL-Free)

Farscape generates Platform.Bindings module functions from C headers:

```fsharp
// Farscape-generated from liboqs or pq-crystals headers
module Platform.Bindings.Kyber =
    /// Generate Kyber-512 keypair
    let keypair (pk: nativeint) (sk: nativeint) : int =
        Unchecked.defaultof<int>

    /// Encapsulate shared secret
    let encapsulate (ct: nativeint) (ss: nativeint) (pk: nativeint) : int =
        Unchecked.defaultof<int>

    /// Decapsulate shared secret
    let decapsulate (ss: nativeint) (ct: nativeint) (sk: nativeint) : int =
        Unchecked.defaultof<int>

module Platform.Bindings.Dilithium =
    /// Generate Dilithium-2 keypair
    let keypair (pk: nativeint) (sk: nativeint) : int =
        Unchecked.defaultof<int>

    /// Sign message
    let sign (sig: nativeint) (siglen: nativeint) (msg: nativeint) (msglen: int) (sk: nativeint) : int =
        Unchecked.defaultof<int>

    /// Verify signature
    let verify (sig: nativeint) (siglen: int) (msg: nativeint) (msglen: int) (pk: nativeint) : int =
        Unchecked.defaultof<int>
```

Alex recognizes calls to `Platform.Bindings.Kyber.*` and emits LLVM declarations that link against the PQC library (liboqs, pq-crystals, or pqm4 depending on target).

### Layer 2: Type Definitions

Safe wrappers with proper sizing (using BAREWire FixedArray):

```fsharp
module Fidelity.PQC.Types

/// Kyber-512 public key (800 bytes)
[<Struct>]
type Kyber512PublicKey = { Data: FixedArray<byte, 800> }

/// Kyber-512 secret key (1632 bytes)
[<Struct>]
type Kyber512SecretKey = { Data: FixedArray<byte, 1632> }

/// Kyber-512 ciphertext (768 bytes)
[<Struct>]
type Kyber512Ciphertext = { Data: FixedArray<byte, 768> }

/// Shared secret (32 bytes)
[<Struct>]
type SharedSecret = { Data: FixedArray<byte, 32> }

/// Dilithium-2 signature (2420 bytes)
[<Struct>]
type Dilithium2Signature = { Data: FixedArray<byte, 2420> }
```

### Layer 3: Functional Wrappers

Idiomatic F# API with proper entropy integration:

```fsharp
module Fidelity.PQC.Kyber

open Platform.Bindings

/// Generate a Kyber-512 keypair using hardware entropy
let generateKeypair (entropy: EntropyPool) : Result<Kyber512PublicKey * Kyber512SecretKey, PQCError> =
    if entropy.Available < 64 then
        Error InsufficientEntropy
    else
        let pk = Kyber512PublicKey.Allocate()
        let sk = Kyber512SecretKey.Allocate()
        // Seed the internal RNG with hardware entropy
        seedRng entropy 64
        match Kyber.keypair(pk.Pointer, sk.Pointer) with
        | 0 -> Ok (pk, sk)
        | e -> Error (KeyGenFailed e)

/// Encapsulate a shared secret for a recipient's public key
let encapsulate (pk: Kyber512PublicKey) (entropy: EntropyPool)
    : Result<Kyber512Ciphertext * SharedSecret, PQCError> =
    // ...

/// Decapsulate a shared secret using the secret key
let decapsulate (sk: Kyber512SecretKey) (ct: Kyber512Ciphertext)
    : Result<SharedSecret, PQCError> =
    // ...
```

## Air-Gapped Communication: Infrared Protocol

Both devices communicate via infrared transceivers, providing:

- **Physical air gap**: No network stack, no radio emissions
- **Line-of-sight requirement**: Prevents remote interception
- **Intentional communication**: User must physically align devices

### IR Protocol Considerations

The IR channel must handle:

1. **Framing**: Packet boundaries in the optical stream
2. **Error Detection**: CRC or similar for noisy optical channel
3. **Flow Control**: Half-duplex coordination
4. **Payload Types**: Key exchange, signed assertions, certificates

A potential protocol stack:

```
┌─────────────────────────────────────────┐
│ Application: Credential Exchange        │
│   - Certificate requests/responses      │
│   - Signed assertions                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Security: ML-KEM + ML-DSA               │
│   - Key encapsulation for encryption    │
│   - Digital signatures for authenttic   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Framing: COBS or similar                │
│   - Packet delimiting                   │
│   - CRC-16 error detection              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Physical: IR Transceiver                │
│   - 38kHz carrier (standard IR)         │
│   - Bit-banged or UART-driven           │
└─────────────────────────────────────────┘
```

## Platform-Specific Implementation

### QuantumCredential (STM32L5 Nucleo)

True unikernel deployment as described in "Hardware Showcase Roadmap":

- **Entropy**: Native 12-bit ADC sampling zener noise
- **PQC Library**: pqm4 Cortex-M optimized implementations
- **IR**: GPIO bit-bang or timer-based PWM for 38kHz carrier
- **Target**: `thumbv8m.main-none-eabi` (Cortex-M33)

Memory constraints require careful implementation:
- Kyber-512: ~2.5KB for keypair
- Dilithium-2: ~2.5KB public key, ~4KB secret key
- Stack usage during operations: 10-20KB typical

### Keystation (Sweet Potato + ADC)

Linux userspace application:

- **Entropy**: Piggyback ADC (I2C/SPI) sampling zener noise
- **PQC Library**: liboqs or pq-crystals reference
- **IR**: GPIO via Linux sysfs or gpiod
- **Target**: `aarch64-unknown-linux-gnu`

Additional responsibilities:
- Touchscreen UI for credential management
- Secure storage of generated credentials
- User-facing certificate inspection

## Credential Format

Generated credentials should be compatible with emerging PQC certificate standards:

- **X.509 with PQC**: IETF drafts for ML-DSA in X.509
- **CBOR-based**: Compact binary format for constrained devices
- **Custom**: Application-specific if interoperability not required

Example credential structure:

```
Credential {
    Version: 1
    Subject: device-specific identifier
    PublicKey: ML-KEM-512 or ML-DSA-44 public key
    ValidFrom: timestamp
    ValidUntil: timestamp
    Issuer: self-signed or Keystation-signed
    Signature: ML-DSA signature
    EntropyAttestation: hash of entropy source samples (optional)
}
```

## Implementation Phases

### Phase 1: Entropy Subsystem

- Implement ADC sampling for both platforms
- Port NIST SP 800-90B health tests
- Implement SHA-3 conditioning
- Validate entropy quality with external tools (ent, dieharder)

### Phase 2: PQC Bindings via Farscape

- Parse pq-crystals headers for Layer 1 generation
- Define F# type wrappers (Layer 2)
- Implement functional API with entropy integration (Layer 3)
- Test against known answer tests (KATs)

### Phase 3: IR Communication

- Implement physical layer (GPIO/timer for STM32, gpiod for Linux)
- Define framing protocol with error detection
- Implement basic ping/pong for link verification

### Phase 4: Credential Generation

- Implement key generation with hardware entropy
- Define credential format (X.509 or custom)
- Implement signing and verification
- Cross-device credential exchange over IR

### Phase 5: Integration and UI

- Keystation touchscreen interface for credential management
- QuantumCredential status indicators (LEDs)
- End-to-end credential generation and transfer demo

## Security Considerations

### Entropy Quality

- Continuous health monitoring during operation
- Fail-safe: refuse to generate keys if entropy tests fail
- Consider entropy from multiple uncorrelated sources

### Side-Channel Protection

- pqm4 uses constant-time implementations where available
- Avoid branching on secret data
- Consider masking for high-security applications

### Key Storage

- STM32L5 TrustZone for secure key storage
- Linux: consider kernel keyring or TPM integration
- Never store secret keys in plain flash

### Physical Security

- Tamper detection (optional)
- Secure boot chain
- Memory zeroization on tamper or power-off

## References

### NIST Standards

- FIPS 203: ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)
- FIPS 204: ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
- FIPS 205: SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)
- SP 800-90B: Recommendation for the Entropy Sources Used for Random Bit Generation

### Reference Implementations

- PQ-CRYSTALS Kyber: https://github.com/pq-crystals/kyber
- PQ-CRYSTALS Dilithium: https://github.com/pq-crystals/dilithium
- liboqs: https://github.com/open-quantum-safe/liboqs
- pqm4: https://github.com/mupq/pqm4

### Related Firefly Documentation

- `docs/Hardware_Showcase_Roadmap.md` - Platform deployment models
- `docs/Farscape_GIR_Integration.md` - Binding generation architecture
- `docs/Demo_UI_Stretch_Goal.md` - Keystation UI plans

### SpeakEZ Blog Posts

- "The Farscape Bridge" - Binding architecture principles
- "Binding F# to C++ in Farscape" - Functional wrapper design
- "Memory Management by Choice" - Stack allocation for crypto buffers
