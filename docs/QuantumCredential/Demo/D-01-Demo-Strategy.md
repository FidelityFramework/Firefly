# YoshiPi Demo Strategy: Linux Symmetry

> **The Key Insight**: Both the YoshiPi device and the desktop Keystation are Linux systems. This means the same Firefly compilation pipeline, the same native F# APIs, and the same WebView UI approach work on both ends.

---

## Executive Summary

The QuantumCredential demo consists of two components:

| Component | Platform | Role |
|-----------|----------|------|
| **Credential Generator** | YoshiPi (ARM64 Linux) | Sample entropy, generate PQC credentials |
| **Keystation** | Desktop (x86_64 Linux) | Receive, verify, display credentials |

Both run:
- Linux kernel
- Firefly-compiled native binaries
- WebView-based UI (via system webview)
- Same FNCS APIs

The "it's just Linux" symmetry dramatically reduces complexity compared to bare-metal embedded approaches.

---

## Architecture: Symmetric Linux Stacks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YoshiPi (Credential Generator)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  F# Application (Firefly-compiled to ARM64 Linux ELF)             │ │
│  │                                                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │ │
│  │  │ Entropy     │  │ PQC Engine  │  │ Credential Generator    │   │ │
│  │  │ Sampling    │  │ (ML-KEM,    │  │ • Key generation        │   │ │
│  │  │ • ADC read  │─►│  ML-DSA)    │─►│ • Signing               │   │ │
│  │  │ • Condition │  │             │  │ • Framing for transfer  │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │ │
│  │         │                                      │                  │ │
│  │         ▼                                      ▼                  │ │
│  │  ┌─────────────┐                    ┌─────────────────────────┐   │ │
│  │  │ FNCS        │                    │ WebView Monitor UI      │   │ │
│  │  │ Platform.   │                    │ • Status display        │   │ │
│  │  │ Bindings    │                    │ • Entropy visualization │   │ │
│  │  └─────────────┘                    │ • Credential preview    │   │ │
│  │         │                           └─────────────────────────┘   │ │
│  └─────────│─────────────────────────────────────│───────────────────┘ │
│            │                                     │                     │
│            ▼                                     ▼                     │
│  ┌─────────────────────┐              ┌─────────────────────────────┐ │
│  │ Linux Kernel        │              │ WebKitGTK (ARM64)           │ │
│  │ • /dev/gpiochip0    │              │ (or framebuffer alternative)│ │
│  │ • /sys/bus/iio/...  │              └─────────────────────────────┘ │
│  └─────────────────────┘                                              │
│            │                                                          │
│            ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Hardware: Pi Zero 2 W + YoshiPi Carrier                          │ │
│  │ • Avalanche circuit → Analog input                               │ │
│  │ • Touchscreen display                                            │ │
│  │ • USB/WiFi for credential transfer                               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

                              │ USB/WiFi
                              │ (BAREWire-framed credentials)
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                    Desktop Keystation                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  F# Application (Firefly-compiled to x86_64 Linux ELF)            │ │
│  │                                                                   │ │
│  │  ┌─────────────────┐  ┌─────────────────────────────────────────┐│ │
│  │  │ Credential      │  │ WebView UI (Partas.Solid)               ││ │
│  │  │ Receiver        │  │ • Credential display                    ││ │
│  │  │ • USB/network   │─►│ • Verification status                   ││ │
│  │  │ • Verification  │  │ • Storage management                    ││ │
│  │  └─────────────────┘  └─────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                   │
│                                     ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ WebKitGTK (x86_64)                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Works

### Same Compiler, Same Libraries, Same Patterns

| Aspect | YoshiPi | Desktop | Shared |
|--------|---------|---------|--------|
| **Target** | Linux/ARM64 | Linux/x86_64 | Linux ABI |
| **Syscalls** | ARM64 syscall convention | x86_64 syscall convention | Same syscall numbers |
| **FNCS** | Platform.Bindings | Platform.Bindings | Same API |
| **WebView** | WebKitGTK (ARM) | WebKitGTK (x86_64) | Same library |
| **IPC** | BAREWire schemas | BAREWire schemas | Same types |

### What Firefly Already Does

Firefly already compiles to Linux/x86_64 (the console samples work). The delta to Linux/ARM64:

1. **LLVM target triple**: `aarch64-unknown-linux-gnu` instead of `x86_64-unknown-linux-gnu`
2. **Syscall convention**: Different registers, but same syscall numbers for most operations
3. **That's it**: Same MLIR → LLVM → native pipeline

### What Needs to Be Added

| Component | Work Required | Complexity |
|-----------|---------------|------------|
| ARM64 LLVM target | Configure target triple | LOW |
| GPIO Platform.Bindings | `openDevice`, `ioctl` for GPIO | LOW |
| IIO/ADC Platform.Bindings | Read from `/sys/bus/iio/...` | LOW |
| WebView bindings | Same as desktop (already designed) | MEDIUM |
| Cross-compilation setup | Toolchain configuration | LOW |

---

## Hardware Integration: YoshiPi Carrier

### Analog Inputs for Avalanche Circuit

The YoshiPi carrier provides 4x 10-bit analog inputs. The avalanche circuit connects to one of these.

**Linux ADC Access Pattern:**
```
/sys/bus/iio/devices/iio:device0/
├── in_voltage0_raw     # Read raw ADC value
├── in_voltage_scale    # Scale factor
└── sampling_frequency  # Sample rate configuration
```

**Platform Binding:**
```fsharp
module Platform.Bindings =
    /// Read raw ADC value from IIO device
    let readADC (devicePath: string) : uint16 =
        Unchecked.defaultof<uint16>

    /// Configure ADC sampling rate
    let setADCSampleRate (devicePath: string) (hz: int) : int =
        Unchecked.defaultof<int>
```

### GPIO for Status Indicators

**Linux GPIO Access Pattern:**
```
/dev/gpiochip0          # Character device for GPIO
```

Via `ioctl` calls (or libgpiod):
- `GPIO_GET_LINEINFO_IOCTL` - Query line info
- `GPIO_GET_LINEHANDLE_IOCTL` - Acquire line handle
- `GPIOHANDLE_SET_LINE_VALUES_IOCTL` - Set output values

**Platform Binding:**
```fsharp
module Platform.Bindings =
    /// Open GPIO chip device
    let openGPIOChip (chipPath: string) : int =
        Unchecked.defaultof<int>

    /// Set GPIO line value
    let setGPIOLine (fd: int) (line: int) (value: int) : int =
        Unchecked.defaultof<int>
```

### Touchscreen Display

The YoshiPi has an integrated touchscreen. Options:

1. **WebKitGTK** - Full WebView, same as desktop (preferred)
2. **Framebuffer** - Direct `/dev/fb0` rendering (fallback)
3. **DRM/KMS** - Modern Linux graphics (if WebKit unavailable)

**Recommendation**: Use WebKitGTK if available on Pi Zero 2 W. The UI code is then 100% shared with desktop.

---

## Demo Flow

### Phase 1: Entropy Sampling

```fsharp
module Entropy =
    let sampleAvalanche (samples: int) : NativeArray<uint16> =
        let adcPath = "/sys/bus/iio/devices/iio:device0/in_voltage0_raw"n
        let buffer = NativeArray.stackalloc<uint16> samples

        for i in 0 .. samples - 1 do
            buffer.[i] <- Platform.Bindings.readADC adcPath

        buffer

    let condition (raw: NativeArray<uint16>) : NativeArray<byte> =
        // SHAKE-256 conditioning
        Crypto.SHAKE256.squeeze raw 32
```

### Phase 2: Credential Generation

```fsharp
module Credential =
    let generate (entropy: NativeArray<byte>) : QuantumCredential =
        // Seed PQC RNG with hardware entropy
        let rng = PQC.seedRNG entropy

        // Generate ML-KEM keypair
        let kemPublic, kemPrivate = PQC.MLKEM.keygen rng

        // Generate ML-DSA keypair
        let dsaPublic, dsaPrivate = PQC.MLDSA.keygen rng

        // Create and sign credential
        let cred = {
            KEMPublicKey = kemPublic
            DSAPublicKey = dsaPublic
            Timestamp = Time.now()
            Nonce = entropy.[0..15]
        }

        let signature = PQC.MLDSA.sign dsaPrivate (BAREWire.encode cred)

        { Credential = cred; Signature = signature }
```

### Phase 3: UI Display

```fsharp
module MonitorUI =
    let updateStatus (webview: nativeint) (status: Status) =
        let json = sprintf """{"entropy": %d, "credentials": %d, "state": "%s"}"""
                          status.EntropyBits
                          status.CredentialsGenerated
                          status.CurrentState

        Webview.eval webview $"updateStatus({json})"
```

### Phase 4: Transfer to Desktop

```fsharp
module Transfer =
    let sendCredential (cred: QuantumCredential) =
        // Encode with BAREWire
        let encoded = BAREWire.encode credentialSchema cred

        // Send via USB serial or network
        let fd = Platform.Bindings.openDevice "/dev/ttyGS0"n  // USB gadget
        Platform.Bindings.write fd encoded.Pointer encoded.Length
```

---

## Build Configuration

### YoshiPi .fidproj

```toml
[package]
name = "QuantumCredentialGenerator"

[compilation]
memory_model = "stack_only"
target = "linux-arm64"      # Cross-compile to ARM64

[desktop]
frontend = "src/MonitorUI"   # Partas.Solid UI
backend = "src/Generator"    # Entropy + PQC + Credential
embed_assets = true

[dependencies]
# Uses FNCS intrinsics (no separate dependency)

[build]
output = "qc-generator"
output_kind = "desktop"      # Uses WebView
```

### Desktop Keystation .fidproj

```toml
[package]
name = "Keystation"

[compilation]
memory_model = "stack_only"
target = "linux-x86_64"

[desktop]
frontend = "src/KeystationUI"
backend = "src/Receiver"
embed_assets = true

[dependencies]
# Uses FNCS intrinsics (no separate dependency)

[build]
output = "keystation"
output_kind = "desktop"
```

---

## Comparison: YoshiPi vs Phase 2 (STM32L5)

| Aspect | YoshiPi (Phase 1) | STM32L5 (Phase 2) |
|--------|-------------------|-------------------|
| **OS** | Linux (Debian) | NuttX RTOS or bare-metal |
| **Bindings** | Same as desktop | Farscape-generated CMSIS |
| **UI** | WebView (shared code) | Custom or none |
| **Risk** | LOW | HIGH |
| **Demo value** | HIGH | HIGHER (if working) |
| **Architecture depth** | Shallow | Deep (quotations, active patterns) |

**Phase 1 (YoshiPi)** proves the concept with minimal new infrastructure.

**Phase 2 (STM32L5)** demonstrates the full architectural vision with Farscape, quotation-based memory, and bare-metal compilation.

---

## Success Criteria

### Demo Day Requirements

- [ ] F# code compiles to ARM64 Linux binary
- [ ] Binary runs on YoshiPi, samples ADC
- [ ] Entropy conditioning produces quality randomness
- [ ] PQC credential generation completes
- [ ] Monitor UI displays status on YoshiPi touchscreen
- [ ] Credential transfers to desktop Keystation
- [ ] Keystation verifies and displays credential

### Narrative Points

1. **"Same F# code, same compiler"** - Firefly targets both platforms
2. **"Real hardware entropy"** - Avalanche circuit provides quantum randomness
3. **"Post-quantum security"** - ML-KEM and ML-DSA algorithms
4. **"Native performance"** - No runtime, no GC, direct hardware access
5. **"Unified UI model"** - WebView works on device and desktop

---

## Cross-References

### This Folder
- [02_YoshiPi_Architecture](./02_YoshiPi_Architecture.md) - Hardware details, GPIO, ADC
- [03_MLIR_Dialect_Strategy](./03_MLIR_Dialect_Strategy.md) - Compilation path: standard dialects for demo
- [04_Linux_Hardware_Bindings](./04_Linux_Hardware_Bindings.md) - Platform.Bindings patterns for Linux hardware
- [05_PostQuantum_Architecture](./05_PostQuantum_Architecture.md) - PQC algorithms, entropy design
- [06_January_Roadmap](./06_January_Roadmap.md) - Timeline and sprints

### Phase 2 (STM32L5 Path)
- [Phase2_STM32L5/](./Phase2_STM32L5/) - Future bare-metal path documentation

### Architecture Documents
- [WebView_Desktop_Architecture](../WebView_Desktop_Architecture.md) - WebView stack (applies to both!)
- [WebView_Build_Integration](../WebView_Build_Integration.md) - Firefly build orchestration
