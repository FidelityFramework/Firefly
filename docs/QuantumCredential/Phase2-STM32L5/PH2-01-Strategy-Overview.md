# Integrated Demo Strategy: Desktop and Embedded Paths

> **Purpose**: This document synthesizes the architectural requirements for demonstrating the Fidelity framework across two target environments: desktop (WebView-based UI) and embedded (STM32L5 QuantumCredential).

---

## The "Cheat" Pattern: Pragmatic Shortcuts for Demo

Both demo paths employ strategic shortcuts. Understanding these shortcuts - what they enable and what they defer - is essential for planning.

### Desktop Path: Dynamic Webview Bindings

| Aspect | What the "Cheat" Provides | What It Defers |
|--------|---------------------------|----------------|
| **Mechanism** | Hand-write `Platform.Bindings.Webview` module, Alex generates library calls | Farscape-based automated binding generation |
| **Why it works** | Desktop OSes have mature dynamic linking infrastructure | - |
| **Binding surface** | ~15 webview functions (create, setHtml, bind, return, etc.) | Complex callback/closure handling |
| **Risk level** | LOW - well-understood pattern, limited scope | - |

### Embedded Path: RTOS Abstraction Layer

| Aspect | What the "Cheat" Provides | What It Defers |
|--------|---------------------------|----------------|
| **Mechanism** | NuttX RTOS provides POSIX layer; Alex generates syscalls | Bare-metal CMSIS bindings via Farscape |
| **Why it works** | NuttX exposes GPIO/ADC/USB as `/dev` character devices | The full quotation-based memory architecture |
| **Binding surface** | Standard POSIX: `open`, `read`, `write`, `ioctl`, `close` | Hardware-specific register layouts |
| **Risk level** | MEDIUM - requires NuttX port verification, unfamiliar territory | - |

---

## Architectural Comparison: Two Demo Paths

### Desktop WebView Demo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Desktop WebView Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frontend (Partas.Solid)        │  Backend (Firefly-compiled)          │
│  ┌───────────────────────┐      │  ┌────────────────────────────────┐  │
│  │ F# → Fable → SolidJS  │ IPC  │  │ F# → PSG → Alex → MLIR → bin  │  │
│  │ Reactive UI           │◄────►│  │ Application logic              │  │
│  └───────────────────────┘      │  │                                │  │
│                                 │  │ Platform.Bindings.Webview      │  │
│  Webview Runtime                │  │   ↓                            │  │
│  (WebKitGTK/WebView2)           │  │ Alex generates library calls   │  │
│                                 │  │   ↓                            │  │
│                                 │  │ Dynamic linking at runtime     │  │
│                                 │  └────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The "native" part is minimal - just library calls to the system webview. The complexity is in build orchestration (Fable + Vite + Firefly) and IPC (BAREWire-over-webview).

### Embedded QuantumCredential Demo (with RTOS)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Embedded QuantumCredential Architecture (RTOS)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Application (Firefly-compiled)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ F# → PSG → Alex → MLIR → ARM Cortex-M33 binary                   │   │
│  │                                                                   │   │
│  │ Platform.Bindings (POSIX-like)                                    │   │
│  │   open("/dev/adc0")   → ADC for entropy                           │   │
│  │   read(fd, buf, n)    → Sample zener noise                        │   │
│  │   write(usb_fd, ...)  → USB output                                │   │
│  │   ioctl(gpio_fd, ...) → GPIO control                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  NuttX RTOS Layer (30-50KB overhead)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ POSIX syscall interface                                          │   │
│  │ Character device drivers (/dev/adc0, /dev/gpio0, /dev/ttyACM0)   │   │
│  │ Scheduler (flat build - single task)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  Hardware (STM32L5 Cortex-M33)                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ADC (zener noise) │ GPIO (status) │ USB (output) │ RNG           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: NuttX acts as a HAL shim. We don't need Farscape to generate CMSIS bindings because NuttX abstracts the hardware into POSIX character devices. This is a significant simplification - at the cost of 30-50KB RTOS overhead.

### Embedded QuantumCredential Demo (Bare-Metal Future Path)

```
┌─────────────────────────────────────────────────────────────────────────┐
│          Embedded QuantumCredential Architecture (Bare-Metal)           │
│                       [FUTURE - Full Architecture]                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Application (Firefly-compiled with quotation-based memory)             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ F# → PSG (with MemoryConstraint) → Alex → MLIR → binary          │   │
│  │                                                                   │   │
│  │ Memory access carries quoted constraints:                         │   │
│  │   <@ { Region = Peripheral; Access = WriteOnly; Volatile } @>     │   │
│  │                                                                   │   │
│  │ Active patterns recognize hardware operations:                    │   │
│  │   | GPIOPinWrite (port, pin, value) →                             │   │
│  │       emitVolatileStore base + offset value                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  Farscape-Generated Plugin                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Expr<PeripheralDescriptor> - quoted hardware layouts             │   │
│  │ Active patterns - hardware operation recognition                 │   │
│  │ MemoryModel record - integration point                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  Hardware (direct register access)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ GPIOA: 0x48000000 + BSRR: 0x18 → volatile store                  │   │
│  │ ADC1:  0x42028000 + DR: 0x40 → volatile load                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The full bare-metal path requires the complete quotation-based memory architecture to be functional: Farscape generating quoted descriptors, BAREWire providing interpretation infrastructure, fsnative nanopasses handling constraint validation, and Alex emitting correct volatile memory access. This is NOT a shortcut.

---

## What Needs to Exist for Each Path

### Desktop WebView Demo (Milestone 2-5 from WebView plan)

| Component | Status | Required Work |
|-----------|--------|---------------|
| Fidelity.Platform.Webview | Not started | Add ~15 webview conduit functions |
| Fidelity.Platform Webview module | Not started | Create BCL-sympathetic wrapper API |
| Alex Bindings/Webview | Not started | Generate library calls (not syscalls) |
| Firefly build orchestration | Not started | Invoke Fable, Vite, embed HTML |
| IPC (BAREWire-over-webview) | Design exists | Implement base64 encoding bridge |

**Risk**: LOW - All pieces are well-understood extensions of existing patterns.

### Embedded Demo with RTOS (Pragmatic Path)

| Component | Status | Required Work |
|-----------|--------|---------------|
| NuttX port for STM32L5 | Exists, needs verification | Configure and test |
| Platform.Bindings.POSIX | Partially exists | Add open/close/ioctl |
| Alex Bindings for ARM | Partially exists | Extend for POSIX syscalls on NuttX |
| ADC driver config | NuttX provides | Enable CONFIG_STM32_ADC1 |
| USB CDC driver config | NuttX provides | Enable CONFIG_CDCACM |
| PQC library | External | Link liboqs or pqm4 |
| Entropy conditioning | Not started | Implement SHAKE/SHA-3 in F# |

**Risk**: MEDIUM - Unfamiliar territory, requires hardware testing.

### Embedded Demo Bare-Metal (Full Architecture Path)

| Component | Status | Required Work |
|-----------|--------|---------------|
| fsnative-spec | Early stage | Define native types, memory layout |
| fsnative (FNCS) | Planned | Fork FCS, implement native types |
| BAREWire quotation infrastructure | Design exists | Implement interpretation framework |
| Farscape quotation output | Design exists | Generate Expr<PeripheralDescriptor> |
| Farscape active pattern generation | Design exists | Generate recognition patterns |
| fsnative PSG extensions | Not started | Add MemoryConstraint field |
| fsnative constraint nanopasses | Not started | Attachment, validation, classification |
| Alex operation class emission | Not started | VolatileStore, VolatileLoad, etc. |
| Baker quotation handling | Not started | Flow quotations through typed tree |

**Risk**: HIGH - Significant architectural work required across multiple components.

---

## Recommended Demo Strategy

### Phase 1: Desktop WebView Demo (Lower Risk)

**Goal**: Demonstrate "F# compiles to native desktop app with reactive UI"

**Narrative**: "We compile Partas.Solid F# components to JavaScript for the UI, and the F# backend to native code. A single `firefly build` command orchestrates everything into one executable."

**Timeline**: Can be achieved with Milestones 2-5 from WebView plan.

### Phase 2: Embedded Demo with RTOS (Medium Risk)

**Goal**: Demonstrate "F# compiles to ARM microcontroller with hardware access"

**Narrative**: "The same Fidelity compiler that builds desktop apps also targets embedded. We run F# on an STM32L5, sampling hardware entropy and generating post-quantum credentials."

**Key Shortcut**: NuttX provides POSIX abstraction, avoiding the need for Farscape-generated CMSIS bindings.

**Timeline**: Requires NuttX porting, ARM target validation, PQC integration.

### Phase 3: Full Quotation Architecture (Long-Term)

**Goal**: Demonstrate "F# carries memory semantics through compilation"

**Narrative**: "Quotations describe hardware memory layouts. Active patterns recognize hardware operations. The compiler validates memory access at compile time and emits precisely correct code."

**Timeline**: Post-demo, strategic investment.

---

## The RTOS Approach: Why It's a Valid "Cheat"

### What NuttX Provides

1. **POSIX Compliance**: `open`, `read`, `write`, `ioctl`, `close` - the same patterns FNCS already uses for console I/O.

2. **Character Device Model**: Hardware peripherals appear as `/dev` entries:
   - `/dev/adc0` - ADC for entropy sampling
   - `/dev/gpio0` - GPIO control
   - `/dev/ttyACM0` - USB CDC for output

3. **Driver Abstraction**: NuttX drivers handle volatile register access, timing, and hardware quirks.

4. **Footprint**: 30-50KB for the RTOS - acceptable for STM32L5 with 256KB+ flash.

### How It Maps to Fidelity

```fsharp
// F# with FNCS (same as desktop console I/O pattern)
module Platform.Bindings =
    let openDevice (path: string) : int = Unchecked.defaultof<int>
    let closeDevice (fd: int) : int = Unchecked.defaultof<int>
    let read (fd: int) (buffer: nativeptr<byte>) (count: int) : int = Unchecked.defaultof<int>
    let write (fd: int) (buffer: nativeptr<byte>) (count: int) : int = Unchecked.defaultof<int>
    let ioctl (fd: int) (cmd: int) (arg: nativeint) : int = Unchecked.defaultof<int>

// Application code
module QuantumCredential =
    let readEntropy () =
        let fd = Platform.Bindings.openDevice "/dev/adc0"n
        let buffer = NativeArray.stackalloc<uint16> 256
        let bytesRead = Platform.Bindings.read fd (NativePtr.ofArray buffer) 512
        Platform.Bindings.closeDevice fd
        buffer
```

**Alex generates**: Standard syscall sequences (`svc #0` on ARM Cortex-M with NuttX).

### What This Defers

1. **Quotation-based memory validation**: No compile-time checking of register access patterns.

2. **Hardware-specific optimization**: NuttX abstracts away register-level details.

3. **Farscape integration**: No automated binding generation from CMSIS headers.

4. **Direct register access**: All hardware access goes through NuttX drivers.

These are acceptable trade-offs for a demo that proves "Fidelity compiles to embedded."

---

## Integration Points: Desktop ↔ Embedded

The QuantumCredential project spans both demo targets:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QuantumCredential System Architecture                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐           IR/USB           ┌──────────────────┐│
│  │ STM32L5 Unikernel   │ ────────────────────────► │ Desktop Keystation││
│  │                     │                            │                  ││
│  │ • Entropy sampling  │                            │ • Credential     ││
│  │ • PQC key gen       │                            │   display        ││
│  │ • Credential issue  │                            │ • Verification   ││
│  │                     │                            │ • Storage        ││
│  └─────────────────────┘                            └──────────────────┘│
│    (Embedded demo)                                    (Desktop demo)     │
│    NuttX + Firefly                                   WebView + Firefly   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Shared Components

| Component | Embedded | Desktop | Shared |
|-----------|----------|---------|--------|
| PQC algorithms | liboqs/pqm4 | liboqs | Algorithm code |
| BAREWire schemas | For IR framing | For IPC | Schema definitions |
| Credential types | F# record types | Same | Type definitions |
| FNCS | Core modules | Full library | Core + Console |

### Credential Exchange Protocol

Both sides use BAREWire for serialization:

```fsharp
// Shared/Credential.fs
type Credential = {
    PublicKey: byte[]     // ML-KEM public key
    Signature: byte[]     // ML-DSA signature
    Timestamp: int64
    Nonce: byte[]
}

let credentialSchema = BAREWire.schema<Credential>
```

**Embedded**: Encodes to BAREWire, transmits via IR/USB.
**Desktop**: Receives, decodes, verifies, displays.

---

## Action Items by Priority

### Immediate (Demo Enablers)

1. **Validate NuttX STM32L5 port** - Get LED blink working on hardware
2. **Add POSIX bindings to Fidelity.Platform** - `openDevice`, `closeDevice`, `ioctl`
3. **Extend Alex for ARM NuttX** - Syscall generation for POSIX operations
4. **Complete WebView Platform.Bindings** - Desktop demo foundation

### Near-Term (Demo Components)

5. **PQC integration** - Link liboqs, create F# wrappers
6. **Entropy pipeline** - ADC sampling → conditioning → PQC seeding
7. **Firefly build orchestration** - Fable + Vite integration
8. **BAREWire IPC** - Base64 bridge for webview communication

### Long-Term (Full Architecture)

9. **fsnative-spec** - Define native type semantics
10. **BAREWire quotation infrastructure** - Interpretation framework
11. **Farscape quotation output** - Generate Expr<PeripheralDescriptor>
12. **fsnative PSG extensions** - MemoryConstraint, nanopasses
13. **Baker quotation flow** - Typed tree overlay with memory semantics

---

## Success Criteria

### Desktop Demo Success

- [ ] Single `firefly build` produces native executable
- [ ] Executable displays reactive UI via system webview
- [ ] F# application logic runs natively (no .NET runtime)
- [ ] Frontend/backend IPC functional

### Embedded Demo Success

- [ ] F# code compiles to ARM Cortex-M33 binary
- [ ] Binary runs on STM32L5 with NuttX
- [ ] ADC samples entropy from hardware
- [ ] PQC key generation completes
- [ ] Credential output via USB

### System Demo Success

- [ ] Embedded device generates credential
- [ ] Desktop app receives and verifies credential
- [ ] Both ends use the same Fidelity compiler
- [ ] "F# from source to silicon" narrative demonstrated

---

## Cross-References

### Within QC_Demo
- [02_January_Roadmap](./02_January_Roadmap.md) - Original timeline and risk assessment
- [03_Hardware_Platforms](./03_Hardware_Platforms.md) - Hardware platform details
- [04_PostQuantum_Architecture](./04_PostQuantum_Architecture.md) - PQC and entropy design
- [05_Farscape_Assessment](./05_Farscape_Assessment.md) - Farscape gap analysis
- [06_UI_Stretch_Goals](./06_UI_Stretch_Goals.md) - UI framework options

### Architecture Documents (parent /docs/)
- [WebView_Desktop_Architecture.md](../WebView_Desktop_Architecture.md) - Desktop stack architecture
- [WebView_Build_Integration.md](../WebView_Build_Integration.md) - Build orchestration
- [Quotation_Based_Memory_Architecture.md](../Quotation_Based_Memory_Architecture.md) - Full memory architecture
