# Stretch Goals: Enhanced Demo Experience

> **Status**: Stretch goals beyond core demo functionality
>
> **Visual Impact**: Two embedded devices with touchscreen UIs exchanging quantum-safe credentials makes a powerful investor demo

---

## The Hardware Symmetry Insight

Both the YoshiPi (Credential Generator) and the Sweet Potato (Keystation) share the **same analog front end**:

| Component | YoshiPi | Sweet Potato | Purpose |
|-----------|---------|--------------|---------|
| **Avalanche Circuit** | Yes | Yes | True random entropy |
| **IR Transmitter** | Yes | Yes | Credential transmission |
| **IR Receiver** | Yes | Yes | Credential reception |
| **ADC Input** | Yes | Yes | Entropy sampling |
| **Touchscreen** | Yes | Yes | User interaction |

**The role distinction (Generator vs Keystation) is purely software-defined.**

This enables:
- **Same PCB design** - One hardware platform, two software roles
- **Bidirectional communication** - Either device can TX or RX
- **Mutual authentication** - Both generate credentials, both verify
- **Role reversal** - The Keystation could generate, the Generator could receive
- **Peer-to-peer exchange** - Two "peers" exchanging credentials

### The "Dirty Little Secret": Self-Sovereign CA

Here's the profound implication: **Any QuantumCredential device can serve as its own Certificate Authority.**

The device has:
- **Hardware entropy** (avalanche circuit) - True randomness, not PRNG
- **PQC key generation** (ML-KEM, ML-DSA) - Quantum-resistant cryptography
- **Signing capability** - Can sign any credential

This means a QuantumCredential device can:
1. Generate its own root key pair (self-signed)
2. Sign credentials for OTHER devices
3. Create a trust hierarchy rooted in quantum-derived randomness

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trust Hierarchy                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌─────────────────────┐                            │
│              │  Root CA Device     │                            │
│              │  (any QC device)    │                            │
│              │                     │                            │
│              │  Avalanche → Root   │                            │
│              │  Entropy     Key    │                            │
│              └──────────┬──────────┘                            │
│                         │ Signs                                  │
│            ┌────────────┼────────────┐                          │
│            ▼            ▼            ▼                          │
│     ┌───────────┐ ┌───────────┐ ┌───────────┐                   │
│     │ Device A  │ │ Device B  │ │ Device C  │                   │
│     │ (signed)  │ │ (signed)  │ │ (signed)  │                   │
│     └───────────┘ └───────────┘ └───────────┘                   │
│                                                                 │
│     Any device with the analog front end can BE the root CA     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**UI Model: Air-Gapped vs Connected**

| Mode | Device | UI | Focus |
|------|--------|-----|-------|
| **Air-gapped** | YoshiPi/Sweet Potato (touchscreen) | Visual, touch-friendly | Generate, receive, verify |
| **Connected** | Desktop/mobile app | Full feature set | CA management, signing, advanced ops |

The touchscreen Keystation keeps the **visual demo impact** - tap to accept, swipe through credentials, watch entropy flow. The CA capability is there but stays tucked away for advanced users accessing via the connected desktop/mobile app.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Demo Experience                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Air-Gapped (Visual Demo)              Connected (Power User)   │
│  ┌─────────────────────┐               ┌─────────────────────┐ │
│  │  Touchscreen UI     │               │  Desktop/Mobile App │ │
│  │                     │               │                     │ │
│  │  • Entropy viz      │               │  • CA management    │ │
│  │  • Tap to generate  │     USB/BT    │  • Sign other devs  │ │
│  │  • Swipe to accept  │◄─────────────►│  • Trust hierarchy  │ │
│  │  • Visual verify    │               │  • Batch operations │ │
│  │                     │               │  • Export/backup    │ │
│  └─────────────────────┘               └─────────────────────┘ │
│                                                                 │
│  Investor sees the touchscreen magic.                           │
│  Power users unlock the CA via connected app.                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Demo narrative enhancement**:
> "What you're seeing is the visual experience - tap to generate, beam to transfer, swipe to accept. But there's hidden power here: connect this device to the desktop app and it becomes a full certificate authority. It can sign credentials for any device in your network. The trust root is quantum randomness, not some corporate CA. Self-sovereign identity in your pocket."

This transforms the demo from "credential generator" to "decentralized PKI infrastructure" - with the advanced features appropriately hidden behind the connected interface.

```
┌─────────────────────┐                  ┌─────────────────────┐
│  Device A           │                  │  Device B           │
│  (any role)         │                  │  (any role)         │
│                     │                  │                     │
│  Avalanche ──► PQC  │  ◄── IR TX ───►  │  Avalanche ──► PQC  │
│  IR TX/RX           │  ◄── IR RX ───►  │  IR TX/RX           │
│  Touch UI           │                  │  Touch UI           │
└─────────────────────┘                  └─────────────────────┘

        Same hardware. Same code. Role is just configuration.
```

---

## Overview: The Full Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     QuantumCredential Demo (Full Vision)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐   │
│  │  YoshiPi                │         │  Libre Sweet Potato             │   │
│  │  (Credential Generator) │         │  (Keystation)                   │   │
│  │                         │         │                                 │   │
│  │  ┌───────────────────┐  │         │  ┌───────────────────────────┐ │   │
│  │  │  Touchscreen UI   │  │  ─────► │  │  Ultra-Wide Touchscreen   │ │   │
│  │  │  • Entropy viz    │  │   USB   │  │  • Credential display     │ │   │
│  │  │  • Status display │  │   IR    │  │  • Verification status    │ │   │
│  │  │  • Touch to gen   │  │   QR    │  │  • Touch to accept        │ │   │
│  │  └───────────────────┘  │         │  └───────────────────────────┘ │   │
│  │                         │         │                                 │   │
│  │  Avalanche ──► PQC      │         │  Verify ──► Store              │   │
│  │  Circuit       Keygen   │         │                                 │   │
│  └─────────────────────────┘         └─────────────────────────────────┘   │
│                                                                             │
│  Both devices: Linux + WebKitGTK + Firefly-compiled F# + Same UI code      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stretch Goal 1: Libre Sweet Potato Keystation

### Hardware: Libre Computer Sweet Potato

| Specification | Value |
|---------------|-------|
| **SoC** | Amlogic S905X (quad-core Cortex-A53) |
| **Memory** | 2GB DDR3 |
| **OS** | Linux (Debian/Ubuntu) |
| **Display** | HDMI or DSI panel support |
| **USB** | USB 2.0 host ports |
| **GPIO** | 40-pin header (Pi-compatible) |

**Why Sweet Potato for Keystation:**
- More RAM (2GB vs 512MB) - comfortable for WebKitGTK
- Same Linux/ARM64 target as YoshiPi
- Same native F# APIs, same WebView UI
- Can drive an ultra-wide touchscreen via DSI or USB

### Ultra-Wide Touchscreen

Options for dramatic visual impact:

| Display | Resolution | Interface | Touch |
|---------|------------|-----------|-------|
| Waveshare 7.9" Ultra-Wide | 1280x400 | HDMI + USB | Capacitive |
| Waveshare 11.9" Ultra-Wide | 320x1480 | HDMI + USB | Capacitive |
| Custom DSI panel | Various | DSI | I2C touch |

The ultra-wide format is ideal for:
- Credential list display (vertical scroll)
- Status bar with entropy visualization
- Touch-friendly large buttons

### Software Stack (Identical to YoshiPi)

```fsharp
// Keystation main - runs on Sweet Potato
let main () =
    let w = Webview.create true
    Webview.setTitle w "Keystation"
    Webview.setSize w 1280 400  // Ultra-wide
    Webview.setHtml w EmbeddedAssets.KeystationHtml
    Webview.run w
    0
```

**The entire UI codebase is shared** - Partas.Solid components work identically on:
- YoshiPi (480x320 touchscreen)
- Sweet Potato (1280x400 ultra-wide)
- Desktop (any resolution)

Only the CSS layout adapts to screen dimensions.

---

## Stretch Goal 2: Touch Interaction

### Linux Touch Input Stack

Touch input on Linux comes through:

```
/dev/input/eventN        # Raw touch events
    ↓
libinput                 # Input processing
    ↓
Wayland/X11              # Window system
    ↓
WebKitGTK                # Browser engine
    ↓
JavaScript touch events  # Your Partas.Solid handlers
```

**Good news**: WebKitGTK handles touch automatically. Touch events appear as standard browser touch/pointer events in your SolidJS code.

### Touch in Partas.Solid

```fsharp
[<SolidComponent>]
let GenerateButton () =
    let generating, setGenerating = createSignal false

    button(
        class' = "generate-btn",
        // Standard DOM events - work with touch!
        onClick = fun _ ->
            setGenerating true
            generateCredential () |> ignore
        // Optional: explicit touch handling
        onTouchStart = fun e ->
            e.preventDefault()  // Prevent scroll
            setGenerating true
    ) {
        if generating() then "Generating..." else "Generate Credential"
    }
```

### Platform.Bindings for Direct Touch (Optional)

If you need raw touch data (e.g., for entropy from touch timing):

```fsharp
module Platform.Bindings =
    /// Open input device for raw events
    let openInputDevice (path: nativeint) : int =
        Unchecked.defaultof<int>

    /// Read input event (struct input_event)
    let readInputEvent (fd: int) (event: nativeint) : int =
        Unchecked.defaultof<int>

module Touch =
    open Platform.Bindings

    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type InputEvent =
        val mutable tv_sec: int64
        val mutable tv_usec: int64
        val mutable type_: uint16
        val mutable code: uint16
        val mutable value: int32

    // Event types
    let EV_ABS = 0x03us
    let ABS_MT_POSITION_X = 0x35us
    let ABS_MT_POSITION_Y = 0x36us

    /// Read touch position (for entropy mixing)
    let readTouchEvent (fd: int) : (int * int) option =
        let event = NativeArray.stackalloc<InputEvent> 1
        let bytesRead = readInputEvent fd (NativePtr.toNativeInt event)
        if bytesRead = sizeof<InputEvent> then
            if event.[0].type_ = EV_ABS then
                Some (int event.[0].code, event.[0].value)
            else None
        else None
```

### Touch Interaction Model

| Device | Touch Capability | Primary Interactions |
|--------|------------------|---------------------|
| **YoshiPi** | Small touchscreen | • "Generate" button<br>• Status tap to expand<br>• Simple gestures |
| **Sweet Potato** | Ultra-wide touch | • Credential list scroll<br>• Accept/reject swipe<br>• Detail expansion |

---

## Stretch Goal 3: Alternative Transfer Methods

### Current: USB Serial (Primary)

USB gadget mode remains the primary transfer method - reliable, fast, and already designed.

### Alternative 1: QR Code Transfer

**Visual impact**: Credential appears as QR code on YoshiPi, Sweet Potato camera scans it.

```
YoshiPi                          Sweet Potato
┌─────────────┐                  ┌─────────────────┐
│  ┌───────┐  │                  │  ┌───────────┐  │
│  │ ▓▓▓▓▓ │  │  ────────────►   │  │  Camera   │  │
│  │ ▓   ▓ │  │   Camera scan    │  │   View    │  │
│  │ ▓▓▓▓▓ │  │                  │  └───────────┘  │
│  └───────┘  │                  │                 │
│   QR Code   │                  │  "Scanning..."  │
└─────────────┘                  └─────────────────┘
```

**Implementation considerations:**
- QR generation: Pure F# (no external lib needed for basic QR)
- QR scanning: Requires camera + decoding library (zbar or similar)
- Credential size: ML-KEM public keys are ~1KB - need multiple QR codes or compression
- Adds hardware requirement: USB camera on Sweet Potato

**Verdict**: High visual impact, but adds complexity. Consider as stretch-stretch goal.

### Alternative 2: IR Transceiver (Bidirectional)

**Visual impact**: Devices "beam" credentials via infrared - **in both directions**.

Since both devices have the same analog front end (IR TX + IR RX), communication is inherently bidirectional:

```
Device A                              Device B
┌─────────────────┐                  ┌─────────────────┐
│                 │                  │                 │
│    ◉ IR TX  ────┼──────────────────┼───► ◉ IR RX    │
│                 │   Credential     │                 │
│    ◉ IR RX  ◄───┼──────────────────┼──── ◉ IR TX    │
│                 │   Acknowledgment │                 │
└─────────────────┘                  └─────────────────┘

        Either device can initiate. Either can respond.
```

**Hardware (same on both devices):**
- IR LED + driver on GPIO (TX)
- IR receiver module on GPIO (RX)
- Modulation/demodulation in software

**Protocol options:**
- IrDA-style framing (standard, but complex)
- Custom simple protocol (easier to implement)
- LIRC kernel module for standard IR

```fsharp
module IR =
    /// Transceiver config (identical on both devices)
    type TransceiverConfig = {
        TxGpio: int      // IR LED GPIO line
        RxGpio: int      // IR receiver GPIO line
        CarrierHz: int   // Typically 38000
    }

    /// Send byte via IR (bit-banged on GPIO)
    let sendByte (cfg: TransceiverConfig) (b: byte) : unit =
        for bit in 7 .. -1 .. 0 do
            let value = (b >>> bit) &&& 1uy
            GPIO.setValue cfg.TxGpio (int value) |> ignore
            // 38kHz carrier modulation...
            busyWait pulseWidth

    /// Receive byte (decode pulse timing)
    let receiveByte (cfg: TransceiverConfig) : byte option =
        // Sample RX GPIO, decode pulse widths
        // ...
        None
```

**Demo flow** (leveraging hardware symmetry):
1. Device A generates credential, beams via IR
2. Device B receives, verifies signature
3. Device B beams acknowledgment back
4. Both touchscreens show success

**Verdict**: With symmetric hardware, IR becomes more compelling. Good stretch goal - visually dramatic "beaming" between devices.

### Alternative 3: BLE (Bluetooth Low Energy)

**Both devices have BLE** (Pi Zero 2 W and most ARM SBCs).

```fsharp
module Platform.Bindings =
    /// Open Bluetooth HCI device
    let openBluetooth () : int =
        Unchecked.defaultof<int>

    /// BLE GATT operations...
```

**Verdict**: Standard and reliable, but less visually dramatic than QR/IR. Good fallback.

---

## Stretch Goal 4: Enhanced Entropy Visualization

### Real-Time Avalanche Display

Show the raw entropy on the YoshiPi screen:

```
┌─────────────────────────────────────────┐
│  QuantumCredential Generator            │
├─────────────────────────────────────────┤
│                                         │
│  Entropy Source: ████████████ HEALTHY   │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ▁▃▅▇█▆▄▂▁▃▅▇█▆▄▂▁▃▅▇█▆▄▂▁▃▅▇█▆ │    │  ← Live waveform
│  │ ▂▄▆█▇▅▃▁▂▄▆█▇▅▃▁▂▄▆█▇▅▃▁▂▄▆█▇▅ │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Samples: 1,024,576  Entropy: 7.98 b/B  │
│                                         │
│         [ GENERATE CREDENTIAL ]         │
│                                         │
└─────────────────────────────────────────┘
```

**Implementation**:
- Sample ADC continuously in background
- Send samples to WebView via IPC
- Canvas or SVG visualization in Partas.Solid
- Calculate entropy estimate (bits per byte)

```fsharp
[<SolidComponent>]
let EntropyWaveform (samples: Accessor<uint16 array>) =
    canvas(
        id = "entropy-canvas",
        width = 320,
        height = 100,
        ref = fun el ->
            createEffect (fun () ->
                drawWaveform el (samples())
            )
    ) { }
```

---

## Priority and Dependencies

### Core Demo (Must Have)

| Component | Status | Risk |
|-----------|--------|------|
| YoshiPi entropy sampling | Design complete | LOW |
| PQC key generation | Algorithm selected | MEDIUM |
| USB credential transfer | Standard Linux | LOW |
| Basic WebView UI | Architecture ready | LOW |

### Stretch Goal Priority

| Goal | Visual Impact | Effort | Dependencies |
|------|---------------|--------|--------------|
| **1. Touch on YoshiPi** | Medium | LOW | WebKitGTK handles it |
| **2. Sweet Potato Keystation** | HIGH | MEDIUM | Same code, different screen |
| **3. Entropy visualization** | HIGH | MEDIUM | Canvas + IPC |
| **4. Touch on Sweet Potato** | Medium | LOW | Same as YoshiPi |
| **5. QR code transfer** | HIGH | HIGH | QR gen + camera + decode |
| **6. IR transfer** | HIGH | HIGH | GPIO + modulation |

**Recommended order**: 1 → 2 → 3 → 4 → (5 or 6 if time)

---

## Hardware Shopping List

### Core Demo

| Item | Purpose | Approx Cost |
|------|---------|-------------|
| YoshiPi carrier board | Credential generator platform | ~$40 |
| Raspberry Pi Zero 2 W | Compute module | ~$15 |
| Avalanche diodes (5x) | True random entropy | ~$10 |
| USB cable | Credential transfer | ~$5 |

### Stretch Goals

| Item | Purpose | Approx Cost |
|------|---------|-------------|
| Libre Sweet Potato | Keystation platform | ~$35 |
| Ultra-wide touchscreen | Dramatic display | ~$60-100 |
| USB camera (optional) | QR scanning | ~$20 |
| IR LED + receiver (optional) | IR transfer | ~$5 |

---

## Demo Narrative

### With Stretch Goals

> "Watch as the YoshiPi samples quantum noise from its avalanche circuit - you can see the entropy flowing in real-time on the display. When I tap 'Generate', it uses this true randomness to create a post-quantum credential.
>
> The credential appears as a QR code. The Keystation scans it with its camera... verified! The ML-DSA signature checks out. This credential is now stored and ready to use.
>
> Both of these devices are running the same F# code, compiled to native ARM binaries by Firefly. No runtime, no garbage collection, no security vulnerabilities from managed code. Just pure, verified, post-quantum security."

### Without Stretch Goals (Core Demo)

> "The YoshiPi generates credentials from hardware entropy. They transfer via USB to the desktop Keystation. Same principle, simpler setup - but the code is identical. When we add the embedded Keystation hardware, it's just a recompile to ARM."

---

## Cross-References

- [01_YoshiPi_Demo_Strategy](./01_YoshiPi_Demo_Strategy.md) - Core demo strategy
- [02_YoshiPi_Architecture](./02_YoshiPi_Architecture.md) - YoshiPi hardware details
- [03_MLIR_Dialect_Strategy](./03_MLIR_Dialect_Strategy.md) - Compilation path and parallel execution
- [04_Linux_Hardware_Bindings](./04_Linux_Hardware_Bindings.md) - Platform bindings
- [05_PostQuantum_Architecture](./05_PostQuantum_Architecture.md) - PQC algorithms
