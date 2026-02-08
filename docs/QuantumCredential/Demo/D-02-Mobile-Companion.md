# D-02: Mobile Companion App (Android)

## Overview

The QuantumCredential device is small, purpose-built hardware. It generates and stores credentials but has limited display and no camera. A phone connected via USB serves as the device's display surface and camera, enabling human-to-human credential exchange through physically constrained channels.

The phone is a dumb terminal. It never touches key material.

---

## The Air-Gap Model

The patent architecture (US 63/780,027) specifies air-gapped credential distribution. This means credentials cross between parties only through physically constrained channels where interception requires physical presence:

| Channel | Medium | Air-Gapped | Why |
|---------|--------|------------|-----|
| **QR code** | Visible light (screen to camera) | Yes | Requires line-of-sight, physically observable interception |
| **IR transceiver** | Infrared (LED to photodiode) | Yes | Requires line-of-sight, short range |
| **USB cable** | Electrical (copper conductor) | Yes | Requires physical access to the wire |
| BLE / WiFi / NFC | Radio | **No** | Interceptable at distance, relay attacks, no physical constraint |

BLE is radio. Radio is not air-gapping. The entire point of the QuantumCredential architecture is that credential distribution never traverses a channel where a remote attacker could intercept, replay, or relay without being physically present.

---

## Why the Phone

Two people want to exchange quantum-safe credentials. Each has a QC device (small, pocketable). The exchange looks like:

```
Person A                                          Person B
┌──────────────────┐                              ┌──────────────────┐
│  Phone (display)  │     ◄── visual gap ──►      │  Phone (camera)   │
│  ┌──────────────┐ │                              │  ┌──────────────┐ │
│  │  QR sequence  │ │  ── light photons ──►       │  │  Camera feed  │ │
│  │  ▓▓▓ ▓ ▓▓▓   │ │                              │  │  [scanning]   │ │
│  │  ▓     ▓      │ │                              │  │              │ │
│  │  ▓▓▓ ▓ ▓▓▓   │ │                              │  │              │ │
│  └──────────────┘ │                              │  └──────────────┘ │
│        │ USB       │                              │        │ USB       │
│  ┌─────┴────────┐ │                              │  ┌─────┴────────┐ │
│  │  QC Device   │ │                              │  │  QC Device   │ │
│  │  (generates) │ │                              │  │  (receives)  │ │
│  └──────────────┘ │                              │  └──────────────┘ │
└──────────────────┘                              └──────────────────┘
```

The phone provides what the QC device lacks:
- A large, bright screen for QR code display
- A camera for QR code scanning
- A touch interface for user confirmation

The phone does **not** provide:
- Key generation (that is the QC device, using hardware entropy)
- Signing (that is the QC device)
- Key storage (credentials live on the QC device)
- Any cryptographic operation whatsoever

---

## Architecture

### Separation of Concerns

| Responsibility | Where | Why |
|---------------|-------|-----|
| Entropy sampling | QC device | Hardware avalanche circuit |
| Key generation | QC device | PQC algorithms seeded by hardware entropy |
| Credential signing | QC device | Private keys never leave device |
| Credential storage | QC device | Tamper-resistant hardware (Phase 2: HUK-wrapped) |
| QR code generation | Phone | Large screen, high resolution |
| QR code scanning | Phone | Camera with autofocus |
| User interaction | Phone | Touch UI, visual feedback |
| BAREWire framing | QC device | Credential serialization before display |

### Data Flow: Sending a Credential

```
QC Device (sender)                  Phone (sender)
─────────────────                   ──────────────
1. Generate credential
2. Sign with ML-DSA
3. BAREWire-encode
4. Frame for QR transfer ──USB──►  5. Render QR sequence on screen
                                    6. Display frame N of M
                                    7. Advance on tap or timer
```

### Data Flow: Receiving a Credential

```
Phone (receiver)                    QC Device (receiver)
────────────────                    ────────────────────
1. Open camera
2. Scan QR frames
3. Reassemble payload  ──USB──►    4. BAREWire-decode
                                    5. Verify ML-DSA signature
                                    6. Store credential
                                    7. Send verification result ──USB──► 8. Display result
```

The phone is a viewport. All cryptographic decisions happen on the QC device.

---

## USB Connection

### QC Device Side: Linux USB Gadget Mode

The YoshiPi (Pi Zero 2 W) has a USB OTG port. Using the Linux USB gadget subsystem via configfs, the device presents itself as a serial device when plugged into a phone:

```
# On the QC device at boot (via init script or systemd service)
modprobe libcomposite

# Configure USB gadget
mkdir -p /sys/kernel/config/usb_gadget/qcdevice
cd /sys/kernel/config/usb_gadget/qcdevice

echo 0x1d6b > idVendor    # Linux Foundation (placeholder)
echo 0x0104 > idProduct    # QC Device identifier
echo "QuantumCredential" > strings/0x409/product
echo "Fidelity" > strings/0x409/manufacturer

# CDC-ACM function (appears as /dev/ttyACM0 on phone)
mkdir -p functions/acm.usb0
mkdir -p configs/c.1
ln -s functions/acm.usb0 configs/c.1/

# Bind to UDC
echo "$(ls /sys/class/udc)" > UDC
```

The device then communicates via `/dev/ttyGS0` (gadget serial). BAREWire-framed messages flow over this serial link.

### Phone Side: Android USB Host

Android detects the USB device by vendor/product ID and can auto-launch the companion app:

```xml
<!-- AndroidManifest.xml -->
<activity android:name=".CompanionActivity">
    <intent-filter>
        <action android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED"/>
    </intent-filter>
    <meta-data
        android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED"
        android:resource="@xml/device_filter"/>
</activity>
```

```xml
<!-- res/xml/device_filter.xml -->
<resources>
    <usb-device vendor-id="7523" product-id="260"/>
</resources>
```

When the user plugs in the QC device for the first time, Android prompts "Open with QuantumCredential?" with a "Always" checkbox. After that, plugging in the device launches the app automatically.

The Android app opens the USB device as a serial port via the USB Host API or a thin library like usb-serial-for-android.

---

## The Mobile Shell

The companion app is minimal. The phone side consists of:

1. A native Android Activity (Kotlin, approximately 50 lines)
2. A WebView filling the Activity
3. A USB serial connection to the QC device
4. Camera access for QR scanning (standard Android Camera2 API)

No framework is needed. No WRY, no Tauri, no Capacitor, no React Native. The app is:
- A WebView that loads the Partas.Solid UI bundle
- A thin Kotlin bridge exposing USB serial read/write and camera frames to JavaScript

### Why No Framework

| Consideration | Assessment |
|--------------|------------|
| Cross-platform abstraction | Not needed; Android-only for now |
| WebView wrapper | Android provides `android.webkit.WebView` natively |
| Native API access | USB and Camera require platform-specific Kotlin regardless |
| Bundle size | WebView is already on the device; app is just HTML + JS + Kotlin glue |
| Maintenance | 50 lines of Kotlin is simpler than any framework dependency |

### Shared UI Bundle

The Partas.Solid UI compiled for the QC device touchscreen, the desktop keystation, and the phone companion app is the same bundle. The only difference is viewport dimensions and which capabilities are available:

| Platform | Screen | USB | Camera | Entropy | Crypto |
|----------|--------|-----|--------|---------|--------|
| YoshiPi touchscreen | 480x320 | Gadget (device side) | No | Yes | Yes |
| Desktop keystation | Any | Host | Optional | No | Verify only |
| Phone companion | Phone-sized | Host | Yes | No | No |

The UI detects available capabilities via the JavaScript bridge and shows relevant controls.

---

## QR Transfer Protocol

ML-KEM public keys are approximately 1,184 bytes (ML-KEM-768) or 1,568 bytes (ML-KEM-1024). A signed credential with metadata exceeds what a single QR code can carry at error correction level M.

### Multi-Frame QR Sequence

The credential is split into frames:

| Field | Size | Purpose |
|-------|------|---------|
| Frame index | 1 byte | Which frame (0-indexed) |
| Total frames | 1 byte | How many frames total |
| Sequence ID | 2 bytes | Links frames to one credential |
| Payload | variable | Credential chunk |
| CRC-16 | 2 bytes | Per-frame integrity |

Each frame encodes as a QR code at a size the phone camera can reliably scan. The sending phone displays frames in sequence; the receiving phone scans them and reassembles.

### Scanning UX

The receiving phone shows:
- Live camera preview
- Progress indicator (3 of 7 frames scanned)
- Frames can arrive out of order (the scanner just fills slots)
- Haptic feedback on each successful frame scan
- Completion triggers automatic verification on the QC device

---

## Boot Sequence

The QC device does not wait for the phone. It is always ready:

```
Power on
    │
    ├── Start entropy sampling (continuous)
    ├── Initialize USB gadget (CDC-ACM)
    ├── Load credential store
    │
    ▼
Ready (LED indicator)
    │
    ├── Phone connects via USB ──► Send device info, capability list
    ├── Phone disconnects ──► Continue sampling, no state change
    │
    ▼
Idle (entropy accumulates, device is self-sufficient)
```

The phone connection is a viewport attachment. Disconnecting the phone does not interrupt any device operation.

---

## Scope Decisions

### Android Only (for Now)

| Platform | Status | Rationale |
|----------|--------|-----------|
| **Android** | Active | USB Host API is straightforward; no certification needed |
| iOS | Deferred | MFi certification required for USB accessories; adds cost and timeline |

iOS support may be revisited when the product matures. The WebView UI bundle and QC device firmware are platform-agnostic; only the native shell differs.

### What the Phone App Does NOT Do

To be explicit about boundaries:

- Does not generate keys
- Does not store private keys
- Does not perform signing or verification
- Does not connect to any network on behalf of the QC device
- Does not cache credentials beyond the current session
- Does not require an account, login, or cloud service

The phone is glass and silicon in service of the QC device. Nothing more.

---

## Implementation Phases

### Phase 1: USB Serial + Status Display

| Task | Complexity |
|------|-----------|
| Android Activity + WebView shell | Low |
| USB serial connection (usb-serial-for-android) | Low |
| BAREWire message parsing in JS bridge | Medium |
| Status display UI (shared Partas.Solid bundle) | Low (already exists) |

This phase proves the phone-as-viewport model. The QC device generates credentials and the phone displays them.

### Phase 2: QR Display (Sending)

| Task | Complexity |
|------|-----------|
| Multi-frame QR generation (JS library) | Low |
| QR sequence display with tap-to-advance | Low |
| Frame progress indicator | Low |

### Phase 3: QR Scanning (Receiving)

| Task | Complexity |
|------|-----------|
| Camera2 API integration | Medium |
| QR frame decoding (ML Kit or ZXing) | Low |
| Frame reassembly and forwarding to QC device via USB | Medium |
| Verification result display | Low |

### Phase 4: IR Support (if hardware available)

If the phone has an IR blaster (some Android phones do), or if an IR transceiver is added to the QC device's USB interface, IR transfer becomes available as a second air-gapped channel.

---

## Cross-References

### This Folder
- [D-01-Demo-Strategy](./D-01-Demo-Strategy.md) - Core demo architecture (YoshiPi + desktop keystation)

### QuantumCredential Docs
- [07_Stretch_Goals](../07_Stretch_Goals.md) - QR and IR transfer as stretch goals; self-sovereign CA
- [05_PostQuantum_Architecture](../05_PostQuantum_Architecture.md) - ML-KEM and ML-DSA credential structure

### Phase 2
- [PH2-00-Embedded-Strategy](../Phase2-Embedded/PH2-00-Embedded-Strategy.md) - Renesas RA6M5 with HUK for device-bound credentials

### Patents
- US 63/780,027 - Air-Gapped Dual Network Architecture (core transfer model)
- US 63/780,055 - Quantum-Resistant Hardware Security Module (device identity)
