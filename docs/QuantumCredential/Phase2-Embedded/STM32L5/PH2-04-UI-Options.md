# Demo Day UI Stretch Goal: FidelityUI

This document outlines a stretch goal for demo day: building graphical user interfaces for both the Keystation (Libre Sweet Potato with WaveShare display) and a desktop companion app for QuantumCredential operations. The goal is to demonstrate the full Fidelity vision: F# functional UI code compiled to native binaries, running without runtime overhead.

## Executive Summary

The stretch goal is to have **two FidelityUI applications** demonstrating QuantumCredential operations:

1. **Keystation UI** - Running on the Libre Sweet Potato with WaveShare 7" touchscreen (LVGL backend)
2. **Desktop Companion App** - Running on Linux x86_64 for QuantumCredential demo interactions (GTK4 backend)

Both applications share the same **F# MVU codebase**, compiled to their respective native targets via Firefly with different rendering backends. This demonstrates the true Fidelity promise: **F# is the universal syntax: what it binds to is an implementation detail.**

## The Multi-Backend Strategy

Rather than forcing LVGL (an embedded toolkit) onto the Linux desktop, FidelityUI uses appropriate native backends:

```
┌─────────────────────────────────────────────────────────────┐
│ F# Application Code (Fabulous-style MVU API)               │
│   - Same Model, View, Update for all platforms             │
│   - Shared business logic and UI structure                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FidelityUI Abstraction Layer                                │
│   - Widget descriptions                                     │
│   - Layout specifications                                   │
│   - Event dispatching                                       │
└─────────────────────────────────────────────────────────────┘
          ↓                              ↓
┌─────────────────────┐        ┌─────────────────────┐
│ LVGL Backend        │        │ GTK4 Backend        │
│ (Keystation)        │        │ (Desktop)           │
│ - Embedded Linux    │        │ - Native Linux look │
│ - Touch-optimized   │        │ - System theme      │
│ - Farscape C hdrs   │        │ - Farscape GIR      │
└─────────────────────┘        └─────────────────────┘
          ↓                              ↓
    LVGL + fbdev              GTK4 + libgtk-4.so.1
```

| Target | Backend | Why |
|--------|---------|-----|
| **Keystation (Sweet Potato)** | LVGL | Touch-optimized, framebuffer direct, embedded-friendly |
| **Desktop Companion** | GTK4 | Native Linux look, system theme integration, HiDPI |
| **STM32L5 (future)** | LVGL | Resource-constrained MCU |
| **macOS (future)** | AppKit | Native macOS experience |
| **Windows (future)** | WinUI | Native Windows experience |

## Why LVGL for Embedded?

LVGL (Light and Versatile Graphics Library) is the correct choice for Fidelity's **embedded** UI story:

| Characteristic | LVGL Advantage |
|----------------|----------------|
| **Footprint** | ~64KB Flash, ~16KB RAM minimum |
| **No Runtime** | Pure C library, no interpreter or GC |
| **Rich Widgets** | Buttons, labels, charts, keyboards, calendars |
| **Touch Support** | Built-in gesture recognition, input handling |
| **Layout System** | Flexbox and Grid layouts |
| **Theming** | Style system with compile-time customization |

LVGL is battle-tested in production embedded systems and aligns perfectly with Fidelity's "no runtime" philosophy for resource-constrained targets.

## Why GTK4 for Desktop?

For Linux desktop applications, GTK4 provides the **native experience**:

| Characteristic | GTK4 Advantage |
|----------------|----------------|
| **Native Look** | Integrates with GNOME/system themes |
| **HiDPI Support** | Automatic scaling, fractional scaling |
| **Accessibility** | Screen reader support, keyboard navigation |
| **Rich Ecosystem** | libadwaita, GStreamer integration |
| **GIR Bindings** | Auto-generated via Farscape from introspection data |

GTK4 bindings are generated via Farscape's GIR integration (see `docs/Farscape_GIR_Integration.md`).

## FidelityUI Architecture

From the SpeakEZ blog documentation, FidelityUI follows this architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ F# Application Code (Fabulous-style MVU API)               │
│   - Model: Application state                                │
│   - View: Declarative widget descriptions                   │
│   - Update: Pure state transitions                          │
└─────────────────────────────────────────────────────────────┘
                              ↓ Firefly Compiler
┌─────────────────────────────────────────────────────────────┐
│ MLIR Transformation (Progressive Lowering)                  │
│   - Widget descriptions → Direct LVGL calls                 │
│   - Event handlers → Static function pointers               │
│   - Layout → LVGL flex/grid configuration                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Platform Targets                                            │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│   │ Linux x86_64│  │ AArch64     │  │ Cortex-M (future)   ││
│   │ Desktop     │  │ Sweet Potato│  │ STM32L5            ││
│   │ LVGL + SDL  │  │ LVGL + fbdev│  │ LVGL + direct LCD  ││
│   └─────────────┘  └─────────────┘  └─────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

From "Building User Interfaces with the Fidelity Framework":

1. **Widget descriptions exist only at compile time** - The Firefly compiler transforms functional descriptions into efficient native code
2. **No heap allocations** - All state management uses stack/arena allocation
3. **Event handlers become static function pointers** - No closure allocations
4. **Same F# API across all targets** - Developer writes Fabulous-style code

### MVU Pattern Preserved

```fsharp
// Developer writes standard MVU code
type Model = {
    CredentialStatus: CredentialStatus
    LastOperation: string
    IsConnected: bool
}

type Msg =
    | Connect
    | Disconnect
    | ReadCredential
    | CredentialRead of CredentialData
    | Error of string

let update msg model =
    match msg with
    | Connect -> { model with IsConnected = true }
    | CredentialRead data ->
        { model with
            CredentialStatus = Ready data
            LastOperation = "Credential read successfully" }
    | Error msg ->
        { model with LastOperation = $"Error: {msg}" }
    // ...

let view model dispatch =
    VStack(spacing = 16.) {
        Label("QuantumCredential Demo")
            .fontSize(24.)

        Label($"Status: {model.CredentialStatus}")

        if model.IsConnected then
            Button("Read Credential", fun () -> dispatch ReadCredential)
                .buttonStyle(ButtonStyle.Primary)
        else
            Button("Connect Device", fun () -> dispatch Connect)

        Label(model.LastOperation)
            .textColor(Color.Gray)
    }
```

This compiles to direct LVGL calls without runtime overhead.

## Target 1: Keystation on Libre Sweet Potato

### Hardware Setup

```
┌──────────────────────────────────────────────────────────────┐
│ Libre Sweet Potato (AML-S905X-CC-V2)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ WaveShare 7" HDMI Touch Display                        │ │
│  │ - 1024x600 IPS panel                                   │ │
│  │ - 5-point capacitive touch                             │ │
│  │ - USB touch interface (HID protocol)                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  USB #1: Display touch (hid-multitouch)                     │
│  HDMI: Display video output                                  │
│  USB #2: Optional keyboard for setup                         │
│  USB-C: Power (5V/3A)                                       │
└──────────────────────────────────────────────────────────────┘
```

### LVGL Backend Configuration

For the Sweet Potato running embedded Linux without a desktop environment:

```c
// LVGL Linux framebuffer configuration
#define LV_USE_LINUX_FBDEV 1
#define LV_LINUX_FBDEV_DEVICE "/dev/fb0"

// Touch input via evdev
#define LV_USE_EVDEV 1
// Auto-discovers touch device, or set explicitly:
// LV_LINUX_EVDEV_POINTER_DEVICE="/dev/input/event0"
```

### WaveShare Touch Driver

Modern [WaveShare 7" displays](https://www.waveshare.com/wiki/7inch_HDMI_LCD_(C)) (Rev 2.1+) use standard HID protocol: no custom driver needed. For older revisions, the [community driver](https://github.com/derekhe/waveshare-7inch-touchscreen-driver) provides support.

The key is ensuring `hid-multitouch` kernel module is available:

```bash
# Check if module is loaded
lsmod | grep hid_multitouch

# Load if needed
sudo modprobe hid-multitouch

# Verify touch device
cat /proc/bus/input/devices | grep -A5 "eGalax\|waveshare"
evtest /dev/input/event0  # Test touch input
```

### Keystation UI Layout

```fsharp
// Keystation main interface
let keystationView model dispatch =
    Grid(
        coldefs = [Star 1.; Star 2.],
        rowdefs = [Pixel 60.; Star 1.; Pixel 40.]
    ) {
        // Header
        Label("Keystation")
            .gridColumn(0).gridColumnSpan(2)
            .fontSize(28.)
            .horizontalAlignment(Center)
            .backgroundColor(Color.DarkBlue)
            .textColor(Color.White)

        // Navigation sidebar
        VStack(spacing = 8.) {
            NavigationButton("Credentials", CredentialsView)
            NavigationButton("Settings", SettingsView)
            NavigationButton("About", AboutView)
        }
        .gridRow(1).gridColumn(0)
        .padding(8.)

        // Content area
        ContentView(model.CurrentView, model, dispatch)
            .gridRow(1).gridColumn(1)

        // Status bar
        Label($"Connected: {model.DeviceCount} devices")
            .gridRow(2).gridColumnSpan(2)
            .fontSize(12.)
            .backgroundColor(Color.LightGray)
    }
```

### Compilation Target

```bash
firefly compile Keystation.fidproj --target aarch64-none-elf
```

The generated binary:
- Links against LVGL (compiled for AArch64)
- Uses Linux framebuffer for display
- Uses evdev for touch input
- Runs as a unikernel process (no desktop environment)

## Target 2: Desktop Companion App

### Purpose

A Linux x86_64 application for:
- Demonstrating QuantumCredential operations from a desktop
- Testing credential read/write workflows
- Providing a "client perspective" during demos

### LVGL Backend Configuration

For desktop, LVGL uses SDL2 as the display/input backend:

```c
// LVGL SDL2 configuration for desktop
#define LV_USE_SDL 1
#define LV_SDL_INCLUDE_PATH <SDL2/SDL.h>
```

### Desktop UI Layout

```fsharp
// Desktop companion app - same MVU pattern
let desktopView model dispatch =
    Window(
        "QuantumCredential Demo",
        width = 800,
        height = 600
    ) {
        VStack(spacing = 16.) {
            // Connection status panel
            Card {
                HStack(spacing = 12.) {
                    StatusIndicator(model.IsConnected)
                    VStack {
                        Label("QuantumCredential")
                            .fontSize(18.)
                        Label(
                            if model.IsConnected then "Connected"
                            else "Not connected"
                        )
                        .textColor(
                            if model.IsConnected then Color.Green
                            else Color.Red
                        )
                    }
                    Spacer()
                    Button(
                        if model.IsConnected then "Disconnect" else "Connect",
                        fun () -> dispatch (if model.IsConnected then Disconnect else Connect)
                    )
                }
            }
            .padding(16.)

            // Operations panel
            if model.IsConnected then
                Card {
                    VStack(spacing = 12.) {
                        Label("Credential Operations")
                            .fontSize(16.)
                            .fontWeight(Bold)

                        HStack(spacing = 8.) {
                            Button("Read", fun () -> dispatch ReadCredential)
                                .buttonStyle(Primary)
                            Button("Write", fun () -> dispatch WriteCredential)
                                .buttonStyle(Secondary)
                            Button("Verify", fun () -> dispatch VerifyCredential)
                                .buttonStyle(Secondary)
                        }

                        // Result display
                        match model.LastResult with
                        | Some result ->
                            ResultCard(result)
                        | None ->
                            Label("No operation performed yet")
                                .textColor(Color.Gray)
                    }
                }
                .padding(16.)

            // Log panel
            ScrollView {
                VStack {
                    for entry in model.Log do
                        LogEntry(entry)
                }
            }
            .height(200.)
        }
        .padding(16.)
    }
```

### Compilation Target

```bash
firefly compile DesktopDemo.fidproj --target x86_64-unknown-linux-gnu
```

The generated binary:
- Links against LVGL + SDL2
- Runs as a standard desktop application
- Communicates with QuantumCredential via USB HID

## Shared UI Components

The power of FidelityUI is code reuse. Common widgets compile to LVGL regardless of target:

```fsharp
// Shared module - used by both Keystation and Desktop
module CredentialUI =

    /// Status indicator with color coding
    let StatusIndicator isConnected =
        Circle(radius = 8.)
            .fill(if isConnected then Color.Green else Color.Red)
            .shadow(if isConnected then Shadow.Glow(Color.Green) else Shadow.None)

    /// Credential card display
    let CredentialCard (cred: CredentialData) dispatch =
        Card {
            VStack(spacing = 8.) {
                HStack {
                    Label(cred.Name)
                        .fontWeight(Bold)
                    Spacer()
                    Label(cred.Type.ToString())
                        .textColor(Color.Gray)
                        .fontSize(12.)
                }

                Label($"ID: {cred.Id}")
                    .fontSize(12.)
                    .textColor(Color.DarkGray)

                HStack(spacing = 4.) {
                    Button("Use", fun () -> dispatch (UseCredential cred.Id))
                        .buttonStyle(Primary)
                    Button("Details", fun () -> dispatch (ShowDetails cred.Id))
                        .buttonStyle(Secondary)
                }
            }
        }
        .padding(12.)

    /// Operation result display
    let ResultCard (result: OperationResult) =
        let (bgColor, icon) =
            match result.Status with
            | Success -> (Color.LightGreen, "✓")
            | Failure -> (Color.LightRed, "✗")
            | Pending -> (Color.LightYellow, "⋯")

        Card {
            HStack(spacing = 8.) {
                Label(icon).fontSize(24.)
                VStack {
                    Label(result.Operation)
                        .fontWeight(Bold)
                    Label(result.Message)
                        .fontSize(12.)
                }
            }
        }
        .backgroundColor(bgColor)
        .padding(8.)
```

## Technical Challenges

### Challenge 1: Touch Calibration on Sweet Potato

The WaveShare display may need touch calibration. Without a full Linux desktop, this requires:

1. Using `evtest` to verify touch coordinates
2. If needed, applying coordinate transformation in LVGL:

```c
// LVGL touch coordinate transformation if needed
static void touch_transform(lv_indev_data_t *data) {
    // Apply calibration matrix if touch is offset
    int32_t x = data->point.x;
    int32_t y = data->point.y;

    // Example: flip Y axis if display is rotated
    data->point.y = LV_VER_RES - y;
}
```

### Challenge 2: Framebuffer Setup Without Desktop

The Sweet Potato needs to boot directly to framebuffer mode:

```bash
# /etc/rc.local or systemd service
# Disable console blanking
echo 0 > /sys/class/graphics/fb0/blank

# Set framebuffer resolution (if needed)
fbset -xres 1024 -yres 600 -depth 32

# Start Keystation application
/opt/keystation/keystation &
```

### Challenge 3: USB Communication with QuantumCredential

Both applications need to communicate with the QuantumCredential device via USB HID. This requires:

1. **Desktop**: libusb or hidapi bindings via Farscape
2. **Keystation**: Same, running on embedded Linux

```fsharp
// USB HID communication - Platform.Bindings pattern (BCL-free)
// Farscape generates from hidapi headers, Alex provides MLIR emission
module Platform.Bindings.HID =
    /// Open HID device by vendor/product ID
    let hidOpen (vendorId: uint16) (productId: uint16) : int =
        Unchecked.defaultof<int>

    /// Write to HID device
    let hidWrite (device: int) (data: nativeint) (length: int) : int =
        Unchecked.defaultof<int>

    /// Read from HID device with timeout
    let hidReadTimeout (device: int) (data: nativeint) (length: int) (timeout: int) : int =
        Unchecked.defaultof<int>

// Higher-level wrapper using Platform.Bindings
module QuantumCredentialUSB =
    open Platform.Bindings.HID

    let connect () : Result<Device, Error> =
        let handle = hidOpen VENDOR_ID PRODUCT_ID
        if handle > 0 then Ok { Handle = handle }
        else Error DeviceNotFound

    let sendCommand (device: Device) (cmd: Command) : Result<Response, Error> =
        use buffer = stackalloc<byte> 64
        serializeCommand cmd buffer
        let written = hidWrite device.Handle (NativePtr.toNativeInt buffer) 64
        if written < 0 then Error WriteFailed
        else
            let read = hidReadTimeout device.Handle (NativePtr.toNativeInt buffer) 64 1000
            if read < 0 then Error ReadFailed
            else Ok (parseResponse buffer read)
```

## Implementation Phases

### Phase A: LVGL Integration Foundation

1. **Farscape bindings for LVGL** - Generate F# bindings from `lvgl.h`
2. **Basic widget wrappers** - Label, Button, Container
3. **Event handling** - Touch/click events → MVU dispatch
4. **Build system** - Link LVGL into Fidelity output

### Phase B: Platform Backends

1. **SDL2 backend** for desktop development/testing
2. **Framebuffer backend** for embedded Linux
3. **Input device handling** - Mouse, keyboard, touch

### Phase C: FidelityUI Core

1. **MVU runtime** - Model/View/Update loop
2. **Virtual DOM diffing** - Efficient updates
3. **Layout engine** - Map F# layouts to LVGL flex/grid

### Phase D: Demo Applications

1. **Desktop companion** - Full CRUD operations
2. **Keystation UI** - Touch-optimized interface
3. **Integration testing** - End-to-end workflows

## Fallback Strategy

If full FidelityUI isn't ready for demo day:

### Option 1: .NET Avalonia Desktop App

Use standard Avalonia.FuncUI for the desktop companion:
- Proven, working solution
- Same MVU pattern
- Demonstrates the "intent" of FidelityUI

### Option 2: Direct LVGL C Application

Write the Keystation UI directly in C with LVGL:
- Proves the hardware works
- Shows the target output of FidelityUI
- Can be replaced with F# version later

### Option 3: TUI (Terminal UI)

Use simple text-based interface for demos:
- Works everywhere
- Zero risk
- Less impressive but functional

## Success Criteria

The stretch goal is **successful** if:

1. **Keystation** displays a graphical UI on the WaveShare screen
2. **Touch input** works for basic navigation
3. **Desktop app** can initiate QuantumCredential operations
4. **Both apps** share significant F# code

The stretch goal is **partially successful** if:

1. Desktop app works with FidelityUI
2. Keystation uses fallback (C/LVGL or TUI)

The stretch goal is **deferred** if:

1. Both use fallback solutions
2. FidelityUI architecture documented but not implemented

## References

### SpeakEZ Blog Posts
- "Building User Interfaces with the Fidelity Framework" - FidelityUI architecture
- "Scaling FidelityUI: The Actor Model" - Olivier/Prospero actor system
- "Leveraging Fabulous for Native UI" - Fabulous → FidelityUI translation
- "A Window Layout System for Fidelity" - Layout engine design

### External Resources
- [LVGL Documentation](https://docs.lvgl.io/master/) - Official LVGL docs
- [LVGL Linux Framebuffer Driver](https://docs.lvgl.io/master/details/integration/embedded_linux/drivers/fbdev.html)
- [LVGL Evdev Touch Driver](https://docs.lvgl.io/9.2/integration/driver/touchpad/evdev.html)
- [lv_port_linux](https://github.com/lvgl/lv_port_linux) - LVGL Linux port
- [WaveShare 7" Display Wiki](https://www.waveshare.com/wiki/7inch_HDMI_LCD_(C))
- [WaveShare Touch Driver](https://github.com/derekhe/waveshare-7inch-touchscreen-driver)

### Related Firefly Documentation
- `docs/Hardware_Showcase_Roadmap.md` - Full hardware target vision
- `docs/Farscape_GIR_Integration.md` - GTK4 binding generation via GIR
- `docs/Architecture_Canonical.md` - Fidelity layer architecture
