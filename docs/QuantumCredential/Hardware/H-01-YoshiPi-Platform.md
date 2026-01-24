# YoshiPi Hardware Architecture

> **Platform**: Raspberry Pi Zero 2 W on YoshiPi carrier board
>
> **OS**: Raspberry Pi OS (Debian-based) or minimal Buildroot Linux

---

## Hardware Overview

### Raspberry Pi Zero 2 W

| Specification | Value |
|---------------|-------|
| **Processor** | Broadcom BCM2710A1, quad-core Cortex-A53 (64-bit) @ 1GHz |
| **Memory** | 512MB LPDDR2 SDRAM |
| **Connectivity** | 2.4GHz 802.11 b/g/n Wi-Fi, Bluetooth 4.2 BLE |
| **GPIO** | 40-pin header (directly on YoshiPi carrier) |
| **USB** | Micro USB OTG |

### YoshiPi Carrier Board

| Feature | Specification | Demo Use |
|---------|---------------|----------|
| **Analog Inputs** | 4x 10-bit ADC channels | Avalanche circuit input |
| **GPIO** | 10 general purpose I/O | Status LEDs, control |
| **Relays** | 2x dry contact | Not needed for demo |
| **Display** | Integrated touchscreen | Monitor UI |
| **Proto Area** | Breadboard-style prototyping | Avalanche circuit mount |
| **Expansion** | Mikrobus, Grove I2C, Qwiic I2C | Future sensors |

---

## Avalanche Circuit Integration

### Quad-Channel Architecture

The YoshiPi carrier provides 4 ADC channels, enabling a quad avalanche circuit design. Four independent noise sources feed four ADC inputs, allowing parallel entropy sampling that maps directly to the Pi Zero 2 W's quad-core processor.

```
┌─────────────────────────────────────────────────────────────────────┐
│  YoshiPi Carrier Board                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ Proto Area: Quad Avalanche Circuit                      │  │  │
│  │  │                                                         │  │  │
│  │  │   ┌───┐    ┌───┐    ┌───┐    ┌───┐                     │  │  │
│  │  │   │Z0 │    │Z1 │    │Z2 │    │Z3 │  Avalanche Diodes   │  │  │
│  │  │   └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘                     │  │  │
│  │  │     │        │        │        │                        │  │  │
│  │  │   ┌─▼─┐    ┌─▼─┐    ┌─▼─┐    ┌─▼─┐                     │  │  │
│  │  │   │Amp│    │Amp│    │Amp│    │Amp│  Quad Op-Amp        │  │  │
│  │  │   └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘                     │  │  │
│  │  │     │        │        │        │                        │  │  │
│  │  └─────┼────────┼────────┼────────┼────────────────────────┘  │  │
│  │        │        │        │        │                            │  │
│  │        ▼        ▼        ▼        ▼                            │  │
│  │   ┌────────┬────────┬────────┬────────┐                       │  │
│  │   │ ADC 0  │ ADC 1  │ ADC 2  │ ADC 3  │  4x 10-bit channels   │  │
│  │   └────────┴────────┴────────┴────────┘                       │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Parallel Sampling

The quad avalanche architecture enables true parallel execution on the quad-core processor. Sampling four independent noise sources is ideal for `scf.parallel`:

| Characteristic | Implication |
|----------------|-------------|
| **Referentially transparent** | Each channel's sample is independent |
| **No cross-channel dependencies** | No operation needs another's result |
| **Pure data transformation** | Voltage reading to entropy bits |
| **Perfect core mapping** | 4 ADC channels to 4 Cortex-A53 cores |

The Firefly compiler emits `scf.parallel` for this pattern, generating code that executes simultaneously across all four cores with no synchronization overhead. See [03_MLIR_Dialect_Strategy.md](./03_MLIR_Dialect_Strategy.md) for details on the standard MLIR dialect approach.

### Interleaved Entropy

Rather than treating four channels as a simple throughput multiplier, interleaving bits from independent quantum sources improves statistical properties:

```
Channel 0: a₁ a₂ a₃ a₄ ...
Channel 1: b₁ b₂ b₃ b₄ ...
Channel 2: c₁ c₂ c₃ c₄ ...
Channel 3: d₁ d₂ d₃ d₄ ...

Interleaved: a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ ...
```

This breaks any temporal autocorrelation within a single circuit's output stream. The interleaving operation itself is also pure and parallelizable.

### Single-Channel Avalanche (Simplified Reference)

Each of the four channels uses an identical avalanche circuit topology:

```
     3.3V
      │
      ├──┬──────────────────────────────────────┐
      │  │                                      │
     [R1]│                                     [R4]
      │  │                                      │
      ├──┼───► Stage 1 Noise ───► Amp ───┐     │
      │  │                               │     │
     ┌┴┐ │                               ▼     │
     │Z│ ▼                           ┌───────┐ │
     │ │ Avalanche                   │ Final │ │
     └┬┘ Diode                       │ Stage │─┴──► To ADC
      │                               │ Amp   │
      ▼                               └───────┘
     GND
```

The quad op-amp (e.g., TL074, LM324) provides four matched amplifier stages. Zener diodes and capacitors are commodity parts, making the quad circuit only marginally more complex than a single-channel design.

The amplified noise signals (0-3.3V range) connect to Analog Inputs 0-3.

---

## Linux Device Access

### ADC via Industrial I/O (IIO) Subsystem

The YoshiPi's ADC appears under the Linux IIO subsystem:

```
/sys/bus/iio/devices/iio:device0/
├── name                        # "yoshipi-adc" or similar
├── in_voltage0_raw             # Channel 0 raw value (0-1023)
├── in_voltage1_raw             # Channel 1 raw value
├── in_voltage2_raw             # Channel 2 raw value
├── in_voltage3_raw             # Channel 3 raw value
├── in_voltage_scale            # Scale factor for mV conversion
├── sampling_frequency          # Current sample rate
└── sampling_frequency_available # Available sample rates
```

**Reading ADC:**
```bash
# Read single sample
cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw
# Returns: 512 (example value, 0-1023 range)

# Read scale
cat /sys/bus/iio/devices/iio:device0/in_voltage_scale
# Returns: 3.222656 (mV per LSB)
```

**High-Speed Sampling:**

For continuous sampling, use the IIO buffer interface:
```
/dev/iio:device0                # Character device for buffered access
```

Or trigger-based sampling via:
```
/sys/bus/iio/devices/iio:device0/buffer/
├── enable                      # 1 to start, 0 to stop
├── length                      # Buffer length in samples
└── watermark                   # Watermark for poll/select
```

### GPIO via gpiochip

The Pi's GPIO appears as:
```
/dev/gpiochip0                  # Main GPIO controller
```

**GPIO ioctl interface:**
```c
// Structures (for reference - Alex generates these)
struct gpioline_info {
    uint32_t line_offset;
    uint32_t flags;
    char name[32];
    char consumer[32];
};

struct gpiohandle_request {
    uint32_t lineoffsets[64];
    uint32_t flags;
    uint8_t default_values[64];
    char consumer_label[32];
    uint32_t lines;
    int fd;
};

// ioctl commands
#define GPIO_GET_LINEINFO_IOCTL    _IOWR(0xB4, 0x02, struct gpioline_info)
#define GPIO_GET_LINEHANDLE_IOCTL  _IOWR(0xB4, 0x03, struct gpiohandle_request)
#define GPIOHANDLE_GET_LINE_VALUES_IOCTL _IOWR(0xB4, 0x08, struct gpiohandle_data)
#define GPIOHANDLE_SET_LINE_VALUES_IOCTL _IOWR(0xB4, 0x09, struct gpiohandle_data)
```

---

## Display Subsystem

### Option 1: WebKitGTK (Preferred)

If GTK and WebKitGTK are available on the Pi Zero 2 W image:

```fsharp
// Same pattern as desktop
let w = Webview.create true
Webview.setTitle w "QC Monitor"
Webview.setSize w 480 320  // Match display resolution
Webview.setHtml w EmbeddedAssets.MonitorHtml
Webview.run w
```

**Considerations:**
- Pi Zero 2 W has limited RAM (512MB)
- WebKitGTK is memory-hungry
- May need minimal X11/Wayland setup
- Consider lightweight alternatives if memory is tight

### Option 2: Framebuffer Direct (Fallback)

If WebKitGTK is too heavy:

```
/dev/fb0                        # Framebuffer device
```

**Framebuffer info:**
```c
struct fb_var_screeninfo {
    uint32_t xres, yres;        // Visible resolution
    uint32_t bits_per_pixel;    // 16, 24, or 32
    // ...
};
```

Can render directly with memory-mapped writes. Would need custom UI rendering (not shared with desktop).

### Option 3: DRM/KMS (Modern)

```
/dev/dri/card0                  # DRM device
```

More complex but better performance. Still not shared with desktop UI.

**Recommendation:** Try WebKitGTK first. If memory is too tight, fall back to a simple framebuffer status display.

---

## USB Gadget Mode

The Pi Zero 2 W supports USB OTG, enabling gadget mode for credential transfer:

```
# Configure USB gadget (CDC ACM - serial port)
modprobe libcomposite
# ... gadget configuration ...
```

Result:
```
/dev/ttyGS0                     # USB gadget serial port (device side)
```

On the desktop, this appears as `/dev/ttyACM0` or similar.

**Credential Transfer Flow:**
```
YoshiPi                         Desktop
   │                               │
   │ /dev/ttyGS0                   │ /dev/ttyACM0
   │     │                         │     │
   └─────┼─────── USB Cable ───────┼─────┘
         │                         │
    write(fd, cred)           read(fd, buf)
```

---

## Memory Map

### Process Memory (Linux User Space)

```
┌─────────────────────────────────────────────────────────────────────┐
│  User Space Process (Firefly-compiled binary)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stack (grows down)                                           │   │
│  │ • Local variables                                            │   │
│  │ • NativeArray.stackalloc buffers                             │   │
│  │ • Function call frames                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│                       (grows)                                       │
│                                                                     │
│                       (grows)                                       │
│                          ▲                                          │
│                          │                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Heap (optional - stack_only mode avoids this)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ mmap regions                                                 │   │
│  │ • /dev/fb0 framebuffer (if used)                             │   │
│  │ • Shared memory for IPC (if needed)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Data segment                                                 │   │
│  │ • Embedded HTML/JS/CSS (WebView assets)                      │   │
│  │ • Static string constants                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Text segment (read-only, executable)                         │   │
│  │ • Compiled F# code                                           │   │
│  │ • PQC algorithm implementations                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Power Considerations

### Pi Zero 2 W Power Budget

| Component | Typical Draw |
|-----------|-------------|
| Pi Zero 2 W (idle) | ~100mA @ 5V |
| Pi Zero 2 W (load) | ~300mA @ 5V |
| Display | ~50-100mA |
| Avalanche circuit | ~10mA |
| **Total** | ~400-500mA @ 5V |

**Power Source Options:**
- USB power (standard micro USB)
- Battery pack (portable demo)
- Bench supply (lab demo)

---

## Software Stack

### Minimal Linux Image

For demo, consider a minimal image:

```
Base: Raspberry Pi OS Lite (no desktop)
Add:
  - WebKitGTK (if used for UI)
  - GTK3 dependencies
  - libiio (for ADC access)
  - libgpiod (for GPIO)

Remove:
  - Desktop environment
  - Unnecessary services
  - Development tools
```

Target image size: < 1GB

### Required Kernel Modules

```
# ADC
iio-core
yoshipi-adc  # Or whatever the carrier's ADC driver is

# GPIO
gpio-core

# USB Gadget
libcomposite
usb_f_acm

# Display (if using framebuffer)
fb
drm
vc4
```

---

## Cross-References

- [01_YoshiPi_Demo_Strategy](./01_YoshiPi_Demo_Strategy.md) - Overall strategy
- [03_MLIR_Dialect_Strategy](./03_MLIR_Dialect_Strategy.md) - Compilation path and parallel execution
- [04_Linux_Hardware_Bindings](./04_Linux_Hardware_Bindings.md) - Fidelity.Platform bindings for Linux hardware
- [05_PostQuantum_Architecture](./05_PostQuantum_Architecture.md) - PQC algorithms
