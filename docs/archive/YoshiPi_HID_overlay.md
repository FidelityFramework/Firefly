# USB HID Prototype Development Specification

## Raspberry Pi Zero 2 W as a USB Peripheral Device

### Executive Summary

This document describes the architecture and development workflow for prototyping a USB Human Interface Device (HID) using a Raspberry Pi Zero 2 W on a YoshiPi carrier board. The device will communicate with host systems using the BAREWire binary protocol, demonstrating a principled approach to hardware security token design. The configuration leverages Linux's USB gadget framework to present the Pi as a composite USB device, combining a custom HID interface for primary functionality with a CDC ACM serial interface for development telemetry. The YoshiPi's integrated ILI9341 display provides on-device visual feedback during development and demonstration.

### Background and Motivation

#### The Problem with Abstraction Frameworks

Development frameworks like Wilderness Labs Meadow provide valuable abstractions for IoT development. They offer managed runtimes, hardware abstraction layers, and integrated toolchains that simplify the path from prototype to production. However, these conveniences come with costs that may be unacceptable for certain classes of applications.

For security devices, the abstraction stack represents attack surface. Each layer between your code and the hardware introduces complexity that must be audited, maintained, and trusted. The Meadow stack, for instance, layers a .NET runtime atop a custom RTOS, with hardware drivers mediated through the Meadow.Foundation abstraction. This is appropriate for general IoT applications where developer velocity outweighs security concerns.

For a hardware security module or authentication device, the calculus differs. The fewer moving parts, the better. A direct relationship between application code and kernel interfaces reduces the surface area that requires formal verification and security audit.

#### The Pi as "Just Another Linux System"

The Raspberry Pi Zero 2 W, when configured as described in this document, becomes indistinguishable from any other Linux system running on ARM hardware. This is a feature, not a limitation.

The implications are significant:

1. **Toolchain independence.** Any language or runtime that targets Linux ARM64 can be used. F#, Rust, C, Go; the choice is yours. No vendor lock-in to a specific SDK or deployment mechanism.

2. **Standard debugging.** GDB, strace, perf, and every other Linux diagnostic tool works normally. No specialized debugger protocols or IDE plugins required.

3. **Deployment simplicity.** Copying files via SCP or rsync constitutes deployment. No firmware flashing, no bootloader negotiations, no proprietary update mechanisms.

4. **Reproducibility.** The entire system can be version-controlled as configuration files and scripts. Rebuilding from scratch is a matter of running a shell script against a stock Raspberry Pi OS image.

Compare this to the Meadow deployment model, which requires the Meadow CLI, specific Visual Studio extensions, and a managed connection to the device for code deployment. The Meadow approach optimizes for a specific developer experience; the Linux approach optimizes for transparency and control.

---

## Part 1: YoshiPi Hardware Setup

### Hardware Overview

The YoshiPi is a carrier board for the Raspberry Pi Zero 2 W featuring:

- ILI9341 3.2" 320x240 TFT SPI display with touchscreen
- MCP23008 I/O expander for backlight control and additional GPIO
- DS3231 RTC on I2C bus 1
- 4 analog inputs via ADC
- 2 dry contact relays
- Mikrobus, Grove I2C, and Qwiic I2C connectors
- Proto-board area for custom circuits

#### Display Hardware Mapping

| Signal | GPIO | Notes |
|--------|------|-------|
| SPI MOSI | GPIO10 | Standard SPI0 |
| SPI SCLK | GPIO11 | Standard SPI0 |
| Chip Select | GPIO4 | Non-standard; requires overlay |
| Data/Command | GPIO23 | Display command/data select |
| Reset | GPIO24 | Display hardware reset |
| Backlight | MCP23008 GP4 | Via I/O expander |

#### I2C Device Map

| Bus | Address | Device |
|-----|---------|--------|
| 1 | 0x20 | MCP23008 I/O Expander |
| 1 | 0x68 | DS3231 RTC |
| 2 | 0x37, 0x3a | Touchscreen controller |
| 2 | 0x4a, 0x4b | ADC/IO |
| 2 | 0x50 | EEPROM |

### Initial Boot Configuration

#### Prerequisites

- Raspberry Pi Zero 2 W
- YoshiPi carrier board (fully assembled)
- MicroSD card (32GB recommended) with Raspberry Pi OS (Bookworm or later)
- External HDMI monitor via mini-HDMI adapter
- Bluetooth keyboard and mouse (or USB via OTG adapter)
- WiFi network credentials

#### First Boot Procedure

1. Flash Raspberry Pi OS to the SD card using Raspberry Pi Imager
2. Insert SD card into the Pi Zero 2 W mounted on the YoshiPi
3. Connect external HDMI monitor to the mini-HDMI port
4. Connect power to the PWR IN micro-USB port
5. Complete the Raspberry Pi OS setup wizard:
   - Set locale and timezone
   - Configure WiFi network
   - Create user account
   - Allow system updates to complete
6. Pair Bluetooth keyboard and mouse through the desktop interface

#### Enable Required Interfaces

Open a terminal and run:

```bash
sudo raspi-config
```

Navigate to **Interface Options** and enable:

- **I2C** - Required for MCP23008 backlight control and touchscreen
- **SPI** - Required for display communication

Reboot when prompted.

#### Verify Interface Configuration

After reboot, confirm the interfaces are enabled:

```bash
# Check I2C buses
ls /dev/i2c*
# Expected: /dev/i2c-1 /dev/i2c-2

# Check SPI
ls /dev/spi*
# Expected: /dev/spidev0.0 /dev/spidev0.1

# Scan I2C bus 1 for MCP23008 and RTC
sudo i2cdetect -y 1
# Expected: 0x20 (MCP23008), 0x68 (RTC)
```

---

## Part 2: USB Gadget Configuration

### Physical Configuration

The Raspberry Pi Zero 2 W presents two micro-USB ports:

| Port | Label | Function |
|------|-------|----------|
| Edge | PWR | Power input only; no data lines |
| Center | USB | OTG-capable data port with power input |

For this application, the center USB port serves dual duty. When connected to a host system, it:

1. Draws power from the host USB bus (eliminating the need for separate power)
2. Presents as a composite USB device to the host

During development with the display active, connect power to both ports: PWR IN for adequate current, USB for data communication.

### USB Composite Gadget Architecture

Linux's USB gadget framework, accessed through ConfigFS, allows a single USB device controller to present multiple logical devices to the host. This prototype uses a composite gadget with two functions:

**Function 1: Custom HID**

The Human Interface Device class was designed for keyboards, mice, and similar input devices, but its specification accommodates arbitrary data exchange through vendor-defined usage pages. HID offers several advantages for security token applications:

- No driver installation required on any major operating system
- Well-defined interrupt transfer semantics with guaranteed latency bounds
- Bidirectional communication through input and output reports
- Feature reports available for larger, polled data transfers

The HID descriptor defines a vendor-specific device with 64-byte input and output reports. This payload size aligns with USB full-speed interrupt endpoint maximums and provides sufficient bandwidth for authentication protocols.

**Function 2: CDC ACM (Virtual Serial Port)**

The Communications Device Class Abstract Control Model presents as a standard serial port to the host operating system. On Linux hosts, this appears as `/dev/ttyACMx`; on Windows, as a COM port; on macOS, as `/dev/tty.usbmodemXXXX`.

This interface serves exclusively for development and debugging:

- Real-time telemetry from the device application
- Log output during protocol exchanges
- Interactive debugging console when needed
- Performance metrics and diagnostic data

### Critical Configuration Details

The Raspberry Pi Zero 2 W requires specific configuration to operate in USB peripheral mode. The default Raspberry Pi OS image ships with settings that force USB host mode.

#### config.txt Configuration

Edit `/boot/firmware/config.txt`. The `dtoverlay=dwc2,dr_mode=peripheral` line must appear in the `[all]` section, not under device-specific sections.

Locate the end of the file and ensure it contains:

```ini
[cm4]
otg_mode=0

[cm5]

[all]
dtoverlay=dwc2,dr_mode=peripheral
```

If `otg_mode=1` appears anywhere, change it to `otg_mode=0` or remove it.

#### cmdline.txt Configuration

Edit `/boot/firmware/cmdline.txt` and add `modules-load=dwc2` after `rootwait`:

```
console=serial0,115200 console=tty1 root=PARTUUID=XXXXXXXX-02 rootfstype=ext4 fsck.repair=yes rootwait modules-load=dwc2 quiet splash plymouth.ignore-serial-consoles
```

**Critical:** The entire cmdline.txt must remain on a single line.

#### USB Gadget Setup Script

Create the gadget configuration script:

```bash
sudo nano /usr/local/bin/usb-gadget-setup.sh
```

Contents:

```bash
#!/bin/bash
# USB Composite Gadget: HID + CDC ACM

modprobe libcomposite

cd /sys/kernel/config/usb_gadget/
mkdir -p yoshipi
cd yoshipi

echo 0x1d6b > idVendor  # Linux Foundation
echo 0x0104 > idProduct # Multifunction Composite Gadget
echo 0x0100 > bcdDevice
echo 0x0200 > bcdUSB

mkdir -p strings/0x409
echo "SpeakEZ" > strings/0x409/manufacturer
echo "BAREWire HID" > strings/0x409/product
echo "000001" > strings/0x409/serialnumber

# HID function
mkdir -p functions/hid.usb0
echo 1 > functions/hid.usb0/protocol
echo 1 > functions/hid.usb0/subclass
echo 64 > functions/hid.usb0/report_length

# Vendor-defined HID descriptor: 64-byte input, 64-byte output
echo -ne '\x06\x00\xff\x09\x01\xa1\x01\x09\x02\x15\x00\x26\xff\x00\x75\x08\x95\x40\x81\x02\x09\x03\x15\x00\x26\xff\x00\x75\x08\x95\x40\x91\x02\xc0' > functions/hid.usb0/report_desc

# CDC ACM function (serial)
mkdir -p functions/acm.usb0

# Configuration
mkdir -p configs/c.1/strings/0x409
echo "HID + Serial" > configs/c.1/strings/0x409/configuration
echo 250 > configs/c.1/MaxPower

ln -s functions/hid.usb0 configs/c.1/
ln -s functions/acm.usb0 configs/c.1/

ls /sys/class/udc > UDC
```

Make executable:

```bash
sudo chmod +x /usr/local/bin/usb-gadget-setup.sh
```

#### Systemd Service for USB Gadget

Create the service file:

```bash
sudo nano /etc/systemd/system/usb-gadget.service
```

Contents:

```ini
[Unit]
Description=USB Composite Gadget
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/usb-gadget-setup.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable usb-gadget.service
```

#### Verification

Reboot the system:

```bash
sudo reboot
```

After reboot, verify USB gadget operation:

```bash
# Confirm dwc2 module loaded
lsmod | grep dwc2

# Confirm UDC available
ls /sys/class/udc/
# Expected: 3f980000.usb

# Check gadget interfaces exist
ls /dev/hidg0
ls /dev/ttyGS0
```

On the host system:

```bash
# Linux host
lsusb | grep "Linux Foundation"
# Expected: ID 1d6b:0104 Linux Foundation Multifunction Composite Gadget

ls /dev/ttyACM*
# Expected: /dev/ttyACM0

ls /dev/hidraw*
# Note the new hidraw device
```

---

## Part 3: ILI9341 Display Configuration

The YoshiPi's 3.2" ILI9341 display requires the modern `panel-mipi-dbi` DRM driver. This section documents the complete configuration path.

### Blacklist Legacy Driver

Prevent the deprecated fbtft driver from loading:

```bash
echo "blacklist fb_ili9341" | sudo tee /etc/modprobe.d/blacklist-fbtft.conf
```

### Create Display Firmware Binary

The panel-mipi-dbi driver requires a firmware file containing the ILI9341 initialization sequence. The format consists of a 15-byte magic header, 1-byte version, followed by command sequences.

```bash
sudo python3 << 'EOF'
magic = bytes([0x4D, 0x49, 0x50, 0x49, 0x20, 0x44, 0x42, 0x49, 0, 0, 0, 0, 0, 0, 0])
version = bytes([1])
commands = bytes([
    0x01, 0x00,             # Software reset
    0x00, 0x01, 150,        # Delay 150ms
    0x11, 0x00,             # Sleep out
    0x00, 0x01, 255,        # Delay 255ms
    0x3A, 0x01, 0x55,       # Pixel format: 16bpp RGB565
    0x36, 0x01, 0xE8,       # MADCTL: MY+MX+MV+BGR (landscape, correct orientation)
    0x21, 0x00,             # Display inversion on
    0x13, 0x00,             # Normal display mode
    0x29, 0x00,             # Display on
    0x00, 0x01, 100,        # Delay 100ms
])

with open('/lib/firmware/ili9341.bin', 'wb') as f:
    f.write(magic + version + commands)

print("Created /lib/firmware/ili9341.bin")
EOF
```

### Configure Device Tree Overlays

Edit `/boot/firmware/config.txt` and add the following in the `[all]` section, after the dwc2 overlay:

```ini
# YoshiPi ILI9341 3.2" 320x240 Display
dtoverlay=spi0-cs,cs0_pin=4
dtoverlay=mipi-dbi-spi,spi0-0,speed=32000000
dtparam=compatible=ili9341\0panel-mipi-dbi-spi
dtparam=write-only
dtparam=width=320,height=240
dtparam=reset-gpio=24,dc-gpio=23
```

**Note:** The `spi0-cs` overlay remaps the chip select from the default GPIO8 to GPIO4, which is where the YoshiPi routes the display CS line.

### Configure Driver Auto-Loading

Enable automatic loading of the display driver at boot:

```bash
echo "panel-mipi-dbi" | sudo tee /etc/modules-load.d/panel-mipi-dbi.conf
```

### Create Backlight Initialization Service

The MCP23008 I/O expander controls the display backlight. GPIO17 must be held high to release the MCP23008 from reset, then the backlight pin (GP4) must be configured as output and set high.

Create the initialization script:

```bash
sudo nano /usr/local/bin/yoshipi-display-init.sh
```

Contents:

```bash
#!/bin/bash
# YoshiPi Display Backlight Initialization
# Releases MCP23008 from reset and enables backlight

# Set GPIO17 high to release MCP23008 from reset
gpioset -c 0 17=1 &
GPIOSET_PID=$!
sleep 0.2

# Configure MCP23008: Set GP4 as output (bit 4 = 0 in IODIR register)
i2cset -y 1 0x20 0x00 0xEF

# Enable backlight: Set GP4 high (bit 4 = 1 in GPIO register)
i2cset -y 1 0x20 0x09 0x10

# Keep gpioset running to maintain GPIO17 state
wait $GPIOSET_PID
```

Make executable:

```bash
sudo chmod +x /usr/local/bin/yoshipi-display-init.sh
```

Create the systemd service:

```bash
sudo nano /etc/systemd/system/yoshipi-display.service
```

Contents:

```ini
[Unit]
Description=YoshiPi Display Initialization
After=local-fs.target
Before=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/yoshipi-display-init.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable yoshipi-display.service
```

### Install Required Tools

```bash
sudo apt install -y gpiod i2c-tools
```

### Verification

Reboot the system:

```bash
sudo reboot
```

After reboot, verify display operation:

```bash
# Check driver loaded
lsmod | grep panel

# Check framebuffer exists
ls -la /dev/fb1
# Expected: character device, not regular file

# Check display recognized by X11
DISPLAY=:0 xrandr
# Expected: SPI-1 connected 320x240+1920+0 (or similar)

# Check backlight service running
sudo systemctl status yoshipi-display.service
```

The display should show the desktop extended from the primary HDMI output. Moving the mouse to the right edge of the HDMI display will move it onto the SPI display.

### Display Test

Test the display with a simple colored window:

```bash
DISPLAY=:0 python3 << 'EOF'
import tkinter as tk

root = tk.Tk()
root.geometry("320x240+1920+0")  # Position on SPI display
root.overrideredirect(True)
root.configure(bg='#000080')

label = tk.Label(root, text="Hello from\nFidelity Framework!", 
                 font=("DejaVu Sans", 24, "bold"),
                 fg="white", bg="#000080")
label.pack(expand=True)

root.mainloop()
EOF
```

Press Ctrl+C to exit.

---

## Part 4: System Architecture

### Network Architecture

WiFi connectivity persists independently of the USB gadget configuration. The Pi connects to the local network as any other wireless client, obtaining an IP address via DHCP.

This dual-path architecture provides:

| Path | Purpose | Protocol |
|------|---------|----------|
| WiFi | Administration, deployment, bulk transfers | SSH, SCP, rsync |
| USB HID | Product functionality | BAREWire |
| USB CDC ACM | Development telemetry | Plain text, structured logs |
| SPI Display | On-device visual feedback | DRM framebuffer |

During demonstrations, WiFi can be disabled entirely, presenting the device as a pure USB peripheral with no network capability.

### Development Workflow

The steady-state development loop:

1. **Edit** source code on the development machine using preferred tooling
2. **Build** targeting `linux-arm64` (for 64-bit OS) or `linux-arm` (for 32-bit OS)
3. **Deploy** via SCP or rsync over WiFi to the Pi
4. **Test** the HID interface via USB connection to the same or different host
5. **Monitor** telemetry on the CDC ACM serial interface
6. **Observe** status on the integrated SPI display
7. **Debug** via SSH if deeper inspection is needed

This cycle completes in seconds for incremental changes.

### Debugging Facilities

Standard Linux tools apply:

- **journalctl** for system logs and service status
- **dmesg** for kernel messages including USB gadget events
- **strace** for system call tracing
- **gdb** for interactive debugging (local or remote)

The CDC ACM serial interface provides application-level telemetry without interfering with HID protocol exchanges. A simple terminal emulator (minicom, screen, or picocom) displays real-time output:

```bash
# On host system
screen /dev/ttyACM0 115200
```

---

## Part 5: BAREWire Protocol Integration

### Protocol Overview

BAREWire is a binary protocol designed for zero-copy memory operations, inter-process communication, and network interchange. Its schema-driven approach enables formal verification of message structures and efficient serialization without runtime reflection.

Applying BAREWire to the USB HID transport demonstrates the protocol's versatility beyond its primary IPC use case. The same message definitions and serialization logic operate unchanged across:

- Shared memory between processes on a single host
- USB HID reports between device and host
- WebSocket frames between distributed systems
- Any other byte-oriented transport

### HID Framing Layer

USB HID imposes a 64-byte maximum on interrupt transfer reports. BAREWire messages may exceed this limit. A thin framing layer bridges this constraint.

**Frame Structure (2-byte header, 62-byte payload maximum):**

```
Byte 0: [START:1][END:1][SEQ:6]
Byte 1: [LENGTH:8]
Bytes 2-63: Payload (0-62 bytes)
```

**Flag Semantics:**

| START | END | Meaning |
|-------|-----|---------|
| 1 | 1 | Complete message in single frame |
| 1 | 0 | First fragment of multi-frame message |
| 0 | 0 | Continuation fragment |
| 0 | 1 | Final fragment of multi-frame message |

The 6-bit sequence number (0-63) enables detection of dropped or reordered frames without requiring acknowledgment overhead.

### HID Report Descriptor

The binary HID descriptor defines a vendor-specific device:

| Bytes | Meaning |
|-------|---------|
| `06 00 ff` | Usage Page (Vendor Defined 0xFF00) |
| `09 01` | Usage (Vendor Usage 1) |
| `a1 01` | Collection (Application) |
| `09 02` | Usage (Vendor Usage 2) |
| `15 00` | Logical Minimum (0) |
| `26 ff 00` | Logical Maximum (255) |
| `75 08` | Report Size (8 bits) |
| `95 40` | Report Count (64) |
| `81 02` | Input (Data, Variable, Absolute) |
| `09 03` | Usage (Vendor Usage 3) |
| `15 00` | Logical Minimum (0) |
| `26 ff 00` | Logical Maximum (255) |
| `75 08` | Report Size (8 bits) |
| `95 40` | Report Count (64) |
| `91 02` | Output (Data, Variable, Absolute) |
| `c0` | End Collection |

---

## Part 6: Complete Configuration Reference

### /boot/firmware/config.txt

The complete `[all]` section should contain:

```ini
[all]
# USB Gadget Mode
dtoverlay=dwc2,dr_mode=peripheral

# YoshiPi ILI9341 3.2" 320x240 Display
dtoverlay=spi0-cs,cs0_pin=4
dtoverlay=mipi-dbi-spi,spi0-0,speed=32000000
dtparam=compatible=ili9341\0panel-mipi-dbi-spi
dtparam=write-only
dtparam=width=320,height=240
dtparam=reset-gpio=24,dc-gpio=23
```

### /boot/firmware/cmdline.txt

Single line containing:

```
console=serial0,115200 console=tty1 root=PARTUUID=XXXXXXXX-02 rootfstype=ext4 fsck.repair=yes rootwait modules-load=dwc2 quiet splash plymouth.ignore-serial-consoles
```

### Service Files

| Service | Purpose | File |
|---------|---------|------|
| usb-gadget.service | USB composite device setup | /etc/systemd/system/usb-gadget.service |
| yoshipi-display.service | Display backlight initialization | /etc/systemd/system/yoshipi-display.service |

### Module Configuration

| File | Contents |
|------|----------|
| /etc/modules-load.d/panel-mipi-dbi.conf | `panel-mipi-dbi` |
| /etc/modprobe.d/blacklist-fbtft.conf | `blacklist fb_ili9341` |

### Firmware Files

| File | Purpose |
|------|---------|
| /lib/firmware/ili9341.bin | Display controller initialization sequence |

---

## Part 7: Implementation Checklist

### Phase 1: Initial Setup

- [ ] Flash Raspberry Pi OS (Bookworm or later) to SD card
- [ ] Mount Pi Zero 2 W on YoshiPi carrier board
- [ ] Connect external HDMI monitor
- [ ] Boot and complete OS setup wizard
- [ ] Pair Bluetooth keyboard and mouse
- [ ] Configure WiFi network
- [ ] Enable SSH: `sudo systemctl enable ssh`
- [ ] Enable I2C and SPI via raspi-config
- [ ] Verify I2C devices: `sudo i2cdetect -y 1`
- [ ] Install tools: `sudo apt install -y gpiod i2c-tools`

### Phase 2: USB Gadget Configuration

- [ ] Edit `/boot/firmware/config.txt`:
  - [ ] Ensure `otg_mode=0` (not `otg_mode=1`)
  - [ ] Add `dtoverlay=dwc2,dr_mode=peripheral` under `[all]`
- [ ] Edit `/boot/firmware/cmdline.txt`:
  - [ ] Add `modules-load=dwc2` after `rootwait`
- [ ] Create `/usr/local/bin/usb-gadget-setup.sh`
- [ ] Create `/etc/systemd/system/usb-gadget.service`
- [ ] Enable service: `sudo systemctl enable usb-gadget.service`
- [ ] Reboot and verify:
  - [ ] `lsmod | grep dwc2` shows module
  - [ ] `ls /sys/class/udc/` shows controller
  - [ ] Host sees composite gadget via `lsusb`

### Phase 3: Display Configuration

- [ ] Blacklist legacy driver: `/etc/modprobe.d/blacklist-fbtft.conf`
- [ ] Create firmware: `/lib/firmware/ili9341.bin`
- [ ] Add display overlays to `/boot/firmware/config.txt`
- [ ] Enable driver auto-load: `/etc/modules-load.d/panel-mipi-dbi.conf`
- [ ] Create `/usr/local/bin/yoshipi-display-init.sh`
- [ ] Create `/etc/systemd/system/yoshipi-display.service`
- [ ] Enable service: `sudo systemctl enable yoshipi-display.service`
- [ ] Reboot and verify:
  - [ ] `/dev/fb1` exists as character device
  - [ ] `DISPLAY=:0 xrandr` shows SPI-1 output
  - [ ] Display shows desktop or test pattern

### Phase 4: Application Development

- [ ] Implement BAREWire framing layer (device side)
- [ ] Implement BAREWire framing layer (host side)
- [ ] Define HID report structures
- [ ] Implement device application main loop
- [ ] Implement host detection and connection logic
- [ ] Integrate telemetry output on CDC ACM
- [ ] Implement display status updates
- [ ] End-to-end protocol validation

### Phase 5: Demonstration Preparation

- [ ] Document demonstration procedure
- [ ] Prepare fallback configurations
- [ ] Test WiFi-disabled operation
- [ ] Rehearse demo scenario
- [ ] Package device with appropriate enclosure or mounting

---

## Conclusion

The YoshiPi carrier board with Raspberry Pi Zero 2 W, configured as described in this document, provides an effective platform for prototyping USB peripheral devices with integrated visual feedback. The combination of HID for primary functionality, CDC ACM for development telemetry, and the ILI9341 display for on-device status supports efficient iteration while maintaining protocol separation.

By treating the Pi as a standard Linux system, this approach avoids vendor-specific tooling dependencies and maintains full transparency into the software stack. The same deployment and debugging techniques apply whether the target is a Pi Zero, a Jetson Nano, or any other Linux ARM system.

For security-focused applications where minimizing trusted computing base matters, this directness is a feature. Every layer between application code and hardware is standard, auditable, and replaceable.
