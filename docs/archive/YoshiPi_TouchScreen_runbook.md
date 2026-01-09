# YoshiPi Touchscreen Configuration Runbook

## Overview

This document provides complete configuration instructions for enabling the XPT2046 resistive touchscreen on the YoshiPi platform running Raspberry Pi OS. The touchscreen shares the SPI bus infrastructure with the ILI9341 display but requires a userspace daemon for operation due to non-standard chip select routing through an I2C GPIO expander.

## Hardware Architecture

### Display Subsystem
| Component | Connection | Details |
|-----------|------------|---------|
| ILI9341 Display | SPI0 | 320x240, 3.2", RGB565 |
| Display CS | GPIO4 | Active low |
| Display DC | GPIO23 | Data/Command select |
| Display Reset | GPIO24 | Active low reset |
| Backlight Control | MCP23008 GP4 | Via I2C GPIO expander |

### Touchscreen Subsystem
| Component | Connection | Details |
|-----------|------------|---------|
| XPT2046 Controller | SPI1 | Resistive touch ADC |
| Touch CS | MCP23008 GP7 | Via I2C GPIO expander |
| Touch IRQ | GPIO26 | Active low interrupt |

### I2C GPIO Expander
| Device | Bus | Address | Purpose |
|--------|-----|---------|---------|
| MCP23008 | I2C-1 | 0x20 | GPIO expansion for backlight and touch CS |
| MCP23008 Reset | GPIO17 | Active low reset line |

## Prerequisites

### Required Packages
```bash
sudo apt install -y python3-spidev python3-libgpiod python3-evdev
```

If python3-evdev fails via apt:
```bash
pip3 install evdev --break-system-packages
```

### Required Kernel Modules
- `spi-bcm2835` (SPI controller)
- `pinctrl_mcp23s08` (MCP23008 GPIO driver)

## Configuration

### Boot Configuration (/boot/firmware/config.txt)

Add or verify the following entries:

```ini
# Core interfaces
dtparam=spi=on
dtparam=i2c_arm=on

# Display configuration
dtoverlay=spi0-cs,cs0_pin=4
dtoverlay=mipi-dbi-spi,spi0-0,speed=32000000
dtparam=compatible=ili9341\0panel-mipi-dbi-spi

# Touchscreen SPI bus
dtoverlay=spi1-1cs

# MCP23008 GPIO expander
dtoverlay=mcp23008-gpio
```

### MCP23008 Device Tree Overlay

Create the overlay source file:

```bash
cat << 'EOF' | sudo tee /boot/firmware/overlays/mcp23008-gpio.dtbo.dts
/dts-v1/;
/plugin/;

/ {
    compatible = "brcm,bcm2835";

    fragment@0 {
        target = <&i2c1>;
        __overlay__ {
            #address-cells = <1>;
            #size-cells = <0>;
            status = "okay";

            mcp23008: mcp23008@20 {
                compatible = "microchip,mcp23008";
                reg = <0x20>;
                gpio-controller;
                #gpio-cells = <2>;
                status = "okay";
            };
        };
    };
};
EOF
```

Compile the overlay:

```bash
sudo dtc -@ -I dts -O dtb -o /boot/firmware/overlays/mcp23008-gpio.dtbo \
    /boot/firmware/overlays/mcp23008-gpio.dtbo.dts
```

## Touch Daemon

### Daemon Script (/usr/local/bin/yoshipi-touch-daemon.py)

```python
#!/usr/bin/env python3
"""
YoshiPi XPT2046 Touchscreen Daemon

Reads touch coordinates from XPT2046 via SPI1, controls chip select
via MCP23008 GP7, and injects input events via uinput.

Calibrated for 270-degree display rotation.
"""

import spidev
import gpiod
from gpiod.line import Direction, Value
import time
from evdev import UInput, AbsInfo, ecodes as e

# Screen dimensions (after 270° rotation)
SCREEN_W, SCREEN_H = 320, 240

# Calibration values from corner touch test
# Touch X axis maps to screen Y, Touch Y axis maps to screen X
TX_MIN, TX_MAX = 256, 3850   # Raw touch X range
TY_MIN, TY_MAX = 360, 3855   # Raw touch Y range

# Initialize SPI1 for XPT2046
spi = spidev.SpiDev()
spi.open(1, 0)  # SPI1, CE0
spi.max_speed_hz = 1000000
spi.mode = 0

# Initialize MCP23008 GP7 for touch chip select
chip = gpiod.Chip("/dev/gpiochip1")
cs_req = chip.request_lines(
    consumer="touch_cs",
    config={7: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE)}
)

# Create uinput device for touch event injection
cap = {
    e.EV_KEY: [e.BTN_TOUCH],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(value=0, min=0, max=SCREEN_W, fuzz=0, flat=0, resolution=0)),
        (e.ABS_Y, AbsInfo(value=0, min=0, max=SCREEN_H, fuzz=0, flat=0, resolution=0)),
    ]
}
ui = UInput(cap, name='yoshipi-touch', version=0x1)


def read_touch():
    """Read raw X/Y coordinates from XPT2046."""
    cs_req.set_value(7, Value.INACTIVE)  # CS low (active)
    time.sleep(0.0001)
    
    # XPT2046 command 0xD0: Read X position (12-bit, differential)
    rx = spi.xfer2([0xD0, 0x00, 0x00])
    raw_x = ((rx[1] << 8) | rx[2]) >> 3
    
    # XPT2046 command 0x90: Read Y position (12-bit, differential)
    ry = spi.xfer2([0x90, 0x00, 0x00])
    raw_y = ((ry[1] << 8) | ry[2]) >> 3
    
    cs_req.set_value(7, Value.ACTIVE)  # CS high (inactive)
    return raw_x, raw_y


def scale(val, in_min, in_max, out_min, out_max):
    """Linear interpolation between ranges."""
    if in_max == in_min:
        return out_min
    return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


print("YoshiPi touch daemon running...")
touching = False

try:
    while True:
        raw_x, raw_y = read_touch()
        
        # Valid touch: X > 100 and Y < 4000 (idle state is X=0, Y=4095)
        if raw_x > 100 and raw_y < 4000:
            # Axis swap and mapping for 270° rotation:
            # Touch Y -> Screen X, Touch X -> Screen Y
            screen_x = scale(raw_y, TY_MIN, TY_MAX, 0, SCREEN_W)
            screen_y = scale(raw_x, TX_MIN, TX_MAX, 0, SCREEN_H)
            
            # Clamp to screen bounds
            screen_x = max(0, min(SCREEN_W, screen_x))
            screen_y = max(0, min(SCREEN_H, screen_y))
            
            if not touching:
                ui.write(e.EV_KEY, e.BTN_TOUCH, 1)
                touching = True
            
            ui.write(e.EV_ABS, e.ABS_X, screen_x)
            ui.write(e.EV_ABS, e.ABS_Y, screen_y)
            ui.syn()
        else:
            if touching:
                ui.write(e.EV_KEY, e.BTN_TOUCH, 0)
                ui.syn()
                touching = False
        
        time.sleep(0.02)  # 50Hz polling rate

except KeyboardInterrupt:
    print("\nShutting down")

ui.close()
cs_req.release()
chip.close()
spi.close()
```

### Systemd Service (/etc/systemd/system/yoshipi-touch.service)

```ini
[Unit]
Description=YoshiPi Touchscreen Daemon
After=multi-user.target

[Service]
Type=simple
ExecStartPre=/bin/bash -c 'echo 1-0020 > /sys/bus/i2c/drivers/mcp230xx/bind 2>/dev/null || true'
ExecStartPre=/bin/sleep 0.5
ExecStartPre=/usr/bin/gpioset -z -c 1 4=1
ExecStart=/usr/bin/python3 /usr/local/bin/yoshipi-touch-daemon.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The service performs three startup operations:
1. Binds the MCP23008 I2C device to the GPIO driver (may fail silently if already bound)
2. Waits for the GPIO chip to initialize
3. Enables the display backlight (GP4 high)

## Installation

### Step 1: Install Dependencies

```bash
sudo apt update
sudo apt install -y python3-spidev python3-libgpiod python3-evdev
```

### Step 2: Create MCP23008 Overlay

```bash
# Create overlay source
cat << 'EOF' | sudo tee /boot/firmware/overlays/mcp23008-gpio.dtbo.dts
/dts-v1/;
/plugin/;

/ {
    compatible = "brcm,bcm2835";

    fragment@0 {
        target = <&i2c1>;
        __overlay__ {
            #address-cells = <1>;
            #size-cells = <0>;
            status = "okay";

            mcp23008: mcp23008@20 {
                compatible = "microchip,mcp23008";
                reg = <0x20>;
                gpio-controller;
                #gpio-cells = <2>;
                status = "okay";
            };
        };
    };
};
EOF

# Compile overlay
sudo dtc -@ -I dts -O dtb -o /boot/firmware/overlays/mcp23008-gpio.dtbo \
    /boot/firmware/overlays/mcp23008-gpio.dtbo.dts
```

### Step 3: Update Boot Configuration

```bash
# Add SPI1 and MCP23008 overlay to config.txt
echo "dtoverlay=spi1-1cs" | sudo tee -a /boot/firmware/config.txt
echo "dtoverlay=mcp23008-gpio" | sudo tee -a /boot/firmware/config.txt
```

### Step 4: Install Touch Daemon

```bash
# Copy daemon script
sudo cp yoshipi-touch-daemon.py /usr/local/bin/
sudo chmod +x /usr/local/bin/yoshipi-touch-daemon.py

# Create systemd service
sudo tee /etc/systemd/system/yoshipi-touch.service << 'EOF'
[Unit]
Description=YoshiPi Touchscreen Daemon
After=multi-user.target

[Service]
Type=simple
ExecStartPre=/bin/bash -c 'echo 1-0020 > /sys/bus/i2c/drivers/mcp230xx/bind 2>/dev/null || true'
ExecStartPre=/bin/sleep 0.5
ExecStartPre=/usr/bin/gpioset -z -c 1 4=1
ExecStart=/usr/bin/python3 /usr/local/bin/yoshipi-touch-daemon.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable yoshipi-touch.service
sudo systemctl start yoshipi-touch.service
```

### Step 5: Reboot and Verify

```bash
sudo reboot
```

After reboot:

```bash
# Check service status
sudo systemctl status yoshipi-touch.service

# Verify gpiochip1 exists (MCP23008)
sudo gpiodetect

# Check touch input device
ls /dev/input/event*
cat /proc/bus/input/devices | grep -A5 "yoshipi-touch"
```

## Calibration

If touch accuracy is poor, run the calibration script to obtain new values:

```bash
cat << 'EOF' > ~/touch_calibrate.py
#!/usr/bin/env python3
import spidev
import gpiod
from gpiod.line import Direction, Value
import time

spi = spidev.SpiDev()
spi.open(1, 0)
spi.max_speed_hz = 1000000
spi.mode = 0

chip = gpiod.Chip("/dev/gpiochip1")
cs_req = chip.request_lines(
    consumer="touch_cs",
    config={7: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE)}
)

def read_touch_avg(samples=10):
    xs, ys = [], []
    for _ in range(samples):
        cs_req.set_value(7, Value.INACTIVE)
        time.sleep(0.0001)
        rx = spi.xfer2([0xD0, 0x00, 0x00])
        x = ((rx[1] << 8) | rx[2]) >> 3
        ry = spi.xfer2([0x90, 0x00, 0x00])
        y = ((ry[1] << 8) | ry[2]) >> 3
        cs_req.set_value(7, Value.ACTIVE)
        if x > 100 and y < 4000:
            xs.append(x)
            ys.append(y)
        time.sleep(0.02)
    if xs and ys:
        return sum(xs)//len(xs), sum(ys)//len(ys)
    return None, None

def wait_for_touch():
    print("  Waiting for touch...", end="", flush=True)
    while True:
        x, y = read_touch_avg(5)
        if x is not None:
            print(f" Got it: raw X={x}, Y={y}")
            time.sleep(0.5)
            return x, y
        time.sleep(0.05)

def wait_for_release():
    print("  Release...", end="", flush=True)
    while True:
        cs_req.set_value(7, Value.INACTIVE)
        time.sleep(0.0001)
        rx = spi.xfer2([0xD0, 0x00, 0x00])
        x = ((rx[1] << 8) | rx[2]) >> 3
        cs_req.set_value(7, Value.ACTIVE)
        if x < 100:
            print(" OK")
            time.sleep(0.3)
            return
        time.sleep(0.05)

print("=" * 50)
print("TOUCHSCREEN CALIBRATION")
print("=" * 50)
print("\nTouch each corner when prompted.\n")

corners = {}

for name, label in [('tl', 'TOP-LEFT'), ('tr', 'TOP-RIGHT'), 
                     ('bl', 'BOTTOM-LEFT'), ('br', 'BOTTOM-RIGHT')]:
    print(f"Touch {label} corner of screen")
    corners[name] = wait_for_touch()
    wait_for_release()
    print()

print("=" * 50)
print("RESULTS")
print("=" * 50)
for name, label in [('tl', 'Top-Left'), ('tr', 'Top-Right'),
                    ('bl', 'Bottom-Left'), ('br', 'Bottom-Right')]:
    print(f"{label:14s}: X={corners[name][0]:4d}, Y={corners[name][1]:4d}")

x_min = min(corners['tl'][0], corners['bl'][0])
x_max = max(corners['tr'][0], corners['br'][0])
y_min = min(corners['tl'][1], corners['tr'][1])
y_max = max(corners['bl'][1], corners['br'][1])

print(f"\nUpdate daemon with:")
print(f"TX_MIN, TX_MAX = {x_min}, {x_max}")
print(f"TY_MIN, TY_MAX = {y_min}, {y_max}")

cs_req.release()
chip.close()
spi.close()
EOF

# Stop service before calibrating
sudo systemctl stop yoshipi-touch.service
sudo python3 ~/touch_calibrate.py
```

Update the calibration values in `/usr/local/bin/yoshipi-touch-daemon.py` and restart the service.

## Troubleshooting

### MCP23008 Probe Failure

If `dmesg | grep mcp` shows "probe with driver mcp230xx failed with error -5":

```bash
# Ensure GPIO17 (MCP23008 reset) is high
gpioset -c 0 17=1

# Manually bind the driver
echo "1-0020" | sudo tee /sys/bus/i2c/drivers/mcp230xx/bind

# Verify gpiochip1 appears
sudo gpiodetect
```

### Touch Not Responding

1. Verify SPI1 is available:
   ```bash
   ls /dev/spidev1.0
   ```

2. Check MCP23008 is accessible:
   ```bash
   sudo i2cdetect -y 1  # Should show 0x20
   ```

3. Verify gpiochip1 exists:
   ```bash
   sudo gpiodetect  # Should show gpiochip1 [mcp23008]
   ```

4. Test raw touch reading:
   ```bash
   sudo python3 ~/touch_calibrate.py
   ```

### Backlight Off After Reboot

The backlight is controlled by MCP23008 GP4. If the service fails to start:

```bash
# Manual backlight enable
sudo gpioset -z -c 1 4=1

# Check service logs
journalctl -xeu yoshipi-touch.service
```

### Input Device Not Created

Verify evdev and uinput:

```bash
# Check uinput module
lsmod | grep uinput
sudo modprobe uinput

# Check input devices
cat /proc/bus/input/devices
```

## Technical Notes

### Why Userspace Daemon?

The XPT2046 touchscreen uses a non-standard chip select arrangement where CS is routed through the MCP23008 I2C GPIO expander. Standard Linux SPI drivers (ads7846) expect CS to be a native GPIO pin controllable by the SPI controller. The userspace approach allows:

1. Direct control of the MCP23008 GPIO for chip select
2. Flexibility in coordinate transformation for display rotation
3. Custom calibration without kernel module recompilation

### SPI Bus Assignment

- **SPI0**: Display (ILI9341) - hardware CS via GPIO4
- **SPI1**: Touch (XPT2046) - software CS via MCP23008 GP7

This separation ensures display and touch operations don't interfere with each other.

### Coordinate Transformation

The display operates in 270-degree rotation mode (MADCTL 0xE8). The touch panel's physical orientation requires:

1. Axis swap: Touch X → Screen Y, Touch Y → Screen X
2. No inversion needed after swap (calibration accounts for this)

### Polling Rate

The daemon polls at 50Hz (20ms interval), providing responsive touch input while minimizing CPU usage. This can be adjusted in the daemon's `time.sleep(0.02)` call.

## GTK and WebKit Availability

The following GUI libraries are available for application development:

| Library | Package | Status |
|---------|---------|--------|
| GTK3 Runtime | libgtk-3-0t64 | Installed |
| GTK4 Runtime | libgtk-4-1 | Installed |
| WebKit2GTK 4.1 | libwebkit2gtk-4.1-0 | Installed |
| PyGObject GTK3 | gir1.2-gtk-3.0 | Installed |
| PyGObject WebKit2 | gir1.2-webkit2-4.1 | Installed |

### Installation (if missing)

```bash
# WebKit2GTK runtime and Python bindings
sudo apt install -y libwebkit2gtk-4.1-0 gir1.2-webkit2-4.1

# GTK development headers (for compiling native apps)
sudo apt install -y libgtk-3-dev libgtk-4-dev libwebkit2gtk-4.1-dev
```

### Example: Touch-Enabled WebView Application

```python
#!/usr/bin/env python3
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.1')
from gi.repository import Gtk, WebKit2

win = Gtk.Window(title="YoshiPi WebView")
win.set_default_size(320, 240)
win.connect("destroy", Gtk.main_quit)

webview = WebKit2.WebView()
webview.load_uri("https://example.com")
win.add(webview)
win.fullscreen()  # Optional: run in fullscreen mode
win.show_all()

Gtk.main()
```

## Kiosk and Fullscreen Options

For demo and embedded applications, you may want to hide the desktop panel and run your app in fullscreen mode. Several approaches are available:

### Option 1: GTK Fullscreen (Simplest)

Add `fullscreen()` to your GTK window. Covers everything including the panel. Exit with Alt+F4 or programmatic gesture.

```python
win = Gtk.Window(title="App")
win.fullscreen()  # Covers everything including panel
```

### Option 2: Hide Panel Temporarily

Control the LXDE panel visibility from the command line:

```bash
# Hide LXDE panel
pkill lxpanel

# Bring it back later
lxpanel --profile LXDE-pi &
```

### Option 3: Undecorated Window

Remove window decorations and position at screen origin:

```python
win = Gtk.Window()
win.set_decorated(False)  # No title bar
win.set_default_size(320, 240)
win.move(0, 0)
```

### Option 4: Full Kiosk Mode at Boot

For dedicated appliance deployments, configure the system to launch your app directly at boot without the desktop environment:

```bash
# Create autostart entry for kiosk app
mkdir -p ~/.config/autostart
cat << 'EOF' > ~/.config/autostart/kiosk.desktop
[Desktop Entry]
Type=Application
Name=Kiosk App
Exec=/usr/bin/python3 /home/hhh/myapp.py
X-GNOME-Autostart-enabled=true
EOF

# Optionally disable panel entirely
echo "hide=1" >> ~/.config/lxpanel/LXDE-pi/panels/panel
```

### Recommendation

For interactive demos, `win.fullscreen()` provides the cleanest experience with easy recovery. For production kiosk deployments, Option 4 with a watchdog service ensures the app restarts on failure.

## References

- YoshiPi Hardware Repository: https://github.com/yoshimoshi-garage/yoshipi
- XPT2046 Datasheet: Touch controller command reference
- MCP23008 Datasheet: I2C GPIO expander register map
- Linux Input Subsystem: uinput interface documentation

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2024-12-24 | 1.0 | Initial configuration and daemon implementation |
