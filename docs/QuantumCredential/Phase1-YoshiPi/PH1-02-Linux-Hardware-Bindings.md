# Linux Hardware Bindings for YoshiPi

> **Audience**: Firefly developers implementing hardware access for YoshiPi
>
> **Key Insight**: Linux provides uniform interfaces for hardware. GPIO, ADC, and serial all reduce to file descriptors and syscalls that Fidelity.Platform already knows how to handle.

---

## The Platform.Bindings Pattern

All hardware access follows the established Platform.Bindings pattern in Fidelity.Platform:

```fsharp
module Platform.Bindings =
    /// Function signature - Alex provides the implementation
    let someHardwareOp (args...) : returnType =
        Unchecked.defaultof<returnType>
```

**How it works:**
1. Fidelity.Platform declares the function signature with `Unchecked.defaultof<T>` body
2. Alex recognizes `Platform.Bindings.*` calls during PSG traversal
3. Alex emits platform-specific MLIR (syscalls for Linux, API calls for Windows)
4. The generated code calls the actual OS interface

**For YoshiPi (Linux/ARM64)**, all hardware access reduces to:
- `open()` - Get file descriptor for device
- `read()`/`write()` - Data transfer
- `ioctl()` - Device control
- `close()` - Release file descriptor

---

## Core Device Access Bindings

### File Descriptor Operations

These extend the existing `writeBytes`/`readBytes` pattern:

```fsharp
module Platform.Bindings =

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Access Bindings
    // ═══════════════════════════════════════════════════════════════════════════

    /// Open a device or file by path.
    /// Alex implementations: Linux: open() syscall
    /// Returns file descriptor (>= 0) or -1 on error
    let openDevice (path: nativeint) (flags: int) : int =
        Unchecked.defaultof<int>

    /// Close a file descriptor.
    /// Alex implementations: Linux: close() syscall
    let closeDevice (fd: int) : int =
        Unchecked.defaultof<int>

    /// Perform ioctl on a file descriptor.
    /// Alex implementations: Linux: ioctl() syscall
    /// The 'arg' parameter is type-punned based on the request
    let ioctl (fd: int) (request: uint64) (arg: nativeint) : int =
        Unchecked.defaultof<int>

    /// Seek to position in file.
    /// Alex implementations: Linux: lseek() syscall
    let lseek (fd: int) (offset: int64) (whence: int) : int64 =
        Unchecked.defaultof<int64>
```

### Open Flags (Constants)

```fsharp
module Platform.Constants =
    // Linux open() flags
    let O_RDONLY   = 0
    let O_WRONLY   = 1
    let O_RDWR     = 2
    let O_CREAT    = 0o100
    let O_NONBLOCK = 0o4000

    // lseek() whence values
    let SEEK_SET = 0
    let SEEK_CUR = 1
    let SEEK_END = 2
```

---

## GPIO Access via /dev/gpiochip

### Linux GPIO Character Device Interface

The YoshiPi exposes GPIO through `/dev/gpiochip0`. Access uses ioctl:

```
/dev/gpiochip0                  # Main GPIO controller
```

### GPIO ioctl Commands

```fsharp
module Platform.Constants.GPIO =
    // ioctl request codes (from linux/gpio.h)
    // These encode direction, size, and type using _IOC macro
    let GPIO_GET_CHIPINFO_IOCTL       = 0x8044B401UL  // _IOR(0xB4, 0x01, gpiochip_info)
    let GPIO_GET_LINEINFO_IOCTL       = 0xC048B402UL  // _IOWR(0xB4, 0x02, gpioline_info)
    let GPIO_GET_LINEHANDLE_IOCTL     = 0xC16CB403UL  // _IOWR(0xB4, 0x03, gpiohandle_request)
    let GPIO_GET_LINEEVENT_IOCTL      = 0xC030B404UL  // _IOWR(0xB4, 0x04, gpioevent_request)
    let GPIOHANDLE_GET_LINE_VALUES    = 0xC040B408UL  // _IOWR(0xB4, 0x08, gpiohandle_data)
    let GPIOHANDLE_SET_LINE_VALUES    = 0xC040B409UL  // _IOWR(0xB4, 0x09, gpiohandle_data)

    // Line flags
    let GPIOHANDLE_REQUEST_INPUT      = 0x01u
    let GPIOHANDLE_REQUEST_OUTPUT     = 0x02u
    let GPIOHANDLE_REQUEST_ACTIVE_LOW = 0x04u
```

### GPIO Data Structures

These structures must be laid out exactly as Linux expects:

```fsharp
/// GPIO line info (queried via GPIO_GET_LINEINFO_IOCTL)
[<Struct; StructLayout(LayoutKind.Sequential)>]
type GpioLineInfo =
    val mutable line_offset: uint32
    val mutable flags: uint32
    val mutable name: NativeArray<byte>       // char[32]
    val mutable consumer: NativeArray<byte>   // char[32]

/// GPIO handle request (sent via GPIO_GET_LINEHANDLE_IOCTL)
[<Struct; StructLayout(LayoutKind.Sequential)>]
type GpioHandleRequest =
    val mutable lineoffsets: NativeArray<uint32>   // uint32_t[64]
    val mutable flags: uint32
    val mutable default_values: NativeArray<byte>  // uint8_t[64]
    val mutable consumer_label: NativeArray<byte>  // char[32]
    val mutable lines: uint32
    val mutable fd: int32                          // Returned handle fd

/// GPIO handle data (for get/set values)
[<Struct; StructLayout(LayoutKind.Sequential)>]
type GpioHandleData =
    val mutable values: NativeArray<byte>  // uint8_t[64]
```

### Higher-Level GPIO Module

```fsharp
module GPIO =
    open Platform.Bindings
    open Platform.Constants.GPIO

    /// Open GPIO chip and return file descriptor
    let openChip (chipPath: string) : int =
        openDevice chipPath.Pointer O_RDWR

    /// Request a GPIO line as output
    let requestOutput (chipFd: int) (line: int) (initialValue: int) : int =
        let request = NativeArray.stackalloc<GpioHandleRequest> 1
        request.[0].lineoffsets.[0] <- uint32 line
        request.[0].flags <- GPIOHANDLE_REQUEST_OUTPUT
        request.[0].default_values.[0] <- byte initialValue
        request.[0].lines <- 1u

        let result = ioctl chipFd GPIO_GET_LINEHANDLE_IOCTL (NativePtr.toNativeInt request)
        if result < 0 then -1
        else request.[0].fd

    /// Set output line value
    let setValue (lineFd: int) (value: int) : int =
        let data = NativeArray.stackalloc<GpioHandleData> 1
        data.[0].values.[0] <- byte value
        ioctl lineFd GPIOHANDLE_SET_LINE_VALUES (NativePtr.toNativeInt data)

    /// Request a GPIO line as input
    let requestInput (chipFd: int) (line: int) : int =
        let request = NativeArray.stackalloc<GpioHandleRequest> 1
        request.[0].lineoffsets.[0] <- uint32 line
        request.[0].flags <- GPIOHANDLE_REQUEST_INPUT
        request.[0].lines <- 1u

        let result = ioctl chipFd GPIO_GET_LINEHANDLE_IOCTL (NativePtr.toNativeInt request)
        if result < 0 then -1
        else request.[0].fd

    /// Get input line value
    let getValue (lineFd: int) : int =
        let data = NativeArray.stackalloc<GpioHandleData> 1
        let result = ioctl lineFd GPIOHANDLE_GET_LINE_VALUES (NativePtr.toNativeInt data)
        if result < 0 then -1
        else int data.[0].values.[0]
```

### GPIO Usage Example

```fsharp
let controlStatusLED () =
    // Open GPIO chip
    let chip = GPIO.openChip "/dev/gpiochip0"n
    if chip < 0 then failwith "Cannot open GPIO chip"

    // Request GPIO 17 as output, initially off
    let led = GPIO.requestOutput chip 17 0
    if led < 0 then failwith "Cannot request GPIO line"

    // Blink LED
    for _ in 1..5 do
        GPIO.setValue led 1 |> ignore
        Platform.Bindings.sleep 500
        GPIO.setValue led 0 |> ignore
        Platform.Bindings.sleep 500

    closeDevice led |> ignore
    closeDevice chip |> ignore
```

---

## ADC Access via IIO Subsystem

### Linux Industrial I/O (IIO)

The YoshiPi's ADC appears under the Linux IIO subsystem:

```
/sys/bus/iio/devices/iio:device0/
├── name                        # Device name
├── in_voltage0_raw             # Channel 0 raw value (0-1023)
├── in_voltage1_raw             # Channel 1 raw value
├── in_voltage_scale            # mV per LSB
└── sampling_frequency          # Sample rate (if supported)
```

### Sysfs Read Pattern

For IIO, we read ASCII values from sysfs files:

```fsharp
module IIO =
    open Platform.Bindings

    /// Read raw ADC value from sysfs
    /// Returns raw value (0-1023 for 10-bit ADC) or -1 on error
    let readRaw (channelPath: string) : int =
        let fd = openDevice channelPath.Pointer O_RDONLY
        if fd < 0 then -1
        else
            // Read ASCII digits
            let buffer = NativeArray.stackalloc<byte> 16
            let bytesRead = readBytes fd (NativePtr.toNativeInt buffer) 16

            closeDevice fd |> ignore

            if bytesRead <= 0 then -1
            else
                // Parse ASCII to int
                parseAsciiInt buffer bytesRead

    /// Read scale factor
    let readScale (scalePath: string) : float32 =
        let fd = openDevice scalePath.Pointer O_RDONLY
        if fd < 0 then 0.0f
        else
            let buffer = NativeArray.stackalloc<byte> 32
            let bytesRead = readBytes fd (NativePtr.toNativeInt buffer) 32
            closeDevice fd |> ignore

            if bytesRead <= 0 then 0.0f
            else parseAsciiFloat buffer bytesRead

    /// Helper: Parse ASCII integer from buffer
    let private parseAsciiInt (buffer: NativeArray<byte>) (len: int) : int =
        let mutable result = 0
        let mutable i = 0
        while i < len && buffer.[i] >= byte '0' && buffer.[i] <= byte '9' do
            result <- result * 10 + int (buffer.[i] - byte '0')
            i <- i + 1
        result

    /// Helper: Parse ASCII float (simplified - no exponent)
    let private parseAsciiFloat (buffer: NativeArray<byte>) (len: int) : float32 =
        // Simplified implementation - parse integer.fraction
        // Full implementation would handle more formats
        0.0f  // TODO: implement
```

### High-Speed ADC Sampling

For continuous sampling (entropy collection), use the IIO buffer interface:

```fsharp
module IIO.Buffered =
    open Platform.Bindings

    /// Configure IIO buffer for continuous sampling
    let configureBuffer (devicePath: string) (length: int) : int =
        // Enable channel
        let scanEnablePath = String.concat [devicePath; "/scan_elements/in_voltage0_en"n]
        let enableFd = openDevice scanEnablePath.Pointer O_WRONLY
        if enableFd >= 0 then
            let one = "1"n
            writeBytes enableFd one.Pointer 1 |> ignore
            closeDevice enableFd |> ignore

        // Set buffer length
        let bufLenPath = String.concat [devicePath; "/buffer/length"n]
        let lenFd = openDevice bufLenPath.Pointer O_WRONLY
        if lenFd >= 0 then
            let lenStr = intToAscii length
            writeBytes lenFd lenStr.Pointer lenStr.Length |> ignore
            closeDevice lenFd |> ignore

        0  // Success

    /// Enable buffered capture
    let enableBuffer (devicePath: string) : int =
        let enablePath = String.concat [devicePath; "/buffer/enable"n]
        let fd = openDevice enablePath.Pointer O_WRONLY
        if fd < 0 then -1
        else
            let one = "1"n
            let result = writeBytes fd one.Pointer 1
            closeDevice fd |> ignore
            result

    /// Read samples from buffer device
    let readSamples (buffer: NativeArray<uint16>) (count: int) : int =
        let fd = openDevice "/dev/iio:device0"n.Pointer O_RDONLY
        if fd < 0 then -1
        else
            let bytesWanted = count * 2  // 16-bit samples
            let bytesRead = readBytes fd (NativePtr.toNativeInt buffer) bytesWanted
            closeDevice fd |> ignore
            bytesRead / 2  // Return sample count
```

### Entropy Sampling Module

```fsharp
module Entropy =
    open IIO

    let adcPath = "/sys/bus/iio/devices/iio:device0/in_voltage0_raw"n

    /// Sample avalanche noise from ADC
    let sampleAvalanche (count: int) : NativeArray<uint16> =
        let buffer = NativeArray.stackalloc<uint16> count

        for i in 0 .. count - 1 do
            let raw = readRaw adcPath
            buffer.[i] <- uint16 (if raw < 0 then 0 else raw)

        buffer

    /// Condition raw samples into seed bytes using XOR folding
    /// (Simple conditioning - production would use SHAKE-256)
    let condition (samples: NativeArray<uint16>) : NativeArray<byte> =
        let outputLen = samples.Length / 8  // 8 samples per output byte
        let output = NativeArray.stackalloc<byte> outputLen

        for i in 0 .. outputLen - 1 do
            let mutable b = 0uy
            for j in 0 .. 7 do
                let sample = samples.[i * 8 + j]
                // XOR LSBs from each sample
                b <- b ^^^ byte (sample &&& 0xFFus)
            output.[i] <- b

        output
```

---

## USB Serial Access (Gadget Mode)

### USB Gadget Configuration

The Pi Zero 2 W supports USB OTG, allowing it to appear as a USB device (gadget) to the desktop.

```bash
# On YoshiPi: Configure as USB serial device (CDC ACM)
modprobe libcomposite
# (gadget configuration script creates /dev/ttyGS0)
```

Once configured, `/dev/ttyGS0` is a standard serial device:

```fsharp
module USBGadget =
    open Platform.Bindings

    /// Open USB gadget serial port
    let openSerial () : int =
        openDevice "/dev/ttyGS0"n.Pointer O_RDWR

    /// Write credential data to USB
    let writeCredential (fd: int) (data: NativeArray<byte>) : int =
        writeBytes fd (NativePtr.toNativeInt data) data.Length

    /// Read acknowledgment from desktop
    let readAck (fd: int) : int =
        let buffer = NativeArray.stackalloc<byte> 16
        readBytes fd (NativePtr.toNativeInt buffer) 16
```

### Desktop Side

On the desktop, the YoshiPi appears as `/dev/ttyACM0` (or similar):

```fsharp
module CredentialReceiver =
    open Platform.Bindings

    /// Open USB serial port to YoshiPi
    let openDevice () : int =
        openDevice "/dev/ttyACM0"n.Pointer O_RDWR

    /// Read credential from YoshiPi
    let readCredential (fd: int) (maxLen: int) : NativeArray<byte> =
        let buffer = NativeArray.stackalloc<byte> maxLen
        let bytesRead = readBytes fd (NativePtr.toNativeInt buffer) maxLen
        buffer  // Caller handles actual length
```

---

## Alex Implementation Notes

### Syscall Numbers (Linux ARM64)

Alex generates syscall instructions for these bindings:

```fsharp
// Alex/Bindings/LinuxSyscalls.fs
module LinuxSyscalls =
    // ARM64 syscall numbers (same as x86_64 for most)
    let SYS_read    = 63L
    let SYS_write   = 64L
    let SYS_openat  = 56L    // open() is implemented via openat(AT_FDCWD, ...)
    let SYS_close   = 57L
    let SYS_lseek   = 62L
    let SYS_ioctl   = 29L
```

### MLIR Generation Pattern

For `ioctl`, Alex generates:

```mlir
// ioctl(fd, request, arg) -> result
%fd = ...
%request = arith.constant 0xC040B409 : i64   // GPIOHANDLE_SET_LINE_VALUES
%arg = ...
%result = llvm.inline_asm "svc #0",
    "=r,{x8},{x0},{x1},{x2}"
    (i64 29, i32 %fd, i64 %request, i64 %arg) : i32
```

### Binding Registration

```fsharp
// Alex/Bindings/DeviceBindings.fs
module DeviceBindings =
    let registerBindings () =
        // Register for Linux ARM64
        PlatformDispatch.register Linux ARM64 "openDevice"
            (fun prim -> emitSyscall SYS_openat prim)

        PlatformDispatch.register Linux ARM64 "closeDevice"
            (fun prim -> emitSyscall SYS_close prim)

        PlatformDispatch.register Linux ARM64 "ioctl"
            (fun prim -> emitSyscall SYS_ioctl prim)

        PlatformDispatch.register Linux ARM64 "lseek"
            (fun prim -> emitSyscall SYS_lseek prim)

        // Also register for x86_64 (desktop testing)
        PlatformDispatch.register Linux X86_64 "openDevice"
            (fun prim -> emitSyscall SYS_openat_x86 prim)
        // ... etc
```

---

## Summary: What This Enables

With these bindings, the YoshiPi demo can:

| Operation | FNCS Module | Linux Interface |
|-----------|--------------|-----------------|
| LED control | `GPIO.setValue` | ioctl on `/dev/gpiochip0` |
| Button input | `GPIO.getValue` | ioctl on `/dev/gpiochip0` |
| ADC sampling | `IIO.readRaw` | read from sysfs |
| Fast entropy | `IIO.Buffered.readSamples` | read from `/dev/iio:device0` |
| USB transfer | `USBGadget.writeCredential` | write to `/dev/ttyGS0` |

**Key Advantage**: All of this is standard Linux programming. The same Firefly compiler, same FNCS intrinsics, same patterns as desktop - just different device paths and ioctl constants.

---

## Cross-References

- [01_YoshiPi_Demo_Strategy](./01_YoshiPi_Demo_Strategy.md) - Overall strategy
- [02_YoshiPi_Architecture](./02_YoshiPi_Architecture.md) - Hardware details
- [03_MLIR_Dialect_Strategy](./03_MLIR_Dialect_Strategy.md) - Compilation path and parallel execution
- [05_PostQuantum_Architecture](./05_PostQuantum_Architecture.md) - PQC algorithms
- [Architecture_Canonical](../Architecture_Canonical.md) - Platform binding pattern
- Farscape docs (`~/repos/Farscape/docs/`) - Quotation-based binding architecture
