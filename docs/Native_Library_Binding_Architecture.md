# Native Library Binding Architecture

## From C Headers to Linked Binaries: The Complete Fidelity Binding Story

This document describes how the Fidelity framework enables F# programs to consume native C/C++ libraries—from parsing headers through code generation, compilation, and final linking. While the January 2025 demo focuses on STM32L5 microcontrollers and the CMSIS HAL library, the architecture generalizes to any native library binding scenario.

---

## The Problem We're Solving

Native libraries present a fundamental challenge for high-level languages: the library exists as compiled machine code with a C header file describing its interface. To use the library from F#, we need:

1. **Type-safe F# declarations** that mirror the C function signatures
2. **Memory layout compatibility** for structures passed across the boundary
3. **Correct calling conventions** so the compiled code interoperates
4. **Link-time resolution** connecting our compiled code to the library

Traditional .NET approaches rely on runtime P/Invoke and marshalling. Fidelity takes a different path: **compile-time binding with static linking**. The F# code is compiled to native machine code that directly calls library functions, with no runtime overhead.

---

## The Three Pillars

The binding architecture rests on three cooperating systems:

### Farscape: The Binding Generator

Farscape parses C/C++ headers and generates F# source code. It understands:
- Function declarations and their signatures
- Structure layouts with field offsets and alignment
- Enumerations and their underlying values
- Preprocessor macros (constants, bit positions, masks)
- Platform-specific qualifiers (`__IO`, `volatile`, etc.)

Farscape's output is **pure F# source code**—not binaries, not metadata, just `.fs` files that become part of the compilation.

### BAREWire: The Memory Description System

BAREWire provides the type system for describing memory layouts. For native bindings, this means:
- **Peripheral descriptors** for memory-mapped hardware
- **Structure schemas** matching C struct layouts
- **Access qualifiers** (read-only, write-only, read-write)
- **Memory region classification** (RAM, flash, peripheral, DMA)

BAREWire types are consumed at compile time. They carry semantic information that influences code generation without imposing runtime cost.

### Firefly/Alex: The Compilation and Binding Engine

Firefly compiles F# through FCS and the PSG to MLIR and LLVM. Alex, the targeting layer, handles:
- **Extern dispatch** for library function calls
- **Volatile semantics** for hardware register access
- **Calling convention** translation
- **Link directive emission** for the final binary

Alex doesn't need library-specific knowledge. It responds to patterns in the PSG—extern declarations, memory region markers, access qualifiers—and emits appropriate MLIR/LLVM.

---

## The Binding Lifecycle

### Phase 1: Header Analysis (Farscape)

Farscape ingests C headers and builds a semantic model of the library:

```
stm32l552xx.h          (device registers)
stm32l5xx_hal_gpio.h   (GPIO HAL API)
stm32l5xx_hal_uart.h   (UART HAL API)
core_cm33.h            (ARM Cortex-M33 core)
        ↓
   Farscape Parser
        ↓
   Declaration Model:
   - 247 peripheral structs
   - 16,000+ register definitions
   - 89 HAL functions
   - 15 interrupt vectors
```

The parser handles CMSIS-specific patterns:
- `__IO`, `__I`, `__O` qualifiers → access constraints
- `_Pos` and `_Msk` macro suffixes → bit field definitions
- `typedef struct { ... } XXX_TypeDef` → peripheral layouts
- `#define XXX_BASE (addr)` → memory-mapped addresses

### Phase 2: Binding Generation (Farscape → F# Source)

Farscape generates two categories of F# output:

#### Category A: High-Level API (Fidelity.[Target])

The developer-facing F# library with idiomatic types and functions:

```fsharp
// Fidelity.STM32L5/GPIO.fs
module Fidelity.STM32L5.GPIO

/// GPIO port enumeration
type Port = GPIOA | GPIOB | GPIOC | GPIOD | GPIOE | GPIOF | GPIOG | GPIOH

/// GPIO pin mode
type Mode = Input | Output | Alternate | Analog

/// Initialize a GPIO pin
let init (port: Port) (pin: int) (mode: Mode) : Result<unit, GpioError> =
    let gpioBase = Descriptors.GPIO.baseAddress port
    let initStruct = createInitStruct pin mode
    match HAL.GPIO_Init(gpioBase, initStruct) with
    | HAL_OK -> Ok ()
    | status -> Error (GpioError.InitFailed status)

/// Write a pin state
let inline writePin (port: Port) (pin: int) (state: bool) : unit =
    let gpioBase = Descriptors.GPIO.baseAddress port
    if state then
        Registers.BSRR.set gpioBase pin
    else
        Registers.BRR.set gpioBase pin
```

This layer provides F# idioms: discriminated unions, Result types, named parameters, inline functions for zero-cost abstractions.

#### Category B: Memory Descriptors (BAREWire.[Target])

Compile-time descriptions of the hardware memory map:

```fsharp
// BAREWire.STM32L5/Descriptors.fs
module BAREWire.STM32L5.Descriptors

open BAREWire.Core

/// GPIO peripheral descriptor
let GPIO : PeripheralDescriptor = {
    Name = "GPIO"
    Region = MemoryRegionKind.Peripheral
    Instances = Map.ofList [
        ("GPIOA", 0x4800_0000un)
        ("GPIOB", 0x4800_0400un)
        ("GPIOC", 0x4800_0800un)
        // ...
    ]
    Registers = [
        { Name = "MODER";  Offset = 0x00; Width = 32; Access = ReadWrite; BitFields = [...] }
        { Name = "OTYPER"; Offset = 0x04; Width = 32; Access = ReadWrite; BitFields = [...] }
        { Name = "OSPEEDR"; Offset = 0x08; Width = 32; Access = ReadWrite; BitFields = [...] }
        { Name = "PUPDR"; Offset = 0x0C; Width = 32; Access = ReadWrite; BitFields = [...] }
        { Name = "IDR";   Offset = 0x10; Width = 32; Access = ReadOnly;  BitFields = [...] }
        { Name = "ODR";   Offset = 0x14; Width = 32; Access = ReadWrite; BitFields = [...] }
        { Name = "BSRR";  Offset = 0x18; Width = 32; Access = WriteOnly; BitFields = [...] }
        { Name = "BRR";   Offset = 0x28; Width = 32; Access = WriteOnly; BitFields = [...] }
        // ...
    ]
}

/// Interrupt vector table
let Interrupts : (string * int) list = [
    ("Reset", -15)
    ("NMI", -14)
    ("HardFault", -13)
    // ... core exceptions ...
    ("WWDG", 0)
    ("PVD_PVM", 1)
    // ... peripheral interrupts ...
    ("USART1", 37)
    ("USART2", 38)
    // ...
]
```

#### Category C: Platform Bindings (HAL Function Bindings)

Platform.Bindings module functions that Alex recognizes and provides platform-specific MLIR emission:

```fsharp
// Fidelity.STM32L5/HAL.fs
module Fidelity.STM32L5.HAL

/// HAL status codes
type HAL_StatusTypeDef = HAL_OK = 0 | HAL_ERROR = 1 | HAL_BUSY = 2 | HAL_TIMEOUT = 3

// GPIO HAL functions - Platform.Bindings pattern (BCL-free)
// Alex provides MLIR emission that links against HAL library
module Platform.Bindings.HAL.GPIO =
    let init (gpioX: nativeint) (gpioInit: nativeint) : HAL_StatusTypeDef =
        Unchecked.defaultof<HAL_StatusTypeDef>

    let deInit (gpioX: nativeint) (gpioPin: uint32) : HAL_StatusTypeDef =
        Unchecked.defaultof<HAL_StatusTypeDef>

    let writePin (gpioX: nativeint) (gpioPin: uint16) (pinState: int) : unit =
        ()

    let readPin (gpioX: nativeint) (gpioPin: uint16) : int =
        Unchecked.defaultof<int>

// UART HAL functions
module Platform.Bindings.HAL.UART =
    let init (huart: nativeint) : HAL_StatusTypeDef =
        Unchecked.defaultof<HAL_StatusTypeDef>

    let transmit (huart: nativeint) (pData: nativeint) (size: uint16) (timeout: uint32) : HAL_StatusTypeDef =
        Unchecked.defaultof<HAL_StatusTypeDef>

    let receive (huart: nativeint) (pData: nativeint) (size: uint16) (timeout: uint32) : HAL_StatusTypeDef =
        Unchecked.defaultof<HAL_StatusTypeDef>
```

### Phase 3: Compilation (Firefly)

The developer writes application code using the generated bindings:

```fsharp
// Application: Blink.fs
module Blink

open Fidelity.STM32L5.GPIO
open Fidelity.STM32L5.Time

let main () =
    // Initialize LED pin (PA5 on Nucleo board)
    init GPIOA 5 Output |> ignore

    // Blink forever
    while true do
        writePin GPIOA 5 true
        delay 500
        writePin GPIOA 5 false
        delay 500
```

Firefly compiles this through its pipeline:

```
Blink.fs + Fidelity.STM32L5 + BAREWire.STM32L5
                        ↓
                      FNCS
                        ↓
              PSG (Program Semantic Graph)
                        ↓
                      Alex
                        ↓
                      MLIR
                        ↓
                      LLVM IR
                        ↓
                    Object File
```

### Phase 4: PSG Analysis and Code Generation (Alex)

Alex traverses the PSG and makes code generation decisions based on semantic markers:

#### Extern Function Calls

When Alex encounters a Platform.Bindings function call:

```fsharp
// PSG shows: FunctionCall to Platform.Bindings.HAL.GPIO.init
```

Alex emits:
1. An LLVM `declare` for the external function
2. A `call` instruction at the use site
3. Library reference metadata for the linker

```llvm
; External declaration
declare i32 @HAL_GPIO_Init(ptr %GPIOx, ptr %GPIO_Init)

; Call site
%status = call i32 @HAL_GPIO_Init(ptr %gpioa_base, ptr %init_struct)
```

#### Direct Register Access

When code accesses memory through BAREWire descriptors marked as `MemoryRegionKind.Peripheral`:

```fsharp
// PSG shows: Store to address 0x48000018 (GPIOA->BSRR), region = Peripheral, access = WriteOnly
```

Alex emits a volatile store:

```llvm
; Volatile store to set pin (BSRR register)
store volatile i32 %pin_mask, ptr inttoptr (i64 1207959576 to ptr), align 4
```

The `volatile` qualifier ensures:
- No reordering with other memory operations
- No elimination even if the value isn't "used"
- Actual memory access on every execution

#### Access Constraint Enforcement

The `AccessKind` from BAREWire descriptors enables compile-time checking:

- `ReadOnly` registers: Alex warns/errors on store operations
- `WriteOnly` registers: Alex warns/errors on load operations
- `ReadWrite` registers: Both operations permitted

This catches hardware programming errors at compile time rather than runtime.

### Phase 5: Linking

The final step combines:
1. **Firefly-compiled objects** - the F# application code
2. **HAL library** - pre-compiled `libstm32l5xx_hal.a`
3. **Startup code** - vector table, initialization
4. **Linker script** - memory layout for the target

```
arm-none-eabi-ld \
    -T STM32L552.ld \
    -o blink.elf \
    blink.o \
    libstm32l5xx_hal.a \
    startup_stm32l552.o \
    -lc -lnosys
```

The linker resolves:
- `HAL_GPIO_Init` → address in `libstm32l5xx_hal.a`
- `HAL_UART_Transmit` → address in `libstm32l5xx_hal.a`
- Memory sections → flash and RAM addresses from linker script

Output: A standalone `.elf` binary ready to flash to the microcontroller.

---

## The Information Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENERATION TIME (Farscape)                        │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ C/C++ Headers   │                                                        │
│  │ stm32l552xx.h   │──┐                                                     │
│  │ stm32l5xx_hal.h │  │                                                     │
│  │ core_cm33.h     │  │                                                     │
│  └─────────────────┘  │                                                     │
│                       ▼                                                     │
│              ┌─────────────────┐                                            │
│              │    Farscape     │                                            │
│              │  (C/C++ Parser) │                                            │
│              └────────┬────────┘                                            │
│                       │                                                     │
│         ┌─────────────┼─────────────┐                                       │
│         ▼             ▼             ▼                                       │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐                          │
│  │ Fidelity.   │ │ BAREWire.    │ │ Platform.    │                          │
│  │ STM32L5     │ │ STM32L5      │ │ Bindings     │                          │
│  │ (F# API)    │ │ (Descriptors)│ │ (BCL-free)   │                          │
│  └─────────────┘ └──────────────┘ └──────────────┘                          │
│         │             │                   │                                 │
└─────────┼─────────────┼───────────────────┼─────────────────────────────────┘
          │             │                   │
          └─────────────┼───────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPILATION TIME (Firefly)                          │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ Application.fs  │                                                        │
│  │ + Fidelity.*    │                                                        │
│  │ + BAREWire.*    │                                                        │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │      FNCS       │  (F# Native Compiler Services)                         │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │      PSG        │  (Program Semantic Graph)                              │
│  │  - Extern refs  │                                                        │
│  │  - Memory marks │                                                        │
│  │  - Access kinds │                                                        │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │     Alex        │  (Targeting Layer)                                     │
│  │  - Extern emit  │                                                        │
│  │  - Volatile gen │                                                        │
│  │  - Link refs    │                                                        │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │     MLIR        │                                                        │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │    LLVM IR      │                                                        │
│  └────────┬────────┘                                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   Object File   │  (.o)                                                  │
│  └────────┬────────┘                                                        │
│           │                                                                 │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             LINK TIME                                       │
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │   Application   │     │   HAL Library   │     │  Startup Code   │        │
│  │      .o         │     │      .a         │     │      .o         │        │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘        │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   ▼                                         │
│                          ┌─────────────────┐                                │
│                          │     Linker      │                                │
│                          │  (arm-none-eabi │                                │
│                          │      -ld)       │                                │
│                          └────────┬────────┘                                │
│                                   ▼                                         │
│                          ┌─────────────────┐                                │
│                          │   Final Binary  │                                │
│                          │     (.elf)      │                                │
│                          └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Static vs Dynamic Binding: The MLIR → LLVM Flow

The Fidelity framework supports **both** static and dynamic library binding. This section details exactly how each model flows through MLIR to LLVM, with concrete examples.

### The Two Binding Models

| Model | Use Case | Link Time | Runtime |
|-------|----------|-----------|---------|
| **Static** | Embedded/unikernel, security-critical | Library code merged into binary | No external dependencies |
| **Dynamic** | Desktop/server with OS, plugins | Symbol references recorded | Library loaded at runtime |

### Configuration-Driven Binding Strategy

The binding strategy is declared in the project configuration, not embedded in code:

```toml
[dependencies]
# Static: merged into binary (security-critical crypto)
crypto_lib = { version = "1.2.0", binding = "static" }

# Dynamic: loaded at runtime (OS-provided HAL)
stm32l5_hal = { version = "2.1.5", binding = "dynamic" }

# Dynamic: system library (always dynamic)
libc = { binding = "dynamic" }

[profiles.embedded]
# Override: everything static for unikernel
binding.default = "static"
binding.exceptions = ["device_hal"]

[profiles.linux-desktop]
# Override: everything dynamic except crypto
binding.default = "dynamic"
binding.static_override = ["crypto_lib"]
```

### Static Binding: MLIR → LLVM Flow

For static binding, the library function becomes a **direct call** with external linkage resolved at link time.

**Step 1: PSG captures Platform.Binding with static binding strategy**

```
PSG Node: FunctionCall
  Target: Platform.Bindings.HAL.GPIO.init
  Attributes:
    - PlatformBinding: { module: HAL.GPIO, function: init }
    - BindingStrategy: Static  ← from project config
```

**Step 2: Alex emits MLIR with external function declaration**

```mlir
// MLIR: External function declaration (will be resolved at link time)
func.func private @HAL_GPIO_Init(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32
    attributes {
        llvm.linkage = #llvm.linkage<external>,
        fidelity.binding = "static",
        fidelity.library = "stm32l5_hal"
    }

// MLIR: Call site
func.func @initGpio() {
    %gpioa = llvm.mlir.constant(0x48000000 : i64) : i64
    %gpioa_ptr = llvm.inttoptr %gpioa : i64 to !llvm.ptr
    %init_struct = llvm.alloca ...
    %result = func.call @HAL_GPIO_Init(%gpioa_ptr, %init_struct) : (!llvm.ptr, !llvm.ptr) -> i32
    ...
}
```

**Step 3: MLIR lowers to LLVM IR**

```llvm
; LLVM IR: External declaration (symbol resolved by linker)
declare i32 @HAL_GPIO_Init(ptr %GPIOx, ptr %GPIO_Init)

; LLVM IR: Call site - direct call, no indirection
define void @initGpio() {
entry:
    %gpioa_ptr = inttoptr i64 1207959552 to ptr  ; 0x48000000
    %init_struct = alloca %GPIO_InitTypeDef, align 4
    ; ... populate init_struct ...
    %result = call i32 @HAL_GPIO_Init(ptr %gpioa_ptr, ptr %init_struct)
    ; ...
}
```

**Step 4: Linker resolves symbol**

```bash
arm-none-eabi-ld -o app.elf app.o libstm32l5_hal.a
```

The linker:
1. Finds `HAL_GPIO_Init` symbol in `libstm32l5_hal.a`
2. Extracts the object file containing it
3. Merges it into the final binary
4. Resolves the call address

**Result**: Single self-contained binary with no external dependencies.

### Dynamic Binding: MLIR → LLVM Flow

For dynamic binding, the library function is called through a **runtime-resolved reference**.

**Step 1: PSG captures Platform.Binding with dynamic binding strategy**

```
PSG Node: FunctionCall
  Target: Platform.Bindings.GTK.Window.create
  Attributes:
    - PlatformBinding: { module: GTK.Window, function: create }
    - BindingStrategy: Dynamic  ← from project config
```

**Step 2: Alex emits MLIR with dynamic linkage markers**

```mlir
// MLIR: External function with dso_local marker for dynamic linking
func.func private @gtk_window_new(%arg0: i32) -> !llvm.ptr
    attributes {
        llvm.linkage = #llvm.linkage<external>,
        fidelity.binding = "dynamic",
        fidelity.library = "gtk-4",
        llvm.dso_local = false  // Symbol resolved at runtime
    }

// MLIR: Call site (same as static - the difference is in linkage)
func.func @createWindow() -> !llvm.ptr {
    %window_type = llvm.mlir.constant(0 : i32) : i32
    %window = func.call @gtk_window_new(%window_type) : (i32) -> !llvm.ptr
    return %window : !llvm.ptr
}
```

**Step 3: MLIR lowers to LLVM IR with PLT/GOT considerations**

```llvm
; LLVM IR: External declaration (resolved via PLT at runtime)
declare ptr @gtk_window_new(i32) #0

; Function attributes for dynamic linking
attributes #0 = { "frame-pointer"="all" }

; LLVM IR: Call site - goes through PLT
define ptr @createWindow() {
entry:
    %window = call ptr @gtk_window_new(i32 0)
    ret ptr %window
}
```

**Step 4: Linker creates dynamic references**

```bash
gcc -o app app.o -lgtk-4
```

The linker:
1. Records `gtk_window_new` as an undefined symbol requiring `libgtk-4.so`
2. Creates PLT (Procedure Linkage Table) entry for lazy resolution
3. Creates GOT (Global Offset Table) entry for the resolved address
4. Records `gtk-4` in the dynamic section (DT_NEEDED)

**Step 5: Runtime loads library**

At program startup (or first call with lazy binding):
1. Dynamic linker (`ld.so`) loads `libgtk-4.so`
2. Resolves `gtk_window_new` address
3. Patches GOT entry
4. Subsequent calls go directly to resolved address

**Result**: Smaller binary, shared library, updateable separately.

### Hybrid Binding in Practice

Real applications often use both models simultaneously:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Binary                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Statically Linked                                        │   │
│  │  - crypto_lib (security: no substitution attacks)        │   │
│  │  - core_algorithms (performance: inlining enabled)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Dynamic References (PLT/GOT)                             │   │
│  │  - libgtk-4.so (GUI framework, shared by many apps)      │   │
│  │  - libssl.so (system crypto, security updates)           │   │
│  │  - libc.so (always dynamic, OS-provided)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The January Demo: Pure Static (Unikernel Model)

The STM32L5 "QuantumCredential" demo uses **pure static binding** because:

1. **No OS**: There's no dynamic linker on bare metal
2. **No filesystem**: Can't load `.so` files
3. **Security**: Post-quantum crypto must not be substitutable
4. **Determinism**: Exact code is known at compile time

```toml
[package]
name = "quantum_credential"
target = "thumbv8m.main-none-eabi"

[binding]
default = "static"  # Everything statically linked

[dependencies]
stm32l5_hal = { binding = "static" }
pqcrypto = { binding = "static" }
```

### The Libre Sweet Potato Future: Hybrid Model

A future desktop application on Libre Sweet Potato (ARM SBC with Linux) would use **hybrid binding**:

```toml
[package]
name = "fidelity_desktop_app"
target = "aarch64-unknown-linux-gnu"

[binding]
default = "dynamic"  # OS integration
static_override = ["crypto_lib"]  # Security-critical stays static

[dependencies]
gtk = { binding = "dynamic" }      # Shared GUI framework
gpiod = { binding = "dynamic" }    # OS-managed GPIO access
crypto_lib = { binding = "static" } # Can't risk substitution
```

The application would:
- Call GTK through PLT → shared `libgtk-4.so`
- Call GPIO through PLT → shared `libgpiod.so`
- Call crypto directly → code merged into binary

### MLIR Transformation Based on Binding Strategy

Alex's transformation is straightforward - the same extern declaration produces different MLIR based on the project configuration:

```fsharp
// Alex pseudo-code for extern handling
let emitExternFunction (extern: ExternDeclaration) (config: ProjectConfig) =
    let bindingStrategy = config.GetBindingStrategy extern.LibraryName

    mlir {
        // Function declaration is the same structure
        yield! declareExternalFunction extern.Name extern.Signature

        // Attributes differ based on binding strategy
        match bindingStrategy with
        | Static ->
            yield! addAttribute "llvm.linkage" "#llvm.linkage<external>"
            yield! addAttribute "fidelity.binding" "static"
            // Static: will be resolved at link time against .a

        | Dynamic ->
            yield! addAttribute "llvm.linkage" "#llvm.linkage<external>"
            yield! addAttribute "llvm.dso_local" "false"
            yield! addAttribute "fidelity.binding" "dynamic"
            // Dynamic: will go through PLT/GOT at runtime
    }
```

The key insight: **The MLIR structure is nearly identical.** The difference is in metadata attributes that guide LLVM's code generation and the linker's behavior.

### Link-Time Optimization Implications

Static binding enables **Link-Time Optimization (LTO)** across the boundary:

```
Without LTO:          With LTO (static only):
┌──────────┐          ┌──────────┐
│ app.o    │          │ app.o    │
│  call ───┼──┐       │  ───┐    │
└──────────┘  │       └─────┼────┘
              │             │ inlined!
┌──────────┐  │       ┌─────┼────┐
│ lib.o    │◄─┘       │ lib.o    │
│  func    │          │  (merged)│
└──────────┘          └──────────┘
```

Dynamic binding **cannot** do this - the library code isn't available at compile time.

---

## Generalization Beyond Embedded

While the January demo targets STM32L5, the architecture supports any native library:

### Desktop Libraries

```fsharp
// Generated by Farscape from sqlite3.h - Platform.Bindings pattern
module Platform.Bindings.SQLite =
    let open' (filename: nativeint) (ppDb: nativeint) : int =
        Unchecked.defaultof<int>

    let exec (db: nativeint) (sql: nativeint) (callback: nativeint) (arg: nativeint) (errmsg: nativeint) : int =
        Unchecked.defaultof<int>
```

### System Libraries

```fsharp
// Fidelity.Platform bindings
module Platform.Bindings =
    let writeBytes (fd: int) (buf: nativeint) (count: int) : int =
        Unchecked.defaultof<int>

// Platform-specific bindings in Fidelity.Platform/<Target>
// Alex provides platform-specific MLIR emission for each
```

### Graphics/Compute Libraries

```fsharp
// Generated by Farscape from vulkan.h - Platform.Bindings pattern
module Platform.Bindings.Vulkan =
    let createInstance (pCreateInfo: nativeint) (pAllocator: nativeint) (pInstance: nativeint) : VkResult =
        Unchecked.defaultof<VkResult>
```

The pattern is consistent:
1. **Farscape** parses headers → generates Platform.Bindings modules (BCL-free)
2. **Firefly** compiles F# → LLVM objects with external references (Alex provides MLIR)
3. **Linker** resolves references → final binary

---

## The Role of Each Component

### Farscape's Responsibilities

- Parse C/C++ headers (via libclang or native invocation)
- Extract function signatures, struct layouts, enums, constants
- Generate idiomatic F# types (DUs for enums, records for structs)
- Generate extern declarations with correct calling conventions
- Generate BAREWire descriptors for memory-mapped regions
- Handle platform-specific qualifiers and macros

Farscape does **NOT**:
- Generate MLIR or LLVM code
- Know about Alex's internal patterns
- Make platform-specific code generation decisions

### BAREWire's Responsibilities

- Define abstract types for memory description
- Provide schema system for binary layouts
- Carry semantic markers (access kind, memory region)
- Enable compile-time layout calculation

BAREWire does **NOT**:
- Implement platform-specific memory operations
- Contain syscall bindings or hardware access code
- Know about specific chips or peripherals (that's BAREWire.[Target])

### Alex's Responsibilities

- Recognize extern declarations in the PSG
- Emit LLVM external function declarations
- Recognize memory region markers (Peripheral, etc.)
- Emit volatile operations where required
- Emit linker directives for library references
- Handle calling convention translation

Alex does **NOT**:
- Parse C headers
- Know about specific libraries (no "if HAL_GPIO_Init" checks)
- Generate library-specific code paths

### The Linker's Responsibilities

- Resolve external symbol references
- Combine object files with libraries
- Apply memory layout from linker script
- Produce final executable

---

## January Demo: STM32L5 Blink

The concrete deliverable demonstrates the full pipeline:

### Input Files

1. **CMSIS Headers**: `stm32l552xx.h`, `stm32l5xx_hal_*.h`, `core_cm33.h`
2. **HAL Library**: Pre-compiled `libstm32l5xx_hal.a`
3. **Application**: `Blink.fs` using GPIO to blink an LED

### Generated Files (by Farscape)

1. `Fidelity.STM32L5/*.fs` - High-level GPIO, UART, Timer APIs
2. `BAREWire.STM32L5/*.fs` - Peripheral descriptors, register layouts
3. Extern declarations for HAL functions

### Output

A working `blink.elf` that:
- Initializes GPIO pin PA5 as output
- Toggles the pin in a loop with delays
- Runs standalone on STM32L552 Nucleo board

### Success Criteria

1. LED blinks at expected rate
2. No runtime crashes or undefined behavior
3. Binary size comparable to equivalent C implementation
4. Code structure is idiomatic F#

---

## Future Directions

### Automatic Library Discovery

Farscape could scan standard locations for headers and libraries:
```
farscape discover --target arm-none-eabi --sdk /path/to/stm32cube
```

### Multi-Target Binding Sets

Generate bindings for multiple chips from a single HAL:
```
farscape generate stm32l5xx_hal.h --targets STM32L552,STM32L562,STM32L5A6
```

### Inline Assembly Integration

For performance-critical paths, allow inline assembly in F#:
```fsharp
[<InlineAsm("mrs %0, primask", "=r")>]
extern uint32 getPrimask()
```

### Hardware Abstraction Layers

Build higher-level abstractions on top of chip-specific bindings:
```fsharp
// Fidelity.HAL.GPIO (chip-agnostic)
type IGpioPin =
    abstract Set: bool -> unit
    abstract Get: unit -> bool

// Implementation for STM32L5
type STM32L5Pin(port: Port, pin: int) =
    interface IGpioPin with
        member _.Set value = GPIO.writePin port pin value
        member _.Get () = GPIO.readPin port pin
```

---

## Conclusion

The Fidelity native binding architecture enables F# to consume any native C/C++ library through a principled pipeline:

1. **Farscape** transforms headers into F# source
2. **BAREWire** provides the memory description vocabulary
3. **Firefly/Alex** compiles to LLVM with proper external references
4. **The linker** produces the final binary

Each component has a clear responsibility. Information flows forward through the pipeline without backward dependencies. The result is native F# code that links directly against C libraries with no runtime marshalling overhead.

The January demo proves this architecture with a concrete, working example. The generalization to desktop libraries, system APIs, and compute frameworks follows the same pattern—only the headers and libraries change.
