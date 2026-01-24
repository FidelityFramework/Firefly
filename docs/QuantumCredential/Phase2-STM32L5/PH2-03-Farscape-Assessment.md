# Farscape Assessment: Maturation Path for January Demo

> **Document Purpose**: This assessment captures the current state of Farscape, identifies gaps blocking the January demo (CMSIS HAL binding generation for STM32L5 unikernel), and provides a prioritized maturation roadmap informed by Firefly's architectural learnings.

## Executive Summary

Farscape is a C/C++ binding generator that aims to produce F# interop code from native headers. The architecture is sound: it uses [CppSharp](https://github.com/mono/CppSharp) (built on libclang) for parsing and has a reasonable pipeline from AST to code generation. However, **the core parsing functionality is currently non-functional**: all headers except a hardcoded cJSON.h example return empty declaration lists.

For the January demo goal of compiling F# code that interfaces with CMSIS HAL on STM32L5, four gaps must be addressed:

1. **Parser is broken** (BLOCKING): CppSharp integration exists but is bypassed
2. **No macro support**: CMSIS HAL depends heavily on `#define` constants
3. **Wrong output format**: Generates P/Invoke for .NET runtime, not FNCS-style intrinsics
4. **No ARM bindings in Alex**: Need bare-metal register access, not syscalls

---

## 1. Understanding What Farscape Is

### 1.1 Design Intent

Farscape's purpose is to automate the tedious and error-prone task of writing F# bindings for native C/C++ libraries. Rather than manually declaring each extern function, struct, and constant, Farscape:

1. **Parses** C/C++ header files using CppSharp/libclang
2. **Maps** C types to F# equivalents
3. **Generates** F# source files with proper interop attributes
4. **Creates** a complete F# project structure

This follows the same philosophy as [CppSharp's own goal](https://github.com/mono/CppSharp): "Tools and libraries to glue C/C++ APIs to high-level languages."

### 1.2 Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Farscape Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  C/C++ Header    ──►  CppParser.fs   ──►  TypeMapper.fs                │
│  (e.g., gpio.h)       (CppSharp)          (C → F# types)               │
│                            │                    │                       │
│                            ▼                    ▼                       │
│                    Declaration list      TypeMapping list               │
│                            │                    │                       │
│                            └──────┬─────────────┘                       │
│                                   ▼                                     │
│                          CodeGenerator.fs                               │
│                          (P/Invoke generation)                          │
│                                   │                                     │
│                                   ▼                                     │
│                          BindingGenerator.fs                            │
│                          (Project orchestration)                        │
│                                   │                                     │
│                                   ▼                                     │
│                           F# Project Output                             │
│                           - NativeBindings.fs                           │
│                           - Wrappers.fs                                 │
│                           - StructWrappers.fs                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Source Files Overview

| File | Purpose | Status |
|------|---------|--------|
| `CppParser.fs` | Parse C/C++ headers via CppSharp | **BROKEN** - Returns empty lists |
| `TypeMapper.fs` | Map C types to F# equivalents | Working |
| `CodeGenerator.fs` | Generate P/Invoke declarations | Working (wrong format) |
| `BindingGenerator.fs` | Orchestrate full pipeline | Working |
| `Project.fs` | Generate .fsproj and solution | Working |
| `Types.fs` | Common types (OperationStatus) | Working |
| `MemoryManager.fs` | Memory management utilities | Exists |
| `DelegatePointer.fs` | Function pointer handling | Exists |
| `ProjectOptions.fs` | Configuration types | Working |

---

## 2. The C/C++ Language Question

### 2.1 Does CppSharp Handle Pure C?

**Yes, absolutely.** CppSharp is built on [libclang](https://clang.llvm.org/doxygen/group__CINDEX.html), which is Clang's C interface to the full C/C++ parser. As the libclang documentation states:

> "The C Interface to Clang provides a relatively small API that exposes facilities for parsing source code into an abstract syntax tree (AST), loading already-parsed ASTs, traversing the AST..."

Clang itself is the reference compiler for both C and C++, supporting all modern C standards (C89, C99, C11, C17, C23) and C++ standards. When parsing a `.h` file, Clang determines the language based on:

1. File extension (`.h` is typically treated as C unless in C++ mode)
2. Explicit language flags (`-x c` or `-x c++`)
3. Presence of C++ constructs in the file

### 2.2 CMSIS HAL Headers Are Pure C

The STM32 CMSIS HAL headers we need to parse are **pure C**:

```c
// stm32l5xx_hal_gpio.h
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t Pin;
  uint32_t Mode;
  uint32_t Pull;
  uint32_t Speed;
  uint32_t Alternate;
} GPIO_InitTypeDef;

void HAL_GPIO_Init(GPIO_TypeDef *GPIOx, const GPIO_InitTypeDef *GPIO_Init);
GPIO_PinState HAL_GPIO_ReadPin(const GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
void HAL_GPIO_WritePin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState);

#ifdef __cplusplus
}
#endif
```

The `extern "C"` wrapper ensures C linkage even when included from C++, but the content is standard C. CppSharp/libclang handles this perfectly.

### 2.3 What Farscape's DeclarationVisitor Handles

Looking at `CppParser.fs`, the `DeclarationVisitor` class properly handles:

| C/C++ Construct | Handler Method | F# Declaration Type |
|-----------------|----------------|---------------------|
| Functions | `VisitFunctionDecl` | `Function of FunctionDecl` |
| Structs/Classes | `VisitClassDecl` | `Struct of StructDecl` or `Class of ClassDecl` |
| Enumerations | `VisitEnumDecl` | `Enum of EnumDecl` |
| Namespaces | `VisitNamespace` | `Namespace of NamespaceDecl` |
| Typedefs | (implicit) | `Typedef of TypedefDecl` |

**Missing from DeclarationVisitor:**
- **Macros/Preprocessor definitions** - Critical for CMSIS
- **Global variables** - Less common in headers
- **Unions** - Treated as structs currently

---

## 3. The Critical Problem: Parser is Non-Functional

### 3.1 What the Code Shows

The `parseHeader` function in `CppParser.fs` (lines 142-520) reveals the problem:

```fsharp
let parseHeader (options: HeaderParserOptions) =
    printfn $"Manually parsing header: {options.HeaderFile}"

    if options.HeaderFile.EndsWith("cJSON.h") then
        // 360+ lines of hardcoded cJSON declarations
        printfn "Using manual declarations for cJSON.h"
        [
            cJSON_Hooks_Struct
            cJSON_Struct
            typedef_bool
            yield! enumDeclarations
            yield! functionDeclarations
        ]
    else
        // THIS IS THE PROBLEM
        printfn "Creating empty declarations for header: %s" options.HeaderFile
        let extension = System.IO.Path.GetExtension(options.HeaderFile).ToLowerInvariant()

        match extension with
        | ".h" | ".hpp" | ".hxx" ->
            printfn "C/C++ header detected, returning empty list until parser is fixed"
            []  // <-- EVERY HEADER RETURNS EMPTY!
        | _ ->
            printfn "Unsupported file type, returning empty list"
            []
```

### 3.2 The Unused Infrastructure

The irony is that `DeclarationVisitor` (lines 56-134) is properly implemented:

```fsharp
type DeclarationVisitor() =
    inherit AstVisitor()

    let declarations = ResizeArray<Declaration>()

    member _.GetDeclarations() = declarations |> List.ofSeq

    override _.VisitFunctionDecl(func: CppSharp.AST.Function) =
        let parameters =
            func.Parameters
            |> Seq.map (fun p -> p.Name, p.Type.ToString())
            |> List.ofSeq

        declarations.Add(
            Function {
                Name = func.Name
                ReturnType = func.ReturnType.ToString()
                Parameters = parameters
                Documentation =
                    if isNull func.Comment then None
                    else Option.ofObj func.Comment.BriefText
                IsVirtual = ...
                IsStatic = ...
            })
        true

    // ... handlers for Class, Enum, Namespace
```

This visitor knows how to walk a CppSharp AST and extract declarations. It's simply **never invoked**.

### 3.3 What Should Happen

The fix requires wiring up CppSharp's parser to feed the `DeclarationVisitor`:

```fsharp
// Pseudocode for what parseHeader should do
let parseHeader (options: HeaderParserOptions) =
    // 1. Configure CppSharp parser
    let parserOptions = ParserOptions()
    parserOptions.AddSourceFile(options.HeaderFile)
    for path in options.IncludePaths do
        parserOptions.AddIncludeDirs(path)

    // 2. Parse the header
    let result = ClangParser.ParseHeader(parserOptions)

    // 3. Walk the AST with our visitor
    let visitor = DeclarationVisitor()
    for tu in result.TranslationUnits do
        visitor.VisitTranslationUnit(tu)

    // 4. Return collected declarations
    visitor.GetDeclarations()
```

---

## 4. Gap Analysis for January Demo

### 4.1 Gap 1: Parser Non-Functional (BLOCKING)

**Impact**: Complete blocker. No header can be parsed.

**Root Cause**: `parseHeader` bypasses CppSharp entirely, returning hardcoded or empty lists.

**Fix Complexity**: Medium. The infrastructure exists (DeclarationVisitor), needs wiring.

**Acceptance Criteria**:
```fsharp
// This should return non-empty list with GPIO_InitTypeDef, HAL_GPIO_Init, etc.
let declarations = CppParser.parse
    "helpers/cmsis/STM32L5xx_HAL_Driver/Inc/stm32l5xx_hal_gpio.h"
    ["helpers/cmsis/CMSIS/Core/Include"
     "helpers/cmsis/STM32L5xx/Include"
     "helpers/cmsis/STM32L5xx_HAL_Driver/Inc"]
    true

declarations |> List.length |> should be greaterThan 0
```

### 4.2 Gap 2: No Macro Support

**Impact**: High. CMSIS HAL is unusable without constants.

**Evidence from `stm32l5xx_hal_gpio.h`**:
```c
#define GPIO_PIN_0    ((uint16_t)0x0001)
#define GPIO_PIN_1    ((uint16_t)0x0002)
// ... 14 more pins
#define GPIO_PIN_All  ((uint16_t)0xFFFF)

#define GPIO_MODE_INPUT      MODE_INPUT
#define GPIO_MODE_OUTPUT_PP  (MODE_OUTPUT | OUTPUT_PP)
// ... 10 more modes

#define GPIO_SPEED_FREQ_LOW        0x00000000u
#define GPIO_SPEED_FREQ_MEDIUM     0x00000001u
#define GPIO_SPEED_FREQ_HIGH       0x00000002u
#define GPIO_SPEED_FREQ_VERY_HIGH  0x00000003u

#define GPIO_NOPULL    0x00000000u
#define GPIO_PULLUP    0x00000001u
#define GPIO_PULLDOWN  0x00000002u
```

**Current Declaration Type** (missing Macro):
```fsharp
type Declaration =
    | Function of FunctionDecl
    | Struct of StructDecl
    | Enum of EnumDecl
    | Typedef of TypedefDecl
    | Namespace of NamespaceDecl
    | Class of ClassDecl
    // Missing: | Macro of MacroDecl
```

**Fix Complexity**: Low-Medium. CppSharp can extract macros; need to add type and visitor logic.

**Alternative**: [CppAst.NET](https://github.com/xoofx/CppAst.NET) explicitly advertises "access to the full AST, comments and macros" and may be easier for macro extraction.

### 4.3 Gap 3: Wrong Output Format

**Impact**: High. Generated code won't work with Firefly.

**Current Output** (P/Invoke for .NET runtime - INCOMPATIBLE with Fidelity):
```fsharp
// WRONG: DllImport is a BCL dependency Fidelity cannot use
module CMSIS.NativeBindings
open System.Runtime.InteropServices

[<DllImport("libstm32l5", CallingConvention = CallingConvention.Cdecl)>]
extern void HAL_GPIO_Init(nativeint GPIOx, nativeint GPIO_Init)
```

**Required Output** (Platform.Bindings pattern for Fidelity):
```fsharp
namespace CMSIS.STM32L5

module GPIO =
    [<Struct>]
    type GPIO_InitTypeDef = {
        Pin: uint32
        Mode: uint32
        Pull: uint32
        Speed: uint32
        Alternate: uint32
    }

// Platform.Bindings - BCL-free, Alex provides MLIR emission
module Platform.Bindings.HAL.GPIO =
    /// Initialize GPIO peripheral
    let init (gpio: nativeint) (initStruct: nativeint) : unit =
        ()  // Alex emits register writes or HAL calls

    /// Write GPIO pin
    let writePin (gpio: nativeint) (pin: uint16) (state: int) : unit =
        ()  // Alex emits BSRR register write
```

**Key Differences**:
1. NO BCL dependencies (no `System.Runtime.InteropServices`)
2. `Unchecked.defaultof<T>` or `()` as placeholder for Alex-provided implementation
3. Structs use `[<Struct>]` attribute for value semantics
4. Module naming follows `Platform.Bindings.*` convention for Alex recognition

**Fix Complexity**: Medium. New generator mode or separate `FidelityCodeGenerator.fs`.

### 4.4 Gap 4: No ARM/BareMetal Bindings in Alex

**Impact**: Medium. Without this, bindings compile but don't run on STM32.

**Current Alex Binding Infrastructure**:
```fsharp
// BindingTypes.fs - These already exist!
type OSFamily =
    | Linux | Windows | MacOS | FreeBSD
    | BareMetal  // <-- For embedded
    | WASM

type Architecture =
    | X86_64 | ARM64
    | ARM32_Thumb  // <-- For Cortex-M
    | RISCV64 | RISCV32 | WASM32
```

**What's Missing**: Actual binding implementations in `Alex/Bindings/ARM/`

**HAL Functions Need Memory-Mapped Register Access**:
```fsharp
// Example: HAL_GPIO_WritePin implementation for STM32L5
let bindGpioWritePin (platform: TargetPlatform) (prim: ExternPrimitive) = mlir {
    match prim.Args with
    | [gpioBase; pin; state] ->
        // GPIO BSRR register is at offset 0x18 from base
        let! bsrrOffset = arith.constant 0x18L I32
        let! bsrrAddr = llvm.getelementptr gpioBase [bsrrOffset]

        // If state == SET, write to lower 16 bits (set)
        // If state == RESET, write to upper 16 bits (reset)
        let! shiftAmount =
            match state with
            | { Value = 0 } -> arith.constant 16L I32  // RESET
            | _ -> arith.constant 0L I32               // SET
        let! bitPattern = arith.shli pin shiftAmount

        // Store to BSRR (write-only register)
        do! llvm.store bitPattern bsrrAddr
        return EmittedVoid
    | _ ->
        return NotSupported "WritePin requires (gpio, pin, state)"
}
```

**Fix Complexity**: High. Requires:
- Understanding STM32L5 memory map
- Implementing register access patterns
- Testing on actual hardware or QEMU

---

## 5. What CMSIS HAL Parsing Requires

### 5.1 Header Dependency Chain

Parsing `stm32l5xx_hal_gpio.h` requires resolving includes:

```
stm32l5xx_hal_gpio.h
└── stm32l5xx_hal_def.h
    ├── stm32l5xx.h
    │   ├── stm32l552xx.h (or stm32l562xx.h)
    │   │   └── core_cm33.h
    │   │       └── cmsis_gcc.h (or cmsis_armclang.h)
    │   └── system_stm32l5xx.h
    └── stm32l5xx_hal_conf.h
```

This means the parser needs proper include path configuration:
```
-I helpers/cmsis/CMSIS/Core/Include
-I helpers/cmsis/STM32L5xx/Include
-I helpers/cmsis/STM32L5xx_HAL_Driver/Inc
-D STM32L552xx
-D USE_HAL_DRIVER
```

### 5.2 Declaration Inventory

From `stm32l5xx_hal_gpio.h`, Farscape must extract:

| Category | Count | Examples |
|----------|-------|----------|
| Structs | 1 | `GPIO_InitTypeDef` |
| Enums | 1 | `GPIO_PinState` |
| Functions | 12 | `HAL_GPIO_Init`, `HAL_GPIO_ReadPin`, etc. |
| Macros | ~50 | `GPIO_PIN_0`-`GPIO_PIN_15`, modes, speeds |

### 5.3 Special Considerations

**Volatile Registers**: GPIO peripheral structs use `__IO` (volatile) qualifiers:
```c
typedef struct {
  __IO uint32_t MODER;    // Mode register
  __IO uint32_t OTYPER;   // Output type register
  __IO uint32_t OSPEEDR;  // Output speed register
  __IO uint32_t PUPDR;    // Pull-up/pull-down register
  __IO uint32_t IDR;      // Input data register
  __IO uint32_t ODR;      // Output data register
  __IO uint32_t BSRR;     // Bit set/reset register
  // ... more registers
} GPIO_TypeDef;
```

Firefly must preserve volatile semantics in generated MLIR.

**Pointer-to-Peripheral**: HAL functions take `GPIO_TypeDef*` which is actually a memory-mapped address:
```c
#define GPIOA  ((GPIO_TypeDef *) GPIOA_BASE)
#define GPIOA_BASE  (PERIPH_BASE + 0x00020000UL)
```

The F# binding doesn't need to resolve these; they're passed at runtime.

---

## 6. Comparison with Firefly's Binding Pattern

### 6.1 How FNCS/Alex Bindings Work

Firefly's current binding system (for console I/O, time, etc.) follows the **Platform.Bindings pattern** (BCL-free):

**Step 1: Fidelity.Platform declares bindings** (no DllImport!)
```fsharp
// Fidelity.Platform - BCL-free platform bindings
module Platform.Bindings =
    /// Write bytes to file descriptor
    let writeBytes (fd: int) (buffer: nativeint) (count: int) : int =
        Unchecked.defaultof<int>  // Placeholder - Alex provides implementation

    /// Read bytes from file descriptor
    let readBytes (fd: int) (buffer: nativeint) (maxCount: int) : int =
        Unchecked.defaultof<int>
```

**Step 2: Alex registers bindings by module/function name**
```fsharp
// Alex/Bindings/Console/ConsoleBindings.fs
let registerBindings () =
    ExternDispatch.register Linux X86_64 "writeBytes"
        (fun ext -> bindWriteBytes TargetPlatform.linux_x86_64 ext)
```

**Step 3: Alex generates platform-specific MLIR**
```fsharp
let bindWriteBytes (platform: TargetPlatform) (prim: ExternPrimitive) = mlir {
    match platform.OS with
    | Linux ->
        // Generate syscall instruction
        let! result = emitUnixWriteSyscall 1L fd buf count
        return Emitted result
    | MacOS ->
        let! result = emitUnixWriteSyscall 0x2000004L fd buf count
        return Emitted result
}
```

### 6.2 What Farscape Must Generate for CMSIS

Following the Platform.Bindings pattern (BCL-free):

**Farscape generates** (like Fidelity.Platform):
```fsharp
// Generated: CMSIS.STM32L5.GPIO.fs
namespace CMSIS.STM32L5

/// Platform.Bindings for HAL GPIO - Alex provides MLIR emission
module Platform.Bindings.HAL.GPIO =
    /// Initialize GPIO peripheral
    let init (gpio: nativeint) (initStruct: nativeint) : unit =
        ()  // Alex emits HAL_GPIO_Init call or direct register access

    /// Write GPIO pin state
    let writePin (gpio: nativeint) (pin: uint16) (state: int) : unit =
        ()  // Alex emits HAL_GPIO_WritePin or BSRR register write

    /// Read GPIO pin state
    let readPin (gpio: nativeint) (pin: uint16) : int =
        Unchecked.defaultof<int>
```

**Alex provides ARM bindings** (new module):
```fsharp
// Alex/Bindings/ARM/GPIOBindings.fs
let registerGPIOBindings () =
    ExternDispatch.register BareMetal ARM32_Thumb "Platform.Bindings.HAL.GPIO.init"
        (fun ext -> bindGpioInit ext)
    ExternDispatch.register BareMetal ARM32_Thumb "Platform.Bindings.HAL.GPIO.writePin"
        (fun ext -> bindGpioWritePin ext)
```

---

## 7. Recommended Maturation Path

### 7.1 Phase 1: Fix Parser (Week 1-2)

**Priority**: CRITICAL - Blocks everything else

**Tasks**:
1. Study CppSharp documentation and examples
2. Implement proper `ParserOptions` configuration
3. Wire `ClangParser.ParseHeader` to `DeclarationVisitor`
4. Handle include path resolution
5. Add error reporting for parse failures

**Verification**:
```bash
# Parse GPIO header successfully
dotnet run -- parse helpers/cmsis/STM32L5xx_HAL_Driver/Inc/stm32l5xx_hal_gpio.h \
    --include helpers/cmsis/CMSIS/Core/Include \
    --include helpers/cmsis/STM32L5xx/Include \
    --include helpers/cmsis/STM32L5xx_HAL_Driver/Inc \
    --define STM32L552xx \
    --verbose
```

### 7.2 Phase 2: Add Macro Support (Week 2)

**Priority**: HIGH - Required for CMSIS

**Tasks**:
1. Add `Macro of MacroDecl` to Declaration type
2. Extend visitor to collect preprocessor definitions
3. Handle macro value parsing (numeric, expressions)
4. Consider [CppAst.NET](https://github.com/xoofx/CppAst.NET) if CppSharp macro support is insufficient

**Verification**:
```fsharp
// Should find all GPIO_PIN_* constants
let macros = declarations |> List.choose (function Macro m -> Some m | _ -> None)
macros |> List.exists (fun m -> m.Name = "GPIO_PIN_0") |> should be true
```

### 7.3 Phase 3: Fidelity Output Mode (Week 2-3)

**Priority**: HIGH - Required for Firefly integration

**Options**:
- **A) Separate generator**: `FidelityCodeGenerator.fs` alongside `CodeGenerator.fs`
- **B) Mode flag**: Add `OutputFormat` option (PInvoke | Fidelity)

**Tasks**:
1. Design output format matching Fidelity.Platform pattern
2. Generate `Platform.Bindings.*` module functions with `Unchecked.defaultof<T>`
3. Use `nativeint` for pointer arguments (BCL-free)
4. Emit proper struct definitions with `[<Struct>]`
5. Generate module structure following `Platform.Bindings.*` convention

### 7.4 Phase 4: ARM/BareMetal Bindings (Week 3-4)

**Priority**: MEDIUM - Required for demo execution

**Location**: `src/Alex/Bindings/ARM/`

**Tasks**:
1. Create `GPIOBindings.fs` for HAL_GPIO_* functions
2. Implement memory-mapped register access patterns
3. Handle volatile semantics in MLIR generation
4. Test with QEMU ARM emulation or hardware

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CppSharp integration harder than expected | Medium | High | Consider CppAst.NET fallback |
| Macro extraction incomplete | Medium | High | Manual supplementation for demo |
| ARM binding complexity underestimated | High | Medium | Focus on minimal GPIO subset |
| Include path resolution issues | Medium | Medium | Pre-process headers; use C LSP |
| Demo timeline too aggressive | Medium | High | Scope to GPIO only; defer other HAL modules |

---

## 9. Resources

### 9.1 Documentation

- [CppSharp Wiki](https://github.com/mono/CppSharp/wiki) - Getting started, developer manual
- [CppAst.NET](https://github.com/xoofx/CppAst.NET) - Alternative with explicit macro support
- [libclang Documentation](https://clang.llvm.org/doxygen/group__CINDEX.html) - Low-level API reference

### 9.2 Reference Code

- **Firefly bindings pattern**: `src/Alex/Bindings/Console/ConsoleBindings.fs`
- **FNCS primitives**: Core type operations
- **CMSIS headers**: `helpers/cmsis/STM32L5xx_HAL_Driver/Inc/`
- **Full STM32CubeL5**: `~/repos/STM32CubeL5/`

### 9.3 Tools Available

- **C LSP (clangd)**: Configured in Serena for CMSIS header inspection
- **F# LSP**: Available for Farscape code navigation
- **MLIR LSP**: Available for generated IR inspection

---

## 10. Success Criteria for January Demo

### Minimum Viable Demo

```fsharp
// F# source file: Blink.fs
open CMSIS.STM32L5.GPIO

[<EntryPoint>]
let main () =
    // Configure PA5 as output (LED on Nucleo-L552ZE-Q)
    let init = GPIO_InitTypeDef(
        Pin = GPIO_PIN_5,
        Mode = GPIO_MODE_OUTPUT_PP,
        Pull = GPIO_NOPULL,
        Speed = GPIO_SPEED_FREQ_LOW
    )
    halGpioInit(GPIOA, &init)

    // Blink forever
    while true do
        halGpioTogglePin(GPIOA, GPIO_PIN_5)
        // delay would need separate implementation
    0
```

**Compilation succeeds**:
```bash
Firefly compile Blink.fidproj --target thumbv8m.main-none-eabihf -o blink.elf
```

**Output runs on STM32L5** (or QEMU):
- LED toggles
- No runtime errors
- Binary size reasonable (<10KB for GPIO-only)

---

## Appendix A: Farscape File Inventory

```
helpers/Farscape/
├── src/
│   └── Farscape.Core/
│       ├── Farscape.Core.fsproj    # Package: CppSharp 1.0.45.22293
│       ├── ProjectOptions.fs        # Configuration types
│       ├── Types.fs                 # OperationStatus enum
│       ├── CppParser.fs             # BROKEN - needs fix
│       ├── TypeMapper.fs            # C→F# type mapping (working)
│       ├── MemoryManager.fs         # Memory utilities
│       ├── DelegatePointer.fs       # Function pointer handling
│       ├── CodeGenerator.fs         # P/Invoke generation (wrong format)
│       ├── Project.fs               # .fsproj generation
│       └── BindingGenerator.fs      # Pipeline orchestration
└── README.md
```

## Appendix B: CMSIS Header Inventory

```
helpers/cmsis/
├── CMSIS/Core/Include/              # ARM CMSIS core (26 headers)
│   ├── core_cm33.h                  # Cortex-M33 definitions
│   ├── cmsis_gcc.h                  # GCC intrinsics
│   └── ...
├── STM32L5xx/Include/               # Device headers (6 headers)
│   ├── stm32l552xx.h                # STM32L552 peripherals (GPIO_TypeDef, etc.)
│   └── ...
├── STM32L5xx_HAL_Driver/
│   ├── Inc/                         # HAL headers (85 headers)
│   │   ├── stm32l5xx_hal.h          # Main HAL include
│   │   ├── stm32l5xx_hal_gpio.h     # GPIO HAL (our target)
│   │   └── ...
│   └── Src/                         # HAL source (76 files, for reference)
└── compile_commands.json            # For clangd
```
