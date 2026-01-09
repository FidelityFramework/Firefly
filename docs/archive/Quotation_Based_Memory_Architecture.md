# Quotation-Based Memory Architecture

## Executive Summary

This document describes an architectural approach where F# quotations and active patterns serve as the **foundational substrate** for memory model representation across the Fidelity framework. Rather than treating these as convenience features, we propose they become the core interchange format between components.

The key insight: **F#'s unique features (quotations, active patterns, computation expressions) can define not just what Fidelity compiles, but how Fidelity compiles.**

## The Four-Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Quotation-Based Memory Architecture                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Farscape (C/C++ Binding Generator)                              │   │
│  │                                                                  │   │
│  │  Input: CMSIS/HAL headers                                        │   │
│  │  Output: Quoted memory descriptors + Generated active patterns   │   │
│  │                                                                  │   │
│  │  NOT just F# types - CODE that represents hardware knowledge     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire (Memory Descriptor Infrastructure)                     │   │
│  │                                                                  │   │
│  │  Provides: Quotation interpretation framework                    │
│  │            Active pattern composition primitives                 │   │
│  │            Memory region algebra                                 │   │
│  │                                                                  │   │
│  │  The "runtime" for quoted memory descriptions                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  fsnative (Native F# Compiler)                                   │   │
│  │                                                                  │   │
│  │  Ingests: Quoted descriptors from plugins                        │   │
│  │  Transforms: Quotations through PSG nanopasses                   │   │
│  │  Recognizes: Memory patterns via active patterns                 │   │
│  │  Emits: MLIR with correct memory semantics                       │   │
│  │                                                                  │   │
│  │  Quotations flow THROUGH the PSG, transformed at each pass       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Alloy (Base Native Library)                                     │   │
│  │                                                                  │   │
│  │  Consumes: Memory primitives with quoted constraints             │   │
│  │  Expresses: User-facing API that carries semantic guarantees     │   │
│  │  Does NOT: Define memory regions or hardware layouts             │   │
│  │                                                                  │   │
│  │  BCL-analog that is transparent to the memory architecture       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Quotations as Interchange Format

Memory constraints are not opaque type annotations. They are **inspectable, transformable expressions**:

```fsharp
// Traditional: Opaque type annotation
type GPIO_ODR = NativePtr<uint32, peripheral, readWrite>

// Quotation-based: Inspectable constraint
let GPIO_ODR_Constraint = <@
    { Address = 0x48000014un
      Region = Peripheral
      Access = ReadWrite
      Volatile = true
      Width = Bits32
      Documentation = "GPIO Output Data Register" }
@>
```

The quotation can be:
- **Inspected** at compile time to validate constraints
- **Transformed** by nanopasses to specialize for targets
- **Composed** with other constraints
- **Evaluated** to emit correct MLIR

### 2. Active Patterns as Recognition Substrate

Active patterns are not syntactic sugar. They are **computed pattern matching** that enables:

```fsharp
// Recognition is computed, not structural
let (|VolatilePeripheralWrite|_|) (node: PSGNode) =
    match node with
    | FunctionCall (WriteOp, [ptr; value]) ->
        match ptr.MemoryConstraint with
        | Some <@ { Volatile = true; Region = Peripheral } @> ->
            Some (VolatilePeripheralWrite (ptr, value))
        | _ -> None
    | _ -> None
```

This separates:
- **What** to recognize (the pattern)
- **How** to recognize it (the computation)
- **Where** the knowledge comes from (quotations from plugins)

### 3. Plugins Provide Knowledge, Core Provides Mechanism

The core framework (fsnative + BAREWire) provides:
- Quotation interpretation infrastructure
- Active pattern composition framework
- PSG transformation pipeline
- MLIR emission machinery

Plugins (Farscape-generated, hardware-specific) provide:
- Quoted memory region descriptions
- Generated active patterns for hardware recognition
- Peripheral-specific access patterns

### 4. Pure F# Idioms, No BCL Patterns

Fidelity deliberately avoids .NET/BCL conventions that don't serve native compilation:

- **Records over interfaces**: `MemoryModel` is a record, not an `IMemoryModel` interface. Records are first-class values that compose naturally.
- **Functions over abstract methods**: Recognition is a function field `Recognize: PSGNode -> MemoryOperation option`, not an abstract method requiring inheritance.
- **No `[<AbstractClass>]`**: Inheritance hierarchies are BCL patterns. Composition via records and functions is more idiomatic.
- **Minimal annotations**: F* proof annotations will be significant; we avoid accumulating unrelated metadata attributes.

This keeps the code visually clean and semantically focused on the domain rather than the runtime machinery.

### 5. Alloy as Transparent Consumption Surface

Alloy does not define memory semantics. It **consumes** them:

```fsharp
// Alloy provides the API
module Alloy.Memory =
    let inline read<'T, [<Measure>] 'region, [<Measure>] 'access
                    when 'access :> readable>
        (ptr: NativePtr<'T, 'region, 'access>) : 'T =
        Platform.Bindings.read ptr

// The constraint ('access :> readable) is checked against quoted descriptors
// Alloy doesn't know what "readable" means for STM32 vs GPU
// That knowledge lives in the plugin's quotations
```

---

## Part 1: Farscape - Generating Quotations and Active Patterns

### Current State

Farscape currently generates F# type definitions:

```fsharp
// Current Farscape output
[<Struct; StructLayout(LayoutKind.Sequential)>]
type GPIO_TypeDef = {
    MODER: NativePtr<uint32, peripheral, readWrite>
    IDR: NativePtr<uint32, peripheral, readOnly>
    BSRR: NativePtr<uint32, peripheral, writeOnly>
}
```

### Proposed State

Farscape generates **three artifacts**:

#### Artifact 1: Quoted Descriptors

```fsharp
// Generated: CMSIS.STM32L5.GPIO.Descriptors.fs
module CMSIS.STM32L5.GPIO.Descriptors

open BAREWire.Quotations
open Microsoft.FSharp.Quotations

/// Quoted peripheral descriptor for GPIO family
let gpioDescriptor: Expr<PeripheralDescriptor> = <@
    { Name = "GPIO"
      Instances =
        [ "GPIOA", 0x48000000un
          "GPIOB", 0x48000400un
          "GPIOC", 0x48000800un ]
      Layout =
        { Size = 0x400
          Alignment = 4
          Fields =
            [ { Name = "MODER";  Offset = 0x00; Width = 32; Access = ReadWrite }
              { Name = "OTYPER"; Offset = 0x04; Width = 32; Access = ReadWrite }
              { Name = "IDR";    Offset = 0x10; Width = 32; Access = ReadOnly }
              { Name = "ODR";    Offset = 0x14; Width = 32; Access = ReadWrite }
              { Name = "BSRR";   Offset = 0x18; Width = 32; Access = WriteOnly } ] }
      MemoryRegion = Peripheral
      Volatile = true }
@>

/// Quoted constraint for individual registers
let moderConstraint: Expr<RegisterConstraint> = <@
    { PeripheralFamily = "GPIO"
      RegisterName = "MODER"
      Offset = 0x00
      Access = ReadWrite
      BitFields =
        [ for pin in 0..15 do
            { Name = sprintf "MODE%d" pin
              Position = pin * 2
              Width = 2
              Values = ["Input"; "Output"; "AltFunc"; "Analog"] } ] }
@>

/// All register constraints for GPIO
let allConstraints: Expr<RegisterConstraint> list =
    [ moderConstraint; otyperConstraint; idrConstraint; ... ]
```

#### Artifact 2: Generated Active Patterns

```fsharp
// Generated: CMSIS.STM32L5.GPIO.Patterns.fs
module CMSIS.STM32L5.GPIO.Patterns

open BAREWire.Patterns
open Microsoft.FSharp.Quotations

/// Active pattern to recognize GPIO port access
let (|GPIOPortAccess|_|) (node: PSGNode) =
    match node.MemoryConstraint with
    | Some expr ->
        match expr with
        | <@ { PeripheralFamily = "GPIO"; RegisterName = name } @> ->
            let instance = extractInstance node
            Some (GPIOPortAccess (instance, name))
        | _ -> None
    | None -> None

/// Active pattern to recognize GPIO pin write
let (|GPIOPinWrite|_|) (node: PSGNode) =
    match node with
    | FunctionCall (target, [port; pin; value])
        when isWriteOp target ->
        match port with
        | GPIOPortAccess (instance, "BSRR") ->
            Some (GPIOPinWrite (instance, pin, value))
        | GPIOPortAccess (instance, "ODR") ->
            Some (GPIOPinWriteODR (instance, pin, value))
        | _ -> None
    | _ -> None

/// Active pattern to recognize GPIO pin read
let (|GPIOPinRead|_|) (node: PSGNode) =
    match node with
    | FunctionCall (target, [port; pin])
        when isReadOp target ->
        match port with
        | GPIOPortAccess (instance, "IDR") ->
            Some (GPIOPinRead (instance, pin))
        | _ -> None
    | _ -> None

/// Composed pattern for any GPIO operation
let (|GPIOOperation|_|) (node: PSGNode) =
    match node with
    | GPIOPinWrite (inst, pin, value) -> Some (GPIOWrite (inst, pin, value))
    | GPIOPinRead (inst, pin) -> Some (GPIORead (inst, pin))
    | _ -> None
```

#### Artifact 3: Plugin Registration

```fsharp
// Generated: CMSIS.STM32L5.Plugin.fs
module CMSIS.STM32L5.Plugin

open BAREWire.Plugins
open CMSIS.STM32L5.GPIO.Descriptors
open CMSIS.STM32L5.GPIO.Patterns

/// STM32L5 memory model - provides all hardware knowledge as a record value
let stm32l5 : MemoryModel = {
    TargetFamily = "stm32l5"

    PeripheralDescriptors = [
        gpioDescriptor
        usartDescriptor
        spiDescriptor
        i2cDescriptor
        // ... all peripherals
    ]

    RegisterConstraints =
        GPIO.Descriptors.allConstraints @
        USART.Descriptors.allConstraints @
        // ...
        []

    Recognize = function
        // GPIO patterns
        | GPIOOperation op -> Some op
        // USART patterns
        | USARTTransmit op -> Some op
        | USARTReceive op -> Some op
        // SPI patterns
        | SPITransfer op -> Some op
        // No match
        | _ -> None

    Regions = <@
        [ { Name = "Flash"
            Base = 0x08000000un
            Size = 0x80000un
            Properties = ReadExecute ||| Cacheable }
          { Name = "SRAM1"
            Base = 0x20000000un
            Size = 0x30000un
            Properties = ReadWrite ||| Cacheable }
          { Name = "SRAM2"
            Base = 0x20030000un
            Size = 0x10000un
            Properties = ReadWrite ||| Cacheable }
          { Name = "Peripheral"
            Base = 0x40000000un
            Size = 0x20000000un
            Properties = ReadWrite ||| Volatile ||| NoCache } ]
    @>

    CacheTopology = None  // Could add L1 I/D cache description
    CoherencyModel = None  // Single core, no coherency needed
}
```

### Farscape Code Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Farscape Generation Pipeline                         │
│                                                                         │
│  Stage 1: Parse                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  XParsec parses C/C++ headers                                     │ │
│  │  Extracts: structs, fields, qualifiers, macros                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Stage 2: Analyze                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Identify peripheral families                                     │ │
│  │  Extract register layouts and access constraints                  │ │
│  │  Parse bit field definitions from macros                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Stage 3: Generate Quotations                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Emit F# source containing quoted descriptors                     │ │
│  │  Each peripheral family → Expr<PeripheralDescriptor>              │ │
│  │  Each register → Expr<RegisterConstraint>                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Stage 4: Generate Active Patterns                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Emit F# source containing active pattern definitions             │ │
│  │  Patterns reference the quoted descriptors                        │ │
│  │  Patterns compose hierarchically (register → peripheral → family) │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Stage 5: Generate Plugin                                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Emit plugin registration that aggregates all artifacts           │ │
│  │  Plugin provides MemoryModel record to BAREWire                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Output: Compiled plugin assembly (.dll)                                │
│  Contains: Quoted descriptors + Active patterns + Plugin type           │
└─────────────────────────────────────────────────────────────────────────┘
```

### XParsec's Continued Role

XParsec remains central to Farscape for **parsing C/C++ headers**. The change is in what Farscape **outputs**:

| Before | After |
|--------|-------|
| F# type definitions | Quoted descriptors |
| Platform.Bindings stubs | Generated active patterns |
| BAREWire descriptor values | Plugin registration |

XParsec is the **input** machinery. Quotations/Active Patterns are the **output** format.

---

## Part 2: BAREWire - Quotation Infrastructure

### Current State

BAREWire provides (or will provide) hardware descriptor types:

```fsharp
// Current BAREWire types
type PeripheralDescriptor = {
    Name: string
    Instances: Map<string, unativeint>
    Layout: PeripheralLayout
    MemoryRegion: MemoryRegionKind
}
```

### Proposed Restructuring

BAREWire becomes the **infrastructure for quotation-based memory description**:

#### Core Quotation Types

```fsharp
// BAREWire.Quotations.fs
module BAREWire.Quotations

open Microsoft.FSharp.Quotations

/// Core descriptor types (these appear INSIDE quotations)
type MemoryProperties =
    | ReadOnly
    | WriteOnly
    | ReadWrite
    | Volatile
    | Cacheable
    | NoCache
    | Execute
    // Composable via |||

type RegionDescriptor = {
    Name: string
    Base: unativeint
    Size: unativeint
    Properties: MemoryProperties
}

type FieldDescriptor = {
    Name: string
    Offset: int
    Width: int  // bits
    Access: MemoryProperties
}

type PeripheralDescriptor = {
    Name: string
    Instances: (string * unativeint) list
    Layout: FieldDescriptor list
    MemoryRegion: string  // references a RegionDescriptor
    Volatile: bool
}

type RegisterConstraint = {
    PeripheralFamily: string
    RegisterName: string
    Offset: int
    Access: MemoryProperties
    BitFields: BitFieldDescriptor list
}

type BitFieldDescriptor = {
    Name: string
    Position: int
    Width: int
    Values: string list option  // Named values if enum-like
}
```

#### Quotation Interpretation Framework

```fsharp
// BAREWire.Interpretation.fs
module BAREWire.Interpretation

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns

/// Interpret a quoted peripheral descriptor at compile time
let interpretPeripheral (expr: Expr<PeripheralDescriptor>) : PeripheralInfo =
    match expr with
    | Value (desc, _) ->
        // Direct value - evaluate immediately
        extractInfo desc
    | NewRecord (_, fields) ->
        // Record construction - extract field values
        let fieldValues = fields |> List.map interpretField
        buildPeripheralInfo fieldValues
    | _ ->
        failwithf "Unsupported quotation structure: %A" expr

/// Interpret a quoted memory region
let interpretRegion (expr: Expr<RegionDescriptor>) : RegionInfo =
    match expr with
    | NewRecord (_, [name; base'; size; props]) ->
        { Name = interpretString name
          Base = interpretUnativeint base'
          Size = interpretUnativeint size
          Properties = interpretProperties props }
    | _ ->
        failwithf "Unsupported region quotation: %A" expr

/// Compose multiple region descriptors
let composeRegions (exprs: Expr<RegionDescriptor> list) : Expr<RegionDescriptor list> =
    let interpreted = exprs |> List.map interpretRegion
    // Validate non-overlapping, build combined model
    validateRegions interpreted
    <@ interpreted @>
```

#### Active Pattern Composition Framework

```fsharp
// BAREWire.Patterns.fs
module BAREWire.Patterns

/// Base active patterns that plugins build upon
let (|VolatileAccess|CacheableAccess|StackAccess|UnknownAccess|)
    (constraint': Expr<RegisterConstraint>) =
    match constraint' with
    | <@ { Access = props } @> when hasFlag props Volatile ->
        VolatileAccess (interpretAccess props)
    | <@ { Access = props } @> when hasFlag props Cacheable ->
        CacheableAccess (interpretAccess props)
    | <@ { Access = props } @> when hasFlag props StackLocal ->
        StackAccess
    | _ ->
        UnknownAccess constraint'

/// Compose plugin-provided patterns with base patterns
let composePatterns
    (basePatterns: (PSGNode -> 'a option) list)
    (pluginPatterns: (PSGNode -> 'a option) list)
    : PSGNode -> 'a option =
    fun node ->
        // Try plugin patterns first (more specific)
        pluginPatterns
        |> List.tryPick (fun p -> p node)
        |> Option.orElseWith (fun () ->
            // Fall back to base patterns
            basePatterns |> List.tryPick (fun p -> p node))

/// Framework for building parameterized patterns
let buildPeripheralPattern
    (family: string)
    (registers: string list)
    : (PSGNode -> PeripheralAccess option) =
    fun node ->
        match node.MemoryConstraint with
        | Some <@ { PeripheralFamily = f; RegisterName = r } @>
            when f = family && List.contains r registers ->
            Some { Family = family; Register = r; Node = node }
        | _ -> None
```

#### Plugin Interface

```fsharp
// BAREWire.Plugins.fs
module BAREWire.Plugins

open Microsoft.FSharp.Quotations

/// Memory model provided by target-specific plugins
/// This is a record, not an interface - pure F# idiom, no BCL inheritance patterns
type MemoryModel = {
    /// Target family identifier (e.g., "stm32l5", "riscv-sifive-u74")
    TargetFamily: string

    /// Quoted peripheral descriptors
    PeripheralDescriptors: Expr<PeripheralDescriptor> list

    /// Quoted register constraints
    RegisterConstraints: Expr<RegisterConstraint> list

    /// Quoted memory region map
    Regions: Expr<RegionDescriptor list>

    /// Active pattern recognition for memory operations
    Recognize: PSGNode -> MemoryOperation option

    /// Optional: Cache topology for cache-aware compilation
    CacheTopology: Expr<CacheLevel list> option

    /// Optional: Coherency model for multi-core targets
    CoherencyModel: Expr<CoherencyPolicy> option
}

/// Plugin loader
module PluginLoader =
    /// Load memory model for target specified in fidproj
    let load (target: string) : MemoryModel =
        // Discovery mechanism - could be:
        // 1. Well-known assembly names
        // 2. NuGet packages
        // 3. Local plugin directory
        ...

    /// Compose multiple memory models (for heterogeneous targets)
    let compose (models: MemoryModel list) : MemoryModel =
        { TargetFamily =
              models |> List.map (_.TargetFamily) |> String.concat "+"
          PeripheralDescriptors =
              models |> List.collect (_.PeripheralDescriptors)
          RegisterConstraints =
              models |> List.collect (_.RegisterConstraints)
          Regions =
              composeRegionQuotations (models |> List.map (_.Regions))
          Recognize = fun node ->
              models |> List.tryPick (fun m -> m.Recognize node)
          CacheTopology =
              models |> List.tryPick (_.CacheTopology)
          CoherencyModel =
              models |> List.tryPick (_.CoherencyModel)
        }
```

### BAREWire Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BAREWire Restructured Architecture                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire.Quotations                                             │   │
│  │  • Core descriptor types (inside quotations)                     │   │
│  │  • PeripheralDescriptor, RegionDescriptor, RegisterConstraint    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire.Interpretation                                         │   │
│  │  • Quotation → concrete value interpretation                     │   │
│  │  • Compile-time evaluation of quoted descriptors                 │   │
│  │  • Validation and composition                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire.Patterns                                               │   │
│  │  • Base active patterns for memory classification                │   │
│  │  • Pattern composition framework                                 │   │
│  │  • Parameterized pattern builders                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire.Plugins                                                │   │
│  │  • MemoryModel record type                                       │   │
│  │  • Plugin discovery and loading                                  │   │
│  │  • Multi-model composition for heterogeneous targets             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BAREWire.Serialization (existing, extended)                     │   │
│  │  • Wire format for memory descriptors                            │   │
│  │  • Schema system integration                                     │   │
│  │  • IPC for distributed compilation                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: fsnative - Nanopass Infrastructure for Quotations

### Core Design

fsnative's PSG nanopasses transform quotations through the compilation pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              fsnative Quotation Transformation Pipeline                 │
│                                                                         │
│  fidproj specifies: target = "stm32l5"                                  │
│                                                                         │
│  Phase 0: Plugin Loading                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Load MemoryModel for target                                       │ │
│  │  Collect: quoted descriptors, active patterns, region map         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 1: PSG Construction (existing)                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  SynExpr → PSG nodes + ChildOf edges                              │ │
│  │  NEW: Nodes get MemoryConstraint: Expr<_> option = None           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 2: Symbol Correlation (existing)                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Attach FSharpSymbol from FCS                                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 3: Memory Constraint Attachment (NEW)                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  For each pointer operation node:                                 │ │
│  │    - Look up address in plugin's quoted region map                │ │
│  │    - Attach matching Expr<RegisterConstraint> to node             │ │
│  │                                                                   │ │
│  │  Quotations are ATTACHED, not yet interpreted                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 4: Typed Tree Overlay (existing)                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Zipper correlates FSharpExpr with PSG nodes                      │ │
│  │  Captures resolved types, constraints, SRTP                       │ │
│  │  NEW: Type constraints inform memory constraint refinement        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 5: Constraint Validation (NEW)                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Interpret quotations to validate:                                │ │
│  │    - Read-only pointers are not written                           │ │
│  │    - Write-only pointers are not read                             │ │
│  │    - Region constraints are satisfied                             │ │
│  │                                                                   │ │
│  │  Emit FS8001-FS8003 errors for violations                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 6: Quotation Specialization (NEW)                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Partial evaluation of quotations for target:                     │ │
│  │    - Resolve symbolic addresses to concrete values                │ │
│  │    - Evaluate conditional constraints                             │ │
│  │    - Simplify composed constraints                                │ │
│  │                                                                   │ │
│  │  Quotations become more concrete as they flow through             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 7: Active Pattern Recognition (NEW)                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Apply plugin's active patterns to classify operations:           │ │
│  │    - GPIOPinWrite → volatile 32-bit store                         │ │
│  │    - USARTTransmit → volatile byte store + status check           │ │
│  │    - DMATransfer → memory barrier + descriptor setup              │ │
│  │                                                                   │ │
│  │  Classification attached to nodes for Alex consumption            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Phase 8+: Existing Enrichment Passes                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Def-use edges, operation classification, etc.                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Alex/XParsec: MLIR Emission                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Traverse PSG with Zipper                                         │ │
│  │  At each node:                                                    │ │
│  │    - Check OperationClassification (from Phase 7)                 │ │
│  │    - Emit appropriate MLIR based on classification                │ │
│  │                                                                   │ │
│  │  Classifications came from active patterns                        │ │
│  │  Active patterns came from plugins                                │ │
│  │  Plugins were generated by Farscape                               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Extended PSG Node Structure

```fsharp
// Extended PSGNode with quotation support
type PSGNode = {
    // Existing fields
    Id: PSGNodeId
    SyntaxKind: string
    SourceRange: range
    Children: PSGNode list
    FSharpSymbol: FSharpSymbol option

    // NEW: Quotation-based memory constraint
    MemoryConstraint: Expr<RegisterConstraint> option

    // NEW: Classified operation (after active pattern recognition)
    OperationClass: MemoryOperationClass option
}

and MemoryOperationClass =
    | VolatileLoad of region: string * width: int
    | VolatileStore of region: string * width: int
    | CacheableLoad of cachePolicy: CachePolicy
    | CacheableStore of cachePolicy: CachePolicy * writePolicy: WritePolicy
    | StackAlloc of size: int * align: int
    | ArenaAlloc of arenaId: string * size: int
    | DMAOperation of dmaDescriptor: Expr<DMAConstraint>
    | AtomicOperation of atomicKind: AtomicKind
    | FenceOperation of fenceKind: FenceKind
    | Unclassified  // Fallback - use default semantics
```

### Nanopass Implementations

#### Phase 3: Memory Constraint Attachment

```fsharp
// fsnative.Nanopasses.MemoryConstraintAttachment.fs
module fsnative.Nanopasses.MemoryConstraintAttachment

open BAREWire.Plugins
open BAREWire.Interpretation

/// Attach quoted memory constraints to pointer operation nodes
let attachConstraints (model: MemoryModel) : PSG -> PSG =
    let regionMap = interpretRegions model.Regions
    let constraintMap =
        model.RegisterConstraints
        |> List.map (fun c -> (interpretConstraintKey c, c))
        |> Map.ofList

    PSG.mapNodes (fun node ->
        match node with
        | PointerOperation ptr ->
            // Determine which region this pointer targets
            let address = extractAddress ptr
            match findRegion regionMap address with
            | Some region ->
                // Look up constraint for this region/register
                let key = (region.Name, extractRegisterName node)
                match Map.tryFind key constraintMap with
                | Some constraint' ->
                    { node with MemoryConstraint = Some constraint' }
                | None ->
                    // No specific constraint - attach region-level constraint
                    { node with MemoryConstraint = Some (regionToConstraint region) }
            | None ->
                // Address not in any known region - likely stack or dynamic
                node
        | _ -> node
    )
```

#### Phase 5: Constraint Validation

```fsharp
// fsnative.Nanopasses.ConstraintValidation.fs
module fsnative.Nanopasses.ConstraintValidation

open BAREWire.Interpretation

/// Validate memory operations against quoted constraints
let validateConstraints : PSG -> Result<PSG, CompilerError list> =
    fun psg ->
        let errors = ResizeArray<CompilerError>()

        psg |> PSG.iterNodes (fun node ->
            match node.MemoryConstraint with
            | Some constraint' ->
                let access = interpretAccess constraint'

                // Check read operations
                if isReadOperation node && not (canRead access) then
                    errors.Add {
                        Code = "FS8001"
                        Message = sprintf "Cannot read from write-only register '%s'"
                                         (extractRegisterName node)
                        Range = node.SourceRange
                    }

                // Check write operations
                if isWriteOperation node && not (canWrite access) then
                    errors.Add {
                        Code = "FS8002"
                        Message = sprintf "Cannot write to read-only register '%s'"
                                         (extractRegisterName node)
                        Range = node.SourceRange
                    }

                // Check width constraints
                let opWidth = extractOperationWidth node
                let regWidth = interpretWidth constraint'
                if opWidth <> regWidth then
                    errors.Add {
                        Code = "FS8004"
                        Message = sprintf "Access width mismatch: operation is %d bits, register '%s' requires %d bits"
                                         opWidth (extractRegisterName node) regWidth
                        Range = node.SourceRange
                    }
            | None -> ()
        )

        if errors.Count = 0 then Ok psg
        else Error (errors |> Seq.toList)
```

#### Phase 7: Active Pattern Recognition

```fsharp
// fsnative.Nanopasses.PatternRecognition.fs
module fsnative.Nanopasses.PatternRecognition

open BAREWire.Plugins
open BAREWire.Patterns

/// Apply memory model's recognition function to classify operations
let classifyOperations (model: MemoryModel) : PSG -> PSG =
    let recognize = model.Recognize

    PSG.mapNodes (fun node ->
        match node.MemoryConstraint with
        | Some _ ->
            // Try to classify using memory model's recognition function
            match recognize node with
            | Some (VolatileWrite (region, width, value)) ->
                { node with OperationClass = Some (VolatileStore (region, width)) }
            | Some (VolatileRead (region, width)) ->
                { node with OperationClass = Some (VolatileLoad (region, width)) }
            | Some (DMATransfer desc) ->
                { node with OperationClass = Some (DMAOperation desc) }
            | Some (AtomicRMW kind) ->
                { node with OperationClass = Some (AtomicOperation kind) }
            | None ->
                // No specific pattern matched - use default based on constraint
                let defaultClass = deriveDefaultClass node.MemoryConstraint.Value
                { node with OperationClass = Some defaultClass }
        | None ->
            // No constraint - leave unclassified
            { node with OperationClass = Some Unclassified }
    )

/// Default classification from constraint when no pattern matches
let private deriveDefaultClass (constraint': Expr<RegisterConstraint>) : MemoryOperationClass =
    match constraint' with
    | VolatileAccess access ->
        match access with
        | Read -> VolatileLoad ("unknown", 32)
        | Write -> VolatileStore ("unknown", 32)
        | ReadWrite -> VolatileStore ("unknown", 32)  // Assume store for RMW
    | CacheableAccess policy ->
        CacheableStore (policy, WriteBack)
    | StackAccess ->
        StackAlloc (0, 4)  // Will be refined by later passes
    | UnknownAccess _ ->
        Unclassified
```

### Alex Integration

```fsharp
// Alex.Emission.MemoryOperations.fs
module Alex.Emission.MemoryOperations

open fsnative.PSG

/// Emit MLIR for classified memory operations
let emitMemoryOp (node: PSGNode) : MLIRBuilder<unit> =
    mlir {
        match node.OperationClass with
        | Some (VolatileStore (region, width)) ->
            let! ptr = emitPointer node
            let! value = emitValue (getStoreValue node)
            do! emitVolatileStore width ptr value

        | Some (VolatileLoad (region, width)) ->
            let! ptr = emitPointer node
            let! result = emitVolatileLoad width ptr
            do! bindResult node.Id result

        | Some (DMAOperation dmaDesc) ->
            // DMA requires memory barriers and descriptor setup
            do! emitFence Acquire
            let! desc = emitDMADescriptor dmaDesc
            do! emitDMAStart desc
            do! emitFence Release

        | Some (AtomicOperation kind) ->
            let! ptr = emitPointer node
            let! value = emitValue (getStoreValue node)
            let! result = emitAtomic kind ptr value
            do! bindResult node.Id result

        | Some (CacheableStore (policy, writePolicy)) ->
            let! ptr = emitPointer node
            let! value = emitValue (getStoreValue node)
            match writePolicy with
            | WriteThrough -> do! emitStoreWriteThrough ptr value
            | WriteBack -> do! emitStore ptr value  // Normal store

        | Some Unclassified | None ->
            // Default emission - normal load/store
            do! emitDefaultMemoryOp node
    }
```

---

## Part 4: Alloy - Base Native Library Articulation

### Design Principle

Alloy is the **consumption surface**, not the **definition surface** for memory semantics. It provides:

1. **User-facing API** that is BCL-sympathetic
2. **Type constraints** that reference phantom measures
3. **Platform binding declarations** that Alex recognizes
4. **No knowledge** of specific hardware targets

### Alloy's Relationship to Memory Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Alloy in the Memory Architecture                     │
│                                                                         │
│  User Code                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  let value = Memory.read ptr                                      │ │
│  │  Memory.write ptr newValue                                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Alloy.Memory (API Layer)                                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Provides: Generic memory operations with type constraints        │ │
│  │  Enforces: 'access :> readable, 'access :> writable, etc.        │ │
│  │  Does NOT: Know what "readable" means for specific hardware       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  Alloy.Platform.Bindings (Declaration Layer)                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Declares: Primitive operations as Unchecked.defaultof<_>         │ │
│  │  Examples: read, write, fence, atomicCAS, etc.                    │ │
│  │  Alex recognizes and implements these                             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  fsnative + Plugins (Semantic Layer)                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Plugin provides: Quoted constraints for hardware                 │ │
│  │  Nanopass attaches: Constraints to operation nodes                │ │
│  │  Active patterns classify: Based on plugin knowledge              │ │
│  │  Alex emits: Correct MLIR for classified operations               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Alloy Memory Module

```fsharp
// Alloy/Memory.fs
module Alloy.Memory

open Alloy.Measures

// ============================================================================
// Type Constraints (Alloy defines the vocabulary)
// ============================================================================

/// Readable memory access
type readable = interface end

/// Writable memory access
type writable = interface end

/// Read-write access (combines readable and writable)
type readWrite =
    interface
        inherit readable
        inherit writable
    end

/// Read-only access
type readOnly =
    interface
        inherit readable
    end

/// Write-only access
type writeOnly =
    interface
        inherit writable
    end

// ============================================================================
// Pointer Types (Alloy defines the shape)
// ============================================================================

/// Native pointer with region and access phantom types
/// The actual measures are defined by plugins, not Alloy
[<Struct>]
type NativePtr<'T, [<Measure>] 'region, 'access when 'access :> readable or 'access :> writable> =
    val mutable private address: unativeint

    /// Create pointer from address (unsafe - requires plugin validation)
    static member inline ofAddress (addr: unativeint) : NativePtr<'T, 'region, 'access> =
        let mutable p = Unchecked.defaultof<NativePtr<'T, 'region, 'access>>
        p.address <- addr
        p

// ============================================================================
// Memory Operations (Alloy provides API, Alex provides implementation)
// ============================================================================

/// Read from memory
let inline read<'T, [<Measure>] 'region, 'access when 'access :> readable>
    (ptr: NativePtr<'T, 'region, 'access>) : 'T =
    Platform.Bindings.memoryRead ptr

/// Write to memory
let inline write<'T, [<Measure>] 'region, 'access when 'access :> writable>
    (ptr: NativePtr<'T, 'region, 'access>) (value: 'T) : unit =
    Platform.Bindings.memoryWrite ptr value

/// Volatile read (explicit - bypasses cache)
let inline volatileRead<'T, [<Measure>] 'region, 'access when 'access :> readable>
    (ptr: NativePtr<'T, 'region, 'access>) : 'T =
    Platform.Bindings.volatileRead ptr

/// Volatile write (explicit - bypasses cache)
let inline volatileWrite<'T, [<Measure>] 'region, 'access when 'access :> writable>
    (ptr: NativePtr<'T, 'region, 'access>) (value: 'T) : unit =
    Platform.Bindings.volatileWrite ptr value

/// Memory fence
let inline fence (kind: FenceKind) : unit =
    Platform.Bindings.fence kind

/// Atomic compare-and-swap
let inline atomicCAS<'T, [<Measure>] 'region, 'access when 'access :> readWrite>
    (ptr: NativePtr<'T, 'region, 'access>)
    (expected: 'T)
    (desired: 'T) : bool =
    Platform.Bindings.atomicCAS ptr expected desired

// ============================================================================
// Platform Bindings (Alex recognizes and implements these)
// ============================================================================

module Platform.Bindings =
    let memoryRead<'T, [<Measure>] 'region, 'access>
        (ptr: NativePtr<'T, 'region, 'access>) : 'T =
        Unchecked.defaultof<'T>

    let memoryWrite<'T, [<Measure>] 'region, 'access>
        (ptr: NativePtr<'T, 'region, 'access>) (value: 'T) : unit =
        ()

    let volatileRead<'T, [<Measure>] 'region, 'access>
        (ptr: NativePtr<'T, 'region, 'access>) : 'T =
        Unchecked.defaultof<'T>

    let volatileWrite<'T, [<Measure>] 'region, 'access>
        (ptr: NativePtr<'T, 'region, 'access>) (value: 'T) : unit =
        ()

    let fence (kind: FenceKind) : unit = ()

    let atomicCAS<'T, [<Measure>] 'region, 'access>
        (ptr: NativePtr<'T, 'region, 'access>) (expected: 'T) (desired: 'T) : bool =
        Unchecked.defaultof<bool>
```

### Alloy Does NOT Know About Hardware

Critical distinction:

```fsharp
// WRONG - Alloy should NOT contain this
module Alloy.Hardware.STM32 =
    let GPIOA_BASE = 0x48000000un  // NO! Hardware knowledge in Alloy

// RIGHT - Hardware knowledge comes from plugins
// User code references plugin-provided values:
open CMSIS.STM32L5.GPIO

let led = GPIOA  // Defined in plugin-generated code, not Alloy
Memory.write led.ODR 0x20u  // Alloy API, plugin constraint, Alex emission
```

Alloy provides:
- `Memory.read`, `Memory.write` - generic operations
- `NativePtr<'T, 'region, 'access>` - typed pointer structure
- `readable`, `writable`, `readWrite` - constraint vocabulary

Alloy does NOT provide:
- Peripheral base addresses
- Register layouts
- Memory region definitions
- Cache policies
- Hardware-specific anything

### How Constraints Flow Through

```fsharp
// User writes:
let value = Memory.read gpio.IDR

// Alloy provides the API shape:
//   read<'T, 'region, 'access when 'access :> readable>

// Plugin provides the constraint:
//   gpio.IDR : NativePtr<uint32, peripheral, readOnly>
//   Quoted: <@ { Region = "Peripheral"; Access = ReadOnly; Volatile = true } @>

// fsnative validates:
//   readOnly :> readable ✓
//   ReadOnly does not include writable ✓

// Active pattern classifies:
//   GPIOPinRead (GPIOA, IDR) → VolatileLoad ("Peripheral", 32)

// Alex emits:
//   %ptr = llvm.mlir.constant(0x48000010 : i64) : i64
//   %result = llvm.load volatile %ptr : !llvm.ptr -> i32
```

### Alloy's Measure Vocabulary

Alloy defines the **vocabulary** of measures, but plugins define the **instances**:

```fsharp
// Alloy/Measures.fs - Vocabulary
module Alloy.Measures

/// Memory region measure (abstract - instances from plugins)
[<Measure>] type region

/// Access kind measure (abstract - instances from plugins)
[<Measure>] type access

// Plugins provide concrete measures:
// [<Measure>] type peripheral  // STM32 plugin
// [<Measure>] type sram        // STM32 plugin
// [<Measure>] type globalMem   // GPU plugin
// [<Measure>] type sharedMem   // GPU plugin
```

This separation ensures:
1. Alloy remains hardware-agnostic
2. New hardware targets don't require Alloy changes
3. Type constraints are validated against plugin-provided measures
4. The vocabulary is consistent across all targets

---

## Part 5: Integration Example - Complete Flow

### Scenario: GPIO Pin Write on STM32L5

#### Step 1: User Code (uses Alloy API)

```fsharp
// User application
open Alloy.Memory
open CMSIS.STM32L5.GPIO

let setLed () =
    Memory.write GPIOA.BSRR GPIO_PIN_5  // Set pin 5 high
```

#### Step 2: Plugin Provides Constraints (Farscape-generated)

```fsharp
// CMSIS.STM32L5.GPIO.Descriptors (generated by Farscape)
let bsrrConstraint = <@
    { PeripheralFamily = "GPIO"
      RegisterName = "BSRR"
      Offset = 0x18
      Access = WriteOnly
      Width = 32
      Volatile = true
      Documentation = "Bit Set/Reset Register - write 1 to set/reset pins" }
@>

// CMSIS.STM32L5.GPIO.Patterns (generated by Farscape)
let (|GPIOPinSet|_|) node =
    match node with
    | FunctionCall (WriteOp, [port; value]) ->
        match port with
        | GPIOPortAccess (instance, "BSRR") ->
            Some (GPIOPinSet (instance, value))
        | _ -> None
    | _ -> None
```

#### Step 3: fsnative Nanopasses Transform

```
PSG after Phase 1 (Construction):
  FunctionCall
    ├── Symbol: Memory.write
    ├── Arg0: MemberAccess (GPIOA, BSRR)
    └── Arg1: Const (GPIO_PIN_5 = 0x20)

PSG after Phase 3 (Constraint Attachment):
  FunctionCall
    ├── Symbol: Memory.write
    ├── Arg0: MemberAccess (GPIOA, BSRR)
    │         MemoryConstraint: <@ { Access = WriteOnly; Volatile = true; ... } @>
    └── Arg1: Const (0x20)

PSG after Phase 5 (Validation):
  ✓ WriteOnly :> writable - OK
  ✓ Width 32 matches uint32 - OK

PSG after Phase 7 (Classification):
  FunctionCall
    ├── OperationClass: VolatileStore ("Peripheral", 32)
    └── ...
```

#### Step 4: Alex Emits MLIR

```mlir
// Generated MLIR
%base = llvm.mlir.constant(0x48000000 : i64) : i64
%offset = llvm.mlir.constant(0x18 : i64) : i64
%addr = llvm.add %base, %offset : i64
%ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
%value = llvm.mlir.constant(0x20 : i32) : i32
llvm.store volatile %value, %ptr : i32, !llvm.ptr
```

#### Step 5: LLVM Lowers to ARM

```asm
; Generated ARM assembly
ldr r0, =0x48000018    ; GPIOA->BSRR address
mov r1, #0x20          ; GPIO_PIN_5
str r1, [r0]           ; Volatile store (no reordering)
```

---

## Summary: Component Responsibilities

| Component | Provides | Consumes | Does NOT |
|-----------|----------|----------|----------|
| **Farscape** | Quoted descriptors, Generated active patterns | C/C++ headers | Know about PSG or MLIR |
| **BAREWire** | Quotation infrastructure, Pattern composition, MemoryModel type | Plugin definitions | Know about specific hardware |
| **fsnative** | Nanopass pipeline, Constraint validation, Pattern recognition | Plugins, Alloy code | Define hardware layouts |
| **Alloy** | API vocabulary, Type constraints, Platform bindings | fsnative measures | Define memory regions |
| **Alex** | MLIR emission | Classified PSG nodes | Know about quotation details |

---

## Implementation Roadmap

### Phase 1: BAREWire Foundation
1. Define quotation types in `BAREWire.Quotations`
2. Implement interpretation framework in `BAREWire.Interpretation`
3. Define `MemoryModel` record type in `BAREWire.Plugins`
4. Create base active patterns in `BAREWire.Patterns`

### Phase 2: Farscape Code Generation
1. Extend Farscape to emit quoted descriptors
2. Add active pattern generation from parsed headers
3. Generate plugin registration code
4. Test with STM32L5 CMSIS headers

### Phase 3: fsnative Nanopass Integration
1. Extend PSGNode with `MemoryConstraint` field
2. Implement constraint attachment nanopass
3. Implement constraint validation nanopass
4. Implement pattern recognition nanopass
5. Integrate with existing typed tree overlay

### Phase 4: Alloy Articulation
1. Define measure vocabulary in `Alloy.Measures`
2. Define `NativePtr` with phantom types
3. Define memory operation API
4. Ensure Platform.Bindings align with Alex recognition

### Phase 5: Alex Integration
1. Extend Alex to consume `OperationClass`
2. Implement MLIR emission for each operation class
3. Test end-to-end with GPIO example

---

## Related Documents

| Document | Location |
|----------|----------|
| Staged Memory Model | `/docs/Staged_Memory_Model.md` |
| Memory Interlock Requirements | `/docs/Memory_Interlock_Requirements.md` |
| PSG Nanopass Architecture | `/docs/PSG_Nanopass_Architecture.md` |
| BAREWire Hardware Descriptors | `~/repos/BAREWire/docs/08 Hardware Descriptors.md` |
| Farscape Architecture | `~/repos/Farscape/docs/01_Architecture_Overview.md` |
