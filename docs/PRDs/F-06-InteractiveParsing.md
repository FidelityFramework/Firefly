# F-06: Interactive Parsing

> **Sample**: `06_AddNumbersInteractive` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample extends DU handling with string parsing, type detection, and interactive user input. It demonstrates multi-way pattern matching with tuple patterns and type promotion.

**Key Achievement**: Established `Parse` intrinsics (`Parse.int`, `Parse.float`) and compound control flow with multiple pattern matches.

---

## 2. Surface Feature

```fsharp
module AddNumbersInteractive

type Number =
    | IntVal of int
    | FloatVal of float

let parseNumber (s: string) : Number =
    if String.contains s '.' then
        FloatVal (Parse.float s)
    else
        IntVal (Parse.int s)

let add (a: Number) (b: Number) =
    match a, b with
    | IntVal x, IntVal y -> IntVal (x + y)
    | FloatVal x, FloatVal y -> FloatVal (x + y)
    | IntVal x, FloatVal y -> FloatVal (float x + y)
    | FloatVal x, IntVal y -> FloatVal (x + float y)

let arena = Arena.create<'a> 1024
let input1 = Console.readlnFrom &arena
let input2 = Console.readlnFrom &arena
let result = add (parseNumber input1) (parseNumber input2)
Console.writeln $"Result: {result}"
```

---

## 3. Infrastructure Contributions

### 3.1 String.contains Intrinsic

Character search for decimal point detection:

```fsharp
// In FNCS CheckExpressions.fs
| "String.contains" ->
    // string -> char -> bool
    NativeType.TFun(env.Globals.StringType,
        NativeType.TFun(env.Globals.CharType, env.Globals.BoolType))
```

**Implementation**: Linear scan through UTF-8 bytes looking for target character.

### 3.2 Parse Intrinsics

String-to-number conversion:

```fsharp
| "Parse.int" ->
    // string -> int
    NativeType.TFun(env.Globals.StringType, env.Globals.IntType)

| "Parse.float" ->
    // string -> float
    NativeType.TFun(env.Globals.StringType, env.Globals.Float64Type)
```

**Implementation**:
- `Parse.int`: ASCII digit accumulation with sign handling
- `Parse.float`: Mantissa + exponent parsing (IEEE 754)

### 3.3 Type Promotion

The `float` function converts int to float:

```fsharp
| FloatVal x, IntVal y -> FloatVal (x + float y)
```

**Intrinsic**:
```fsharp
| "float" ->
    // int -> float
    NativeType.TFun(env.Globals.IntType, env.Globals.Float64Type)
```

**MLIR**: `arith.sitofp` (signed int to floating point)

### 3.4 Multiple Console Reads

Each `Console.readlnFrom` call gets independent SSA:

```mlir
%input1 = func.call @Console.readln() : () -> memref<?xi8>
%input2 = func.call @Console.readln() : () -> memref<?xi8>
// %input1 and %input2 are distinct SSA values
```

### 3.5 Tuple Pattern Matching

The `match a, b with` creates tuple patterns:

```
Match
├── Targets: Tuple[(a), (b)]
└── Cases:
    ├── Pattern: Tuple[IntVal x, IntVal y]
    │   └── Body: ...
    └── ...
```

Baker lowers this to nested conditionals checking both components.

---

## 4. PSG Representation

```
ModuleOrNamespace: AddNumbersInteractive
├── TypeDefinition: Number
│
├── LetBinding: parseNumber
│   └── Lambda: s ->
│       └── IfThenElse
│           ├── Condition: String.contains s '.'
│           ├── Then: FloatVal (Parse.float s)
│           └── Else: IntVal (Parse.int s)
│
├── LetBinding: add
│   └── Lambda: a, b ->
│       └── Match (tuple pattern)
│
├── LetBinding: arena
├── LetBinding: input1
├── LetBinding: input2
├── LetBinding: result
│   └── Application: add (parseNumber input1) (parseNumber input2)
│
└── Application: Console.writeln
```

---

## 5. Control Flow Graph

```
┌─────────────────┐
│ parseNumber(s)  │
└────────┬────────┘
         │
    ┌────▼────┐
    │contains?│
    └────┬────┘
      T  │  F
   ┌─────┼─────┐
   ▼           ▼
┌──────┐   ┌──────┐
│float │   │ int  │
│parse │   │parse │
└──┬───┘   └──┬───┘
   │          │
   ▼          ▼
┌──────┐   ┌──────┐
│Float │   │ Int  │
│Val   │   │ Val  │
└──────┘   └──────┘
```

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all bindings and intermediate values |
| PatternBindings | SSA for `x`, `y` in match patterns |

---

## 7. MLIR Output Pattern

```mlir
// parseNumber function
// String is memref<?xi8>. Number DU is memref<9xi8> (1 tag + 8 payload).
func.func @parseNumber(%s: memref<?xi8>) -> memref<9xi8> {
  // Check for decimal point — String.contains is a byte-scan loop (Alex pattern)
  %c46_i8 = arith.constant 46 : i8  // '.'
  // ... (scf.while loop scanning %s for %c46_i8, produces %has_dot : i1)

  // Expression-valued scf.if — returns the DU memref
  %result = scf.if %has_dot -> memref<9xi8> {
    %f = func.call @Parse.float(%s) : (memref<?xi8>) -> f64
    // Construct FloatVal: tag=1, payload=f64
    %du = memref.alloca() : memref<9xi8>
    %c1_tag = arith.constant 1 : i8
    %tag_ref = memref.reinterpret_cast %du to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
    memref.store %c1_tag, %tag_ref[%c0] : memref<1xi8>
    %payload_ref = memref.view %du[%c1][] : memref<9xi8> to memref<1xf64>
    memref.store %f, %payload_ref[%c0] : memref<1xf64>
    scf.yield %du : memref<9xi8>
  } else {
    %i = func.call @Parse.int(%s) : (memref<?xi8>) -> i64
    // Construct IntVal: tag=0, payload=i64
    %du = memref.alloca() : memref<9xi8>
    %c0_tag = arith.constant 0 : i8
    %tag_ref = memref.reinterpret_cast %du to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
    memref.store %c0_tag, %tag_ref[%c0] : memref<1xi8>
    %payload_ref = memref.view %du[%c1][] : memref<9xi8> to memref<1xi64>
    memref.store %i, %payload_ref[%c0] : memref<1xi64>
    scf.yield %du : memref<9xi8>
  }
  func.return %result : memref<9xi8>
}
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/06_AddNumbersInteractive
/path/to/Firefly compile AddNumbers.fidproj
echo -e "10\n2.5" | ./AddNumbers
# Output: Result: FloatVal 12.5
```

---

## 9. Architectural Lessons

1. **Parse as Intrinsics**: No runtime parsing library - compiler provides primitives
2. **Compound Control Flow**: Multiple patterns + conditionals compose cleanly
3. **SSA Isolation**: Each read gets distinct SSA, no aliasing confusion
4. **Type Promotion in MLIR**: `float` conversion uses `sitofp` instruction

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-08**: Option type (parsing that may fail)
- **F-09**: Result type (parsing with error info)
- **I-01**: Socket parsing (network protocols)

---

## 11. Related Documents

- [F-05-DiscriminatedUnions](F-05-DiscriminatedUnions.md) - DU foundation
- [F-08-OptionType](F-08-OptionType.md) - Option for fallible parsing
- [FNCS_Architecture.md](../FNCS_Architecture.md) - Intrinsic design
