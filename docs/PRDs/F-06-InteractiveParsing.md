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

**MLIR**: `llvm.sitofp` (signed int to floating point)

### 3.4 Multiple Console Reads

Each `Console.readlnFrom` call gets independent SSA:

```mlir
%input1 = llvm.call @console_readln_from(%arena)
%input2 = llvm.call @console_readln_from(%arena)
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
llvm.func @parseNumber(%s: !llvm.ptr, %s_len: i64) -> !number_t {
  // Check for decimal point
  %dot = llvm.mlir.constant(46 : i8)  // '.'
  %has_dot = llvm.call @string_contains(%s, %s_len, %dot)

  llvm.cond_br %has_dot, ^float_case, ^int_case

^float_case:
  %f = llvm.call @parse_float(%s, %s_len)
  %fbits = llvm.bitcast %f : f64 to i64
  // Construct FloatVal
  ...
  llvm.br ^return(%floatval)

^int_case:
  %i = llvm.call @parse_int(%s, %s_len)
  // Construct IntVal
  ...
  llvm.br ^return(%intval)

^return(%result: !number_t):
  llvm.return %result
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
