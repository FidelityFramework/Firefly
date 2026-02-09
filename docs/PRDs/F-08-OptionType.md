# F-08: Option Type

> **Sample**: `08_Option` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample implements the canonical F# `Option<'T>` type as a homogeneous discriminated union. Option represents nullable values without null references, using `Some` and `None` constructors.

**Key Achievement**: Established inline struct representation for homogeneous DUs and `OptionRecipes` in Baker for HOF decomposition.

---

## 2. Surface Feature

```fsharp
module OptionTest

let maybeValue = Some 42
let noValue: int option = None

let describe (opt: int option) =
    match opt with
    | Some x -> Console.writeln $"Value: {x}"
    | None -> Console.writeln "No value"

describe maybeValue
describe noValue

// Option operations
let doubled = Option.map (fun x -> x * 2) maybeValue
```

---

## 3. Infrastructure Contributions

### 3.1 Option Type Representation

Option is a homogeneous DU with inline struct layout:

```
Option<T> Layout
┌─────────────────────────────────────┐
│ tag: i8     (0 = None, 1 = Some)   │
│ value: T    (payload slot)         │
└─────────────────────────────────────┘
```

**Properties**:
- Stack-allocated (no arena needed)
- Fixed size: `sizeof(i8) + alignment + sizeof(T)`
- `None` has undefined value slot (tag = 0 is sufficient)

### 3.2 FNCS Type Definition

```fsharp
// Option<'T> as NTU type
| "Option" ->
    NativeType.TUnion [
        ("None", NativeType.TUnit)
        ("Some", typeParam)
    ]
```

### 3.3 Option Intrinsics

| Intrinsic | Type | Purpose |
|-----------|------|---------|
| `Option.isSome` | `'T option -> bool` | Check if Some |
| `Option.isNone` | `'T option -> bool` | Check if None |
| `Option.get` | `'T option -> 'T` | Extract value (unchecked) |
| `Option.defaultValue` | `'T -> 'T option -> 'T` | Value or default |

**Implementation**:

```mlir
// Option.isSome — Option is memref<9xi8> (1 tag + 8 payload)
%tag_ref = memref.reinterpret_cast %opt to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
%tag = memref.load %tag_ref[%c0] : memref<1xi8>
%c1 = arith.constant 1 : i8
%is_some = arith.cmpi eq, %tag, %c1 : i8

// Option.get — extract payload via memref.view
%payload_ref = memref.view %opt[%c1][] : memref<9xi8> to memref<1xi64>
%value = memref.load %payload_ref[%c0] : memref<1xi64>
```

### 3.4 OptionRecipes in Baker

Higher-order Option operations decompose in Baker:

```fsharp
Option.map f opt
```

Decomposes to:

```fsharp
match opt with
| Some x -> Some (f x)
| None -> None
```

**Recipe Definitions**:
```fsharp
// In Baker OptionRecipes
| "Option.map" -> decomposeToMatch (fun x -> Some (f x)) None
| "Option.bind" -> decomposeToMatch (fun x -> f x) None
| "Option.filter" -> decomposeToMatch (fun x -> if pred x then Some x else None) None
```

---

## 4. PSG Representation

```
ModuleOrNamespace: OptionTest
├── LetBinding: maybeValue
│   └── UnionCase: Some
│       └── Argument: 42
│
├── LetBinding: noValue
│   └── UnionCase: None
│
├── LetBinding: describe
│   └── Lambda: opt ->
│       └── Match
│           ├── Case: Some x -> ...
│           └── Case: None -> ...
│
├── Application: describe maybeValue
├── Application: describe noValue
│
└── LetBinding: doubled
    └── Application: Option.map
        ├── Lambda: x -> x * 2
        └── maybeValue
```

---

## 5. Pattern Match Lowering

After Baker saturation:

```
Match (opt)
└── IfThenElse
    ├── Condition: opt.tag == 1  (is Some)
    ├── Then:
    │   └── Let x = opt.value
    │       └── Console.writeln ...
    └── Else:
        └── Console.writeln "No value"
```

---

## 6. MLIR Patterns

### 6.1 Option Type

Option is represented as `memref<Nxi8>` (flat byte buffer: 1 byte tag + payload):

```mlir
// Option<i32> is memref<5xi8> (1 tag + 4 payload, assuming i32 alignment)
// In practice with alignment: memref<9xi8> (1 tag + padding + 8 payload for i64)
```

### 6.2 Construction

```mlir
// Some 42
%opt = memref.alloca() : memref<9xi8>
%tag_ref = memref.reinterpret_cast %opt to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
%tag_some = arith.constant 1 : i8
memref.store %tag_some, %tag_ref[%c0] : memref<1xi8>
%payload_ref = memref.view %opt[%c1][] : memref<9xi8> to memref<1xi32>
%value = arith.constant 42 : i32
memref.store %value, %payload_ref[%c0] : memref<1xi32>

// None
%none = memref.alloca() : memref<9xi8>
%none_tag_ref = memref.reinterpret_cast %none to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
%tag_none = arith.constant 0 : i8
memref.store %tag_none, %none_tag_ref[%c0] : memref<1xi8>
// value slot left undefined
```

### 6.3 Pattern Dispatch

```mlir
%tag_ref = memref.reinterpret_cast %opt to offset: [0], sizes: [1], strides: [1] : memref<9xi8> to memref<1xi8>
%tag = memref.load %tag_ref[%c0] : memref<1xi8>
%c1_tag = arith.constant 1 : i8
%is_some = arith.cmpi eq, %tag, %c1_tag : i8

%result = scf.if %is_some -> (!result_type) {
  // Some case — extract payload via memref.view
  %payload_ref = memref.view %opt[%c1][] : memref<9xi8> to memref<1xi32>
  %x = memref.load %payload_ref[%c0] : memref<1xi32>
  // use x
  scf.yield %some_result : !result_type
} else {
  // None handling
  scf.yield %none_result : !result_type
}
```

---

## 7. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for bindings and extractions |
| PatternBindings | SSA for `x` in `Some x` pattern |

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/08_Option
/path/to/Firefly compile OptionTest.fidproj
./OptionTest
# Output:
# Value: 42
# No value
```

---

## 9. Architectural Lessons

1. **Homogeneous DU Inline**: Same-sized payloads fit in inline struct
2. **No Null References**: Option is the only way to represent "maybe no value"
3. **Baker Decomposition**: HOFs like `map` lower to primitives
4. **Type Safety**: `Option.get` on None is undefined - use pattern matching

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-09**: Result type (extends DU pattern with error case)
- **C-04**: Collection operations returning Option
- **C-06, C-07**: Seq operations (Seq.tryFind returns Option)

---

## 11. Related Documents

- [F-05-DiscriminatedUnions](F-05-DiscriminatedUnions.md) - DU foundation
- [F-09-ResultType](F-09-ResultType.md) - Error-carrying alternative
- [Baker_Saturation_Architecture.md](../Baker_Saturation_Architecture.md) - Recipe decomposition
