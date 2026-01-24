# F-05: Discriminated Unions

> **Sample**: `05_AddNumbers` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces discriminated unions (DUs), the fundamental sum type of F#. A homogeneous DU (`Number = IntVal | FloatVal`) demonstrates tag-based dispatch, pattern matching, and payload extraction.

**Key Achievement**: Established DU representation with inline struct layout `{tag, payload}` and Baker's `MatchRecipes` for pattern match lowering.

---

## 2. Surface Feature

```fsharp
module AddNumbers

type Number =
    | IntVal of int
    | FloatVal of float

let add (a: Number) (b: Number) =
    match a, b with
    | IntVal x, IntVal y -> IntVal (x + y)
    | FloatVal x, FloatVal y -> FloatVal (x + y)
    | IntVal x, FloatVal y -> FloatVal (float x + y)
    | FloatVal x, IntVal y -> FloatVal (x + float y)

let result = add (IntVal 1) (FloatVal 2.5)
```

---

## 3. Infrastructure Contributions

### 3.1 DU Representation

Homogeneous DUs (all cases have same payload size) use inline struct layout:

```
Number DU Layout
┌─────────────────────────────────────┐
│ tag: i8    (0 = IntVal, 1 = FloatVal)
│ payload: i64  (union slot)          │
└─────────────────────────────────────┘
```

**Size**: 1 (tag) + 7 (padding) + 8 (payload) = 16 bytes

### 3.2 Homogeneous vs Heterogeneous

| Type | Payload Types | Layout | Allocation |
|------|---------------|--------|------------|
| Homogeneous | All same size | Inline struct | Stack |
| Heterogeneous | Different sizes | Arena pointer | Arena |

`Number` is homogeneous because both `int` and `float` fit in 8 bytes.

### 3.3 Bits Coercion

Float payloads stored in int64 slots via bit reinterpretation:

```fsharp
// Storing FloatVal 2.5
let bits = Bits.float64ToInt64Bits 2.5  // Reinterpret, no conversion
// bits = 4612811918334230528L (IEEE 754 representation)
```

This enables uniform payload slot without type-specific variants.

### 3.4 Pattern Match Lowering

`MatchRecipes` in Baker lowers pattern matches to decision trees:

```
match a, b with
| IntVal x, IntVal y -> ...
```

Becomes:

```
IfThenElse
├── Condition: a.tag == 0 && b.tag == 0
├── Then: extract x from a.payload, y from b.payload, ...
└── Else: next pattern
```

### 3.5 Tag Extraction

Tags are extracted via `TupleGet` on the DU struct:

```mlir
%tag = llvm.extractvalue %du[0] : !llvm.struct<(i8, i64)>
%is_intval = llvm.icmp "eq" %tag, %zero : i8
```

### 3.6 Payload Extraction

Payloads are extracted with type-aware casting:

```mlir
// For IntVal (already i64)
%payload = llvm.extractvalue %du[1] : !llvm.struct<(i8, i64)>

// For FloatVal (need bitcast)
%payload_bits = llvm.extractvalue %du[1]
%payload_float = llvm.bitcast %payload_bits : i64 to f64
```

---

## 4. PSG Representation

```
ModuleOrNamespace: AddNumbers
├── TypeDefinition: Number (DU)
│   ├── Case: IntVal of int
│   └── Case: FloatVal of float
│
├── LetBinding: add
│   └── Lambda: a, b ->
│       └── Match
│           ├── Targets: [(a, b)]
│           └── Cases:
│               ├── (IntVal x, IntVal y) -> ...
│               ├── (FloatVal x, FloatVal y) -> ...
│               ├── (IntVal x, FloatVal y) -> ...
│               └── (FloatVal x, IntVal y) -> ...
│
└── LetBinding: result
    └── Application: add (IntVal 1) (FloatVal 2.5)
```

---

## 5. Baker Saturation

`DUConstruct` lowering transforms union case construction:

```
Before (PSG):
  UnionCase: IntVal
  └── Argument: 1

After (Baker):
  DUConstruct
  ├── Tag: 0
  └── Payload: 1 (coerced to i64 slot)
```

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all nodes including pattern bindings |
| PatternBindings | SSA assignments for `x`, `y` in pattern matches |

**PatternBindings** is critical - it ensures extracted values get SSA slots.

---

## 7. MLIR Output Pattern

```mlir
// DU struct type
!number_t = !llvm.struct<(i8, i64)>

// Construct IntVal 1
%tag_int = llvm.mlir.constant(0 : i8)
%payload_int = llvm.mlir.constant(1 : i64)
%intval = llvm.mlir.undef : !number_t
%intval_1 = llvm.insertvalue %tag_int, %intval[0]
%intval_2 = llvm.insertvalue %payload_int, %intval_1[1]

// Pattern match dispatch
%a_tag = llvm.extractvalue %a[0] : !number_t
%is_int = llvm.icmp "eq" %a_tag, %zero
llvm.cond_br %is_int, ^intval_case, ^floatval_case

^intval_case:
  %a_payload = llvm.extractvalue %a[1] : !number_t
  // ... continue matching b
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/05_AddNumbers
/path/to/Firefly compile AddNumbers.fidproj
./AddNumbers
# Output: Result: FloatVal 3.5
```

---

## 9. Architectural Lessons

1. **Representation Fidelity**: DU layout computed by compiler, not library conventions
2. **Bits Coercion**: Type reinterpretation enables uniform payload slots
3. **Baker Decomposition**: Pattern matches lowered before Alex sees them
4. **PatternBindings Coeffect**: Extracted values need explicit SSA tracking

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-07**: Bits intrinsics (extends coercion patterns)
- **F-08**: Option type (canonical homogeneous DU)
- **F-09**: Result type (heterogeneous DU extension)
- **C-04**: Collection types with DU elements

---

## 11. Related Documents

- [F-07-BitsIntrinsics](F-07-BitsIntrinsics.md) - Bit-level operations
- [F-08-OptionType](F-08-OptionType.md) - Homogeneous DU specialization
- [F-09-ResultType](F-09-ResultType.md) - Heterogeneous DU extension
- [Discriminated_Union_Architecture.md](../Discriminated_Union_Architecture.md) - DU design
