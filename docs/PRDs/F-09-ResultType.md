# F-09: Result Type

> **Sample**: `09_Result` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample implements `Result<'T, 'E>` as a heterogeneous discriminated union. Unlike Option's homogeneous layout, Result requires arena allocation when `'T` and `'E` have different sizes.

**Key Achievement**: Established `DULayout` coeffect for arena-allocated heterogeneous DUs and demonstrated type-safe error handling.

---

## 2. Surface Feature

```fsharp
module ResultTest

let divide (a: int) (b: int) : Result<int, string> =
    if b = 0 then
        Error "Division by zero"
    else
        Ok (a / b)

let arena = Arena.create<'a> 1024

let result1 = divide 10 2
let result2 = divide 10 0

match result1 with
| Ok value -> Console.writeln $"Success: {value}"
| Error msg -> Console.writeln $"Error: {msg}"
```

---

## 3. Infrastructure Contributions

### 3.1 Heterogeneous DU Detection

Result is heterogeneous when `sizeof('T) ≠ sizeof('E)`:

| 'T | 'E | Homogeneous? |
|----|----|----|
| int | int | Yes (both 8 bytes) |
| int | string | No (int=8, string=16) |
| string | string | Yes (both 16 bytes) |

### 3.2 Arena-Allocated Layout

For heterogeneous Result:

```
Result Pointer (8 bytes on stack)
    │
    ▼
Arena Memory:
┌─────────────────────────────────────┐
│ tag: i8      (0 = Ok, 1 = Error)   │
│ padding: [alignment bytes]         │
│ payload: T or E (sized by case)    │
└─────────────────────────────────────┘
```

**Properties**:
- Stack holds pointer to arena-allocated case struct
- Each case struct sized independently
- Arena affinity tracks which arena owns the Result

### 3.3 DULayout Coeffect

The `DULayout` coeffect pre-computes:
- Whether DU is homogeneous or heterogeneous
- Case struct layouts with offsets
- Arena requirements for construction

```fsharp
type DULayout = {
    IsHomogeneous: bool
    CaseLayouts: Map<string, CaseLayout>
    ArenaRequired: bool
}
```

### 3.4 Result Intrinsics

| Intrinsic | Type | Purpose |
|-----------|------|---------|
| `Result.isOk` | `Result<'T,'E> -> bool` | Check if Ok |
| `Result.isError` | `Result<'T,'E> -> bool` | Check if Error |
| `Result.get` | `Result<'T,'E> -> 'T` | Extract value (unchecked) |
| `Result.getError` | `Result<'T,'E> -> 'E` | Extract error (unchecked) |

### 3.5 ResultRecipes in Baker

```fsharp
Result.map f result
```

Decomposes to:

```fsharp
match result with
| Ok x -> Ok (f x)
| Error e -> Error e
```

---

## 4. PSG Representation

```
ModuleOrNamespace: ResultTest
├── LetBinding: divide
│   └── Lambda: a, b ->
│       └── IfThenElse
│           ├── Condition: b = 0
│           ├── Then: UnionCase Error "Division by zero"
│           └── Else: UnionCase Ok (a / b)
│
├── LetBinding: arena
├── LetBinding: result1 = divide 10 2
├── LetBinding: result2 = divide 10 0
│
└── Match (result1)
    ├── Case: Ok value -> ...
    └── Case: Error msg -> ...
```

---

## 5. Construction with Arena

For `Error "Division by zero"`:

```mlir
// 1. Compute case struct size
%error_size = llvm.mlir.constant(24 : i64)  // tag + padding + string

// 2. Bump allocate from arena
%ptr = llvm.call @arena_alloc(%arena, %error_size)

// 3. Store tag
%tag_ptr = llvm.getelementptr %ptr[0]
%error_tag = llvm.mlir.constant(1 : i8)
llvm.store %error_tag, %tag_ptr

// 4. Store payload
%payload_ptr = llvm.getelementptr %ptr[8]  // After alignment
llvm.store %msg_ptr, %payload_ptr
llvm.store %msg_len, %payload_ptr_len

// 5. Result value is the arena pointer
// %ptr is the Result<int, string>
```

---

## 6. Pattern Match on Heterogeneous DU

```mlir
// Load tag from arena pointer
%tag_ptr = llvm.getelementptr %result[0]
%tag = llvm.load %tag_ptr : i8
%is_ok = llvm.icmp "eq" %tag, 0

llvm.cond_br %is_ok, ^ok_case, ^error_case

^ok_case:
  // Load int payload (starts at offset 8)
  %value_ptr = llvm.getelementptr %result[8]
  %value = llvm.load %value_ptr : i64
  // use value
  llvm.br ^merge

^error_case:
  // Load string payload (starts at offset 8)
  %msg_ptr_ptr = llvm.getelementptr %result[8]
  %msg_ptr = llvm.load %msg_ptr_ptr : !llvm.ptr
  %msg_len_ptr = llvm.getelementptr %result[16]
  %msg_len = llvm.load %msg_len_ptr : i64
  // use msg
  llvm.br ^merge
```

---

## 7. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all nodes |
| PatternBindings | SSA for pattern extractions |
| DULayout | Pre-computed DU representation info |
| ArenaAffinity | Track arena ownership |

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/09_Result
/path/to/Firefly compile ResultTest.fidproj
./ResultTest
# Output:
# Success: 5
# Error: Division by zero
```

---

## 9. Homogeneous vs Heterogeneous Comparison

| Aspect | Option (F-08) | Result (F-09) |
|--------|---------------|---------------|
| Layout | Always inline | Depends on types |
| Stack size | Fixed (1 + sizeof T) | 8 bytes (pointer) |
| Arena needed | No | When heterogeneous |
| DULayout coeffect | Simple | Full computation |

---

## 10. Architectural Lessons

1. **Type-Driven Layout**: DU representation depends on payload types
2. **DULayout Coeffect**: Pre-compute representation before emission
3. **Arena Integration**: Heterogeneous DUs need explicit memory
4. **Uniform Extraction**: Pattern match works regardless of layout

---

## 11. Downstream Dependencies

This sample's infrastructure enables:
- **F-06**: Parsing with Result (fallible operations)
- **C-04**: Collection operations with Result
- **A-01**: Async with error handling

---

## 12. Related Documents

- [F-08-OptionType](F-08-OptionType.md) - Homogeneous DU pattern
- [F-02-ArenaAllocation](F-02-ArenaAllocation.md) - Arena infrastructure
- [Discriminated_Union_Architecture.md](../Discriminated_Union_Architecture.md) - DU design
