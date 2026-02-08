# F-05: Discriminated Unions

> **Sample**: `05_AddNumbers` | **Status**: Implemented | **Category**: Foundation

## 1. Executive Summary

This sample introduces discriminated unions (DUs), the fundamental sum type of F#. A homogeneous DU (`Number = IntVal | FloatVal`) demonstrates tag-based dispatch, pattern matching, payload extraction, and numeric formatting.

**Key Achievement**: Established DU representation with byte-level `memref<Nxi8>` layout, `memref.view` for typed payload access, expression-valued `scf.if` for match results, and Baker's `MatchRecipes` for pattern match lowering.

---

## 2. Surface Feature

```fsharp
module AddNumbers

type Number =
    | IntVal of int
    | FloatVal of float

let formatNumber (n: Number) : string =
    match n with
    | IntVal x -> Format.int x
    | FloatVal x -> Format.float x

let runDemo () : int =
    Console.writeln "=== DU Pattern Match Test ==="
    let a = IntVal 42
    Console.write "IntVal 42 -> "
    Console.writeln (formatNumber a)
    let b = FloatVal 3.14
    Console.write "FloatVal 3.14 -> "
    Console.writeln (formatNumber b)
    Console.writeln "Done!"
    0

[<EntryPoint>]
let main argv = runDemo ()
```

---

## 3. Infrastructure Contributions

### 3.1 DU Representation

DUs use byte-level `memref` allocation — no LLVM struct types:

```
Number DU Layout: memref<9xi8>
┌──────────┬────────────────────────┐
│ tag: i8  │ payload: i64 or f64    │
│ offset 0 │ offset 1 (8 bytes)     │
└──────────┴────────────────────────┘
```

**Size**: 1 (tag) + 8 (payload) = 9 bytes. No padding — byte-level access via `memref.view`.

### 3.2 Homogeneous vs Heterogeneous

| Type | Payload Types | Layout | Allocation |
|------|---------------|--------|------------|
| Homogeneous | All same size | `memref<Nxi8>` inline | Stack |
| Heterogeneous | Different sizes | Arena-allocated | Arena |

`Number` is homogeneous because both `int` (i64) and `float` (f64) are 8 bytes.

### 3.3 Typed Access via memref.view

Payloads stored as raw bytes, accessed via typed views:

```mlir
// Storing IntVal 42: view bytes as i64, store directly
%c1 = arith.constant 1 : index
%payload_view = memref.view %du[%c1][] : memref<9xi8> to memref<1xi64>
memref.store %value, %payload_view[%c0] : memref<1xi64>

// Storing FloatVal 3.14: view bytes as f64, store directly
%c1 = arith.constant 1 : index
%payload_view = memref.view %du[%c1][] : memref<9xi8> to memref<1xf64>
memref.store %value, %payload_view[%c0] : memref<1xf64>
```

No bit-level coercion needed — `memref.view` reinterprets the underlying bytes as the target type.

**Two access operations**:
- **`memref.reinterpret_cast`**: Same element type (i8→i8). Used for tag access.
- **`memref.view`**: Different element type (i8→i64, i8→f64). Used for payload access.

### 3.4 Pattern Match Lowering

`MatchRecipes` in Baker lowers pattern matches to decision trees:

```
match n with
| IntVal x -> ...
| FloatVal x -> ...
```

Becomes:

```
IfThenElse
├── Condition: n.tag == 0
├── Then: extract x from n.payload (as i64), ...
└── Else: extract x from n.payload (as f64), ...
```

### 3.5 Tag Extraction

Tags extracted via `memref.reinterpret_cast` (same element type) + `memref.load`:

```mlir
%tag_view = memref.reinterpret_cast %du to offset: [0], sizes: [1], strides: [1]
    : memref<9xi8> to memref<1xi8>
%tag = memref.load %tag_view[%c0] : memref<1xi8>
%is_intval = arith.cmpi eq, %tag, %c0_i8 : i8
```

### 3.6 Payload Extraction

Payloads extracted via `memref.view` (different element type) + `memref.load`:

```mlir
// For IntVal (i64 payload)
%c1 = arith.constant 1 : index
%view_i64 = memref.view %du[%c1][] : memref<9xi8> to memref<1xi64>
%int_val = memref.load %view_i64[%c0] : memref<1xi64>

// For FloatVal (f64 payload)
%c1 = arith.constant 1 : index
%view_f64 = memref.view %du[%c1][] : memref<9xi8> to memref<1xf64>
%float_val = memref.load %view_f64[%c0] : memref<1xf64>
```

### 3.7 Expression-Valued Pattern Match

Pattern matches that return values use MLIR's expression-valued `scf.if`:

```mlir
%result = scf.if %is_intval -> (memref<?xi8>) {
    // IntVal branch: extract i64, call Format.int
    %int_str = func.call @Format.int(%int_val) : (i64) -> memref<?xi8>
    scf.yield %int_str : memref<?xi8>
} else {
    // FloatVal branch: extract f64, call Format.float
    %float_str = func.call @Format.float(%float_val) : (f64) -> memref<?xi8>
    scf.yield %float_str : memref<?xi8>
}
func.return %result : memref<?xi8>
```

This is distinct from void `scf.if` (used for side-effect-only branches). The ControlFlowWitness
checks `node.Type` — if non-unit, it emits the expression-valued form with `scf.yield` in each branch.

### 3.8 Format as Platform Library

`Format.int` and `Format.float` are F# code in `Fidelity.Platform` — same compilation pattern as
`Console.write`. They compile through the full pipeline (FNCS → PSG → Baker → Alex) naturally.
Alex witnesses them like any other function. No push-model construction.

This supersedes the retired `FormatOps.fs` (861 lines of imperative MLIR in Alex).

---

## 4. PSG Representation

```
ModuleOrNamespace: AddNumbers
├── TypeDefinition: Number (DU)
│   ├── Case: IntVal of int
│   └── Case: FloatVal of float
│
├── LetBinding: formatNumber
│   └── Lambda: n ->
│       └── Match
│           ├── Targets: [(n)]
│           └── Cases:
│               ├── IntVal x -> Application(Format.int, x)
│               └── FloatVal x -> Application(Format.float, x)
│
├── LetBinding: runDemo
│   └── Lambda: () ->
│       └── Sequential [Console.writeln ..., DUConstruct, formatNumber, ...]
│
└── LetBinding: main
    └── Lambda: argv -> Application(runDemo, ())
```

---

## 5. Baker Saturation

`DUConstruct` lowering transforms union case construction:

```
Before (PSG):
  UnionCase: IntVal
  └── Argument: 42

After (Baker):
  DUConstruct
  ├── Tag: 0 (i8)
  └── Payload: 42 (stored via memref.view into byte buffer)
```

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all nodes including pattern bindings |
| PatternBindings | SSA assignments for `x` in pattern matches |

**PatternBindings** is critical — it ensures extracted values get SSA slots.

**SSA Costs**:
- DUGetTag: 3 SSAs (reinterpret_cast + zero_index + load)
- DUEliminate: 4 SSAs (offset_const + memref.view + zero_index + load)
- DUConstruct: 4 + 3*N SSAs (alloca + per-field: offset + view + store)

---

## 7. MLIR Output Pattern

```mlir
// DU function signature — accepts static-sized byte memref
func.func @AddNumbers.formatNumber(%arg0: memref<9xi8>) -> memref<?xi8> {
    // Tag extraction via reinterpret_cast (same element type)
    %tag_view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1]
        : memref<9xi8> to memref<1xi8>
    %tag = memref.load %tag_view[%c0] : memref<1xi8>
    %is_intval = arith.cmpi eq, %tag, %c0_i8 : i8

    // Expression-valued scf.if — returns the match result
    %result = scf.if %is_intval -> (memref<?xi8>) {
        // Payload extraction via memref.view (different element type)
        %view_i64 = memref.view %arg0[%c1][] : memref<9xi8> to memref<1xi64>
        %int_val = memref.load %view_i64[%c0] : memref<1xi64>
        %str = func.call @Format.int(%int_val) : (i64) -> memref<?xi8>
        scf.yield %str : memref<?xi8>
    } else {
        %view_f64 = memref.view %arg0[%c1][] : memref<9xi8> to memref<1xf64>
        %float_val = memref.load %view_f64[%c0] : memref<1xf64>
        %str = func.call @Format.float(%float_val) : (f64) -> memref<?xi8>
        scf.yield %str : memref<?xi8>
    }
    func.return %result : memref<?xi8>
}

// DU construction — alloca + typed stores
%du = memref.alloca() : memref<9xi8>
%tag_view = memref.reinterpret_cast %du to offset: [0], sizes: [1], strides: [1]
    : memref<9xi8> to memref<1xi8>
memref.store %tag_val, %tag_view[%c0] : memref<1xi8>
%payload_view = memref.view %du[%c1][] : memref<9xi8> to memref<1xi64>
memref.store %payload_val, %payload_view[%c0] : memref<1xi64>
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/05_AddNumbers
/home/hhh/repos/Firefly/src/bin/Debug/net10.0/Firefly compile AddNumbers.fidproj
./AddNumbers
# Output:
# === DU Pattern Match Test ===
# IntVal 42 -> 42
# FloatVal 3.14 -> 3.14
# Done!
```

---

## 9. Architectural Lessons

1. **Byte-Level Representation**: DUs use `memref<Nxi8>` — no LLVM struct types needed. `memref.view` provides typed access to payload bytes.
2. **Two Access Operations**: `memref.reinterpret_cast` for same-element-type (tag), `memref.view` for different-element-type (payload). This distinction is an MLIR semantic constraint.
3. **Expression-Valued scf.if**: Pattern matches that return values need `scf.yield` in each branch. The ControlFlowWitness determines this by checking `node.Type` for non-unit.
4. **`[<RequireQualifiedAccess>]` Gotcha**: `NTUKind` has this attribute. Bare `NTUunit` in a pattern is a variable binding (matches anything!), not a DU case match. Must use `NTUKind.NTUunit`.
5. **Baker Decomposition**: Pattern matches lowered to IfThenElse + DUGetTag + DUEliminate before Alex sees them.
6. **PatternBindings Coeffect**: Extracted values need explicit SSA tracking.
7. **Platform Library Pattern**: Format.int/float are F# in Fidelity.Platform, not imperative MLIR construction in Alex.
8. **Pointer Offset Adjustment (MemRef.add fusion)**: When `NativePtr.add buf offset` flows into `NativeStr.fromPointer`, the witness must fuse them. Baker transforms `NativePtr.add` → `MemRef.add` (marker op). The MemRef.add witness returns only the offset (`TIndex`), discarding the base. Consumers like `NativeStr.fromPointer` detect this pattern, extract both base + offset, then compute the adjusted source pointer: `%adjusted = arith.addi %base_ptr, %offset : index`. Same fusion pattern exists in `MemRef.store`.
9. **Elements Take Explicit Types**: Arithmetic Elements must receive their operand type as an explicit parameter from the Pattern — never pull from `state.Current.Type`. The PSG node type can differ from operand types (e.g., comparison nodes have type `bool`/`i1` while operands are `i64`/`f64`; nativeint arithmetic nodes resolve to `i64` while operands are `index`).
10. **Deterministic Compilation**: The compiler MUST produce byte-identical output for identical input. Two non-determinism sources were eliminated: (a) `Async.Parallel` in Baker's `FanOut.fs` caused race conditions on the global `NodeId.fresh()` counter — replaced with sequential processing; (b) `String.GetHashCode()` is randomized per-process in .NET — replaced with deterministic FNV-1a hash for string global names.

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-08**: Option type (canonical homogeneous DU)
- **F-09**: Result type (heterogeneous DU extension)
- **C-04**: Collection types with DU elements
- **Future**: Tuple pattern matching over multiple DU args (PRD vision: `add` function)

---

## 11. Related Documents

- [F-08-OptionType](F-08-OptionType.md) - Homogeneous DU specialization
- [F-09-ResultType](F-09-ResultType.md) - Heterogeneous DU extension
- Serena memory: `discriminated_union_architecture` - Full DU design
- Serena memory: `composite_type_representation_architecture` - memref.view patterns
- Serena memory: `format_int_implementation_plan_feb2026` - Format platform library approach
