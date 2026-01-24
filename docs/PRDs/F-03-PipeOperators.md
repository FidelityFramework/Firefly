# F-03: Pipe Operators

> **Sample**: `03_HelloWorldHalfCurried` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces the F# pipe operator (`|>`) and demonstrates that F# syntactic sugar is resolved in PSG transformations, not during code generation. The `ReducePipeOperators` nanopass normalizes data flow syntax into direct function applications.

**Key Achievement**: Established the nanopass pattern for syntactic sugar reduction, ensuring Alex sees normalized structure.

---

## 2. Surface Feature

```fsharp
module HelloWorld

let greet name =
    $"Hello, {name}!" |> Console.writeln

let arena = Arena.create<'a> 1024
Console.readlnFrom &arena |> greet
```

**Data Flow**:
```
readlnFrom → greet → writeln
```

---

## 3. Infrastructure Contributions

### 3.1 Pipe Operator Semantics

The pipe operator `|>` is syntactic sugar for function application:

```fsharp
x |> f  ≡  f x
x |> f |> g  ≡  g (f x)
```

This is NOT a runtime operation - it's compile-time restructuring.

### 3.2 ReducePipeOperators Nanopass

**File**: `src/Core/PSG/Nanopass/ReducePipeOperators.fs`

The nanopass transforms pipe expressions during PSG enrichment:

```
Before:
  Application
  ├── Operator: |>
  ├── Left: x
  └── Right: f

After:
  Application
  ├── Function: f
  └── Argument: x
```

**Algorithm**:
1. Pattern match on `Application` nodes with `|>` operator
2. Extract left operand (value) and right operand (function)
3. Reconstruct as direct application
4. Recurse for chained pipes

### 3.3 Back-Pipe Operator

The back-pipe operator `<|` is also handled:

```fsharp
f <| x  ≡  f x
```

Reduction is symmetric - extract function and argument, reconstruct application.

### 3.4 Partial Application Preparation

When pipes involve partial application:

```fsharp
values |> List.map toString
```

The nanopass preserves partial application structure:

```
Application
├── Function: List.map
├── Argument: toString
└── [partial - awaiting values]
```

This prepares for closure creation in C-01.

---

## 4. PSG Transformation

**Before ReducePipeOperators**:
```
StatementSequence
├── LetBinding: greet
│   └── Lambda: name ->
│       └── Application
│           ├── Operator: |>
│           ├── Left: InterpolatedString
│           └── Right: Console.writeln
└── Application
    ├── Operator: |>
    ├── Left: Application (readlnFrom)
    └── Right: Ident (greet)
```

**After ReducePipeOperators**:
```
StatementSequence
├── LetBinding: greet
│   └── Lambda: name ->
│       └── Application
│           ├── Function: Console.writeln
│           └── Argument: InterpolatedString
└── Application
    ├── Function: greet
    └── Argument: Application (readlnFrom)
```

---

## 5. Nanopass Architecture

This sample establishes the nanopass design pattern:

```
┌─────────────────────────────────────┐
│ PSG (with pipe operators)          │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ ReducePipeOperators Nanopass       │
│ - Pattern match |> and <|          │
│ - Restructure to direct apps       │
│ - Preserve partial application     │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ PSG (normalized function calls)    │
└─────────────────────────────────────┘
```

**Key Properties**:
- Single-purpose: Only handles pipe operators
- Composable: Other nanopasses can run before/after
- Inspectable: `-k` flag shows intermediate PSG

---

## 6. Coeffects

No new coeffects introduced. This sample relies on:

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for let bindings and applications |

---

## 7. MLIR Impact

The normalized PSG produces straightforward MLIR:

```mlir
// greet function
llvm.func @greet(%name: !llvm.ptr, %name_len: i64) {
  // Build interpolated string
  // Call Console.writeln directly (no pipe indirection)
}

// main
llvm.func @main() -> i32 {
  %name = llvm.call @console_readln_from(...)
  llvm.call @greet(%name, %name_len)  // Direct call, no pipe
  llvm.return %zero
}
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/03_HelloWorldHalfCurried
/path/to/Firefly compile HelloWorld.fidproj
echo "Pipes" | ./HelloWorld
# Output: Hello, Pipes!
```

---

## 9. Architectural Lessons

1. **Sugar in PSG, Not Alex**: Syntactic sugar resolved during PSG enrichment
2. **Nanopass Modularity**: Each nanopass has single responsibility
3. **Structure Preservation**: Pipes become direct calls, but partial application preserved
4. **Alex Simplicity**: Code generation sees normalized structure only

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-04**: Currying and lambdas (extends function application patterns)
- **C-02**: Higher-order functions (pipes common in HOF chains)
- **C-06, C-07**: Sequence operations (`seq |> Seq.map |> Seq.filter`)

---

## 11. Related Documents

- [F-02-ArenaAllocation](F-02-ArenaAllocation.md) - Memory patterns used here
- [PSG_Nanopass_Architecture.md](../PSG_Nanopass_Architecture.md) - Nanopass design
- [C-02-HigherOrderFunctions](C-02-HigherOrderFunctions.md) - HOF patterns with pipes
