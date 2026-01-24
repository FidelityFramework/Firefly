# C-05: Lazy Values (Thunks)

> **Sample**: `14_Lazy` | **Status**: Planned | **Depends On**: C-01 (Closures)

**Foundation of the Lazy Stack**: This PRD establishes deferred computation with memoization. `Lazy<'T>` is a simpler primitive than `Seq<'T>` - it computes once and caches. Sequences (C-06) build on this foundation.

## 1. Executive Summary

Lazy values defer computation until explicitly forced. Unlike sequences (which may yield multiple values), a lazy value produces exactly one value that is then cached. This enables efficient memoization and avoidance of unnecessary computation.

**Key Insight**: A lazy value is an **extended flat closure** - the thunk's captured variables are inlined directly into the lazy struct, following MLKit-style flat closure principles (see "Gaining Closure" blog post). There is no `env_ptr`, no null pointers, no indirection.

### 1.1 Flat Closure Alignment

The C-01 closure architecture established that closures are self-contained:

```
Closure = {code_ptr, capture₀, capture₁, ...}
```

Lazy extends this pattern by prepending memoization state:

```
Lazy<T> = {computed: i1, value: T, code_ptr: ptr, capture₀, capture₁, ...}
```

This means **each `lazy { ... }` expression produces a concrete struct type** based on its capture set. The "type" `Lazy<int>` is actually a family of types parameterized by captures - the same pattern as closures themselves.

## 2. Language Feature Specification

### 2.1 Lazy Creation

```fsharp
let expensive = lazy {
    Console.writeln "Computing..."
    42
}
```

The computation is deferred - "Computing..." is NOT printed yet.

### 2.2 Lazy Force

```fsharp
let v1 = Lazy.force expensive  // Prints "Computing...", returns 42
let v2 = Lazy.force expensive  // Returns 42 immediately (cached)
```

### 2.3 Lazy.value (Shorthand)

```fsharp
let v = expensive.Value  // Same as Lazy.force
```

## 3. Architectural Principles

### 3.1 No Nulls, No `env_ptr`

Following the flat closure model, the lazy struct contains:

| Field | Type | Purpose |
|-------|------|---------|
| `computed` | `i1` | Has thunk been evaluated? |
| `value` | `T` | Cached result (valid when computed=true) |
| `code_ptr` | `ptr` | Thunk function pointer |
| `capture₀...captureₙ` | varies | Inlined captured variables |

**There is no `env_ptr` field.** Captured variables are stored directly in the struct.

### 3.2 Capture Semantics (from C-01)

| Variable Kind | Capture Mode | Storage in Lazy |
|---------------|--------------|-----------------|
| Immutable | ByValue | Copy of value |
| Mutable | ByRef | Pointer to storage location |

For mutable captures that escape (e.g., lazy value returned from function), the storage is hoisted to arena allocation per C-01 patterns.

### 3.3 Struct Layout Examples

**No captures** (`lazy 42`):
```
{computed: i1, value: i32, code_ptr: ptr}
```

**With immutable captures** (`lazy (x * y)` where x, y are `int`):
```
{computed: i1, value: i32, code_ptr: ptr, x: i32, y: i32}
```

**With mutable capture** (`lazy (count <- count + 1; count)` where count is mutable):
```
{computed: i1, value: i32, code_ptr: ptr, count_ptr: ptr}
```
Note: `count_ptr` points to the mutable's storage location (stack or arena).

## 4. FNCS Layer Implementation

### 4.1 NativeType Extension

```fsharp
// In NativeTypes.fs
type NativeType =
    // ... existing types ...
    | TLazy of elementType: NativeType
```

**Note**: `TLazy` represents the semantic type. The concrete MLIR struct layout varies by capture set - this is resolved during Alex lowering, not in FNCS.

### 4.2 SemanticKind.LazyExpr

```fsharp
type SemanticKind =
    | LazyExpr of body: NodeId * captures: CaptureInfo list
```

The `captures` list uses the same `CaptureInfo` type as Lambda (C-01):
- Name, Type, IsMutable, SourceNodeId
- FNCS computes captures during type checking (scope is known)

### 4.3 SemanticKind.LazyForce

```fsharp
type SemanticKind =
    | LazyForce of lazyValue: NodeId
```

Force is a semantic operation, not just a function call - it involves checking the computed flag and potentially invoking the thunk.

### 4.4 Lazy Intrinsics

```fsharp
// In CheckExpressions.fs
| "Lazy.force" ->
    // Lazy<'a> -> 'a
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TLazy(aVar), aVar)

| "Lazy.isValueCreated" ->
    // Lazy<'a> -> bool
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TLazy(aVar), env.Globals.BoolType)
```

### 4.5 Checking `lazy { }` Expressions

```fsharp
/// Check a lazy expression
let checkLazyExpr
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (body: SynExpr)
    (range: range)
    : SemanticNode =

    // 1. Check body expression
    let bodyNode = checkExpr env builder body

    // 2. Collect captures (reuse closure capture analysis from C-01)
    let captures = collectCaptures env builder bodyNode.Id

    // 3. Create LazyExpr node
    builder.Create(
        SemanticKind.LazyExpr(bodyNode.Id, captures),
        NativeType.TLazy(bodyNode.Type),
        range,
        children = [bodyNode.Id])
```

**Key Point**: Capture analysis is reused from C-01. FNCS already knows how to identify captured variables during lambda checking - the same logic applies to lazy.

### 4.6 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `TLazy` type constructor |
| `SemanticGraph.fs` | MODIFY | Add `LazyExpr`, `LazyForce` SemanticKinds |
| `CheckExpressions.fs` | MODIFY | Add `Lazy.force`, `Lazy.isValueCreated` intrinsics |
| `Expressions/Coordinator.fs` | MODIFY | Handle `lazy { }` expressions |

## 5. Alex Layer Implementation

### 5.1 LazyLayout Coeffect

Following the `ClosureLayout` pattern from C-01, compute lazy struct layouts as coeffects:

**File**: `src/Alex/Preprocessing/LazyLayout.fs`

```fsharp
/// Layout information for a lazy expression
type LazyLayout = {
    /// NodeId of the LazyExpr
    LazyId: NodeId
    /// Element type (T in Lazy<T>)
    ElementType: MLIRType
    /// Capture layouts (reuse from closure)
    Captures: CaptureLayout list
    /// Total struct size
    StructType: MLIRType
    /// Offset of code_ptr field
    CodePtrOffset: int
    /// SSAs for lazy struct operations
    LazySSA: SSA
    ComputedFlagSSA: SSA
    ValueSSA: SSA
}

/// Compute lazy layouts for all LazyExpr nodes
let run (graph: SemanticGraph) (closureLayouts: ClosureLayoutCoeffect) : Map<NodeId, LazyLayout> =
    // For each LazyExpr, build layout based on captures
    // Reuse capture layout computation from C-01
```

### 5.2 Lazy Struct Type Generation

```fsharp
/// Generate MLIR struct type for a lazy expression
let lazyStructType (elementType: MLIRType) (captureTypes: MLIRType list) : MLIRType =
    // {computed: i1, value: T, code_ptr: ptr, cap₀, cap₁, ...}
    TStruct ([TInt I1; elementType; TPtr] @ captureTypes)
```

### 5.3 LazyWitness - Creation

```fsharp
/// Emit MLIR for lazy expression creation
let witnessLazyCreate
    (z: PSGZipper)
    (layout: LazyLayout)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let structType = layout.StructType
    let ssas = requireNodeSSAs layout.LazyId z

    // Pre-assigned SSAs
    let falseSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withComputedSSA = ssas.[2]
    let withCodePtrSSA = ssas.[3]
    // SSAs for captures: ssas.[4..]

    let ops = [
        // Create false constant for computed flag
        MLIROp.ArithOp (ArithOp.ConstI (falseSSA, 0L, MLIRTypes.i1))

        // Create undef lazy struct
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, structType))

        // Insert computed=false at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (withComputedSSA, undefSSA, falseSSA, [0], structType))

        // Get thunk function address and insert at index 2
        MLIROp.LLVMOp (LLVMOp.AddressOf (layout.CodePtrSSA, GFunc layout.ThunkFuncName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withComputedSSA, layout.CodePtrSSA, [2], structType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps =
        captureVals
        |> List.indexed
        |> List.collect (fun (i, capVal) ->
            let prevSSA = if i = 0 then withCodePtrSSA else ssas.[3 + i]
            let nextSSA = ssas.[4 + i]
            [MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [3 + i], structType))])

    let resultSSA = if captureVals.IsEmpty then withCodePtrSSA else ssas.[3 + captureVals.Length]

    (ops @ captureOps, TRValue { SSA = resultSSA; Type = structType })
```

### 5.4 LazyWitness - Force

```fsharp
/// Emit MLIR for Lazy.force
let witnessLazyForce
    (z: PSGZipper)
    (layout: LazyLayout)
    (lazyVal: Val)
    : (MLIROp list * TransferResult) =

    let structType = layout.StructType
    let elemType = layout.ElementType
    let ssas = requireNodeSSAs layout.LazyId z

    // SSA assignments for force operation
    let computedSSA = ssas.[0]      // Extracted computed flag
    let cachedValueSSA = ssas.[1]   // Extracted cached value (if computed)
    let codePtrSSA = ssas.[2]       // Extracted code pointer
    // ssas.[3..] for extracted captures
    let computedValueSSA = ssas.[3 + layout.Captures.Length]
    let resultSSA = ssas.[4 + layout.Captures.Length]

    // Extract computed flag
    let checkOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (computedSSA, lazyVal.SSA, [0], structType))
    ]

    // SCF.if for branching
    // If computed: extract and return value from field 1
    // If not computed: extract code_ptr and captures, call thunk, cache result

    // The thunk signature depends on captures:
    // No captures: () -> T
    // With captures: (cap₀, cap₁, ...) -> T

    let extractCaptureOps =
        layout.Captures
        |> List.mapi (fun i _ ->
            let capSSA = ssas.[3 + i]
            MLIROp.LLVMOp (LLVMOp.ExtractValue (capSSA, lazyVal.SSA, [3 + i], structType)))

    let captureArgs =
        layout.Captures
        |> List.mapi (fun i cap -> { SSA = ssas.[3 + i]; Type = cap.Type })

    // Build force logic using SCF.if
    // ... (detailed branching logic)

    (checkOps @ extractCaptureOps @ branchOps, TRValue { SSA = resultSSA; Type = elemType })
```

### 5.5 Thunk Function Generation

Each `lazy { ... }` generates a thunk function that takes captured values as parameters:

```fsharp
/// Generate thunk function for a lazy expression
let emitThunkFunction
    (lazyId: NodeId)
    (layout: LazyLayout)
    (bodyEmitter: unit -> MLIRBuilder)
    : MLIROp list =

    // Thunk signature: (cap₀: T₀, cap₁: T₁, ...) -> T
    // No captures: () -> T
    let paramTypes = layout.Captures |> List.map (fun c -> c.Type)
    let returnType = layout.ElementType

    // Emit function with body
    // ...
```

**Key insight**: The thunk function takes captures as **parameters**, not through an env pointer. When forced, the captured values are extracted from the lazy struct and passed to the thunk.

### 5.6 SSA Cost Computation

```fsharp
/// SSA cost for LazyExpr with N captures
let lazyExprSSACost (numCaptures: int) : int =
    // 1: false constant
    // 1: undef struct
    // 1: insert computed flag
    // 1: addressof code_ptr
    // 1: insert code_ptr
    // N: insert each capture
    5 + numCaptures

/// SSA cost for LazyForce with N captures
let lazyForceSSACost (numCaptures: int) : int =
    // 1: extract computed flag
    // 1: extract cached value
    // 1: extract code_ptr
    // N: extract each capture
    // 1: computed value from thunk call
    // 1: result (phi or direct)
    5 + numCaptures
```

### 5.7 Files to Create/Modify (Alex)

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/LazyLayout.fs` | CREATE | Compute lazy struct layouts |
| `Alex/Witnesses/LazyWitness.fs` | CREATE | Emit lazy creation and force MLIR |
| `Alex/Preprocessing/SSAAssignment.fs` | MODIFY | Add LazyExpr, LazyForce SSA costs |
| `Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle LazyExpr, LazyForce |
| `Alex/CodeGeneration/TypeMapping.fs` | MODIFY | Map TLazy to concrete struct types |

## 6. MLIR Output Specification

### 6.1 No-Capture Example: `lazy 42`

```mlir
// Lazy struct type (no captures)
!lazy_int_0 = !llvm.struct<(i1, i32, ptr)>

// Thunk function: () -> i32
llvm.func @lazy_42_thunk() -> i32 {
    %c42 = arith.constant 42 : i32
    llvm.return %c42 : i32
}

// Creation
%false = arith.constant false
%undef = llvm.mlir.undef : !lazy_int_0
%with_computed = llvm.insertvalue %false, %undef[0] : !lazy_int_0
%code_ptr = llvm.mlir.addressof @lazy_42_thunk : !llvm.ptr
%lazy_val = llvm.insertvalue %code_ptr, %with_computed[2] : !lazy_int_0
```

### 6.2 With-Capture Example: `lazy (x * y)`

```mlir
// Lazy struct type (captures x: i32, y: i32)
!lazy_int_2 = !llvm.struct<(i1, i32, ptr, i32, i32)>

// Thunk function: (i32, i32) -> i32
llvm.func @lazy_xy_thunk(%x: i32, %y: i32) -> i32 {
    %result = arith.muli %x, %y : i32
    llvm.return %result : i32
}

// Creation (assuming %x_val and %y_val are the captured values)
%false = arith.constant false
%undef = llvm.mlir.undef : !lazy_int_2
%s0 = llvm.insertvalue %false, %undef[0] : !lazy_int_2
%code_ptr = llvm.mlir.addressof @lazy_xy_thunk : !llvm.ptr
%s1 = llvm.insertvalue %code_ptr, %s0[2] : !lazy_int_2
%s2 = llvm.insertvalue %x_val, %s1[3] : !lazy_int_2
%lazy_val = llvm.insertvalue %y_val, %s2[4] : !lazy_int_2
```

### 6.3 Force Example

```mlir
// Force a lazy value
llvm.func @force_lazy(%lazy: !lazy_int_2) -> i32 {
    // Extract computed flag
    %computed = llvm.extractvalue %lazy[0] : !lazy_int_2

    // Branch based on computed
    %result = scf.if %computed -> i32 {
        // Already computed - return cached value
        %cached = llvm.extractvalue %lazy[1] : !lazy_int_2
        scf.yield %cached : i32
    } else {
        // Not computed - extract code_ptr and captures, call thunk
        %code_ptr = llvm.extractvalue %lazy[2] : !lazy_int_2
        %cap_x = llvm.extractvalue %lazy[3] : !lazy_int_2
        %cap_y = llvm.extractvalue %lazy[4] : !lazy_int_2
        %computed_val = llvm.call %code_ptr(%cap_x, %cap_y) : (i32, i32) -> i32
        // Note: Caching requires alloca; see Section 7
        scf.yield %computed_val : i32
    }

    llvm.return %result : i32
}
```

## 7. Memoization and Mutability

### 7.1 The Caching Challenge

True memoization requires mutating the lazy struct to:
1. Set `computed = true`
2. Store the computed value

With by-value lazy structs, this requires the lazy value to be stored in mutable memory (stack alloca or arena).

### 7.2 Implementation Options

**By-pointer memoization**
- Lazy values are always `ptr` to stack/arena allocated struct
- Force mutates through the pointer
- Natural memoization

**Functional update (copy-on-force)**
- Force returns `(value, updated_lazy_struct)`
- Caller decides whether to use updated struct
- Pure but awkward API

**Deferred memoization (pure thunk)**
- Initial implementation: always recompute (no caching)
- Add memoization when arena PRDs (20-22) are complete
- Simpler starting point

**Decision**: Start with **deferred memoization** for C-05. The semantics are correct for pure thunks (same result each time). True memoization with caching will be added after arena support (A-04 to A-06) provides the memory management foundation.

### 7.3 Pure Thunk Semantics (Initial Implementation)

For this PRD, `Lazy.force` always evaluates the thunk:

```mlir
// Simplified force (no caching)
llvm.func @force_lazy_pure(%lazy: !lazy_int_2) -> i32 {
    %code_ptr = llvm.extractvalue %lazy[2] : !lazy_int_2
    %cap_x = llvm.extractvalue %lazy[3] : !lazy_int_2
    %cap_y = llvm.extractvalue %lazy[4] : !lazy_int_2
    %result = llvm.call %code_ptr(%cap_x, %cap_y) : (i32, i32) -> i32
    llvm.return %result : i32
}
```

This is semantically correct for pure computations. The memoization optimization comes later.

## 8. Validation

### 8.1 Sample Code

```fsharp
module LazyValuesSample

let expensive = lazy {
    Console.writeln "Computing expensive value..."
    42
}

let lazyAdd a b = lazy {
    Console.writeln "Adding..."
    a + b
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Lazy Values Test ==="

    Console.writeln "--- First Force ---"
    let v1 = Lazy.force expensive
    Console.write "Result: "
    Console.writeln (Format.int v1)

    Console.writeln "--- Second Force ---"
    let v2 = Lazy.force expensive
    Console.write "Result: "
    Console.writeln (Format.int v2)

    Console.writeln "--- Lazy with captures ---"
    let sum = lazyAdd 10 20
    Console.write "Sum: "
    Console.writeln (Format.int (Lazy.force sum))

    0
```

### 8.2 Expected Output (Initial - No Memoization)

```
=== Lazy Values Test ===
--- First Force ---
Computing expensive value...
Result: 42
--- Second Force ---
Computing expensive value...
Result: 42
--- Lazy with captures ---
Adding...
Sum: 30
```

Note: "Computing expensive value..." appears TWICE because initial implementation doesn't cache. This is correct behavior for pure thunks and will be optimized when arena support enables memoization.

### 8.3 Expected Output (With Memoization - Future)

```
=== Lazy Values Test ===
--- First Force ---
Computing expensive value...
Result: 42
--- Second Force (cached) ---
Result: 42
--- Lazy with captures ---
Adding...
Sum: 30
```

## 9. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add `TLazy` to NativeTypes
- [ ] Add `LazyExpr`, `LazyForce` to SemanticKind
- [ ] Implement `lazy { }` checking with capture analysis (reuse C-01 logic)
- [ ] Add `Lazy.force` intrinsic
- [ ] FNCS builds successfully

### Phase 2: Alex Implementation
- [ ] Create `LazyLayout.fs` coeffect computation
- [ ] Create `LazyWitness.fs` with flat closure model
- [ ] Update SSAAssignment for LazyExpr, LazyForce
- [ ] Update TypeMapping for TLazy → concrete struct
- [ ] Handle LazyExpr, LazyForce in FNCSTransfer
- [ ] Generate thunk functions with capture parameters
- [ ] Firefly builds successfully

### Phase 3: Validation
- [ ] Sample 14 compiles without errors
- [ ] Binary executes correctly (thunks evaluate)
- [ ] Captures work (lazyAdd captures a, b)
- [ ] Samples 01-13 still pass (regression)

### Phase 4: Memoization (Future - requires A-04 to A-06)
- [ ] Arena allocation for lazy struct
- [ ] Force mutates struct to cache result
- [ ] Second force returns cached value

## 10. Related PRDs

- **C-01**: Closures - Lazy uses same flat closure model
- **C-03**: Recursion - Pre-creation pattern for NodeIds
- **C-06**: SimpleSeq - Sequences build on lazy foundation
- **C-07**: SeqOperations - Higher-order sequence functions
- **A-04 to A-06**: Regions/Arena - Enable true memoization

## 11. Architectural Alignment

This PRD aligns with the flat closure architecture documented in:
- "Gaining Closure" blog post (SpeakEZ)
- `closure_architecture_corrected` Serena memory
- `true_flat_closures_implementation` Serena memory

**Key principles maintained:**
1. **No nulls** - Every field initialized
2. **No env_ptr** - Captures inlined directly
3. **Self-contained structs** - No pointer chains
4. **Coeffect-based layout** - SSA computed before witnessing
5. **Capture reuse** - Same analysis as C-01 closures

## 12. Implementation Lessons (January 2026)

### 12.1 The "Compose from Standing Art" Principle

> **New features MUST compose from recently established patterns, not invent parallel mechanisms.**

C-05 implementation initially went wrong by creating a `{code_ptr, env_ptr}` closure model with null env_ptr for no-capture cases. This completely ignored that **flat closures had just been established** in C-01.

**The correct approach:**
1. Identify what existing patterns the feature needs (closures, capture analysis)
2. Check what was RECENTLY established (C-01 flat closures)
3. EXTEND the existing pattern, don't reinvent

```
WRONG: Lazy-specific closure model with nulls
RIGHT: Lazy = C-01 Flat Closure + memoization state
       {computed: i1, value: T, code_ptr: ptr, cap₀, cap₁, ...}
```

This principle applies to ALL future PRDs. See Serena memory: `compose_from_standing_art_principle`

### 12.2 Thunk Calling Convention: Struct Pointer Passing

A key architectural decision was choosing between two calling conventions:

| Convention | Thunk Signature | Force Complexity |
|------------|-----------------|------------------|
| **Parameter Passing** | `(cap₀, cap₁, ...) -> T` | Must extract and pass captures |
| **Struct Pointer Passing** | `(ptr) -> T` | Uniform - just passes pointer |

**Decision: Struct Pointer Passing** - Thunk receives pointer to lazy struct, extracts its own captures.

**Why struct pointer passing wins:**
- Force is UNIFORM regardless of capture count
- No def-use tracking needed at force site
- Clean separation: force handles invocation, thunk handles extraction
- SSA cost for force is FIXED (4 ops) not variable

See Serena memory: `lazy_thunk_calling_convention`

### 12.3 Capture Analysis Reuse

FNCS already had capture analysis for Lambda (C-01). The correct approach was to **reuse it**:

```fsharp
// In Applications.fs - made public for reuse
let computeCaptures (builder: NodeBuilder) (env: TypeEnv) (bodyNodeId: NodeId) (excludeNames: Set<string>) : CaptureInfo list

// checkLambda uses it
let captures = computeCaptures builder env bodyNode.Id paramNames

// checkLazy reuses the SAME function
let captures = computeCaptures builder env innerNode.Id (Set.singleton "_unit")
```

**Lesson:** Before writing new code, ask "Does this already exist for a similar feature?"

### 12.4 SSA Cost Determinism

SSA assignment must be 100% deterministic - no synthetic SSAs, no runtime decisions.

**LazyExpr SSA cost:** `5 + numCaptures` (variable, but deterministic from PSG)
**LazyForce SSA cost:** `4` (fixed - this is the struct pointer passing benefit)

The coeffect pattern means witnesses OBSERVE pre-computed SSAs, they don't INVENT them.

### 12.5 Process Lessons for Future PRDs

Before implementing ANY new feature:

1. **Review the last 2-3 PRDs** for patterns that might apply
2. **Read Serena memories** for architectural decisions
3. **Identify composition points** - what existing code/patterns to extend?
4. **Document the composition** in the PRD explicitly
5. **Course correct early** if implementation diverges from patterns

> "The standing art composes up. Use it."

### 12.6 Files Modified (Reference)

**FNCS:**
- `Applications.fs` - Made `collectVarRefs` public, added `computeCaptures`
- `Coordinator.fs` - `checkLazy` uses `computeCaptures`

**Alex:**
- `LazyWitness.fs` - Struct pointer passing calling convention
- `SSAAssignment.fs` - SSA costs (LazyExpr: 5+N, LazyForce: 4)
- `FNCSTransfer.fs` - Simplified LazyForce (uniform, no capture tracking)
