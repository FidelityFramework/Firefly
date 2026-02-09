# C-06: Simple Sequence Expressions

> **Sample**: `15_SimpleSeq` | **Status**: Planned | **Depends On**: C-05 (Lazy), C-01 (Closures)

## 1. Executive Summary

Sequence expressions (`seq { }`) provide lazy, on-demand iteration in F#. Unlike `Lazy<'T>` (single deferred value), `Seq<'T>` produces multiple values through resumable computation.

**Key Insight**: A sequence is an **extended flat closure with state machine fields**. Like Lazy (C-05), captures are inlined directly into the struct. The sequence adds state tracking for resumable computation at yield points.

**Builds on C-05**: Both `Lazy<'T>` and `Seq<'T>` are flat closures with extra state. Lazy has `{computed, value, code_ptr, captures...}`. Seq has `{state, current, code_ptr, captures...}`.

## 2. Language Feature Specification

### 2.1 Basic Sequence Expression

```fsharp
let numbers = seq {
    yield 1
    yield 2
    yield 3
}
```

Produces values 1, 2, 3 on demand.

### 2.2 Sequence with Computation

```fsharp
let squares n = seq {
    let mutable i = 1
    while i <= n do
        yield i * i
        i <- i + 1
}
```

### 2.3 Sequence with Conditional Yields

```fsharp
let evens max = seq {
    let mutable n = 0
    while n <= max do
        if n % 2 = 0 then
            yield n
        n <- n + 1
}
```

### 2.4 Sequence Consumption

```fsharp
for x in numbers do
    Console.writeln (Format.int x)
```

The `for...in` construct drives the state machine.

### 2.5 Semantic Laws

1. **Laziness**: Elements computed on demand, not eagerly
2. **Pull-based**: Consumer controls iteration pace
3. **Restartability**: Iterating a seq twice re-executes from the beginning
4. **Capture semantics**: Variables captured at creation time (flat closure model)

## 3. Architectural Principles

### 3.1 Flat Closure Extension (No `env_ptr`)

Following the flat closure model from C-01 and C-05:

```
Seq<T> = {state: i32, current: T, code_ptr: ptr, capture₀, capture₁, ...}
```

| Field | Type | Purpose |
|-------|------|---------|
| `state` | `i32` | Current state: 0=initial, N=after yield N, -1=done |
| `current` | `T` | Current value (valid after MoveNext returns true) |
| `code_ptr` | `ptr` | MoveNext function pointer |
| `capture₀...captureₙ` | varies | Inlined captured variables |

**There is no `env_ptr` field.** Captures are stored directly in the struct.

### 3.2 Capture Semantics (from C-01)

| Variable Kind | Capture Mode | Storage in Seq |
|---------------|--------------|----------------|
| Immutable | ByValue | Copy of value |
| Mutable | ByRef | Pointer to storage location |

### 3.3 Struct Layout Examples

**No captures** (`seq { yield 1; yield 2; yield 3 }`):
```
{state: i32, current: i32, code_ptr: ptr}
```

**With captures** (`seq { for i in 1..n do yield i * factor }` where n, factor are captured):
```
{state: i32, current: i32, code_ptr: ptr, n: i32, factor: i32}
```

### 3.4 State Machine Model

Each `yield` becomes a state transition point. The MoveNext function:
1. Switches on current state
2. Executes code until next yield (or end)
3. Stores yielded value in `current` field
4. Updates state to next yield point
5. Returns `true` (has value) or `false` (done)

### 3.5 Internal State vs. Captures (Critical Distinction)

Sequence expressions can have TWO kinds of variables that persist across yields:

| Category | Definition | Storage | Lifetime | Example |
|----------|------------|---------|----------|---------|
| **Captures** | Variables from enclosing scope | Inlined in seq struct at creation | Created at seq creation, immutable copies | `factor` in `seq { yield x * factor }` |
| **Internal State** | Mutable variables declared inside seq body | Additional fields in seq struct | Created at first MoveNext, mutated between yields | `i` in `seq { let mutable i = 1; while ... }` |

**Captures** (from enclosing scope):
```fsharp
let factor = 10
let scaled = seq {           // 'factor' is CAPTURED here
    yield 1 * factor
    yield 2 * factor
}
// factor is copied into the seq struct at creation
```

**Internal State** (mutable variables inside body):
```fsharp
let countUp max = seq {
    let mutable i = 1        // INTERNAL STATE - lives in seq struct
    while i <= max do
        yield i
        i <- i + 1           // Mutated between yields
}
// 'i' is a field in the seq struct, initialized on first MoveNext
```

**Combined Example**:
```fsharp
let multiplesOf factor count = seq {
    let mutable i = 1        // Internal state (mutable, in struct)
    while i <= count do      // 'count' is captured (immutable copy)
        yield i * factor     // 'factor' is captured (immutable copy)
        i <- i + 1
}
```

Resulting struct layout:
```
{state: i32, current: i32, code_ptr: ptr, factor: i32, count: i32, i: i32}
|<-- standard seq fields -->|<-- captures -->|<-- internal state -->|
```

**Why This Matters for Alex**:
- Captures are initialized at seq creation (in `witnessSeqCreate`)
- Internal state is initialized in the MoveNext function (state 0 block)
- Both are accessed via fixed struct offsets
- Internal state requires read-modify-write in MoveNext; captures are read-only

## 4. FNCS Layer Implementation

### 4.1 NTUKind and NativeType Extensions

```fsharp
// In NativeTypes.fs
type NTUKind =
    // ... existing kinds ...
    | NTUseq   // Sequence/generator

type NativeType =
    // ... existing types ...
    | TSeq of elementType: NativeType
```

### 4.2 SemanticKind Extensions

```fsharp
type SemanticKind =
    // ... existing kinds ...

    /// Sequence expression - a resumable flat closure producing values on demand
    /// body: The sequence body containing yields
    /// captures: Variables captured from enclosing scope (flat closure model)
    | SeqExpr of body: NodeId * captures: CaptureInfo list

    /// Yield point within a sequence expression
    /// value: The expression to yield
    | Yield of value: NodeId

    /// For-each loop consuming an enumerable
    /// loopVar: Name of the loop variable
    /// varType: Type of the loop variable
    /// source: The sequence/enumerable to iterate
    /// body: Loop body executed for each element
    | ForEach of loopVar: string * varType: NativeType * source: NodeId * body: NodeId
```

**Note**: No `stateIndex` in Yield - that's computed by Alex as a coeffect (YieldStateIndices).

### 4.3 TypeEnv Extension

Following C-03's environment enrichment pattern:

```fsharp
type TypeEnv = {
    // ... existing fields ...

    /// Enclosing sequence expression (None at top level)
    /// Used to validate that yields appear inside seq { }
    EnclosingSeqExpr: NodeId option
}
```

### 4.4 Checking `seq { }` Expressions

**LambdaContext Extension**: Following C-01's flat closure model, sequence generators use `LambdaContext.SeqGenerator` to distinguish them from regular closures and lazy thunks:

```fsharp
/// Extended LambdaContext (from C-01, extended in C-05, C-06)
type LambdaContext =
    | RegularClosure       // C-01: {code_ptr, cap₀, cap₁, ...}
    | LazyThunk            // C-05: {computed, value, code_ptr, cap₀, ...}
    | SeqGenerator         // C-06: {state, current, code_ptr, cap₀, ...}

/// The LambdaContext determines the extraction base index for captures:
/// - RegularClosure: base = 1 (after code_ptr)
/// - LazyThunk: base = 3 (after computed, value, code_ptr)
/// - SeqGenerator: base = 3 (after state, current, code_ptr)
```

```fsharp
/// Check a sequence expression
let checkSeqExpr
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (seqBody: SynExpr)
    (range: range)
    : SemanticNode =

    // STEP 1: Pre-create SeqExpr node to get NodeId (C-03 pattern)
    let elementType = freshTypeVar range
    let seqNode = builder.Create(
        SemanticKind.SeqExpr(NodeId.Empty, []),  // Placeholder
        NativeType.TSeq(elementType),
        range,
        children = [])

    // STEP 2: Extend environment with enclosing seq context
    //         AND set LambdaContext for capture extraction
    let seqEnv = { env with
                    EnclosingSeqExpr = Some seqNode.Id
                    LambdaContext = Some LambdaContext.SeqGenerator }

    // STEP 3: Check body - yields will validate against EnclosingSeqExpr
    let bodyNode = checkExpr seqEnv builder seqBody

    // STEP 4: Collect captures (reuse closure logic from C-01)
    let captures = collectCaptures env builder bodyNode.Id

    // STEP 5: Update SeqExpr node with actual body and captures
    builder.SetChildren(seqNode.Id, [bodyNode.Id])
    builder.UpdateKind(seqNode.Id, SemanticKind.SeqExpr(bodyNode.Id, captures))

    seqNode
```

### 4.5 Checking `yield` Expressions

```fsharp
/// Check a yield expression within a sequence
let checkYield
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (valueExpr: SynExpr)
    (range: range)
    : SemanticNode =

    // Validate we're inside a seq { }
    match env.EnclosingSeqExpr with
    | None ->
        failwith "yield can only appear inside a sequence expression"
    | Some seqId ->
        let valueNode = checkExpr env builder valueExpr

        builder.Create(
            SemanticKind.Yield(valueNode.Id),
            NativeType.TUnit,  // yield itself returns unit
            range,
            children = [valueNode.Id])
```

### 4.6 Checking `for...in` Expressions

```fsharp
/// Check a for-each loop
let checkForEach
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (loopVar: string)
    (sourceExpr: SynExpr)
    (bodyExpr: SynExpr)
    (range: range)
    : SemanticNode =

    // Check the source sequence
    let sourceNode = checkExpr env builder sourceExpr

    // Extract element type from TSeq
    let elemType =
        match sourceNode.Type with
        | NativeType.TSeq elemTy -> elemTy
        | _ -> failwith "for...in requires a sequence source"

    // Extend environment with loop variable
    let loopEnv = env.AddLocal(loopVar, elemType)

    // Check body
    let bodyNode = checkExpr loopEnv builder bodyExpr

    builder.Create(
        SemanticKind.ForEach(loopVar, elemType, sourceNode.Id, bodyNode.Id),
        NativeType.TUnit,
        range,
        children = [sourceNode.Id; bodyNode.Id])
```

### 4.7 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `NTUseq`, `TSeq` |
| `SemanticGraph.fs` | MODIFY | Add `SeqExpr`, `Yield`, `ForEach` |
| `Types.fs` | MODIFY | Add `EnclosingSeqExpr` to TypeEnv |
| `Expressions/Computations.fs` | MODIFY | Seq/yield/for-in checking |
| `Expressions/Coordinator.fs` | MODIFY | Route expressions |
| `Unify.fs` | MODIFY | Add TSeq bridge cases |

### 4.8 Type Unification Bridge Cases (Critical Architecture)

**This section documents critical architectural knowledge for C-06 through T-04.**

#### 4.8.1 The Type Representation Duality

Types like `Seq<T>`, `Lazy<T>`, `nativeptr<T>`, and `byref<T>` exist in **TWO representations** within FNCS:

| Representation | Source | Example | Memory Model |
|----------------|--------|---------|--------------|
| **Direct Form** | Programmatic type construction (`mkSeqType`, `mkLazyType`) | `TSeq elem`, `TLazy elem` | Has proper struct layout info |
| **TApp Form** | Parsing type syntax (`seq<int>`, `Lazy<string>`) | `TApp(seqTyCon, [elem])` | Generic type application |

**Why both exist:**
- When FNCS constructs types programmatically (e.g., `mkSeqType int`), it creates `TSeq int` directly
- When FCS parses type syntax like `seq<int>`, it creates `TApp(seqTyCon, [int])`
- Both represent the **same semantic type** but through different construction paths

#### 4.8.2 The Unification Problem

Without explicit handling, `TSeq int` and `TApp(seq, [int])` would **NOT unify**:

```fsharp
// This would fail without bridge cases:
let x: seq<int> = seq { yield 1 }  // seq { } creates TSeq int
//    ^^^^^^^^^                     // type annotation parses to TApp(seq, [int])
// Error: Cannot unify TSeq int with TApp(seq, [int])
```

#### 4.8.3 Bridge Cases in Unify.fs

The solution is explicit "bridge cases" in the `unify` function that make both representations equivalent:

```fsharp
// In Unify.fs - the unify function

// TSeq directly unifies with itself
| NativeType.TSeq elem1, NativeType.TSeq elem2 ->
    unify elem1 elem2 range

// Bridge: TSeq <-> TApp(seq, [elem])
| NativeType.TSeq elem1, NativeType.TApp(tc, [elem2]) when tc.Name = "seq" ->
    unify elem1 elem2 range
| NativeType.TApp(tc, [elem1]), NativeType.TSeq elem2 when tc.Name = "seq" ->
    unify elem1 elem2 range
```

#### 4.8.4 Existing Bridge Cases (Pre-C-06)

This pattern already existed for `nativeptr` and `byref`:

```fsharp
// TNativePtr <-> TApp(nativeptr, [elem])
| NativeType.TNativePtr elem1, NativeType.TApp(tc, [elem2])
    when tc.Name = "nativeptr" || tc.Name = "NativePtr" ->
    unify elem1 elem2 range

// TByref <-> TApp(byref/inref/outref, [elem])
| NativeType.TByref elem1, NativeType.TApp(tc, [elem2])
    when tc.Name = "byref" || tc.Name = "inref" || tc.Name = "outref" ->
    unify elem1 elem2 range
```

#### 4.8.5 Bridge Cases Added in C-06

| Direct Form | TApp Names | Purpose |
|-------------|------------|---------|
| `TSeq elem` | `"seq"` | Sequence expressions |
| `TLazy elem` | `"Lazy"`, `"lazy"` | Lazy thunks (gap fixed from C-05) |

**Important Discovery**: TLazy was missing its bridge case before C-06 - this was a pre-existing gap that has been fixed as part of this implementation.

#### 4.8.6 When to Add Bridge Cases

**Add a bridge case when introducing ANY new direct-form type case** that:
1. Represents a parameterized type (has element type(s))
2. Can also appear as a type application from parsed syntax
3. Needs to unify with both representations

**Future PRDs that may need bridge cases:**
- `TAsync<T>` if introduced as a direct case
- `TResult<T, E>` if introduced as a direct case
- Any new "struct wrapper" types

#### 4.8.7 The canUnify Function

Note: The `canUnify` function (used for type compatibility checks without side effects) must also have corresponding bridge cases. Ensure both `unify` and `canUnify` are updated together.

#### 4.8.8 Why Direct Forms Exist

Direct forms (`TSeq`, `TLazy`, `TNativePtr`, `TByref`) exist because they carry **additional semantic information** beyond generic type application:

| Direct Form | Semantic Information |
|-------------|---------------------|
| `TSeq elem` | State machine struct layout, MoveNext pattern |
| `TLazy elem` | Memoization fields, force pattern |
| `TNativePtr elem` | Pointer semantics, no GC tracking |
| `TByref elem` | Reference semantics, stack discipline |

The `TApp` form is more general but loses this domain-specific knowledge. Bridge cases ensure both representations are considered equivalent while preserving the richer semantics when available.

## 5. Alex Layer Implementation

### 5.1 YieldStateIndices Coeffect

Following the SSAAssignment pattern, assign state indices as coeffects:

**File**: `src/Alex/Preprocessing/YieldStateIndices.fs`

**Yield Numbering Order**: Yields are numbered **in document (syntactic) order** - a pre-order traversal of the PSG body. This provides deterministic, predictable state indices:

| State | Meaning |
|-------|---------|
| 0 | Initial state (before first yield) |
| 1 | After first yield in document order |
| 2 | After second yield in document order |
| N | After Nth yield |
| -1 | Completed (no more values) |

**Example**:
```fsharp
seq {
    yield 1       // State transition: 0 -> 1
    if cond then
        yield 2   // State transition: 1 -> 2 (if taken)
    yield 3       // State transition: 1 -> 3 or 2 -> 3
}
```

The `collectYieldsInBody` function performs **pre-order PSG traversal**, visiting children left-to-right in document order.

```fsharp
/// Coeffect: Maps each Yield NodeId to its state index
type YieldStateCoeffect = {
    /// SeqExpr NodeId -> (Yield NodeId -> state index)
    StateIndices: Map<NodeId, Map<NodeId, int>>
}

/// Collect yields in document (pre-order) traversal order
let rec collectYieldsInBody (graph: SemanticGraph) (nodeId: NodeId) : NodeId list =
    let node = graph.Nodes.[nodeId]
    let childYields = node.Children |> List.collect (collectYieldsInBody graph)
    match node.Kind with
    | SemanticKind.Yield _ -> nodeId :: childYields  // This yield + nested yields
    | _ -> childYields

/// Assign state indices to all yield points in document order
let run (graph: SemanticGraph) : YieldStateCoeffect =
    let mutable indices = Map.empty

    for node in graph.Nodes.Values do
        match node.Kind with
        | SemanticKind.SeqExpr(bodyId, _) ->
            let yields = collectYieldsInBody graph bodyId
            let yieldIndices =
                yields
                |> List.mapi (fun i yieldId -> (yieldId, i + 1))  // States 1, 2, 3...
                |> Map.ofList
            indices <- Map.add node.Id yieldIndices indices
        | _ -> ()

    { StateIndices = indices }
```

### 5.2 SeqLayout Coeffect

Following C-05's LazyLayout pattern:

**File**: `src/Alex/Preprocessing/SeqLayout.fs`

```fsharp
/// Layout information for a sequence expression
type SeqLayout = {
    /// NodeId of the SeqExpr
    SeqId: NodeId
    /// Element type (T in Seq<T>)
    ElementType: MLIRType
    /// Capture layouts (reuse from closure)
    Captures: CaptureLayout list
    /// Internal state layouts (mutable vars in seq body)
    InternalState: StateFieldLayout list
    /// Total struct type
    StructType: MLIRType
    /// Number of yield points
    NumYields: int
    /// MoveNext function name
    MoveNextFuncName: string
}

/// Generate MLIR struct type for a sequence expression
let seqStructType (elementType: MLIRType) (captureTypes: MLIRType list) (internalStateTypes: MLIRType list) : MLIRType =
    // {state: i32, current: T, code_ptr: ptr, cap₀, cap₁, ..., state₀, state₁, ...}
    TStruct ([TInt I32; elementType; TPtr] @ captureTypes @ internalStateTypes)
```

**MoveNextFuncName Generation Pattern**: MoveNext function names are deterministically generated from the SeqExpr's NodeId to ensure uniqueness and debuggability:

```fsharp
/// Generate MoveNext function name for a sequence
let generateMoveNextName (seqId: NodeId) (enclosingFuncName: string option) : string =
    // Pattern: seq_<nodeid>_moveNext or <enclosingFunc>_seq_<nodeid>_moveNext
    match enclosingFuncName with
    | Some funcName -> sprintf "%s_seq_%d_moveNext" funcName seqId.Value
    | None -> sprintf "seq_%d_moveNext" seqId.Value

// Examples:
// - Top-level: "seq_42_moveNext"
// - Inside 'main': "main_seq_42_moveNext"
// - Inside 'countUp': "countUp_seq_87_moveNext"
```

This naming convention:
1. **Uniqueness**: NodeId ensures no collisions
2. **Debuggability**: Enclosing function name provides context in stack traces
3. **Determinism**: Same input always produces same name

### 5.3 SeqWitness - Creation

```fsharp
/// Emit MLIR for sequence expression initialization
let witnessSeqCreate
    (z: PSGZipper)
    (layout: SeqLayout)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let structType = layout.StructType
    let ssas = requireNodeSSAs layout.SeqId z

    let ops = [
        // Create initial state = 0
        MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 0L, MLIRTypes.i32))

        // Create undef seq struct
        MLIROp.LLVMOp (LLVMOp.Undef (ssas.[1], structType))

        // Insert state=0 at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (ssas.[2], ssas.[1], ssas.[0], [0], structType))

        // Get MoveNext function address and insert at index 2
        MLIROp.LLVMOp (LLVMOp.AddressOf (ssas.[3], GFunc layout.MoveNextFuncName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (ssas.[4], ssas.[2], ssas.[3], [2], structType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps = ... // Same pattern as LazyWitness

    (ops @ captureOps, TRValue { SSA = resultSSA; Type = structType })
```

### 5.4 MoveNext Function Generation

The MoveNext function is generated as a state machine:

```fsharp
/// Generate MoveNext function for a sequence
/// Signature: (ptr-to-seq-struct) -> i1
/// Mutates the struct (state, current) through the pointer
let emitMoveNextFunction
    (seqId: NodeId)
    (layout: SeqLayout)
    (yieldIndices: Map<NodeId, int>)
    (bodyEmitter: int -> MLIRBuilder)  // state -> code for that state
    : MLIROp list =

    // MoveNext takes a POINTER to the seq struct (for mutation)
    // Extracts current state, switches, executes code, stores new state/current

    let numYields = layout.NumYields

    // Generate:
    // 1. Load state from struct
    // 2. Switch on state: 0 -> ^state0, 1 -> ^state1, ...
    // 3. Each state block: execute code until yield, store current, update state, return true
    // 4. Done block: set state = -1, return false

    ...
```

**Key insight**: MoveNext takes a **pointer** to the seq struct because it needs to mutate `state` and `current`. This is different from force (which can work with by-value for pure computation).

### 5.5 ForEachWitness

```fsharp
/// Emit MLIR for for-each loop
let witnessForEach
    (z: PSGZipper)
    (loopVar: string)
    (seqLayout: SeqLayout)
    (seqVal: Val)
    (bodyEmitter: Val -> MLIROp list)  // current value -> body ops
    : MLIROp list =

    // 1. Allocate seq struct on stack (if not already a pointer)
    // 2. Loop: call MoveNext, check result, extract current, emit body, repeat

    [
        // Alloca for seq struct
        MLIROp.LLVMOp (LLVMOp.Alloca (seqAllocaSSA, one, seqLayout.StructType, None))
        MLIROp.LLVMOp (LLVMOp.Store (seqVal.SSA, seqAllocaSSA, seqLayout.StructType, NotAtomic))

        // Loop header
        // %has_next = call @moveNext(%seq_alloca)
        // cond_br %has_next, ^body, ^done

        // Loop body
        // %current_ptr = gep %seq_alloca[0, 1]
        // %current = load %current_ptr
        // ... body using %current ...
        // br ^header

        // Done
        // ...
    ]
```

### 5.6 SSA Cost Computation

```fsharp
/// SSA cost for SeqExpr with N captures
let seqExprSSACost (numCaptures: int) : int =
    // 1: state constant (0)
    // 1: undef struct
    // 1: insert state
    // 1: addressof code_ptr
    // 1: insert code_ptr
    // N: insert each capture
    5 + numCaptures

/// SSA cost for ForEach
let forEachSSACost : int =
    // 1: alloca for seq
    // 1: store seq value
    // 1: moveNext result
    // 1: current_ptr (gep)
    // 1: current value (load)
    5
```

### 5.7 Files to Create/Modify (Alex)

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/YieldStateIndices.fs` | CREATE | State index coeffects |
| `Alex/Preprocessing/SeqLayout.fs` | CREATE | Seq struct layout coeffects |
| `Alex/Witnesses/SeqWitness.fs` | CREATE | Seq creation, MoveNext emission |
| `Alex/Witnesses/ForEachWitness.fs` | CREATE | For-each loop emission |
| `Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle SeqExpr, Yield, ForEach |

## 6. MLIR Output Specification

### 6.1 Simple Sequence: `seq { yield 1; yield 2; yield 3 }`

```mlir
// Sequence struct is memref<Nxi8> (flat byte buffer)
// Layout: [state: i32, current: i32, code_ptr: index]
// No captures for this simple sequence.

// MoveNext function — takes seq struct as memref, returns i1
func.func @seq_123_moveNext(%self: memref<?xi8>) -> i1 {
    // Load current state via memref.view at offset 0
    %state_ref = memref.reinterpret_cast %self to offset: [0], sizes: [1], strides: [4]
        : memref<?xi8> to memref<1xi32>
    %state = memref.load %state_ref[%c0] : memref<1xi32>

    // Current value field at offset 4
    %curr_ref = memref.view %self[%c4][] : memref<?xi8> to memref<1xi32>

    // State machine via nested scf.if (or scf.index_switch when available)
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %is_s0 = arith.cmpi eq, %state, %c0_i32 : i32
    %result = scf.if %is_s0 -> i1 {
        // state0: yield 1
        %c1 = arith.constant 1 : i32
        memref.store %c1, %curr_ref[%c0] : memref<1xi32>
        memref.store %c1_i32, %state_ref[%c0] : memref<1xi32>
        scf.yield %true : i1
    } else {
        %is_s1 = arith.cmpi eq, %state, %c1_i32 : i32
        %r1 = scf.if %is_s1 -> i1 {
            // state1: yield 2
            %c2 = arith.constant 2 : i32
            memref.store %c2, %curr_ref[%c0] : memref<1xi32>
            memref.store %c2_i32, %state_ref[%c0] : memref<1xi32>
            scf.yield %true : i1
        } else {
            %is_s2 = arith.cmpi eq, %state, %c2_i32 : i32
            %r2 = scf.if %is_s2 -> i1 {
                // state2: yield 3
                %c3 = arith.constant 3 : i32
                memref.store %c3, %curr_ref[%c0] : memref<1xi32>
                %c3_state = arith.constant 3 : i32
                memref.store %c3_state, %state_ref[%c0] : memref<1xi32>
                scf.yield %true : i1
            } else {
                // done
                %neg1 = arith.constant -1 : i32
                memref.store %neg1, %state_ref[%c0] : memref<1xi32>
                scf.yield %false : i1
            }
            scf.yield %r2 : i1
        }
        scf.yield %r1 : i1
    }
    func.return %result : i1
}
```

### 6.2 Sequence with Captures: `seq { for i in 1..n do yield i * factor }`

```mlir
// Sequence struct is memref<Nxi8> (flat byte buffer with captures)
// Layout: [state: i32, current: i32, code_ptr: index, n: i32, factor: i32, i: i32]
// Captures: n, factor. Mutable loop state: i.

// MoveNext function
func.func @seq_factors_moveNext(%self: memref<?xi8>) -> i1 {
    // Load state via memref.view at offset 0
    %state_ref = memref.view %self[%c0][] : memref<?xi8> to memref<1xi32>
    %state = memref.load %state_ref[%c0] : memref<1xi32>

    // Load captures and mutable state via memref.view at computed offsets
    %n_ref = memref.view %self[%off_n][] : memref<?xi8> to memref<1xi32>
    %n = memref.load %n_ref[%c0] : memref<1xi32>
    %factor_ref = memref.view %self[%off_factor][] : memref<?xi8> to memref<1xi32>
    %factor = memref.load %factor_ref[%c0] : memref<1xi32>
    %i_ref = memref.view %self[%off_i][] : memref<?xi8> to memref<1xi32>
    %curr_ref = memref.view %self[%c4][] : memref<?xi8> to memref<1xi32>

    // State machine: init (state 0) or advance (state 1), then check + yield
    %c0_i32 = arith.constant 0 : i32
    %is_init = arith.cmpi eq, %state, %c0_i32 : i32
    scf.if %is_init {
        %one = arith.constant 1 : i32
        memref.store %one, %i_ref[%c0] : memref<1xi32>
    } else {
        %i_val = memref.load %i_ref[%c0] : memref<1xi32>
        %one = arith.constant 1 : i32
        %i_next = arith.addi %i_val, %one : i32
        memref.store %i_next, %i_ref[%c0] : memref<1xi32>
    }

    // Check: i > n?
    %i_current = memref.load %i_ref[%c0] : memref<1xi32>
    %done_cond = arith.cmpi sgt, %i_current, %n : i32
    %result = scf.if %done_cond -> i1 {
        // Done
        %neg1 = arith.constant -1 : i32
        memref.store %neg1, %state_ref[%c0] : memref<1xi32>
        scf.yield %false : i1
    } else {
        // Yield: current = i * factor
        %product = arith.muli %i_current, %factor : i32
        memref.store %product, %curr_ref[%c0] : memref<1xi32>
        %c1_i32 = arith.constant 1 : i32
        memref.store %c1_i32, %state_ref[%c0] : memref<1xi32>
        scf.yield %true : i1
    }
    func.return %result : i1
}
```

### 6.3 For-Each Loop: `for x in numbers do ...`

```mlir
// Allocate seq struct on stack — memref<Nxi8> flat buffer
%seq_buf = memref.alloca() : memref<Nxi8>
// Initialize state=0 via memref.store at offset 0

// Iteration via scf.while
scf.while : () -> () {
    %has_next = func.call @seq_123_moveNext(%seq_buf) : (memref<?xi8>) -> i1
    scf.condition(%has_next)
} do {
    // Extract current value via memref.view at current field offset
    %curr_ref = memref.view %seq_buf[%c4][] : memref<Nxi8> to memref<1xi32>
    %x = memref.load %curr_ref[%c0] : memref<1xi32>
    // ... loop body using %x ...
    scf.yield
}
```

## 7. Validation

### 7.1 Sample Code

```fsharp
/// Sample 15: Simple Sequence Expressions
module SimpleSeqSample

// Basic sequence - literal yields
let threeNumbers = seq {
    yield 1
    yield 2
    yield 3
}

// Sequence with while loop
let countUp (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

// Sequence with conditional yields
let evenNumbersUpTo (max: int) = seq {
    let mutable n = 0
    while n <= max do
        if n % 2 = 0 then
            yield n
        n <- n + 1
}

// Sequence with captures
let multiplesOf (factor: int) (count: int) = seq {
    let mutable i = 1
    while i <= count do
        yield factor * i
        i <- i + 1
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 15: Simple Sequences ==="
    Console.writeln ""

    Console.writeln "--- Basic Sequence ---"
    for x in threeNumbers do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Count 1 to 5 ---"
    for x in countUp 1 5 do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Evens up to 10 ---"
    for x in evenNumbersUpTo 10 do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Multiples of 3 (first 5) ---"
    for x in multiplesOf 3 5 do
        Console.writeln (Format.int x)

    0
```

### 7.2 Expected Output

```
=== Sample 15: Simple Sequences ===

--- Basic Sequence ---
1
2
3

--- Count 1 to 5 ---
1
2
3
4
5

--- Evens up to 10 ---
0
2
4
6
8
10

--- Multiples of 3 (first 5) ---
3
6
9
12
15
```

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add `NTUseq` to NTUKind enum
- [ ] Add `TSeq` to NativeType
- [ ] Add `SeqExpr`, `Yield`, `ForEach` to SemanticKind
- [ ] Add `EnclosingSeqExpr` to TypeEnv
- [ ] Implement seq { } expression checking with capture analysis
- [ ] Implement yield checking
- [ ] Implement for...in checking
- [ ] FNCS builds successfully

### Phase 2: Alex Implementation
- [ ] Create `YieldStateIndices.fs` coeffect pass
- [ ] Create `SeqLayout.fs` coeffect pass
- [ ] Create `SeqWitness.fs` with flat closure model
- [ ] Create `ForEachWitness.fs`
- [ ] Handle SeqExpr, Yield, ForEach in FNCSTransfer
- [ ] Generate MoveNext functions with state machines
- [ ] Firefly builds successfully

### Phase 3: Validation
- [ ] Sample 15 compiles without errors
- [ ] Binary executes correctly
- [ ] State machine transitions verified
- [ ] Captures work correctly
- [ ] Samples 01-14 still pass (regression)

## 9. Lessons Applied

| Lesson | Application |
|--------|-------------|
| Flat closure model (C-01) | Captures inlined in seq struct |
| Pre-creation pattern (C-03) | SeqExpr node created before checking body |
| Environment enrichment (C-03) | `EnclosingSeqExpr` added to TypeEnv |
| Coeffect pattern (C-05) | YieldStateIndices, SeqLayout computed before witnessing |
| Extended closure (C-05) | Seq struct = closure + state machine fields |

## 10. Related PRDs

- **C-01**: Closures - Sequences reuse flat closure model
- **C-03**: Recursion - Pre-creation and environment enrichment patterns
- **C-05**: Lazy - Foundation for extended flat closures
- **C-07**: SeqOperations - `Seq.map`, `Seq.filter`, etc.
- **A-01**: Async - Builds on same deferred computation model

## 11. Architectural Alignment

This PRD aligns with the flat closure architecture:

**Key principles maintained:**
1. **No nulls** - Every field initialized
2. **No env_ptr** - Captures inlined directly
3. **Self-contained structs** - No pointer chains
4. **Coeffect-based layout** - State indices and layout computed before witnessing
5. **Capture reuse** - Same analysis as C-01 closures
6. **MoveNext by pointer** - Mutation requires pointer to struct (stack-allocated for for-each)

> **Critical:** See Serena memory `compose_from_standing_art_principle` for why composing from C-01/14 patterns is essential. New features MUST extend standing art, not reinvent.
