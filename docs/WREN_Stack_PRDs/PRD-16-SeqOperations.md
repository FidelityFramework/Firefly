# PRD-16: Sequence Operations

> **Sample**: `16_SeqOperations` | **Status**: Planned | **Depends On**: PRD-15 (SimpleSeq), PRD-14 (Lazy), PRD-12 (HOFs), PRD-11 (Closures)

---

## CONTEXT WINDOW RESET PROTOCOL

> **At the START of every new context window, IMMEDIATELY review this section to establish bearings.**

### Quick Start Checklist

1. **Activate Firefly project**: `mcp__serena-local__activate_project "Firefly"`
2. **Read progress memory**: `mcp__serena-local__read_memory "prd16_seqoperations_progress"`
3. **Review this PRD**: Understand the architectural through-line
4. **Check standing art files**: Review the key implementation files listed below

### Standing Art Files (PRD-11, PRD-14, PRD-15)

These files contain the patterns that PRD-16 MUST compose from:

#### Witness Layer (`src/Alex/Witnesses/`)

| File | Purpose | Key Exports to Study |
|------|---------|---------------------|
| `SeqWitness.fs` | Seq struct creation & MoveNext | `seqStructTypeFull`, `witnessSeqCreateFull`, `witnessMoveNextWhileBased`, `WhileBasedMoveNextInfo`, `YieldBlockInfo` |
| `LazyWitness.fs` | Lazy thunk pattern (simpler precursor) | `lazyStructType`, `witnessLazyCreate`, `witnessLazyForce` |
| `LambdaWitness.fs` | Flat closure construction | `buildClosureConstruction`, `buildCaptureExtractionOps`, `witness` |

#### Preprocessing Layer (`src/Alex/Preprocessing/`)

| File | Purpose | Key Exports to Study |
|------|---------|---------------------|
| `SSAAssignment.fs` | SSA allocation + ClosureLayout coeffect | `ClosureLayout`, `CaptureSlot`, `CaptureMode` (ByValue/ByRef), `computeSeqExprSSACost`, `buildClosureLayout`, `computeLambdaSSACost` |
| `YieldStateIndices.fs` | Seq body structure analysis | `SeqYieldInfo`, `WhileBodyInfo`, `InternalStateField`, `YieldInfo`, `analyzeBodyStructure`, `collectMutableBindings` |

#### Transfer/Traversal Layer (`src/Alex/Traversal/`)

| File | Purpose | Key Patterns |
|------|---------|--------------|
| `FNCSTransfer.fs` | PSG traversal → MLIR emission | `SemanticKind.SeqExpr` handling (~line 408, ~1128), WhileBased pattern with `loadVar`/`storeVar`, capture extraction |

#### Type Mapping (`src/Alex/CodeGeneration/`)

| File | Purpose |
|------|---------|
| `TypeMapping.fs` | `NativeType` → `MLIRType` conversion, `mapNativeTypeWithGraphForArch` |

### Architectural Through-Line

```
PRD-11 (Closures)     → Flat closure: {code_ptr, cap₀, cap₁, ...}
         ↓ extends (adds state prefix)
PRD-14 (Lazy)         → Extended closure: {computed: i1, value: T, code_ptr, cap₀...}
         ↓ extends (adds internal state suffix)
PRD-15 (SimpleSeq)    → State machine: {state: i32, current: T, code_ptr, cap₀..., internalState₀...}
         ↓ composes (nests inner structures)
PRD-16 (SeqOperations)→ Wrapper sequences: {state, current, code_ptr, inner_seq, closure}
```

### Key Serena Memories

- `architecture_principles` - Layer separation, non-dispatch model
- `negative_examples` - Anti-patterns to avoid
- `lazy_seq_flat_closure_architecture` - Flat closure model specifics
- `true_flat_closures_implementation` - PRD-11 closure patterns
- `fncs_functional_decomposition_principle` - How intrinsics should decompose
- `prd16_seqoperations_progress` - **EPHEMERAL**: Current implementation progress

---

## NORMATIVE SPEC REFERENCES

> **The fsnative-spec repository contains the authoritative specifications. These documents are the "north star" for implementation.**

### Primary Spec Chapters (fsnative-spec/spec/)

| Chapter | Path | PRD-16 Relevance |
|---------|------|------------------|
| **Closure Representation** | `spec/closure-representation.md` | §3: Flat closure struct layout; §8: Nested functions vs escaping closures (mappers/predicates are ESCAPING) |
| **Lazy Representation** | `spec/lazy-representation.md` | §4: Struct pointer passing convention; foundation pattern for MoveNext |
| **Seq Representation** | `spec/seq-representation.md` | §3: Base seq struct layout; §4: MoveNext calling convention; §5: Sequential flattening |
| **Backend Lowering** | `spec/drafts/backend-lowering-architecture.md` | §3: Flat closure pattern requires backend-specific MLIR (addressof, indirect call) |

### NEW: Seq Operations Representation (Draft)

**Path**: `fsnative-spec/spec/drafts/seq-operations-representation.md`

This draft chapter was created to fill a spec gap identified during PRD-16 development. It specifies:

- §4: Wrapper struct layouts (MapSeq, FilterSeq, TakeSeq, CollectSeq)
- §5: Copy semantics - inner seq and closure copied by VALUE
- §6: Composition model - nested structs for pipelines
- §7: MoveNext algorithms for each operation
- §8: Seq.fold as eager consumer (no wrapper)

**ACTION REQUIRED**: This draft should be promoted to a formal spec chapter before PRD-16 implementation is complete.

### Key Spec Constraints for PRD-16

From the spec audit, these constraints MUST be honored:

1. **Flat Representation** (closure-representation.md §2.2): Wrappers use flat closures, NO `env_ptr`
2. **Copy Semantics** (seq-operations-representation.md §5): Inner seq and closure copied by value at wrapper creation
3. **Escaping Closure Classification** (closure-representation.md §8.3): Mappers/predicates are escaping (passed as values), use closure struct not parameter-passing
4. **Struct Pointer Passing** (lazy-representation.md §4.1): MoveNext receives pointer to containing struct
5. **Module-Level Exclusion** (lazy-representation.md §5.1): Module-level bindings are NOT captured
6. **Backend-Specific Operations** (backend-lowering.md §3): Taking function address requires `llvm.mlir.addressof`, struct manipulation requires `llvm.insertvalue`/`llvm.extractvalue`

### Spec Gap Identified and Addressed

**Gap**: The original `seq-representation.md` covered PRD-15 (seq expressions) but NOT PRD-16 (Seq module operations).

**Resolution**: Created `spec/drafts/seq-operations-representation.md` covering:
- Wrapper sequence structures
- Copy semantics rationale
- Composition (nested struct) model
- MoveNext algorithm specifications
- SSA cost formulas

---

## 1. Executive Summary

This PRD covers the core sequence operations: `Seq.map`, `Seq.filter`, `Seq.take`, `Seq.fold`, etc. These are higher-order functions over sequences - they transform or consume lazy enumerations.

**Key Insight**: Seq operations create **composed flat closures**. `Seq.map f xs` produces a new seq struct that contains:
1. The original sequence (inlined, not by pointer)
2. The mapper closure (flat closure, no `env_ptr`)
3. Its own state machine fields

**Builds on PRD-15**: Seq operations wrap inner sequences. The wrapper is itself a flat closure with state machine fields. The inner sequence and transformation closure are inlined captures.

## 2. Language Feature Specification

### 2.1 Seq.map

```fsharp
let doubled = Seq.map (fun x -> x * 2) numbers
```

Transforms each element lazily.

### 2.2 Seq.filter

```fsharp
let evens = Seq.filter (fun x -> x % 2 = 0) numbers
```

Yields only elements matching predicate.

### 2.3 Seq.take

```fsharp
let first5 = Seq.take 5 infiniteSeq
```

Yields at most N elements.

### 2.4 Seq.fold

```fsharp
let sum = Seq.fold (fun acc x -> acc + x) 0 numbers
```

Eager reduction to single value (consumes sequence).

### 2.5 Seq.collect (flatMap)

```fsharp
let flattened = Seq.collect (fun x -> seq { yield x; yield x * 10 }) numbers
```

Maps then flattens.

## 3. Architectural Principles

### 3.1 Composed Flat Closures (No `env_ptr`)

Seq operations create wrapper sequences. Following the flat closure model:

```
MapSeq<A,B> = {state: i32, current: B, moveNext_ptr: ptr, inner_seq: Seq<A>, mapper: (A -> B)}
```

Both `inner_seq` and `mapper` are **inlined** (flat), not stored by pointer.

| Field | Type | Purpose |
|-------|------|---------|
| `state` | `i32` | Wrapper's own state |
| `current` | `B` | Current transformed value |
| `moveNext_ptr` | `ptr` | Wrapper's MoveNext function |
| `inner_seq` | `Seq<A>` | Inlined inner sequence struct |
| `mapper` | `(A -> B)` | Inlined mapper closure |

**No `env_ptr` anywhere.** The mapper closure itself is flat: `{code_ptr, cap₀, cap₁, ...}`.

### 3.2 Struct Layout Examples

**Seq.map** (mapper with no captures):
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 mapper: {mapper_code_ptr: ptr}}
```

**Seq.map** (mapper captures `factor`):
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 mapper: {mapper_code_ptr: ptr, factor: i32}}
```

**Seq.filter**:
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 predicate: {pred_code_ptr: ptr, ...captures...}}
```

**Seq.take**:
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 remaining: i32}
```

### 3.3 The Composition Challenge

When sequences are composed (e.g., `Seq.take 5 (Seq.map f (Seq.filter p xs))`), the struct grows:

```
TakeSeq {
    state, current, moveNext_ptr,
    inner: MapSeq {
        state, current, moveNext_ptr,
        inner: FilterSeq {
            state, current, moveNext_ptr,
            inner: OriginalSeq { ... },
            predicate: {...}
        },
        mapper: {...}
    },
    remaining: i32
}
```

This is a **compile-time known** nested struct. No heap allocation. Each composition adds to the struct size, but the exact size is known at compile time.

### 3.4 Why Inline Instead of Pointer?

**Flat closure philosophy**: Self-contained, no indirection.

If we stored inner sequences by pointer:
- Need arena/heap allocation for the inner sequence
- Add indirection (cache misses)
- Lifetime management complexity

With inlining:
- Single contiguous struct
- Stack allocation works
- Predictable memory layout
- No lifetime issues (everything has same lifetime)

**Trade-off**: Struct size grows with composition depth. For typical pipeline depths (3-5), this is acceptable. Very deep pipelines would benefit from alternative strategies (future optimization).

### 3.5 Copy Semantics (Critical for Correctness)

**Seq operation wrappers COPY the inner seq and closure structs by value, not by pointer.**

When creating a wrapper sequence (e.g., `Seq.map`):

```fsharp
let mapped = Seq.map mapper innerSeq
```

The wrapper struct creation performs **value copies**:

```mlir
// Create map wrapper struct - COPIES both inner and mapper
%undef = llvm.mlir.undef : !map_seq_type
%s0 = llvm.insertvalue %zero, %undef[0] : !map_seq_type     // state = 0
%s1 = llvm.insertvalue %code_ptr, %s0[2] : !map_seq_type    // code_ptr
%s2 = llvm.insertvalue %inner_seq_VAL, %s1[3] : !map_seq_type  // COPY inner seq
%s3 = llvm.insertvalue %mapper_VAL, %s2[4] : !map_seq_type     // COPY mapper closure
```

**Why this matters:**
1. **Independence**: Each wrapper owns its own copy of the inner seq's state
2. **Iteration**: Multiple iterations of the same wrapper are independent
3. **No aliasing**: No shared mutable state between wrappers

**Consequence for closure invocation**: When invoking the mapper/predicate closure, we extract the flat closure value and pass captures as arguments:

```mlir
// Extract mapper closure (value copy in struct)
%mapper_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr
%mapper = llvm.load %mapper_ptr : !mapper_closure_type

// Extract code_ptr and captures (if any)
%code_ptr = llvm.extractvalue %mapper[0] : !mapper_closure_type
%cap0 = llvm.extractvalue %mapper[1] : !mapper_closure_type  // if captures exist

// Invoke: code_ptr(captures..., value)
%result = llvm.call %code_ptr(%cap0, %inner_val) : (...) -> !output_type
```

**Critical**: The closure is stored **by value** in the wrapper struct. When invoking, we extract the code pointer and captures from the inlined closure, not from a pointer indirection.

## 4. FNCS Layer Implementation

### 4.1 Seq Module Intrinsics

**File**: `~/repos/fsnative/src/Compiler/NativeTypedTree/Expressions/Intrinsics.fs`

> **Note (January 2026)**: `Seq.empty` was added early to FNCS to unblock BAREWire dependency resolution.
> This is a foundational sequence producer that belongs conceptually between PRD-15 (seq expressions)
> and PRD-16 (seq operations). Added here for convenience given shared PRD boundaries.

```fsharp
// Seq intrinsic module
| "Seq.empty" ->
    // seq<'T> - Returns an empty sequence (polymorphic value)
    let ty = NativeType.TForall([tyParamSpecT], seqT)
    Resolved (mkIntrinsic IntrinsicModule.Seq op IntrinsicCategory.Pure fullName, ty)

| "Seq.map" ->
    // ('a -> 'b) -> seq<'a> -> seq<'b>
    let aVar = freshTypeVar ()
    let bVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(aVar, bVar),
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(bVar)))

| "Seq.filter" ->
    // ('a -> bool) -> seq<'a> -> seq<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(aVar, env.Globals.BoolType),
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(aVar)))

| "Seq.take" ->
    // int -> seq<'a> -> seq<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        env.Globals.IntType,
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(aVar)))

| "Seq.fold" ->
    // ('state -> 'a -> 'state) -> 'state -> seq<'a> -> 'state
    let stateVar = freshTypeVar ()
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(stateVar, NativeType.TFun(aVar, stateVar)),
        NativeType.TFun(stateVar,
            NativeType.TFun(NativeType.TSeq(aVar), stateVar)))
```

### 4.2 SemanticKind for Seq Operations

Use existing `Application` with `IntrinsicInfo` marking:

```fsharp
// In SemanticGraph.fs
type IntrinsicModule =
    | Console
    | Format
    | Sys
    | NativePtr
    | Math
    | Lazy
    | Seq  // NEW

type IntrinsicInfo = {
    Module: IntrinsicModule
    Operation: string
    // ...
}
```

The PSG represents `Seq.map f xs` as:
```
Application(
    Application(VarRef "Seq.map", [mapperNode]),
    [sourceSeqNode])
```

With intrinsic info attached marking it as `{Module = Seq; Operation = "map"}`.

### 4.3 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Seq.map, filter, take, fold intrinsics |
| `SemanticGraph.fs` | MODIFY | Add `Seq` to IntrinsicModule |
| `NativeGlobals.fs` | MODIFY | Seq module registration |

### 4.4 Type Unification Considerations

PRD-16 operations produce and consume `TSeq` types. The type unification bridge cases documented in **PRD-15 Section 4.8** ensure that:

1. `Seq.map f xs` where `xs: seq<int>` (TApp form from type annotation) correctly unifies with `TSeq int`
2. The result type `seq<'b>` can be used in contexts expecting either representation
3. Chained operations like `Seq.take 5 (Seq.map f xs)` work regardless of how types are constructed

**No new bridge cases are needed for PRD-16** - it uses the existing `TSeq` type and its bridge cases from PRD-15.

## 5. Alex Layer Implementation

### 5.1 SeqOpLayout Coeffect

Compute layouts for sequence operation wrappers:

**File**: `src/Alex/Preprocessing/SeqOpLayout.fs`

```fsharp
/// Layout for a Seq.map wrapper
type MapSeqLayout = {
    ElementType: MLIRType           // Output element type (B)
    InnerElementType: MLIRType      // Input element type (A)
    InnerSeqLayout: SeqLayout       // Layout of inner sequence (inlined)
    MapperLayout: ClosureLayout     // Layout of mapper closure (flat)
    StructType: MLIRType            // Complete wrapper struct type
    MoveNextFuncName: string
}

/// Layout for a Seq.filter wrapper
type FilterSeqLayout = {
    ElementType: MLIRType
    InnerSeqLayout: SeqLayout
    PredicateLayout: ClosureLayout
    StructType: MLIRType
    MoveNextFuncName: string
}

/// Layout for a Seq.take wrapper
type TakeSeqLayout = {
    ElementType: MLIRType
    InnerSeqLayout: SeqLayout
    StructType: MLIRType
    MoveNextFuncName: string
}

/// Generate struct type for MapSeq
let mapSeqStructType (outElemType: MLIRType) (innerSeqType: MLIRType) (mapperType: MLIRType) : MLIRType =
    // {state: i32, current: B, moveNext_ptr: ptr, inner: InnerSeq, mapper: Mapper}
    TStruct [TInt I32; outElemType; TPtr; innerSeqType; mapperType]
```

### 5.2 Seq.map MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.map wrapper
/// Algorithm:
/// 1. Call inner.MoveNext()
/// 2. If false, return false
/// 3. Get inner.current
/// 4. Apply mapper to get transformed value
/// 5. Store in self.current
/// 6. Return true
let emitMapMoveNext (layout: MapSeqLayout) : MLIROp list =
    // MoveNext signature: (ptr) -> i1

    mlir {
        // Get pointer to inner sequence (inlined at fixed offset)
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"

        // Call inner's MoveNext
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^transform, ^done"

        // Transform block
        yield "^transform:"
        // Get inner's current value
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%inner_val = llvm.load %inner_curr_ptr"

        // === MAPPER CLOSURE INVOCATION (Flat Closure Pattern) ===
        // The mapper is stored as a flat closure: {code_ptr, cap₀, cap₁, ...}
        // We extract code_ptr and each capture, then call with captures prepended

        // Get mapper closure (inlined at offset 4)
        yield "%mapper_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%mapper = llvm.load %mapper_ptr : !mapper_closure_type"

        // Extract code_ptr (always at index 0)
        yield "%mapper_code = llvm.extractvalue %mapper[0] : !mapper_closure_type"

        // Extract captures (indices 1, 2, 3, ... based on MapperLayout.Captures)
        // For each capture at index i (1-based in closure struct):
        for i, cap in layout.MapperLayout.Captures |> List.indexed do
            yield sprintf "%%cap_%d = llvm.extractvalue %%mapper[%d] : !mapper_closure_type" i (i + 1)

        // Call mapper: code_ptr(cap₀, cap₁, ..., inner_val) -> B
        // Following flat closure calling convention from PRD-11:
        // - Captures come FIRST (prepended)
        // - Original parameters come LAST
        let capArgs = layout.MapperLayout.Captures |> List.mapi (fun i _ -> sprintf "%%cap_%d" i) |> String.concat ", "
        let callArgs = if capArgs = "" then "%inner_val" else sprintf "%s, %%inner_val" capArgs
        yield sprintf "%%result = llvm.call %%mapper_code(%s) : (...) -> !output_elem_type" callArgs

        // Store transformed value
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %result, %curr_ptr"
        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        // Done block
        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

**Mapper Invocation SSA Breakdown** (for N captures):
| SSA | Purpose |
|-----|---------|
| 1 | GEP to mapper location |
| 1 | Load mapper closure |
| 1 | Extract code_ptr |
| N | Extract each capture |
| 1 | Call mapper |
| **Total** | **4 + N** |

### 5.3 Seq.filter MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.filter wrapper
/// Algorithm:
/// 1. Loop: call inner.MoveNext()
/// 2. If false, return false
/// 3. Get inner.current
/// 4. Apply predicate
/// 5. If true: store in self.current, return true
/// 6. If false: continue loop
let emitFilterMoveNext (layout: FilterSeqLayout) : MLIROp list =
    mlir {
        yield "llvm.br ^loop"

        yield "^loop:"
        // Call inner MoveNext
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^check, ^done"

        yield "^check:"
        // Get inner current
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%val = llvm.load %inner_curr_ptr"

        // Apply predicate (flat closure)
        yield "%pred_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%pred = llvm.load %pred_ptr"
        yield "%pred_code = llvm.extractvalue %pred[0]"
        yield "%matches = llvm.call %pred_code(..., %val) : (...) -> i1"
        yield "llvm.cond_br %matches, ^yield, ^loop"

        yield "^yield:"
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %val, %curr_ptr"
        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

### 5.4 Seq.take MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.take wrapper
/// Algorithm:
/// 1. Check remaining > 0
/// 2. If false, return false
/// 3. Call inner.MoveNext()
/// 4. If false, return false
/// 5. Copy inner.current to self.current
/// 6. Decrement remaining
/// 7. Return true
let emitTakeMoveNext (layout: TakeSeqLayout) : MLIROp list =
    mlir {
        // Check remaining
        yield "%remaining_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%remaining = llvm.load %remaining_ptr : i32"
        yield "%zero = arith.constant 0 : i32"
        yield "%has_remaining = arith.cmpi sgt, %remaining, %zero : i32"
        yield "llvm.cond_br %has_remaining, ^try_inner, ^done"

        yield "^try_inner:"
        // Call inner MoveNext
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^yield, ^done"

        yield "^yield:"
        // Copy current
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%val = llvm.load %inner_curr_ptr"
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %val, %curr_ptr"

        // Decrement remaining
        yield "%one = arith.constant 1 : i32"
        yield "%new_remaining = arith.subi %remaining, %one : i32"
        yield "llvm.store %new_remaining, %remaining_ptr"

        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

### 5.5 Seq.fold (Eager Consumer)

Fold is NOT a sequence transformer - it consumes the sequence eagerly:

```fsharp
/// Emit Seq.fold - consumes sequence to produce single value
let witnessSeqFold
    (z: PSGZipper)
    (folderClosure: Val)       // Flat closure: (state, elem) -> state
    (initialVal: Val)          // Initial accumulator
    (seqVal: Val)              // Source sequence
    (layout: SeqLayout)
    : (MLIROp list * TransferResult) =

    mlir {
        // Allocate seq on stack for mutation
        yield "%seq_alloca = llvm.alloca 1 x !seq_type : !llvm.ptr"
        yield "llvm.store %seq_val, %seq_alloca"

        // Allocate accumulator
        yield "%acc_alloca = llvm.alloca 1 x !state_type : !llvm.ptr"
        yield "llvm.store %initial_val, %acc_alloca"

        yield "llvm.br ^loop"

        yield "^loop:"
        // Call MoveNext
        yield "%moveNext_ptr = llvm.getelementptr %seq_alloca[0, 2] : !llvm.ptr"
        yield "%moveNext = llvm.load %moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %moveNext(%seq_alloca) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^body, ^done"

        yield "^body:"
        // Get current element
        yield "%curr_ptr = llvm.getelementptr %seq_alloca[0, 1] : !llvm.ptr"
        yield "%elem = llvm.load %curr_ptr"

        // Get current accumulator
        yield "%acc = llvm.load %acc_alloca"

        // Apply folder (flat closure)
        yield "%folder_code = llvm.extractvalue %folder[0]"
        // ... extract captures ...
        yield "%new_acc = llvm.call %folder_code(..., %acc, %elem)"

        // Store new accumulator
        yield "llvm.store %new_acc, %acc_alloca"
        yield "llvm.br ^loop"

        yield "^done:"
        yield "%result = llvm.load %acc_alloca"
        yield "llvm.return %result"
    }
```

### 5.6 SSA Cost Formulas

Following PRD-14's coeffect-based SSA pre-computation, each seq operation has deterministic SSA requirements:

#### 5.6.1 Wrapper Creation SSA Costs

| Operation | Formula | Breakdown |
|-----------|---------|-----------|
| **Seq.map** | `5 + sizeof(inner) + sizeof(mapper)` | state(1) + code_ptr(1) + insertvalue×3 + inner fields + mapper fields |
| **Seq.filter** | `5 + sizeof(inner) + sizeof(predicate)` | state(1) + code_ptr(1) + insertvalue×3 + inner fields + predicate fields |
| **Seq.take** | `6 + sizeof(inner)` | state(1) + remaining(1) + code_ptr(1) + insertvalue×3 + inner fields |
| **Seq.fold** | `5` (no wrapper created) | alloca(1) + store(1) + acc_alloca(1) + acc_store(1) + result_load(1) |

```fsharp
/// SSA cost for Seq.map wrapper creation
let mapWrapperSSACost (innerSeqSize: int) (mapperSize: int) : int =
    // 1: constant 0 for state
    // 1: undef wrapper struct
    // 1: insert state
    // 1: addressof moveNext
    // 1: insert moveNext ptr
    // innerSeqSize: copy inner seq into wrapper (insertvalue chain)
    // mapperSize: copy mapper closure into wrapper
    5 + innerSeqSize + mapperSize

/// SSA cost for Seq.filter wrapper creation
let filterWrapperSSACost (innerSeqSize: int) (predicateSize: int) : int =
    5 + innerSeqSize + predicateSize

/// SSA cost for Seq.take wrapper creation
let takeWrapperSSACost (innerSeqSize: int) : int =
    // Same as map/filter, plus 1 for the "remaining" count
    6 + innerSeqSize

/// Seq.fold doesn't create a wrapper - it's an eager consumer
/// Returns SSA cost for the fold loop setup (not per-iteration)
let foldSetupSSACost : int = 5
```

#### 5.6.2 MoveNext Function SSA Costs (Per Invocation)

| Operation | Formula | Notes |
|-----------|---------|-------|
| **map MoveNext** | `10 + N_mapper_caps` | GEP×3 + load×3 + call×2 + extract(1+N) + store + constants |
| **filter MoveNext** | `12 + N_pred_caps` | Same as map + loop branch overhead |
| **take MoveNext** | `14` | remaining check + all of map's cost |
| **fold (per iteration)** | `8 + N_folder_caps` | No wrapper, direct iteration |

```fsharp
/// SSA cost for map MoveNext function body
let mapMoveNextSSACost (numMapperCaptures: int) : int =
    // Inner sequence operations: gep(1) + load moveNext ptr(1) + call(1) + gep curr(1) + load curr(1)
    // Mapper invocation: gep(1) + load(1) + extract code(1) + extract caps(N) + call(1)
    // Store result: gep(1) + store(1)
    // Constants and branch: 2
    10 + numMapperCaptures

/// SSA cost for filter MoveNext (includes loop)
let filterMoveNextSSACost (numPredicateCaptures: int) : int =
    // Same as map, plus loop control
    12 + numPredicateCaptures

/// SSA cost for take MoveNext
let takeMoveNextSSACost : int =
    // Remaining check: load(1) + cmp(1) + constant(1)
    // Plus map-equivalent cost for pass-through
    14

/// SSA cost for fold per-iteration body
let foldIterationSSACost (numFolderCaptures: int) : int =
    8 + numFolderCaptures
```

#### 5.6.3 Composed Pipeline SSA Analysis

For a composition like `Seq.take 5 (Seq.map f (Seq.filter p xs))`:

**Creation phase SSA cost**:
```
filterWrapper = 5 + innerSize + predSize
mapWrapper = 5 + filterWrapperSize + mapperSize
takeWrapper = 6 + mapWrapperSize
```

**Per-iteration SSA cost** (worst case - filter matches):
```
take.MoveNext calls map.MoveNext calls filter.MoveNext calls inner.MoveNext
= takeMoveNext + mapMoveNext + filterMoveNext + innerMoveNext
```

### 5.7 Files to Create/Modify (Alex)

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/SeqOpLayout.fs` | CREATE | Wrapper sequence layouts |
| `Alex/Witnesses/SeqWitness.fs` | MODIFY | Add map, filter, take, fold witnesses |

## 6. MLIR Output Specification

### 6.1 Seq.map Example

```mlir
// Source: Seq.map (fun x -> x * 2) numbers
// where numbers: seq { yield 1; yield 2; yield 3 }

// Inner seq type (from PRD-15)
!inner_seq = !llvm.struct<(i32, i32, ptr)>

// Mapper closure type (no captures - just code_ptr)
!mapper = !llvm.struct<(ptr)>

// MapSeq wrapper type
!map_seq = !llvm.struct<(i32, i32, ptr, !inner_seq, !mapper)>

// Map MoveNext function
llvm.func @map_double_moveNext(%self: !llvm.ptr) -> i1 {
    // Get inner seq pointer
    %inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr

    // Call inner MoveNext
    %inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr
    %inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^transform, ^done

^transform:
    // Get inner current
    %inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr
    %inner_val = llvm.load %inner_curr_ptr : i32

    // Apply mapper (x * 2)
    %two = arith.constant 2 : i32
    %result = arith.muli %inner_val, %two : i32

    // Store result
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    llvm.store %result, %curr_ptr : i32

    %true = arith.constant true
    llvm.return %true : i1

^done:
    %false = arith.constant false
    llvm.return %false : i1
}

// Creation: Seq.map (fun x -> x * 2) numbers
// 1. Create inner seq (numbers)
%inner_seq = ... // from PRD-15

// 2. Create mapper closure (just code_ptr for no-capture lambda)
%mapper_code = llvm.mlir.addressof @double_func : !llvm.ptr
%mapper_undef = llvm.mlir.undef : !mapper
%mapper = llvm.insertvalue %mapper_code, %mapper_undef[0] : !mapper

// 3. Create map wrapper
%zero = arith.constant 0 : i32
%undef = llvm.mlir.undef : !map_seq
%s0 = llvm.insertvalue %zero, %undef[0] : !map_seq
%moveNext_ptr = llvm.mlir.addressof @map_double_moveNext : !llvm.ptr
%s1 = llvm.insertvalue %moveNext_ptr, %s0[2] : !map_seq
%s2 = llvm.insertvalue %inner_seq, %s1[3] : !map_seq
%map_seq_val = llvm.insertvalue %mapper, %s2[4] : !map_seq
```

### 6.2 Seq.filter Example

```mlir
// Source: Seq.filter (fun x -> x % 2 = 0) numbers

!filter_seq = !llvm.struct<(i32, i32, ptr, !inner_seq, !predicate)>

llvm.func @filter_even_moveNext(%self: !llvm.ptr) -> i1 {
    llvm.br ^loop

^loop:
    %inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr
    %inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr
    %inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^check, ^done

^check:
    %inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr
    %val = llvm.load %inner_curr_ptr : i32

    // Check x % 2 == 0
    %two = arith.constant 2 : i32
    %rem = arith.remsi %val, %two : i32
    %zero = arith.constant 0 : i32
    %is_even = arith.cmpi eq, %rem, %zero : i32
    llvm.cond_br %is_even, ^yield, ^loop

^yield:
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    llvm.store %val, %curr_ptr : i32
    %true = arith.constant true
    llvm.return %true : i1

^done:
    %false = arith.constant false
    llvm.return %false : i1
}
```

### 6.3 Seq.fold Example

```mlir
// Source: Seq.fold (fun acc x -> acc + x) 0 numbers

llvm.func @fold_sum(%folder: !folder_closure, %initial: i32, %seq_val: !seq_type) -> i32 {
    // Allocate seq for mutation
    %one = arith.constant 1 : i64
    %seq_alloca = llvm.alloca %one x !seq_type : !llvm.ptr
    llvm.store %seq_val, %seq_alloca : !seq_type

    // Accumulator
    %acc_alloca = llvm.alloca %one x i32 : !llvm.ptr
    llvm.store %initial, %acc_alloca : i32

    llvm.br ^loop

^loop:
    %moveNext_ptr = llvm.getelementptr %seq_alloca[0, 2] : !llvm.ptr
    %moveNext = llvm.load %moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %moveNext(%seq_alloca) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^body, ^done

^body:
    %curr_ptr = llvm.getelementptr %seq_alloca[0, 1] : !llvm.ptr
    %elem = llvm.load %curr_ptr : i32
    %acc = llvm.load %acc_alloca : i32

    // acc + x
    %new_acc = arith.addi %acc, %elem : i32
    llvm.store %new_acc, %acc_alloca : i32
    llvm.br ^loop

^done:
    %result = llvm.load %acc_alloca : i32
    llvm.return %result : i32
}
```

## 7. Validation and Sample Coverage Requirements

> **CRITICAL**: The sample MUST exercise ALL feature variants. Incomplete samples lead to incomplete implementations.

### 7.1 Sample Structure Overview

Sample 16 (`16_SeqOperations`) must comprehensively test:

| Part | Feature Area | Coverage Goal |
|------|--------------|---------------|
| 1 | Source sequences | PRD-15 constructs as inputs |
| 2 | Seq.map | No-capture and with-capture variants |
| 3 | Seq.filter | No-capture and with-capture variants |
| 4 | Seq.take | Normal case and edge cases |
| 5 | Seq.fold | No-capture and with-capture variants |
| 6 | Composed pipelines (no captures) | Multiple operation chains |
| 7 | Manual comparison | Verify equivalence |
| **8** | **Closures with captures** | **CRITICAL: Tests flat closure model** |
| **9** | **Seq.collect (flatMap)** | **Nested sequence production** |
| **10** | **Composed pipelines with captures** | **Full integration test** |
| **11** | **Edge cases** | **Empty, single, boundary conditions** |
| **12** | **Deep composition** | **3+ operations chained** |

### 7.2 Part-by-Part Test Specifications

#### Part 1: Source Sequences (PRD-15 Foundation)
```fsharp
let range (start: int) (stop: int) = seq { ... }
let naturals (n: int) = range 1 n
```
**Purpose**: Establish PRD-15 sequences as inputs for transformation operations.

#### Part 2: Seq.map - Basic (No Captures)
```fsharp
let doubled = Seq.map (fun x -> x * 2) (naturals 5)      // Expected: 2 4 6 8 10
let squared = Seq.map (fun x -> x * x) (naturals 5)      // Expected: 1 4 9 16 25
let addTen = Seq.map (fun x -> x + 10) (naturals 5)      // Expected: 11 12 13 14 15
```
**Validates**: Basic transformation with inline computation.

#### Part 3: Seq.filter - Basic (No Captures)
```fsharp
let evens = Seq.filter (fun x -> x % 2 = 0) (naturals 10)       // Expected: 2 4 6 8 10
let odds = Seq.filter (fun x -> x % 2 = 1) (naturals 10)        // Expected: 1 3 5 7 9
let greaterThan5 = Seq.filter (fun x -> x > 5) (naturals 10)    // Expected: 6 7 8 9 10
```
**Validates**: Predicate filtering with inline conditions.

#### Part 4: Seq.take
```fsharp
let firstThree = Seq.take 3 (naturals 100)    // Expected: 1 2 3
let exactlyFive = Seq.take 5 (naturals 5)     // Expected: 1 2 3 4 5 (boundary)
let takeMoreThanAvailable = Seq.take 10 (naturals 3)  // Expected: 1 2 3 (graceful)
```
**Validates**: Count limiting with boundary conditions.

#### Part 5: Seq.fold - Basic (No Captures)
```fsharp
let sum = Seq.fold (fun acc x -> acc + x) 0 (naturals 10)           // Expected: 55
let product = Seq.fold (fun acc x -> acc * x) 1 (naturals 5)        // Expected: 120
let findMax = Seq.fold (fun acc x -> if x > acc then x else acc) 0 (naturals 10)  // Expected: 10
let countElements = Seq.fold (fun acc _ -> acc + 1) 0 (naturals 10) // Expected: 10
```
**Validates**: Eager consumption with various accumulation patterns.

#### Part 6: Composed Pipelines (No Captures)
```fsharp
let evensSquared = naturals 10 |> Seq.filter (fun x -> x % 2 = 0) |> Seq.map (fun x -> x * x)
// Expected: 4 16 36 64 100

let squaresOver10 = naturals 10 |> Seq.map (fun x -> x * x) |> Seq.filter (fun x -> x > 10)
// Expected: 16 25 36 49 64 81 100

let first3EvensDoubled = naturals 100 
    |> Seq.filter (fun x -> x % 2 = 0) 
    |> Seq.map (fun x -> x * 2) 
    |> Seq.take 3
// Expected: 4 8 12

let sumEvenSquares = naturals 10 
    |> Seq.filter (fun x -> x % 2 = 0) 
    |> Seq.map (fun x -> x * x) 
    |> Seq.fold (fun acc x -> acc + x) 0
// Expected: 220 (4+16+36+64+100)
```
**Validates**: Multiple operations compose correctly with nested structs.

#### Part 7: Manual Comparison
```fsharp
let manualSum (s: seq<int>) : int = ...
let manualCount (s: seq<int>) : int = ...
```
**Validates**: Seq.fold behaves equivalently to manual for-loop consumption.

#### Part 8: Closures with Captures (CRITICAL)

> **This part exercises the flat closure model. Without it, capture handling is untested.**

```fsharp
// === Seq.map with captured value ===
let scale (factor: int) (xs: seq<int>) = 
    Seq.map (fun x -> x * factor) xs  // 'factor' is captured

let scaledBy3 = scale 3 (naturals 5)
// Expected: 3 6 9 12 15

let scaledBy7 = scale 7 (naturals 3)
// Expected: 7 14 21

// === Seq.filter with captured value ===
let aboveThreshold (threshold: int) (xs: seq<int>) =
    Seq.filter (fun x -> x > threshold) xs  // 'threshold' is captured

let above5 = aboveThreshold 5 (naturals 10)
// Expected: 6 7 8 9 10

let above0 = aboveThreshold 0 (naturals 5)
// Expected: 1 2 3 4 5

// === Seq.fold with captured value ===
let sumWithOffset (offset: int) (xs: seq<int>) =
    Seq.fold (fun acc x -> acc + x + offset) 0 xs  // 'offset' captured in folder

let sumPlus10Each = sumWithOffset 10 (naturals 5)
// Expected: 65 (1+10 + 2+10 + 3+10 + 4+10 + 5+10 = 15 + 50)

// === Multiple captures ===
let rangeTransform (lo: int) (hi: int) (xs: seq<int>) =
    Seq.map (fun x -> x * lo + hi) xs  // Both 'lo' and 'hi' captured

let transformed = rangeTransform 2 100 (naturals 3)
// Expected: 102 104 106
```

**Validates**: 
- Flat closure struct includes capture fields
- Capture extraction at invocation time
- Multiple captures work correctly
- Captures don't interfere with inner seq state

#### Part 9: Seq.collect (flatMap)

```fsharp
// === Basic flatMap ===
let expandDouble = Seq.collect (fun x -> seq { yield x; yield x * 2 }) (naturals 3)
// Expected: 1 2 2 4 3 6

let expandTriple = Seq.collect (fun x -> seq { yield x; yield x; yield x }) (naturals 2)
// Expected: 1 1 1 2 2 2

// === flatMap with captured value ===
let expandWithFactor (factor: int) (xs: seq<int>) =
    Seq.collect (fun x -> seq { yield x; yield x * factor }) xs

let expandBy10 = expandWithFactor 10 (naturals 3)
// Expected: 1 10 2 20 3 30

// === flatMap producing variable-length sequences ===
let repeat (xs: seq<int>) =
    Seq.collect (fun x -> seq {
        let mutable i = 0
        while i < x do
            yield x
            i <- i + 1
    }) xs

let repeated = repeat (range 1 4)
// Expected: 1  2 2  3 3 3  4 4 4 4
```

**Validates**:
- Inner sequence production from mapper
- Nested iteration (outer advances only after inner exhausted)
- Captures in the mapper that produces sequences

#### Part 10: Composed Pipelines with Captures

```fsharp
// === Pipeline where each operation captures ===
let complexPipeline (threshold: int) (multiplier: int) (count: int) =
    naturals 20
    |> Seq.filter (fun x -> x > threshold)      // captures 'threshold'
    |> Seq.map (fun x -> x * multiplier)        // captures 'multiplier'
    |> Seq.take count                            // captures 'count'

let result1 = complexPipeline 5 2 5
// naturals 20 = 1..20
// filter > 5 = 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
// map * 2 = 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
// take 5 = 12 14 16 18 20
// Expected: 12 14 16 18 20

// === Fold at the end with captures throughout ===
let sumFilteredScaled (minVal: int) (factor: int) (xs: seq<int>) =
    xs
    |> Seq.filter (fun x -> x >= minVal)
    |> Seq.map (fun x -> x * factor)
    |> Seq.fold (fun acc x -> acc + x) 0

let total = sumFilteredScaled 3 10 (naturals 5)
// filter >= 3: 3 4 5
// map * 10: 30 40 50
// fold sum: 120
// Expected: 120
```

**Validates**:
- Each wrapper correctly captures its own closure
- Nested struct contains multiple closure instances
- Captures from different pipeline stages don't interfere

#### Part 11: Edge Cases

```fsharp
// === Empty source sequence ===
let emptySource = seq { if false then yield 0 }
let mapEmpty = Seq.map (fun x -> x * 2) emptySource
let filterEmpty = Seq.filter (fun x -> x > 0) emptySource
let foldEmpty = Seq.fold (fun acc x -> acc + x) 42 emptySource
// Expected: mapEmpty yields nothing, filterEmpty yields nothing, foldEmpty = 42

// === Single element ===
let singleElement = seq { yield 99 }
let mapSingle = Seq.map (fun x -> x + 1) singleElement
// Expected: 100

// === Filter removes all ===
let filterNone = Seq.filter (fun x -> x > 100) (naturals 10)
// Expected: (empty)

// === Take zero ===
let takeZero = Seq.take 0 (naturals 10)
// Expected: (empty)

// === Take from empty ===
let takeFromEmpty = Seq.take 5 emptySource
// Expected: (empty)
```

**Validates**: Boundary conditions handled gracefully without crashes.

#### Part 12: Deep Composition (3+ Operations)

```fsharp
// === Four operations chained ===
let deepPipeline1 =
    naturals 50
    |> Seq.filter (fun x -> x % 2 = 0)    // evens: 2 4 6 8 ... 50
    |> Seq.map (fun x -> x / 2)           // halved: 1 2 3 4 ... 25
    |> Seq.filter (fun x -> x % 3 = 0)    // div by 3: 3 6 9 12 15 18 21 24
    |> Seq.take 5
// Expected: 3 6 9 12 15

// === Five operations with fold ===
let deepPipelineWithFold =
    naturals 100
    |> Seq.filter (fun x -> x % 5 = 0)    // 5 10 15 20 ... 100 (20 elements)
    |> Seq.map (fun x -> x * 2)           // 10 20 30 40 ... 200
    |> Seq.filter (fun x -> x > 50)       // 60 70 80 ... 200 (15 elements)
    |> Seq.take 5                          // 60 70 80 90 100
    |> Seq.fold (fun acc x -> acc + x) 0
// Expected: 400 (60+70+80+90+100)
```

**Validates**: Struct nesting works correctly at depth 4-5.

### 7.3 Complete Expected Output Summary

```
=== Sample 16: Sequence Operations ===

--- Part 2: Seq.map ---
naturals 5: 1 2 3 4 5
doubled (x*2): 2 4 6 8 10
squared (x*x): 1 4 9 16 25
addTen (x+10): 11 12 13 14 15

--- Part 3: Seq.filter ---
evens from 1..10: 2 4 6 8 10
odds from 1..10: 1 3 5 7 9
greaterThan5 from 1..10: 6 7 8 9 10

--- Part 4: Seq.take ---
firstThree from 1..100: 1 2 3
exactlyFive from 1..5: 1 2 3 4 5
takeMoreThanAvailable: 1 2 3

--- Part 5: Seq.fold ---
sum of 1..10: 55
product of 1..5: 120
max of 1..10: 10
count of 1..10: 10

--- Part 6: Composed Operations ---
evensSquared: 4 16 36 64 100
squaresOver10: 16 25 36 49 64 81 100
first3EvensDoubled: 4 8 12
sumEvenSquares: 220

--- Part 7: Manual vs Seq.fold ---
manualSum 1..10: 55
manualCount 1..10: 10

--- Part 8: Closures with Captures ---
scaledBy3: 3 6 9 12 15
scaledBy7: 7 14 21
above5: 6 7 8 9 10
above0: 1 2 3 4 5
sumPlus10Each: 65
transformed (2*x+100): 102 104 106

--- Part 9: Seq.collect ---
expandDouble: 1 2 2 4 3 6
expandTriple: 1 1 1 2 2 2
expandBy10: 1 10 2 20 3 30
repeated: 1 2 2 3 3 3 4 4 4 4

--- Part 10: Composed with Captures ---
complexPipeline 5 2 5: 12 14 16 18 20
sumFilteredScaled 3 10: 120

--- Part 11: Edge Cases ---
mapEmpty: (done)
filterEmpty: (done)
foldEmpty: 42
mapSingle: 100
filterNone: (done)
takeZero: (done)
takeFromEmpty: (done)

--- Part 12: Deep Composition ---
deepPipeline1: 3 6 9 12 15
deepPipelineWithFold: 400
```

### 7.4 Validation Checklist

- [ ] **Parts 1-7**: All basic operations work with no-capture lambdas
- [ ] **Part 8**: Closure with captures works for map, filter, fold
- [ ] **Part 9**: Seq.collect (flatMap) works with nested iteration
- [ ] **Part 10**: Composed pipelines with captures don't interfere
- [ ] **Part 11**: All edge cases handled gracefully
- [ ] **Part 12**: Deep composition (4-5 operations) works correctly
- [ ] **Regression**: Samples 01-15 still pass

## 8. Implementation Checklist

### Phase 0: SSA Cost Infrastructure
- [ ] Implement `mapWrapperSSACost` function
- [ ] Implement `filterWrapperSSACost` function
- [ ] Implement `takeWrapperSSACost` function
- [ ] Implement `foldSetupSSACost` function
- [ ] Implement MoveNext SSA cost functions
- [ ] Verify SSA cost formulas match actual generation

### Phase 1: Seq.map
- [ ] Add Seq.map intrinsic to FNCS
- [ ] Create MapSeqLayout coeffect (with SSA cost)
- [ ] Implement MapSeq wrapper creation with copy semantics
- [ ] Implement MapSeq MoveNext generation
- [ ] Implement mapper closure invocation with explicit capture extraction
- [ ] Test: map doubles values

### Phase 2: Seq.filter
- [ ] Add Seq.filter intrinsic
- [ ] Create FilterSeqLayout coeffect (with SSA cost)
- [ ] Implement FilterSeq wrapper creation with copy semantics
- [ ] Implement FilterSeq MoveNext generation with loop
- [ ] Implement predicate closure invocation with capture extraction
- [ ] Test: filter for evens

### Phase 3: Seq.take
- [ ] Add Seq.take intrinsic
- [ ] Create TakeSeqLayout coeffect (with SSA cost)
- [ ] Implement TakeSeq wrapper creation (includes remaining counter)
- [ ] Implement TakeSeq MoveNext generation
- [ ] Test: take limits sequence

### Phase 4: Seq.fold
- [ ] Add Seq.fold intrinsic
- [ ] Implement fold as eager consumer (not a wrapper)
- [ ] Implement folder closure invocation with capture extraction
- [ ] Test: fold sums sequence

### Validation
- [ ] Sample 16 compiles without errors
- [ ] Sample 16 produces correct output
- [ ] Composed pipelines work (e.g., `take (map (filter xs))`)
- [ ] Copy semantics verified (independent iteration)
- [ ] SSA counts match cost formulas
- [ ] Samples 01-15 still pass

## 9. Related PRDs

- **PRD-11**: Closures - Mapper/predicate are flat closures
- **PRD-12**: HOFs - Seq operations are HOFs
- **PRD-14**: Lazy - Foundation for deferred computation
- **PRD-15**: SimpleSeq - Foundation for sequences

## 10. Architectural Alignment

This PRD aligns with the flat closure architecture:

**Key principles maintained:**
1. **No nulls** - Every field initialized
2. **No env_ptr** - Closures and inner sequences are flat
3. **Self-contained structs** - Wrappers inline their dependencies
4. **Coeffect-based layout** - Wrapper layouts computed before witnessing
5. **Composition = nesting** - Composed sequences are nested structs

**Trade-off acknowledged:**
- Struct size grows with composition depth
- For typical depths (3-5 operations), this is acceptable
- Very deep pipelines may need alternative strategies (future optimization)

> **Critical:** See Serena memory `compose_from_standing_art_principle` for why composing from PRD-11/14/15 patterns is essential. New features MUST extend standing art, not reinvent.
