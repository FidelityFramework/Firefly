# PRD-15: Sequence Operations

> **Sample**: `15_SeqOperations` | **Status**: Planned | **Depends On**: PRD-14 (SimpleSeq)

## 1. Executive Summary

This PRD covers the core sequence operations: `Seq.map`, `Seq.filter`, `Seq.take`, `Seq.fold`, etc. These are higher-order functions over sequences - they transform or consume lazy enumerations.

**Key Insight**: Seq operations are **composed iterators**. `Seq.map f xs` doesn't create a new sequence struct - it wraps the original with a transformation. Each operation is another state machine layer.

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

Eager reduction to single value.

### 2.5 Seq.collect (flatMap)

```fsharp
let flattened = Seq.collect (fun x -> seq { yield x; yield x * 10 }) numbers
```

Maps then flattens.

## 3. FNCS Layer Implementation

### 3.1 Seq Module Intrinsics

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/CheckExpressions.fs`

```fsharp
// Seq intrinsic module
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

### 3.2 SemanticKind for Seq Operations

Option 1: Use existing `Intrinsic` kind with Seq operation tag:
```fsharp
| Intrinsic of op: IntrinsicOp * args: NodeId list

type IntrinsicOp =
    | SeqMap | SeqFilter | SeqTake | SeqFold | SeqCollect | ...
```

Option 2: Specific Seq kinds:
```fsharp
| SeqMap of mapper: NodeId * source: NodeId
| SeqFilter of predicate: NodeId * source: NodeId
| SeqTake of count: NodeId * source: NodeId
| SeqFold of folder: NodeId * initial: NodeId * source: NodeId
```

**Recommendation**: Option 1 is more extensible.

## 4. Firefly/Alex Layer Implementation

### 4.1 Seq.map State Machine

`Seq.map` wraps an inner sequence and transforms its output:

```fsharp
type MapSeq<'a, 'b> = {
    State: int
    Current: 'b
    Inner: seq<'a>          // The wrapped sequence
    Mapper: 'a -> 'b        // The transformation closure
}

let mapMoveNext (self: MapSeq) =
    if innerMoveNext self.Inner then
        self.Current <- self.Mapper self.Inner.Current
        true
    else
        false
```

### 4.2 Seq.filter State Machine

`Seq.filter` skips elements that don't match:

```fsharp
type FilterSeq<'a> = {
    State: int
    Current: 'a
    Inner: seq<'a>
    Predicate: 'a -> bool
}

let filterMoveNext (self: FilterSeq) =
    while innerMoveNext self.Inner do
        let x = self.Inner.Current
        if self.Predicate x then
            self.Current <- x
            return true
    false
```

### 4.3 Seq.take State Machine

`Seq.take` counts elements:

```fsharp
type TakeSeq<'a> = {
    State: int
    Current: 'a
    Inner: seq<'a>
    Remaining: int
}

let takeMoveNext (self: TakeSeq) =
    if self.Remaining > 0 && innerMoveNext self.Inner then
        self.Current <- self.Inner.Current
        self.Remaining <- self.Remaining - 1
        true
    else
        false
```

### 4.4 Seq.fold (Eager Consumer)

`Seq.fold` consumes the entire sequence:

```fsharp
let seqFold folder initial source =
    let mutable acc = initial
    while moveNext source do
        acc <- folder acc source.Current
    acc
```

## 5. MLIR Output Specification

### 5.1 Map Sequence Struct

```mlir
// Seq.map (fun x -> x * 2) numbers
!map_seq = !llvm.struct<(
    i32,                    // state
    i32,                    // current (output type)
    !llvm.ptr,              // inner sequence pointer
    !closure_type           // mapper closure
)>
```

### 5.2 Map MoveNext

```mlir
llvm.func @map_moveNext(%self: !llvm.ptr) -> i1 {
    // Get inner sequence
    %inner_ptr = llvm.getelementptr %self[0, 2]
    %inner = llvm.load %inner_ptr : !llvm.ptr

    // Move inner
    %has_next = llvm.call @inner_moveNext(%inner) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^transform, ^done

^transform:
    // Get inner current
    %inner_curr_ptr = llvm.getelementptr %inner[0, 1]
    %inner_val = llvm.load %inner_curr_ptr : i32

    // Apply mapper
    %mapper_ptr = llvm.getelementptr %self[0, 3]
    %mapper = llvm.load %mapper_ptr : !closure_type
    %code = llvm.extractvalue %mapper[0]
    %env = llvm.extractvalue %mapper[1]
    %result = llvm.call %code(%env, %inner_val) : (!llvm.ptr, i32) -> i32

    // Store result
    %curr_ptr = llvm.getelementptr %self[0, 1]
    llvm.store %result, %curr_ptr

    llvm.return %true : i1

^done:
    llvm.return %false : i1
}
```

### 5.3 Fold Implementation

```mlir
// Seq.fold (fun acc x -> acc + x) 0 numbers
llvm.func @seq_fold_add(%folder: !closure_type, %initial: i32, %seq: !llvm.ptr) -> i32 {
    %acc = llvm.alloca 1 x i32
    llvm.store %initial, %acc
    llvm.br ^loop

^loop:
    %has_next = llvm.call @moveNext(%seq) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^body, ^done

^body:
    %curr_ptr = llvm.getelementptr %seq[0, 1]
    %x = llvm.load %curr_ptr : i32
    %acc_val = llvm.load %acc : i32

    // Apply folder
    %code = llvm.extractvalue %folder[0]
    %env = llvm.extractvalue %folder[1]
    %new_acc = llvm.call %code(%env, %acc_val, %x) : (!llvm.ptr, i32, i32) -> i32

    llvm.store %new_acc, %acc
    llvm.br ^loop

^done:
    %result = llvm.load %acc : i32
    llvm.return %result : i32
}
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module SeqOperationsSample

let numbers = seq { for i in 1..10 do yield i }

[<EntryPoint>]
let main _ =
    Console.writeln "=== Seq Operations Test ==="

    Console.writeln "--- Seq.map ---"
    let doubled = Seq.map (fun x -> x * 2) numbers
    for x in Seq.take 5 doubled do
        Console.writeln (Format.int x)

    Console.writeln "--- Seq.filter ---"
    let evens = Seq.filter (fun x -> x % 2 = 0) numbers
    for x in evens do
        Console.writeln (Format.int x)

    Console.writeln "--- Seq.fold ---"
    let sum = Seq.fold (fun acc x -> acc + x) 0 numbers
    Console.write "Sum: "
    Console.writeln (Format.int sum)

    0
```

### 6.2 Expected Output

```
=== Seq Operations Test ===
--- Seq.map ---
2
4
6
8
10
--- Seq.filter ---
2
4
6
8
10
--- Seq.fold ---
Sum: 55
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Seq.map, Seq.filter, etc. intrinsics |
| `NativeGlobals.fs` | MODIFY | Seq module registration |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/SeqWitness.fs` | MODIFY | Add map, filter, take, fold witnesses |
| `src/Alex/CodeGeneration/SeqStateTypes.fs` | CREATE | Define wrapper sequence struct types |

## 8. Implementation Checklist

### Phase 1: Seq.map
- [ ] Add Seq.map intrinsic to FNCS
- [ ] Implement MapSeq state machine in Alex
- [ ] Test: map doubles values

### Phase 2: Seq.filter
- [ ] Add Seq.filter intrinsic
- [ ] Implement FilterSeq state machine
- [ ] Test: filter for evens

### Phase 3: Seq.take
- [ ] Add Seq.take intrinsic
- [ ] Implement TakeSeq state machine
- [ ] Test: take limits infinite sequence

### Phase 4: Seq.fold
- [ ] Add Seq.fold intrinsic
- [ ] Implement fold as eager consumer
- [ ] Test: fold sums sequence

### Validation
- [ ] Sample 15 compiles without errors
- [ ] Sample 15 produces correct output
- [ ] Samples 01-14 still pass

## 9. Related PRDs

- **PRD-14**: SimpleSeq - Foundation for sequences
- **PRD-12**: HOFs - Seq operations are HOFs
- **PRD-11**: Closures - Mapper/predicate are closures
