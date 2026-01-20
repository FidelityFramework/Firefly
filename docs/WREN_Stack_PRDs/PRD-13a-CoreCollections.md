# PRD-13a: Core Collections and Range Expressions

> **Sample**: `13a_Collections` | **Status**: Planned | **Depends On**: PRD-11 (Closures), PRD-13 (Recursion)

**Foundation for Eager Collections and Idiomatic F# Syntax**: This PRD establishes the core collection types and range expressions that BAREWire and most F# programs require. Unlike Seq (lazy, pull-based), these are eager, fully-materialized data structures.

## 1. Executive Summary

F# programs fundamentally rely on:

**Core Collection Types:**
- **List<'T>** - Immutable singly-linked list (F#'s workhorse collection)
- **Map<'K, 'V>** - Immutable key-value dictionary
- **Set<'T>** - Immutable set of unique values

**Range Expressions** (extremely common F# syntax):
- `[1..10]` - List from 1 to 10
- `[1..2..10]` - Stepped list: 1, 3, 5, 7, 9
- `[|1..10|]` - Array range
- `seq { 1..10 }` - Lazy sequence range

These are NOT lazy sequences - collections are fully materialized in memory. Range expressions provide idiomatic initialization syntax.

### 1.1 Why This PRD?

**BAREWire Blockers**: The BAREWire serialization library requires:
- `Map.tryFind` (4 occurrences)
- `Map.values` (2 occurrences)
- `Set.empty`, `Set.contains`, `Set.add` (3 occurrences)
- `Option.map` (1 occurrence)

**Conspicuous Absence**: Range expressions like `[1..10]` are among the most common F# patterns, yet currently error with "requires slice support" in FNCS.

**Glaring Omission**: The PRD roadmap (11-31) covers closures, lazy, seq, async, regions, networking, desktop - but NO PRD covers the fundamental eager collection types that virtually every F# program uses.

### 1.2 Design Philosophy

Following FNCS principles:
1. **Monomorphization** - Collections are specialized per element type (no uniform representation)
2. **No Boxing** - Element types are stored directly, not as `obj`
3. **Arena-Friendly** - Collections allocate from regions when available (PRD-20+)
4. **Structural Sharing** - Immutable operations share unchanged substructure

## 2. Type Definitions

### 2.1 List<'T>

F# lists are immutable singly-linked lists with structural sharing.

```
// Cons cell layout
ListCell<T> = {
    head: T
    tail: ptr<ListCell<T>>  // null = end of list
}

// List is a pointer to first cell (or null for empty)
List<T> = ptr<ListCell<T>>
```

**Note**: Unlike .NET's two-pointer FSharpList (for IEnumerable), native lists are single-pointer cons cells. This is the classic ML/Lisp representation.

### 2.2 Map<'K, 'V>

Maps are immutable balanced binary search trees (AVL or Red-Black).

```
// Map node layout
MapNode<K, V> = {
    key: K
    value: V
    left: ptr<MapNode<K, V>>   // null = no left child
    right: ptr<MapNode<K, V>>  // null = no right child
    height: i32                 // For AVL balancing
}

// Map is a pointer to root (or null for empty)
Map<K, V> = ptr<MapNode<K, V>>
```

**Key Constraint**: Keys must support comparison (`IComparable` in BCL terms). In FNCS, this is expressed via SRTP: `'K when 'K : comparison`.

### 2.3 Set<'T>

Sets are immutable balanced binary search trees (same structure as Map without values).

```
// Set node layout
SetNode<T> = {
    value: T
    left: ptr<SetNode<T>>
    right: ptr<SetNode<T>>
    height: i32
}

// Set is a pointer to root (or null for empty)
Set<T> = ptr<SetNode<T>>
```

### 2.4 NTUKind Extensions

```fsharp
type NTUKind =
    // ... existing kinds ...
    | NTUlist      // List<'T>
    | NTUmap       // Map<'K, 'V>
    | NTUset       // Set<'T>
```

### 2.5 NativeType Extensions

```fsharp
type NativeType =
    // ... existing types ...
    | TList of elementType: NativeType
    | TMap of keyType: NativeType * valueType: NativeType
    | TSet of elementType: NativeType
```

### 2.6 Range Expressions

Range expressions are idiomatic F# syntax for generating sequences of values. They desugar to collection initialization.

#### 2.6.1 Syntax Forms

| Syntax | Meaning | Desugars To |
|--------|---------|-------------|
| `[1..10]` | List 1 to 10 inclusive | `List.ofSeq (seq { 1..10 })` or loop |
| `[1..2..10]` | List 1,3,5,7,9 (step 2) | `List.ofSeq (seq { 1..2..10 })` |
| `[10..-1..1]` | Countdown 10,9,8,...,1 | `List.ofSeq (seq { 10..-1..1 })` |
| `[|1..10|]` | Array 1 to 10 | `Array.init 10 (fun i -> i + 1)` |
| `[|1..2..10|]` | Array with step | Loop-based initialization |
| `seq { 1..10 }` | Lazy sequence | State machine (PRD-15) |
| `{ 1..10 }` | Sequence (implicit) | Same as `seq { 1..10 }` |

#### 2.6.2 FCS Representation

FCS parses range expressions as `SynExpr.IndexRange`:

```fsharp
// From FCS SyntaxTree.fs
| IndexRange of
    expr1: SynExpr option *    // Start (None = unbounded)
    opm: range *               // Range of .. operator
    expr2: SynExpr option *    // End (None = unbounded)
    range1: range *
    range2: range *
    range: range

// Stepped ranges use nested IndexRange or special handling
```

**Current FNCS Status**: `IndexRange` currently errors with "requires slice support". This PRD implements proper range expression handling.

#### 2.6.3 FNCS Implementation

**SemanticKind Extension:**

```fsharp
type SemanticKind =
    // ... existing kinds ...

    /// Range expression: start..finish or start..step..finish
    /// Produces a sequence/list/array depending on context
    | RangeExpr of
        start: NodeId *
        finish: NodeId *
        step: NodeId option *
        targetKind: RangeTargetKind

/// What collection type the range produces
type RangeTargetKind =
    | RangeToList    // [1..10]
    | RangeToArray   // [|1..10|]
    | RangeToSeq     // seq { 1..10 }
```

**Type Checking:**

```fsharp
/// Check a range expression
let checkRangeExpr
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (startOpt: SynExpr option)
    (finishOpt: SynExpr option)
    (stepOpt: SynExpr option)
    (targetKind: RangeTargetKind)
    (range: SourceRange)
    : SemanticNode =

    // Start and finish must be same numeric type
    let startNode = startOpt |> Option.map (checkExpr env builder)
    let finishNode = finishOpt |> Option.map (checkExpr env builder)
    let stepNode = stepOpt |> Option.map (checkExpr env builder)

    // Infer element type (int by default, or from expressions)
    let elemType =
        match startNode, finishNode with
        | Some s, _ -> s.Type
        | _, Some f -> f.Type
        | None, None -> env.Globals.IntType

    // Unify all range bounds to same type
    [startNode; finishNode; stepNode]
    |> List.choose id
    |> List.iter (fun n ->
        addConstraint (Constraint.Equals(n.Type, elemType, range)) env)

    // Result type depends on target
    let resultType =
        match targetKind with
        | RangeToList -> mkListType elemType
        | RangeToArray -> mkArrayType elemType
        | RangeToSeq -> mkSeqType elemType

    let childIds = [startNode; finishNode; stepNode] |> List.choose (Option.map (fun n -> n.Id))
    builder.Create(
        SemanticKind.RangeExpr(
            startNode |> Option.map (fun n -> n.Id) |> Option.defaultValue NodeId.Empty,
            finishNode |> Option.map (fun n -> n.Id) |> Option.defaultValue NodeId.Empty,
            stepNode |> Option.map (fun n -> n.Id),
            targetKind),
        resultType,
        range,
        children = childIds)
```

#### 2.6.4 Alex Code Generation

**List Range** `[1..10]`:
```mlir
// Eager: allocate and fill list in reverse, then reverse
// Or: generate as seq, then Seq.toList
llvm.func @range_to_list(%start: i32, %finish: i32) -> !list_int {
    // Simple approach: loop building cons cells
    %result = ... // List.empty
    %i = %finish
    loop:
        %cell = call @list_cons(%i, %result)
        %i_next = arith.subi %i, 1
        %done = arith.cmpi slt, %i_next, %start
        cond_br %done, ^exit, ^loop
    exit:
        return %result
}
```

**Array Range** `[|1..10|]`:
```mlir
// Efficient: calculate size, allocate, fill
llvm.func @range_to_array(%start: i32, %finish: i32) -> !array_int {
    %size = arith.subi %finish, %start
    %size_plus_1 = arith.addi %size, 1
    %arr = call @array_zeroCreate(%size_plus_1)
    %i = 0
    %val = %start
    loop:
        call @array_set(%arr, %i, %val)
        %i_next = arith.addi %i, 1
        %val_next = arith.addi %val, 1
        %done = arith.cmpi sge, %i_next, %size_plus_1
        cond_br %done, ^exit, ^loop
    exit:
        return %arr
}
```

**Stepped Range** `[1..2..10]`:
```mlir
// With step parameter
llvm.func @range_stepped_to_array(%start: i32, %step: i32, %finish: i32) -> !array_int {
    // Calculate size: ((finish - start) / step) + 1
    %diff = arith.subi %finish, %start
    %count = arith.divsi %diff, %step
    %size = arith.addi %count, 1
    %arr = call @array_zeroCreate(%size)
    // Fill with stepped values
    ...
}
```

#### 2.6.5 Intrinsics for Range Support

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `Range.toList` | `int -> int -> List<int>` | `[start..finish]` |
| `Range.toListStep` | `int -> int -> int -> List<int>` | `[start..step..finish]` |
| `Range.toArray` | `int -> int -> int[]` | `[|start..finish|]` |
| `Range.toArrayStep` | `int -> int -> int -> int[]` | `[|start..step..finish|]` |
| `Range.toSeq` | `int -> int -> seq<int>` | `seq { start..finish }` |
| `Range.toSeqStep` | `int -> int -> int -> seq<int>` | `seq { start..step..finish }` |

**Note**: These intrinsics are generic over numeric types supporting `(+)` and comparison. In practice, `int` is most common.

## 3. FNCS Intrinsics

### 3.1 List Intrinsics

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `List.empty` | `List<'T>` | Empty list (null pointer) |
| `List.isEmpty` | `List<'T> -> bool` | Check if list is empty |
| `List.head` | `List<'T> -> 'T` | Get first element (fails on empty) |
| `List.tail` | `List<'T> -> List<'T>` | Get rest of list (fails on empty) |
| `List.cons` | `'T -> List<'T> -> List<'T>` | Prepend element |
| `List.length` | `List<'T> -> int` | Count elements |
| `List.rev` | `List<'T> -> List<'T>` | Reverse list |
| `List.append` | `List<'T> -> List<'T> -> List<'T>` | Concatenate lists |
| `List.map` | `('T -> 'U) -> List<'T> -> List<'U>` | Transform elements |
| `List.filter` | `('T -> bool) -> List<'T> -> List<'T>` | Keep matching elements |
| `List.fold` | `('S -> 'T -> 'S) -> 'S -> List<'T> -> 'S` | Left fold |
| `List.foldBack` | `('T -> 'S -> 'S) -> List<'T> -> 'S -> 'S` | Right fold |
| `List.tryHead` | `List<'T> -> Option<'T>` | Safe head |
| `List.tryFind` | `('T -> bool) -> List<'T> -> Option<'T>` | Find first match |
| `List.forall` | `('T -> bool) -> List<'T> -> bool` | All elements match |
| `List.exists` | `('T -> bool) -> List<'T> -> bool` | Any element matches |

**F# List Syntax Sugar**:
- `[1; 2; 3]` desugars to `List.cons 1 (List.cons 2 (List.cons 3 List.empty))`
- `x :: xs` desugars to `List.cons x xs`

### 3.2 Map Intrinsics

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `Map.empty` | `Map<'K, 'V>` | Empty map (null pointer) |
| `Map.isEmpty` | `Map<'K, 'V> -> bool` | Check if map is empty |
| `Map.add` | `'K -> 'V -> Map<'K, 'V> -> Map<'K, 'V>` | Add/replace key-value |
| `Map.remove` | `'K -> Map<'K, 'V> -> Map<'K, 'V>` | Remove key |
| `Map.tryFind` | `'K -> Map<'K, 'V> -> Option<'V>` | Lookup by key |
| `Map.find` | `'K -> Map<'K, 'V> -> 'V` | Lookup (fails if missing) |
| `Map.containsKey` | `'K -> Map<'K, 'V> -> bool` | Check key exists |
| `Map.count` | `Map<'K, 'V> -> int` | Number of entries |
| `Map.keys` | `Map<'K, 'V> -> List<'K>` | All keys as list |
| `Map.values` | `Map<'K, 'V> -> List<'V>` | All values as list |
| `Map.toList` | `Map<'K, 'V> -> List<'K * 'V>` | Convert to key-value pairs |
| `Map.ofList` | `List<'K * 'V> -> Map<'K, 'V>` | Create from key-value pairs |
| `Map.map` | `('K -> 'V -> 'U) -> Map<'K, 'V> -> Map<'K, 'U>` | Transform values |
| `Map.filter` | `('K -> 'V -> bool) -> Map<'K, 'V> -> Map<'K, 'V>` | Keep matching entries |
| `Map.fold` | `('S -> 'K -> 'V -> 'S) -> 'S -> Map<'K, 'V> -> 'S` | Fold over entries |

### 3.3 Set Intrinsics

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `Set.empty` | `Set<'T>` | Empty set (null pointer) |
| `Set.isEmpty` | `Set<'T> -> bool` | Check if set is empty |
| `Set.add` | `'T -> Set<'T> -> Set<'T>` | Add element |
| `Set.remove` | `'T -> Set<'T> -> Set<'T>` | Remove element |
| `Set.contains` | `'T -> Set<'T> -> bool` | Check membership |
| `Set.count` | `Set<'T> -> int` | Number of elements |
| `Set.union` | `Set<'T> -> Set<'T> -> Set<'T>` | Set union |
| `Set.intersect` | `Set<'T> -> Set<'T> -> Set<'T>` | Set intersection |
| `Set.difference` | `Set<'T> -> Set<'T> -> Set<'T>` | Set difference |
| `Set.isSubset` | `Set<'T> -> Set<'T> -> bool` | Subset test |
| `Set.toList` | `Set<'T> -> List<'T>` | Convert to list |
| `Set.ofList` | `List<'T> -> Set<'T>` | Create from list |
| `Set.map` | `('T -> 'U) -> Set<'T> -> Set<'U>` | Transform elements |
| `Set.filter` | `('T -> bool) -> Set<'T> -> Set<'T>` | Keep matching elements |
| `Set.fold` | `('S -> 'T -> 'S) -> 'S -> Set<'T> -> 'S` | Fold over elements |

### 3.4 Option Intrinsics (Enhancement)

Option already exists but needs these operations:

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `Option.map` | `('T -> 'U) -> Option<'T> -> Option<'U>` | Transform if Some |
| `Option.bind` | `('T -> Option<'U>) -> Option<'T> -> Option<'U>` | Flatmap |
| `Option.defaultValue` | `'T -> Option<'T> -> 'T` | Get with default |
| `Option.defaultWith` | `(unit -> 'T) -> Option<'T> -> 'T` | Get with lazy default |
| `Option.isSome` | `Option<'T> -> bool` | Check if Some |
| `Option.isNone` | `Option<'T> -> bool` | Check if None |
| `Option.get` | `Option<'T> -> 'T` | Unwrap (fails on None) |
| `Option.toList` | `Option<'T> -> List<'T>` | Convert to 0-or-1 element list |

### 3.5 Additional Small Intrinsics (BAREWire Needs)

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `max` | `'T -> 'T -> 'T` | Maximum (comparison) |
| `min` | `'T -> 'T -> 'T` | Minimum (comparison) |
| `fst` | `'T * 'U -> 'T` | First of tuple |
| `snd` | `'T * 'U -> 'U` | Second of tuple |
| `Array.blit` | `'T[] -> int -> 'T[] -> int -> int -> unit` | Copy array segment |
| `String.concat` | `string -> List<string> -> string` | Join strings with separator |

## 4. Implementation Strategy

### 4.1 Phase 1: Minimal Viable (BAREWire Unblocking)

Focus on the specific operations BAREWire needs:

**Priority 1 - BAREWire Blockers:**
- `Map.tryFind`
- `Map.values`
- `Set.empty`, `Set.contains`, `Set.add`
- `Option.map`
- `max`, `snd`
- `Array.blit`
- `String.concat`

This minimal set unblocks BAREWire compilation.

### 4.2 Phase 2: List Foundation

Implement List intrinsics:
- Core: `empty`, `isEmpty`, `head`, `tail`, `cons`, `length`
- Transforms: `map`, `filter`, `fold`, `rev`, `append`
- Queries: `tryHead`, `tryFind`, `forall`, `exists`

### 4.3 Phase 3: Map/Set Complete

Implement remaining Map and Set operations:
- Tree operations: `add`, `remove`, rebalancing
- Bulk operations: `ofList`, `toList`
- Set operations: `union`, `intersect`, `difference`

### 4.4 Phase 4: Integration with Regions (PRD-20+)

Once regions are available:
- Collection nodes allocate from current region
- Structural sharing works across region boundaries
- Deterministic cleanup when region closes

## 5. Alex Layer Implementation

### 5.1 List Operations

**List.cons** (most common operation):
```mlir
// Allocate cons cell
%cell_ptr = llvm.call @arena_alloc(%arena, %cell_size) : (!llvm.ptr, i64) -> !llvm.ptr

// Store head
%head_ptr = llvm.getelementptr %cell_ptr[0, 0] : !llvm.ptr
llvm.store %new_head, %head_ptr : !element_type

// Store tail
%tail_ptr = llvm.getelementptr %cell_ptr[0, 1] : !llvm.ptr
llvm.store %existing_list, %tail_ptr : !llvm.ptr

// Result is pointer to new cell
```

**List.head**:
```mlir
// Check for null (empty list) - debug builds only
// Extract head from cons cell
%head_ptr = llvm.getelementptr %list[0, 0] : !llvm.ptr
%head = llvm.load %head_ptr : !element_type
```

### 5.2 Map Operations (AVL Tree)

**Map.tryFind**:
```mlir
llvm.func @map_tryFind(%map: !llvm.ptr, %key: !key_type) -> !option_value_type {
    // Binary search down tree
    %current = %map
    llvm.br ^loop

^loop:
    %is_null = llvm.icmp eq %current, %null : !llvm.ptr
    llvm.cond_br %is_null, ^not_found, ^check

^check:
    %node_key_ptr = llvm.getelementptr %current[0, 0] : !llvm.ptr
    %node_key = llvm.load %node_key_ptr : !key_type
    %cmp = call @compare(%key, %node_key) : (!key_type, !key_type) -> i32
    %is_less = arith.cmpi slt, %cmp, %zero : i32
    %is_greater = arith.cmpi sgt, %cmp, %zero : i32
    llvm.cond_br %is_less, ^go_left, ^check_equal

^check_equal:
    llvm.cond_br %is_greater, ^go_right, ^found

^go_left:
    %left_ptr = llvm.getelementptr %current[0, 2] : !llvm.ptr
    %left = llvm.load %left_ptr : !llvm.ptr
    %current = %left
    llvm.br ^loop

^go_right:
    %right_ptr = llvm.getelementptr %current[0, 3] : !llvm.ptr
    %right = llvm.load %right_ptr : !llvm.ptr
    %current = %right
    llvm.br ^loop

^found:
    %value_ptr = llvm.getelementptr %current[0, 1] : !llvm.ptr
    %value = llvm.load %value_ptr : !value_type
    %some = ... // Construct Some(value)
    llvm.return %some : !option_value_type

^not_found:
    %none = ... // Construct None
    llvm.return %none : !option_value_type
}
```

### 5.3 Comparison Constraint

Map and Set require comparable keys/elements. In FNCS, this is expressed via SRTP:

```fsharp
// Map.add requires comparison constraint on key type
let add<'K, 'V when 'K : comparison> (key: 'K) (value: 'V) (map: Map<'K, 'V>) : Map<'K, 'V> = ...
```

Alex resolves SRTP to concrete comparison implementations:
- `int`, `int64`, etc. → built-in comparison ops
- `string` → lexicographic comparison
- Custom types → require explicit `compare` function

## 6. Memory Management

### 6.1 Pre-Region (Stack/Global Arena)

Before PRD-20 regions:
- Collections allocate from a global arena
- Arena cleared on program exit
- No early deallocation (acceptable for short-lived programs)

### 6.2 Post-Region (PRD-20+)

With regions:
- Collections allocate from the enclosing region
- Region close deallocates all collection nodes
- Structural sharing respects region lifetimes

### 6.3 Structural Sharing

Immutable operations share unchanged structure:

```fsharp
let xs = [1; 2; 3]      // Allocates 3 cells
let ys = 0 :: xs        // Allocates 1 cell, shares xs
let zs = List.tail xs   // No allocation! Just pointer to second cell
```

```
xs: [1] -> [2] -> [3] -> null
     ^
ys: [0] -+

zs:      [2] -> [3] -> null  (same as xs.tail)
```

## 7. Files to Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `NTUlist`, `NTUmap`, `NTUset`, `TList`, `TMap`, `TSet` |
| `Intrinsics.fs` | MODIFY | Add List.*, Map.*, Set.*, Option.* intrinsics |
| `Unify.fs` | MODIFY | Add bridge cases for TList, TMap, TSet |

### 7.2 Alex

| File | Action | Purpose |
|------|--------|---------|
| `Witnesses/ListWitness.fs` | CREATE | List operation MLIR generation |
| `Witnesses/MapWitness.fs` | CREATE | Map operation MLIR generation |
| `Witnesses/SetWitness.fs` | CREATE | Set operation MLIR generation |
| `FNCSTransfer.fs` | MODIFY | Handle collection intrinsics |

## 8. Validation

### 8.1 Sample Code

```fsharp
module CollectionsSample

[<EntryPoint>]
let main _ =
    Console.writeln "=== Core Collections Test ==="

    // Range expressions - the idiomatic F# way
    Console.writeln "--- Range Expressions ---"

    // Array range
    let arr = [|1..5|]
    Console.write "Array [|1..5|]: "
    for x in arr do
        Console.write (Format.int x + " ")
    Console.writeln ""

    // List range
    let lst = [1..5]
    Console.write "List [1..5]: "
    for x in lst do
        Console.write (Format.int x + " ")
    Console.writeln ""

    // Stepped range
    let odds = [|1..2..10|]
    Console.write "Odds [|1..2..10|]: "
    for x in odds do
        Console.write (Format.int x + " ")
    Console.writeln ""

    // Countdown
    let countdown = [5..-1..1]
    Console.write "Countdown [5..-1..1]: "
    for x in countdown do
        Console.write (Format.int x + " ")
    Console.writeln ""

    // List operations
    Console.writeln "--- List ---"
    let xs = [1; 2; 3]
    let ys = 0 :: xs
    Console.write "ys length: "
    Console.writeln (Format.int (List.length ys))

    for x in xs do
        Console.writeln (Format.int x)

    // Map operations
    Console.writeln "--- Map ---"
    let m = Map.empty
            |> Map.add "one" 1
            |> Map.add "two" 2
            |> Map.add "three" 3

    match Map.tryFind "two" m with
    | Some v -> Console.writeln ("Found: " + Format.int v)
    | None -> Console.writeln "Not found"

    // Set operations
    Console.writeln "--- Set ---"
    let s = Set.empty
            |> Set.add 1
            |> Set.add 2
            |> Set.add 3

    if Set.contains 2 s then
        Console.writeln "Set contains 2"

    0
```

### 8.2 Expected Output

```
=== Core Collections Test ===
--- Range Expressions ---
Array [|1..5|]: 1 2 3 4 5
List [1..5]: 1 2 3 4 5
Odds [|1..2..10|]: 1 3 5 7 9
Countdown [5..-1..1]: 5 4 3 2 1
--- List ---
ys length: 4
1
2
3
--- Map ---
Found: 2
--- Set ---
Set contains 2
```

## 9. Implementation Checklist

### Phase 1: BAREWire Unblocking (Priority)
- [ ] Add `TMap`, `TSet` to NativeTypes
- [ ] Add `Map.tryFind` intrinsic
- [ ] Add `Map.values` intrinsic
- [ ] Add `Set.empty`, `Set.contains`, `Set.add` intrinsics
- [ ] Add `Option.map` intrinsic
- [ ] Add `max`, `snd` intrinsics
- [ ] Add `Array.blit` intrinsic
- [ ] Add `String.concat` intrinsic
- [ ] BAREWire compiles

### Phase 2: Range Expressions (High Visibility)
- [ ] Add `RangeExpr` SemanticKind with `RangeTargetKind`
- [ ] Update FNCS Coordinator to handle `SynExpr.IndexRange`
- [ ] Implement `Range.toArray` intrinsic (simplest case)
- [ ] Implement `Range.toList` intrinsic
- [ ] Implement stepped range variants (`Range.toArrayStep`, etc.)
- [ ] Alex witnesses for range code generation
- [ ] `[|1..10|]` compiles and produces correct array
- [ ] `[1..10]` compiles and produces correct list
- [ ] `[1..2..10]` stepped ranges work

### Phase 3: List Foundation
- [ ] Add `TList` to NativeTypes
- [ ] Implement core List intrinsics (empty, cons, head, tail, length)
- [ ] Implement List.map, filter, fold
- [ ] List sample compiles and runs

### Phase 4: Complete Collections
- [ ] Implement remaining Map operations
- [ ] Implement remaining Set operations
- [ ] Implement tree rebalancing (AVL)
- [ ] Full sample compiles and runs

## 10. Relationship to Other PRDs

| PRD | Relationship |
|-----|--------------|
| PRD-11 (Closures) | List.map, filter, fold take closure parameters |
| PRD-13 (Recursion) | Collection operations are naturally recursive |
| PRD-14 (Lazy) | `List.tryHead` returns `Option`, not `Lazy` |
| PRD-15 (Seq) | `Seq.toList`, `List.toSeq` convert between |
| PRD-16 (SeqOps) | Many Seq ops mirror List ops |
| PRD-20 (Regions) | Collections allocate from regions |

## 11. Anti-Patterns

| Pattern | Why Wrong | Correct Approach |
|---------|-----------|------------------|
| `ResizeArray` | BCL mutable type | Use `List` or `Array` |
| `Dictionary` | BCL mutable type | Use `Map` |
| `HashSet` | BCL mutable type | Use `Set` |
| `box`/`unbox` | No `obj` in native | Use proper generic types |
| Uniform representation | Performance cost | Monomorphize per type |

## 12. Notes on BAREWire Refactoring

With this PRD, BAREWire needs these changes:

1. **Replace `ResizeArray`** → Use `Array.zeroCreate` + explicit resizing, or use `List` and convert
2. **Remove `box`/`unbox`** → Use proper typed accessors or generic dispatch
3. **Update Map/Set usage** → Already correct, just needs intrinsics

The `box`/`unbox` usage in `Memory/View.fs` (lines 459-503) is the critical blocker. This must be refactored to use typed read/write functions dispatched by NTUKind match, not runtime boxing.
