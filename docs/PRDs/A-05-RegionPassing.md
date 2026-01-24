# A-05: Region Parameters

> **Sample**: `21_RegionPassing` | **Status**: Planned | **Depends On**: A-04 (BasicRegion)

## 1. Executive Summary

This PRD covers passing regions as function parameters and tracking which region allocated data belongs to. Functions can accept regions to allocate their results into, enabling caller-controlled memory management.

**Key Insight**: Region-parameterized functions enable "output allocation" - the caller provides the region, the callee allocates into it. This avoids allocation/copy overhead when composing operations.

## 2. Language Feature Specification

### 2.1 Region as Parameter

```fsharp
let allocateBuffer (r: Region) (size: int) : nativeptr<byte> =
    Region.alloc<byte> r size
```

### 2.2 Region-Parameterized Operations

```fsharp
let transform (r: Region) (input: nativeptr<int>) (len: int) : nativeptr<int> =
    let output = Region.alloc<int> r len
    for i in 0..(len-1) do
        NativePtr.set output i (NativePtr.get input i * 2)
    output
```

### 2.3 Borrowed Region Semantics

When a region is passed to a function:
- The function may allocate into it
- The function MUST NOT release it
- The caller retains ownership

This is "borrowing" the region - not ownership transfer.

## 3. FNCS Layer Implementation

### 3.1 Region Parameter Tracking

```fsharp
type BorrowedRegionCoeffect = {
    ParamName: string
    ParamNodeId: NodeId
    Allocations: NodeId list  // Region.alloc calls using this param
}
```

### 3.2 Ownership Analysis

FNCS ensures:
1. Region parameters are not released by the callee
2. Data allocated in borrowed region may escape (it's the caller's region)

```fsharp
let checkRegionRelease env builder regionExpr =
    let regionNode = checkExpr env builder regionExpr
    match lookupBindingKind regionNode with
    | Parameter ->
        error "Cannot release a borrowed region (passed as parameter)"
    | Local ->
        // OK - local regions can be released
        ...
```

### 3.3 Return Type Annotation (Future)

For more precise tracking, return types could indicate region provenance:

```fsharp
// Conceptual (internal to compiler)
val transform : r:Region -> input:nativeptr<int> -> len:int -> nativeptr<int>@r
```

The `@r` annotation (not user-visible syntax) indicates the result is allocated in region `r`.

## 4. Firefly/Alex Layer Implementation

### 4.1 Region Parameter SSA

Region parameters are treated like any other parameter:

```fsharp
// In SSAAssignment for function parameters
| paramType when paramType = TRegion ->
    let paramSSA = freshSSA ()
    bindParameter name paramSSA TRegion
```

### 4.2 No Special Witness Logic

Region parameters use the same MLIR as local regions - they're just pointers to Region structs. The only difference is ownership tracking (no release).

### 4.3 BorrowedRegion Coeffect

The coeffect system tracks borrowed regions to prevent accidental release:

```fsharp
type FunctionCoeffect = {
    BorrowedRegions: Set<string>  // Parameter names
    // ...
}

let emitRegionRelease z regionSSA =
    if isBorrowedRegion regionSSA z then
        // Compiler error - should have been caught in FNCS
        failwith "Cannot release borrowed region"
    else
        emitMunmap z regionSSA
```

## 5. MLIR Output Specification

### 5.1 Function with Region Parameter

```mlir
// let transform (r: Region) (input: nativeptr<int>) (len: int)
llvm.func @transform(%r: !llvm.ptr, %input: !llvm.ptr, %len: i32) -> !llvm.ptr {
    // Allocate output in caller's region
    %elem_size = arith.constant 4 : i64
    %bytes = arith.muli %len, %elem_size : i64

    %used_ptr = llvm.getelementptr %r[0, 2]
    %used = llvm.load %used_ptr : i64
    %new_used = arith.addi %used, %bytes : i64
    llvm.store %new_used, %used_ptr

    %base_ptr = llvm.getelementptr %r[0, 0]
    %base = llvm.load %base_ptr : !llvm.ptr
    %output = llvm.getelementptr %base[%used]

    // Transform loop
    // ...

    llvm.return %output : !llvm.ptr
}
```

### 5.2 Caller Site

```mlir
// let region = Region.create 4
// let result = transform region input 100
%region = ...  // create region
%result = llvm.call @transform(%region, %input, %c100)
// Region still owned by caller - will be released at scope exit
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module RegionPassingSample

let allocateAndFill (r: Region) (size: int) (value: int) : nativeptr<int> =
    let buffer = Region.alloc<int> r size
    for i in 0..(size-1) do
        NativePtr.set buffer i value
    buffer

let doubleValues (r: Region) (input: nativeptr<int>) (len: int) : nativeptr<int> =
    let output = Region.alloc<int> r len
    for i in 0..(len-1) do
        NativePtr.set output i (NativePtr.get input i * 2)
    output

[<EntryPoint>]
let main _ =
    Console.writeln "=== Region Passing Test ==="

    let region = Region.create 4

    // Allocate and fill in same region
    let data = allocateAndFill region 5 10
    Console.write "Initial: "
    for i in 0..4 do
        Console.write (Format.int (NativePtr.get data i))
        Console.write " "
    Console.writeln ""

    // Transform into same region
    let doubled = doubleValues region data 5
    Console.write "Doubled: "
    for i in 0..4 do
        Console.write (Format.int (NativePtr.get doubled i))
        Console.write " "
    Console.writeln ""

    // Region.release region ← compiler-inserted
    0
```

### 6.2 Expected Output

```
=== Region Passing Test ===
Initial: 10 10 10 10 10
Doubled: 20 20 20 20 20
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `ScopeAnalysis.fs` | MODIFY | Track borrowed vs owned regions |
| `CheckExpressions.fs` | MODIFY | Validate no release on borrowed |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/RegionOwnership.fs` | CREATE | Track borrowed region coeffects |

## 8. Implementation Checklist

### Phase 1: FNCS Ownership Tracking
- [ ] Distinguish owned vs borrowed regions
- [ ] Error on releasing borrowed region
- [ ] Track allocations per region parameter

### Phase 2: Alex Implementation
- [ ] Add BorrowedRegion coeffect
- [ ] Verify release not emitted for borrowed

### Phase 3: Validation
- [ ] Sample 21 compiles without errors
- [ ] Sample 21 produces correct output
- [ ] Compiler rejects invalid release of borrowed region
- [ ] Samples 01-20 still pass

## 9. Related PRDs

- **A-04**: BasicRegion - Foundation
- **A-06**: RegionEscape - Data escaping regions
- **T-03 to T-05**: MailboxProcessor - Workers with borrowed regions
