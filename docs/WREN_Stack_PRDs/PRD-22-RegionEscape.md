# PRD-22: Region Escape Analysis

> **Sample**: `22_RegionEscape` | **Status**: Planned | **Depends On**: PRD-20-21 (BasicRegion, RegionPassing)

## 1. Executive Summary

This PRD covers **escape analysis** - detecting when region-allocated data would outlive its region, and providing `Region.copyOut` for safe extraction.

**Key Insight**: Data allocated in a region becomes invalid when the region is released. If data needs to survive beyond the region's scope, it must be explicitly copied out.

## 2. Language Feature Specification

### 2.1 The Escape Problem

```fsharp
let dangerous () : nativeptr<int> =
    let region = Region.create 1
    let data = Region.alloc<int> region 10
    // Region.release region ← compiler-inserted
    data  // ERROR: data escapes its region!
```

The returned pointer would be invalid - pointing to deallocated memory.

### 2.2 Region.copyOut

```fsharp
let safe () : int[] =
    let region = Region.create 1
    let temp = Region.alloc<int> region 10
    // ... process temp ...
    let result = Region.copyOut<int> temp 10  // Copy to managed array
    // Region.release region ← compiler-inserted
    result  // OK: result is independent of region
```

### 2.3 copyOut Variations

```fsharp
// Copy to caller-provided region
Region.copyTo<'T> : nativeptr<'T> -> int -> Region -> nativeptr<'T>

// Copy to new heap allocation (when needed)
Region.copyToHeap<'T> : nativeptr<'T> -> int -> 'T[]
```

## 3. FNCS Layer Implementation

### 3.1 Escape Detection

During scope exit analysis, FNCS checks if any region-allocated pointer could escape:

```fsharp
type RegionAllocInfo = {
    RegionNodeId: NodeId
    AllocNodeId: NodeId
    DataType: NativeType
}

let checkForEscapes scope =
    for returnExpr in scope.ReturnExprs do
        match getProvenance returnExpr with
        | RegionAllocated regionId when regionId.IsLocalTo scope ->
            error $"Data from region escapes scope - use Region.copyOut"
        | _ ->
            // OK - not region-allocated or region outlives scope
            ()
```

### 3.2 Provenance Tracking

Track where pointers come from:

```fsharp
type DataProvenance =
    | Stack                        // alloca
    | RegionAllocated of NodeId    // Region.alloc result
    | HeapAllocated                // Explicit heap allocation
    | Unknown                      // External/parameter
```

### 3.3 copyOut Intrinsic

```fsharp
// In CheckExpressions.fs
| "Region.copyOut" ->
    // nativeptr<'T> -> int -> 'T[]
    let tVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TNativePtr(tVar),
        NativeType.TFun(env.Globals.IntType, NativeType.TArray(tVar)))

| "Region.copyTo" ->
    // nativeptr<'T> -> int -> Region -> nativeptr<'T>
    let tVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TNativePtr(tVar),
        NativeType.TFun(
            env.Globals.IntType,
            NativeType.TFun(NativeType.TRegion, NativeType.TNativePtr(tVar))))
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Escape Analysis Nanopass

**File**: `src/Alex/Preprocessing/EscapeAnalysis.fs`

```fsharp
type EscapeAnalysisCoeffect = {
    EscapingExpressions: (NodeId * string) list  // Expr and reason
}

let analyzeEscapes (graph: SemanticGraph) =
    for scope in allScopes graph do
        let localRegions = scope.LocalRegions
        for returnExpr in scope.Returns do
            match provenance returnExpr with
            | RegionAllocated r when Set.contains r localRegions ->
                emit Warning "Potential escape from local region"
```

### 4.2 copyOut Witness

```fsharp
let witnessCopyOut z srcPtrSSA countSSA elemSize =
    let resultSSA = freshSSA ()

    // Calculate byte size
    emit $"  %%bytes = arith.muli %%{countSSA}, {elemSize} : i64"

    // Allocate destination array (or region allocation)
    emit $"  %%{resultSSA} = llvm.call @array_create(%%{countSSA})"

    // Get destination data pointer
    emit $"  %%dest = llvm.call @array_data_ptr(%%{resultSSA})"

    // memcpy
    emit $"  llvm.call @llvm.memcpy.p0.p0.i64(%%dest, %%{srcPtrSSA}, %%bytes, i1 false)"

    TRValue { SSA = resultSSA; Type = TArray elemType }
```

### 4.3 copyTo Witness (Region-to-Region)

```fsharp
let witnessCopyTo z srcPtrSSA countSSA destRegionSSA elemSize =
    // Allocate in destination region
    let destPtrSSA = emitRegionAlloc z destRegionSSA countSSA elemSize

    // memcpy
    emit $"  %%bytes = arith.muli %%{countSSA}, {elemSize} : i64"
    emit $"  llvm.call @llvm.memcpy.p0.p0.i64(%%{destPtrSSA}, %%{srcPtrSSA}, %%bytes, i1 false)"

    TRValue { SSA = destPtrSSA; Type = TNativePtr elemType }
```

## 5. MLIR Output Specification

### 5.1 copyOut to Array

```mlir
// let result = Region.copyOut<int> temp 10
%count = arith.constant 10 : i32
%elem_size = arith.constant 4 : i64
%bytes = arith.muli %count, %elem_size : i64

// Allocate managed array
%result = llvm.call @array_create_int(%count) : (i32) -> !llvm.ptr
%dest = llvm.call @array_data_ptr(%result) : (!llvm.ptr) -> !llvm.ptr

// Copy data
llvm.call @llvm.memcpy.p0.p0.i64(%dest, %temp, %bytes, i1 false)
```

### 5.2 copyTo Region

```mlir
// let copy = Region.copyTo temp 10 otherRegion
%count = arith.constant 10 : i64
%elem_size = arith.constant 4 : i64
%bytes = arith.muli %count, %elem_size : i64

// Allocate in destination region
%used_ptr = llvm.getelementptr %otherRegion[0, 2]
%used = llvm.load %used_ptr : i64
%new_used = arith.addi %used, %bytes : i64
llvm.store %new_used, %used_ptr
%base = llvm.load %base_ptr : !llvm.ptr
%dest = llvm.getelementptr %base[%used]

// Copy data
llvm.call @llvm.memcpy.p0.p0.i64(%dest, %temp, %bytes, i1 false)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module RegionEscapeSample

let processAndExtract (input: int[]) : int[] =
    let region = Region.create 4

    // Temporary processing buffer
    let temp = Region.alloc<int> region (Array.length input)

    // Process: square each value
    for i in 0..(Array.length input - 1) do
        NativePtr.set temp i (input.[i] * input.[i])

    // Extract result before region release
    let result = Region.copyOut<int> temp (Array.length input)

    // Region.release region ← compiler-inserted
    result  // Safe: result is independent

let transformInPlace (r: Region) (data: nativeptr<int>) (len: int) =
    // Temporary buffer in same region (no escape issue)
    let temp = Region.alloc<int> r len

    for i in 0..(len-1) do
        NativePtr.set temp i (NativePtr.get data i + 1)

    // Copy back to original
    for i in 0..(len-1) do
        NativePtr.set data i (NativePtr.get temp i)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Region Escape Test ==="

    let input = [| 1; 2; 3; 4; 5 |]
    Console.write "Input: "
    for x in input do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    let squared = processAndExtract input
    Console.write "Squared: "
    for x in squared do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    0
```

### 6.2 Expected Output

```
=== Region Escape Test ===
Input: 1 2 3 4 5
Squared: 1 4 9 16 25
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `ScopeAnalysis.fs` | MODIFY | Track data provenance |
| `CheckExpressions.fs` | MODIFY | Add copyOut/copyTo intrinsics |
| `EscapeDetection.fs` | CREATE | Detect unsafe escapes |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/EscapeAnalysis.fs` | CREATE | Escape analysis nanopass |
| `src/Alex/Witnesses/RegionWitness.fs` | MODIFY | Add copyOut/copyTo witnesses |

## 8. Implementation Checklist

### Phase 1: FNCS Escape Detection
- [ ] Implement data provenance tracking
- [ ] Detect escaping region pointers
- [ ] Add copyOut/copyTo intrinsics

### Phase 2: Alex Implementation
- [ ] Create EscapeAnalysis nanopass
- [ ] Implement copyOut witness
- [ ] Implement copyTo witness

### Phase 3: Validation
- [ ] Sample 22 compiles without errors
- [ ] Sample 22 produces correct output
- [ ] Compiler rejects unsafe escapes
- [ ] Samples 01-21 still pass

## 9. Escape Analysis Complexity

This is **simple escape analysis** - lexical scope based:

```
Escapes if: data.region.scope ⊂ return.scope
```

NOT full Rust-style borrow checking. Simpler, but requires explicit copyOut for any potential escape.

## 10. Related PRDs

- **PRD-20**: BasicRegion - Foundation
- **PRD-21**: RegionPassing - Borrowed regions
- **PRD-29-31**: MailboxProcessor - Message extraction from worker regions
