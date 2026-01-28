# A-04: Basic Region Allocation

> **Sample**: `20_BasicRegion` | **Status**: Planned | **Depends On**: C-01 to A-03

## 1. Executive Summary

Scoped Regions provide dynamic memory allocation with **compiler-inferred deterministic disposal**. Unlike GC-based allocation, regions have lexically-scoped lifetimes - the compiler inserts deallocation at scope exit.

**Key Insight**: Regions are MLKit-style memory management. All allocations in a region are freed together when the region is released. This is bulk deallocation - no per-object tracking, no GC pauses.

**Reference**: See `scoped_regions_architecture` memory for design details.

## 2. Language Feature Specification

### 2.1 Region Creation

```fsharp
let region = Region.create 4  // 4 pages (16KB)
```

Creates a region backed by OS virtual memory (mmap/VirtualAlloc).

### 2.2 Region Allocation

```fsharp
let buffer = Region.alloc<byte> region 1024
```

Bump-pointer allocation within the region.

### 2.3 Automatic Disposal

```fsharp
let processData () =
    let region = Region.create 4
    let temp = Region.alloc<int> region 100

    // ... use temp ...

    // Region.release region  ← COMPILER-INSERTED at scope exit
```

The compiler tracks Region bindings and inserts `Region.release` at all scope exit points.

### 2.4 No IDisposable

Regions are NOT IDisposable. There's no `use` keyword. The compiler manages lifetime automatically based on lexical scope.

## 3. FNCS Layer Implementation

### 3.1 Region Type

```fsharp
// In NativeTypes.fs
| TRegion  // Opaque region handle

// Region is a built-in type, not parameterized
```

### 3.2 Region Intrinsics

```fsharp
// In CheckExpressions.fs
| "Region.create" ->
    // int -> Region (pages argument)
    NativeType.TFun(env.Globals.IntType, NativeType.TRegion)

| "Region.alloc" ->
    // Region -> int -> nativeptr<'T>
    // Type parameter 'T determines element size
    let tVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TRegion,
        NativeType.TFun(env.Globals.IntType, NativeType.TNativePtr(tVar)))

| "Region.release" ->
    // Region -> unit
    NativeType.TFun(NativeType.TRegion, env.Globals.UnitType)
```

### 3.3 Linear Resource Tracking

FNCS marks Region bindings as linear resources:

```fsharp
type BindingKind =
    | Normal
    | LinearResource of resourceType: LinearResourceType

type LinearResourceType =
    | Region
    // Future: FileHandle, Socket, etc.
```

During scope exit analysis (a nanopass), the compiler ensures each linear resource is consumed exactly once.

### 3.4 Scope Exit Coeffect

```fsharp
type ScopeExitCoeffect = {
    ScopeId: NodeId
    LinearResources: (string * NodeId) list  // Name and release point
}
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Region Struct

```fsharp
type Region = {
    Base: nativeptr<byte>    // mmap'd memory
    Capacity: int64          // Total bytes
    Used: int64              // Bump pointer offset
    Growable: bool           // Can expand?
}
```

### 4.2 Region.create Witness

```fsharp
let witnessRegionCreate z pagesSSA =
    let regionSSA = freshSSA ()

    // Calculate size
    emit $"  %%size = arith.muli %%{pagesSSA}, 4096 : i64"

    // mmap (addr=0 means OS chooses address)
    emit "  %addr_zero = llvm.mlir.zero : !llvm.ptr"
    emit $"  %%base = llvm.call @mmap(%%addr_zero, %%size, i32 3, i32 34, i32 -1, i64 0)"

    // Allocate Region struct
    emit $"  %%{regionSSA} = llvm.alloca 1 x !region_type"
    emit $"  %%base_ptr = llvm.getelementptr %%{regionSSA}[0, 0]"
    emit "  llvm.store %base, %base_ptr"
    emit $"  %%cap_ptr = llvm.getelementptr %%{regionSSA}[0, 1]"
    emit "  llvm.store %size, %cap_ptr"
    emit $"  %%used_ptr = llvm.getelementptr %%{regionSSA}[0, 2]"
    emit "  llvm.store 0, %used_ptr"

    TRValue { SSA = regionSSA; Type = TRegion }
```

### 4.3 Region.alloc Witness

```fsharp
let witnessRegionAlloc z regionSSA countSSA elemSize =
    let resultSSA = freshSSA ()

    // Calculate byte size
    emit $"  %%bytes = arith.muli %%{countSSA}, {elemSize} : i64"

    // Get current used offset
    emit $"  %%used_ptr = llvm.getelementptr %%{regionSSA}[0, 2]"
    emit "  %used = llvm.load %used_ptr : i64"

    // Bump pointer
    emit "  %new_used = arith.addi %used, %bytes : i64"
    emit "  llvm.store %new_used, %used_ptr"

    // Calculate result pointer
    emit $"  %%base_ptr = llvm.getelementptr %%{regionSSA}[0, 0]"
    emit "  %base = llvm.load %base_ptr : !llvm.ptr"
    emit $"  %%{resultSSA} = llvm.getelementptr %%base[%%used]"

    TRValue { SSA = resultSSA; Type = TNativePtr elemType }
```

### 4.4 Region.release Witness

```fsharp
let witnessRegionRelease z regionSSA =
    // Get base and capacity
    emit $"  %%base_ptr = llvm.getelementptr %%{regionSSA}[0, 0]"
    emit "  %base = llvm.load %base_ptr : !llvm.ptr"
    emit $"  %%cap_ptr = llvm.getelementptr %%{regionSSA}[0, 1]"
    emit "  %cap = llvm.load %cap_ptr : i64"

    // munmap
    emit "  llvm.call @munmap(%base, %cap)"

    TRVoid
```

### 4.5 Automatic Release Insertion

The `ScopeExitInsertion` nanopass adds `Region.release` calls:

```fsharp
let insertScopeExits (graph: SemanticGraph) =
    for scope in allScopes graph do
        for (regionName, regionNodeId) in scope.LinearResources do
            // For each exit point of scope, insert release
            for exitPoint in scope.ExitPoints do
                insertBefore exitPoint (RegionRelease regionNodeId)
```

## 5. MLIR Output Specification

### 5.1 Region Type

```mlir
!region_type = !llvm.struct<(
    ptr,     // base pointer
    i64,     // capacity
    i64,     // used
    i1       // growable
)>
```

### 5.2 Region Create

```mlir
// let region = Region.create 4
%size = arith.muli %pages, %c4096 : i64
%addr_zero = llvm.mlir.zero : !llvm.ptr  // OS chooses address
%base = llvm.call @mmap(
    %addr_zero,   // addr (0 = OS chooses)
    %size,        // length
    i32 3,        // PROT_READ | PROT_WRITE
    i32 34,       // MAP_PRIVATE | MAP_ANONYMOUS
    i32 -1,       // fd (unused)
    i64 0         // offset
) : (!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr

%region = llvm.alloca 1 x !region_type
%base_slot = llvm.getelementptr %region[0, 0]
llvm.store %base, %base_slot
%cap_slot = llvm.getelementptr %region[0, 1]
llvm.store %size, %cap_slot
%used_slot = llvm.getelementptr %region[0, 2]
llvm.store %c0, %used_slot
```

### 5.3 Region Alloc (Bump Pointer)

```mlir
// let buffer = Region.alloc<int> region 100
%elem_size = arith.constant 4 : i64  // sizeof(int)
%bytes = arith.muli %count, %elem_size : i64

%used_ptr = llvm.getelementptr %region[0, 2]
%used = llvm.load %used_ptr : i64
%new_used = arith.addi %used, %bytes : i64
llvm.store %new_used, %used_ptr

%base_ptr = llvm.getelementptr %region[0, 0]
%base = llvm.load %base_ptr : !llvm.ptr
%result = llvm.getelementptr %base[%used] : (!llvm.ptr, i64) -> !llvm.ptr
```

### 5.4 Region Release

```mlir
// Region.release region (compiler-inserted)
%base = llvm.load %base_ptr : !llvm.ptr
%cap = llvm.load %cap_ptr : i64
llvm.call @munmap(%base, %cap) : (!llvm.ptr, i64) -> i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicRegionSample

let processInRegion () =
    let region = Region.create 4  // 16KB

    // Allocate temporary buffer
    let buffer = Region.alloc<int> region 100

    // Use buffer
    for i in 0..99 do
        NativePtr.set buffer i (i * i)

    // Compute sum
    let mutable sum = 0
    for i in 0..99 do
        sum <- sum + NativePtr.get buffer i

    // Region.release region  ← compiler-inserted
    sum

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Region Test ==="

    let result = processInRegion ()
    Console.write "Sum of squares 0-99: "
    Console.writeln (Format.int result)

    0
```

### 6.2 Expected Output

```
=== Basic Region Test ===
Sum of squares 0-99: 328350
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TRegion type |
| `CheckExpressions.fs` | MODIFY | Add Region intrinsics |
| `SemanticGraph.fs` | MODIFY | Add LinearResource binding kind |
| `ScopeAnalysis.fs` | CREATE | Track linear resources and exit points |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/ScopeExitInsertion.fs` | CREATE | Insert Region.release at exits |
| `src/Alex/Witnesses/RegionWitness.fs` | CREATE | Emit region MLIR |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TRegion to NativeTypes
- [ ] Add Region.create/alloc/release intrinsics
- [ ] Implement linear resource tracking
- [ ] Create ScopeAnalysis pass

### Phase 2: Alex Implementation
- [ ] Create ScopeExitInsertion nanopass
- [ ] Create RegionWitness
- [ ] Implement mmap/munmap platform bindings

### Phase 3: Validation
- [ ] Sample 20 compiles without errors
- [ ] Sample 20 produces correct output
- [ ] Memory is properly released (no leaks)
- [ ] Samples 01-19 still pass

## 9. Platform Bindings

| Operation | Linux | Windows | Embedded |
|-----------|-------|---------|----------|
| Create | mmap | VirtualAlloc | Static buffer |
| Grow | mremap | VirtualAlloc | ERROR |
| Release | munmap | VirtualFree | No-op |

## 10. Related PRDs

- **A-05**: Region Passing - Regions as parameters
- **A-06**: Region Escape - copyOut for escaping data
- **T-03 to T-05**: MailboxProcessor - Region per message batch
