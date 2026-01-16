# PRD-16: Lazy Values

> **Sample**: `16_LazyValues` | **Status**: Planned | **Depends On**: PRD-11 (Closures)

## 1. Executive Summary

Lazy values defer computation until explicitly forced. Unlike sequences (which may yield multiple values), a lazy value produces exactly one value that is then cached. This enables efficient memoization and avoidance of unnecessary computation.

**Key Insight**: A lazy value is a thunk (suspended computation) plus a cache. The struct holds: `{ Computed: bool, Value: 'T, Thunk: unit -> 'T }`. First `force` evaluates the thunk and caches; subsequent forces return the cached value.

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

## 3. FNCS Layer Implementation

### 3.1 Lazy Type

```fsharp
// In NativeTypes.fs
| TLazy of elementType: NativeType

// Type constructor
| "Lazy" -> fun elemTy -> NativeType.TLazy(elemTy)
```

### 3.2 SemanticKind.LazyExpr

```fsharp
type SemanticKind =
    | LazyExpr of body: NodeId * captures: CaptureInfo list
```

Similar to Lambda, but:
- Always takes `unit` (no parameters)
- Has implicit caching semantics

### 3.3 Lazy Intrinsics

```fsharp
// In CheckExpressions.fs
| "Lazy.force" ->
    // Lazy<'a> -> 'a
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TLazy(aVar), aVar)

| "Lazy.value" ->
    // Same as force (property accessor compiled to function)
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TLazy(aVar), aVar)
```

### 3.4 Capture Analysis

Lazy expressions capture variables from enclosing scope, just like lambdas:

```fsharp
let checkLazyExpr env builder lazyBody =
    // 1. Check body
    let bodyNode = checkExpr env builder lazyBody

    // 2. Collect captures (reuse closure logic)
    let captures = collectCaptures env builder bodyNode.Id

    // 3. Create LazyExpr
    builder.Create(
        SemanticKind.LazyExpr(bodyNode.Id, captures),
        NativeType.TLazy(bodyNode.Type),
        range)
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Lazy Struct Layout

```fsharp
type LazyFrame<'T> = {
    Computed: bool      // Has the thunk been evaluated?
    Value: 'T           // Cached result (valid only if Computed)
    Thunk: unit -> 'T   // The suspended computation (closure)
}
```

### 4.2 Lazy Creation Witness

```fsharp
let witnessLazyExpr z lazyNodeId =
    let coeffect = lookupLazyCoeffect lazyNodeId z

    // 1. Allocate lazy struct
    emit $"  %%{coeffect.LazySSA} = llvm.alloca 1 x {coeffect.LazyStructType}"

    // 2. Initialize computed = false
    emit $"  %%computed_ptr = llvm.getelementptr %%{coeffect.LazySSA}[0, 0]"
    emit "  llvm.store %false, %computed_ptr"

    // 3. Build thunk closure (captures computation)
    let thunkClosure = emitThunkClosure z lazyNodeId coeffect.Captures
    emit $"  %%thunk_ptr = llvm.getelementptr %%{coeffect.LazySSA}[0, 2]"
    emit $"  llvm.store %%{thunkClosure}, %%thunk_ptr"

    TRValue { SSA = coeffect.LazySSA; Type = TLazy }
```

### 4.3 Lazy.force Witness

```fsharp
let witnessLazyForce z lazySSA =
    let resultSSA = freshSynthSSA z

    // Check if already computed
    emit $"  %%computed_ptr = llvm.getelementptr %%{lazySSA}[0, 0]"
    emit "  %computed = llvm.load %computed_ptr : i1"
    emit "  llvm.cond_br %computed, ^cached, ^compute"

    // Cached path
    emit "^cached:"
    emit $"  %%value_ptr = llvm.getelementptr %%{lazySSA}[0, 1]"
    emit $"  %%{resultSSA}_cached = llvm.load %%value_ptr"
    emit "  llvm.br ^done"

    // Compute path
    emit "^compute:"
    emit $"  %%thunk_ptr = llvm.getelementptr %%{lazySSA}[0, 2]"
    emit "  %thunk = llvm.load %thunk_ptr : !closure_type"
    emit "  %code = llvm.extractvalue %thunk[0]"
    emit "  %env = llvm.extractvalue %thunk[1]"
    emit $"  %%{resultSSA}_computed = llvm.call %%code(%%env)"

    // Cache the result
    emit $"  llvm.store %%{resultSSA}_computed, %%value_ptr"
    emit "  llvm.store %true, %computed_ptr"
    emit "  llvm.br ^done"

    // Merge
    emit "^done:"
    emit $"  %%{resultSSA} = llvm.phi [%%{resultSSA}_cached, ^cached], [%%{resultSSA}_computed, ^compute]"

    TRValue { SSA = resultSSA; Type = valueType }
```

## 5. MLIR Output Specification

### 5.1 Lazy Struct Type

```mlir
// lazy { Console.writeln "Computing..."; 42 }
!lazy_int = !llvm.struct<(
    i1,             // computed flag
    i32,            // cached value
    !closure_type   // thunk closure
)>
```

### 5.2 Lazy Creation

```mlir
// let expensive = lazy { ... }
%expensive = llvm.alloca 1 x !lazy_int

// Set computed = false
%computed_ptr = llvm.getelementptr %expensive[0, 0]
llvm.store %false, %computed_ptr

// Build thunk closure
%thunk = ...  // closure with body computation
%thunk_ptr = llvm.getelementptr %expensive[0, 2]
llvm.store %thunk, %thunk_ptr
```

### 5.3 Lazy Force

```mlir
llvm.func @lazy_force_int(%lazy: !llvm.ptr) -> i32 {
    %computed_ptr = llvm.getelementptr %lazy[0, 0]
    %computed = llvm.load %computed_ptr : i1
    llvm.cond_br %computed, ^cached, ^compute

^cached:
    %value_ptr = llvm.getelementptr %lazy[0, 1]
    %cached_val = llvm.load %value_ptr : i32
    llvm.br ^done(%cached_val : i32)

^compute:
    %thunk_ptr = llvm.getelementptr %lazy[0, 2]
    %thunk = llvm.load %thunk_ptr : !closure_type
    %code = llvm.extractvalue %thunk[0]
    %env = llvm.extractvalue %thunk[1]
    %computed_val = llvm.call %code(%env) : (!llvm.ptr) -> i32

    // Cache result
    llvm.store %computed_val, %value_ptr
    llvm.store %true, %computed_ptr
    llvm.br ^done(%computed_val : i32)

^done(%result: i32):
    llvm.return %result : i32
}
```

## 6. Thread Safety Consideration

**Single-threaded implementation** (for now):
- No locking around the computed check
- Multiple threads could compute simultaneously (benign race if pure)

**Future (with threading PRD-27-28)**:
- Add mutex for thread-safe lazy initialization
- Or use compare-and-swap for lock-free initialization

## 7. Validation

### 7.1 Sample Code

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

    Console.writeln "--- Second Force (cached) ---"
    let v2 = Lazy.force expensive
    Console.write "Result: "
    Console.writeln (Format.int v2)

    Console.writeln "--- Lazy with captures ---"
    let sum = lazyAdd 10 20
    Console.write "Sum: "
    Console.writeln (Format.int (Lazy.force sum))

    0
```

### 7.2 Expected Output

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

Note: "Computing expensive value..." appears only ONCE - the second force uses the cached value.

## 8. Files to Create/Modify

### 8.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TLazy type constructor |
| `SemanticGraph.fs` | MODIFY | Add LazyExpr SemanticKind |
| `CheckExpressions.fs` | MODIFY | Add Lazy.force intrinsic |
| `Expressions/Coordinator.fs` | MODIFY | Handle lazy { } expressions |

### 8.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/LazyWitness.fs` | CREATE | Emit lazy struct and force MLIR |
| `src/Alex/Preprocessing/SSAAssignment.fs` | MODIFY | Handle LazyExpr SSAs |
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle LazyExpr, Lazy.force |

## 9. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TLazy to NativeTypes
- [ ] Add LazyExpr to SemanticKind
- [ ] Implement lazy { } checking with capture analysis
- [ ] Add Lazy.force intrinsic

### Phase 2: Alex Implementation
- [ ] Create LazyWitness
- [ ] Implement lazy struct allocation
- [ ] Implement thunk closure creation
- [ ] Implement force with caching

### Phase 3: Validation
- [ ] Sample 16 compiles without errors
- [ ] Sample 16 produces correct output (caching verified)
- [ ] Samples 01-15 still pass

## 10. Related PRDs

- **PRD-11**: Closures - Thunks are closures
- **PRD-14**: Sequences - Both are deferred computation
- **PRD-27-28**: Threading - Thread-safe lazy initialization
