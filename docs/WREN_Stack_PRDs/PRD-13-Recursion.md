# PRD-13: Recursive Bindings

> **Sample**: `13_Recursion` | **Status**: Planned | **Depends On**: PRD-11 (Closures), PRD-12 (HOFs)

## 1. Executive Summary

Recursive functions (`let rec`) are foundational to functional programming. This PRD covers both simple recursion and mutual recursion (`let rec ... and ...`). The key challenge is that recursive functions reference themselves before their definition is complete.

**Key Insight**: Recursive bindings require forward declaration - the function must be "visible" in its own body. FNCS must handle this in scope construction, and Alex must handle self-referential function pointers.

## 2. Language Feature Specification

### 2.1 Simple Recursion

```fsharp
let rec factorial (n: int) : int =
    if n <= 1 then 1
    else n * factorial (n - 1)
```

The binding `factorial` must be visible within its own body.

### 2.2 Tail Recursion

```fsharp
let factorialTail (n: int) : int =
    let rec loop acc n =
        if n <= 1 then acc
        else loop (acc * n) (n - 1)
    loop 1 n
```

Tail-recursive functions can be optimized to loops (tail call elimination).

### 2.3 Mutual Recursion

```fsharp
let rec isEven (n: int) : bool =
    if n = 0 then true
    else isOdd (n - 1)
and isOdd (n: int) : bool =
    if n = 0 then false
    else isEven (n - 1)
```

Both `isEven` and `isOdd` must be visible in both bodies.

## 3. FNCS Layer Implementation

### 3.1 Recursive Binding Scope

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/Expressions/Bindings.fs`

For `let rec`, the binding must be added to the environment BEFORE checking the body:

```fsharp
let checkRecBinding env builder name paramTypes bodyExpr =
    // 1. Create placeholder type variable
    let funcType = createFunctionTypeFromParams paramTypes

    // 2. Add binding to environment BEFORE checking body
    let envWithSelf = addBinding name funcType true env  // true = recursive

    // 3. Check body in extended environment
    let bodyNode = checkExpr envWithSelf builder bodyExpr

    // 4. Unify placeholder with actual body type
    unify funcType (inferredType bodyNode)
```

### 3.2 SemanticKind for Recursive Binding

Option 1: Mark existing `LetBinding` with recursive flag:
```fsharp
| LetBinding of name: string * isRec: bool * value: NodeId
```

Option 2: Separate kind:
```fsharp
| RecursiveBinding of name: string * value: NodeId
```

**Recommendation**: Use Option 1 - simpler, less code duplication.

### 3.3 Mutual Recursion Handling

For `let rec ... and ...`:

```fsharp
let checkMutualRecBindings env builder bindings =
    // 1. Collect all binding names and create placeholder types
    let placeholders =
        bindings |> List.map (fun (name, params, _) ->
            name, createFunctionTypeFromParams params)

    // 2. Add ALL bindings to environment
    let envWithAll = placeholders |> List.fold addBinding env

    // 3. Check each body in the extended environment
    let checkedBodies =
        bindings |> List.map (fun (_, _, body) ->
            checkExpr envWithAll builder body)

    // 4. Unify placeholders with actual types
    List.iter2 unifyPlaceholder placeholders checkedBodies
```

### 3.4 VarRef to Self

When a recursive function references itself, FNCS creates a `VarRef` that points to the function's own `LetBinding` node:

```fsharp
// In checkExpr for identifier
| SynExpr.Ident(name) ->
    match lookupBinding name env with
    | Some binding ->
        builder.Create(
            SemanticKind.VarRef(name, binding.NodeId),
            binding.Type,
            range)
```

For recursive bindings, `binding.NodeId` may point to a node still being constructed.

## 4. Firefly/Alex Layer Implementation

### 4.1 Recursive Function SSA

Self-references within a recursive function use the function's own address:

```fsharp
// For recursive function 'factorial'
// When we see VarRef to 'factorial' inside factorial's body:
| SemanticKind.VarRef(name, _) when isRecursiveSelfRef name z ->
    // Use the function's own label/address
    let funcSSA = getCurrentFunctionSSA z
    TRValue { SSA = funcSSA; Type = functionType }
```

### 4.2 Tail Call Optimization

LLVM handles tail call optimization when:
1. The call is in tail position
2. The `musttail` or `tail` attribute is applied

```mlir
// Tail-recursive call
llvm.call @loop(%new_acc, %new_n) {tail} : (i32, i32) -> i32
```

**Nanopass consideration**: A `TailPosition` coeffect could mark calls in tail position, enabling `tail` attribute emission.

### 4.3 Mutual Recursion - Forward Declarations

For mutually recursive functions, emit forward declarations:

```mlir
// Forward declarations
llvm.func @isEven(i32) -> i1
llvm.func @isOdd(i32) -> i1

// Definitions
llvm.func @isEven(%n: i32) -> i1 {
    // ... can call @isOdd ...
}

llvm.func @isOdd(%n: i32) -> i1 {
    // ... can call @isEven ...
}
```

## 5. MLIR Output Specification

### 5.1 Simple Recursion

```mlir
llvm.func @factorial(%n: i32) -> i32 {
    %cmp = arith.cmpi sle, %n, %c1 : i32
    llvm.cond_br %cmp, ^base, ^recurse

^base:
    llvm.return %c1 : i32

^recurse:
    %n_minus_1 = arith.subi %n, %c1 : i32
    %sub_result = llvm.call @factorial(%n_minus_1) : (i32) -> i32
    %result = arith.muli %n, %sub_result : i32
    llvm.return %result : i32
}
```

### 5.2 Tail Recursion

```mlir
llvm.func @loop(%acc: i32, %n: i32) -> i32 {
    %cmp = arith.cmpi sle, %n, %c1 : i32
    llvm.cond_br %cmp, ^done, ^continue

^done:
    llvm.return %acc : i32

^continue:
    %new_acc = arith.muli %acc, %n : i32
    %new_n = arith.subi %n, %c1 : i32
    // Tail call - LLVM can optimize to jump
    %result = llvm.call @loop(%new_acc, %new_n) {tail} : (i32, i32) -> i32
    llvm.return %result : i32
}
```

### 5.3 Mutual Recursion

```mlir
// Forward declarations first
llvm.func @isEven(i32) -> i1
llvm.func @isOdd(i32) -> i1

llvm.func @isEven(%n: i32) -> i1 {
    %is_zero = arith.cmpi eq, %n, %c0 : i32
    llvm.cond_br %is_zero, ^true_case, ^recurse

^true_case:
    llvm.return %true : i1

^recurse:
    %n_minus_1 = arith.subi %n, %c1 : i32
    %result = llvm.call @isOdd(%n_minus_1) : (i32) -> i1
    llvm.return %result : i1
}

llvm.func @isOdd(%n: i32) -> i1 {
    %is_zero = arith.cmpi eq, %n, %c0 : i32
    llvm.cond_br %is_zero, ^false_case, ^recurse

^false_case:
    llvm.return %false : i1

^recurse:
    %n_minus_1 = arith.subi %n, %c1 : i32
    %result = llvm.call @isEven(%n_minus_1) : (i32) -> i1
    llvm.return %result : i1
}
```

## 6. Validation

### 6.1 Sample Code

**File**: `samples/console/FidelityHelloWorld/13_Recursion/Recursion.fs`

Key test cases:
- `factorial` - simple recursion
- `factorialTail` - tail recursion with nested `let rec`
- `fibonacci` / `fibonacciTail` - double recursion
- `gcd` - naturally tail recursive
- `isEven` / `isOdd` - mutual recursion

### 6.2 Expected Output

```
=== Recursion Test ===
--- Factorial ---
factorial 5: 120
factorialTail 5: 120
factorial 10: 3628800

--- Fibonacci ---
fibonacci 10: 55
fibonacciTail 10: 55
fibonacciTail 20: 6765

--- Sum ---
sum 1 to 10: 55
sum 1 to 100: 5050

--- Count Digits ---
digits in 12345: 5
digits in 7: 1

--- GCD ---
gcd 48 18: 6
gcd 100 35: 5

--- Power ---
2^10: 1024
3^5: 243

--- Mutual Recursion (Even/Odd) ---
isEven 10: true
isOdd 10: false
isEven 7: false
isOdd 7: true
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `Expressions/Bindings.fs` | MODIFY | Handle `let rec` scope construction |
| `SemanticGraph.fs` | MODIFY | Add `isRec` flag to LetBinding or new RecursiveBinding kind |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle self-referential VarRefs |
| `src/Alex/Witnesses/BindingWitness.fs` | MODIFY | Emit forward declarations for mutual recursion |
| `src/Alex/Preprocessing/SSAAssignment.fs` | MODIFY | Handle recursive binding SSAs |

## 8. Implementation Checklist

### Phase 1: Simple Recursion
- [ ] FNCS: Add binding to scope before checking body
- [ ] FNCS: Handle VarRef to self in recursive function
- [ ] Alex: Emit self-call correctly
- [ ] Test: `factorial` works

### Phase 2: Tail Recursion
- [ ] Add TailPosition coeffect (optional optimization)
- [ ] Emit `tail` attribute on tail calls
- [ ] Test: `factorialTail` works with large inputs (no stack overflow)

### Phase 3: Mutual Recursion
- [ ] FNCS: Handle `let rec ... and ...` binding group
- [ ] Alex: Emit forward declarations
- [ ] Test: `isEven`/`isOdd` work

### Validation
- [ ] Sample 13 compiles without errors
- [ ] Sample 13 produces correct output
- [ ] Samples 01-12 still pass

## 9. Related PRDs

- **PRD-11**: Closures - Nested `let rec` in closures
- **PRD-14-15**: Sequences - Recursive sequence generation
- **PRD-17-19**: Async - Recursive async loops
