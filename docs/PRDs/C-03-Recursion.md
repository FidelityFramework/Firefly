# C-03: Recursive Bindings

> **Sample**: `13_Recursion` | **Status**: In Progress | **Depends On**: C-01 (Closures), C-02 (HOFs)

## 1. Executive Summary

Recursive functions (`let rec`) require that a binding be visible within its own body. This PRD implements:
1. **Simple recursion** - `let rec f x = ... f ...`
2. **Nested recursion** - `let f x = let rec loop y = ... loop ... in loop x`
3. **Mutual recursion** - `let rec f x = ... g ... and g y = ... f ...`

## 2. The Core Challenge

A recursive function references itself before its definition is complete:

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)  // VarRef to 'factorial' - but we're still defining it!
```

The VarRef to `factorial` inside the body needs a `defId` pointing to the Binding node. But at the time we check the body, the Binding node doesn't exist yet.

## 3. FNCS Implementation

### 3.1 Current State (Incomplete)

In `checkLetOrUse` for `let rec`:
```fsharp
let bindingEnv =
    if isRec then
        bindings |> List.fold (fun env binding ->
            let name = getBindingName binding
            let ty = freshTypeVar range
            addBinding name ty false None env  // NodeId = None!
        ) env
```

This adds bindings to the environment with `NodeId = None`, so self-referential VarRefs get `defId = None`.

### 3.2 Required Implementation

**Pre-create Binding nodes** to obtain NodeIds before checking bodies:

```fsharp
let checkLetOrUse ... =
    let bindings = letOrUse.Bindings
    let isRec = letOrUse.IsRecursive

    if isRec then
        // STEP 1: Pre-create Binding nodes to get NodeIds
        let bindingNodes = bindings |> List.map (fun binding ->
            let name = getBindingName binding
            let ty = freshTypeVar range
            let node = builder.Create(
                SemanticKind.Binding(name, false, true, false),  // isRec = true
                ty,
                range)
            (binding, name, ty, node))

        // STEP 2: Add ALL bindings to environment WITH their NodeIds
        let envWithBindings = bindingNodes |> List.fold (fun env (_, name, ty, node) ->
            addBinding name ty false (Some node.Id) env
        ) env

        // STEP 3: Check each body in the extended environment
        let checkedBindings = bindingNodes |> List.map (fun (binding, name, ty, bindingNode) ->
            // Check the body - VarRefs to 'name' now resolve to bindingNode.Id
            let bodyNode = checkBinding ... envWithBindings ...
            // Update the binding node with the actual body
            (bindingNode, bodyNode))

        // STEP 4: Build result
        ...
    else
        // Non-recursive: existing logic
        ...
```

### 3.3 Key Principle

> **The NodeId must exist before we need to reference it.**

For recursive bindings:
1. Create the Binding node (get NodeId)
2. Add to environment with that NodeId
3. Check body (VarRefs resolve via environment)
4. Connect body to Binding node

### 3.4 Files to Modify

| File | Change |
|------|--------|
| `Expressions/Bindings.fs` | Restructure `checkLetOrUse` for recursive bindings |

**No other FNCS files need changes.** The SemanticKind.Binding already has an `isRec` flag. VarRef already supports `defId: NodeId option`.

## 4. Firefly Implementation

### 4.1 SSAAssignment: Nested Function Naming

For nested functions with the same name (e.g., `loop` in multiple functions), qualify names by walking the Parent chain:

```fsharp
// In collectLambdas:
let findEnclosingFunctionName (startId: NodeId) : string option =
    let rec walk nodeId passedFirstLambda =
        match graph.Nodes.TryFind nodeId with
        | None -> None
        | Some n ->
            match n.Kind with
            | SemanticKind.Lambda _ when passedFirstLambda ->
                // Found enclosing Lambda - get parent Binding's name
                match n.Parent with
                | Some pid ->
                    match graph.Nodes.[pid].Kind with
                    | SemanticKind.Binding(name, _, _, _) -> Some name
                    | _ -> None
                | None -> None
            | _ ->
                match n.Parent with
                | Some pid -> walk pid true
                | None -> None
    walk startId false

// Usage: qualify nested names
match findEnclosingFunctionName lambdaId with
| Some enclosing -> sprintf "%s_%s" enclosing baseName
| None -> baseName
```

### 4.2 No Witness Changes Needed

With proper PSG (VarRefs have defIds), the existing witness code handles recursive calls correctly:
- VarRef has defId → lookup in lambdaNames → emit call

## 5. FNCS: Nested Function Captures (Issue Found Jan 2026)

### 5.1 The Problem

Nested recursive functions may **capture variables from enclosing scope**, but the current implementation in `checkSingleBinding` hardcodes `captures = []` for ALL named function bindings:

```fsharp
// Bindings.fs line ~223 - CURRENT (BUGGY)
SemanticKind.Lambda(lambdaParams, bodyNode.Id, [], env.EnclosingFunction, LambdaContext.RegularClosure),
//                                               ^^ captures hardcoded to empty!
```

The comment says:
```fsharp
// Named function bindings don't capture from outer scope (they ARE the outer scope)
```

This is **correct for top-level functions** but **wrong for nested functions**.

### 5.2 Example: sumTo vs factorialTail

```fsharp
// factorialTail - WORKS (no capture needed)
let factorialTail (n: int) : int =
    let rec loop acc n =     // 'n' is a PARAMETER, shadows outer n
        if n <= 1 then acc
        else loop (acc * n) (n - 1)
    loop 1 n

// sumTo - BROKEN (capture needed but not computed)
let sumTo (n: int) : int =
    let rec loop acc i =     // only 'acc' and 'i' are parameters
        if i > n then acc    // 'n' CAPTURED from outer scope!
        else loop (acc + i) (i + 1)
    loop 0 1
```

In `sumTo`, the nested `loop` references `n` from the enclosing function. This is a capture that must flow to MLIR.

**Current MLIR output (wrong):**
```mlir
llvm.func @sumTo_loop(%arg0: i64, %arg1: i64) -> i64 {
    %cmp = arith.cmpi sgt, %arg1, %arg0 : i64   // comparing i > acc, NOT i > n!
```

The condition `i > n` became `i > acc` because `n` wasn't passed.

### 5.3 The Fix

When `env.EnclosingFunction.IsSome`, we're inside a function, so this is a nested function that may need captures. Call `computeCaptures` (from Applications.fs) instead of passing `[]`:

```fsharp
// Bindings.fs - CORRECTED
let paramNames = lambdaParams |> List.map (fun (name, _, _) -> name) |> Set.ofList
// Also exclude the function's own name (for recursive self-reference)
let excludeNames = Set.add name paramNames

// Compute captures only for nested functions (top-level never captures)
let captures =
    if env.EnclosingFunction.IsSome then
        computeCaptures builder env bodyNode.Id excludeNames
    else
        []

let lambdaNode = builder.Create(
    SemanticKind.Lambda(lambdaParams, bodyNode.Id, captures, env.EnclosingFunction, LambdaContext.RegularClosure),
    funcType,
    range,
    children = lambdaChildren)
```

### 5.4 Alex Impact

Once captures flow through the PSG, Alex/LambdaWitness must handle them for nested recursive functions:
- Either pass captures as additional parameters (parameter-passing style)
- Or create a closure environment (flat closure style, as in C-01)

For tail-recursive nested functions that don't escape, parameter-passing is more efficient.

## 6. Verification

### 6.1 PSG Check
```
VarRef ("factorial", Some (NodeId 547))  // Self-reference has defId
```

### 6.2 MLIR Check
```mlir
llvm.func @factorial(%n: i32) -> i32 {
    ...
    %result = llvm.call @factorial(%n_minus_1) : (i32) -> i32  // Recursive call works
    ...
}
```

### 6.3 Execution Check
```
factorial 5: 120
factorialTail 5: 120
sumTo 10: 55        // After capture fix
```

## 7. Implementation Checklist

### Phase 1: FNCS - Recursive Binding NodeIds
- [x] Restructure `checkLetOrUse` to pre-create Binding nodes for `let rec`
- [x] Add NodeIds to environment before checking bodies
- [x] Verify: VarRef to self has `defId = Some nodeId`
- [x] FNCS builds

### Phase 2: Firefly - Nested Function Naming (Parent Links)
- [x] Bindings.fs: `buildSequential` sets parent on all children
- [x] Bindings.fs: Lambda creation sets parent on params and body
- [x] TypeOperations.fs: TypeAnnotation sets parent on inner node
- [x] SSAAssignment uses Parent chain for qualified names
- [x] Verify: `@factorialTail_loop` not `@loop`
- [x] Firefly builds

### Phase 3: FNCS - Nested Function Captures (NEW)
- [ ] Import `computeCaptures` into Bindings.fs (or move to shared module)
- [ ] Call `computeCaptures` when `env.EnclosingFunction.IsSome`
- [ ] Exclude function's own name and parameters from capture set
- [ ] Verify: sumTo's loop Lambda has `captures = [{Name="n"; ...}]`
- [ ] FNCS builds

### Phase 4: Alex - Handle Nested Function Captures
- [ ] LambdaWitness: pass captures as additional parameters for non-escaping nested functions
- [ ] Generate `@sumTo_loop(n, acc, i)` not `@sumTo_loop(acc, i)`
- [ ] Verify: sumTo returns 55 for n=10

### Phase 5: Validation
- [ ] RecursionSimple compiles and executes
- [ ] Recursion.fidproj (full) compiles and executes with correct sumTo results
- [ ] Samples 01-14 still pass

## 8. Related PRDs

- **C-01**: Closures - nested functions may capture
- **C-02**: HOFs - recursive functions as values
