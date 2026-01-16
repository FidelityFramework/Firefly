# PRD-11: MLKit-Style Flat Closures

> **Sample**: `11_Closures` | **Status**: In Progress | **Depends On**: Samples 01-10 (complete)

## 1. Executive Summary

This PRD specifies the implementation of MLKit-style flat closures for F# Native compilation. Closures are foundational to functional programming - they enable functions to capture variables from enclosing scopes. This feature unlocks higher-order functions (Sample 12), sequences (Samples 14-15), async (Samples 17-19), and ultimately the MailboxProcessor capstone (Samples 29-31).

**Key Architectural Decision**: Capture analysis is scope analysis. Scope is resolved during type checking. Therefore, **capture analysis belongs in FNCS**, not Firefly. FNCS computes captures during PSG construction; Alex handles only SSA assignment and MLIR emission.

## 2. Language Feature Specification

### 2.1 F# Closure Semantics

A closure is a function bundled with its captured environment:

```fsharp
let makeCounter (start: int) : (unit -> int) =
    let mutable count = start
    fun () ->
        count <- count + 1
        count
```

The inner lambda `fun () -> ...` captures the mutable variable `count`. Each call to `makeCounter` produces an independent closure with its own `count` state.

### 2.2 Capture Modes

| Variable Kind | Capture Mode | Environment Entry | Semantics |
|---------------|--------------|-------------------|-----------|
| Immutable `let x = ...` | ByValue | `T` | Copy value into closure |
| Mutable `let mutable x = ...` | ByRef | `ptr<T>` | Pointer to original slot |

**Critical**: Mutable variables MUST be captured by reference to enable mutation through the closure. Multiple closures capturing the same mutable variable share the same storage.

### 2.3 Flat vs Linked Closures

Based on Shao & Appel (1994) "Space-Efficient Closure Representations":

| Property | Flat Closures (MLKit) | Linked Closures |
|----------|----------------------|-----------------|
| Memory layout | All captures inline | Pointer to outer env |
| Access time | O(1) direct offset | O(depth) chain walk |
| Space safety | Safe for GC | Keeps outer env alive |
| Creation cost | Copy all captures | Store one pointer |
| Cache behavior | Contiguous | Scattered |

**Fidelity uses FLAT closures** because:
1. No GC - space safety is structural, not runtime
2. Region allocation - closures live in regions, not heap
3. Predictable performance - no chain traversal

### 2.4 Memory Layout

```
Closure Structure (Flat)
+---------------------+----------------------------------+
| code_ptr (8 bytes)  | env: captured values inline      |
|                     |   capture_0: T0                  |
|                     |   capture_1: T1                  |
|                     |   ...                            |
+---------------------+----------------------------------+
```

For `makeCounter`, the returned closure:
```
+---------------------+----------------------------------+
| ptr<increment_impl> | count_ref: ptr<int>              |
+---------------------+----------------------------------+
```

## 3. FNCS Layer Implementation

### 3.1 Type Definitions

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/SemanticGraph.fs`

```fsharp
/// Information about a captured variable
type CaptureInfo = {
    Name: string
    Type: NativeType
    IsMutable: bool
    SourceNodeId: NodeId option  // Where the captured binding is defined
}

/// SemanticKind.Lambda now includes captures
type SemanticKind =
    // ...
    | Lambda of
        parameters: (string * NativeType * NodeId) list *
        body: NodeId *
        captures: CaptureInfo list
```

### 3.2 Capture Analysis Algorithm

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/Expressions/Applications.fs`

The `checkLambda` function computes captures during type checking:

```fsharp
let checkLambda env builder lambdaExpr =
    // 1. Create parameter bindings in extended environment
    let envWithParams = addParameterBindings params env

    // 2. Check body expression in extended environment
    let bodyNode = checkExpr envWithParams builder bodyExpr

    // 3. Collect VarRefs in body (recursive traversal)
    let bodyVarRefs = collectVarRefs builder bodyNode.Id

    // 4. Identify captures: VarRefs not in parameter set
    let paramNames = params |> List.map fst |> Set.ofList
    let capturedNames = Set.difference bodyVarRefs paramNames

    // 5. Build CaptureInfo list from outer scope bindings
    let captures =
        capturedNames
        |> Set.toList
        |> List.choose (fun name ->
            match tryLookupBinding name env with  // Original env!
            | Some binding ->
                Some { Name = name
                       Type = binding.Type
                       IsMutable = binding.IsMutable
                       SourceNodeId = binding.NodeId }
            | None -> None)

    // 6. Create Lambda node with captures
    builder.Create(
        SemanticKind.Lambda(lambdaParams, bodyNode.Id, captures),
        funcType,
        range,
        children = paramNodeIds @ [bodyNode.Id])
```

**Key insight**: Captures are bindings referenced in the body that exist in the OUTER environment (before parameters were added), not the extended environment.

### 3.3 VarRef Collection Helper

```fsharp
let private collectVarRefs (builder: NodeBuilder) (nodeId: NodeId) : Set<string> =
    let nodes = builder.Nodes
    let rec collect (nodeId: NodeId) (acc: Set<string>) : Set<string> =
        match Map.tryFind nodeId nodes with
        | None -> acc
        | Some node ->
            let acc =
                match node.Kind with
                | SemanticKind.VarRef(name, _) -> Set.add name acc
                | _ -> acc
            node.Children |> List.fold (fun a childId -> collect childId a) acc
    collect nodeId Set.empty
```

### 3.4 Unit-Parameterized Lambda Types

**Issue**: `fun () -> body` must have type `unit -> bodyType`, not just `bodyType`.

**Fix**: In `checkLambda`, handle empty parameter list specially:

```fsharp
let funcType =
    if List.isEmpty paramTypes then
        NativeType.TFun(env.Globals.UnitType, bodyNode.Type)
    else
        mkFunctionType paramTypes bodyNode.Type
```

### 3.5 Lambda Children Structure

Lambda nodes must include parameter PatternBinding NodeIds in Children for proper traversal:

```fsharp
let paramNodeIds = lambdaParams |> List.map (fun (_, _, nodeId) -> nodeId)
builder.Create(
    SemanticKind.Lambda(lambdaParams, bodyNode.Id, captures),
    funcType,
    range,
    children = paramNodeIds @ [bodyNode.Id])  // Params THEN body
```

### 3.6 Traversal Updates

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/SemanticGraph.fs`

The `foldWithSCFRegions` function must walk Lambda parameter PatternBindings before the body region:

```fsharp
| SemanticKind.Lambda (params', bodyId, _captures), Some hook ->
    let parentId = node.Id
    // Walk parameter PatternBindings first (for SSA assignment)
    let paramNodeIds = params' |> List.map (fun (_, _, nodeId) -> nodeId)
    let state = paramNodeIds |> List.fold walk state
    // Lambda body is a region
    let state = hook.BeforeRegion state parentId LambdaBodyRegion
    let state = walk state bodyId
    let state = hook.AfterRegion state parentId LambdaBodyRegion
    state
```

## 4. Firefly/Alex Layer Implementation

### 4.1 SSA Assignment

**File**: `src/Alex/Preprocessing/SSAAssignment.fs`

For Lambda nodes with captures, SSAAssignment allocates:

```fsharp
// For captured mutable variables (need address SSA for ByRef)
%var_value    // SSA for the value (already assigned)
%var_addr     // SSA for the address (NEW - for ByRef capture)

// For Lambda with captures
%env_alloca   // SSA for llvm.alloca of env struct
%env_slot_N   // SSA for GEP to capture slot N
%closure      // SSA for final closure struct
```

**Implementation approach**: When visiting a Lambda node, check if `captures` is non-empty. If so, allocate additional SSAs for environment construction.

### 4.2 Closure Coeffect

**File**: `src/Alex/Preprocessing/SSAAssignment.fs` (or new `ClosureLayout.fs`)

```fsharp
type CaptureEntry = {
    Name: string
    SourceSSA: SSA           // SSA of captured variable's value
    AddressSSA: SSA option   // SSA of address (for ByRef captures)
    CaptureKind: CaptureKind // ByValue or ByRef
    OffsetInEnv: int         // Byte offset in struct
    SlotSSA: SSA             // SSA for GEP to this slot
}

type ClosureCoeffect = {
    LambdaNodeId: NodeId
    Captures: CaptureEntry list
    EnvStructType: string       // MLIR struct type string
    EnvAllocaSSA: SSA
    ClosureSSA: SSA
    CodePtrSSA: SSA
}
```

### 4.3 Lambda Witness - Closure Construction

**File**: `src/Alex/Witnesses/LambdaWitness.fs`

```fsharp
let witnessLambda (z: PSGZipper) (lambdaNodeId: NodeId) =
    match getLambdaCaptures lambdaNodeId z with
    | [] ->
        // No captures - simple function pointer
        let codePtr = getCodePointerForLambda lambdaNodeId z
        TRValue { SSA = codePtr; Type = TFunctionPointer }
    | captures ->
        // Has captures - emit flat closure
        emitFlatClosure z lambdaNodeId captures

let emitFlatClosure (z: PSGZipper) (lambdaId: NodeId) (captures: CaptureInfo list) =
    let coeffect = lookupClosureCoeffect lambdaId z

    // 1. Allocate environment struct
    emit $"  %%{coeffect.EnvAllocaSSA} = llvm.alloca 1 x {coeffect.EnvStructType}"

    // 2. Store each capture
    for entry in coeffect.Captures do
        emit $"  %%{entry.SlotSSA} = llvm.getelementptr %%{coeffect.EnvAllocaSSA}[0, {entry.OffsetInEnv}]"

        match entry.CaptureKind with
        | ByRef ->
            // Store pointer to mutable variable
            emit $"  llvm.store %%{entry.AddressSSA.Value}, %%{entry.SlotSSA}"
        | ByValue ->
            // Copy immutable value
            emit $"  llvm.store %%{entry.SourceSSA}, %%{entry.SlotSSA}"

    // 3. Build closure struct { code_ptr, env_ptr }
    emit $"  %%tmp = llvm.insertvalue undef[0], %%{coeffect.CodePtrSSA}"
    emit $"  %%{coeffect.ClosureSSA} = llvm.insertvalue %%tmp[1], %%{coeffect.EnvAllocaSSA}"

    TRValue { SSA = coeffect.ClosureSSA; Type = TClosureStruct }
```

### 4.4 Closure Invocation Witness

When a closure is **called**, extract and invoke:

```fsharp
let witnessClosureCall (z: PSGZipper) (closureSSA: SSA) (args: TRValue list) =
    let extractCode = freshSynthSSA z
    let extractEnv = freshSynthSSA z
    let resultSSA = freshSynthSSA z

    // Extract code pointer and environment
    emit $"  %%{extractCode} = llvm.extractvalue %%{closureSSA}[0]"
    emit $"  %%{extractEnv} = llvm.extractvalue %%{closureSSA}[1]"

    // Call with env as first argument
    let argList =
        extractEnv :: (args |> List.map (fun a -> a.SSA))
        |> List.map (sprintf "%%%s")
        |> String.concat ", "
    emit $"  %%{resultSSA} = llvm.call %%{extractCode}({argList})"

    TRValue { SSA = resultSSA; Type = returnType }
```

### 4.5 Lambda Body Code Generation

The lambda's implementation function receives the environment as its first parameter:

```fsharp
// Lambda: fun () -> count <- count + 1; count
// Generated function signature:
llvm.func @lambda_impl(%env: !llvm.ptr) -> i32 {
    // Load captured mutable variable (ByRef - it's a pointer)
    %count_ptr_addr = llvm.getelementptr %env[0, 0]
    %count_ptr = llvm.load %count_ptr_addr : !llvm.ptr

    // Read current value
    %count_val = llvm.load %count_ptr : i32

    // Increment
    %new_val = arith.addi %count_val, 1

    // Store back through pointer
    llvm.store %new_val, %count_ptr

    // Return new value
    llvm.return %new_val
}
```

### 4.6 PatternBinding Handler Update

**File**: `src/Alex/Traversal/FNCSTransfer.fs`

Lambda parameter PatternBindings need special handling - they define bindings but don't have values from the outer scope:

```fsharp
| SemanticKind.PatternBinding name ->
    match Map.tryFind name z.State.VarBindings with
    | Some (ssa, ty) ->
        z, TRValue { SSA = ssa; Type = ty }
    | None when name = "_" || name.StartsWith("_") ->
        z, TRVoid  // Discarded binding
    | None ->
        // Check if this is a Lambda parameter definition
        match z.Path with
        | step :: _ when isLambdaNode step.Parent ->
            z, TRVoid  // Lambda param - binding created elsewhere
        | _ ->
            z, TRError (sprintf "PatternBinding '%s' not found" name)
```

## 5. MLIR Output Specification

### 5.1 Closure Struct Type

```mlir
// Closure type: { function_ptr, env_ptr }
!closure_type = !llvm.struct<(ptr, ptr)>

// Environment struct (example for makeCounter)
!counter_env = !llvm.struct<(ptr)>  // Single ptr<int> for mutable count
```

### 5.2 Closure Construction

```mlir
// makeCounter returning a closure
llvm.func @makeCounter(%start: i32) -> !closure_type {
    // Allocate stack slot for mutable 'count'
    %count_slot = llvm.alloca 1 x i32
    llvm.store %start, %count_slot

    // Allocate environment struct
    %env = llvm.alloca 1 x !counter_env

    // Store pointer to count in environment
    %env_slot0 = llvm.getelementptr %env[0, 0]
    llvm.store %count_slot, %env_slot0

    // Build closure struct
    %closure_tmp = llvm.insertvalue undef : !closure_type[0], @counter_increment_impl
    %closure = llvm.insertvalue %closure_tmp[1], %env

    llvm.return %closure
}
```

### 5.3 Closure Implementation Function

```mlir
// The actual increment function
llvm.func @counter_increment_impl(%env: !llvm.ptr) -> i32 {
    // Get pointer to count from environment
    %count_ptr_addr = llvm.getelementptr %env[0, 0] : (!llvm.ptr) -> !llvm.ptr
    %count_ptr = llvm.load %count_ptr_addr : !llvm.ptr -> !llvm.ptr

    // Load, increment, store
    %old_val = llvm.load %count_ptr : !llvm.ptr -> i32
    %one = arith.constant 1 : i32
    %new_val = arith.addi %old_val, %one : i32
    llvm.store %new_val, %count_ptr : i32, !llvm.ptr

    llvm.return %new_val : i32
}
```

### 5.4 Closure Invocation

```mlir
// Calling counter()
%code_ptr = llvm.extractvalue %closure[0] : !closure_type -> !llvm.ptr
%env_ptr = llvm.extractvalue %closure[1] : !closure_type -> !llvm.ptr
%result = llvm.call %code_ptr(%env_ptr) : (!llvm.ptr) -> i32
```

## 6. fsnative-spec Updates

**File**: `~/repos/fsnative-spec/spec/closure-representation.md`

Document the normative closure specification:
- Flat closure requirement
- Capture mode determination (ByValue vs ByRef)
- Environment struct layout rules
- Calling convention (env as first parameter)

## 7. Reference Patterns

### 7.1 MLKit (FNCS Reference)

MLKit's `ClosExp.sml` shows flat closure construction:
- All captured values copied into environment record
- Environment allocated in current region
- Function pointer paired with environment pointer

**Key insight**: MLKit determines captures during lambda compilation, not as a separate pass.

### 7.2 FStar Free Variable Analysis

FStar's `FStarC_Syntax_Free.ml` provides the traversal pattern:
```ocaml
let rec free_names_and_uvs' tm =
  match tm with
  | Tm_name x -> singleton_bv x  (* Variable = potential free var *)
  | Tm_abs { bs; body; _ } ->
      let body_fvs = free_names_and_uvars body in
      aux_binders bs body_fvs  (* Exclude lambda params *)
```

### 7.3 Triton-CPU (MLIR Reference)

Triton-CPU's closure-like patterns in `TritonToTritonGPU/`:
- Struct type construction for captured state
- GEP-based field access
- Function pointer extraction and indirect calls

## 8. Validation

### 8.1 Sample Code

**File**: `samples/console/FidelityHelloWorld/11_Closures/Closures.fs`

```fsharp
let makeCounter (start: int) : (unit -> int) =
    let mutable count = start
    fun () ->
        count <- count + 1
        count

let makeGreeter (name: string) : (string -> string) =
    fun greeting -> $"{greeting}, {name}!"

let makeAccumulator (initial: int) : (int -> int) =
    let mutable total = initial
    fun n ->
        total <- total + n
        total

let makeRangeChecker (min: int) (max: int) : (int -> bool) =
    fun x -> x >= min && x <= max
```

### 8.2 Expected Output

```
=== Closures Test ===
--- Counter ---
First call: 1
Second call: 2
Third call: 3

--- Greeter ---
Hello, Alice!
Goodbye, Alice!
Welcome, Bob!

--- Accumulator ---
Add 10: 110
Add 25: 135
Add 5: 140

--- Range Checker ---
5 in range 10-20: false
15 in range 10-20: true
25 in range 10-20: false

--- Independent Closures ---
counter1: 1
counter2: 101
counter1: 2
counter2: 102
```

### 8.3 Regression Tests

ALL samples 01-10 must continue to pass after closure implementation:

```bash
for i in 01 02 03 04 05 06 07 08 09 10; do
  cd ~/repos/Firefly/samples/console/FidelityHelloWorld/${i}_*/
  ~/repos/Firefly/src/bin/Debug/net10.0/Firefly compile *.fidproj
  ./target/* || echo "FAIL: Sample $i"
done
```

## 9. Files to Create/Modify

### 9.1 FNCS (~/repos/fsnative/)

| File | Action | Purpose |
|------|--------|---------|
| `src/Compiler/Checking.Native/SemanticGraph.fs` | MODIFY | Add CaptureInfo type, update Lambda SemanticKind, fix traversal |
| `src/Compiler/Checking.Native/Expressions/Applications.fs` | MODIFY | Add collectVarRefs, implement capture analysis in checkLambda |
| `src/Compiler/Checking.Native/Expressions/Bindings.fs` | MODIFY | Update Lambda creation sites for Children structure |
| `src/Compiler/Checking.Native/Expressions/Coordinator.fs` | MODIFY | Update any Lambda pattern matches |
| `src/Compiler/Checking.Native/Expressions/Collections.fs` | MODIFY | Update any Lambda pattern matches |
| `src/Compiler/Checking.Native/FSharpNativeExpr.fs` | MODIFY | Update Lambda pattern matches for 3-tuple |

### 9.2 Firefly (~/repos/Firefly/)

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Preprocessing/SSAAssignment.fs` | MODIFY | Add closure-aware SSA assignments |
| `src/Alex/Traversal/FNCSTransfer.fs` | MODIFY | Update Lambda pattern match, fix PatternBinding handler |
| `src/Alex/Witnesses/LambdaWitness.fs` | MODIFY | Implement flat closure MLIR emission |
| `src/Alex/XParsec/PSGCombinators.fs` | MODIFY | Update Lambda pattern matching |
| `src/Firefly.fsproj` | MODIFY | Any new file references |
| `docs/Closure_Nanopass_Architecture.md` | UPDATED | Reflects correct architecture |

## 10. Implementation Checklist

### Phase 1: FNCS Changes (COMPLETE)
- [x] Define CaptureInfo type in SemanticGraph.fs
- [x] Update SemanticKind.Lambda to 3-tuple with captures
- [x] Implement collectVarRefs helper in Applications.fs
- [x] Add capture analysis to checkLambda
- [x] Fix unit-parameterized lambda types (`fun () -> ...`)
- [x] Fix Lambda Children to include parameter PatternBindings
- [x] Update Lambda traversal in foldWithSCFRegions
- [x] Update all Lambda pattern matches in FNCS

### Phase 2: Firefly Basic Support (COMPLETE)
- [x] Update Lambda pattern matches in FNCSTransfer.fs
- [x] Update Lambda pattern matches in PSGCombinators.fs
- [x] Fix PatternBinding handler for Lambda parameters
- [x] Verify samples 01-10 still pass

### Phase 3: Closure MLIR Emission (IN PROGRESS)
- [ ] Add closure SSA assignments in SSAAssignment.fs
- [ ] Define ClosureCoeffect type
- [ ] Implement closure struct type generation
- [ ] Implement environment allocation MLIR
- [ ] Implement capture store MLIR (ByValue and ByRef)
- [ ] Implement closure struct construction MLIR
- [ ] Implement closure invocation MLIR

### Phase 4: Lambda Body Generation (PENDING)
- [ ] Generate lambda implementation functions
- [ ] Handle environment parameter in lambda body
- [ ] Implement capture load from environment
- [ ] Handle mutable capture access (through pointer)

### Phase 5: Validation (PENDING)
- [ ] Sample 11 compiles without errors
- [ ] Sample 11 produces correct output
- [ ] All regression tests pass (samples 01-10)

## 11. Academic References

1. Shao & Appel (1994), "Space-Efficient Closure Representations" - Flat vs linked closures
2. Tofte & Talpin (1997), "Region-Based Memory Management" - Closures in regions
3. MLKit Programming with Regions (Elsman, 2021) - Production implementation
4. Perconti & Ahmed (2019), "Closure Conversion is Safe for Space" - Formal verification

## 12. Related PRDs

- **PRD-12**: Higher-Order Functions - Builds on closures for functions as values
- **PRD-17-19**: Async - Uses closures for continuation callbacks
- **PRD-29-31**: MailboxProcessor - Synthesizes closures + async + threading
