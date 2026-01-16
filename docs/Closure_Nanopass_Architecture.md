# Closure Architecture

> **MLKit-style flat closures with FNCS-computed captures.**

## 1. Executive Summary

F# Native uses **MLKit-style flat closures** where all captured variables are stored inline in the closure struct, not via pointer chains to enclosing environments.

**Key Architectural Decision**: Capture analysis is **scope analysis**, and scope is resolved during type checking. Therefore, **capture analysis belongs in FNCS**, not Firefly. FNCS computes captures during PSG construction and includes them directly in `SemanticKind.Lambda`. Alex only handles SSA assignment and struct layout for code generation.

## 2. Layer Responsibilities

| Layer | Responsibility |
|-------|---------------|
| **FNCS** | Compute captures during scope analysis, embed in `SemanticKind.Lambda` |
| **Alex/SSAAssignment** | Assign SSAs for closure structs and environment slots |
| **Alex/Witnesses** | Emit MLIR for closure construction and invocation |

**Capture analysis is NOT a Firefly nanopass.** The PSG arrives from FNCS with complete capture information.

## 3. Academic Background

### 3.1 Why Flat Closures?

Based on Shao & Appel (1994), "Space-Efficient Closure Representations":

| Property | Flat Closures | Linked Closures |
|----------|---------------|-----------------|
| Memory layout | All captures inline | Pointer to outer env |
| Access time | O(1) direct offset | O(depth) chain walk |
| Space safety | ✅ Safe for GC | ❌ Keeps outer env alive |
| Creation cost | Copy all captures | Store one pointer |
| Cache behavior | Contiguous | Scattered |

**Flat is superior for fsnative** because:
1. No GC - space safety is structural
2. Region allocation - no heap fragmentation
3. Predictable performance - no chain traversal

### 3.2 MLKit Heritage

The MLKit compiler (Tofte, Elsman et al.) pioneered:
- Region-based memory management for ML
- All values (including closures) allocated in regions
- Compile-time lifetime inference

F# Native follows this model: closures are stack or region allocated, never GC heap.

## 4. Memory Layout

### 4.1 Flat Closure Structure

```
Closure Structure
┌─────────────────────────────────────────────────────────┐
│ code_ptr (8 bytes): pointer to lambda implementation   │
├─────────────────────────────────────────────────────────┤
│ env (variable size): captured values inline            │
│   ├── capture_0: T₀                                    │
│   ├── capture_1: T₁                                    │
│   └── ...                                              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Capture Modes

| Variable Kind | Capture Mode | In Env | Semantics |
|---------------|--------------|--------|-----------|
| Immutable `let x = ...` | ByValue | `T` | Copy value |
| Mutable `let mutable x = ...` | ByRef | `ptr<T>` | Pointer to slot |
| Ref cell `let x = ref ...` | ByValue | `ref<T>` | Copy cell ptr |

**Critical**: Mutable variables are captured by **reference** (pointer to stack slot), not by value. This enables mutation through the closure.

## 5. FNCS Capture Analysis

### 5.1 SemanticKind.Lambda Definition

FNCS computes captures during `checkLambda` in Applications.fs:

```fsharp
type CaptureInfo = {
    Name: string
    Type: NativeType
    IsMutable: bool
    SourceNodeId: NodeId option  // Where the captured var is defined
}

// Lambda now includes captures computed during scope analysis:
| Lambda of parameters: (string * NativeType * NodeId) list
         * body: NodeId
         * captures: CaptureInfo list
```

### 5.2 Capture Computation Algorithm

FNCS uses free variable analysis during lambda checking:

```fsharp
let checkLambda ... =
    // 1. Create parameter bindings in extended environment
    let envWithParams = addParameterBindings params env

    // 2. Check body expression in extended environment
    let bodyNode = checkExpr envWithParams builder bodyExpr

    // 3. Collect VarRefs in body
    let bodyVarRefs = collectVarRefs builder bodyNode.Id

    // 4. Identify captures: VarRefs not in parameter set
    let paramNames = params |> List.map fst |> Set.ofList
    let capturedNames = Set.difference bodyVarRefs paramNames

    // 5. Build CaptureInfo list from outer scope bindings
    let captures =
        capturedNames
        |> Set.toList
        |> List.choose (fun name ->
            match tryLookupBinding name env with  // Original env, not envWithParams
            | Some binding ->
                Some { Name = name
                       Type = binding.Type
                       IsMutable = binding.IsMutable
                       SourceNodeId = binding.NodeId }
            | None -> None)

    // 6. Create Lambda node with captures
    builder.Create(SemanticKind.Lambda(paramNodes, bodyNode.Id, captures), ...)
```

**Key insight**: Captures are determined by checking which VarRefs in the body refer to bindings in the *outer* environment (before parameters were added).

## 6. Alex SSA Assignment

### 6.1 Lambda SSA Requirements

For each Lambda with captures, SSAAssignment allocates:

```fsharp
// For a Lambda capturing 2 variables:
%env_alloca    // SSA for llvm.alloca of env struct
%env_slot_0    // SSA for GEP to first capture slot
%env_slot_1    // SSA for GEP to second capture slot
%closure       // SSA for final closure struct
```

### 6.2 Captured Variable SSA

For mutable variables that are captured:
```fsharp
%var_value     // SSA for the value (already assigned)
%var_addr      // SSA for the address (for by-ref capture)
```

### 6.3 Reading Captures from PSG

SSAAssignment reads captures directly from `SemanticKind.Lambda`:

```fsharp
match node.Kind with
| SemanticKind.Lambda(params, body, captures) when not (List.isEmpty captures) ->
    // Assign SSAs for closure construction
    let envAllocaSSA = freshSSA ()
    let closureSSA = freshSSA ()
    let slotSSAs = captures |> List.mapi (fun i _ -> freshSSA ())
    // ... record in coeffect map
```

## 7. Witness Emission

### 7.1 Lambda Witness

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
```

### 7.2 Flat Closure Emission

```fsharp
let emitFlatClosure (z: PSGZipper) (lambdaId: NodeId) (captures: CaptureInfo list) =
    let envAllocaSSA = lookupEnvAllocaSSA lambdaId z
    let closureSSA = lookupClosureSSA lambdaId z

    // 1. Allocate environment struct
    emit $"  %{envAllocaSSA} = llvm.alloca 1 x {envStructType captures}"

    // 2. Store each capture
    for (i, cap) in List.indexed captures do
        let slotSSA = lookupCaptureSlotSSA lambdaId i z
        emit $"  %{slotSSA} = llvm.getelementptr %{envAllocaSSA}[0, {i}]"

        if cap.IsMutable then
            // Store pointer to mutable variable
            let addrSSA = lookupAddressSSA cap.SourceNodeId z
            emit $"  llvm.store %{addrSSA}, %{slotSSA}"
        else
            // Copy immutable value
            let valueSSA = lookupValueSSA cap.SourceNodeId z
            emit $"  llvm.store %{valueSSA}, %{slotSSA}"

    // 3. Build closure struct { code_ptr, env_ptr }
    let codePtr = getCodePointerForLambda lambdaId z
    emit $"  %tmp = llvm.insertvalue undef[0], %{codePtr}"
    emit $"  %{closureSSA} = llvm.insertvalue %tmp[1], %{envAllocaSSA}"

    TRValue { SSA = closureSSA; Type = TClosureStruct }
```

### 7.3 Closure Invocation

```fsharp
let witnessClosureCall (z: PSGZipper) (closureSSA: SSA) (args: SSA list) =
    // Extract code pointer and environment
    let extractCode = freshSynthSSA z
    let extractEnv = freshSynthSSA z
    emit $"  %{extractCode} = llvm.extractvalue %{closureSSA}[0]"
    emit $"  %{extractEnv} = llvm.extractvalue %{closureSSA}[1]"

    // Call with env as first argument
    let resultSSA = freshSynthSSA z
    let argList = extractEnv :: args |> List.map (sprintf "%%") |> String.concat ", "
    emit $"  %{resultSSA} = llvm.call %{extractCode}({argList})"

    TRValue { SSA = resultSSA; Type = returnType }
```

## 8. Pipeline Flow

```
F# Source
    │
    ▼
FNCS (checkLambda with scope analysis)
    │
    ├─ Computes captures via free variable analysis
    ├─ Creates SemanticKind.Lambda(params, body, captures)
    │
    ▼
PSG with complete closure information
    │
    ▼
Alex/SSAAssignment
    │
    ├─ Reads captures from SemanticKind.Lambda
    ├─ Assigns env_alloca, slot, and closure SSAs
    │
    ▼
Alex/Witnesses
    │
    ├─ Observes Lambda node with captures
    ├─ Emits flat closure MLIR
    │
    ▼
MLIR → LLVM → Native Binary
```

## 9. References

- Shao & Appel (1994), "Space-Efficient Closure Representations"
- MLKit Programming with Regions (Tofte, Elsman)
- fsnative-spec `spec/closure-representation.md`
