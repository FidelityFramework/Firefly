# PSG to MLIR Emission Engine Design

## Overview

This document describes the design for the emission engine that transforms an enriched PSG (Program Semantic Graph) into MLIR text. This is the "output side" of the compiler pipeline, consuming the PSG that has been constructed and enriched by the nanopass pipeline.

**Key Principle**: The emission engine is a *transcription* layer, not a *transformation* layer. The PSG, enriched by nanopasses, contains all the semantic information needed. The emitter's job is to faithfully render that information as MLIR.

## Related Documentation

- **Architecture**: `docs/Architecture_Canonical.md` - The two-layer model and non-dispatch principle
- **PSG Pipeline**: `docs/PSG_Nanopass_Architecture.md` - How the PSG is constructed and enriched
- **Alex Overview**: `docs/Alex_Architecture_Overview.md` - The targeting layer's role
- **Serena Memories**: `alex_zipper_architecture`, `architecture_principles`, `negative_examples`

## Design Principles

### 1. The Non-Dispatch Model

From `Architecture_Canonical.md`:

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

There is **NO central handler registry**. The emission engine:
- Uses the Zipper to traverse PSG structure
- Uses XParsec for local pattern matching at each node
- Dispatches extern primitives via `ExternDispatch.dispatch`
- Accumulates output via `MLIRBuilder`

The PSG structure itself drives emission. There is no routing table.

### 2. MLIR's Concrete SSA Mechanics

MLIR uses SSA (Static Single Assignment) form with specific mechanics:

```mlir
// Values are defined once with %name = operation
%0 = arith.constant 42 : i32
%1 = arith.addi %0, %0 : i32

// Block arguments handle control flow merges
^bb1(%arg0: i32, %arg1: i32):
  %2 = arith.addi %arg0, %arg1 : i32
  cf.br ^bb2(%2 : i32)

// Operations produce typed results
%result = func.call @foo(%arg) : (i32) -> i64
```

Key mechanics we must handle:
- **SSA numbering**: Fresh `%N` for each defined value
- **Type annotations**: Every value carries its type
- **Block structure**: Labels, arguments, terminators
- **Operation syntax**: `%result = dialect.op operands : types`

### 3. Layer Separation

The emission engine does NOT:
- Pattern match on symbol names (e.g., "Alloy.Console.Write")
- Know about specific libraries
- Make targeting decisions based on function names
- Contain library-specific special cases

The emission engine DOES:
- Follow PSG structure (node kinds, children, edges)
- Use `node.Operation` classification (set by nanopass)
- Dispatch extern primitives by entry point name
- Generate MLIR based on structural patterns

## Architecture

### Component Roles

```
PSG (enriched)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PSGZipper                                              │
│  - Provides "attention" (focus + context)               │
│  - Carries EmissionState (SSA counter, strings, etc.)   │
│  - foldPreOrder / foldPostOrder for traversal           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Emission Logic (at each node)                          │
│  - Pattern match on node.SyntaxKind                     │
│  - Use XParsec to match children structure              │
│  - Generate MLIR via MLIRBuilder monad                  │
└─────────────────────────────────────────────────────────┘
    │
    ├──▶ Regular nodes → Direct MLIR emission
    │
    └──▶ Extern primitives → ExternDispatch.dispatch
                                    │
                                    ▼
                            Platform Bindings
                            (syscalls, API calls)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  MLIRBuilder                                            │
│  - Accumulates MLIR text                                │
│  - Manages SSA counter                                  │
│  - Handles indentation                                  │
│  - Registers global strings                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
MLIR Module (text)
```

### Existing Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| `PSGZipper` | `Alex/Traversal/PSGZipper.fs` | Bidirectional PSG traversal |
| `PSGXParsec` | `Alex/Traversal/PSGXParsec.fs` | Child pattern matching combinators |
| `MLIRBuilder` | `Alex/CodeGeneration/MLIRBuilder.fs` | MLIR monad and operations |
| `ExternDispatch` | `Alex/Bindings/BindingTypes.fs` | Extern primitive registry |
| `ConsoleBindings` | `Alex/Bindings/Console/ConsoleBindings.fs` | Console I/O bindings |
| `TimeBindings` | `Alex/Bindings/Time/TimeBindings.fs` | Time primitive bindings |
| `PSGPatterns` | `Alex/Patterns/PSGPatterns.fs` | Node predicates and extractors |

### Implementation Location: CompilationOrchestrator

The emission logic lives **inside Alex**, not in a separate emission module. Specifically, the `generateMLIRViaAlex` function in `CompilationOrchestrator.fs` orchestrates the emission by:

1. Using `PSGZipper` to traverse from entry points
2. Applying local XParsec patterns at each node
3. Dispatching extern primitives via `ExternDispatch`
4. Accumulating output via `MLIRBuilder`

**IMPORTANT**: There is NO separate "emitter" or "emission module". Creating such a module would recreate the antipattern that was removed (PSGEmitter, PSGScribe). The emission is an internal concern of Alex, expressed through the Zipper fold.

## Emission Strategy

### Entry Point

```fsharp
/// Emit MLIR from an enriched PSG
let emitMLIR (psg: ProgramSemanticGraph) (config: EmissionConfig) : MLIR<unit>
```

The emitter:
1. Creates a Zipper at each entry point
2. Traverses reachable nodes
3. Emits MLIR for each node based on its kind
4. Returns accumulated MLIR text

### Traversal Order

For a function body, we use **pre-order traversal** with the Zipper's fold:

```fsharp
let emitFunction (zipper: PSGZipper) : MLIR<Val> = mlir {
    // Pre-order: emit node, then children
    PSGZipper.foldPreOrder emitNode zipper
}
```

Pre-order ensures:
- Parent context is established before children
- Let bindings are emitted before their uses
- Sequential expressions flow top-to-bottom

### Node Kind Dispatch

At each node, pattern match on `SyntaxKind` to determine emission:

```fsharp
let emitNode (zipper: PSGZipper) : MLIR<Val option> = mlir {
    let node = zipper.Focus

    // Skip unreachable nodes
    if not node.IsReachable then
        return None
    else
        match node.SyntaxKind with
        | "Binding:EntryPoint" ->
            let! result = emitEntryPointBinding zipper
            return Some result
        | "Binding" | "Binding:Function" ->
            let! result = emitFunctionBinding zipper
            return Some result
        | "Sequential" ->
            let! result = emitSequential zipper
            return Some result
        | "App:FunctionCall" ->
            let! result = emitFunctionCall zipper
            return Some result
        | sk when sk.StartsWith("Const:") ->
            let! result = emitConstant zipper
            return Some result
        | sk when sk.StartsWith("Ident") ->
            let! result = emitIdentifier zipper
            return Some result
        // ... other node kinds
        | _ ->
            do! comment $"// Unhandled: {node.SyntaxKind}"
            return None
}
```

This is NOT a central dispatch table - it's local pattern matching at each traversal step.

### Function Call Handling

When encountering `App:FunctionCall`, the emitter must determine how to handle it:

```fsharp
let emitFunctionCall (zipper: PSGZipper) : MLIR<Val> = mlir {
    let node = zipper.Focus

    // Check if this is an extern primitive call
    match tryExtractExternPrimitive node with
    | Some extern ->
        // Dispatch to platform binding
        let! result = ExternDispatch.dispatch extern
        match result with
        | Emitted val -> return val
        | NotSupported msg ->
            do! errorComment msg
            return! fail msg

    | None ->
        // Regular function call - check Operation classification
        match node.Operation with
        | Some (OperationKind.Console op) ->
            // Inline the console operation implementation
            let! result = emitConsoleOp op zipper
            return result

        | Some (OperationKind.Arithmetic op) ->
            // Emit arithmetic operation
            let! result = emitArithmeticOp op zipper
            return result

        | Some (OperationKind.NativePtr op) ->
            // Emit pointer operation
            let! result = emitPointerOp op zipper
            return result

        | _ ->
            // Follow call edge to callee, emit inline
            let! result = emitInlinedCall zipper
            return result
}
```

### Platform Binding Detection

Platform bindings are detected by examining the FCS symbol's containing module. The `Platform.Bindings` module convention (BCL-free) replaces the old DllImport pattern:

```fsharp
/// Extract platform binding info from a PSG node if it's in Platform.Bindings module
let tryExtractPlatformBinding (node: PSGNode) : PlatformBinding option =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        // Check if function is in a Platform.Bindings module
        let containingModule = mfv.DeclaringEntity
        match containingModule with
        | Some entity when isPlatformBindingsModule entity.FullName ->
            // Extract binding info from module structure
            // e.g., "Platform.Bindings.Console.writeBytes" → module: Console, function: writeBytes
            let modulePath = extractModulePath entity.FullName
            let functionName = mfv.LogicalName

            Some {
                ModulePath = modulePath      // e.g., ["Console"] or ["HAL"; "GPIO"]
                FunctionName = functionName  // e.g., "writeBytes" or "init"
                Args = []                    // Populated during emission
                ReturnType = mapType mfv.ReturnParameter.Type
            }
        | _ -> None
    | _ -> None

/// Check if a module path indicates Platform.Bindings
let isPlatformBindingsModule (fullName: string) : bool =
    fullName.StartsWith("Platform.Bindings") ||
    fullName.StartsWith("Alloy.Platform.Bindings")
```

### Inlining Strategy

For reachable non-extern functions, we inline their implementation:

```fsharp
let emitInlinedCall (zipper: PSGZipper) : MLIR<Val> = mlir {
    // Find the callee's definition in the PSG
    let calleeNode = findCalleeDefinition zipper

    // Create a new zipper focused on the callee
    let calleeZipper = PSGZipper.create zipper.Graph calleeNode.Id

    // Emit the callee's body inline
    let! result = emitFunctionBody calleeZipper
    return result
}
```

This approach:
- Follows the PSG structure (no library-specific logic)
- Enables whole-program optimization
- Naturally handles Alloy's decomposition to primitives

## Emission Patterns

### Pattern 1: Entry Point Function

```fsharp
// PSG: Binding:EntryPoint [main]
//   └── Pattern:LongIdent:main
//       └── Pattern:Named:argv
//   └── Sequential (body)

let emitEntryPointBinding (zipper: PSGZipper) : MLIR<Val> = mlir {
    do! emitLine "llvm.func @main() -> i32 {"
    do! pushIndent

    // Emit function body
    let bodyZipper = PSGZipper.downTo zipper "Sequential"
    let! result = emitSequential bodyZipper

    // Ensure return
    do! emitLine $"llvm.return {result.SSA} : i32"

    do! popIndent
    do! emitLine "}"

    return result
}
```

### Pattern 2: Sequential Expressions

```fsharp
// PSG: Sequential
//   └── child1 (executed for effect)
//   └── child2 (executed for effect)
//   └── childN (result value)

let emitSequential (zipper: PSGZipper) : MLIR<Val> = mlir {
    let children = PSGZipper.children zipper

    let mutable lastResult = Val.unit
    for childId in children do
        let childZipper = PSGZipper.downTo zipper childId
        let! result = emitNode childZipper
        lastResult <- result |> Option.defaultValue Val.unit

    return lastResult
}
```

### Pattern 3: String Constants

```fsharp
// PSG: Const:String "Hello, World!"

let emitStringConstant (zipper: PSGZipper) : MLIR<Val> = mlir {
    let node = zipper.Focus
    match node.ConstantValue with
    | Some (StringValue s) ->
        // Register string as global
        let! globalName = registerString s

        // Get pointer to string data
        let! ptr = llvm.addressof globalName (Ptr (Int I8))

        // Build string struct {ptr, length} - native string semantics
        let! str = buildStringStruct ptr (String.length s)
        return str

    | _ -> return! fail "Expected string constant"
}
```

### Pattern 4: Let Bindings

```fsharp
// PSG: LetOrUse:Let
//   └── Binding [name]
//       └── Pattern:Named:name
//       └── (rhs expression)
//   └── (body expression)

let emitLetBinding (zipper: PSGZipper) : MLIR<Val> = mlir {
    // Get binding and body children
    let bindingZipper = PSGZipper.downTo zipper "Binding"
    let bodyZipper = PSGZipper.right bindingZipper |> Option.get

    // Emit RHS expression
    let rhsZipper = getBindingRHS bindingZipper
    let! rhsVal = emitNode rhsZipper

    // Record SSA for the bound name
    let boundName = extractBoundName bindingZipper
    do! recordNodeSSA boundName.Id rhsVal

    // Emit body with binding in scope
    let! bodyVal = emitNode bodyZipper
    return bodyVal
}
```

### Pattern 5: Mutable Bindings

```fsharp
// PSG: Binding:Mutable [name]
//   └── Pattern:Named:name
//   └── (initial value)

let emitMutableBinding (zipper: PSGZipper) : MLIR<Val> = mlir {
    // Get initial value
    let initZipper = getBindingRHS zipper
    let! initVal = emitNode initZipper

    // Allocate stack slot
    let! slot = llvm.alloca initVal.Type

    // Store initial value
    do! llvm.store initVal slot

    // Record the slot (not the value) for this binding
    let boundName = extractBoundName zipper
    do! recordNodeSSA boundName.Id slot

    return Val.unit
}
```

### Pattern 6: Mutable Assignment

```fsharp
// PSG: MutableSet [name]
//   └── (new value expression)

let emitMutableSet (zipper: PSGZipper) : MLIR<Val> = mlir {
    // Get the slot for this mutable
    let varName = extractMutableSetName zipper.Focus
    let! slot = lookupNodeSSA varName

    // Emit new value
    let valueZipper = PSGZipper.down zipper |> Option.get
    let! newVal = emitNode valueZipper

    // Store to slot
    do! llvm.store newVal slot

    return Val.unit
}
```

## Validation Samples

The emission engine must successfully compile these samples:

### 01_HelloWorldDirect

```fsharp
module Examples.HelloWorldDirect
open Alloy

[<EntryPoint>]
let main argv =
    Console.Write "Hello, World!"
    Console.WriteLine ""
    0
```

**Expected MLIR structure**:
```mlir
module {
    llvm.mlir.global private constant @str0("Hello, World!\00") { alignment = 1 }
    llvm.mlir.global private constant @str1("\00") { alignment = 1 }

    llvm.func @main() -> i32 {
        // Console.Write "Hello, World!"
        // ... inlines to writeBytes syscall ...

        // Console.WriteLine ""
        // ... inlines to writeBytes + newline ...

        %ret = arith.constant 0 : i32
        llvm.return %ret : i32
    }
}
```

### 02_HelloWorldSaturated

Adds let bindings and string interpolation.

### 03_HelloWorldHalfCurried

Adds pipe operators and partial application.

## Implementation Roadmap

### Phase 1: Core Infrastructure

1. Implement extern primitive detection from FCS attributes in `PSGPatterns.fs`
2. Wire up Zipper traversal in `generateMLIRViaAlex` (CompilationOrchestrator.fs)
3. Add emission helper functions to `PSGZipper.fs` if needed

### Phase 2: Basic Emission

4. Emit entry point function structure
5. Emit constants (Int, String, Unit, Byte)
6. Emit sequential expressions
7. Emit identifier references (SSA lookup)

### Phase 3: Function Calls

8. Emit function call structure
9. Implement inlining of reachable functions
10. Connect ExternDispatch for primitives

### Phase 4: Control Flow

11. Emit let bindings
12. Emit mutable bindings and assignments
13. Emit conditionals (if/then/else)
14. Emit loops (while, for)

### Phase 5: Validation

15. Validate 01_HelloWorldDirect end-to-end
16. Validate 02_HelloWorldSaturated end-to-end
17. Validate 03_HelloWorldHalfCurried end-to-end
18. Document lessons learned

## Lessons for PSG Construction

After validating the samples, we will document:

1. What PSG structure the emitter needs
2. Where the emitter had to "reach around" missing information
3. What nanopass enrichments would simplify emission
4. How a "front-end zipper" could mirror the emission zipper

These lessons will inform the architectural revision of PSG construction to achieve symmetry between input and output sides of the pipeline.

---

*This design follows the Option C approach: implement the output side first, learn from it, then apply lessons to refactor the input side for architectural symmetry.*
