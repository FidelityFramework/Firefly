# NEGATIVE EXAMPLES: What NOT To Do

These are real mistakes made during development. **DO NOT REPEAT THEM.**

## Mistake 1: Adding Alloy-specific logic to MLIR generation

```fsharp
// WRONG - MLIR generation should not know about Alloy
match symbolName with
| Some name when name = "Alloy.Console.Write" ->
    generateConsoleWrite psg ctx node  // Special case!
| Some name when name = "Alloy.Console.WriteLine" ->
    generateConsoleWriteLine psg ctx node  // Another special case!
```

**Why this is wrong**: MLIR generation is now coupled to Alloy's namespace structure. If Alloy changes, the compiler breaks.

**The fix**: Alloy functions should have real implementations. The PSG should contain the full call graph. The Zipper walks the graph and Bindings generate MLIR based on node structure.

## Mistake 2: Stub implementations in Alloy

```fsharp
// WRONG - This is a stub that expects compiler magic
let inline WriteLine (s: string) : unit =
    () // Placeholder - Firefly compiler handles this
```

**Why this is wrong**: The PSG will show `Const:Unit` as the function body. There's no semantic structure for Alex to work with.

**The fix**: Real implementation that decomposes to primitives:
```fsharp
// RIGHT - Real implementation using lower-level functions
let inline WriteLine (s: string) : unit =
    writeln s  // Calls writeStrOut -> writeBytes (the actual syscall primitive)
```

## Mistake 3: Putting nanopass logic in MLIR generation

```fsharp
// WRONG - Importing nanopass modules into code generation
open Core.PSG.Nanopass.DefUseEdges

// WRONG - Building indices during MLIR generation
let defIndex = buildDefinitionIndex psg
```

**Why this is wrong**: Nanopasses run BEFORE MLIR generation. They enrich the PSG. Code generation should consume the enriched PSG, not run nanopass logic.

## Mistake 4: Adding mutable state tracking to code generation

```fsharp
// WRONG - Code generation tracking mutable bindings
type GenerationContext = {
    // ...
    MutableBindings: Map<string, Val>  // NO! This is transformation logic
}
```

**Why this is wrong**: Mutable variable handling should be resolved in the PSG via nanopasses. Code generation should just follow edges to find values.

## Mistake 5: Creating a Central Dispatch/Emitter/Scribe

```fsharp
// WRONG - Central dispatch registry
module PSGEmitter =
    let handlers = Dictionary<string, NodeHandler>()

    let registerHandler prefix handler =
        handlers.[prefix] <- handler

    let emit node =
        let prefix = getKindPrefix node.SyntaxKind
        match handlers.TryGetValue(prefix) with
        | true, handler -> handler node
        | _ -> defaultHandler node
```

**Why this is wrong**:
- This antipattern was removed TWICE (PSGEmitter, then PSGScribe)
- It collects "special case" routing too early in the pipeline
- It inevitably attracts library-aware logic ("if ConsoleWrite then...")
- The centralization belongs at OUTPUT (MLIR Builder), not at DISPATCH

**The fix**: NO central dispatcher. The Zipper folds over PSG structure. XParsec matches locally at each node. Bindings are looked up by extern primitive entry point. MLIR Builder accumulates the output.

## Mistake 6: String-based parsing or name matching

```fsharp
// WRONG - String matching on symbol names
if symbolName.Contains("Console.Write") then ...

// WRONG - Hardcoded library paths
| Some name when name.StartsWith("Alloy.") -> ...

// RIGHT - Pattern match on PSG node structure
match node.SyntaxKind with
| "App:FunctionCall" -> processCall zipper bindings
| "WhileLoop" -> processWhileLoop zipper bindings
```

## Mistake 7: Premature Centralization

Pooling decision-making logic too early in the pipeline.

**Wrong**: Creating a router/dispatcher that decides what to do with each node kind
**Right**: Let PSG structure drive emission; centralization only at MLIR output

The PSG, enriched by nanopasses, carries enough information that emission is deterministic. No routing decisions needed.

## Mistake 8: Silent Failures in Code Generation (CRITICAL)

```fsharp
// WRONG - Silently return Void when function not found
and emitInlinedCall (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    // ...
    match funcBinding with
    | Some binding -> // ... emit code
    | None ->
        printfn "[GEN] Function not found in PSG: %s" name  // Just prints!
        Void  // *** SILENT FAILURE - continues compilation ***
```

**What happened**: During HelloWorldDirect compilation, the output showed:
```
[GEN] Function not found in PSG: System.Object.ReferenceEquals
[GEN] Function not found in PSG: Microsoft.FSharp.Core.Operators.``not``
```

The code printed warnings but returned `Void` and continued. The result:
- Conditional check silently failed
- Only a newline was written (not "Hello, World!")
- Binary segfaulted on `ret` instruction

**Why this is wrong**: 
- Compilers exist to surface errors. Silent failures hide bugs behind more bugs.
- The root cause (unresolved function) manifested as a symptom (segfault)
- Hours were spent chasing the segfault instead of fixing the real issue

**The fix**:
```fsharp
// RIGHT - Return EmitError and propagate it
| None ->
    EmitError (sprintf "Function not found in PSG: %s - cannot generate code" name)
    // Caller MUST handle EmitError and fail compilation
```

**The principle**: When code generation cannot proceed, it MUST emit an error that halts compilation. Never swallow failures with `printfn` + `Void`.

**When you see this pattern**: STOP EVERYTHING. Do not run the binary. Do not chase symptoms. Fix the error propagation first.

---

## Mistake 9: Hardcoding Types Instead of Using Architectural Type Flow (CRITICAL)

This antipattern was discovered and fixed in January 2026 during the "fat pointer string" debugging session.

```fsharp
// WRONG - Hardcoding type mappings instead of using FNCS type information
let rec mapType (ty: NativeType) : MLIRType =
    match ty with
    | ...
    | "string" -> Pointer  // IGNORES that FNCS knows strings are fat pointers!
```

```fsharp
// WRONG - Hardcoding function signatures instead of using node.Type
| SemanticKind.Lambda _ ->
    let funcName = name
    let signature = "(!llvm.ptr) -> !llvm.ptr"  // IGNORES valueNode.Type!
    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
    zipper', TRValue ("@" + funcName, "!llvm.ptr")
```

**The Principled Architecture**:
- FNCS defines types with correct layouts: `stringTyCon = mkTypeConRef "string" 0 (TypeLayout.Inline(16, 8))` — fat pointer
- PSG nodes carry `Type: NativeType` with full type information
- `mapType` converts NativeType → MLIRType
- `Serialize.mlirType` converts MLIRType → string

**What Was Wrong**:
1. `mapType` returned `Pointer` for strings instead of the fat pointer struct `NativeStrType`
2. External function signatures were hardcoded as `"(!llvm.ptr) -> !llvm.ptr"` instead of derived from `valueNode.Type`
3. Four separate locations had the same hardcoded signature cruft
4. The type information from FNCS was being discarded at the Alex boundary

**Why This Pattern Emerges**:
- When something doesn't work, the instinct is to add a "fallback" or "default"
- Fallbacks accumulate as cruft that ignores the principled type flow
- Each patch makes the next problem harder to diagnose

**The Fix Pattern**:
1. **Trace upstream**: The type information exists in FNCS. Where is it discarded?
2. **Remove, don't fix**: Don't make the fallback use correct types. REMOVE the fallback.
3. **Trust the zipper**: The codata/pull model means the graph contains everything
4. **Use existing architecture**: `mapType` + `Serialize.mlirType` on actual `node.Type`

```fsharp
// RIGHT - Use the type information the architecture provides
| "string" -> NativeStrType  // Fat pointer {ptr: *u8, len: i64}

// RIGHT - Derive signature from actual node type
| SemanticKind.Lambda _ ->
    let funcName = name
    let signature =
        match valueNode.Type with
        | NativeType.TFun(paramTy, retTy) ->
            sprintf "(%s) -> %s"
                (Serialize.mlirType (mapType paramTy))
                (Serialize.mlirType (mapType retTy))
        | _ -> // Error, not fallback
    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
    ...
```

**The Principle**: The architecture provides type information at every layer. When code ignores this and hardcodes types, it's always a bug that will manifest as type mismatches downstream. The fix is never to "improve the hardcoding" but to REMOVE it and use what the architecture provides.

**Remediation Checklist**:
1. Consult Serena memories on architecture before any fix
2. Identify where type information flows from FNCS through PSG to Alex
3. Find where it's being discarded or ignored
4. Remove cruft entirely - don't patch it
5. Trust the zipper's attention mechanism over the PSG

---

## Mistake 10: Imperative "Push" Patterns vs Codata "Pull" Model

Related to Mistake 9, this antipattern involves adding imperative traversal logic and fallback paths instead of trusting the zipper's pull model.

```fsharp
// WRONG - "Wasn't traversed yet" fallback with imperative assumption
| None ->
    // DEFERRED RESOLUTION: The binding's value wasn't traversed yet
    // Check if the value is a Literal (constant) or Lambda (function)
    match SemanticGraph.tryGetNode valueNodeId graph with
    | Some valueNode ->
        // Create extern declaration as workaround...
```

**Why this is wrong**:
- The zipper provides "attention" to any part of the graph
- There's no "wasn't traversed yet" if you use the zipper correctly
- The graph contains everything; the zipper lets you navigate to it

**The Codata/Pull Principle**:
- Don't track what was "already traversed"
- Don't create fallbacks for "not yet seen" nodes
- The graph is complete; witness what you need when you need it
- The zipper carries accumulated observations; recall prior observations via state

**When you find yourself writing "if not traversed yet, then fallback"**:
STOP. You're not using the zipper correctly. The information is available.

---

## Mistake 11: Wrong Binding Layer (BCL Stubs for Platform Operations)

This antipattern was identified in January 2026 during the unified binding architecture analysis.

```fsharp
// WRONG - Platform.Bindings with BCL stubs
module Platform.Bindings =
    let writeBytes fd buffer count : int = Unchecked.defaultof<int>  // BCL!
    let readBytes fd buffer maxCount : int = Unchecked.defaultof<int>
```

**Why this is wrong**:
1. `Unchecked.defaultof` is a BCL function - violates BCL-free principle
2. These are FNCS intrinsics (`Sys.write`, `Sys.read`) - Alloy shouldn't re-declare them
3. Creates semantic vacuum - PSG shows stub body, not meaningful operation
4. Forces Alex to do name-based dispatch ("if Platform.Bindings.writeBytes then...")

**The Three-Layer Architecture**:

| Layer | What | Mechanism |
|-------|------|-----------|
| Layer 1 | FNCS Intrinsics | FNCS emits directly - `Sys.write`, `NativePtr.set` |
| Layer 2 | Binding Libraries | Quotation semantic carriers - Farscape-generated |
| Layer 3 | User Code | Uses Layer 1 & 2 - Alloy, applications |

**The Fix**:
```fsharp
// RIGHT - Alloy uses FNCS intrinsics
module Console =
    let Write (s: string) =
        let ptr = String.asPtr s
        let len = String.length s
        Sys.write 1 ptr len |> ignore  // FNCS intrinsic, not stub
```

**When to use each layer**:
- **Layer 1 (Intrinsics)**: Operations native to the type universe - `Sys.*`, `NativePtr.*`
- **Layer 2 (Binding Libraries)**: External bindings with rich metadata - GTK, CMSIS
- **Layer 3 (User Code)**: Everything else - uses Layer 1 & 2

**See**: Firefly `binding_architecture_unified` memory for complete architecture.

---

## Mistake 12: TVar → Pointer Default for Polymorphic Operators

Discovered in January 2026 during SCF dialect work when `concat2` (string concatenation) failed.

```fsharp
// In mapType:
| NativeType.TVar _ -> Pointer  // ALL type variables become pointers
```

```fsharp
// What happens with op_Addition : 'a -> 'a -> 'a
let len1 = s1.Length    // Type: int (i64)
let len2 = s2.Length    // Type: int (i64)
let total = len1 + len2 // Type: 'a -> 'a -> 'a instantiated to int
```

**The Problem**:
1. `op_Addition` has polymorphic type `'a -> 'a -> 'a` in PSG
2. When curried application creates a Lambda, param types come from `TVar`
3. `mapType(TVar _) = Pointer` → Lambda gets `(!llvm.ptr, !llvm.ptr) -> !llvm.ptr`
4. At call site, `i64` values get converted via `inttoptr` to match signature
5. Result is `!llvm.ptr`, used where `i64` expected → type error

**MLIR Output**:
```mlir
// WRONG - Generated lambda for curried (+)
llvm.func internal @lambda_13(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
    %v0 = llvm.call @op_Addition(%arg0) : (!llvm.ptr) -> !llvm.ptr
    ...
}

// Call site converts i64 lengths to pointers
%v0 = llvm.extractvalue %arg0[1] : !llvm.struct<(!llvm.ptr, i64)>  // i64
%v1 = llvm.inttoptr %v0 : i64 to !llvm.ptr   // WRONG
%v2 = llvm.call @lambda_13(%v1, ...) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
```

**Why This Is Wrong**:
- The SRTP resolution should have resolved `'a` to `int` at this call site
- Typed tree overlay should carry concrete instantiated types
- Code generation receives `TVar` instead of resolved concrete type

**The Architectural Fix (NOT YET IMPLEMENTED)**:
The typed tree (`FSharpExpr`) contains SRTP resolution information. When F# compiler resolves `+` on integers, it knows the concrete instantiation. This information must be captured in the PSG's typed tree overlay and used during code generation.

**Current Workaround** (partial):
- Added `tryEmitPrimitiveBinaryOp` for when both args are primitive types at a single Application
- Doesn't help curried applications where Lambda is created with wrong types

**When You See This Pattern**:
1. Type mismatch involving `!llvm.ptr` and primitive types
2. `inttoptr`/`ptrtoint` conversions in generated MLIR
3. Polymorphic operators being treated as external functions

**Investigation Path**:
1. Check PSG node's `Type` field - is it `TVar` or concrete?
2. If `TVar`, the typed tree overlay isn't working for this node
3. Trace to Baker (typed tree zipper) to see where resolution is lost

**See**: `srtp_resolution_findings` memory for typed tree overlay architecture.

---

#---

## Mistake 13: Transform Logic in the Emission Layer (January 2026 Refactoring)

During the MLIR Transform Refactoring (Phases 1-10), we discovered and documented these patterns:

**Marker strings as workarounds**:
```fsharp
// TECHNICAL DEBT - Marker strings for incomplete FNCS transforms
let marker = sprintf "$pipe:%s:%s" argSSA argType
let marker = sprintf "$partial:%s:%s" funcName argsEncoded
let marker = sprintf "$platform:%s:%s" entryPoint argsEncoded
```

**Why these exist**: FNCS doesn't yet fully reduce pipe operators or provide complete intrinsic metadata. Alex uses markers to encode information about partially-processed constructs.

**The Fix**: Complete FNCS transforms upstream so Alex never sees these patterns:
- Pipe reduction in FNCS → no `$pipe:` markers
- Full intrinsic metadata → no `$platform:` lookups by name
- Application saturation → no `$partial:` markers

**When you see marker strings**: They indicate incomplete upstream transforms. Don't add more markers - fix FNCS.

**sprintf vs Templates**:
```fsharp
// WRONG - sprintf scattered throughout witnesses
sprintf "%s = arith.addi %s, %s : %s" res lhs rhs ty

// RIGHT - Template-based emission
let params = { Result = res; Lhs = lhs; Rhs = rhs; Type = ty }
render ArithTemplates.Quot.Binary.addI params
```

**Current metrics** (January 2026):
- 57 sprintf remaining in Witnesses (down from 138)
- 47 sprintf remaining in MLIRZipper (down from 53)
- Remaining sprintf: SCF nested regions, error messages, markers

---

---

## Mistake 14: Band-Aid Fixes to Config Defaults Instead of Architectural Fixes (January 2026)

Discovered during TRecord→TApp unification when WrenHello failed without `-k` flag but succeeded with it.

```fsharp
// WRONG - "Fixing" reachability by changing config default
let defaultConfig : NanopassConfig = {
    // ...
    SoftDeleteReachability = true  // "Must be true - hard-delete breaks field lookups"
}
```

**What happened**:
1. After unifying record types to use `TApp` with `SemanticGraph.tryGetRecordFields` lookup
2. WrenHello failed: "Type 'Platform.WebView' not found in SemanticGraph.Types"
3. But succeeded with `-k` flag (which enables soft-delete reachability)
4. Initial "fix": change default from `false` to `true`

**Why this is wrong**:
- This masks the real bug: reachability analysis was NOT following type references
- TypeDef nodes for record types were being pruned as "unreachable"
- But they ARE reachable - through the types of reachable nodes!
- Changing the default just hides the architectural deficiency

**The Principled Fix**:
The reachability analysis (`computeReachable`) must follow **type references**, not just expression edges:

```fsharp
/// Extract type names from a NativeType (for reachability of TypeDef nodes)
let rec getTypeNames (ty: NativeType) : string list =
    match ty with
    | NativeType.TApp(tycon, args) ->
        // TApp with FieldCount > 0 indicates a record type needing TypeDef
        let tyconNames = if tycon.FieldCount > 0 then [tycon.Name] else []
        tyconNames @ (args |> List.collect getTypeNames)
    // ... recurse through TFun, TTuple, etc.

/// Get TypeDef NodeIds for types referenced by a node
let getTypeDefRefs (node: SemanticNode) (graph: SemanticGraph) : NodeId list =
    getTypeNames node.Type
    |> List.choose (fun name -> SemanticGraph.recallType name graph)

/// Compute reachable - follows structural, semantic, AND type references
let computeReachable (graph: SemanticGraph) (entries: NodeId list) : Set<NodeId> =
    let rec walk visited nodeId =
        // ...
        let refs = getSemanticReferences node
        let typeRefs = getTypeDefRefs node graph  // NEW: follow type graph
        let allRefs = (node.Children @ refs @ typeRefs) |> List.distinct
        allRefs |> List.fold walk visited
```

**The Principle**: When a reachable node's type references a TypeDef (record, union), that TypeDef is reachable too. This follows FCS's TyconRef.Deref pattern - type references create edges in the reachability graph.

**When you see config-based "fixes"**:
- STOP - this is hiding a bug, not fixing it
- Ask: "What architectural invariant is being violated?"
- Trace the actual data flow to find where the invariant breaks
- Fix the architecture, not the config

---

## Mistake 15: Fresh Type Variables for Pattern Bindings (January 2026)

Discovered during sample 06 (AddNumbersInteractive) failure after TRecord→TApp unification.

```fsharp
// WRONG - Pattern matching creates fresh type variables
| SynArgPats.Pats pats ->
    let (argPatterns, argBindings) =
        pats
        |> List.map (fun p ->
            let argTy = freshTypeVar range  // WRONG: ignores constructor type
            checkPattern env p argTy range)
        |> List.unzip
```

**What happened**:
1. Pattern `| IntVal x, FloatVal y -> FloatVal ((float x) + y)`
2. Binding `x` got type `TVar { Parent = Unbound }` 
3. But `y` got type `TVar { Parent = Bound(float) }`
4. `float x` conversion failed with "Unbound type variable '?11'"

**Why this is wrong**:
- DU constructors have function types: `IntVal : int -> Number`
- Pattern matching should extract payload type from constructor
- Creating fresh type variables ignores known type information
- Relies on downstream constraint solving that may not happen

**The Principled Fix (FCS TyconRef.Deref Pattern)**:
Look up the constructor binding and extract payload types from its function type:

```fsharp
| SynArgPats.Pats pats ->
    // Look up constructor binding to get payload types
    let payloadTypes =
        match tryLookupBinding caseName env with
        | Some binding ->
            // Extract domain types from constructor's function type
            let rec extractDomains ty acc =
                match ty with
                | NativeType.TFun(domain, range) -> extractDomains range (domain :: acc)
                | _ -> List.rev acc
            extractDomains binding.Type []
        | None ->
            pats |> List.map (fun _ -> freshTypeVar range)  // Fallback only

    let (argPatterns, argBindings) =
        List.zip pats payloadTypes
        |> List.map (fun (p, argTy) -> checkPattern env p argTy range)
        |> List.unzip
```

**The Principle**: Type information flows from definitions to uses. Constructor types define payload types. Pattern matching USES this information - it doesn't create new type information.

---

## Mistake 16: TRecord/TApp Dual Representation (January 2026 - ROOT CAUSE)

This was the architectural pollution that caused Mistakes 14 and 15.

```fsharp
// WRONG - Two representations for the same semantic concept
| TRecord of tycon: TypeConRef * fields: (string * NativeType) list
| TApp of tycon: TypeConRef * args: NativeType list
```

**The Problem**:
- `TRecord` embedded field info directly in the type
- `TApp` referenced a type constructor, fields accessed via `SemanticGraph.Types`
- Unification failed: "expected TApp, got TRecord" or vice versa
- Some code paths used `TRecord`, others used `TApp`
- Field access worked for `TRecord` but broke for `TApp`

**Why this happened**:
- Incremental development added `TRecord` as a convenience
- But FCS uses single representation: `TType_app(tyconRef, typeInstantiation, nullness)`
- Field info accessed via `TyconRef.Deref` in FCS, not embedded in type ref

**The Principled Fix**:
Align with FCS architecture - single representation with deferred field lookup:

1. **Add FieldCount to TypeConRef** (not Fields - avoids forward reference):
```fsharp
type TypeConRef = {
    Name: string
    Module: ModulePath
    ParamKinds: TypeParamKind list
    Layout: TypeLayout
    NTUKind: NTUKind option
    FieldCount: int  // 0 = not a record, >0 = record with N fields
}
```

2. **Use TApp consistently** for record types:
```fsharp
let recordType = mkSimpleType (mkRecordTypeConRef name path layout fieldCount)
// NOT: TRecord(tycon, fields)
```

3. **Lookup fields via SemanticGraph.Types** (FCS TyconRef.Deref pattern):
```fsharp
let tryGetRecordFields (typeName: string) (graph: SemanticGraph) =
    match recallType typeName graph with
    | Some nodeId ->
        match tryGetNode nodeId graph with
        | Some { Kind = SemanticKind.TypeDef(_, TypeDefKind.RecordDef fields, _) } ->
            Some fields
        | _ -> None
    | None -> None
```

**The Cascade Effect**:
Fixing this root cause revealed downstream pollution:
- Mistake 14: Reachability didn't follow type references
- Mistake 15: Pattern bindings didn't use constructor types

**The Principle**: FCS patterns exist for good reasons. When diverging from FCS creates type representation inconsistencies, align back with FCS. The "convenience" of embedded data creates pollution that cascades downstream.

---

## Mistake 17: Incomplete Conversion Witness Coverage (January 2026)

When adding new intrinsics to FNCS, the corresponding Alex witnesses must cover ALL operations in that intrinsic category.

```fsharp
// FNCS defines 11 Convert intrinsics:
// toFloat, toInt, toInt64, toByte, toSByte, toInt16, toUInt16, toUInt32, toUInt64, toFloat32, toChar

// WRONG - ApplicationWitness.fs only handled 2:
| ConvertOp "toFloat", [(argSSA, argType)] -> ...
| ConvertOp "toInt", [(argSSA, argType)] -> ...
// Missing: toInt64, toByte, toSByte, toInt16, toUInt16, toUInt32, toUInt64, toFloat32, toChar
```

**How the gap was discovered**: Test code using `uint32 0x12345678` failed with "Unknown intrinsic: uint32 with 1 args". The `uint32` conversion function was recognized by FNCS as `Convert.toUInt32` but had no Alex witness.

**Why this matters**:
- FNCS defines intrinsics with full type signatures
- Alex witnesses must handle ALL intrinsic operations, not just common ones
- Without complete coverage, valid F# code fails at MLIR generation

**The fix**: Add witnesses for ALL conversion operations with proper type-based dispatch:

```fsharp
| ConvertOp "toUInt32", [(argSSA, argType)] ->
    match argType with
    | "i32" -> zipper, TRValue (argSSA, "i32")  // Identity - no conversion
    | _ ->
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let convText =
            match argType with
            | "f32" | "f64" -> render Quot.Conversion.fpToUI { ... }
            | "i64" -> render Quot.Conversion.truncI { ... }
            | "i8" | "i16" -> render Quot.Conversion.extUI { ... }
            | other -> failwithf "toUInt32: unsupported source type %s" other
        ...
```

**Additional lesson - Identity conversions**: MLIR does not allow no-op conversions like `arith.extui %x : i32 to i32`. When source and target types match, return the input value directly without emitting any MLIR operation.

**The Principle**: When adding intrinsic support, create a verification checklist:
1. List ALL operations in the intrinsic module
2. Add witnesses for EVERY operation
3. Test with code that exercises each operation
4. Identity cases must be handled specially (pass through, don't convert)

---

---

## Mistake 18: sprintf-Based MLIR Generation Instead of Composable Templates (CRITICAL - January 2026)

This is the **FOUNDATIONAL ARCHITECTURAL VIOLATION** that polluted the entire Alex layer.

```fsharp
// WRONG - sprintf everywhere in templates and witnesses
let addI = <@ fun p -> sprintf "%s = arith.addi %s, %s : %s" p.Res p.Lhs p.Rhs p.Ty @>

// WRONG - Direct sprintf in witnesses
let text = sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" result globalName
```

**The Horror Show**:
- 532+ sprintf calls scattered throughout Alex
- Templates wrap sprintf in quotations (architectural nonsense)
- XParsec reduced to pattern matching instead of template composition
- MLIR becomes "stringly typed" instead of structured
- SSA values treated as strings instead of typed references

**Why This Is FUNDAMENTALLY WRONG**:

1. **MLIR HAS STRUCTURE** - SSA is NOT stringly typed
   - MLIR operations are structured with typed operands and results
   - sprintf erases all type information into strings
   - Type errors manifest at runtime (MLIR verifier) instead of compile-time

2. **Templates Should Compose UPWARD**
   - The "Library of Alexandria" metaphor: miniaturized templates compose into larger constructs
   - Like LEMMAS composing into PROOFS
   - sprintf produces flat strings that cannot compose

3. **XParsec Is THE Glue Layer**
   - XParsec should COMPOSE templates at build time
   - Instead it's reduced to pattern matching, calling sprintf
   - The composition engine was stripped out

**The Correct Architecture**:

```
Dialect Templates (STRUCTURED TYPES, not strings)
    └── Arith/AddI, SubI, etc.  ← Returns MLIROp.Arith.AddI record
    └── LLVM/AddressOf, Load    ← Returns MLIROp.LLVM.AddressOf record
    └── SCF/If, While, For      ← Returns MLIROp.SCF.If with regions
              ↓
XParsec (COMPOSITION ENGINE)
    - Matches PSG patterns
    - COMPOSES templates upward into larger structures
    - Like assembling lemmas into proofs
    - Type-safe: SSA types must match at composition boundaries
              ↓
Boundary (SERIALIZATION - ONLY HERE)
    - Structured MLIROp → MLIR text string
    - THE ONLY PLACE sprintf exists
```

**What Was Wrong**:
```fsharp
// sprintf at EVERY level (scattered 500+ times)
let addressOf result globalName =
    sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" result globalName

// XParsec reduced to calling sprintf
let emitAddressOf = 
    pGlobalRef >>= fun ref ->
    let text = LLVMTemplates.addressOf resultSSA ref.Name  // Returns string!
    witnessOp text  // Accumulates strings!
```

**What Should Be**:
```fsharp
// Templates return STRUCTURED TYPES
type LLVMAddressOf = { Result: SSA; GlobalName: string }

let addressOf result globalName : LLVMOp =
    LLVMOp.AddressOf { Result = result; GlobalName = globalName }

// XParsec COMPOSES templates into larger structures
let pLoadGlobalString : XParsec<MLIRExpr> =
    pGlobalStringRef >>= fun globalRef ->
    template LLVM.addressOf globalRef.Name >>= fun addrOp ->  // Returns LLVMOp
    template LLVM.load addrOp.Result >>= fun loadOp ->        // Returns LLVMOp
    return (MLIRExpr.Composed [addrOp; loadOp])               // Composed structure!

// Serialization ONLY at boundary
module Serialize =
    let emit (op: MLIROp) : string =
        match op with
        | LLVMOp.AddressOf a -> 
            sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" 
                (ssaName a.Result) a.GlobalName  // sprintf ONLY HERE
```

**Dialect Folder Structure (Target State)**:
```
Alex/Dialects/
├── Core/
│   ├── Types.fs          # MLIROp, SSA, Val, Block, Region
│   └── Serialize.fs      # THE ONLY PLACE FOR sprintf
├── Arith/
│   ├── Types.fs          # ArithOp DU
│   ├── Templates.fs      # Returns structured ArithOp values
│   └── Combinators.fs    # XParsec combinators that compose templates
├── LLVM/
│   ├── Types.fs          # LLVMOp DU
│   ├── Templates.fs      # Returns structured LLVMOp values
│   └── Combinators.fs    # XParsec combinators
├── SCF/
│   ├── Types.fs          # SCFOp DU with Region support
│   ├── Templates.fs      # Region-aware structured templates
│   └── Combinators.fs    # XParsec combinators for control flow
└── Func/
    ├── Types.fs          # FuncOp DU
    ├── Templates.fs      # Function-level structured templates
    └── Combinators.fs    # XParsec combinators
```

**Why sprintf Pollution Is So Damaging**:

1. **No Compile-Time Type Safety** - String concatenation doesn't know about MLIR types
2. **No Composition** - Can't compose "arith.addi" string with "llvm.load" string meaningfully
3. **Deferred Errors** - Malformed MLIR discovered by verifier, not compiler
4. **Testing Nightmare** - Can't unit test template composition, only string output
5. **Refactoring Impossible** - Changing a template requires global search-replace

**The Remediation Scope**:

- 500+ sprintf calls must be eliminated
- Each dialect needs structured types (MLIROp variants)
- XParsec combinators must compose templates, not call sprintf
- Serialize.fs becomes the ONLY place strings are produced
- MLIRZipper witnesses structured ops, not text strings

**When You Find Yourself Writing sprintf in Alex**:
STOP. You are about to pollute the architecture. Create a structured type for the operation and compose it via XParsec. Serialization happens ONLY at the boundary.

---

## Mistake 19: Quotations in Alex Templates (Related to Mistake 18)

```fsharp
// WRONG - Quotations wrapping sprintf in templates
type MLIRTemplate<'Params> = {
    Quotation: Expr<'Params -> string>  // WHY is this a quotation?
    ...
}

let addI = <@ fun p -> sprintf "%s = arith.addi ..." @>  // Quotation wrapping sprintf!
```

**What's Wrong**:
- Quotations have a SPECIFIC role in a SPECIFIC stratum
- Quotations define semantic structure at the BINDING level (libraries → PSG)
- Quotations do NOT belong in Alex templates
- Wrapping sprintf in a quotation provides ZERO benefit

**Where Quotations Belong**:
- In platform bindings at library level
- Captured INTO the PSG as semantic structure
- Represent WHAT operations mean at source level

**Where Quotations Do NOT Belong**:
- Alex templates (templates are structured types)
- MLIR generation (generation uses type-safe composition)

**The Confusion**:
Someone thought "quotations are cool, let's use them everywhere." But quotations are for **defining semantics at the source level**, not for **generating MLIR at the target level**. Using quotations to wrap sprintf is cargo-culting without understanding.

## Mistake 20: Imperative MLIR Construction for FNCS Intrinsics - "Kick the Can Down the Road" (January 2026)

This is a CRITICAL architectural violation discovered during sample 04 (HelloWorldFullCurried) failure.

**The Symptom**:
```
error: %arg0 is already in use
```
Sample 04 fails because the while loop iteration variable (`Arg 0`) collides with the function's first parameter (also `%arg0`).

**The Location**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Console/ConsoleBindings.fs` (lines 312-429)

```fsharp
// WRONG - 488 lines of IMPERATIVE MLIR CONSTRUCTION
let generateReadLine (zipper: MLIRZipper) (targetType: MLIRType) =
    // ... manual buffer allocation ...
    // ... manual while loop construction ...
    let posArg = { SSA = Arg 0; Type = MLIRTypes.i64 }  // THE BUG: hardcodes Arg 0
    // ... manual syscall invocation ...
    // ... imperative op list building ...
```

**What FNCS Provides** (in `CheckExpressions.fs`):
```fsharp
| "readln" ->
    // unit -> string
    // Reads a line from stdin (fd 0)
    NativeType.TFun(env.Globals.UnitType, env.Globals.StringType)  // Just a TYPE SIGNATURE!
```

**WHY THIS IS FUNDAMENTALLY WRONG**:

1. **FNCS punts implementation to Alex**: FNCS provides only a type signature (`unit -> string`), expecting Alex to "figure out" how to implement readline. This is architectural negligence.

2. **Alex does IMPERATIVE construction**: ConsoleBindings.fs builds MLIR imperatively - allocating buffers, constructing while loops manually, hardcoding SSA names. This violates the Photographer Principle (witnesses OBSERVE, don't BUILD).

3. **Hardcoded SSA names cause collision**: Line 348 uses `Arg 0` for the while loop iteration variable. When the function ALSO has parameters, `%arg0` is already in use. MLIR verification fails.

4. **Layer violation**: Alex shouldn't know "how readline works." Alex witnesses PSG structure and generates MLIR from what it observes. Alex cannot INVENT structure that isn't in the PSG.

**THE ROOT CAUSE**: Previous Claude session "aped" platform binding patterns without understanding the functional principles. It saw syscall patterns in Bindings and thought "readln needs a while loop, I'll build one here." This is WRONG.

**THE CORRECT ARCHITECTURE**:

FNCS intrinsics must be expressed as FUNCTIONAL CONSTRUCTS, not type signatures:

```fsharp
// CORRECT - In FNCS (not Alex!), Console.readln should decompose to:
let readln () =
    let rec readLoop acc =
        let byte = Sys.readByte 0  // stdin, single byte read (FNCS intrinsic)
        if byte = '\n'B then
            String.ofBytes (List.rev acc)  // Convert accumulated bytes to string
        else
            readLoop (byte :: acc)  // Accumulate and continue
    readLoop []

// Or using unfold (codata pattern):
let readln () =
    Seq.unfold (fun () ->
        let b = Sys.readByte 0
        if b = '\n'B then None
        else Some(b, ())
    ) ()
    |> String.ofSeq
```

**WHY FUNCTIONAL DECOMPOSITION WORKS**:

1. **PSG carries structure**: When FNCS provides a recursive definition, the PSG contains the full semantic structure.

2. **Alex witnesses, not builds**: Alex observes the recursive structure in PSG and generates corresponding MLIR. No imperative construction.

3. **SSA flows naturally**: The functional structure maps to SSA. Each recursive call becomes a proper scf.while with SSA values that don't collide with function parameters.

4. **Layer separation preserved**: FNCS defines WHAT readln means (semantics). Alex observes HOW to generate MLIR for that structure.

**THE FIX PATH**:

1. **In FNCS** (`CheckExpressions.fs`): Define Console.readln with REAL functional semantics, not just a type signature

2. **In Alex** (`ConsoleBindings.fs`): DELETE the imperative while-loop construction. Alex should witness the PSG structure that FNCS creates.

3. **Add FNCS primitive**: `Sys.readByte : int -> byte` (read single byte from fd) - This IS appropriate for Alex to witness as a syscall binding.

**THE PRINCIPLE**: 

> **FNCS intrinsics must decompose to functional constructs. Alex witnesses structure; Alex does not construct structure.**

The "kick the can down the road" pattern - where FNCS provides only type signatures and expects downstream to "figure it out" - is an architectural FAILURE. It causes:
- Layer violations (Alex building structure)
- SSA conflicts (hardcoded names)
- Untestable code paths (can't validate FNCS semantics)
- Coupling (Alex knows "how readline works")

**WHEN YOU SEE THIS PATTERN**:
- Imperative MLIR construction in Alex
- Hardcoded `Arg N` or manual SSA allocation
- While loops built in Bindings
- FNCS intrinsics with only type signatures

STOP. The fix is UPSTREAM in FNCS, not downstream patching in Alex.

---

# The Acid Test

Before committing any change, ask:

> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.

> "Am I creating a central dispatch mechanism?"

If yes, STOP. This is the antipattern that was removed twice.
