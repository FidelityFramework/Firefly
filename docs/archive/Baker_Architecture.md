# Baker Architecture: Post-Reachability Type Resolution Component Library

## Overview

**Baker** is the companion component library to **Alex**, providing symmetric architecture around the PSG. Both are **consolidation component libraries** - Baker consolidates type-level transforms on the "front" of the PSG, while Alex consolidates code-generation transforms on the "back".

```
FCS Output (SynExpr, CheckedImplFile, FSharpExpr)
        ↓
    [PSG Construction: Phase 1-3]
        ↓
    PSG with structure, symbols, reachability marks
        ↓
    [BAKER]  ← Type-level transforms on NARROWED graph
        ↓
    [PSG: Fully Enriched]
        ↓
    [ALEX]   ← Code-level transforms, MLIR generation
        ↓
    MLIR
```

**Critical Architectural Principle**: Baker operates **AFTER reachability analysis** (Phase 3). It only processes the narrowed compute graph - the subgraph of nodes that are actually reachable from entry points. This is both a performance optimization and a semantic guarantee: Baker's expensive type correlation work is only performed on code that will actually be compiled.

The name "Baker" suggests "baking in" semantic information - the typed tree overlay that enriches PSG nodes with resolved types, SRTP resolutions, and member body mappings.

## Design Rationale

### The Current Gap

The PSG construction pipeline has five phases as documented in `PSG_Nanopass_Architecture.md`:

1. **Phase 1: Structural Construction** - SynExpr → PSG with nodes + edges
2. **Phase 2: Symbol Correlation** - Attach FSharpSymbol via FCS symbol uses
3. **Phase 3: Soft-Delete Reachability** - Mark unreachable nodes
4. **Phase 4: Typed Tree Overlay [BAKER]** - **PARTIALLY IMPLEMENTED**
5. **Phase 5+: Enrichment Nanopasses** - def-use edges, operation classification

Phase 4 is the critical gap. It should:
- Use a zipper to correlate `FSharpExpr` (typed tree) with `PSGNode` by range
- Capture resolved types after inference
- Capture **SRTP resolutions** (TraitCall → resolved member)
- Enable static member body lookup

Currently:
- `TypeIntegration.fs` builds a range index from typed expressions
- `ResolveSRTP.fs` uses reflection to access FCS internals for TraitCall data
- Static member bindings lack symbol correlation (Phase 2 failure)
- Body lookup for SRTP-resolved members fails

### Why Baker Operates After Reachability

**Nanopass Dependency Ordering**: Baker's dependency on reachability is not arbitrary coupling - it's a semantic dependency made explicit. The nanopass framework (Keep 2013) recognizes that passes may have ordering constraints when one pass's output is another's precondition:

```
Phase 3 (Reachability) outputs: narrowed compute graph
Baker's precondition: narrowed compute graph
```

This is proper nanopass design: the dependency keeps work focused on the right part of the scaffolding.

**Performance**: The typed tree correlation is expensive. By waiting until after reachability:
- We only process nodes marked `IsReachable = true`
- Dead code (unreachable from entry points) is never analyzed
- For large codebases like Alloy, this can be a significant reduction

**Semantic Correctness**: Baker enriches nodes that will actually be compiled:
- No wasted effort on unused functions
- SRTP resolutions are only computed for actual call sites
- Member body mappings are demand-driven by the narrowed graph

**Zipper Coherence**: Both Baker and Alex use zippers to traverse AST/PSG:
- Baker's zipper correlates typed tree with PSG (narrowed graph)
- Alex's zipper traverses enriched PSG to emit MLIR
- Both operate on the same narrowed scope - no scope mismatch

### Why Baker as a Component Library?

Baker consolidates all "semantic enrichment from FCS" logic:
- **Single Responsibility**: All type-level transforms in one place
- **Symmetric Design**: Balances Alex at the code-generation end
- **Clear Interfaces**: Baker owns enrichment, Alex owns generation
- **Testability**: Baker's output (enriched PSG) is inspectable
- **Consolidation**: Like Alex consolidates code-gen, Baker consolidates type resolution

While Alex "fans out" to support multiple platforms and architectures (the Library of Alexandria), Baker "focuses in" to resolve types and SRTP for the narrowed application graph. Both are component libraries with clear, complementary scopes.

## Baker Components

### 1. TypedTreeZipper

A two-tree zipper that synchronizes navigation between:
- **FSharpExpr** (typed tree from FCS)
- **PSGNode** (semantic graph node)

```fsharp
type TypedTreeZipper = {
    /// Current FSharpExpr position
    TypedFocus: FSharpExpr option
    /// Current PSGNode position
    PSGFocus: PSGNode
    /// Zipper context for backtracking
    Context: ZipperContext
    /// Correlation state
    Correlation: CorrelationState
}

type CorrelationState = {
    /// Range-based matches
    RangeMatches: Map<RangeKey, FSharpExpr>
    /// SRTP resolutions found
    SRTPResolutions: Map<NodeId, SRTPResolution>
    /// Type overlays ready to apply
    TypeOverlays: Map<NodeId, FSharpType>
}
```

**Correlation Strategy:**

1. **Range-based primary** - Match by source range
2. **Structure-based secondary** - When ranges don't match exactly:
   - For `App` nodes, match by argument count and types
   - For `Let` nodes, match by binding name
   - For `TraitCall`, extract resolution regardless of range

### 2. SRTPResolver

Extracts SRTP resolutions from the typed tree:

```fsharp
type SRTPResolution = {
    /// The TraitCall site (PSG node where SRTP operator appears)
    CallSite: NodeId
    /// The resolved member (FSharpMemberOrFunctionOrValue)
    ResolvedMember: FSharpMemberOrFunctionOrValue
    /// The resolved member's body location (for inlining)
    BodyLocation: NodeId option
    /// Type substitutions from call site
    TypeSubstitutions: Map<string, FSharpType>
}
```

**Resolution Process:**

1. Walk `FSharpExpr` looking for `TraitCall` patterns
2. For each TraitCall:
   - Extract the constraint (`MemberConstraintInfo`)
   - Get the resolved member from FCS
   - Find the corresponding PSG node by range
   - Record the resolution with type substitutions

### 3. MemberBodyMapper

Maps member declarations to their typed expression bodies:

```fsharp
type MemberBodyMapping = {
    /// Member symbol
    Member: FSharpMemberOrFunctionOrValue
    /// Declaration location (for correlation)
    DeclarationRange: range
    /// Typed expression body
    Body: FSharpExpr
    /// PSG binding node (after correlation)
    PSGBinding: NodeId option
}
```

**Process:**

1. Walk `FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, args, expr)`
2. Create mapping from member to body
3. Correlate with PSG binding nodes by range
4. Store for body lookup during Alex emission

### 4. TypeOverlay

Overlays resolved types onto PSG nodes:

```fsharp
/// Overlay resolved types from typed tree onto PSG nodes
let overlayTypes (psg: ProgramSemanticGraph) (index: ResolvedTypeIndex) : ProgramSemanticGraph =
    psg.Nodes
    |> Map.map (fun nodeId node ->
        match findTypeForNode node index with
        | Some resolvedType ->
            { node with Type = Some resolvedType }
        | None -> node)
    |> fun nodes -> { psg with Nodes = nodes }
```

## Integration with PSG Pipeline

Baker operates as Phase 4 of the nanopass pipeline, **after reachability has narrowed the graph**:

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Structural Construction                            │
│   SynExpr → PSG with nodes + ChildOf edges                  │
│   (Full library structure captured)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Symbol Correlation                                 │
│   + FSharpSymbol attachments via FCS symbol uses            │
│   (Full library symbols correlated)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Soft-Delete Reachability                           │
│   + IsReachable marks (structure preserved!)                │
│   *** GRAPH IS NOW NARROWED TO APPLICATION SCOPE ***        │
│   Only nodes reachable from entry points remain active      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: BAKER - Typed Tree Overlay (on NARROWED graph)     │
│                                                             │
│   Input: PSG (with reachability - only process reachable!)  │
│        + FSharpCheckProjectResults                          │
│        + FSharpImplementationFileContents                   │
│                                                             │
│   CRITICAL: Baker ONLY processes nodes where IsReachable    │
│                                                             │
│   Components:                                               │
│   1. TypedTreeZipper - correlate FSharpExpr with PSGNode    │
│   2. SRTPResolver - extract TraitCall → resolved member     │
│   3. MemberBodyMapper - map static members to bodies        │
│   4. TypeOverlay - apply resolved types                     │
│                                                             │
│   Output: PSG enriched with (reachable nodes only):         │
│   - Resolved types on reachable nodes                       │
│   - SRTP resolutions for reachable call sites               │
│   - Member body mappings for inlined functions              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5+: Enrichment Nanopasses                             │
│   Def-use edges, operation classification, etc.             │
│   (Also operates on narrowed graph)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ ALEX: Code Generation (on NARROWED, ENRICHED graph)         │
│   Zipper traversal → XParsec patterns → Bindings → MLIR    │
│   Same narrowed scope as Baker - coherent zippers           │
└─────────────────────────────────────────────────────────────┘
```

**Zipper Coherence**: Baker's TypedTreeZipper and Alex's PSGZipper both operate on the same narrowed graph. This ensures:
- No scope mismatch between type resolution and code generation
- SRTP resolutions found by Baker are exactly those needed by Alex
- Member bodies mapped by Baker are exactly those inlined by Alex

## PSG Extensions

Baker requires PSG to store additional enrichment data:

```fsharp
type ProgramSemanticGraph = {
    // Existing fields...
    Nodes: Map<int, PSGNode>
    Edges: Edge list
    SymbolTable: Map<string, FSharpSymbol>
    EntryPoints: NodeId list

    // Baker additions:
    /// SRTP resolutions (call site → resolution)
    SRTPResolutions: Map<NodeId, SRTPResolution>
    /// Member body mappings (member symbol → body expr/node)
    MemberBodies: Map<string, MemberBodyMapping>
    /// Type substitution index for inlined functions
    InlineTypeSubstitutions: Map<NodeId, Map<string, FSharpType>>
}
```

## Example: op_Dollar Resolution

Consider the Alloy Console code:

```fsharp
// In Alloy/Console.fs (with FNCS - string has native semantics)
let inline Write s = WritableString $ s

// In WritableString type
// Note: With FNCS, string has native semantics - single overload handles all strings
static member inline ( $ ) (_, s: string) = writeString s
```

**Current Problem:**
1. PSG builds `LongIdent:op_Dollar` node at call site
2. Symbol correlation fails (Phase 2) - no FCS symbol use at that range
3. SRTP nanopass finds TraitCall but can't map to PSG body
4. Alex can't find body for op_Dollar

**Baker Solution:**

1. **TypedTreeZipper** walks `Write s` call in typed tree:
   - Finds `TraitCall` for `$` operator
   - Extracts resolution: `WritableString.op_Dollar` with `^a = string`
   - Note: With FNCS, `string` has native semantics (UTF-8 fat pointer)

2. **SRTPResolver** records:
   ```fsharp
   {
       CallSite = <node id of "WritableString $ s">
       ResolvedMember = <FSharpMemberOrFunctionOrValue for op_Dollar(string)>
       TypeSubstitutions = Map ["^a", typeof<string>]
   }
   ```

3. **MemberBodyMapper** maps `WritableString.op_Dollar`:
   - For string (with native semantics): body = `writeString s`
   - Correlates by declaration range

4. **Alex** emission:
   - Looks up SRTP resolution for call site
   - Gets resolved member and type substitutions
   - Finds body via MemberBodyMapper
   - Emits inlined body with type substitutions applied

## File Organization

```
src/
├── Baker/
│   ├── Baker.fs           # Main entry point, orchestration
│   ├── TypedTreeZipper.fs # Two-tree zipper implementation
│   ├── SRTPResolver.fs    # TraitCall extraction and resolution
│   ├── MemberBodyMapper.fs # Member → body mapping
│   ├── TypeOverlay.fs     # Type application to PSG
│   └── Types.fs           # Baker-specific types
│
├── Core/
│   └── PSG/
│       ├── Types.fs       # Extended with Baker fields
│       └── ...
│
└── Alex/
    └── ...
```

## Implementation Phases

### Phase A: Foundation (Current Sprint)

1. Create `src/Baker/` directory structure
2. Define types in `Types.fs`
3. Implement `MemberBodyMapper` - extract member bodies from FCS
4. Wire into pipeline after Phase 3

### Phase B: SRTP Resolution

1. Implement `SRTPResolver` using FCS public API where possible
2. Add `SRTPResolutions` field to PSG
3. Update Alex to use SRTP resolutions for body lookup

### Phase C: TypedTreeZipper

1. Implement full zipper with correlation strategies
2. Replace reflection-based ResolveSRTP with Baker implementation
3. Add comprehensive type overlay

### Phase D: Cleanup

1. Remove `Core/PSG/TypeIntegration.fs` (absorbed into Baker)
2. Remove `Core/PSG/Nanopass/ResolveSRTP.fs` (absorbed into Baker)
3. Update documentation

## Key Principles

1. **Post-Reachability Only** - Baker ONLY processes nodes marked `IsReachable = true`. This is non-negotiable.
2. **Narrowed Graph Scope** - The compute graph has been narrowed to application scope before Baker runs.
3. **Use FCS Public API First** - Only use reflection for data not exposed publicly
4. **Range-Based Correlation** - Primary strategy for typed tree ↔ PSG matching
5. **Preserve PSG Structure** - Baker enriches, doesn't restructure
6. **Inspectable Output** - Baker's enrichments visible in `-k` intermediate output
7. **Single Pass Where Possible** - Walk typed tree once, extract all enrichments
8. **Zipper Coherence** - Baker's zipper and Alex's zipper operate on the same narrowed scope

## Relationship to Alex

| Aspect | Baker | Alex |
|--------|-------|------|
| **Role** | Consolidation component library | Consolidation component library |
| **Position** | Post-reachability, pre-enrichment | Post-enrichment, MLIR generation |
| **Input** | PSG (narrowed) + FCS typed tree | Enriched PSG (narrowed) |
| **Output** | Enriched PSG | MLIR |
| **Scope** | Focuses IN on application graph | Fans OUT to platform targets |
| **Zipper** | TypedTreeZipper (correlates typed tree) | PSGZipper (emits MLIR) |
| **Key Abstractions** | TypedTreeZipper, SRTPResolver | Zipper, XParsec, Bindings |
| **Knowledge Domain** | F# type system, SRTP | Platform targets, syscalls |

**Symmetric Component Libraries**: Baker and Alex are both consolidation component libraries that provide coherent transformation on opposite sides of the enriched PSG:

- **Baker consolidates** type-level transforms: typed tree correlation, SRTP resolution, member body mapping
- **Alex consolidates** code-level transforms: platform targeting, extern dispatch, MLIR generation

**Zipper Coherence**: Both use zippers to traverse the same narrowed graph:
- Baker's TypedTreeZipper correlates FSharpExpr with PSG nodes (enrichment)
- Alex's PSGZipper traverses enriched PSG to emit MLIR (generation)
- Same scope, complementary purposes, coherent design
