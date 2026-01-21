# PSG Enrichment Architecture

> **Enrichment** is the parent concept encompassing **Elaboration** (PLT term) and **Saturation** (Fidelity term) - both describing the process of synthesizing PSG structure that wasn't written by the user.

## Overview

The Fidelity compiler transforms F# source code to native binaries through a multi-phase pipeline. A critical aspect of this pipeline is **enrichment** - the process by which the Program Semantic Graph (PSG) gains structure beyond what the user explicitly wrote.

This document establishes the canonical terminology and architecture for enrichment in the Fidelity framework.

## Terminology: PLT vs Fidelity

### Elaboration (Programming Language Theory)

In PLT, **elaboration** refers to the process of "fleshing out" a program with implicit structure:
- Type annotations made explicit
- Implicit arguments inserted
- Syntactic sugar desugared
- Pattern matching expanded to decision trees

The term emphasizes making the implicit explicit - taking a surface-level representation and producing a fully-explicit internal representation.

### Saturation (Fidelity Framework)

**Saturation** is the Fidelity-specific term for filling the PSG "to the brim" before lowering:
- Every node has complete type information
- Every intrinsic operation has its implementation structure
- Every higher-order function has been decomposed to primitives
- Every closure has explicit capture representation

The metaphor is of a sponge absorbing water until it can hold no more - the PSG absorbs semantic information until it is fully saturated and ready for lowering to MLIR.

### Enrichment (The Parent Concept)

**Enrichment** encompasses both elaboration and saturation. When we say a node is "enriched," we mean:
1. It may have been elaborated (in the PLT sense) - e.g., pattern matching expanded
2. It may have been saturated (in the Fidelity sense) - e.g., intrinsic implementation synthesized

Both result in PSG structure that wasn't in the original source but is necessary for compilation.

## Two Kinds of Enrichment

The Fidelity compiler performs two distinct kinds of enrichment, tracked via metadata:

### 1. Intrinsic Elaboration (`Elaboration.Kind = "Intrinsic"`)

When the compiler encounters an intrinsic operation (defined in FNCS), it may need to synthesize PSG structure to implement that operation's semantics.

**Examples:**
- `Console.write "Hello"` - The user wrote a simple call, but the intrinsic implementation requires:
  - String length extraction
  - Buffer pointer calculation
  - Syscall invocation with fd, buffer, length
  - Return value handling

- `a + b` for generic numeric types - SRTP resolution may require:
  - Type-specific operation dispatch
  - Overflow checking infrastructure (if enabled)

**Key characteristic:** The user wrote something high-level; the compiler synthesizes the low-level implementation structure.

### 2. Baker Saturation (`Elaboration.Kind = "Baker"`)

Baker is responsible for **decomposing language features to primitives**. This includes:
- HOF decomposition (List.map, Seq.fold)
- Seq expression state machines
- Lazy thunk structures
- Async workflows

**Examples:**
- `List.map f xs` decomposes to:
  - Empty check (`List.isEmpty`)
  - Recursive traversal (`List.head`, `List.tail`)
  - Result construction (`List.cons`)
  - Recursive call structure

- `seq { for x in xs do yield f x }` decomposes to:
  - State machine structure
  - State variable nodes
  - Yield point tracking
  - MoveNext implementation

- `lazy expr` decomposes to:
  - Thunk structure (computed flag + value + code)
  - Force implementation

**Key characteristic:** The user wrote a high-level expression; Baker synthesizes the implementation structure.

## Metadata Architecture

Enriched nodes carry metadata that enables "pierce the veil" debugging:

```fsharp
module ElaborationMetadata =
    /// What kind of enrichment: "Intrinsic" or "Baker"
    [<Literal>]
    let Kind = "Elaboration.Kind"

    /// What construct triggered enrichment (e.g., "List.map", "Console.write")
    [<Literal>]
    let For = "Elaboration.For"

    /// Links related nodes from the same enrichment expansion
    [<Literal>]
    let Id = "Elaboration.Id"
```

### Source-Based vs Enriched Nodes

| Node Type | Metadata | Meaning |
|-----------|----------|---------|
| Source-based | None | User wrote this directly |
| Intrinsic-elaborated | Kind="Intrinsic" | Synthesized to implement intrinsic semantics |
| Baker-saturated | Kind="Baker" | Synthesized for HOF decomposition |

### The Expansion ID Pattern

When enrichment creates multiple related nodes, they share an `Elaboration.Id`:

```
Node 42: Kind="Baker", For="List.map", Id=7
Node 43: Kind="Baker", For="List.map", Id=7   ← Same expansion
Node 44: Kind="Baker", For="List.map", Id=7   ← Same expansion
Node 45: (no metadata)                         ← Source-based
Node 46: Kind="Intrinsic", For="syscall", Id=8 ← Different expansion
```

This enables tooling to:
- Group related enriched nodes
- Show expansion boundaries in debugging views
- Reconstruct the "user wrote this / compiler synthesized this" boundary

## Pipeline Placement

```
Source Code
    ↓
  FCS (Parse + Type Check)
    ↓
  PSG Construction (Phase 1-3)
    ↓
  FNCS Type Resolution (Phase 4)
    ↓
┌─────────────────────────────────────┐
│         PSG SATURATION              │
│  ┌─────────────────────────────┐   │
│  │ Intrinsic Elaboration       │   │  ← Expands intrinsic operations
│  │ (SemanticGraph.Elaboration) │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ Baker Saturation            │   │  ← Decomposes HOFs
│  │ (Baker/Recipes)             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
  PSG Elaboration (Nanopasses)
    ↓
  Coeffect Analysis                    ← Separate concept (see Coeffect_Analysis_Architecture.md)
    ↓
  Alex/Zipper → MLIR → LLVM → Binary
```

### Why Saturation Happens in PSG Layer

Saturation must happen BEFORE coeffect analysis because:

1. **Coeffects analyze the saturated graph** - SSA assignment, mutability analysis, and yield state tracking need to see the full structure, including synthesized nodes.

2. **"Only pay for what you use"** - Coeffect analyses only run when their results are needed. A saturated PSG that doesn't use sequences doesn't incur yield state analysis cost.

3. **Lowering strategy depends on structure** - The control-flow ↔ dataflow pivot (see Coeffect_Analysis_Architecture.md) requires the complete structure to make informed decisions.

## Implementation: Marking API

The unified marking API in `SemanticGraph.Elaboration`:

```fsharp
/// Mark a node as Baker-saturated
let markBaker (forConstruct: string) (id: int) (node: SemanticNode) : SemanticNode

/// Mark a node as Intrinsic-elaborated
let markIntrinsic (forConstruct: string) (id: int) (node: SemanticNode) : SemanticNode

/// Generate a fresh expansion ID for grouping related nodes
let freshId () : int
```

### Usage in Baker

```fsharp
let ctx = mkContext range elemType platform "List.map" inspiringNode
let expandedNode = mkExpandedNode ctx (SemanticKind.Application (...)) resultType
// expandedNode automatically has Kind="Baker", For="List.map", Id=ctx.ExpansionId
```

### Usage in Intrinsic Elaboration

```fsharp
let expansionId = Elaboration.freshId()
let synthesizedNode = baseNode |> markIntrinsic "Console.write" expansionId
```

## Phase Output Serialization

The FNCS phase emitter includes enrichment metadata in JSON output:

```json
{
  "id": 42,
  "kind": "Application",
  "type": "unit",
  "elaborationKind": "Baker",
  "elaborationFor": "List.map",
  "elaborationId": 7
}
```

This enables external tooling (IDEs, debuggers, visualization) to understand enrichment boundaries.

## Design Principles

### 1. Transparency Over Magic

Every synthesized node is marked. There is no hidden structure. Developers can always see what the compiler added.

### 2. Separation of Concerns

- **Intrinsic elaboration** handles operation semantics
- **Baker saturation** handles algorithmic structure
- **Coeffect analysis** (separate) handles metadata for lowering

### 3. Composability

Baker-saturated structure composes with intrinsic elaboration:
- `List.map Console.write xs` has Baker structure (map decomposition) containing Intrinsic nodes (Console.write implementation)

### 4. Debuggability

The expansion ID links enable:
- "Collapse" views that hide enriched structure
- "Expand" views that show full detail
- Step-through debugging that skips synthesized code

## Related Documents

- [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md) - Coeffect analysis (separate from enrichment)
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) - PSG construction pipeline
- [FNCS_Architecture.md](./FNCS_Architecture.md) - F# Native Compiler Services
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Overall system architecture

## Serena Memories

- `elaboration_infrastructure` - Condensed reference for enrichment metadata
- `baker_component` - Baker saturation component details
- `psg_elaboration_architecture` - PSG elaboration nanopass details
