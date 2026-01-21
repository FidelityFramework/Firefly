# Coeffect Analysis Architecture

> **Coeffect Analysis** is the computation of metadata about the PSG that enables lowering strategy decisions. It is fundamentally distinct from **Enrichment** (node synthesis) - coeffects analyze structure, they don't create it.

## Overview

In functional programming theory, **effects** describe what a computation does to its environment (mutation, I/O, exceptions), while **coeffects** describe what a computation requires from its environment (context, resources, capabilities).

In the Fidelity compiler, **Coeffect Analysis** extends this concept to encompass all metadata computation that informs lowering strategies. This includes:
- SSA assignment (variable lifetimes and versions)
- Mutability analysis (which bindings are mutable, which are addressed)
- Yield state tracking (for sequence expressions)
- Pattern binding analysis (for match expressions)
- String table construction (for string literals)

## The Critical Distinction

### Enrichment: Synthesizing Structure

Enrichment (elaboration + saturation) creates PSG nodes:
- Input: PSG with some nodes
- Output: PSG with more nodes
- The structure of the graph changes

### Coeffect Analysis: Computing Metadata

Coeffect analysis computes information about existing structure:
- Input: Saturated PSG
- Output: Metadata/indices/maps about the PSG
- The structure of the graph is unchanged

```
┌──────────────────────┐     ┌──────────────────────┐
│     ENRICHMENT       │     │  COEFFECT ANALYSIS   │
│  (Node Synthesis)    │     │  (Metadata Compute)  │
├──────────────────────┤     ├──────────────────────┤
│ • Intrinsic elab     │     │ • SSA assignment     │
│ • Baker saturation   │     │ • Mutability analysis│
│ • Pattern expansion  │     │ • Yield state indices│
│                      │     │ • Pattern bindings   │
├──────────────────────┤     ├──────────────────────┤
│ Creates nodes        │     │ Analyzes nodes       │
│ Changes PSG shape    │     │ PSG shape unchanged  │
│ Happens BEFORE lower │     │ Happens BEFORE lower │
└──────────────────────┘     └──────────────────────┘
```

## Why Coeffects Matter for Fidelity

### The Control-Flow ↔ Dataflow Duality

This is the essence of the "Fidelity" in Fidelity framework.

F# source code expresses computations declaratively (dataflow style):
```fsharp
let result = items |> List.map transform |> List.filter predicate
```

But native code execution is inherently control-flow oriented:
```
loop:
  load item
  call transform
  test predicate
  branch ...
```

Coeffect analysis provides the information needed to **pivot between these representations** while preserving semantic fidelity.

### The Pivot Decision

Given a computation, the compiler can lower it via:
1. **Control-flow emphasis** - Explicit loops, branches, state machines
2. **Dataflow emphasis** - SIMD operations, vectorization, fusion

The choice depends on:
- Target platform capabilities (SIMD width, cache structure)
- Input characteristics (known size, streaming vs materialized)
- Operation semantics (can it vectorize? does order matter?)

**Coeffect analysis provides the data needed to make this decision correctly.**

## Coeffect Analysis Components

### 1. SSA Assignment

Computes Single Static Assignment form for the PSG:
- Each variable binding gets a unique SSA name
- Tracks variable versions through mutations
- Enables register allocation and optimization

```fsharp
type SSAAssignment = {
    NodeToSSA: Map<NodeId, SSA>
    BindingVersions: Map<string, int>
}
```

**Why it's a coeffect:** SSA names are metadata about bindings, not new nodes. The PSG structure is unchanged; we're computing a mapping.

### 2. Mutability Analysis

Identifies mutable state patterns:
- Which bindings are mutable (`let mutable`)
- Which mutable bindings have their address taken (`&x`)
- Which variables are modified in loop bodies
- Module-level mutable bindings (require special initialization)

```fsharp
type MutabilityAnalysis = {
    AddressedMutableBindings: Set<NodeId>
    ModifiedVarsInLoopBodies: Set<NodeId>
    ModuleLevelMutableBindings: (string * NodeId * NodeId) list
}
```

**Why it's a coeffect:** This is analysis of existing structure. We're classifying bindings, not creating new ones.

### 3. Yield State Indices

For sequence expressions, tracks:
- Number of yield points
- State machine structure (Sequential vs WhileBased)
- Internal state variable mappings

```fsharp
type YieldStateAnalysis = {
    SeqExprId: NodeId
    NumYields: int
    BodyKind: SeqBodyKind
    Yields: (NodeId * int * NodeId) list  // YieldId, StateIndex, ValueId
    InternalState: (string * NodeId * int) list  // Name, BindingId, StructIndex
}
```

**Why it's a coeffect:** Yield indices are metadata for state machine generation. The seq expression nodes already exist; we're computing their state machine layout.

### 4. Pattern Binding Analysis

For match expressions, tracks:
- Entry pattern bindings (top-level matches)
- Case-specific bindings (per-branch bindings)
- Binding lifetimes and scopes

```fsharp
type PatternBindingAnalysis = {
    EntryPatternBindings: (NodeId * string * NativeType) list
    CasePatternBindings: Map<int, (NodeId * string * NativeType) list>
}
```

**Why it's a coeffect:** Pattern bindings are structural analysis. The match expression exists; we're computing how values flow through it.

### 5. String Table

Collects string literals for data section emission:
- Global name assignment
- Content and byte length
- Deduplication

```fsharp
type StringEntry = {
    GlobalName: string
    Content: string
    ByteLength: int
}
```

**Why it's a coeffect:** String literals exist in the PSG. The table is metadata enabling efficient data section layout.

## Pipeline Placement

```
Source Code → FCS → PSG Construction
                          ↓
                    FNCS Resolution
                          ↓
┌─────────────────────────────────────┐
│          PSG SATURATION             │  ← Enrichment happens here
│   (Intrinsic Elab + Baker)          │
└─────────────────────────────────────┘
                          ↓
                    PSG Elaboration Nanopasses
                          ↓
┌─────────────────────────────────────┐
│        COEFFECT ANALYSIS            │  ← Analysis happens here
│   (SSA, Mutability, YieldState...)  │
└─────────────────────────────────────┘
                          ↓
                    Alex/Zipper → MLIR
```

### Why Analysis Happens After Saturation

1. **Complete structure required** - SSA assignment needs to see ALL bindings, including those synthesized by Baker.

2. **Yield states need full seq body** - Seq expression analysis requires the complete state machine structure, which Baker provides.

3. **Mutability must see enriched code** - Baker may introduce mutable accumulators; mutability analysis must see them.

## The "Only Pay for What You Use" Principle

Coeffect analyses are demand-driven:

| If the PSG contains... | Then run... |
|------------------------|-------------|
| Seq expressions | YieldStateAnalysis |
| Mutable bindings | MutabilityAnalysis |
| Match expressions | PatternBindingAnalysis |
| String literals | StringTableCollection |
| Any bindings | SSAAssignment (always) |

This is efficient: a simple program with no sequences doesn't pay for sequence analysis.

## Serialization: `alex_coeffects.json`

All coeffect analysis results are serialized for debugging:

```json
{
  "version": "2.0",
  "ssaAssignment": {
    "nodeToSSA": { "42": "v0", "43": "v1" },
    "bindingVersions": { "x": 2, "y": 1 }
  },
  "mutability": {
    "addressedMutableBindings": [55, 67],
    "moduleLevelMutableBindings": [
      { "name": "counter", "bindingId": 55, "initialValueId": 56 }
    ]
  },
  "yieldStates": {
    "seqYields": [
      {
        "seqExprId": 100,
        "numYields": 3,
        "bodyKind": "WhileBased",
        "yields": [...]
      }
    ]
  },
  "strings": [
    { "globalName": "@str_abc123", "content": "Hello", "byteLength": 5 }
  ]
}
```

This enables:
- Debugging lowering decisions
- Understanding state machine layout
- Verifying SSA correctness
- Auditing mutability handling

## The Control-Flow ↔ Dataflow Pivot in Practice

### Example: Sequence Expression

Source (dataflow style):
```fsharp
seq {
    for x in items do
        if predicate x then
            yield transform x
}
```

Coeffect analysis provides:
- Yield count: 1
- Body kind: WhileBased
- State variables: iterator position, current value

With this metadata, Alex can choose:
- **Control-flow lowering:** State machine with explicit transitions
- **Dataflow lowering:** (if items is known-size array) Vectorized filter-map

### Example: List Operations

Source (dataflow style):
```fsharp
items |> List.map f |> List.filter p |> List.fold g init
```

Coeffect analysis provides:
- Operation chain structure
- Intermediate type sizes
- Potential fusion opportunities

With this metadata, Alex can choose:
- **Control-flow lowering:** Three separate traversals
- **Dataflow lowering:** Fused single-pass with deforestation

## Why This is "Fidelity"

The Fidelity framework preserves semantic correctness while enabling optimal lowering:

1. **F# semantics preserved** - The user's code means exactly what F# says it means
2. **Native efficiency achieved** - Lowering exploits platform capabilities
3. **Pivot informed by coeffects** - Analysis provides data for correct decisions

Without coeffect analysis, the compiler would either:
- Always use conservative lowering (inefficient)
- Make uninformed optimization decisions (potentially incorrect)

Coeffects enable **informed, correct optimization**.

## Design Principles

### 1. Analysis Not Transformation

Coeffect analysis never modifies the PSG. It only computes metadata. This ensures:
- PSG remains the single source of truth
- Analysis results can be recomputed
- Debugging shows original structure

### 2. Demand-Driven Computation

Analyses run only when needed. A module with no sequences has no sequence analysis overhead.

### 3. Serializable Results

All coeffect data serializes to JSON for debugging and tooling integration.

### 4. Pipeline Phase Discipline

Coeffect analysis runs AFTER enrichment, BEFORE lowering. This ordering is architecturally required.

## Related Documents

- [PSG_Enrichment_Architecture.md](./PSG_Enrichment_Architecture.md) - Enrichment (distinct from coeffects)
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Overall system architecture
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) - PSG construction and nanopasses

## Serena Memories

- `psg_elaboration_architecture` - PSGElaboration nanopass architecture
- `deterministic_memory_management` - Memory management decisions (coeffect-informed)
- `lazy_seq_flat_closure_architecture` - Lazy/Seq implementations (use yield state analysis)
