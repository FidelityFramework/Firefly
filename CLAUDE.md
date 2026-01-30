# Firefly Compiler - Claude Context

## Active Assignment: Alex XParsec Remediation (January 2026)

**STATUS:** Planning Complete, Ready for Implementation
**PRIORITY:** This is THE ONLY active assignment.

### Context Window Protocol

**START of every context window:**
1. Read checklist: `mcp__serena-local__read_memory` → `alex_remediation_checklist_2026jan`
2. Read plan: `/home/hhh/.claude/plans/elegant-marinating-summit.md`
3. Read core memories: `alex_element_pattern_witness_architecture`, `xparsec_correct_usage_pattern`, `alex_xparsec_throughout_architecture`, `codata_photographer_principle`
4. Review canonical example: `src/Alex/Witnesses/LazyWitness.fs` (38 lines)
5. Check progress in checklist

**END of every context window:**
1. Update checklist (✅/🔄)
2. Document blockers in "Session Notes"
3. Commit if compilable
4. `mcp__serena-local__edit_memory` to persist progress

### Assignment Details

**Goal:** Refactor Alex to leverage XParsec throughout Element/Pattern/Witness architecture

**Targets:** 56% code reduction (8K→3.5K lines), 90% witness reduction (5.7K→600), 100% direct MLIR op elimination (564→0)

**Canonical Example:** `src/Alex/Witnesses/LazyWitness.fs` — ALL witnesses follow this ~20-line pattern

**Tracking:** Checklist in Serena memory `alex_remediation_checklist_2026jan`, plan at path above

### Golden Rules
1. NO code changes without reading plan and checklist first
2. LazyWitness.fs is the template for all witnesses
3. XParsec throughout — Elements, Patterns, AND Witnesses
4. Codata principle — witnesses observe and return, never build or compute
5. Gap emergence — if transform logic needed, return `TRError` and fix in FNCS
6. Incremental validation — compile + test after EACH witness

### Alex Architecture
```
Elements/    (module internal)  →  Atomic MLIR ops with XParsec state threading
Patterns/    (public)           →  Composable elision templates (~50 lines each)
Witnesses/   (public)           →  Thin observers (~20 lines each)
```
Elements are `module internal` — witnesses physically cannot import them.

---

## Architectural Principles

### Consult Serena Memories Before Acting

At any architectural decision point, read relevant memories FIRST:
- `architecture_principles`, `negative_examples` — Core constraints and real mistakes
- `fncs_architecture`, `fncs_functional_decomposition_principle` — FNCS design
- `alex_zipper_architecture` — Zipper + XParsec + Bindings model
- `baker_component` — Type resolution (Phase 4), SRTP
- `native_binding_architecture` — Platform bindings flow
- `compose_from_standing_art_principle` — Extend recent patterns, don't reinvent

### The Cardinal Rule: Fix Upstream

**Never patch where symptoms appear.** This is a multi-stage compiler pipeline. Trace upstream to find the root cause:

```
Native Binary ← LLVM ← MLIR ← Alex/Zipper ← Nanopasses ← PSG ← FCS ← FNCS ← F# Source
```

Fix at the EARLIEST pipeline stage where the defect exists. Before any fix, answer:
1. Have I traced through the full pipeline?
2. Am I fixing the ROOT CAUSE or patching a SYMPTOM?
3. Am I adding library-specific logic to a layer that shouldn't know about libraries?
4. Does my fix require code generation to "know" about specific function names?

If #3 or #4 is "yes", STOP. You're about to violate layer separation.

### Layer Separation

| Layer | Does | Does NOT |
|-------|------|----------|
| **FNCS** | Define native types (NTUKind) and intrinsic ops | Generate code or know targets |
| **FCS** | Parse, type-check, resolve symbols | Transform or generate code |
| **PSG Builder** | Construct semantic graph from FCS | Make targeting decisions |
| **Nanopasses** | Enrich PSG with edges/classifications | Generate MLIR or know targets |
| **Alex/Zipper** | Traverse PSG, emit MLIR via bindings | Pattern-match on symbol names |
| **Bindings** | Platform-specific MLIR generation | Know about F# syntax |

### Compose from Standing Art

New features MUST compose from recently established patterns, not invent parallel mechanisms. Before implementing anything: What patterns from the last 2-3 PRDs apply? Am I extending existing code or writing parallel implementations? If it feels like special-case handling, STOP.

---

## Pipeline Overview

```
F# Source → FCS → PSG (Nanopass Pipeline) → Alex/Zipper → MLIR → LLVM → Native Binary
```

### PSG Nanopass Pipeline

> See `docs/PSG_Nanopass_Architecture.md` for details.

```
Phase 1: Structural Construction    SynExpr → PSG with nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments (via FCS)
Phase 3: Soft-Delete Reachability   + IsReachable marks (structure preserved!)
Phase 4: Typed Tree Overlay         + Type, Constraints, SRTP resolution (Zipper)
Phase 5+: Enrichment Nanopasses     + def-use edges, operation classification, etc.
```

Key: Soft-delete reachability (never hard-delete — zipper needs full structure). Typed tree overlay captures SRTP resolution into PSG. Each phase inspectable via `-k`.

### Core Components

- **FCS** (`/src/Core/FCS/`) — Parsing, type checking, semantic analysis. Both syntax and typed trees used.
- **PSG** (`/src/Core/PSG/`) — Unified IR correlating syntax with semantics. THE single source of truth downstream.
- **Nanopasses** (`/src/Core/PSG/Nanopass/`) — Single-purpose PSG enrichment passes.
- **Alex** (`/src/Alex/`) — Zipper traversal + XParsec pattern matching + platform Bindings → MLIR.
  - `Traversal/` — Zipper and XParsec-based PSG traversal
  - `Pipeline/` — Orchestration, lowering, optimization
  - `Bindings/` — Platform-aware code generation
  - `CodeGeneration/` — Type mapping, MLIR builders
- **FNCS Intrinsics** (external: `~/repos/fsnative/src/Compiler/NativeTypedTree/Expressions/`) — NTUKind type universe, intrinsic operations, platform resolution.

### The Zipper + XParsec + Bindings Model

NO central dispatch hub. The model:

```
Zipper.create(psg, entryNode) → fold over structure → at each node: XParsec pattern → MLIR emission → MLIR Builder accumulates
```

- **Zipper**: Bidirectional PSG traversal, purely navigational, carries state
- **XParsec**: Composable pattern matchers on PSG structure, local decisions
- **Bindings**: Platform-specific MLIR, organized by (OSFamily, Architecture, BindingFunction), are DATA not routing
- **MLIR Builder**: Where centralization correctly occurs (output, not dispatch)

---

## Negative Examples (Real Mistakes)

1. **Symbol-name matching in codegen** — `match symbolName with "Console.Write" -> ...` couples codegen to namespaces. Use PSG node types and FNCS intrinsic markers instead.

2. **Unmarked intrinsics** — Operations must be defined in FNCS `Intrinsics.fs` to be recognized. If it's not there, Alex can't generate code for it.

3. **Nanopass logic in codegen** — Don't import nanopass modules or build indices during MLIR generation. Nanopasses run before; codegen consumes the enriched PSG.

4. **Mutable state in codegen** — Mutable variable handling belongs in PSG nanopasses, not in a `GenerationContext`.

5. **Central dispatch hub** — Handler registries routing on node kinds (PSGEmitter, PSGScribe — removed twice). Zipper folds, XParsec matches locally, Bindings provide implementations.

6. **Hard-deleting unreachable nodes** — Breaks typed tree zipper. Use soft-delete (IsReachable = false).

7. **Mixing nanopass scopes** — Pipe operators (`|>`) use `ReducePipeOperators`. SRTP is separate. Don't mix.

8. **BCL/runtime dependencies** — Types and operations are FNCS intrinsics (compiler-level), not library code.

---

## Reference Resources

| Resource | Path | When |
|----------|------|------|
| F# Compiler Source | `~/repos/fsharp` | AST/syntax issues, FCS internals |
| F# Language Spec | `~/repos/fslang-spec` | Type system, evaluation rules |
| Nanopass Framework | `~/repos/nanopass-framework-scheme` | Nanopass architecture (see `doc/user-guide.pdf`) |
| Triton CPU | `~/triton-cpu` | MLIR dialect patterns, optimization |
| MLIR Haskell Bindings | `~/repos/mlir-hs` | Alternative MLIR binding approach |
| Alloy | `~/repos/Alloy` | HISTORICAL — absorbed into FNCS Jan 2026 |
| Firefly Docs | `/docs/` | PRIMARY architecture docs |
| SpeakEZ Blog | `~/repos/SpeakEZ/hugo/content/blog` | Design philosophy |

### Key Documentation

| Document | Purpose |
|----------|---------|
| `Architecture_Canonical.md` | AUTHORITATIVE: Two-layer model, platform bindings, nanopass pipeline |
| `FNCS_Architecture.md` | F# Native Compiler Services |
| `PSG_Nanopass_Architecture.md` | True nanopass pipeline, typed tree overlay, SRTP |
| `TypedTree_Zipper_Design.md` | Zipper for FSharpExpr/PSG correlation |
| `XParsec_PSG_Architecture.md` | XParsec integration with Zipper |

---

## Build & Test

### Quick Commands

```bash
# Build compiler
cd /home/hhh/repos/Firefly/src && dotnet build

# Compile a sample
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
/home/hhh/repos/Firefly/src/bin/Debug/net10.0/Firefly compile HelloWorld.fidproj

# Execute and validate
./HelloWorld

# Keep intermediates for debugging
Firefly compile HelloWorld.fidproj -k
```

### Regression Runner (PRIMARY)

```bash
cd /home/hhh/repos/Firefly/tests/regression
dotnet fsi Runner.fsx                              # Full suite
dotnet fsi Runner.fsx -- --parallel --verbose       # Fast + detailed
dotnet fsi Runner.fsx -- --sample 05_AddNumbers     # Specific sample
```

A change is NOT complete until the regression runner passes AND binaries execute correctly.

### Sample Progression

| Sample | Tests |
|--------|-------|
| `01_HelloWorldDirect` | Static strings, basic Console calls |
| `02_HelloWorldSaturated` | Let bindings, string interpolation |
| `03_HelloWorldHalfCurried` | Pipe operators, function values |
| `04_HelloWorldFullCurried` | Full currying, Result.map, lambdas |
| `TimeLoop` | Mutable state, while loops, Sleep |

### Intermediate Artifacts

After any runner run, intermediates are at `samples/console/FidelityHelloWorld/<sample>/target/intermediates/`:

| Artifact | Stage |
|----------|-------|
| `01_psg0.json` | Initial PSG with reachability |
| `02_intrinsic_recipes.json` | Intrinsic elaboration recipes |
| `03_psg1.json` | PSG after intrinsic fold-in |
| `04_saturation_recipes.json` | Baker saturation recipes |
| `05_psg2.json` | Final saturated PSG to Alex |
| `06_coeffects.json` | Coeffect analysis |
| `07_output.mlir` | MLIR output |
| `08_output.ll` | LLVM IR |

When debugging, inspect in pipeline order to find WHERE a bug originates.

---

## Key Files

| File | Purpose |
|------|---------|
| `/src/Firefly.fsproj` | Main compiler project |
| `/src/Core/IngestionPipeline.fs` | Pipeline orchestration |
| `/src/Core/PSG/Builder.fs` | PSG construction |
| `/src/Core/PSG/Nanopass/*.fs` | PSG enrichment passes |
| `/src/Core/PSG/Reachability.fs` | Dead code elimination |
| `/src/Alex/Traversal/PSGZipper.fs` | Zipper traversal |
| `/src/Alex/Bindings/*.fs` | Platform-specific MLIR |
| `/src/Alex/Pipeline/CompilationOrchestrator.fs` | Full compilation |

## Project Configuration

`.fidproj` files (TOML):
```toml
[package]
name = "ProjectName"
[compilation]
memory_model = "stack_only"
target = "native"
[build]
sources = ["Main.fs"]
output = "binary_name"
output_kind = "freestanding"  # or "console"
```

## Serena Projects

```
mcp__serena-local__activate_project "Firefly"       # Main compiler
mcp__serena-local__activate_project "fsnative"      # FNCS implementation
mcp__serena-local__activate_project "fsnative-spec"  # F# Native spec
```

Use Serena tools (not bash grep/find) for code understanding. Use bash for git, build, and system commands.
