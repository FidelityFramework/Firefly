# Architecture Principles

> **ARCHITECTURE UPDATE (January 2026)**: Alloy has been absorbed into FNCS.
> Types are NTUKind compiler intrinsics. Operations are FNCS intrinsic modules.
> See blog entry: "Absorbing Alloy: When Your Library Becomes Your Language"

## The "Fidelity" Principle

> **CORE PRINCIPLE**: The entire point of Fidelity is that **the F# compiler controls memory layout - not MLIR, not LLVM**.

Types (memory regions, access kinds, etc.) carry semantic meaning through the ENTIRE compilation pipeline, guiding every decision. They ARE erased - but at the **LAST possible lowering stage**, after Fidelity has made all decisions. **Fidelity dictates; LLVM implements.**

---

## CRITICAL: Silent Failures Are Architectural Violations

> **A compiler's PRIMARY job is to surface errors. Silent failures are the most severe architectural violations.**

When you observe ANY of the following during development or testing:
- Functions "not found" that don't emit errors
- Code generation returning `Void` for unhandled cases
- `printfn` warnings without corresponding error propagation
- Conditions that fail silently (e.g., null checks that don't work)
- Missing symbols that don't halt compilation

**STOP EVERYTHING IMMEDIATELY.**

Do NOT:
- Continue to "see what happens"
- Run the binary to check output
- Chase downstream symptoms (segfaults, wrong output)
- Add workarounds or special cases

Instead:
1. Recognize this as a ROOT CAUSE, not a symptom
2. Fix the code generation to EMIT ERRORS, not swallow them
3. Ensure `EmitError` or equivalent propagates up
4. Add tests to prevent regression
5. Update memories if this reveals a pattern

**Example of the violation:**
```
[GEN] Function not found in PSG: System.Object.ReferenceEquals
[GEN] Function not found in PSG: Microsoft.FSharp.Core.Operators.``not``
```
This was printed, then ignored. The code returned `Void` and continued. The result was a silent failure that manifested as a segfault downstream. The correct behavior: emit a compilation ERROR and halt.

**The Principle:** Failures are GOOD in a compiler. They are signals. Silent failures are bugs that hide bugs.

---

## CRITICAL: Consult Memories at Decision Points

When encountering issues "downstream" in the pipeline, ALWAYS consult Serena memories BEFORE attempting fixes. The memories encode lessons learned from past mistakes.

Key memories to consult:
- `alex_zipper_architecture` - The Non-Dispatch Model
- `negative_examples` - Real mistakes to avoid
- `native_binding_architecture` - Extern primitive flow

## FOUNDATIONAL INSIGHT: SSA IS Functional Programming

From Appel's seminal 1998 paper and the SpeakEZ blog "Why F# is a Natural Fit for MLIR":

> F#'s functional structure maps **directly** to MLIR/SSA without reconstruction.

This is why Alex is powerful - it doesn't need to "figure out" what F# code does; the PSG already expresses the SSA structure that MLIR needs.

- **F# ≅ SSA** - The functional structure IS the compilation target
- **PSG captures intent** - Semantic primitives express WHAT, not HOW
- **Alex chooses implementation** - Target-aware decisions (x86 AVX, ARM NEON, embedded, WAMI)

**Alex IS a transformer** - it transforms PSG into MLIR compute graphs using XParsec and parser combinator monads. This IS its job. The issue is when cruft accumulates in the transformer that should be moved upstream into PSG enrichment.

## The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

The correct model has NO central dispatcher:

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)
    ↓
Fold over structure
    ↓
At each node: XParsec pattern → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates (correct centralization)
    ↓
Output: Complete MLIR module
```

The PSG structure drives emission. There's no routing table.

## The Layer Separation Principle

> **Each layer has ONE responsibility. Do not mix concerns across layers.**

| Layer | Responsibility | DOES NOT |
|-------|---------------|----------|
| **FNCS** | Types (NTUKind), intrinsic modules, PSG construction | Generate MLIR |
| **Firefly/Nanopasses** | Enrich PSG (post-FNCS lowering) | Generate MLIR or know targets |
| **Alex/Zipper** | Traverse PSG | Route or dispatch |
| **XParsec** | Local pattern matching | Central routing |
| **Bindings** | Platform MLIR generation | Know F# syntax |
| **MLIR Builder** | Accumulate emissions | Route or dispatch |

## ANTIPATTERN: Central Dispatch (Emitter/Scribe)

**NEVER create:**
- Handler registries that route by node kind
- Central dispatch mechanisms
- Pattern matching on symbol/function names
- "Emitter" or "Scribe" abstraction layers

These were removed twice (PSGEmitter, PSGScribe). They collect routing logic too early and attract library-aware special cases.

## The PSG Nanopass Pipeline

PSG construction follows a **true nanopass architecture**. Each pass is single-purpose, independently testable, and called by the orchestrator:

```
STRUCTURAL CONSTRUCTION     SynExpr → PSG nodes + ChildOf edges     [FULL GRAPH]
SYMBOL CORRELATION          + FSharpSymbol attachments              [FULL GRAPH]
REACHABILITY               + IsReachable marks                      [NARROWS SCOPE]
BAKER                      + Types, SRTP, Member bodies             [NARROWED]
ENRICHMENT NANOPASSES      + def-use, classifications, lowerings    [NARROWED]
```

**Reachability is a semantic boundary**, not just an optimization. Everything after reachability operates on the narrowed compute graph (~30 symbols for HelloWorld instead of ~780).

**Soft-delete reachability**: Nodes are marked `IsReachable = false`, not removed. Baker's two-tree zipper needs full structure for navigation.

**See `nanopass_pipeline` memory for complete details.**

## Baker Component Library

**Baker** is the symmetric **consolidation component library** to Alex, handling Phase 4 (typed tree overlay):

```
FCS Output → [PSG Phase 1-3] → [BAKER] → [PSG Enriched] → [ALEX] → MLIR
                                  ↑
                    Operates on NARROWED graph (post-reachability)
```

**CRITICAL**: Baker operates **AFTER reachability** (Phase 3). It only processes the narrowed compute graph - nodes marked `IsReachable = true`.

**Components:**
- `TypedTreeZipper` - correlates FSharpExpr with PSGNode by range
- `SRTPResolver` - extracts TraitCall → resolved member mappings
- `MemberBodyMapper` - maps static members to their typed expression bodies
- `TypeOverlay` - applies resolved types to PSG nodes

**Key Principles:**
1. Post-reachability only - narrowed graph scope
2. Zipper coherence - Baker's zipper and Alex's zipper operate on same narrowed scope
3. Consolidation - all type-level transforms in one place

**Symmetric Design:** 
- Baker consolidates type-level transforms (focuses IN on application graph)
- Alex consolidates code-level transforms (fans OUT to platform targets)

See `docs/Baker_Architecture.md` for full design.

## FNCS: Native Type Resolution at the Source

**CRITICAL**: BCL type contamination cannot be fixed downstream. The fix must happen in the type system.

### The IL Dependency Cone Discovery (December 2025)

Cascade deletion analysis revealed a fundamental truth: **the IL import assumption permeates the ENTIRE F# compiler type-checking layer** - 3.2MB across 59 files.

This means FNCS cannot be created by pruning. The type checker must be **rebuilt** for the native type universe.

**Why reducing surface area matters**:
- Every IL assumption is cognitive overhead for the entire framework
- Native type purity enables clean architectural decisions that ripple positively
- BCL contamination ripples negatively; removing it enables organic growth
- Focused surface area = focused development = sustainable platform

**FNCS (F# Native Compiler Services)** is a minimal FCS fork that provides native-first type semantics:
- String literals → `string` with native semantics (UTF-8 fat pointer, not `System.String`)
- Option → `option<'T>` with value semantics (stack-allocated, non-nullable)
- SRTP → Native witness hierarchy

**Key Principle**: Users write standard F# type names. FNCS provides native semantics transparently. There are NO wrapper types like `NativeStr`.

**Layer Separation with FNCS**:
| Layer | Responsibility |
|-------|---------------|
| **FNCS** | Type universe, literal typing, SRTP, null-free |
| **Alloy** | Library implementations using FNCS types |
| **Firefly/PSG** | Semantic graph from FNCS output |
| **Firefly/Alex** | Platform-aware MLIR generation |

See `fncs_architecture` memory, `fncs_ecosystem` memory, and:
- `/docs/FNCS_Architecture.md` - Integration with Firefly
- `~/repos/fsnative/docs/fidelity/FNCS_Pruning_Plan.md` - Implementation roadmap
- `~/repos/fsnative-spec/spec/` - Normative specification chapters
- `/docs/FNCS_Ecosystem.md` - Three-repository architecture overview
- `~/repos/SpeakEZ/hugo/content/proposals/From Bridged To Self Hosted.md` - Full strategic proposal

**Absorption Strategy** (December 2025 update):
- **fsil** patterns become intrinsic: inline-by-default, implicit SRTP, collection protocols
- **UMX** patterns become intrinsic: non-numeric measures, memory regions, access kinds
- See `fsil_inline_semantics` and `umx_phantom_types` memories for details

> **Note on UMX**: Memory region types are NOT "phantom types" that vanish early. They carry semantic meaning through the ENTIRE pipeline, guiding code generation. Erasure happens at the LAST possible lowering stage. See the Fidelity principle above.

**Alloy Evolution** (clarified December 2025):
- As FNCS provides native type semantics, Alloy transforms from workaround-heavy into focused BCL replacement
- **There is NO separate "fsnative library"** - Alloy IS the library, it evolves
- With FNCS: Standard F# types have native semantics; Alloy provides BCL-sympathetic API + Platform.Bindings
- BCL Sympathy means API idioms (not runtime semantics) - null-free, `option` with value semantics, Result-based errors

## Platform Operations: Three-Layer Architecture

Platform operations follow the **three-layer binding architecture**:

| Layer | What | Examples |
|-------|------|----------|
| **Layer 1** | FNCS Intrinsics | `Sys.write`, `Sys.read`, `Sys.exit`, `NativePtr.set` |
| **Layer 2** | Binding Libraries | Farscape-generated (GTK, CMSIS, etc.) |
| **Layer 3** | User Code | Alloy, applications |

**Layer 1 - FNCS Intrinsics** (core syscalls):
```fsharp
// FNCS intrinsics - recognized by module pattern, no declaration needed
Sys.write : int -> nativeptr<byte> -> int -> int
Sys.read  : int -> nativeptr<byte> -> int -> int
Sys.exit  : int -> 'T
```

**Layer 3 - Alloy uses intrinsics** (no stubs!):
```fsharp
// Correct - Alloy uses FNCS intrinsics
module Console =
    let Write (s: string) =
        Sys.write 1 s.Pointer s.Length |> ignore
```

**DEPRECATED** - The old `Platform.Bindings` with `Unchecked.defaultof` stubs is deprecated. See `binding_architecture_unified` memory for the canonical architecture.

## Coeffects and Codata

From the SpeakEZ blog:
- **Coeffects** track what code NEEDS from its environment (requirements, not products)
- **Codata** is demand-driven - Zipper "observes" nodes as it traverses

The Zipper doesn't decide what to do. The PSG node carries enough information (via nanopass enrichment) that emission is deterministic.

**Coeffect-based compilation decisions:**
```fsharp
type ResourceCoeffect =
    | NoResources                        // Pure computation - aggressive optimization
    | StackBounded of size: int          // Stack-allocated, auto-cleanup
    | ContinuationBounded of resources   // Cleanup at continuation completion
    | ExternalResources of handles       // Requires explicit management
```

## Delimited Continuations: The Unifying Abstraction

Delimited continuations are the connective tissue between:
- **Async expressions** - continuations with I/O-triggered resumption
- **Actor model** - continuations with message-triggered resumption  
- **Computation expressions** - syntax sugar for continuation manipulation

All compile through the **DCont MLIR dialect**, share optimization passes, benefit from similar representations.

**The DCont/Inet Duality:**
- **DCont (Sequential)**: Each step depends on previous (`async`, monads)
- **Inet (Parallel)**: No dependencies (`query`, list comprehensions)

The compiler identifies boundaries between pure regions (Inet) and effectful regions (DCont).

**RAII Through Continuation Completion:** Resource cleanup occurs when continuations terminate, not at arbitrary textual boundaries. This aligns resource lifetimes with computation lifetimes.

See `delimited_continuations_architecture` memory and `/docs/DCont_Pipeline_Roadmap.md` for details.

## Zoom Out Before Fixing

When a symptom appears downstream:

1. **Consult Serena memories** on architecture
2. **Trace upstream** through the full pipeline
3. **Find root cause** at the earliest point
4. **Fix upstream** - don't patch symptoms
5. **Validate end-to-end**

The symptom location and fix location are usually different.
