# Firefly Compiler - Claude Context

## CRITICAL: Consult Serena Memories at Architectural Decision Points

> **When encountering issues "downstream" in the pipeline, ALWAYS consult Serena memories on architecture BEFORE attempting fixes.**

The Serena MCP server maintains authoritative memories about architectural decisions, negative examples, and design principles. At any decision point where you're tempted to:
- Add special-case logic for specific functions or libraries
- Create a central dispatch/routing mechanism
- Pattern-match on symbol names rather than PSG structure
- Work around a PSG deficiency in code generation

**STOP and consult these Serena memories:**
- `architecture_principles` - Core architectural constraints
- `fncs_architecture` - FNCS (F# Native Compiler Services) design
- `alex_zipper_architecture` - The correct Zipper + XParsec + Bindings model
- `baker_component` - Type resolution layer (Phase 4), SRTP handling
- `negative_examples` - Real mistakes to avoid repeating
- `native_binding_architecture` - How platform bindings flow to native code

Use `mcp__serena__read_memory` to review relevant memories before proceeding.

---

## CRITICAL: Context Window Reset Protocol

> **At the START of every new context window (after compaction/reset), IMMEDIATELY perform an architectural review.**

Before taking ANY action on code, you MUST:

1. **Activate the Firefly project** via Serena: `mcp__serena-local__activate_project "Firefly"`
2. **List available memories**: `mcp__serena-local__list_memories`
3. **Read critical memories** - at minimum:
   - `architecture_principles` - Core architectural constraints
   - `negative_examples` - Real mistakes with explanations (ESSENTIAL)
   - `fncs_functional_decomposition_principle` - Why intrinsics must decompose
4. **Read task-relevant memories** based on the work at hand
5. **Confirm the layered architecture** - where does this change belong?

This is **NON-NEGOTIABLE**. The codebase contains patterns that LOOK reasonable but are architectural violations:
- Semantic hollowing (type signatures with no functional structure)
- Imperative MLIR construction in Alex (building structure that should come from PSG)
- "Kick the can" intrinsics (expecting downstream to implement)
- Central dispatch/routing mechanisms

Without reviewing memories, you WILL repeat these mistakes. The patterns exist in the code because previous sessions made them before learning better.

**The memories encode hard-won lessons. Read them BEFORE acting.**

---

## CRITICAL: Deliberate, Didactic Approach Required

> **SLOW DOWN. Be deliberate. Be didactic. Use agents prolifically to gather context before acting.**

This is a sophisticated compiler project with deep architectural constraints. Fast, intuitive fixes are almost always wrong. The correct approach requires:

1. **Consult Serena Memories First** - Before any architectural decision, read relevant memories. The memories encode lessons learned from past mistakes.

2. **Use Agents Before Acting** - Before making any change, spawn multiple agents to explore relevant reference materials, documentation, and related codebases. Do not rely solely on the files immediately visible.

3. **Confirm Intent** - When encountering ambiguity or when the path forward isn't crystal clear, stop and confirm with the user rather than making assumptions.

4. **Trace Full Pipeline** - Every issue must be traced through the complete compilation pipeline before proposing a fix. Symptoms rarely appear where root causes exist.

5. **Understand Before Implementing** - Read all relevant documentation, explore reference implementations, and understand the architectural constraints before writing code.

### Anti-Pattern: "Going Too Fast"

The following behavior is WRONG:
- Seeing an error and immediately patching where it manifests
- Adding stub implementations to "make it work"
- Using BCL/runtime dependencies when native implementations exist
- Making changes without exploring reference materials first

The correct behavior:
- Pause and spawn agents to explore context
- Understand WHY something works a certain way before changing it
- Look at how similar problems are solved in reference implementations
- Recognize when operations need to be FNCS intrinsics (types and operations are compiler-level, not library-level)

---

## Primary Reference Resources

These resources are ESSENTIAL for understanding the project architecture and making correct decisions. **Use agents prolifically to explore these resources before making changes.**

### Language & Compiler References

| Resource | Path | Purpose |
|----------|------|---------|
| **F# Compiler Source** | `~/repos/fsharp` | F# compiler implementation, FSharp.Compiler.Service internals, AST structures |
| **F# Language Specification** | `~/repos/fslang-spec` | Authoritative F# language semantics, type system, evaluation rules |
| **Nanopass Framework** | `~/repos/nanopass-framework-scheme` | Reference implementation of nanopass compiler architecture (Scheme). Key resource for understanding `define-language`, `define-pass`, catamorphisms. See `doc/user-guide.pdf` |

### MLIR & Code Generation References

| Resource | Path | Purpose |
|----------|------|---------|
| **Triton CPU** | `~/triton-cpu` | Production MLIR implementation, dialect patterns, optimization passes |
| **MLIR Haskell Bindings** | `~/repos/mlir-hs` | Alternative MLIR binding approach, type-safe IR construction |

### Fidelity Ecosystem

| Resource | Path | Purpose |
|----------|------|---------|
| **Alloy** | `~/repos/Alloy` | HISTORICAL ARCHIVE - Absorbed into FNCS (January 2026). Preserved as reference for patterns it established. |
| **BAREWire** | `~/repos/BAREWire` | Binary serialization - FUTURE. Memory-efficient wire protocol |
| **Farscape** | `~/repos/Farscape` | Distributed compute - FUTURE. Native F# distributed processing |

### Documentation

| Resource | Path | Purpose |
|----------|------|---------|
| **Firefly Docs** | `/docs/` | PRIMARY: Architecture, design decisions, PSG, Alex, lessons learned |
| **SpeakEZ Blog** | `~/repos/SpeakEZ/hugo/content/blog` | SECONDARY: Articles explaining design philosophy, architectural thinking |

### When to Use Each Resource

- **Encountering F# AST/syntax issues**: Explore `~/repos/fsharp` for FCS implementation details
- **Type system questions**: Reference `~/repos/fslang-spec` for language semantics
- **MLIR dialect patterns**: Look at `~/triton-cpu` for production examples
- **Native type implementation**: Types are FNCS intrinsics (NTUKind), not library code. See `~/repos/fsnative/src/Compiler/Checking.Native/`
- **Architectural decisions**: Read `/docs/Architecture_Canonical.md` first
- **Understanding "why"**: Check `~/repos/SpeakEZ/hugo/content/blog` for philosophy (including "Absorbing Alloy")

### Agent Usage Protocol

Before making any non-trivial change:

1. **Spawn an Explore agent** to understand the codebase area being modified
2. **Spawn reference agents** to explore relevant reference materials (e.g., how does FCS handle this? How do FNCS intrinsics work?)
3. **Read documentation** in `/docs/` related to the area of change
4. **Synthesize understanding** before proposing changes

Example agent tasks:
- "Explore how FNCS defines string intrinsics in CheckExpressions.fs"
- "Search ~/repos/fsharp for how FCS represents extern declarations"
- "Find examples of syscall bindings in ~/triton-cpu"
- "Look up the F# spec section on statically resolved type parameters"
- "Explore NTUKind type mappings in NativeTypes.fs"

---

## Project Overview

Firefly is an ahead-of-time (AOT) compiler for F# targeting native binary output without the .NET runtime. The project compiles F# source code through MLIR to LLVM IR and finally to native executables.

**Primary Goal**: Compile F# code to efficient, standalone native binaries that can run without any runtime dependencies (freestanding mode) or with minimal libc dependency (console mode).

**Key Constraint**: "Compiles" means a working native binary that executes correctly, not just successful parsing or IR generation.

## The "Fidelity" Mission

The framework is named "Fidelity" because it **preserves memory and type safety** through the entire compilation pipeline to native code. This is fundamentally different from:

- **.NET/CLR**: Relies on a managed runtime with garbage collection
- **Fable**: Transforms F# AST to target language AST (JavaScript, Rust, Python), delegating memory management to the target runtime

Fidelity/Firefly:
- **Preserves type fidelity**: F# types map to precise native representations, never erased
- **Preserves memory fidelity**: Compiler-verified lifetimes, deterministic allocation (stack/arena)
- **No runtime**: The generated binary has the same safety properties as the source, enforced at compile time

The PSG is not merely an AST - it is a **semantic graph carrying proofs** about memory lifetimes, type safety, resource ownership, and coeffects. Alex consumes this rich semantic information to generate MLIR that preserves all guarantees.

## Architecture Overview

The compilation pipeline flows through these major phases:

```
F# Source → FCS → PSG (True Nanopass Pipeline) → Alex/Zipper → MLIR → LLVM → Native Binary
```

### The PSG Nanopass Pipeline (CRITICAL)

> **See: `docs/PSG_Nanopass_Architecture.md` for authoritative details.**

PSG construction is itself a **true nanopass pipeline**, not a monolithic operation. Each phase does ONE thing:

```
Phase 1: Structural Construction    SynExpr → PSG with nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments (via FCS)
Phase 3: Soft-Delete Reachability   + IsReachable marks (structure preserved!)
Phase 4: Typed Tree Overlay         + Type, Constraints, SRTP resolution (Zipper)
Phase 5+: Enrichment Nanopasses     + def-use edges, operation classification, etc.
```

**CRITICAL Principles:**

1. **Soft-delete reachability** - Mark unreachable nodes but preserve structure. The typed tree zipper needs full structure for navigation.

2. **Typed tree overlay via zipper** - A zipper correlates `FSharpExpr` (typed tree) with PSG nodes by range, capturing:
   - Resolved types (after inference)
   - Resolved constraints (after solving)
   - **SRTP resolution** (TraitCall → resolved member) - THIS IS ESSENTIAL

3. **Each phase is inspectable** - Intermediate PSGs can be examined independently via `-k` flag.

**Why the Typed Tree Matters:**

Without the typed tree overlay, SRTP (Statically Resolved Type Parameters) cannot be resolved:

```fsharp
// Example with SRTP constraint
let inline write< ^T when ^T : (member Length : int)> (x: ^T) = ...
```

- Syntax tree sees: `App [write, x]`
- Typed tree knows: `TraitCall` resolving the SRTP constraint to the specific member access

The typed tree zipper captures this resolution INTO the PSG. Downstream passes (including Alex) use it directly.

> **See: `docs/TypedTree_Zipper_Design.md` for zipper implementation details.**

### Core Components

1. **FCS (F# Compiler Services)** - `/src/Core/FCS/`
   - Provides parsing (`SynExpr`), type checking (`FSharpExpr`), and semantic analysis
   - Single source of truth for F# semantics
   - **Both** syntax and typed trees are used during PSG construction
   - See `docs/FCS_*.md` for detailed documentation

2. **PSG (Program Semantic Graph)** - `/src/Core/PSG/`
   - Unified intermediate representation correlating syntax with semantics
   - **THE SINGLE SOURCE OF TRUTH** for all downstream stages
   - Built through a true nanopass pipeline
   - Contains typed tree overlay for SRTP resolution
   - See `docs/PSG_Nanopass_Architecture.md` (CANONICAL) and `docs/PSG_architecture.md`

3. **Nanopasses** - `/src/Core/PSG/Nanopass/`
   - Small, single-purpose transformations that enrich the PSG
   - Each pass does ONE thing (add def-use edges, classify operations, etc.)
   - Passes are composable and can be inspected independently
   - **Critical distinction**: Some nanopasses are part of PSG construction (Phase 4: Typed Tree Overlay), others enrich afterwards (Phase 5+)
   - See `docs/PSG_Nanopass_Architecture.md`

4. **Alex** - `/src/Alex/`
   - Multi-dimensional hardware targeting layer ("Library of Alexandria")
   - **Zipper**: The traversal engine - the "attention" of Alex
   - **XParsec**: Pattern matching combinators for PSG structures
   - **Bindings**: Platform-aware MLIR generation
   - Maps F# semantics to platform-optimized code patterns
   - Contains:
     - `Traversal/` - Zipper and XParsec-based PSG traversal
     - `Pipeline/` - Orchestration, FCS integration, lowering, optimization
     - `Bindings/` - Platform-aware code generation patterns
     - `CodeGeneration/` - Type mapping, MLIR builders

5. **FNCS Intrinsics** - External at `~/repos/fsnative/src/Compiler/Checking.Native/`
   - Types ARE the language: NTUKind defines the native type universe
   - Operations ARE intrinsics: Sys.*, NativePtr.*, Array.*, String.*, Math.*, Console.*, etc.
   - Platform resolution via quotations at compile time
   - Key files:
     - `NativeTypes.fs` - NTUKind enum defining all native types
     - `NativeGlobals.fs` - Type constructors with NTU metadata
     - `CheckExpressions.fs` - Intrinsic module definitions

## CRITICAL: The Layer Separation Principle

> **Each layer in the pipeline has ONE responsibility. Do not mix concerns across layers.**

### Layer Responsibilities

| Layer | Responsibility | DOES NOT |
|-------|---------------|----------|
| **FNCS** | Define native types (NTUKind) and intrinsic operations | Generate code or know about targets |
| **FCS** | Parse, type-check, resolve symbols | Transform or generate code |
| **PSG Builder** | Construct semantic graph from FCS output | Make targeting decisions |
| **Nanopasses** | Enrich PSG with edges, classifications | Generate MLIR or know about targets |
| **Alex/Zipper** | Traverse PSG, generate MLIR via bindings | Pattern-match on symbol names |
| **Bindings** | Platform-specific MLIR generation | Know about F# syntax |
| **MLIR/LLVM** | Lower and optimize IR | Know about F# |

### The Zipper + Bindings Architecture (Non-Dispatch Model)

**Alex generates MLIR through Zipper traversal and platform Bindings. There is NO central dispatch hub.**

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

The antipattern is a central handler registry that routes based on node kinds. This collects special-case knowledge too early and becomes a dumping ground for library-aware logic.

The correct model:

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)
    ↓
Fold over structure (pre-order or post-order)
    ↓
At each node: XParsec pattern → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates
    ↓
Output: Complete MLIR module
```

**The Zipper:**
- Traverses the PSG structure bidirectionally
- Provides "attention" - focus on current node with context
- Is purely navigational - no routing or dispatch logic
- Carries state through traversal (SSA counters, emitted blocks)

**XParsec Combinators:**
- Match PSG node structure at each position
- Composable patterns, not a routing table
- Local decision-making based on node data

**The Bindings:**
- Contain platform-specific MLIR generation
- Looked up by `Platform.Bindings` module structure (e.g., `writeBytes`, `readBytes`)
- Are DATA (syscall numbers, calling conventions), not routing logic
- Organized by `(OSFamily, Architecture, BindingFunction)`

**MLIR Builder:**
- The natural "pool" where emissions accumulate
- This is where centralization correctly occurs
- Type-safe MLIR construction via computation expression

**MLIR generation should NEVER:**
- Pattern-match on function names or module paths
- Have special cases for specific namespaces
- Contain conditional logic based on symbol names
- Create a central dispatch/routing mechanism
- Require knowledge of user-level module structure

## NEGATIVE EXAMPLES: What NOT To Do

These are real mistakes made during development. **DO NOT REPEAT THEM.**

### Mistake 1: Adding namespace-specific logic to MLIR generation

```fsharp
// WRONG - MLIR generation should not match on namespaces
match symbolName with
| Some name when name = "MyApp.Console.Write" ->
    generateConsoleWrite psg ctx node  // Special case!
| Some name when name.StartsWith("SomeLib.") ->
    generateLibraryCall psg ctx node  // Another special case!
```

**Why this is wrong**: MLIR generation is now coupled to namespace structure. The code generation should be driven by PSG node types and FNCS intrinsic markers, not symbol names.

**The fix**: Alex recognizes FNCS intrinsics (marked with `SemanticKind.Intrinsic`) and generates MLIR based on the intrinsic type, not the symbol name.

### Mistake 2: Expecting FNCS intrinsics without proper recognition

```fsharp
// WRONG - Calling something that looks like an intrinsic but isn't marked
let result = Console.writeln "Hello"  // If Console.writeln isn't a recognized FNCS intrinsic...
```

**Why this is wrong**: If an operation isn't defined in FNCS CheckExpressions.fs as an intrinsic, FNCS doesn't know how to type it or mark it for Alex.

**The fix**: All operations that should compile to native code must be defined in FNCS as intrinsics:
```fsharp
// In CheckExpressions.fs - FNCS intrinsic module
| "Console.writeln" ->
    NativeType.TFun(env.Globals.StringType, env.Globals.UnitType)
```

Note: `string` has native semantics (UTF-8 fat pointer with Pointer/Length members) via FNCS. Users write `string` and FNCS provides native semantics transparently.

### Mistake 3: Putting nanopass logic in MLIR generation

```fsharp
// WRONG - Importing nanopass modules into code generation
open Core.PSG.Nanopass.DefUseEdges

// WRONG - Building indices during MLIR generation
let defIndex = buildDefinitionIndex psg
```

**Why this is wrong**: Nanopasses run BEFORE MLIR generation. They enrich the PSG. Code generation should consume the enriched PSG, not run nanopass logic.

### Mistake 4: Adding mutable state tracking to code generation

```fsharp
// WRONG - Code generation tracking mutable bindings
type GenerationContext = {
    // ...
    MutableBindings: Map<string, Val>  // NO! This is transformation logic
}
```

**Why this is wrong**: Mutable variable handling should be resolved in the PSG via nanopasses. Code generation should just follow edges to find values.

### Mistake 5: Creating a central "Emitter" or "Scribe" dispatch hub

```fsharp
// WRONG - Central dispatch that routes based on node kinds
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
- This is the antipattern that was removed twice (PSGEmitter, then PSGScribe)
- It collects "special case" routing too early in the pipeline
- It inevitably attracts library-aware logic ("if ConsoleWrite then...")
- The centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH

**The fix**: No central dispatcher. The Zipper folds over PSG structure. XParsec matches locally at each node. Bindings are looked up by Platform.Bindings module structure. MLIR Builder accumulates the output.

## Platform Bindings (FNCS Intrinsics)

Platform operations are FNCS intrinsics in the `Sys` module:

```fsharp
// FNCS CheckExpressions.fs - Sys intrinsic module
| "Sys.write" ->
    // fd:int -> buffer:nativeptr<byte> -> count:int -> int
    NativeType.TFun(env.Globals.IntType,
        NativeType.TFun(NativeType.TNativePtr Types.uint8Type,
            NativeType.TFun(env.Globals.IntType, env.Globals.IntType)))

| "Sys.clock_gettime" ->
    // unit -> int64 (ticks since epoch)
    NativeType.TFun(env.Globals.UnitType, env.Globals.Int64Type)
```

Alex provides platform-specific implementations for each `Sys` intrinsic based on the target platform (Linux x86_64, ARM64, Windows, etc.).

**Why intrinsics?** Following ML, Rust, and Triton-CPU patterns, platform operations belong in the compiler, not a library. FNCS defines the operation signatures; Alex provides implementations.

## Critical Working Principle: Zoom Out Before Fixing

> **THIS IS A COMPILER. Compiler development has fundamentally different requirements than typical application development. The pipeline is a directed, multi-stage transformation where upstream decisions have cascading downstream effects. You MUST understand this before making any changes.**

### The Cardinal Rule

**When encountering ANY adverse finding that is NOT a baseline F# syntax error, you MUST stop and review the ENTIRE compiler pipeline before attempting a fix.**

### Why This Matters

**DO NOT "patch in place."** The instinct to fix a problem where it manifests is almost always wrong in compiler development. A symptom appearing in MLIR generation may have its root cause in:
- Missing intrinsic definition in FNCS
- Incorrect symbol capture in FCS ingestion
- Failed correlation in PSG construction
- Missing nanopass transformation
- Wrong reachability decisions

**Patching downstream creates technical debt that compounds.** A "fix" that works around a PSG deficiency:
- Masks the real problem
- Creates implicit dependencies on broken behavior
- Makes future fixes harder as the codebase grows
- Violates the architectural separation of concerns

### The Correct Approach

1. **Observe the symptom** - Note exactly what's wrong (wrong output, missing symbol, incorrect behavior)

2. **Trace upstream** - Walk backwards through the pipeline:
   ```
   Native Binary ← LLVM ← MLIR ← Alex/Zipper ← Nanopasses ← PSG ← FCS ← FNCS ← F# Source
   ```

3. **Find the root cause** - The fix belongs at the EARLIEST point in the pipeline where the defect exists

4. **Fix upstream** - Correct the root cause, then verify the fix propagates correctly through all downstream stages

5. **Validate end-to-end** - Confirm the native binary behaves correctly

### Forcing Functions

Before proposing any fix for a non-syntax issue, answer these questions:

1. "Have I read the relevant docs in `/docs/`?"
2. "Have I traced this issue through the full pipeline?"
3. "Am I fixing the ROOT CAUSE or patching a SYMPTOM?"
4. "Is my fix at the earliest possible point in the pipeline?"
5. "Will this fix work correctly as the compiler evolves, or am I creating hidden coupling?"
6. **"Am I adding library-specific logic to a layer that shouldn't know about libraries?"**
7. **"Does my fix require code generation to 'know' about specific function names?"**

If you cannot confidently answer all questions, you have not yet understood the problem well enough to fix it.

**If the answer to question 6 or 7 is "yes", STOP. You are about to make the same mistake again.**

### Pipeline Review Checklist

When a non-syntax issue arises:

1. **FNCS Intrinsics Level**
   - Is the operation defined as an intrinsic in CheckExpressions.fs?
   - Does the intrinsic have the correct type signature?
   - Is the NTUKind mapping correct for the types involved?

2. **FCS Ingestion Level**
   - Is the symbol being captured correctly?
   - Is the type information complete and accurate?
   - Are all dependencies being resolved?

3. **PSG Construction Level**
   - Is the function reachable from the entry point?
   - Are call edges being created correctly?
   - Is symbol correlation working?
   - Does the PSG show the full decomposed structure?

4. **Nanopass Level**
   - Are def-use edges being created?
   - Are operations being classified correctly?
   - Is the PSG fully enriched before MLIR generation?

5. **Alex/Zipper Level**
   - Is the traversal following the PSG structure?
   - Is MLIR generation based on node kinds, not symbol names?
   - Are there NO library-specific special cases?

6. **MLIR/LLVM Level**
   - Is the generated IR valid?
   - Are external declarations correct?
   - Is the calling convention appropriate?

## XParsec and the Zipper

> **The Zipper is the "attention" of Alex. XParsec provides type-safe pattern matching. Together they traverse the PSG and drive MLIR generation through Bindings.**

### The Zipper's Role

The Zipper traverses the PSG bidirectionally. It:
- Moves through the graph structure
- Provides context about the current position
- Enables focused operations at specific nodes
- Carries state through traversal

### XParsec's Role

XParsec provides composable pattern matchers that:
- Match against typed PSG structures
- Preserve type information through transformations
- Enable backtracking and alternatives

### Bindings' Role

Bindings contain platform-specific MLIR generation:
- Organized by platform (Linux_x86_64, Windows_x86_64, etc.)
- Looked up by PSG node structure
- Generate MLIR for syscalls, memory operations, etc.

### What NOT To Do

**DO NOT fall back to string-based parsing or pattern matching on stringified representations.**

```fsharp
// WRONG - String matching on symbol names
if symbolName.Contains("Console.Write") then ...

// WRONG - Hardcoded namespace paths
| Some name when name.StartsWith("MyLib.") -> ...

// RIGHT - Pattern match on PSG node structure and FNCS intrinsic markers
match node.SyntaxKind with
| "App:FunctionCall" -> processCall zipper bindings
| "Intrinsic:Sys.write" -> processSysWrite zipper bindings  // FNCS intrinsic
| "WhileLoop" -> processWhileLoop zipper bindings
```

## Essential Documentation

Before making changes, review these documents in `/docs/`:

| Document | Purpose |
|----------|---------|
| **`FNCS_Architecture.md`** | **CRITICAL: F# Native Compiler Services - native type resolution at source** |
| **`Architecture_Canonical.md`** | **AUTHORITATIVE: Two-layer model, platform bindings (BCL-free), nanopass pipeline** |
| **`PSG_Nanopass_Architecture.md`** | **CANONICAL: True nanopass pipeline, typed tree overlay, SRTP** |
| **`TypedTree_Zipper_Design.md`** | **Zipper implementation for FSharpExpr/PSG correlation** |
| `PSG_architecture.md` | PSG design decisions, node identity |
| `Alex_Architecture_Overview.md` | Alex overview (references canonical doc) |
| `XParsec_PSG_Architecture.md` | XParsec integration with Zipper |
| `HelloWorld_Lessons_Learned.md` | Common pitfalls and solutions |


## Sample Projects

Located in `/samples/console/`:

- `HelloWorld/` - Minimal validation sample
- `FidelityHelloWorld/` - Progressive complexity samples:
  - `01_HelloWorldDirect/` - Direct module calls
  - `02_HelloWorldSaturated/` - Saturated function calls
  - `03_HelloWorldHalfCurried/` - Pipe operators, partial application
  - `04_HelloWorldFullCurried/` - Full currying, Result.map, lambdas
- `TimeLoop/` - Mutable state, while loops, DateTime, Sleep

## Build, Test, and Validation Commands

> **CRITICAL: Compilation is NOT validation. The samples MUST be executed and their output validated.**

A change is NOT complete until:
1. The compiler builds successfully (`dotnet build`)
2. Sample programs compile to native binaries
3. **The native binaries execute correctly and produce expected output**

### Validation Protocol

After ANY change to the compiler pipeline, you MUST:

1. **Build the compiler**
2. **Compile a relevant sample** (start with `01_HelloWorldDirect`)
3. **Execute the binary interactively** and verify output
4. **Progress through samples** in order (01 → 02 → 03 → 04) as each succeeding sample tests more complex F# features

```bash
# Build the compiler
cd /home/hhh/repos/Firefly/src
dotnet build

# Compile sample 01
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile HelloWorld.fidproj

# CRITICAL: Execute and validate output
./HelloWorld
# Expected: "Hello, World!" (or prompts for input, then greeting)

# If sample 01 passes, progress to sample 02, 03, 04
cd ../02_HelloWorldSaturated
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile HelloWorld.fidproj
./HelloWorld

# With verbose output (for debugging)
Firefly compile HelloWorld.fidproj --verbose

# Keep intermediate files for debugging (.mlir, .ll files)
Firefly compile HelloWorld.fidproj -k
```

### Sample Progression

| Sample | Tests | Expected Behavior |
|--------|-------|-------------------|
| `01_HelloWorldDirect` | Static strings, basic Console calls | Prints greeting |
| `02_HelloWorldSaturated` | Let bindings, string interpolation | Prints formatted greeting |
| `03_HelloWorldHalfCurried` | Pipe operators, function values | Prints greeting with pipes |
| `04_HelloWorldFullCurried` | Full currying, Result.map, lambdas | Interactive name input, greeting |
| `TimeLoop` | Mutable state, while loops, Sleep | Countdown timer |

**Do NOT claim a feature works until you have executed the binary and verified the output.**

## Key Files

| File | Purpose |
|------|---------|
| `/src/Firefly.fsproj` | Main compiler project |
| `/src/Core/IngestionPipeline.fs` | Pipeline orchestration |
| `/src/Core/PSG/Builder.fs` | PSG construction |
| `/src/Core/PSG/Nanopass/*.fs` | PSG enrichment passes |
| `/src/Core/PSG/Reachability.fs` | Dead code elimination |
| `/src/Alex/Traversal/PSGZipper.fs` | Zipper traversal |
| `/src/Alex/Bindings/*.fs` | Platform-specific MLIR generation |
| `/src/Alex/Pipeline/CompilationOrchestrator.fs` | Full compilation |

## Common Pitfalls

1. **Missing Intrinsics**: Operations that aren't defined in FNCS CheckExpressions.fs. If an operation should compile to native code, it must be a recognized FNCS intrinsic.

2. **Namespace-Specific Logic**: Adding `if functionName = "X.Y"` logic anywhere in code generation. Alex should recognize intrinsic markers, not symbol names.

3. **Symbol Correlation**: FCS symbol correlation can fail silently. Check `[BUILDER] Warning:` messages in verbose output.

4. **Missing Nanopass**: If the PSG doesn't have the information you need, add a nanopass to enrich it. Don't compute it during MLIR generation.

5. **Layer Violations**: Any time you find yourself importing a module from a different pipeline stage, stop and reconsider.

6. **SRTP Blindness**: The syntax tree (`SynExpr`) shows SRTP operators as simple identifiers. Only the typed tree (`FSharpExpr.TraitCall`) knows the resolution. If your code doesn't have SRTP info, it means the typed tree overlay is incomplete.

7. **Hard-Delete Reachability**: Physically removing unreachable nodes breaks the typed tree zipper. Use soft-delete (mark IsReachable = false) to preserve structure for correlation.

8. **Wrong Nanopass Scope**: Different operator classes need different nanopasses. Pipe operators (`|>`, `<|`) are reduced by `ReducePipeOperators`. SRTP operators are a separate class. Don't mix concerns.

9. **Central Dispatch/Emitter**: Creating a handler registry that routes based on node kinds. This antipattern (PSGEmitter, PSGScribe) was removed twice. The Zipper folds, XParsec matches locally, Bindings provide platform implementations, MLIR Builder accumulates. NO routing table.

10. **Premature Centralization**: Pooling decision-making logic too early. Centralization is natural and correct at the OUTPUT (MLIR text), but wrong at DISPATCH (traversal logic). The PSG structure drives emission; there's no router.

## Project Configuration

Projects use `.fidproj` files (TOML format):

```toml
[package]
name = "ProjectName"

[compilation]
memory_model = "stack_only"
target = "native"

# No dependencies section needed - types and operations are FNCS intrinsics

[build]
sources = ["Main.fs"]
output = "binary_name"
output_kind = "freestanding"  # or "console"
```

## When in Doubt: The Zoom-Out Protocol

**Stop. Do not write code yet.**

1. **Read the docs** - Review all relevant documentation in `/docs/` completely
2. **Trace the pipeline** - Follow data flow from F# source to native binary
3. **Identify the layer** - Determine which pipeline stage contains the ROOT CAUSE:
   - FNCS (intrinsic definitions, type mappings)
   - FCS (parsing, type checking, symbol resolution)
   - PSG (semantic graph construction)
   - Nanopasses (PSG enrichment)
   - Alex/Zipper (traversal and MLIR generation)
   - MLIR/LLVM (IR lowering, optimization)
4. **Fix upstream** - Apply the fix at the earliest point where the defect exists
5. **Validate downstream** - Verify the fix propagates correctly through all stages to produce a working binary

**Remember**: In compiler development, the symptom location and the fix location are usually different. Resist the temptation to patch where you see the problem. Find where the problem originates and fix it there.

## The Acid Test

Before committing any change, ask:

> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.
