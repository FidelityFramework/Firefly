# Alex Zipper Architecture (Updated January 2026)

## Current Architecture: Observation-Driven Emission

The Alex emission layer uses an **observation-driven model** based on codata and coeffects:

```
PSG from FNCS (already transformed)
         ↓
Alex Preprocessing (analysis → coeffects)
         ↓
MLIRZipper fold (observation → template selection → MLIR)
         ↓
MLIR Output
```

**Key Principle**: "The photographer doesn't build the scene - they arrive AFTER the scene is complete and witness it."

## Core Components

### 1. MLIRZipper (`Alex/Traversal/MLIRZipper.fs`)

The MLIRZipper provides:
- **Navigation**: Move through PSG structure
- **State threading**: SSA counters, NodeSSA map, string literals
- **Fold operations**: `foldPostOrder` for emission traversal
- **NO transform logic** - pure navigation

```fsharp
type MLIRZipper = {
    Focus: MLIRFocus
    Path: MLIRPath
    CurrentOps: MLIROp list
    State: MLIRState
    Globals: MLIRGlobal list
}
```

### 2. Witnesses (`Alex/Witnesses/`)

Each witness:
1. **OBSERVES** the focused node
2. **OBSERVES** coeffects (pre-computed in Preprocessing)
3. **SELECTS** appropriate template
4. **RETURNS** updated state with emitted MLIR

Witness files:
- `LiteralWitness.fs` - Integer, float, string constants
- `BindingWitness.fs` - Let bindings (mutable/immutable)
- `ApplicationWitness.fs` - Function applications, intrinsics
- `ControlFlowWitness.fs` - If/while/for, SCF operations
- `MemoryWitness.fs` - Load/store, alloca, GEP
- `LambdaWitness.fs` - Lambda expressions

### 3. Templates (`Alex/Dialects/` - TARGET ARCHITECTURE)

**CRITICAL: Templates are STRUCTURED TYPES that COMPOSE UPWARD, not sprintf wrappers.**

Each dialect has its own folder with:
- `Types.fs` - DU defining structured operations for that dialect
- `Templates.fs` - Functions returning structured MLIROp values
- `Combinators.fs` - XParsec combinators that compose templates

```
Alex/Dialects/
├── Core/
│   ├── Types.fs          # MLIROp, SSA, Val, Block, Region
│   └── Serialize.fs      # THE ONLY PLACE for string generation
├── Arith/
│   ├── Types.fs          # type ArithOp = AddI of BinaryParams | SubI of ... 
│   ├── Templates.fs      # let addI p = ArithOp.AddI p (returns structured type!)
│   └── Combinators.fs    # let pBinaryArith = ... (XParsec composing templates)
├── LLVM/
│   ├── Types.fs          # type LLVMOp = AddressOf of ... | Load of ...
│   ├── Templates.fs      # let addressOf r n = LLVMOp.AddressOf { Result=r; Name=n }
│   └── Combinators.fs    # let pLoadGlobal = ... (composes addressOf + load)
├── SCF/
│   └── ...
└── Func/
    └── ...
```

**Templates are LEMMAS, XParsec COMPOSES them into PROOFS**:
```fsharp
// Template returns STRUCTURED TYPE (not string!)
let addressOf (result: SSA) (globalName: string) : LLVMOp =
    LLVMOp.AddressOf { Result = result; GlobalName = globalName }

// XParsec COMPOSES templates upward
let pLoadGlobalString : XParsec<MLIRExpr> =
    pGlobalStringRef >>= fun globalRef ->
    template LLVM.addressOf globalRef.Name >>= fun addrOp ->
    template LLVM.load addrOp.Result >>= fun loadOp ->
    return (MLIRExpr.Composed [addrOp; loadOp])

// Serialization ONLY at boundary (Serialize.fs)
let emit (op: MLIROp) : string = sprintf ... // ONLY HERE
```

**WRONG (Current Polluted State)**:
```fsharp
// DON'T DO THIS - sprintf wrapped in quotation
let addI = <@ fun p -> sprintf "%s = arith.addi %s, %s : %s" ... @>
```

### 4. XParsec Combinators (`Alex/XParsec/`) - THE GLUE LAYER

**XParsec is THE composition engine. It composes templates UPWARD into larger structures.**

XParsec is NOT just pattern matching - it is the glue that:
1. Matches PSG patterns
2. COMPOSES dialect templates into larger MLIR structures
3. Threads SSA state through composition
4. Produces structured MLIROp trees (NOT strings)

```fsharp
// XParsec composes templates like lemmas compose into proofs
let pLoadStructField : XParsec<MLIRExpr> =
    pStructPtr >>= fun structPtr ->           // Match: we have struct ptr
    pFieldIndex >>= fun fieldIdx ->           // Match: we have field index
    template LLVM.gep structPtr fieldIdx >>= fun gepOp ->   // Compose: GEP template
    template LLVM.load gepOp.Result >>= fun loadOp ->       // Compose: Load template
    return (MLIRExpr.Sequence [gepOp; loadOp])              // Result: composed structure

// Complex control flow - XParsec composes SCF templates
let pWhileLoop : XParsec<MLIRExpr> =
    pCondition >>= fun cond ->
    pBody >>= fun body ->
    template SCF.whileOp cond body >>= fun whileOp ->  // SCF template
    return (MLIRExpr.ControlFlow whileOp)
```

**Files**:
- `PSGCombinators.fs` - PSG pattern matching combinators
- `TemplateComposers.fs` - Combinators that compose dialect templates
- `MLIRExpr.fs` - Structured MLIR expression type (result of composition)

### 5. Preprocessing (`Alex/Preprocessing/`)

**Analysis only, no transforms**:
- `MutabilityAnalysis.fs` - Which bindings need alloca vs SSA
- `SSAAssignment.fs` - Pre-assign SSA names
- `PatternBindingAnalysis.fs` - Pattern binding extraction

Output: Coeffects (metadata attached to nodes)

### 6. Transfer (`Alex/Traversal/FNCSTransfer.fs`)

The main transfer pipeline that:
- Receives SemanticGraph from FNCS
- Creates MLIRZipper from entry point
- Folds with witness functions
- Produces MLIR output

## The Four Pillars

| Pillar | Role | Location |
|--------|------|----------|
| **Coeffects** | Pre-computed analysis results | `Alex/Preprocessing/` |
| **Templates** | MLIR structure as data | `Alex/Templates/` |
| **Zipper** | Navigation + fold | `Alex/Traversal/MLIRZipper.fs` |
| **XParsec** | Combinator-based recognition | `Alex/XParsec/` |

## What Changed (January 2026)

| Before | After |
|--------|-------|
| Multiple transfer files | Single `FNCSTransfer.fs` |
| Transform logic in witnesses | Observation only |
| sprintf throughout | Template-based emission |
| String matching on names | SemanticKind/IntrinsicInfo dispatch |
| No XParsec infrastructure | Full combinator library |

## Current Metrics

- **sprintf in Witnesses**: 57 (down from 138)
- **sprintf in MLIRZipper**: 47 (down from 53)
- **MLIRZipper lines**: 1493 (SCF complexity)

## Marker Strings (Technical Debt)

Current workarounds for incomplete FNCS transforms:
- `$pipe:` - Pipe operator chains
- `$partial:` - Partial application
- `$platform:` - Platform bindings

These markers are necessary until FNCS provides complete semantic transforms.

## Removed Files

- `PSGEmitter.fs` - Central dispatch (antipattern)
- `PSGScribe.fs` - Recreated same antipattern
- `MLIRTransfer.fs` - Replaced by FNCSTransfer
- `SaturateApplications.fs` - Moved to FNCS

## Validation

Samples 01-04 compile and run correctly with this architecture.
