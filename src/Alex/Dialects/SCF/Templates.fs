/// SCF (Structured Control Flow) Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → SCFOp
/// NO sprintf. NO string formatting. Just data construction.
module Alex.Dialects.SCF.Templates

open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// CONDITIONAL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// If-then-else: scf.if
let scfIf (results: SSA list) (cond: SSA) (thenRegion: Region) (elseRegion: Region option) (resultTypes: MLIRType list) : SCFOp =
    SCFOp.If (results, cond, thenRegion, elseRegion, resultTypes)

/// Simple if without else (void result)
let scfIfSimple (cond: SSA) (thenRegion: Region) : SCFOp =
    SCFOp.If ([], cond, thenRegion, None, [])

/// If with else (both branches must yield same types)
let scfIfElse (results: SSA list) (cond: SSA) (thenRegion: Region) (elseRegion: Region) (resultTypes: MLIRType list) : SCFOp =
    SCFOp.If (results, cond, thenRegion, Some elseRegion, resultTypes)

// ═══════════════════════════════════════════════════════════════════════════
// LOOP OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// While loop: scf.while
let scfWhile (results: SSA list) (condRegion: Region) (bodyRegion: Region) (iterArgs: Val list) : SCFOp =
    SCFOp.While (results, condRegion, bodyRegion, iterArgs)

/// Simple while without iter_args
let scfWhileSimple (condRegion: Region) (bodyRegion: Region) : SCFOp =
    SCFOp.While ([], condRegion, bodyRegion, [])

/// For loop: scf.for
let scfFor (results: SSA list) (iv: SSA) (start: SSA) (stop: SSA) (step: SSA) (bodyRegion: Region) (iterArgs: Val list) : SCFOp =
    SCFOp.For (results, iv, start, stop, step, bodyRegion, iterArgs)

/// Simple for loop without iter_args
let scfForSimple (iv: SSA) (start: SSA) (stop: SSA) (step: SSA) (bodyRegion: Region) : SCFOp =
    SCFOp.For ([], iv, start, stop, step, bodyRegion, [])

// ═══════════════════════════════════════════════════════════════════════════
// REGION TERMINATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Yield values from region: scf.yield
let scfYield (values: Val list) : SCFOp =
    SCFOp.Yield values

/// Yield no values (void)
let scfYieldVoid : SCFOp =
    SCFOp.Yield []

/// Yield single value
let scfYieldVal (value: Val) : SCFOp =
    SCFOp.Yield [value]

/// While condition: scf.condition
let scfCondition (cond: SSA) (args: Val list) : SCFOp =
    SCFOp.Condition (cond, args)

/// Simple while condition without forwarding args
let scfConditionSimple (cond: SSA) : SCFOp =
    SCFOp.Condition (cond, [])

// ═══════════════════════════════════════════════════════════════════════════
// REGION BUILDERS
// ═══════════════════════════════════════════════════════════════════════════

/// Create an empty region with a single entry block
let emptyRegion (entryLabel: string) : Region =
    {
        Blocks = [
            {
                Label = BlockRef entryLabel
                Args = []
                Ops = []
            }
        ]
    }

/// Create a region with a single block containing operations
let singleBlockRegion (label: string) (args: BlockArg list) (ops: MLIROp list) : Region =
    {
        Blocks = [
            {
                Label = BlockRef label
                Args = args
                Ops = ops
            }
        ]
    }

/// Create a region with implicit entry block (no label, no args)
/// Used for SCF while/for regions where block args are defined by the operation
let implicitEntryRegion (ops: MLIROp list) : Region =
    {
        Blocks = [
            {
                Label = BlockRef ""  // Empty = no label (implicit entry)
                Args = []            // Args defined by operation signature
                Ops = ops
            }
        ]
    }

/// Create a block
let block (label: string) (args: BlockArg list) (ops: MLIROp list) : Block =
    {
        Label = BlockRef label
        Args = args
        Ops = ops
    }

/// Create a block argument
let blockArg (ssa: SSA) (ty: MLIRType) : BlockArg =
    { SSA = ssa; Type = ty }

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a while loop structure for mutable iteration
/// condOps should end with scf.condition
/// bodyOps should end with scf.yield
let whilePattern (condOps: MLIROp list) (bodyOps: MLIROp list) (iterArgs: Val list) : SCFOp =
    let condRegion = singleBlockRegion "before" (iterArgs |> List.map (fun v -> blockArg v.SSA v.Type)) condOps
    let bodyRegion = singleBlockRegion "do" (iterArgs |> List.map (fun v -> blockArg v.SSA v.Type)) bodyOps
    SCFOp.While (iterArgs |> List.map (fun v -> v.SSA), condRegion, bodyRegion, iterArgs)

/// Create a for loop structure
/// bodyOps should end with scf.yield
let forPattern (iv: SSA) (start: SSA) (stop: SSA) (step: SSA) (bodyOps: MLIROp list) (iterArgs: Val list) : SCFOp =
    let bodyRegion = singleBlockRegion "body" (blockArg iv MLIRTypes.index :: (iterArgs |> List.map (fun v -> blockArg v.SSA v.Type))) bodyOps
    SCFOp.For (iterArgs |> List.map (fun v -> v.SSA), iv, start, stop, step, bodyRegion, iterArgs)

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap SCFOp in MLIROp
let wrap (op: SCFOp) : MLIROp = MLIROp.SCFOp op
