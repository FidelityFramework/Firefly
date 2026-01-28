/// ArithWitness - Witness arithmetic/comparison/bitwise operations to MLIR via XParsec
///
/// Uses XParsec combinators to match PSG structure, delegates to Patterns for MLIR elision.
/// Handles: binary arithmetic, comparisons, bitwise operations, boolean logic, unary operations.
///
/// NANOPASS: This witness handles ONLY arithmetic/comparison/bitwise nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ArithWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness arithmetic operations - category-selective (handles only arithmetic/comparison/bitwise nodes)
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try each arithmetic pattern in sequence
    // Patterns need implementation - skipping for now
    match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None ->
        match tryMatch pComparison ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None ->
            match tryMatch pBitwise ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None ->
                match tryMatch pUnary ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None ->
                    // Not an arithmetic operation - skip
                    WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Arithmetic nanopass - witnesses binary/unary arithmetic, comparisons, bitwise ops
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
