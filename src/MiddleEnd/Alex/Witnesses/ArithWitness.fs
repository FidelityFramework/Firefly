/// ArithWitness - Witness arithmetic/comparison/bitwise operations to MLIR
///
/// Uses XParsec combinators to match PSG structure, delegates to Patterns for MLIR elision.
/// Handles: binary arithmetic, comparisons, bitwise operations, boolean logic, unary operations.
module Alex.Witnesses.ArithWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators

/// Witness binary arithmetic operation (addition, subtraction, multiplication, division, modulo)
let witnessBinaryArith (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBinaryArith ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Binary arithmetic pattern match failed"

/// Witness comparison operation (<, <=, >, >=, ==, !=)
let witnessComparison (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pComparison ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Comparison pattern match failed"

/// Witness bitwise operation (&, |, ^, <<, >>)
let witnessBitwise (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBitwise ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Bitwise operation pattern match failed"

/// Witness unary operation (-, not, ~)
let witnessUnary (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pUnary ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "Unary operation pattern match failed"
