/// ArithWitness - Witness arithmetic/comparison/bitwise operations to MLIR via XParsec
///
/// Uses XParsec combinators to match PSG structure, delegates to Patterns for MLIR elision.
/// Handles: binary arithmetic, comparisons, bitwise operations, boolean logic, unary operations.
///
/// NANOPASS: This witness handles ONLY arithmetic/comparison/bitwise nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ArithWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness arithmetic operations - category-selective (handles only arithmetic/comparison/bitwise nodes)
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pClassifiedIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((info, category), _) ->
        match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
        | None -> WitnessOutput.error "Arithmetic: No SSA assigned"
        | Some resultSSA ->
            let arch = ctx.Coeffects.Platform.TargetArch

            // Extract binary operation operands from children
            // Post-order traversal ensures children are already witnessed
            match node.Children with
            | [lhsId; rhsId] ->
                match MLIRAccumulator.recallNode lhsId ctx.Accumulator, MLIRAccumulator.recallNode rhsId ctx.Accumulator with
                | Some (lhsSSA, _), Some (rhsSSA, _) ->
                    match category with
                    | BinaryArith _ ->
                        match tryMatch (pBuildBinaryArith resultSSA lhsSSA rhsSSA arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Binary arithmetic pattern emission failed"

                    | Comparison _ ->
                        match tryMatch (pBuildComparison resultSSA lhsSSA rhsSSA arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Comparison pattern emission failed"

                    | _ -> WitnessOutput.skip

                | _ -> WitnessOutput.error $"Arithmetic operands not yet witnessed for {info.FullName}"

            | _ -> WitnessOutput.error $"Unexpected operand count for binary operation {info.FullName}"

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Arithmetic nanopass - witnesses binary/unary arithmetic, comparisons, bitwise ops
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
