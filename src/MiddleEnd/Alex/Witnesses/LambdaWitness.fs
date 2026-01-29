/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
let private witnessLambda (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((params', bodyId, captureInfos), _) ->
        match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
        | None -> WitnessOutput.error "Lambda: No SSAs assigned"
        | Some ssas ->
            // Extract captures as Val list
            let captures =
                captureInfos
                |> List.choose (fun capture ->
                    capture.SourceNodeId
                    |> Option.bind (fun id -> MLIRAccumulator.recallNode id ctx.Accumulator)
                    |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty }))

            // Witness lambda body as sub-graph (returns operation list)
            // For now, return error - body witnessing needs sub-graph combinator
            WitnessOutput.error "Lambda body witnessing requires sub-graph combinator - not yet implemented"

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Lambda nanopass - witnesses Lambda nodes
let nanopass : Nanopass = {
    Name = "Lambda"
    Witness = witnessLambda
}
