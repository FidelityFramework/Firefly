/// LazyWitness - Witness Lazy<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lazy-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy operations - category-selective (handles only Lazy nodes)
let private witnessLazy (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try LazyExpr pattern
    match tryMatch pLazyExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((bodyId, captureInfos), _) ->
        match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
        | None -> WitnessOutput.error "LazyExpr: No SSAs assigned"
        | Some ssas ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let captures =
                captureInfos
                |> List.choose (fun capture ->
                    capture.SourceNodeId
                    |> Option.bind (fun id -> MLIRAccumulator.recallNode id ctx.Accumulator)
                    |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty }))

            match MLIRAccumulator.recallNode bodyId ctx.Accumulator with
            | None -> WitnessOutput.error "LazyExpr: Body not yet witnessed"
            | Some (codePtr, _) ->
                match tryMatch (pBuildLazyStruct codePtr captures ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "LazyExpr pattern emission failed"

    | None ->
        // Try LazyForce pattern
        match tryMatch pLazyForce ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (lazyNodeId, _) ->
            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "LazyForce: No SSAs assigned"
            | Some ssas ->
                let arch = ctx.Coeffects.Platform.TargetArch
                match MLIRAccumulator.recallNode lazyNodeId ctx.Accumulator with
                | None -> WitnessOutput.error "LazyForce: Lazy value not yet witnessed"
                | Some (lazySSA, _) ->
                    match tryMatch (pBuildLazyForce lazySSA ssas.[3] ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "LazyForce pattern emission failed"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Lazy nanopass - witnesses LazyExpr and LazyForce nodes
let nanopass : Nanopass = {
    Name = "Lazy"
    Witness = witnessLazy
}
