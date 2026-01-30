/// SequentialWitness - Witness sequential composition nodes
///
/// Sequential nodes don't emit MLIR - they forward the last child's SSA.
/// All child operations are already emitted by child witnesses (post-order).
///
/// NANOPASS: This witness handles ONLY Sequential nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SequentialWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness sequential nodes - forwards last child's SSA
let private witnessSequential (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pSequential ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (childIds, _) ->
        // Sequential forwards the last child's result
        match List.tryLast childIds with
        | Some lastId ->
            match MLIRAccumulator.recallNode lastId ctx.Accumulator with
            | Some (ssa, ty) ->
                // Forward last child's SSA
                { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
            | None ->
                // Last child has no SSA (e.g., unit literal) - return void
                // This is normal for sequences ending in unit expressions
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        | None ->
            // Empty sequential - return void
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Sequential nanopass - witnesses sequential composition
let nanopass : Nanopass = {
    Name = "Sequential"
    Phase = ContentPhase
    Witness = witnessSequential
}
