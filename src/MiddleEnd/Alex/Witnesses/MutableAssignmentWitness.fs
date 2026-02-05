/// MutableAssignmentWitness - Witness mutable variable assignment (x <- value)
///
/// Handles F# mutable assignment via SemanticKind.Set nodes.
/// Emits memref.store to update the mutable variable.
///
/// NANOPASS: This witness handles ONLY Set nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MutableAssignmentWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness mutable assignment nodes (x <- value)
let private witnessMutableAssignment (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pSet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((targetId, valueId), _) ->
        // Get memref SSA from target (VarRef to mutable binding)
        // Get value SSA from value node
        match MLIRAccumulator.recallNode targetId ctx.Accumulator,
              MLIRAccumulator.recallNode valueId ctx.Accumulator with
        | Some (memrefSSA, TMemRef elemType), Some (valueSSA, _) ->
            // Emit memref.store to update mutable variable
            let (NodeId nodeIdInt) = node.Id
            match tryMatch (pStoreMutableVariable nodeIdInt memrefSSA valueSSA elemType)
                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) ->
                { InlineOps = ops; TopLevelOps = []; Result = result }
            | None ->
                WitnessOutput.error "Mutable assignment: Store pattern emission failed"
        | Some (_, ty), Some _ ->
            WitnessOutput.error $"Mutable assignment: Target is not a mutable variable (type: {ty})"
        | Some _, None ->
            WitnessOutput.error "Mutable assignment: Value not yet witnessed"
        | None, _ ->
            WitnessOutput.error "Mutable assignment: Target variable not yet witnessed"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// MutableAssignment nanopass - witnesses mutable variable assignment
let nanopass : Nanopass = {
    Name = "MutableAssignment"
    Witness = witnessMutableAssignment
}
