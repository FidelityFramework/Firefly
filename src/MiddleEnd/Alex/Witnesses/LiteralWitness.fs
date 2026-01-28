/// Literal Witness - Witness literal values to MLIR via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Literal nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

/// Get result SSA for a node (the final SSA from its allocation)
let private getSingleSSA (nodeId: SemanticGraph.Types.NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "No result SSA for node %A" nodeId

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Literal nodes - category-selective (handles only Literal nodes)
let private witnessLiteralNode (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Use XParsec combinator to match literal
    match tryMatch pLiteral ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (lit, _) ->
        // Get SSA from coeffects
        let ssa = getSingleSSA node.Id ctx.Coeffects.SSA
        let arch = ctx.Coeffects.Platform.TargetArch

        // Call pattern to build MLIR (pattern returns PSGParser, need to execute it)
        match tryMatch (pBuildLiteral lit ssa arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "Literal pattern emission failed"

    | None ->
        // Not a literal node - skip
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Literal nanopass - witnesses Literal nodes (int, bool, char, float, etc.)
let nanopass : Nanopass = {
    Name = "Literal"
    Witness = witnessLiteralNode
}
