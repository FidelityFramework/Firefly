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

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy operations - category-selective (handles only Lazy nodes)
let private witnessLazy (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try LazyExpr pattern
    match tryMatch pLazyExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((bodyId, captures), _) ->
        // TODO: Integrate with pLazyStruct pattern
        WitnessOutput.error "LazyExpr matched - Pattern integration needed"
    | None ->
        // Try LazyForce pattern
        match tryMatch pLazyForce ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (lazyNodeId, _) ->
            // TODO: Integrate with pLazyForce pattern
            WitnessOutput.error "LazyForce matched - Pattern integration needed"
        | None ->
            // Not a lazy node - skip
            WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Lazy nanopass - witnesses LazyExpr and LazyForce nodes
let nanopass : Nanopass = {
    Name = "Lazy"
    Witness = witnessLazy
}
