/// LazyWitness - Witness Lazy<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators

/// Witness LazyExpr - uses XParsec to match, Patterns to elide
let witnessLazyExpr (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // tryMatch needs: graph, node, zipper, platform
    match PSGZipper.create ctx.Graph node.Id with
    | None ->
        WitnessOutput.error "Could not create zipper for LazyExpr"
    | Some zipper ->
        match tryMatch pLazyExpr ctx.Graph node zipper ctx.Coeffects.Platform with
        | Some ((bodyId, captures), _) ->
            // TODO: Call Pattern to build lazy struct
            WitnessOutput.error "LazyExpr matched via XParsec - Pattern integration next"
        | None ->
            WitnessOutput.error "LazyExpr XParsec pattern match failed"

/// Witness LazyForce - uses XParsec to match, Patterns to elide
let witnessLazyForce (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match PSGZipper.create ctx.Graph node.Id with
    | None ->
        WitnessOutput.error "Could not create zipper for LazyForce"
    | Some zipper ->
        match tryMatch pLazyForce ctx.Graph node zipper ctx.Coeffects.Platform with
        | Some (lazyNodeId, _) ->
            // TODO: Call Pattern to force lazy
            WitnessOutput.error "LazyForce matched via XParsec - Pattern integration next"
        | None ->
            WitnessOutput.error "LazyForce XParsec pattern match failed"
