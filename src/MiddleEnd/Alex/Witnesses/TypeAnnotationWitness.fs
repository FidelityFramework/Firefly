/// TypeAnnotationWitness - Transparent wrapper witness using XParsec
///
/// TypeAnnotation nodes are transparent wrappers that annotate expressions with types.
/// This witness forwards the wrapped node's result without generating operations.
///
/// ARCHITECTURAL PRINCIPLE: This witness demonstrates proper XParsec monadic structure:
/// - ONLY uses tryMatch on XParsec patterns
/// - NO direct graph API access
/// - NO fallback computation logic
/// - Proper debug tooling for transparency
module Alex.Witnesses.TypeAnnotationWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns  // For findLastValueNode

// ═══════════════════════════════════════════════════════════
// WITNESS IMPLEMENTATION (XParsec patterns only)
// ═══════════════════════════════════════════════════════════

/// Witness TypeAnnotation nodes - transparent wrapper pattern
/// This witness demonstrates proper monadic parser combinator structure
let private witnessTypeAnnotation (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Use XParsec pattern to extract wrapped node ID
    match tryMatch pTypeAnnotation ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((wrappedId, annotatedType), _) ->
        // Traverse Sequential structure to find actual value-producing node
        // Sequential nodes are structural scaffolding - not witnesses
        let actualValueNode = Alex.Patterns.ElisionPatterns.findLastValueNode wrappedId ctx.Graph
        
        // Recall the actual value node's result from accumulator
        // TypeAnnotation is transparent - it forwards whatever the wrapped node produced
        match MLIRAccumulator.recallNode actualValueNode ctx.Accumulator with
        | Some (wrappedSSA, wrappedType) ->
            // Bind this node to the wrapped result (transparent pass-through)
            MLIRAccumulator.bindNode node.Id wrappedSSA wrappedType ctx.Accumulator
            // Return empty - TypeAnnotation doesn't generate operations, just forwards results
            WitnessOutput.empty

        | None ->
            // Wrapped node produced no result (e.g., unit value, void expression)
            // TypeAnnotation forwards TRVoid as well - this is normal for unit-typed expressions
            WitnessOutput.empty

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION
// ═══════════════════════════════════════════════════════════

/// TypeAnnotation nanopass - ContentPhase (runs during body witnessing)
let nanopass : Nanopass = {
    Name = "TypeAnnotation"
    Witness = witnessTypeAnnotation
}
