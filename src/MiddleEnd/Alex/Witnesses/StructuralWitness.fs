/// StructuralWitness - Transparent witness for structural container nodes
///
/// Structural nodes (ModuleDef, Sequential, PatternBinding) organize the PSG tree but don't emit MLIR.
/// They need to be "witnessed" per the Domain Responsibility Principle to avoid coverage gaps.
///
/// NANOPASS: This witness handles ONLY structural container nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.StructuralWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness structural nodes transparently - they organize but don't emit ops
let private witnessStructural (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.ModuleDef (moduleName, _) ->
        printfn "[StructuralWitness] Handling ModuleDef: %s (node %A)" moduleName node.Id
        // Module definition - structural container, children already witnessed
        // Return TRVoid per Domain Responsibility Principle
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.Sequential _ ->
        // Sequential node - structural container for imperative sequences
        // Children already witnessed in post-order, return TRVoid
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.PatternBinding _ ->
        // Function parameter binding - SSA pre-assigned in coeffects (no MLIR generation)
        // This is a structural marker (like .NET/F# idiom for argv, destructured params)
        // VarRef nodes lookup these bindings from coeffects
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | _ ->
        // Not a structural node - skip for other witnesses
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Structural nanopass - transparent witness for container nodes
let nanopass : Nanopass = {
    Name = "Structural"
    Witness = witnessStructural
}
