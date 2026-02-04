/// IntrinsicWitness - Skip bare intrinsic nodes (function references)
///
/// Intrinsic nodes represent function values when not being applied.
/// ApplicationWitness handles intrinsic applications.
/// This witness skips bare intrinsic references.
///
/// NANOPASS: This witness handles ONLY bare Intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.IntrinsicWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness intrinsic nodes - compile-time function references
let private witnessIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Intrinsic _ ->
        // Intrinsic function reference (compile-time, no SSA, no MLIR)
        // ApplicationWitness handles applications
        // Return void to mark as witnessed but produce no value
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
    | _ ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Intrinsic nanopass - skips bare intrinsic nodes
let nanopass : Nanopass = {
    Name = "Intrinsic"
    Witness = witnessIntrinsic
}
