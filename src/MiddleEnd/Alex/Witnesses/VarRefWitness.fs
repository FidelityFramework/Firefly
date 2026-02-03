/// VarRefWitness - Witness variable reference nodes
///
/// Variable references don't emit MLIR - they forward the binding's SSA.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// FUNCTION REFERENCES: VarRef nodes pointing to function bindings (Lambda nodes) are SKIPPED.
/// ApplicationWitness handles extracting function names directly from VarRef nodes.
///
/// NANOPASS: This witness handles ONLY VarRef nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.VarRefWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.CodeGeneration.TypeMapping

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, bindingIdOpt), _) ->
        match bindingIdOpt with
        | Some bindingId ->
            // Check if the binding references a function (Lambda node)
            match SemanticGraph.tryGetNode bindingId ctx.Graph with
            | Some bindingNode ->
                // Check binding type
                match bindingNode.Kind with
                | SemanticKind.PatternBinding _ ->
                    // Parameter binding - SSA is in coeffects, not accumulator
                    match SSAAssign.lookupSSA bindingId ctx.Coeffects.SSA with
                    | Some ssa ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let ty = mapNativeTypeForArch arch bindingNode.Type
                        { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                    | None ->
                        WitnessOutput.error $"VarRef '{name}': PatternBinding has no SSA in coeffects"

                | SemanticKind.Binding _ ->
                    // Check if binding's child is a Lambda (function binding)
                    let isFunctionBinding =
                        bindingNode.Children
                        |> List.tryHead
                        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId ctx.Graph)
                        |> Option.map (fun childNode -> childNode.Kind.ToString().StartsWith("Lambda"))
                        |> Option.defaultValue false

                    if isFunctionBinding then
                        // Function reference - ApplicationWitness handles this
                        WitnessOutput.skip
                    else
                        // Value binding - post-order: binding already witnessed, recall its SSA
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // Forward the binding's SSA - VarRef doesn't emit ops
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                        | None ->
                            WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"

                | _ ->
                    WitnessOutput.error $"VarRef '{name}': Unexpected binding kind {bindingNode.Kind}"
            | None ->
                WitnessOutput.error $"VarRef '{name}': Binding node not found"
        | None ->
            WitnessOutput.error $"VarRef '{name}': No binding ID (unresolved reference)"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// VarRef nanopass - witnesses variable references
let nanopass : Nanopass = {
    Name = "VarRef"
    Witness = witnessVarRef
}
