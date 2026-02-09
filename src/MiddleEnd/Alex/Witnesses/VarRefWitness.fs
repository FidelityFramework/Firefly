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
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns  // pLoadMutableVariable
open Alex.Dialects.Core.Types  // TMemRef
open Alex.CodeGeneration.TypeMapping
open XParsec
open XParsec.Parsers
open XParsec.Combinators

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
                    // Extract SSA monadically
                    let patternBindingPattern =
                        parser {
                            let! ssa = getNodeSSA bindingId
                            let! state = getUserState
                            let arch = state.Coeffects.Platform.TargetArch
                            let ty = mapNativeTypeForArch arch bindingNode.Type
                            return ([], TRValue { SSA = ssa; Type = ty })
                        }

                    match tryMatch patternBindingPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error $"VarRef '{name}': PatternBinding has no SSA in coeffects"

                | SemanticKind.Binding (_, isMut, _, _) ->
                    // Check if binding's child is a Lambda (function binding)
                    let isFunctionBinding =
                        bindingNode.Children
                        |> List.tryHead
                        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId ctx.Graph)
                        |> Option.map (fun childNode -> childNode.Kind.ToString().StartsWith("Lambda"))
                        |> Option.defaultValue false

                    if isFunctionBinding then
                        // Function reference - structural node, no SSA value produced
                        // ApplicationWitness navigates to VarRef for name resolution independently
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    elif Set.contains bindingId ctx.Coeffects.CurryFlattening.PartialAppBindings then
                        // Partial application binding - no value SSA available
                        // ApplicationWitness handles saturated calls through the coeffect
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    else
                        // Value binding - post-order: binding already witnessed, recall its SSA
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // Auto-load ONLY if the Binding is mutable (isMut from PSG).
                            // Mutable bindings hold memref<1xT> cells that need memref.load.
                            // Immutable bindings (including NativePtr.stackalloc results) forward as-is.
                            if isMut then
                                // Mutable cell — extract element type for auto-load
                                let elemTypeOpt =
                                    match ty with
                                    | TMemRef elemType -> Some elemType
                                    | TMemRefStatic (_, elemType) -> Some elemType
                                    | _ -> None
                                match elemTypeOpt with
                                | Some elemType ->
                                    let (NodeId nodeIdInt) = node.Id
                                    match tryMatch (pLoadMutableVariable nodeIdInt ssa elemType)
                                                  ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                    | Some ((ops, result), _) ->
                                        { InlineOps = ops; TopLevelOps = []; Result = result }
                                    | None ->
                                        WitnessOutput.error $"VarRef '{name}': Mutable variable load failed"
                                | None ->
                                    WitnessOutput.error $"VarRef '{name}': Mutable cell has unexpected type {ty}"
                            else
                                // Immutable value (including buffers): forward directly
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
