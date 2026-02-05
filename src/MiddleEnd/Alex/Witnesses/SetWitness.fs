/// SetWitness - Witness Set<'T> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY Set intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SetWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.CollectionPatterns
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Set operations - pure XParsec monadic observer
let private witnessSet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Set -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let setType = mapNativeTypeForArch arch node.Type

            match tryMatch (pSetEmpty node.Id setType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "Set.empty pattern emission failed"

        | "isEmpty" ->
            WitnessOutput.error "Set.isEmpty: Baker decomposes to structural check"

        | "value" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (setSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let elemType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pSetValue node.Id setSSA elemType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Set.value pattern emission failed"
                | None -> WitnessOutput.error "Set.value: Set not yet witnessed"
            | _ -> WitnessOutput.error $"Set.value: Expected 1 child, got {node.Children.Length}"

        | "left" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (setSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let subtreeType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pSetLeft node.Id setSSA subtreeType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Set.left pattern emission failed"
                | None -> WitnessOutput.error "Set.left: Set not yet witnessed"
            | _ -> WitnessOutput.error $"Set.left: Expected 1 child, got {node.Children.Length}"

        | "right" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (setSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let subtreeType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pSetRight node.Id setSSA subtreeType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Set.right pattern emission failed"
                | None -> WitnessOutput.error "Set.right: Set not yet witnessed"
            | _ -> WitnessOutput.error $"Set.right: Expected 1 child, got {node.Children.Length}"

        | "height" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (setSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let heightType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pSetHeight node.Id setSSA heightType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Set.height pattern emission failed"
                | None -> WitnessOutput.error "Set.height: Set not yet witnessed"
            | _ -> WitnessOutput.error $"Set.height: Expected 1 child, got {node.Children.Length}"

        | "node" ->
            match node.Children with
            | [elementId; leftId; rightId; heightId] ->
                match MLIRAccumulator.recallNode elementId ctx.Accumulator,
                      MLIRAccumulator.recallNode leftId ctx.Accumulator,
                      MLIRAccumulator.recallNode rightId ctx.Accumulator,
                      MLIRAccumulator.recallNode heightId ctx.Accumulator with
                | Some (elementSSA, elementType), Some (leftSSA, leftType), Some (rightSSA, rightType), Some _ ->
                    let element = { SSA = elementSSA; Type = elementType }
                    let left = { SSA = leftSSA; Type = leftType }
                    let right = { SSA = rightSSA; Type = rightType }
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let setType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pSetAdd node.Id element left right setType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Set.node pattern emission failed"
                | _ -> WitnessOutput.error "Set.node: One or more children not yet witnessed"
            | _ -> WitnessOutput.error $"Set.node: Expected 4 children, got {node.Children.Length}"

        | op -> WitnessOutput.error $"Unknown Set operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Set nanopass - witnesses Set intrinsic nodes
let nanopass : Nanopass = {
    Name = "Set"
    Witness = witnessSet
}
