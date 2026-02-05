/// MapWitness - Witness Map<'K,'V> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY Map intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MapWitness

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

/// Witness Map operations - pure XParsec monadic observer
let private witnessMap (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Map -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let mapType = mapNativeTypeForArch arch node.Type

            match tryMatch (pMapEmpty node.Id mapType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "Map.empty pattern emission failed"

        | "isEmpty" ->
            WitnessOutput.error "Map.isEmpty: Baker decomposes to structural check"

        | "key" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let keyType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapKey node.Id mapSSA keyType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.key pattern emission failed"
                | None -> WitnessOutput.error "Map.key: Map not yet witnessed"
            | _ -> WitnessOutput.error $"Map.key: Expected 1 child, got {node.Children.Length}"

        | "value" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let valueType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapValue node.Id mapSSA valueType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.value pattern emission failed"
                | None -> WitnessOutput.error "Map.value: Map not yet witnessed"
            | _ -> WitnessOutput.error $"Map.value: Expected 1 child, got {node.Children.Length}"

        | "left" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let subtreeType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapLeft node.Id mapSSA subtreeType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.left pattern emission failed"
                | None -> WitnessOutput.error "Map.left: Map not yet witnessed"
            | _ -> WitnessOutput.error $"Map.left: Expected 1 child, got {node.Children.Length}"

        | "right" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let subtreeType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapRight node.Id mapSSA subtreeType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.right pattern emission failed"
                | None -> WitnessOutput.error "Map.right: Map not yet witnessed"
            | _ -> WitnessOutput.error $"Map.right: Expected 1 child, got {node.Children.Length}"

        | "height" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let heightType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapHeight node.Id mapSSA heightType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.height pattern emission failed"
                | None -> WitnessOutput.error "Map.height: Map not yet witnessed"
            | _ -> WitnessOutput.error $"Map.height: Expected 1 child, got {node.Children.Length}"

        | "node" ->
            match node.Children with
            | [keyId; valueId; leftId; rightId; heightId] ->
                match MLIRAccumulator.recallNode keyId ctx.Accumulator,
                      MLIRAccumulator.recallNode valueId ctx.Accumulator,
                      MLIRAccumulator.recallNode leftId ctx.Accumulator,
                      MLIRAccumulator.recallNode rightId ctx.Accumulator,
                      MLIRAccumulator.recallNode heightId ctx.Accumulator with
                | Some (keySSA, keyType), Some (valueSSA, valueType), Some (leftSSA, leftType), Some (rightSSA, rightType), Some _ ->
                    let key = { SSA = keySSA; Type = keyType }
                    let value = { SSA = valueSSA; Type = valueType }
                    let left = { SSA = leftSSA; Type = leftType }
                    let right = { SSA = rightSSA; Type = rightType }
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mapType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pMapAdd node.Id key value left right mapType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Map.node pattern emission failed"
                | _ -> WitnessOutput.error "Map.node: One or more children not yet witnessed"
            | _ -> WitnessOutput.error $"Map.node: Expected 5 children, got {node.Children.Length}"

        | "add" | "tryFind" | "remove" | "containsKey" ->
            WitnessOutput.error $"Map.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown Map operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Map nanopass - witnesses Map intrinsic nodes
let nanopass : Nanopass = {
    Name = "Map"
    Witness = witnessMap
}
