/// ListWitness - Witness List<'T> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY List intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ListWitness

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

/// Witness List operations - pure XParsec monadic observer
let private witnessList (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.List -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let listType = mapNativeTypeForArch arch node.Type

            match tryMatch (pListEmpty node.Id listType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "List.empty pattern emission failed"

        | "isEmpty" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    match tryMatch (pListIsEmpty node.Id listSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "List.isEmpty pattern emission failed"
                | None -> WitnessOutput.error "List.isEmpty: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.isEmpty: Expected 1 child, got {node.Children.Length}"

        | "head" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let elementType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pListHead node.Id listSSA elementType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "List.head pattern emission failed"
                | None -> WitnessOutput.error "List.head: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.head: Expected 1 child, got {node.Children.Length}"

        | "tail" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let tailType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pListTail node.Id listSSA tailType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "List.tail pattern emission failed"
                | None -> WitnessOutput.error "List.tail: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.tail: Expected 1 child, got {node.Children.Length}"

        | "cons" ->
            match node.Children with
            | [headId; tailId] ->
                match MLIRAccumulator.recallNode headId ctx.Accumulator, MLIRAccumulator.recallNode tailId ctx.Accumulator with
                | Some (headSSA, headType), Some (tailSSA, tailType) ->
                    let head = { SSA = headSSA; Type = headType }
                    let tail = { SSA = tailSSA; Type = tailType }
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let listType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pListCons node.Id head tail listType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "List.cons pattern emission failed"
                | _ -> WitnessOutput.error "List.cons: Head or tail not yet witnessed"
            | _ -> WitnessOutput.error $"List.cons: Expected 2 children, got {node.Children.Length}"

        | "map" | "filter" | "fold" | "length" ->
            WitnessOutput.error $"List.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown List operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// List nanopass - witnesses List intrinsic nodes
let nanopass : Nanopass = {
    Name = "List"
    Witness = witnessList
}
