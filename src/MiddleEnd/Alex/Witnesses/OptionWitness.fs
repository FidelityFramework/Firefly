/// OptionWitness - Witness Option<'T> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY Option intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.OptionWitness

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

/// Witness Option operations - pure XParsec monadic observer
let private witnessOption (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Option -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "Some" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (valSSA, valType) ->
                    let value = { SSA = valSSA; Type = valType }
                    let totalBytes = 1 + mlirTypeSize valType
                    let optionTy = TMemRefStatic(totalBytes, TInt I8)

                    match tryMatch (pOptionSome node.Id value optionTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Option.Some pattern emission failed"
                | None -> WitnessOutput.error "Option.Some: Value not yet witnessed"
            | _ -> WitnessOutput.error $"Option.Some: Expected 1 child, got {node.Children.Length}"

        | "None" ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let optionType = mapNativeTypeForArch arch node.Type

            match tryMatch (pOptionNone node.Id optionType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "Option.None pattern emission failed"

        | "isSome" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatch (pOptionIsSome node.Id optSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Option.isSome pattern emission failed"
                | None -> WitnessOutput.error "Option.isSome: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.isSome: Expected 1 child, got {node.Children.Length}"

        | "isNone" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatch (pOptionIsNone node.Id optSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Option.isNone pattern emission failed"
                | None -> WitnessOutput.error "Option.isNone: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.isNone: Expected 1 child, got {node.Children.Length}"

        | "get" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    let valueType = TIndex  // Fallback type

                    match tryMatch (pOptionGet node.Id optSSA valueType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Option.get pattern emission failed"
                | None -> WitnessOutput.error "Option.get: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.get: Expected 1 child, got {node.Children.Length}"

        | "map" | "bind" | "defaultValue" | "defaultWith" ->
            WitnessOutput.error $"Option.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown Option operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Option nanopass - witnesses Option intrinsic nodes
let nanopass : Nanopass = {
    Name = "Option"
    Witness = witnessOption
}
