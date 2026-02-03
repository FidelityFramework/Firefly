/// OptionWitness - Witness Option<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
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

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Option operations - category-selective (handles only Option intrinsic nodes)
let private witnessOption (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Option -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "Some" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [childId], Some ssas ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (valSSA, valType) ->
                    let value = { SSA = valSSA; Type = valType }
                    let optionTy = TStruct [TInt I8; valType]
                    match tryMatch (pOptionSome value ssas optionTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = optionTy } }
                    | None -> WitnessOutput.error "Option.Some pattern emission failed"
                | None -> WitnessOutput.error "Option.Some: Value not yet witnessed"
            | _ -> WitnessOutput.error "Option.Some: Invalid children or SSAs"

        | "None" ->
            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "Option.None: No SSAs assigned"
            | Some ssas ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let optionType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                match tryMatch (pOptionNone ssas optionType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some (ops, _) ->
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = optionType } }
                | None -> WitnessOutput.error "Option.None pattern emission failed"

        | "isSome" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [childId], Some ssas when ssas.Length >= 3 ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatch (pOptionIsSome optSSA ssas.[0] ssas.[1] ssas.[2]) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[2]; Type = TInt I1 } }
                    | None -> WitnessOutput.error "Option.isSome pattern emission failed"
                | None -> WitnessOutput.error "Option.isSome: Option not yet witnessed"
            | _ -> WitnessOutput.error "Option.isSome: Invalid children or SSAs"

        | "isNone" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [childId], Some ssas when ssas.Length >= 3 ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatch (pOptionIsNone optSSA ssas.[0] ssas.[1] ssas.[2]) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[2]; Type = TInt I1 } }
                    | None -> WitnessOutput.error "Option.isNone pattern emission failed"
                | None -> WitnessOutput.error "Option.isNone: Option not yet witnessed"
            | _ -> WitnessOutput.error "Option.isNone: Invalid children or SSAs"

        | "get" ->
            match node.Children, SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | [childId], Some resultSSA ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, optType) ->
                    let valueType = match optType with
                                    | TStruct [_; vt] -> vt
                                    | _ -> TPtr
                    match tryMatch (pOptionGet optSSA resultSSA valueType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) ->
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = valueType } }
                    | None -> WitnessOutput.error "Option.get pattern emission failed"
                | None -> WitnessOutput.error "Option.get: Option not yet witnessed"
            | _ -> WitnessOutput.error "Option.get: Invalid children or SSAs"

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
