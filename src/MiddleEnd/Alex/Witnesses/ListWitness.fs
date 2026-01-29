/// ListWitness - Witness List<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY List intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ListWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

let private witnessEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
    | None -> WitnessOutput.error "List.empty: No SSA assigned"
    | Some resultSSA ->
        let arch = ctx.Coeffects.Platform.TargetArch
        let codePtr = resultSSA
        match tryMatch (pFlatClosure codePtr [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, _) ->
            let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
            { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
        | None -> WitnessOutput.error "List.empty: pFlatClosure pattern failed"

let private witnessIsEmpty (_ctx: WitnessContext) (_node: SemanticNode) : WitnessOutput =
    WitnessOutput.error "List.isEmpty: Baker decomposes to structural check"

let private witnessHead (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
    | [listNodeId], Some ssas when ssas.Length >= 2 ->
        match MLIRAccumulator.recallNode listNodeId ctx.Accumulator with
        | Some (listSSA, _) ->
            match tryMatch (pFieldAccess listSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
            | None -> WitnessOutput.error "List.head: pFieldAccess pattern failed"
        | None -> WitnessOutput.error "List.head: List node not yet witnessed"
    | _ -> WitnessOutput.error "List.head: Invalid children or SSAs"

let private witnessTail (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
    | [listNodeId], Some ssas when ssas.Length >= 2 ->
        match MLIRAccumulator.recallNode listNodeId ctx.Accumulator with
        | Some (listSSA, _) ->
            match tryMatch (pFieldAccess listSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
            | None -> WitnessOutput.error "List.tail: pFieldAccess pattern failed"
        | None -> WitnessOutput.error "List.tail: List node not yet witnessed"
    | _ -> WitnessOutput.error "List.tail: Invalid children or SSAs"

let private witnessCons (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
    | [headId; tailId], Some ssas when ssas.Length > 0 ->
        let childVals =
            [headId; tailId]
            |> List.choose (fun childId ->
                MLIRAccumulator.recallNode childId ctx.Accumulator
                |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty }))

        if childVals.Length <> 2 then
            WitnessOutput.error "List.cons: Could not retrieve all child SSAs"
        else
            let arch = ctx.Coeffects.Platform.TargetArch
            let codePtr = ssas.[0]
            match tryMatch (pFlatClosure codePtr childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) ->
                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = mlirType } }
            | None -> WitnessOutput.error "List.cons: pFlatClosure pattern failed"
    | _ -> WitnessOutput.error "List.cons: Invalid children or SSAs"

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

let private witnessList (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.List -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" -> witnessEmpty ctx node
        | "isEmpty" -> witnessIsEmpty ctx node
        | "head" -> witnessHead ctx node
        | "tail" -> witnessTail ctx node
        | "cons" -> witnessCons ctx node
        | "map" | "filter" | "fold" | "length" ->
            WitnessOutput.error $"List.{info.Operation} requires Baker decomposition"
        | op -> WitnessOutput.error $"Unknown List operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

let nanopass : Nanopass = {
    Name = "List"
    Witness = witnessList
}
