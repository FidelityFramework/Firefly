/// MapWitness - Witness Map<'K,'V> primitive operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Map intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses.
module Alex.Witnesses.MapWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Map operations - category-selective (handles only Map intrinsic nodes)
let private witnessMap (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Map -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "Map.empty: No SSA assigned"
            | Some resultSSA ->
                let arch = ctx.Coeffects.Platform.TargetArch
                match tryMatch (pFlatClosure resultSSA [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
                | None -> WitnessOutput.error "Map.empty: pFlatClosure pattern failed"

        | "isEmpty" ->
            WitnessOutput.error "Map.isEmpty: Baker decomposes to structural check"

        | "key" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [mapNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode mapNodeId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    match tryMatch (pFieldAccess mapSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Map.key: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Map.key: Map node not yet witnessed"
            | _ -> WitnessOutput.error "Map.key: Invalid children or SSAs"

        | "value" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [mapNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode mapNodeId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    match tryMatch (pFieldAccess mapSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Map.value: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Map.value: Map node not yet witnessed"
            | _ -> WitnessOutput.error "Map.value: Invalid children or SSAs"

        | "left" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [mapNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode mapNodeId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    match tryMatch (pFieldAccess mapSSA 2 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Map.left: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Map.left: Map node not yet witnessed"
            | _ -> WitnessOutput.error "Map.left: Invalid children or SSAs"

        | "right" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [mapNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode mapNodeId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    match tryMatch (pFieldAccess mapSSA 3 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Map.right: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Map.right: Map node not yet witnessed"
            | _ -> WitnessOutput.error "Map.right: Invalid children or SSAs"

        | "height" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [mapNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode mapNodeId ctx.Accumulator with
                | Some (mapSSA, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    match tryMatch (pFieldAccess mapSSA 4 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Map.height: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Map.height: Map node not yet witnessed"
            | _ -> WitnessOutput.error "Map.height: Invalid children or SSAs"

        | "node" ->
            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "Map.node: No SSAs assigned"
            | Some ssas when ssas.Length < 6 -> WitnessOutput.error $"Map.node: Expected 6 SSAs, got {ssas.Length}"
            | Some ssas ->
                if node.Children.Length <> 5 then
                    WitnessOutput.error $"Map.node: Expected 5 children, got {node.Children.Length}"
                else
                    let childVals =
                        node.Children
                        |> List.choose (fun childId ->
                            match MLIRAccumulator.recallNode childId ctx.Accumulator with
                            | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                            | None -> None)
                    if childVals.Length <> 5 then
                        WitnessOutput.error $"Map.node: Could not retrieve all child SSAs (got {childVals.Length}/5)"
                    else
                        let arch = ctx.Coeffects.Platform.TargetArch
                        match tryMatch (pRecordStruct childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some (ops, _) ->
                            let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                            { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[5]; Type = mlirType } }
                        | None -> WitnessOutput.error "Map.node: pRecordStruct pattern failed"

        | "add" | "tryFind" | "remove" | "containsKey" ->
            WitnessOutput.error $"Map.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown Map operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Map nanopass - witnesses Map primitive operations
let nanopass : Nanopass = {
    Name = "Map"
    Witness = witnessMap
}
