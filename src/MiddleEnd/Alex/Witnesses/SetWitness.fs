/// SetWitness - Witness Set<'T> primitive operations to MLIR
///
/// PRD-13a: Core Collections - Set operations
///
/// ARCHITECTURAL PRINCIPLES (January 2026):
/// - Witnesses are THIN compositional layers (~15-25 lines per primitive)
/// - Delegate MLIR emission to shared Patterns (Elements→Patterns→Witnesses)
/// - Observe PSG structure (SSAs pre-assigned by PSGElaboration)
/// - Baker decomposes high-level ops (union, intersect) to primitives
/// - Alex witnesses ONLY primitives: empty, value, left, right, height, node
///
/// NANOPASS: This witness handles ONLY Set-related intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses.
module Alex.Witnesses.SetWitness

open XParsec
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Set operations - category-selective (handles only Set intrinsic nodes)
let private witnessSet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Set -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "Set.empty: No SSA assigned"
            | Some resultSSA ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let codePtrTy = TPtr
                match tryMatch (pFlatClosure resultSSA codePtrTy [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
                | None -> WitnessOutput.error "Set.empty: pFlatClosure pattern failed"

        | "isEmpty" ->
            WitnessOutput.error "Set.isEmpty requires Baker decomposition"

        | "value" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [setNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode setNodeId ctx.Accumulator with
                | Some (setSSA, _) ->
                    match tryMatch (pFieldAccess setSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Set.value: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Set.value: Set node not yet witnessed"
            | _ -> WitnessOutput.error "Set.value: Invalid children or SSAs"

        | "left" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [setNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode setNodeId ctx.Accumulator with
                | Some (setSSA, _) ->
                    match tryMatch (pFieldAccess setSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Set.left: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Set.left: Set node not yet witnessed"
            | _ -> WitnessOutput.error "Set.left: Invalid children or SSAs"

        | "right" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [setNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode setNodeId ctx.Accumulator with
                | Some (setSSA, _) ->
                    match tryMatch (pFieldAccess setSSA 2 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Set.right: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Set.right: Set node not yet witnessed"
            | _ -> WitnessOutput.error "Set.right: Invalid children or SSAs"

        | "height" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | [setNodeId], Some ssas when ssas.Length >= 2 ->
                match MLIRAccumulator.recallNode setNodeId ctx.Accumulator with
                | Some (setSSA, _) ->
                    match tryMatch (pFieldAccess setSSA 3 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                    | None -> WitnessOutput.error "Set.height: pFieldAccess pattern failed"
                | None -> WitnessOutput.error "Set.height: Set node not yet witnessed"
            | _ -> WitnessOutput.error "Set.height: Invalid children or SSAs"

        | "node" ->
            match node.Children, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | children, Some ssas when children.Length = 4 && ssas.Length >= 5 ->
                let childVals =
                    children
                    |> List.choose (fun childId ->
                        MLIRAccumulator.recallNode childId ctx.Accumulator
                        |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty }))
                if childVals.Length <> 4 then
                    WitnessOutput.error $"Set.node: Could not retrieve all child SSAs (got {childVals.Length}/4)"
                else
                    match tryMatch (pRecordStruct childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[4]; Type = mlirType } }
                    | None -> WitnessOutput.error "Set.node: pRecordStruct pattern failed"
            | _ -> WitnessOutput.error "Set.node: Invalid children or SSAs"

        | op -> WitnessOutput.error $"Unknown Set operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Set nanopass - witnesses Set primitive operations
let nanopass : Nanopass = {
    Name = "Set"
    Witness = witnessSet
}
