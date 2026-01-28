/// MapWitness - Witness Map<'K,'V> primitive operations to MLIR
///
/// PRD-13a: Core Collections - Map operations
///
/// ARCHITECTURAL PRINCIPLES (January 2026):
/// - Witnesses are THIN compositional layers (~15-25 lines per primitive)
/// - Delegate MLIR emission to shared Patterns (Elements→Patterns→Witnesses)
/// - Observe PSG structure (SSAs pre-assigned by PSGElaboration)
/// - Baker decomposes high-level ops (add, tryFind) to primitives
/// - Alex witnesses ONLY primitives: empty, key, value, left, right, height, node
///
/// NANOPASS: This witness handles ONLY Map-related intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses.
module Alex.Witnesses.MapWitness

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
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get all SSAs for a node (for operations with multiple intermediates)
let private getNodeSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> []

/// Get single SSA for a node
let private getNodeSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA option =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some ssa -> Some ssa
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Map.empty - flat closure with zero captures
/// SSA cost: 1 (Undef)
/// Pattern: pFlatClosure with zero captures
let private witnessEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match getNodeSSA node.Id ctx.Coeffects.SSA with
    | Some resultSSA ->
        let arch = ctx.Coeffects.Platform.TargetArch
        // Map.empty = flat closure {code_ptr} with zero captures
        // Delegate to pFlatClosure pattern with empty capture list
        let codePtr = resultSSA  // Placeholder - actual code_ptr from coeffects
        match tryMatch (pFlatClosure codePtr [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, _) ->
            let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
            { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
        | None ->
            WitnessOutput.error "Map.empty: pFlatClosure pattern failed"
    | None ->
        WitnessOutput.error "Map.empty: No SSA assigned"

/// Witness Map.isEmpty - Baker decomposes to structural check
/// This operation is decomposed by Baker into control flow
/// Witness observes the decomposed structure (IfThenElse, etc.)
let private witnessIsEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    WitnessOutput.error "Map.isEmpty: Baker decomposes to structural check - witness the decomposed control flow"

/// Witness Map.key - extract key field from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 0
let private witnessKey (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Map.key: Expected 2 SSAs (GEP + Load)"
    else
        // Get map SSA from child node (the map being accessed)
        match node.Children with
        | [mapNodeId] ->
            let mapIdVal = NodeId.value mapNodeId
            match MLIRAccumulator.recallNode mapIdVal ctx.Accumulator with
            | Some (mapSSA, _) ->
                // Delegate to pFieldAccess: field index 0 (key field in AVL struct)
                match tryMatch (pFieldAccess mapSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.key: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Map.key: Map node not yet witnessed"
        | _ ->
            WitnessOutput.error "Map.key: Expected 1 child (map node)"

/// Witness Map.value - extract value field from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 1
let private witnessValue (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Map.value: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [mapNodeId] ->
            let mapIdVal = NodeId.value mapNodeId
            match MLIRAccumulator.recallNode mapIdVal ctx.Accumulator with
            | Some (mapSSA, _) ->
                match tryMatch (pFieldAccess mapSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.value: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Map.value: Map node not yet witnessed"
        | _ ->
            WitnessOutput.error "Map.value: Expected 1 child (map node)"

/// Witness Map.left - extract left subtree from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 2
let private witnessLeft (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Map.left: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [mapNodeId] ->
            let mapIdVal = NodeId.value mapNodeId
            match MLIRAccumulator.recallNode mapIdVal ctx.Accumulator with
            | Some (mapSSA, _) ->
                match tryMatch (pFieldAccess mapSSA 2 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.left: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Map.left: Map node not yet witnessed"
        | _ ->
            WitnessOutput.error "Map.left: Expected 1 child (map node)"

/// Witness Map.right - extract right subtree from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 3
let private witnessRight (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Map.right: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [mapNodeId] ->
            let mapIdVal = NodeId.value mapNodeId
            match MLIRAccumulator.recallNode mapIdVal ctx.Accumulator with
            | Some (mapSSA, _) ->
                match tryMatch (pFieldAccess mapSSA 3 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.right: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Map.right: Map node not yet witnessed"
        | _ ->
            WitnessOutput.error "Map.right: Expected 1 child (map node)"

/// Witness Map.height - extract height field from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 4
let private witnessHeight (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Map.height: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [mapNodeId] ->
            let mapIdVal = NodeId.value mapNodeId
            match MLIRAccumulator.recallNode mapIdVal ctx.Accumulator with
            | Some (mapSSA, _) ->
                match tryMatch (pFieldAccess mapSSA 4 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.height: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Map.height: Map node not yet witnessed"
        | _ ->
            WitnessOutput.error "Map.height: Expected 1 child (map node)"

/// Witness Map.node - construct AVL node {key, value, left, right, height}
/// SSA cost: 6 (Undef + 5 InsertValues)
/// Pattern: pRecordStruct with 5 fields
let private witnessNode (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 6 then
        WitnessOutput.error $"Map.node: Expected 6 SSAs (Undef + 5 InsertValues), got {ssas.Length}"
    else
        // Children: [keyNode, valueNode, leftNode, rightNode, heightNode]
        if node.Children.Length <> 5 then
            WitnessOutput.error $"Map.node: Expected 5 children, got {node.Children.Length}"
        else
            // Get SSAs for all child nodes (key, value, left, right, height)
            let childVals =
                node.Children
                |> List.choose (fun childId ->
                    let childIdVal = NodeId.value childId
                    match MLIRAccumulator.recallNode childIdVal ctx.Accumulator with
                    | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                    | None -> None)

            if childVals.Length <> 5 then
                WitnessOutput.error $"Map.node: Could not retrieve all child SSAs (got {childVals.Length}/5)"
            else
                // Delegate to pRecordStruct: builds struct via Undef + InsertValue chain
                match tryMatch (pRecordStruct childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[5]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Map.node: pRecordStruct pattern failed"

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Map operations - category-selective (handles only Map intrinsic nodes)
let private witnessMap (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try to match Intrinsic node with Map module
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (info, _) when info.Module = IntrinsicModule.Map ->
        // Dispatch to primitive witness based on operation
        match info.Operation with
        | "empty" -> witnessEmpty ctx node
        | "isEmpty" -> witnessIsEmpty ctx node
        | "key" -> witnessKey ctx node
        | "value" -> witnessValue ctx node
        | "left" -> witnessLeft ctx node
        | "right" -> witnessRight ctx node
        | "height" -> witnessHeight ctx node
        | "node" -> witnessNode ctx node
        | _ -> WitnessOutput.skip  // Unknown operation - not a Baker primitive
    | _ -> WitnessOutput.skip  // Not a Map intrinsic

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Map nanopass - witnesses Map primitive operations
let nanopass : Nanopass = {
    Name = "Map"
    Witness = witnessMap
}
