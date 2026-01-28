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
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get all SSAs for a node
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

/// Witness Set.empty - flat closure with zero captures
/// SSA cost: 1 (Undef)
/// Pattern: pFlatClosure with zero captures
let private witnessEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match getNodeSSA node.Id ctx.Coeffects.SSA with
    | Some resultSSA ->
        let arch = ctx.Coeffects.Platform.TargetArch
        // Set.empty = flat closure {code_ptr} with zero captures
        let codePtr = resultSSA  // Placeholder - actual code_ptr from coeffects
        match tryMatch (pFlatClosure codePtr [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, _) ->
            let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
            { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
        | None ->
            WitnessOutput.error "Set.empty: pFlatClosure pattern failed"
    | None ->
        WitnessOutput.error "Set.empty: No SSA assigned"

/// Witness Set.isEmpty - Baker decomposes to structural check
/// This operation is decomposed by Baker into control flow
let private witnessIsEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    WitnessOutput.error "Set.isEmpty: Baker decomposes to structural check - witness the decomposed control flow"

/// Witness Set.value - extract value field from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 0
let private witnessValue (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Set.value: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [setNodeId] ->
            let setIdVal = NodeId.value setNodeId
            match MLIRAccumulator.recallNode setIdVal ctx.Accumulator with
            | Some (setSSA, _) ->
                // Delegate to pFieldAccess: field index 0 (value field in AVL struct)
                match tryMatch (pFieldAccess setSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Set.value: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Set.value: Set node not yet witnessed"
        | _ ->
            WitnessOutput.error "Set.value: Expected 1 child (set node)"

/// Witness Set.left - extract left subtree from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 1
let private witnessLeft (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Set.left: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [setNodeId] ->
            let setIdVal = NodeId.value setNodeId
            match MLIRAccumulator.recallNode setIdVal ctx.Accumulator with
            | Some (setSSA, _) ->
                match tryMatch (pFieldAccess setSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Set.left: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Set.left: Set node not yet witnessed"
        | _ ->
            WitnessOutput.error "Set.left: Expected 1 child (set node)"

/// Witness Set.right - extract right subtree from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 2
let private witnessRight (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Set.right: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [setNodeId] ->
            let setIdVal = NodeId.value setNodeId
            match MLIRAccumulator.recallNode setIdVal ctx.Accumulator with
            | Some (setSSA, _) ->
                match tryMatch (pFieldAccess setSSA 2 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Set.right: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Set.right: Set node not yet witnessed"
        | _ ->
            WitnessOutput.error "Set.right: Expected 1 child (set node)"

/// Witness Set.height - extract height field from AVL node
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 3
let private witnessHeight (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "Set.height: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [setNodeId] ->
            let setIdVal = NodeId.value setNodeId
            match MLIRAccumulator.recallNode setIdVal ctx.Accumulator with
            | Some (setSSA, _) ->
                match tryMatch (pFieldAccess setSSA 3 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Set.height: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "Set.height: Set node not yet witnessed"
        | _ ->
            WitnessOutput.error "Set.height: Expected 1 child (set node)"

/// Witness Set.node - construct AVL node {value, left, right, height}
/// SSA cost: 5 (Undef + 4 InsertValues)
/// Pattern: pRecordStruct with 4 fields
let private witnessNode (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 5 then
        WitnessOutput.error $"Set.node: Expected 5 SSAs (Undef + 4 InsertValues), got {ssas.Length}"
    else
        // Children: [valueNode, leftNode, rightNode, heightNode]
        if node.Children.Length <> 4 then
            WitnessOutput.error $"Set.node: Expected 4 children, got {node.Children.Length}"
        else
            // Get SSAs for all child nodes (value, left, right, height)
            let childVals =
                node.Children
                |> List.choose (fun childId ->
                    let childIdVal = NodeId.value childId
                    match MLIRAccumulator.recallNode childIdVal ctx.Accumulator with
                    | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                    | None -> None)

            if childVals.Length <> 4 then
                WitnessOutput.error $"Set.node: Could not retrieve all child SSAs (got {childVals.Length}/4)"
            else
                // Delegate to pRecordStruct: builds struct via Undef + InsertValue chain
                match tryMatch (pRecordStruct childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[4]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "Set.node: pRecordStruct pattern failed"

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Set operations - category-selective (handles only Set intrinsic nodes)
let private witnessSet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try to match Intrinsic node with Set module
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (info, _) when info.Module = IntrinsicModule.Set ->
        // Dispatch to primitive witness based on operation
        match info.Operation with
        | "empty" -> witnessEmpty ctx node
        | "isEmpty" -> witnessIsEmpty ctx node
        | "value" -> witnessValue ctx node
        | "left" -> witnessLeft ctx node
        | "right" -> witnessRight ctx node
        | "height" -> witnessHeight ctx node
        | "node" -> witnessNode ctx node
        | _ -> WitnessOutput.skip  // Unknown operation - not a Baker primitive
    | _ -> WitnessOutput.skip  // Not a Set intrinsic

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Set nanopass - witnesses Set primitive operations
let nanopass : Nanopass = {
    Name = "Set"
    Witness = witnessSet
}
