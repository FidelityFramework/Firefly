/// ListWitness - Witness List<'T> primitive operations to MLIR
///
/// PRD-13a: Core Collections - List operations
///
/// ARCHITECTURAL PRINCIPLES (January 2026):
/// - Witnesses are THIN compositional layers (~15-25 lines per primitive)
/// - Delegate MLIR emission to shared Patterns (Elements→Patterns→Witnesses)
/// - Observe PSG structure (SSAs pre-assigned by PSGElaboration)
/// - Baker decomposes high-level ops (map, filter, fold) to primitives
/// - Alex witnesses ONLY primitives: empty, head, tail, cons
///
/// NANOPASS: This witness handles ONLY List-related intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses.
module Alex.Witnesses.ListWitness

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

/// Witness List.empty - flat closure with zero captures
/// SSA cost: 1 (Undef)
/// Pattern: pFlatClosure with zero captures
let private witnessEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match getNodeSSA node.Id ctx.Coeffects.SSA with
    | Some resultSSA ->
        let arch = ctx.Coeffects.Platform.TargetArch
        // List.empty = flat closure {code_ptr} with zero captures
        let codePtr = resultSSA  // Placeholder - actual code_ptr from coeffects
        match tryMatch (pFlatClosure codePtr [] [resultSSA]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, _) ->
            let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
            { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = mlirType } }
        | None ->
            WitnessOutput.error "List.empty: pFlatClosure pattern failed"
    | None ->
        WitnessOutput.error "List.empty: No SSA assigned"

/// Witness List.isEmpty - Baker decomposes to structural check
/// This operation is decomposed by Baker into control flow
let private witnessIsEmpty (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    WitnessOutput.error "List.isEmpty: Baker decomposes to structural check - witness the decomposed control flow"

/// Witness List.head - extract head element from cons cell
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 0
let private witnessHead (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "List.head: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [listNodeId] ->
            let listIdVal = NodeId.value listNodeId
            match MLIRAccumulator.recallNode listIdVal ctx.Accumulator with
            | Some (listSSA, _) ->
                // Delegate to pFieldAccess: field index 0 (head field in cons cell)
                match tryMatch (pFieldAccess listSSA 0 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "List.head: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "List.head: List node not yet witnessed"
        | _ ->
            WitnessOutput.error "List.head: Expected 1 child (list node)"

/// Witness List.tail - extract tail pointer from cons cell
/// SSA cost: 2 (GEP + Load)
/// Pattern: pFieldAccess on field 1
let private witnessTail (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.Length < 2 then
        WitnessOutput.error "List.tail: Expected 2 SSAs (GEP + Load)"
    else
        match node.Children with
        | [listNodeId] ->
            let listIdVal = NodeId.value listNodeId
            match MLIRAccumulator.recallNode listIdVal ctx.Accumulator with
            | Some (listSSA, _) ->
                match tryMatch (pFieldAccess listSSA 1 ssas.[0] ssas.[1]) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = ssas.[1]; Type = mlirType } }
                | None ->
                    WitnessOutput.error "List.tail: pFieldAccess pattern failed"
            | None ->
                WitnessOutput.error "List.tail: List node not yet witnessed"
        | _ ->
            WitnessOutput.error "List.tail: Expected 1 child (list node)"

/// Witness List.cons - construct cons cell {code_ptr, head, tail}
/// SSA cost: varies (flat closure construction)
/// Pattern: pFlatClosure with 2 captures (head, tail)
let private witnessCons (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
    if ssas.IsEmpty then
        WitnessOutput.error "List.cons: No SSAs assigned"
    else
        // Children: [headNode, tailNode]
        if node.Children.Length <> 2 then
            WitnessOutput.error $"List.cons: Expected 2 children (head, tail), got {node.Children.Length}"
        else
            // Get SSAs for head and tail
            let childVals =
                node.Children
                |> List.choose (fun childId ->
                    let childIdVal = NodeId.value childId
                    match MLIRAccumulator.recallNode childIdVal ctx.Accumulator with
                    | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                    | None -> None)

            if childVals.Length <> 2 then
                WitnessOutput.error $"List.cons: Could not retrieve all child SSAs (got {childVals.Length}/2)"
            else
                let arch = ctx.Coeffects.Platform.TargetArch
                // List.cons = flat closure {code_ptr, head, tail}
                let codePtr = ssas.[0]  // Placeholder - actual code_ptr from coeffects
                match tryMatch (pFlatClosure codePtr childVals ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) ->
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    let finalSSA = ssas.[ssas.Length - 1]
                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = finalSSA; Type = mlirType } }
                | None ->
                    WitnessOutput.error "List.cons: pFlatClosure pattern failed"

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness List operations - category-selective (handles only List intrinsic nodes)
let private witnessList (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try to match Intrinsic node with List module
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some (info, _) when info.Module = IntrinsicModule.List ->
        // Dispatch to primitive witness based on operation
        match info.Operation with
        | "empty" -> witnessEmpty ctx node
        | "isEmpty" -> witnessIsEmpty ctx node
        | "head" -> witnessHead ctx node
        | "tail" -> witnessTail ctx node
        | "cons" -> witnessCons ctx node
        | _ -> WitnessOutput.skip  // Unknown operation - not a Baker primitive
    | _ -> WitnessOutput.skip  // Not a List intrinsic

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// List nanopass - witnesses List primitive operations
let nanopass : Nanopass = {
    Name = "List"
    Witness = witnessList
}
