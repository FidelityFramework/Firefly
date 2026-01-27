/// CoeffectLookup - Helper functions for coefficient access
///
/// VISIBILITY: module internal - Witnesses CANNOT import this
///
/// Coeffects are pre-computed data (SSA assignments, mutability analysis,
/// platform resolution, etc.) that witnesses READ but never modify.
/// These helpers provide safe access patterns for witnesses via Patterns.
module internal Alex.Elements.CoeffectLookup

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// SSA LOOKUP
// ═══════════════════════════════════════════════════════════════════════════

/// Get single SSA for a node (fails if not found)
let requireSSA (nodeId: NodeId) (coeffects: TransferCoeffects) : SSA =
    match SSAAssign.lookupSSA nodeId coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get list of SSAs for a node (fails if not found)
let requireSSAs (nodeId: NodeId) (coeffects: TransferCoeffects) : SSA list =
    match SSAAssign.lookupSSAs nodeId coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Lookup SSA (returns option)
let tryLookupSSA (nodeId: NodeId) (coeffects: TransferCoeffects) : SSA option =
    SSAAssign.lookupSSA nodeId coeffects.SSA

/// Lookup SSAs (returns option)
let tryLookupSSAs (nodeId: NodeId) (coeffects: TransferCoeffects) : SSA list option =
    SSAAssign.lookupSSAs nodeId coeffects.SSA
