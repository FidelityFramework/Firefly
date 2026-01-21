/// SetWitness - Witness Set operations to MLIR
///
/// PRD-13a: Core Collections - Set<'T>
///
/// ARCHITECTURAL PRINCIPLES:
/// - Set is an AVL tree (like Map, without values): {value: T, left: ptr, right: ptr, height: i32}
/// - Empty set = null pointer
/// - Balanced tree ensures O(log n) operations
/// - Structural sharing: subtrees reused on insert/remove
///
/// PRIMITIVE OPERATIONS (Alex witnesses directly):
/// - empty: returns null pointer
/// - isEmpty: null check
///
/// DECOMPOSABLE OPERATIONS (FNCS saturation):
/// - add, remove, contains - tree traversal with comparison
/// - union, intersect, difference - tree merge operations
/// - toList, fold - in-order traversal
module Alex.Witnesses.SetWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build the AVL node struct type for a set
/// Layout: {value: T, left: ptr, right: ptr, height: i32}
let setNodeType (elementType: MLIRType) : MLIRType =
    TStruct [elementType; TPtr; TPtr; TInt I32]

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSA for a node
let private requireSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "SetWitness: No SSA for node %A" nodeId

/// Get pre-assigned SSAs for a node
let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "SetWitness: No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Set.empty<'T> - returns null pointer
/// SSA cost: 1
let witnessEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    : MLIROp list * TransferResult =
    
    let resultSSA = requireSSA nodeId ssa
    let op = MLIROp.LLVMOp (LLVMOp.NullPtr resultSSA)
    [op], TRValue { SSA = resultSSA; Type = TPtr }

/// Witness Set.isEmpty - null pointer check
/// SSA cost: 2 (null constant + icmp)
let witnessIsEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (setVal: Val)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let nullSSA = ssas.[0]
    let resultSSA = ssas.[1]
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.NullPtr nullSSA)
        MLIROp.LLVMOp (LLVMOp.ICmp (resultSSA, ICmpPred.Eq, setVal.SSA, nullSSA))
    ]
    ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 }

/// Witness Set.contains - check if element exists
/// Requires tree traversal with comparison
let witnessContains
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (valueVal: Val)
    (setVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.contains requires tree traversal - complex implementation pending"

/// Witness Set.add - insert element into set
/// Returns a new set (structural sharing)
let witnessAdd
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (valueVal: Val)
    (setVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.add requires tree traversal with rebalancing - complex implementation pending"

/// Witness Set.remove - remove element from set
/// Returns a new set (structural sharing)
let witnessRemove
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (valueVal: Val)
    (setVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.remove requires tree traversal with rebalancing - complex implementation pending"

/// Witness Set.count - return number of elements
let witnessCount
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (setVal: Val)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.count requires tree traversal - use functional decomposition"

/// Witness Set.union - combine two sets
let witnessUnion
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (set1Val: Val)
    (set2Val: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.union requires tree merge - complex implementation pending"

/// Witness Set.intersect - intersection of two sets
let witnessIntersect
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (set1Val: Val)
    (set2Val: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.intersect requires tree traversal - complex implementation pending"

/// Witness Set.difference - elements in set1 but not set2
let witnessDifference
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (set1Val: Val)
    (set2Val: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Set.difference requires tree traversal - complex implementation pending"

// ═══════════════════════════════════════════════════════════════════════════
// SSA COST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// SSA cost for Set.empty
let emptySSACost : int = 1

/// SSA cost for Set.isEmpty
let isEmptySSACost : int = 2

/// SSA cost for Set.contains (placeholder)
let containsSSACost : int = 15

/// SSA cost for Set.add (placeholder)
let addSSACost : int = 20

/// SSA cost for Set.remove (placeholder)
let removeSSACost : int = 20

/// SSA cost for Set.union (placeholder)
let unionSSACost : int = 30

/// SSA cost for Set.intersect (placeholder)
let intersectSSACost : int = 30

/// SSA cost for Set.difference (placeholder)
let differenceSSACost : int = 30
