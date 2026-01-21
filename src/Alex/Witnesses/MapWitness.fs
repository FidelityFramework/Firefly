/// MapWitness - Witness Map operations to MLIR
///
/// PRD-13a: Core Collections - Map<'K, 'V>
///
/// ARCHITECTURAL PRINCIPLES:
/// - Map is an AVL tree: {key: K, value: V, left: ptr, right: ptr, height: i32}
/// - Empty map = null pointer
/// - Balanced tree ensures O(log n) operations
/// - Structural sharing: subtrees reused on insert/remove
///
/// PRIMITIVE OPERATIONS (Alex witnesses directly):
/// - empty: returns null pointer
/// - isEmpty: null check
///
/// DECOMPOSABLE OPERATIONS (FNCS saturation):
/// - add, remove, tryFind, find, containsKey - tree traversal with comparison
/// - keys, values, toList, fold - in-order traversal
module Alex.Witnesses.MapWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build the AVL node struct type for a map
/// Layout: {key: K, value: V, left: ptr, right: ptr, height: i32}
let mapNodeType (keyType: MLIRType) (valueType: MLIRType) : MLIRType =
    TStruct [keyType; valueType; TPtr; TPtr; TInt I32]

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSA for a node
let private requireSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "MapWitness: No SSA for node %A" nodeId

/// Get pre-assigned SSAs for a node
let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "MapWitness: No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Map.empty<'K, 'V> - returns null pointer
/// SSA cost: 1
let witnessEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    : MLIROp list * TransferResult =
    
    let resultSSA = requireSSA nodeId ssa
    let op = MLIROp.LLVMOp (LLVMOp.NullPtr resultSSA)
    [op], TRValue { SSA = resultSSA; Type = TPtr }

/// Witness Map.isEmpty - null pointer check
/// SSA cost: 2 (null constant + icmp)
let witnessIsEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapVal: Val)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let nullSSA = ssas.[0]
    let resultSSA = ssas.[1]
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.NullPtr nullSSA)
        MLIROp.LLVMOp (LLVMOp.ICmp (resultSSA, ICmpPred.Eq, mapVal.SSA, nullSSA))
    ]
    ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 }

/// Witness Map.count - return number of elements
/// For empty map, returns 0. For non-empty, would need tree traversal.
/// Placeholder - full implementation requires tree traversal
let witnessCount
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapVal: Val)
    : MLIROp list * TransferResult =
    
    // Placeholder - decompose to fold-based counting in FNCS
    [], TRError "Map.count requires tree traversal - use functional decomposition"

/// Witness Map.tryFind - lookup key in map, return Option<V>
/// This is a complex operation requiring tree traversal and Option construction
/// Placeholder for now - full implementation requires control flow
let witnessTryFind
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (keyVal: Val)
    (mapVal: Val)
    (keyType: MLIRType)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    // For a cold implementation, we return error - complex tree traversal needed
    [], TRError "Map.tryFind requires tree traversal with comparison - complex implementation pending"

/// Witness Map.add - insert/update key-value pair
/// Returns a new map (structural sharing)
/// Placeholder - requires tree traversal and rebalancing
let witnessAdd
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (keyVal: Val)
    (valueVal: Val)
    (mapVal: Val)
    (keyType: MLIRType)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Map.add requires tree traversal with rebalancing - complex implementation pending"

/// Witness Map.containsKey - check if key exists
/// Wrapper around tryFind that returns bool
let witnessContainsKey
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (keyVal: Val)
    (mapVal: Val)
    (keyType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Map.containsKey requires tree traversal - complex implementation pending"

/// Witness Map.values - extract all values as List<V>
/// Requires in-order traversal
let witnessValues
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Map.values requires in-order traversal - complex implementation pending"

/// Witness Map.keys - extract all keys as List<K>
/// Requires in-order traversal
let witnessKeys
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapVal: Val)
    (keyType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Map.keys requires in-order traversal - complex implementation pending"

// ═══════════════════════════════════════════════════════════════════════════
// SSA COST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// SSA cost for Map.empty
let emptySSACost : int = 1

/// SSA cost for Map.isEmpty
let isEmptySSACost : int = 2

/// SSA cost for Map.tryFind (placeholder - complex)
let tryFindSSACost : int = 20

/// SSA cost for Map.add (placeholder - complex)
let addSSACost : int = 25

/// SSA cost for Map.containsKey
let containsKeySSACost : int = 15

/// SSA cost for Map.values/keys
let valuesSSACost : int = 30
let keysSSACost : int = 30
