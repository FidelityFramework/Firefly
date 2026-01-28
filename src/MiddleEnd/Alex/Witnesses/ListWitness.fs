/// ListWitness - Witness List operations to MLIR
///
/// PRD-13a: Core Collections - List<'T>
///
/// ARCHITECTURAL PRINCIPLES:
/// - List is a singly-linked cons cell: {head: T, tail: ptr<List<T>>}
/// - Empty list = null pointer
/// - Structural sharing: tail pointer reuse
/// - All allocations from arena (or global arena pre-PRD-20)
///
/// PRIMITIVE OPERATIONS (Alex witnesses directly):
/// - empty: flat closure with zero captures { code_ptr }
/// - isEmpty: Baker decomposes to structural check
/// - head: GEP to field 0, load
/// - tail: GEP to field 1, load
/// - cons: arena alloc + store head + store tail
///
/// DECOMPOSABLE OPERATIONS (FNCS saturation):
/// - map, filter, fold, length, rev, append - decompose to recursion over primitives
module Alex.Witnesses.ListWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build the cons cell struct type for a list element type
/// Layout: {head: ElementType, tail: ptr}
let listCellType (elementType: MLIRType) : MLIRType =
    TStruct [elementType; TPtr]

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSA for a node
let private requireSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "ListWitness: No SSA for node %A" nodeId

/// Get pre-assigned SSAs for a node
let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "ListWitness: No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness List.empty<'T> - returns flat closure with zero captures
/// SSA cost: 1 (Undef)
let witnessEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    : MLIROp list * TransferResult =

    let resultSSA = requireSSA nodeId ssa
    // Flat closure: { code_ptr }
    let closureType = TStruct [TPtr]
    let op = MLIROp.LLVMOp (LLVMOp.Undef (resultSSA, closureType))
    [op], TRValue { SSA = resultSSA; Type = closureType }

/// Witness List.isEmpty - Baker decomposes to structural check
/// SSA cost: Depends on Baker's decomposition
let witnessIsEmpty
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (listVal: Val)
    : MLIROp list * TransferResult =

    // isEmpty is decomposed by Baker into structural checks
    // Witness what Baker produces, not the high-level operation
    [], TRError "List.isEmpty requires Baker decomposition to structural check"

/// Witness List.head - extract first element (assumes non-empty)
/// SSA cost: 2 (GEP + load)
let witnessHead
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (listVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let gepSSA = ssas.[0]
    let resultSSA = ssas.[1]
    let cellType = listCellType elementType
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (gepSSA, listVal.SSA, 0, cellType))
        MLIROp.LLVMOp (LLVMOp.Load (resultSSA, gepSSA, elementType, AtomicOrdering.NotAtomic))
    ]
    ops, TRValue { SSA = resultSSA; Type = elementType }

/// Witness List.tail - extract rest of list (assumes non-empty)
/// SSA cost: 2 (GEP + load)
let witnessTail
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (listVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let gepSSA = ssas.[0]
    let resultSSA = ssas.[1]
    let cellType = listCellType elementType
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.StructGEP (gepSSA, listVal.SSA, 1, cellType))
        MLIROp.LLVMOp (LLVMOp.Load (resultSSA, gepSSA, TPtr, AtomicOrdering.NotAtomic))
    ]
    ops, TRValue { SSA = resultSSA; Type = TPtr }

/// Witness List.cons - prepend element to list
/// SSA cost: 4 (alloca/arena_alloc + GEP head + GEP tail)
/// Note: Uses stack allocation for now; will use arena when PRD-20 lands
let witnessCons
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (headVal: Val)
    (tailVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let oneSSA = ssas.[0]
    let cellPtrSSA = ssas.[1]
    let headGepSSA = ssas.[2]
    let tailGepSSA = ssas.[3]
    let cellType = listCellType elementType
    
    let ops = [
        // Allocate cons cell (stack for now, arena later)
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (cellPtrSSA, oneSSA, cellType, None))
        // Store head
        MLIROp.LLVMOp (LLVMOp.StructGEP (headGepSSA, cellPtrSSA, 0, cellType))
        MLIROp.LLVMOp (LLVMOp.Store (headVal.SSA, headGepSSA, elementType, AtomicOrdering.NotAtomic))
        // Store tail
        MLIROp.LLVMOp (LLVMOp.StructGEP (tailGepSSA, cellPtrSSA, 1, cellType))
        MLIROp.LLVMOp (LLVMOp.Store (tailVal.SSA, tailGepSSA, TPtr, AtomicOrdering.NotAtomic))
    ]
    ops, TRValue { SSA = cellPtrSSA; Type = TPtr }

/// Witness List.length - count elements (iterative)
/// This is a primitive that generates an inline loop
/// SSA cost: ~15 (loop with counter)
let witnessLength
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (listVal: Val)
    (elementType: MLIRType)
    : MLIROp list * TransferResult =
    
    // For now, return placeholder - full implementation requires SCF.While
    // The decomposition approach (recursive via fold) is preferred
    [], TRError "List.length requires SCF.While - use functional decomposition"

// ═══════════════════════════════════════════════════════════════════════════
// SSA COST FUNCTIONS (for SSAAssignment nanopass)
// ═══════════════════════════════════════════════════════════════════════════

/// SSA cost for List.empty
let emptySSACost : int = 1

/// SSA cost for List.isEmpty
let isEmptySSACost : int = 2

/// SSA cost for List.head
let headSSACost : int = 2

/// SSA cost for List.tail
let tailSSACost : int = 2

/// SSA cost for List.cons
let consSSACost : int = 4  // oneSSA, cellPtrSSA, headGepSSA, tailGepSSA
