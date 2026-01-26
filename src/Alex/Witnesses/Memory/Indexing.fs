/// Memory Indexing Witness - Array/collection indexing and address-of
///
/// SCOPE: IndexGet, IndexSet, AddressOf operations
/// DOES NOT: Field access, aggregates, DUs (separate witnesses)
module Alex.Witnesses.Memory.Indexing

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module SSAAssignment = PSGElaboration.SSAAssignment

/// Get all pre-assigned SSAs for a node
let requireNodeSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssignment.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get result SSA for a node
let requireNodeSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssignment.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No result SSA for node %A" nodeId

/// Check if a type is the native string type (fat pointer)
let isNativeStrType (ty: MLIRType) : bool =
    match ty with
    | TStruct [TPtr; TInt _] -> true
    | _ -> false

let witnessIndexGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (collSSA: SSA)
    (collType: MLIRType)
    (indexSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

    // GEP to compute element address
    let ptrSSA = ssas.[0]
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, collSSA, [(indexSSA, MLIRTypes.i64)], elemType))

    // Load the element
    let loadSSA = ssas.[1]
    let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, ptrSSA, elemType, NotAtomic))

    [gepOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }

/// Witness array/collection index set
/// Generates GEP + Store sequence
/// Uses 1 pre-assigned SSA: gep[0]
let witnessIndexSet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (collSSA: SSA)
    (indexSSA: SSA)
    (valueSSA: SSA)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

    // GEP to compute element address
    let ptrSSA = ssas.[0]
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, collSSA, [(indexSSA, MLIRTypes.i64)], elemType))

    // Store the value
    let storeOp = MLIROp.LLVMOp (LLVMOp.Store (valueSSA, ptrSSA, elemType, NotAtomic))

    [gepOp; storeOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// ADDRESS-OF OPERATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Witness address-of operator (&expr)
/// For mutable bindings, the SSA already represents the alloca address
/// For immutable values, we need to allocate and store
/// Uses up to 2 pre-assigned SSAs: const[0], alloca[1]
let witnessAddressOf
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (exprSSA: SSA)
    (exprType: MLIRType)
    (isMutable: bool)
    : MLIROp list * TransferResult =

    if isMutable then
        // Mutable binding already has an address (alloca)
        // The SSA is the pointer to the mutable slot
        [], TRValue { SSA = exprSSA; Type = TPtr }
    else
        let ssas = requireNodeSSAs nodeId ctx
        // Need to allocate stack space and store the value
        let oneSSA = ssas.[0]
        let allocaSSA = ssas.[1]

        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, oneSSA, exprType, None))
            MLIROp.LLVMOp (LLVMOp.Store (exprSSA, allocaSSA, exprType, NotAtomic))
        ]

        ops, TRValue { SSA = allocaSSA; Type = TPtr }

// ═══════════════════════════════════════════════════════════════════════════
// FIELD ACCESS
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve field name to struct index based on type
/// For records, uses SemanticGraph.tryGetRecordFields to look up field order
