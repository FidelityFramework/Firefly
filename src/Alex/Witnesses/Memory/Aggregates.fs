/// Memory Aggregates Witness - Tuple, Record, Array, List construction
///
/// SCOPE: witnessTupleExpr, witnessRecordExpr, witnessRecordCopyUpdate, witnessArrayExpr, witnessListExpr
/// DOES NOT: Indexing, fields, DUs (separate witnesses)
module Alex.Witnesses.Memory.Aggregates

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Witnesses.Memory.Indexing

let witnessTupleExpr
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (elements: Val list)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
    let elemTypes = elements |> List.map (fun v -> v.Type)
    let tupleType = TStruct elemTypes

    // Start with undef
    let undefSSA = ssas.[0]
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, tupleType))

    // Insert each element
    let insertOps, finalSSA =
        elements
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, elem) ->
            let newSSA = ssas.[idx + 1]  // SSAs 1..N are for inserts
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, elem.SSA, [idx], tupleType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = tupleType }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness record construction
/// Records are represented as structs with fields in declaration order
/// Uses N+1 pre-assigned SSAs: undef[0], insert[1..N]
let witnessRecordExpr
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (fields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

    // Start with undef
    let undefSSA = ssas.[0]
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, recordType))

    // Insert each field
    let insertOps, finalSSA =
        fields
        |> List.indexed
        |> List.fold (fun (ops, prevSSA) (idx, (_, fieldVal)) ->
            let newSSA = ssas.[idx + 1]  // SSAs 1..N are for inserts
            let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, fieldVal.SSA, [idx], recordType))
            (ops @ [insertOp], newSSA)
        ) ([], undefSSA)

    [undefOp] @ insertOps, TRValue { SSA = finalSSA; Type = recordType }

/// Witness record copy-and-update expression { orig with field1 = val1 }
/// Uses original record as base and updates only specified fields
/// Uses N pre-assigned SSAs for insertvalue operations (N = number of updated fields)
let witnessRecordCopyUpdate
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (origVal: Val)
    (fieldDefs: (string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType) list)
    (updatedFields: (string * Val) list)
    (recordType: MLIRType)
    : MLIROp list * TransferResult =

    if List.isEmpty updatedFields then
        // No updates - just return the original
        [], TRValue origVal
    else
        let ssas = requireNodeSSAs nodeId ctx

        // Build map of field names to their indices
        let fieldIndexMap =
            fieldDefs
            |> List.mapi (fun idx (name, _) -> (name, idx))
            |> Map.ofList

        // Start with original record, insert each updated field at its correct index
        let insertOps, finalSSA =
            updatedFields
            |> List.indexed
            |> List.fold (fun (ops, prevSSA) (i, (fieldName, fieldVal)) ->
                let newSSA = ssas.[i]
                let fieldIdx = Map.find fieldName fieldIndexMap  // Look up correct index
                let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newSSA, prevSSA, fieldVal.SSA, [fieldIdx], recordType))
                (ops @ [insertOp], newSSA)
            ) ([], origVal.SSA)

        insertOps, TRValue { SSA = finalSSA; Type = recordType }

// ═══════════════════════════════════════════════════════════════════════════
// ARRAY CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array expression [| e1; e2; ... |]
/// Allocates array and stores each element
/// Uses 5 + 2*N pre-assigned SSAs: count[0], alloca[1], (idx,gep)*N, undef, withPtr, result
let witnessArrayExpr
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    let count = List.length elements
    let countSSA = nextSSA ()
    let allocaSSA = nextSSA ()

    // Allocate array on stack
    let allocOps = [
        MLIROp.ArithOp (ArithOp.ConstI (countSSA, int64 count, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, countSSA, elemType, None))
    ]

    // Store each element
    let storeOps =
        elements
        |> List.indexed
        |> List.collect (fun (idx, elem) ->
            let idxSSA = nextSSA ()
            let ptrSSA = nextSSA ()
            [
                MLIROp.ArithOp (ArithOp.ConstI (idxSSA, int64 idx, MLIRTypes.i64))
                MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, allocaSSA, [(idxSSA, MLIRTypes.i64)], elemType))
                MLIROp.LLVMOp (LLVMOp.Store (elem.SSA, ptrSSA, elemType, NotAtomic))
            ]
        )

    // Build fat array struct { ptr, count }
    let undefSSA = nextSSA ()
    let withPtrSSA = nextSSA ()
    let resultSSA = nextSSA ()

    let arrayType = TStruct [TPtr; MLIRTypes.i64]

    let buildArrayOps = [
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, allocaSSA, [0], arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, countSSA, [1], arrayType))
    ]

    allocOps @ storeOps @ buildArrayOps, TRValue { SSA = resultSSA; Type = arrayType }

// ═══════════════════════════════════════════════════════════════════════════
// LIST CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness list expression [e1; e2; ...]
/// Lists are represented as linked cons cells or array-backed
/// For now, using array representation for simplicity
/// Uses same SSAs as witnessArrayExpr (delegates)
let witnessListExpr
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (elements: Val list)
    (elemType: MLIRType)
    : MLIROp list * TransferResult =

    // Use same representation as arrays for now
    witnessArrayExpr nodeId ctx elements elemType

// ═══════════════════════════════════════════════════════════════════════════
// UNION CASE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness discriminated union case construction
/// DU layout: { tag, payload... } where types come from unionType struct
/// For 2-element DUs (option), insert payload at [1]
/// For multi-payload DUs (result), insert payload at [tag+1]
/// Uses 3-5 pre-assigned SSAs depending on payload conversion needs
