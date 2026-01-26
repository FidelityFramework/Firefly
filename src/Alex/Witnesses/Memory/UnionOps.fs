/// Memory UnionOps Witness - DU tag operations
///
/// SCOPE: witnessDUGetTag, witnessDUEliminate, witnessDUConstruct
/// DOES NOT: Union case construction (separate witness)
module Alex.Witnesses.Memory.UnionOps

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Witnesses.Memory.Indexing

let witnessDUGetTag
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (duSSA: SSA)
    (duType: NativeType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx
    let tagType = MLIRTypes.i8

    // Map the DU type to get the MLIR type (graph-aware for consistency)
    let duMlirType = mapType duType ctx

    // Check if this is a pointer-based DU (heterogeneous like Result)
    match duMlirType with
    | TPtr ->
        // Pointer-based DU: need to load the struct first, then extract tag
        // We need the case struct type for the load - use minimal {i8} since we only need tag
        // Actually, we need to load the full struct to get proper alignment
        // For tag extraction, we can load as {i8, ...} but simplest is to load i8 directly
        // Since tag is always at offset 0, we can load just the tag byte
        let loadSSA = ssas.[0]
        let tagSSA = ssas.[1]
        
        // Load the tag byte directly from the pointer (tag is at offset 0)
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, duSSA, tagType, NotAtomic))
        
        [loadOp], TRValue { SSA = loadSSA; Type = tagType }

    | _ ->
        // Inline struct DU: extract tag directly from index 0
        let tagSSA = ssas.[0]
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, duSSA, [0], duMlirType))
        [extractOp], TRValue { SSA = tagSSA; Type = tagType }

/// Witness DU payload elimination (type-safe extraction)
/// This is the CASE ELIMINATOR pattern:
/// 1. Use the ACTUAL DU struct type for extractvalue (all cases share same layout)
/// 2. If the extracted slot type differs from desired payload type, bitcast
///
/// For example, DU Number = IntVal of int | FloatVal of float
/// - Runtime struct: (i8 tag, i64 payload) - i64 holds both int and float bits
/// - IntVal extraction: extractvalue[1] gives i64, use directly
/// - FloatVal extraction: extractvalue[1] gives i64, bitcast to f64
///
/// Uses 1-2 pre-assigned SSAs: extract[0], optional bitcast[1]
let witnessDUEliminate
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (duSSA: SSA)
    (duType: MLIRType)
    (caseIndex: int)
    (caseName: string)
    (payloadType: MLIRType)
    : MLIROp list * TransferResult =

    let ssas = requireNodeSSAs nodeId ctx

    // Check if this is a pointer-based DU (heterogeneous like Result)
    match duType with
    | TPtr ->
        // Pointer-based DU: load the case-specific struct, then extract payload
        // Case struct type is {i8 tag, PayloadType}
        let caseStructType = TStruct [TInt I8; payloadType]
        
        let loadSSA = ssas.[0]
        let extractSSA = ssas.[1]
        
        // Load the case-specific struct from the pointer
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, duSSA, caseStructType, NotAtomic))
        
        // Extract payload from index 1 (index 0 is tag)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (extractSSA, loadSSA, [1], caseStructType))
        
        [loadOp; extractOp], TRValue { SSA = extractSSA; Type = payloadType }

    | _ ->
        // Inline struct DU: extract directly
        let extractSSA = ssas.[0]

        // Get the actual slot type from the DU struct
        let slotType =
            match duType with
            | TStruct [_tagTy; payloadSlotTy] -> payloadSlotTy
            | _ -> payloadType

        // Extract using the ACTUAL DU type
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (extractSSA, duSSA, [1], duType))

        // Check if we need to bitcast (e.g., i64 -> f64 for FloatVal)
        if slotType = payloadType then
            [extractOp], TRValue { SSA = extractSSA; Type = payloadType }
        else
            let bitcastSSA = ssas.[1]
            let bitcastOp = MLIROp.LLVMOp (LLVMOp.Bitcast (bitcastSSA, extractSSA, slotType, payloadType))
            [extractOp; bitcastOp], TRValue { SSA = bitcastSSA; Type = payloadType }

/// Witness DU construction - creates a DU value with the given tag and optional payload
/// SSA layout: Nullary = 3, With payload = 4
let witnessDUConstruct
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (caseName: string)
    (caseIndex: int)
    (payloadOpt: Val option)
    (duType: MLIRType)
    : MLIROp list * TransferResult =

    // Check for DULayout coeffect (heterogeneous DUs needing arena allocation)
    match SSAAssignment.lookupDULayout nodeId ctx.Coeffects.SSA with
    | Some layout ->
        // Arena allocation path for heterogeneous DUs (e.g., Result<'T, 'E>)
        // Uses pre-computed SSAs from DULayout (coeffect pattern)

        // 1. Build case-specific struct: {i8 tag, PayloadType}
        let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (layout.StructUndefSSA, layout.CaseStructType))
        let tagOp = MLIROp.ArithOp (ArithOp.ConstI (layout.TagConstSSA, int64 caseIndex, MLIRTypes.i8))
        let withTagOp = MLIROp.LLVMOp (LLVMOp.InsertValue (layout.WithTagSSA, layout.StructUndefSSA, layout.TagConstSSA, [0], layout.CaseStructType))

        let structOps, finalStructSSA =
            match payloadOpt, layout.WithPayloadSSA with
            | Some payload, Some withPayloadSSA ->
                let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withPayloadSSA, layout.WithTagSSA, payload.SSA, [1], layout.CaseStructType))
                [undefOp; tagOp; withTagOp; insertPayloadOp], withPayloadSSA
            | None, None ->
                [undefOp; tagOp; withTagOp], layout.WithTagSSA
            | _ ->
                failwithf "DULayout payload mismatch for case %s" caseName

        // 2. Compute size using GEP null trick (all SSAs from layout)
        let sizeOps = [
            MLIROp.LLVMOp (LLVMOp.NullPtr layout.SizeNullPtrSSA)
            MLIROp.ArithOp (ArithOp.ConstI (layout.SizeOneSSA, 1L, MLIRTypes.i32))
            MLIROp.LLVMOp (LLVMOp.GEP (layout.SizeGepSSA, layout.SizeNullPtrSSA, [(layout.SizeOneSSA, MLIRTypes.i32)], layout.CaseStructType))
            MLIROp.LLVMOp (LLVMOp.PtrToInt (layout.SizeSSA, layout.SizeGepSSA, MLIRTypes.i64))
        ]

        // 3. Allocate from closure_heap arena (bump allocation)
        let allocOps = [
            MLIROp.LLVMOp (LLVMOp.AddressOf (layout.HeapPosPtrSSA, GFunc "closure_pos"))
            MLIROp.LLVMOp (LLVMOp.Load (layout.HeapPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
            MLIROp.LLVMOp (LLVMOp.AddressOf (layout.HeapBaseSSA, GFunc "closure_heap"))
            MLIROp.LLVMOp (LLVMOp.GEP (layout.HeapResultPtrSSA, layout.HeapBaseSSA, [(layout.HeapPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
            MLIROp.ArithOp (ArithOp.AddI (layout.HeapNewPosSSA, layout.HeapPosSSA, layout.SizeSSA, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Store (layout.HeapNewPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
        ]

        // 4. Store struct to arena
        let storeOp = MLIROp.LLVMOp (LLVMOp.Store (finalStructSSA, layout.HeapResultPtrSSA, layout.CaseStructType, NotAtomic))

        // Result is the pointer to arena-allocated case struct
        let allOps = structOps @ sizeOps @ allocOps @ [storeOp]
        allOps, TRValue { SSA = layout.HeapResultPtrSSA; Type = TPtr }

    | None ->
        // Inline path for homogeneous DUs (e.g., Option<'T>)
        // Direct struct representation - no arena allocation needed
        let ssas = requireNodeSSAs nodeId ctx
        let mutable ssaIdx = 0
        let nextSSA () =
            let ssa = ssas.[ssaIdx]
            ssaIdx <- ssaIdx + 1
            ssa

        // Start with undef
        let undefSSA = nextSSA ()
        let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, duType))

        // Create tag constant and insert at index 0
        let tagConstSSA = nextSSA ()
        let withTagSSA = nextSSA ()
        let tagOps = [
            MLIROp.ArithOp (ArithOp.ConstI (tagConstSSA, int64 caseIndex, MLIRTypes.i8))
            MLIROp.LLVMOp (LLVMOp.InsertValue (withTagSSA, undefSSA, tagConstSSA, [0], duType))
        ]

        // Insert payload if present
        match payloadOpt with
        | Some payload ->
            let resultSSA = nextSSA ()
            let insertPayloadOp = MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withTagSSA, payload.SSA, [1], duType))
            [undefOp] @ tagOps @ [insertPayloadOp], TRValue { SSA = resultSSA; Type = duType }
        | None ->
            // Nullary case - just tag, no payload
            [undefOp] @ tagOps, TRValue { SSA = withTagSSA; Type = duType }

