/// OptionWitness - Witness Option operations to MLIR
///
/// PRD-13a: Core Collections - Option<'T>
///
/// ARCHITECTURAL PRINCIPLES:
/// - Option is a discriminated union: {tag: i8, value: T}
/// - None: tag = 0, value undefined
/// - Some x: tag = 1, value = x
/// - ValueOption uses same layout but may be stack-optimized
///
/// PRIMITIVE OPERATIONS (Alex witnesses directly):
/// - None: construct with tag 0
/// - Some: construct with tag 1 and value
/// - isSome: tag == 1
/// - isNone: tag == 0
/// - get: extract value (assumes Some)
///
/// DECOMPOSABLE OPERATIONS (FNCS saturation):
/// - map, bind, defaultValue, defaultWith - pattern match on tag
module Alex.Witnesses.OptionWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Option struct type for a value type
/// Layout: {tag: i8, value: T}
/// Tag values: 0 = None, 1 = Some
let optionType (valueType: MLIRType) : MLIRType =
    TStruct [TInt I8; valueType]

/// None tag value
let noneTag : int64 = 0L

/// Some tag value
let someTag : int64 = 1L

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSA for a node
let private requireSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "OptionWitness: No SSA for node %A" nodeId

/// Get pre-assigned SSAs for a node
let private requireSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "OptionWitness: No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE WITNESSES
// ═══════════════════════════════════════════════════════════════════════════

/// Witness None<'T> - construct empty option
/// SSA cost: 3 (tag constant + undef + insertvalue)
let witnessNone
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let tagSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let resultSSA = ssas.[2]
    let optType = optionType valueType
    
    let ops = [
        MLIROp.ArithOp (ArithOp.ConstI (tagSSA, noneTag, TInt I8))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, optType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, undefSSA, tagSSA, [0], optType))
    ]
    ops, TRValue { SSA = resultSSA; Type = optType }

/// Witness Some x - construct option with value
/// SSA cost: 4 (tag constant + undef + insertvalue tag + insertvalue value)
let witnessSome
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (valueVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let tagSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withTagSSA = ssas.[2]
    let resultSSA = ssas.[3]
    let optType = optionType valueType
    
    let ops = [
        MLIROp.ArithOp (ArithOp.ConstI (tagSSA, someTag, TInt I8))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, optType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withTagSSA, undefSSA, tagSSA, [0], optType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withTagSSA, valueVal.SSA, [1], optType))
    ]
    ops, TRValue { SSA = resultSSA; Type = optType }

/// Witness Option.isSome - check if option has value
/// SSA cost: 3 (extractvalue + constant + icmp)
let witnessIsSome
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (optionVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let tagSSA = ssas.[0]
    let oneSSA = ssas.[1]
    let resultSSA = ssas.[2]
    let optType = optionType valueType
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, optionVal.SSA, [0], optType))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, someTag, TInt I8))
        MLIROp.ArithOp (ArithOp.CmpI (resultSSA, ICmpPred.Eq, tagSSA, oneSSA, TInt I8))
    ]
    ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 }

/// Witness Option.isNone - check if option is empty
/// SSA cost: 3 (extractvalue + constant + icmp)
let witnessIsNone
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (optionVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let tagSSA = ssas.[0]
    let zeroSSA = ssas.[1]
    let resultSSA = ssas.[2]
    let optType = optionType valueType
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, optionVal.SSA, [0], optType))
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, noneTag, TInt I8))
        MLIROp.ArithOp (ArithOp.CmpI (resultSSA, ICmpPred.Eq, tagSSA, zeroSSA, TInt I8))
    ]
    ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 }

/// Witness Option.get - extract value (assumes Some)
/// SSA cost: 1 (extractvalue)
let witnessGet
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (optionVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let resultSSA = requireSSA nodeId ssa
    let optType = optionType valueType
    
    let ops = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, optionVal.SSA, [1], optType))
    ]
    ops, TRValue { SSA = resultSSA; Type = valueType }

/// Witness Option.defaultValue - get value or default
/// Uses arith.select for branchless conditional
/// SSA cost: 5
let witnessDefaultValue
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (defaultVal: Val)
    (optionVal: Val)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =
    
    let ssas = requireSSAs nodeId ssa
    let tagSSA = ssas.[0]
    let oneSSA = ssas.[1]
    let isSomeSSA = ssas.[2]
    let valueSSA = ssas.[3]
    let resultSSA = ssas.[4]
    let optType = optionType valueType
    
    let ops = [
        // Extract tag and check if Some
        MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, optionVal.SSA, [0], optType))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, someTag, TInt I8))
        MLIROp.ArithOp (ArithOp.CmpI (isSomeSSA, ICmpPred.Eq, tagSSA, oneSSA, TInt I8))
        // Extract value
        MLIROp.LLVMOp (LLVMOp.ExtractValue (valueSSA, optionVal.SSA, [1], optType))
        // Select: isSome ? value : default
        MLIROp.ArithOp (ArithOp.Select (resultSSA, isSomeSSA, valueSSA, defaultVal.SSA, valueType))
    ]
    ops, TRValue { SSA = resultSSA; Type = valueType }

/// Witness Option.map - transform value if Some
/// Requires conditional and closure invocation
let witnessMap
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (mapperVal: Val)
    (optionVal: Val)
    (inputType: MLIRType)
    (outputType: MLIRType)
    : MLIROp list * TransferResult =
    
    // Complex - requires conditional + closure call
    // For cold implementation, return error - should decompose in FNCS
    [], TRError "Option.map requires conditional + closure call - use functional decomposition"

/// Witness Option.bind - flatMap for options
/// Requires conditional and closure invocation
let witnessBind
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (binderVal: Val)
    (optionVal: Val)
    (inputType: MLIRType)
    (outputType: MLIRType)
    : MLIROp list * TransferResult =
    
    [], TRError "Option.bind requires conditional + closure call - use functional decomposition"

// ═══════════════════════════════════════════════════════════════════════════
// SSA COST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// SSA cost for None
let noneSSACost : int = 3

/// SSA cost for Some
let someSSACost : int = 4

/// SSA cost for isSome
let isSomeSSACost : int = 3

/// SSA cost for isNone
let isNoneSSACost : int = 3

/// SSA cost for get
let getSSACost : int = 1

/// SSA cost for defaultValue
let defaultValueSSACost : int = 5

/// SSA cost for map (placeholder - requires control flow)
let mapSSACost : int = 15

/// SSA cost for bind (placeholder - requires control flow)
let bindSSACost : int = 20
