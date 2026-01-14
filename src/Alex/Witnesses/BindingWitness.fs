/// Binding Witness - Witness variable bindings to MLIR
///
/// ARCHITECTURAL PRINCIPLE: Witnesses OBSERVE and RETURN structured MLIROp.
/// They do NOT emit. The FOLD accumulates via withOps.
/// Uses coeffects from preprocessing (SSAAssignment, MutabilityAnalysis).
module Alex.Witnesses.BindingWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.CodeGeneration.TypeMapping
open Alex.Preprocessing.SSAAssignment
open Alex.Preprocessing.MutabilityAnalysis
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS CONTEXT (Coeffects)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness context carries coeffects from preprocessing
type WitnessContext = {
    SSA: SSAAssignment
    Mutability: MutabilityAnalysisResult
    Graph: SemanticGraph
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a binding is module-level mutable
let private isModuleLevelMutable (nodeId: int) (ctx: WitnessContext) : bool =
    ctx.Mutability.ModuleLevelMutableBindings
    |> List.exists (fun m -> m.BindingId = nodeId)

/// Check if a binding needs alloca (addressed mutable)
let private isAddressedMutable (nodeId: int) (ctx: WitnessContext) : bool =
    Set.contains nodeId ctx.Mutability.AddressedMutableBindings

/// Get module-level mutable info
let private getModuleLevelMutable (name: string) (ctx: WitnessContext) : ModuleLevelMutable option =
    ctx.Mutability.ModuleLevelMutableBindings
    |> List.tryFind (fun m -> m.Name = name)

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a binding and generate corresponding MLIR
/// The value has already been witnessed (post-order traversal)
/// Returns (ops, result) - caller accumulates ops via withOps
let witness
    (ctx: WitnessContext)
    (z: PSGZipper)
    (node: SemanticNode)
    (name: string)
    (isMutable: bool)
    (valueNodeId: NodeId)
    (valueResult: TransferResult)
    : MLIROp list * TransferResult =

    let nodeIdVal = NodeId.value node.Id

    // Check if this is a module-level mutable (needs LLVM global, not SSA)
    if isMutable && isModuleLevelMutable nodeIdVal ctx then
        // Module-level mutable: emit LLVM global definition
        match getModuleLevelMutable name ctx with
        | Some mlm ->
            let mlirType = mapNativeType node.Type
            let globalName = sprintf "g_%s" name
            // Global definition with zero initialization
            // (actual initialization requires an init function for non-const values)
            let globalOp = MLIROp.LLVMOp (GlobalDef (globalName, "zeroinitializer", mlirType, false))
            [globalOp], TRVoid
        | None ->
            [], TRError (sprintf "Module-level mutable '%s' not found in analysis" name)

    // Addressed mutable: needs alloca + store
    // Requires 2 SSAs: oneConst + allocaResult
    elif isMutable && isAddressedMutable nodeIdVal ctx then
        match valueResult with
        | TRValue valueVal ->
            // Look up the pre-assigned SSAs for this binding
            match lookupSSAs node.Id ctx.SSA with
            | Some ssas when ssas.Length >= 2 ->
                let oneSSA = ssas.[0]
                let allocaSSA = ssas.[1]
                let elemType = valueVal.Type
                // 1. Alloca for the element type
                let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
                let allocaOp = MLIROp.LLVMOp (Alloca (allocaSSA, oneSSA, elemType, None))
                // 2. Store initial value
                let storeOp = MLIROp.LLVMOp (Store (valueVal.SSA, allocaSSA, elemType, NotAtomic))
                [oneOp; allocaOp; storeOp], TRValue { SSA = allocaSSA; Type = MLIRTypes.ptr }
            | _ ->
                [], TRError (sprintf "No SSAs assigned for mutable binding '%s'" name)
        | TRVoid ->
            [], TRError (sprintf "Mutable binding '%s' has void value" name)
        | TRError msg ->
            [], TRError msg
        | TRBuiltin (bname, _) ->
            [], TRError (sprintf "Cannot bind builtin '%s' to mutable '%s'" bname name)

    // Regular binding: pure SSA (no ops needed, value flows through)
    else
        match valueResult with
        | TRValue _ ->
            // Value result flows through - the binding just "names" it
            [], valueResult
        | TRVoid ->
            // Unit binding - valid
            [], TRVoid
        | TRError msg ->
            [], TRError msg
        | TRBuiltin (bname, args) ->
            // Builtin flows through
            [], TRBuiltin (bname, args)

// ═══════════════════════════════════════════════════════════════════════════
// VARIABLE REFERENCE WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a variable reference (look up its SSA from coeffect)
/// INVARIANT: PSG guarantees definitions are traversed before uses
/// varRefNodeId: The NodeId of the VarRef node itself (for SSA lookup)
let witnessVarRef
    (ctx: WitnessContext)
    (z: PSGZipper)
    (varRefNodeId: NodeId)
    (name: string)
    (defId: NodeId option)
    : MLIROp list * TransferResult =

    // Check if this references a module-level mutable
    match getModuleLevelMutable name ctx with
    | Some mlm ->
        // Module-level mutable: emit addressof + load (needs 2 SSAs)
        let ssas = requireNodeSSAs varRefNodeId z
        let ptrSSA = ssas.[0]
        let loadSSA = ssas.[1]
        
        let globalName = sprintf "g_%s" name
        let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GFunc globalName))

        // Get element type from definition node
        let elemType =
            match defId with
            | Some nodeId ->
                match SemanticGraph.tryGetNode nodeId ctx.Graph with
                | Some node -> mapNativeType node.Type
                | None -> MLIRTypes.i32
            | None -> MLIRTypes.i32

        let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
        [addrOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }

    | None ->
        // Local variable: look up SSA from coeffect
        match defId with
        | Some nodeId ->
            let nodeIdVal = NodeId.value nodeId

            // Check if it's an addressed mutable (need to load from alloca)
            if isAddressedMutable nodeIdVal ctx then
                match lookupSSAs nodeId ctx.SSA with
                | Some ssas when ssas.Length >= 2 ->
                    // The binding has: [oneConst, allocaPtr]
                    // We need the allocaPtr (index 1) to load from
                    let allocaSSA = ssas.[1]
                    // The VarRef node gets its own load result SSA
                    let loadSSA = requireNodeSSA varRefNodeId z
                    match SemanticGraph.tryGetNode nodeId ctx.Graph with
                    | Some defNode ->
                        let elemType = mapNativeType defNode.Type
                        let loadOp = MLIROp.LLVMOp (Load (loadSSA, allocaSSA, elemType, NotAtomic))
                        [loadOp], TRValue { SSA = loadSSA; Type = elemType }
                    | None ->
                        [], TRError (sprintf "Definition node %d not found" nodeIdVal)
                | _ ->
                    [], TRError (sprintf "No SSA for addressed mutable '%s'" name)
            else
                // Regular variable: look up runtime value from NodeBindings first
                // This captures the actual value SSA set during traversal (e.g., from stackalloc result)
                match recallNodeResult nodeIdVal z with
                | Some (ssa, ty) ->
                    [], TRValue { SSA = ssa; Type = ty }
                | None ->
                    // Fallback: try SSAAssignment for parameters and special cases
                    match lookupSSA nodeId ctx.SSA with
                    | Some ssa ->
                        // Get type from definition node
                        match SemanticGraph.tryGetNode nodeId ctx.Graph with
                        | Some defNode ->
                            let mlirType = mapNativeType defNode.Type
                            [], TRValue { SSA = ssa; Type = mlirType }
                        | None ->
                            [], TRError (sprintf "Definition node %d not found" nodeIdVal)
                    | None ->
                        // Function references don't get SSAs - called by name via Application witness
                        match SemanticGraph.tryGetNode nodeId ctx.Graph with
                        | Some defNode ->
                            match defNode.Kind with
                            | SemanticKind.Lambda _ -> [], TRVoid
                            | SemanticKind.Binding _ ->
                                match defNode.Children with
                                | [childId] ->
                                    match SemanticGraph.tryGetNode childId ctx.Graph with
                                    | Some cn when (match cn.Kind with SemanticKind.Lambda _ -> true | _ -> false) -> [], TRVoid
                                    | _ -> [], TRError (sprintf "No SSA assigned for variable '%s'" name)
                                | _ -> [], TRError (sprintf "No SSA assigned for variable '%s'" name)
                            | _ -> [], TRError (sprintf "No SSA assigned for variable '%s'" name)
                        | None -> [], TRError (sprintf "No SSA assigned for variable '%s'" name)
        | None ->
            [], TRError (sprintf "Variable '%s' has no definition" name)
