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
    elif isMutable && isAddressedMutable nodeIdVal ctx then
        match valueResult with
        | TRValue valueVal ->
            // Look up the pre-assigned SSA for this binding's alloca
            match lookupSSA node.Id ctx.SSA with
            | Some allocaSSA ->
                let elemType = valueVal.Type
                // 1. Alloca for the element type
                let oneSSA = freshSynthSSA z
                let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
                let allocaOp = MLIROp.LLVMOp (Alloca (allocaSSA, oneSSA, elemType, None))
                // 2. Store initial value
                let storeOp = MLIROp.LLVMOp (Store (valueVal.SSA, allocaSSA, elemType, NotAtomic))
                [oneOp; allocaOp; storeOp], TRValue { SSA = allocaSSA; Type = MLIRTypes.ptr }
            | None ->
                [], TRError (sprintf "No SSA assigned for mutable binding '%s'" name)
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
let witnessVarRef
    (ctx: WitnessContext)
    (z: PSGZipper)
    (name: string)
    (defId: NodeId option)
    : MLIROp list * TransferResult =

    // Check if this references a module-level mutable
    match getModuleLevelMutable name ctx with
    | Some mlm ->
        // Module-level mutable: emit addressof + load
        let globalName = sprintf "g_%s" name
        let ptrSSA = freshSynthSSA z
        let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GFunc globalName))

        // Get element type from definition node
        let elemType =
            match defId with
            | Some nodeId ->
                match SemanticGraph.tryGetNode nodeId ctx.Graph with
                | Some node -> mapNativeType node.Type
                | None -> MLIRTypes.i32
            | None -> MLIRTypes.i32

        let loadSSA = freshSynthSSA z
        let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
        [addrOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }

    | None ->
        // Local variable: look up SSA from coeffect
        match defId with
        | Some nodeId ->
            let nodeIdVal = NodeId.value nodeId

            // Check if it's an addressed mutable (need to load from alloca)
            if isAddressedMutable nodeIdVal ctx then
                match lookupSSA nodeId ctx.SSA with
                | Some allocaSSA ->
                    // The SSA is the alloca pointer - need to load
                    match SemanticGraph.tryGetNode nodeId ctx.Graph with
                    | Some defNode ->
                        let elemType = mapNativeType defNode.Type
                        let loadSSA = freshSynthSSA z
                        let loadOp = MLIROp.LLVMOp (Load (loadSSA, allocaSSA, elemType, NotAtomic))
                        [loadOp], TRValue { SSA = loadSSA; Type = elemType }
                    | None ->
                        [], TRError (sprintf "Definition node %d not found" nodeIdVal)
                | None ->
                    [], TRError (sprintf "No SSA for addressed mutable '%s'" name)
            else
                // Regular variable: look up its SSA directly
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
                    [], TRError (sprintf "No SSA assigned for variable '%s'" name)
        | None ->
            [], TRError (sprintf "Variable '%s' has no definition" name)
