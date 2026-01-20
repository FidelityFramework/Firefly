/// Binding Witness - Witness variable bindings to MLIR
///
/// ARCHITECTURAL PRINCIPLE: Witnesses OBSERVE and RETURN structured MLIROp.
/// They do NOT emit. The FOLD accumulates via withOps.
/// Uses coeffects from preprocessing (SSAAssignment, MutabilityAnalysis).
module Alex.Witnesses.BindingWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.SSAAssignment
open PSGElaboration.MutabilityAnalysis
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
            let mlirType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch ctx.Graph node.Type
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
/// Witness a VarRef node - resolve variable to its SSA value
///
/// STRATIFIED LOOKUP ORDER (January 2026):
/// 1. Module-level mutables: addressof + load from global
/// 2. Addressed mutable (local): load from alloca
/// 3. VarBindings + NodeBindings (stratified):
///    - Captured mutables (ByRef): VarBindings has pointer, load through it
///    - Function parameters (PatternBinding): VarBindings directly
///    - VarBindings with name match: Check NodeBindings first by defId
///      - If NodeBindings has it: use it (local let, not captured)
///      - If not: use VarBindings (captured variable)
/// 4. NodeBindings (by defId) - for vars not in VarBindings
/// 5. SSAAssignment (by defId) - pre-computed coeffects
/// 6. Function reference - Lambda/Binding to Lambda returns TRVoid (called by name)
///
/// The stratified approach handles both:
/// - Name shadowing: local let `buffer` shadows parameter `buffer` from other function
/// - Closure captures: captured `min` uses VarBindings (not NodeBindings) inside closure
let witnessVarRef
    (ctx: WitnessContext)
    (z: PSGZipper)
    (varRefNodeId: NodeId)
    (name: string)
    (defId: NodeId option)
    : MLIROp list * TransferResult =

    // Helper: get type from definition node
    let getDefType nodeId =
        match SemanticGraph.tryGetNode nodeId ctx.Graph with
        | Some node -> mapNativeTypeWithGraphForArch z.State.Platform.TargetArch ctx.Graph node.Type
        | None -> MLIRTypes.i32

    // Helper: check if defNode is a function reference (Lambda or Binding to Lambda)
    let isFunctionRef nodeId =
        match PSGElaboration.SSAAssignment.lookupLambdaName nodeId ctx.SSA with
        | Some _ -> true
        | None -> false

    // Helper: check if defId points to a PatternBinding (function parameter)
    // vs a Binding (let expression) - this determines if VarBindings should be used
    let isParameterRef nodeId =
        match SemanticGraph.tryGetNode nodeId ctx.Graph with
        | Some defNode ->
            match defNode.Kind with
            | SemanticKind.PatternBinding _ -> true
            | _ -> false
        | None -> false

    // 1. Module-level mutable: addressof + load
    match getModuleLevelMutable name ctx with
    | Some _ ->
        let ssas = requireNodeSSAs varRefNodeId z
        let ptrSSA, loadSSA = ssas.[0], ssas.[1]
        let globalName = sprintf "g_%s" name
        let elemType = defId |> Option.map getDefType |> Option.defaultValue MLIRTypes.i32
        let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GFunc globalName))
        let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
        [addrOp; loadOp], TRValue { SSA = loadSSA; Type = elemType }
    | None ->

    // Need defId for most lookups
    match defId with
    | None ->
        // No defId - try VarBindings by name as last resort
        match recallVarSSA name z with
        | Some (ssa, ty) -> [], TRValue { SSA = ssa; Type = ty }
        | None -> [], TRError (sprintf "Variable '%s' has no definition" name)
    | Some nodeId ->

    let nodeIdVal = NodeId.value nodeId

    // 2. Addressed mutable (local): load from alloca
    if isAddressedMutable nodeIdVal ctx then
        match lookupSSAs nodeId ctx.SSA with
        | Some ssas when ssas.Length >= 2 ->
            let allocaSSA = ssas.[1]
            let loadSSA = requireNodeSSA varRefNodeId z
            let elemType = getDefType nodeId
            let loadOp = MLIROp.LLVMOp (Load (loadSSA, allocaSSA, elemType, NotAtomic))
            [loadOp], TRValue { SSA = loadSSA; Type = elemType }
        | _ -> [], TRError (sprintf "No alloca SSA for addressed mutable '%s'" name)
    else

    // 3. Captured variables (marked explicitly): use VarBindings
    // Captures are marked by LambdaWitness when entering closure scope
    // They must use VarBindings (capture extraction SSA), NOT NodeBindings
    if isCapturedVariable name z then
        match recallVarSSA name z with
        | Some (ptrSSA, _ptrTy) when isCapturedMutable name z ->
            // Captured mutable (ByRef): ptrSSA is pointer to the value, need to load
            let elemType = getDefType nodeId
            let loadSSA = requireNodeSSA varRefNodeId z
            let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
            [loadOp], TRValue { SSA = loadSSA; Type = elemType }
        | Some (ssa, ty) ->
            // Captured by value: use VarBindings directly
            [], TRValue { SSA = ssa; Type = ty }
        | None ->
            [], TRError (sprintf "Captured variable '%s' not bound in VarBindings" name)
    else

    // 4. Function parameters: use VarBindings
    match recallVarSSA name z with
    | Some (ssa, ty) when isParameterRef nodeId ->
        // Function parameter: use VarBindings directly (defId is PatternBinding)
        [], TRValue { SSA = ssa; Type = ty }
    | _ ->

    // 5. NodeBindings: let-bound values in current function scope
    match recallNodeResult nodeIdVal z with
    | Some (ssa, ty) -> [], TRValue { SSA = ssa; Type = ty }
    | None ->

    // 6. SSAAssignment: pre-computed coeffects (for non-parameter refs not in NodeBindings)
    match lookupSSA nodeId ctx.SSA with
    | Some ssa -> [], TRValue { SSA = ssa; Type = getDefType nodeId }
    | None ->

    // 7. Function reference - return builtin marker for Application to handle
    // This covers PlatformBindings, Intrinsics, and other external functions
    // that don't have SSAs in the current scope but are called by name
    [], TRBuiltin (name, [])

// ═══════════════════════════════════════════════════════════════════════════
// MUTABLE SET WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a mutable variable assignment (Set)
/// Generates STORE to the appropriate location (global or alloca)
/// setNodeId: The NodeId of the Set node itself (for SSA lookup if needed)
let witnessSet
    (ctx: WitnessContext)
    (z: PSGZipper)
    (setNodeId: NodeId)
    (name: string)
    (defId: NodeId option)
    (valueSSA: SSA)
    (valueType: MLIRType)
    : MLIROp list * TransferResult =

    // Check if this targets a module-level mutable
    match getModuleLevelMutable name ctx with
    | Some mlm ->
        // Module-level mutable: emit addressof + store
        let addrSSA = requireNodeSSA setNodeId z
        let globalName = sprintf "g_%s" name
        let addrOp = MLIROp.LLVMOp (AddressOf (addrSSA, GFunc globalName))
        let storeOp = MLIROp.LLVMOp (Store (valueSSA, addrSSA, valueType, NotAtomic))
        [addrOp; storeOp], TRVoid

    | None ->
        // Check if this is a captured mutable (ByRef capture)
        if isCapturedMutable name z then
            // Captured mutable: VarBindings has the pointer, store through it
            match recallVarSSA name z with
            | Some (ptrSSA, _ptrTy) ->
                let storeOp = MLIROp.LLVMOp (Store (valueSSA, ptrSSA, valueType, NotAtomic))
                [storeOp], TRVoid
            | None ->
                [], TRError (sprintf "Captured mutable '%s' not bound in scope" name)
        else
            // Local variable: check if addressed mutable
            match defId with
            | Some nodeId ->
                let nodeIdVal = NodeId.value nodeId

                if isAddressedMutable nodeIdVal ctx then
                    // Addressed mutable: store to the alloca
                    match lookupSSAs nodeId ctx.SSA with
                    | Some ssas when ssas.Length >= 2 ->
                        // The binding has: [oneConst, allocaPtr]
                        // We store to the allocaPtr (index 1)
                        let allocaSSA = ssas.[1]
                        let storeOp = MLIROp.LLVMOp (Store (valueSSA, allocaSSA, valueType, NotAtomic))
                        [storeOp], TRVoid
                    | _ ->
                        [], TRError (sprintf "No alloca SSA for addressed mutable '%s'" name)
                else
                    // Not an addressed mutable - this shouldn't happen for Set
                    // (Set targets must be mutable bindings)
                    [], TRError (sprintf "Set target '%s' is not a mutable binding" name)

            | None ->
                [], TRError (sprintf "Set target '%s' has no definition" name)
