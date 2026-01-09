/// Binding Witness - Witness variable bindings to MLIR
///
/// Observes binding PSG nodes and generates corresponding MLIR.
/// Handles both immutable SSA bindings and mutable alloca bindings.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.BindingWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Traversal.MLIRZipper
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.MLIRTypes
module LitWitness = Alex.Witnesses.LiteralWitness
module Patterns = Alex.Patterns.SemanticPatterns

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a binding and generate corresponding MLIR
/// The value has already been witnessed (post-order traversal)
let witness 
    (name: string) 
    (isMutable: bool) 
    (valueNodeId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let nodeIdVal = NodeId.value node.Id
    
    // Check if this is a module-level mutable (needs LLVM global, not SSA)
    // Module-level mutables are identified during MutabilityAnalysis preprocessing
    if isMutable && MLIRZipper.isModuleLevelMutable name zipper then
        // Module-level mutable: emit LLVM global
        // The initial value was witnessed but we need to store it separately
        match MLIRZipper.lookupModuleLevelMutable name zipper with
        | Some (_bindingId, globalName) ->
            // Get the type from the node - use llvmType for LLVM dialect globals
            let mlirType = Serialize.llvmType (mapNativeType node.Type)
            // Observe the global (will be emitted at module level)
            let zipper1 = MLIRZipper.observeMutableGlobal globalName mlirType zipper
            // Note: Initial value handling is complex - for now, use zero-init
            // The initial value SSA was computed, but storing to global requires
            // a different pattern (can't be done at module scope, needs init func)
            // For simple cases like `let mutable x = 0`, zero-init is correct
            zipper1, TRVoid
        | None ->
            failwithf "CODEGEN ERROR: Module-level mutable '%s' not found in analysis" name
    else
        // The value has already been witnessed (post-order)
        match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
        | Some (ssa, ty) ->
            // Check if this is an addressed mutable (needs alloca)
            if isMutable && MLIRZipper.isAddressedMutable nodeIdVal zipper then
                // Addressed mutable: use alloca + store instead of pure SSA
                // 1. Emit alloca for the element type (ty is already a string)
                let allocaSSA, zipper1 = MLIRZipper.witnessAllocaStr ty zipper
                // 2. Store initial value into alloca
                let zipper2 = MLIRZipper.witnessStore ssa ty allocaSSA zipper1
                // 3. Record alloca for this binding (store element type for later loads)
                let zipper3 = MLIRZipper.recordMutableAlloca nodeIdVal allocaSSA ty zipper2
                // 4. Bind the *pointer* to the node (so AddressOf can find it)
                let zipper4 = MLIRZipper.bindNodeSSA (string nodeIdVal) allocaSSA "!llvm.ptr" zipper3
                // 5. Bind var name to the alloca pointer for VarRef to find
                let zipper5 = MLIRZipper.bindVar name allocaSSA "!llvm.ptr" zipper4
                zipper5, TRValue (allocaSSA, "!llvm.ptr")
            else
                // Immutable or non-addressed mutable: use pure SSA
                // For non-addressed mutable vars, Set operations will rebind to new SSA values
                // This enables SCF iter_args for loops (no alloca/load/store)
                let zipper' = MLIRZipper.bindNodeSSA (string nodeIdVal) ssa ty zipper
                let zipper'' = MLIRZipper.bindVar name ssa ty zipper'
                zipper'', TRValue (ssa, ty)
        | None ->
            // HARD STOP: Binding's value expression didn't produce an SSA
            // This means either:
            // 1. The value is genuinely unit (OK)
            // 2. The value expression isn't implemented (BUG - should fail at source)
            // Check the node's type to distinguish
            match node.Type with
            | NativeType.TApp (tycon, _) when tycon.Name = "unit" ->
                // Unit binding - no SSA needed, but still bind the node for consistency
                zipper, TRVoid
            | _ ->
                // Non-unit value didn't produce SSA - this is an implementation gap
                failwithf "CODEGEN ERROR: Binding '%s' (node %d) has type %A but value expression produced no SSA. The value expression is not yet implemented for native compilation." name (NodeId.value node.Id) node.Type

// ═══════════════════════════════════════════════════════════════════════════
// VARIABLE REFERENCE WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: Build function signature and observe extern func reference
/// Returns (zipper, TRValue) for the function reference
let private observeFuncRef (name: string) (ty: NativeType) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    let signature =
        match ty with
        | NativeType.TFun(paramTy, retTy) ->
            sprintf "(%s) -> %s"
                (Serialize.mlirType (mapNativeType paramTy))
                (Serialize.mlirType (mapNativeType retTy))
        | _ ->
            sprintf "(%s) -> %s"
                (Serialize.mlirType (mapNativeType ty))
                "!llvm.ptr"
    let retType =
        match ty with
        | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapNativeType retTy)
        | _ -> "!llvm.ptr"
    let zipper' = MLIRZipper.observeExternFunc name signature zipper
    zipper', TRValue ("@" + name, retType)

/// Check if a name is a built-in operator
let private isBuiltInOperator (name: string) =
    match name with
    | Patterns.BuiltInOperator _ -> true
    | _ -> false

/// Witness a variable reference (recall its SSA from prior observation)
/// INVARIANT: FNCS guarantees definitions are traversed before uses
/// If this fails, it's an FNCS graph construction bug - hard stop with diagnostic
let witnessVarRef (name: string) (defId: NodeId option) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Check if this is a module-level mutable - if so, we need addressof + load
    match MLIRZipper.lookupModuleLevelMutable name zipper with
    | Some (_bindingId, globalName) ->
        // Module-level mutable: use composable global load
        // Get the type from the definition node - use llvmType for LLVM dialect
        let elemType =
            match defId with
            | Some nodeId ->
                match SemanticGraph.tryGetNode nodeId graph with
                | Some node -> Serialize.llvmType (mapNativeType node.Type)
                | None -> "i32"  // Fallback
            | None -> "i32"  // Fallback

        // Composable: addressof + load
        let loadedSSA, zipper' = MLIRZipper.witnessGlobalLoad globalName elemType zipper
        zipper', TRValue (loadedSSA, elemType)
    | None ->
        // Not a module-level mutable - check if it's an addressed local mutable
        let isAddressedMut =
            match defId with
            | Some nodeId -> MLIRZipper.isAddressedMutable (NodeId.value nodeId) zipper
            | None -> false

        // First try to look up by variable name (for Lambda parameters)
        match MLIRZipper.recallVar name zipper with
        | Some (ssaName, mlirType) ->
            if isAddressedMut then
                // Addressed mutable: ssaName is the alloca pointer, we need to load the value
                match defId with
                | Some nodeId ->
                    match MLIRZipper.lookupMutableAlloca (NodeId.value nodeId) zipper with
                    | Some (_, elementType) ->
                        let loadedSSA, zipper' = MLIRZipper.witnessLoadStr ssaName elementType zipper
                        zipper', TRValue (loadedSSA, elementType)
                    | None -> zipper, TRValue (ssaName, mlirType)
                | None -> zipper, TRValue (ssaName, mlirType)
            else
                zipper, TRValue (ssaName, mlirType)
        | None ->
            // Not a bound variable - try definition node
            match defId with
            | Some nodeId ->
                match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
                | Some (ssaName, mlirType) ->
                    zipper, TRValue (ssaName, mlirType)
                | None ->
                    // Definition node wasn't traversed - check if it's a module-level binding
                    match SemanticGraph.tryGetNode nodeId graph with
                    | None ->
                        failwithf "FNCS GRAPH ERROR: Variable '%s' references node %d which is NOT in graph." name (NodeId.value nodeId)
                    | Some defNode ->
                        // Check if this is a closure capture (variable from outer scope)
                        // This happens when we're in a nested Lambda and reference an outer binding
                        match zipper.Focus, defNode.Kind with
                        | MLIRFocus.InFunction funcName, SemanticKind.Binding (_, _, _, _) when funcName <> "main" ->
                            // We're inside a nested function and referencing a binding
                            // Check if this binding is NOT a module-level binding (those are OK)
                            let isModuleLevel = MLIRZipper.isModuleLevelMutable name zipper ||
                                                MLIRZipper.lookupModuleLevelMutable name zipper |> Option.isSome
                            if not isModuleLevel then
                                // CLOSURE CAPTURE DETECTED - fail with clear diagnostic
                                // Per fsnative-spec/spec/reactive-signals.md Section 2.2:
                                // "Function pointers can only be created from top-level functions (no closures)"
                                failwithf "CLOSURE CAPTURE ERROR: Variable '%s' is captured from outer scope in function '%s'.\n\nFnPtr.ofFunction only supports top-level functions without captures.\nPer FNCS specification, closures are not supported for function pointers.\n\nTo fix:\n  1. Move '%s' to module level, OR\n  2. Pass '%s' as an explicit parameter to the function.\n\nSee: fsnative-spec/spec/reactive-signals.md Section 2.2" name funcName name name
                            else
                                observeFuncRef name defNode.Type zipper
                        | _ ->
                            match defNode.Kind, defNode.Children with
                            | SemanticKind.Binding (_, _, _, _), valueNodeId :: _ ->
                                // Check if the binding's value (Lambda) was traversed
                                match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
                                | Some (lambdaSSA, lambdaType) ->
                                    zipper, TRValue (lambdaSSA, lambdaType)
                                | None ->
                                    // Try looking for function name marker
                                    match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId) + "_lambdaName") zipper with
                                    | Some (lambdaName, _) ->
                                        zipper, TRValue ("@" + lambdaName, "!llvm.ptr")
                                    | None ->
                                        // DEFERRED RESOLUTION: Check value node
                                        match SemanticGraph.tryGetNode valueNodeId graph with
                                        | Some valueNode ->
                                            match valueNode.Kind with
                                            | SemanticKind.Literal lit ->
                                                LitWitness.witness lit zipper
                                            | SemanticKind.Lambda _ ->
                                                observeFuncRef name valueNode.Type zipper
                                            | _ ->
                                                observeFuncRef name valueNode.Type zipper
                                        | None ->
                                            observeFuncRef name defNode.Type zipper
                            | _ ->
                                observeFuncRef name defNode.Type zipper
            | None ->
                // No definition node - check if it's a built-in
                if isBuiltInOperator name then
                    zipper, TRBuiltin name
                else
                    zipper, TRError (sprintf "Variable '%s' has no definition node" name)
