/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// SPECIAL CASE: Entry point Lambdas need to witness function bodies (sub-graphs)
/// that can contain ANY category of nodes. Uses combinator to fold over
/// all registered witnesses.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.Traversal.ScopeContext
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns
open Alex.XParsec.PSGCombinators  // For findLastValueNode
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// Y-COMBINATOR PATTERN
// ═══════════════════════════════════════════════════════════
//
// Lambda witnesses need to handle nested lambdas (closures, higher-order functions).
// This requires recursive self-reference: the combinator must include itself.
//
// Solution: Y-combinator fixed point via thunk (unit -> Combinator)
// The combinator getter is passed from WitnessRegistry, allowing deferred evaluation
// and creating a proper fixed point where witnesses can recursively invoke themselves.

// ═══════════════════════════════════════════════════════════
// CURRY FLATTENING SUPPORT
// ═══════════════════════════════════════════════════════════

/// Unroll through N levels of TFun to find the innermost return type.
/// For a flattened Lambda with N params: TFun(t1, TFun(t2, ... TFun(tN, retType)...)) → retType
let rec private unrollReturnType (nParams: int) (ty: NativeType) : NativeType =
    if nParams <= 0 then ty
    else
        match ty with
        | NativeType.TFun (_, inner) -> unrollReturnType (nParams - 1) inner
        | _ -> ty

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
/// Takes combinator getter (Y-combinator thunk) for recursive self-reference
let private witnessLambdaWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Get the full combinator (including ourselves) via Y-combinator fixed point
    let combinator = getCombinator()

    match tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((params', bodyId, captureInfos), _) ->
        // FIRST: Visit parameter nodes (PatternBindings) to mark them as witnessed
        // ALL Lambdas must visit their parameters for coverage validation
        // Parameters are structural (SSA comes from coeffects), but must be visited
        for (_, _, paramNodeId) in params' do
            match SemanticGraph.tryGetNode paramNodeId ctx.Graph with
            | Some paramNode ->
                // Visit parameter with sub-graph combinator (will hit StructuralWitness)
                visitAllNodes combinator ctx paramNode ctx.GlobalVisited
            | None -> ()

        // Check if this is an entry point Lambda
        let nodeIdValue = NodeId.value node.Id
        let isEntryPoint = Set.contains nodeIdValue ctx.Coeffects.EntryPointLambdaIds

        if isEntryPoint then
            // Entry point Lambda: generate func.func @main wrapper

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedSSATypes = ctx.Accumulator.SSATypes
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            let argvType = TMemRef (TInt I8)
            MLIRAccumulator.registerSSAType (SSA.Arg 0) argvType ctx.Accumulator

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

            // Witness body nodes with child scope context
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper; ScopeContext = bodyScopeRef }
                    visitAllNodes combinator bodyCtx bodyNode ctx.GlobalVisited
                | None -> ()
            | None -> ()

            // Restore parent's SSATypes (isolate this function's registrations)
            ctx.Accumulator.SSATypes <- savedSSATypes

            // Extract operations from child scope ref (NOT from parent!)
            let bodyOps = ScopeContext.getOps !bodyScopeRef

            // Get body result for return value
            // Traverse Sequential structure to find actual value-producing node
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun
            let arch = ctx.Coeffects.Platform.TargetArch
            let innerReturnNativeType = unrollReturnType (List.length params') node.Type
            let expectedReturnType =
                Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch innerReturnNativeType

            // Handle bodyResult based on return type
            let returnSSA, returnType =
                match bodyResult with
                | Some (ssa, ty) -> (Some ssa, ty)
                | None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match innerReturnNativeType with
                    | NativeType.TApp ({ NTUKind = Some NTUunit }, []) ->
                        // Unit-returning function - no result SSA is expected
                        (None, expectedReturnType)
                    | _ ->
                        // Non-unit function should have produced a result
                        let bodyNodeKindStr =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode ->
                                let kindStr = sprintf "%A" bodyNode.Kind |> fun s -> s.Split('\n').[0]
                                let typeStr = sprintf "%A" bodyNode.Type
                                sprintf "Body node %d is %s (type: %s)" (NodeId.value actualValueNode) kindStr typeStr
                            | None ->
                                sprintf "Body node %d not found in graph" (NodeId.value actualValueNode)
                        let hint =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode when bodyNode.Kind.ToString().StartsWith("Lambda") ->
                                " [HINT: Body is a nested Lambda — Lambda produces TRVoid (emits FuncDef as side-effect). " +
                                "Returning a function value (currying/thunk) is not yet implemented]"
                            | _ -> ""
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some "Entry point return")
                                    (sprintf "%s — produced no result.%s" bodyNodeKindStr hint)
                        MLIRAccumulator.addError err ctx.Accumulator
                        (None, expectedReturnType)

            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))
            let completeBody = bodyOps @ [returnOp]

            // Build func.func @main wrapper (portable MLIR)
            // Parameters: argv as memref<?xi8> (dynamic-sized buffer)
            let argvType = TMemRef (TInt I8)
            let funcParams = [(SSA.Arg 0, argvType)]
            let funcDef = FuncOp.FuncDef("main", funcParams, returnType, completeBody, Public)

            // Add FuncDef to parent scope (ctx.ScopeContext, which is root for entry points)
            let updatedParentScope = ScopeContext.addOp (MLIROp.FuncOp funcDef) !ctx.ScopeContext
            ctx.ScopeContext := updatedParentScope

            // Return empty - FuncDef already added to parent scope
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        else
            // Non-entry-point Lambda: Generate FuncDef for module-level function
            // Extract QUALIFIED function name from parent Binding + ModuleDef (if present)
            // Same logic as ApplicationWitness for qualified name resolution
            let funcName =
                // Lambda's parent should be Binding node - extract name directly from PSG
                match node.Parent with
                | Some bindingId ->
                    match SemanticGraph.tryGetNode bindingId ctx.Graph with
                    | Some bindingNode ->
                        match bindingNode.Kind with
                        | SemanticKind.Binding (bindingName, _, _, _) ->
                            // Got binding name - check if Binding has ModuleDef parent for qualification
                            match bindingNode.Parent with
                            | Some moduleParentId ->
                                match SemanticGraph.tryGetNode moduleParentId ctx.Graph with
                                | Some moduleParent ->
                                    match moduleParent.Kind with
                                    | SemanticKind.ModuleDef (moduleName, _) ->
                                        // Qualified name: Module.Function
                                        sprintf "%s.%s" moduleName bindingName
                                    | _ -> bindingName  // No ModuleDef parent, use binding name
                                | None -> bindingName
                            | None -> bindingName
                        | _ ->
                            // Parent is not a Binding - use generic lambda name
                            sprintf "lambda_%d" nodeIdValue
                    | None -> sprintf "lambda_%d" nodeIdValue
                | None -> sprintf "lambda_%d" nodeIdValue

            // Map parameters to MLIR types and build parameter list with SSAs
            let arch = ctx.Coeffects.Platform.TargetArch

            // Extract parameter SSAs monadically
            let extractParamSSAs =
                parser {
                    let rec extractParams ps =
                        parser {
                            match ps with
                            | [] -> return []
                            | (paramName, paramType, paramNodeId) :: rest ->
                                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch paramType
                                let! paramSSA = getNodeSSA paramNodeId
                                let! restParams = extractParams rest
                                return (paramSSA, mlirType) :: restParams
                        }
                    return! extractParams params'
                }

            let mlirParams =
                match tryMatch extractParamSSAs ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some (paramList, _) -> paramList
                | None ->
                    // ARCHITECTURAL VIOLATION REMOVED: Was generating SSAs at runtime
                    // SSAs MUST come from SSAAssignment coeffects only
                    // If this fails, SSAAssignment nanopass needs to be fixed
                    printfn "[ERROR] LambdaWitness: Parameter SSAs not found in coeffects for Lambda node %A" (NodeId.value node.Id)
                    printfn "[ERROR] This indicates SSAAssignment nanopass failed to pre-allocate parameter SSAs"
                    printfn "[ERROR] Parameters: %A" params'
                    []  // Return empty list - will cause compilation to fail with proper error

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedSSATypes = ctx.Accumulator.SSATypes
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            for (paramSSA, mlirType) in mlirParams do
                MLIRAccumulator.registerSSAType paramSSA mlirType ctx.Accumulator

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

            // Witness body nodes with child scope context
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper; ScopeContext = bodyScopeRef }
                    visitAllNodes combinator bodyCtx bodyNode ctx.GlobalVisited
                | None -> ()
            | None -> ()

            // Restore parent's SSATypes (isolate this function's registrations)
            ctx.Accumulator.SSATypes <- savedSSATypes

            // Extract operations from child scope ref (NOT from parent!)
            let bodyOps = ScopeContext.getOps !bodyScopeRef

            // Get body result for return value
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun
            let innerReturnNativeType2 = unrollReturnType (List.length params') node.Type
            let returnType =
                Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch innerReturnNativeType2

            // Handle bodyResult based on return type
            let returnSSA =
                match bodyResult with
                | Some (ssa, _) -> Some ssa
                | None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match innerReturnNativeType2 with
                    | NativeType.TApp ({ NTUKind = Some NTUunit }, []) ->
                        // Unit-returning function - no result SSA is expected
                        None
                    | _ ->
                        // Non-unit function should have produced a result
                        // Enrich diagnostic with body node kind to diagnose nested-lambda / currying gaps
                        let bodyNodeKindStr =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode ->
                                let kindStr = sprintf "%A" bodyNode.Kind |> fun s -> s.Split('\n').[0]
                                let typeStr = sprintf "%A" bodyNode.Type
                                sprintf "Body node %d is %s (type: %s)" (NodeId.value actualValueNode) kindStr typeStr
                            | None ->
                                sprintf "Body node %d not found in graph" (NodeId.value actualValueNode)
                        let hint =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode when bodyNode.Kind.ToString().StartsWith("Lambda") ->
                                " [HINT: Body is a nested Lambda — Lambda produces TRVoid (emits FuncDef as side-effect). " +
                                "Returning a function value (currying/thunk) is not yet implemented]"
                            | _ -> ""
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some (sprintf "%s return" funcName))
                                    (sprintf "%s — produced no result.%s" bodyNodeKindStr hint)
                        MLIRAccumulator.addError err ctx.Accumulator
                        None

            // Build return operation
            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))

            let completeBody = bodyOps @ [returnOp]

            // Build FuncDef for module-level function
            let funcDef = FuncOp.FuncDef(funcName, mlirParams, returnType, completeBody, Public)

            // Add FuncDef to ROOT scope (module level - all FuncDefs are top-level in MLIR)
            let updatedRootScope = ScopeContext.addOp (MLIROp.FuncOp funcDef) !ctx.RootScopeContext
            ctx.RootScopeContext := updatedRootScope

            // Return empty - FuncDef already added to root scope
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create Lambda nanopass with Y-combinator thunk for recursive self-reference
/// The combinator getter allows deferred evaluation, creating a fixed point where
/// this witness can handle nested lambdas (closures, higher-order functions)
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass = {
    Name = "Lambda"
    Witness = witnessLambdaWith getCombinator
}

/// Placeholder nanopass export - will be replaced by createNanopass call in registry
let nanopass : Nanopass = {
    Name = "Lambda"
    Witness = fun _ _ -> WitnessOutput.error "Lambda nanopass not properly initialized - use createNanopass with Y-combinator"
}
