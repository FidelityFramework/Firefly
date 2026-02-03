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
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns
open Alex.XParsec.PSGCombinators  // For findLastValueNode

module SSAAssign = PSGElaboration.SSAAssignment

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
                visitAllNodes combinator ctx paramNode ctx.Accumulator ctx.GlobalVisited
            | None -> ()

        // Check if this is an entry point Lambda
        let nodeIdValue = NodeId.value node.Id
        let isEntryPoint = Set.contains nodeIdValue ctx.Coeffects.EntryPointLambdaIds

        if isEntryPoint then
            // Entry point Lambda: generate func.func @main wrapper
            // Create NESTED accumulator for body operations
            // Bindings still go to GLOBAL NodeBindings (shared with parent)
            let bodyAcc = MLIRAccumulator.empty()

            // THEN: Witness body nodes into nested accumulator with GLOBAL visited set
            // This prevents duplicate visitation by top-level traversal
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    // Body context: nested accumulator, but shared global bindings via ctx.Accumulator.NodeAssoc
                    let bodyCtx = { ctx with Zipper = bodyZipper; Accumulator = bodyAcc }
                    visitAllNodes combinator bodyCtx bodyNode bodyAcc ctx.GlobalVisited
                    
                    // Copy bindings from body accumulator to parent (for cross-scope lookups)
                    bodyAcc.NodeAssoc |> Map.iter (fun nodeId binding ->
                        ctx.Accumulator.NodeAssoc <- Map.add nodeId binding ctx.Accumulator.NodeAssoc)
                | None -> ()
            | None -> ()

            // Separate body ops from module-level ops, and REVERSE to get correct order
            // bodyAcc.AllOps accumulates in reverse (newest first), so reverse to get def-before-use order
            let bodyOps = 
                bodyAcc.AllOps 
                |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> false | _ -> true)
                |> List.rev
            let moduleOps = 
                bodyAcc.AllOps 
                |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> true | _ -> false)
                |> List.rev

            // Get body result for return value
            // Traverse Sequential structure to find actual value-producing node
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator

            // Determine return type from Lambda type signature
            let arch = ctx.Coeffects.Platform.TargetArch
            let expectedReturnType =
                match node.Type with
                | NativeType.TFun (_, retType) -> Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch retType
                | _ -> TInt I32  // Fallback

            // Handle bodyResult based on return type
            let returnSSA, returnType =
                match bodyResult with
                | Some (ssa, ty) -> (Some ssa, ty)
                | None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match node.Type with
                    | NativeType.TFun (_, NativeType.TApp ({ NTUKind = Some NTUunit }, [])) ->
                        // Unit-returning function - no result SSA is expected
                        (None, expectedReturnType)
                    | _ ->
                        // Non-unit function should have produced a result
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some "Entry point return")
                                    (sprintf "Body (node %d, actual value node %d) produced no result - fix upstream witness" 
                                             (NodeId.value bodyId) (NodeId.value actualValueNode))
                        MLIRAccumulator.addError err ctx.Accumulator
                        (None, expectedReturnType)

            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))
            let completeBody = bodyOps @ [returnOp]

            // Build func.func @main wrapper (portable MLIR)
            // Parameters: argv as memref<?xi8> (dynamic-sized buffer)
            let argvType = TMemRef (TInt I8)
            let funcParams = [(SSA.Arg 0, argvType)]
            let funcDef = FuncOp.FuncDef("main", funcParams, returnType, completeBody, Public)

            // Return FuncDef and module-level ops in TopLevelOps
            // visitAllNodes will add these to RootAccumulator
            let topLevelOps = MLIROp.FuncOp funcDef :: moduleOps

            { InlineOps = []; TopLevelOps = topLevelOps; Result = TRVoid }
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

            // Create NESTED accumulator for body operations
            let bodyAcc = MLIRAccumulator.empty()
            
            // Witness body nodes into nested accumulator with GLOBAL visited set
            // This prevents duplicate visitation by top-level traversal
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    // Body context: nested accumulator, shared global bindings
                    let bodyCtx = { ctx with Zipper = bodyZipper; Accumulator = bodyAcc }
                    visitAllNodes combinator bodyCtx bodyNode bodyAcc ctx.GlobalVisited
                    
                    // Copy bindings from body accumulator to parent
                    bodyAcc.NodeAssoc |> Map.iter (fun nodeId binding ->
                        ctx.Accumulator.NodeAssoc <- Map.add nodeId binding ctx.Accumulator.NodeAssoc)
                | None -> ()
            | None -> ()

            // Separate body ops from module-level ops, and REVERSE to get correct order
            // bodyAcc.AllOps accumulates in reverse (newest first), so reverse to get def-before-use order
            let bodyOps = 
                bodyAcc.AllOps 
                |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> false | _ -> true)
                |> List.rev
            let moduleOps = 
                bodyAcc.AllOps 
                |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> true | _ -> false)
                |> List.rev

            // Get body result for return value
            // Traverse Sequential structure to find actual value-producing node
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator

            // Map parameters to MLIR types and build parameter list with SSAs
            let arch = ctx.Coeffects.Platform.TargetArch
            let mlirParams =
                params'
                |> List.map (fun (paramName, paramType, paramNodeId) ->
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch paramType
                    // Lookup SSA for parameter from coeffects
                    match SSAAssign.lookupSSA paramNodeId ctx.Coeffects.SSA with
                    | Some paramSSA -> (paramSSA, mlirType)
                    | None ->
                        // Fallback: create fresh SSA (shouldn't happen if coeffects are correct)
                        let paramSSA = SSA.V (NodeId.value paramNodeId)
                        (paramSSA, mlirType))

            // Determine return type from Lambda type signature
            let returnType =
                match node.Type with
                | NativeType.TFun (_, retType) -> Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch retType
                | _ -> TInt I32  // Fallback

            // Handle bodyResult based on return type
            let returnSSA =
                match bodyResult with
                | Some (ssa, _) -> Some ssa
                | None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match node.Type with
                    | NativeType.TFun (_, NativeType.TApp ({ NTUKind = Some NTUunit }, [])) ->
                        // Unit-returning function - no result SSA is expected
                        None
                    | _ ->
                        // Non-unit function should have produced a result
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some (sprintf "%s return" funcName))
                                    (sprintf "Body (node %d, actual value node %d) produced no result - fix upstream witness" 
                                             (NodeId.value bodyId) (NodeId.value actualValueNode))
                        MLIRAccumulator.addError err ctx.Accumulator
                        None

            // Build return operation
            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))

            let completeBody = bodyOps @ [returnOp]

            // Build FuncDef for module-level function
            let funcDef = FuncOp.FuncDef(funcName, mlirParams, returnType, completeBody, Public)

            // Return FuncDef and module-level ops in TopLevelOps
            // visitAllNodes will add these to RootAccumulator
            let topLevelOps = MLIROp.FuncOp funcDef :: moduleOps

            { InlineOps = []; TopLevelOps = topLevelOps; Result = TRVoid }

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
