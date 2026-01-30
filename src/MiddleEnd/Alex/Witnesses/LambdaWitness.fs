/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// SPECIAL CASE: Entry point Lambdas need to witness function bodies (sub-graphs)
/// that can contain ANY category of nodes. Uses subGraphCombinator to fold over
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

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// SUB-GRAPH COMBINATOR
// ═══════════════════════════════════════════════════════════

/// Build sub-graph combinator from nanopass list
/// Folds over all witnesses, returning first non-skip result
let private makeSubGraphCombinator (nanopasses: Nanopass list) : (WitnessContext -> SemanticNode -> WitnessOutput) =
    fun ctx node ->
        let rec tryWitnesses witnesses =
            match witnesses with
            | [] -> WitnessOutput.skip
            | nanopass :: rest ->
                match nanopass.Witness ctx node with
                | output when output = WitnessOutput.skip ->
                    tryWitnesses rest
                | output -> output
        tryWitnesses nanopasses

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
/// Takes nanopass list to build sub-graph combinator for body witnessing
let private witnessLambdaWith (nanopasses: Nanopass list) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // DEBUG: Check if we're being called for the Lambda node
    if node.Id = NodeId 536 then
        printfn "[DEBUG] LambdaWitness called for node 536, kind: %A" node.Kind

    // Filter to ONLY ContentPhase witnesses for sub-graph traversal
    // This prevents StructuralPhase witnesses from recursing and causing double-witnessing
    let contentWitnesses = nanopasses |> List.filter (fun np -> np.Phase = ContentPhase)
    let subGraphCombinator = makeSubGraphCombinator contentWitnesses

    match tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((params', bodyId, captureInfos), _) ->
        // Check if this is an entry point Lambda
        let nodeIdValue = NodeId.value node.Id
        let isEntryPoint = Set.contains nodeIdValue ctx.Coeffects.EntryPointLambdaIds

        if node.Id = NodeId 536 then
            printfn "[DEBUG] Pattern matched! isEntryPoint=%b, bodyId=%d" isEntryPoint (NodeId.value bodyId)

        if isEntryPoint then
            // Entry point Lambda: generate func.func @main wrapper
            let scopeLabel = sprintf "func_%d" nodeIdValue
            let scopeKind = FunctionScope "main"

            if node.Id = NodeId 536 then printfn "[DEBUG] Entering entry point path, adding scope markers"

            // Mark scope entry
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeEnter (scopeKind, scopeLabel))) ctx.Accumulator

            // Witness body nodes into shared accumulator with FRESH visited set
            // Use fresh visited set because we WANT to re-visit body nodes (already visited by ContentPhase)
            // to capture their operations within this scope
            let freshVisited = ref Set.empty
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper }
                    visitAllNodes subGraphCombinator bodyCtx bodyNode ctx.Accumulator freshVisited
                | None -> ()
            | None -> ()

            // Mark scope exit
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeExit (scopeKind, scopeLabel))) ctx.Accumulator

            // Extract operations between markers
            let scopeOps = MLIRAccumulator.extractScope scopeKind scopeLabel ctx.Accumulator
            if node.Id = NodeId 536 then printfn "[DEBUG] Entry point: Extracted %d operations from scope" (List.length scopeOps)

            // Separate body ops from module-level ops
            let bodyOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ -> false | _ -> true)
            let moduleOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ -> true | _ -> false)

            // Get body result for return value
            let bodyResult = MLIRAccumulator.recallNode bodyId ctx.Accumulator

            // Trust bodyResult - emit error if None
            let returnSSA, returnType =
                match bodyResult with
                | Some (ssa, ty) -> (Some ssa, ty)
                | None ->
                    let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some "Entry point return")
                                (sprintf "Body (node %d) produced no result - fix upstream witness" (NodeId.value bodyId))
                    MLIRAccumulator.addError err ctx.Accumulator
                    (None, TInt I32)

            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))
            let completeBody = bodyOps @ [returnOp]

            // Build func.func @main wrapper (portable)
            // Parameters: argv as !llvm.struct<(!llvm.ptr, i64)> fat pointer
            let argvType = TStruct [TPtr; TInt I64]
            let funcParams = [(SSA.Arg 0, argvType)]
            let funcDef = FuncOp.FuncDef("main", funcParams, returnType, completeBody, Public)

            // Replace scope with FuncDef
            MLIRAccumulator.replaceScope scopeKind scopeLabel (MLIROp.FuncOp funcDef) ctx.Accumulator

            // Add module-level ops separately
            MLIRAccumulator.addOps moduleOps ctx.Accumulator

            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        else
            // Non-entry-point Lambda: Generate FuncDef for module-level function
            // Extract function name from parent Binding node using XParsec pattern
            let funcName =
                match tryMatch pLambdaWithBinding ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((bindingName, _, _, _), _) ->
                    // No @ prefix - serializer will add it
                    bindingName
                | None ->
                    sprintf "lambda_%d" nodeIdValue

            let scopeLabel = sprintf "func_%d" nodeIdValue
            let scopeKind = FunctionScope funcName

            // Mark scope entry
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeEnter (scopeKind, scopeLabel))) ctx.Accumulator

            // Witness body nodes into shared accumulator with FRESH visited set
            // Use fresh visited set because we WANT to re-visit body nodes (already visited by ContentPhase)
            // to capture their operations within this scope
            let freshVisited = ref Set.empty
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper }
                    visitAllNodes subGraphCombinator bodyCtx bodyNode ctx.Accumulator freshVisited
                | None -> ()
            | None -> ()

            // Mark scope exit
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeExit (scopeKind, scopeLabel))) ctx.Accumulator

            // Extract operations between markers
            let scopeOps = MLIRAccumulator.extractScope scopeKind scopeLabel ctx.Accumulator

            // Separate body ops from module-level ops (GlobalStrings, etc.)
            let bodyOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> false | _ -> true)
            let moduleOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ | MLIROp.FuncOp (FuncOp.FuncDef _) -> true | _ -> false)

            // Get body result for return value
            let bodyResult = MLIRAccumulator.recallNode bodyId ctx.Accumulator

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

            // Trust bodyResult - emit error if None
            let returnSSA =
                match bodyResult with
                | Some (ssa, _) -> Some ssa
                | None ->
                    let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some (sprintf "%s return" funcName))
                                (sprintf "Body (node %d) produced no result - fix upstream witness" (NodeId.value bodyId))
                    MLIRAccumulator.addError err ctx.Accumulator
                    None

            // Build return operation
            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))

            let completeBody = bodyOps @ [returnOp]

            // Build FuncDef for module-level function
            let funcDef = FuncOp.FuncDef(funcName, mlirParams, returnType, completeBody, Public)

            // Replace scope with FuncDef
            MLIRAccumulator.replaceScope scopeKind scopeLabel (MLIROp.FuncOp funcDef) ctx.Accumulator

            // Add module-level ops separately
            MLIRAccumulator.addOps moduleOps ctx.Accumulator

            { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create Lambda nanopass with nanopass list for sub-graph traversal (body witnessing)
/// This must be called AFTER all other nanopasses are registered
let createNanopass (nanopasses: Nanopass list) : Nanopass = {
    Name = "Lambda"
    Phase = StructuralPhase
    Witness = witnessLambdaWith nanopasses
}

/// Placeholder nanopass export - will be replaced by createNanopass call in registry
let nanopass : Nanopass = {
    Name = "Lambda"
    Phase = StructuralPhase
    Witness = fun _ _ -> WitnessOutput.error "Lambda nanopass not properly initialized - use createNanopass"
}
