/// FNCSTransfer - Transfer SemanticGraph to MLIR via witnesses
///
/// This is NOT a layer. It is orchestration code that:
/// 1. Sets up coeffects (SSA assignment, mutability analysis)
/// 2. Runs FNCS traversal with witnesses
/// 3. Serializes accumulated ops to MLIR text
///
/// Region ops for SCF constructs are collected in LOCAL state,
/// not in the zipper. The zipper remains pure navigation + emission.
module Alex.Traversal.FNCSTransfer

open System.Collections.Generic
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.PSGZipper
open Alex.Bindings.BindingTypes
open Alex.Bindings.PlatformTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns
open Alex.Preprocessing.PlatformConfig

// Witness modules
module LitWitness = Alex.Witnesses.LiteralWitness
module PreprocessingSerializer = Alex.Preprocessing.PreprocessingSerializer
module BindWitness = Alex.Witnesses.BindingWitness
module AppWitness = Alex.Witnesses.Application.Witness
module CFWitness = Alex.Witnesses.ControlFlowWitness
module MemWitness = Alex.Witnesses.MemoryWitness
module LambdaWitness = Alex.Witnesses.LambdaWitness

// Preprocessing modules (coeffects)
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis
module SSAAssign = Alex.Preprocessing.SSAAssignment
module StringCollect = Alex.Preprocessing.StringCollection
module PlatformResolution = Alex.Preprocessing.PlatformBindingResolution

/// Map FNCS NativeType to MLIR type
let private mapType = mapNativeType

/// Convert RegionKind to int for dictionary key
let private regionKindToInt (kind: RegionKind) : int =
    match kind with
    | RegionKind.GuardRegion -> 0
    | RegionKind.BodyRegion -> 1
    | RegionKind.ThenRegion -> 2
    | RegionKind.ElseRegion -> 3
    | RegionKind.StartExprRegion -> 4
    | RegionKind.EndExprRegion -> 5
    | RegionKind.LambdaBodyRegion -> 6
    | RegionKind.MatchCaseRegion idx -> 100 + idx

/// Core transfer implementation
let private transferGraphCore
    (graph: SemanticGraph)
    (isFreestanding: bool)
    (collectErrors: bool)
    (intermediatesDir: string option)
    : string * string list =

    // ═══════════════════════════════════════════════════════════════════════
    // COEFFECTS (computed ONCE before transfer)
    // All preprocessing passes run here - the zipper is PASSIVE (no mutation)
    // ═══════════════════════════════════════════════════════════════════════
    let entryPointLambdaIds = MutAnalysis.findEntryPointLambdaIds graph
    let analysisResult = MutAnalysis.analyze graph
    let ssaAssignment = SSAAssign.assignSSA graph
    let stringTable = StringCollect.collect graph  // Pre-collect all strings

    // Platform resolution (nanopass)
    let hostPlatform = TargetPlatform.detectHost()
    let runtimeMode = if isFreestanding then Freestanding else Console
    let platformResolution = PlatformResolution.analyze graph runtimeMode hostPlatform.OS hostPlatform.Arch

    // Serialize preprocessing results to intermediates (if path provided)
    match intermediatesDir with
    | Some dir -> PreprocessingSerializer.serialize dir ssaAssignment entryPointLambdaIds graph
    | None -> ()

    // Create WitnessContext for binding witnesses
    let witnessCtx: BindWitness.WitnessContext = {
        SSA = ssaAssignment
        Mutability = analysisResult
        Graph = graph
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LOCAL STATE for region ops (NOT in zipper)
    // ═══════════════════════════════════════════════════════════════════════
    let regionOps = Dictionary<int * int, MLIROp list>()
    let mutable errors: string list = []

    // ═══════════════════════════════════════════════════════════════════════
    // SCF REGION HOOK (closes over regionOps)
    // ═══════════════════════════════════════════════════════════════════════
    let scfHook: SCFRegionHook<PSGZipper> = {
        BeforeRegion = fun z _parentId _kind ->
            enterRegion z
            z
        AfterRegion = fun z parentId kind ->
            let ops = exitRegion z
            let key = (NodeId.value parentId, regionKindToInt kind)
            regionOps.[key] <- ops
            z
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HELPER: Resolve NodeId to Val
    // ═══════════════════════════════════════════════════════════════════════
    let resolveNodeToVal (nodeId: NodeId) (z: PSGZipper) : Val option =
        match recallNodeResult (NodeId.value nodeId) z with
        | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
        | None -> None

    // ═══════════════════════════════════════════════════════════════════════
    // WITNESS DISPATCH (closes over regionOps, graph, witnessCtx)
    // ═══════════════════════════════════════════════════════════════════════
    let witnessNode (z: PSGZipper) (node: SemanticNode) : PSGZipper =
        let nodeIdVal = NodeId.value node.Id

        let z', result =
            match node.Kind with
            // ─────────────────────────────────────────────────────────────────
            // Literals
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Literal lit ->
                let ops, result = LitWitness.witness z node.Id lit
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Variable references - dispatch to witness (handles addressed mutables)
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.VarRef (name, defIdOpt) ->
                let ops, result = BindWitness.witnessVarRef witnessCtx z node.Id name defIdOpt
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Platform bindings (marker only - Application handles emission)
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.PlatformBinding _ ->
                z, TRVoid

            // ─────────────────────────────────────────────────────────────────
            // Intrinsics (marker only - Application handles emission)
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Intrinsic _ ->
                z, TRVoid

            // ─────────────────────────────────────────────────────────────────
            // Function applications
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Application (funcId, argIds) ->
                let ops, result = AppWitness.witness node.Id funcId argIds node.Type z
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Sequential expressions
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Sequential nodeIds ->
                let ops, result = CFWitness.witnessSequential z nodeIds
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Bindings
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Binding (name, isMutable, _isRecursive, _isEntryPoint) ->
                match node.Children with
                | valueId :: _ ->
                    // Get the value result from already-processed child
                    let valueResult =
                        match recallNodeResult (NodeId.value valueId) z with
                        | Some (ssa, ty) -> TRValue { SSA = ssa; Type = ty }
                        | None -> TRVoid
                    let ops, result = BindWitness.witness witnessCtx z node name isMutable valueId valueResult
                    emitAll ops z
                    z, result
                | [] ->
                    z, TRError "Binding has no value child"

            // ─────────────────────────────────────────────────────────────────
            // Module/Type definitions (containers, no code)
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.ModuleDef _ ->
                z, TRVoid

            | SemanticKind.TypeDef _ ->
                z, TRVoid

            // ─────────────────────────────────────────────────────────────────
            // Record expressions
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.RecordExpr (fields, _copyFrom) ->
                // fields is (string * NodeId) list - resolve each field value
                let fieldVals =
                    fields
                    |> List.choose (fun (name, valueId) ->
                        match resolveNodeToVal valueId z with
                        | Some v -> Some (name, v)
                        | None -> None)
                let recordType = mapType node.Type
                let ops, result = MemWitness.witnessRecordExpr node.Id z fieldVals recordType
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Lambdas
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Lambda (params', bodyId) ->
                // Get accumulated body ops from regionOps (populated by scfHook)
                let bodyKey = (nodeIdVal, regionKindToInt RegionKind.LambdaBodyRegion)
                let bodyOps = regionOps.GetValueOrDefault(bodyKey, [])

                // Witness returns: (funcDef option, localOps, result)
                // Photographer Principle: witness RETURNS, we ACCUMULATE
                let funcDefOpt, localOps, result = LambdaWitness.witness params' bodyId node bodyOps z

                // Add function definition to top-level (if present)
                match funcDefOpt with
                | Some funcDef -> emitTopLevel funcDef z
                | None -> ()

                // Emit local ops (addressof) to current scope
                emitAll localOps z

                // ARCHITECTURAL FIX: Exit function scope to restore parent's CurrentOps
                let z' = exitFunction z
                z', result

            // ─────────────────────────────────────────────────────────────────
            // Type annotations (semantically transparent)
            // TypeAnnotation's node.Type IS the annotated type - provided by FNCS
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.TypeAnnotation (exprId, _) ->
                // Check if this TypeAnnotation is unit-typed using the node's own Type
                // FNCS sets node.Type = annotatedType during construction
                let isUnit =
                    match node.Type with
                    | NativeType.TApp(tc, []) ->
                        match tc.NTUKind with
                        | Some NTUKind.NTUunit -> true
                        | _ -> false
                    | _ -> false
                if isUnit then
                    // Unit-typed annotation - no value to pass through
                    z, TRVoid
                else
                    // Non-unit annotation - pass through the inner expression's result
                    // If inner is a marker (Intrinsic, PlatformBinding), it has no result - that's OK
                    match recallNodeResult (NodeId.value exprId) z with
                    | Some (ssa, ty) ->
                        z, TRValue { SSA = ssa; Type = ty }
                    | None ->
                        // Inner is a marker - TypeAnnotation is also transparent
                        z, TRVoid

            // ─────────────────────────────────────────────────────────────────
            // Mutable set
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Set (targetId, valueId) ->
                match SemanticGraph.tryGetNode targetId graph with
                | Some targetNode ->
                    match targetNode.Kind with
                    | SemanticKind.VarRef (name, _) ->
                        match recallNodeResult (NodeId.value valueId) z with
                        | Some (valueSSA, valueType) ->
                            match lookupModuleLevelMutable name z with
                            | Some (_, globalName) ->
                                // Set node needs 1 SSA for addressof intermediate
                                let addrSSA = requireNodeSSA node.Id z
                                let addrOp = MLIROp.LLVMOp (LLVMOp.AddressOf (addrSSA, GlobalRef.GNamed globalName))
                                let storeOp = MLIROp.LLVMOp (LLVMOp.Store (valueSSA, addrSSA, valueType, AtomicOrdering.NotAtomic))
                                emit addrOp z
                                emit storeOp z
                                z, TRVoid
                            | None ->
                                let z' = bindVarSSA name valueSSA valueType z
                                z', TRVoid
                        | None ->
                            z, TRError (sprintf "Set: value not computed for '%s'" name)
                    | _ ->
                        match recallNodeResult (NodeId.value targetId) z,
                              recallNodeResult (NodeId.value valueId) z with
                        | Some (targetSSA, _), Some (valueSSA, valueType) ->
                            let storeOp = MLIROp.LLVMOp (LLVMOp.Store (valueSSA, targetSSA, valueType, AtomicOrdering.NotAtomic))
                            emit storeOp z
                            z, TRVoid
                        | _ ->
                            z, TRError "Set: target or value not computed"
                | None ->
                    z, TRError "Set: target node not found"

            // ─────────────────────────────────────────────────────────────────
            // Control flow - use collected region ops
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.IfThenElse (guardId, thenId, elseIdOpt) ->
                let thenKey = (nodeIdVal, regionKindToInt RegionKind.ThenRegion)
                let elseKey = (nodeIdVal, regionKindToInt RegionKind.ElseRegion)

                // Get guard result from node bindings
                let condSSA =
                    match recallNodeResult (NodeId.value guardId) z with
                    | Some (ssa, _) -> ssa
                    | None -> failwithf "IfThenElse: guard node %d result not found" (NodeId.value guardId)

                let thenOps = regionOps.GetValueOrDefault(thenKey, [])
                let thenResultSSA =
                    match recallNodeResult (NodeId.value thenId) z with
                    | Some (ssa, _) -> Some ssa
                    | None -> None

                let elseOps, elseResultSSA =
                    match elseIdOpt with
                    | Some elseId ->
                        let ops = regionOps.GetValueOrDefault(elseKey, [])
                        let ssa = match recallNodeResult (NodeId.value elseId) z with
                                  | Some (s, _) -> Some s
                                  | None -> None
                        Some ops, ssa
                    | None -> None, None

                let resultType =
                    let mlirType = mapType node.Type
                    if mlirType = TUnit then None else Some mlirType

                let ops, result = CFWitness.witnessIfThenElse node.Id z condSSA thenOps thenResultSSA elseOps elseResultSSA resultType
                emitAll ops z
                z, result

            | SemanticKind.WhileLoop (guardId, _bodyId) ->
                let guardKey = (nodeIdVal, regionKindToInt RegionKind.GuardRegion)
                let bodyKey = (nodeIdVal, regionKindToInt RegionKind.BodyRegion)

                let condOps = regionOps.GetValueOrDefault(guardKey, [])
                let bodyOps = regionOps.GetValueOrDefault(bodyKey, [])

                let condResultSSA =
                    match recallNodeResult (NodeId.value guardId) z with
                    | Some (ssa, _) -> ssa
                    | None -> failwithf "WhileLoop: guard node %d result not found" (NodeId.value guardId)

                // Get iter args from mutability analysis
                let iterArgs: Val list = []  // TODO: from coeffects

                let ops, result = CFWitness.witnessWhileLoop node.Id z condOps condResultSSA bodyOps iterArgs
                emitAll ops z
                z, result

            | SemanticKind.ForLoop (_varName, startId, finishId, isUp, _bodyId) ->
                let bodyKey = (nodeIdVal, regionKindToInt RegionKind.BodyRegion)
                let bodyOps = regionOps.GetValueOrDefault(bodyKey, [])

                // Get pre-assigned SSAs for ForLoop: [ivSSA; stepSSA]
                let ssas = requireNodeSSAs node.Id z
                let ivSSA = ssas.[0]
                let stepSSA = ssas.[1]

                let startSSA =
                    match recallNodeResult (NodeId.value startId) z with
                    | Some (ssa, _) -> ssa
                    | None -> failwithf "ForLoop: start node %d result not found" (NodeId.value startId)
                let stopSSA =
                    match recallNodeResult (NodeId.value finishId) z with
                    | Some (ssa, _) -> ssa
                    | None -> failwithf "ForLoop: finish node %d result not found" (NodeId.value finishId)

                // Step is 1 or -1 depending on isUp
                let stepVal = if isUp then 1L else -1L
                let stepOp = MLIROp.ArithOp (ArithOp.ConstI (stepSSA, stepVal, MLIRTypes.index))
                emit stepOp z

                let iterArgs: Val list = []  // TODO: from coeffects

                let ops, result = CFWitness.witnessForLoop node.Id z ivSSA startSSA stopSSA stepSSA bodyOps iterArgs
                emitAll ops z
                z, result

            | SemanticKind.Match (scrutineeId, cases) ->
                // Get scrutinee SSA - MUST exist, processed before this node
                let scrutineeSSA, scrutineeType =
                    match recallNodeResult (NodeId.value scrutineeId) z with
                    | Some (ssa, ty) -> ssa, ty
                    | None -> failwithf "Match: scrutinee node %d result not found" (NodeId.value scrutineeId)

                // Build cases with collected region ops
                // Use case index as tag (actual tag resolution would come from Pattern)
                let caseOps =
                    cases
                    |> List.mapi (fun idx case ->
                        let caseKey = (nodeIdVal, regionKindToInt (RegionKind.MatchCaseRegion idx))
                        let ops = regionOps.GetValueOrDefault(caseKey, [])
                        // case.Body is NodeId (not option)
                        let resultSSA =
                            match recallNodeResult (NodeId.value case.Body) z with
                            | Some (ssa, _) -> Some ssa
                            | None -> None
                        (idx, ops, resultSSA))

                let resultType =
                    let mlirType = mapType node.Type
                    if mlirType = TUnit then None else Some mlirType

                let ops, result = CFWitness.witnessMatch node.Id z scrutineeSSA scrutineeType caseOps resultType
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // Other node kinds
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.InterpolatedString parts ->
                // Resolve ExprPart NodeIds to their already-witnessed Val results
                let resolveExpr exprNodeId =
                    recallNodeResult (NodeId.value exprNodeId) z
                    |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })

                let ops, result = LitWitness.witnessInterpolated node.Id z parts resolveExpr
                emitAll ops z
                z, result

            | SemanticKind.IndexGet (collectionId, indexId) ->
                match resolveNodeToVal collectionId z, resolveNodeToVal indexId z with
                | Some collVal, Some indexVal ->
                    // Determine element type from collection type
                    let elemType =
                        match collVal.Type with
                        | TStruct [TPtr; _] -> TPtr  // Fat pointer - element type unknown, assume ptr
                        | _ -> TPtr
                    let ops, result = MemWitness.witnessIndexGet node.Id z collVal.SSA collVal.Type indexVal.SSA elemType
                    emitAll ops z
                    z, result
                | _ ->
                    z, TRError "IndexGet: collection or index not computed"

            | SemanticKind.IndexSet (collectionId, indexId, valueId) ->
                match resolveNodeToVal collectionId z, resolveNodeToVal indexId z, resolveNodeToVal valueId z with
                | Some collVal, Some indexVal, Some valV ->
                    let ops, result = MemWitness.witnessIndexSet node.Id z collVal.SSA indexVal.SSA valV.SSA valV.Type
                    emitAll ops z
                    z, result
                | _ ->
                    z, TRError "IndexSet: collection, index, or value not computed"

            | SemanticKind.AddressOf (exprId, isMutable) ->
                match resolveNodeToVal exprId z with
                | Some exprVal ->
                    let ops, result = MemWitness.witnessAddressOf node.Id z exprVal.SSA exprVal.Type isMutable
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError "AddressOf: expr not computed"

            | SemanticKind.TupleExpr elementIds ->
                let elements =
                    elementIds
                    |> List.choose (fun id -> resolveNodeToVal id z)
                let ops, result = MemWitness.witnessTupleExpr node.Id z elements
                emitAll ops z
                z, result

            | SemanticKind.TraitCall (memberName, _typeArgs, argId) ->
                match resolveNodeToVal argId z with
                | Some receiverVal ->
                    let memberType = mapType node.Type
                    let ops, result = MemWitness.witnessTraitCall node.Id z receiverVal.SSA receiverVal.Type memberName memberType
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError "TraitCall: receiver not computed"

            | SemanticKind.ArrayExpr elementIds ->
                let elements =
                    elementIds
                    |> List.choose (fun id -> resolveNodeToVal id z)
                let elemType =
                    match elements with
                    | first :: _ -> first.Type
                    | [] -> TPtr
                let ops, result = MemWitness.witnessArrayExpr node.Id z elements elemType
                emitAll ops z
                z, result

            | SemanticKind.ListExpr elementIds ->
                let elements =
                    elementIds
                    |> List.choose (fun id -> resolveNodeToVal id z)
                let elemType =
                    match elements with
                    | first :: _ -> first.Type
                    | [] -> TPtr
                let ops, result = MemWitness.witnessListExpr node.Id z elements elemType
                emitAll ops z
                z, result

            | SemanticKind.FieldGet (exprId, fieldName) ->
                match resolveNodeToVal exprId z with
                | Some exprVal ->
                    // Resolve field index and type from field name
                    // For strings (NativeStr = {ptr, i64}):
                    //   Pointer → index 0, type ptr
                    //   Length → index 1, type i64 (NOT F# int!)
                    // For general records: lookup from type definition
                    let fieldIndex, fieldType =
                        match exprVal.Type, fieldName with
                        | TStruct [TPtr; TInt I64], "Pointer" -> 0, MLIRTypes.ptr
                        | TStruct [TPtr; TInt I64], "Length" -> 1, MLIRTypes.i64
                        | _, "Pointer" -> 0, MLIRTypes.ptr
                        | _, "Length" -> 1, mapType node.Type
                        | _ -> 0, mapType node.Type
                    let ops, result = MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type fieldIndex fieldType
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError (sprintf "FieldGet '%s': expr not computed" fieldName)

            | SemanticKind.FieldSet (exprId, fieldName, valueId) ->
                match resolveNodeToVal exprId z, resolveNodeToVal valueId z with
                | Some exprVal, Some valV ->
                    // Resolve field index from field name
                    let fieldIndex =
                        match fieldName with
                        | "Pointer" -> 0
                        | "Length" -> 1
                        | _ -> 0  // TODO: Look up from type definition
                    let ops, result = MemWitness.witnessFieldSet node.Id z exprVal.SSA exprVal.Type fieldIndex valV.SSA
                    emitAll ops z
                    z, result
                | _ ->
                    z, TRError (sprintf "FieldSet '%s': expr or value not computed" fieldName)

            | SemanticKind.PatternBinding name ->
                match Map.tryFind name z.State.VarBindings with
                | Some (ssa, ty) ->
                    z, TRValue { SSA = ssa; Type = ty }
                | None ->
                    z, TRError (sprintf "PatternBinding '%s' not found" name)

            | SemanticKind.UnionCase (_caseName, caseIndex, payloadOpt) ->
                let payload =
                    match payloadOpt with
                    | Some payloadId -> resolveNodeToVal payloadId z
                    | None -> None
                let unionType = mapType node.Type
                let ops, result = MemWitness.witnessUnionCase node.Id z caseIndex payload unionType
                emitAll ops z
                z, result

            | SemanticKind.Error msg ->
                z, TRError msg

            | kind ->
                z, TRError (sprintf "SemanticKind not implemented: %A" kind)

        // Bind result to node for future reference
        match result with
        | TRValue { SSA = ssa; Type = ty } ->
            bindNodeResult nodeIdVal ssa ty z'
        | TRError msg when collectErrors ->
            errors <- msg :: errors
            z'
        | _ ->
            z'

    // ═══════════════════════════════════════════════════════════════════════
    // CREATE ZIPPER AND RUN TRAVERSAL
    // ═══════════════════════════════════════════════════════════════════════
    match fromEntryPoint graph ssaAssignment analysisResult stringTable platformResolution entryPointLambdaIds with
    | None ->
        "", ["No entry point found in graph"]

    | Some initialZipper ->
        // Run traversal with SCF hooks
        let finalZipper =
            Traversal.foldWithSCFRegions
                LambdaWitness.preBindParams
                (Some scfHook)
                witnessNode
                initialZipper
                graph

        // ═══════════════════════════════════════════════════════════════════
        // SERIALIZE OUTPUT
        // ═══════════════════════════════════════════════════════════════════

        // Collect string constants as globals
        // Name must match GString serialization: @str_<hash>
        let stringOps =
            getStrings finalZipper
            |> List.map (fun (hash, content, len) ->
                let globalName = sprintf "str_%u" hash
                MLIROp.LLVMOp (LLVMOp.GlobalString (globalName, content, len)))

        // Get top-level ops (functions)
        let topLevelOps = getTopLevelOps finalZipper

        // Get _start wrapper if freestanding mode
        let startWrapperOps =
            match getStartWrapperOps finalZipper with
            | Some ops -> ops
            | None -> []

        // Combine and serialize: strings, _start (if any), then user functions
        let allOps = stringOps @ startWrapperOps @ (List.rev topLevelOps)
        let body = opsToString allOps
        let mlir = sprintf "module {\n%s}\n" body

        mlir, List.rev errors

/// Transfer a SemanticGraph to MLIR text
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    fst (transferGraphCore graph isFreestanding false None)

/// Transfer with diagnostics
let transferGraphWithDiagnostics (graph: SemanticGraph) (isFreestanding: bool) (intermediatesDir: string option) : string * string list =
    transferGraphCore graph isFreestanding true intermediatesDir
