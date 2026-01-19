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
open FSharp.Native.Compiler.PSG.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.PSGZipper
open Alex.Bindings.BindingTypes
open Alex.Bindings.PlatformTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns
open PSGElaboration.PlatformConfig
open Core.FNCS.Integration

// Witness modules
module LitWitness = Alex.Witnesses.LiteralWitness
module PreprocessingSerializer = PSGElaboration.PreprocessingSerializer
module BindWitness = Alex.Witnesses.BindingWitness
module AppWitness = Alex.Witnesses.Application.Witness
module CFWitness = Alex.Witnesses.ControlFlowWitness
module MemWitness = Alex.Witnesses.MemoryWitness
module LambdaWitness = Alex.Witnesses.LambdaWitness
module LazyWitness = Alex.Witnesses.LazyWitness
module SeqWitness = Alex.Witnesses.SeqWitness

// Preprocessing modules (coeffects)
module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PlatformResolution = PSGElaboration.PlatformBindingResolution
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices



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

    // Platform resolution FIRST - architecture is needed by SSA assignment
    let hostPlatform = TargetPlatform.detectHost()
    let runtimeMode = if isFreestanding then Freestanding else Console
    let platformResolution = PlatformResolution.analyze graph runtimeMode hostPlatform.OS hostPlatform.Arch

    let entryPointLambdaIds = MutAnalysis.findEntryPointLambdaIds graph
    let analysisResult = MutAnalysis.analyze graph
    let ssaAssignment = SSAAssign.assignSSA hostPlatform.Arch graph
    let stringTable = StringCollect.collect graph  // Pre-collect all strings

    // Pattern binding analysis (coeffect for match case bindings)
    let patternBindingAnalysis = PatternAnalysis.analyze graph

    // Yield state indices (coeffect for seq expression state machines)
    let yieldStateCoeffect = YieldStateIndices.run graph

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

    // ═══════════════════════════════════════════════════════════════════════════
    // LOCAL STATE for region ops (NOT in zipper)
    // ═══════════════════════════════════════════════════════════════════════
    let regionOps = Dictionary<int * int, MLIROp list>()
    let mutable errors: string list = []
    
    // Global visited set for the entire graph traversal to handle DAGs correctly
    let visited = System.Collections.Generic.HashSet<int>()
    let emittedFunctions = System.Collections.Generic.HashSet<string>()

    // ═══════════════════════════════════════════════════════════════════════════
    // SCF REGION HOOK (closes over regionOps, graph, patternBindingAnalysis)

    /// Bind pattern variables for a match case before body traversal
    /// FOUR PILLARS: This is "wiring-up" - connecting coeffects to VarBindings
    /// - Coeffects: PatternBindingAnalysis, SSAAssignment (pre-computed, immutable)
    /// - Zipper: bindVarSSA adds to VarBindings (observation/witnessing)
    /// - Witnesses: MemWitness.witnessPayloadExtract handles MLIR emission
    /// - Active Patterns: Match on SemanticKind (semantic meaning)
    let bindMatchCasePatterns (z: PSGZipper) (matchNodeId: NodeId) (caseIdx: int) : PSGZipper =
        // COEFFECT LOOKUP: Get Match node structure
        match SemanticGraph.tryGetNode matchNodeId graph with
        | Some matchNode ->
            match matchNode.Kind with
            | SemanticKind.Match (scrutineeId, cases) when caseIdx < List.length cases ->
                let case = cases.[caseIdx]

                // COEFFECT LOOKUP: Get scrutinee SSA (already in NodeBindings from prior traversal)
                match recallNodeResult (NodeId.value scrutineeId) z with
                | Some (scrutineeSSA, scrutineeType) ->
                    // PHOTOGRAPHER PRINCIPLE: Walk pattern structure to find each binding's extraction path
                    // For tuple patterns: need to extract tuple element FIRST, then DU payload
                    let mutable z' = z

                    // Collect extraction info: (pbNodeId, tupleIndex option, duSSA, duType)
                    // For simple DU patterns: tupleIndex = None, use scrutinee directly
                    // For tuple patterns: tupleIndex = Some i, extract tuple[i] first
                    let rec collectBindingsWithPath (pattern: Pattern) (tupleIdx: int option) : (NodeId * int option) list =
                        match pattern with
                        | Pattern.Tuple elements ->
                            // Tuple pattern: each element has a tuple index
                            elements
                            |> List.mapi (fun i elem -> collectBindingsWithPath elem (Some i))
                            |> List.concat
                        | Pattern.Union (_, _, Some (Pattern.Tuple [Pattern.Var _]), _) ->
                            // DU with single payload - find corresponding PatternBinding
                            case.PatternBindings
                            |> List.tryItem (match tupleIdx with Some i -> i | None -> 0)
                            |> Option.map (fun pbId -> [(pbId, tupleIdx)])
                            |> Option.defaultValue []
                        | Pattern.Union (_, _, Some (Pattern.Tuple payloads), _) ->
                            // DU with multiple payloads - currently just take first var
                            payloads
                            |> List.tryPick (function Pattern.Var _ -> Some tupleIdx | _ -> None)
                            |> Option.map (fun ti ->
                                case.PatternBindings
                                |> List.tryItem (match ti with Some i -> i | None -> 0)
                                |> Option.map (fun pbId -> [(pbId, ti)])
                                |> Option.defaultValue [])
                            |> Option.defaultValue []
                        | Pattern.Union (_, _, Some payload, _) ->
                            collectBindingsWithPath payload tupleIdx
                        | Pattern.Union (_, _, None, _) -> []
                        | Pattern.Var _ -> []
                        | Pattern.Wildcard -> []
                        | _ -> []

                    // For simple patterns, just iterate pattern bindings with no tuple index
                    let isTuplePattern = match case.Pattern with Pattern.Tuple _ -> true | _ -> false

                    if isTuplePattern then
                        // Tuple pattern: extract each element, then extract payload
                        match case.Pattern with
                        | Pattern.Tuple elements ->
                            for i, elem in List.indexed elements do
                                // For each tuple element, find bindings and extract
                                let rec processElement (pattern: Pattern) (elemIdx: int) =
                                    match pattern with
                                    | Pattern.Union (_, _, Some (Pattern.Tuple [Pattern.Var (name, _)]), _) ->
                                        // Find the PatternBinding node for this name
                                        case.PatternBindings
                                        |> List.tryFind (fun pbId ->
                                            match SemanticGraph.tryGetNode pbId graph with
                                            | Some n -> match n.Kind with SemanticKind.PatternBinding n' -> n' = name | _ -> false
                                            | None -> false)
                                        |> Option.iter (fun pbNodeId ->
                                            match SemanticGraph.tryGetNode pbNodeId graph with
                                            | Some pbNode ->
                                                match SSAAssign.lookupSSAs pbNodeId ssaAssignment with
                                                | Some ssas when not (List.isEmpty ssas) ->
                                                    let patternType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph pbNode.Type
                                                    // WITNESS: Extract tuple element, then DU payload
                                                    let ops, resultVal = MemWitness.witnessTuplePatternExtract ssas scrutineeSSA scrutineeType elemIdx patternType
                                                    for op in ops do emit op z'
                                                    z' <- bindVarSSA name resultVal.SSA resultVal.Type z'
                                                | _ -> ()
                                            | None -> ())
                                    | Pattern.Var (name, _) ->
                                        // Simple variable binding in tuple
                                        case.PatternBindings
                                        |> List.tryFind (fun pbId ->
                                            match SemanticGraph.tryGetNode pbId graph with
                                            | Some n -> match n.Kind with SemanticKind.PatternBinding n' -> n' = name | _ -> false
                                            | None -> false)
                                        |> Option.iter (fun pbNodeId ->
                                            match SemanticGraph.tryGetNode pbNodeId graph with
                                            | Some pbNode ->
                                                match SSAAssign.lookupSSAs pbNodeId ssaAssignment with
                                                | Some ssas when not (List.isEmpty ssas) ->
                                                    let patternType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph pbNode.Type
                                                    // WITNESS: Extract tuple element directly
                                                    // We can reuse witnessTuplePatternExtract but need to know if it handles simple var?
                                                    // witnessTuplePatternExtract extracts elem, then does payload extraction.
                                                    // If we just want the element, we should emit extractvalue directly.
                                                    
                                                    // Manual extract logic here to be safe and simple
                                                    // Extract element at elemIdx from scrutinee tuple
                                                    let ssa = ssas.[0] // Use first SSA for result
                                                    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (ssa, scrutineeSSA, [elemIdx], scrutineeType))
                                                    emit extractOp z'
                                                    z' <- bindVarSSA name ssa patternType z'
                                                | _ -> ()
                                            | None -> ())
                                    | Pattern.Union (_, _, Some payload, _) ->
                                        processElement payload elemIdx
                                    | _ -> ()
                                processElement elem i
                        | _ -> ()
                    else
                        // Simple DU pattern: extract payload directly
                        // Get union case index from pattern (for multi-payload DUs like Result)
                        let unionCaseIndex =
                            match case.Pattern with
                            | Pattern.Union (_, tagIndex, _, _) -> tagIndex
                            | _ -> 0  // Default for non-union patterns
                        for pbNodeId in case.PatternBindings do
                            match SemanticGraph.tryGetNode pbNodeId graph with
                            | Some pbNode ->
                                match pbNode.Kind with
                                | SemanticKind.PatternBinding name ->
                                    match SSAAssign.lookupSSAs pbNodeId ssaAssignment with
                                    | Some ssas when not (List.isEmpty ssas) ->
                                        let patternType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph pbNode.Type
                                        let ops, resultVal = MemWitness.witnessPayloadExtract ssas scrutineeSSA scrutineeType patternType unionCaseIndex
                                        for op in ops do emit op z'
                                        z' <- bindVarSSA name resultVal.SSA resultVal.Type z'
                                    | _ -> ()
                                | _ -> ()
                            | None -> ()
                    z'
                | None ->
                    // Scrutinee not yet in NodeBindings - shouldn't happen in correct traversal
                    z
            | _ -> z
        | None -> z

    let scfHook: SCFRegionHook<PSGZipper> = {
        BeforeRegion = fun z parentId kind ->
            match kind with
            | RegionKind.MatchCaseRegion idx ->
                enterRegion z
                bindMatchCasePatterns z parentId idx
            | _ ->
                // Unified stack management: Always enter region to isolate op collection.
                // enterFunction (from preBindParams) handles scope/bindings.
                enterRegion z
                z

        AfterRegion = fun z parentId kind ->
            // Unified stack management: Always exit region to retrieve collected ops.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // MUTUAL RECURSION: foldSubtree AND witnessNode
    // ═══════════════════════════════════════════════════════════════════════════

    let rec foldSubtree 
        (nodeId: NodeId)
        (z: PSGZipper) 
        : PSGZipper =
        
        let rec walk state nodeId =
            let nodeIdVal = NodeId.value nodeId
            if visited.Contains(nodeIdVal) then
                state
            else
                visited.Add(nodeIdVal) |> ignore
                match SemanticGraph.tryGetNode nodeId graph with
                | None -> state
                | Some node ->
                    // 1. Follow semantic dependencies (VarRef definitions)
                    let state =
                        match node.Kind with
                        | SemanticKind.VarRef (_, Some defId) ->
                            walk state defId
                        | _ -> state

                    // 2. Pre-bind Lambda parameters and ForEach loop variables
                    let state =
                        match node.Kind with
                        | SemanticKind.Lambda _ -> LambdaWitness.preBindParams state node
                        | SemanticKind.ForEach (varName, collectionId, _bodyId) ->
                            // Pre-bind the loop variable to currentSSA before walking the body
                            // currentSSA is at index 6 in ForEach's SSA allocation
                            match SSAAssign.lookupSSAs node.Id ssaAssignment with
                            | Some ssas when ssas.Length >= 7 ->
                                let currentSSA = ssas.[6]
                                // Get element type from collection node's type
                                let elemType =
                                    match SemanticGraph.tryGetNode collectionId graph with
                                    | Some collNode ->
                                        match collNode.Type with
                                        | NativeType.TSeq elemNativeType ->
                                            mapNativeTypeWithGraphForArch state.State.Platform.TargetArch graph elemNativeType
                                        | _ -> MLIRTypes.i64
                                    | None -> MLIRTypes.i64
                                // Bind the loop variable to currentSSA
                                bindVarSSA varName currentSSA elemType state
                            | _ -> state
                        | _ -> state

                    // 3. Process children with SCF hooks
                    let state =
                        match node.Kind with
                        | SemanticKind.WhileLoop (guardId, bodyId) ->
                            let parentId = node.Id
                            let state = scfHook.BeforeRegion state parentId GuardRegion
                            let state = walk state guardId
                            let state = scfHook.AfterRegion state parentId GuardRegion
                            let state = scfHook.BeforeRegion state parentId BodyRegion
                            let state = walk state bodyId
                            let state = scfHook.AfterRegion state parentId BodyRegion
                            state

                        | SemanticKind.ForLoop (_, startId, endId, _, bodyId) ->
                            let parentId = node.Id
                            let state = scfHook.BeforeRegion state parentId StartExprRegion
                            let state = walk state startId
                            let state = scfHook.AfterRegion state parentId StartExprRegion
                            let state = scfHook.BeforeRegion state parentId EndExprRegion
                            let state = walk state endId
                            let state = scfHook.AfterRegion state parentId EndExprRegion
                            let state = scfHook.BeforeRegion state parentId BodyRegion
                            let state = walk state bodyId
                            let state = scfHook.AfterRegion state parentId BodyRegion
                            state

                        | SemanticKind.IfThenElse (guardId, thenId, elseIdOpt) ->
                            let parentId = node.Id
                            let state = walk state guardId
                            let state = scfHook.BeforeRegion state parentId ThenRegion
                            let state = walk state thenId
                            let state = scfHook.AfterRegion state parentId ThenRegion
                            match elseIdOpt with
                            | Some elseId ->
                                let state = scfHook.BeforeRegion state parentId ElseRegion
                                let state = walk state elseId
                                scfHook.AfterRegion state parentId ElseRegion
                            | None -> state

                        | SemanticKind.Match (scrutineeId, cases) ->
                            let parentId = node.Id
                            let state = walk state scrutineeId
                            cases
                            |> List.fold (fun (state, idx) case ->
                                let state = scfHook.BeforeRegion state parentId (MatchCaseRegion idx)
                                let state = case.PatternBindings |> List.fold walk state
                                let state =
                                    match case.Guard with
                                    | Some guardId -> walk state guardId
                                    | None -> state
                                let state = walk state case.Body
                                let state = scfHook.AfterRegion state parentId (MatchCaseRegion idx)
                                (state, idx + 1)
                            ) (state, 0)
                            |> fst

                        // Lambda: No special handling needed - body has SeparateFunction flag.
                        // Default children traversal walks params; body is skipped via EmissionStrategy check.
                        // The LambdaWitness handles body traversal during witnessNode.

                        | SemanticKind.ForEach (_varName, collectionId, bodyId) ->
                            // ForEach: Walk collection first (outside region), then body as region
                            // Collection produces the seq value, body consumes current element
                            let parentId = node.Id
                            let state = walk state collectionId
                            let state = scfHook.BeforeRegion state parentId BodyRegion
                            let state = walk state bodyId
                            let state = scfHook.AfterRegion state parentId BodyRegion
                            state

                        // SeqExpr: No special handling needed - body has SeparateFunction flag.
                        // Default children traversal walks nothing (SeqExpr has Lambda as child, not body directly).
                        // The SeqWitness handles MoveNext generation during witnessNode.

                        | _ ->
                            let semanticRefs = Reachability.getSemanticReferences node
                            let state = semanticRefs |> List.fold walk state
                            // Architectural fix (January 2026): Filter out nodes with special emission strategies.
                            // - SeparateFunction: Lambda/SeqExpr bodies - parent witness handles them
                            // - MainPrologue: Module-level value bindings - main's Lambda handles them
                            let childrenToWalk =
                                node.Children
                                |> List.filter (fun childId ->
                                    match SemanticGraph.tryGetNode childId graph with
                                    | Some childNode ->
                                        match childNode.EmissionStrategy with
                                        | EmissionStrategy.SeparateFunction _ -> false
                                        | EmissionStrategy.MainPrologue -> false
                                        | EmissionStrategy.Inline -> true
                                    | None -> true)
                            childrenToWalk |> List.fold walk state

                    // 4. Post-order witness
                    witnessNode state node
        
        walk z nodeId

    and witnessNode (z: PSGZipper) (node: SemanticNode) : PSGZipper =
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
            | SemanticKind.RecordExpr (fields, copyFrom) ->
                // fields is (string * NodeId) list - resolve each field value
                let fieldVals =
                    fields
                    |> List.choose (fun (name, valueId) ->
                        match resolveNodeToVal valueId z with
                        | Some v -> Some (name, v)
                        | None -> None)
                // Use graph-aware type mapping for records (looks up field types from TypeDef)
                let recordType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch z.Graph node.Type

                // Get record field definitions from graph
                let recordFieldDefs =
                    match node.Type with
                    | NativeType.TApp(tycon, _) when tycon.FieldCount > 0 ->
                        SemanticGraph.tryGetRecordFields tycon.Name graph
                    | _ -> None

                match copyFrom, recordFieldDefs with
                | Some origId, Some fieldDefs ->
                    // Copy-and-update: { orig with field1 = val1; field2 = val2 }
                    // Start from original record and update only the specified fields
                    match resolveNodeToVal origId z with
                    | Some origVal ->
                        // Use specialized witness for copy-and-update
                        // It will insert updated fields at their correct indices into the original
                        let ops, result = MemWitness.witnessRecordCopyUpdate node.Id z origVal fieldDefs fieldVals recordType
                        emitAll ops z
                        z, result
                    | None ->
                        z, TRError "RecordExpr: copyFrom source not computed"
                | _, Some fieldDefs ->
                    // New record construction - use field definitions for correct indices
                    // Build map of provided fields
                    let fieldMap = fieldVals |> Map.ofList
                    // Build values in definition order
                    let orderedFieldVals =
                        fieldDefs |> List.choose (fun (fieldName, _) ->
                            Map.tryFind fieldName fieldMap
                            |> Option.map (fun v -> (fieldName, v)))
                    let ops, result = MemWitness.witnessRecordExpr node.Id z orderedFieldVals recordType
                    emitAll ops z
                    z, result
                | _, None ->
                    // Non-record type or unknown type - fallback to original behavior
                    let ops, result = MemWitness.witnessRecordExpr node.Id z fieldVals recordType
                    emitAll ops z
                    z, result

            // ─────────────────────────────────────────────────────────────────
            // Lambdas
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Lambda (params', bodyId, _captures, _, _context) ->
                // Check if this is main (for MainPrologue processing)
                let lambdaName =
                    SSAAssign.lookupLambdaName node.Id ssaAssignment
                    |> Option.defaultValue ""
                let isMain = (lambdaName = "main")
                
                // Find MainPrologue siblings (module-level value bindings)
                // These need to be processed in main's scope before the body
                let mainPrologueSiblings =
                    if isMain then
                        // Find parent ModuleDef and get its MainPrologue children
                        let siblings =
                            graph.EntryPoints
                            |> List.collect (fun entryId ->
                                match SemanticGraph.tryGetNode entryId graph with
                                | Some entryNode ->
                                    match entryNode.Kind with
                                    | SemanticKind.ModuleDef (_, memberIds) ->
                                        memberIds
                                        |> List.filter (fun memberId ->
                                            match SemanticGraph.tryGetNode memberId graph with
                                            | Some memberNode ->
                                                match memberNode.EmissionStrategy with
                                                | EmissionStrategy.MainPrologue -> true
                                                | _ -> false
                                            | None -> false)
                                    | _ -> []
                                | None -> [])
                        printfn "[DEBUG] MainPrologue siblings for main: %A" (siblings |> List.map NodeId.value)
                        siblings
                    else []
                
                // RECURSIVE WITNESS STRATEGY:
                // We define a callback that traverses the body subtree and captures the output.
                // For main, also process MainPrologue siblings first.
                
                let witnessBody (zFunc: PSGZipper) : MLIROp list * TransferResult =
                    // zFunc is already in the function scope (enterFunction called).
                    
                    // For main: process MainPrologue bindings FIRST (module-level value bindings)
                    // Their ops go to CurrentOps and become part of main's prologue
                    let zAfterPrologue =
                        mainPrologueSiblings
                        |> List.fold (fun z prologueId -> foldSubtree prologueId z) zFunc
                    
                    // Now traverse the body subtree
                    let zAfter = foldSubtree bodyId zAfterPrologue
                    
                    // Retrieve the accumulated ops
                    let ops = List.rev zAfter.State.CurrentOps
                    zAfter.State.CurrentOps <- []
                    
                    // Retrieve the result
                    let result = 
                        match resolveNodeToVal bodyId zAfter with
                        | Some v -> TRValue v
                        | None -> TRVoid
                        
                    ops, result

                // Call witness with the callback
                let funcDefOpt, localOps, result = LambdaWitness.witness params' bodyId node witnessBody z

                // ARCHITECTURAL FIX: Exit function scope to restore parent's CurrentOps FIRST
                let z' = exitFunction z

                // Add function definition to top-level (if present and not already emitted)
                match funcDefOpt with
                | Some funcDef -> 
                    // Use active pattern or manual extraction to get the function name
                    let funcNameOpt =
                        match funcDef with
                        | MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> Some name
                        | MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (name, _, _, _, _)) -> Some name
                        | _ -> None
                    
                    match funcNameOpt with
                    | Some name ->
                        if not (emittedFunctions.Contains(name)) then
                            emittedFunctions.Add(name) |> ignore
                            emitTopLevel funcDef z'
                    | None -> ()
                | None -> ()

                // Now emit local ops (closure construction) to parent scope
                emitAll localOps z'

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
            // Mutable set - dispatch to witness
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.Set (targetId, valueId) ->
                match SemanticGraph.tryGetNode targetId graph with
                | Some targetNode ->
                    match targetNode.Kind with
                    | SemanticKind.VarRef (name, defIdOpt) ->
                        // VarRef target: dispatch to BindingWitness.witnessSet
                        match recallNodeResult (NodeId.value valueId) z with
                        | Some (valueSSA, valueType) ->
                            let witnessCtx: BindWitness.WitnessContext = {
                                SSA = ssaAssignment
                                Mutability = analysisResult
                                Graph = graph
                            }
                            let ops, result = BindWitness.witnessSet witnessCtx z node.Id name defIdOpt valueSSA valueType
                            emitAll ops z
                            z, result
                        | None ->
                            z, TRError (sprintf "Set: value not computed for '%s'" name)
                    | _ ->
                        // Non-VarRef target (e.g., pointer dereference): direct store
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
                    let mlirType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
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

                // Build cases with Pattern + collected region ops + guard SSA
                // The witness will inspect Pattern to determine match strategy
                // (Four Pillars: Pattern IS the classification - witness derives strategy)
                let caseOps =
                    cases
                    |> List.mapi (fun idx case ->
                        let caseKey = (nodeIdVal, regionKindToInt (RegionKind.MatchCaseRegion idx))
                        let ops = regionOps.GetValueOrDefault(caseKey, [])
                        let resultSSA =
                            match recallNodeResult (NodeId.value case.Body) z with
                            | Some (ssa, _) -> Some ssa
                            | None -> None
                        // Look up guard result SSA (if guard exists)
                        let guardSSA =
                            case.Guard
                            |> Option.bind (fun guardId ->
                                recallNodeResult (NodeId.value guardId) z
                                |> Option.map fst)
                        // Pass Pattern + guard SSA - witness derives strategy from it
                        (case.Pattern, guardSSA, ops, resultSSA))

                let resultType =
                    let mlirType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
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
                // For AddressOf of mutable bindings, we need the alloca pointer, NOT the loaded value.
                // The VarRef witness loads values from allocas, but AddressOf needs the address itself.
                let exprNode = SemanticGraph.tryGetNode exprId graph
                let isTargetAddressedMutable =
                    match exprNode with
                    | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
                        EmissionState.isAddressedMutable (NodeId.value defId) z.State
                    | _ -> false

                if isTargetAddressedMutable then
                    // Get the alloca pointer directly from the definition's SSA assignment
                    match exprNode with
                    | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
                        match SSAAssign.lookupSSAs defId ssaAssignment with
                        | Some ssas when ssas.Length >= 2 ->
                            // For addressed mutables: [oneConst, allocaPtr]
                            let allocaSSA = ssas.[1]
                            let z' = bindNodeResult nodeIdVal allocaSSA TPtr z
                            z', TRValue { SSA = allocaSSA; Type = TPtr }
                        | _ ->
                            z, TRError "AddressOf: mutable binding has no alloca SSA"
                    | _ ->
                        z, TRError "AddressOf: expected VarRef for mutable target"
                else
                    // Non-mutable or non-VarRef: use standard path
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
                    let memberType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
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
                    // Resolve field index(es) and type from field name
                    // For strings (NativeStr = {ptr, i64}):
                    //   Pointer → index 0, type ptr
                    //   Length → index 1, type i64 (NOT F# int!)
                    // For records: lookup from TypeDef in graph
                    // For nested access (e.g., "Person.Name"): compute index path

                    // Helper to look up a single field index and type from a record type
                    let lookupFieldIndex (typeName: string) (fldName: string) : (int * NativeType) option =
                        match SemanticGraph.tryGetRecordFields typeName graph with
                        | Some fields ->
                            fields
                            |> List.tryFindIndex (fun (name, _) -> name = fldName)
                            |> Option.map (fun idx -> idx, snd fields.[idx])
                        | None -> None

                    // Get the record type name from exprNode
                    let exprTypeName =
                        match SemanticGraph.tryGetNode exprId graph with
                        | Some exprNode ->
                            match exprNode.Type with
                            | NativeType.TApp(tycon, _) when tycon.FieldCount > 0 -> Some tycon.Name
                            | _ -> None
                        | None -> None

                    // Handle field access (simple or nested)
                    let ops, result =
                        match exprVal.Type, fieldName with
                        | TStruct [TPtr; TInt I64], "Pointer" ->
                            MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type 0 MLIRTypes.ptr
                        | TStruct [TPtr; TInt I64], "Length" ->
                            MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type 1 MLIRTypes.i64
                        | _, "Pointer" ->
                            MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type 0 MLIRTypes.ptr
                        | _, "Length" ->
                            MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type 1 (mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type)
                        | _ when fieldName.Contains(".") ->
                            // Nested field access - parse path and compute indices
                            let pathParts = fieldName.Split('.') |> Array.toList
                            match exprTypeName with
                            | Some startTypeName ->
                                // Walk through the path to compute indices
                                let rec computeIndices (currentTypeName: string) (parts: string list) (acc: int list) : (int list * MLIRType) option =
                                    match parts with
                                    | [] -> None  // Empty path - shouldn't happen
                                    | [lastField] ->
                                        // Last field in path - get index and final type
                                        match lookupFieldIndex currentTypeName lastField with
                                        | Some (idx, fieldTy) ->
                                            Some (List.rev (idx :: acc), mapNativeTypeForArch z.State.Platform.TargetArch fieldTy)
                                        | None -> None
                                    | field :: rest ->
                                        // Intermediate field - get index and continue
                                        match lookupFieldIndex currentTypeName field with
                                        | Some (idx, fieldTy) ->
                                            // Get the next type name from fieldTy
                                            match fieldTy with
                                            | NativeType.TApp(nextTycon, _) when nextTycon.FieldCount > 0 ->
                                                computeIndices nextTycon.Name rest (idx :: acc)
                                            | _ ->
                                                None  // Intermediate field is not a record
                                        | None -> None
                                match computeIndices startTypeName pathParts [] with
                                | Some (indices, fieldType) ->
                                    MemWitness.witnessNestedFieldGet node.Id z exprVal.SSA exprVal.Type indices fieldType
                                | None ->
                                    [], TRError (sprintf "FieldGet: cannot resolve nested path '%s' from type '%s'" fieldName startTypeName)
                            | None ->
                                [], TRError (sprintf "FieldGet: nested path '%s' but expr is not a record type" fieldName)
                        | _ ->
                            // Simple field access on a record
                            match exprTypeName with
                            | Some typeName ->
                                match lookupFieldIndex typeName fieldName with
                                | Some (idx, fieldTy) ->
                                    MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type idx (mapNativeTypeForArch z.State.Platform.TargetArch fieldTy)
                                | None ->
                                    [], TRError (sprintf "FieldGet: field '%s' not found in record type '%s'" fieldName typeName)
                            | None ->
                                // Not a record type - use node.Type (result type) for field type
                                MemWitness.witnessFieldGet node.Id z exprVal.SSA exprVal.Type 0 (mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type)
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError (sprintf "FieldGet '%s': expr not computed" fieldName)

            | SemanticKind.FieldSet (exprId, fieldName, valueId) ->
                match resolveNodeToVal exprId z, resolveNodeToVal valueId z with
                | Some exprVal, Some valV ->
                    // Resolve field index from field name
                    // For strings: Pointer → 0, Length → 1
                    // For records: look up from TypeDef in graph
                    // Note: Nested FieldSet (e.g., "Person.Name") is not supported

                    // Helper to look up a single field index from a record type
                    let lookupFieldIndex (typeName: string) (fldName: string) : int option =
                        match SemanticGraph.tryGetRecordFields typeName graph with
                        | Some fields ->
                            fields |> List.tryFindIndex (fun (name, _) -> name = fldName)
                        | None -> None

                    let fieldIndex =
                        if fieldName.Contains(".") then
                            failwithf "FieldSet: nested path '%s' not supported - use intermediate bindings" fieldName
                        else
                            match fieldName with
                            | "Pointer" -> 0
                            | "Length" -> 1
                            | _ ->
                                // Look up field index from record's NativeType via graph
                                match SemanticGraph.tryGetNode exprId graph with
                                | Some exprNode ->
                                    match exprNode.Type with
                                    | NativeType.TApp(tycon, _) when tycon.FieldCount > 0 ->
                                        // Record type - look up field index
                                        match lookupFieldIndex tycon.Name fieldName with
                                        | Some idx -> idx
                                        | None ->
                                            failwithf "FieldSet: field '%s' not found in record type '%s'" fieldName tycon.Name
                                    | _ -> 0  // Not a record type - default
                                | None -> 0  // Node not found - fallback
                    let ops, result = MemWitness.witnessFieldSet node.Id z exprVal.SSA exprVal.Type fieldIndex valV.SSA
                    emitAll ops z
                    z, result
                | _ ->
                    z, TRError (sprintf "FieldSet '%s': expr or value not computed" fieldName)

            | SemanticKind.PatternBinding name ->
                // PatternBindings can be:
                // 1. Lambda parameters - already bound by preBindParams, or discarded
                // 2. Match case bindings - bound during match processing
                // 3. Discarded bindings ('_') - no value needed
                match Map.tryFind name z.State.VarBindings with
                | Some (ssa, ty) ->
                    z, TRValue { SSA = ssa; Type = ty }
                | None when name = "_" || name.StartsWith("_") ->
                    // Discarded binding or synthetic parameter - no value to return
                    z, TRVoid
                | None ->
                    // Check if this is a Lambda parameter (definition, not use)
                    // Lambda parameters are definitions - they don't need lookup
                    // Use focus mode as fallback if Path is not available (e.g. during fold)
                    let isLambdaContext = 
                        match z.State.Focus with 
                        | InFunction _ -> true 
                        | _ -> 
                            match z.Path with
                            | step :: _ when (match step.Parent.Kind with SemanticKind.Lambda _ -> true | _ -> false) -> true
                            | _ -> false
                    
                    if isLambdaContext then
                        z, TRVoid
                    else
                        z, TRError (sprintf "PatternBinding '%s' not found" name)

            | SemanticKind.UnionCase (_caseName, caseIndex, payloadOpt) ->
                let payload =
                    match payloadOpt with
                    | Some payloadId -> resolveNodeToVal payloadId z
                    | None -> None
                let unionType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
                let ops, result = MemWitness.witnessUnionCase node.Id z caseIndex payload unionType
                emitAll ops z
                z, result

            // ─────────────────────────────────────────────────────────────────
            // PRD-14: Lazy values
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.LazyExpr (bodyId, captures) ->
                // bodyId points to the thunk lambda
                // For simple thunks (no captures), Lambda returns TRVoid, so we construct
                // the closure struct {funcPtr, null} ourselves using the thunk's name
                let resultType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
                let elementType =
                    match resultType with
                    | TStruct [_; elemTy; _] -> elemTy  // {i1, T, closure} -> T
                    | _ -> MLIRTypes.i64  // Fallback

                // Get thunk function name
                match SSAAssign.lookupLambdaName bodyId ssaAssignment with
                | Some thunkName ->
                    // Resolve capture values - FLAT CLOSURE: captures are inlined, not in env_ptr
                    let captureVals =
                        captures
                        |> List.choose (fun cap ->
                            match cap.SourceNodeId with
                            | Some srcId -> resolveNodeToVal srcId z
                            | None -> None)
                    
                    let ops, result = LazyWitness.witnessLazyCreate node.Id z thunkName elementType captureVals
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError (sprintf "LazyExpr: thunk lambda name not found for bodyId %d" (NodeId.value bodyId))

            | SemanticKind.LazyForce lazyValueId ->
                match resolveNodeToVal lazyValueId z with
                | Some lazyVal ->
                    // Get element type (T) - this is the result type of force
                    let elementType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph node.Type
                    let ops, result = LazyWitness.witnessLazyForce node.Id z lazyVal elementType
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError "LazyForce: lazy value not computed"

            // ─────────────────────────────────────────────────────────────────
            // PRD-15: Sequence expressions
            // ─────────────────────────────────────────────────────────────────
            | SemanticKind.SeqExpr (_bodyId, captures) ->
                // PRD-15: Generate MoveNext function and seq struct
                // 1. Generate function name from node ID (deterministic)
                let moveNextName = sprintf "seq_%d_moveNext" (NodeId.value node.Id)

                // 2. Get element type from TSeq
                let elementType =
                    match node.Type with
                    | NativeType.TSeq elemType ->
                        mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph elemType
                    | _ -> MLIRTypes.i32  // Fallback for non-TSeq (shouldn't happen)

                // 3. Resolve capture values (for seq struct creation)
                let captureVals =
                    captures
                    |> List.choose (fun cap ->
                        match cap.SourceNodeId with
                        | Some srcId ->
                            match recallNodeResult (NodeId.value srcId) z with
                            | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                            | None -> None
                        | None -> None)
                let captureTypes = captureVals |> List.map (fun v -> v.Type)

                // 4. Get yield info from coeffect
                match YieldStateIndices.tryGetSeqYields node.Id yieldStateCoeffect with
                | Some seqYieldInfo ->
                    // 5. Dispatch based on body structure
                    match seqYieldInfo.BodyKind with
                    | YieldStateIndices.Sequential ->
                        // SEQUENTIAL PATTERN: Each yield is a separate state
                        // Build YieldBlockInfo for each yield
                        let yieldBlocks =
                            seqYieldInfo.Yields
                            |> List.choose (fun yieldInfo ->
                                // Get the yield node and extract the literal value
                                match SemanticGraph.tryGetNode yieldInfo.YieldId graph with
                                | Some yieldNode ->
                                    match yieldNode.Kind with
                                    | SemanticKind.Yield valueId ->
                                        // Get the value node to extract the literal
                                        match SemanticGraph.tryGetNode valueId graph with
                                        | Some valueNode ->
                                            match valueNode.Kind with
                                            | SemanticKind.Literal lit ->
                                                // Convert literal to int64 for IntLiteral
                                                let intValue =
                                                    match lit with
                                                    | LiteralValue.Int32 v -> int64 v
                                                    | LiteralValue.Int64 v -> v
                                                    | LiteralValue.UInt32 v -> int64 v
                                                    | LiteralValue.UInt64 v -> int64 v
                                                    | LiteralValue.UInt8 v -> int64 v
                                                    | LiteralValue.Int8 v -> int64 v
                                                    | LiteralValue.Int16 v -> int64 v
                                                    | LiteralValue.UInt16 v -> int64 v
                                                    | _ -> 0L  // Non-integer literals not supported in PRD-15 SimpleSeq
                                                Some ({
                                                    SeqWitness.StateIndex = yieldInfo.StateIndex
                                                    SeqWitness.ValueSource = SeqWitness.IntLiteral intValue
                                                    SeqWitness.ValueType = elementType
                                                } : SeqWitness.YieldBlockInfo)
                                            | _ ->
                                                // Non-literal yields not supported in sequential pattern
                                                None
                                        | None -> None
                                    | _ -> None
                                | None -> None)

                        if yieldBlocks.Length <> seqYieldInfo.NumYields then
                            // Some yields couldn't be converted (non-literal values)
                            z, TRError "SeqExpr: Sequential pattern only supports literal yields"
                        else
                            let seqStructTy = SeqWitness.seqStructType elementType captureTypes

                            // Generate MoveNext function
                            let moveNextFuncOp = SeqWitness.witnessMoveNext moveNextName seqStructTy elementType yieldBlocks

                            // Add MoveNext to top-level (if not already emitted)
                            if not (emittedFunctions.Contains(moveNextName)) then
                                emittedFunctions.Add(moveNextName) |> ignore
                                emitTopLevel moveNextFuncOp z

                            // Create seq struct
                            let ops, result = SeqWitness.witnessSeqCreate node.Id z moveNextName elementType captureVals
                            emitAll ops z
                            z, result

                    | YieldStateIndices.WhileBased whileInfo ->
                        // WHILE-BASED PATTERN: Two-state model with shared check/yield/done blocks
                        // Internal state fields need to be part of the seq struct
                        let internalStateTypes =
                            seqYieldInfo.InternalState
                            |> List.map (fun field ->
                                mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph field.Type)

                        // Build full seq struct type with internal state
                        let seqStructTy = SeqWitness.seqStructTypeFull elementType captureTypes internalStateTypes

                        // Map capture names to struct indices
                        // Layout: {state, current, moveNext_ptr, cap0, cap1, ..., state0, state1, ...}
                        let numCaptures = List.length captures
                        let captureNameToIndex =
                            captures
                            |> List.mapi (fun i cap -> (cap.Name, 3 + i))
                            |> Map.ofList

                        // Map internal state names to struct indices
                        let internalStateNameToIndex =
                            seqYieldInfo.InternalState
                            |> List.map (fun field -> (field.Name, field.StructIndex))
                            |> Map.ofList

                        // Combined name-to-index for all variables accessible in MoveNext
                        let varNameToIndex =
                            Map.fold (fun acc k v -> Map.add k v acc) captureNameToIndex internalStateNameToIndex

                        // Build WhileBasedMoveNextInfo
                        let whileBasedInfo: SeqWitness.WhileBasedMoveNextInfo = {
                            MoveNextName = moveNextName
                            SeqStructType = seqStructTy
                            ElementType = elementType
                            WhileInfo = whileInfo
                            InternalState = seqYieldInfo.InternalState
                            NumCaptures = numCaptures
                            CaptureTypes = captures |> List.map (fun c -> c.Type)
                        }

                        // Generate MoveNext using while-based generator
                        // For now, we use a simplified approach that generates ops directly
                        // based on the PSG structure analysis

                        // SSA counter for MoveNext function-local SSAs (starting after entry block SSAs)
                        let mutable moveNextSSA = 10  // Start after entry block SSAs

                        // Helper: emit ops to load a variable from struct
                        let loadVar (varName: string) (seqPtrSSA: SSA) (varType: MLIRType) : (MLIROp list * SSA) =
                            match Map.tryFind varName varNameToIndex with
                            | Some fieldIdx ->
                                let ptrSSA = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                let valSSA = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                let ops = [
                                    MLIROp.LLVMOp (LLVMOp.StructGEP (ptrSSA, seqPtrSSA, fieldIdx, seqStructTy))
                                    MLIROp.LLVMOp (LLVMOp.Load (valSSA, ptrSSA, varType, AtomicOrdering.NotAtomic))
                                ]
                                (ops, valSSA)
                            | None ->
                                // Variable not found - emit error-like placeholder
                                let valSSA = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                ([MLIROp.ArithOp (ArithOp.ConstI (valSSA, 0L, varType))], valSSA)

                        // Helper: emit ops to store a value to a variable in struct
                        let storeVar (varName: string) (seqPtrSSA: SSA) (valueSSA: SSA) (varType: MLIRType) : MLIROp list =
                            match Map.tryFind varName varNameToIndex with
                            | Some fieldIdx ->
                                let ptrSSA = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                [
                                    MLIROp.LLVMOp (LLVMOp.StructGEP (ptrSSA, seqPtrSSA, fieldIdx, seqStructTy))
                                    MLIROp.LLVMOp (LLVMOp.Store (valueSSA, ptrSSA, varType, AtomicOrdering.NotAtomic))
                                ]
                            | None -> []

                        // The seqPtr in MoveNext is %0 (first function arg)
                        let seqPtrSSA = V 0

                        // EMIT FUNCTIONS for the while-based state machine

                        // emitInit: Initialize internal state fields
                        // For each internal state field, store its initial value
                        // Initial values come from either captures or constants
                        let emitInit () =
                            seqYieldInfo.InternalState
                            |> List.collect (fun field ->
                                // Find the initialization expression for this field
                                // In `let mutable i = start`, the init value is `start`
                                match SemanticGraph.tryGetNode field.BindingId graph with
                                | Some bindingNode ->
                                    match bindingNode.Children with
                                    | [valueId] ->
                                        match SemanticGraph.tryGetNode valueId graph with
                                        | Some valueNode ->
                                            match valueNode.Kind with
                                            | SemanticKind.VarRef (refName, _) ->
                                                // Init from a capture
                                                let fieldType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph field.Type
                                                let (loadOps, loadedSSA) = loadVar refName seqPtrSSA fieldType
                                                let storeOps = storeVar field.Name seqPtrSSA loadedSSA fieldType
                                                loadOps @ storeOps
                                            | SemanticKind.Literal lit ->
                                                // Init from a constant
                                                let fieldType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph field.Type
                                                let constSSA = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                let intValue =
                                                    match lit with
                                                    | LiteralValue.Int32 v -> int64 v
                                                    | LiteralValue.Int64 v -> v
                                                    | _ -> 0L
                                                let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, fieldType))
                                                let storeOps = storeVar field.Name seqPtrSSA constSSA fieldType
                                                [constOp] @ storeOps
                                            | _ -> []
                                        | None -> []
                                    | _ -> []
                                | None -> [])

                        // emitCondition: Evaluate while condition
                        // The condition is typically a comparison like `i <= stop`
                        let emitCondition () =
                            match SemanticGraph.tryGetNode whileInfo.ConditionId graph with
                            | Some condNode ->
                                match condNode.Kind with
                                | SemanticKind.Application (funcId, [leftId; rightId]) ->
                                    // Binary operation like <= or >=
                                    match SemanticGraph.tryGetNode funcId graph with
                                    | Some funcNode ->
                                        match funcNode.Kind with
                                        | SemanticKind.Intrinsic intrinsicInfo ->
                                            // Get left operand - carry type through for proper NTU mapping
                                            let (leftOps, leftSSA, leftType) =
                                                match SemanticGraph.tryGetNode leftId graph with
                                                | Some leftNode ->
                                                    let mappedType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph leftNode.Type
                                                    match leftNode.Kind with
                                                    | SemanticKind.VarRef (varName, _) ->
                                                        let (ops, ssa) = loadVar varName seqPtrSSA mappedType
                                                        (ops, ssa, mappedType)
                                                    | SemanticKind.Literal lit ->
                                                        let constSSA = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | LiteralValue.Int64 v -> v | _ -> 0L
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, mappedType))], constSSA, mappedType)
                                                    | _ ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, mappedType))], ssaPlaceholder, mappedType)
                                                | None ->
                                                    let ssaPlaceholder = V moveNextSSA
                                                    moveNextSSA <- moveNextSSA + 1
                                                    // Fallback to platform word type from Fidelity.Platform
                                                    let fallbackType = platformWordType z.State.Platform.TargetArch
                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, fallbackType))], ssaPlaceholder, fallbackType)

                                            // Get right operand - use left operand's type for consistency
                                            let (rightOps, rightSSA) =
                                                match SemanticGraph.tryGetNode rightId graph with
                                                | Some rightNode ->
                                                    let mappedType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph rightNode.Type
                                                    match rightNode.Kind with
                                                    | SemanticKind.VarRef (varName, _) ->
                                                        loadVar varName seqPtrSSA mappedType
                                                    | SemanticKind.Literal lit ->
                                                        let constSSA = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | LiteralValue.Int64 v -> v | _ -> 0L
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, mappedType))], constSSA)
                                                    | _ ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, leftType))], ssaPlaceholder)
                                                | None ->
                                                    let ssaPlaceholder = V moveNextSSA
                                                    moveNextSSA <- moveNextSSA + 1
                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, leftType))], ssaPlaceholder)

                                            // Generate comparison using the operand type (from NTU mapping)
                                            let cmpSSA = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            // Map IntrinsicInfo.Operation to ICmpPred
                                            let cmpPred =
                                                match intrinsicInfo.Operation with
                                                | "op_LessThanOrEqual" -> ICmpPred.Sle
                                                | "op_GreaterThanOrEqual" -> ICmpPred.Sge
                                                | "op_LessThan" -> ICmpPred.Slt
                                                | "op_GreaterThan" -> ICmpPred.Sgt
                                                | "op_Equality" -> ICmpPred.Eq
                                                | "op_Inequality" -> ICmpPred.Ne
                                                | _ -> ICmpPred.Sle  // Default

                                            // Use leftType for comparison - properly mapped through NTU
                                            let cmpOp = MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, cmpPred, leftSSA, rightSSA, leftType))
                                            (leftOps @ rightOps @ [cmpOp], cmpSSA)
                                        | _ ->
                                            // Not an intrinsic - return placeholder
                                            let ssaPlaceholder = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                                    | None ->
                                        let ssaPlaceholder = V moveNextSSA
                                        moveNextSSA <- moveNextSSA + 1
                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                                | _ ->
                                    let ssaPlaceholder = V moveNextSSA
                                    moveNextSSA <- moveNextSSA + 1
                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                            | None ->
                                let ssaPlaceholder = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)

                        // emitYieldValue: Compute yield value
                        let emitYieldValue () =
                            match SemanticGraph.tryGetNode whileInfo.YieldValueId graph with
                            | Some valueNode ->
                                match valueNode.Kind with
                                | SemanticKind.VarRef (varName, _) ->
                                    // Simple yield of a variable
                                    let varType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph valueNode.Type
                                    loadVar varName seqPtrSSA varType
                                | SemanticKind.Literal lit ->
                                    // Yield of a constant
                                    let constSSA = V moveNextSSA
                                    moveNextSSA <- moveNextSSA + 1
                                    let intValue =
                                        match lit with
                                        | LiteralValue.Int32 v -> int64 v
                                        | LiteralValue.Int64 v -> v
                                        | _ -> 0L
                                    ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, elementType))], constSSA)
                                | SemanticKind.Application (funcId, argIds) ->
                                    // Computed yield (e.g., i * factor)
                                    match SemanticGraph.tryGetNode funcId graph with
                                    | Some funcNode ->
                                        match funcNode.Kind with
                                        | SemanticKind.Intrinsic intrinsicInfo ->
                                            match argIds with
                                            | [leftId; rightId] ->
                                                // Binary operation
                                                let (leftOps, leftSSA) =
                                                    match SemanticGraph.tryGetNode leftId graph with
                                                    | Some leftNode ->
                                                        match leftNode.Kind with
                                                        | SemanticKind.VarRef (varName, _) ->
                                                            let varType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph leftNode.Type
                                                            loadVar varName seqPtrSSA varType
                                                        | SemanticKind.Literal lit ->
                                                            let constSSA = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, elementType))], constSSA)
                                                        | _ ->
                                                            let ssaPlaceholder = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                                                    | None ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)

                                                let (rightOps, rightSSA) =
                                                    match SemanticGraph.tryGetNode rightId graph with
                                                    | Some rightNode ->
                                                        match rightNode.Kind with
                                                        | SemanticKind.VarRef (varName, _) ->
                                                            let varType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph rightNode.Type
                                                            loadVar varName seqPtrSSA varType
                                                        | SemanticKind.Literal lit ->
                                                            let constSSA = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, elementType))], constSSA)
                                                        | _ ->
                                                            let ssaPlaceholder = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                                                    | None ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)

                                                // Generate arithmetic operation based on IntrinsicInfo.Operation
                                                let resultSSA = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                let arithOp =
                                                    match intrinsicInfo.Operation with
                                                    | "op_Addition" ->
                                                        MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, elementType))
                                                    | "op_Subtraction" ->
                                                        MLIROp.ArithOp (ArithOp.SubI (resultSSA, leftSSA, rightSSA, elementType))
                                                    | "op_Multiply" ->
                                                        MLIROp.ArithOp (ArithOp.MulI (resultSSA, leftSSA, rightSSA, elementType))
                                                    | _ ->
                                                        MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, elementType))
                                                (leftOps @ rightOps @ [arithOp], resultSSA)
                                            | _ ->
                                                let ssaPlaceholder = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                                        | _ ->
                                            let ssaPlaceholder = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                                    | None ->
                                        let ssaPlaceholder = V moveNextSSA
                                        moveNextSSA <- moveNextSSA + 1
                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                                | _ ->
                                    let ssaPlaceholder = V moveNextSSA
                                    moveNextSSA <- moveNextSSA + 1
                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)
                            | None ->
                                let ssaPlaceholder = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, elementType))], ssaPlaceholder)

                        // emitPostYield: Execute post-yield code (e.g., i <- i + 1)
                        // Supports LetBinding for local immutable bindings (e.g., let temp = a + b)
                        let emitPostYield () =
                            // Track local immutable bindings (not in struct, computed locally)
                            let mutable localBindings: Map<string, SSA> = Map.empty
                            
                            // Helper to load a variable - checks local bindings first, then struct
                            let loadVarOrLocal (name: string) (varType: MLIRType) =
                                match Map.tryFind name localBindings with
                                | Some ssa -> ([], ssa)  // Local binding - no ops needed, already have SSA
                                | None -> loadVar name seqPtrSSA varType  // Struct field - emit load ops
                            
                            // Helper to evaluate a value expression
                            let rec emitValueExpr (valueId: NodeId) (resultType: MLIRType) : (MLIROp list * SSA) =
                                match SemanticGraph.tryGetNode valueId graph with
                                | Some valueNode ->
                                    match valueNode.Kind with
                                    | SemanticKind.Application (funcId, [leftId; rightId]) ->
                                        // Binary operation like a + b
                                        let leftType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph valueNode.Type
                                        let (leftOps, leftSSA) = emitOperand leftId leftType
                                        let (rightOps, rightSSA) = emitOperand rightId leftType
                                        
                                        match SemanticGraph.tryGetNode funcId graph with
                                        | Some funcNode ->
                                            match funcNode.Kind with
                                            | SemanticKind.Intrinsic intrinsicInfo ->
                                                let resultSSA = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                let arithOp =
                                                    match intrinsicInfo.Operation with
                                                    | "op_Addition" -> MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, leftType))
                                                    | "op_Subtraction" -> MLIROp.ArithOp (ArithOp.SubI (resultSSA, leftSSA, rightSSA, leftType))
                                                    | "op_Multiply" -> MLIROp.ArithOp (ArithOp.MulI (resultSSA, leftSSA, rightSSA, leftType))
                                                    | _ -> MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, leftType))
                                                (leftOps @ rightOps @ [arithOp], resultSSA)
                                            | _ -> emitDefault resultType
                                        | None -> emitDefault resultType
                                    | SemanticKind.VarRef (vName, _) ->
                                        loadVarOrLocal vName resultType
                                    | SemanticKind.Literal lit ->
                                        let constSSA = V moveNextSSA
                                        moveNextSSA <- moveNextSSA + 1
                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, resultType))], constSSA)
                                    | _ -> emitDefault resultType
                                | None -> emitDefault resultType
                            
                            and emitOperand (nodeId: NodeId) (opType: MLIRType) : (MLIROp list * SSA) =
                                match SemanticGraph.tryGetNode nodeId graph with
                                | Some node ->
                                    match node.Kind with
                                    | SemanticKind.VarRef (vName, _) -> loadVarOrLocal vName opType
                                    | SemanticKind.Literal lit ->
                                        let constSSA = V moveNextSSA
                                        moveNextSSA <- moveNextSSA + 1
                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, opType))], constSSA)
                                    | _ -> emitDefault opType
                                | None -> emitDefault opType
                            
                            and emitDefault (ty: MLIRType) =
                                let ssa = V moveNextSSA
                                moveNextSSA <- moveNextSSA + 1
                                ([MLIROp.ArithOp (ArithOp.ConstI (ssa, 0L, ty))], ssa)
                            
                            // Process each post-yield expression in order
                            whileInfo.PostYieldExprs
                            |> List.collect (fun exprId ->
                                match SemanticGraph.tryGetNode exprId graph with
                                | Some exprNode ->
                                    match exprNode.Kind with
                                    | SemanticKind.Binding (name, isMutable, _, _) when not isMutable ->
                                        // Immutable let binding like: let temp = a + b
                                        // The value expression is in Children[0]
                                        // Compute value and store in local bindings map (not in struct)
                                        match exprNode.Children with
                                        | valueId :: _ ->
                                            let bindingType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph exprNode.Type
                                            let (valueOps, valueSSA) = emitValueExpr valueId bindingType
                                            localBindings <- Map.add name valueSSA localBindings
                                            valueOps  // Emit the computation ops, no store
                                        | [] -> []
                                    | SemanticKind.Set (targetId, valueId) ->
                                        // Mutable assignment like: i <- i + 1 or b <- temp
                                        match SemanticGraph.tryGetNode targetId graph with
                                        | Some targetNode ->
                                            match targetNode.Kind with
                                            | SemanticKind.VarRef (varName, _) ->
                                                let varType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph targetNode.Type
                                                let (valueOps, valueSSA) = emitValueExpr valueId varType
                                                valueOps @ storeVar varName seqPtrSSA valueSSA varType
                                            | _ -> []
                                        | None -> []
                                    | _ -> []
                                | None -> [])

                        // emitPreYield: Execute pre-yield code (e.g., sum <- sum + i)
                        // PRD-15: Pre-yield expressions execute BEFORE yielding the value
                        // This is critical for computed sequences like triangularNumbers
                        let emitPreYield () =
                            whileInfo.PreYieldExprs
                            |> List.collect (fun exprId ->
                                match SemanticGraph.tryGetNode exprId graph with
                                | Some exprNode ->
                                    match exprNode.Kind with
                                    | SemanticKind.Set (targetId, valueId) ->
                                        // Assignment like sum <- sum + i
                                        match SemanticGraph.tryGetNode targetId graph with
                                        | Some targetNode ->
                                            match targetNode.Kind with
                                            | SemanticKind.VarRef (varName, _) ->
                                                let varType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph targetNode.Type
                                                // Evaluate the value expression
                                                let (valueOps, valueSSA) =
                                                    match SemanticGraph.tryGetNode valueId graph with
                                                    | Some valueNode ->
                                                        match valueNode.Kind with
                                                        | SemanticKind.Application (funcId, [leftId; rightId]) ->
                                                            // Binary operation like sum + i
                                                            let (leftOps, leftSSA) =
                                                                match SemanticGraph.tryGetNode leftId graph with
                                                                | Some leftNode ->
                                                                    match leftNode.Kind with
                                                                    | SemanticKind.VarRef (vName, _) ->
                                                                        loadVar vName seqPtrSSA varType
                                                                    | SemanticKind.Literal lit ->
                                                                        let constSSA = V moveNextSSA
                                                                        moveNextSSA <- moveNextSSA + 1
                                                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, varType))], constSSA)
                                                                    | _ ->
                                                                        let ssaPlaceholder = V moveNextSSA
                                                                        moveNextSSA <- moveNextSSA + 1
                                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                                | None ->
                                                                    let ssaPlaceholder = V moveNextSSA
                                                                    moveNextSSA <- moveNextSSA + 1
                                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)

                                                            let (rightOps, rightSSA) =
                                                                match SemanticGraph.tryGetNode rightId graph with
                                                                | Some rightNode ->
                                                                    match rightNode.Kind with
                                                                    | SemanticKind.VarRef (vName, _) ->
                                                                        loadVar vName seqPtrSSA varType
                                                                    | SemanticKind.Literal lit ->
                                                                        let constSSA = V moveNextSSA
                                                                        moveNextSSA <- moveNextSSA + 1
                                                                        let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                                                        ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, varType))], constSSA)
                                                                    | _ ->
                                                                        let ssaPlaceholder = V moveNextSSA
                                                                        moveNextSSA <- moveNextSSA + 1
                                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                                | None ->
                                                                    let ssaPlaceholder = V moveNextSSA
                                                                    moveNextSSA <- moveNextSSA + 1
                                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)

                                                            // Determine operation from intrinsic
                                                            match SemanticGraph.tryGetNode funcId graph with
                                                            | Some funcNode ->
                                                                match funcNode.Kind with
                                                                | SemanticKind.Intrinsic intrinsicInfo ->
                                                                    let resultSSA = V moveNextSSA
                                                                    moveNextSSA <- moveNextSSA + 1
                                                                    let arithOp =
                                                                        match intrinsicInfo.Operation with
                                                                        | "op_Addition" ->
                                                                            MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, varType))
                                                                        | "op_Subtraction" ->
                                                                            MLIROp.ArithOp (ArithOp.SubI (resultSSA, leftSSA, rightSSA, varType))
                                                                        | _ ->
                                                                            MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, varType))
                                                                    (leftOps @ rightOps @ [arithOp], resultSSA)
                                                                | _ ->
                                                                    let ssaPlaceholder = V moveNextSSA
                                                                    moveNextSSA <- moveNextSSA + 1
                                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                            | None ->
                                                                let ssaPlaceholder = V moveNextSSA
                                                                moveNextSSA <- moveNextSSA + 1
                                                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                        | SemanticKind.VarRef (vName, _) ->
                                                            loadVar vName seqPtrSSA varType
                                                        | SemanticKind.Literal lit ->
                                                            let constSSA = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            let intValue = match lit with | LiteralValue.Int32 v -> int64 v | _ -> 0L
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, varType))], constSSA)
                                                        | _ ->
                                                            let ssaPlaceholder = V moveNextSSA
                                                            moveNextSSA <- moveNextSSA + 1
                                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                    | None ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, varType))], ssaPlaceholder)
                                                // Store the result
                                                valueOps @ storeVar varName seqPtrSSA valueSSA varType
                                            | _ -> []
                                        | None -> []
                                    | _ -> []
                                | None -> [])

                        // emitYieldCondition: Evaluate if-condition(s) for conditional yield
                        // For nested ifs: `if A then if B then yield x` → evaluate A && B
                        let emitYieldCondition =
                            match whileInfo.ConditionalYield with
                            | Some condYieldInfo when not (List.isEmpty condYieldInfo.ConditionIds) ->
                                Some (fun () ->
                                    // Helper to evaluate operands (VarRef, Literal, nested Application)
                                    let rec evalOperand (nodeId: NodeId) : (MLIROp list * SSA * MLIRType) =
                                        match SemanticGraph.tryGetNode nodeId graph with
                                        | Some opNode ->
                                            let mappedType = mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph opNode.Type
                                            match opNode.Kind with
                                            | SemanticKind.VarRef (varName, _) ->
                                                let (ops, ssa) = loadVar varName seqPtrSSA mappedType
                                                (ops, ssa, mappedType)
                                            | SemanticKind.Literal lit ->
                                                let constSSA = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                let intValue = match lit with | LiteralValue.Int32 v -> int64 v | LiteralValue.Int64 v -> v | _ -> 0L
                                                ([MLIROp.ArithOp (ArithOp.ConstI (constSSA, intValue, mappedType))], constSSA, mappedType)
                                            | SemanticKind.Application (innerFuncId, [innerLeftId; innerRightId]) ->
                                                // Nested operation like n % 2
                                                match SemanticGraph.tryGetNode innerFuncId graph with
                                                | Some innerFuncNode ->
                                                    match innerFuncNode.Kind with
                                                    | SemanticKind.Intrinsic innerIntrinsicInfo ->
                                                        let (leftOps, leftSSA, leftType) = evalOperand innerLeftId
                                                        let (rightOps, rightSSA, _) = evalOperand innerRightId
                                                        let resultSSA = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        let arithOp =
                                                            match innerIntrinsicInfo.Operation with
                                                            | "op_Modulus" ->
                                                                MLIROp.ArithOp (ArithOp.RemSI (resultSSA, leftSSA, rightSSA, leftType))
                                                            | "op_Addition" ->
                                                                MLIROp.ArithOp (ArithOp.AddI (resultSSA, leftSSA, rightSSA, leftType))
                                                            | "op_Subtraction" ->
                                                                MLIROp.ArithOp (ArithOp.SubI (resultSSA, leftSSA, rightSSA, leftType))
                                                            | "op_Multiply" ->
                                                                MLIROp.ArithOp (ArithOp.MulI (resultSSA, leftSSA, rightSSA, leftType))
                                                            | _ ->
                                                                MLIROp.ArithOp (ArithOp.RemSI (resultSSA, leftSSA, rightSSA, leftType))
                                                        (leftOps @ rightOps @ [arithOp], resultSSA, leftType)
                                                    | _ ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, mappedType))], ssaPlaceholder, mappedType)
                                                | None ->
                                                    let ssaPlaceholder = V moveNextSSA
                                                    moveNextSSA <- moveNextSSA + 1
                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, mappedType))], ssaPlaceholder, mappedType)
                                            | _ ->
                                                let ssaPlaceholder = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, mappedType))], ssaPlaceholder, mappedType)
                                        | None ->
                                            let fallbackType = platformWordType z.State.Platform.TargetArch
                                            let ssaPlaceholder = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 0L, fallbackType))], ssaPlaceholder, fallbackType)

                                    // Helper to evaluate a single boolean condition
                                    let evalSingleCondition (condId: NodeId) : (MLIROp list * SSA) =
                                        match SemanticGraph.tryGetNode condId graph with
                                        | Some condNode ->
                                            match condNode.Kind with
                                            | SemanticKind.Application (funcId, [leftId; rightId]) ->
                                                match SemanticGraph.tryGetNode funcId graph with
                                                | Some funcNode ->
                                                    match funcNode.Kind with
                                                    | SemanticKind.Intrinsic intrinsicInfo ->
                                                        let (leftOps, leftSSA, leftType) = evalOperand leftId
                                                        let (rightOps, rightSSA, _) = evalOperand rightId
                                                        let cmpSSA = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        let cmpPred =
                                                            match intrinsicInfo.Operation with
                                                            | "op_Equality" -> ICmpPred.Eq
                                                            | "op_Inequality" -> ICmpPred.Ne
                                                            | "op_LessThan" -> ICmpPred.Slt
                                                            | "op_LessThanOrEqual" -> ICmpPred.Sle
                                                            | "op_GreaterThan" -> ICmpPred.Sgt
                                                            | "op_GreaterThanOrEqual" -> ICmpPred.Sge
                                                            | _ -> ICmpPred.Eq
                                                        let cmpOp = MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, cmpPred, leftSSA, rightSSA, leftType))
                                                        (leftOps @ rightOps @ [cmpOp], cmpSSA)
                                                    | _ ->
                                                        let ssaPlaceholder = V moveNextSSA
                                                        moveNextSSA <- moveNextSSA + 1
                                                        ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                                                | None ->
                                                    let ssaPlaceholder = V moveNextSSA
                                                    moveNextSSA <- moveNextSSA + 1
                                                    ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                                            | _ ->
                                                let ssaPlaceholder = V moveNextSSA
                                                moveNextSSA <- moveNextSSA + 1
                                                ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)
                                        | None ->
                                            let ssaPlaceholder = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            ([MLIROp.ArithOp (ArithOp.ConstI (ssaPlaceholder, 1L, MLIRTypes.i1))], ssaPlaceholder)

                                    // Evaluate all conditions and AND them together
                                    let mutable allOps = []
                                    let mutable resultSSA = V 0  // Will be set below

                                    for i, condId in List.indexed condYieldInfo.ConditionIds do
                                        let (condOps, condSSA) = evalSingleCondition condId
                                        allOps <- allOps @ condOps
                                        if i = 0 then
                                            resultSSA <- condSSA
                                        else
                                            // AND with previous result
                                            let andSSA = V moveNextSSA
                                            moveNextSSA <- moveNextSSA + 1
                                            allOps <- allOps @ [MLIROp.ArithOp (ArithOp.AndI (andSSA, resultSSA, condSSA, MLIRTypes.i1))]
                                            resultSSA <- andSSA

                                    (allOps, resultSSA))
                            | _ -> None

                        // Generate MoveNext function using while-based generator
                        let moveNextFuncOp = SeqWitness.witnessMoveNextWhileBased
                                                whileBasedInfo
                                                emitCondition
                                                emitYieldValue
                                                emitPostYield
                                                emitPreYield
                                                emitInit
                                                emitYieldCondition

                        // Add MoveNext to top-level
                        if not (emittedFunctions.Contains(moveNextName)) then
                            emittedFunctions.Add(moveNextName) |> ignore
                            emitTopLevel moveNextFuncOp z

                        // Create seq struct with captures AND internal state fields
                        // Internal state fields are initialized to 0, MoveNext state 0 will set actual values
                        let ops, result = SeqWitness.witnessSeqCreateFull node.Id z moveNextName elementType captureVals internalStateTypes
                        emitAll ops z
                        z, result

                | None ->
                    // No yields in this seq expression - empty sequence
                    // For empty seq, create a struct with state=-1 (already done)
                    let moveNextName = sprintf "seq_%d_moveNext" (NodeId.value node.Id)
                    let captureVals = []
                    let seqStructTy = SeqWitness.seqStructTypeNoCaptures elementType

                    // Generate empty MoveNext that immediately returns false
                    let emptyYieldBlocks = []
                    let moveNextFuncOp = SeqWitness.witnessMoveNext moveNextName seqStructTy elementType emptyYieldBlocks

                    if not (emittedFunctions.Contains(moveNextName)) then
                        emittedFunctions.Add(moveNextName) |> ignore
                        emitTopLevel moveNextFuncOp z

                    let ops, result = SeqWitness.witnessSeqCreate node.Id z moveNextName elementType captureVals
                    emitAll ops z
                    z, result

            | SemanticKind.Yield _valueId ->
                // PRD-15: Yield is a MARKER node - its code is generated inside MoveNext
                // The yield value is extracted from the graph by SeqExpr handler
                // During normal traversal, Yield nodes are inside SeqExpr bodies which
                // are NOT traversed (see walk function's SeqExpr case)
                // If we reach here, it means a Yield is outside a seq { } - that's an error
                // But FNCS should have caught this, so we just return void to be safe
                z, TRVoid

            | SemanticKind.YieldBang _seqId ->
                // PRD-15 SimpleSeq: yield! (flattening nested sequences) is not supported
                // This will be implemented in a future PRD when we need Seq.collect semantics
                z, TRError "YieldBang (yield!) is not supported in PRD-15 SimpleSeq"

            | SemanticKind.ForEach (varName, collectionId, bodyId) ->
                // ForEach consumes a sequence
                match resolveNodeToVal collectionId z with
                | Some seqVal ->
                    // Get element type from seq type
                    let elementType =
                        match SemanticGraph.tryGetNode collectionId graph with
                        | Some collNode ->
                            match collNode.Type with
                            | NativeType.TSeq elemType ->
                                mapNativeTypeWithGraphForArch z.State.Platform.TargetArch graph elemType
                            | _ -> MLIRTypes.i64
                        | None -> MLIRTypes.i64

                    // Get body ops from region collection
                    let bodyKey = (nodeIdVal, regionKindToInt RegionKind.BodyRegion)
                    let bodyOps = regionOps.GetValueOrDefault(bodyKey, [])

                    let ops, result = SeqWitness.witnessForEach node.Id z seqVal elementType bodyOps
                    emitAll ops z
                    z, result
                | None ->
                    z, TRError "ForEach: collection not computed"

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
    match fromEntryPoint graph ssaAssignment analysisResult stringTable platformResolution patternBindingAnalysis entryPointLambdaIds with
    | None ->
        "", ["No entry point found in graph"]

    | Some initialZipper ->
        // Run traversal using recursive subtree walker starting from entry points
        let finalZipper =
            graph.EntryPoints
            |> List.fold (fun z id -> foldSubtree id z) initialZipper

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

        // GLOBAL CLOSURE ARENA (Native "Heap" for Closures)
        // 1MB static buffer for closure environments.
        // This avoids malloc/libc and provides a simple bump allocator.
        let closureArenaOps = [
            MLIROp.LLVMOp (GlobalDef ("closure_heap", "dense<0> : vector<1048576xi8>", TArray (1048576, TInt I8), false))
            MLIROp.LLVMOp (GlobalDef ("closure_pos", "0", TInt I64, false))
        ]

        // Combine and serialize: strings, arena, _start (if any), then user functions
        let allOps = stringOps @ closureArenaOps @ startWrapperOps @ (List.rev topLevelOps)
        let body = opsToString allOps
        let mlir = sprintf "module {\n%s}\n" body

        mlir, List.rev errors

/// Transfer a SemanticGraph to MLIR text
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    fst (transferGraphCore graph isFreestanding false None)

/// Transfer with diagnostics
let transferGraphWithDiagnostics (graph: SemanticGraph) (isFreestanding: bool) (intermediatesDir: string option) : string * string list =
    transferGraphCore graph isFreestanding true intermediatesDir
