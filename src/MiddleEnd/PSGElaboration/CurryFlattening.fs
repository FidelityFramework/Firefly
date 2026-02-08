/// CurryFlattening - Normalize curried Lambda chains and resolve partial applications
///
/// This pass runs BEFORE coeffect analysis (SSAAssignment, etc.) so that
/// downstream passes see flattened multi-parameter Lambdas.
///
/// Two phases:
///   1. Graph normalization: Flatten Lambda(a) → Lambda(b) → body into Lambda(a,b) → body
///   2. Coeffect analysis: Detect partial applications and their saturated call sites
///
/// Architecture note: Phase 1 is a structural normalization (like Phases 1-4 of PSG construction).
/// Phase 2 is a coeffect (metadata about the flattened structure). This separation ensures
/// SSAAssignment sees the correct flattened structure and assigns correct Arg SSAs.
module PSGElaboration.CurryFlattening

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Info about a partial application of a flattened curried function
type PartialApplicationInfo = {
    /// The binding NodeId that owns the target flattened function
    TargetBindingId: NodeId
    /// Arguments supplied at the partial application site
    SuppliedArgNodes: NodeId list
    /// Total parameter count of the flattened function
    TotalParams: int
}

/// Info about a call site that fully saturates a partial application
type SaturatedCallInfo = {
    /// The binding NodeId that owns the target flattened function
    TargetBindingId: NodeId
    /// All arguments in order: from partial app + call site
    AllArgNodes: NodeId list
}

/// Result of curry flattening analysis
type CurryFlatteningResult = {
    /// Partial application Application NodeIds → info
    PartialApplications: Map<NodeId, PartialApplicationInfo>
    /// Saturated call Application NodeIds → direct call info
    SaturatedCalls: Map<NodeId, SaturatedCallInfo>
    /// Binding NodeIds that hold partial applications (witnesses return TRVoid)
    PartialAppBindings: Set<NodeId>
    /// Lambda NodeIds absorbed by flattening (marked unreachable)
    AbsorbedLambdas: Set<NodeId>
    /// Argument node IDs from partial applications whose InlineOps are deferred
    /// to the saturated call site (avoids MLIR region isolation violations when
    /// partial app is at module scope but saturated call is inside a function)
    DeferredArgNodes: Set<NodeId>
}

module CurryFlatteningResult =
    let empty = {
        PartialApplications = Map.empty
        SaturatedCalls = Map.empty
        PartialAppBindings = Set.empty
        AbsorbedLambdas = Set.empty
        DeferredArgNodes = Set.empty
    }

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 1: Graph normalization — flatten nested Lambda chains
// ═══════════════════════════════════════════════════════════════════════════

/// Recursively collect parameters from a chain of nested Lambdas.
/// Returns (innerParams, innermostBodyId, absorbedLambdaIds).
let rec private collectNestedLambdaParams
    (graph: SemanticGraph)
    (nodeId: NodeId)
    : (string * NativeType * NodeId) list * NodeId * NodeId list =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node when node.IsReachable ->
        match node.Kind with
        | SemanticKind.Lambda (params', bodyId, _, _, _) ->
            // This node IS a Lambda — absorb its params and recurse into its body
            let (deeperParams, innermostBody, deeperAbsorbed) =
                collectNestedLambdaParams graph bodyId
            (params' @ deeperParams, innermostBody, nodeId :: deeperAbsorbed)
        | _ ->
            // Not a Lambda — this is the innermost body
            ([], nodeId, [])
    | _ -> ([], nodeId, [])

/// Flatten the PSG graph: merge nested Lambda chains into single multi-param Lambdas.
/// Returns the modified graph and the set of absorbed (now-unreachable) Lambda NodeIds.
let flatten (graph: SemanticGraph) : SemanticGraph * Set<NodeId> =
    let mutable newNodes = graph.Nodes
    let mutable absorbedSet = Set.empty

    for kvp in graph.Nodes do
        let node = kvp.Value
        if not node.IsReachable then () else
        // Only process Lambda nodes that haven't already been absorbed
        if Set.contains node.Id absorbedSet then () else
        match node.Kind with
        | SemanticKind.Lambda (outerParams, bodyId, captures, encFn, ctx) ->
            // Check if body is also a Lambda (curried chain)
            match SemanticGraph.tryGetNode bodyId graph with
            | Some bodyNode when bodyNode.IsReachable ->
                match bodyNode.Kind with
                | SemanticKind.Lambda _ ->
                    // Collect ALL nested params and find innermost non-Lambda body
                    let (innerParams, innermostBody, absorbedIds) =
                        collectNestedLambdaParams graph bodyId
                    let allParams = outerParams @ innerParams

                    // Create flattened Lambda node
                    let newKind =
                        SemanticKind.Lambda(allParams, innermostBody, captures, encFn, ctx)
                    let paramNodeIds = allParams |> List.map (fun (_, _, nid) -> nid)
                    let newChildren = paramNodeIds @ [innermostBody]
                    let newNode = { node with Kind = newKind; Children = newChildren }
                    newNodes <- Map.add node.Id newNode newNodes

                    // Mark absorbed inner Lambdas as unreachable
                    for absId in absorbedIds do
                        match Map.tryFind absId newNodes with
                        | Some absNode ->
                            let marked = { absNode with IsReachable = false }
                            newNodes <- Map.add absId marked newNodes
                            absorbedSet <- Set.add absId absorbedSet
                        | None -> ()

                    // Reparent inner Lambda parameter nodes to the outer Lambda
                    for (_, _, paramNodeId) in innerParams do
                        match Map.tryFind paramNodeId newNodes with
                        | Some paramNode ->
                            let reparented = { paramNode with Parent = Some node.Id }
                            newNodes <- Map.add paramNodeId reparented newNodes
                        | None -> ()

                    // Reparent innermost body to point to outer Lambda
                    match Map.tryFind innermostBody newNodes with
                    | Some bodyNode ->
                        let reparented = { bodyNode with Parent = Some node.Id }
                        newNodes <- Map.add innermostBody reparented newNodes
                    | None -> ()
                | _ -> ()
            | _ -> ()
        | _ -> ()

    ({ graph with Nodes = newNodes }, absorbedSet)

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2: Coeffect — partial application analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve a Binding to its Lambda child (if it's a function binding)
let private resolveBindingToLambda (graph: SemanticGraph) (bindingId: NodeId) : NodeId option =
    match SemanticGraph.tryGetNode bindingId graph with
    | Some bindingNode ->
        match bindingNode.Kind with
        | SemanticKind.Binding _ ->
            bindingNode.Children
            |> List.tryHead
            |> Option.bind (fun childId ->
                match SemanticGraph.tryGetNode childId graph with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ when childNode.IsReachable -> Some childId
                    | _ -> None
                | None -> None)
        | _ -> None
    | None -> None

/// Count the total parameters of a (flattened) Lambda
let private countLambdaParams (graph: SemanticGraph) (lambdaNodeId: NodeId) : int option =
    match SemanticGraph.tryGetNode lambdaNodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Lambda (params', _, _, _, _) -> Some (List.length params')
        | _ -> None
    | None -> None

/// Analyze partial applications and saturated call sites in the flattened graph.
let analyze (graph: SemanticGraph) (absorbedLambdas: Set<NodeId>) : CurryFlatteningResult =
    let mutable partialApps : Map<NodeId, PartialApplicationInfo> = Map.empty
    let mutable saturatedCalls : Map<NodeId, SaturatedCallInfo> = Map.empty
    let mutable partialAppBindings : Set<NodeId> = Set.empty

    // Phase 1: Find partial applications
    // A partial application: Application(VarRef(f), args) where f is a multi-param function
    // and args.Length < f's total param count
    for kvp in graph.Nodes do
        let node = kvp.Value
        if not node.IsReachable then () else
        match node.Kind with
        | SemanticKind.Application (funcNodeId, argNodeIds) ->
            match SemanticGraph.tryGetNode funcNodeId graph with
            | Some funcNode ->
                match funcNode.Kind with
                | SemanticKind.VarRef (_, Some bindingId) ->
                    match resolveBindingToLambda graph bindingId with
                    | Some lambdaId ->
                        match countLambdaParams graph lambdaId with
                        | Some totalParams when totalParams > List.length argNodeIds ->
                            // Under-saturated call = partial application
                            let info = {
                                TargetBindingId = bindingId
                                SuppliedArgNodes = argNodeIds
                                TotalParams = totalParams
                            }
                            partialApps <- Map.add node.Id info partialApps

                            // The parent Binding of this Application holds a partial app
                            match node.Parent with
                            | Some parentId ->
                                match SemanticGraph.tryGetNode parentId graph with
                                | Some parentNode ->
                                    match parentNode.Kind with
                                    | SemanticKind.Binding _ ->
                                        partialAppBindings <- Set.add parentId partialAppBindings
                                    | _ -> ()
                                | None -> ()
                            | None -> ()
                        | _ -> ()
                    | None -> ()
                | _ -> ()
            | None -> ()
        | _ -> ()

    // Phase 2: Find call sites that saturate partial applications
    // Pattern: Application(VarRef(partialAppBinding), remainingArgs)
    for kvp in graph.Nodes do
        let node = kvp.Value
        if not node.IsReachable then () else
        match node.Kind with
        | SemanticKind.Application (funcNodeId, argNodeIds) ->
            match SemanticGraph.tryGetNode funcNodeId graph with
            | Some funcNode ->
                match funcNode.Kind with
                | SemanticKind.VarRef (_, Some bindingId)
                    when Set.contains bindingId partialAppBindings ->
                    // Call uses a partial application binding — find the partial app info
                    match SemanticGraph.tryGetNode bindingId graph with
                    | Some bindingNode ->
                        match bindingNode.Children |> List.tryHead with
                        | Some appId ->
                            match Map.tryFind appId partialApps with
                            | Some partialInfo ->
                                let allArgs = partialInfo.SuppliedArgNodes @ argNodeIds
                                if List.length allArgs = partialInfo.TotalParams then
                                    // Fully saturated
                                    let satInfo = {
                                        TargetBindingId = partialInfo.TargetBindingId
                                        AllArgNodes = allArgs
                                    }
                                    saturatedCalls <- Map.add node.Id satInfo saturatedCalls
                            | None -> ()
                        | None -> ()
                    | None -> ()
                | _ -> ()
            | None -> ()
        | _ -> ()

    // Collect argument node IDs from partial applications
    // These nodes' InlineOps are deferred to the saturated call site
    let deferredArgNodes =
        partialApps
        |> Map.fold (fun acc _ info ->
            info.SuppliedArgNodes |> List.fold (fun s nid -> Set.add nid s) acc
        ) Set.empty

    {
        PartialApplications = partialApps
        SaturatedCalls = saturatedCalls
        PartialAppBindings = partialAppBindings
        AbsorbedLambdas = absorbedLambdas
        DeferredArgNodes = deferredArgNodes
    }
