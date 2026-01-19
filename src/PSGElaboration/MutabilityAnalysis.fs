/// Mutability Analysis - Pre-transfer analysis for mutable bindings
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It computes:
/// - Which mutable bindings have their address taken (need alloca, not SSA)
/// - Which variables are modified in each loop body (need iter_args for SCF)
///
/// This eliminates on-demand tree traversal during transfer, adhering to
/// the photographer principle: observe the structure, don't compute during transfer.
module PSGElaboration.MutabilityAnalysis

open FSharp.Native.Compiler.PSG.SemanticGraph

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Information about a module-level mutable binding
type ModuleLevelMutable = {
    /// The binding's NodeId
    BindingId: int
    /// The variable name
    Name: string
    /// The initial value NodeId (first child of binding)
    InitialValueId: int
}

/// Result of mutability analysis for a semantic graph
type MutabilityAnalysisResult = {
    /// Set of NodeId values for mutable bindings whose address is taken (&amp;&amp; operator)
    /// These need alloca instead of pure SSA
    AddressedMutableBindings: Set<int>
    
    /// Map from loop body NodeId to list of variable names modified in that body
    /// Used for SCF iter_args generation
    ModifiedVarsInLoopBodies: Map<int, string list>
    
    /// Module-level mutable bindings that need LLVM globals
    /// These cannot use SSA (function-scoped) - they need addressof + load/store
    ModuleLevelMutableBindings: ModuleLevelMutable list
}

// ═══════════════════════════════════════════════════════════════════════════
// Addressed Mutable Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Find all mutable bindings whose address is taken (&amp;&amp; operator)
/// These need alloca instead of pure SSA
/// Returns a set of NodeIds of the mutable Binding nodes
let findAddressedMutableBindings (graph: SemanticGraph) : Set<int> =
    let mutableBindingIds = System.Collections.Generic.HashSet<int>()
    
    // Find all AddressOf nodes and check if they reference VarRef to mutable bindings
    for KeyValue(nodeId, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.AddressOf (exprId, _) ->
            // Check if exprId is a VarRef to a mutable binding
            match SemanticGraph.tryGetNode exprId graph with
            | Some exprNode ->
                match exprNode.Kind with
                | SemanticKind.VarRef (_, Some targetBindingId) ->
                    // Check if target is a mutable binding
                    match SemanticGraph.tryGetNode targetBindingId graph with
                    | Some bindingNode ->
                        match bindingNode.Kind with
                        | SemanticKind.Binding (_, isMutable, _, _) when isMutable ->
                            mutableBindingIds.Add(NodeId.value targetBindingId) |> ignore
                        | _ -> ()
                    | None -> ()
                | _ -> ()
            | None -> ()
        | SemanticKind.Set (targetId, _) ->
            // Any variable that is mutated via Set must be stack-allocated (unless we do mem2reg/iter_args)
            // So we treat it as an "addressed mutable"
            match SemanticGraph.tryGetNode targetId graph with
            | Some targetNode ->
                match targetNode.Kind with
                | SemanticKind.VarRef (_, Some defId) ->
                    mutableBindingIds.Add(NodeId.value defId) |> ignore
                | _ -> ()
            | None -> ()
        | _ -> ()
    
    mutableBindingIds |> Set.ofSeq

// ═══════════════════════════════════════════════════════════════════════════
// Modified Variables Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze a subtree to find all Set nodes and extract their target variable names.
/// This is used to determine which mutable vars need iter_args in SCF loops.
/// Uses the SemanticGraph structure directly - this is pure analysis.
let findModifiedVarsInSubtree (graph: SemanticGraph) (rootNodeId: NodeId) : string list =
    let rec walk (nodeId: NodeId) (acc: Set<string>) : Set<string> =
        match SemanticGraph.tryGetNode nodeId graph with
        | None -> acc
        | Some node ->
            let acc =
                match node.Kind with
                | SemanticKind.Set (targetId, _) ->
                    // Get target node to extract variable name
                    match SemanticGraph.tryGetNode targetId graph with
                    | Some targetNode ->
                        match targetNode.Kind with
                        | SemanticKind.VarRef (name, _) -> Set.add name acc
                        | _ -> acc
                    | None -> acc
                | _ -> acc
            // Walk children
            node.Children |> List.fold (fun a c -> walk c a) acc
    walk rootNodeId Set.empty |> Set.toList

/// Find all loop bodies in the graph and compute which variables are modified in each
/// Returns a map from loop body NodeId to list of modified variable names
let findAllModifiedVarsInLoops (graph: SemanticGraph) : Map<int, string list> =
    let mutable result = Map.empty
    
    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.WhileLoop (_, bodyId) ->
            let modifiedVars = findModifiedVarsInSubtree graph bodyId
            if not (List.isEmpty modifiedVars) then
                result <- Map.add (NodeId.value bodyId) modifiedVars result
        | SemanticKind.ForLoop (_, _, _, _, bodyId) ->
            let modifiedVars = findModifiedVarsInSubtree graph bodyId
            if not (List.isEmpty modifiedVars) then
                result <- Map.add (NodeId.value bodyId) modifiedVars result
        | _ -> ()
    
    result

// ═══════════════════════════════════════════════════════════════════════════
// Module-Level Mutable Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Find all module-level mutable bindings.
/// These are mutable bindings that are direct children of ModuleDef nodes.
/// They need to be emitted as LLVM globals (not SSA values) because:
/// 1. SSA values are function-scoped
/// 2. Module-level mutables are accessed across function boundaries
/// 3. They need addressof + load/store semantics
let findModuleLevelMutableBindings (graph: SemanticGraph) : ModuleLevelMutable list =
    let result = System.Collections.Generic.List<ModuleLevelMutable>()
    
    // Find all ModuleDef nodes and check their direct children
    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.ModuleDef (_, memberIds) ->
            // Check each member for mutable bindings
            for memberId in memberIds do
                match SemanticGraph.tryGetNode memberId graph with
                | Some memberNode ->
                    match memberNode.Kind with
                    | SemanticKind.Binding (name, isMutable, _, _) when isMutable ->
                        // Found a module-level mutable binding
                        match memberNode.Children with
                        | initialValueId :: _ ->
                            result.Add({
                                BindingId = NodeId.value memberId
                                Name = name
                                InitialValueId = NodeId.value initialValueId
                            })
                        | [] -> ()  // Binding without value - shouldn't happen
                    | _ -> ()
                | None -> ()
        | _ -> ()
    
    result |> List.ofSeq

// ═══════════════════════════════════════════════════════════════════════════
// Main Analysis Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Perform complete mutability analysis on a semantic graph
/// This should be called ONCE before transfer begins
let analyze (graph: SemanticGraph) : MutabilityAnalysisResult =
    {
        AddressedMutableBindings = findAddressedMutableBindings graph
        ModifiedVarsInLoopBodies = findAllModifiedVarsInLoops graph
        ModuleLevelMutableBindings = findModuleLevelMutableBindings graph
    }

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Lambda Discovery
// ═══════════════════════════════════════════════════════════════════════════

/// Find entry point Lambda IDs from the semantic graph
/// Entry point Bindings have Lambda children - those are the entry point Lambdas
let findEntryPointLambdaIds (graph: SemanticGraph) : Set<int> =
    graph.EntryPoints
    |> List.collect (fun epId ->
        match SemanticGraph.tryGetNode epId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding (_, _, _, _) ->
                // Entry point Binding - its children include the Lambda
                node.Children
                |> List.choose (fun childId ->
                    match SemanticGraph.tryGetNode childId graph with
                    | Some child when (match child.Kind with SemanticKind.Lambda _ -> true | _ -> false) ->
                        Some (NodeId.value childId)
                    | _ -> None)
            | SemanticKind.Lambda _ ->
                // Entry point is directly a Lambda
                [NodeId.value epId]
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Check module members for main binding
                memberIds
                |> List.collect (fun memberId ->
                    match SemanticGraph.tryGetNode memberId graph with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding (name, _, _, _) when name = "main" ->
                            memberNode.Children
                            |> List.choose (fun childId ->
                                match SemanticGraph.tryGetNode childId graph with
                                | Some child ->
                                    match child.Kind with
                                    | SemanticKind.Lambda _ -> Some (NodeId.value childId)
                                    | _ -> None
                                | None -> None)
                        | _ -> []
                    | None -> [])
            | _ -> []
        | None -> [])
    |> Set.ofList
