/// NanopassArchitecture - Parallel nanopass framework
///
/// Each witness = one nanopass (complete PSG traversal)
/// Nanopasses run in parallel via IcedTasks
/// Results overlay/fold into cohesive MLIR graph
module Alex.Traversal.NanopassArchitecture

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// Nanopass execution phase
/// ContentPhase: Witnesses run first, respect scope boundaries, don't recurse into scopes
/// StructuralPhase: Witnesses run second, wrap content in scope structures (FuncDef, SCFOp)
type NanopassPhase =
    | ContentPhase      // Literal, Arith, Lazy, Collections - traverse but respect scopes
    | StructuralPhase   // Lambda, ControlFlow - handle scope boundaries

/// A nanopass is a complete PSG traversal that selectively witnesses nodes
type Nanopass = {
    /// Nanopass name (e.g., "Literal", "Arithmetic", "ControlFlow")
    Name: string

    /// Execution phase (ContentPhase runs before StructuralPhase)
    Phase: NanopassPhase

    /// The witnessing function for this nanopass
    /// Returns skip for nodes it doesn't handle
    Witness: WitnessContext -> SemanticNode -> WitnessOutput
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAVERSAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Check if this node defines a scope boundary (owns its children)
/// Scope boundaries prevent child recursion during global traversal.
/// Instead, scope-owning witnesses use witnessSubgraph to handle their children.
let private isScopeBoundary (node: SemanticNode) : bool =
    match node.Kind with
    | SemanticKind.Lambda _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.WhileLoop _ -> true
    | SemanticKind.ForLoop _ -> true
    | SemanticKind.ForEach _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.TryWith _ -> true
    | _ -> false

/// Visit all nodes in PSG via zipper traversal
/// Each nanopass traverses entire graph, selectively witnessing
let rec private visitAllNodes
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    (visitedCtx: WitnessContext)
    (currentNode: SemanticNode)
    (accumulator: MLIRAccumulator)
    : unit =

    // Check if already visited
    if MLIRAccumulator.isVisited currentNode.Id accumulator then
        ()
    else
        // Mark as visited
        MLIRAccumulator.markVisited currentNode.Id accumulator

        // Focus zipper on this node
        match PSGZipper.focusOn currentNode.Id visitedCtx.Zipper with
        | None -> ()
        | Some focusedZipper ->
            // Shadow context with focused zipper
            let focusedCtx = { visitedCtx with Zipper = focusedZipper }

            // Witness this node (may skip if not relevant to this nanopass)
            let output = witness focusedCtx currentNode

            // Add operations to accumulator (both inline and top-level go into TopLevelOps)
            MLIRAccumulator.addTopLevelOps output.InlineOps accumulator
            MLIRAccumulator.addTopLevelOps output.TopLevelOps accumulator

            // Bind result if value
            match output.Result with
            | TRValue v ->
                MLIRAccumulator.bindNode currentNode.Id v.SSA v.Type accumulator
            | TRVoid -> ()
            | TRError diag ->
                MLIRAccumulator.addError diag accumulator

            // Recursively visit children ONLY if not a scope boundary
            // Scope boundaries delegate child witnessing to witnessSubgraph
            if not (isScopeBoundary currentNode) then
                for childId in currentNode.Children do
                    match SemanticGraph.tryGetNode childId visitedCtx.Graph with
                    | Some childNode -> visitAllNodes witness focusedCtx childNode accumulator
                    | None -> ()

/// Witness a sub-graph rooted at a given node, returning collected operations
///
/// This is used by control flow witnesses to materialize branch sub-graphs
/// as operation lists for structured control flow regions (SCF.If, SCF.While, etc.)
///
/// Unlike visitAllNodes which accumulates into a shared accumulator, this:
/// 1. Creates an isolated accumulator for the sub-graph
/// 2. Witnesses all reachable nodes from the root
/// 3. Returns operations as a list (for SCF region nesting)
let witnessSubgraph
    rootNodeId
    (ctx: WitnessContext)
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    : MLIROp list =

    // Create isolated accumulator for this sub-graph
    let subAccumulator = MLIRAccumulator.empty()

    // Get root node
    match SemanticGraph.tryGetNode rootNodeId ctx.Graph with
    | None -> []
    | Some rootNode ->
        // Focus zipper on root of sub-graph
        match PSGZipper.focusOn rootNodeId ctx.Zipper with
        | None -> []
        | Some focusedZipper ->
            // Create sub-context with isolated accumulator
            let subCtx = {
                Graph = ctx.Graph
                Coeffects = ctx.Coeffects
                Accumulator = subAccumulator
                Zipper = focusedZipper
            }

            // Traverse sub-graph starting from root
            visitAllNodes witness subCtx rootNode subAccumulator

            // Return collected operations
            List.rev subAccumulator.TopLevelOps

/// Witness a sub-graph rooted at a given node, returning body ops, module ops, and root result
///
/// Like witnessSubgraph, but separates operations into:
/// - Body operations (for function/branch bodies)
/// - Module-level operations (GlobalString declarations)
/// - Root result (SSA binding for the sub-graph root)
///
/// Used by Lambda witness to properly distribute operations between function body and module level.
let witnessSubgraphWithResult
    rootNodeId
    (ctx: WitnessContext)
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    : MLIROp list * MLIROp list * (SSA * MLIRType) option =

    // Create isolated accumulator for this sub-graph
    let subAccumulator = MLIRAccumulator.empty()

    // Get root node
    match SemanticGraph.tryGetNode rootNodeId ctx.Graph with
    | None -> ([], [], None)
    | Some rootNode ->
        // Focus zipper on root of sub-graph
        match PSGZipper.focusOn rootNodeId ctx.Zipper with
        | None -> ([], [], None)
        | Some focusedZipper ->
            // Create sub-context with isolated accumulator
            let subCtx = {
                Graph = ctx.Graph
                Coeffects = ctx.Coeffects
                Accumulator = subAccumulator
                Zipper = focusedZipper
            }

            // Traverse sub-graph starting from root
            visitAllNodes witness subCtx rootNode subAccumulator

            // Get result binding for root node
            let rootResult = MLIRAccumulator.recallNode rootNodeId subAccumulator

            // Separate operations into body ops and module-level ops
            let allOps = List.rev subAccumulator.TopLevelOps
            let bodyOps = allOps |> List.filter (fun op ->
                match op with
                | MLIROp.GlobalString _ -> false  // Module-level
                | _ -> true  // Body operation
            )
            let moduleOps = allOps |> List.filter (fun op ->
                match op with
                | MLIROp.GlobalString _ -> true  // Module-level
                | _ -> false
            )

            // Return body ops, module ops, and root result
            (bodyOps, moduleOps, rootResult)

/// Run a single nanopass over entire PSG
let runNanopass
    (nanopass: Nanopass)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator =

    // Create fresh accumulator
    let freshAcc = MLIRAccumulator.empty()

    // Visit ALL reachable nodes, not just entry-point-reachable nodes
    // This ensures nodes like Console.write/writeln (reachable via VarRef but not via child edges) are witnessed
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        if node.IsReachable && not (MLIRAccumulator.isVisited nodeId freshAcc) then
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = freshAcc
                    Zipper = initialZipper
                }
                // Visit this reachable node
                visitAllNodes nanopass.Witness nodeCtx node freshAcc

    freshAcc

// ═══════════════════════════════════════════════════════════════════════════
// OVERLAY MERGE (Associative)
// ═══════════════════════════════════════════════════════════════════════════

/// Merge/overlay two accumulators from parallel nanopasses
/// MUST be associative: merge (merge a b) c = merge a (merge b c)
let overlayAccumulators (acc1: MLIRAccumulator) (acc2: MLIRAccumulator) : MLIRAccumulator =
    let merged = MLIRAccumulator.empty()

    // Merge top-level ops (order may vary, but all are included)
    MLIRAccumulator.addTopLevelOps acc1.TopLevelOps merged
    MLIRAccumulator.addTopLevelOps acc2.TopLevelOps merged

    // Merge visited sets
    merged.Visited <- Set.union acc1.Visited acc2.Visited

    // Merge bindings (should be disjoint - different nanopasses handle different nodes)
    merged.CurrentScope <-
        { merged.CurrentScope with
            NodeAssoc =
                Map.fold (fun m k v -> Map.add k v m)
                    acc1.CurrentScope.NodeAssoc
                    acc2.CurrentScope.NodeAssoc }

    // Merge errors
    merged.Errors <- acc1.Errors @ acc2.Errors

    merged

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

/// Registry of all nanopasses (populated by witnesses)
type NanopassRegistry = {
    /// All registered nanopasses
    Nanopasses: Nanopass list
}

module NanopassRegistry =
    let empty = { Nanopasses = [] }

    let register (nanopass: Nanopass) (registry: NanopassRegistry) =
        { registry with Nanopasses = nanopass :: registry.Nanopasses }

    let registerAll (nanopasses: Nanopass list) (registry: NanopassRegistry) =
        { registry with Nanopasses = nanopasses @ registry.Nanopasses }
