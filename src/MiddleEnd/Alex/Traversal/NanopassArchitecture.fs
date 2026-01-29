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

/// A nanopass is a complete PSG traversal that selectively witnesses nodes
type Nanopass = {
    /// Nanopass name (e.g., "Literal", "Arithmetic", "ControlFlow")
    Name: string

    /// The witnessing function for this nanopass
    /// Returns skip for nodes it doesn't handle
    Witness: WitnessContext -> SemanticNode -> WitnessOutput
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAVERSAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

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
            | TRError msg ->
                MLIRAccumulator.addError msg accumulator

            // Recursively visit children
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

/// Run a single nanopass over entire PSG
let runNanopass
    (nanopass: Nanopass)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator =

    // Create fresh accumulator
    let freshAcc = MLIRAccumulator.empty()

    // Start from entry points
    for entryId in graph.EntryPoints do
        match PSGZipper.create graph entryId with
        | None -> ()
        | Some initialZipper ->
            match SemanticGraph.tryGetNode entryId graph with
            | None -> ()
            | Some entryNode ->
                // Create context for this entry point
                let entryCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = freshAcc
                    Zipper = initialZipper
                }

                // Traverse from this entry point
                visitAllNodes nanopass.Witness entryCtx entryNode freshAcc

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
