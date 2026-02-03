/// NanopassArchitecture - Parallel nanopass framework
///
/// Each witness = one nanopass (complete PSG traversal)
/// Nanopasses run in parallel via IcedTasks
/// Results overlay/fold into cohesive MLIR graph
module Alex.Traversal.NanopassArchitecture

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper
open Alex.Traversal.CoverageValidation

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// A nanopass is a complete PSG traversal that selectively witnesses nodes
/// Single-phase post-order traversal: all witnesses run during one traversal
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

/// Check if this node defines a scope boundary (owns its children)
/// Scope boundaries control recursion: scope-owning witnesses handle their own children
/// via explicit scope markers (ScopeEnter/ScopeExit) instead of automatic child traversal.
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

/// Visit all nodes in post-order (children before parents)
/// PUBLIC: Used by Lambda/ControlFlow witnesses for sub-graph traversal
/// Post-order ensures children's SSA bindings are available when parent witnesses
let rec visitAllNodes
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    (visitedCtx: WitnessContext)
    (currentNode: SemanticNode)
    (accumulator: MLIRAccumulator)
    (visited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across all nanopasses)
    : unit =

    // Check if already visited
    if Set.contains currentNode.Id !visited then
        printfn "[visitAllNodes] Node %A already visited - skipping" currentNode.Id
        ()
    else
        // Mark as visited
        visited := Set.add currentNode.Id !visited

        // Focus zipper on this node
        match PSGZipper.focusOn currentNode.Id visitedCtx.Zipper with
        | None ->
            printfn "[visitAllNodes] WARNING: Failed to focus zipper on node %A" currentNode.Id
            ()
        | Some focusedZipper ->
            printfn "[visitAllNodes] Successfully focused on node %A" currentNode.Id
            // Shadow context with focused zipper
            let focusedCtx = { visitedCtx with Zipper = focusedZipper }

            // POST-ORDER Phase 1: Visit children FIRST (tree edges)
            // This ensures children's SSA bindings are available when parent witnesses
            if not (isScopeBoundary currentNode) then
                printfn "[visitAllNodes] Node %A is NOT a scope boundary - visiting %d children" currentNode.Id currentNode.Children.Length
                for childId in currentNode.Children do
                    printfn "[visitAllNodes] Visiting child %A of parent %A" childId currentNode.Id
                    match SemanticGraph.tryGetNode childId visitedCtx.Graph with
                    | Some childNode -> visitAllNodes witness focusedCtx childNode accumulator visited
                    | None ->
                        printfn "[visitAllNodes] WARNING: Child node %A not found in graph!" childId

            // POST-ORDER Phase 2: Visit VarRef binding targets (reference edges)
            // VarRef nodes reference Bindings that may not be in child structure - visit those too
            match currentNode.Kind with
            | SemanticKind.VarRef (_, Some bindingId) ->
                if not (Set.contains bindingId !visited) then
                    match SemanticGraph.tryGetNode bindingId visitedCtx.Graph with
                    | Some bindingNode -> visitAllNodes witness focusedCtx bindingNode accumulator visited
                    | None -> ()
            | _ -> ()

            // THEN witness current node (after ALL dependencies - children AND references)
            printfn "[visitAllNodes] Invoking witness on node %A" currentNode.Id
            let output = witness focusedCtx currentNode
            printfn "[visitAllNodes] Witness returned %A for node %A" output.Result currentNode.Id

            // Add operations to appropriate accumulators
            // InlineOps go to current scope accumulator (may be nested)
            MLIRAccumulator.addOps output.InlineOps accumulator
            // TopLevelOps go to ROOT accumulator (module-level only)
            if not (List.isEmpty output.TopLevelOps) then
                let funcDefCount = output.TopLevelOps |> List.filter (fun op -> match op with MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> true | _ -> false) |> List.length
                printfn "[DEBUG] Node %d: Adding %d TopLevelOps (%d FuncDefs) to RootAccumulator" (NodeId.value currentNode.Id) (List.length output.TopLevelOps) funcDefCount
            MLIRAccumulator.addOps output.TopLevelOps focusedCtx.RootAccumulator

            // Bind result if value (global binding)
            match output.Result with
            | TRValue v ->
                MLIRAccumulator.bindNode currentNode.Id v.SSA v.Type accumulator
            | TRVoid -> ()
            | TRError diag ->
                MLIRAccumulator.addError diag accumulator
            | TRSkip -> ()  // Should never reach here (combineWitnesses filters out TRSkip)

/// REMOVED: witnessSubgraph and witnessSubgraphWithResult
///
/// These functions created isolated accumulators which broke SSA binding resolution.
/// With the flat accumulator architecture, scope boundaries are handled via ScopeMarkers
/// instead of isolated accumulators.
///
/// Lambda and ControlFlow witnesses now use:
/// 1. ScopeMarker (ScopeEnter) to mark scope start
/// 2. Witness body nodes into shared accumulator
/// 3. ScopeMarker (ScopeExit) to mark scope end
/// 4. extractScope to get operations between markers
/// 5. Wrap extracted ops in FuncDef/SCFOp
/// 6. replaceScope to substitute wrapped operation for markers+contents

/// Run a single nanopass over entire PSG with SHARED accumulator and GLOBAL visited set
let runNanopass
    (nanopass: Nanopass)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)  // SHARED accumulator (ops, bindings, errors)
    (globalVisited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across ALL nanopasses)
    : unit =

    // Visit ALL reachable nodes, not just entry-point-reachable nodes
    // This ensures nodes like Console.write/writeln (reachable via VarRef but not via child edges) are witnessed
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        if node.IsReachable && not (Set.contains nodeId !globalVisited) then
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = sharedAcc  // SHARED accumulator
                    RootAccumulator = sharedAcc  // Root accumulator (same as shared for top-level)
                    Zipper = initialZipper
                    GlobalVisited = globalVisited  // GLOBAL visited set
                }
                // Visit this reachable node (post-order) with GLOBAL visited set
                visitAllNodes nanopass.Witness nodeCtx node sharedAcc globalVisited

// ═══════════════════════════════════════════════════════════════════════════
// REMOVED: overlayAccumulators
// ═══════════════════════════════════════════════════════════════════════════

/// REMOVED: overlayAccumulators
///
/// This function merged separate accumulators from parallel nanopasses.
/// With the flat accumulator architecture, ALL nanopasses share a single accumulator,
/// so no merging is needed. Operations and bindings accumulate directly during traversal.

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

// ═══════════════════════════════════════════════════════════════════════════
// COMBINED WITNESS EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Combine multiple nanopass witnesses into a single witness that tries each in order
/// WITH COVERAGE VALIDATION: Reports error if no witness handles a node (prevents silent gaps)
let private combineWitnesses (nanopasses: Nanopass list) : (WitnessContext -> SemanticNode -> WitnessOutput) =
    fun ctx node ->
        let rec tryWitnesses remaining =
            match remaining with
            | [] ->
                // NO WITNESS HANDLED THIS NODE - Report error for coverage validation
                // This prevents silent gaps where nodes are skipped without any witness
                // processing them, which leads to empty MLIR output.
                // Structural nodes (ModuleDef, Sequential) should have transparent witnesses.
                WitnessOutput.error (sprintf "No witness handled node %A (%A)" node.Id node.Kind)
            | nanopass :: rest ->
                let result = nanopass.Witness ctx node
                match result.Result with
                | TRSkip ->
                    // This witness doesn't handle this node kind - try next witness
                    tryWitnesses rest
                | _ ->
                    // Witness handled the node (TRValue, TRVoid, or TRError) - stop trying
                    result
        tryWitnesses nanopasses

/// Run all nanopasses in single post-order traversal with shared accumulator
let runAllNanopasses
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)
    (globalVisited: ref<Set<NodeId>>)
    : unit =

    // Create combined witness that tries all nanopasses at each node
    let combinedWitness = combineWitnesses nanopasses

    // Visit only TOP-LEVEL reachable nodes (entry points)
    // Scope-internal nodes (Lambda bodies, branch bodies) are visited by their parent witnesses
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        // Only visit if reachable, not yet visited, AND parent is None or ModuleDef (top-level)
        let isTopLevel =
            match node.Parent with
            | None -> true  // No parent = top-level
            | Some parentId ->
                match SemanticGraph.tryGetNode parentId graph with
                | Some parentNode ->
                    match parentNode.Kind with
                    | SemanticKind.ModuleDef _ -> true  // Module-level binding
                    | _ -> false  // Inside a scope (Lambda, If, etc.)
                | None -> false
        if node.IsReachable && not (Set.contains nodeId !globalVisited) && isTopLevel then
            printfn "[DEBUG] Processing top-level node %d (%A)" (NodeId.value nodeId) node.Kind
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = sharedAcc
                    RootAccumulator = sharedAcc  // Root same as shared for top-level
                    Zipper = initialZipper
                    GlobalVisited = globalVisited
                }
                visitAllNodes combinedWitness nodeCtx node sharedAcc globalVisited

/// Main entry point: Execute all nanopasses and return accumulator
let executeNanopasses
    (registry: NanopassRegistry)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : MLIRAccumulator =

    if List.isEmpty registry.Nanopasses then
        MLIRAccumulator.empty()
    else
        // Create SINGLE shared accumulator for ALL nanopasses
        let sharedAcc = MLIRAccumulator.empty()

        // Create SINGLE global visited set for ALL nanopasses
        let globalVisited = ref Set.empty

        printfn "[Alex] Single-phase execution: %d registered nanopasses" (List.length registry.Nanopasses)

        // Run all nanopasses together in single traversal
        runAllNanopasses registry.Nanopasses graph coeffects sharedAcc globalVisited

        // TODO: Serialize results if intermediatesDir provided

        // Coverage validation - ensure all reachable nodes were witnessed
        let coverageDiagnostics = CoverageValidation.validateCoverage graph !globalVisited
        if not (List.isEmpty coverageDiagnostics) then
            // Add coverage errors to accumulator
            for diag in coverageDiagnostics do
                MLIRAccumulator.addError diag sharedAcc

        sharedAcc
