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
open Alex.Traversal.ScopeContext
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

/// Debug tracing flag for visitAllNodes — set to true for detailed traversal logging
let mutable private traceTraversal = false

/// Visit all nodes in post-order (children before parents)
/// PUBLIC: Used by Lambda/ControlFlow witnesses for sub-graph traversal
/// Post-order ensures children's SSA bindings are available when parent witnesses
let rec visitAllNodes
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    (visitedCtx: WitnessContext)
    (currentNode: SemanticNode)
    (visited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across all nanopasses)
    : unit =

    // Check if already visited
    if Set.contains currentNode.Id !visited then
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
            if traceTraversal then printfn "[visitAllNodes] Focused on node %A" currentNode.Id
            // Shadow context with focused zipper
            let focusedCtx = { visitedCtx with Zipper = focusedZipper }

            // POST-ORDER Phase 1: Visit children FIRST (tree edges)
            // This ensures children's SSA bindings are available when parent witnesses
            if not (isScopeBoundary currentNode) then
                if traceTraversal then printfn "[visitAllNodes] Node %A: visiting %d children" currentNode.Id currentNode.Children.Length
                for childId in currentNode.Children do
                    match SemanticGraph.tryGetNode childId visitedCtx.Graph with
                    | Some childNode -> visitAllNodes witness focusedCtx childNode visited
                    | None ->
                        printfn "[visitAllNodes] WARNING: Child node %A not found in graph!" childId

            // POST-ORDER Phase 2: Visit VarRef binding targets (reference edges)
            // VarRef nodes reference Bindings that may not be in child structure - visit those too
            match currentNode.Kind with
            | SemanticKind.VarRef (_, Some bindingId) ->
                if not (Set.contains bindingId !visited) then
                    match SemanticGraph.tryGetNode bindingId visitedCtx.Graph with
                    | Some bindingNode -> visitAllNodes witness focusedCtx bindingNode visited
                    | None -> ()
            | _ -> ()

            // THEN witness current node (after ALL dependencies - children AND references)
            let output = witness focusedCtx currentNode
            if traceTraversal then printfn "[visitAllNodes] Node %A: witness returned %A" currentNode.Id output.Result

            // Add operations to appropriate scope contexts (principled accumulation)
            // InlineOps go to current scope (may be nested function body)
            // EXCEPTION: Deferred arg nodes (partial app arguments) have their InlineOps
            // stored in the accumulator for re-emission at the saturated call site.
            // This prevents MLIR region isolation violations when partial app is at module
            // scope but saturated call is inside a function body.
            let isDeferredArg = Set.contains currentNode.Id focusedCtx.Coeffects.CurryFlattening.DeferredArgNodes
            if isDeferredArg && not (List.isEmpty output.InlineOps) then
                MLIRAccumulator.deferInlineOps currentNode.Id output.InlineOps focusedCtx.Accumulator
            else
                let updatedCurrentScope = ScopeContext.addOps output.InlineOps !focusedCtx.ScopeContext
                focusedCtx.ScopeContext := updatedCurrentScope

            // TopLevelOps go to ROOT scope (module level: GlobalString, nested FuncDef)
            if not (List.isEmpty output.TopLevelOps) then
                if traceTraversal then
                    let funcDefCount = output.TopLevelOps |> List.filter (fun op -> match op with MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> true | _ -> false) |> List.length
                    printfn "[visitAllNodes] Node %d: Adding %d TopLevelOps (%d FuncDefs) to RootScopeContext" (NodeId.value currentNode.Id) (List.length output.TopLevelOps) funcDefCount
                let updatedRootScope = ScopeContext.addOps output.TopLevelOps !focusedCtx.RootScopeContext
                focusedCtx.RootScopeContext := updatedRootScope

            // Bind result if value (global binding)
            match output.Result with
            | TRValue v ->
                MLIRAccumulator.bindNode currentNode.Id v.SSA v.Type focusedCtx.Accumulator
            | TRVoid -> ()
            | TRError diag ->
                MLIRAccumulator.addError diag focusedCtx.Accumulator
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
    (rootScope: ref<ScopeContext>)  // Shared root scope for operation accumulation
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
                    Accumulator = sharedAcc  // SHARED accumulator (for SSA bindings)
                    RootAccumulator = sharedAcc  // Root accumulator (same as shared for top-level)
                    ScopeContext = rootScope  // Shared root scope (mutable)
                    RootScopeContext = rootScope  // Root scope for TopLevelOps (same as ScopeContext at top-level)
                    Zipper = initialZipper
                    GlobalVisited = globalVisited  // GLOBAL visited set
                }
                // Visit this reachable node (post-order) with GLOBAL visited set
                visitAllNodes nanopass.Witness nodeCtx node globalVisited

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
                //
                // Enrich the error with contextual information to aid diagnosis:
                let kindStr = sprintf "%A" node.Kind
                let typeStr = sprintf "%A" node.Type
                let contextInfo =
                    match node.Kind with
                    | SemanticKind.VarRef (name, Some bindingId) ->
                        // For VarRef: show what the binding resolves to
                        match SemanticGraph.tryGetNode bindingId ctx.Graph with
                        | Some bindingNode ->
                            let bindingChildKind =
                                bindingNode.Children
                                |> List.tryHead
                                |> Option.bind (fun cid -> SemanticGraph.tryGetNode cid ctx.Graph)
                                |> Option.map (fun cn -> sprintf "%A" cn.Kind |> fun s -> s.Split('\n').[0])
                                |> Option.defaultValue "no children"
                            sprintf "VarRef '%s' -> Binding %d (child: %s). Type: %s" name (NodeId.value bindingId) bindingChildKind typeStr
                        | None ->
                            sprintf "VarRef '%s' -> Binding %d (not found in graph). Type: %s" name (NodeId.value bindingId) typeStr
                    | _ ->
                        sprintf "Kind: %s. Type: %s" (kindStr.Split('\n').[0]) typeStr
                WitnessOutput.error (sprintf "No witness handled node %A — %s" node.Id contextInfo)
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
    (rootScope: ref<ScopeContext>)
    (globalVisited: ref<Set<NodeId>>)
    : unit =

    // Create combined witness that tries all nanopasses at each node
    let combinedWitness = combineWitnesses nanopasses

    // Process a single structural root node
    let processRoot (nodeId: NodeId) =
        if not (Set.contains nodeId !globalVisited) then
            match SemanticGraph.tryGetNode nodeId graph with
            | Some node when node.IsReachable ->
                if traceTraversal then printfn "[DEBUG] Processing root node %d (%A)" (NodeId.value nodeId) node.Kind
                match PSGZipper.create graph nodeId with
                | None -> ()
                | Some initialZipper ->
                    let nodeCtx = {
                        Graph = graph
                        Coeffects = coeffects
                        Accumulator = sharedAcc
                        RootAccumulator = sharedAcc
                        ScopeContext = rootScope
                        RootScopeContext = rootScope
                        Zipper = initialZipper
                        GlobalVisited = globalVisited
                    }
                    visitAllNodes combinedWitness nodeCtx node globalVisited
            | _ -> ()

    // Visit structural roots from ModuleClassifications (semantically-ordered, not NodeId-ordered)
    // This decouples traversal order from NodeId allocation order.
    let classifications = graph.ModuleClassifications.Value
    for kvp in classifications do
        let moduleDefId = kvp.Key
        let classification = kvp.Value
        // Module-init first (prologue bindings)
        for nodeId in classification.ModuleInit do
            processRoot nodeId
        // Then definitions in source order (includes entry point)
        for nodeId in classification.Definitions do
            processRoot nodeId
        // Finally the ModuleDef node itself (for coverage validation)
        processRoot moduleDefId

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

        // Create root scope for operation accumulation
        let rootScope = ref (ScopeContext.root())

        if traceTraversal then printfn "[Alex] Single-phase execution: %d registered nanopasses" (List.length registry.Nanopasses)

        // Run all nanopasses together in single traversal
        runAllNanopasses registry.Nanopasses graph coeffects sharedAcc rootScope globalVisited

        // TODO: Serialize results if intermediatesDir provided

        // Coverage validation - ensure all reachable nodes were witnessed
        let coverageDiagnostics = CoverageValidation.validateCoverage graph !globalVisited
        if not (List.isEmpty coverageDiagnostics) then
            // Add coverage errors to accumulator
            for diag in coverageDiagnostics do
                MLIRAccumulator.addError diag sharedAcc

        // Extract operations from root scope and add to accumulator (Phase 7)
        let rootOps = ScopeContext.getOps !rootScope
        if traceTraversal then printfn "[DEBUG] Extracted %d operations from rootScope" (List.length rootOps)
        MLIRAccumulator.addOps rootOps sharedAcc

        sharedAcc
