/// ParallelNanopass - IcedTasks orchestration for parallel nanopasses
///
/// Runs multiple nanopasses in parallel via IcedTasks.ColdTask
/// Collects results in envelope pass
/// Overlays into cohesive MLIR graph
module Alex.Traversal.ParallelNanopass

open System.IO
open System.Text.Json
open IcedTasks
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════════════════════
// PARALLEL EXECUTION (Reactive - results collected as they arrive)
// ═══════════════════════════════════════════════════════════════════════════

// DESIGN DECISION: Full-fat parallel execution without pre-discovery
//
// We run ALL registered nanopasses in parallel, even if some will be empty
// (traverse and skip all nodes). This is the simple, correct approach.
//
// FUTURE OPTIMIZATION: For very large projects (10,000+ nodes), a discovery
// pass could scan node types and filter nanopasses before parallel execution.
// This optimization is intentionally deferred until profiling shows need.
//
// Trade-off: Empty nanopass cost (~microseconds for HelloWorld) vs. discovery
// pass cost (another full traversal). Current strategy favors simplicity.

/// Run multiple nanopasses in parallel via IcedTasks
/// Results collected AS THEY COMPLETE (order doesn't matter)
let runNanopassesParallel
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator list =

    // Fan out: Execute all nanopasses in parallel using IcedTasks coldTask
    // ColdTask is unit -> Task<'T>, so we create them, then invoke all with Task.WhenAll
    nanopasses
    |> List.map (fun nanopass ->
        coldTask { return runNanopass nanopass graph coeffects })
    |> List.map (fun ct -> ct())  // Invoke all coldTasks to start them
    |> fun tasks -> System.Threading.Tasks.Task.WhenAll(tasks).GetAwaiter().GetResult()
    |> List.ofArray

/// Run multiple nanopasses sequentially (for debugging/validation)
let runNanopassesSequential
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator list =

    nanopasses
    |> List.map (fun nanopass -> runNanopass nanopass graph coeffects)

// ═══════════════════════════════════════════════════════════════════════════
// ENVELOPE PASS (Reactive Result Collection)
// ═══════════════════════════════════════════════════════════════════════════

/// Envelope pass: Reactively collect results AS THEY ARRIVE
/// Since nanopasses are referentially transparent and merge is associative,
/// order doesn't matter - merge results as they complete
let collectEnvelopeReactive (nanopassResults: MLIRAccumulator list) : MLIRAccumulator =
    match nanopassResults with
    | [] -> MLIRAccumulator.empty()
    | [single] -> single
    | many ->
        // Overlay all results associatively
        // Order of arrival doesn't matter (associativity + referential transparency)
        many
        |> List.reduce overlayAccumulators

/// Envelope pass (alias for backwards compatibility)
let collectEnvelope = collectEnvelopeReactive

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ORCHESTRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for parallel execution
type ParallelConfig = {
    /// Enable parallel execution (false for debugging)
    EnableParallel: bool

    /// FUTURE: Enable discovery pass for large projects
    /// Threshold where pre-scanning node types becomes cheaper than empty traversals
    /// Current: Always false (full-fat parallel execution)
    EnableDiscovery: bool

    /// FUTURE: Node count threshold to trigger discovery
    /// Current: Not used (discovery disabled)
    DiscoveryThreshold: int
}

let defaultConfig = {
    EnableParallel = true
    EnableDiscovery = false  // Deferred optimization
    DiscoveryThreshold = 10000  // Placeholder for future tuning
}

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS RESULT SERIALIZATION (for -k flag debugging)
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize a nanopass result to JSON for debugging
let private serializeNanopassResult (intermediatesDir: string) (nanopass: Nanopass) (accumulator: MLIRAccumulator) : unit =
    let fileName = sprintf "07_%s_witness.json" (nanopass.Name.ToLower())
    let filePath = Path.Combine(intermediatesDir, fileName)

    let summary = {|
        NanopassName = nanopass.Name
        OperationCount = List.length accumulator.TopLevelOps
        ErrorCount = List.length accumulator.Errors
        VisitedNodes = accumulator.Visited |> Set.toList |> List.map (fun nodeId -> NodeId.value nodeId)
        Errors = accumulator.Errors
        Operations = accumulator.TopLevelOps |> List.map (fun op -> sprintf "%A" op)
    |}

    let json = JsonSerializer.Serialize(summary, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(filePath, json)
    printfn "[Alex] Wrote nanopass result: %s" fileName

/// Main entry point: Run all nanopasses and collect results
let executeNanopasses
    (config: ParallelConfig)
    (registry: NanopassRegistry)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : MLIRAccumulator =

    if List.isEmpty registry.Nanopasses then
        // No nanopasses registered - empty result
        MLIRAccumulator.empty()
    else
        // FUTURE: Discovery optimization (currently disabled)
        // if config.EnableDiscovery && graph.Nodes.Count > config.DiscoveryThreshold then
        //     let presentKinds = discoverPresentNodeTypes graph
        //     let filteredRegistry = filterRelevantNanopasses registry presentKinds
        //     registry <- filteredRegistry

        // Run nanopasses (parallel or sequential)
        // Current: ALL registered nanopasses run (full-fat strategy)
        let nanopassResults =
            if config.EnableParallel then
                runNanopassesParallel registry.Nanopasses graph coeffects
            else
                runNanopassesSequential registry.Nanopasses graph coeffects

        // Serialize each nanopass result for debugging (if keeping intermediates)
        match intermediatesDir with
        | Some dir ->
            List.zip registry.Nanopasses nanopassResults
            |> List.iter (fun (nanopass, accumulator) ->
                serializeNanopassResult dir nanopass accumulator)
        | None -> ()

        // Envelope pass: Collect and overlay results
        collectEnvelope nanopassResults
