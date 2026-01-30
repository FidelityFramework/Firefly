/// MLIRTransfer - Parallel Nanopass Orchestration
///
/// This file orchestrates parallel witness execution via the nanopass registry.
/// NO manual dispatch - each witness is a nanopass that runs in parallel.
///
/// ARCHITECTURE: Witnesses register themselves in WitnessRegistry. This module
/// simply calls executeNanopasses to run all witnesses in parallel and collect results.

module Alex.Traversal.MLIRTransfer

open System.IO
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.WitnessRegistry
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer PSG to MLIR via parallel nanopass execution
///
/// ARCHITECTURE:
/// 1. Initialize witness registry (all witnesses register their nanopasses)
/// 2. Execute all nanopasses in parallel via IcedTasks
/// 3. Reactive envelope collects results as they arrive (random order OK)
/// 4. Return cohesive MLIR graph
///
/// NO DISPATCH LOGIC - Each nanopass traverses the entire PSG, witnessing only
/// its category of nodes and returning WitnessOutput.skip for others.
let transfer
    (graph: SemanticGraph)
    (entryNodeId: NodeId)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : Result<MLIROp list * MLIROp list, string> =

    // Initialize witness registry (populates all migrated nanopasses)
    initializeRegistry()

    // Check entry point exists
    match SemanticGraph.tryGetNode entryNodeId graph with
    | None ->
        Result.Error (sprintf "Entry node %d not found" (NodeId.value entryNodeId))
    | Some _ ->
        // Execute all nanopasses in single-phase traversal
        let accumulator = executeNanopasses globalRegistry graph coeffects intermediatesDir

        // Debug: Print accumulator stats with recursive counts
        let totalOps = MLIRAccumulator.totalOperations accumulator
        let allOpsCount = List.length accumulator.AllOps
        printfn "[MLIRTransfer] Accumulator: %d total ops in %d stream items, %d errors"
            totalOps
            allOpsCount
            (List.length accumulator.Errors)

        // Strip scope markers and prepare operations for output
        let cleanedOps =
            accumulator.AllOps
            |> List.rev

        // Write partial MLIR to intermediate file for debugging (even with errors)
        match intermediatesDir with
        | Some dir ->
            let mlirText = moduleToString "main" cleanedOps
            let mlirPath = Path.Combine(dir, "07_output.mlir")
            File.WriteAllText(mlirPath, mlirText)
        | None -> ()

        // Check for errors accumulated during nanopass execution
        match accumulator.Errors with
        | [] ->
            // Success - return accumulated MLIR operations
            Result.Ok (cleanedOps, [])
        | errors ->
            // Errors occurred - format and report them (MLIR already written above)
            let formattedErrors = errors |> List.map Diagnostic.format |> String.concat "\n"
            Result.Error formattedErrors

