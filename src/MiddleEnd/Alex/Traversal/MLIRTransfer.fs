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
        printfn "[MLIRTransfer] Accumulator (hash: %d): %d total ops in %d stream items, %d errors"
            (accumulator.GetHashCode())
            totalOps
            allOpsCount
            (List.length accumulator.Errors)

        // Strip scope markers and prepare operations for output  
        let cleanedOps =
            accumulator.AllOps
            |> List.rev
        
        printfn "[DEBUG] cleanedOps has %d items" (List.length cleanedOps)
        cleanedOps |> List.iteri (fun i op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) ->
                printfn "[DEBUG] cleanedOps[%d]: FuncDef %s" i name
            | MLIROp.GlobalString (name, _, _) ->
                printfn "[DEBUG] cleanedOps[%d]: GlobalString %s" i name
            | _ ->
                printfn "[DEBUG] cleanedOps[%d]: %A" i op)

        // Write partial MLIR to intermediate file for debugging (even with errors)
        printfn "[DEBUG] About to write 07_output.mlir, intermediatesDir=%A" intermediatesDir
        match intermediatesDir with
        | Some dir ->
            printfn "[DEBUG] Calling moduleToString with %d cleanedOps" (List.length cleanedOps)
            let mlirText = moduleToString "main" cleanedOps
            printfn "[DEBUG] mlirText has %d lines" (mlirText.Split('\n').Length)
            let funcDefLines = mlirText.Split('\n') |> Array.filter (fun line -> line.Contains("func.func @lambda") || line.Contains("func.func @main"))
            printfn "[DEBUG] mlirText has %d func.func lines" funcDefLines.Length
            let mlirPath = Path.Combine(dir, "07_output.mlir")
            File.WriteAllText(mlirPath, mlirText)
            printfn "[DEBUG] Wrote MLIR to %s" mlirPath
        | None -> 
            printfn "[DEBUG] intermediatesDir is None, skipping 07_output.mlir write"

        // Check for errors accumulated during nanopass execution
        match accumulator.Errors with
        | [] ->
            // Success - return accumulated MLIR operations
            Result.Ok (cleanedOps, [])
        | errors ->
            // Errors occurred - format and report them (MLIR already written above)
            let formattedErrors = errors |> List.map Diagnostic.format |> String.concat "\n"
            Result.Error formattedErrors

