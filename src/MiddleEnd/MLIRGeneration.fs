/// MLIRGeneration - MiddleEnd orchestration layer
///
/// Firefly Pipeline Context:
///   FrontEnd (FNCS) → PSG → MiddleEnd → MLIR text → BackEnd (mliropt/LLVM)
///
/// This module is the PUBLIC API for the MiddleEnd. It orchestrates:
///   1. PSGElaboration: Compute coeffects (SSA, mutability, yields, etc.)
///   2. Alex transfer: Witnesses traverse PSG → structured MLIROp
///   3. Serialization: MLIROp → MLIR text (exit point)
///
/// Clean signature: PSG + PlatformContext → MLIR text
module MiddleEnd.MLIRGeneration

open System.IO
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.MLIRTransfer

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MLIR from PSG
/// This is the single entry point for the MiddleEnd
let generate
    (graph: SemanticGraph)
    (platformCtx: PlatformContext)
    (outputKind: Core.Types.Dialects.OutputKind)
    (intermediatesDir: string option)
    : Result<string, string> =

    // Parse platform info from FNCS context
    let (os, arch) =
        match platformCtx.PlatformId with
        | id when id.Contains("Linux") && id.Contains("x86_64") -> (Linux, X86_64)
        | id when id.Contains("Linux") && id.Contains("ARM64") -> (Linux, ARM64)
        | id when id.Contains("Windows") && id.Contains("x86_64") -> (Windows, X86_64)
        | id when id.Contains("MacOS") && id.Contains("x86_64") -> (MacOS, X86_64)
        | id when id.Contains("MacOS") && id.Contains("ARM64") -> (MacOS, ARM64)
        | _ -> (Linux, X86_64)

    let runtimeMode =
        match outputKind with
        | Core.Types.Dialects.OutputKind.Freestanding -> PSGElaboration.PlatformConfig.Freestanding
        | Core.Types.Dialects.OutputKind.Console -> PSGElaboration.PlatformConfig.Console
        | Core.Types.Dialects.OutputKind.Library -> PSGElaboration.PlatformConfig.Console
        | Core.Types.Dialects.OutputKind.Embedded -> PSGElaboration.PlatformConfig.Freestanding

    // Phase 0: Flatten curried lambdas (graph normalization BEFORE coeffect analysis)
    // Merges Lambda(a) → Lambda(b) → body into Lambda(a,b) → body
    // and identifies partial application / saturated call patterns
    let (flattenedGraph, absorbedLambdas) = PSGElaboration.CurryFlattening.flatten graph
    let curryFlatteningResult = PSGElaboration.CurryFlattening.analyze flattenedGraph absorbedLambdas

    // Compute effective arg counts for saturated calls (SSAAssignment needs correct SSA counts)
    let saturatedCallArgCounts =
        curryFlatteningResult.SaturatedCalls
        |> Map.map (fun _ info -> List.length info.AllArgNodes)

    // Compute coeffects on flattened graph (SSAs reflect flattened parameter structure)
    let ssaAssignment = PSGElaboration.SSAAssignment.assignSSA arch flattenedGraph saturatedCallArgCounts
    let mutability = PSGElaboration.MutabilityAnalysis.analyze flattenedGraph
    let yieldStates = PSGElaboration.YieldStateIndices.run flattenedGraph
    let patternBindings = PSGElaboration.PatternBindingAnalysis.analyze flattenedGraph
    let strings = PSGElaboration.StringCollection.collect flattenedGraph
    let platformResolution = PSGElaboration.PlatformBindingResolution.analyze flattenedGraph runtimeMode os arch
    let escapeAnalysis = PSGElaboration.EscapeAnalysis.analyzeGraph flattenedGraph

    // Serialize coeffects if keeping intermediates
    match intermediatesDir with
    | Some dir ->
        PSGElaboration.PreprocessingSerializer.serializeAll
            dir ssaAssignment mutability yieldStates patternBindings strings
            ssaAssignment.EntryPointLambdas flattenedGraph
    | None -> ()

    // Build TransferCoeffects
    let coeffects : TransferCoeffects = {
        SSA = ssaAssignment
        Platform = platformResolution
        Mutability = mutability
        PatternBindings = patternBindings
        Strings = strings
        YieldStates = yieldStates
        EscapeAnalysis = escapeAnalysis
        CurryFlattening = curryFlatteningResult
        EntryPointLambdaIds = ssaAssignment.EntryPointLambdas
    }

    // Execute Alex transfer (parallel nanopasses)
    match flattenedGraph.EntryPoints with
    | [] -> Result.Error "No entry points found in PSG"
    | entryId :: _ ->
        match transfer flattenedGraph entryId coeffects intermediatesDir with
        | Result.Ok (topLevelOps, _) ->
            // Apply MLIR nanopasses (MLIR→MLIR transformations)
            // This is the integration point for dual witness infrastructure:
            // - PSG witnesses emit portable MLIR (memref, func.call)
            // - MLIR nanopasses transform for backends (FFI conversion, DCont/Inet lowering)
            let transformedOps = Alex.Pipeline.MLIRNanopass.applyPasses topLevelOps platformResolution intermediatesDir

            // Serialize MLIROp → MLIR text (exit point of MiddleEnd)
            let mlirText = moduleToString "main" transformedOps

            // Write final MLIR output (renamed to 10_output.mlir for nanopass visibility)
            match intermediatesDir with
            | Some dir ->
                let finalPath = Path.Combine(dir, "10_output.mlir")
                File.WriteAllText(finalPath, mlirText)
                printfn "[Alex] Wrote final MLIR: 10_output.mlir"
            | None -> ()

            Result.Ok mlirText
        | Result.Error msg -> Result.Error msg
