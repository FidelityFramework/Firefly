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

    // Compute coeffects (PSGElaboration)
    let ssaAssignment = PSGElaboration.SSAAssignment.assignSSA arch graph
    let mutability = PSGElaboration.MutabilityAnalysis.analyze graph
    let yieldStates = PSGElaboration.YieldStateIndices.run graph
    let patternBindings = PSGElaboration.PatternBindingAnalysis.analyze graph
    let strings = PSGElaboration.StringCollection.collect graph
    let platformResolution = PSGElaboration.PlatformBindingResolution.analyze graph runtimeMode os arch

    // Serialize coeffects if keeping intermediates
    match intermediatesDir with
    | Some dir ->
        PSGElaboration.PreprocessingSerializer.serializeAll
            dir ssaAssignment mutability yieldStates patternBindings strings
            ssaAssignment.EntryPointLambdas graph
    | None -> ()

    // Build TransferCoeffects
    let coeffects : TransferCoeffects = {
        SSA = ssaAssignment
        Platform = platformResolution
        Mutability = mutability
        PatternBindings = patternBindings
        Strings = strings
        YieldStates = yieldStates
        EntryPointLambdaIds = ssaAssignment.EntryPointLambdas
    }

    // Execute Alex transfer (parallel nanopasses)
    match graph.EntryPoints with
    | [] -> Result.Error "No entry points found in PSG"
    | entryId :: _ ->
        match transfer graph entryId coeffects intermediatesDir with
        | Result.Ok (topLevelOps, _) ->
            // Apply MLIR nanopasses (MLIR→MLIR transformations)
            // This is the integration point for dual witness infrastructure:
            // - PSG witnesses emit portable MLIR (memref, func.call)
            // - MLIR nanopasses transform for backends (FFI conversion, DCont/Inet lowering)
            let transformedOps = Alex.Pipeline.MLIRNanopass.applyPasses topLevelOps platformResolution

            // Serialize MLIROp → MLIR text (exit point of MiddleEnd)
            let mlirText = moduleToString "main" transformedOps
            Result.Ok mlirText
        | Result.Error msg -> Result.Error msg
