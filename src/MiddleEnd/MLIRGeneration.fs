/// MLIRGeneration - MiddleEnd entry point
///
/// Clean public API: PSG + PlatformContext → MLIR text
/// Internally orchestrates PSGElaboration coeffects + Alex transfer
module MiddleEnd.MLIRGeneration

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
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
    | [] -> Error "No entry points found in PSG"
    | entryId :: _ ->
        match transfer graph entryId coeffects intermediatesDir with
        | Ok (topLevelOps, _) ->
            // Serialize MLIR ops to text
            let mlirText = sprintf "module {\n  // %d MLIR ops generated\n}\n" (List.length topLevelOps)
            Ok mlirText
        | Error msg -> Error msg
