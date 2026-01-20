module Alex.Pipeline.CompilationTypes

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open System

// ===================================================================
// Compilation Pipeline Types
// ===================================================================

/// Phases of the compilation pipeline
type CompilationPhase =
    | Initialization
    | Parsing
    | TypeChecking
    | SymbolExtraction
    | ReachabilityAnalysis
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | IntermediateGeneration
    | Finalization

/// Error severity levels
type ErrorSeverity =
    | Error
    | Warning
    | Info

/// Compiler error with context
type CompilerError = {
    Phase: string
    Message: string
    Location: string option
    Severity: ErrorSeverity
}

/// Compilation statistics
type CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}

/// Empty statistics for initialization
module CompilationStatistics =
    let empty = {
        TotalFiles = 0
        TotalSymbols = 0
        ReachableSymbols = 0
        EliminatedSymbols = 0
        CompilationTimeMs = 0.0
    }

/// Configuration for the compilation pipeline
type CompilationConfig = {
    EnableClosureElimination: bool
    EnableStackAllocation: bool
    EnableReachabilityAnalysis: bool
    PreserveIntermediateASTs: bool
    VerboseOutput: bool
}

/// Progress callback for reporting compilation status
type ProgressCallback = CompilationPhase -> string -> unit

// ===================================================================
// Project Loading Types (for FCS integration)
// ===================================================================

/// Project loading phase (subset of compilation phases)
type ProjectLoadingPhase =
    | ProjectLoading
    | FCSProcessing
    | SymbolCollection
    | IntermediateGeneration

// Note: Removed projectPhaseToCompilationPhase as it was causing type issues
// The phases are handled differently in the orchestrator

// ===================================================================
// Emission Error Collection
// ===================================================================

/// An error that occurred during MLIR emission
type EmissionError = {
    NodeKind: string
    NodeSymbol: string option
    Message: string
    Phase: CompilationPhase
}

/// Mutable collector for emission errors
/// Used during emission to accumulate errors while continuing to emit
/// what we can (for intermediate file generation)
type EmissionErrorCollector() =
    let mutable errors: EmissionError list = []

    member _.Add(error: EmissionError) =
        errors <- error :: errors

    member _.AddError(nodeKind: string, symbol: string option, message: string) =
        errors <- { NodeKind = nodeKind; NodeSymbol = symbol; Message = message; Phase = MLIRGeneration } :: errors

    member _.Errors = List.rev errors

    member _.HasErrors = not (List.isEmpty errors)

    member _.Clear() = errors <- []

    /// Convert to CompilerError list for reporting
    member _.ToCompilerErrors() : CompilerError list =
        errors
        |> List.rev
        |> List.map (fun e ->
            {
                Phase = "MLIR Emission"
                Message = sprintf "[%s] %s%s"
                    e.NodeKind
                    (match e.NodeSymbol with Some s -> s + ": " | None -> "")
                    e.Message
                Location = None
                Severity = ErrorSeverity.Error
            })

/// Global emission error collector (thread-local in future if needed)
module EmissionErrors =
    let mutable private collector = EmissionErrorCollector()

    let reset() = collector <- EmissionErrorCollector()
    let add nodeKind symbol message = collector.AddError(nodeKind, symbol, message)
    let errors() = collector.Errors
    let hasErrors() = collector.HasErrors
    let toCompilerErrors() = collector.ToCompilerErrors()