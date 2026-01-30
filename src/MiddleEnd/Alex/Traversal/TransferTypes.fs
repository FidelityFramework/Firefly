/// Transfer Types - Core types for MLIR Transfer
///
/// CANONICAL ARCHITECTURE (January 2026):
/// This file defines the types that witnesses receive. It compiles BEFORE
/// witnesses so they can elegantly take `ctx: WitnessContext` rather than
/// explicit parameter threading.
///
/// The Three Concerns:
/// - PSGZipper: Pure navigation (Focus, Path, Graph) - defined in PSGZipper.fs
/// - TransferCoeffects: Pre-computed, immutable coeffects
/// - MLIRAccumulator: Mutable fold state
///
/// See: mlir_transfer_canonical_architecture memory
module Alex.Traversal.TransferTypes

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES (for type definitions)
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER COEFFECTS (Pre-computed, Immutable)
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed coeffects - computed ONCE before traversal, NEVER modified
type TransferCoeffects = {
    SSA: SSAAssign.SSAAssignment
    Platform: PlatformResolutionResult
    Mutability: MutAnalysis.MutabilityAnalysisResult
    PatternBindings: PatternAnalysis.PatternBindingAnalysisResult
    Strings: StringCollect.StringTable
    YieldStates: YieldStateIndices.YieldStateCoeffect
    EntryPointLambdaIds: Set<int>
}

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTION TRACE (For Debugging Pattern Failures)
// ═══════════════════════════════════════════════════════════════════════════

/// Execution trace entry - records each step in pattern execution
type ExecutionTrace = {
    /// Hierarchy depth: 0=Witness, 1=Pattern, 2=Element
    Depth: int
    
    /// Component name: "LiteralWitness", "pBuildStringLiteral", "pAddressOf"
    ComponentName: string
    
    /// PSG NodeId (if witness-level, otherwise None)
    NodeId: NodeId option
    
    /// Serialized parameters for inspection
    Parameters: string
    
    /// Sequential execution order
    Timestamp: int
}

module ExecutionTrace =
    /// Format trace entry for display
    let format (trace: ExecutionTrace) : string =
        let indent = String.replicate trace.Depth "  "
        let nodeInfo = match trace.NodeId with Some nid -> sprintf "[Node %d] " (NodeId.value nid) | None -> ""
        sprintf "%s%s%s(%s)" indent nodeInfo trace.ComponentName trace.Parameters

/// Trace collector - mutable accumulator for execution traces
type TraceCollector = ResizeArray<ExecutionTrace>

module TraceCollector =
    let create () : TraceCollector = ResizeArray<ExecutionTrace>()
    
    let add (depth: int) (componentName: string) (nodeId: NodeId option) (parameters: string) (collector: TraceCollector) =
        collector.Add({
            Depth = depth
            ComponentName = componentName
            NodeId = nodeId
            Parameters = parameters
            Timestamp = collector.Count
        })
    
    let toList (collector: TraceCollector) : ExecutionTrace list =
        collector |> Seq.toList

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURED DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Diagnostic severity levels
type DiagnosticSeverity =
    | Error
    | Warning
    | Info

/// Structured diagnostic capturing WHERE and WHAT went wrong
type Diagnostic = {
    /// Severity level
    Severity: DiagnosticSeverity

    /// NodeId where error occurred (if known)
    NodeId: NodeId option

    /// Source component (e.g., "Literal", "Arithmetic", "ControlFlow")
    Source: string option

    /// Phase/operation that failed (e.g., "pBuildStringLiteral", "SSA lookup")
    Phase: string option

    /// Human-readable message
    Message: string

    /// Optional: Expected vs Actual for validation errors
    Details: (string * string) option
}

module Diagnostic =
    /// Create an error diagnostic with full context
    let error nodeId source phase message =
        { Severity = Error
          NodeId = nodeId
          Source = source
          Phase = phase
          Message = message
          Details = None }

    /// Create an error diagnostic with just a message
    let errorSimple message =
        error None None None message

    /// Create an error diagnostic with expected/actual details
    let errorWithDetails nodeId source phase message expected actual =
        { Severity = Error
          NodeId = nodeId
          Source = source
          Phase = phase
          Message = message
          Details = Some (expected, actual) }

    /// Format diagnostic to human-readable string
    let format (diag: Diagnostic) : string =
        let parts = [
            // Severity
            match diag.Severity with
            | Error -> Some "[ERROR]"
            | Warning -> Some "[WARNING]"
            | Info -> Some "[INFO]"

            // NodeId
            match diag.NodeId with
            | Some nid -> Some (sprintf "Node %d" (NodeId.value nid))
            | None -> None

            // Source
            match diag.Source with
            | Some src -> Some (sprintf "(%s)" src)
            | None -> None

            // Phase
            match diag.Phase with
            | Some phase -> Some (sprintf "in %s" phase)
            | None -> None

            // Message
            Some diag.Message

            // Details
            match diag.Details with
            | Some (expected, actual) ->
                Some (sprintf "Expected: %s, Actual: %s" expected actual)
            | None -> None
        ]
        parts
        |> List.choose id
        |> String.concat " "

// ═══════════════════════════════════════════════════════════════════════════
// MLIR ACCUMULATOR (Mutable Fold State)
// ═══════════════════════════════════════════════════════════════════════════

/// Flat accumulator - all operations in single stream with scope markers
/// SSA bindings are global (shared across all witnesses and scopes)
/// NOTE: Visited set is NOT in accumulator - each nanopass gets its own visited set
type MLIRAccumulator = {
    mutable AllOps: MLIROp list                      // Flat operation stream with markers
    mutable Errors: Diagnostic list
    mutable NodeAssoc: Map<NodeId, SSA * MLIRType>  // Global SSA bindings (PSG nodes)
    mutable MLIRTempCounter: int                      // For MLIR-level temporary SSAs (NOT PSG nodes)

    // Witnessing Coordination State (Dependent Transparency)
    mutable EmittedGlobals: Set<string>              // Track emitted global strings (by symbol name)
    // NOTE: Function declarations now handled by MLIR Declaration Collection Pass (no coordination needed)
}

module MLIRAccumulator =
    let empty () : MLIRAccumulator =
        {
            AllOps = []
            Errors = []
            NodeAssoc = Map.empty
            MLIRTempCounter = 0
            EmittedGlobals = Set.empty
        }

    /// Add a single operation to the flat stream
    let addOp (op: MLIROp) (acc: MLIRAccumulator) =
        acc.AllOps <- op :: acc.AllOps

    /// Add multiple operations to the flat stream
    let addOps (ops: MLIROp list) (acc: MLIRAccumulator) =
        acc.AllOps <- List.rev ops @ acc.AllOps

    /// Add an error diagnostic
    let addError (err: Diagnostic) (acc: MLIRAccumulator) =
        acc.Errors <- err :: acc.Errors

    /// Bind a PSG node to its SSA value (global binding)
    let bindNode (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.NodeAssoc <- Map.add nodeId (ssa, ty) acc.NodeAssoc

    /// Recall the SSA binding for a PSG node (global lookup)
    let recallNode (nodeId: NodeId) (acc: MLIRAccumulator) =
        Map.tryFind nodeId acc.NodeAssoc

    // ═══════════════════════════════════════════════════════════
    // WITNESSING COORDINATION (Dependent Transparency Support)
    // ═══════════════════════════════════════════════════════════

    /// Try to emit a global string (returns Some op if not already emitted, None if duplicate)
    /// This implements dependent transparency coordination: witnesses check before emitting module-level declarations
    let tryEmitGlobal (name: string) (content: string) (byteLength: int) (acc: MLIRAccumulator) : MLIROp option =
        if Set.contains name acc.EmittedGlobals then
            None  // Already emitted by another witness
        else
            acc.EmittedGlobals <- Set.add name acc.EmittedGlobals
            Some (MLIROp.GlobalString (name, content, byteLength))

    /// NOTE: Function declaration coordination removed - now handled by MLIR Declaration Collection Pass
    /// This eliminates "first witness wins" race condition and separates concerns:
    /// - Witnesses emit FuncCall operations (codata)
    /// - Declaration Collection Pass analyzes calls and emits FuncDecl (structural MLIR transformation)

    /// NOTE: Scope markers removed - single-phase execution with nested accumulators
    /// Scope-owning witnesses (Lambda, ControlFlow) create nested accumulators for body operations.
    /// Operations naturally nest; bindings remain global for cross-scope lookups.

    /// Generate fresh SSA for MLIR-level temporary (not associated with PSG node)
    /// Used for implementation-level operations like FFI marshaling
    /// Uses high numbers (10000+) to avoid collision with PSG-assigned SSAs
    let freshMLIRTemp (acc: MLIRAccumulator) : SSA =
        let n = acc.MLIRTempCounter
        acc.MLIRTempCounter <- n + 1
        SSA.V (10000 + n)

    /// Backward compatibility aliases
    let addTopLevelOp = addOp
    let addTopLevelOps = addOps

    /// Property accessor for compatibility
    let topLevelOps (acc: MLIRAccumulator) = acc.AllOps

    /// Recursively count all operations (including nested in FuncDef, SCFOp, etc.)
    let rec countOperations (ops: MLIROp list) : int =
        ops |> List.sumBy (fun op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
                1 + countOperations body
            | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps)) ->
                let elseCount = match elseOps with Some ops -> countOperations ops | None -> 0
                1 + countOperations thenOps + elseCount
            | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
                1 + countOperations condOps + countOperations bodyOps
            | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) ->
                1 + countOperations bodyOps
            | MLIROp.Block (_, blockOps) ->
                1 + countOperations blockOps
            | MLIROp.Region ops ->
                1 + countOperations ops
            | _ -> 1)

    /// Get total operation count from accumulator (including all nested operations)
    let totalOperations (acc: MLIRAccumulator) : int =
        countOperations acc.AllOps

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER RESULT (Result of witnessing a node)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of witnessing a PSG node
type TransferResult =
    | TRValue of Val                    // Produces a value (SSA + type)
    | TRVoid                             // Produces no value (effect only)
    | TRError of Diagnostic              // Error with structured context

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS OUTPUT (What witnesses return)
// ═══════════════════════════════════════════════════════════════════════════

/// Codata returned by witnesses
type WitnessOutput = {
    InlineOps: MLIROp list
    TopLevelOps: MLIROp list
    Result: TransferResult
}

module WitnessOutput =
    let empty = { InlineOps = []; TopLevelOps = []; Result = TRVoid }
    let inline' ops result = { InlineOps = ops; TopLevelOps = []; Result = result }
    let value v = { InlineOps = []; TopLevelOps = []; Result = TRValue v }

    /// Create error output with simple message
    let error msg = { InlineOps = []; TopLevelOps = []; Result = TRError (Diagnostic.errorSimple msg) }

    /// Create error output with full diagnostic context
    let errorDiag diag = { InlineOps = []; TopLevelOps = []; Result = TRError diag }

    /// Skip this node (not handled by this nanopass)
    let skip = empty

    let withTopLevel topOps (output: WitnessOutput) : WitnessOutput = 
        { output with TopLevelOps = topOps @ output.TopLevelOps }
    let combine (a: WitnessOutput) (b: WitnessOutput) =
        { InlineOps = a.InlineOps @ b.InlineOps
          TopLevelOps = a.TopLevelOps @ b.TopLevelOps
          Result = b.Result }
    let combineAll outputs = outputs |> List.fold combine empty

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS CONTEXT (What witnesses receive)
// ═══════════════════════════════════════════════════════════════════════════

/// Context passed to witnesses - the elegant single parameter
type WitnessContext = {
    Coeffects: TransferCoeffects
    Accumulator: MLIRAccumulator     // Current scope's accumulator (changes for nested scopes)
    RootAccumulator: MLIRAccumulator // Root/module-level accumulator (constant across all scopes)
    Graph: SemanticGraph
    Zipper: PSGZipper                // Navigation state (created ONCE by fold)
    GlobalVisited: ref<Set<NodeId>>  // Global visited set (shared across all nanopasses and function bodies)
}

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Convenience functions)
// ═══════════════════════════════════════════════════════════════════════════

/// Get single pre-assigned SSA for a node
let requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get all pre-assigned SSAs for a node
let requireSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssign.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get target architecture from coeffects
let targetArch (ctx: WitnessContext) : Architecture =
    ctx.Coeffects.Platform.TargetArch

/// CANONICAL TYPE MAPPING - the ONLY way to map NativeType to MLIRType
/// Uses graph-aware mapping that correctly handles records by looking up
/// field types from TypeDef nodes. ALL type mapping should go through this.
let mapType (ty: NativeType) (ctx: WitnessContext) : MLIRType =
    mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph ty

/// Get platform-aware word width for string length, array length, etc.
let wordWidth (ctx: WitnessContext) : IntWidth =
    platformWordWidth ctx.Coeffects.Platform.TargetArch
