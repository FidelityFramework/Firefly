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
open Alex.Traversal.ScopeContext

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES (for type definitions)
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices
module EscapeAnalysis = PSGElaboration.EscapeAnalysis
module CurryFlat = PSGElaboration.CurryFlattening

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
    EscapeAnalysis: EscapeAnalysis.EscapeAnalysisResult
    CurryFlattening: CurryFlat.CurryFlatteningResult
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
/// CRITICAL: This is a CLASS (reference type) not a record, so mutations propagate correctly
[<AllowNullLiteral>]
type MLIRAccumulator() =
    member val AllOps: MLIROp list = [] with get, set                      // Flat operation stream with markers
    member val Errors: Diagnostic list = [] with get, set
    member val NodeAssoc: Map<NodeId, SSA * MLIRType> = Map.empty with get, set  // Global SSA bindings (PSG nodes)
    member val SSATypes: Map<SSA, MLIRType> = Map.empty with get, set            // SSA → type reverse index (for monadic type derivation in Elements)
    member val MLIRTempCounter: int = 0 with get, set                      // For MLIR-level temporary SSAs (NOT PSG nodes)

    // Witnessing Coordination State (Dependent Transparency)
    member val EmittedGlobals: Set<string> = Set.empty with get, set              // Track emitted global strings (by symbol name)
    // NOTE: Function declarations now handled by MLIR Declaration Collection Pass (no coordination needed)

    // Deferred InlineOps: Partial app arguments whose InlineOps are suppressed at their
    // original scope and re-emitted at the saturated call site (MLIR region isolation)
    member val DeferredInlineOps: System.Collections.Generic.Dictionary<int, MLIROp list> = System.Collections.Generic.Dictionary<int, MLIROp list>() with get

module MLIRAccumulator =
    let empty () : MLIRAccumulator =
        MLIRAccumulator()

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
    /// Also populates SSATypes reverse index for monadic type derivation in Elements
    let bindNode (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.NodeAssoc <- Map.add nodeId (ssa, ty) acc.NodeAssoc
        // Preserve physical SSA type if already registered by an Element (pAlloca, pAlloc, etc.)
        // Elements register physical types (TMemRefStatic from alloca); bindNode carries semantic types
        // (TMemRef for mutable cells). pLoad/pLoadFrom derive memrefType from SSATypes.
        match Map.tryFind ssa acc.SSATypes with
        | Some existingTy when existingTy <> ty ->
            // Type collision detected — same SSA registered with different type.
            // Benign case: TMemRefStatic(n, elem) vs TMemRef(elem) — physical vs semantic type.
            // Elements (pAlloca) register physical TMemRefStatic; bindNode carries semantic TMemRef.
            // Physical type is correct for pLoadFrom — preserve it silently.
            // Real collision: fundamentally different types — indicates SSATypes scope leak.
            let isBenignMemRefRefinement =
                match existingTy, ty with
                | TMemRefStatic (_, elemA), TMemRef elemB when elemA = elemB -> true
                | TMemRef elemA, TMemRefStatic (_, elemB) when elemA = elemB -> true
                | _ -> false
            if not isBenignMemRefRefinement then
                let ssaStr = match ssa with | V n -> sprintf "%%v%d" n | Arg n -> sprintf "%%arg%d" n
                let diag = Diagnostic.errorWithDetails (Some nodeId) (Some "SSATypes") (Some "bindNode")
                            (sprintf "SSA type collision: %s already registered as %A, new type %A (keeping existing)" ssaStr existingTy ty)
                            (sprintf "%A" existingTy) (sprintf "%A" ty)
                acc.Errors <- diag :: acc.Errors
        | Some _ -> () // Same type — no conflict
        | None ->
            acc.SSATypes <- Map.add ssa ty acc.SSATypes

    /// Recall the SSA binding for a PSG node (global lookup)
    let recallNode (nodeId: NodeId) (acc: MLIRAccumulator) =
        Map.tryFind nodeId acc.NodeAssoc

    /// Recall the type of an SSA value (reverse index lookup)
    /// Used by Elements (e.g. pLoad) to derive memref types monadically from the accumulator
    let recallSSAType (ssa: SSA) (acc: MLIRAccumulator) =
        Map.tryFind ssa acc.SSATypes

    /// Register an SSA value's type directly (for intermediate SSAs created by Elements)
    /// Called by Elements like pAlloca/pAlloc that create new SSAs not bound to PSG nodes
    let registerSSAType (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.SSATypes <- Map.add ssa ty acc.SSATypes

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

    /// Store deferred InlineOps for a node (suppressed at original scope, re-emitted at saturated call site)
    let deferInlineOps (nodeId: NodeId) (ops: MLIROp list) (acc: MLIRAccumulator) =
        let key = NodeId.value nodeId
        acc.DeferredInlineOps.[key] <- ops

    /// Retrieve deferred InlineOps for a node (returns empty list if none)
    let getDeferredInlineOps (nodeId: NodeId) (acc: MLIRAccumulator) : MLIROp list =
        let key = NodeId.value nodeId
        match acc.DeferredInlineOps.TryGetValue(key) with
        | true, ops -> ops
        | false, _ -> []

    /// NOTE: Function declaration coordination removed - now handled by MLIR Declaration Collection Pass
    /// This eliminates "first witness wins" race condition and separates concerns:
    /// - Witnesses emit FuncCall operations (codata)
    /// - Declaration Collection Pass analyzes calls and emits FuncDecl (structural MLIR transformation)

    /// NOTE: Scope markers removed - single-phase execution with nested accumulators
    /// Scope-owning witnesses (Lambda, ControlFlow) create nested accumulators for body operations.
    /// Operations naturally nest; bindings remain global for cross-scope lookups.

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
            | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps, _)) ->
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
    | TRSkip                             // Node not handled (try next witness)

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
    let skip = { InlineOps = []; TopLevelOps = []; Result = TRSkip }

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
    Accumulator: MLIRAccumulator     // Shared for SSA bindings (global)
    RootAccumulator: MLIRAccumulator // Root/module-level accumulator (constant across all scopes)
    ScopeContext: ref<ScopeContext>  // Current scope for operation accumulation (mutable for traversal)
    RootScopeContext: ref<ScopeContext>  // Root module-level scope (constant, for TopLevelOps like GlobalString, nested FuncDef)
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
