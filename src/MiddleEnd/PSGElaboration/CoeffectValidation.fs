/// Coeffect Validation - Early detection of semantic errors
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Validation runs AFTER coeffect analysis, BEFORE transfer.
/// This catches structural errors early with meaningful context,
/// rather than discovering them as cryptic MLIR failures.
///
/// DESIGN FOR GROWTH:
/// This module establishes the validation PATTERN. As memory layout
/// coeffects (cache-aware, arena, sentinel strategies) grow complex,
/// this infrastructure will be ready to validate those too.
///
/// Current validations:
/// - Variable Binding Completeness: Every Variable node must have a binding source
module PSGElaboration.CoeffectValidation

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core

module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// Validation Error Types
// ═══════════════════════════════════════════════════════════════════════════

/// A validation error with context for debugging
type ValidationError = {
    Category: string
    Message: string
    NodeId: int option
    SourceRange: (int * int) option  // (startLine, startCol)
}

/// Result of validation - either success or list of errors
type ValidationResult =
    | Valid
    | Invalid of ValidationError list

module ValidationResult =
    let combine (a: ValidationResult) (b: ValidationResult) =
        match a, b with
        | Valid, Valid -> Valid
        | Valid, Invalid errs -> Invalid errs
        | Invalid errs, Valid -> Invalid errs
        | Invalid errs1, Invalid errs2 -> Invalid (errs1 @ errs2)

    let combineAll results =
        results |> List.fold combine Valid

// ═══════════════════════════════════════════════════════════════════════════
// Variable Binding Completeness
// ═══════════════════════════════════════════════════════════════════════════

/// Collects all sources that can bind variables
type BindingSources = {
    /// Variables bound by lambda parameters (name -> nodeId of lambda)
    LambdaParams: Map<string, int>
    /// Variables bound by let expressions (name -> nodeId of binding)
    LetBindings: Map<string, int>
    /// Variables bound by patterns (name -> nodeId of case body)
    PatternBindings: Map<string, int>
}

/// Build a map of all binding sources in the graph
let collectBindingSources
    (graph: SemanticGraph)
    (patternResult: PatternAnalysis.PatternBindingAnalysisResult)
    : BindingSources =

    let mutable lambdaParams = Map.empty
    let mutable letBindings = Map.empty
    let mutable patternBindings = Map.empty

    // Collect lambda parameters
    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.Lambda (params', _, _, _, _) ->
            for (name, _, _) in params' do
                lambdaParams <- Map.add name (NodeId.value node.Id) lambdaParams
        | SemanticKind.Binding (name, _, _, _) ->
            letBindings <- Map.add name (NodeId.value node.Id) letBindings
        | _ -> ()

    // Collect pattern bindings from the analysis result
    // First, include the directly-scanned PatternBinding nodes (most authoritative)
    for KeyValue(name, nodeId) in patternResult.AllPatternBindings do
        patternBindings <- Map.add name nodeId patternBindings

    // Then overlay with case-level bindings (for backward compatibility)
    for KeyValue(caseBodyId, bindings) in patternResult.CasePatternBindings do
        for binding in bindings do
            patternBindings <- Map.add binding.Name caseBodyId patternBindings

    for KeyValue(lambdaId, bindings) in patternResult.EntryPatternBindings do
        for binding in bindings do
            patternBindings <- Map.add binding.Name lambdaId patternBindings

    { LambdaParams = lambdaParams
      LetBindings = letBindings
      PatternBindings = patternBindings }

/// Check if a variable name has a binding source
let hasBindingSource (name: string) (sources: BindingSources) : bool =
    Map.containsKey name sources.LambdaParams ||
    Map.containsKey name sources.LetBindings ||
    Map.containsKey name sources.PatternBindings

/// Check if a name is a local variable (not a qualified module reference)
let private isLocalVariable (name: string) : bool =
    // Qualified names contain dots (e.g., "Console.writeln", "Format.int")
    // Local variables don't have dots
    not (name.Contains('.'))

/// Check if a VarRef has a valid binding source
/// Uses the VarRef's definition field first (most authoritative),
/// then falls back to name-based lookup
let private hasValidBinding
    (graph: SemanticGraph)
    (name: string)
    (definition: NodeId option)
    (sources: BindingSources)
    : bool =
    // First check: if VarRef has a definition, verify it points to a valid binding
    match definition with
    | Some defNodeId ->
        // The VarRef knows its definition - check if that node exists
        match SemanticGraph.tryGetNode defNodeId graph with
        | Some defNode ->
            // Valid if it's a PatternBinding or the node is reachable
            match defNode.Kind with
            | SemanticKind.PatternBinding _ -> true
            | SemanticKind.Binding _ -> true
            | _ -> defNode.IsReachable  // Other definitions (e.g., Lambda params)
        | None -> false  // Definition points to non-existent node
    | None ->
        // No definition attached - fall back to name-based lookup
        hasBindingSource name sources

/// Validate that all Variable nodes have binding sources
let validateVariableBindings
    (graph: SemanticGraph)
    (patternResult: PatternAnalysis.PatternBindingAnalysisResult)
    : ValidationResult =

    let sources = collectBindingSources graph patternResult
    let mutable errors = []

    // Find all VarRef nodes and check each has a binding
    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.VarRef (name, definition) ->
            // VarRef nodes represent variable references
            // Only check reachable nodes AND local variables (not qualified module refs)
            if node.IsReachable && isLocalVariable name && not (hasValidBinding graph name definition sources) then
                let range = (node.Range.Start.Line, node.Range.Start.Column)
                errors <- {
                    Category = "UnboundVariable"
                    Message = $"Variable '{name}' is referenced but has no binding source (not a parameter, let-binding, or pattern binding)"
                    NodeId = Some (NodeId.value node.Id)
                    SourceRange = Some range
                } :: errors
        | _ -> ()

    if List.isEmpty errors then Valid
    else Invalid (List.rev errors)

// ═══════════════════════════════════════════════════════════════════════════
// SSA Completeness (future validation)
// ═══════════════════════════════════════════════════════════════════════════

/// Validate that all nodes requiring SSA have assignments
/// (Placeholder for future implementation)
let validateSSACompleteness
    (_graph: SemanticGraph)
    (_ssaResult: SSAAssign.SSAAssignment)
    : ValidationResult =
    // TODO: Implement SSA completeness checking
    // For each node that should have an SSA (expressions, intermediates),
    // verify it has an assignment in the SSA coeffect
    Valid

// ═══════════════════════════════════════════════════════════════════════════
// Main Validation Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Run all validations on a graph with computed coeffects
/// Returns Valid if all checks pass, or Invalid with all errors found
let validate
    (graph: SemanticGraph)
    (patternResult: PatternAnalysis.PatternBindingAnalysisResult)
    (ssaResult: SSAAssign.SSAAssignment)
    : ValidationResult =

    ValidationResult.combineAll [
        validateVariableBindings graph patternResult
        validateSSACompleteness graph ssaResult
    ]

/// Format validation errors for display
let formatErrors (errors: ValidationError list) : string =
    errors
    |> List.map (fun err ->
        let location =
            match err.NodeId, err.SourceRange with
            | Some nodeId, Some (line, col) ->
                $"[NodeId {nodeId}, line {line}:{col}]"
            | Some nodeId, None ->
                $"[NodeId {nodeId}]"
            | None, Some (line, col) ->
                $"[line {line}:{col}]"
            | None, None ->
                ""
        $"[{err.Category}] {location} {err.Message}")
    |> String.concat "\n"
