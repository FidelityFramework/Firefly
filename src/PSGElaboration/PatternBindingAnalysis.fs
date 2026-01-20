/// Pattern Binding Analysis - Pre-transfer analysis for pattern bindings
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It computes:
/// - Which bindings emerge from each pattern (name, type pairs)
/// - Pre-indexed by pattern location for O(1) lookup during emission
///
/// This eliminates on-demand pattern traversal during transfer, adhering to
/// the photographer principle: observe the structure, don't compute during transfer.
///
/// NANOPASS PLACEMENT:
/// - This is ANALYSIS (read-only observation of PSG structure)
/// - Runs BEFORE transfer begins
/// - Produces coeffects that witnesses PULL during emission
module PSGElaboration.PatternBindingAnalysis

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// A single pattern binding: variable name and its type
type PatternBinding = {
    Name: string
    Type: NativeType
}

/// Result of pattern binding analysis for a semantic graph
type PatternBindingAnalysisResult = {
    /// Map from Match case NodeId to list of bindings introduced by that pattern
    /// Used during emission to know what variables to bind
    CasePatternBindings: Map<int, PatternBinding list>
    
    /// Map from Lambda entry point NodeId to bindings from entry patterns (e.g., argv)
    /// Used for entry point argument pattern bindings
    EntryPatternBindings: Map<int, PatternBinding list>
}

// ═══════════════════════════════════════════════════════════════════════════
// Pure Pattern Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Extract all bindings from a pattern (pure recursive analysis)
/// Returns list of (name, type) pairs for variables introduced by the pattern
let rec extractPatternBindings (pattern: Pattern) : PatternBinding list =
    match pattern with
    | Pattern.Var (name, ty) -> 
        [{ Name = name; Type = ty }]
    
    | Pattern.Tuple elements ->
        elements |> List.collect extractPatternBindings
    
    | Pattern.As (inner, _name) ->
        // As pattern binds the whole value AND the inner pattern
        // The outer 'as' binding is handled separately by FNCS
        extractPatternBindings inner
    
    | Pattern.Union (_, _tagIndex, Some payload, _) ->
        extractPatternBindings payload

    | Pattern.Union (_, _tagIndex, None, _) ->
        []
    
    | Pattern.Record (fields, _) ->
        fields |> List.collect (fun (_, p) -> extractPatternBindings p)
    
    | Pattern.Array elements ->
        elements |> List.collect extractPatternBindings
    
    | Pattern.Or (p1, _p2) ->
        // Both branches should bind same vars (F# enforces this)
        extractPatternBindings p1
    
    | Pattern.And (p1, p2) ->
        extractPatternBindings p1 @ extractPatternBindings p2
    
    | Pattern.Const _ | Pattern.Wildcard | Pattern.Null | Pattern.IsType _ | Pattern.Exception _ ->
        []

// ═══════════════════════════════════════════════════════════════════════════
// Graph-Level Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Find all match cases in the graph and compute their pattern bindings
/// Returns a map from case body NodeId to list of bindings
let findAllCasePatternBindings (graph: SemanticGraph) : Map<int, PatternBinding list> =
    let mutable result = Map.empty
    
    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.Match (_, cases) ->
            // Each case has a pattern and body
            for case in cases do
                let bindings = extractPatternBindings case.Pattern
                if not (List.isEmpty bindings) then
                    result <- Map.add (NodeId.value case.Body) bindings result
        | _ -> ()
    
    result

/// Find entry point Lambda patterns and compute their bindings
/// Entry points may have argv patterns like [|arg1|] that introduce bindings
let findEntryPatternBindings (graph: SemanticGraph) : Map<int, PatternBinding list> =
    let mutable result = Map.empty
    
    // Entry points are typically Bindings containing Lambdas
    for epId in graph.EntryPoints do
        match SemanticGraph.tryGetNode epId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding (_, _, _, _) ->
                // Look for Lambda children
                for childId in node.Children do
                    match SemanticGraph.tryGetNode childId graph with
                    | Some child ->
                        match child.Kind with
                        | SemanticKind.Lambda (params', _, _captures, _, _) ->
                            // params' contains the parameter patterns (now includes NodeId)
                            let bindings = params' |> List.collect (fun (name, ty, _nodeId) ->
                                [{ Name = name; Type = ty }])
                            if not (List.isEmpty bindings) then
                                result <- Map.add (NodeId.value childId) bindings result
                        | _ -> ()
                    | None -> ()
            | SemanticKind.Lambda (params', _, _captures, _, _) ->
                // Entry point is directly a Lambda
                let bindings = params' |> List.collect (fun (name, ty, _nodeId) ->
                    [{ Name = name; Type = ty }])
                if not (List.isEmpty bindings) then
                    result <- Map.add (NodeId.value epId) bindings result
            | _ -> ()
        | None -> ()
    
    result

// ═══════════════════════════════════════════════════════════════════════════
// Main Analysis Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Perform complete pattern binding analysis on a semantic graph
/// This should be called ONCE before transfer begins
let analyze (graph: SemanticGraph) : PatternBindingAnalysisResult =
    {
        CasePatternBindings = findAllCasePatternBindings graph
        EntryPatternBindings = findEntryPatternBindings graph
    }

// ═══════════════════════════════════════════════════════════════════════════
// Lookup Helpers (for witnesses to PULL coeffects)
// ═══════════════════════════════════════════════════════════════════════════

/// Get bindings for a match case body (if any)
let getCaseBindings (caseBodyId: int) (result: PatternBindingAnalysisResult) : PatternBinding list =
    Map.tryFind caseBodyId result.CasePatternBindings
    |> Option.defaultValue []

/// Get bindings for an entry point Lambda (if any)
let getEntryBindings (lambdaId: int) (result: PatternBindingAnalysisResult) : PatternBinding list =
    Map.tryFind lambdaId result.EntryPatternBindings
    |> Option.defaultValue []

/// Check if a pattern has any bindings
let hasBindings (pattern: Pattern) : bool =
    not (List.isEmpty (extractPatternBindings pattern))
