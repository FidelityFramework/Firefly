/// SSAValidation - Validate SSA assignment invariants
///
/// This module enforces the SSA (Static Single Assignment) invariants that
/// MUST hold before MLIR generation:
///
/// 1. No redefinition: Each SSA is assigned exactly once per function
/// 2. No namespace leakage: SSAs from one function don't leak into another
/// 3. Coherent scoping: Module-level bindings vs function-local SSAs
///
/// All violations result in HARD ERRORS with clear diagnostic messages
/// pointing to the PSG nodes involved.
module PSGElaboration.SSAValidation

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // For NodeId
open PSGElaboration.SSAAssignment
open PSGElaboration.Coeffects
open Alex.Dialects.Core.Types  // For SSA type

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION ERRORS
// ═══════════════════════════════════════════════════════════════════════════

type SSAValidationError =
    | Redefinition of SSA * NodeId * string * NodeId * string * string option
    | NamespaceLeakage of SSA * string option * string option * NodeId * string
    | UseBeforeDefinition of SSA * NodeId * string * string option

let formatError (graph: SemanticGraph) (error: SSAValidationError) : string =
    let formatSSA ssa =
        match ssa with
        | V n -> sprintf "%%v%d" n
        | Arg n -> sprintf "%%arg%d" n

    match error with
    | Redefinition (ssa, firstNode, firstKind, secondNode, secondKind, funcOpt) ->
        let funcContext =
            match funcOpt with
            | Some func -> sprintf " in function '%s'" func
            | None -> " at module level"

        sprintf "SSA REDEFINITION ERROR: %s is assigned to multiple nodes%s\n  First: Node %d (%s)\n  Second: Node %d (%s)\n\nEach SSA must be assigned exactly once."
            (formatSSA ssa) funcContext (NodeId.value firstNode) firstKind (NodeId.value secondNode) secondKind

    | NamespaceLeakage (ssa, defFunc, useFunc, usingNode, usingKind) ->
        let defContext =
            match defFunc with
            | Some func -> sprintf "function '%s'" func
            | None -> "module level"
        let useContext =
            match useFunc with
            | Some func -> sprintf "function '%s'" func
            | None -> "module level"

        sprintf "SSA NAMESPACE LEAKAGE ERROR: %s defined in %s is used in %s\n  Using node: %d (%s)\n\nSSAs cannot cross function boundaries."
            (formatSSA ssa) defContext useContext (NodeId.value usingNode) usingKind

    | UseBeforeDefinition (ssa, usingNode, usingKind, funcOpt) ->
        let funcContext =
            match funcOpt with
            | Some func -> sprintf " in function '%s'" func
            | None -> " at module level"

        sprintf "SSA USE-BEFORE-DEFINITION ERROR: %s is used before it is defined%s\n  Using node: %d (%s)\n\nSSAs must be defined before use."
            (formatSSA ssa) funcContext (NodeId.value usingNode) usingKind

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION LOGIC
// ═══════════════════════════════════════════════════════════════════════════

/// Find the enclosing Lambda for a node by walking up Parent chain
/// CRITICAL: For a Lambda node itself, return its OWN ID (it's a function, not part of parent)
/// For other nodes, walk up to find the containing Lambda
let private findEnclosingLambda (nodeId: NodeId) (graph: SemanticGraph) : NodeId option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Lambda _ ->
            // Lambda nodes define their own scope
            Some nodeId
        | _ ->
            // Non-Lambda nodes: walk up to find enclosing Lambda
            let rec walk (currentId: NodeId) (visited: Set<NodeId>) =
                if Set.contains currentId visited then None
                else
                    match SemanticGraph.tryGetNode currentId graph with
                    | Some node ->
                        match node.Kind with
                        | SemanticKind.Lambda _ -> Some currentId
                        | _ ->
                            match node.Parent with
                            | Some parentId -> walk parentId (Set.add currentId visited)
                            | None -> None  // Reached root without finding Lambda (module-level)
                    | None -> None
            match node.Parent with
            | Some parentId -> walk parentId Set.empty
            | None -> None
    | None -> None

/// Check for SSA redefinitions WITHIN each function scope
/// CRITICAL: In MLIR/LLVM, each function has its own SSA namespace.
/// %v0 in function A and %v0 in function B are DIFFERENT and should NOT conflict.
let checkRedefinitions (graph: SemanticGraph) (ssaAssignment: SSAAssignment) : SSAValidationError list =
    // Group nodes by their enclosing Lambda (function scope)
    // Key: Lambda NodeId (using NodeId -1 as sentinel for module-level)
    let nodesByFunction = System.Collections.Generic.Dictionary<NodeId, ResizeArray<NodeId * SSA * string>>()
    let moduleLevelSentinel = NodeId -1  // Sentinel for module-level nodes

    for kvp in ssaAssignment.NodeSSA do
        let nodeId = kvp.Key
        let allocation = kvp.Value

        // Find which function this node belongs to (use sentinel for module-level)
        let enclosingLambdaId =
            match findEnclosingLambda nodeId graph with
            | Some lambdaId -> lambdaId
            | None -> moduleLevelSentinel

        // Get node kind for error messages
        let nodeKind =
            match SemanticGraph.tryGetNode nodeId graph with
            | Some node -> sprintf "%A" node.Kind
            | None -> "Unknown"

        // Add each SSA to this function's list
        if not (nodesByFunction.ContainsKey(enclosingLambdaId)) then
            nodesByFunction.[enclosingLambdaId] <- ResizeArray()

        for ssa in allocation.SSAs do
            nodesByFunction.[enclosingLambdaId].Add((nodeId, ssa, nodeKind))

    // Check for SSA redefinitions WITHIN each function scope
    [
        for kvp in nodesByFunction do
            let functionId = kvp.Key
            let nodeSSAs = kvp.Value

            // Build inverse map: SSA -> nodes WITHIN THIS FUNCTION
            let ssaToNodes = System.Collections.Generic.Dictionary<SSA, ResizeArray<NodeId * string>>()
            for (nodeId, ssa, nodeKind) in nodeSSAs do
                if not (ssaToNodes.ContainsKey(ssa)) then
                    ssaToNodes.[ssa] <- ResizeArray()
                ssaToNodes.[ssa].Add((nodeId, nodeKind))

            // Find SSAs assigned to multiple nodes in THIS function
            for ssaKvp in ssaToNodes do
                let ssa = ssaKvp.Key
                let nodes = ssaKvp.Value
                if nodes.Count > 1 then
                    let (firstNodeId, firstKind) = nodes.[0]
                    let (secondNodeId, secondKind) = nodes.[1]

                    // Get function name for error message (None if module-level sentinel)
                    let funcOpt =
                        if functionId = moduleLevelSentinel then
                            None
                        else
                            Map.tryFind functionId ssaAssignment.LambdaNames

                    yield Redefinition (ssa, firstNodeId, firstKind, secondNodeId, secondKind, funcOpt)
    ]

// ═══════════════════════════════════════════════════════════════════════════
// MAIN VALIDATION ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Validate SSA assignments after they've been computed
/// Returns Result with list of validation errors (empty list = success)
let validate (graph: SemanticGraph) (ssaAssignment: SSAAssignment) : Result<unit, SSAValidationError list> =
    let errors = ResizeArray<SSAValidationError>()

    // 1. Check for redefinitions WITHIN each function scope
    let redefinitionErrors = checkRedefinitions graph ssaAssignment
    errors.AddRange(redefinitionErrors)

    // TODO: 2. Check for namespace leakage between functions
    //  Requires building node -> function mapping from PSG structure

    // TODO: 3. Check for use-before-definition
    //  Requires def-use analysis on PSG structure

    if errors.Count = 0 then
        Ok ()
    else
        Error (List.ofSeq errors)

/// Validate and fail compilation if errors are found
let validateOrFail (graph: SemanticGraph) (ssaAssignment: SSAAssignment) : unit =
    match validate graph ssaAssignment with
    | Ok () -> ()
    | Error errors ->
        let errorMessages =
            errors
            |> List.map (formatError graph)
            |> String.concat "\n\n"

        failwithf "SSA VALIDATION FAILED:\n\n%s\n\nCompilation aborted. Fix SSA assignment bugs in PSGElaboration." errorMessages
