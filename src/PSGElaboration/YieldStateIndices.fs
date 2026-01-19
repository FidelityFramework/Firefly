/// YieldStateIndices - Coeffect for assigning state indices to yield points
///
/// PRD-15: Each yield in a sequence expression becomes a state transition point.
/// This coeffect pre-computes the state index for each yield in document order.
///
/// FOUR PILLARS - This is a COEFFECT:
/// - Pre-computed during preprocessing (immutable after)
/// - Consumed by SeqWitness during MoveNext generation
/// - Yield witness looks up its state index, doesn't compute it
///
/// STATE NUMBERING:
/// - State 0: Initial state (before first yield)
/// - State 1: After first yield (ready for second)
/// - State N: After Nth yield
/// - State -1: Done (no more values)
///
/// The yield at state N-1 transitions TO state N.
/// When MoveNext is called in state N-1, it executes and stores N as next state.
///
/// BODY STRUCTURE (TWO PATTERNS):
/// 1. Sequential: seq { yield 1; yield 2 } - each yield is a separate state
/// 2. While-based: seq { while cond do yield x } - two-state model:
///    - State 0: Initialize internal state, jump to check
///    - State 1: Execute post-yield code, jump to check
///    - Shared blocks: check, yield, done
module PSGElaboration.YieldStateIndices

open FSharp.Native.Compiler.PSG.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL STATE TRACKING
// ═══════════════════════════════════════════════════════════════════════════

/// Information about an internal mutable binding in a seq body
/// These become fields in the seq struct (after captures)
type InternalStateField = {
    /// Name of the mutable variable
    Name: string
    /// Type of the variable
    Type: NativeType
    /// NodeId of the Binding node
    BindingId: NodeId
    /// Index in the seq struct (3 + numCaptures + index)
    StructIndex: int
}

// ═══════════════════════════════════════════════════════════════════════════
// BODY STRUCTURE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Structure of a while-based sequence body
/// seq { <init>; while <cond> do <before-yield>; yield <value>; <post-yield> }
type WhileBodyInfo = {
    /// NodeIds of expressions before the while loop (initialization)
    InitExprs: NodeId list
    /// NodeId of the WhileLoop node
    WhileNodeId: NodeId
    /// NodeId of the while condition
    ConditionId: NodeId
    /// NodeIds of expressions before the yield (inside while body)
    PreYieldExprs: NodeId list
    /// The single yield point (while-based seqs have one yield per while)
    YieldNodeId: NodeId
    /// NodeId of the value expression being yielded
    YieldValueId: NodeId
    /// NodeIds of expressions after the yield (inside while body)
    PostYieldExprs: NodeId list
    /// Whether there's a conditional around the yield (if x then yield y)
    ConditionalYield: ConditionalYieldInfo option
}

/// Information about conditional yields (if inside while)
/// Supports nested conditionals: if A then if B then yield x → ConditionIds = [A; B]
and ConditionalYieldInfo = {
    /// NodeId of the outermost IfThenElse
    IfNodeId: NodeId
    /// NodeIds of all conditions (for nested ifs, multiple conditions ANDed together)
    ConditionIds: NodeId list
}

/// The kind of sequence body structure
type SeqBodyKind =
    /// Sequential yields: seq { yield 1; yield 2; yield 3 }
    /// Each yield gets its own state (0 -> 1 -> 2 -> ... -> -1)
    | Sequential
    /// While-based: seq { while cond do yield x }
    /// Two-state model (state 0 = init, state 1 = post-yield)
    | WhileBased of WhileBodyInfo

// ═══════════════════════════════════════════════════════════════════════════
// YIELD INFO TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Information about a single yield point
type YieldInfo = {
    /// NodeId of the Yield node
    YieldId: NodeId
    /// State index (1, 2, 3, ...)
    /// This is the state AFTER this yield executes
    StateIndex: int
    /// NodeId of the value expression being yielded
    ValueId: NodeId
}

/// Yield state information for a single SeqExpr
type SeqYieldInfo = {
    /// NodeId of the SeqExpr
    SeqExprId: NodeId
    /// All yields in document order with their state indices
    Yields: YieldInfo list
    /// Total number of yields
    NumYields: int
    /// Body structure (sequential vs while-based)
    BodyKind: SeqBodyKind
    /// Internal mutable state fields (let mutable inside seq body)
    InternalState: InternalStateField list
}

/// The coeffect: maps SeqExpr NodeIds to their yield information
type YieldStateCoeffect = {
    /// SeqExprId -> yield information
    SeqYields: Map<int, SeqYieldInfo>  // int is NodeId.value
}

// ═══════════════════════════════════════════════════════════════════════════
// YIELD COLLECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is a literal boolean value
let private isLiteralBool (graph: SemanticGraph) (nodeId: NodeId) : bool option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Literal (LiteralValue.Bool b) -> Some b
        | _ -> None
    | None -> None

/// Collect all Yield nodes in a subtree in document (pre-order) order
/// This traverses children left-to-right, which matches source code order
/// Yields inside `if false then` branches are excluded (dead code)
let rec private collectYieldsInSubtree (graph: SemanticGraph) (nodeId: NodeId) : (NodeId * NodeId) list =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> []
    | Some node ->
        // Handle IfThenElse with constant conditions - skip dead branches
        match node.Kind with
        | SemanticKind.IfThenElse (condId, thenId, elseIdOpt) ->
            match isLiteralBool graph condId with
            | Some false ->
                // `if false then` - skip then branch entirely, only collect from else
                match elseIdOpt with
                | Some elseId -> collectYieldsInSubtree graph elseId
                | None -> []
            | Some true ->
                // `if true then` - skip else branch, only collect from then
                collectYieldsInSubtree graph thenId
            | None ->
                // Non-constant condition - collect from both branches
                let thenYields = collectYieldsInSubtree graph thenId
                let elseYields =
                    match elseIdOpt with
                    | Some elseId -> collectYieldsInSubtree graph elseId
                    | None -> []
                thenYields @ elseYields
        | SemanticKind.Yield valueId ->
            // This is a yield node
            [(nodeId, valueId)]
        | _ ->
            // For other nodes, collect from children (document order = left-to-right)
            node.Children
            |> List.collect (fun childId -> collectYieldsInSubtree graph childId)

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL STATE COLLECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Collect all mutable bindings in a subtree (internal state fields)
let rec private collectMutableBindings (graph: SemanticGraph) (nodeId: NodeId) : (string * NativeType * NodeId) list =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> []
    | Some node ->
        let thisBinding =
            match node.Kind with
            | SemanticKind.Binding (name, isMutable, _, _) when isMutable ->
                [(name, node.Type, node.Id)]
            | _ -> []

        let childBindings =
            node.Children
            |> List.collect (fun childId -> collectMutableBindings graph childId)

        thisBinding @ childBindings

// ═══════════════════════════════════════════════════════════════════════════
// BODY STRUCTURE ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is a WhileLoop
let private isWhileLoop (graph: SemanticGraph) (nodeId: NodeId) : (NodeId * NodeId) option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.WhileLoop (guardId, bodyId) -> Some (guardId, bodyId)
        | _ -> None
    | None -> None

/// Check if a node is a Sequential
let private isSequential (graph: SemanticGraph) (nodeId: NodeId) : NodeId list option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Sequential nodes -> Some nodes
        | _ -> None
    | None -> None

/// Flatten nested Sequentials into a single list of non-Sequential nodes
/// e.g., [A, Sequential([B, Sequential([C, D])])] → [A, B, C, D]
let rec private flattenSequentials (graph: SemanticGraph) (nodeIds: NodeId list) : NodeId list =
    nodeIds
    |> List.collect (fun nodeId ->
        match isSequential graph nodeId with
        | Some innerNodes -> flattenSequentials graph innerNodes
        | None -> [nodeId])

/// Check if a node is an IfThenElse containing a yield, collecting all nested conditions
/// For `if A then if B then yield x`, returns ConditionIds = [A; B]
let private isConditionalYield (graph: SemanticGraph) (nodeId: NodeId) : ConditionalYieldInfo option =
    // Recursively collect conditions from nested IfThenElse nodes
    let rec collectConditions (nId: NodeId) (accConditions: NodeId list) : (NodeId * NodeId list) option =
        match SemanticGraph.tryGetNode nId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.IfThenElse (condId, thenId, _) ->
                let newConditions = accConditions @ [condId]
                // Check if then branch is another IfThenElse (nested conditional)
                match SemanticGraph.tryGetNode thenId graph with
                | Some thenNode ->
                    match thenNode.Kind with
                    | SemanticKind.IfThenElse _ ->
                        // Nested if - recurse
                        collectConditions thenId newConditions
                    | SemanticKind.Yield _ ->
                        // Direct yield - done
                        Some (nId, newConditions)
                    | _ ->
                        // Check if subtree contains a yield
                        let yields = collectYieldsInSubtree graph thenId
                        if not (List.isEmpty yields) then
                            Some (nId, newConditions)
                        else
                            None
                | None -> None
            | _ -> None
        | None -> None

    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.IfThenElse _ ->
            match collectConditions nodeId [] with
            | Some (outerIfId, conditions) ->
                Some { IfNodeId = outerIfId; ConditionIds = conditions }
            | None -> None
        | _ -> None
    | None -> None

/// Analyze the structure of a sequence body
/// Returns Sequential for simple yields, WhileBased for while-loop patterns
let private analyzeBodyStructure
    (graph: SemanticGraph)
    (bodyId: NodeId)
    (yields: (NodeId * NodeId) list)
    : SeqBodyKind =

    // The bodyId might be a Lambda (MoveNext thunk) - unwrap to get actual body
    let actualBodyId =
        match SemanticGraph.tryGetNode bodyId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Lambda(_, lambdaBodyId, _, _, _) -> lambdaBodyId
            | _ -> bodyId
        | None -> bodyId

    // Helper to find a WhileLoop in a list of nodes, recursively searching into nested Sequentials
    let rec findWhileInSequence (nodes: NodeId list) (accInit: NodeId list) : (NodeId list * (NodeId * NodeId * NodeId)) option =
        match nodes with
        | [] -> None
        | nodeId :: remaining ->
            match isWhileLoop graph nodeId with
            | Some (guardId, whileBodyId) ->
                // Found while loop - accInit contains all nodes before it
                Some (List.rev accInit, (nodeId, guardId, whileBodyId))
            | None ->
                // Not a while loop - check if it's a nested Sequential we should descend into
                match isSequential graph nodeId with
                | Some nestedNodes ->
                    // Try to find while inside this nested Sequential
                    match findWhileInSequence nestedNodes (nodeId :: accInit) with
                    | Some result -> Some result
                    | None ->
                        // No while in nested, continue searching siblings
                        findWhileInSequence remaining (nodeId :: accInit)
                | None ->
                    // Not a Sequential either, continue with siblings
                    findWhileInSequence remaining (nodeId :: accInit)

    // Check if body is a Sequential containing a WhileLoop
    match isSequential graph actualBodyId with
    | Some nodes ->
        match findWhileInSequence nodes [] with
        | Some (initExprs, (whileNodeId, conditionId, whileBodyId)) ->
            // Found a while loop - analyze its body for yield structure
            match isSequential graph whileBodyId with
            | Some whileBodyNodes ->
                // Find the yield(s) in the while body
                let whileYields = collectYieldsInSubtree graph whileBodyId
                match whileYields with
                | [(yieldNodeId, yieldValueId)] ->
                    // Single yield - typical pattern
                    // First flatten all nested Sequentials in the while body
                    // This ensures pre/post yield expressions are properly separated even when deeply nested
                    let flattenedBody = flattenSequentials graph whileBodyNodes
                    
                    // Split flattened body into pre-yield and post-yield
                    let rec splitAtYield (nodes: NodeId list) (pre: NodeId list) =
                        match nodes with
                        | [] -> (List.rev pre, [])
                        | nodeId :: rest ->
                            let nodeYields = collectYieldsInSubtree graph nodeId
                            if not (List.isEmpty nodeYields) then
                                // This node contains the yield - everything after is post-yield
                                // (Since we flattened, this is the actual yield node or conditional)
                                (List.rev pre, rest)
                            else
                                splitAtYield rest (nodeId :: pre)
                    let (preYield, postYield) = splitAtYield flattenedBody []

                    // Check for conditional yield pattern
                    let conditionalYield =
                        whileBodyNodes
                        |> List.tryPick (fun nodeId -> isConditionalYield graph nodeId)

                    WhileBased {
                        InitExprs = initExprs
                        WhileNodeId = whileNodeId
                        ConditionId = conditionId
                        PreYieldExprs = preYield
                        YieldNodeId = yieldNodeId
                        YieldValueId = yieldValueId
                        PostYieldExprs = postYield
                        ConditionalYield = conditionalYield
                    }
                | _ ->
                    // Multiple yields in while - treat as sequential for now
                    // (This would require more complex state machine)
                    Sequential
            | None ->
                // While body is not a Sequential - check if it's a single yield or if
                match SemanticGraph.tryGetNode whileBodyId graph with
                | Some whileBodyNode ->
                    match whileBodyNode.Kind with
                    | SemanticKind.Yield valueId ->
                        // Simple: while cond do yield x
                        WhileBased {
                            InitExprs = initExprs
                            WhileNodeId = whileNodeId
                            ConditionId = conditionId
                            PreYieldExprs = []
                            YieldNodeId = whileBodyId
                            YieldValueId = valueId
                            PostYieldExprs = []
                            ConditionalYield = None
                        }
                    | SemanticKind.IfThenElse (condId, thenId, _) ->
                        // Conditional yield: while cond do if x then yield y
                        let yields = collectYieldsInSubtree graph thenId
                        match yields with
                        | [(yieldId, valueId)] ->
                            WhileBased {
                                InitExprs = initExprs
                                WhileNodeId = whileNodeId
                                ConditionId = conditionId
                                PreYieldExprs = []
                                YieldNodeId = yieldId
                                YieldValueId = valueId
                                PostYieldExprs = []
                                ConditionalYield = Some { IfNodeId = whileBodyId; ConditionIds = [condId] }
                            }
                        | _ -> Sequential  // Complex pattern
                    | _ -> Sequential  // Unrecognized pattern
                | None -> Sequential
        | None ->
            // No while loop found - sequential yields
            Sequential
    | None ->
        // Body is not a Sequential - check if it's a direct WhileLoop
        match isWhileLoop graph actualBodyId with
        | Some (conditionId, whileBodyId) ->
            // Direct while loop as body
            let whileYields = collectYieldsInSubtree graph whileBodyId
            match whileYields with
            | [(yieldNodeId, yieldValueId)] ->
                WhileBased {
                    InitExprs = []
                    WhileNodeId = actualBodyId
                    ConditionId = conditionId
                    PreYieldExprs = []
                    YieldNodeId = yieldNodeId
                    YieldValueId = yieldValueId
                    PostYieldExprs = []
                    ConditionalYield = None
                }
            | _ -> Sequential
        | None ->
            // Not a while-based pattern
            Sequential

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Compute yield state indices for all SeqExprs in the graph
let run (graph: SemanticGraph) : YieldStateCoeffect =
    let mutable seqYields = Map.empty

    // Find all SeqExpr nodes and compute their yield indices
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.SeqExpr (bodyId, captures) ->
            // Collect all yields in the body in document order
            let yields = collectYieldsInSubtree graph bodyId

            // Analyze body structure (sequential vs while-based)
            let bodyKind = analyzeBodyStructure graph bodyId yields

            // Collect internal mutable bindings
            let mutableBindings = collectMutableBindings graph bodyId
            let numCaptures = List.length captures

            // Create internal state fields with struct indices
            // Struct layout: {state: i32, current: T, code_ptr: ptr, cap₀, ..., capₙ, state₀, ..., stateₘ}
            let internalState =
                mutableBindings
                |> List.mapi (fun i (name, ty, bindingId) ->
                    {
                        Name = name
                        Type = ty
                        BindingId = bindingId
                        StructIndex = 3 + numCaptures + i  // After standard fields and captures
                    })

            // Assign state indices: 1, 2, 3, ...
            // State N means "after yield N has executed"
            let yieldInfos =
                yields
                |> List.mapi (fun i (yieldId, valueId) ->
                    {
                        YieldId = yieldId
                        StateIndex = i + 1  // States 1, 2, 3, ...
                        ValueId = valueId
                    })

            let seqInfo = {
                SeqExprId = node.Id
                Yields = yieldInfos
                NumYields = List.length yields
                BodyKind = bodyKind
                InternalState = internalState
            }

            seqYields <- Map.add (NodeId.value node.Id) seqInfo seqYields
        | _ -> ()

    { SeqYields = seqYields }

// ═══════════════════════════════════════════════════════════════════════════
// LOOKUP FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Get yield info for a SeqExpr
let tryGetSeqYields (seqExprId: NodeId) (coeffect: YieldStateCoeffect) : SeqYieldInfo option =
    Map.tryFind (NodeId.value seqExprId) coeffect.SeqYields

/// Get state index for a specific Yield node
let tryGetYieldStateIndex (yieldId: NodeId) (coeffect: YieldStateCoeffect) : int option =
    coeffect.SeqYields
    |> Map.tryPick (fun _ seqInfo ->
        seqInfo.Yields
        |> List.tryFind (fun yi -> yi.YieldId = yieldId)
        |> Option.map (fun yi -> yi.StateIndex))

/// Get the SeqExpr that contains a Yield
let tryGetEnclosingSeq (yieldId: NodeId) (coeffect: YieldStateCoeffect) : NodeId option =
    coeffect.SeqYields
    |> Map.tryPick (fun seqIdVal seqInfo ->
        if seqInfo.Yields |> List.exists (fun yi -> yi.YieldId = yieldId) then
            Some seqInfo.SeqExprId
        else
            None)
