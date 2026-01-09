/// SSA Assignment Pass - Alex preprocessing for MLIR emission
///
/// This pass assigns SSA values to PSG nodes BEFORE MLIR emission.
/// SSA is an MLIR/LLVM concern, not F# semantics, so it lives in Alex.
///
/// Key design:
/// - SSA counter resets at each Lambda boundary (per-function scoping)
/// - Post-order traversal ensures values are assigned before uses
/// - Returns Map<NodeId, NodeSSAAllocation> that witnesses read (coeffect lookup, no generation during emission)
/// - Uses structured SSA type (V of int | Arg of int), not strings
/// - Knows MLIR expansion costs: one PSG node may need multiple SSAs
module Alex.Preprocessing.SSAAssignment

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA ALLOCATION FOR NODES
// ═══════════════════════════════════════════════════════════════════════════

/// SSA allocation for a node - supports multi-SSA expansion
/// SSAs are in emission order; Result is the final SSA (what gets returned/used)
type NodeSSAAllocation = {
    /// All SSAs for this node in emission order
    SSAs: SSA list
    /// The result SSA (always the last one)
    Result: SSA
}

module NodeSSAAllocation =
    let single (ssa: SSA) = { SSAs = [ssa]; Result = ssa }
    let multi (ssas: SSA list) =
        match ssas with
        | [] -> failwith "NodeSSAAllocation requires at least one SSA"
        | _ -> { SSAs = ssas; Result = List.last ssas }

// ═══════════════════════════════════════════════════════════════════════════
// MLIR EXPANSION COSTS
// ═══════════════════════════════════════════════════════════════════════════

/// Get the number of SSAs needed for a literal value
let private literalExpansionCost (lit: LiteralValue) : int =
    match lit with
    | LiteralValue.String _ -> 5  // addressof, constI(len), undef, insertvalue(ptr), insertvalue(len)
    | LiteralValue.Unit -> 1     // constI
    | LiteralValue.Bool _ -> 1   // constI
    | LiteralValue.Int8 _ | LiteralValue.Int16 _ | LiteralValue.Int32 _ | LiteralValue.Int64 _ -> 1
    | LiteralValue.UInt8 _ | LiteralValue.UInt16 _ | LiteralValue.UInt32 _ | LiteralValue.UInt64 _ -> 1
    | LiteralValue.NativeInt _ | LiteralValue.UNativeInt _ -> 1
    | LiteralValue.Char _ -> 1
    | LiteralValue.Float32 _ | LiteralValue.Float64 _ -> 1
    | _ -> 1  // Default

/// Get the number of SSAs needed for a node based on its kind
let private nodeExpansionCost (kind: SemanticKind) : int =
    match kind with
    | SemanticKind.Literal lit -> literalExpansionCost lit
    // Most nodes produce one result
    | _ -> 1

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION SCOPE STATE
// ═══════════════════════════════════════════════════════════════════════════

/// SSA assignment state for a single function scope
type private FunctionScope = {
    Counter: int
    Assignments: Map<int, NodeSSAAllocation>  // NodeId.value -> SSA allocation
}

module private FunctionScope =
    let empty = { Counter = 0; Assignments = Map.empty }

    /// Yield a single SSA
    let yieldSSA (scope: FunctionScope) : SSA * FunctionScope =
        let ssa = V scope.Counter
        ssa, { scope with Counter = scope.Counter + 1 }

    /// Yield multiple SSAs based on expansion cost
    let yieldSSAs (count: int) (scope: FunctionScope) : SSA list * FunctionScope =
        let ssas = List.init count (fun i -> V (scope.Counter + i))
        ssas, { scope with Counter = scope.Counter + count }

    /// Assign a node's SSA allocation
    let assign (nodeId: NodeId) (alloc: NodeSSAAllocation) (scope: FunctionScope) : FunctionScope =
        { scope with Assignments = Map.add (NodeId.value nodeId) alloc scope.Assignments }

/// Check if a node kind produces an SSA value
let private producesValue (kind: SemanticKind) : bool =
    match kind with
    | SemanticKind.Literal _ -> true
    | SemanticKind.VarRef _ -> true
    | SemanticKind.Application _ -> true
    | SemanticKind.Lambda _ -> true
    | SemanticKind.Binding _ -> true
    | SemanticKind.Sequential _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.TupleExpr _ -> true
    | SemanticKind.RecordExpr _ -> true
    | SemanticKind.UnionCase _ -> true
    | SemanticKind.ArrayExpr _ -> true
    | SemanticKind.ListExpr _ -> true
    | SemanticKind.FieldGet _ -> true
    | SemanticKind.IndexGet _ -> true
    | SemanticKind.Upcast _ -> true
    | SemanticKind.Downcast _ -> true
    | SemanticKind.TypeTest _ -> true
    | SemanticKind.AddressOf _ -> true
    | SemanticKind.Deref _ -> true
    | SemanticKind.TraitCall _ -> true
    | SemanticKind.Intrinsic _ -> true
    | SemanticKind.PlatformBinding _ -> true
    | SemanticKind.InterpolatedString _ -> true
    // These don't produce values (statements/void)
    | SemanticKind.Set _ -> false
    | SemanticKind.FieldSet _ -> false
    | SemanticKind.IndexSet _ -> false
    | SemanticKind.NamedIndexedPropertySet _ -> false
    | SemanticKind.WhileLoop _ -> false
    | SemanticKind.ForLoop _ -> false
    | SemanticKind.ForEach _ -> false
    | SemanticKind.TryWith _ -> false
    | SemanticKind.TryFinally _ -> false
    | SemanticKind.Quote _ -> false
    | SemanticKind.ObjectExpr _ -> false
    | SemanticKind.ModuleDef _ -> false
    | SemanticKind.TypeDef _ -> false
    | SemanticKind.MemberDef _ -> false
    | SemanticKind.TypeAnnotation _ -> true  // Passes through the inner value
    | SemanticKind.PatternBinding _ -> true  // Pattern binding introduces a variable
    | SemanticKind.Error _ -> false

/// Result of SSA assignment pass
type SSAAssignment = {
    /// Map from NodeId.value to SSA allocation (supports multi-SSA expansion)
    NodeSSA: Map<int, NodeSSAAllocation>
    /// Map from Lambda NodeId.value to its function name
    LambdaNames: Map<int, string>
    /// Set of entry point Lambda IDs
    EntryPointLambdas: Set<int>
}

/// Assign SSA names to all nodes in a function body
/// Returns updated scope with assignments
let rec private assignFunctionBody
    (graph: SemanticGraph)
    (scope: FunctionScope)
    (nodeId: NodeId)
    : FunctionScope =

    match Map.tryFind nodeId graph.Nodes with
    | None -> scope
    | Some node ->
        // Post-order: process children first
        let scopeAfterChildren =
            node.Children |> List.fold (fun s childId -> assignFunctionBody graph s childId) scope

        // Special handling for nested Lambdas - they get their own scope
        // (but we still assign this Lambda node an SSA in parent scope)
        match node.Kind with
        | SemanticKind.Lambda(_, bodyId) ->
            // Process Lambda body in a NEW scope (SSA counter resets)
            let _innerScope = assignFunctionBody graph FunctionScope.empty bodyId
            // Lambda itself gets an SSA in the PARENT scope (function pointer)
            if producesValue node.Kind then
                let ssaName, scopeWithSSA = FunctionScope.yieldSSA scopeAfterChildren
                FunctionScope.assign node.Id (NodeSSAAllocation.single ssaName) scopeWithSSA
            else
                scopeAfterChildren
        | _ ->
            // Regular node - assign SSAs based on expansion cost
            if producesValue node.Kind then
                let cost = nodeExpansionCost node.Kind
                let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                let alloc = NodeSSAAllocation.multi ssas
                FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                scopeAfterChildren

/// Collect all Lambdas in the graph and assign function names
let private collectLambdas (graph: SemanticGraph) : Map<int, string> * Set<int> =
    let mutable lambdaCounter = 0
    let mutable lambdaNames = Map.empty
    let mutable entryPoints = Set.empty

    // First, identify entry point lambdas
    for entryId in graph.EntryPoints do
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding(_, _, _, isEntryPoint) when isEntryPoint ->
                // The binding's first child is typically the Lambda
                match node.Children with
                | lambdaId :: _ -> entryPoints <- Set.add (NodeId.value lambdaId) entryPoints
                | _ -> ()
            | SemanticKind.Lambda _ ->
                entryPoints <- Set.add (NodeId.value entryId) entryPoints
            | _ -> ()
        | None -> ()

    // Now assign names to all Lambdas
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda _ ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                if Set.contains nodeIdVal entryPoints then
                    "main"
                else
                    let n = sprintf "lambda_%d" lambdaCounter
                    lambdaCounter <- lambdaCounter + 1
                    n
            lambdaNames <- Map.add nodeIdVal name lambdaNames
        | _ -> ()

    lambdaNames, entryPoints

/// Main entry point: assign SSA names to all nodes in the graph
let assignSSA (graph: SemanticGraph) : SSAAssignment =
    let lambdaNames, entryPoints = collectLambdas graph

    // For each Lambda, assign SSAs to its body in its own scope
    let mutable allAssignments = Map.empty

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId) ->
            // Start with parameter SSAs (%arg0, %arg1, etc.)
            let paramScope =
                params'
                |> List.mapi (fun i (name, _ty) -> i, name)
                |> List.fold (fun (scope: FunctionScope) (i, _name) ->
                    // Parameters use %argN naming, don't count toward SSA counter
                    scope
                ) FunctionScope.empty

            // Assign SSAs to body nodes
            let bodyScope = assignFunctionBody graph paramScope bodyId

            // Merge into global assignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
        | _ -> ()

    // Also process top-level nodes (module bindings, etc.)
    let topLevelScope =
        graph.EntryPoints
        |> List.fold (fun scope entryId -> assignFunctionBody graph scope entryId) FunctionScope.empty

    for kvp in topLevelScope.Assignments do
        allAssignments <- Map.add kvp.Key kvp.Value allAssignments

    {
        NodeSSA = allAssignments
        LambdaNames = lambdaNames
        EntryPointLambdas = entryPoints
    }

/// Look up the full SSA allocation for a node (coeffect lookup)
let lookupAllocation (nodeId: NodeId) (assignment: SSAAssignment) : NodeSSAAllocation option =
    Map.tryFind (NodeId.value nodeId) assignment.NodeSSA

/// Look up just the result SSA for a node (most common use case)
let lookupSSA (nodeId: NodeId) (assignment: SSAAssignment) : SSA option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.Result)

/// Look up all SSAs for a node (for witnesses that need intermediates)
let lookupSSAs (nodeId: NodeId) (assignment: SSAAssignment) : SSA list option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.SSAs)

/// Look up the function name for a Lambda
let lookupLambdaName (nodeId: NodeId) (assignment: SSAAssignment) : string option =
    Map.tryFind (NodeId.value nodeId) assignment.LambdaNames

/// Check if a Lambda is an entry point
let isEntryPoint (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Set.contains (NodeId.value nodeId) assignment.EntryPointLambdas
