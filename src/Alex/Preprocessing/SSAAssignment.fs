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

/// Calculate SSA cost for interpolated string based on parts
let private interpolatedStringCost (parts: InterpolatedPart list) : int =
    // Count string parts (each needs 5 SSAs for fat pointer construction)
    let stringPartCount =
        parts |> List.sumBy (fun p ->
            match p with
            | InterpolatedPart.StringPart _ -> 1
            | InterpolatedPart.ExprPart _ -> 0)  // Already computed, no new SSAs

    // Concatenations: each needs 10 SSAs (4 extract, 1 add, 1 alloca, 1 gep, 3 build)
    let concatCount = max 0 (List.length parts - 1)

    // Total: 5 per string part + 10 per concat
    (stringPartCount * 5) + (concatCount * 10)

/// Get the number of SSAs needed for a node based on its kind
let private nodeExpansionCost (kind: SemanticKind) : int =
    match kind with
    | SemanticKind.Literal lit -> literalExpansionCost lit
    // InterpolatedString needs dynamic SSA count based on parts
    | SemanticKind.InterpolatedString parts -> interpolatedStringCost parts
    // ForLoop needs 2 SSAs: ivSSA (induction variable) + stepSSA (step constant)
    | SemanticKind.ForLoop _ -> 2
    // Lambda needs 1 SSA for potential return value synthesis (type reconciliation)
    | SemanticKind.Lambda _ -> 1
    // Application nodes may call syscalls (5 SSAs), string ops (10+ SSAs),
    // or Format operations (intToString ~40, floatToString ~70, stringToInt ~30)
    // Allocate generously - unused SSAs are harmless
    | SemanticKind.Application _ -> 80
    // IfThenElse needs up to 3 SSAs: result + zero for void then + zero for void else
    | SemanticKind.IfThenElse _ -> 3
    // Match needs many SSAs: discrim + (tag+cmp per case) + zero yields + result
    // Allocate generously for typical match expressions
    | SemanticKind.Match _ -> 15
    // Binding may need SSAs for mutable alloca
    | SemanticKind.Binding _ -> 3
    // Memory operations
    | SemanticKind.IndexGet _ -> 2        // gep + load
    | SemanticKind.IndexSet _ -> 1        // gep (store doesn't produce SSA)
    | SemanticKind.AddressOf _ -> 2       // const + alloca (if immutable)
    // VarRef may need SSAs for loading mutable values:
    // - Module-level mutable: addressof + load (2 SSAs)
    // - Addressed mutable: load result (1 SSA)
    | SemanticKind.VarRef _ -> 2
    | SemanticKind.FieldGet _ -> 1        // extract
    | SemanticKind.FieldSet _ -> 1        // insert
    | SemanticKind.TraitCall _ -> 1       // extract typically
    // Data structure construction - use generous estimates
    | SemanticKind.TupleExpr _ -> 10      // undef + inserts
    | SemanticKind.RecordExpr _ -> 10     // undef + inserts
    | SemanticKind.ArrayExpr _ -> 20      // allocs + stores + fat pointer
    | SemanticKind.ListExpr _ -> 20       // same as array
    | SemanticKind.UnionCase _ -> 4       // tag + undef + insert(s)
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
    | SemanticKind.VarRef _ -> true
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
        | SemanticKind.Lambda(_params, bodyId) ->
            // Process Lambda body in a NEW scope (SSA counter resets)
            let _innerScope = assignFunctionBody graph FunctionScope.empty bodyId
            // Lambda itself gets an SSA in the PARENT scope (function pointer)
            if producesValue node.Kind then
                let ssaName, scopeWithSSA = FunctionScope.yieldSSA scopeAfterChildren
                FunctionScope.assign node.Id (NodeSSAAllocation.single ssaName) scopeWithSSA
            else
                scopeAfterChildren
        // VarRef now gets SSAs for mutable variable loads
        // (Regular VarRefs to immutable values may not use their SSAs, but unused SSAs are harmless)

        // ForLoop needs SSAs for internal operation (ivSSA + stepSSA)
        // even though it doesn't "produce a value" in the semantic sense
        | SemanticKind.ForLoop _ ->
            let cost = nodeExpansionCost node.Kind  // Returns 2
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            FunctionScope.assign node.Id alloc scopeWithSSAs

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
    // Entry points may be ModuleDef nodes containing Binding nodes with isEntryPoint=true
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
            | SemanticKind.ModuleDef (_, memberIds) ->
                // ModuleDef entry point - check members for entry point Binding
                for memberId in memberIds do
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding(_, _, _, isEntryPoint) when isEntryPoint ->
                            match memberNode.Children with
                            | lambdaId :: _ -> entryPoints <- Set.add (NodeId.value lambdaId) entryPoints
                            | _ -> ()
                        | _ -> ()
                    | None -> ()
            | _ -> ()
        | None -> ()

    // Build reverse index: Lambda NodeId -> Binding name
    // This handles the case where Parent field isn't set on Lambdas
    let mutable lambdaToBindingName = Map.empty
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Binding (bindingName, _, _, _) ->
            // Check if first child is a Lambda
            match node.Children with
            | childId :: _ ->
                match Map.tryFind childId graph.Nodes with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ ->
                        // This Binding contains a Lambda - record the mapping
                        lambdaToBindingName <- Map.add (NodeId.value childId) bindingName lambdaToBindingName
                    | _ -> ()
                | None -> ()
            | [] -> ()
        | _ -> ()

    // Now assign names to all Lambdas
    // ARCHITECTURAL PRINCIPLE: When a Lambda is bound via `let name = ...`,
    // use the binding name as the function name. This preserves programmer intent.
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda _ ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                if Set.contains nodeIdVal entryPoints then
                    "main"
                else
                    // First check our reverse index (handles Parent = None case)
                    match Map.tryFind nodeIdVal lambdaToBindingName with
                    | Some bindingName -> bindingName
                    | None ->
                        // Fallback: check Parent field directly
                        match node.Parent with
                        | Some parentId ->
                            match Map.tryFind parentId graph.Nodes with
                            | Some parentNode ->
                                match parentNode.Kind with
                                | SemanticKind.Binding (bindingName, _, _, _) ->
                                    bindingName
                                | _ ->
                                    let n = sprintf "lambda_%d" lambdaCounter
                                    lambdaCounter <- lambdaCounter + 1
                                    n
                            | None ->
                                let n = sprintf "lambda_%d" lambdaCounter
                                lambdaCounter <- lambdaCounter + 1
                                n
                        | None ->
                            let n = sprintf "lambda_%d" lambdaCounter
                            lambdaCounter <- lambdaCounter + 1
                            n
            lambdaNames <- Map.add nodeIdVal name lambdaNames
            // CRITICAL: Also add the parent Binding's NodeId with the same name
            // VarRefs point to Bindings, not Lambdas. Both must resolve to the function name.
            match node.Parent with
            | Some parentId ->
                match Map.tryFind parentId graph.Nodes with
                | Some parentNode ->
                    match parentNode.Kind with
                    | SemanticKind.Binding _ ->
                        lambdaNames <- Map.add (NodeId.value parentId) name lambdaNames
                    | _ -> ()
                | None -> ()
            | None -> ()
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
            // Assign SSAs to parameter PatternBindings (Arg 0, Arg 1, etc.)
            // This allows VarRefs to parameters to look up their SSAs
            let paramScope =
                params'
                |> List.mapi (fun i (_name, _ty, nodeId) -> i, nodeId)
                |> List.fold (fun (scope: FunctionScope) (i, nodeId) ->
                    // Parameters get Arg N SSAs, mapped to their PatternBinding NodeId
                    FunctionScope.assign nodeId (NodeSSAAllocation.single (Arg i)) scope
                ) FunctionScope.empty

            // Assign SSAs to body nodes
            let bodyScope = assignFunctionBody graph paramScope bodyId

            // Merge into global assignments (including parameter SSAs)
            for kvp in paramScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments

            // CRITICAL: Lambda node itself needs an SSA for return value synthesis
            // Used by createDefaultReturn and reconcileReturnType in LambdaWitness
            // Use next available SSA in the Lambda's scope
            let lambdaSSA = V bodyScope.Counter
            allAssignments <- Map.add (NodeId.value node.Id) (NodeSSAAllocation.single lambdaSSA) allAssignments
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
