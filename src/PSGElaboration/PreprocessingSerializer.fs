/// Preprocessing Serializer - Serialize PSGElaboration results to intermediates
///
/// This module serializes preprocessing results (SSA assignment, Lambda names,
/// entry points, etc.) to JSON for inspection. Following the FNCS phase emission
/// pattern, this enables architectural debugging through intermediates rather than
/// console output.
///
/// Note: These results are "coeffects" in the sense that they are pre-computed
/// requirements that the traversal reads from (not writes to).
///
/// CRITICAL: This serializer provides FULL visibility into SSA assignments.
/// When debugging "wrong SSA" issues, the alex_coeffects.json file shows
/// exactly what SSA was assigned to each node and WHY.
module PSGElaboration.PreprocessingSerializer

open System.IO
open System.Text.Json
open FSharp.Native.Compiler.PSG.SemanticGraph
open PSGElaboration.SSAAssignment
open Alex.Dialects.Core.Types

/// Detailed lambda information for debugging
type LambdaInfoJson = {
    NodeId: int
    Name: string
    Type: string
    ParamCount: int
}

/// Serializable SSA value
type SSAJson = {
    Kind: string      // "V" or "Arg"
    Index: int        // The numeric index
    Display: string   // "%v42" or "%arg0"
}

/// Serializable node SSA allocation
type NodeSSAAllocationJson = {
    NodeId: int
    NodeKind: string
    SSAs: SSAJson list
    ResultSSA: SSAJson
    ParentFunction: string option  // Which function scope this belongs to
}

/// Serializable closure layout
type ClosureLayoutJson = {
    LambdaNodeId: int
    LambdaName: string
    CaptureCount: int
    Captures: {| Name: string; SlotIndex: int; CaptureKind: string; SourceSSA: string |} list
    ClosureStructType: string
}

/// Serializable representation of SSA assignment
type SSAAssignmentJson = {
    LambdaNames: (int * string) list
    Lambdas: LambdaInfoJson list
    EntryPointLambdas: int list
    NodeSSACount: int
    /// FULL SSA assignments - essential for debugging
    NodeSSAAssignments: NodeSSAAllocationJson list
    /// Closure layouts for lambdas with captures
    ClosureLayouts: ClosureLayoutJson list
}

/// Convert SSA to JSON representation
let private ssaToJson (ssa: SSA) : SSAJson =
    match ssa with
    | V n -> { Kind = "V"; Index = n; Display = sprintf "%%v%d" n }
    | Arg n -> { Kind = "Arg"; Index = n; Display = sprintf "%%arg%d" n }

/// Get a short description of a SemanticKind
let private kindToShortString (kind: SemanticKind) : string =
    match kind with
    | SemanticKind.Binding (name, isMut, _, _) ->
        sprintf "Binding(%s%s)" name (if isMut then ", mutable" else "")
    | SemanticKind.VarRef (name, defId) ->
        sprintf "VarRef(%s -> %s)" name (defId |> Option.map (fun id -> string (NodeId.value id)) |> Option.defaultValue "?")
    | SemanticKind.Lambda (params', _, captures, _, _) ->
        sprintf "Lambda(%d params, %d captures)" (List.length params') (List.length captures)
    | SemanticKind.Application (funcId, argIds) ->
        sprintf "Application(%d, [%s])" (NodeId.value funcId) (argIds |> List.map (NodeId.value >> string) |> String.concat ", ")
    | SemanticKind.Literal lit -> sprintf "Literal(%A)" lit
    | SemanticKind.Intrinsic info -> sprintf "Intrinsic(%s)" info.FullName
    | SemanticKind.Sequential ids -> sprintf "Sequential(%d items)" (List.length ids)
    | SemanticKind.IfThenElse _ -> "IfThenElse"
    | SemanticKind.WhileLoop _ -> "WhileLoop"
    | SemanticKind.ForLoop _ -> "ForLoop"
    | SemanticKind.Set (targetId, valueId) -> sprintf "Set(%d <- %d)" (NodeId.value targetId) (NodeId.value valueId)
    | SemanticKind.TypeAnnotation (inner, _) -> sprintf "TypeAnnotation(%d)" (NodeId.value inner)
    | SemanticKind.PatternBinding name -> sprintf "PatternBinding(%s)" name
    | kind -> sprintf "%A" kind |> fun s -> if s.Length > 50 then s.Substring(0, 47) + "..." else s

/// Find which function a node belongs to (for context)
let private findParentFunction (nodeId: int) (graph: SemanticGraph) (lambdaNames: Map<int, string>) : string option =
    // Walk up parent chain to find enclosing Lambda
    let rec findLambda currentId visited =
        if Set.contains currentId visited then None
        else
            match Map.tryFind (NodeId currentId) graph.Nodes with
            | Some node ->
                match node.Kind with
                | SemanticKind.Lambda _ ->
                    lambdaNames |> Map.tryFind currentId
                | _ ->
                    match node.Parent with
                    | Some parentId -> findLambda (NodeId.value parentId) (Set.add currentId visited)
                    | None -> None
            | None -> None
    findLambda nodeId Set.empty

/// Serialize SSA assignment to JSON-friendly structure
let serializeSSAAssignment (ssa: SSAAssignment) (graph: SemanticGraph) : SSAAssignmentJson =
    let lambdas =
        graph.Nodes.Values
        |> Seq.choose (fun node ->
            match node.Kind with
            | SemanticKind.Lambda(params', _, _captures, _, _) ->
                let nodeIdVal = NodeId.value node.Id
                let name = ssa.LambdaNames |> Map.tryFind nodeIdVal |> Option.defaultValue "unknown"
                Some {
                    NodeId = nodeIdVal
                    Name = name
                    Type = sprintf "%A" node.Type
                    ParamCount = List.length params'
                }
            | _ -> None)
        |> Seq.toList

    // FULL SSA assignments with node context
    let nodeSSAAssignments =
        ssa.NodeSSA
        |> Map.toList
        |> List.sortBy fst
        |> List.map (fun (nodeId, alloc) ->
            let nodeKind =
                match Map.tryFind (NodeId nodeId) graph.Nodes with
                | Some node -> kindToShortString node.Kind
                | None -> "Unknown"
            let parentFunc = findParentFunction nodeId graph ssa.LambdaNames
            {
                NodeId = nodeId
                NodeKind = nodeKind
                SSAs = alloc.SSAs |> List.map ssaToJson
                ResultSSA = ssaToJson alloc.Result
                ParentFunction = parentFunc
            })

    // Closure layouts
    let closureLayouts =
        ssa.ClosureLayouts
        |> Map.toList
        |> List.map (fun (nodeId, layout) ->
            let lambdaName = ssa.LambdaNames |> Map.tryFind nodeId |> Option.defaultValue "unknown"
            {
                LambdaNodeId = nodeId
                LambdaName = lambdaName
                CaptureCount = List.length layout.Captures
                Captures =
                    layout.Captures
                    |> List.map (fun cap ->
                        {| Name = cap.Name
                           SlotIndex = cap.SlotIndex
                           CaptureKind = sprintf "%A" cap.Mode
                           SourceSSA = cap.SourceNodeId |> Option.map (fun id -> sprintf "Node %d" (NodeId.value id)) |> Option.defaultValue "?" |})
                ClosureStructType = sprintf "%A" layout.ClosureStructType
            })

    {
        LambdaNames = ssa.LambdaNames |> Map.toList
        Lambdas = lambdas
        EntryPointLambdas = ssa.EntryPointLambdas |> Set.toList
        NodeSSACount = Map.count ssa.NodeSSA
        NodeSSAAssignments = nodeSSAAssignments
        ClosureLayouts = closureLayouts
    }

/// Serialize preprocessing results to intermediates directory
let serialize
    (intermediatesDir: string)
    (ssaAssignment: SSAAssignment)
    (entryPointLambdaIds: Set<int>)
    (graph: SemanticGraph)
    : unit =
    
    let coeffectsPath = Path.Combine(intermediatesDir, "alex_coeffects.json")
    
    let data = {|
        version = "1.0"
        description = "PSGElaboration coeffects"
        ssaAssignment = serializeSSAAssignment ssaAssignment graph
        entryPointLambdaIds = entryPointLambdaIds |> Set.toList
    |}
    
    let options = JsonSerializerOptions(WriteIndented = true)
    let json = JsonSerializer.Serialize(data, options)
    File.WriteAllText(coeffectsPath, json)
