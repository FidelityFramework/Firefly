/// Preprocessing Serializer - Serialize Alex preprocessing results to intermediates
///
/// This module serializes preprocessing results (SSA assignment, Lambda names,
/// entry points, etc.) to JSON for inspection. Following the FNCS phase emission
/// pattern, this enables architectural debugging through intermediates rather than
/// console output.
///
/// Note: These results are "coeffects" in the sense that they are pre-computed
/// requirements that the traversal reads from (not writes to).
module Alex.Preprocessing.PreprocessingSerializer

open System.IO
open System.Text.Json
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Preprocessing.SSAAssignment

/// Detailed lambda information for debugging
type LambdaInfoJson = {
    NodeId: int
    Name: string
    Type: string
    ParamCount: int
}

/// Serializable representation of SSA assignment
type SSAAssignmentJson = {
    LambdaNames: (int * string) list
    Lambdas: LambdaInfoJson list
    EntryPointLambdas: int list
    NodeSSACount: int
    NodeSSAKeys: int list
}

/// Serialize SSA assignment to JSON-friendly structure
let serializeSSAAssignment (ssa: SSAAssignment) (graph: SemanticGraph) : SSAAssignmentJson =
    let lambdas = 
        graph.Nodes.Values
        |> Seq.choose (fun node ->
            match node.Kind with
            | SemanticKind.Lambda(params', _) ->
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

    {
        LambdaNames = ssa.LambdaNames |> Map.toList
        Lambdas = lambdas
        EntryPointLambdas = ssa.EntryPointLambdas |> Set.toList
        NodeSSACount = Map.count ssa.NodeSSA
        NodeSSAKeys = ssa.NodeSSA |> Map.keys |> Seq.sort |> Seq.toList
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
        description = "Alex preprocessing coeffects"
        ssaAssignment = serializeSSAAssignment ssaAssignment graph
        entryPointLambdaIds = entryPointLambdaIds |> Set.toList
    |}
    
    let options = JsonSerializerOptions(WriteIndented = true)
    let json = JsonSerializer.Serialize(data, options)
    File.WriteAllText(coeffectsPath, json)
