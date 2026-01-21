/// Preprocessing Serializer - Serialize PSGElaboration results to intermediates
///
/// This module serializes ALL preprocessing coeffects to JSON for inspection.
/// Following the FNCS phase emission pattern, this enables architectural
/// debugging through intermediates rather than console output.
///
/// "Pierce the Veil" Infrastructure (January 2026):
/// - SSA assignments: Which SSA was assigned to each node and WHY
/// - Mutability analysis: Which bindings need alloca, which vars are modified in loops
/// - Yield state indices: Seq state machine structure (critical for seq debugging)
/// - Pattern bindings: What variables emerge from each pattern match
/// - String table: All string literals and their global names
///
/// All coeffects are serialized to alex_coeffects.json when -k flag is used.
module PSGElaboration.PreprocessingSerializer

open System.IO
open System.Text.Json
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open PSGElaboration.Coeffects
open PSGElaboration.SSAAssignment
open PSGElaboration.MutabilityAnalysis
open PSGElaboration.YieldStateIndices
open PSGElaboration.PatternBindingAnalysis
open PSGElaboration.StringCollection
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

// ═══════════════════════════════════════════════════════════════════════════
// MUTABILITY ANALYSIS JSON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Serializable module-level mutable binding
type ModuleLevelMutableJson = {
    BindingId: int
    Name: string
    InitialValueId: int
}

/// Serializable mutability analysis result
type MutabilityAnalysisJson = {
    /// Mutable bindings whose address is taken (need alloca)
    AddressedMutableBindings: int list
    /// Map from loop body NodeId to modified variable names
    ModifiedVarsInLoopBodies: (int * string list) list
    /// Module-level mutable bindings (need LLVM globals)
    ModuleLevelMutableBindings: ModuleLevelMutableJson list
}

// ═══════════════════════════════════════════════════════════════════════════
// YIELD STATE INDICES JSON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Serializable yield info
type YieldInfoJson = {
    YieldId: int
    StateIndex: int
    ValueId: int
}

/// Serializable internal state field
type InternalStateFieldJson = {
    Name: string
    Type: string
    BindingId: int
    StructIndex: int
}

/// Serializable while body info
type WhileBodyInfoJson = {
    InitExprs: int list
    WhileNodeId: int
    ConditionId: int
    PreYieldExprs: int list
    YieldNodeId: int
    YieldValueId: int
    PostYieldExprs: int list
    HasConditionalYield: bool
    ConditionalConditionIds: int list
}

/// Serializable seq yield info
type SeqYieldInfoJson = {
    SeqExprId: int
    Yields: YieldInfoJson list
    NumYields: int
    BodyKind: string  // "Sequential" or "WhileBased"
    WhileBodyInfo: WhileBodyInfoJson option
    InternalState: InternalStateFieldJson list
}

/// Serializable yield state coeffect
type YieldStateCoeffectJson = {
    SeqYields: SeqYieldInfoJson list
}

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN BINDING ANALYSIS JSON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Serializable pattern binding
type PatternBindingJson = {
    Name: string
    Type: string
}

/// Serializable pattern binding analysis
type PatternBindingAnalysisJson = {
    /// Map from case body NodeId to bindings
    CasePatternBindings: (int * PatternBindingJson list) list
    /// Map from Lambda NodeId to entry bindings
    EntryPatternBindings: (int * PatternBindingJson list) list
}

// ═══════════════════════════════════════════════════════════════════════════
// STRING TABLE JSON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Serializable string entry
type StringEntryJson = {
    Hash: uint32
    GlobalName: string
    Content: string
    ByteLength: int
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

// ═══════════════════════════════════════════════════════════════════════════
// MUTABILITY ANALYSIS SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize mutability analysis to JSON-friendly structure
let serializeMutabilityAnalysis (mutability: MutabilityAnalysisResult) : MutabilityAnalysisJson =
    {
        AddressedMutableBindings = mutability.AddressedMutableBindings |> Set.toList
        ModifiedVarsInLoopBodies = mutability.ModifiedVarsInLoopBodies |> Map.toList
        ModuleLevelMutableBindings =
            mutability.ModuleLevelMutableBindings
            |> List.map (fun m -> {
                BindingId = m.BindingId
                Name = m.Name
                InitialValueId = m.InitialValueId
            })
    }

// ═══════════════════════════════════════════════════════════════════════════
// YIELD STATE INDICES SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize yield state coeffect to JSON-friendly structure
let serializeYieldStateCoeffect (yieldState: YieldStateCoeffect) : YieldStateCoeffectJson =
    let seqYields =
        yieldState.SeqYields
        |> Map.toList
        |> List.map (fun (_, seqInfo) ->
            let bodyKind, whileInfo =
                match seqInfo.BodyKind with
                | SeqBodyKind.Sequential -> "Sequential", None
                | SeqBodyKind.WhileBased info ->
                    let whileJson = {
                        InitExprs = info.InitExprs |> List.map NodeId.value
                        WhileNodeId = NodeId.value info.WhileNodeId
                        ConditionId = NodeId.value info.ConditionId
                        PreYieldExprs = info.PreYieldExprs |> List.map NodeId.value
                        YieldNodeId = NodeId.value info.YieldNodeId
                        YieldValueId = NodeId.value info.YieldValueId
                        PostYieldExprs = info.PostYieldExprs |> List.map NodeId.value
                        HasConditionalYield = Option.isSome info.ConditionalYield
                        ConditionalConditionIds =
                            info.ConditionalYield
                            |> Option.map (fun c -> c.ConditionIds |> List.map NodeId.value)
                            |> Option.defaultValue []
                    }
                    "WhileBased", Some whileJson

            {
                SeqExprId = NodeId.value seqInfo.SeqExprId
                Yields =
                    seqInfo.Yields
                    |> List.map (fun yi -> {
                        YieldId = NodeId.value yi.YieldId
                        StateIndex = yi.StateIndex
                        ValueId = NodeId.value yi.ValueId
                    })
                NumYields = seqInfo.NumYields
                BodyKind = bodyKind
                WhileBodyInfo = whileInfo
                InternalState =
                    seqInfo.InternalState
                    |> List.map (fun field -> {
                        Name = field.Name
                        Type = sprintf "%A" field.Type
                        BindingId = NodeId.value field.BindingId
                        StructIndex = field.StructIndex
                    })
            })
    { SeqYields = seqYields }

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN BINDING ANALYSIS SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize pattern binding analysis to JSON-friendly structure
let serializePatternBindingAnalysis (patternBindings: PatternBindingAnalysisResult) : PatternBindingAnalysisJson =
    let serializeBindings (bindings: PatternBindingAnalysis.PatternBinding list) =
        bindings |> List.map (fun b -> { Name = b.Name; Type = sprintf "%A" b.Type })
    {
        CasePatternBindings =
            patternBindings.CasePatternBindings
            |> Map.toList
            |> List.map (fun (nodeId, bindings) -> (nodeId, serializeBindings bindings))
        EntryPatternBindings =
            patternBindings.EntryPatternBindings
            |> Map.toList
            |> List.map (fun (nodeId, bindings) -> (nodeId, serializeBindings bindings))
    }

// ═══════════════════════════════════════════════════════════════════════════
// STRING TABLE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize string table to JSON-friendly structure
let serializeStringTable (strings: StringTable) : StringEntryJson list =
    strings
    |> Map.toList
    |> List.map (fun (hash, entry) -> {
        Hash = hash
        GlobalName = entry.GlobalName
        Content = entry.Content
        ByteLength = entry.ByteLength
    })

// ═══════════════════════════════════════════════════════════════════════════
// MAIN SERIALIZATION ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize ALL preprocessing coeffects to intermediates directory
/// This is the "pierce the veil" infrastructure for debugging the nanopass pipeline
let serialize
    (intermediatesDir: string)
    (ssaAssignment: SSAAssignment)
    (entryPointLambdaIds: Set<int>)
    (graph: SemanticGraph)
    : unit =
    
    let coeffectsPath = Path.Combine(intermediatesDir, "alex_coeffects.json")
    
    // Note: Full coeffects are passed via the new serializeAll function
    // This legacy signature is maintained for backward compatibility
    let data = {|
        version = "1.0"
        description = "PSGElaboration coeffects (legacy - use serializeAll for full coeffects)"
        ssaAssignment = serializeSSAAssignment ssaAssignment graph
        entryPointLambdaIds = entryPointLambdaIds |> Set.toList
    |}
    
    let options = JsonSerializerOptions(WriteIndented = true)
    let json = JsonSerializer.Serialize(data, options)
    File.WriteAllText(coeffectsPath, json)

/// Serialize ALL preprocessing coeffects to intermediates directory
/// "Pierce the Veil" - complete visibility into nanopass infrastructure
let serializeAll
    (intermediatesDir: string)
    (ssaAssignment: SSAAssignment)
    (mutability: MutabilityAnalysisResult)
    (yieldStates: YieldStateCoeffect)
    (patternBindings: PatternBindingAnalysisResult)
    (strings: StringTable)
    (entryPointLambdaIds: Set<int>)
    (graph: SemanticGraph)
    : unit =
    
    let coeffectsPath = Path.Combine(intermediatesDir, "alex_coeffects.json")
    
    let data = {|
        version = "2.0"
        description = "PSGElaboration coeffects - complete nanopass visibility"
        
        // SSA Assignment (existing)
        ssaAssignment = serializeSSAAssignment ssaAssignment graph
        entryPointLambdaIds = entryPointLambdaIds |> Set.toList
        
        // Mutability Analysis (new)
        mutability = serializeMutabilityAnalysis mutability
        
        // Yield State Indices (new - critical for seq debugging)
        yieldStates = serializeYieldStateCoeffect yieldStates
        
        // Pattern Binding Analysis (new)
        patternBindings = serializePatternBindingAnalysis patternBindings
        
        // String Table (new)
        strings = serializeStringTable strings
    |}
    
    let options = JsonSerializerOptions(WriteIndented = true)
    let json = JsonSerializer.Serialize(data, options)
    File.WriteAllText(coeffectsPath, json)
