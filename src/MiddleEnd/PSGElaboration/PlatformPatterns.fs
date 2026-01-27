/// Platform operation recognition patterns (following Farscape pattern)
///
/// These active patterns recognize Sys.* intrinsic calls in the PSG and extract
/// their arguments for quotation-based MLIR generation.
module PSGElaboration.PlatformPatterns

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// HELPER PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Match an intrinsic node with specific module and operation
let (|SysIntrinsic|_|) (operation: string) (node: SemanticNode) : unit option =
    match node.Kind with
    | SemanticKind.Intrinsic info when
        info.Module = IntrinsicModule.Sys && info.Operation = operation ->
        Some ()
    | _ -> None

/// Extract SSA value from a node
let extractSSA (node: SemanticNode) : SSA =
    match node.AdditionalData.TryFind "SSA" with
    | Some (SSAValue ssa) -> ssa
    | _ -> failwithf "Node %d has no SSA value" node.Id

/// Extract integer constant from a node
let extractIntConstant (node: SemanticNode) : int option =
    match node.Kind with
    | SemanticKind.Literal lit ->
        match lit.Value with
        | LiteralValue.Int i -> Some i
        | _ -> None
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// SYS.* INTRINSIC PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Recognize Sys.write: fd -> buffer -> count -> int
/// Application node with Intrinsic { Module = Sys, Operation = "write" } + 3 args
let (|SysWrite|_|) (graph: SemanticGraph) (node: SemanticNode) : (int * SSA * SSA) option =
    match node.Kind with
    | SemanticKind.Application app ->
        // Get the function node (first child)
        match Map.tryFind app.FuncId graph.Nodes with
        | Some funcNode ->
            match funcNode with
            | SysIntrinsic "write" _ ->
                // Extract arguments: fd, buffer, count
                match app.Args with
                | [fdId; bufId; countId] ->
                    match Map.tryFind fdId graph.Nodes,
                          Map.tryFind bufId graph.Nodes,
                          Map.tryFind countId graph.Nodes with
                    | Some fdNode, Some bufNode, Some countNode ->
                        // Try to extract fd as a constant
                        match extractIntConstant fdNode with
                        | Some fd ->
                            let bufSSA = extractSSA bufNode
                            let countSSA = extractSSA countNode
                            Some (fd, bufSSA, countSSA)
                        | None -> None
                    | _ -> None
                | _ -> None
            | _ -> None
        | None -> None
    | _ -> None

/// Recognize Sys.read: fd -> buffer -> count -> int
let (|SysRead|_|) (graph: SemanticGraph) (node: SemanticNode) : (int * SSA * SSA) option =
    match node.Kind with
    | SemanticKind.Application app ->
        match Map.tryFind app.FuncId graph.Nodes with
        | Some funcNode ->
            match funcNode with
            | SysIntrinsic "read" _ ->
                match app.Args with
                | [fdId; bufId; countId] ->
                    match Map.tryFind fdId graph.Nodes,
                          Map.tryFind bufId graph.Nodes,
                          Map.tryFind countId graph.Nodes with
                    | Some fdNode, Some bufNode, Some countNode ->
                        match extractIntConstant fdNode with
                        | Some fd ->
                            let bufSSA = extractSSA bufNode
                            let countSSA = extractSSA countNode
                            Some (fd, bufSSA, countSSA)
                        | None -> None
                    | _ -> None
                | _ -> None
            | _ -> None
        | None -> None
    | _ -> None

/// Recognize Sys.exit: code -> unit
let (|SysExit|_|) (graph: SemanticGraph) (node: SemanticNode) : SSA option =
    match node.Kind with
    | SemanticKind.Application app ->
        match Map.tryFind app.FuncId graph.Nodes with
        | Some funcNode ->
            match funcNode with
            | SysIntrinsic "exit" _ ->
                match app.Args with
                | [codeId] ->
                    match Map.tryFind codeId graph.Nodes with
                    | Some codeNode ->
                        Some (extractSSA codeNode)
                    | None -> None
                | _ -> None
            | _ -> None
        | None -> None
    | _ -> None

/// Recognize Sys.nanosleep: req -> rem -> int
let (|SysNanosleep|_|) (graph: SemanticGraph) (node: SemanticNode) : (SSA * SSA) option =
    match node.Kind with
    | SemanticKind.Application app ->
        match Map.tryFind app.FuncId graph.Nodes with
        | Some funcNode ->
            match funcNode with
            | SysIntrinsic "nanosleep" _ ->
                match app.Args with
                | [reqId; remId] ->
                    match Map.tryFind reqId graph.Nodes,
                          Map.tryFind remId graph.Nodes with
                    | Some reqNode, Some remNode ->
                        let reqSSA = extractSSA reqNode
                        let remSSA = extractSSA remNode
                        Some (reqSSA, remSSA)
                    | _ -> None
                | _ -> None
            | _ -> None
        | None -> None
    | _ -> None

/// Recognize Sys.clock_gettime: unit -> int64
let (|SysClockGetTime|_|) (graph: SemanticGraph) (node: SemanticNode) : unit option =
    match node.Kind with
    | SemanticKind.Application app ->
        match Map.tryFind app.FuncId graph.Nodes with
        | Some funcNode ->
            match funcNode with
            | SysIntrinsic "clock_gettime" _ ->
                Some ()
            | _ -> None
        | None -> None
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITE RECOGNITION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Recognize any platform operation (analogous to Farscape's recognizeMemoryOperation)
let recognizePlatformOperation (graph: SemanticGraph) (node: SemanticNode) : PlatformOperation option =
    match node with
    | SysWrite (graph) (fd, buffer, count) ->
        Some (PlatformOperation.SysWrite (fd, buffer, count))
    | SysRead (graph) (fd, buffer, count) ->
        Some (PlatformOperation.SysRead (fd, buffer, count))
    | SysExit (graph) code ->
        Some (PlatformOperation.SysExit code)
    | SysNanosleep (graph) (req, rem) ->
        Some (PlatformOperation.SysNanosleep (req, rem))
    | SysClockGetTime (graph) _ ->
        Some PlatformOperation.SysClockGetTime
    | _ -> None
