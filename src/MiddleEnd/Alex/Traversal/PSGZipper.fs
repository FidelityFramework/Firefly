/// PSGZipper - True Huet Zipper for PSG Navigation
///
/// A zipper is a data structure for navigating and modifying immutable trees.
/// It pairs your current focus with a path (breadcrumbs) showing how you got there.
///
/// References:
/// - Huet, "The Zipper" (1997) - original paper
/// - Tomas Petricek, "Tree zipper query expressions" - F# computation expressions
/// - Mark Seemann, "Zippers" (2024) - bidirectional navigation in functional code
///
/// CANONICAL ARCHITECTURE (January 2026):
/// The PSGZipper is PURELY navigational. It contains:
/// - Focus: The current SemanticNode we're examining
/// - Path: Breadcrumbs recording siblings left behind when navigating down
/// - Graph: The full SemanticGraph for node lookups
///
/// NOTHING ELSE goes in the zipper. Coeffects and accumulator state are separate.
/// See: mlir_transfer_canonical_architecture memory
module Alex.Traversal.PSGZipper

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// ZIPPER PATH (Breadcrumbs)
// ═══════════════════════════════════════════════════════════════════════════

/// A single step in the path, recording what we left behind when going down.
/// When navigating to child N, we record:
/// - The parent node we came from
/// - The siblings to the left of our chosen child
/// - The siblings to the right of our chosen child
type PathStep = {
    /// The parent node we descended from
    Parent: SemanticNode
    /// Siblings to the LEFT of our position (in order)
    LeftSiblings: NodeId list
    /// Siblings to the RIGHT of our position (in order)
    RightSiblings: NodeId list
}

/// The path from root to current focus (list of steps, most recent first)
type ZipperPath = PathStep list

// ═══════════════════════════════════════════════════════════════════════════
// PSG ZIPPER (Pure Navigation)
// ═══════════════════════════════════════════════════════════════════════════

/// The PSG Zipper - true Huet zipper for NAVIGATION ONLY
/// 
/// ARCHITECTURAL INVARIANT: This type contains Focus, Path, and Graph.
/// NO coeffects. NO accumulator state. NO mutable fields.
type PSGZipper = {
    /// The current focus node
    Focus: SemanticNode

    /// Path back to root (breadcrumbs)
    Path: ZipperPath

    /// The full semantic graph (for node lookups)
    Graph: SemanticGraph
}

// ═══════════════════════════════════════════════════════════════════════════
// ZIPPER CREATION
// ═══════════════════════════════════════════════════════════════════════════

/// Create a zipper focused on a specific node
let create (graph: SemanticGraph) (focusId: NodeId) : PSGZipper option =
    match SemanticGraph.tryGetNode focusId graph with
    | Some node ->
        Some {
            Focus = node
            Path = []  // At root, no path
            Graph = graph
        }
    | None -> None

/// Create a zipper at the first entry point
let fromEntryPoint (graph: SemanticGraph) : PSGZipper option =
    match graph.EntryPoints with
    | entryId :: _ -> create graph entryId
    | [] -> None

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════

/// Move UP to the parent node
/// Returns None if already at root
let up (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, cannot go up
    | step :: restPath ->
        Some {
            Focus = step.Parent
            Path = restPath
            Graph = z.Graph
        }

/// Move DOWN to a specific child by index
/// Records siblings left behind in the path
let down (childIndex: int) (z: PSGZipper) : PSGZipper option =
    let children = z.Focus.Children
    if childIndex < 0 || childIndex >= List.length children then
        None
    else
        let childId = List.item childIndex children
        match SemanticGraph.tryGetNode childId z.Graph with
        | None -> None
        | Some childNode ->
            let leftSiblings = List.take childIndex children
            let rightSiblings = List.skip (childIndex + 1) children
            let step = {
                Parent = z.Focus
                LeftSiblings = leftSiblings
                RightSiblings = rightSiblings
            }
            Some {
                Focus = childNode
                Path = step :: z.Path
                Graph = z.Graph
            }

/// Move DOWN to the first child
let downFirst (z: PSGZipper) : PSGZipper option =
    down 0 z

/// Move LEFT to the previous sibling
let left (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, no siblings
    | step :: restPath ->
        match List.tryLast step.LeftSiblings with
        | None -> None  // No left siblings
        | Some leftId ->
            match SemanticGraph.tryGetNode leftId z.Graph with
            | None -> None
            | Some leftNode ->
                let newLeftSiblings = List.take (List.length step.LeftSiblings - 1) step.LeftSiblings
                let newRightSiblings = z.Focus.Id :: step.RightSiblings
                let newStep = {
                    Parent = step.Parent
                    LeftSiblings = newLeftSiblings
                    RightSiblings = newRightSiblings
                }
                Some {
                    Focus = leftNode
                    Path = newStep :: restPath
                    Graph = z.Graph
                }

/// Move RIGHT to the next sibling
let right (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, no siblings
    | step :: restPath ->
        match step.RightSiblings with
        | [] -> None  // No right siblings
        | rightId :: remainingRight ->
            match SemanticGraph.tryGetNode rightId z.Graph with
            | None -> None
            | Some rightNode ->
                let newLeftSiblings = step.LeftSiblings @ [z.Focus.Id]
                let newStep = {
                    Parent = step.Parent
                    LeftSiblings = newLeftSiblings
                    RightSiblings = remainingRight
                }
                Some {
                    Focus = rightNode
                    Path = newStep :: restPath
                    Graph = z.Graph
                }

/// Navigate to a specific node by ID (re-roots the zipper there)
let focusOn (nodeId: NodeId) (z: PSGZipper) : PSGZipper option =
    match SemanticGraph.tryGetNode nodeId z.Graph with
    | None -> None
    | Some node ->
        Some {
            Focus = node
            Path = []  // Re-rooted, path cleared
            Graph = z.Graph
        }

// ═══════════════════════════════════════════════════════════════════════════
// FOCUS QUERIES
// ═══════════════════════════════════════════════════════════════════════════

/// Get the current focus node
let focus (z: PSGZipper) : SemanticNode = z.Focus

/// Get the focus node's ID
let focusId (z: PSGZipper) : NodeId = z.Focus.Id

/// Get the focus node's Kind
let focusKind (z: PSGZipper) : SemanticKind = z.Focus.Kind

/// Get the focus node's Type
let focusType (z: PSGZipper) : NativeType = z.Focus.Type

/// Check if at root (no path)
let isAtRoot (z: PSGZipper) : bool = List.isEmpty z.Path

/// Get depth in tree (path length)
let depth (z: PSGZipper) : int = List.length z.Path

/// Get child count of focus
let childCount (z: PSGZipper) : int = List.length z.Focus.Children

/// Check if focus has children
let hasChildren (z: PSGZipper) : bool = not (List.isEmpty z.Focus.Children)

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH LOOKUPS
// ═══════════════════════════════════════════════════════════════════════════

/// Get a node from the graph by ID
let getNode (nodeId: NodeId) (z: PSGZipper) : SemanticNode option =
    SemanticGraph.tryGetNode nodeId z.Graph

/// Get a node, failing if not found
let requireNode (nodeId: NodeId) (z: PSGZipper) : SemanticNode =
    match getNode nodeId z with
    | Some node -> node
    | None -> failwithf "Node %A not found in graph" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// PATH INSPECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Find the enclosing Lambda by walking up the zipper's path.
/// This is the DEFINITIVE source of truth for "are we inside a function?"
/// Returns Some (lambdaNode, lambdaParams) if inside a Lambda, None if at module level.
let findEnclosingLambda (z: PSGZipper) : SemanticNode option =
    // First check if the current focus IS a Lambda
    match z.Focus.Kind with
    | SemanticKind.Lambda _ -> Some z.Focus
    | _ ->
        // Walk up the path looking for a Lambda parent
        z.Path
        |> List.tryPick (fun step ->
            match step.Parent.Kind with
            | SemanticKind.Lambda _ -> Some step.Parent
            | _ -> None)
