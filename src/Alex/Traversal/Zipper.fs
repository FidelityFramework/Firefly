/// Zipper - True Huet Zipper for PSG Navigation
///
/// ARCHITECTURAL PRINCIPLE:
/// The Zipper is the "camera" - it moves focus through the PSG.
/// It carries read-only Coeffects for "What is?" lookups.
/// It does NOT carry mutable state or accumulators.
///
/// The fold ACCUMULATES what witnesses RETURN.
/// The Zipper just provides focus and context.
module Alex.Traversal.Zipper

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.Coeffects

// ═══════════════════════════════════════════════════════════════════════════
// ZIPPER PATH (Breadcrumbs)
// ═══════════════════════════════════════════════════════════════════════════

/// A single step in the path, recording what we left behind when going down.
type PathStep = {
    /// The parent node we descended from
    Parent: SemanticNode
    /// Siblings to the LEFT of our position
    LeftSiblings: NodeId list
    /// Siblings to the RIGHT of our position
    RightSiblings: NodeId list
}

/// The path from root to current focus (most recent first)
type ZipperPath = PathStep list

// ═══════════════════════════════════════════════════════════════════════════
// THE ZIPPER
// ═══════════════════════════════════════════════════════════════════════════

/// The PSG Zipper - navigation + read-only context
/// This is the "camera" that witnesses use to observe the scene.
type Zipper = {
    /// Current focus node
    Focus: SemanticNode

    /// Path back to root (breadcrumbs)
    Path: ZipperPath

    /// The full semantic graph (for node lookups)
    Graph: SemanticGraph

    /// Read-only coeffects from nanopasses
    Coeffects: Coeffects
}

// ═══════════════════════════════════════════════════════════════════════════
// CREATION
// ═══════════════════════════════════════════════════════════════════════════

/// Create a zipper focused on a specific node
let create (graph: SemanticGraph) (focusId: NodeId) (coeffects: Coeffects) : Zipper option =
    match Map.tryFind focusId graph.Nodes with
    | Some node ->
        Some {
            Focus = node
            Path = []
            Graph = graph
            Coeffects = coeffects
        }
    | None -> None

/// Create a zipper focused on the entry point
let fromEntryPoint (graph: SemanticGraph) (coeffects: Coeffects) : Zipper option =
    match graph.EntryPoints with
    | [] -> None
    | entryId :: _ -> create graph entryId coeffects

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════

/// Move down to the first child
let down (z: Zipper) : Zipper option =
    match z.Focus.Children with
    | [] -> None
    | firstChild :: rest ->
        match Map.tryFind firstChild z.Graph.Nodes with
        | Some childNode ->
            Some {
                z with
                    Focus = childNode
                    Path = {
                        Parent = z.Focus
                        LeftSiblings = []
                        RightSiblings = rest
                    } :: z.Path
            }
        | None -> None

/// Move down to a specific child by index
let downTo (index: int) (z: Zipper) : Zipper option =
    let children = z.Focus.Children
    if index < 0 || index >= List.length children then
        None
    else
        let leftSibs = List.take index children
        let target = List.item index children
        let rightSibs = List.skip (index + 1) children
        match Map.tryFind target z.Graph.Nodes with
        | Some childNode ->
            Some {
                z with
                    Focus = childNode
                    Path = {
                        Parent = z.Focus
                        LeftSiblings = leftSibs
                        RightSiblings = rightSibs
                    } :: z.Path
            }
        | None -> None

/// Move up to the parent
let up (z: Zipper) : Zipper option =
    match z.Path with
    | [] -> None
    | step :: rest ->
        Some {
            z with
                Focus = step.Parent
                Path = rest
        }

/// Move right to the next sibling
let right (z: Zipper) : Zipper option =
    match z.Path with
    | [] -> None
    | step :: rest ->
        match step.RightSiblings with
        | [] -> None
        | nextSib :: remaining ->
            match Map.tryFind nextSib z.Graph.Nodes with
            | Some sibNode ->
                Some {
                    z with
                        Focus = sibNode
                        Path = {
                            step with
                                LeftSiblings = step.LeftSiblings @ [z.Focus.Id]
                                RightSiblings = remaining
                        } :: rest
                }
            | None -> None

/// Move left to the previous sibling
let left (z: Zipper) : Zipper option =
    match z.Path with
    | [] -> None
    | step :: rest ->
        match List.tryLast step.LeftSiblings with
        | None -> None
        | Some prevSib ->
            match Map.tryFind prevSib z.Graph.Nodes with
            | Some sibNode ->
                Some {
                    z with
                        Focus = sibNode
                        Path = {
                            step with
                                LeftSiblings = List.take (List.length step.LeftSiblings - 1) step.LeftSiblings
                                RightSiblings = z.Focus.Id :: step.RightSiblings
                        } :: rest
                }
            | None -> None

/// Move to root
let rec root (z: Zipper) : Zipper =
    match up z with
    | Some parent -> root parent
    | None -> z

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT LOOKUPS (delegated for convenience)
// ═══════════════════════════════════════════════════════════════════════════

/// Look up SSA for the focused node
let focusSSA (z: Zipper) : SSA option =
    Coeffects.lookupSSA z.Focus.Id z.Coeffects

/// Look up SSAs for the focused node
let focusSSAs (z: Zipper) : SSA list option =
    Coeffects.lookupSSAs z.Focus.Id z.Coeffects

/// Look up Lambda name if focused on a Lambda
let focusLambdaName (z: Zipper) : string option =
    Coeffects.lookupLambdaName z.Focus.Id z.Coeffects

/// Check if focused Lambda is entry point
let focusIsEntryPoint (z: Zipper) : bool =
    Coeffects.isEntryPoint z.Focus.Id z.Coeffects

// ═══════════════════════════════════════════════════════════════════════════
// NODE LOOKUPS
// ═══════════════════════════════════════════════════════════════════════════

/// Get a node by ID from the graph
let getNode (nodeId: NodeId) (z: Zipper) : SemanticNode option =
    Map.tryFind nodeId z.Graph.Nodes

/// Get children nodes
let getChildren (z: Zipper) : SemanticNode list =
    z.Focus.Children
    |> List.choose (fun id -> Map.tryFind id z.Graph.Nodes)

/// Get parent node
let getParent (z: Zipper) : SemanticNode option =
    match z.Path with
    | [] -> None
    | step :: _ -> Some step.Parent
