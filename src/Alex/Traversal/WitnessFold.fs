/// WitnessFold - The accumulator pattern for witnessing
///
/// ARCHITECTURAL PRINCIPLE:
/// Witnesses OBSERVE and RETURN MLIROp lists.
/// The fold ACCUMULATES what witnesses return.
/// There is ONE output: the accumulated ops.
///
/// "The photographer doesn't build the scene - they arrive AFTER
/// the scene is complete and witness it."
module Alex.Traversal.WitnessFold

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.Zipper

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// What a witness returns after observing a node.
/// The ops are accumulated by the fold.
type WitnessResult = {
    /// MLIR operations this node produces
    Ops: MLIROp list

    /// Optional result value (SSA + type) for parent nodes to reference
    Result: Val option
}

module WitnessResult =
    let empty = { Ops = []; Result = None }
    let ops lst = { Ops = lst; Result = None }
    let withResult ops result = { Ops = ops; Result = Some result }
    let value v = { Ops = []; Result = Some v }

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS SIGNATURE
// ═══════════════════════════════════════════════════════════════════════════

/// A witness function observes a node and returns ops.
/// It receives the zipper (for focus and coeffect lookups)
/// and child results (from post-order traversal).
type Witness = Zipper -> WitnessResult list -> WitnessResult

// ═══════════════════════════════════════════════════════════════════════════
// ACCUMULATOR
// ═══════════════════════════════════════════════════════════════════════════

/// The accumulator for the witness fold.
/// This is what we're building - a collection of MLIR operations.
type Accumulator = {
    /// Top-level operations (functions, globals)
    TopLevel: MLIROp list

    /// String literals encountered (hash -> content, length)
    Strings: Map<uint32, string * int>
}

module Accumulator =
    let empty = { TopLevel = []; Strings = Map.empty }

    let addOps (ops: MLIROp list) (acc: Accumulator) =
        { acc with TopLevel = acc.TopLevel @ ops }

    let addString (hash: uint32) (content: string) (len: int) (acc: Accumulator) =
        { acc with Strings = Map.add hash (content, len) acc.Strings }

// ═══════════════════════════════════════════════════════════════════════════
// POST-ORDER FOLD
// ═══════════════════════════════════════════════════════════════════════════

/// Fold over the PSG in post-order, accumulating witness results.
/// Post-order ensures children are witnessed before parents.
let rec private foldNode
    (witness: Witness)
    (z: Zipper)
    : WitnessResult =

    // First, fold over all children (post-order)
    let childResults =
        z.Focus.Children
        |> List.mapi (fun i _ -> i)
        |> List.choose (fun i ->
            match downTo i z with
            | Some childZipper -> Some (foldNode witness childZipper)
            | None -> None)

    // Then witness this node with child results
    witness z childResults

/// Main entry point: fold over the entire graph from entry point
let fold
    (witness: Witness)
    (graph: SemanticGraph)
    (coeffects: Coeffects.Coeffects)
    : Accumulator =

    match fromEntryPoint graph coeffects with
    | None ->
        Accumulator.empty
    | Some zipper ->
        let result = foldNode witness zipper
        Accumulator.addOps result.Ops Accumulator.empty

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-ENTRY FOLD (for all Lambdas)
// ═══════════════════════════════════════════════════════════════════════════

/// Fold over all Lambda nodes in the graph
let foldAllLambdas
    (witness: Witness)
    (graph: SemanticGraph)
    (coeffects: Coeffects.Coeffects)
    : Accumulator =

    // Find all Lambda nodes
    let lambdaNodes =
        graph.Nodes
        |> Map.toList
        |> List.filter (fun (_, node) ->
            match node.Kind with
            | SemanticKind.Lambda _ -> true
            | _ -> false)

    // Fold each Lambda, accumulating results
    lambdaNodes
    |> List.fold (fun acc (nodeId, _) ->
        match create graph nodeId coeffects with
        | Some zipper ->
            let result = foldNode witness zipper
            Accumulator.addOps result.Ops acc
        | None -> acc
    ) Accumulator.empty
