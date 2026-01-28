/// MLIRTransfer - Thin fold from SemanticGraph to MLIR
///
/// This file ONLY orchestrates: fold PSG structure, dispatch to witnesses.
/// Witnesses use XParsec + Patterns + Elements to elide PSG → MLIR.

module Alex.Traversal.MLIRTransfer

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS MODULES
// ═══════════════════════════════════════════════════════════════════════════

module LazyWitness = Alex.Witnesses.LazyWitness

// ═══════════════════════════════════════════════════════════════════════════
// THE FOLD (Core Transfer Logic)
// ═══════════════════════════════════════════════════════════════════════════

/// Fold over PSG node - witnesses use XParsec to pattern match and elide
let rec visitNode (ctx: WitnessContext) (nodeId: NodeId) (acc: MLIRAccumulator) : WitnessOutput =

    // Check if already processed
    match MLIRAccumulator.isVisited (NodeId.value nodeId) acc with
    | true ->
        match MLIRAccumulator.recallNode (NodeId.value nodeId) acc with
        | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
        | None -> WitnessOutput.error (sprintf "Node %d visited but no binding" (NodeId.value nodeId))
    | false ->
        MLIRAccumulator.markVisited (NodeId.value nodeId) acc

        // Focus zipper on this node for witnesses to navigate from
        match PSGZipper.focusOn nodeId ctx.Zipper with
        | None -> WitnessOutput.error (sprintf "Node %d not found in graph" (NodeId.value nodeId))
        | Some zipper ->
            let ctx = { ctx with Zipper = zipper }  // Shadow with focused zipper
            let node = PSGZipper.focus zipper

            match node.Kind with

            // ═════════════════════════════════════════════════════════════════
            // LAZY (PRD-14) - XParsec Architecture Pilot
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.LazyExpr (bodyId, captures) ->
                LazyWitness.witnessLazyExpr ctx node

            | SemanticKind.LazyForce lazyValueId ->
                LazyWitness.witnessLazyForce ctx node

            // ═════════════════════════════════════════════════════════════════
            // LITERALS - needs LiteralWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Literal _ ->
                WitnessOutput.error "Literal - needs LiteralWitness (XParsec + Patterns)"

            // ═════════════════════════════════════════════════════════════════
            // BINDINGS AND LAMBDAS - needs LambdaWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Binding _ ->
                WitnessOutput.error "Binding - structural traversal needed"

            | SemanticKind.Lambda _ ->
                WitnessOutput.error "Lambda - needs LambdaWitness (XParsec + Patterns)"

            // ═════════════════════════════════════════════════════════════════
            // APPLICATION - needs ApplicationWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Application _ ->
                WitnessOutput.error "Application - needs ApplicationWitness (XParsec + Patterns)"

            // ═════════════════════════════════════════════════════════════════
            // CONTROL FLOW - needs ControlFlowWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.IfThenElse _ ->
                WitnessOutput.error "IfThenElse - needs ControlFlowWitness"

            | SemanticKind.WhileLoop _ ->
                WitnessOutput.error "WhileLoop - needs ControlFlowWitness"

            | SemanticKind.ForLoop _ ->
                WitnessOutput.error "ForLoop - needs ControlFlowWitness"

            | SemanticKind.ForEach _ ->
                WitnessOutput.error "ForEach - needs ControlFlowWitness"

            | SemanticKind.Match _ ->
                WitnessOutput.error "Match - needs ControlFlowWitness"

            | SemanticKind.TryWith _ ->
                WitnessOutput.error "TryWith - needs ControlFlowWitness"

            | SemanticKind.TryFinally _ ->
                WitnessOutput.error "TryFinally - needs ControlFlowWitness"

            // ═════════════════════════════════════════════════════════════════
            // SEQUENCES (PRD-15) - needs SeqWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.SeqExpr _ ->
                WitnessOutput.error "SeqExpr - needs SeqWitness (XParsec + Patterns)"

            | SemanticKind.Yield _ ->
                WitnessOutput.error "Yield - needs SeqWitness"

            | SemanticKind.YieldBang _ ->
                WitnessOutput.error "YieldBang - needs SeqWitness"

            // ═════════════════════════════════════════════════════════════════
            // REFERENCES - needs ReferenceWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.VarRef _ ->
                WitnessOutput.error "VarRef - needs ReferenceWitness"

            | SemanticKind.Set _ ->
                WitnessOutput.error "Set - needs ReferenceWitness"

            // ═════════════════════════════════════════════════════════════════
            // TUPLES - needs TupleWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.TupleExpr _ ->
                WitnessOutput.error "TupleExpr - needs TupleWitness"

            | SemanticKind.TupleGet _ ->
                WitnessOutput.error "TupleGet - needs TupleWitness"

            // ═════════════════════════════════════════════════════════════════
            // RECORDS - needs RecordWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.RecordExpr _ ->
                WitnessOutput.error "RecordExpr - needs RecordWitness"

            | SemanticKind.FieldGet _ ->
                WitnessOutput.error "FieldGet - needs RecordWitness"

            | SemanticKind.FieldSet _ ->
                WitnessOutput.error "FieldSet - needs RecordWitness"

            // ═════════════════════════════════════════════════════════════════
            // UNIONS - needs UnionWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.UnionCase _ ->
                WitnessOutput.error "UnionCase - needs UnionWitness"

            | SemanticKind.DUGetTag _ ->
                WitnessOutput.error "DUGetTag - needs UnionWitness"

            | SemanticKind.DUEliminate _ ->
                WitnessOutput.error "DUEliminate - needs UnionWitness"

            | SemanticKind.DUConstruct _ ->
                WitnessOutput.error "DUConstruct - needs UnionWitness"

            // ═════════════════════════════════════════════════════════════════
            // ARRAYS AND LISTS - needs CollectionWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.ArrayExpr _ ->
                WitnessOutput.error "ArrayExpr - needs CollectionWitness"

            | SemanticKind.ListExpr _ ->
                WitnessOutput.error "ListExpr - needs CollectionWitness"

            | SemanticKind.IndexGet _ ->
                WitnessOutput.error "IndexGet - needs CollectionWitness"

            | SemanticKind.IndexSet _ ->
                WitnessOutput.error "IndexSet - needs CollectionWitness"

            | SemanticKind.NamedIndexedPropertySet _ ->
                WitnessOutput.error "NamedIndexedPropertySet - needs CollectionWitness"

            // ═════════════════════════════════════════════════════════════════
            // TYPE OPERATIONS
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.TypeAnnotation (exprId, _) ->
                // Type annotations are metadata - process expression
                visitNode ctx exprId acc

            | SemanticKind.Upcast _ ->
                WitnessOutput.error "Upcast - needs CastWitness"

            | SemanticKind.Downcast _ ->
                WitnessOutput.error "Downcast - needs CastWitness"

            | SemanticKind.TypeTest _ ->
                WitnessOutput.error "TypeTest - needs CastWitness"

            // ═════════════════════════════════════════════════════════════════
            // POINTERS - needs PointerWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.AddressOf _ ->
                WitnessOutput.error "AddressOf - needs PointerWitness"

            | SemanticKind.Deref _ ->
                WitnessOutput.error "Deref - needs PointerWitness"

            // ═════════════════════════════════════════════════════════════════
            // INTRINSICS - needs IntrinsicWitness
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Intrinsic info ->
                WitnessOutput.error (sprintf "Intrinsic %A - needs IntrinsicWitness" info)

            | SemanticKind.PlatformBinding name ->
                WitnessOutput.error (sprintf "PlatformBinding %s - needs PlatformWitness" name)

            | SemanticKind.TraitCall _ ->
                WitnessOutput.error "TraitCall - needs TraitWitness"

            // ═════════════════════════════════════════════════════════════════
            // SPECIAL CONSTRUCTS
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Sequential _ ->
                WitnessOutput.error "Sequential - needs sequential processing"

            | SemanticKind.Quote _ ->
                WitnessOutput.error "Quote - quotations not supported"

            | SemanticKind.InterpolatedString _ ->
                WitnessOutput.error "InterpolatedString - needs StringWitness"

            | SemanticKind.PatternBinding _ ->
                WitnessOutput.error "PatternBinding - needs PatternWitness"

            // ═════════════════════════════════════════════════════════════════
            // DEFINITIONS (metadata nodes - should not be visited)
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.ObjectExpr _ ->
                WitnessOutput.error "ObjectExpr - not supported"

            | SemanticKind.ModuleDef _ ->
                WitnessOutput.error "ModuleDef - metadata node"

            | SemanticKind.TypeDef _ ->
                WitnessOutput.error "TypeDef - metadata node"

            | SemanticKind.MemberDef _ ->
                WitnessOutput.error "MemberDef - metadata node"

            // ═════════════════════════════════════════════════════════════════
            // ERRORS
            // ═════════════════════════════════════════════════════════════════

            | SemanticKind.Error msg ->
                WitnessOutput.error (sprintf "PSG Error: %s" msg)

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer PSG to MLIR starting from entry point
let transfer
    (graph: SemanticGraph)
    (entryNodeId: NodeId)
    (coeffects: TransferCoeffects)
    : Result<MLIROp list * MLIROp list, string> =

    match SemanticGraph.tryGetNode entryNodeId graph with
    | None ->
        Error (sprintf "Entry node %d not found" (NodeId.value entryNodeId))
    | Some _ ->
        // Create zipper ONCE at entry point (Huet zipper pattern)
        match PSGZipper.create graph entryNodeId with
        | None ->
            Error (sprintf "Could not create zipper for entry node %d" (NodeId.value entryNodeId))
        | Some zipper ->
            let acc = MLIRAccumulator.empty ()
            let ctx = {
                Graph = graph
                Coeffects = coeffects
                Accumulator = acc
                Zipper = zipper            // Navigation state (created ONCE)
            }

            let output = visitNode ctx entryNodeId acc

            MLIRAccumulator.addTopLevelOps output.InlineOps acc
            MLIRAccumulator.addTopLevelOps output.TopLevelOps acc

            match output.Result with
            | TRError msg -> Error msg
            | _ -> Ok (List.rev acc.TopLevelOps, [])
