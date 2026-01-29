/// SeqWitness - Witness Seq<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Seq-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SeqWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq operations - category-selective (handles only Seq nodes)
let private witnessSeq (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pSeqExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((bodyId, captures), _) ->
        match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
        | None -> WitnessOutput.error "SeqExpr: No SSAs assigned"
        | Some ssas ->
            let arch = ctx.Coeffects.Platform.TargetArch
            let captureVals =
                captures
                |> List.choose (fun cap ->
                    cap.SourceNodeId
                    |> Option.bind (fun id -> SSAAssign.lookupSSA id ctx.Coeffects.SSA)
                    |> Option.map (fun ssa ->
                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch cap.Type
                        { SSA = ssa; Type = mlirType }))

            let rec extractMutableBindings nodeId =
                match SemanticGraph.tryGetNode nodeId ctx.Graph with
                | None -> []
                | Some n ->
                    let thisVal =
                        match n.Kind with
                        | SemanticKind.Binding (_, true, _, _) ->
                            SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA
                            |> Option.map (fun ssa ->
                                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch n.Type
                                [{ SSA = ssa; Type = mlirType }])
                            |> Option.defaultValue []
                        | _ -> []
                    thisVal @ (n.Children |> List.collect extractMutableBindings)

            let internalState = extractMutableBindings bodyId

            match MLIRAccumulator.recallNode bodyId ctx.Accumulator with
            | None -> WitnessOutput.error "SeqExpr: Body not yet witnessed"
            | Some (codePtr, codePtrTy) ->
                // Get Seq<T> type from node, extract current type T from struct field [1]
                let seqTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                let currentTy = match seqTy with
                                | TStruct (_::ct::_) -> ct  // {state: i32, current: T, ...}
                                | _ -> TPtr  // fallback
                match tryMatch (pBuildSeqStruct currentTy codePtrTy codePtr captureVals internalState ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "SeqExpr pattern emission failed"

    | None ->
        match tryMatch pForEach ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((_, collectionId, _), _) ->
            match MLIRAccumulator.recallNode collectionId ctx.Accumulator with
            | None -> WitnessOutput.error "ForEach: Collection not yet witnessed"
            | Some (collectionSSA, _) ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let bodyOps = []
                match tryMatch (pBuildForEachLoop collectionSSA bodyOps arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "ForEach pattern emission failed"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Seq nanopass - witnesses SeqExpr and ForEach nodes
let nanopass : Nanopass = {
    Name = "Seq"
    Witness = witnessSeq
}
