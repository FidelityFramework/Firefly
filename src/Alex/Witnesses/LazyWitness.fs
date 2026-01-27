/// LazyWitness - Witness Lazy<'T> operations
///
/// PRD-14: Deferred computation with memoization
/// Layout: {computed: i1, value: T, code_ptr: ptr, cap0, cap1, ...}
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns
open Alex.Elements.CoeffectLookup
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS: LazyExpr (creation)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness LazyExpr - Build lazy struct with thunk and captures
/// thunkLambdaOutput: Already-visited Lambda child (thunk function)
let witnessLazyExpr (ctx: WitnessContext) (z: PSGZipper) (node: SemanticNode) (thunkLambdaOutput: WitnessOutput) : WitnessOutput =
    // Use XParsec to extract LazyExpr data
    match tryMatch pLazyExpr ctx.Graph node z ctx.Coeffects.Platform with
    | Some (bodyId, captures) ->
        // Get capture nodes
        let captureNodes =
            captures
            |> List.choose (fun c -> c.SourceNodeId)
            |> List.choose (fun id -> SemanticGraph.tryGetNode id ctx.Graph)

        // Convert nodes to Vals by looking up ACTUAL SSAs from accumulator
        let captureVals =
            List.zip captures captureNodes
            |> List.map (fun (info, captureNode) ->
                match MLIRAccumulator.recallNode (NodeId.value captureNode.Id) ctx.Accumulator with
                | Some (ssa, ty) -> { SSA = ssa; Type = ty }
                | None ->
                    // Fallback: try coeffects (for cases where node wasn't visited yet)
                    let ssa = requireSSA captureNode.Id ctx.Coeffects
                    let ty = mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph info.Type
                    { SSA = ssa; Type = ty })

        // Extract thunk function name from elided Lambda
        // The thunk Lambda was elided as "lambda_<id>" - use that name
        let thunkName = sprintf "lambda_%d" (NodeId.value bodyId)

        // Extract element type from TLazy wrapper
        let elementType =
            match node.Type with
            | NativeType.TLazy elemTy ->
                mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph elemTy
            | _ ->
                failwithf "LazyExpr node %d does not have TLazy type" (NodeId.value node.Id)
        let ssas = requireSSAs node.Id ctx.Coeffects
        let ops = buildLazyStruct thunkName elementType captureVals ssas

        let captureTypes = captureVals |> List.map (fun v -> v.Type)
        let lazyType = MLIRType.TStruct ([MLIRType.TInt IntBitWidth.I1; elementType; MLIRType.TPtr] @ captureTypes)
        let resultSSA = List.last ssas

        { InlineOps = ops
          TopLevelOps = thunkLambdaOutput.TopLevelOps  // Include thunk function
          Result = TRValue { SSA = resultSSA; Type = lazyType } }

    | None ->
        WitnessOutput.error (sprintf "LazyExpr pattern match failed (node %d)" (NodeId.value node.Id))

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS: LazyForce (evaluation)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness LazyForce - Evaluate lazy thunk
/// lazyOutput is the already-visited lazy value
let witnessLazyForce (ctx: WitnessContext) (node: SemanticNode) (lazyOutput: WitnessOutput) : WitnessOutput =
    match lazyOutput.Result with
    | TRValue lazyVal ->
        // Extract element type from lazy struct
        let elementType =
            match lazyVal.Type with
            | MLIRType.TStruct fields when fields.Length >= 2 -> fields.[1]
            | _ -> failwithf "LazyForce: expected lazy struct type (node %d)" (NodeId.value node.Id)

        // Force evaluation
        let ssas = requireSSAs node.Id ctx.Coeffects
        let ops = forceLazy lazyVal elementType ssas
        let resultSSA = List.last ssas

        { InlineOps = lazyOutput.InlineOps @ ops
          TopLevelOps = lazyOutput.TopLevelOps
          Result = TRValue { SSA = resultSSA; Type = elementType } }

    | TRError msg ->
        { lazyOutput with Result = TRError msg }

    | _ ->
        WitnessOutput.error (sprintf "LazyForce: lazy value has no result (node %d)" (NodeId.value node.Id))
