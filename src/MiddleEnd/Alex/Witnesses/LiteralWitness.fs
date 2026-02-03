/// Literal Witness - Witness literal values to MLIR via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Literal nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.LiteralPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Literal nodes - category-selective (handles only Literal nodes)
let private witnessLiteralNode (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pLiteral ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some (lit, _) ->
        let arch = ctx.Coeffects.Platform.TargetArch

        // String literals need special handling (5 SSAs vs 1 SSA for other literals)
        match lit with
        | NativeLiteral.String content ->
            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | None ->
                let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "SSA lookup") "String literal: No SSAs assigned"
                WitnessOutput.errorDiag diag
            | Some ssas when ssas.Length >= 5 ->
                // Use trace-enabled variant to capture full execution path
                match tryMatchWithTrace (pBuildStringLiteral content ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok (((inlineOps, globalName, strContent, byteLength), result), _, _trace) ->
                    // Success - emit GlobalString via coordination (dependent transparency)
                    let topLevelOps =
                        match MLIRAccumulator.tryEmitGlobal globalName strContent byteLength ctx.Accumulator with
                        | Some globalOp -> [globalOp]
                        | None -> []  // Already emitted by another witness
                    { InlineOps = inlineOps; TopLevelOps = topLevelOps; Result = result }
                | Result.Error (err, trace) ->
                    // Failure - serialize trace for debugging
                    // TODO: Serialize trace to intermediates/07_literal_witness_nodeXXX_trace.json
                    let traceMsg = trace |> List.map ExecutionTrace.format |> String.concat "\n"
                    let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "pBuildStringLiteral") 
                                    (sprintf "String literal pattern emission failed:\nXParsec Error: %A\nExecution Trace:\n%s" err traceMsg)
                    WitnessOutput.errorDiag diag
            | Some ssas ->
                let diag = Diagnostic.errorWithDetails (Some node.Id) (Some "Literal") (Some "SSA validation")
                                "String literal: Incorrect SSA count" "5 SSAs" $"{ssas.Length} SSAs"
                WitnessOutput.errorDiag diag

        | _ ->
            // Other literals (int, bool, float, char, etc.) use single SSA
            match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | None ->
                let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "SSA lookup") "Literal: No SSA assigned"
                WitnessOutput.errorDiag diag
            | Some ssa ->
                match tryMatch (pBuildLiteral lit ssa arch) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None ->
                    let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "pBuildLiteral") "Literal pattern emission failed"
                    WitnessOutput.errorDiag diag

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Literal nanopass - witnesses Literal nodes (int, bool, char, float, etc.)
let nanopass : Nanopass = {
    Name = "Literal"
    Witness = witnessLiteralNode
}
