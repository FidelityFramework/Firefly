/// MemoryWitness - Witness memory and data structure operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to MemoryPatterns for MLIR elision.
///
/// NANOPASS: This witness handles memory-related operations.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MemoryWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness memory operations - category-selective
let private witnessMemory (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pDUGetTag ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((duValueId, _duType), _) ->
        match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
        | Some (duSSA, duType), Some tagSSA ->
            match tryMatch (pExtractDUTag duSSA duType tagSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = tagSSA; Type = TInt I8 } }
            | None -> WitnessOutput.error "DUGetTag pattern emission failed"
        | _ -> WitnessOutput.error "DUGetTag: DU value or tag SSA not available"

    | None ->
        match tryMatch pDUEliminate ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((duValueId, caseIndex, _caseName, _payloadType), _) ->
            match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
            | Some (duSSA, duType), Some ssas ->
                match tryMatch (pExtractDUPayload duSSA duType caseIndex duType ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = duType } }
                | None -> WitnessOutput.error "DUEliminate pattern emission failed"
            | _ -> WitnessOutput.error "DUEliminate: DU value or SSAs not available"

        | None ->
            match tryMatch pDUConstruct ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((_caseName, caseIndex, payloadOpt, _arenaHintOpt), _) ->
                match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                | None -> WitnessOutput.error "DUConstruct: No SSAs assigned"
                | Some ssas ->
                    let tag = int64 caseIndex
                    let payload =
                        match payloadOpt with
                        | Some payloadId ->
                            match MLIRAccumulator.recallNode payloadId ctx.Accumulator with
                            | Some (ssa, ty) -> [{ SSA = ssa; Type = ty }]
                            | None -> []
                        | None -> []

                    match tryMatch (pDUCase tag payload ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = TPtr } }
                    | None -> WitnessOutput.error "DUConstruct pattern emission failed"

            | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Memory nanopass - witnesses memory-related operations
let nanopass : Nanopass = {
    Name = "Memory"
    Witness = witnessMemory
}
