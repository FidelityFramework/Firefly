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

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness memory operations - handles FieldGet and DU operations
/// Intrinsic applications (NativePtr.*) are handled by ApplicationWitness
let private witnessMemory (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Skip intrinsic nodes - ApplicationWitness handles intrinsic applications
    match node.Kind with
    | SemanticKind.Intrinsic _ -> WitnessOutput.skip
    | _ ->
        // Try FieldGet first (struct field access like s.Pointer, s.Length)
        match tryMatch pFieldGet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((structId, fieldName), _) ->
            // Recall struct SSA
            match MLIRAccumulator.recallNode structId ctx.Accumulator with
            | Some (structSSA, structTy) ->
                // Witness all SSAs pre-assigned by coeffects
                match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                | Some ssas ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let fieldTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    match tryMatch (pStructFieldGet ssas structSSA fieldName structTy fieldTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error $"FieldGet pattern failed for field '{fieldName}'"
                | None -> WitnessOutput.error $"FieldGet: No SSAs assigned (field '{fieldName}')"
            | None -> WitnessOutput.error $"FieldGet: Struct value not yet witnessed (field '{fieldName}')"

        | None ->
            // Not FieldGet - try DU operations: GetTag, Eliminate, Construct
            match tryMatch pDUGetTag ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((duValueId, _duType), _) ->
                match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                | Some (duSSA, duType), Some ssas when ssas.Length >= 2 ->
                    let indexZeroSSA = ssas.[0]
                    let tagSSA = ssas.[1]
                    match tryMatch (pExtractDUTag duSSA duType tagSSA indexZeroSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = tagSSA; Type = TInt I8 } }
                    | None -> WitnessOutput.error "DUGetTag pattern emission failed"
                | _ -> WitnessOutput.error "DUGetTag: DU value or tag SSA not available"

            | None ->
            match tryMatch pDUEliminate ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((duValueId, caseIndex, _caseName, _payloadType), _) ->
                match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                | Some (duSSA, duType), Some ssas ->
                    match tryMatch (pExtractDUPayload duSSA duType caseIndex duType ssas) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = duType } }
                    | None -> WitnessOutput.error "DUEliminate pattern emission failed"
                | _ -> WitnessOutput.error "DUEliminate: DU value or SSAs not available"

            | None ->
                match tryMatch pDUConstruct ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
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

                        let arch = ctx.Coeffects.Platform.TargetArch
                        let duTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        match tryMatch (pDUCase tag payload ssas duTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = duTy } }
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
