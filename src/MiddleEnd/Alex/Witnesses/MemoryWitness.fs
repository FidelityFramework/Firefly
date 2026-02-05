/// MemoryWitness - Witness memory and data structure operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
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
                let arch = ctx.Coeffects.Platform.TargetArch
                let fieldTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type

                match tryMatch (pStructFieldGet node.Id structSSA fieldName structTy fieldTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error $"FieldGet pattern failed for field '{fieldName}'"
            | None -> WitnessOutput.error $"FieldGet: Struct value not yet witnessed (field '{fieldName}')"

        | None ->
            // Not FieldGet - try DU operations: GetTag, Eliminate, Construct
            match tryMatch pDUGetTag ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((duValueId, _duType), _) ->
                match MLIRAccumulator.recallNode duValueId ctx.Accumulator with
                | Some (duSSA, duType) ->
                    match tryMatch (pExtractDUTag node.Id duSSA duType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "DUGetTag pattern emission failed"
                | None -> WitnessOutput.error "DUGetTag: DU value not available"

            | None ->
            match tryMatch pDUEliminate ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((duValueId, caseIndex, _caseName, _payloadType), _) ->
                match MLIRAccumulator.recallNode duValueId ctx.Accumulator with
                | Some (duSSA, duType) ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let payloadType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type

                    match tryMatch (pExtractDUPayload node.Id duSSA duType caseIndex payloadType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "DUEliminate pattern emission failed"
                | None -> WitnessOutput.error "DUEliminate: DU value not available"

            | None ->
                match tryMatch pDUConstruct ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((_caseName, caseIndex, payloadOpt, _arenaHintOpt), _) ->
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

                    match tryMatch (pDUCase node.Id tag payload duTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
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
