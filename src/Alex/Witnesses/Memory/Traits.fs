/// Memory Traits Witness - SRTP trait call resolution
///
/// SCOPE: witnessTraitCall (SRTP member access)
/// DOES NOT: Indexing, fields, aggregates, DUs (separate witnesses)
module Alex.Witnesses.Memory.Traits

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Witnesses.Memory.Indexing

let witnessTraitCall
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (receiverSSA: SSA)
    (receiverType: MLIRType)
    (memberName: string)
    (memberType: MLIRType)
    : MLIROp list * TransferResult =

    // For property-like traits (e.g., Length), generate field access
    // For method-like traits, this would need to generate a call
    // The specific implementation depends on the resolved member

    let resultSSA = requireNodeSSA nodeId ctx

    match memberName with
    | "Length" when isNativeStrType receiverType ->
        // String.Length - extract length field (index 1) from fat pointer
        // Length is platform word sized (i64 on 64-bit, i32 on 32-bit)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TInt (wordWidth ctx) }

    | "Length" ->
        // Array/collection Length - extract count field (index 1) from fat pointer
        // Length is platform word sized (i64 on 64-bit, i32 on 32-bit)
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [1], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TInt (wordWidth ctx) }

    | "Pointer" when isNativeStrType receiverType ->
        // String.Pointer - extract pointer field (index 0) from fat pointer
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = TPtr }

    | _ ->
        // Generic field access - assume field index 0 for unknown traits
        // This is a fallback and should be refined based on type information
        let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, receiverSSA, [0], receiverType))
        [extractOp], TRValue { SSA = resultSSA; Type = memberType }

// ═══════════════════════════════════════════════════════════════════════════
// DISCRIMINATED UNION OPERATIONS (January 2026)
//
// DU Architecture: Pointer-based with case eliminators
// - DUs are represented as pointers to region-allocated storage
// - Each case has its own typed eliminator for payload extraction
// - Pointer bitcast allows case-specific struct types (transliteration)
//
// For now (inline struct phase), DUs use (tag, payload) inline structs.
// The case-specific typing ensures extractvalue uses correct payload type.
// ═══════════════════════════════════════════════════════════════════════════

/// Witness DU tag extraction
/// Extracts the tag (discriminator) from a DU value
/// Tag is at field index 0, stored as i8 (or i16 for >256 cases)
/// Uses 1 pre-assigned SSA: extract[0]
