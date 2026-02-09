/// MemRefElements - Atomic MemRef dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides memory operations (alloca, load, store) via XParsec state threading.
///
/// Elements derive memref types monadically from the accumulator's SSA type index.
/// When a memref SSA was created (by pAlloca, pAlloc, etc.) or bound (by witness traversal),
/// its type was registered in the accumulator. pLoad uses this to derive the correct memref type
/// without callers having to push it as a parameter.
module internal Alex.Elements.MemRefElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.TransferTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit memref.load operation (derives types monadically from PSG node + accumulator)
/// elemType: derived from Current PSG node type (the load result type)
/// memrefType: derived from accumulator SSA type index (the source memref's type)
let pLoad (ssa: SSA) (memref: SSA) (indices: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let elemType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        match MLIRAccumulator.recallSSAType memref state.Accumulator with
        | Some memrefType ->
            return MLIROp.MemRefOp (MemRefOp.Load (ssa, memref, indices, elemType, memrefType))
        | None ->
            return! fail (Message $"pLoad: memref SSA {memref} has no registered type in accumulator (elemType={elemType})")
    }

/// Emit memref.load with explicit element type (memref type derived from accumulator)
/// For cases where the element type differs from the Current PSG node type
/// (e.g., loading TIndex from an index buffer, loading TInt I8 from a string)
let pLoadFrom (ssa: SSA) (memref: SSA) (indices: SSA list) (elemType: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        match MLIRAccumulator.recallSSAType memref state.Accumulator with
        | Some memrefType ->
            // Validate: memrefType must be a memref type (TMemRef or TMemRefStatic), not a scalar
            match memrefType with
            | TMemRef _ | TMemRefStatic _ ->
                return MLIROp.MemRefOp (MemRefOp.Load (ssa, memref, indices, elemType, memrefType))
            | _ ->
                let ssaStr = match memref with | V n -> sprintf "%%v%d" n | Arg n -> sprintf "%%arg%d" n
                return! fail (Message $"pLoadFrom: SSA {ssaStr} has type {memrefType} — expected memref type. This indicates an SSATypes scope leak (cross-function SSA collision).")
        | None ->
            return! fail (Message $"pLoadFrom: memref SSA {memref} has no registered type in accumulator (elemType={elemType})")
    }

/// Emit memref.store operation
/// elemType: element type of the memref (must match value type)
/// memrefType: full memref type for serialization
let pStore (value: SSA) (memref: SSA) (indices: SSA list) (elemType: MLIRType) (memrefType: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.MemRefOp (MemRefOp.Store (value, memref, indices, elemType, memrefType))
    }

/// Emit memref.alloca operation (stack allocation with compile-time size)
/// Registers the created SSA's memref type in the accumulator for downstream pLoad derivation
let pAlloca (ssa: SSA) (count: int) (elemType: MLIRType) (alignment: int option) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let memrefType = TMemRefStatic (count, elemType)
        MLIRAccumulator.registerSSAType ssa memrefType state.Accumulator
        return MLIROp.MemRefOp (MemRefOp.Alloca (ssa, memrefType, alignment))
    }

/// Emit memref.alloc operation (heap allocation with runtime size)
/// Registers the created SSA's memref type in the accumulator for downstream pLoad derivation
let pAlloc (ssa: SSA) (sizeSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let memrefType = TMemRef elemType
        MLIRAccumulator.registerSSAType ssa memrefType state.Accumulator
        return MLIROp.MemRefOp (MemRefOp.Alloc (ssa, sizeSSA, elemType))
    }

/// Emit memref.subview operation (replaces GEP for arrays)
let pSubView (ssa: SSA) (source: SSA) (offsets: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let memrefType = TMemRef ty
        return MLIROp.MemRefOp (MemRefOp.SubView (ssa, source, offsets, memrefType))
    }

/// Extract base pointer from memref for FFI boundaries
/// Uses builtin.unrealized_conversion_cast - standard MLIR operation for boundary crossings
/// This is for cases where portable memref needs to be passed to external C functions (syscalls, FFI)
let pExtractBasePtr (result: SSA) (memref: SSA) (memrefTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.MemRefOp (MemRefOp.ExtractBasePtr (result, memref, memrefTy))
    }

/// Get reference to global memref
/// Registers the created SSA's memref type in the accumulator for downstream pLoad derivation
let pMemRefGetGlobal (result: SSA) (globalName: string) (memrefType: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        MLIRAccumulator.registerSSAType result memrefType state.Accumulator
        return MLIROp.MemRefOp (MemRefOp.GetGlobal (result, globalName, memrefType))
    }

/// Get memref dimension size (replaces struct length extraction)
/// Emits: %result = memref.dim %memref, %dimIndex : memref<...>
/// Used to extract string length from memref descriptor for FFI/syscalls
let pMemRefDim (result: SSA) (memref: SSA) (dimIndex: SSA) (memrefType: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.MemRefOp (MemRefOp.Dim (result, memref, dimIndex, memrefType))
    }

/// Cast memref type (e.g., static → dynamic dimensions)
/// Emits: %result = memref.cast %source : srcType to destType
let pMemRefCast (result: SSA) (source: SSA) (srcType: MLIRType) (destType: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.MemRefOp (MemRefOp.Cast (result, source, srcType, destType))
    }
