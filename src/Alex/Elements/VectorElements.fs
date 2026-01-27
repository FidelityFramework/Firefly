/// VectorElements - Atomic Vector dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides ALL Vector dialect operations from Types.fs (exhaustive, no convenience wrappers).
module internal Alex.Elements.VectorElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// BROADCAST
// ═══════════════════════════════════════════════════════════

let pBroadcast (ssa: SSA) (source: SSA) (resultTy: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Broadcast (ssa, source, resultTy)) }

// ═══════════════════════════════════════════════════════════
// EXTRACT/INSERT
// ═══════════════════════════════════════════════════════════

let pExtract (ssa: SSA) (vector: SSA) (position: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Extract (ssa, vector, position)) }

let pInsert (ssa: SSA) (source: SSA) (dest: SSA) (position: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Insert (ssa, source, dest, position)) }

let pExtractStrided (ssa: SSA) (vector: SSA) (offsets: int list) (sizes: int list) (strides: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ExtractStrided (ssa, vector, offsets, sizes, strides)) }

let pInsertStrided (ssa: SSA) (source: SSA) (dest: SSA) (offsets: int list) (strides: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.InsertStrided (ssa, source, dest, offsets, strides)) }

// ═══════════════════════════════════════════════════════════
// SHAPE OPERATIONS
// ═══════════════════════════════════════════════════════════

let pShapeCast (ssa: SSA) (source: SSA) (resultTy: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ShapeCast (ssa, source, resultTy)) }

let pTranspose (ssa: SSA) (vector: SSA) (transp: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Transpose (ssa, vector, transp)) }

let pFlattenTranspose (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.FlattenTranspose (ssa, vector)) }

// ═══════════════════════════════════════════════════════════
// REDUCTION OPERATIONS
// ═══════════════════════════════════════════════════════════

let pReductionAdd (ssa: SSA) (vector: SSA) (acc: SSA option) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionAdd (ssa, vector, acc)) }

let pReductionMul (ssa: SSA) (vector: SSA) (acc: SSA option) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMul (ssa, vector, acc)) }

let pReductionAnd (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionAnd (ssa, vector)) }

let pReductionOr (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionOr (ssa, vector)) }

let pReductionXor (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionXor (ssa, vector)) }

let pReductionMinSI (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMinSI (ssa, vector)) }

let pReductionMinUI (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMinUI (ssa, vector)) }

let pReductionMaxSI (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMaxSI (ssa, vector)) }

let pReductionMaxUI (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMaxUI (ssa, vector)) }

let pReductionMinF (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMinF (ssa, vector)) }

let pReductionMaxF (ssa: SSA) (vector: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ReductionMaxF (ssa, vector)) }

// ═══════════════════════════════════════════════════════════
// FMA (Fused Multiply-Add)
// ═══════════════════════════════════════════════════════════

let pFMA (ssa: SSA) (lhs: SSA) (rhs: SSA) (acc: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.FMA (ssa, lhs, rhs, acc)) }

// ═══════════════════════════════════════════════════════════
// SPLAT
// ═══════════════════════════════════════════════════════════

let pSplat (ssa: SSA) (value: SSA) (resultTy: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Splat (ssa, value, resultTy)) }

// ═══════════════════════════════════════════════════════════
// LOAD/STORE
// ═══════════════════════════════════════════════════════════

let pVectorLoad (ssa: SSA) (basePtr: SSA) (indices: SSA list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.VectorLoad (ssa, basePtr, indices)) }

let pVectorStore (valueToStore: SSA) (basePtr: SSA) (indices: SSA list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.VectorStore (valueToStore, basePtr, indices)) }

let pMaskedLoad (ssa: SSA) (basePtr: SSA) (indices: SSA list) (mask: SSA) (passthru: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.MaskedLoad (ssa, basePtr, indices, mask, passthru)) }

let pMaskedStore (valueToStore: SSA) (basePtr: SSA) (indices: SSA list) (mask: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.MaskedStore (valueToStore, basePtr, indices, mask)) }

let pGather (ssa: SSA) (basePtr: SSA) (indices: SSA) (indexVec: SSA) (mask: SSA) (passthru: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Gather (ssa, basePtr, indices, indexVec, mask, passthru)) }

let pScatter (valueToStore: SSA) (basePtr: SSA) (indices: SSA) (indexVec: SSA) (mask: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Scatter (valueToStore, basePtr, indices, indexVec, mask)) }

// ═══════════════════════════════════════════════════════════
// MASK OPERATIONS
// ═══════════════════════════════════════════════════════════

let pCreateMask (ssa: SSA) (operands: SSA list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.CreateMask (ssa, operands)) }

let pConstantMask (ssa: SSA) (maskDimSizes: int list) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.ConstantMask (ssa, maskDimSizes)) }

// ═══════════════════════════════════════════════════════════
// PRINT (debugging)
// ═══════════════════════════════════════════════════════════

let pPrint (source: SSA) (punctuation: string option) : PSGParser<MLIROp> =
    parser { return MLIROp.VectorOp (VectorOp.Print (source, punctuation)) }
