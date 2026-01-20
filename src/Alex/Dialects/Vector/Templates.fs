/// Vector Dialect Templates - Structured operation constructors for SIMD
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → VectorOp
/// NO sprintf. NO string formatting. Just data construction.
///
/// Source: /usr/include/mlir/Dialect/Vector/IR/VectorOps.td
module Alex.Dialects.Vector.Templates

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// BROADCAST
// ═══════════════════════════════════════════════════════════════════════════

/// Broadcast a scalar to a vector: vector.broadcast
let broadcast (result: SSA) (source: SSA) (resultTy: MLIRType) : VectorOp =
    VectorOp.Broadcast (result, source, resultTy)

// ═══════════════════════════════════════════════════════════════════════════
// EXTRACT/INSERT
// ═══════════════════════════════════════════════════════════════════════════

/// Extract an element from a vector: vector.extract
let extract (result: SSA) (vector: SSA) (position: int list) : VectorOp =
    VectorOp.Extract (result, vector, position)

/// Extract a scalar element at a single position
let extractScalar (result: SSA) (vector: SSA) (position: int) : VectorOp =
    VectorOp.Extract (result, vector, [position])

/// Insert an element into a vector: vector.insert
let insert (result: SSA) (source: SSA) (dest: SSA) (position: int list) : VectorOp =
    VectorOp.Insert (result, source, dest, position)

/// Insert a scalar element at a single position
let insertScalar (result: SSA) (source: SSA) (dest: SSA) (position: int) : VectorOp =
    VectorOp.Insert (result, source, dest, [position])

/// Extract a strided slice: vector.extract_strided_slice
let extractStrided (result: SSA) (vector: SSA) (offsets: int list) (sizes: int list) (strides: int list) : VectorOp =
    VectorOp.ExtractStrided (result, vector, offsets, sizes, strides)

/// Insert a strided slice: vector.insert_strided_slice
let insertStrided (result: SSA) (source: SSA) (dest: SSA) (offsets: int list) (strides: int list) : VectorOp =
    VectorOp.InsertStrided (result, source, dest, offsets, strides)

// ═══════════════════════════════════════════════════════════════════════════
// SHAPE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Cast a vector to a different shape: vector.shape_cast
let shapeCast (result: SSA) (source: SSA) (resultTy: MLIRType) : VectorOp =
    VectorOp.ShapeCast (result, source, resultTy)

/// Transpose a vector: vector.transpose
let transpose (result: SSA) (vector: SSA) (transp: int list) : VectorOp =
    VectorOp.Transpose (result, vector, transp)

/// Flatten and transpose a 2D vector to 1D: vector.flat_transpose
let flatTranspose (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.FlattenTranspose (result, vector)

// ═══════════════════════════════════════════════════════════════════════════
// REDUCTION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Sum reduction: vector.reduction <add>
let reductionAdd (result: SSA) (vector: SSA) (acc: SSA option) : VectorOp =
    VectorOp.ReductionAdd (result, vector, acc)

/// Sum reduction without accumulator
let reduceAdd (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionAdd (result, vector, None)

/// Product reduction: vector.reduction <mul>
let reductionMul (result: SSA) (vector: SSA) (acc: SSA option) : VectorOp =
    VectorOp.ReductionMul (result, vector, acc)

/// Product reduction without accumulator
let reduceMul (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMul (result, vector, None)

/// AND reduction: vector.reduction <and>
let reductionAnd (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionAnd (result, vector)

/// OR reduction: vector.reduction <or>
let reductionOr (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionOr (result, vector)

/// XOR reduction: vector.reduction <xor>
let reductionXor (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionXor (result, vector)

/// Signed minimum reduction: vector.reduction <minsi>
let reductionMinSI (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMinSI (result, vector)

/// Unsigned minimum reduction: vector.reduction <minui>
let reductionMinUI (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMinUI (result, vector)

/// Signed maximum reduction: vector.reduction <maxsi>
let reductionMaxSI (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMaxSI (result, vector)

/// Unsigned maximum reduction: vector.reduction <maxui>
let reductionMaxUI (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMaxUI (result, vector)

/// Float minimum reduction: vector.reduction <minimumf>
let reductionMinF (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMinF (result, vector)

/// Float maximum reduction: vector.reduction <maximumf>
let reductionMaxF (result: SSA) (vector: SSA) : VectorOp =
    VectorOp.ReductionMaxF (result, vector)

// ═══════════════════════════════════════════════════════════════════════════
// FMA (Fused Multiply-Add)
// ═══════════════════════════════════════════════════════════════════════════

/// Fused multiply-add: vector.fma
/// Computes: lhs * rhs + acc
let fma (result: SSA) (lhs: SSA) (rhs: SSA) (acc: SSA) : VectorOp =
    VectorOp.FMA (result, lhs, rhs, acc)

// ═══════════════════════════════════════════════════════════════════════════
// SPLAT
// ═══════════════════════════════════════════════════════════════════════════

/// Splat a scalar to a vector: vector.splat
/// Creates a vector where all elements are the same scalar value
let splat (result: SSA) (value: SSA) (resultTy: MLIRType) : VectorOp =
    VectorOp.Splat (result, value, resultTy)

// ═══════════════════════════════════════════════════════════════════════════
// LOAD/STORE
// ═══════════════════════════════════════════════════════════════════════════

/// Load a vector from memory: vector.load
let load (result: SSA) (base': SSA) (indices: SSA list) : VectorOp =
    VectorOp.VectorLoad (result, base', indices)

/// Load a vector from a single base pointer (no indices)
let loadSimple (result: SSA) (base': SSA) : VectorOp =
    VectorOp.VectorLoad (result, base', [])

/// Store a vector to memory: vector.store
let store (valueToStore: SSA) (base': SSA) (indices: SSA list) : VectorOp =
    VectorOp.VectorStore (valueToStore, base', indices)

/// Store a vector to a single base pointer (no indices)
let storeSimple (valueToStore: SSA) (base': SSA) : VectorOp =
    VectorOp.VectorStore (valueToStore, base', [])

/// Masked load: vector.maskedload
let maskedLoad (result: SSA) (base': SSA) (indices: SSA list) (mask: SSA) (passthru: SSA) : VectorOp =
    VectorOp.MaskedLoad (result, base', indices, mask, passthru)

/// Masked store: vector.maskedstore
let maskedStore (valueToStore: SSA) (base': SSA) (indices: SSA list) (mask: SSA) : VectorOp =
    VectorOp.MaskedStore (valueToStore, base', indices, mask)

/// Gather elements: vector.gather
let gather (result: SSA) (base': SSA) (indices: SSA) (indexVec: SSA) (mask: SSA) (passthru: SSA) : VectorOp =
    VectorOp.Gather (result, base', indices, indexVec, mask, passthru)

/// Scatter elements: vector.scatter
let scatter (valueToStore: SSA) (base': SSA) (indices: SSA) (indexVec: SSA) (mask: SSA) : VectorOp =
    VectorOp.Scatter (valueToStore, base', indices, indexVec, mask)

// ═══════════════════════════════════════════════════════════════════════════
// MASK OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a mask from runtime values: vector.create_mask
let createMask (result: SSA) (operands: SSA list) : VectorOp =
    VectorOp.CreateMask (result, operands)

/// Create a constant mask: vector.constant_mask
let constantMask (result: SSA) (maskDimSizes: int list) : VectorOp =
    VectorOp.ConstantMask (result, maskDimSizes)

// ═══════════════════════════════════════════════════════════════════════════
// PRINT (debugging)
// ═══════════════════════════════════════════════════════════════════════════

/// Print a vector (for debugging): vector.print
let print (source: SSA) (punctuation: string option) : VectorOp =
    VectorOp.Print (source, punctuation)

/// Print a vector with default punctuation
let printSimple (source: SSA) : VectorOp =
    VectorOp.Print (source, None)

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap VectorOp in MLIROp
let wrap (op: VectorOp) : MLIROp = MLIROp.VectorOp op
