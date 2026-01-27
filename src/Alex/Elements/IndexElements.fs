/// IndexElements - Atomic Index dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides ALL Index dialect operations from Types.fs (exhaustive, no convenience wrappers).
module internal Alex.Elements.IndexElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════

let pIndexConst (ssa: SSA) (value: int64) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexConst (ssa, value)) }

let pIndexBoolConst (ssa: SSA) (value: bool) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexBoolConst (ssa, value)) }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC
// ═══════════════════════════════════════════════════════════

let pIndexAdd (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexAdd (ssa, lhs, rhs)) }

let pIndexSub (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexSub (ssa, lhs, rhs)) }

let pIndexMul (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexMul (ssa, lhs, rhs)) }

let pIndexDivS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexDivS (ssa, lhs, rhs)) }

let pIndexDivU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexDivU (ssa, lhs, rhs)) }

let pIndexCeilDivS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexCeilDivS (ssa, lhs, rhs)) }

let pIndexCeilDivU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexCeilDivU (ssa, lhs, rhs)) }

let pIndexFloorDivS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexFloorDivS (ssa, lhs, rhs)) }

let pIndexRemS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexRemS (ssa, lhs, rhs)) }

let pIndexRemU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexRemU (ssa, lhs, rhs)) }

// ═══════════════════════════════════════════════════════════
// MIN/MAX
// ═══════════════════════════════════════════════════════════

let pIndexMaxS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexMaxS (ssa, lhs, rhs)) }

let pIndexMaxU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexMaxU (ssa, lhs, rhs)) }

let pIndexMinS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexMinS (ssa, lhs, rhs)) }

let pIndexMinU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexMinU (ssa, lhs, rhs)) }

// ═══════════════════════════════════════════════════════════
// BITWISE
// ═══════════════════════════════════════════════════════════

let pIndexShl (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexShl (ssa, lhs, rhs)) }

let pIndexShrS (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexShrS (ssa, lhs, rhs)) }

let pIndexShrU (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexShrU (ssa, lhs, rhs)) }

let pIndexAnd (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexAnd (ssa, lhs, rhs)) }

let pIndexOr (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexOr (ssa, lhs, rhs)) }

let pIndexXor (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexXor (ssa, lhs, rhs)) }

// ═══════════════════════════════════════════════════════════
// COMPARISON
// ═══════════════════════════════════════════════════════════

let pIndexCmp (ssa: SSA) (pred: IndexCmpPred) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexCmp (ssa, pred, lhs, rhs)) }

// ═══════════════════════════════════════════════════════════
// CASTS
// ═══════════════════════════════════════════════════════════

let pIndexCastS (ssa: SSA) (operand: SSA) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexCastS (ssa, operand, toTy)) }

let pIndexCastU (ssa: SSA) (operand: SSA) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexCastU (ssa, operand, toTy)) }

// ═══════════════════════════════════════════════════════════
// SIZE OF
// ═══════════════════════════════════════════════════════════

let pIndexSizeOf (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser { return MLIROp.IndexOp (IndexOp.IndexSizeOf (ssa, ty)) }
