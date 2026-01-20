/// Index Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → IndexOp
/// NO sprintf. NO string formatting. Just data construction.
module Alex.Dialects.Index.Templates

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Index constant: index.constant
let indexConst (result: SSA) (value: int64) : IndexOp =
    IndexOp.IndexConst (result, value)

/// Index constant 0
let indexZero (result: SSA) : IndexOp =
    IndexOp.IndexConst (result, 0L)

/// Index constant 1
let indexOne (result: SSA) : IndexOp =
    IndexOp.IndexConst (result, 1L)

// ═══════════════════════════════════════════════════════════════════════════
// ARITHMETIC OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Index addition: index.add
let indexAdd (result: SSA) (lhs: SSA) (rhs: SSA) : IndexOp =
    IndexOp.IndexAdd (result, lhs, rhs)

/// Index subtraction: index.sub
let indexSub (result: SSA) (lhs: SSA) (rhs: SSA) : IndexOp =
    IndexOp.IndexSub (result, lhs, rhs)

/// Index multiplication: index.mul
let indexMul (result: SSA) (lhs: SSA) (rhs: SSA) : IndexOp =
    IndexOp.IndexMul (result, lhs, rhs)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CASTS
// ═══════════════════════════════════════════════════════════════════════════

/// Cast index to integer (signed): index.casts
let indexCastS (result: SSA) (operand: SSA) (toTy: MLIRType) : IndexOp =
    IndexOp.IndexCastS (result, operand, toTy)

/// Cast index to integer (unsigned): index.castu
let indexCastU (result: SSA) (operand: SSA) (toTy: MLIRType) : IndexOp =
    IndexOp.IndexCastU (result, operand, toTy)

/// Cast index to i64 (common pattern)
let indexToI64 (result: SSA) (operand: SSA) : IndexOp =
    IndexOp.IndexCastS (result, operand, MLIRTypes.i64)

/// Cast index to i32 (common pattern)
let indexToI32 (result: SSA) (operand: SSA) : IndexOp =
    IndexOp.IndexCastS (result, operand, MLIRTypes.i32)

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap IndexOp in MLIROp
let wrap (op: IndexOp) : MLIROp = MLIROp.IndexOp op
