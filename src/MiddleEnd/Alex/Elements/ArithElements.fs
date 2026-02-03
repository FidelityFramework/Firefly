/// ArithElements - Atomic arithmetic operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides integer and floating-point arithmetic operations via XParsec state threading.
module internal Alex.Elements.ArithElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// INTEGER ARITHMETIC
// ═══════════════════════════════════════════════════════════

/// Emit AddI (integer addition)
let pAddI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.AddI (ssa, lhs, rhs, ty))
    }

/// Emit SubI (integer subtraction)
let pSubI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.SubI (ssa, lhs, rhs, ty))
    }

/// Emit MulI (integer multiplication)
let pMulI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.MulI (ssa, lhs, rhs, ty))
    }

/// Emit DivSI (signed integer division)
let pDivSI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.DivSI (ssa, lhs, rhs, ty))
    }

/// Emit DivUI (unsigned integer division)
let pDivUI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.DivUI (ssa, lhs, rhs, ty))
    }

/// Emit RemSI (signed integer remainder/modulo)
let pRemSI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.RemSI (ssa, lhs, rhs, ty))
    }

/// Emit RemUI (unsigned integer remainder/modulo)
let pRemUI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.RemUI (ssa, lhs, rhs, ty))
    }

/// Emit CmpI (integer comparison)
/// Takes operand type explicitly since comparison result type (i1) differs from operand type (i64/i32/etc)
let pCmpI (ssa: SSA) (pred: ICmpPred) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.CmpI (ssa, pred, lhs, rhs, operandTy))
    }

// ═══════════════════════════════════════════════════════════
// FLOATING-POINT ARITHMETIC
// ═══════════════════════════════════════════════════════════

/// Emit AddF (floating-point addition)
let pAddF (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.AddF (ssa, lhs, rhs, ty))
    }

/// Emit SubF (floating-point subtraction)
let pSubF (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.SubF (ssa, lhs, rhs, ty))
    }

/// Emit MulF (floating-point multiplication)
let pMulF (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.MulF (ssa, lhs, rhs, ty))
    }

/// Emit DivF (floating-point division)
let pDivF (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.DivF (ssa, lhs, rhs, ty))
    }

/// Emit CmpF (floating-point comparison)
let pCmpF (ssa: SSA) (pred: FCmpPred) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.CmpF (ssa, pred, lhs, rhs, ty))
    }

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSIONS
// ═══════════════════════════════════════════════════════════

/// Emit ExtSI (sign-extend integer)
let pExtSI (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.ExtSI (ssa, value, fromTy, toTy))
    }

/// Emit ExtUI (zero-extend integer)
let pExtUI (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.ExtUI (ssa, value, fromTy, toTy))
    }

/// Emit TruncI (truncate integer)
let pTruncI (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.TruncI (ssa, value, fromTy, toTy))
    }

/// Emit SIToFP (signed integer to floating-point)
let pSIToFP (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.SIToFP (ssa, value, fromTy, toTy))
    }

/// Emit FPToSI (floating-point to signed integer)
let pFPToSI (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.FPToSI (ssa, value, fromTy, toTy))
    }

// ═══════════════════════════════════════════════════════════
// OTHER OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit Select (conditional select)
let pSelect (ssa: SSA) (cond: SSA) (trueVal: SSA) (falseVal: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.Select (ssa, cond, trueVal, falseVal, ty))
    }

// ═══════════════════════════════════════════════════════════
// BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════
// NOTE: These replace LLVM dialect bitwise ops (llvm.and, llvm.or, etc.)
// with standard MLIR Arith dialect operations for backend flexibility.

/// Emit arith.andi (bitwise AND)
let pAndI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.AndI (ssa, lhs, rhs, ty))
    }

/// Emit arith.ori (bitwise OR)
let pOrI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.OrI (ssa, lhs, rhs, ty))
    }

/// Emit arith.xori (bitwise XOR)
let pXorI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.XorI (ssa, lhs, rhs, ty))
    }

/// Emit arith.shli (shift left)
let pShLI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.ShLI (ssa, lhs, rhs, ty))
    }

/// Emit arith.shrui (logical shift right - unsigned)
let pShRUI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.ShRUI (ssa, lhs, rhs, ty))
    }

/// Emit arith.shrsi (arithmetic shift right - signed)
let pShRSI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.ShRSI (ssa, lhs, rhs, ty))
    }
