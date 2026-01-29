/// MLIRElements - Atomic MLIR operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides atomic MLIR op construction via XParsec-compatible interface.
module internal Alex.Elements.MLIRElements

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// Elements accept type from caller - patterns know the type and pass it explicitly

/// ExtractValue - caller provides type
let pExtractValue (ssa: SSA) (value: SSA) (indices: int list) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.ExtractValue (ssa, value, indices, ty))
    }

/// InsertValue - caller provides type
let pInsertValue (ssa: SSA) (struct_: SSA) (value: SSA) (indices: int list) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.InsertValue (ssa, struct_, value, indices, ty))
    }

/// Undef - caller provides type
let pUndef (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.Undef (ssa, ty))
    }

/// ConstI - caller provides type
let pConstI (ssa: SSA) (value: int64) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.ConstI (ssa, value, ty))
    }

/// ConstF - caller provides type
let pConstF (ssa: SSA) (value: float) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.ConstF (ssa, value, ty))
    }
