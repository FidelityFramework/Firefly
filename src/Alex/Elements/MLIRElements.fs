/// MLIRElements - Atomic MLIR operation emission
///
/// VISIBILITY: module internal - Witnesses CANNOT import this
///
/// This module provides atomic MLIR operation builders. It is intentionally
/// internal to enforce the architectural firewall: witnesses must call
/// Patterns (which compose these Elements), never Elements directly.
module internal Alex.Elements.MLIRElements

open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// LLVM OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit ExtractValue operation
let emitExtractValue (ssa: SSA) (value: SSA) (indices: int list) (ty: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.ExtractValue (ssa, value, indices, ty))

/// Emit InsertValue operation
let emitInsertValue (ssa: SSA) (struct_: SSA) (value: SSA) (indices: int list) (ty: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.InsertValue (ssa, struct_, value, indices, ty))

/// Emit Undef (uninitialized value)
let emitUndef (ssa: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Undef (ssa, ty))

/// Emit Load operation
let emitLoad (ssa: SSA) (ptr: SSA) (ty: MLIRType) (ordering: AtomicOrdering) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Load (ssa, ptr, ty, ordering))

/// Emit Store operation
let emitStore (value: SSA) (ptr: SSA) (ty: MLIRType) (ordering: AtomicOrdering) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Store (value, ptr, ty, ordering))

/// Emit GEP (GetElementPtr)
let emitGEP (ssa: SSA) (ptr: SSA) (indices: (SSA * MLIRType) list) (ty: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.GEP (ssa, ptr, indices, ty))

/// Emit StructGEP (typed struct field access)
let emitStructGEP (ssa: SSA) (ptr: SSA) (index: int) (ty: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.StructGEP (ssa, ptr, index, ty))

/// Emit IndirectCall
let emitIndirectCall (resultSSA: SSA option) (funcPtr: SSA) (args: Val list) (retTy: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.IndirectCall (resultSSA, funcPtr, args, retTy))

/// Emit Alloca (stack allocation)
let emitAlloca (ssa: SSA) (count: SSA) (ty: MLIRType) (alignment: int option) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Alloca (ssa, count, ty, alignment))

/// Emit AddressOf (function pointer)
let emitAddressOf (ssa: SSA) (target: GlobalRef) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.AddressOf (ssa, target))

/// Emit Bitcast
let emitBitcast (ssa: SSA) (value: SSA) (fromTy: MLIRType) (toTy: MLIRType) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Bitcast (ssa, value, fromTy, toTy))

// ═══════════════════════════════════════════════════════════════════════════
// ARITH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit integer constant
let emitConstI (ssa: SSA) (value: int64) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.ConstI (ssa, value, ty))

/// Emit float constant
let emitConstF (ssa: SSA) (value: float) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.ConstF (ssa, value, ty))

/// Emit AddI (integer addition)
let emitAddI (ssa: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.AddI (ssa, lhs, rhs, ty))

/// Emit SubI (integer subtraction)
let emitSubI (ssa: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.SubI (ssa, lhs, rhs, ty))

/// Emit MulI (integer multiplication)
let emitMulI (ssa: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.MulI (ssa, lhs, rhs, ty))

/// Emit DivSI (signed integer division)
let emitDivSI (ssa: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.DivSI (ssa, lhs, rhs, ty))

/// Emit RemSI (signed integer remainder)
let emitRemSI (ssa: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : MLIROp =
    MLIROp.ArithOp (ArithOp.RemSI (ssa, lhs, rhs, ty))

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit direct function call
let emitFuncCall (resultSSA: SSA option) (funcName: string) (args: Val list) (retTy: MLIRType) : MLIROp =
    MLIROp.FuncOp (FuncOp.FuncCall (resultSSA, funcName, args, retTy))

/// Emit function return
let emitReturn (valueSSA: SSA option) (ty: MLIRType option) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Return (valueSSA, ty))
