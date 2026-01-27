/// LLVMElements - Atomic LLVM dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides memory, call, branch, and block operations via XParsec state threading.
module internal Alex.Elements.LLVMElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit Load operation
let pLoad (ssa: SSA) (ptr: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Load (ssa, ptr, ty, AtomicOrdering.NotAtomic))
    }

/// Emit Store operation
let pStore (value: SSA) (ptr: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Store (value, ptr, ty, AtomicOrdering.NotAtomic))
    }

/// Emit GEP (GetElementPtr) operation
let pGEP (ssa: SSA) (ptr: SSA) (indices: (SSA * MLIRType) list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.GEP (ssa, ptr, indices, ty, None))
    }

/// Emit StructGEP (struct field pointer) operation
let pStructGEP (ssa: SSA) (ptr: SSA) (fieldIndex: int) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let structTy = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type  // The struct type being indexed
        return MLIROp.LLVMOp (LLVMOp.StructGEP (ssa, ptr, fieldIndex, structTy, ty))
    }

/// Emit Alloca (stack allocation) operation
let pAlloca (ssa: SSA) (size: int option) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Alloca (ssa, ty, size))
    }

/// Emit BitCast operation
let pBitCast (ssa: SSA) (value: SSA) (targetTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.BitCast (ssa, value, targetTy))
    }

/// Emit IntToPtr operation
let pIntToPtr (ssa: SSA) (value: SSA) (targetTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.IntToPtr (ssa, value, targetTy))
    }

/// Emit PtrToInt operation
let pPtrToInt (ssa: SSA) (ptr: SSA) (targetTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.PtrToInt (ssa, ptr, targetTy))
    }

// ═══════════════════════════════════════════════════════════
// CALL OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit Call (direct function call)
let pCall (ssa: SSA) (funcName: string) (args: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let returnTy = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Call (ssa, funcName, args, returnTy))
    }

/// Emit IndirectCall (call via function pointer)
let pIndirectCall (ssa: SSA) (funcPtr: SSA) (args: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let returnTy = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.IndirectCall (ssa, funcPtr, args, returnTy))
    }

/// Emit Return (function return)
let pReturn (value: SSA option) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.Return value)
    }

/// Emit Phi (SSA phi node for control flow merge)
let pPhi (ssa: SSA) (incoming: (SSA * string) list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Phi (ssa, incoming, ty))
    }

// ═══════════════════════════════════════════════════════════
// BRANCH OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit CondBranch (conditional branch)
let pCondBranch (cond: SSA) (thenLabel: string) (elseLabel: string) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.CondBranch (cond, thenLabel, elseLabel))
    }

/// Emit Branch (unconditional branch)
let pBranch (label: string) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.Branch label)
    }

// ═══════════════════════════════════════════════════════════
// STRUCTURAL OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit Block with label and operations
let pBlock (label: string) (ops: MLIROp list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.Block (label, ops)
    }

/// Emit Region with blocks
let pRegion (blocks: MLIROp list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.Region blocks
    }

// ═══════════════════════════════════════════════════════════
// BITWISE OPERATIONS (LLVM dialect)
// ═══════════════════════════════════════════════════════════

/// Emit And (bitwise and)
let pAndI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.And (ssa, lhs, rhs, ty))
    }

/// Emit Or (bitwise or)
let pOrI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Or (ssa, lhs, rhs, ty))
    }

/// Emit Xor (bitwise xor)
let pXorI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Xor (ssa, lhs, rhs, ty))
    }

/// Emit Shl (shift left)
let pShLI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Shl (ssa, lhs, rhs, ty))
    }

/// Emit LShr (logical shift right - unsigned)
let pShRUI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.LShr (ssa, lhs, rhs, ty))
    }

/// Emit AShr (arithmetic shift right - signed)
let pShRSI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.AShr (ssa, lhs, rhs, ty))
    }
