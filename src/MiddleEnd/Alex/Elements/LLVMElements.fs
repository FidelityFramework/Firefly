/// LLVMElements - Atomic LLVM dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides LLVM-specific operations (C ABI calls, pointer manipulation, inline asm, etc.)
/// via XParsec state threading.
///
/// CRITICAL: LLVM dialect should ONLY be used for operations with no standard MLIR equivalent.
/// See /docs/LLVM_Dialect_Reference.md for the 11 approved LLVM-specific operations.
/// Use MemRef dialect for memory ops, Arith dialect for bitwise ops, CF dialect for branches.
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
// COMPATIBILITY WRAPPERS - DEPRECATED
// ═══════════════════════════════════════════════════════════
// These functions provide backward compatibility for Pattern files still using
// LLVM dialect signatures. They delegate to MemRefElements or remain as LLVM-specific.
// TODO: Migrate Pattern files to use MemRefElements directly, then remove these wrappers.

open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements

/// DEPRECATED: Use MemRefElements.pLoad instead
/// Wrapper for backward compatibility with Pattern files
let pLoad (ssa: SSA) (ptr: SSA) : PSGParser<MLIROp> =
    MemRefElements.pLoad ssa ptr []

/// DEPRECATED: For simple array indexing, use MemRefElements.pSubView instead
/// For C struct layout, use pStructGEP below
/// Wrapper for backward compatibility with Pattern files
let pGEP (ssa: SSA) (ptr: SSA) (indices: (SSA * MLIRType) list) : PSGParser<MLIROp> =
    let ssaIndices = indices |> List.map fst
    MemRefElements.pSubView ssa ptr ssaIndices

// ═══════════════════════════════════════════════════════════
// POINTER OPERATIONS (C struct layout, type conversions)
// ═══════════════════════════════════════════════════════════

/// Emit StructGEP (struct field pointer) operation (C struct layout)
let pStructGEP (ssa: SSA) (ptr: SSA) (fieldIndex: int) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let structTy = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type  // The struct type being indexed
        return MLIROp.LLVMOp (LLVMOp.StructGEP (ssa, ptr, fieldIndex, structTy, ty))
    }

/// Emit BitCast operation (pointer type conversions)
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
let pCall (ssa: SSA) (funcName: string) (args: (SSA * MLIRType) list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let returnTy = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Call (ssa, funcName, args, returnTy))
    }

/// Emit IndirectCall (call via function pointer)
let pIndirectCall (ssa: SSA) (funcPtr: SSA) (args: (SSA * MLIRType) list) : PSGParser<MLIROp> =
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
// STRUCTURAL OPERATIONS (Block/Region - NOT dialect-specific)
// ═══════════════════════════════════════════════════════════
// NOTE: Branch operations migrated to CFElements (CF dialect).

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
// BITWISE OPERATION WRAPPERS - DEPRECATED
// ═══════════════════════════════════════════════════════════
// These functions provide backward compatibility for Pattern files still using
// LLVM dialect signatures for bitwise operations.
// They delegate to ArithElements (Arith dialect).
// TODO: Migrate Pattern files to use ArithElements directly, then remove these wrappers.

/// DEPRECATED: Use ArithElements.pAndI instead
let pAndI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pAndI ssa lhs rhs

/// DEPRECATED: Use ArithElements.pOrI instead
let pOrI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pOrI ssa lhs rhs

/// DEPRECATED: Use ArithElements.pXorI instead
let pXorI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pXorI ssa lhs rhs

/// DEPRECATED: Use ArithElements.pShLI instead
let pShLI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pShLI ssa lhs rhs

/// DEPRECATED: Use ArithElements.pShRUI instead
let pShRUI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pShRUI ssa lhs rhs

/// DEPRECATED: Use ArithElements.pShRSI instead
let pShRSI (ssa: SSA) (lhs: SSA) (rhs: SSA) : PSGParser<MLIROp> =
    ArithElements.pShRSI ssa lhs rhs
