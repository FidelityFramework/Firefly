/// MemRefElements - Atomic MemRef dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides memory operations (alloca, load, store) via XParsec state threading.
///
/// NOTE: This replaces LLVM dialect memory operations (llvm.alloca, llvm.load, llvm.store)
/// with standard MLIR MemRef dialect operations for backend flexibility.
module internal Alex.Elements.MemRefElements

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

/// Emit memref.load operation
let pLoad (ssa: SSA) (memref: SSA) (indices: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.MemRefOp (MemRefOp.Load (ssa, memref, indices, ty))
    }

/// Emit memref.store operation
/// elemType: element type of the memref (must match value type)
/// memrefType: full memref type for serialization
let pStore (value: SSA) (memref: SSA) (indices: SSA list) (elemType: MLIRType) (memrefType: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.MemRefOp (MemRefOp.Store (value, memref, indices, elemType, memrefType))
    }

/// Emit memref.alloca operation (stack allocation with compile-time size)
/// count: number of elements to allocate
/// elemType: explicit element type for the memref (e.g., TInt I8, TInt I32, TMemRefStatic(...))
let pAlloca (ssa: SSA) (count: int) (elemType: MLIRType) (alignment: int option) : PSGParser<MLIROp> =
    parser {
        let memrefType = TMemRefStatic (count, elemType)
        return MLIROp.MemRefOp (MemRefOp.Alloca (ssa, memrefType, alignment))
    }

/// Emit memref.alloc operation (heap allocation with runtime size)
/// sizeSSA: index-typed SSA value representing the number of elements
/// elemType: element type for the memref (e.g., TInt I8)
/// Returns a dynamic memref (memref<?xelemType>)
let pAlloc (ssa: SSA) (sizeSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp> =
    parser {
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
/// Emits: %result = memref.get_global @globalName : memrefType
let pMemRefGetGlobal (result: SSA) (globalName: string) (memrefType: MLIRType) : PSGParser<MLIROp> =
    parser {
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
