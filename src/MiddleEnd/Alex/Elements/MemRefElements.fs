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
let pStore (value: SSA) (memref: SSA) (indices: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.MemRefOp (MemRefOp.Store (value, memref, indices, ty))
    }

/// Emit memref.alloca operation (stack allocation)
let pAlloca (ssa: SSA) (alignment: int option) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let memrefType = TMemRef ty
        return MLIROp.MemRefOp (MemRefOp.Alloca (ssa, memrefType, alignment))
    }

/// Emit memref.subview operation (replaces GEP for arrays)
let pSubView (ssa: SSA) (source: SSA) (offsets: SSA list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        let memrefType = TMemRef ty
        return MLIROp.MemRefOp (MemRefOp.SubView (ssa, source, offsets, memrefType))
    }
