/// MLIRAtomics - Atomic MLIR operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides atomic MLIR op construction via XParsec-compatible interface.
module internal Alex.Elements.MLIRAtomics

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// Elements accept type from caller - patterns know the type and pass it explicitly

// ═══════════════════════════════════════════════════════════
// STRUCT OFFSET COMPUTATION
// ═══════════════════════════════════════════════════════════

/// SCAB REMOVED: This function should not exist - callers must use coeffect-provided SSAs directly
/// Keeping as compiler error signal - if called, tells us where pattern needs refactoring
let private computeStructOffset (indices: int list) : SSA =
    failwith "ARCHITECTURAL ERROR: computeStructOffset called - this function should have been removed entirely. Caller must use coeffects."

// ═══════════════════════════════════════════════════════════
// PORTABLE MLIR STRUCT OPERATIONS (MemRef-based)
// ═══════════════════════════════════════════════════════════

/// ExtractValue - memref.load with int field index
/// Emits constant + load operations
let pExtractValue (ssa: SSA) (structMemref: SSA) (fieldIndex: int) (offsetSSA: SSA) (ty: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! emitTrace "pExtractValue" (sprintf "ssa=%A, memref=%A, field=%d, ty=%A" ssa structMemref fieldIndex ty)
        let offsetOp = ArithOp.ConstI (offsetSSA, int64 fieldIndex, TIndex) |> MLIROp.ArithOp
        let loadOp = MemRefOp.Load (ssa, structMemref, [offsetSSA], ty) |> MLIROp.MemRefOp
        return [offsetOp; loadOp]
    }

/// InsertValue - memref.store with int field index
/// Emits constant + store operations
let pInsertValue (resultSSA: SSA) (structMemref: SSA) (value: SSA) (fieldIndex: int) (offsetSSA: SSA) (ty: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! emitTrace "pInsertValue" (sprintf "result=%A, memref=%A, value=%A, field=%d, ty=%A" resultSSA structMemref value fieldIndex ty)
        let offsetOp = ArithOp.ConstI (offsetSSA, int64 fieldIndex, TIndex) |> MLIROp.ArithOp
        let memrefType = TMemRefStatic (1, ty)  // 1-element struct field storage
        let storeOp = MemRefOp.Store (value, structMemref, [offsetSSA], ty, memrefType) |> MLIROp.MemRefOp
        return [offsetOp; storeOp]
    }

/// Undef - NOW USES memref.alloca (uninitialized allocation)
/// SEMANTIC CHANGE: Creates memref in stack memory instead of undef SSA value
/// Semantically equivalent: uninitialized memory = undef value
let pUndef (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pUndef" (sprintf "ssa=%A, ty=%A" ssa ty)
        // Allocate uninitialized memref (semantically equivalent to undef)
        return MLIROp.MemRefOp (MemRefOp.Alloca (ssa, ty, None))
    }

/// ConstI - caller provides type
let pConstI (ssa: SSA) (value: int64) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pConstI" (sprintf "ssa=%A, value=%d, ty=%A" ssa value ty)
        return MLIROp.ArithOp (ArithOp.ConstI (ssa, value, ty))
    }

/// ConstF - caller provides type
let pConstF (ssa: SSA) (value: float) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.ArithOp (ArithOp.ConstF (ssa, value, ty))
    }

/// GlobalString - module-level string constant
let pGlobalString (name: string) (content: string) (byteLength: int) : PSGParser<MLIROp> =
    parser {
        return GlobalString (name, content, byteLength)
    }

