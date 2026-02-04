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

/// Compute flat offset SSA from nested struct indices
/// ARCHITECTURAL VIOLATION REMOVED: SSAs must come from SSAAssignment, not generated inline
let private computeStructOffset (indices: int list) : SSA =
    failwith "MLIRAtomics.computeStructOffset: offset SSAs must come from coeffects (removed inline SSA generation)"

// ═══════════════════════════════════════════════════════════
// PORTABLE MLIR STRUCT OPERATIONS (MemRef-based)
// ═══════════════════════════════════════════════════════════

/// ExtractValue - NOW USES memref.load with witnessed offset
/// SEMANTIC CHANGE: Struct is now in memory (memref), not SSA register value
/// offsetSSA: Pre-assigned SSA for the offset constant (from coeffects)
let pExtractValue (ssa: SSA) (structMemref: SSA) (offsetSSA: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pExtractValue" (sprintf "ssa=%A, memref=%A, offset=%A, ty=%A" ssa structMemref offsetSSA ty)
        // Load from memref at witnessed offset
        return MLIROp.MemRefOp (MemRefOp.Load (ssa, structMemref, [offsetSSA], ty))
    }

/// InsertValue - NOW USES memref.store with witnessed offset
/// SEMANTIC CHANGE: Struct is now in memory (memref), not SSA register value
/// offsetSSA: Pre-assigned SSA for the offset constant (from coeffects)
let pInsertValue (resultSSA: SSA) (structMemref: SSA) (value: SSA) (offsetSSA: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pInsertValue" (sprintf "result=%A, memref=%A, value=%A, offset=%A, ty=%A" resultSSA structMemref value offsetSSA ty)
        // Store value into memref at witnessed offset
        // Note: resultSSA is the memref AFTER store (same as input memref in memref semantics)
        return MLIROp.MemRefOp (MemRefOp.Store (value, structMemref, [offsetSSA], ty))
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

