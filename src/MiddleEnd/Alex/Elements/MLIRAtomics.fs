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
/// For now, returns a constant SSA for the first index (simplified implementation)
/// TODO: Full implementation would compute offset from type layout
let private computeStructOffset (indices: int list) : SSA =
    match indices with
    | [] -> SSA.V 0  // No offset
    | [i] -> SSA.V (1000000 + i)  // Simple offset encoding for single-level access
    | indices -> SSA.V (1000000 + (indices |> List.sum))  // Simplified for nested access

// ═══════════════════════════════════════════════════════════
// PORTABLE MLIR STRUCT OPERATIONS (MemRef-based)
// ═══════════════════════════════════════════════════════════

/// ExtractValue - NOW USES memref.load with computed offset
/// SEMANTIC CHANGE: Struct is now in memory (memref), not SSA register value
let pExtractValue (ssa: SSA) (structMemref: SSA) (indices: int list) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pExtractValue" (sprintf "ssa=%A, memref=%A, indices=%A, ty=%A" ssa structMemref indices ty)
        // Compute offset from indices (simplified - actual offset depends on type layout)
        let offsetSSA = computeStructOffset indices
        // Load from memref at computed offset
        return MLIROp.MemRefOp (MemRefOp.Load (ssa, structMemref, [offsetSSA], ty))
    }

/// InsertValue - NOW USES memref.store with computed offset
/// SEMANTIC CHANGE: Struct is now in memory (memref), not SSA register value
let pInsertValue (resultSSA: SSA) (structMemref: SSA) (value: SSA) (indices: int list) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        do! emitTrace "pInsertValue" (sprintf "result=%A, memref=%A, value=%A, indices=%A, ty=%A" resultSSA structMemref value indices ty)
        // Compute offset from indices
        let offsetSSA = computeStructOffset indices
        // Store value into memref at computed offset
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

