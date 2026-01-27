/// CFElements - Atomic CF (Control Flow) dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides ALL CF dialect operations from Types.fs (unstructured control flow).
module internal Alex.Elements.CFElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// ═══════════════════════════════════════════════════════════
// ASSERTIONS
// ═══════════════════════════════════════════════════════════

let pAssert (cond: SSA) (message: string) : PSGParser<MLIROp> =
    parser { return MLIROp.CFOp (CFOp.Assert (cond, message)) }

// ═══════════════════════════════════════════════════════════
// UNCONDITIONAL BRANCH
// ═══════════════════════════════════════════════════════════

let pBr (dest: BlockRef) (destOperands: Val list) : PSGParser<MLIROp> =
    parser { return MLIROp.CFOp (CFOp.Br (dest, destOperands)) }

// ═══════════════════════════════════════════════════════════
// CONDITIONAL BRANCH
// ═══════════════════════════════════════════════════════════

let pCondBr (cond: SSA) (trueDest: BlockRef) (trueDestOperands: Val list)
                (falseDest: BlockRef) (falseDestOperands: Val list)
                (branchWeights: (int * int) option) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CFOp (CFOp.CondBr (cond, trueDest, trueDestOperands,
                                         falseDest, falseDestOperands, branchWeights))
    }

// ═══════════════════════════════════════════════════════════
// SWITCH
// ═══════════════════════════════════════════════════════════

let pSwitch (flag: SSA) (flagTy: MLIRType) (defaultDest: BlockRef) (defaultOperands: Val list)
                (cases: (int64 * BlockRef * Val list) list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CFOp (CFOp.Switch (flag, flagTy, defaultDest, defaultOperands, cases))
    }
