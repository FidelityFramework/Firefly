/// CF (Control Flow) Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → CFOp
/// NO sprintf. NO string formatting. Just data construction.
///
/// Source: /usr/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td
module Alex.Dialects.CF.Templates

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// ASSERTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Assert a condition with a message: cf.assert
/// If condition is false, aborts execution with the given message
let assert' (cond: SSA) (message: string) : CFOp =
    CFOp.Assert (cond, message)

// ═══════════════════════════════════════════════════════════════════════════
// UNCONDITIONAL BRANCH
// ═══════════════════════════════════════════════════════════════════════════

/// Unconditional branch: cf.br
/// Transfers control to the destination block with optional arguments
let br (dest: BlockRef) (destOperands: Val list) : CFOp =
    CFOp.Br (dest, destOperands)

/// Unconditional branch with no arguments
let brSimple (dest: BlockRef) : CFOp =
    CFOp.Br (dest, [])

// ═══════════════════════════════════════════════════════════════════════════
// CONDITIONAL BRANCH
// ═══════════════════════════════════════════════════════════════════════════

/// Conditional branch: cf.cond_br
/// If condition is true, branch to trueDest; otherwise branch to falseDest
let condBr 
    (cond: SSA) 
    (trueDest: BlockRef) (trueDestOperands: Val list) 
    (falseDest: BlockRef) (falseDestOperands: Val list) 
    (branchWeights: (int * int) option) : CFOp =
    CFOp.CondBr (cond, trueDest, trueDestOperands, falseDest, falseDestOperands, branchWeights)

/// Conditional branch with no operands and no weights
let condBrSimple (cond: SSA) (trueDest: BlockRef) (falseDest: BlockRef) : CFOp =
    CFOp.CondBr (cond, trueDest, [], falseDest, [], None)

/// Conditional branch with branch weights
let condBrWeighted 
    (cond: SSA) 
    (trueDest: BlockRef) 
    (falseDest: BlockRef) 
    (trueWeight: int) 
    (falseWeight: int) : CFOp =
    CFOp.CondBr (cond, trueDest, [], falseDest, [], Some (trueWeight, falseWeight))

// ═══════════════════════════════════════════════════════════════════════════
// SWITCH
// ═══════════════════════════════════════════════════════════════════════════

/// Switch statement: cf.switch
/// Multi-way branch based on an integer flag value
let switch 
    (flag: SSA) 
    (flagTy: MLIRType) 
    (defaultDest: BlockRef) 
    (defaultOperands: Val list) 
    (cases: (int64 * BlockRef * Val list) list) : CFOp =
    CFOp.Switch (flag, flagTy, defaultDest, defaultOperands, cases)

/// Simple switch with no operands passed to any destination
let switchSimple 
    (flag: SSA) 
    (flagTy: MLIRType) 
    (defaultDest: BlockRef) 
    (cases: (int64 * BlockRef) list) : CFOp =
    let casesWithOps = cases |> List.map (fun (v, dest) -> (v, dest, []))
    CFOp.Switch (flag, flagTy, defaultDest, [], casesWithOps)

// ═══════════════════════════════════════════════════════════════════════════
// CONVENIENCE PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a conditional that uses true/false branches for if/else style control
let ifElse (cond: SSA) (ifBlock: BlockRef) (elseBlock: BlockRef) : CFOp =
    condBrSimple cond ifBlock elseBlock

/// Create a loop back edge (unconditional branch to loop header)
let loopBackEdge (headerBlock: BlockRef) (iterArgs: Val list) : CFOp =
    br headerBlock iterArgs

/// Create a loop exit (unconditional branch to exit block)
let loopExit (exitBlock: BlockRef) (resultArgs: Val list) : CFOp =
    br exitBlock resultArgs

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap CFOp in MLIROp
let wrap (op: CFOp) : MLIROp = MLIROp.CFOp op
