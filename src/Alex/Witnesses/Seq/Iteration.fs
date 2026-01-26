/// Seq Iteration Witness - Witness ForEach loops over Seq<'T>
///
/// PRD-15: ForEach allocates seq on stack and calls MoveNext repeatedly
///
/// SCOPE: ForEach iteration, placeholder MoveNext
/// DOES NOT: Seq creation, MoveNext state machines (separate witnesses)
module Alex.Witnesses.Seq.Iteration

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Witnesses.Seq.Creation

module SSAAssign = PSGElaboration.SSAAssignment
module SCF = Alex.Dialects.SCF.Templates

let witnessForEach
    (nodeId: NodeId)
    (ssa: SSAAssign.SSAAssignment)
    (seqVal: Val)
    (elementType: MLIRType)
    (bodyOps: MLIROp list)
    : (MLIROp list * TransferResult) =

    let ssas = requireSSAs nodeId ssa
    let seqStructTy = seqVal.Type

    // Pre-assigned SSAs (from SSAAssignment coeffect)
    // Setup SSAs (4)
    let oneSSA = ssas.[0]
    let allocaPtrSSA = ssas.[1]
    let seqLoadSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    // Condition region SSAs (1)
    let hasNextSSA = ssas.[4]
    // Body region SSAs (2)
    let seqReloadSSA = ssas.[5]
    let currentSSA = ssas.[6]
    // Total: 7 SSAs

    // ─────────────────────────────────────────────────────────────────────────
    // SETUP OPS (outside the while loop)
    // ─────────────────────────────────────────────────────────────────────────
    let setupOps = [
        // Create constant 1 for alloca count
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))

        // Alloca space for seq struct on stack
        MLIROp.LLVMOp (LLVMOp.Alloca (allocaPtrSSA, oneSSA, seqStructTy, None))

        // Store seq value to alloca
        MLIROp.LLVMOp (LLVMOp.Store (seqVal.SSA, allocaPtrSSA, seqStructTy, AtomicOrdering.NotAtomic))

        // Load seq struct to extract code_ptr (code_ptr doesn't change during iteration)
        MLIROp.LLVMOp (LLVMOp.Load (seqLoadSSA, allocaPtrSSA, seqStructTy, AtomicOrdering.NotAtomic))

        // Extract MoveNext function pointer from seq struct (at index 2)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, seqLoadSSA, [2], seqStructTy))
    ]

    // ─────────────────────────────────────────────────────────────────────────
    // CONDITION REGION
    // MoveNext receives pointer to seq struct, mutates it, returns i1
    // ─────────────────────────────────────────────────────────────────────────
    let condOps = [
        // Call MoveNext(alloca_ptr) -> i1
        // MoveNext signature: (ptr) -> i1
        MLIROp.LLVMOp (LLVMOp.IndirectCall (
            Some hasNextSSA,
            codePtrSSA,
            [{ SSA = allocaPtrSSA; Type = TPtr }],
            MLIRTypes.i1
        ))
    ]
    let condTerminator = MLIROp.SCFOp (SCF.scfCondition hasNextSSA [])
    let condRegion = SCF.singleBlockRegion "" [] (condOps @ [condTerminator])

    // ─────────────────────────────────────────────────────────────────────────
    // BODY REGION
    // After MoveNext returns true, seq struct has been mutated with new state/current.
    // Load the current value and execute user body.
    // ─────────────────────────────────────────────────────────────────────────
    let bodySetupOps = [
        // Reload seq struct (MoveNext mutated it)
        MLIROp.LLVMOp (LLVMOp.Load (seqReloadSSA, allocaPtrSSA, seqStructTy, AtomicOrdering.NotAtomic))

        // Extract current value from seq struct (at index 1)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (currentSSA, seqReloadSSA, [1], seqStructTy))
    ]

    // bodyOps come from the PSG traversal - they reference the loop variable
    // which is bound to currentSSA
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield [])
    let bodyRegion = SCF.singleBlockRegion "" [] (bodySetupOps @ bodyOps @ [bodyTerminator])

    // ─────────────────────────────────────────────────────────────────────────
    // BUILD SCF.WHILE
    // No iter_args - state is maintained in the seq struct on stack
    // ─────────────────────────────────────────────────────────────────────────
    let whileOp = SCFOp.While ([], condRegion, bodyRegion, [])

    // ForEach returns unit (TRVoid)
    (setupOps @ [MLIROp.SCFOp whileOp], TRVoid)

// ═══════════════════════════════════════════════════════════════════════════
// PLACEHOLDER MOVENEXT (January 2026 - XParsec Integration Pending)
// ═══════════════════════════════════════════════════════════════════════════

/// Placeholder MoveNext for while-based sequences
///
/// TEMPORARY (January 2026):
/// This generates a MoveNext that immediately returns false (empty sequence).
/// Used during FNCSTransfer strip-mining while XParsec integration is in progress.
///
/// The callback-based witnessMoveNextWhileBased was deleted because:
/// - Callbacks inject "build" logic where witnessing should happen
/// - NO dispatch - only witness. NO push - only pull.
/// - Zipper = attention, XParsec = pull
///
/// Proper implementation will use:
/// - Zipper navigation to MoveNext body nodes
/// - XParsec combinators to witness expressions
/// - Coeffect lookup for pre-computed SSAs
///
/// Samples 01-13 (non-Seq) should still work.
/// Samples 14-16 (Seq) will produce empty sequences.
let witnessMoveNextPlaceholder
    (moveNextName: string)
    (seqStructType: MLIRType)
    (elementType: MLIRType)
    : MLIROp =

    // Minimal MoveNext: load state, return false (always done)
    let seqPtrSSA = V 0
    let falseLitSSA = V 1

    let entryOps = [
        // Return false immediately - sequence is "empty"
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))
    ]

    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []
        Ops = entryOps
    }

    let bodyRegion: Region = {
        Blocks = [entryBlock]
    }

    // FLAT CLOSURE PATTERN: llvm.func for addressof compatibility
    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (
        moveNextName,
        [(seqPtrSSA, TPtr)],
        MLIRTypes.i1,
        bodyRegion,
        LLVMPrivate
    ))

// ═══════════════════════════════════════════════════════════════════════════
// SSA COST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// SSA cost for SeqExpr with N captures (no internal state)
let seqExprSSACost (numCaptures: int) : int =
    // 5 base + N captures (same as LazyExpr pattern)
    5 + numCaptures

/// SSA cost for SeqExpr with N captures and M internal state fields
/// PRD-15: While-based sequences with let mutable inside body
let seqExprSSACostFull (numCaptures: int) (numInternalState: int) : int =
    // 5 base + N captures + 2*M internal state (const + insert per field)
    5 + numCaptures + (numInternalState * 2)

/// SSA cost for ForEach
let forEachSSACost : int =
    // 7 fixed: setup (4) + condition (1) + body (2)
    7

/// SSA cost for Yield (inside MoveNext function)
let yieldSSACost : int =
    // 4: gep to current + store + gep to state + store state
    4

/// SSA cost for YieldBang (yield! - flatten nested seq)
/// PRD-15 SimpleSeq: Not supported - returns 0
/// Future PRDs will implement proper nested iteration
let yieldBangSSACost : int = 0
