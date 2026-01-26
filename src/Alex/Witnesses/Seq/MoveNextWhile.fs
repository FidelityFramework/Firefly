/// Seq MoveNext While Witness - While-based yield pattern state machine
///
/// PRD-15: Two-state model for while-based sequences
///
/// SCOPE: MoveNext for seq { while cond do yield expr } patterns
/// DOES NOT: Sequential patterns, ForEach (separate witnesses)
module Alex.Witnesses.Seq.MoveNextWhile

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Witnesses.Seq.Creation
open PSGElaboration.YieldStateIndices

module CF = Alex.Dialects.CF.Templates

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Information for while-based MoveNext generation
type WhileBasedMoveNextInfo = {
    MoveNextName: string
    SeqStructType: MLIRType
    ElementType: MLIRType
    WhileInfo: WhileBodyInfo
    InternalState: InternalStateField list
    NumCaptures: int
    CaptureTypes: NativeType list
}

// WHILE-BASED MOVENEXT STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════════

/// Generate the MoveNext state machine function for while-based sequences
///
/// PRD-15: While-based sequences use a TWO-STATE model:
/// - State 0: Initialize internal state vars, jump to check
/// - State 1: Execute post-yield code, jump to check
/// - Shared blocks: check, yield, done
///
/// DIALECT ARCHITECTURE (January 2026):
/// - Function definition: func.func (portable across MLIR backends)
/// - Control flow: cf.switch, cf.br, cf.cond_br (portable)
/// - Struct access: llvm.getelementptr, llvm.load, llvm.store (necessary - no MLIR memref for structs)
/// - Return: func.return (portable)
///
/// STRUCTURE:
/// ```mlir
/// func.func private @moveNext(%seq_ptr: ptr) -> i1 {
///   %state_ptr = llvm.getelementptr %seq_ptr[0, 0]  // LLVM for struct access
///   %state = llvm.load %state_ptr : i32
///   cf.switch %state [0: ^s0, 1: ^s1], ^done        // CF for control flow
///
/// ^s0:  // Initial: init mutable vars
///   <initialize internal state fields>
///   cf.br ^check
///
/// ^s1:  // After yield: post-yield code
///   <post-yield expressions>
///   cf.br ^check
///
/// ^check:
///   %cond = <evaluate while condition>
///   cf.cond_br %cond, ^yield, ^done
///
/// ^yield:
///   %value = <compute yield value>
///   llvm.store %value, %seq_ptr[0, 1]   // store to current (LLVM for structs)
///   llvm.store 1, %seq_ptr[0, 0]        // always transition to state 1
///   func.return true                     // func.return for portability
///
/// ^done:
///   llvm.store -1, %seq_ptr[0, 0]
///   func.return false
/// }
/// ```
///
/// NOTE: This is a simplified generator that works with the structural information
/// from YieldStateIndices. For computed expressions, we emit placeholder operations
/// that will be filled in by the FNCSTransfer handler using the actual PSG traversal.
let witnessMoveNextWhileBased
    (info: WhileBasedMoveNextInfo)
    (emitCondition: unit -> (MLIROp list * SSA))  // Emit while condition, return ops and result SSA
    (emitYieldValue: unit -> (MLIROp list * SSA)) // Emit yield value, return ops and result SSA
    (emitPostYield: unit -> MLIROp list)          // Emit post-yield ops (e.g., i <- i + 1)
    (emitPreYield: unit -> MLIROp list)           // Emit pre-yield ops (e.g., sum <- sum + i)
    (emitInit: unit -> MLIROp list)               // Emit internal state initialization
    (emitYieldCondition: (unit -> (MLIROp list * SSA)) option)  // Emit if-condition for conditional yield
    : MLIROp =

    // SSA counter for function-local SSAs
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    // Function parameter: seq_ptr
    let seqPtrSSA = nextSSA ()  // %0

    // Entry block: load state
    let statePtrSSA = nextSSA ()     // %1
    let stateValSSA = nextSSA ()     // %2
    let trueLitSSA = nextSSA ()      // %3
    let falseLitSSA = nextSSA ()     // %4
    let oneSSA = nextSSA ()          // %5 (constant 1 for state transition)
    let neg1SSA = nextSSA ()         // %6 (constant -1 for done state)

    // Block references
    let checkBlockRef = BlockRef "check"
    let yieldBlockRef = BlockRef "yield"
    let doneBlockRef = BlockRef "done"

    // Entry block: load state, switch to s0, s1, or done
    let entryOps = [
        // Get pointer to state field (index 0)
        MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA, seqPtrSSA, 0, info.SeqStructType))
        // Load state
        MLIROp.LLVMOp (LLVMOp.Load (stateValSSA, statePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
        // Constants
        MLIROp.ArithOp (ArithOp.ConstI (trueLitSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.ConstI (neg1SSA, -1L, MLIRTypes.i32))
    ]

    // Switch: state 0 -> ^s0, state 1 -> ^s1, default -> ^done
    // Using CF dialect for backend portability (CF can lower to multiple targets)
    let switchOp = MLIROp.CFOp (CF.switchSimple
        stateValSSA
        (TInt I32)
        doneBlockRef
        [(0L, BlockRef "s0"); (1L, BlockRef "s1")])

    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []
        Ops = entryOps @ [switchOp]
    }

    // ^s0: Initialize internal state vars, jump to check
    let initOps = emitInit ()
    let s0Block: Block = {
        Label = BlockRef "s0"
        Args = []
        Ops = initOps @ [MLIROp.CFOp (CF.brSimple checkBlockRef)]
    }

    // ^s1: Post-yield code, jump to check
    let postYieldOps = emitPostYield ()
    let s1Block: Block = {
        Label = BlockRef "s1"
        Args = []
        Ops = postYieldOps @ [MLIROp.CFOp (CF.brSimple checkBlockRef)]
    }

    // ^check: Evaluate while condition, branch to yield or done
    let (condOps, condSSA) = emitCondition ()
    let checkBlock: Block =
        match info.WhileInfo.ConditionalYield with
        | None ->
            // No conditional yield - branch directly to yield or done
            {
                Label = checkBlockRef
                Args = []
                Ops = condOps @ [MLIROp.CFOp (CF.condBrSimple condSSA yieldBlockRef doneBlockRef)]
            }
        | Some condYieldInfo ->
            // Has conditional yield - branch to yield_check first
            {
                Label = checkBlockRef
                Args = []
                Ops = condOps @ [MLIROp.CFOp (CF.condBrSimple condSSA (BlockRef "yield_check") doneBlockRef)]
            }

    // ^yield_check (optional): Evaluate if condition for conditional yield
    let yieldCheckBlock =
        match info.WhileInfo.ConditionalYield, emitYieldCondition with
        | Some _, Some emitCond ->
            // Evaluate the if-condition and branch accordingly
            // If true -> yield, if false -> s1 (post-yield, loop back to check)
            let (yieldCondOps, yieldCondSSA) = emitCond ()
            Some {
                Label = BlockRef "yield_check"
                Args = []
                Ops = yieldCondOps @ [MLIROp.CFOp (CF.condBrSimple yieldCondSSA yieldBlockRef (BlockRef "s1"))]
            }
        | Some _, None ->
            // Conditional yield but no emitter provided - fall back to unconditional
            Some {
                Label = BlockRef "yield_check"
                Args = []
                Ops = [MLIROp.CFOp (CF.brSimple yieldBlockRef)]
            }
        | None, _ -> None

    // ^yield: Execute pre-yield ops, compute value, store to current, set state=1, return true
    // PRD-15: Pre-yield ops (like sum <- sum + i) execute BEFORE yielding
    let preYieldOps = emitPreYield ()
    let (yieldValueOps, valueSSA) = emitYieldValue ()
    let currentPtrSSA = nextSSA ()
    let statePtrSSA2 = nextSSA ()  // Need a fresh SSA for the second GEP

    let yieldStoreOps = [
        // Get pointer to current field (index 1)
        MLIROp.LLVMOp (LLVMOp.StructGEP (currentPtrSSA, seqPtrSSA, 1, info.SeqStructType))
        // Store value to current
        MLIROp.LLVMOp (LLVMOp.Store (valueSSA, currentPtrSSA, info.ElementType, AtomicOrdering.NotAtomic))
        // Get pointer to state field (index 0) - need fresh SSA
        MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA2, seqPtrSSA, 0, info.SeqStructType))
        // Store state = 1 (always transition to state 1 for while-based)
        MLIROp.LLVMOp (LLVMOp.Store (oneSSA, statePtrSSA2, TInt I32, AtomicOrdering.NotAtomic))
        // Return true (llvm.return matches llvm.func - flat closure pattern)
        MLIROp.LLVMOp (LLVMOp.Return (Some trueLitSSA, Some MLIRTypes.i1))
    ]

    let yieldBlock: Block = {
        Label = yieldBlockRef
        Args = []
        Ops = preYieldOps @ yieldValueOps @ yieldStoreOps
    }

    // ^done: Set state = -1, return false
    let donePtrSSA = nextSSA ()
    let doneBlock: Block = {
        Label = doneBlockRef
        Args = []
        Ops = [
            MLIROp.LLVMOp (LLVMOp.StructGEP (donePtrSSA, seqPtrSSA, 0, info.SeqStructType))
            MLIROp.LLVMOp (LLVMOp.Store (neg1SSA, donePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
            MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))
        ]
    }

    // Build the function body region
    let blocks =
        [entryBlock; s0Block; s1Block; checkBlock]
        @ (yieldCheckBlock |> Option.toList)
        @ [yieldBlock; doneBlock]

    let bodyRegion: Region = {
        Blocks = blocks
    }

    // FLAT CLOSURE PATTERN (January 2026):
    // MoveNext uses llvm.func because its address is taken via llvm.mlir.addressof
    // and stored in the Seq struct. This is the same pattern as closures and lazy thunks.
    // See fsnative-spec/spec/drafts/backend-lowering-architecture.md for rationale.
    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (
        info.MoveNextName,
        [(seqPtrSSA, TPtr)],
        MLIRTypes.i1,
        bodyRegion,
        LLVMPrivate
    ))

/// Compute SSAs needed for while-based MoveNext function
let moveNextWhileBasedSSACost (hasConditionalYield: bool) : int =
    // Entry: seqPtrSSA, statePtrSSA, stateValSSA, trueLitSSA, falseLitSSA, oneSSA, neg1SSA = 7
    // Yield block: currentPtrSSA, statePtrSSA2 = 2
    // Done block: donePtrSSA = 1
    // Plus condition ops and value ops (estimated)
    let baseCost = 7 + 2 + 1  // 10
    if hasConditionalYield then baseCost + 2 else baseCost

