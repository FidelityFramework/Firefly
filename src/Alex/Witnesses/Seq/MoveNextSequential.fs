/// Seq MoveNext Sequential Witness - Sequential yield pattern state machine
///
/// PRD-15: Fixed sequential yields (state 0, 1, 2, ..., done)
///
/// SCOPE: MoveNext for seq { yield 1; yield 2; yield 3 } patterns
/// DOES NOT: While-based patterns, ForEach (separate witnesses)
module Alex.Witnesses.Seq.MoveNextSequential

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Witnesses.Seq.Creation

module CF = Alex.Dialects.CF.Templates

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// What value to yield at a state point
type YieldValueSource =
    | IntLiteral of value: int64
    | ComputedValue of valueNodeId: NodeId

/// Information needed to generate a yield block
type YieldBlockInfo = {
    StateIndex: int
    ValueSource: YieldValueSource
    ValueType: MLIRType
}

let witnessMoveNext
    (moveNextName: string)
    (seqStructType: MLIRType)
    (elementType: MLIRType)
    (yieldBlocks: YieldBlockInfo list)
    : MLIROp =

    // SSA counter for function-local SSAs
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    // Function parameter: seq_ptr
    let seqPtrSSA = nextSSA ()  // %0

    // Entry block: load state, switch
    let stateSSA = nextSSA ()        // %1
    let trueLitSSA = nextSSA ()      // %2
    let falseLitSSA = nextSSA ()     // %3
    let neg1SSA = nextSSA ()         // %4

    // Block references
    let doneBlockRef = BlockRef "done"

    // Build entry block ops
    let entryOps = [
        // Load state from seq_ptr[0] (state field is at index 0)
        // Using StructGEP to get pointer to state field, then load
        MLIROp.LLVMOp (LLVMOp.StructGEP (stateSSA, seqPtrSSA, 0, seqStructType))
    ]

    // We need another SSA for the loaded state value
    let stateValSSA = nextSSA ()  // %5
    let entryOps2 = [
        MLIROp.LLVMOp (LLVMOp.Load (stateValSSA, stateSSA, TInt I32, AtomicOrdering.NotAtomic))
        // Constants for return values
        MLIROp.ArithOp (ArithOp.ConstI (trueLitSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (falseLitSSA, 0L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.ConstI (neg1SSA, -1L, MLIRTypes.i32))
    ]

    // Build switch cases: state 0 -> ^s0, state 1 -> ^s1, etc.
    let numYields = List.length yieldBlocks
    let switchCases =
        [0 .. numYields - 1]
        |> List.map (fun stateIdx -> (int64 stateIdx, BlockRef (sprintf "s%d" stateIdx)))

    // Entry block terminator: switch on state (using CF dialect for backend portability)
    let switchOp = MLIROp.CFOp (CF.switchSimple stateValSSA (TInt I32) doneBlockRef switchCases)

    // Entry block - Note: Args are for block parameters, function args are in LLVMFuncDef
    // The terminator is the last op in Ops
    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []  // Entry block has no block args (func args are separate)
        Ops = entryOps @ entryOps2 @ [switchOp]
    }

    // Generate state blocks
    // Each state block N:
    // - Computes the value for yield N (yieldBlocks[N])
    // - Stores value to seq_ptr[1] (current field)
    // - Stores next state to seq_ptr[0] (state field)
    // - Returns true
    let stateBlocks =
        yieldBlocks
        |> List.mapi (fun blockIdx yieldInfo ->
            // This is block for state = blockIdx
            let blockName = sprintf "s%d" blockIdx

            // SSAs for this block
            let valueSSA = nextSSA ()      // SSA for the computed/constant value
            let currentPtrSSA = nextSSA ()
            let nextStateSSA = nextSSA ()
            let statePtrSSA = nextSSA ()

            // Compute next state: last yield transitions to done (-1), others to next block
            let nextState =
                if blockIdx = numYields - 1 then
                    -1L  // Last yield transitions to done state
                else
                    int64 (blockIdx + 1)

            // Generate op to compute the yield value based on YieldValueSource
            let valueOp =
                match yieldInfo.ValueSource with
                | IntLiteral v ->
                    // Generate constant for the literal value
                    MLIROp.ArithOp (ArithOp.ConstI (valueSSA, v, yieldInfo.ValueType))
                | ComputedValue nodeId ->
                    // Sequential pattern only supports literals - computed values require while-based pattern
                    // This is a dispatch error in FNCSTransfer - it should have routed to WhileBased
                    failwithf "SeqWitness.witnessMoveNext: ComputedValue (nodeId=%d) in sequential pattern. Sequential yields must be literals; computed expressions require while-based MoveNext. State=%d"
                        (NodeId.value nodeId) yieldInfo.StateIndex

            let storeOps = [
                // Get pointer to current field (index 1)
                MLIROp.LLVMOp (LLVMOp.StructGEP (currentPtrSSA, seqPtrSSA, 1, seqStructType))
                // Store value to current
                MLIROp.LLVMOp (LLVMOp.Store (valueSSA, currentPtrSSA, elementType, AtomicOrdering.NotAtomic))
                // Constant for next state
                MLIROp.ArithOp (ArithOp.ConstI (nextStateSSA, nextState, MLIRTypes.i32))
                // Get pointer to state field (index 0)
                MLIROp.LLVMOp (LLVMOp.StructGEP (statePtrSSA, seqPtrSSA, 0, seqStructType))
                // Store next state
                MLIROp.LLVMOp (LLVMOp.Store (nextStateSSA, statePtrSSA, TInt I32, AtomicOrdering.NotAtomic))
            ]

            // Return true terminator (using llvm.return to match llvm.func)
            let returnOp = MLIROp.LLVMOp (LLVMOp.Return (Some trueLitSSA, Some MLIRTypes.i1))

            let ops = [valueOp] @ storeOps @ [returnOp]

            {
                Label = BlockRef blockName
                Args = []
                Ops = ops
            } : Block
        )

    // Done block: return false
    let doneBlock: Block = {
        Label = BlockRef "done"
        Args = []
        Ops = [MLIROp.LLVMOp (LLVMOp.Return (Some falseLitSSA, Some MLIRTypes.i1))]
    }

    // Build the function body region
    let bodyRegion: Region = {
        Blocks = [entryBlock] @ stateBlocks @ [doneBlock]
    }

    // FLAT CLOSURE PATTERN (January 2026):
    // MoveNext uses llvm.func because its address is taken via llvm.mlir.addressof
    // and stored in the Seq struct. This is the same pattern as closures and lazy thunks.
    // See fsnative-spec/spec/drafts/backend-lowering-architecture.md for rationale.
    MLIROp.LLVMOp (LLVMOp.LLVMFuncDef (
        moveNextName,
        [(seqPtrSSA, TPtr)],
        MLIRTypes.i1,
        bodyRegion,
        LLVMPrivate
    ))

/// Compute SSAs needed for MoveNext function (sequential pattern)
/// This is more complex because it depends on the number of yields
let moveNextSSACost (numYields: int) : int =
    // Entry block: seqPtrSSA, stateSSA, trueLitSSA, falseLitSSA, neg1SSA, stateValSSA = 6
    // Per state block: valueSSA, currentPtrSSA, nextStateSSA, statePtrSSA = 4 each
    6 + (numYields * 4)

