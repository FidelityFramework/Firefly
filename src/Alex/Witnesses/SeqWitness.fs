/// Seq Witness - Witness Seq<'T> values to MLIR
///
/// PRD-15: Lazy sequence expressions with state machine semantics
///
/// ARCHITECTURAL PRINCIPLES (Four Pillars):
/// 1. Coeffects: SSA assignment is pre-computed, lookup via context
/// 2. Active Patterns: Match on semantic meaning (SeqExpr, ForEach)
/// 3. Zipper: Navigate and accumulate structured ops
/// 4. Templates: Return structured MLIROp types, no sprintf
///
/// FLAT CLOSURE ARCHITECTURE (January 2026):
/// Sequences are "extended flat closures" - captures are inlined directly.
/// NO env_ptr, NO nulls - following MLKit-style flat closure principles.
///
/// SEQ STRUCT LAYOUT:
/// !seq_T = !llvm.struct<(i32, T, ptr, cap₀, cap₁, ...)>
///   - Field 0: State (i32) - 0=initial, N=after yield N, -1=done
///   - Field 1: Current value (T) - valid when MoveNext returns true
///   - Field 2: MoveNext function pointer
///   - Field 3+: Inlined captured values
///
/// MOVENEXT CALLING CONVENTION:
/// MoveNext receives a POINTER to the seq struct (for mutation):
///   func.func private @moveNext(%seq_ptr: ptr) -> i1 {
///       // Uses LLVM ops for struct access (no MLIR memref equivalent for structs)
///       %state_ptr = llvm.getelementptr %seq_ptr[0, 0] : ptr
///       %state = llvm.load %state_ptr : i32
///       // ... state machine logic (uses cf.switch for control flow) ...
///       // Store new state and current value through pointer
///       func.return %has_next : i1
///   }
///
/// FOREACH LOOP:
/// Allocates seq struct on stack, repeatedly calls MoveNext until done.
///
/// OPERATIONS:
/// - witnessSeqCreate: Build seq struct with state=0 and captures
/// - witnessForEach: Emit iteration loop calling MoveNext
module Alex.Witnesses.SeqWitness

open FSharp.Native.Compiler.PSG.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper

// Module aliases for dialect templates
module CF = Alex.Dialects.CF.Templates
module Func = Alex.Dialects.Func.Templates

// ═══════════════════════════════════════════════════════════════════════════
// SEQ STRUCT TYPE (FLAT CLOSURE MODEL)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the MLIR type for Seq<T> with captures and internal state
/// Layout: { state: i32, current: T, moveNext_ptr: ptr, cap₀, ..., capₙ, state₀, ..., stateₘ }
///
/// FIELD INDICES (PRD-15):
/// - 0: state (i32) - state machine state
/// - 1: current (T) - current yielded value
/// - 2: moveNext_ptr (ptr) - function pointer
/// - 3 to 3+numCaptures-1: captured values from enclosing scope
/// - 3+numCaptures onwards: internal mutable state fields
let seqStructTypeFull
    (elementType: MLIRType)
    (captureTypes: MLIRType list)
    (internalStateTypes: MLIRType list)
    : MLIRType =
    // {i32, T, ptr, cap₀, ..., capₙ, state₀, ..., stateₘ}
    TStruct ([TInt I32; elementType; TPtr] @ captureTypes @ internalStateTypes)

/// Build the MLIR type for Seq<T> with N captures (no internal state)
/// Layout: { state: i32, current: T, moveNext_ptr: ptr, cap₀, cap₁, ... }
let seqStructType (elementType: MLIRType) (captureTypes: MLIRType list) : MLIRType =
    seqStructTypeFull elementType captureTypes []

/// Seq struct type with no captures (simplest case)
let seqStructTypeNoCaptures (elementType: MLIRType) : MLIRType =
    // {i32, T, ptr}
    TStruct [TInt I32; elementType; TPtr]

// ═══════════════════════════════════════════════════════════════════════════
// SEQ.CREATE - Build sequence expression (FLAT CLOSURE)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness seq expression creation
///
/// Input:
///   - moveNextName: Name of the MoveNext function
///   - elementType: The type T in Seq<T>
///   - captureVals: Values captured by the sequence expression
///
/// Output:
///   - Seq struct: {state=0, current=undef, moveNext_ptr, cap₀, cap₁, ...}
///
/// SSA cost: 5 + numCaptures
///   - 1: zero constant (state=0)
///   - 1: undef struct
///   - 1: insert state
///   - 1: addressof moveNext_ptr
///   - 1: insert moveNext_ptr
///   - N: insert each capture
let witnessSeqCreate
    (appNodeId: NodeId)
    (z: PSGZipper)
    (moveNextName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let seqType = seqStructType elementType captureTypes
    let ssas = requireNodeSSAs appNodeId z

    // Pre-assigned SSAs (from SSAAssignment coeffect)
    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    // ssas.[5..] for captures

    let baseOps = [
        // Create zero constant for state
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))

        // Create undef seq struct
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, seqType))

        // Insert state=0 at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], seqType))

        // Get MoveNext function address
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))

        // Insert moveNext_ptr at index 2 (skip current at index 1)
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], seqType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps, finalSSA =
        if captureVals.IsEmpty then
            [], withCodePtrSSA
        else
            let ops, lastSSA =
                captureVals
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                    let nextSSA = ssas.[5 + i]
                    let captureIndex = 3 + i
                    let op = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [captureIndex], seqType))
                    (accOps @ [op], nextSSA)
                ) ([], withCodePtrSSA)
            ops, lastSSA

    (baseOps @ captureOps, TRValue { SSA = finalSSA; Type = seqType })

/// Witness seq expression creation with internal state fields
///
/// PRD-15: For while-based sequences with internal mutable state (let mutable inside body),
/// the seq struct must include slots for internal state fields.
///
/// Input:
///   - moveNextName: Name of the MoveNext function
///   - elementType: The type T in Seq<T>
///   - captureVals: Values captured by the sequence expression
///   - internalStateTypes: Types of internal mutable variables (let mutable inside seq body)
///
/// Output:
///   - Seq struct: {state=0, current=undef, moveNext_ptr, cap₀, cap₁, ..., state₀=0, state₁=0, ...}
///
/// IMPORTANT: Internal state fields are initialized to default values (0).
/// MoveNext state 0 (emitInit) will initialize them to actual values.
///
/// SSA cost: 5 + numCaptures + numInternalState
let witnessSeqCreateFull
    (appNodeId: NodeId)
    (z: PSGZipper)
    (moveNextName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    (internalStateTypes: MLIRType list)
    : (MLIROp list * TransferResult) =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let seqType = seqStructTypeFull elementType captureTypes internalStateTypes
    let ssas = requireNodeSSAs appNodeId z
    let numCaptures = List.length captureVals
    let numInternalState = List.length internalStateTypes

    // Pre-assigned SSAs (from SSAAssignment coeffect)
    let zeroSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withStateSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    // ssas.[5..5+numCaptures-1] for captures
    // ssas.[5+numCaptures..] for internal state defaults

    let baseOps = [
        // Create zero constant for state
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i32))

        // Create undef seq struct
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, seqType))

        // Insert state=0 at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (withStateSSA, undefSSA, zeroSSA, [0], seqType))

        // Get MoveNext function address
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc moveNextName))

        // Insert moveNext_ptr at index 2 (skip current at index 1)
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withStateSSA, codePtrSSA, [2], seqType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps, afterCapturesSSA =
        if captureVals.IsEmpty then
            [], withCodePtrSSA
        else
            let ops, lastSSA =
                captureVals
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                    let nextSSA = ssas.[5 + i]
                    let captureIndex = 3 + i
                    let op = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [captureIndex], seqType))
                    (accOps @ [op], nextSSA)
                ) ([], withCodePtrSSA)
            ops, lastSSA

    // Insert default values (0) for each internal state field at indices 3+numCaptures, ...
    // MoveNext state 0 will initialize them to actual values via emitInit()
    let internalStateOps, finalSSA =
        if internalStateTypes.IsEmpty then
            [], afterCapturesSSA
        else
            let ops, lastSSA =
                internalStateTypes
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, stateType) ->
                    // SSA for zero constant for this internal state field
                    let zeroConstSSA = ssas.[5 + numCaptures + i * 2]
                    // SSA for InsertValue result
                    let nextSSA = ssas.[5 + numCaptures + i * 2 + 1]
                    let stateIndex = 3 + numCaptures + i
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroConstSSA, 0L, stateType))
                    let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, zeroConstSSA, [stateIndex], seqType))
                    (accOps @ [zeroOp; insertOp], nextSSA)
                ) ([], afterCapturesSSA)
            ops, lastSSA

    (baseOps @ captureOps @ internalStateOps, TRValue { SSA = finalSSA; Type = seqType })

// ═══════════════════════════════════════════════════════════════════════════
// MOVENEXT STATE MACHINE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

open PSGElaboration.YieldStateIndices
open FSharp.Native.Compiler.Checking.Native.NativeTypes

/// What value to yield at a state point
/// PRD-15 SimpleSeq: supports literals and computed expressions
type YieldValueSource =
    /// Simple integer literal - constant generated in MoveNext
    | IntLiteral of value: int64
    /// Computed value - ops to emit, result SSA is computed dynamically
    /// The ops list should NOT include the SSA assignment - that's handled by the generator
    | ComputedValue of valueNodeId: NodeId

/// Information needed to generate a yield block (for sequential pattern)
type YieldBlockInfo = {
    /// State index for this yield (1, 2, 3, ...)
    StateIndex: int
    /// Source of the value to yield
    ValueSource: YieldValueSource
    /// Type of the yielded value
    ValueType: MLIRType
}

/// Information for while-based MoveNext generation
type WhileBasedMoveNextInfo = {
    /// MoveNext function name
    MoveNextName: string
    /// Seq struct type
    SeqStructType: MLIRType
    /// Element type (T in Seq<T>)
    ElementType: MLIRType
    /// While body info from YieldStateIndices
    WhileInfo: WhileBodyInfo
    /// Internal state fields (mutable vars in seq body)
    InternalState: InternalStateField list
    /// Number of captures
    NumCaptures: int
    /// Capture types (for struct layout)
    CaptureTypes: NativeType list
}

/// Generate the MoveNext state machine function
///
/// PRD-15: MoveNext is a state machine that:
/// 1. Loads current state from seq struct
/// 2. Switches to the appropriate state block
/// 3. Each state block: stores value, updates state, returns true
/// 4. Done block: returns false
///
/// DIALECT ARCHITECTURE (January 2026):
/// - Function definition: func.func (portable across MLIR backends)
/// - Control flow: cf.switch, cf.br (portable)
/// - Struct access: llvm.getelementptr, llvm.load, llvm.store (necessary - no MLIR memref for structs)
/// - Return: func.return (portable)
///
/// STRUCTURE:
/// ```mlir
/// func.func private @moveNext(%seq_ptr: ptr) -> i1 {
///   %state_ptr = llvm.getelementptr %seq_ptr[0, 0]  // LLVM for struct access
///   %state = llvm.load %state_ptr : i32
///   cf.switch %state [0: ^s0, 1: ^s1, ...], ^done   // CF for control flow
///
/// ^s0:
///   %v0 = <compute value 0>
///   llvm.store %v0, %seq_ptr[0, 1]   // store to current (LLVM for structs)
///   llvm.store 1, %seq_ptr[0, 0]     // update state
///   func.return true                  // func.return for portability
///
/// ^done:
///   func.return false
/// }
/// ```
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

// ═══════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════
// FOREACH - Iterate over sequence (using SCF while)
// ═══════════════════════════════════════════════════════════════════════════

module SCF = Alex.Dialects.SCF.Templates

/// Witness for-each loop over a sequence
///
/// PRD-15: ForEach allocates the seq on stack and calls MoveNext repeatedly.
/// Uses SCF while loop for structured control flow.
///
/// ARCHITECTURE:
/// 1. Setup (outside loop): alloca seq, store value, extract code_ptr
/// 2. Condition region: IndirectCall MoveNext(seq_ptr) -> i1, scf.condition
/// 3. Body region: Load current from (mutated) seq struct, execute bodyOps, scf.yield
///
/// The MoveNext function mutates the seq struct in place (updates state and current).
/// Since seq is on stack via alloca, all mutations persist between iterations.
///
/// SSA cost: 7 (fixed)
///   - Setup: oneSSA, allocaPtrSSA, seqLoadSSA, codePtrSSA (4)
///   - Condition region: hasNextSSA (1)
///   - Body region: seqReloadSSA, currentSSA (2)
let witnessForEach
    (nodeId: NodeId)
    (z: PSGZipper)
    (seqVal: Val)
    (elementType: MLIRType)
    (bodyOps: MLIROp list)
    : (MLIROp list * TransferResult) =

    let ssas = requireNodeSSAs nodeId z
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
