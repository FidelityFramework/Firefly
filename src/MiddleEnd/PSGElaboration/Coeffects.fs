/// Coeffects - Pre-computed analysis results consumed by witnesses
///
/// Coeffects are the outputs of PSGElaboration passes. They represent
/// information computed BEFORE emission that witnesses OBSERVE (not compute).
///
/// This follows the "mise-en-place" principle: all prep work is done
/// before emission begins. Witnesses only look up pre-computed values.
///
/// Current coeffects:
/// - NodeSSAAllocation: Which SSAs are assigned to each PSG node
/// - ClosureLayout: How closures are structured (captures, SSAs, types)
///
/// Future coeffects (when SeqMoveNext moves to FNCS):
/// - SeqMoveNextLayout: State machine structure for sequence MoveNext
module PSGElaboration.Coeffects

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA ASSIGNMENT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════

/// SSA allocation for a PSG node - supports multi-SSA expansion
/// One PSG node may expand to multiple MLIR ops, each needing an SSA.
/// SSAs are in emission order; Result is the final SSA (what gets used downstream).
type NodeSSAAllocation = {
    /// All SSAs for this node in emission order
    SSAs: SSA list
    /// The result SSA (always the last one)
    Result: SSA
}

module NodeSSAAllocation =
    let single (ssa: SSA) = { SSAs = [ssa]; Result = ssa }
    let multi (ssas: SSA list) =
        match ssas with
        | [] -> failwith "NodeSSAAllocation requires at least one SSA"
        | _ -> { SSAs = ssas; Result = List.last ssas }

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE LAYOUT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════
//
// For Lambdas with captures, we pre-compute the complete closure layout.
// This is deterministic - derived from CaptureInfo list in PSG (from FNCS).
// Witnesses observe this coeffect; they do NOT compute layout during emission.

/// How a variable is captured in a closure
type CaptureMode =
    | ByValue  // Immutable variable: copy value into closure struct
    | ByRef    // Mutable variable: store pointer to alloca in closure struct

/// Layout information for a single captured variable
type CaptureSlot = {
    /// Name of the captured variable
    Name: string
    /// Index in the closure struct (0 = code_ptr, 1+ = captures)
    SlotIndex: int
    /// MLIR type of the slot (value type for ByValue, ptr for ByRef)
    SlotType: MLIRType
    /// Source NodeId of the captured binding (for SSA lookup)
    SourceNodeId: NodeId option
    /// How the variable is captured
    Mode: CaptureMode
}

/// Complete closure layout for a Lambda with captures.
/// This coeffect tells LambdaWitness exactly how to construct and extract closures.
type ClosureLayout = {
    /// The Lambda node this layout is for
    LambdaNodeId: NodeId
    /// Ordered list of capture slots (matches closure struct field order)
    Captures: CaptureSlot list

    // ─────────────────────────────────────────────────────────────────────────
    // FLAT STRUCT CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for addressof code_ptr
    CodeAddrSSA: SSA
    /// SSA for undef closure struct
    ClosureUndefSSA: SSA
    /// SSA for insertvalue of code_ptr at [0]
    ClosureWithCodeSSA: SSA
    /// SSAs for insertvalue of each capture at [1..N] (one per capture)
    CaptureInsertSSAs: SSA list

    // ─────────────────────────────────────────────────────────────────────────
    // HEAP ALLOCATION SSAs (for escaping closures)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSAs for heap arena allocation (5 SSAs)
    HeapPosPtrSSA: SSA
    HeapPosSSA: SSA
    HeapBaseSSA: SSA
    HeapResultPtrSSA: SSA
    HeapNewPosSSA: SSA

    /// SSAs for size computation (4 SSAs)
    SizeNullPtrSSA: SSA
    SizeGepSSA: SSA
    SizeSSA: SSA
    SizeOneSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // UNIFORM PAIR CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for undef uniform pair {ptr, ptr}
    PairUndefSSA: SSA
    /// SSA for insertvalue code_ptr at [0]
    PairWithCodeSSA: SSA
    /// SSA for final closure result (insertvalue env_ptr at [1])
    ClosureResultSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // CAPTURE EXTRACTION SSAs (for callee/inner function)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for loading the closure struct from env_ptr (Arg 0) in inner function
    StructLoadSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // TYPE INFORMATION
    // ─────────────────────────────────────────────────────────────────────────
    /// MLIR type of the environment struct (kept for compatibility)
    EnvStructType: MLIRType
    /// MLIR type of the closure struct: {ptr, T0, T1, ...} = {code_ptr, captures...}
    ClosureStructType: MLIRType
    /// Lambda context: determines extraction base index and load struct type
    Context: LambdaContext
    /// For LazyThunk: the full lazy struct type {i1, T, ptr, cap0, cap1, ...}
    LazyStructType: MLIRType option
}

/// Get the struct type to load when extracting captures from this closure
let closureLoadStructType (layout: ClosureLayout) : MLIRType =
    match layout.Context with
    | LambdaContext.RegularClosure -> layout.ClosureStructType
    | LambdaContext.LazyThunk ->
        match layout.LazyStructType with
        | Some lazyType -> lazyType
        | None -> failwith "LazyThunk context requires LazyStructType"
    | LambdaContext.SeqGenerator -> layout.ClosureStructType

/// Get the base index for capture extraction based on context
/// Regular closure: captures at indices 1, 2, ... (after code_ptr at [0])
/// Lazy thunk: captures at indices 3, 4, ... (after computed[0], value[1], code_ptr[2])
let closureExtractionBaseIndex (layout: ClosureLayout) : int =
    match layout.Context with
    | LambdaContext.RegularClosure -> 1
    | LambdaContext.LazyThunk -> 3
    | LambdaContext.SeqGenerator -> 3

// ═══════════════════════════════════════════════════════════════════════════
// DU LAYOUT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════
//
// For heterogeneous DUs (like Result<'T, 'E>) that need arena allocation,
// we pre-compute the complete DU layout. This follows the flat closure model:
// build case-specific struct inline, then store to arena, return pointer.
//
// Homogeneous DUs (like Option<'T>) use inline struct representation and
// don't need a DULayout - they're handled directly by witnessDUConstruct.

/// Complete DU layout for a DUConstruct node that needs arena allocation.
/// This coeffect tells MemoryWitness exactly how to construct arena-allocated DUs.
type DULayout = {
    /// The DUConstruct node this layout is for
    DUConstructNodeId: NodeId
    /// Case name (e.g., "Ok", "Error")
    CaseName: string
    /// Case index (0 for first case, 1 for second, etc.)
    CaseIndex: int
    /// Whether this case has a payload
    HasPayload: bool

    // ─────────────────────────────────────────────────────────────────────────
    // CASE-SPECIFIC STRUCT CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for undef case struct
    StructUndefSSA: SSA
    /// SSA for tag constant
    TagConstSSA: SSA
    /// SSA for insertvalue of tag at [0]
    WithTagSSA: SSA
    /// SSA for insertvalue of payload at [1] (only used if HasPayload)
    WithPayloadSSA: SSA option

    // ─────────────────────────────────────────────────────────────────────────
    // SIZE COMPUTATION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for null pointer (GEP base for size computation)
    SizeNullPtrSSA: SSA
    /// SSA for constant 1 (GEP index)
    SizeOneSSA: SSA
    /// SSA for GEP null[1] result
    SizeGepSSA: SSA
    /// SSA for ptrtoint (size in bytes)
    SizeSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // ARENA ALLOCATION SSAs (uses closure_heap arena)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for addressof closure_pos
    HeapPosPtrSSA: SSA
    /// SSA for load current position
    HeapPosSSA: SSA
    /// SSA for addressof closure_heap
    HeapBaseSSA: SSA
    /// SSA for GEP heap_base + pos (result pointer)
    HeapResultPtrSSA: SSA
    /// SSA for pos + size (new position)
    HeapNewPosSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // TYPE INFORMATION
    // ─────────────────────────────────────────────────────────────────────────
    /// MLIR type of the case-specific struct: {i8, PayloadType}
    CaseStructType: MLIRType
    /// MLIR type of the payload (if HasPayload)
    PayloadType: MLIRType option
}
