/// SSA Assignment Pass - PSGElaboration for MLIR emission
///
/// This pass assigns SSA values to PSG nodes BEFORE MLIR emission.
/// SSA is an MLIR/LLVM concern, not F# semantics, so it lives in PSGElaboration.
///
/// Key design:
/// - SSA counter resets at each Lambda boundary (per-function scoping)
/// - Post-order traversal ensures values are assigned before uses
/// - Returns Map<NodeId, NodeSSAAllocation> that witnesses read (coeffect lookup, no generation during emission)
/// - Uses structured SSA type (V of int | Arg of int), not strings
/// - Knows MLIR expansion costs: one PSG node may need multiple SSAs
module PSGElaboration.SSAAssignment

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.UnionFind
open Alex.Dialects.Core.Types
open Alex.Bindings.PlatformTypes
open PSGElaboration.Coeffects

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURAL SSA DERIVATION
// ═══════════════════════════════════════════════════════════════════════════
//
// SSA counts are derived from actual node STRUCTURE, not just node KIND.
// Since the graph is statically resolved, we know exactly what emission will do.
// This eliminates heuristics and prevents "not enough SSAs" errors.
//
// Key insight: SSA count is a deterministic function of instance structure.

/// Get the number of SSAs needed for a literal value
let private literalExpansionCost (lit: LiteralValue) : int =
    match lit with
    | LiteralValue.String _ -> 5  // addressof, constI(len), undef, insertvalue(ptr), insertvalue(len)
    | LiteralValue.Unit -> 1     // constI
    | LiteralValue.Bool _ -> 1   // constI
    | LiteralValue.Int8 _ | LiteralValue.Int16 _ | LiteralValue.Int32 _ | LiteralValue.Int64 _ -> 1
    | LiteralValue.UInt8 _ | LiteralValue.UInt16 _ | LiteralValue.UInt32 _ | LiteralValue.UInt64 _ -> 1
    | LiteralValue.NativeInt _ | LiteralValue.UNativeInt _ -> 1
    | LiteralValue.Char _ -> 1
    | LiteralValue.Float32 _ | LiteralValue.Float64 _ -> 1
    | _ -> 1  // Default

/// Compute exact SSA count for Lambda based on captures list
/// This is DETERMINISTIC - derived directly from PSG structure (captures list from FNCS)
///
/// CLOSURE CONSTRUCTION with heap allocation:
/// - Simple Lambda (0 captures): 0 SSAs (emits func.func, no local value needed)
/// - Closing Lambda (N captures):
///   Flat struct construction (N + 3):
///     - 1 SSA: addressof for code pointer
///     - 1 SSA: undef for flat closure struct
///     - 1 SSA: insertvalue for code_ptr at [0]
///     - N SSAs: insertvalue for each capture at [1..N]
///   Heap allocation (5):
///     - 5 SSAs: posPtrSSA, posSSA, heapBaseSSA, resultPtrSSA, newPosSSA
///   Size computation (4):
///     - 4 SSAs: nullPtrSSA, gepSSA, sizeSSA, oneSSA
///   Uniform pair construction (3):
///     - 3 SSAs: pairUndefSSA, pairWithCodeSSA, closureResultSSA
///   Total: N + 15 SSAs
let private computeLambdaSSACost (captures: CaptureInfo list) : int =
    let n = List.length captures
    if n = 0 then
        0  // Simple function - no closure struct needed
    else
        n + 15  // flat struct (n+3) + heap (5) + size (4) + pair (3)

/// Minimal NativeType to MLIRType mapping for capture slots
/// This is a subset of TypeMapping.mapNativeType, inlined here to avoid
/// circular dependencies (SSAAssignment must compile before TypeMapping)
/// CRITICAL: Must check Layout + NTUKind first to match TypeMapping behavior,
/// especially for PlatformWord types like int which depend on target architecture.
let rec private mapCaptureType (arch: Architecture) (ty: NativeType) : MLIRType =
    let wordWidth = platformWordWidth arch
    match ty with
    | NativeType.TApp(tycon, args) ->
        // FIRST: Check Layout + NTUKind for platform-aware type mapping
        // This mirrors TypeMapping.mapNativeType to ensure consistency
        match tycon.Layout, tycon.NTUKind with
        // Zero-size unit type
        | TypeLayout.Inline (0, 1), Some NTUKind.NTUunit -> TInt I32
        // Boolean: 1-bit
        | TypeLayout.Inline (1, 1), Some NTUKind.NTUbool -> TInt I1
        // Fixed-width integers by NTUKind
        | _, Some NTUKind.NTUint8 -> TInt I8
        | _, Some NTUKind.NTUuint8 -> TInt I8
        | _, Some NTUKind.NTUint16 -> TInt I16
        | _, Some NTUKind.NTUuint16 -> TInt I16
        | _, Some NTUKind.NTUint32 -> TInt I32
        | _, Some NTUKind.NTUuint32 -> TInt I32
        | _, Some NTUKind.NTUint64 -> TInt I64
        | _, Some NTUKind.NTUuint64 -> TInt I64
        // Platform-word integers (int, uint, nativeint, size_t, ptrdiff_t)
        // Size depends on target architecture via platformWordWidth
        | TypeLayout.PlatformWord, Some NTUKind.NTUint
        | TypeLayout.PlatformWord, Some NTUKind.NTUuint
        | TypeLayout.PlatformWord, Some NTUKind.NTUnint
        | TypeLayout.PlatformWord, Some NTUKind.NTUunint
        | TypeLayout.PlatformWord, Some NTUKind.NTUsize
        | TypeLayout.PlatformWord, Some NTUKind.NTUdiff
        | TypeLayout.PlatformWord, None -> TInt wordWidth  // Platform word resolved per architecture
        // Pointers
        | TypeLayout.PlatformWord, Some NTUKind.NTUptr
        | TypeLayout.PlatformWord, Some NTUKind.NTUfnptr -> TPtr
        | _, Some NTUKind.NTUptr -> TPtr
        // Floats
        | _, Some NTUKind.NTUfloat32 -> TFloat F32
        | _, Some NTUKind.NTUfloat64 -> TFloat F64
        // Char (Unicode codepoint)
        | _, Some NTUKind.NTUchar -> TInt I32
        // String (fat pointer)
        | TypeLayout.FatPointer, Some NTUKind.NTUstring -> TStruct [TPtr; TInt I64]
        // SECOND: Name-based fallback for types without proper NTU metadata
        | _ ->
            match tycon.Name with
            | "Ptr" | "nativeptr" | "byref" | "inref" | "outref" -> TPtr
            | "array" -> TStruct [TPtr; TInt I64]  // Fat pointer
            | "option" | "voption" ->
                match args with
                | [innerTy] -> TStruct [TInt I1; mapCaptureType arch innerTy]
                | _ -> TPtr  // Fallback
            | _ ->
                // Records, DUs, unknown types - check if has fields or treat as pointer
                if tycon.FieldCount > 0 then
                    TPtr  // Records are passed by pointer
                else
                    TPtr  // Fallback for other cases
    | NativeType.TFun _ -> TStruct [TPtr; TPtr]  // Function = closure struct
    | NativeType.TTuple (elements, _) ->
        TStruct (elements |> List.map (mapCaptureType arch))
    | NativeType.TVar tvar ->
        // Resolve type variable using Union-Find
        match find tvar with
        | (_, Some boundTy) -> mapCaptureType arch boundTy
        | (_, None) -> TPtr  // Unbound type variable - assume pointer-sized
    | NativeType.TByref _ -> TPtr
    | _ -> TPtr  // Fallback for other cases

/// Compute the MLIR type for a capture slot based on capture mode
let private captureSlotType (arch: Architecture) (capture: CaptureInfo) : MLIRType =
    if capture.IsMutable then
        // Mutable capture: store pointer to the alloca
        TPtr
    else
        // Immutable capture: store the value directly
        mapCaptureType arch capture.Type

/// Build the environment struct type from captures
let private buildEnvStructType (arch: Architecture) (captures: CaptureInfo list) : MLIRType =
    let slotTypes = captures |> List.map (captureSlotType arch)
    TStruct slotTypes

/// Build complete ClosureLayout from Lambda captures and pre-assigned SSAs
/// This is computed once during SSAAssignment - witnesses observe the result
///
/// TRUE FLAT CLOSURE SSA layout for N captures (total N+3 SSAs):
///   ssas[0]           = addressof code_ptr
///   ssas[1]           = undef closure struct
///   ssas[2]           = insertvalue code_ptr at [0]
///   ssas[3..N+2]      = insertvalue for each capture at [1..N]
///   ssas[N+2]         = final result (last insertvalue)
///
/// NOTE: Capture EXTRACTION SSAs (used in callee) are derived at emission time
/// from capture count, not pre-allocated here. This cleanly separates:
/// - Parent scope: closure CONSTRUCTION (these SSAs)
/// - Child scope: capture EXTRACTION (v0..v(N-1), derived from PSG structure)
let private buildClosureLayout
    (arch: Architecture)
    (graph: SemanticGraph)
    (lambdaNodeId: NodeId)
    (bodyNodeId: NodeId)
    (captures: CaptureInfo list)
    (ssas: SSA list)
    (context: LambdaContext)
    : ClosureLayout =

    let n = List.length captures

    // Extract SSAs by position for closure CONSTRUCTION
    // Flat struct construction (0 to N+2)
    let codeAddrSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withCodeSSA = ssas.[2]
    let captureInsertSSAs = ssas.[3..(n + 2)]

    // Heap allocation (N+3 to N+7)
    let heapPosPtrSSA = ssas.[n + 3]
    let heapPosSSA = ssas.[n + 4]
    let heapBaseSSA = ssas.[n + 5]
    let heapResultPtrSSA = ssas.[n + 6]
    let heapNewPosSSA = ssas.[n + 7]

    // Size computation (N+8 to N+11)
    let sizeNullPtrSSA = ssas.[n + 8]
    let sizeGepSSA = ssas.[n + 9]
    let sizeSSA = ssas.[n + 10]
    let sizeOneSSA = ssas.[n + 11]

    // Uniform pair construction (N+12 to N+14)
    let pairUndefSSA = ssas.[n + 12]
    let pairWithCodeSSA = ssas.[n + 13]
    let closureResultSSA = ssas.[n + 14]

    // Build capture slots (structural info only - no SSAs for extraction)
    // Extraction SSAs are derived at emission time from SlotIndex
    let captureSlots =
        captures
        |> List.mapi (fun i capture ->
            {
                Name = capture.Name
                SlotIndex = i
                SlotType = captureSlotType arch capture
                SourceNodeId = capture.SourceNodeId
                Mode = if capture.IsMutable then ByRef else ByValue
            })

    // Build env struct type (for internal tracking, kept for compatibility)
    let envStructType = buildEnvStructType arch captures

    // TRUE FLAT CLOSURE: {code_ptr, capture_0, capture_1, ...}
    // Captures are inlined directly, not via env_ptr indirection
    // This eliminates lifetime issues - closure is returned by value with all state inline
    let captureTypes = captures |> List.map (captureSlotType arch)
    let closureStructType = TStruct (TPtr :: captureTypes)

    // PRD-14 Option B: For lazy thunks, compute the FULL lazy struct type
    // {computed: i1, value: T, code_ptr: ptr, cap0, cap1, ...}
    // T is the Lambda body's return type (the element type of Lazy<T>)
    let lazyStructType =
        match context with
        | LambdaContext.LazyThunk ->
            // Get the body's return type (T in Lazy<T>)
            match SemanticGraph.tryGetNode bodyNodeId graph with
            | Some bodyNode ->
                let elementType = mapCaptureType arch bodyNode.Type
                // Lazy struct: {i1, T, ptr, cap0, cap1, ...}
                Some (TStruct ([TInt I1; elementType; TPtr] @ captureTypes))
            | None ->
                failwithf "LazyThunk Lambda body node %d not found" (NodeId.value bodyNodeId)
        | _ -> None

    // StructLoadSSA is for the CALLEE (inner function) - not from parent scope's ssas
    // It's V(captureCount) because extraction SSAs are v0..v(N-1), body starts at v(N+1)
    let structLoadSSA = V n

    {
        LambdaNodeId = lambdaNodeId
        Captures = captureSlots
        CodeAddrSSA = codeAddrSSA
        ClosureUndefSSA = undefSSA
        ClosureWithCodeSSA = withCodeSSA
        CaptureInsertSSAs = captureInsertSSAs
        HeapPosPtrSSA = heapPosPtrSSA
        HeapPosSSA = heapPosSSA
        HeapBaseSSA = heapBaseSSA
        HeapResultPtrSSA = heapResultPtrSSA
        HeapNewPosSSA = heapNewPosSSA
        SizeNullPtrSSA = sizeNullPtrSSA
        SizeGepSSA = sizeGepSSA
        SizeSSA = sizeSSA
        SizeOneSSA = sizeOneSSA
        PairUndefSSA = pairUndefSSA
        PairWithCodeSSA = pairWithCodeSSA
        ClosureResultSSA = closureResultSSA
        StructLoadSSA = structLoadSSA
        EnvStructType = envStructType
        ClosureStructType = closureStructType
        Context = context
        LazyStructType = lazyStructType
    }

/// Calculate SSA cost for interpolated string based on parts
let private interpolatedStringCost (parts: InterpolatedPart list) : int =
    // Count string parts (each needs 5 SSAs for fat pointer construction)
    let stringPartCount =
        parts |> List.sumBy (fun p ->
            match p with
            | InterpolatedPart.StringPart _ -> 1
            | InterpolatedPart.ExprPart _ -> 0)  // Already computed, no new SSAs

    // Concatenations: each needs 10 SSAs (4 extract, 1 add, 1 alloca, 1 gep, 3 build)
    let concatCount = max 0 (List.length parts - 1)

    // Total: 5 per string part + 10 per concat
    (stringPartCount * 5) + (concatCount * 10)

/// Count tuple elements from a scrutinee type
/// Returns 1 for non-tuple types (single DU), N for N-element tuples
let private countScrutineeTupleElements (graph: SemanticGraph) (scrutineeId: NodeId) : int =
    match Map.tryFind scrutineeId graph.Nodes with
    | Some node ->
        match node.Type with
        | NativeType.TTuple (elements, _) -> List.length elements
        | _ -> 1  // Non-tuple scrutinee = single DU
    | None -> 1

/// Count tags in a pattern (for tuple patterns, count nested Union patterns)
let rec private countPatternTags (pattern: Pattern) : int =
    match pattern with
    | Pattern.Union _ -> 1
    | Pattern.Tuple elements -> elements |> List.sumBy countPatternTags
    | Pattern.Var _ | Pattern.Wildcard -> 0
    | _ -> 0

/// Compute exact SSA count for Match expression from its structure
/// This mirrors what ControlFlowWitness.witnessMatch actually emits
let private computeMatchSSACost (graph: SemanticGraph) (scrutineeId: NodeId) (cases: MatchCase list) : int =
    // Determine pattern complexity from actual cases
    let numTags =
        match cases with
        | case :: _ -> countPatternTags case.Pattern
        | [] -> 1

    let numCases = List.length cases

    // Tag extraction SSAs (mirrors lines 224-251 in ControlFlowWitness.fs)
    let extractionSSAs =
        if numTags <= 1 then
            1  // Single tag extraction
        else
            // Tuple pattern: extract each element (N) + extract each tag (N)
            numTags * 2

    // Tag comparison SSAs per case (mirrors buildTagComparison)
    // For each tag: expectedSSA + cmpSSA = 2
    // For multiple tags: add (numTags - 1) AND operations
    let comparisonSSAsPerCase =
        if numTags <= 1 then
            2  // Single tag: expected + cmp
        else
            (numTags * 2) + max 0 (numTags - 1)  // 2 per tag + ANDs

    // If-chain SSAs (mirrors buildIfChain)
    // Each non-final case: comparison SSAs + potentially if/else structure
    // Final case: just body (no comparison)
    // Plus result phi and zero constants for void cases
    let ifChainSSAs =
        let nonFinalCases = max 0 (numCases - 1)
        // Each branch may need: result accumulation + zero for void
        (nonFinalCases * comparisonSSAsPerCase) + (numCases * 2) + 5

    // Total: extraction + comparisons + if-chain + buffer
    extractionSSAs + ifChainSSAs + 10  // 10 for safety margin

/// Compute exact SSA count for Application based on intrinsic analysis
let private computeApplicationSSACost (graph: SemanticGraph) (node: SemanticNode) : int =
    // Look at what we're applying to determine SSA needs
    match node.Children with
    | funcId :: _ ->
        match Map.tryFind funcId graph.Nodes with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic info ->
                // Intrinsics have known SSA costs based on operation
                match info.Module, info.Operation with
                | IntrinsicModule.Format, "int" -> 45     // intToString
                | IntrinsicModule.Format, "int64" -> 45   // intToString (handles i64 directly)
                | IntrinsicModule.Format, "float" -> 75   // floatToString
                | IntrinsicModule.Parse, "int" -> 35      // stringToInt
                | IntrinsicModule.Parse, "float" -> 250   // stringToFloat (complex)
                | IntrinsicModule.String, "contains" -> 30 // string scanning
                | IntrinsicModule.String, "concat2" -> 15  // concatenation
                | IntrinsicModule.String, "length" -> 3    // extract
                | IntrinsicModule.Sys, _ -> 16             // syscalls (clock_gettime needs 16 for ms computation)
                | IntrinsicModule.DateTime, "now" -> 16    // delegates to clock_gettime
                | IntrinsicModule.DateTime, "utcNow" -> 16 // delegates to clock_gettime
                | IntrinsicModule.DateTime, "hour" -> 5    // const + div + rem + trunc
                | IntrinsicModule.DateTime, "minute" -> 5
                | IntrinsicModule.DateTime, "second" -> 5
                | IntrinsicModule.DateTime, "millisecond" -> 3  // just mod + trunc
                | IntrinsicModule.DateTime, "toTimeString" -> 60  // complex formatting
                | IntrinsicModule.DateTime, "toDateString" -> 60
                | IntrinsicModule.DateTime, "toString" -> 80
                | IntrinsicModule.DateTime, "toDateTimeString" -> 100  // full ISO 8601 format
                | IntrinsicModule.DateTime, _ -> 20        // other DateTime ops
                | IntrinsicModule.TimeSpan, "fromMilliseconds" -> 2
                | IntrinsicModule.TimeSpan, "fromSeconds" -> 3
                | IntrinsicModule.TimeSpan, "hours" -> 5
                | IntrinsicModule.TimeSpan, "minutes" -> 5
                | IntrinsicModule.TimeSpan, "seconds" -> 5
                | IntrinsicModule.TimeSpan, "milliseconds" -> 3
                | IntrinsicModule.TimeSpan, _ -> 10        // other TimeSpan ops
                | IntrinsicModule.Lazy, "create" -> 10     // PRD-14: undef + flag store + thunk store
                | IntrinsicModule.Lazy, "force" -> 20      // PRD-14: check + branch + cached/compute paths + phi
                | IntrinsicModule.Lazy, "isValueCreated" -> 3  // PRD-14: GEP + load flag
                | IntrinsicModule.Lazy, _ -> 10            // other Lazy ops
                // PRD-16: Seq operations - wrapper creation costs
                | IntrinsicModule.Seq, "map" -> 15         // undef + insertvalue x 5 (state, current, moveNext_ptr, inner, mapper)
                | IntrinsicModule.Seq, "filter" -> 15      // same structure as map
                | IntrinsicModule.Seq, "take" -> 15        // undef + insertvalue x 5 (state, current, moveNext_ptr, inner, remaining)
                | IntrinsicModule.Seq, "fold" -> 25        // loop setup: alloca seq + alloca acc + moveNext calls
                | IntrinsicModule.Seq, "collect" -> 20     // complex: outer + mapper + inner seq slot
                | IntrinsicModule.Seq, "iter" -> 20        // loop like fold but no accumulator
                | IntrinsicModule.Seq, "toArray" -> 30     // iteration + dynamic array building
                | IntrinsicModule.Seq, "toList" -> 30      // iteration + list cons
                | IntrinsicModule.Seq, "isEmpty" -> 10     // single MoveNext call
                | IntrinsicModule.Seq, "head" -> 12        // MoveNext + extract current
                | IntrinsicModule.Seq, "length" -> 20      // full iteration with counter
                | IntrinsicModule.Seq, _ -> 15             // default for other Seq ops
                | IntrinsicModule.Bits, "htons" | IntrinsicModule.Bits, "ntohs" -> 2  // byte swap uint16
                | IntrinsicModule.Bits, "htonl" | IntrinsicModule.Bits, "ntohl" -> 2  // byte swap uint32
                | IntrinsicModule.Bits, _ -> 1             // bitcast operations
                | IntrinsicModule.NativePtr, _ -> 5        // pointer ops
                | IntrinsicModule.Array, _ -> 10           // array ops
                | IntrinsicModule.Operators, _ -> 5        // arithmetic
                | IntrinsicModule.Convert, _ -> 3          // type conversions
                | IntrinsicModule.Math, _ -> 5             // math functions
                | _ -> 20  // Default for unknown intrinsics
            | SemanticKind.VarRef _ -> 5  // Function call
            | _ -> 10  // Other applications
        | None -> 10
    | [] -> 5

/// Compute exact SSA count for TupleExpr based on element count
let private computeTupleSSACost (childIds: NodeId list) : int =
    // undef + one insertvalue per element
    1 + List.length childIds

/// Compute exact SSA count for RecordExpr based on field count
let private computeRecordSSACost (fields: (string * NodeId) list) : int =
    // undef + one insertvalue per field
    1 + List.length fields

/// Compute exact SSA count for UnionCase based on payload presence
let private computeUnionCaseSSACost (payloadOpt: NodeId option) : int =
    match payloadOpt with
    | Some _ -> 6  // tag + undef + withTag + payload insert + conversion + result
    | None -> 3    // tag + undef + withTag (no payload)

/// Count mutable bindings in a subtree (internal state fields for seq)
/// PRD-15 THROUGH-LINE: Internal state is unique to Seq - mutable vars declared inside body
/// that persist between MoveNext calls. This is distinct from captures (read-only from enclosing scope).
let rec private countMutableBindingsInSubtree (graph: SemanticGraph) (nodeId: NodeId) : int =
    match Map.tryFind nodeId graph.Nodes with
    | None -> 0
    | Some node ->
        // Count this node if it's a mutable binding
        let thisCount =
            match node.Kind with
            | SemanticKind.Binding (_, isMutable, _, _) when isMutable -> 1
            | _ -> 0
        // Count in children
        let childCount =
            node.Children
            |> List.sumBy (fun childId -> countMutableBindingsInSubtree graph childId)
        thisCount + childCount

/// Compute SSA cost for SeqExpr based on captures and internal state
/// PRD-15: SeqExpr SSA cost = 5 base + numCaptures + (2 * numInternalState)
/// The 2 per internal state is for: const zero + InsertValue
let private computeSeqExprSSACost (graph: SemanticGraph) (bodyId: NodeId) (captures: CaptureInfo list) : int =
    let numCaptures = List.length captures
    let numInternalState = countMutableBindingsInSubtree graph bodyId
    // 5 base (zero, undef, insert state, addressof, insert code_ptr)
    // + 1 per capture (InsertValue)
    // + 2 per internal state (const zero + InsertValue)
    5 + numCaptures + (numInternalState * 2)

/// Get the number of SSAs needed for a node based on its STRUCTURE
/// This is the key function - it analyzes actual instance structure, not just kind
let private nodeExpansionCost (graph: SemanticGraph) (node: SemanticNode) : int =
    match node.Kind with
    // Structural analysis - exact counts from instance
    | SemanticKind.Match (scrutineeId, cases) ->
        computeMatchSSACost graph scrutineeId cases

    | SemanticKind.Application _ ->
        computeApplicationSSACost graph node

    | SemanticKind.TupleExpr childIds ->
        computeTupleSSACost childIds

    | SemanticKind.RecordExpr (fields, _) ->
        computeRecordSSACost fields

    | SemanticKind.UnionCase (_, _, payloadOpt) ->
        computeUnionCaseSSACost payloadOpt

    // Literal-based costs
    | SemanticKind.Literal lit -> literalExpansionCost lit
    | SemanticKind.InterpolatedString parts -> interpolatedStringCost parts

    // Lambda: cost depends on captures (structural analysis)
    | SemanticKind.Lambda (_, _, captures, _, _) ->
        computeLambdaSSACost captures

    // Fixed costs (these don't vary by structure)
    | SemanticKind.ForLoop _ -> 2
    | SemanticKind.IfThenElse _ -> 3
    | SemanticKind.Binding _ -> 3
    | SemanticKind.IndexGet _ -> 2
    | SemanticKind.IndexSet _ -> 1
    | SemanticKind.AddressOf _ -> 2
    | SemanticKind.VarRef _ -> 2
    | SemanticKind.FieldGet _ -> 1
    | SemanticKind.FieldSet _ -> 1
    | SemanticKind.Set _ -> 1  // For module-level mutable address operation
    | SemanticKind.TraitCall _ -> 1
    | SemanticKind.ArrayExpr _ -> 20
    | SemanticKind.ListExpr _ -> 20
    // PatternBinding needs SSAs for extraction + conversion
    // For tuple patterns: elemExtract + payloadExtract + convert = 3
    | SemanticKind.PatternBinding _ -> 3
    // PRD-14: Lazy values - SSA costs derived from PSG structure
    // LazyExpr: 5 base + N captures (per LazyWitness documentation)
    //   - 1: false constant, 1: undef struct, 1: insert computed
    //   - 1: addressof code_ptr, 1: insert code_ptr
    //   - N: insert each capture
    | SemanticKind.LazyExpr (_, captures) -> 5 + List.length captures
    // LazyForce: 4 fixed (per LazyWitness documentation)
    //   - 1: extract code_ptr, 1: const 1 for alloca
    //   - 1: alloca for lazy struct, 1: indirect call result
    | SemanticKind.LazyForce _ -> 4
    // PRD-15: Sequence expressions
    // SeqExpr: 5 base + captures + (2 * internal state fields)
    // Internal state = let mutable bindings inside seq body (through-line from PRD-11)
    | SemanticKind.SeqExpr (bodyId, captures) -> computeSeqExprSSACost graph bodyId captures
    // Yield: 4 (gep current + store value + gep state + store state)
    | SemanticKind.Yield _ -> 4
    // YieldBang: 12 (nested iteration setup)
    | SemanticKind.YieldBang _ -> 12
    // ForEach: 7 (setup: 4 + condition: 1 + body: 2)
    | SemanticKind.ForEach _ -> 7
    | _ -> 1

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION SCOPE STATE
// ═══════════════════════════════════════════════════════════════════════════

/// SSA assignment state for a single function scope
type private FunctionScope = {
    Counter: int
    Assignments: Map<int, NodeSSAAllocation>  // NodeId.value -> SSA allocation
}

module private FunctionScope =
    let empty = { Counter = 0; Assignments = Map.empty }

    /// Yield a single SSA
    let yieldSSA (scope: FunctionScope) : SSA * FunctionScope =
        let ssa = V scope.Counter
        ssa, { scope with Counter = scope.Counter + 1 }

    /// Yield multiple SSAs based on expansion cost
    let yieldSSAs (count: int) (scope: FunctionScope) : SSA list * FunctionScope =
        let ssas = List.init count (fun i -> V (scope.Counter + i))
        ssas, { scope with Counter = scope.Counter + count }

    /// Assign a node's SSA allocation
    let assign (nodeId: NodeId) (alloc: NodeSSAAllocation) (scope: FunctionScope) : FunctionScope =
        { scope with Assignments = Map.add (NodeId.value nodeId) alloc scope.Assignments }

/// Check if a node kind produces an SSA value
let private producesValue (kind: SemanticKind) : bool =
    match kind with
    | SemanticKind.Literal _ -> true
    | SemanticKind.Application _ -> true
    | SemanticKind.Lambda _ -> true
    | SemanticKind.Binding _ -> true
    | SemanticKind.Sequential _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.TupleExpr _ -> true
    | SemanticKind.RecordExpr _ -> true
    | SemanticKind.UnionCase _ -> true
    | SemanticKind.ArrayExpr _ -> true
    | SemanticKind.ListExpr _ -> true
    | SemanticKind.FieldGet _ -> true
    | SemanticKind.IndexGet _ -> true
    | SemanticKind.Upcast _ -> true
    | SemanticKind.Downcast _ -> true
    | SemanticKind.TypeTest _ -> true
    | SemanticKind.AddressOf _ -> true
    | SemanticKind.VarRef _ -> true
    | SemanticKind.Deref _ -> true
    | SemanticKind.TraitCall _ -> true
    | SemanticKind.Intrinsic _ -> true
    | SemanticKind.LazyExpr _ -> true
    | SemanticKind.LazyForce _ -> true
    // PRD-15: Sequence expressions
    | SemanticKind.SeqExpr _ -> true   // Produces seq struct value
    | SemanticKind.Yield _ -> true     // Needs SSAs for state machine operations
    | SemanticKind.YieldBang _ -> true // Needs SSAs for nested iteration
    | SemanticKind.PlatformBinding _ -> true
    | SemanticKind.InterpolatedString _ -> true
    // Set needs SSAs for module-level mutable address operations
    | SemanticKind.Set _ -> true
    | SemanticKind.FieldSet _ -> false
    | SemanticKind.IndexSet _ -> false
    | SemanticKind.NamedIndexedPropertySet _ -> false
    | SemanticKind.WhileLoop _ -> false
    | SemanticKind.ForLoop _ -> false
    | SemanticKind.ForEach _ -> true  // PRD-15: Needs SSAs for loop control
    | SemanticKind.TryWith _ -> false
    | SemanticKind.TryFinally _ -> false
    | SemanticKind.Quote _ -> false
    | SemanticKind.ObjectExpr _ -> false
    | SemanticKind.ModuleDef _ -> false
    | SemanticKind.TypeDef _ -> false
    | SemanticKind.MemberDef _ -> false
    | SemanticKind.TypeAnnotation _ -> true  // Passes through the inner value
    | SemanticKind.PatternBinding _ -> true  // Pattern binding introduces a variable
    | SemanticKind.Error _ -> false

/// Result of SSA assignment pass
type SSAAssignment = {
    /// Map from NodeId.value to SSA allocation (supports multi-SSA expansion)
    NodeSSA: Map<int, NodeSSAAllocation>
    /// Map from Lambda NodeId.value to its function name
    LambdaNames: Map<int, string>
    /// Set of entry point Lambda IDs
    EntryPointLambdas: Set<int>
    /// Closure layouts for Lambdas with captures (NodeId.value -> ClosureLayout)
    /// Empty for simple lambdas (no captures)
    ClosureLayouts: Map<int, ClosureLayout>
}

/// Assign SSA names to all nodes in a function body
/// Returns updated scope with assignments
let rec private assignFunctionBody
    (arch: Architecture)
    (graph: SemanticGraph)
    (closureLayouts: System.Collections.Generic.Dictionary<int, ClosureLayout>)
    (scope: FunctionScope)
    (nodeId: NodeId)
    : FunctionScope =

    match Map.tryFind nodeId graph.Nodes with
    | None -> scope
    | Some node ->
        // Architectural fix (January 2026): Filter out SeparateFunction children.
        // These are Lambda/SeqExpr body nodes - processed by their parent, not during children traversal.
        // This makes SSA assignment deterministic based on EmissionStrategy, not SemanticKind special-cases.
        let childrenToProcess =
            node.Children
            |> List.filter (fun childId ->
                match Map.tryFind childId graph.Nodes with
                | Some child ->
                    match child.EmissionStrategy with
                    | EmissionStrategy.SeparateFunction _ -> false  // Skip, parent handles it
                    | EmissionStrategy.MainPrologue -> false  // Skip, main handles it
                    | EmissionStrategy.Inline -> true
                | None -> true)

        // Post-order: process filtered children first
        let scopeAfterChildren =
            childrenToProcess |> List.fold (fun s childId -> assignFunctionBody arch graph closureLayouts s childId) scope

        // Special handling for nested Lambdas - they get their own scope
        // (but we still assign this Lambda node SSAs in parent scope for closure construction)
        match node.Kind with
        | SemanticKind.Lambda(_params, bodyId, captures, enclosingFuncOpt, context) ->
            // Process Lambda body in a NEW scope.
            // Architectural fix (January 2026): Start SSA counter AFTER capture extraction SSAs + StructLoad.
            // The body's EmissionStrategy.SeparateFunction carries the captureCount.
            // SSA layout in inner function: v0..v(N-1) = extraction, vN = struct load, v(N+1)+ = body
            // So body starts at captureCount + 1 (when there are captures) or 0 (no captures).
            let startCounter =
                match Map.tryFind bodyId graph.Nodes with
                | Some bodyNode ->
                    match bodyNode.EmissionStrategy with
                    | EmissionStrategy.SeparateFunction captureCount ->
                        if captureCount > 0 then captureCount + 1 else 0
                    | _ -> 0  // Shouldn't happen - Lambda bodies are marked SeparateFunction
                | None -> 0
            let innerStartScope = { FunctionScope.empty with Counter = startCounter }
            let _innerScope = assignFunctionBody arch graph closureLayouts innerStartScope bodyId

            // DISTINCTION: Nested NAMED functions with captures use parameter-passing, NOT closure structs.
            // Anonymous lambdas (fun x -> ...) that escape STILL need closure structs.
            // Check: Lambda is nested (enclosingFuncOpt = Some _) AND its parent is a Binding (named function).
            // Anonymous lambdas have non-Binding parents (Application, Sequential, etc.).
            let isNestedNamedFunction =
                Option.isSome enclosingFuncOpt &&
                match node.Parent with
                | Some parentId ->
                    match Map.tryFind parentId graph.Nodes with
                    | Some parentNode ->
                        match parentNode.Kind with
                        | SemanticKind.Binding _ -> true  // Named function definition
                        | _ -> false  // Anonymous lambda (value, argument, etc.)
                    | None -> false
                | None -> false

            if isNestedNamedFunction then
                // Nested function: NO closure layout, captures passed as explicit parameters
                // No SSAs needed in parent scope for closure construction
                scopeAfterChildren
            else
                // Potentially escaping closure: use closure struct model
                // Lambda itself gets SSAs in the PARENT scope for closure struct construction
                // SSA count is deterministic based on captures (from FNCS)
                let cost = computeLambdaSSACost captures
                if cost > 0 then
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    let scopeWithAlloc = FunctionScope.assign node.Id alloc scopeWithSSAs

                    // Compute ClosureLayout immediately using the allocated SSAs from the parent scope
                    // Pass the context so witnesses know how to extract captures
                    // PRD-14: Pass graph and bodyId for lazy struct type computation
                    if not (List.isEmpty captures) then
                        let layout = buildClosureLayout arch graph node.Id bodyId captures ssas context
                        if not (closureLayouts.ContainsKey(NodeId.value node.Id)) then
                            closureLayouts.Add(NodeId.value node.Id, layout)
                    scopeWithAlloc
                else
                    // Simple lambda (no captures) - no SSAs needed in parent scope
                    scopeAfterChildren
        // VarRef now gets SSAs for mutable variable loads
        // (Regular VarRefs to immutable values may not use their SSAs, but unused SSAs are harmless)

        // ForLoop needs SSAs for internal operation (ivSSA + stepSSA)
        // even though it doesn't "produce a value" in the semantic sense
        | SemanticKind.ForLoop _ ->
            let cost = nodeExpansionCost graph node  // Structural derivation
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            FunctionScope.assign node.Id alloc scopeWithSSAs

        | _ ->
            // Regular node - assign SSAs based on structural analysis
            if producesValue node.Kind then
                let cost = nodeExpansionCost graph node  // Structural derivation
                let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                let alloc = NodeSSAAllocation.multi ssas
                FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                scopeAfterChildren

/// Collect all Lambdas in the graph and assign function names
let private collectLambdas (graph: SemanticGraph) : Map<int, string> * Set<int> =
    let mutable lambdaCounter = 0
    let mutable lambdaNames = Map.empty
    let mutable entryPoints = Set.empty

    // First, identify entry point lambdas
    // Entry points may be ModuleDef nodes containing Binding nodes with isEntryPoint=true
    for entryId in graph.EntryPoints do
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding(_, _, _, isEntryPoint) when isEntryPoint ->
                // The binding's first child is typically the Lambda
                match node.Children with
                | lambdaId :: _ -> entryPoints <- Set.add (NodeId.value lambdaId) entryPoints
                | _ -> ()
            | SemanticKind.Lambda _ ->
                entryPoints <- Set.add (NodeId.value entryId) entryPoints
            | SemanticKind.ModuleDef (_, memberIds) ->
                // ModuleDef entry point - check members for entry point Binding
                for memberId in memberIds do
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding(_, _, _, isEntryPoint) when isEntryPoint ->
                            match memberNode.Children with
                            | lambdaId :: _ -> entryPoints <- Set.add (NodeId.value lambdaId) entryPoints
                            | _ -> ()
                        | _ -> ()
                    | None -> ()
            | _ -> ()
        | None -> ()

    // PRD-13: Find enclosing function by walking Parent chain
    // Structure: Lambda -> Binding("loop") -> ... -> Lambda -> Binding("factorialTail")
    let findEnclosingFunctionName (startId: NodeId) : string option =
        let rec walk (nodeId: NodeId) (passedFirstLambda: bool) =
            match Map.tryFind nodeId graph.Nodes with
            | None -> None
            | Some n ->
                match n.Kind with
                | SemanticKind.Lambda _ when passedFirstLambda ->
                    // Found enclosing Lambda - get its parent Binding's name
                    match n.Parent with
                    | Some parentId ->
                        match Map.tryFind parentId graph.Nodes with
                        | Some { Kind = SemanticKind.Binding(name, _, _, _) } -> Some name
                        | _ -> None
                    | None -> None
                | _ ->
                    match n.Parent with
                    | Some parentId -> walk parentId true
                    | None -> None
        walk startId false

    // Assign names to all Lambdas
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda _ ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                if Set.contains nodeIdVal entryPoints then
                    "main"
                else
                    // Get base name from parent Binding
                    let baseName =
                        match node.Parent with
                        | Some parentId ->
                            match Map.tryFind parentId graph.Nodes with
                            | Some { Kind = SemanticKind.Binding(bindingName, _, _, _) } -> Some bindingName
                            | _ -> None
                        | None -> None

                    match baseName with
                    | Some bname ->
                        // Check if nested by walking Parent chain
                        match findEnclosingFunctionName node.Id with
                        | Some enclosing -> sprintf "%s_%s" enclosing bname
                        | None -> bname
                    | None ->
                        let n = sprintf "lambda_%d" lambdaCounter
                        lambdaCounter <- lambdaCounter + 1
                        n

            lambdaNames <- Map.add nodeIdVal name lambdaNames

            // Also store for parent Binding's NodeId (VarRefs point to Bindings)
            match node.Parent with
            | Some parentId ->
                match Map.tryFind parentId graph.Nodes with
                | Some { Kind = SemanticKind.Binding _ } ->
                    lambdaNames <- Map.add (NodeId.value parentId) name lambdaNames
                | _ -> ()
            | None -> ()
        | _ -> ()

    lambdaNames, entryPoints

/// Check if a Binding contains a Lambda (function definition vs value binding)
let private isLambdaBinding (graph: SemanticGraph) (bindingNodeId: NodeId) : bool =
    match Map.tryFind bindingNodeId graph.Nodes with
    | Some node ->
        match node.Kind with
        | SemanticKind.Binding _ ->
            match node.Children with
            | childId :: _ ->
                match Map.tryFind childId graph.Nodes with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ -> true
                    | _ -> false
                | None -> false
            | [] -> false
        | _ -> false
    | None -> false

/// Find the main Lambda NodeId from entry points
let private findMainLambdaId (graph: SemanticGraph) (entryPoints: Set<int>) : NodeId option =
    // Entry points are either:
    // 1. ModuleDef containing Bindings (one of which is main)
    // 2. Direct Lambda node
    graph.EntryPoints
    |> List.tryPick (fun entryId ->
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Lambda _ when Set.contains (NodeId.value entryId) entryPoints ->
                Some entryId
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Find the main Lambda among module members
                memberIds |> List.tryPick (fun memberId ->
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding (_, _, _, isEntryPoint) when isEntryPoint ->
                            // Get the Lambda child of this binding
                            match memberNode.Children with
                            | lambdaId :: _ when Set.contains (NodeId.value lambdaId) entryPoints ->
                                Some lambdaId
                            | _ -> None
                        | _ -> None
                    | None -> None)
            | _ -> None
        | None -> None)

/// Find module-level VALUE bindings (non-Lambda bindings that are siblings of main)
/// These need SSAs in main's scope because they're emitted in main's prologue
let private findModuleLevelValueBindings (graph: SemanticGraph) (mainLambdaId: NodeId) : NodeId list =
    // Find the ModuleDef containing main
    graph.EntryPoints
    |> List.collect (fun entryId ->
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Get all Binding members that are NOT Lambda bindings
                // and are NOT the main binding itself
                memberIds
                |> List.filter (fun memberId ->
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding _ ->
                            // Check if this binding contains a Lambda
                            let isLambda = isLambdaBinding graph memberId
                            // Also check if this binding's child is main
                            let containsMain =
                                match memberNode.Children with
                                | childId :: _ -> childId = mainLambdaId
                                | [] -> false
                            // Include if it's a VALUE binding (not Lambda, not main)
                            not isLambda && not containsMain
                        | _ -> false
                    | None -> false)
            | _ -> []
        | None -> [])

/// Main entry point: assign SSA names to all nodes in the graph
///
/// TWO-PASS SSA ASSIGNMENT:
/// Pass 1: Module-level VALUE bindings get SSAs in main's scope (emitted in main's prologue)
/// Pass 2: Each Lambda body gets its own scope with counter starting at 0
///
/// This prevents SSA collisions between module-level bindings and function bodies.
let assignSSA (arch: Architecture) (graph: SemanticGraph) : SSAAssignment =
    let lambdaNames, entryPoints = collectLambdas graph

    let mutable allAssignments = Map.empty
    let mutableClosureLayouts = System.Collections.Generic.Dictionary<int, ClosureLayout>()

    // Find the main Lambda
    let mainLambdaIdOpt = findMainLambdaId graph entryPoints

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 1: Module-level VALUE bindings (emitted in main's prologue)
    // These share main's SSA namespace, so process them first and continue counter
    // ═══════════════════════════════════════════════════════════════════════════
    let moduleLevelValueBindings =
        match mainLambdaIdOpt with
        | Some mainId -> findModuleLevelValueBindings graph mainId
        | None -> []

    // Assign SSAs to module-level value bindings
    // These will use %v0, %v1, ... and main's body continues from there
    let moduleLevelScope =
        moduleLevelValueBindings
        |> List.fold (fun scope bindingId ->
            assignFunctionBody arch graph mutableClosureLayouts scope bindingId
        ) FunctionScope.empty

    // Track the counter after module-level bindings
    let moduleLevelCounter = moduleLevelScope.Counter

    // Merge module-level assignments
    for kvp in moduleLevelScope.Assignments do
        allAssignments <- Map.add kvp.Key kvp.Value allAssignments

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 2: Lambda bodies (each gets its own scope)
    // Non-main Lambdas start at counter 0
    // Main Lambda starts at moduleLevelCounter (continues from Pass 1)
    // ═══════════════════════════════════════════════════════════════════════════
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId, captures, _, _context) ->
            let nodeIdVal = NodeId.value node.Id
            let isMain = Set.contains nodeIdVal entryPoints &&
                         (mainLambdaIdOpt |> Option.map (fun id -> NodeId.value id = nodeIdVal) |> Option.defaultValue false)

            // Main Lambda continues from module-level counter; others start fresh
            let initialCounter = if isMain then moduleLevelCounter else 0
            let initialScope = { FunctionScope.empty with Counter = initialCounter }

            // Assign SSAs to parameter PatternBindings (Arg 0, Arg 1, etc.)
            let paramScope =
                params'
                |> List.mapi (fun i (_name, _ty, nodeId) -> i, nodeId)
                |> List.fold (fun (scope: FunctionScope) (i, nodeId) ->
                    FunctionScope.assign nodeId (NodeSSAAllocation.single (Arg i)) scope
                ) initialScope

            // Assign SSAs to body nodes
            // This will also compute ClosureLayouts for any nested lambdas found in the body
            let bodyScope = assignFunctionBody arch graph mutableClosureLayouts paramScope bodyId

            // Merge into global assignments (including parameter SSAs)
            for kvp in paramScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments

            // Note: ClosureLayout for THIS lambda itself is computed when its PARENT is visited
            // or if it is visited as part of another lambda's body.
            // If this lambda is a top-level binding value, it might be visited in Pass 1?
            // Or if it's main, it's never "visited" as a child?
            // Main doesn't have captures, so no layout needed.
        | _ -> ()

    // Convert mutable dictionary to immutable map
    let closureLayouts = 
        mutableClosureLayouts
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
        |> Map.ofSeq

    {
        NodeSSA = allAssignments
        LambdaNames = lambdaNames
        EntryPointLambdas = entryPoints
        ClosureLayouts = closureLayouts
    }

/// Look up the full SSA allocation for a node (coeffect lookup)
let lookupAllocation (nodeId: NodeId) (assignment: SSAAssignment) : NodeSSAAllocation option =
    Map.tryFind (NodeId.value nodeId) assignment.NodeSSA

/// Look up just the result SSA for a node (most common use case)
let lookupSSA (nodeId: NodeId) (assignment: SSAAssignment) : SSA option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.Result)

/// Look up all SSAs for a node (for witnesses that need intermediates)
let lookupSSAs (nodeId: NodeId) (assignment: SSAAssignment) : SSA list option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.SSAs)

/// Look up the function name for a Lambda
let lookupLambdaName (nodeId: NodeId) (assignment: SSAAssignment) : string option =
    Map.tryFind (NodeId.value nodeId) assignment.LambdaNames

/// Check if a Lambda is an entry point
let isEntryPoint (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Set.contains (NodeId.value nodeId) assignment.EntryPointLambdas

/// Look up ClosureLayout for a Lambda with captures (coeffect lookup)
/// Returns None for simple lambdas (no captures)
let lookupClosureLayout (nodeId: NodeId) (assignment: SSAAssignment) : ClosureLayout option =
    Map.tryFind (NodeId.value nodeId) assignment.ClosureLayouts

/// Check if a Lambda has captures (is a closure)
let hasClosure (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Map.containsKey (NodeId.value nodeId) assignment.ClosureLayouts

/// PRD-14/PRD-15: Get the actual return type for a function that may return a lazy or seq with captures.
/// If the function body is a LazyExpr with captures, returns the actual lazy struct type
/// including the inlined captures: {i1, T, ptr, cap0, cap1, ...}
/// If the function body is a SeqExpr with captures, returns the actual seq struct type
/// including the inlined captures: {i32, T, ptr, cap0, cap1, ...}
/// Returns None if the function doesn't return a lazy/seq with captures.
let getActualFunctionReturnType (arch: Architecture) (graph: SemanticGraph) (defId: NodeId) (assignment: SSAAssignment) : MLIRType option =
    // defId may be a Binding node - need to find the Lambda child
    let lambdaNode =
        match Map.tryFind defId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Lambda _ -> Some node
            | SemanticKind.Binding _ ->
                // Binding's first child is typically the Lambda
                match node.Children with
                | childId :: _ ->
                    match Map.tryFind childId graph.Nodes with
                    | Some childNode ->
                        match childNode.Kind with
                        | SemanticKind.Lambda _ -> Some childNode
                        | _ -> None
                    | None -> None
                | [] -> None
            | _ -> None
        | None -> None

    match lambdaNode with
    | None -> None
    | Some lambda ->
        match lambda.Kind with
        | SemanticKind.Lambda (_, bodyId, _, _, _) ->
            // Check if body is a LazyExpr
            match Map.tryFind bodyId graph.Nodes with
            | Some bodyNode ->
                match bodyNode.Kind with
                | SemanticKind.LazyExpr (_, captures) when not (List.isEmpty captures) ->
                    // Function returns a lazy with captures
                    // Compute the actual lazy struct type: {i1, T, ptr, cap0, cap1, ...}
                    // Get element type from the LazyExpr's type
                    let elemMlir =
                        match bodyNode.Type with
                        | NativeType.TLazy elemType -> mapCaptureType arch elemType
                        | _ -> TInt I64  // Fallback

                    // Compute capture types using the same logic as closure construction
                    let captureTypes = captures |> List.map (captureSlotType arch)

                    // Build the actual lazy struct type with captures inlined
                    let actualLazyType = TStruct (TInt I1 :: elemMlir :: TPtr :: captureTypes)
                    Some actualLazyType

                // PRD-15: Sequence expressions with captures and/or internal state
                | SemanticKind.SeqExpr (seqBodyId, captures) ->
                    // Function returns a seq - check if it has captures OR internal state
                    let numInternalState = countMutableBindingsInSubtree graph seqBodyId
                    if List.isEmpty captures && numInternalState = 0 then
                        None  // No captures, no internal state - use simple type
                    else
                        // Compute the actual seq struct type: {i32, T, ptr, cap0, ..., state0, ...}
                        // where i32 is state (vs i1 computed flag for lazy)
                        let elemMlir =
                            match bodyNode.Type with
                            | NativeType.TSeq elemType -> mapCaptureType arch elemType
                            | _ -> TInt I64  // Fallback

                        // Compute capture types using the same logic as closure construction
                        let captureTypes = captures |> List.map (captureSlotType arch)

                        // PRD-15 THROUGH-LINE: Internal state fields are also part of struct
                        // They're initialized to default (0), MoveNext state 0 sets actual values
                        // For now, assume all internal state is i64 (platform word size)
                        // A more precise approach would traverse the body to get actual types
                        let internalStateTypes = List.replicate numInternalState (TInt I64)

                        // Build the actual seq struct type with captures + internal state inlined
                        // Layout: {state: i32, current: T, code_ptr: ptr, cap0..., state0...}
                        let actualSeqType = TStruct (TInt I32 :: elemMlir :: TPtr :: captureTypes @ internalStateTypes)
                        Some actualSeqType

                | _ -> None
            | None -> None
        | _ -> None
