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

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.NativeTypedTree.UnionFind
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
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
let private literalExpansionCost (lit: NativeLiteral) : int =
    match lit with
    | NativeLiteral.String _ -> 1  // TypeLayout.Opaque: just memref.get_global
    | NativeLiteral.Unit -> 1     // constI
    | NativeLiteral.Bool _ -> 1   // constI
    | NativeLiteral.Int _ -> 1    // All integer types (int8..int64, uint8..uint64, nativeint)
    | NativeLiteral.UInt _ -> 1   // Unsigned integers > int64.MaxValue
    | NativeLiteral.Char _ -> 1
    | NativeLiteral.Float _ -> 1  // float32 and float64
    | NativeLiteral.Decimal _ -> 1
    | NativeLiteral.ByteArray _ -> 1
    | NativeLiteral.UInt16Array _ -> 1
    | NativeLiteral.BigInt _ -> 1

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
///   Size computation (3):
///     - 3 SSAs: gepSSA, sizeSSA, oneSSA (size computed at compile time, no null GEP trick)
///   Uniform pair construction (3):
///     - 3 SSAs: pairUndefSSA, pairWithCodeSSA, closureResultSSA
///   Total: N + 14 SSAs
let private computeLambdaSSACost (captures: CaptureInfo list) : int =
    let n = List.length captures
    if n = 0 then
        0  // Simple function - no closure struct needed
    else
        n + 14  // flat struct (n+3) + heap (5) + size (3) + pair (3)

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
        | TypeLayout.PlatformWord, Some NTUKind.NTUfnptr -> TIndex
        | _, Some NTUKind.NTUptr -> TIndex
        // Floats
        | _, Some NTUKind.NTUfloat32 -> TFloat F32
        | _, Some NTUKind.NTUfloat64 -> TFloat F64
        // Char (Unicode codepoint)
        | _, Some NTUKind.NTUchar -> TInt I32
        // String (fat pointer)
        | TypeLayout.FatPointer, Some NTUKind.NTUstring ->
            let totalBytes = mlirTypeSize TIndex + mlirTypeSize (TInt I64)
            TMemRefStatic(totalBytes, TInt I8)
        // SECOND: Name-based fallback for types without proper NTU metadata
        | _ ->
            match tycon.Name with
            | "Ptr" | "nativeptr" | "byref" | "inref" | "outref" -> TIndex
            | "array" ->
                let totalBytes = mlirTypeSize TIndex + mlirTypeSize (TInt I64)
                TMemRefStatic(totalBytes, TInt I8)  // Fat pointer
            | "option" | "voption" ->
                match args with
                | [innerTy] ->
                    let innerMlir = mapCaptureType arch innerTy
                    let totalBytes = 1 + mlirTypeSize innerMlir
                    TMemRefStatic(totalBytes, TInt I8)
                | _ -> TIndex  // Fallback
            | _ ->
                // Records, DUs, unknown types - check if has fields or treat as pointer
                if tycon.FieldCount > 0 then
                    TIndex  // Records are passed by pointer
                else
                    TIndex  // Fallback for other cases
    | NativeType.TFun _ ->
        let totalBytes = mlirTypeSize TIndex + mlirTypeSize TIndex
        TMemRefStatic(totalBytes, TInt I8)  // Function = closure struct
    | NativeType.TTuple (elements, _) ->
        let elementTypes = elements |> List.map (mapCaptureType arch)
        let totalBytes = elementTypes |> List.sumBy mlirTypeSize
        TMemRefStatic(totalBytes, TInt I8)
    | NativeType.TVar tvar ->
        // Resolve type variable using Union-Find
        match find tvar with
        | (_, Some boundTy) -> mapCaptureType arch boundTy
        | (_, None) -> TIndex  // Unbound type variable - assume pointer-sized
    | NativeType.TByref _ -> TIndex
    | _ -> TIndex  // Fallback for other cases

/// Compute the MLIR type for a capture slot based on capture mode
let private captureSlotType (arch: Architecture) (capture: CaptureInfo) : MLIRType =
    if capture.IsMutable then
        // Mutable capture: store pointer to the alloca
        TIndex
    else
        // Immutable capture: store the value directly
        mapCaptureType arch capture.Type

/// Build the environment struct type from captures
let private buildEnvStructType (arch: Architecture) (captures: CaptureInfo list) : MLIRType =
    let slotTypes = captures |> List.map (captureSlotType arch)
    let totalBytes = slotTypes |> List.sumBy mlirTypeSize
    TMemRefStatic(totalBytes, TInt I8)

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

    // Size computation (N+8 to N+10) - no null GEP trick, size computed at compile time
    let sizeGepSSA = ssas.[n + 8]
    let sizeSSA = ssas.[n + 9]
    let sizeOneSSA = ssas.[n + 10]

    // Uniform pair construction (N+11 to N+13)
    let pairUndefSSA = ssas.[n + 11]
    let pairWithCodeSSA = ssas.[n + 12]
    let closureResultSSA = ssas.[n + 13]

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
    let fieldTypes = TIndex :: captureTypes
    let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
    let closureStructType = TMemRefStatic(totalBytes, TInt I8)

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
                let fieldTypes = [TInt I1; elementType; TIndex] @ captureTypes
                let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
                Some (TMemRefStatic(totalBytes, TInt I8))
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

/// Build complete DULayout from DUConstruct node and pre-assigned SSAs
/// This follows the flat closure model: build case-specific struct, store to arena, return pointer
///
/// SSA layout for DU with payload (13 SSAs total):
///   ssas[0]           = undef case struct
///   ssas[1]           = tag constant
///   ssas[2]           = insertvalue tag at [0]
///   ssas[3]           = insertvalue payload at [1]
///   ssas[4..6]        = size computation (const1, gep, size from compile-time)
///   ssas[7..11]       = arena allocation (posPtrSSA, posSSA, baseSSA, resultPtrSSA, newPosSSA)
///
/// SSA layout for nullary DU (11 SSAs total):
///   ssas[0]           = undef case struct
///   ssas[1]           = tag constant
///   ssas[2]           = insertvalue tag at [0]
///   ssas[3..5]        = size computation
///   ssas[6..10]       = arena allocation
let private buildDULayout
    (arch: Architecture)
    (graph: SemanticGraph)
    (duConstructNodeId: NodeId)
    (caseName: string)
    (caseIndex: int)
    (payloadOpt: NodeId option)
    (ssas: SSA list)
    : DULayout =

    let hasPayload = Option.isSome payloadOpt

    // Extract SSAs by position
    // Struct construction
    let structUndefSSA = ssas.[0]
    let tagConstSSA = ssas.[1]
    let withTagSSA = ssas.[2]
    let withPayloadSSA, sizeOffset =
        if hasPayload then
            Some ssas.[3], 4
        else
            None, 3

    // Size computation (3 SSAs - no null GEP trick, size computed at compile time)
    let sizeOneSSA = ssas.[sizeOffset]
    let sizeGepSSA = ssas.[sizeOffset + 1]
    let sizeSSA = ssas.[sizeOffset + 2]

    // Arena allocation (5 SSAs)
    let arenaOffset = sizeOffset + 3
    let heapPosPtrSSA = ssas.[arenaOffset]
    let heapPosSSA = ssas.[arenaOffset + 1]
    let heapBaseSSA = ssas.[arenaOffset + 2]
    let heapResultPtrSSA = ssas.[arenaOffset + 3]
    let heapNewPosSSA = ssas.[arenaOffset + 4]

    // Get payload type from payload node if present
    let payloadType =
        payloadOpt
        |> Option.bind (fun payloadId -> Map.tryFind payloadId graph.Nodes)
        |> Option.map (fun node -> mapCaptureType arch node.Type)

    // Build case-specific struct type: {i8, PayloadType} or {i8} for nullary
    let caseStructType =
        match payloadType with
        | Some pType ->
            let totalBytes = 1 + mlirTypeSize pType
            TMemRefStatic(totalBytes, TInt I8)
        | None ->
            TMemRefStatic(1, TInt I8)

    {
        DUConstructNodeId = duConstructNodeId
        CaseName = caseName
        CaseIndex = caseIndex
        HasPayload = hasPayload
        StructUndefSSA = structUndefSSA
        TagConstSSA = tagConstSSA
        WithTagSSA = withTagSSA
        WithPayloadSSA = withPayloadSSA
        SizeOneSSA = sizeOneSSA
        SizeGepSSA = sizeGepSSA
        SizeSSA = sizeSSA
        HeapPosPtrSSA = heapPosPtrSSA
        HeapPosSSA = heapPosSSA
        HeapBaseSSA = heapBaseSSA
        HeapResultPtrSSA = heapResultPtrSSA
        HeapNewPosSSA = heapNewPosSSA
        CaseStructType = caseStructType
        PayloadType = payloadType
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
let private computeApplicationSSACost (graph: SemanticGraph) (node: SemanticNode) (saturatedCallArgCounts: Map<NodeId, int>) : int =
    // Check if this is a saturated call (curry flattening) — use effective arg count
    match Map.tryFind node.Id saturatedCallArgCounts with
    | Some effectiveArgCount ->
        1 + effectiveArgCount  // 1 result + N potential memref casts
    | None ->
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
                | IntrinsicModule.String, "concat2" -> 18  // concatenation (18 SSAs - pure index arithmetic, NO i64 round-trip)
                | IntrinsicModule.Sys, "write" -> 6        // FFI extraction + length (2 ptr + 3 len + 1 result)
                | IntrinsicModule.Sys, "read" -> 6         // FFI extraction + capacity (2 ptr + 3 cap + 1 result)
                | IntrinsicModule.Sys, _ -> 16             // other syscalls (clock_gettime needs 16 for ms computation)
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
                
                // PRD-13a: List operations
                | IntrinsicModule.List, "empty" -> 1           // flat closure (Undef)
                | IntrinsicModule.List, "isEmpty" -> 2         // Baker decomposes to structural check
                | IntrinsicModule.List, "head" -> 2            // GEP + load
                | IntrinsicModule.List, "tail" -> 2            // GEP + load
                | IntrinsicModule.List, "cons" -> 4            // const + alloca + 2 stores
                | IntrinsicModule.List, "length" -> 15         // loop with counter
                | IntrinsicModule.List, "map" -> 20            // recursive structure
                | IntrinsicModule.List, "filter" -> 20         // recursive structure
                | IntrinsicModule.List, "fold" -> 20           // recursive structure
                | IntrinsicModule.List, "rev" -> 15            // iterative reverse
                | IntrinsicModule.List, "append" -> 20         // copy + link
                | IntrinsicModule.List, _ -> 15                // default for other List ops
                
                // PRD-13a: Map operations
                | IntrinsicModule.Map, "empty" -> 1            // flat closure (Undef)
                | IntrinsicModule.Map, "isEmpty" -> 2          // Baker decomposes to structural check
                | IntrinsicModule.Map, "key" -> 2              // offset constant + load
                | IntrinsicModule.Map, "value" -> 2            // offset constant + load
                | IntrinsicModule.Map, "left" -> 2             // offset constant + load
                | IntrinsicModule.Map, "right" -> 2            // offset constant + load
                | IntrinsicModule.Map, "height" -> 2           // offset constant + load
                | IntrinsicModule.Map, "tryFind" -> 20         // tree traversal
                | IntrinsicModule.Map, "find" -> 18            // tree traversal (may fail)
                | IntrinsicModule.Map, "add" -> 25             // tree traversal + rebalance
                | IntrinsicModule.Map, "remove" -> 25          // tree traversal + rebalance
                | IntrinsicModule.Map, "containsKey" -> 15     // tree traversal
                | IntrinsicModule.Map, "count" -> 20           // full traversal
                | IntrinsicModule.Map, "keys" -> 30            // in-order traversal
                | IntrinsicModule.Map, "values" -> 30          // in-order traversal
                | IntrinsicModule.Map, "toList" -> 30          // in-order traversal
                | IntrinsicModule.Map, "ofList" -> 40          // build tree from list
                | IntrinsicModule.Map, _ -> 20                 // default for other Map ops
                
                // PRD-13a: Set operations
                | IntrinsicModule.Set, "empty" -> 1            // flat closure (Undef)
                | IntrinsicModule.Set, "isEmpty" -> 2          // Baker decomposes to structural check
                | IntrinsicModule.Set, "value" -> 2            // offset constant + load
                | IntrinsicModule.Set, "left" -> 2             // offset constant + load
                | IntrinsicModule.Set, "right" -> 2            // offset constant + load
                | IntrinsicModule.Set, "height" -> 2           // offset constant + load
                | IntrinsicModule.Set, "contains" -> 15        // tree traversal
                | IntrinsicModule.Set, "add" -> 20             // tree traversal + rebalance
                | IntrinsicModule.Set, "remove" -> 20          // tree traversal + rebalance
                | IntrinsicModule.Set, "count" -> 20           // full traversal
                | IntrinsicModule.Set, "union" -> 30           // tree merge
                | IntrinsicModule.Set, "intersect" -> 30       // tree filter
                | IntrinsicModule.Set, "difference" -> 30      // tree filter
                | IntrinsicModule.Set, "toList" -> 25          // in-order traversal
                | IntrinsicModule.Set, "ofList" -> 35          // build tree from list
                | IntrinsicModule.Set, _ -> 20                 // default for other Set ops
                
                // PRD-13a: Option operations
                | IntrinsicModule.Option, "isSome" -> 3        // extract + const + icmp
                | IntrinsicModule.Option, "isNone" -> 3        // extract + const + icmp
                | IntrinsicModule.Option, "get" -> 1           // extract value
                | IntrinsicModule.Option, "defaultValue" -> 5  // extract + icmp + select
                | IntrinsicModule.Option, "defaultWith" -> 10  // conditional + closure call
                | IntrinsicModule.Option, "map" -> 15          // conditional + closure call
                | IntrinsicModule.Option, "bind" -> 20         // conditional + closure call + option handling
                | IntrinsicModule.Option, "toList" -> 8        // conditional + list cons
                | IntrinsicModule.Option, _ -> 10              // default for other Option ops
                
                | IntrinsicModule.Bits, "htons" | IntrinsicModule.Bits, "ntohs" -> 2  // byte swap uint16
                | IntrinsicModule.Bits, "htonl" | IntrinsicModule.Bits, "ntohl" -> 2  // byte swap uint32
                | IntrinsicModule.Bits, _ -> 1             // bitcast operations

                // MemRef operations (MLIR memref semantics)
                // Baker has transformed NativePtr → MemRef, these are the target operations
                | IntrinsicModule.MemRef, "alloca" -> 1  // result memref only
                | IntrinsicModule.MemRef, "load" -> 1    // load result (index already from Baker)
                | IntrinsicModule.MemRef, "store" -> 1   // returns unit, but witness requires SSA allocation
                | IntrinsicModule.MemRef, "add" -> 1     // marker: returns offset/index for memref operations
                | IntrinsicModule.MemRef, "copy" -> 1    // memcpy returns void* (result pointer)
                | IntrinsicModule.MemRef, _ -> 1         // safe default
                | IntrinsicModule.Array, _ -> 10           // array ops
                | IntrinsicModule.Operators, _ -> 5        // arithmetic
                | IntrinsicModule.Convert, _ -> 3          // type conversions
                | IntrinsicModule.Math, _ -> 5             // math functions
                | _ -> 20  // Default for unknown intrinsics
            | SemanticKind.VarRef _ ->
                // Function call: 1 result + N potential type compatibility casts (static→dynamic memref)
                // Argument count = Children.Length - 1 (first child is function)
                let argCount = max 0 (node.Children.Length - 1)
                1 + argCount  // 1 for result, argCount for potential casts
            | _ -> 10  // Other applications
        | None -> 10
    | [] -> 5

/// Compute exact SSA count for TupleExpr based on element count
let private computeTupleSSACost (childIds: NodeId list) : int =
    // undef + (offset constant + insertvalue result) per element
    // Each insertvalue needs 2 SSAs in memref semantics
    1 + 2 * List.length childIds

/// Compute exact SSA count for RecordExpr based on field count
let private computeRecordSSACost (fields: (string * NodeId) list) : int =
    // undef + (offset constant + insertvalue result) per field
    // Each insertvalue needs 2 SSAs in memref semantics
    1 + 2 * List.length fields

/// Compute exact SSA count for UnionCase based on payload presence
let private computeUnionCaseSSACost (payloadOpt: NodeId option) : int =
    match payloadOpt with
    | Some _ -> 6  // tag + undef + withTag + payload insert + conversion + result
    | None -> 3    // tag + undef + withTag (no payload)

/// Compute the DU slot type from a DU's NativeType
/// Must match the logic in TypeMapping.fs for consistency
let private getDUSlotType (arch: Architecture) (duType: NativeType) : MLIRType option =
    match duType with
    | NativeType.TApp (tycon, args) ->
        match tycon.Name, tycon.Layout with
        // Option/ValueOption: slot type = payload type (homogeneous)
        | "option", _ | "voption", _ ->
            match args with
            | [innerTy] -> Some (mapCaptureType arch innerTy)
            | _ -> None
        // Result: slot type = max of Ok and Error types (heterogeneous)
        | "result", _ ->
            match args with
            | [okTy; errorTy] ->
                let okMlir = mapCaptureType arch okTy
                let errorMlir = mapCaptureType arch errorTy
                // Pick the larger type (same logic as TypeMapping.maxMLIRType)
                let okSize = Alex.CodeGeneration.TypeMapping.mlirTypeSize okMlir
                let errorSize = Alex.CodeGeneration.TypeMapping.mlirTypeSize errorMlir
                Some (if okSize >= errorSize then okMlir else errorMlir)
            | _ -> None
        // Other DUs with known layout
        | _, TypeLayout.Inline (size, align) when size > 8 ->
            let tagSize = size % align
            let payloadSize = size - tagSize
            let slotType =
                match payloadSize with
                | 1 -> TInt I8
                | 2 -> TInt I16
                | 4 -> TInt I32
                | 8 -> TInt I64
                | n -> TMemRefStatic(n, TInt I8)
            Some slotType
        | _ -> None
    | _ -> None

/// Check if a DU type needs arena allocation
/// Heterogeneous DUs (like Result<'T, 'E>) need arena; homogeneous DUs (like Option<'T>) use inline struct
let private needsDUArenaAllocation (duType: NativeType) : bool =
    match duType with
    | NativeType.TApp (tycon, _) ->
        match tycon.Name with
        | "result" -> true   // Result is heterogeneous, needs arena
        | "option" | "voption" -> false  // Option is homogeneous, inline struct
        | _ -> false  // Default to inline for other DUs
    | _ -> false

/// Compute exact SSA count for DUConstruct based on actual types
///
/// DU construction uses pDUCase pattern which needs: 4 + 2 * payload.Length
/// - 4 base SSAs: undef, tag const, tag offset, tag result (insertvalue for tag)
/// - 2 SSAs per payload field: offset constant + insertvalue result
///
/// Examples:
/// - Option None: 4 + 2*0 = 4 SSAs
/// - Option Some(x): 4 + 2*1 = 6 SSAs
/// - Result Ok(x): 4 + 2*1 = 6 SSAs
/// - Result Error(e): 4 + 2*1 = 6 SSAs
let private computeDUConstructSSACost (_arch: Architecture) (_graph: SemanticGraph) (_duType: NativeType) (payloadOpt: NodeId option) : int =
    // All DUs use the same pattern: 4 base + 2 per payload
    // Arena allocation (if needed) is handled by DULayout coeffect, not SSA count
    let payloadCount = if Option.isSome payloadOpt then 1 else 0
    4 + 2 * payloadCount

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
let private nodeExpansionCost (arch: Architecture) (graph: SemanticGraph) (node: SemanticNode) (saturatedCallArgCounts: Map<NodeId, int>) : int =
    match node.Kind with
    // Structural analysis - exact counts from instance
    | SemanticKind.Match (scrutineeId, cases) ->
        computeMatchSSACost graph scrutineeId cases

    | SemanticKind.Application _ ->
        computeApplicationSSACost graph node saturatedCallArgCounts

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
    | SemanticKind.FieldGet _ -> 3  // Max: extract + optional intermediate + cast (for string fields)
    | SemanticKind.FieldSet _ -> 2  // Offset constant + store
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
    // DU Operations (January 2026)
    // DUGetTag: For pointer-based DUs, need load + extractvalue (2 SSAs)
    //           For inline DUs, just extractvalue (1 SSA)
    | SemanticKind.DUGetTag (_, duType) ->
        if needsDUArenaAllocation duType then 2 else 1
    // DUEliminate: For pointer-based DUs, need load + extractvalue + potential bitcast (2-3 SSAs)
    //              For inline DUs, extractvalue + potential bitcast (1-2 SSAs)
    | SemanticKind.DUEliminate (_, _, _, _) ->
        // Check the DU type from the scrutinee - but we don't have easy access here
        // For now, use 3 SSAs for all cases (covers both paths with bitcast)
        // TODO: Could optimize by checking scrutinee's type
        3
    // DUConstruct: SSA count depends on whether payload type matches slot type
    // Computed deterministically from PSG structure
    | SemanticKind.DUConstruct (_, _, payloadOpt, _) ->
        computeDUConstructSSACost arch graph node.Type payloadOpt
    // Standalone Intrinsic nodes (not applied via Application)
    // These appear in entry point elaboration and other structural expansions
    | SemanticKind.Intrinsic info ->
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "emptyStringArray" -> 5  // zeroPtr + const0 + undef + insertvalue*2
        | IntrinsicModule.Sys, "exit" -> 16             // syscall with full register setup
        | _ -> 1  // Default for standalone intrinsics
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
    | SemanticKind.TupleGet _ -> true
    | SemanticKind.RecordExpr _ -> true
    | SemanticKind.UnionCase _ -> true
    // DU Operations (January 2026)
    | SemanticKind.DUGetTag _ -> true
    | SemanticKind.DUEliminate _ -> true
    | SemanticKind.DUConstruct _ -> true
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
    /// DU layouts for DUConstruct nodes needing arena allocation (NodeId.value -> DULayout)
    /// Empty for homogeneous DUs like Option that use inline struct
    DULayouts: Map<int, DULayout>
}

/// Assign SSA names to all nodes in a function body
/// Returns updated scope with assignments
/// innerScopeAssignments: mutable collection for nested lambda body SSAs (separate MLIR functions)
let rec private assignFunctionBody
    (arch: Architecture)
    (graph: SemanticGraph)
    (closureLayouts: System.Collections.Generic.Dictionary<int, ClosureLayout>)
    (duLayouts: System.Collections.Generic.Dictionary<int, DULayout>)
    (innerScopeAssignments: System.Collections.Generic.Dictionary<int, NodeSSAAllocation>)
    (saturatedCallArgCounts: Map<NodeId, int>)
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
            childrenToProcess |> List.fold (fun s childId -> assignFunctionBody arch graph closureLayouts duLayouts innerScopeAssignments saturatedCallArgCounts s childId) scope

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

            // Assign Arg SSAs for nested lambda parameters (mirrors top-level handling in assignSSA)
            let paramScope =
                _params
                |> List.mapi (fun i (_name, _ty, paramNodeId) -> i, paramNodeId)
                |> List.fold (fun (s: FunctionScope) (i, paramNodeId) ->
                    FunctionScope.assign paramNodeId (NodeSSAAllocation.single (Arg i)) s
                ) innerStartScope

            let innerScope = assignFunctionBody arch graph closureLayouts duLayouts innerScopeAssignments saturatedCallArgCounts paramScope bodyId

            // Merge nested lambda's parameter and body SSAs into the shared collection.
            // These are a separate MLIR function's namespace, collected for the global SSA map.
            for kvp in paramScope.Assignments do
                if not (innerScopeAssignments.ContainsKey(kvp.Key)) then
                    innerScopeAssignments.Add(kvp.Key, kvp.Value)
            for kvp in innerScope.Assignments do
                if not (innerScopeAssignments.ContainsKey(kvp.Key)) then
                    innerScopeAssignments.Add(kvp.Key, kvp.Value)

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
            let cost = nodeExpansionCost arch graph node saturatedCallArgCounts  // Structural derivation
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            FunctionScope.assign node.Id alloc scopeWithSSAs

        // DUConstruct: Build DULayout coeffect for arena-allocated heterogeneous DUs
        | SemanticKind.DUConstruct (caseName, caseIndex, payloadOpt, _guardExpr) ->
            let cost = nodeExpansionCost arch graph node saturatedCallArgCounts
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            let scopeWithAlloc = FunctionScope.assign node.Id alloc scopeWithSSAs

            // Build DULayout for heterogeneous DUs needing arena allocation
            if needsDUArenaAllocation node.Type then
                let layout = buildDULayout arch graph node.Id caseName caseIndex payloadOpt ssas
                if not (duLayouts.ContainsKey(NodeId.value node.Id)) then
                    duLayouts.Add(NodeId.value node.Id, layout)

            scopeWithAlloc

        // ─────────────────────────────────────────────────────────────────────
        // PATTERN BINDING SSA ALIASING (January 2026)
        // Record pattern bindings: `{ Age = a }` creates PatternBinding with FieldGet child.
        // The PatternBinding is just a NAME for the FieldGet result - it should ALIAS
        // the child's SSA, not allocate new SSAs.
        // Lambda parameter PatternBindings: have no children, Lambda assigns them as Arg.
        // ─────────────────────────────────────────────────────────────────────
        | SemanticKind.PatternBinding _ ->
            if not (List.isEmpty node.Children) then
                // Record pattern binding - alias the first child's SSA (the FieldGet)
                let childId = List.head node.Children
                match Map.tryFind (NodeId.value childId) scopeAfterChildren.Assignments with
                | Some childSSA ->
                    // Alias: PatternBinding gets the same SSA as its FieldGet child
                    FunctionScope.assign node.Id childSSA scopeAfterChildren
                | None ->
                    // Child not in assignments - shouldn't happen with post-order
                    // Fall back to normal allocation
                    let cost = nodeExpansionCost arch graph node saturatedCallArgCounts
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                // Lambda parameter - no children, Lambda processing assigns Arg SSAs
                // Just pass through - don't allocate here
                scopeAfterChildren

        | _ ->
            // Regular node - assign SSAs based on structural analysis
            if producesValue node.Kind then
                let cost = nodeExpansionCost arch graph node saturatedCallArgCounts  // Structural derivation
                if cost > 0 then
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs
                else
                    // Node produces a value conceptually but has 0 SSA cost (e.g., MemRef.store returning unit)
                    scopeAfterChildren
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
        | SemanticKind.Lambda (_, _, _, lambdaNameOpt, _) ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                // Get base name from parent Binding first
                let parentBindingName =
                    match node.Parent with
                    | Some parentId ->
                        match Map.tryFind parentId graph.Nodes with
                        | Some { Kind = SemanticKind.Binding(bindingName, _, _, _) } -> Some bindingName
                        | _ -> None
                    | None -> None

                // Only use lambdaNameOpt if it matches the parent Binding name
                // This distinguishes explicit names (like "_start") from inherited context
                // (where lambdaNameOpt = env.EnclosingFunction for capture analysis)
                match lambdaNameOpt, parentBindingName with
                | Some explicitName, Some parentName when explicitName = parentName -> explicitName
                | _, _ ->
                    if Set.contains nodeIdVal entryPoints then
                        "main"
                    else
                        // Use already-computed parentBindingName
                        match parentBindingName with
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
let assignSSA (arch: Architecture) (graph: SemanticGraph) (saturatedCallArgCounts: Map<NodeId, int>) : SSAAssignment =
    let lambdaNames, entryPoints = collectLambdas graph

    let mutable allAssignments = Map.empty
    let mutableClosureLayouts = System.Collections.Generic.Dictionary<int, ClosureLayout>()
    let mutableDULayouts = System.Collections.Generic.Dictionary<int, DULayout>()
    let mutableInnerScopeAssignments = System.Collections.Generic.Dictionary<int, NodeSSAAllocation>()

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
            assignFunctionBody arch graph mutableClosureLayouts mutableDULayouts mutableInnerScopeAssignments saturatedCallArgCounts scope bindingId
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

    // Track module-level SSA counter for top-level Lambda nodes
    let mutable topLevelCounter = moduleLevelCounter

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId, captures, enclosingFuncOpt, _context) ->
            // FIX (January 2026): Only process top-level Lambdas to avoid double-processing
            // Nested Lambdas (with enclosingFuncOpt = Some _) are already handled recursively
            // by assignFunctionBody when processing their enclosing Lambda's body
            let isTopLevel = Option.isNone enclosingFuncOpt

            if isTopLevel then
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
                let bodyScope = assignFunctionBody arch graph mutableClosureLayouts mutableDULayouts mutableInnerScopeAssignments saturatedCallArgCounts paramScope bodyId

                // Merge into global assignments (including parameter SSAs)
                for kvp in paramScope.Assignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments
                for kvp in bodyScope.Assignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments

                // Merge nested lambda scope assignments collected during recursive traversal
                for kvp in mutableInnerScopeAssignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments
                mutableInnerScopeAssignments.Clear()

                // Assign SSAs to the Lambda node itself (for closure value)
                // Top-level Lambdas (not visited during body traversal) need SSA assignments
                // for closure construction if they have captures
                let cost = computeLambdaSSACost captures
                if cost > 0 then
                    // Lambda with captures needs SSAs for closure struct construction
                    let ssas = List.init cost (fun i -> V (topLevelCounter + i))
                    let alloc = NodeSSAAllocation.multi ssas
                    allAssignments <- Map.add nodeIdVal alloc allAssignments
                    topLevelCounter <- topLevelCounter + cost
                // Note: Lambdas with no captures (cost=0) don't need SSA assignments
                // They are emitted as direct function symbols

                // Assign SSAs to parent Binding if this Lambda has one
                // Lambda Bindings are filtered out of Pass 1, so they need SSA assignment here
                match node.Parent with
                | Some parentId ->
                    match Map.tryFind parentId graph.Nodes with
                    | Some parentNode when (match parentNode.Kind with SemanticKind.Binding _ -> true | _ -> false) ->
                        let parentIdVal = NodeId.value parentId
                        if not (Map.containsKey parentIdVal allAssignments) then
                            // Binding needs SSAs (fixed cost of 3)
                            let bindingCost = 3
                            let bindingSSAs = List.init bindingCost (fun i -> V (topLevelCounter + i))
                            let bindingAlloc = NodeSSAAllocation.multi bindingSSAs
                            allAssignments <- Map.add parentIdVal bindingAlloc allAssignments
                            topLevelCounter <- topLevelCounter + bindingCost
                    | _ -> ()
                | None -> ()

        // PRD-15 FIX (January 2026): SeqExpr MoveNext bodies need their own SSA scope
        // MoveNext is a separate function with seqPtr as %arg0, body SSAs start at 1
        | SemanticKind.SeqExpr (bodyId, _captures) ->
            // MoveNext function: %arg0 = seqPtr, body SSAs start at v1
            let initialScope = { FunctionScope.empty with Counter = 1 }
            let bodyScope = assignFunctionBody arch graph mutableClosureLayouts mutableDULayouts mutableInnerScopeAssignments saturatedCallArgCounts initialScope bodyId

            // Merge SeqExpr body assignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments

        | _ -> ()

    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION: Check for unassigned value-producing nodes
    // ═══════════════════════════════════════════════════════════════════════════
    let unassignedNodes =
        graph.Nodes
        |> Map.toList
        |> List.filter (fun (_, node) ->
            let nodeIdVal = NodeId.value node.Id
            if not node.IsReachable then false
            elif not (producesValue node.Kind) then false
            elif Map.containsKey nodeIdVal allAssignments then false
            else
                // Node produces value but has no SSA - check if this is expected
                match node.Kind with
                | SemanticKind.Lambda (_, _, captures, _, _) ->
                    // No-capture Lambdas are emitted as direct function symbols, not SSA values
                    // Only Lambdas with captures need SSAs for closure struct construction
                    not (List.isEmpty captures)
                | SemanticKind.Intrinsic _ ->
                    // Intrinsic function nodes (TFun types) are just references, not values
                    // Only intrinsic CALLS (via Application) produce values
                    match node.Type with
                    | NativeType.TFun _ -> false  // Function reference, not a value
                    | _ -> true  // Non-function intrinsic should have SSA
                | _ -> true)
        |> List.map (fun (_, node) -> NodeId.value node.Id, node.Kind)

    if not (List.isEmpty unassignedNodes) then
        printfn "[SSA VALIDATION] Found %d unassigned value-producing nodes:" (List.length unassignedNodes)
        for (id, kind) in unassignedNodes |> List.take (min 10 (List.length unassignedNodes)) do
            printfn "  Node %d: %A" id kind

    // Convert mutable dictionaries to immutable maps
    let closureLayouts = 
        mutableClosureLayouts
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
        |> Map.ofSeq

    let duLayouts =
        mutableDULayouts
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
        |> Map.ofSeq

    {
        NodeSSA = allAssignments
        LambdaNames = lambdaNames
        EntryPointLambdas = entryPoints
        ClosureLayouts = closureLayouts
        DULayouts = duLayouts
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

/// Look up DULayout for a DUConstruct node needing arena allocation (coeffect lookup)
/// Returns None for homogeneous DUs like Option that use inline struct
let lookupDULayout (nodeId: NodeId) (assignment: SSAAssignment) : DULayout option =
    Map.tryFind (NodeId.value nodeId) assignment.DULayouts

/// Check if a DUConstruct node needs arena allocation
let hasDULayout (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Map.containsKey (NodeId.value nodeId) assignment.DULayouts

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
                    let fieldTypes = TInt I1 :: elemMlir :: TIndex :: captureTypes
                    let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
                    let actualLazyType = TMemRefStatic(totalBytes, TInt I8)
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
                        let fieldTypes = TInt I32 :: elemMlir :: TIndex :: captureTypes @ internalStateTypes
                        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
                        let actualSeqType = TMemRefStatic(totalBytes, TInt I8)
                        Some actualSeqType

                | _ -> None
            | None -> None
        | _ -> None
