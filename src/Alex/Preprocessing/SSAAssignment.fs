/// SSA Assignment Pass - Alex preprocessing for MLIR emission
///
/// This pass assigns SSA values to PSG nodes BEFORE MLIR emission.
/// SSA is an MLIR/LLVM concern, not F# semantics, so it lives in Alex.
///
/// Key design:
/// - SSA counter resets at each Lambda boundary (per-function scoping)
/// - Post-order traversal ensures values are assigned before uses
/// - Returns Map<NodeId, NodeSSAAllocation> that witnesses read (coeffect lookup, no generation during emission)
/// - Uses structured SSA type (V of int | Arg of int), not strings
/// - Knows MLIR expansion costs: one PSG node may need multiple SSAs
module Alex.Preprocessing.SSAAssignment

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA ALLOCATION FOR NODES
// ═══════════════════════════════════════════════════════════════════════════

/// SSA allocation for a node - supports multi-SSA expansion
/// SSAs are in emission order; Result is the final SSA (what gets returned/used)
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
// CLOSURE LAYOUT (Coeffect for Closing Lambdas)
// ═══════════════════════════════════════════════════════════════════════════
//
// For Lambdas with captures, we pre-compute the complete closure layout.
// This is deterministic - derived from CaptureInfo list in PSG.
// Witnesses observe this coeffect; they do NOT compute layout during emission.

/// How a variable is captured in a closure
type CaptureMode =
    | ByValue  // Immutable variable: copy value into env struct
    | ByRef    // Mutable variable: store pointer to alloca in env struct

/// Layout information for a single captured variable
type CaptureSlot = {
    /// Name of the captured variable
    Name: string
    /// Index in the environment struct (0, 1, 2, ...)
    SlotIndex: int
    /// MLIR type of the slot (value type for ByValue, ptr for ByRef)
    SlotType: MLIRType
    /// Source NodeId of the captured binding (for SSA lookup)
    SourceNodeId: NodeId option
    /// How the variable is captured
    Mode: CaptureMode
    /// SSA for GEP to this slot during closure construction
    GepSSA: SSA
}

/// Complete closure layout for a Lambda with captures
/// This is the coeffect that LambdaWitness observes
type ClosureLayout = {
    /// The Lambda node this layout is for
    LambdaNodeId: NodeId
    /// Ordered list of capture slots (matches env struct field order)
    Captures: CaptureSlot list
    /// SSA for addressof code_ptr
    CodeAddrSSA: SSA
    /// SSA for undef closure struct
    ClosureUndefSSA: SSA
    /// SSA for insertvalue of code_ptr at [0]
    ClosureWithCodeSSA: SSA
    /// SSAs for insertvalue of each capture at [1..N] (one per capture)
    CaptureInsertSSAs: SSA list
    /// SSA for final closure result (last insertvalue)
    ClosureResultSSA: SSA
    /// MLIR type of the environment struct (kept for compatibility)
    EnvStructType: MLIRType
    /// MLIR type of the closure struct: {ptr, T0, T1, ...} = {code_ptr, captures...}
    ClosureStructType: MLIRType
}

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
/// TRUE FLAT CLOSURE: For a Lambda with N captures, emission requires:
/// - Simple Lambda (0 captures): 0 SSAs (emits func.func, no local value needed)
/// - Closing Lambda (N captures):
///   - 1 SSA: addressof for code pointer
///   - N SSAs: (reserved for extractvalue in callee, reused as intermediate insertvalue results)
///   - 1 SSA: undef for closure struct
///   - 1 SSA: insertvalue for code_ptr at [0]
///   - N SSAs: insertvalue for each capture at [1..N] (LAST one is final result)
///   Total: 2N + 3 SSAs
///
/// Additionally, for mutable captures (ByRef), we need the address of the captured variable.
/// This is handled by the captured variable's binding (alloca), not here.
let private computeLambdaSSACost (captures: CaptureInfo list) : int =
    let n = List.length captures
    if n = 0 then
        0  // Simple function - no closure struct needed
    else
        2 * n + 3  // addressof + N extracts + undef + (1 + N) insertvalues

/// Minimal NativeType to MLIRType mapping for capture slots
/// This is a subset of TypeMapping.mapNativeType, inlined here to avoid
/// circular dependencies (SSAAssignment must compile before TypeMapping)
let rec private mapCaptureType (ty: NativeType) : MLIRType =
    match ty with
    | NativeType.TApp(tycon, args) ->
        match tycon.Name with
        | "unit" -> TInt I32
        | "bool" -> TInt I1
        | "int8" | "sbyte" -> TInt I8
        | "uint8" | "byte" -> TInt I8
        | "int16" -> TInt I16
        | "uint16" -> TInt I16
        | "int" | "int32" -> TInt I32
        | "uint" | "uint32" -> TInt I32
        | "int64" -> TInt I64
        | "uint64" -> TInt I64
        | "nativeint" | "unativeint" -> TIndex  // Platform-word integer (i64 on 64-bit)
        | "float32" | "single" -> TFloat F32
        | "float" | "double" -> TFloat F64
        | "char" -> TInt I32
        | "string" -> TStruct [TPtr; TInt I64]  // Fat pointer
        | "Ptr" | "nativeptr" | "byref" | "inref" | "outref" -> TPtr
        | "array" -> TStruct [TPtr; TInt I64]  // Fat pointer
        | "option" | "voption" ->
            match args with
            | [innerTy] -> TStruct [TInt I1; mapCaptureType innerTy]
            | _ -> TPtr  // Fallback
        | _ -> TPtr  // Records, DUs, unknown types - treat as pointer-sized
    | NativeType.TFun _ -> TStruct [TPtr; TPtr]  // Function = closure struct
    | NativeType.TTuple (elements, _) ->
        TStruct (elements |> List.map mapCaptureType)
    | NativeType.TVar _ -> TPtr  // Type variable - assume pointer-sized
    | NativeType.TByref _ -> TPtr
    | _ -> TPtr  // Fallback for other cases

/// Compute the MLIR type for a capture slot based on capture mode
let private captureSlotType (capture: CaptureInfo) : MLIRType =
    if capture.IsMutable then
        // Mutable capture: store pointer to the alloca
        TPtr
    else
        // Immutable capture: store the value directly
        mapCaptureType capture.Type

/// Build the environment struct type from captures
let private buildEnvStructType (captures: CaptureInfo list) : MLIRType =
    let slotTypes = captures |> List.map captureSlotType
    TStruct slotTypes

/// Build complete ClosureLayout from Lambda captures and pre-assigned SSAs
/// This is computed once during SSAAssignment - witnesses observe the result
///
/// TRUE FLAT CLOSURE SSA layout for N captures (total 2N+3 SSAs):
///   ssas[0]           = addressof code_ptr
///   ssas[1..N]        = extractvalue SSAs (for callee extraction)
///   ssas[N+1]         = undef closure struct
///   ssas[N+2]         = insertvalue code_ptr at [0]
///   ssas[N+3..2N+2]   = insertvalue for each capture at [1..N]
///   ssas[2N+2]        = final result (last insertvalue)
let private buildClosureLayout
    (lambdaNodeId: NodeId)
    (captures: CaptureInfo list)
    (ssas: SSA list)
    : ClosureLayout =

    let n = List.length captures

    // Extract SSAs by position for TRUE FLAT CLOSURE
    let codeAddrSSA = ssas.[0]
    let gepSSAs = ssas.[1..n]  // N SSAs for extractvalue in callee
    let undefSSA = ssas.[n + 1]
    let withCodeSSA = ssas.[n + 2]
    // InsertValue SSAs for captures: ssas[N+3], ssas[N+4], ..., ssas[2N+2]
    let captureInsertSSAs = ssas.[(n + 3)..(2 * n + 2)]
    // The final result is the last insertvalue SSA
    let resultSSA = ssas.[2 * n + 2]

    // Build capture slots with their GEP SSAs (for extraction in callee)
    let captureSlots =
        captures
        |> List.mapi (fun i capture ->
            {
                Name = capture.Name
                SlotIndex = i
                SlotType = captureSlotType capture
                SourceNodeId = capture.SourceNodeId
                Mode = if capture.IsMutable then ByRef else ByValue
                GepSSA = gepSSAs.[i]
            })

    // Build env struct type (for internal tracking, kept for compatibility)
    let envStructType = buildEnvStructType captures

    // TRUE FLAT CLOSURE: {code_ptr, capture_0, capture_1, ...}
    // Captures are inlined directly, not via env_ptr indirection
    // This eliminates lifetime issues - closure is returned by value with all state inline
    let captureTypes = captures |> List.map captureSlotType
    let closureStructType = TStruct (TPtr :: captureTypes)

    {
        LambdaNodeId = lambdaNodeId
        Captures = captureSlots
        CodeAddrSSA = codeAddrSSA
        ClosureUndefSSA = undefSSA
        ClosureWithCodeSSA = withCodeSSA
        CaptureInsertSSAs = captureInsertSSAs
        ClosureResultSSA = resultSSA
        EnvStructType = envStructType
        ClosureStructType = closureStructType
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
    | SemanticKind.Lambda (_, _, captures) ->
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
    | SemanticKind.TraitCall _ -> 1
    | SemanticKind.ArrayExpr _ -> 20
    | SemanticKind.ListExpr _ -> 20
    // PatternBinding needs SSAs for extraction + conversion
    // For tuple patterns: elemExtract + payloadExtract + convert = 3
    | SemanticKind.PatternBinding _ -> 3
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
    | SemanticKind.PlatformBinding _ -> true
    | SemanticKind.InterpolatedString _ -> true
    // These don't produce values (statements/void)
    | SemanticKind.Set _ -> false
    | SemanticKind.FieldSet _ -> false
    | SemanticKind.IndexSet _ -> false
    | SemanticKind.NamedIndexedPropertySet _ -> false
    | SemanticKind.WhileLoop _ -> false
    | SemanticKind.ForLoop _ -> false
    | SemanticKind.ForEach _ -> false
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
    (graph: SemanticGraph)
    (scope: FunctionScope)
    (nodeId: NodeId)
    : FunctionScope =

    match Map.tryFind nodeId graph.Nodes with
    | None -> scope
    | Some node ->
        // Post-order: process children first
        let scopeAfterChildren =
            node.Children |> List.fold (fun s childId -> assignFunctionBody graph s childId) scope

        // Special handling for nested Lambdas - they get their own scope
        // (but we still assign this Lambda node SSAs in parent scope for closure construction)
        match node.Kind with
        | SemanticKind.Lambda(_params, bodyId, captures) ->
            // Process Lambda body in a NEW scope (SSA counter resets)
            let _innerScope = assignFunctionBody graph FunctionScope.empty bodyId
            // Lambda itself gets SSAs in the PARENT scope for closure struct construction
            // SSA count is deterministic based on captures (from FNCS)
            let cost = computeLambdaSSACost captures
            if cost > 0 then
                let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                let alloc = NodeSSAAllocation.multi ssas
                FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                // Simple lambda (no captures) - no SSAs needed in parent scope
                // The function is emitted as func.func @name, called via func.call @name
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

    // Build reverse index: Lambda NodeId -> Binding name
    // This handles the case where Parent field isn't set on Lambdas
    let mutable lambdaToBindingName = Map.empty
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Binding (bindingName, _, _, _) ->
            // Check if first child is a Lambda
            match node.Children with
            | childId :: _ ->
                match Map.tryFind childId graph.Nodes with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ ->
                        // This Binding contains a Lambda - record the mapping
                        lambdaToBindingName <- Map.add (NodeId.value childId) bindingName lambdaToBindingName
                    | _ -> ()
                | None -> ()
            | [] -> ()
        | _ -> ()

    // Now assign names to all Lambdas
    // ARCHITECTURAL PRINCIPLE: When a Lambda is bound via `let name = ...`,
    // use the binding name as the function name. This preserves programmer intent.
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda _ ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                if Set.contains nodeIdVal entryPoints then
                    "main"
                else
                    // First check our reverse index (handles Parent = None case)
                    match Map.tryFind nodeIdVal lambdaToBindingName with
                    | Some bindingName -> bindingName
                    | None ->
                        // Fallback: check Parent field directly
                        match node.Parent with
                        | Some parentId ->
                            match Map.tryFind parentId graph.Nodes with
                            | Some parentNode ->
                                match parentNode.Kind with
                                | SemanticKind.Binding (bindingName, _, _, _) ->
                                    bindingName
                                | _ ->
                                    let n = sprintf "lambda_%d" lambdaCounter
                                    lambdaCounter <- lambdaCounter + 1
                                    n
                            | None ->
                                let n = sprintf "lambda_%d" lambdaCounter
                                lambdaCounter <- lambdaCounter + 1
                                n
                        | None ->
                            let n = sprintf "lambda_%d" lambdaCounter
                            lambdaCounter <- lambdaCounter + 1
                            n
            lambdaNames <- Map.add nodeIdVal name lambdaNames
            // CRITICAL: Also add the parent Binding's NodeId with the same name
            // VarRefs point to Bindings, not Lambdas. Both must resolve to the function name.
            match node.Parent with
            | Some parentId ->
                match Map.tryFind parentId graph.Nodes with
                | Some parentNode ->
                    match parentNode.Kind with
                    | SemanticKind.Binding _ ->
                        lambdaNames <- Map.add (NodeId.value parentId) name lambdaNames
                    | _ -> ()
                | None -> ()
            | None -> ()
        | _ -> ()

    lambdaNames, entryPoints

/// Main entry point: assign SSA names to all nodes in the graph
let assignSSA (graph: SemanticGraph) : SSAAssignment =
    let lambdaNames, entryPoints = collectLambdas graph

    // For each Lambda, assign SSAs to its body in its own scope
    let mutable allAssignments = Map.empty
    let mutable closureLayouts = Map.empty

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId, captures) ->
            // Assign SSAs to parameter PatternBindings (Arg 0, Arg 1, etc.)
            // This allows VarRefs to parameters to look up their SSAs
            let paramScope =
                params'
                |> List.mapi (fun i (_name, _ty, nodeId) -> i, nodeId)
                |> List.fold (fun (scope: FunctionScope) (i, nodeId) ->
                    // Parameters get Arg N SSAs, mapped to their PatternBinding NodeId
                    FunctionScope.assign nodeId (NodeSSAAllocation.single (Arg i)) scope
                ) FunctionScope.empty

            // Assign SSAs to body nodes
            let bodyScope = assignFunctionBody graph paramScope bodyId

            // Merge into global assignments (including parameter SSAs)
            for kvp in paramScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments

            // Lambda node SSAs are assigned by assignFunctionBody (via nodeExpansionCost)
            // For closing lambdas (with captures), also compute ClosureLayout
            if not (List.isEmpty captures) then
                // Look up the SSAs assigned to this Lambda node
                match Map.tryFind (NodeId.value node.Id) bodyScope.Assignments with
                | Some alloc when alloc.SSAs.Length >= (List.length captures + 4) ->
                    let layout = buildClosureLayout node.Id captures alloc.SSAs
                    closureLayouts <- Map.add (NodeId.value node.Id) layout closureLayouts
                | _ ->
                    // SSAs should have been assigned by assignFunctionBody
                    // If not, something went wrong - but we don't want to crash here
                    ()
        | _ -> ()

    // Also process top-level nodes (module bindings, etc.)
    let topLevelScope =
        graph.EntryPoints
        |> List.fold (fun scope entryId -> assignFunctionBody graph scope entryId) FunctionScope.empty

    for kvp in topLevelScope.Assignments do
        allAssignments <- Map.add kvp.Key kvp.Value allAssignments

    // Compute ClosureLayouts from top-level scope assignments as well
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(_, _, captures) when not (List.isEmpty captures) ->
            // Check if we already computed this layout (from Lambda body processing above)
            if not (Map.containsKey (NodeId.value node.Id) closureLayouts) then
                // Check top-level scope
                match Map.tryFind (NodeId.value node.Id) topLevelScope.Assignments with
                | Some alloc when alloc.SSAs.Length >= (List.length captures + 4) ->
                    let layout = buildClosureLayout node.Id captures alloc.SSAs
                    closureLayouts <- Map.add (NodeId.value node.Id) layout closureLayouts
                | _ -> ()
        | _ -> ()

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
