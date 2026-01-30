/// PSG Combinators - XParsec-based pattern matching for SemanticGraph nodes
///
///
/// XParsec provides:
/// - Parser<'Parsed, 'T, 'State, 'Input, 'InputSlice> type
/// - >>=, |>>, .>>, >>., <|>, parser { } - all standard combinators
/// - getUserState, setUserState, updateUserState - state threading
/// - Reader.ofString - creates cursor with custom state
///
/// We use XParsec for PSG graph navigation by:
/// 1. Threading PSGParserState through XParsec's state mechanism
/// 2. Using empty string as dummy input (PSG doesn't consume characters)
/// 3. Navigating by updating state.Current, not consuming input
///

module Alex.XParsec.PSGCombinators

open XParsec
open XParsec.Parsers     // preturn, fail, getUserState, setUserState, updateUserState
open XParsec.Combinators // >>=, |>>, .>>, >>., <|>, parser { }
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.PSGZipper
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// PSG PARSER STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Parser state threaded through pattern matching
///
/// PLATFORM-AWARE (January 2026):
/// Platform info flows through parser state, enabling type resolution
/// without hard-coded values in witnesses. Use `platformWordType` and
/// `platformWordBits` for architecture-appropriate types.
///
/// The `Platform` field contains `PlatformResolutionResult` which has:
/// - TargetArch: Architecture (X86_64, ARM64, etc.)
/// - PlatformWordType: MLIRType (already resolved!)
/// - TargetOS: OSFamily
/// - RuntimeMode: Freestanding or Console
type PSGParserState = {
    Graph: SemanticGraph
    Zipper: PSGZipper
    /// Currently focused node
    Current: SemanticNode
    /// Platform resolution result (architecture, OS, word type, bindings)
    /// This is a coeffect - already computed, just look it up
    Platform: PlatformResolutionResult
    
    /// Optional execution trace collector (only enabled for diagnostic runs)
    ExecutionTrace: Alex.Traversal.TransferTypes.TraceCollector option
    /// Current depth in hierarchy (0=Witness, 1=Pattern, 2=Element)
    CurrentDepth: int
}

// ═══════════════════════════════════════════════════════════════════════════
// PSG PARSER TYPE (5 type parameters - using XParsec's Parser directly)
// ═══════════════════════════════════════════════════════════════════════════

/// PSG parser type - uses XParsec's Parser with custom state
///
/// PSG parsers don't parse characters - they navigate a graph structure.
/// We use empty string as dummy input, threading PSGParserState through XParsec.
///
/// Type parameters:
/// - 'T: Parsed result type
/// - char: Element type (dummy - we don't consume characters)
/// - PSGParserState: Our custom state
/// - ReadableString: Input type (always empty string)
/// - ReadableStringSlice: Sliceable input type
type PSGParser<'T> = Parser<'T, char, PSGParserState, ReadableString, ReadableStringSlice>

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM-AWARE TYPE RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Get the word-sized integer type for the target platform
/// Already resolved in PlatformResolutionResult - just look it up
let platformWordType (state: PSGParserState) : MLIRType =
    state.Platform.PlatformWordType

/// Get the word width in bits for the target platform
let platformWordBits (state: PSGParserState) : int =
    match platformWordWidth state.Platform.TargetArch with
    | I64 -> 64
    | I32 -> 32
    | I16 -> 16
    | I8 -> 8
    | I1 -> 1

/// Get the target architecture
let targetArch (state: PSGParserState) : Architecture =
    state.Platform.TargetArch

/// Get the appropriate return type for main function
/// This uses platform word size, NOT hard-coded i32
let mainReturnType (state: PSGParserState) : MLIRType =
    state.Platform.PlatformWordType

/// Get the appropriate type for nativeint/unativeint
let nativeIntType (state: PSGParserState) : MLIRType =
    state.Platform.PlatformWordType

/// Map NTUKind to MLIRType with platform awareness
/// Delegates to TypeMapping but provides platform context from state
let mapNTUKindForPlatform (state: PSGParserState) (kind: NTUKind) : MLIRType =
    Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType state.Platform.TargetArch kind

// ═══════════════════════════════════════════════════════════════════════════
// BASIC PSG OPERATIONS (use XParsec's state threading)
// ═══════════════════════════════════════════════════════════════════════════

/// Get current PSG node from state
let getCurrentNode : PSGParser<SemanticNode> =
    getUserState |>> (fun state -> state.Current)

/// Get current graph from state
let getGraph : PSGParser<SemanticGraph> =
    getUserState |>> (fun state -> state.Graph)

/// Get platform info from state
let getPlatform : PSGParser<PlatformResolutionResult> =
    getUserState |>> (fun state -> state.Platform)

/// Set current node in state
let setCurrentNode (node: SemanticNode) : PSGParser<unit> =
    updateUserState (fun state -> { state with Current = node })

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE PARSERS (pattern match on current node)
// ═══════════════════════════════════════════════════════════════════════════

/// Match the current node's kind
let pKind (expected: SemanticKind) : PSGParser<SemanticNode> =
    parser {
        let! node = getCurrentNode
        if node.Kind = expected then
            return node
        else
            return! fail (Message (sprintf "Expected %A but got %A" expected node.Kind))
    }

/// Match any node (always succeeds with current node)
let pAny : PSGParser<SemanticNode> =
    getCurrentNode

/// Match a Literal node
let pLiteral : PSGParser<NativeLiteral> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Literal lit -> return lit
        | _ -> return! fail (Message "Expected Literal")
    }

/// Match a VarRef node (defId is optional)
let pVarRef : PSGParser<string * NodeId option> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.VarRef (name, defIdOpt) -> return (name, defIdOpt)
        | _ -> return! fail (Message "Expected VarRef")
    }

/// Match a FieldGet node (extracts field from struct/tuple)
let pFieldGet : PSGParser<NodeId * string> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.FieldGet (structId, fieldName) -> return (structId, fieldName)
        | _ -> return! fail (Message "Expected FieldGet")
    }

/// Match an Application node
let pApplication : PSGParser<NodeId * NodeId list> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Application (funcId, argIds) -> return (funcId, argIds)
        | _ -> return! fail (Message "Expected Application")
    }

/// Match an Intrinsic node
let pIntrinsic : PSGParser<IntrinsicInfo> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Intrinsic info -> return info
        | _ -> return! fail (Message "Expected Intrinsic")
    }

/// Match a PlatformBinding node
let pPlatformBinding : PSGParser<string> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.PlatformBinding entryPoint -> return entryPoint
        | _ -> return! fail (Message "Expected PlatformBinding")
    }

/// Match a Binding node
let pBinding : PSGParser<string * bool * bool * bool> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Binding (name, isMut, isRec, isEntry) ->
            return (name, isMut, isRec, isEntry)
        | _ -> return! fail (Message "Expected Binding")
    }

/// Match a Lambda node (params are name*type*nodeId tuples for SSA assignment)
let pLambda : PSGParser<(string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType * NodeId) list * NodeId> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Lambda (params', bodyId, _captures, _, _) -> return (params', bodyId)
        | _ -> return! fail (Message "Expected Lambda")
    }

/// Match a Lambda node with captures
/// Returns: (params, bodyId, captures)
let pLambdaWithCaptures : PSGParser<(string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType * NodeId) list * NodeId * CaptureInfo list> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Lambda (params', bodyId, captures, _, _) -> return params', bodyId, captures
        | _ -> return! fail (Message "Expected Lambda")
    }

/// Match a Lambda node with parent Binding name
/// Composes: pLambdaWithCaptures + zipper navigation + pBinding
/// Returns: (bindingName, params, bodyId, captures)
let pLambdaWithBinding : PSGParser<string * (string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType * NodeId) list * NodeId * CaptureInfo list> =
    parser {
        // Get Lambda data from current node
        let! (params', bodyId, captures) = pLambdaWithCaptures
        
        // Navigate to parent using zipper
        let! state = getUserState
        match up state.Zipper with
        | Some parentZipper ->
            // Save current state
            let savedState = state
            
            // Update to parent node
            let parentNode = parentZipper.Focus
            do! setUserState { state with Zipper = parentZipper; Current = parentNode }
            
            // Try to match parent as Binding
            let! bindingResult =
                (parser {
                    let! (name, _, _, _) = pBinding
                    return Some name
                } <|> preturn None)
            
            // Restore original state
            do! setUserState savedState
            
            match bindingResult with
            | Some name -> return (name, params', bodyId, captures)
            | None -> return (sprintf "lambda_%d" (NodeId.value state.Current.Id), params', bodyId, captures)
        | None ->
            // No parent - use node ID as fallback name
            return (sprintf "lambda_%d" (NodeId.value state.Current.Id), params', bodyId, captures)
    }

/// Match an IfThenElse node
let pIfThenElse : PSGParser<NodeId * NodeId * NodeId option> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.IfThenElse (guard, thenB, elseOpt) ->
            return (guard, thenB, elseOpt)
        | _ -> return! fail (Message "Expected IfThenElse")
    }

/// Match a WhileLoop node
let pWhileLoop : PSGParser<NodeId * NodeId> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.WhileLoop (guard, body) -> return (guard, body)
        | _ -> return! fail (Message "Expected WhileLoop")
    }

/// Match a ForLoop node
let pForLoop : PSGParser<string * NodeId * NodeId * bool * NodeId> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.ForLoop (var, start, finish, isUp, body) ->
            return (var, start, finish, isUp, body)
        | _ -> return! fail (Message "Expected ForLoop")
    }

/// Match a Sequential node
let pSequential : PSGParser<NodeId list> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.Sequential childIds -> return childIds
        | _ -> return! fail (Message "Expected Sequential")
    }

/// Match a TypeAnnotation node
/// TypeAnnotation is a transparent wrapper - returns (wrappedNodeId, annotatedType)
let pTypeAnnotation : PSGParser<NodeId * NativeType> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.TypeAnnotation (wrappedId, annotatedType) ->
            return (wrappedId, annotatedType)
        | _ -> return! fail (Message "Expected TypeAnnotation")
    }

/// Match a PatternBinding node
let pPatternBinding : PSGParser<string> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.PatternBinding name -> return name
        | _ -> return! fail (Message "Expected PatternBinding")
    }

// ═══════════════════════════════════════════════════════════════════════════
// DISCRIMINATED UNION PARSERS (January 2026)
// ═══════════════════════════════════════════════════════════════════════════

/// Match a DUGetTag node - extracts tag from DU value
/// Returns: (duValueNodeId, duType)
let pDUGetTag : PSGParser<NodeId * NativeType> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.DUGetTag (duValueId, duType) -> return (duValueId, duType)
        | _ -> return! fail (Message "Expected DUGetTag")
    }

/// Match a DUEliminate node - type-safe payload extraction via case eliminator
/// Returns: (duValueNodeId, caseIndex, caseName, payloadType)
let pDUEliminate : PSGParser<NodeId * int * string * NativeType> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.DUEliminate (duValueId, caseIndex, caseName, payloadType) ->
            return (duValueId, caseIndex, caseName, payloadType)
        | _ -> return! fail (Message "Expected DUEliminate")
    }

/// Match a DUConstruct node - constructs DU value in arena
/// Returns: (caseName, caseIndex, payloadOpt, arenaHintOpt)
let pDUConstruct : PSGParser<string * int * NodeId option * NodeId option> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.DUConstruct (caseName, caseIndex, payload, arenaHint) ->
            return (caseName, caseIndex, payload, arenaHint)
        | _ -> return! fail (Message "Expected DUConstruct")
    }

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION COMBINATORS (use XParsec's state threading)
// ═══════════════════════════════════════════════════════════════════════════

/// Focus on a child node by ID
let focusChild (childId: NodeId) : PSGParser<SemanticNode> =
    parser {
        let! state = getUserState
        match SemanticGraph.tryGetNode childId state.Graph with
        | Some childNode ->
            do! setCurrentNode childNode
            return childNode
        | None ->
            return! fail (Message (sprintf "Child node %A not found" childId))
    }

/// Run parser on a specific child node, then restore focus
let onChild (childId: NodeId) (p: PSGParser<'T>) : PSGParser<'T> =
    parser {
        let! state = getUserState
        match SemanticGraph.tryGetNode childId state.Graph with
        | Some childNode ->
            // Save current position
            let savedCurrent = state.Current
            // Navigate to child
            do! setCurrentNode childNode
            // Run child parser
            let! result = p
            // Restore position
            do! setCurrentNode savedCurrent
            return result
        | None ->
            return! fail (Message (sprintf "Child node %A not found" childId))
    }

/// Run parser on each child and collect results
let onChildren (childIds: NodeId list) (p: PSGParser<'T>) : PSGParser<'T list> =
    parser {
        let! results =
            childIds
            |> List.map (fun cid -> onChild cid p)
            |> List.fold (fun accParser nextParser ->
                parser {
                    let! acc = accParser
                    let! next = nextParser
                    return acc @ [next]
                }) (preturn [])
        return results
    }

// ═══════════════════════════════════════════════════════════════════════════
// ATOMIC OPERATION CLASSIFICATION PATTERNS
// ═══════════════════════════════════════════════════════════════════════════
//
// TERMINOLOGY NOTE (January 2026):
// - FNCS calls them: "Intrinsics" (intrinsic to native type universe)
// - MiddleEnd calls them: "Atomic Operations" (atomic/indivisible at MLIR level)
// - PSG type name remains `SemanticKind.Intrinsic` (can't change FNCS output)
// - Comments and function names use "Atomic Operation" terminology

/// Match atomic operation by module (SemanticKind.Intrinsic)
let pIntrinsicModule (expectedModule: IntrinsicModule) : PSGParser<IntrinsicInfo> =
    parser {
        let! info = pIntrinsic
        if info.Module = expectedModule then
            return info
        else
            return! fail (Message (sprintf "Expected module %A but got %A" expectedModule info.Module))
    }

/// Match atomic operation by full name pattern (SemanticKind.Intrinsic)
let pIntrinsicNamed (fullName: string) : PSGParser<IntrinsicInfo> =
    parser {
        let! info = pIntrinsic
        if info.FullName = fullName then
            return info
        else
            return! fail (Message (sprintf "Expected %s but got %s" fullName info.FullName))
    }

/// Classify atomic operation by category for emission dispatch
type EmissionCategory =
    | BinaryArith of mlirOp: string
    | UnaryArith of mlirOp: string
    | Comparison of mlirOp: string
    | MemoryOp of op: string
    | StringOp of op: string
    // NOTE: ConsoleOp removed - Console is NOT an atomic operation, it's Layer 3 user code
    // in Fidelity.Platform that uses Sys.* atomic operations. See fsnative-spec/spec/platform-bindings.md
    | PlatformOp of op: string
    | DateTimeOp of op: string
    | TimeSpanOp of op: string
    | OtherAtomicOp

let classifyAtomicOp (info: IntrinsicInfo) : EmissionCategory =
    match info.Module, info.Operation with
    // Arithmetic operators
    | IntrinsicModule.Operators, "op_Addition" -> BinaryArith "addi"
    | IntrinsicModule.Operators, "op_Subtraction" -> BinaryArith "subi"
    | IntrinsicModule.Operators, "op_Multiply" -> BinaryArith "muli"
    | IntrinsicModule.Operators, "op_Division" -> BinaryArith "divsi"
    | IntrinsicModule.Operators, "op_Modulus" -> BinaryArith "remsi"
    // Comparison operators
    | IntrinsicModule.Operators, "op_LessThan" -> Comparison "slt"
    | IntrinsicModule.Operators, "op_LessThanOrEqual" -> Comparison "sle"
    | IntrinsicModule.Operators, "op_GreaterThan" -> Comparison "sgt"
    | IntrinsicModule.Operators, "op_GreaterThanOrEqual" -> Comparison "sge"
    | IntrinsicModule.Operators, "op_Equality" -> Comparison "eq"
    | IntrinsicModule.Operators, "op_Inequality" -> Comparison "ne"
    // Memory
    | IntrinsicModule.NativePtr, op -> MemoryOp op
    // String
    | IntrinsicModule.String, op -> StringOp op
    // NOTE: Console is NOT an atomic operation - see fsnative-spec/spec/platform-bindings.md
    // Platform (Sys.* atomic operations from FNCS)
    | IntrinsicModule.Sys, op -> PlatformOp op
    // DateTime operations
    | IntrinsicModule.DateTime, op -> DateTimeOp op
    // TimeSpan operations
    | IntrinsicModule.TimeSpan, op -> TimeSpanOp op
    | _ -> OtherAtomicOp

/// Match and classify atomic operation (SemanticKind.Intrinsic)
let pClassifiedAtomicOp : PSGParser<IntrinsicInfo * EmissionCategory> =
    pIntrinsic |>> fun info -> (info, classifyAtomicOp info)

// ═══════════════════════════════════════════════════════════════════════════
// LAZY VALUE PARSERS (PRD-14, January 2026)
// ═══════════════════════════════════════════════════════════════════════════

/// Match a LazyExpr node - lazy value construction
/// Returns: (bodyId, captures)
let pLazyExpr : PSGParser<NodeId * CaptureInfo list> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.LazyExpr (bodyId, captures) ->
            return (bodyId, captures)
        | _ ->
            return! fail (Message "Expected LazyExpr node")
    }

/// Match a LazyForce node - force lazy evaluation
/// Returns: lazyValueNodeId
let pLazyForce : PSGParser<NodeId> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.LazyForce lazyId ->
            return lazyId
        | _ ->
            return! fail (Message "Expected LazyForce node")
    }

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENCE PARSERS (PRD-15, January 2026)
// ═══════════════════════════════════════════════════════════════════════════

/// Match a SeqExpr node - sequence expression construction
/// Returns: (bodyId, captures)
let pSeqExpr : PSGParser<NodeId * CaptureInfo list> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.SeqExpr (bodyId, captures) ->
            return (bodyId, captures)
        | _ ->
            return! fail (Message "Expected SeqExpr node")
    }

/// Match a ForEach node - for-in loop over sequence
/// Returns: (var, collection, body)
let pForEach : PSGParser<string * NodeId * NodeId> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.ForEach (var, collection, body) ->
            return (var, collection, body)
        | _ ->
            return! fail (Message "Expected ForEach node")
    }

// ═══════════════════════════════════════════════════════════════════════════
// RUNNER (create Reader with empty string and custom state)
// ═══════════════════════════════════════════════════════════════════════════

/// Run a parser on a node with platform context
/// Platform info flows through state for type resolution
///
/// CRITICAL: Uses Reader.ofString with EMPTY STRING.
/// PSG parsers don't consume characters - they navigate graph structure.
let runParser (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    let state = { 
        Graph = graph
        Zipper = zipper
        Current = node
        Platform = platform
        ExecutionTrace = None  // No tracing by default
        CurrentDepth = 0
    }
    let reader = Reader.ofString "" state  // Empty string - we don't parse characters
    parser reader

/// Run parser with execution trace enabled (for diagnostics)
let runParserWithTrace (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) (traceCollector: Alex.Traversal.TransferTypes.TraceCollector) =
    let state = {
        Graph = graph
        Zipper = zipper
        Current = node
        Platform = platform
        ExecutionTrace = Some traceCollector
        CurrentDepth = 0
    }
    let reader = Reader.ofString "" state
    parser reader

/// Try to match a pattern, returning option
/// Platform info flows through state for type resolution
let tryMatch (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    match runParser parser graph node zipper platform with
    | Ok success -> Some (success.Parsed, zipper)
    | Error _ -> None

/// Try to match a pattern with diagnostic error capture
/// Returns Result with detailed error information on failure
/// Use this for debugging pattern match failures
let tryMatchWithDiagnostics (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    match runParser parser graph node zipper platform with
    | Ok success -> Result.Ok (success.Parsed, zipper)
    | Error err -> 
        // XParsec error contains position, expected, and messages
        let errorMsg = sprintf "Pattern match failed at position %d. Error: %A" err.Position.Index err
        Result.Error errorMsg

/// Create initial parser state from zipper
/// Platform must be passed explicitly (coeffect, not part of zipper)
let stateFromZipper (zipper: PSGZipper) (node: SemanticNode) (platform: PlatformResolutionResult) : PSGParserState =
    {
        Graph = zipper.Graph
        Zipper = zipper
        Current = node
        Platform = platform
        ExecutionTrace = None
        CurrentDepth = 0
    }

/// Emit a trace entry (if tracing is enabled)
/// Use this from Elements and Patterns to record execution
let emitTrace (componentName: string) (parameters: string) : PSGParser<unit> =
    fun reader ->
        let state = reader.State
        match state.ExecutionTrace with
        | Some collector ->
            Alex.Traversal.TransferTypes.TraceCollector.add
                state.CurrentDepth
                componentName
                (Some state.Current.Id)
                parameters
                collector
        | None -> ()
        preturn () reader

/// Guard combinator - ensure a condition is true, or fail with message
/// This is the DECLARATIVE alternative to imperative if statements
let ensure (condition: bool) (errorMsg: string) : PSGParser<unit> =
    if condition then
        preturn ()
    else
        fail (Message errorMsg)

/// Run a parser using zipper
/// Platform must be passed explicitly (coeffect, not part of zipper)
let runParserWithZipper (parser: PSGParser<'T>) (zipper: PSGZipper) (node: SemanticNode) (platform: PlatformResolutionResult) =
    let state = stateFromZipper zipper node platform
    let reader = Reader.ofString "" state
    parser reader

/// Try to match with diagnostic trace enabled
/// Returns Result with trace attached on failure
let tryMatchWithTrace (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    let traceCollector = Alex.Traversal.TransferTypes.TraceCollector.create()
    match runParserWithTrace parser graph node zipper platform traceCollector with
    | Ok success -> Result.Ok (success.Parsed, zipper, Alex.Traversal.TransferTypes.TraceCollector.toList traceCollector)
    | Error err ->
        let trace = Alex.Traversal.TransferTypes.TraceCollector.toList traceCollector
        Result.Error (err, trace)

// ═══════════════════════════════════════════════════════════════════════════
// NOTE: Sub-graph witnessing infrastructure needed for control flow
// ═══════════════════════════════════════════════════════════════════════════
//
// ARCHITECTURAL GAP: Control flow witnesses (IfThenElse, WhileLoop, ForLoop) need
// infrastructure to recursively witness branch sub-graphs and collect their operations
// for structured control flow regions.
//
// This combinator cannot be implemented here because:
// 1. PSGCombinators is at the XParsec layer
// 2. Witnessing requires WitnessContext/WitnessOutput from Traversal layer
// 3. This would create a circular dependency
//
// SOLUTION: This infrastructure needs to live in:
// - Option A: A new module in Traversal/ that provides sub-graph witnessing
// - Option B: As a helper in NanopassArchitecture.fs where it can access witness types
// - Option C: In each control flow witness itself as witness-specific logic
//
// The infrastructure needs to:
// - Navigate to a sub-graph root using child edges
// - Recursively witness all nodes in that sub-graph
// - Collect all operations into a flat list
// - Return that list for composition into control flow structures (SCF.If, SCF.While, etc.)
//
// Until this is resolved, ControlFlowWitness, LambdaWitness, and similar witnesses
// that need sub-graph witnessing remain stubbed.
