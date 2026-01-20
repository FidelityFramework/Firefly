/// PSG Combinators - XParsec-style pattern matching for SemanticGraph nodes
///
/// This module provides composable parsers for recognizing PSG node patterns
/// and emitting MLIR fragments. The combinators interlock to form coherent
/// MLIR expressions from PSG structure.
///
/// Key insight: XParsec works on sequential input. We adapt it to work on
/// PSG children sequences, enabling pattern recognition like:
/// - Binary operations: App(App(op, lhs), rhs)
/// - Pipe chains: x |> f |> g
/// - Curried applications: f a b c
module Alex.XParsec.PSGCombinators

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Result of pattern matching on PSG nodes
type PSGMatchResult<'T> =
    | Matched of 'T
    | NoMatch of reason: string

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
}

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

/// A PSG parser that attempts to match a pattern and produce a result
type PSGParser<'T> = PSGParserState -> PSGMatchResult<'T> * PSGParserState

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE PARSERS
// ═══════════════════════════════════════════════════════════════════════════

/// Always succeeds with the given value
let preturn (value: 'T) : PSGParser<'T> =
    fun state -> Matched value, state

/// Always fails with the given reason
let pfail (reason: string) : PSGParser<'T> =
    fun state -> NoMatch reason, state

/// Match the current node's kind
let pKind (expected: SemanticKind) : PSGParser<SemanticNode> =
    fun state ->
        if state.Current.Kind = expected then
            Matched state.Current, state
        else
            NoMatch (sprintf "Expected %A but got %A" expected state.Current.Kind), state

/// Match any node (always succeeds with current node)
let pAny : PSGParser<SemanticNode> =
    fun state -> Matched state.Current, state

/// Match a Literal node
let pLiteral : PSGParser<NativeLiteral> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.Literal lit -> Matched lit, state
        | _ -> NoMatch "Expected Literal", state

/// Match a VarRef node (defId is optional)
let pVarRef : PSGParser<string * NodeId option> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.VarRef (name, defIdOpt) -> Matched (name, defIdOpt), state
        | _ -> NoMatch "Expected VarRef", state

/// Match an Application node
let pApplication : PSGParser<NodeId * NodeId list> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.Application (funcId, argIds) -> Matched (funcId, argIds), state
        | _ -> NoMatch "Expected Application", state

/// Match an Intrinsic node
let pIntrinsic : PSGParser<IntrinsicInfo> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.Intrinsic info -> Matched info, state
        | _ -> NoMatch "Expected Intrinsic", state

/// Match a PlatformBinding node
let pPlatformBinding : PSGParser<string> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.PlatformBinding entryPoint -> Matched entryPoint, state
        | _ -> NoMatch "Expected PlatformBinding", state

/// Match a Binding node
let pBinding : PSGParser<string * bool * bool * bool> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.Binding (name, isMut, isRec, isEntry) ->
            Matched (name, isMut, isRec, isEntry), state
        | _ -> NoMatch "Expected Binding", state

/// Match a Lambda node (params are name*type*nodeId tuples for SSA assignment)
let pLambda : PSGParser<(string * FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NativeType * NodeId) list * NodeId> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.Lambda (params', bodyId, _captures, _, _) -> Matched (params', bodyId), state
        | _ -> NoMatch "Expected Lambda", state

/// Match an IfThenElse node
let pIfThenElse : PSGParser<NodeId * NodeId * NodeId option> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.IfThenElse (guard, thenB, elseOpt) ->
            Matched (guard, thenB, elseOpt), state
        | _ -> NoMatch "Expected IfThenElse", state

/// Match a WhileLoop node
let pWhileLoop : PSGParser<NodeId * NodeId> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.WhileLoop (guard, body) -> Matched (guard, body), state
        | _ -> NoMatch "Expected WhileLoop", state

/// Match a ForLoop node
let pForLoop : PSGParser<string * NodeId * NodeId * bool * NodeId> =
    fun state ->
        match state.Current.Kind with
        | SemanticKind.ForLoop (var, start, finish, isUp, body) ->
            Matched (var, start, finish, isUp, body), state
        | _ -> NoMatch "Expected ForLoop", state

// ═══════════════════════════════════════════════════════════════════════════
// COMBINATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Bind: if p succeeds, run binder on result
let bind (p: PSGParser<'A>) (binder: 'A -> PSGParser<'B>) : PSGParser<'B> =
    fun state ->
        match p state with
        | Matched a, state' -> binder a state'
        | NoMatch reason, state' -> NoMatch reason, state'

/// Infix bind operator
let (>>=) p binder = bind p binder

/// Map: transform successful result
let map (f: 'A -> 'B) (p: PSGParser<'A>) : PSGParser<'B> =
    fun state ->
        match p state with
        | Matched a, state' -> Matched (f a), state'
        | NoMatch reason, state' -> NoMatch reason, state'

/// Infix map operator
let (|>>) p f = map f p

/// Sequence: run both parsers, return second result
let (>>.) (p1: PSGParser<'A>) (p2: PSGParser<'B>) : PSGParser<'B> =
    fun state ->
        match p1 state with
        | Matched _, state' -> p2 state'
        | NoMatch reason, state' -> NoMatch reason, state'

/// Sequence: run both parsers, return first result
let (.>>) (p1: PSGParser<'A>) (p2: PSGParser<'B>) : PSGParser<'A> =
    fun state ->
        match p1 state with
        | Matched a, state' ->
            match p2 state' with
            | Matched _, state'' -> Matched a, state''
            | NoMatch reason, state'' -> NoMatch reason, state''
        | NoMatch reason, state' -> NoMatch reason, state'

/// Sequence: run both parsers, return both results
let (.>>.) (p1: PSGParser<'A>) (p2: PSGParser<'B>) : PSGParser<'A * 'B> =
    fun state ->
        match p1 state with
        | Matched a, state' ->
            match p2 state' with
            | Matched b, state'' -> Matched (a, b), state''
            | NoMatch reason, state'' -> NoMatch reason, state''
        | NoMatch reason, state' -> NoMatch reason, state'

/// Choice: try first parser, if fails try second
let (<|>) (p1: PSGParser<'A>) (p2: PSGParser<'A>) : PSGParser<'A> =
    fun state ->
        match p1 state with
        | Matched a, state' -> Matched a, state'
        | NoMatch _, _ -> p2 state  // Reset to original state

/// Optional: match zero or one
let opt (p: PSGParser<'A>) : PSGParser<'A option> =
    fun state ->
        match p state with
        | Matched a, state' -> Matched (Some a), state'
        | NoMatch _, _ -> Matched None, state

/// Satisfy: match current node if predicate holds
let satisfy (predicate: SemanticNode -> bool) : PSGParser<SemanticNode> =
    fun state ->
        if predicate state.Current then
            Matched state.Current, state
        else
            NoMatch "Predicate not satisfied", state

/// Satisfy with extractor: match and extract if predicate holds
let satisfyMap (f: SemanticNode -> 'T option) : PSGParser<'T> =
    fun state ->
        match f state.Current with
        | Some v -> Matched v, state
        | None -> NoMatch "SatisfyMap returned None", state

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION COMBINATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Focus on a child node by ID
let focusChild (childId: NodeId) : PSGParser<SemanticNode> =
    fun state ->
        match SemanticGraph.tryGetNode childId state.Graph with
        | Some childNode ->
            let state' = { state with Current = childNode }
            Matched childNode, state'
        | None ->
            NoMatch (sprintf "Child node %A not found" childId), state

/// Run parser on a specific child node, then restore focus
let onChild (childId: NodeId) (p: PSGParser<'T>) : PSGParser<'T> =
    fun state ->
        match SemanticGraph.tryGetNode childId state.Graph with
        | Some childNode ->
            let childState = { state with Current = childNode }
            match p childState with
            | Matched v, childState' ->
                // Restore original current but keep zipper changes
                Matched v, { childState' with Current = state.Current }
            | NoMatch reason, _ -> NoMatch reason, state
        | None ->
            NoMatch (sprintf "Child node %A not found" childId), state

/// Run parser on each child and collect results
let onChildren (childIds: NodeId list) (p: PSGParser<'T>) : PSGParser<'T list> =
    fun state ->
        let rec loop ids acc currentState =
            match ids with
            | [] -> Matched (List.rev acc), currentState
            | id :: rest ->
                match SemanticGraph.tryGetNode id state.Graph with
                | Some childNode ->
                    let childState = { currentState with Current = childNode }
                    match p childState with
                    | Matched v, childState' ->
                        loop rest (v :: acc) { childState' with Current = state.Current }
                    | NoMatch reason, _ ->
                        NoMatch reason, state
                | None ->
                    NoMatch (sprintf "Child node %A not found" id), state
        loop childIds [] state

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC CLASSIFICATION PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Match an intrinsic by module
let pIntrinsicModule (expectedModule: IntrinsicModule) : PSGParser<IntrinsicInfo> =
    pIntrinsic >>= fun info ->
        if info.Module = expectedModule then preturn info
        else pfail (sprintf "Expected module %A but got %A" expectedModule info.Module)

/// Match an intrinsic by full name pattern
let pIntrinsicNamed (fullName: string) : PSGParser<IntrinsicInfo> =
    pIntrinsic >>= fun info ->
        if info.FullName = fullName then preturn info
        else pfail (sprintf "Expected %s but got %s" fullName info.FullName)

/// Classify intrinsic by category for emission dispatch
type EmissionCategory =
    | BinaryArith of mlirOp: string
    | UnaryArith of mlirOp: string
    | Comparison of mlirOp: string
    | MemoryOp of op: string
    | StringOp of op: string
    // NOTE: ConsoleOp removed - Console is NOT an intrinsic, it's Layer 3 user code
    // in Fidelity.Platform that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md
    | PlatformOp of op: string
    | DateTimeOp of op: string
    | TimeSpanOp of op: string
    | OtherIntrinsic

let classifyIntrinsic (info: IntrinsicInfo) : EmissionCategory =
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
    // NOTE: Console is NOT an intrinsic - see fsnative-spec/spec/platform-bindings.md
    // Platform (Sys.* intrinsics)
    | IntrinsicModule.Sys, op -> PlatformOp op
    // DateTime operations
    | IntrinsicModule.DateTime, op -> DateTimeOp op
    // TimeSpan operations
    | IntrinsicModule.TimeSpan, op -> TimeSpanOp op
    | _ -> OtherIntrinsic

/// Match and classify intrinsic
let pClassifiedIntrinsic : PSGParser<IntrinsicInfo * EmissionCategory> =
    pIntrinsic |>> fun info -> (info, classifyIntrinsic info)

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTATION EXPRESSION
// ═══════════════════════════════════════════════════════════════════════════

type PSGParserBuilder() =
    member _.Bind(p, f) = bind p f
    member _.Return(x) = preturn x
    member _.ReturnFrom(p) = p
    member _.Zero() = preturn ()
    member _.Combine(p1, p2) = p1 >>. p2
    member _.Delay(f) = fun state -> f () state

let psg = PSGParserBuilder()

// ═══════════════════════════════════════════════════════════════════════════
// RUNNER
// ═══════════════════════════════════════════════════════════════════════════

/// Run a parser on a node with platform context
/// Platform info flows through state for type resolution
let runParser (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    let state = { Graph = graph; Zipper = zipper; Current = node; Platform = platform }
    parser state

/// Try to match a pattern, returning option
/// Platform info flows through state for type resolution
let tryMatch (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    match runParser parser graph node zipper platform with
    | Matched v, state' -> Some (v, state'.Zipper)
    | NoMatch _, _ -> None

/// Create initial parser state from zipper (convenience)
/// Extracts platform from zipper state - this is the common pattern
let stateFromZipper (zipper: PSGZipper) (node: SemanticNode) : PSGParserState =
    {
        Graph = zipper.Graph
        Zipper = zipper
        Current = node
        Platform = zipper.State.Platform
    }

/// Run a parser using zipper (most common usage)
/// Platform is extracted from zipper.State.Platform
let runParserWithZipper (parser: PSGParser<'T>) (zipper: PSGZipper) (node: SemanticNode) =
    let state = stateFromZipper zipper node
    parser state
