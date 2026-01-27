/// PSG Combinators - XParsec-based pattern matching for SemanticGraph nodes
///
/// This module provides composable parsers for recognizing PSG node patterns
/// using the actual XParsec library (not reimplementing combinators by hand).
///
/// Key insight: XParsec works on sequential input. We adapt it to work on
/// PSG structure by threading our custom state (current node, graph, zipper).
module Alex.XParsec.PSGCombinators

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// USER STATE - threaded through XParsec
// ═══════════════════════════════════════════════════════════════════════════

/// Parser state threaded through pattern matching
///
/// PLATFORM-AWARE (January 2026):
/// Platform info flows through parser state, enabling type resolution
/// without hard-coded values in witnesses.
type PSGParserState = {
    Graph: SemanticGraph
    Zipper: PSGZipper
    /// Currently focused node
    Current: SemanticNode
    /// Platform resolution result (architecture, OS, word type, bindings)
    Platform: PlatformResolutionResult
}

// ═══════════════════════════════════════════════════════════════════════════
// PARSER TYPE - Using XParsec's Parser with our custom state
// ═══════════════════════════════════════════════════════════════════════════

/// A PSG parser using XParsec's infrastructure
/// Parser<'Parsed, 'T, 'State, 'Input, 'InputSlice>
/// We use: char as token type, PSGParserState as state, ReadableString as input
type PSGParser<'T> = Parser<'T, char, PSGParserState, ReadableString, ReadableStringSlice>

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM-AWARE TYPE RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Get the word-sized integer type for the target platform
let platformWordType : PSGParser<MLIRType> =
    getUserState |>> fun state -> state.Platform.PlatformWordType

/// Get the word width in bits for the target platform
let platformWordBits : PSGParser<int> =
    getUserState |>> fun state ->
        match platformWordWidth state.Platform.TargetArch with
        | I64 -> 64
        | I32 -> 32
        | I16 -> 16
        | I8 -> 8
        | I1 -> 1

/// Get the target architecture
let targetArch : PSGParser<Architecture> =
    getUserState |>> fun state -> state.Platform.TargetArch

/// Map NTUKind to MLIRType with platform awareness
let mapNTUKindForPlatform (kind: NTUKind) : PSGParser<MLIRType> =
    getUserState |>> fun state ->
        Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType state.Platform.TargetArch kind

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE PARSERS - Match PSG node kinds
// ═══════════════════════════════════════════════════════════════════════════

/// Get the current PSG node
let getCurrentNode : PSGParser<SemanticNode> =
    getUserState |>> fun state -> state.Current

/// Match a specific SemanticKind
let pKind (expected: SemanticKind) : PSGParser<SemanticNode> =
    getUserState >>= fun state ->
        if state.Current.Kind = expected then
            preturn state.Current
        else
            fail (Message (sprintf "Expected %A but got %A" expected state.Current.Kind))

/// Match a Literal node
let pLiteral : PSGParser<NativeLiteral> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.Literal lit -> preturn lit
        | _ -> fail (Message "Expected Literal")

/// Match a VarRef node
let pVarRef : PSGParser<string * NodeId option> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.VarRef (name, defIdOpt) -> preturn (name, defIdOpt)
        | _ -> fail (Message "Expected VarRef")

/// Match an Application node
let pApplication : PSGParser<NodeId * NodeId list> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.Application (funcId, argIds) -> preturn (funcId, argIds)
        | _ -> fail (Message "Expected Application")

/// Match an Intrinsic node
let pIntrinsic : PSGParser<IntrinsicInfo> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.Intrinsic info -> preturn info
        | _ -> fail (Message "Expected Intrinsic")

/// Match a PlatformBinding node
let pPlatformBinding : PSGParser<string> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.PlatformBinding entryPoint -> preturn entryPoint
        | _ -> fail (Message "Expected PlatformBinding")

/// Match a Binding node
let pBinding : PSGParser<string * bool * bool * bool> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.Binding (name, isMut, isRec, isEntry) ->
            preturn (name, isMut, isRec, isEntry)
        | _ -> fail (Message "Expected Binding")

/// Match a Lambda node
let pLambda : PSGParser<(string * NativeType * NodeId) list * NodeId> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.Lambda (params', bodyId, _, _, _) -> preturn (params', bodyId)
        | _ -> fail (Message "Expected Lambda")

/// Match an IfThenElse node
let pIfThenElse : PSGParser<NodeId * NodeId * NodeId option> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.IfThenElse (guard, thenB, elseOpt) ->
            preturn (guard, thenB, elseOpt)
        | _ -> fail (Message "Expected IfThenElse")

/// Match a WhileLoop node
let pWhileLoop : PSGParser<NodeId * NodeId> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.WhileLoop (guard, body) -> preturn (guard, body)
        | _ -> fail (Message "Expected WhileLoop")

/// Match a ForLoop node
let pForLoop : PSGParser<string * NodeId * NodeId * bool * NodeId> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.ForLoop (var, start, finish, isUp, body) ->
            preturn (var, start, finish, isUp, body)
        | _ -> fail (Message "Expected ForLoop")

/// Match a LazyExpr node (deferred computation)
let pLazyExpr : PSGParser<NodeId * CaptureInfo list> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.LazyExpr (bodyId, captures) -> preturn (bodyId, captures)
        | _ -> fail (Message "Expected LazyExpr")

/// Match a LazyForce node (force lazy evaluation)
let pLazyForce : PSGParser<NodeId> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.LazyForce lazyValueId -> preturn lazyValueId
        | _ -> fail (Message "Expected LazyForce")

/// Match a DUGetTag node
let pDUGetTag : PSGParser<NodeId * NativeType> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.DUGetTag (duValueId, duType) -> preturn (duValueId, duType)
        | _ -> fail (Message "Expected DUGetTag")

/// Match a DUEliminate node
let pDUEliminate : PSGParser<NodeId * int * string * NativeType> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.DUEliminate (duValueId, caseIndex, caseName, payloadType) ->
            preturn (duValueId, caseIndex, caseName, payloadType)
        | _ -> fail (Message "Expected DUEliminate")

/// Match a DUConstruct node
let pDUConstruct : PSGParser<string * int * NodeId option * NodeId option> =
    getUserState >>= fun state ->
        match state.Current.Kind with
        | SemanticKind.DUConstruct (caseName, caseIndex, payload, arenaHint) ->
            preturn (caseName, caseIndex, payload, arenaHint)
        | _ -> fail (Message "Expected DUConstruct")

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION COMBINATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Focus on a child node by ID
let focusChild (childId: NodeId) : PSGParser<SemanticNode> =
    getUserState >>= fun state ->
        match SemanticGraph.tryGetNode childId state.Graph with
        | Some childNode ->
            setUserState { state with Current = childNode } >>. preturn childNode
        | None ->
            fail (Message (sprintf "Child node %A not found" childId))

/// Run parser on a specific child node, then restore focus
let onChild (childId: NodeId) (p: PSGParser<'T>) : PSGParser<'T> =
    getUserState >>= fun originalState ->
        match SemanticGraph.tryGetNode childId originalState.Graph with
        | Some childNode ->
            setUserState { originalState with Current = childNode } >>.
            p .>>
            setUserState originalState
        | None ->
            fail (Message (sprintf "Child node %A not found" childId))

/// Run parser on each child and collect results
let onChildren (childIds: NodeId list) (p: PSGParser<'T>) : PSGParser<'T list> =
    let rec loop ids acc =
        match ids with
        | [] -> preturn (List.rev acc)
        | id :: rest ->
            onChild id p >>= fun result ->
            loop rest (result :: acc)
    loop childIds []

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Match an intrinsic by module
let pIntrinsicModule (expectedModule: IntrinsicModule) : PSGParser<IntrinsicInfo> =
    pIntrinsic >>= fun info ->
        if info.Module = expectedModule then preturn info
        else fail (Message (sprintf "Expected module %A but got %A" expectedModule info.Module))

/// Match an intrinsic by full name pattern
let pIntrinsicNamed (fullName: string) : PSGParser<IntrinsicInfo> =
    pIntrinsic >>= fun info ->
        if info.FullName = fullName then preturn info
        else fail (Message (sprintf "Expected %s but got %s" fullName info.FullName))

// ═══════════════════════════════════════════════════════════════════════════
// RUNNER - Initialize state and run parser
// ═══════════════════════════════════════════════════════════════════════════

/// Run a parser on a node with platform context
let runParser (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    let initialState = { Graph = graph; Zipper = zipper; Current = node; Platform = platform }
    // XParsec requires input, but we don't parse strings - we just thread state
    // So we run with empty input
    let reader = Reader.ofString "" initialState
    parser reader

/// Try to match a pattern, returning option
let tryMatch (parser: PSGParser<'T>) (graph: SemanticGraph) (node: SemanticNode) (zipper: PSGZipper) (platform: PlatformResolutionResult) =
    match runParser parser graph node zipper platform with
    | Ok result -> Some result.Parsed
    | Error _ -> None
