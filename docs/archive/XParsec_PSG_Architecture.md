# XParsec, Active Patterns, and Zippers: Right Tool for Right Job

## Architectural Philosophy

The Firefly compiler uses multiple tools for PSG traversal and MLIR generation, each appropriate for its specific purpose:

1. **Active Patterns** - F#'s native idiom for single-node structural recognition
2. **XParsec** - Parser combinators for sequential child parsing with backtracking
3. **PSGZipper** - Bidirectional tree traversal with context preservation
4. **Bindings** - Platform-specific MLIR generation (data, not routing)

**The Principle: Use the simplest tool that expresses the intent.**

| Concern | Right Tool | Why |
|---------|-----------|-----|
| **Single-node classification** | Active patterns / DU matching | Structure is known; recognition is local |
| **Sequential child parsing** | XParsec combinators | May need backtracking; variable length |
| **Tree navigation** | Zipper | Bidirectional; context-preserving |
| **State threading** | Accumulator in fold | Functional; explicit |

## Current Implementation Status (December 2024)

**Transfer.fs** already follows this philosophy correctly:
```fsharp
match node.Kind with
| SKExpr EConst -> witnessConst acc node
| SKExpr EIdent | SKExpr ELongIdent -> witnessIdent acc node
| SKBinding _ -> witnessBinding acc node
```

The typed `SyntaxKindT` discriminated union provides structural recognition without string matching.

**Baker** uses FCS's built-in active patterns correctly:
```fsharp
match expr with
| FSharpFieldGet(_, containingType, field) -> Some { ... }
| _ -> None
```

## The Zipper: Non-Negotiable Core Component

**The PSGZipper is not optional. It is the traversal engine for the PSG.**

From "Hyping Hypergraphs":
> "The bidirectional zipper that traverses the hypergraph becomes a powerful tool for maintaining proofs during transformation. As it navigates through the graph, it carries both the current focus and the proof context."

From "Speed And Safety With Graph Coloring":
> "Our bidirectional zipper traversal enables sophisticated pattern matching across both control flow and data flow"

From "Proof-Aware Compilation":
> "The bidirectional zipper that traverses the hypergraph becomes a powerful tool for maintaining proofs during transformation... Forward navigation accumulates proof obligations... Backward navigation propagates postconditions... Lateral movement discovers related proofs... Context preservation ensures transformations maintain required properties"

The zipper provides:
- **Focus**: The current node being examined
- **Path**: The trail of "breadcrumbs" back to root (parent chain with sibling context)
- **Graph Context**: Access to the full PSG for node lookups
- **Generation State**: SSA counter, locals, types - threaded through traversal
- **Bidirectional Navigation**: up/down/left/right movement with context preservation

### PSGZipper Structure

```fsharp
type PSGPath =
    | Top
    | BindingChild of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath
    | AppArg of func: PSGNode * argIndex: int * otherArgs: NodeId list * path: PSGPath
    | SequenceItem of preceding: NodeId list * following: NodeId list * parent: PSGNode * path: PSGPath
    | MatchCase of scrutinee: PSGNode * caseIndex: int * otherCases: NodeId list * path: PSGPath
    | Child of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath

type PSGZipper = {
    Focus: PSGNode
    Path: PSGPath
    Graph: ProgramSemanticGraph
    State: GenerationState
}
```

### Zipper Operations

```fsharp
val down: PSGZipper -> NavResult<PSGZipper>      // Move to first child
val up: PSGZipper -> NavResult<PSGZipper>        // Move to parent
val left: PSGZipper -> NavResult<PSGZipper>      // Move to previous sibling
val right: PSGZipper -> NavResult<PSGZipper>     // Move to next sibling
val downTo: int -> PSGZipper -> NavResult<PSGZipper>  // Move to nth child
val children: PSGZipper -> NodeId list           // Get child IDs
val childNodes: PSGZipper -> PSGNode list        // Get child nodes
val ancestors: PSGZipper -> PSGNode list         // Get ancestor chain
```

## The Architectural Insight: XParsec Operates Within the Zipper

XParsec is designed for sequential input parsing. The PSG is a tree. But **at each zipper position, children ARE sequential**. This is the key insight.

**XParsec's job is NOT to traverse the tree. The zipper does that.**

**XParsec's job IS to provide principled combinators for:**
1. Matching patterns at the current focus
2. Matching patterns across the sequential list of children
3. Composing these matches monadically
4. Threading state and handling failure/backtracking

## Architecture: XParsec Layered on Zipper

### Layer 1: XParsec Integration Types

```fsharp
/// The "input" for XParsec when parsing PSG children
[<Struct>]
type PSGChildSlice(nodes: PSGNode array, start: int, length: int) =
    interface IReadable<PSGNode, PSGChildSlice> with
        member _.Item index = nodes.[start + int index]
        member _.TryItem index =
            let i = start + int index
            if i < start + length then ValueSome nodes.[i]
            else ValueNone
        member _.Length = int64 length
        member _.Slice(newStart, newLength) =
            PSGChildSlice(nodes, start + int newStart, int newLength)
        member _.SpanSlice(index, count) =
            ReadOnlySpan(nodes, start + int index, count)

/// State threaded through XParsec parsing
type PSGParseState = {
    /// The zipper provides navigation context
    Zipper: PSGZipper
    /// Additional context for generation
    AdditionalContext: Map<string, obj>
}

/// XParsec parser specialized for PSG child sequences
type PSGChildParser<'T> =
    XParsec.Parser<'T, PSGNode, PSGParseState, PSGChildSlice, PSGChildSlice>
```

### Layer 2: Bridging Zipper and XParsec

The zipper navigates the tree. When we want to match against children, we use XParsec:

```fsharp
module PSGXParsec =
    open XParsec
    open XParsec.Parsers
    open XParsec.Combinators

    /// Run an XParsec parser on the children of the current zipper focus
    let parseChildren (parser: PSGChildParser<'T>) (zipper: PSGZipper) : Result<'T, string> =
        let children = PSGZipper.childNodes zipper |> Array.ofList
        let input = PSGChildSlice(children, 0, children.Length)
        let state = { Zipper = zipper; AdditionalContext = Map.empty }
        let reader = XParsec.Reader(input, state, 0L)

        match parser reader with
        | Ok success -> Ok success.Parsed
        | Error err -> Error (sprintf "Parse failed at position %d" err.Position.Index)

    /// Match if current child satisfies predicate
    let satisfyChild (pred: PSGNode -> bool) : PSGChildParser<PSGNode> =
        XParsec.Parsers.satisfy pred

    /// Match child with specific syntax kind prefix
    let childKind (prefix: string) : PSGChildParser<PSGNode> =
        satisfyChild (fun n -> n.SyntaxKind.StartsWith(prefix))

    /// Skip a child and continue
    let skipChild : PSGChildParser<unit> =
        satisfyChild (fun _ -> true) >>% ()

    /// Get remaining children
    let remainingChildren : PSGChildParser<PSGNode list> =
        XParsec.Combinators.many (satisfyChild (fun _ -> true))
        |>> List.ofSeq
```

### Layer 3: Pattern Type Becomes XParsec-Based

Instead of `Pattern<'T> = PSGZipper -> 'T option`, patterns become:

```fsharp
/// A pattern that matches at the current zipper focus
/// Uses XParsec result type for consistency
type FocusPattern<'T> = PSGZipper -> XParsec.ParseResult<'T, PSGNode, PSGParseState>

/// A pattern that matches against children using XParsec
type ChildrenPattern<'T> = PSGChildParser<'T>

module FocusPattern =
    open XParsec

    /// Succeed with value
    let succeed (value: 'T) : FocusPattern<'T> =
        fun _ -> Ok { Parsed = value }

    /// Fail with message
    let fail (msg: string) : FocusPattern<'T> =
        fun z ->
            let state = { Zipper = z; AdditionalContext = Map.empty }
            Error { Position = { Id = ReaderId 0L; Index = 0L; State = state }
                    Errors = Message msg }

    /// Match if focus satisfies predicate
    let satisfies (pred: PSGNode -> bool) : FocusPattern<PSGNode> =
        fun z ->
            if pred z.Focus then Ok { Parsed = z.Focus }
            else fail "Focus does not satisfy predicate" z

    /// Match syntax kind prefix
    let syntaxKind (prefix: string) : FocusPattern<PSGNode> =
        satisfies (fun n -> n.SyntaxKind.StartsWith(prefix))

    /// Bind (monadic composition) - uses XParsec's bind semantics
    let bind (f: 'T -> FocusPattern<'U>) (p: FocusPattern<'T>) : FocusPattern<'U> =
        fun z ->
            match p z with
            | Ok success -> f success.Parsed z
            | Error e -> Error e

    let (>>=) p f = bind f p
    let (|>>) p f = bind (fun x -> succeed (f x)) p
    let (>>.) p1 p2 = p1 >>= fun _ -> p2
    let (.>>) p1 p2 = p1 >>= fun x -> p2 |>> fun _ -> x
    let (<|>) p1 p2 = fun z -> match p1 z with | Ok r -> Ok r | Error _ -> p2 z

    /// Navigate down and apply pattern to children using XParsec
    let withChildren (childParser: ChildrenPattern<'T>) : FocusPattern<'T> =
        fun z ->
            match PSGXParsec.parseChildren childParser z with
            | Ok result -> Ok { Parsed = result }
            | Error msg -> fail msg z
```

### Layer 4: MLIR Generation Uses XParsec State Threading

```fsharp
/// MLIR generation is now a PSG parser that produces MLIR fragments
type MLIRGen<'T> = PSGChildParser<'T>

module MLIRGen =
    open XParsec
    open XParsec.Parsers
    open XParsec.Combinators

    /// Access the zipper from parse state
    let getZipper : MLIRGen<PSGZipper> =
        fun reader -> preturn reader.State.Zipper reader

    /// Update generation state in zipper
    let updateState (f: GenerationState -> GenerationState) : MLIRGen<unit> =
        fun reader ->
            let newZipper = PSGZipper.mapState f reader.State.Zipper
            reader.State <- { reader.State with Zipper = newZipper }
            preturn () reader

    /// Generate fresh SSA name
    let freshSSA : MLIRGen<string> =
        fun reader ->
            let z = reader.State.Zipper
            let newZ, name = PSGZipper.nextSSA z
            reader.State <- { reader.State with Zipper = newZ }
            preturn name reader

    /// Generate MLIR line (side effect through builder in state)
    let line (text: string) : MLIRGen<unit> =
        fun reader ->
            // Actual generation goes through MLIRBuilder
            preturn () reader
```

### Layer 5: Expression Processing as XParsec Parser

```fsharp
/// Main expression processing - an XParsec parser over PSG children
let rec pExpr : MLIRGen<ExprResult> =
    // Use XParsec's choice combinator
    XParsec.Combinators.choiceL [
        // Function calls - dispatch to bindings
        pFunctionCall

        // Arithmetic expressions
        pArithmetic

        // Constants
        pConst

        // Variables
        pIdent

        // Sequential expressions
        pSequential

        // Fallback
        pUnhandled
    ] "Expected expression"

and pFunctionCall : MLIRGen<ExprResult> =
    // Match: App node with function and arguments
    PSGXParsec.childKind "App" >>= fun appNode ->
    // ... pattern continues - dispatch to platform bindings
    MLIRGen.getZipper >>= fun z ->
    // Navigate into app and process
    match PSGZipper.downTo 0 z with
    | NavOk innerZ ->
        // Use XParsec on inner's children
        PSGXParsec.parseChildren pCallTarget innerZ
        |> function
        | Ok callInfo -> processCall callInfo  // Dispatches to Bindings
        | Error _ -> XParsec.Parsers.pzero
    | NavFail _ -> XParsec.Parsers.pzero

and pArithmetic : MLIRGen<ExprResult> =
    // Match curried binary op: App[App[op, left], right]
    PSGXParsec.childKind "App" >>= fun innerApp ->
    // ... pattern continues
    MLIRGen.getZipper >>= fun z ->
    match PSGZipper.downTo 0 z with
    | NavOk innerZ ->
        PSGXParsec.parseChildren pBinaryOp innerZ
        |> function
        | Ok opInfo -> generateBinaryOp opInfo
        | Error _ -> XParsec.Parsers.pzero
    | NavFail _ -> XParsec.Parsers.pzero
```

## The Key Principle: Right Tool for Right Job

### 1. Active Patterns - Single-Node Recognition

F#'s native idiom for structural matching. Use for:
- Single-node classification in `witnessNode` dispatch
- Extracting typed data from nodes
- Composable with `&` and `|` in match expressions

```fsharp
// GOOD: Active pattern for single-node recognition
let (|App|_|) (node: PSGNode) =
    match node.Kind with SKExpr EApp -> Some node | _ -> None

let (|ArithOp|_|) (node: PSGNode) =
    match node.Operation with
    | Some (Arithmetic op) -> Some op
    | _ -> None

// Compose naturally in match expressions
match node with
| App & WithSymbol (_, sym) -> handleAppWithSymbol sym
| ArithOp Add -> handleAdd ()
| _ -> handleDefault ()
```

### 2. XParsec - Sequential Child Parsing

Use when parsing child sequences with:
- Variable-length lists (`pMany`, `pMany1`)
- Backtracking alternatives (`pOr`, `pChoice`)
- Ordered sequences (`pTuple3`, `pPipe4`)
- Separator-based lists (`pSepBy`)

```fsharp
// GOOD: XParsec for sequential child parsing
let pCallArgs = pMany1 (pOr pArgExpr pIdentExpr)
let pBinaryOp = pTuple3 pOperand pOperator pOperand
let pFieldList = pSepBy pField pSemicolon
```

**DO NOT use XParsec for single-node classification:**
```fsharp
// AVOID: XParsec wrapping a simple predicate
let appChild = satisfyChild (fun n -> SyntaxKindT.isApp n.Kind)

// PREFER: Active pattern
let (|AppNode|_|) node = if SyntaxKindT.isApp node.Kind then Some node else None
```

### 3. PSGZipper - Tree Navigation

The bidirectional zipper provides:
- **Focus**: Current node being examined
- **Path**: Breadcrumbs back to root
- **Graph Context**: Access to full PSG for lookups
- **State Threading**: SSA counter, locals, generation state
- **Bidirectional Navigation**: up/down/left/right with context

Use `foldPostOrder` for MLIR generation (children before parents).

### 4. Bindings - Platform-Specific MLIR

Platform differences are **data** (syscall numbers, register conventions), not routing logic:
- Organized by `(OSFamily, Architecture, EntryPoint)`
- Looked up by extern entry point
- Generate MLIR for syscalls, memory operations, etc.

## Remediation Path

### Phase 1: Evolve PSGPatterns.fs to Active Patterns (PRIORITY)

Current `PSGPatterns.fs` provides predicates for `satisfyChild`. Evolve to proper active patterns:

```fsharp
// FROM: Predicate style
let isApp (node: PSGNode) : bool = SyntaxKindT.isApp node.Kind

// TO: Active pattern style
let (|App|_|) (node: PSGNode) =
    match node.Kind with SKExpr EApp -> Some node | _ -> None

let (|ArithOp|_|) (node: PSGNode) =
    match node.Operation with
    | Some (Arithmetic op) -> Some op
    | _ -> None

let (|WithSymbol|_|) (node: PSGNode) =
    node.Symbol |> Option.map (fun s -> node, s)
```

### Phase 2: Slim Down PSGXParsec.fs

Current ~900 lines is heavier than necessary. **Keep only:**
- `PSGChildSlice` (the `IReadable` implementation for XParsec)
- `PSGParseState` (minimal state: SSA counters, focus ID)
- `parseChildren` / `tryParseChildren` (bridge functions)
- `EmitContext` (emission state - consider moving to separate file)
- Genuine sequential parsers for child sequences

**Remove:**
- Combinator wrappers (`pMap`, `pBind`, etc.) - users can import XParsec directly
- Single-node child matchers that just wrap predicates - use active patterns instead

### Phase 3: Use Active Patterns in Witness Functions (DONE)

Transfer.fs already does this correctly with `SyntaxKindT` DU matching. Extend where appropriate:

```fsharp
// Current (predicate lookup)
let funcIdentNode = children |> List.tryFind (fun c -> SyntaxKindT.isLongIdent c.Kind)

// Could become (active pattern)
let funcIdentNode = children |> List.tryPick (function LongIdent n -> Some n | _ -> None)
```

### Phase 4: Reserve XParsec for Genuine Sequential Parsing

Use XParsec only when you genuinely need:
- Variable-length child lists
- Backtracking between alternatives
- Complex sequential patterns

```fsharp
// GOOD: XParsec for parsing argument list
let pCallArgs = pMany1 (pOr pArgExpr pIdentExpr)

// UNNECESSARY: Single child lookup
// let appChild = satisfyChild (fun n -> SyntaxKindT.isApp n.Kind)
// Use active pattern instead
```

## Files Status

| File | Status | Notes |
|------|--------|-------|
| `Alex/Traversal/PSGXParsec.fs` | **Needs slimming** | Remove combinator wrappers, keep XParsec bridge |
| `Alex/Patterns/PSGPatterns.fs` | **Evolve to active patterns** | Convert predicates to `(\|...\|_\|)` form |
| `Alex/Generation/Transfer.fs` | **Already correct** | Uses typed DU matching |
| `Alex/Traversal/PSGZipper.fs` | **Complete** | Tree navigation via fold |
| `Alex/Traversal/MLIRZipper.fs` | **Complete** | MLIR composition |
| `Alex/Bindings/*` | **Working** | Platform-specific MLIR generation |

## What This Achieves

1. **Right tool for right job**: Active patterns for recognition, XParsec for parsing, Zipper for navigation
2. **F# idioms respected**: Active patterns are the native F# way to recognize structure
3. **XParsec preserved**: Sequential child parsing with backtracking where genuinely needed
4. **Zipper preserved**: Tree navigation with context remains fundamental
5. **Thinner glue layer**: PSGXParsec.fs becomes a bridge, not a framework
6. **Composability**: Both active patterns (`&`, `|`) and XParsec compose naturally
7. **Type Safety**: Typed DU matching eliminates string-based dispatch

## Conclusion

The architecture uses three complementary tools:

- **Active Patterns** = single-node structural recognition (F# idiom)
- **XParsec** = sequential child parsing with backtracking
- **Zipper** = tree navigation with context preservation
- **Bindings** = platform-specific MLIR generation (data, not routing)

Each tool is appropriate for its specific purpose. The tension is resolved not by choosing one over another, but by recognizing where each naturally applies. Transfer.fs already demonstrates the correct approach with typed DU matching; the remediation extends this clarity to PSGPatterns.fs and slims PSGXParsec.fs to its essential role as a bridge to XParsec's sequential parsing capabilities.
