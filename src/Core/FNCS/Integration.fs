/// FNCS Integration Layer
/// Thin interface between Firefly and F# Native Compiler Services.
///
/// FNCS provides:
/// - Native type checking with types attached during construction
/// - SRTP resolution during type checking (not post-hoc)
/// - Hard-pruned SemanticGraph (only reachable nodes)
/// - No BCL types, no IL imports, no obj
/// - Baker enrichment (module classification metadata)
///
/// Firefly receives SemanticGraph and applies:
/// - Lowering nanopasses (FlattenApplications, LowerStrings, etc.)
/// - Alex emission (Zipper + XParsec + Bindings → MLIR)
module Core.FNCS.Integration

// Re-export FNCS types for use throughout Firefly
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open FSharp.Native.Compiler.NativeService
// Baker - Post-construction semantic enrichment (peer to Checking.Native)
// ModuleClassifications now computed lazily on SemanticGraph itself
module BakerPipeline = FSharp.Native.Compiler.Baker.Pipeline

/// Type aliases for cleaner code
type FNCSNode = SemanticNode
type FNCSGraph = SemanticGraph
type FNCSType = NativeType
type FNCSKind = SemanticKind
type FNCSNodeId = NodeId
type FNCSLiteralValue = LiteralValue

/// Check result from FNCS
type FNCSCheckResult = CheckResult

/// Module classification (computed lazily on SemanticGraph)
type FNCSModuleClassification = ModuleClassification

/// Extract int from NodeId
let nodeIdToInt (NodeId id) = id

/// Get the ID of a node
let nodeId (node: FNCSNode) : FNCSNodeId = node.Id

/// Get the type of a node
let nodeType (node: FNCSNode) : FNCSType = node.Type

/// Get the kind of a node
let nodeKind (node: FNCSNode) : FNCSKind = node.Kind

/// Get the SRTP resolution if present
let nodeSRTP (node: FNCSNode) : WitnessResolution option = node.SRTPResolution

/// Get children of a node
let nodeChildren (node: FNCSNode) : NodeId list = node.Children

/// Get a node by ID
let getNode (id: NodeId) (graph: FNCSGraph) : FNCSNode option =
    SemanticGraph.tryGetNode id graph

/// Get a node by ID (throws if not found)
let getNodeExn (id: NodeId) (graph: FNCSGraph) : FNCSNode =
    SemanticGraph.getNode id graph

/// Fold over graph in post-order (children before parents - required for SSA emission)
let foldPostOrder (folder: 'State -> FNCSNode -> 'State) (state: 'State) (graph: FNCSGraph) : 'State =
    Traversal.foldPostOrder folder state graph

/// Fold over graph in pre-order
let foldPreOrder (folder: 'State -> FNCSNode -> 'State) (state: 'State) (graph: FNCSGraph) : 'State =
    Traversal.foldPreOrder folder state graph

/// Map over all nodes in the graph
let mapNodes (f: FNCSNode -> FNCSNode) (graph: FNCSGraph) : FNCSGraph =
    Traversal.map f graph

/// Filter nodes in the graph
let filterNodes (predicate: FNCSNode -> bool) (graph: FNCSGraph) : FNCSGraph =
    Traversal.filter predicate graph

/// Get all binding nodes
let bindings (graph: FNCSGraph) : FNCSNode list =
    SemanticGraph.bindings graph

/// Check if result has errors
let hasErrors (result: FNCSCheckResult) : bool =
    CheckResult.hasErrors result

/// Get error diagnostics
let errors (result: FNCSCheckResult) : Diagnostic list =
    CheckResult.errors result

/// Get warning diagnostics
let warnings (result: FNCSCheckResult) : Diagnostic list =
    CheckResult.warnings result

/// Format a type for display
let formatTypeStr (ty: FNCSType) : string =
    formatType ty

/// Check if a node is a platform binding
let isPlatformBinding (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.PlatformBinding _ -> true
    | _ -> false

/// Get platform binding name if present
let platformBindingName (node: FNCSNode) : string option =
    match node.Kind with
    | SemanticKind.PlatformBinding name -> Some name
    | _ -> None

/// Check if a node is a function application
let isApplication (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Application _ -> true
    | _ -> false

/// Check if a node is a binding (let/do)
let isBinding (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Binding _ -> true
    | _ -> false

/// Get binding info (name, isMutable, isRecursive) if present
let bindingInfo (node: FNCSNode) : (string * bool * bool) option =
    match node.Kind with
    | SemanticKind.Binding (name, isMutable, isRec, _isEntryPoint) -> Some (name, isMutable, isRec)
    | _ -> None

/// Check if a node is a literal
let isLiteral (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Literal _ -> true
    | _ -> false

/// Get literal value if present
let literalValue (node: FNCSNode) : LiteralValue option =
    match node.Kind with
    | SemanticKind.Literal v -> Some v
    | _ -> None

/// Check if a node is a lambda
let isLambda (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Lambda _ -> true
    | _ -> false

/// Check if a node is a variable reference
let isVarRef (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.VarRef _ -> true
    | _ -> false

/// Get variable reference name if present
let varRefName (node: FNCSNode) : string option =
    match node.Kind with
    | SemanticKind.VarRef (name, _) -> Some name
    | _ -> None

/// Get variable reference definition node ID if present
let varRefDefinition (node: FNCSNode) : FNCSNodeId option =
    match node.Kind with
    | SemanticKind.VarRef (_, defId) -> defId
    | _ -> None

/// Check if type is a function type
let isFunType (ty: FNCSType) : bool =
    isFunctionType ty

/// Check if type is a type variable
let isTypeVariable (ty: FNCSType) : bool =
    isTypeVar ty

/// Check if a node is an interpolated string
let isInterpolatedString (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.InterpolatedString _ -> true
    | _ -> false

/// Get interpolated string parts if present
let interpolatedStringParts (node: FNCSNode) : InterpolatedPart list option =
    match node.Kind with
    | SemanticKind.InterpolatedString parts -> Some parts
    | _ -> None

/// Re-export InterpolatedPart type for use in Firefly
type FNCSInterpolatedPart = InterpolatedPart

// ═══════════════════════════════════════════════════════════════════════════
// Parsing API - Exposed for ProjectLoader
// ═══════════════════════════════════════════════════════════════════════════

/// Parse options for source files
type FNCSParseOptions = ParseOptions

/// Parse result
type FNCSParseResult = ParseResult

/// Combined parse and check result
type FNCSParseAndCheckResult = ParseAndCheckResult

/// Default parse options (from NativeService module)
let defaultFNCSParseOptions : FNCSParseOptions = FSharp.Native.Compiler.NativeService.defaultParseOptions

/// Parse a source string with default options
let parseSource (source: string) (fileName: string) : FNCSParseResult =
    parseStringWithDefaults source fileName

/// Parse a source string with custom options
let parseSourceWithOptions (source: string) (fileName: string) (options: FNCSParseOptions) : FNCSParseResult =
    parseString source fileName options

/// Parse and type-check a source string in one step
let parseAndCheckSource (source: string) (fileName: string) : FNCSParseAndCheckResult =
    parseAndCheck source fileName

/// Check multiple parsed inputs together with shared type environment
/// This is the correct way to compile multi-file projects - type abbreviations,
/// bindings, etc. from earlier files are visible when checking later files.
let checkMultipleInputs (inputs: FSharp.Native.Compiler.Syntax.ParsedInput list) : FNCSCheckResult =
    checkParsedInputs inputs

/// Check if parse result succeeded
let parseSucceeded (result: FNCSParseResult) : bool =
    match result with
    | ParseSuccess _ -> true
    | ParseError _ -> false

/// Get parsed input from successful parse
let parsedInput (result: FNCSParseResult) : FSharp.Native.Compiler.Syntax.ParsedInput option =
    match result with
    | ParseSuccess input -> Some input
    | ParseError _ -> None

/// Get parse errors
let parseErrors (result: FNCSParseResult) : string list =
    match result with
    | ParseSuccess _ -> []
    | ParseError errs -> errs

// ═══════════════════════════════════════════════════════════════════════════
// Module Classification - Coeffect Observation (computed lazily on PSG)
// ═══════════════════════════════════════════════════════════════════════════

/// Emit Baker intermediates to a directory (for -k flag debug output).
let emitBakerIntermediates (graph: FNCSGraph) (outputDir: string) (baseName: string) : unit =
    BakerPipeline.emitIntermediates graph outputDir baseName

/// Get module classification for a specific module (observes lazy field).
let getModuleClassification (moduleId: FNCSNodeId) (graph: FNCSGraph) : FNCSModuleClassification option =
    Map.tryFind moduleId graph.ModuleClassifications.Value

/// Get all module classifications (observes lazy field).
let allModuleClassifications (graph: FNCSGraph) : Map<FNCSNodeId, FNCSModuleClassification> =
    graph.ModuleClassifications.Value

let moduleInitBindings (classification: FNCSModuleClassification) : FNCSNodeId list =
    classification.ModuleInit

let moduleDefinitions (classification: FNCSModuleClassification) : FNCSNodeId list =
    classification.Definitions

let moduleEntryPoint (classification: FNCSModuleClassification) : FNCSNodeId option =
    classification.EntryPoint
