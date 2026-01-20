/// PSGZipper - True Huet Zipper for PSG Navigation
///
/// A zipper is a data structure for navigating and modifying immutable trees.
/// It pairs your current focus with a path (breadcrumbs) showing how you got there.
///
/// References:
/// - Huet, "The Zipper" (1997) - original paper
/// - Tomas Petricek, "Tree zipper query expressions" - F# computation expressions
/// - Mark Seemann, "Zippers" (2024) - bidirectional navigation in functional code
///
/// ARCHITECTURAL ROLE:
/// - Focus: The current SemanticNode we're examining
/// - Path: Breadcrumbs recording siblings left behind when navigating down
/// - Graph: The full SemanticGraph for node lookups
/// - State: Emission state accumulated during traversal (from TransferContext)
///
/// Navigation is O(1). We can always reconstruct the full tree by walking up.
module Alex.Traversal.PSGZipper

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open PSGElaboration.Coeffects
open PSGElaboration.SSAAssignment
open PSGElaboration.MutabilityAnalysis
open PSGElaboration.StringCollection
open PSGElaboration.PlatformConfig
open PSGElaboration.PatternBindingAnalysis

// ═══════════════════════════════════════════════════════════════════════════
// ZIPPER PATH (Breadcrumbs)
// ═══════════════════════════════════════════════════════════════════════════

/// A single step in the path, recording what we left behind when going down.
/// When navigating to child N, we record:
/// - The parent node we came from
/// - The siblings to the left of our chosen child
/// - The siblings to the right of our chosen child
type PathStep = {
    /// The parent node we descended from
    Parent: SemanticNode
    /// Siblings to the LEFT of our position (in order)
    LeftSiblings: NodeId list
    /// Siblings to the RIGHT of our position (in order)
    RightSiblings: NodeId list
}

/// The path from root to current focus (list of steps, most recent first)
type ZipperPath = PathStep list

// ═══════════════════════════════════════════════════════════════════════════
// EMISSION STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Zipper focus mode
type ZipperFocus =
    | AtNode                          // Default: focused on a PSG node
    | InFunction of funcName: string  // Inside a function scope

/// Emission state accumulated during traversal.
/// This is the "bag" the photographer carries - coeffects + accumulators.
type EmissionState = {
    // ─────────────────────────────────────────────────────────────────────────
    // COEFFECTS (read-only, from preprocessing)
    // ─────────────────────────────────────────────────────────────────────────

    /// Pre-computed SSA assignments for PSG nodes
    SSAAssignment: SSAAssignment

    /// Mutability analysis results
    MutabilityInfo: MutabilityAnalysisResult

    /// Platform binding resolution results (runtime mode, syscall vs libc, etc.)
    Platform: PlatformResolutionResult

    /// Pattern binding analysis results (match case bindings, entry bindings)
    PatternBindings: PatternBindingAnalysisResult

    // ─────────────────────────────────────────────────────────────────────────
    /// Counter for lambda names
    mutable NextLambdaId: int

    // ─────────────────────────────────────────────────────────────────────────
    // ACCUMULATED OUTPUT
    // ─────────────────────────────────────────────────────────────────────────

    /// Operations in current scope (reverse order for efficiency)
    mutable CurrentOps: MLIROp list

    /// Top-level operations (function definitions, globals)
    mutable TopLevel: MLIROp list

    /// Pre-collected string constants (coeffect from preprocessing)
    /// IMMUTABLE - strings are collected ONCE before zipper traversal
    Strings: StringTable

    /// Function visibility (name -> isInternal)
    mutable FuncVisibility: Map<string, bool>

    /// Region stack for nested scopes (SCF.if, SCF.while, etc.)
    mutable RegionStack: (MLIROp list) list

    // ─────────────────────────────────────────────────────────────────────────
    // FUNCTION SCOPE
    // ─────────────────────────────────────────────────────────────────────────

    /// Current function parameters (set when entering function scope)
    mutable CurrentFuncParams: (SSA * MLIRType) list option

    /// Current function return type
    mutable CurrentFuncRetType: MLIRType option

    /// Current focus mode
    mutable Focus: ZipperFocus

    /// Stack of saved CurrentOps for nested function scopes
    mutable OpsStack: (MLIROp list) list

    /// Stack of saved function params for nested function scopes
    mutable FuncParamsStack: ((SSA * MLIRType) list) list

    /// Stack of saved return types for nested function scopes
    mutable FuncRetTypeStack: MLIRType list

    /// Stack of saved Focus states for nested function scopes
    mutable FocusStack: ZipperFocus list

    // ─────────────────────────────────────────────────────────────────────────
    // VARIABLE BINDINGS (structured types, not strings)
    // ─────────────────────────────────────────────────────────────────────────

    /// Variable bindings: name -> (SSA, MLIRType)
    mutable VarBindings: Map<string, SSA * MLIRType>

    /// Node SSA bindings: NodeId value -> (SSA, MLIRType)
    mutable NodeBindings: Map<int, SSA * MLIRType>

    /// Stack of saved VarBindings for nested function scopes
    mutable VarBindingsStack: Map<string, SSA * MLIRType> list

    /// Stack of saved NodeBindings for nested function scopes
    mutable NodeBindingsStack: Map<int, SSA * MLIRType> list

    /// Captured variable names (both ByRef and ByValue captures)
    /// These come from closure environment extraction - VarRef should use VarBindings (not NodeBindings)
    mutable CapturedVariables: Set<string>

    /// Stack of saved CapturedVariables for nested function scopes
    mutable CapturedVariablesStack: Set<string> list

    /// Captured mutable variable names (ByRef captures only)
    /// These are pointers to the original mutable's alloca - VarRef must load, Set must store
    mutable CapturedMutables: Set<string>

    /// Stack of saved CapturedMutables for nested function scopes
    mutable CapturedMutablesStack: Set<string> list

    /// Entry point lambda IDs (for determining "main")
    EntryPointLambdaIds: Set<int>

    // ─────────────────────────────────────────────────────────────────────────
    // MODULE-LEVEL BINDING SUPPORT
}

// ═══════════════════════════════════════════════════════════════════════════
// PSG ZIPPER
// ═══════════════════════════════════════════════════════════════════════════

/// The PSG Zipper - true Huet zipper with emission state
type PSGZipper = {
    /// The current focus node
    Focus: SemanticNode

    /// Path back to root (breadcrumbs)
    Path: ZipperPath

    /// The full semantic graph (for node lookups)
    Graph: SemanticGraph

    /// Emission state accumulated during traversal
    State: EmissionState
}

// ═══════════════════════════════════════════════════════════════════════════
// EMISSION STATE CREATION
// ═══════════════════════════════════════════════════════════════════════════

module EmissionState =
    /// Create emission state from preprocessing results
    /// ARCHITECTURAL PRINCIPLE: All coeffects (SSA, mutability, strings, platform, patterns)
    /// are computed in preprocessing passes and passed here as immutable data.
    let create
        (ssaAssign: SSAAssignment)
        (mutInfo: MutabilityAnalysisResult)
        (stringTable: StringTable)
        (platform: PlatformResolutionResult)
        (patternBindings: PatternBindingAnalysisResult)
        (entryPointLambdaIds: Set<int>)
        : EmissionState =
        // Start lambda ID counter after the highest lambda_N used in SSAAssignment
        // This prevents collision when fallback naming is used during traversal
        let maxLambdaId =
            ssaAssign.LambdaNames
            |> Map.toSeq
            |> Seq.choose (fun (_, name) ->
                if name.StartsWith("lambda_") then
                    match System.Int32.TryParse(name.Substring(7)) with
                    | true, n -> Some n
                    | false, _ -> None
                else None)
            |> Seq.fold max -1  // Start from -1 so first lambda will be 0 if none exist

        {
            SSAAssignment = ssaAssign
            MutabilityInfo = mutInfo
            Platform = platform
            PatternBindings = patternBindings
            NextLambdaId = maxLambdaId + 1
            CurrentOps = []
            TopLevel = []
            Strings = stringTable  // Pre-collected, immutable coeffect
            FuncVisibility = Map.empty
            RegionStack = []
            CurrentFuncParams = None
            CurrentFuncRetType = None
            Focus = AtNode
            OpsStack = []
            FuncParamsStack = []
            FuncRetTypeStack = []
            FocusStack = []
            VarBindings = Map.empty
            NodeBindings = Map.empty
            VarBindingsStack = []
            NodeBindingsStack = []
            CapturedVariables = Set.empty
            CapturedVariablesStack = []
            CapturedMutables = Set.empty
            CapturedMutablesStack = []
            EntryPointLambdaIds = entryPointLambdaIds
        }

    /// Look up pre-computed SSA for a PSG node (coeffect lookup)
    let lookupNodeSSA (nodeId: NodeId) (state: EmissionState) : SSA option =
        lookupSSA nodeId state.SSAAssignment

    /// Look up pre-computed SSA, failing if not found
    let requireNodeSSA (nodeId: NodeId) (state: EmissionState) : SSA =
        match lookupNodeSSA nodeId state with
        | Some ssa -> ssa
        | None -> failwithf "No SSA assignment for node %A" nodeId

    /// Check if a binding needs alloca by NodeId
    /// (AddressedMutableBindings contains NodeId values)
    let isAddressedMutable (nodeIdValue: int) (state: EmissionState) : bool =
        Set.contains nodeIdValue state.MutabilityInfo.AddressedMutableBindings

    /// Check if a variable name is modified in any loop body
    let isModifiedInAnyLoop (varName: string) (state: EmissionState) : bool =
        state.MutabilityInfo.ModifiedVarsInLoopBodies
        |> Map.exists (fun _ names -> List.contains varName names)

    /// Check if a binding needs alloca (by NodeId and name)
    let needsAlloca (nodeIdValue: int) (varName: string) (state: EmissionState) : bool =
        isAddressedMutable nodeIdValue state || isModifiedInAnyLoop varName state

    /// Look up a platform binding resolution by node ID
    let lookupPlatformBinding (nodeIdValue: int) (state: EmissionState) : BindingResolution option =
        lookupBinding nodeIdValue state.Platform

    /// Check if a node has a platform binding resolution
    let hasPlatformBinding (nodeIdValue: int) (state: EmissionState) : bool =
        hasBinding nodeIdValue state.Platform

    /// Get the runtime mode from platform resolution
    let getRuntimeMode (state: EmissionState) : RuntimeMode =
        state.Platform.RuntimeMode

    /// Check if we're in freestanding mode
    let isFreestanding (state: EmissionState) : bool =
        state.Platform.RuntimeMode = Freestanding

// ═══════════════════════════════════════════════════════════════════════════
// ZIPPER CREATION
// ═══════════════════════════════════════════════════════════════════════════

/// Create a zipper focused on a specific node
let create
    (graph: SemanticGraph)
    (focusId: NodeId)
    (state: EmissionState)
    : PSGZipper option =
    match SemanticGraph.tryGetNode focusId graph with
    | Some node ->
        Some {
            Focus = node
            Path = []  // At root, no path
            Graph = graph
            State = state
        }
    | None -> None

/// Create a zipper at the first entry point
/// All coeffects (SSA, mutability, strings, platform, patterns) are pre-computed and passed in.
let fromEntryPoint
    (graph: SemanticGraph)
    (ssaAssign: SSAAssignment)
    (mutInfo: MutabilityAnalysisResult)
    (stringTable: StringTable)
    (platform: PlatformResolutionResult)
    (patternBindings: PatternBindingAnalysisResult)
    (entryPointLambdaIds: Set<int>)
    : PSGZipper option =
    match graph.EntryPoints with
    | entryId :: _ ->
        let state = EmissionState.create ssaAssign mutInfo stringTable platform patternBindings entryPointLambdaIds
        create graph entryId state
    | [] -> None

// ═══════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════

/// Move UP to the parent node
/// Returns None if already at root
let up (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, cannot go up
    | step :: restPath ->
        Some {
            Focus = step.Parent
            Path = restPath
            Graph = z.Graph
            State = z.State
        }

/// Move DOWN to a specific child by index
/// Records siblings left behind in the path
let down (childIndex: int) (z: PSGZipper) : PSGZipper option =
    let children = z.Focus.Children
    if childIndex < 0 || childIndex >= List.length children then
        None
    else
        let childId = List.item childIndex children
        match SemanticGraph.tryGetNode childId z.Graph with
        | None -> None
        | Some childNode ->
            let leftSiblings = List.take childIndex children
            let rightSiblings = List.skip (childIndex + 1) children
            let step = {
                Parent = z.Focus
                LeftSiblings = leftSiblings
                RightSiblings = rightSiblings
            }
            Some {
                Focus = childNode
                Path = step :: z.Path
                Graph = z.Graph
                State = z.State
            }

/// Move DOWN to the first child
let downFirst (z: PSGZipper) : PSGZipper option =
    down 0 z

/// Move LEFT to the previous sibling
let left (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, no siblings
    | step :: restPath ->
        match List.tryLast step.LeftSiblings with
        | None -> None  // No left siblings
        | Some leftId ->
            match SemanticGraph.tryGetNode leftId z.Graph with
            | None -> None
            | Some leftNode ->
                let newLeftSiblings = List.take (List.length step.LeftSiblings - 1) step.LeftSiblings
                let newRightSiblings = z.Focus.Id :: step.RightSiblings
                let newStep = {
                    Parent = step.Parent
                    LeftSiblings = newLeftSiblings
                    RightSiblings = newRightSiblings
                }
                Some {
                    Focus = leftNode
                    Path = newStep :: restPath
                    Graph = z.Graph
                    State = z.State
                }

/// Move RIGHT to the next sibling
let right (z: PSGZipper) : PSGZipper option =
    match z.Path with
    | [] -> None  // At root, no siblings
    | step :: restPath ->
        match step.RightSiblings with
        | [] -> None  // No right siblings
        | rightId :: remainingRight ->
            match SemanticGraph.tryGetNode rightId z.Graph with
            | None -> None
            | Some rightNode ->
                let newLeftSiblings = step.LeftSiblings @ [z.Focus.Id]
                let newStep = {
                    Parent = step.Parent
                    LeftSiblings = newLeftSiblings
                    RightSiblings = remainingRight
                }
                Some {
                    Focus = rightNode
                    Path = newStep :: restPath
                    Graph = z.Graph
                    State = z.State
                }

/// Navigate to a specific node by ID (re-roots the zipper there)
let focusOn (nodeId: NodeId) (z: PSGZipper) : PSGZipper option =
    match SemanticGraph.tryGetNode nodeId z.Graph with
    | None -> None
    | Some node ->
        Some {
            Focus = node
            Path = []  // Re-rooted, path cleared
            Graph = z.Graph
            State = z.State
        }

// ═══════════════════════════════════════════════════════════════════════════
// FOCUS QUERIES
// ═══════════════════════════════════════════════════════════════════════════

/// Get the current focus node
let focus (z: PSGZipper) : SemanticNode = z.Focus

/// Get the focus node's ID
let focusId (z: PSGZipper) : NodeId = z.Focus.Id

/// Get the focus node's Kind
let focusKind (z: PSGZipper) : SemanticKind = z.Focus.Kind

/// Get the focus node's Type
let focusType (z: PSGZipper) : NativeType = z.Focus.Type

/// Check if at root (no path)
let isAtRoot (z: PSGZipper) : bool = List.isEmpty z.Path

/// Get depth in tree (path length)
let depth (z: PSGZipper) : int = List.length z.Path

/// Get child count of focus
let childCount (z: PSGZipper) : int = List.length z.Focus.Children

/// Check if focus has children
let hasChildren (z: PSGZipper) : bool = not (List.isEmpty z.Focus.Children)

// ═══════════════════════════════════════════════════════════════════════════
// SCOPE DERIVATION (from Zipper's Path)
// ═══════════════════════════════════════════════════════════════════════════

/// Find the enclosing Lambda by walking up the zipper's path.
/// This is the DEFINITIVE source of truth for "are we inside a function?"
/// Returns Some (lambdaNode, lambdaName) if inside a Lambda, None if at module level.
let findEnclosingLambda (z: PSGZipper) : (SemanticNode * string) option =
    // First check if the current focus IS a Lambda
    match z.Focus.Kind with
    | SemanticKind.Lambda _ ->
        let name = lookupLambdaName z.Focus.Id z.State.SSAAssignment
                   |> Option.defaultValue "unknown"
        Some (z.Focus, name)
    | _ ->
        // Walk up the path looking for a Lambda parent
        z.Path
        |> List.tryPick (fun step ->
            match step.Parent.Kind with
            | SemanticKind.Lambda _ ->
                let name = lookupLambdaName step.Parent.Id z.State.SSAAssignment
                           |> Option.defaultValue "unknown"
                Some (step.Parent, name)
            | _ -> None)

// ═══════════════════════════════════════════════════════════════════════════
// STATE OPERATIONS (Emission)
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a single operation to current scope.
let emit (op: MLIROp) (z: PSGZipper) : unit =
    z.State.CurrentOps <- op :: z.State.CurrentOps

/// Emit multiple operations to current scope.
let emitAll (ops: MLIROp list) (z: PSGZipper) : unit =
    z.State.CurrentOps <- List.rev ops @ z.State.CurrentOps

/// Emit a top-level operation (function definition, global)
let emitTopLevel (op: MLIROp) (z: PSGZipper) : unit =
    z.State.TopLevel <- op :: z.State.TopLevel

/// Get current ops in correct order
let getCurrentOps (z: PSGZipper) : MLIROp list =
    List.rev z.State.CurrentOps

/// Get top-level ops in correct order
let getTopLevelOps (z: PSGZipper) : MLIROp list =
    List.rev z.State.TopLevel

// ═══════════════════════════════════════════════════════════════════════════
// REGION MANAGEMENT (for SCF nested scopes)
// ═══════════════════════════════════════════════════════════════════════════

/// Enter a new region scope (pushes current ops onto stack)
let enterRegion (z: PSGZipper) : unit =
    z.State.RegionStack <- z.State.CurrentOps :: z.State.RegionStack
    z.State.CurrentOps <- []

/// Exit region scope, returning the region's ops
let exitRegion (z: PSGZipper) : MLIROp list =
    let regionOps = List.rev z.State.CurrentOps
    match z.State.RegionStack with
    | parentOps :: rest ->
        z.State.CurrentOps <- parentOps
        z.State.RegionStack <- rest
        regionOps
    | [] ->
        failwith "exitRegion called without matching enterRegion"

/// Build a region by running a builder function
let buildRegion (builder: unit -> unit) (z: PSGZipper) : Region =
    enterRegion z
    builder ()
    let ops = exitRegion z
    { Blocks = [{ Label = BlockRef "entry"; Args = []; Ops = ops }] }

// ═══════════════════════════════════════════════════════════════════════════
// STRING LITERALS (Pure Derivation + Coeffect Access)
// ═══════════════════════════════════════════════════════════════════════════

/// Derive the global reference for a string literal
/// PURE FUNCTION: content → @str_<hash>
/// No lookup - the mapping is deterministic
let deriveStringGlobalRef (content: string) : string =
    deriveGlobalRef content

/// Derive the byte length for a string literal
/// PURE FUNCTION: content → UTF-8 byte count
let deriveStringByteLength (content: string) : int =
    deriveByteLength content

/// Access the StringTable coeffect - returns all strings for global emission
/// The coeffect answers: "what strings exist in this program?"
let getStrings (z: PSGZipper) : (uint32 * string * int) list =
    toList z.State.Strings

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION VISIBILITY
// ═══════════════════════════════════════════════════════════════════════════

/// Register a function's visibility
let registerFunc (name: string) (isInternal: bool) (z: PSGZipper) : unit =
    z.State.FuncVisibility <- Map.add name isInternal z.State.FuncVisibility

/// Check if a function is internal
let isFuncInternal (name: string) (z: PSGZipper) : bool =
    Map.tryFind name z.State.FuncVisibility |> Option.defaultValue false

// ═══════════════════════════════════════════════════════════════════════════
// SSA SHORTCUTS (delegating to EmissionState)
// ═══════════════════════════════════════════════════════════════════════════

/// Look up result SSA for a PSG node (most common case)
let lookupNodeSSA (nodeId: NodeId) (z: PSGZipper) : SSA option =
    EmissionState.lookupNodeSSA nodeId z.State

/// Look up all SSAs for a PSG node (for witnesses that need intermediates)
let lookupNodeSSAs (nodeId: NodeId) (z: PSGZipper) : SSA list option =
    lookupSSAs nodeId z.State.SSAAssignment

/// Look up full SSA allocation for a PSG node
let lookupNodeAllocation (nodeId: NodeId) (z: PSGZipper) : NodeSSAAllocation option =
    lookupAllocation nodeId z.State.SSAAssignment

/// Require SSA for a PSG node (fails if not found)
let requireNodeSSA (nodeId: NodeId) (z: PSGZipper) : SSA =
    EmissionState.requireNodeSSA nodeId z.State

/// Require all SSAs for a PSG node (fails if not found)
let requireNodeSSAs (nodeId: NodeId) (z: PSGZipper) : SSA list =
    match lookupNodeSSAs nodeId z with
    | Some ssas -> ssas
    | None -> failwithf "No SSA allocation for node %A" nodeId

/// Check if a binding needs alloca (by NodeId value and variable name)
let needsAlloca (nodeIdValue: int) (varName: string) (z: PSGZipper) : bool =
    EmissionState.needsAlloca nodeIdValue varName z.State

/// Look up a pre-resolved platform binding by node ID
let lookupPlatformBinding (nodeIdValue: int) (z: PSGZipper) : BindingResolution option =
    EmissionState.lookupPlatformBinding nodeIdValue z.State

/// Check if a node has a platform binding resolution
let hasPlatformBinding (nodeIdValue: int) (z: PSGZipper) : bool =
    EmissionState.hasPlatformBinding nodeIdValue z.State

/// Get the runtime mode from the zipper's platform resolution
let getRuntimeMode (z: PSGZipper) : RuntimeMode =
    EmissionState.getRuntimeMode z.State

/// Check if we're in freestanding mode
let isFreestanding (z: PSGZipper) : bool =
    EmissionState.isFreestanding z.State

/// Get the _start wrapper ops if needed for freestanding mode
let getStartWrapperOps (z: PSGZipper) : MLIROp list option =
    z.State.Platform.StartWrapperOps

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH LOOKUPS
// ═══════════════════════════════════════════════════════════════════════════

/// Get a node from the graph by ID
let getNode (nodeId: NodeId) (z: PSGZipper) : SemanticNode option =
    SemanticGraph.tryGetNode nodeId z.Graph

/// Get a node, failing if not found
let requireNode (nodeId: NodeId) (z: PSGZipper) : SemanticNode =
    match getNode nodeId z with
    | Some node -> node
    | None -> failwithf "Node %A not found in graph" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// VARIABLE BINDINGS (Structured Types)
// ═══════════════════════════════════════════════════════════════════════════

/// Bind a variable name to an SSA value (structured)
let bindVarSSA (name: string) (ssa: SSA) (ty: MLIRType) (z: PSGZipper) : PSGZipper =
    z.State.VarBindings <- Map.add name (ssa, ty) z.State.VarBindings
    z

/// Look up a variable binding (structured)
let recallVarSSA (name: string) (z: PSGZipper) : (SSA * MLIRType) option =
    Map.tryFind name z.State.VarBindings

/// Mark a variable as captured (either ByRef or ByValue)
/// Captured variables should use VarBindings (not NodeBindings) when resolving VarRef
let markCapturedVariable (name: string) (z: PSGZipper) : PSGZipper =
    z.State.CapturedVariables <- Set.add name z.State.CapturedVariables
    z

/// Check if a variable is captured (either ByRef or ByValue)
let isCapturedVariable (name: string) (z: PSGZipper) : bool =
    Set.contains name z.State.CapturedVariables

/// Mark a variable as a captured mutable (ByRef capture)
/// These need load/store indirection when accessed
let markCapturedMutable (name: string) (z: PSGZipper) : PSGZipper =
    z.State.CapturedMutables <- Set.add name z.State.CapturedMutables
    z

/// Check if a variable is a captured mutable (ByRef capture)
let isCapturedMutable (name: string) (z: PSGZipper) : bool =
    Set.contains name z.State.CapturedMutables

/// Bind a node's SSA result (structured)
let bindNodeResult (nodeId: int) (ssa: SSA) (ty: MLIRType) (z: PSGZipper) : PSGZipper =
    z.State.NodeBindings <- Map.add nodeId (ssa, ty) z.State.NodeBindings
    z

/// Look up a node's SSA result (structured)
let recallNodeResult (nodeId: int) (z: PSGZipper) : (SSA * MLIRType) option =
    Map.tryFind nodeId z.State.NodeBindings

/// Look up a module-level mutable binding by name
/// Returns (bindingId, globalName) if found
/// Global name is computed as "g_{name}"
let lookupModuleLevelMutable (name: string) (z: PSGZipper) : (int * string) option =
    z.State.MutabilityInfo.ModuleLevelMutableBindings
    |> List.tryFind (fun m -> m.Name = name)
    |> Option.map (fun m -> (m.BindingId, sprintf "g_%s" m.Name))

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION SCOPE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

/// Enter function scope
let enterFunction
    (name: string)
    (params': (SSA * MLIRType) list)
    (retType: MLIRType)
    (visibility: FuncVisibility)
    (z: PSGZipper)
    : PSGZipper =
    // Save current ops to stack before entering new function scope
    z.State.OpsStack <- z.State.CurrentOps :: z.State.OpsStack
    z.State.CurrentOps <- []  // Fresh ops for this function
    // Save current function params/retType to stack for nested lambdas
    match z.State.CurrentFuncParams with
    | Some ps -> z.State.FuncParamsStack <- ps :: z.State.FuncParamsStack
    | None -> ()
    match z.State.CurrentFuncRetType with
    | Some rt -> z.State.FuncRetTypeStack <- rt :: z.State.FuncRetTypeStack
    | None -> ()
    // Save current focus to stack
    z.State.FocusStack <- z.State.Focus :: z.State.FocusStack
    // Save current variable bindings to stack for function scoping
    // VarBindings: cleared (parameters will be bound fresh)
    // NodeBindings: PRESERVED (module-level bindings must be accessible)
    // CapturedVariables/Mutables: cleared (capture markers are function-local)
    z.State.VarBindingsStack <- z.State.VarBindings :: z.State.VarBindingsStack
    z.State.NodeBindingsStack <- z.State.NodeBindings :: z.State.NodeBindingsStack
    z.State.CapturedVariablesStack <- z.State.CapturedVariables :: z.State.CapturedVariablesStack
    z.State.CapturedMutablesStack <- z.State.CapturedMutables :: z.State.CapturedMutablesStack
    z.State.VarBindings <- Map.empty  // Fresh bindings for this function (parameters)
    // NodeBindings: NOT cleared - preserve module-level and outer scope bindings
    z.State.CapturedVariables <- Set.empty  // Fresh captured variables set for this function
    z.State.CapturedMutables <- Set.empty  // Fresh captured mutables set for this function
    z.State.Focus <- InFunction name
    z.State.CurrentFuncParams <- Some params'
    z.State.CurrentFuncRetType <- Some retType
    registerFunc name (visibility = FuncVisibility.Private) z
    z

/// Clear current ops (after building function)
let clearCurrentOps (z: PSGZipper) : unit =
    z.State.CurrentOps <- []

/// Exit function scope
let exitFunction (z: PSGZipper) : PSGZipper =
    // Restore saved ops from stack when exiting function scope
    match z.State.OpsStack with
    | saved :: rest ->
        z.State.CurrentOps <- saved
        z.State.OpsStack <- rest
    | [] ->
        z.State.CurrentOps <- []
    z.State.Focus <- AtNode
    // Restore function params from stack (for nested lambdas)
    match z.State.FuncParamsStack with
    | saved :: rest ->
        z.State.CurrentFuncParams <- Some saved
        z.State.FuncParamsStack <- rest
    | [] ->
        z.State.CurrentFuncParams <- None
    // Restore return type from stack (for nested lambdas)
    match z.State.FuncRetTypeStack with
    | saved :: rest ->
        z.State.CurrentFuncRetType <- Some saved
        z.State.FuncRetTypeStack <- rest
    | [] ->
        z.State.CurrentFuncRetType <- None
    // Restore focus from stack
    match z.State.FocusStack with
    | saved :: rest ->
        z.State.Focus <- saved
        z.State.FocusStack <- rest
    | [] ->
        z.State.Focus <- AtNode
    // Restore variable bindings from stack (function scoping)
    match z.State.VarBindingsStack with
    | saved :: rest ->
        z.State.VarBindings <- saved
        z.State.VarBindingsStack <- rest
    | [] ->
        z.State.VarBindings <- Map.empty
    match z.State.NodeBindingsStack with
    | saved :: rest ->
        z.State.NodeBindings <- saved
        z.State.NodeBindingsStack <- rest
    | [] ->
        z.State.NodeBindings <- Map.empty
    match z.State.CapturedVariablesStack with
    | saved :: rest ->
        z.State.CapturedVariables <- saved
        z.State.CapturedVariablesStack <- rest
    | [] ->
        z.State.CapturedVariables <- Set.empty
    match z.State.CapturedMutablesStack with
    | saved :: rest ->
        z.State.CapturedMutables <- saved
        z.State.CapturedMutablesStack <- rest
    | [] ->
        z.State.CapturedMutables <- Set.empty
    z

// ═══════════════════════════════════════════════════════════════════════════
// LAMBDA NAME GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a lambda name for a specific node ID
/// Entry point lambdas become "main", others get unique names
let yieldLambdaNameForNode (nodeIdValue: int) (z: PSGZipper) : string =
    if Set.contains nodeIdValue z.State.EntryPointLambdaIds then
        "main"
    else
        let id = z.State.NextLambdaId
        z.State.NextLambdaId <- id + 1
        sprintf "lambda_%d" id

/// Generate a fresh lambda name (not tied to a specific node)
let yieldLambdaName (z: PSGZipper) : string =
    let id = z.State.NextLambdaId
    z.State.NextLambdaId <- id + 1
    sprintf "lambda_%d" id

// ═══════════════════════════════════════════════════════════════════════════
// SCOPE ACCESSORS
// ═══════════════════════════════════════════════════════════════════════════

/// Get the current scope mode from the zipper
let getScopeMode (z: PSGZipper) : ZipperFocus =
    z.State.Focus
