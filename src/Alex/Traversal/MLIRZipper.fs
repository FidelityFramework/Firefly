/// MLIRZipper - Codata-driven MLIR composition via observation and witnessing
///
/// ARCHITECTURAL FOUNDATION (Coeffects & Codata):
/// - PSGZipper navigates the INPUT (PSG tree structure)
/// - MLIRZipper composes the OUTPUT (MLIR compute graph) via OBSERVATION
/// - Together they form Alex's dual-zipper architecture
///
/// KEY VOCABULARY (from Coeffects & Codata theory):
/// - **witness** - To observe and record a computation (the fold witnesses each node)
/// - **observe** - To note a context requirement (coeffect - what we need)
/// - **yield** - To produce on demand (codata production)
/// - **bind** - To associate an observation with an identity
/// - **recall** - To retrieve a prior observation
/// - **extract** - Comonad operation: collapse accumulated context to final value
///
/// KEY DESIGN PRINCIPLES (from SpeakEZ blogs):
/// 1. SSA IS Functional Programming (Appel 1998) - F# structure maps directly to MLIR
/// 2. Codata/Pull model - observations accumulate; MLIR emerges from witnessed structure
/// 3. Coeffect tracking - observe requirements (string literals, extern funcs)
/// 4. Functional state threading - immutable state passed through
/// 5. NO string pattern matching on symbol names
module Alex.Traversal.MLIRZipper

open Alex.CodeGeneration.MLIRTypes
open Alex.Templates.TemplateTypes
module SSACoeffect = Alex.Preprocessing.SSAAssignment
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis

module ArithTemplates = Alex.Templates.ArithTemplates
module LLVMTemplates = Alex.Templates.LLVMTemplates
module MemTemplates = Alex.Templates.MemoryTemplates

// ═══════════════════════════════════════════════════════════════════
// MLIR Structure Types - What we're building
// ═══════════════════════════════════════════════════════════════════

/// An MLIR operation with its result SSA value(s)
type MLIROp = {
    /// The operation text (e.g., "%v0 = arith.constant 42 : i32")
    Text: string
    /// Result SSA values produced by this op (empty for void ops)
    Results: (string * MLIRType) list
}

/// A basic block within a function
type MLIRBlock = {
    /// Block label (e.g., "bb0", "entry")
    Label: string
    /// Block arguments (for phi-like semantics)
    Arguments: (string * MLIRType) list
    /// Operations within this block (in order)
    Operations: MLIROp list
}

/// A function definition
type MLIRFunc = {
    /// Function name
    Name: string
    /// Parameter names and types
    Parameters: (string * MLIRType) list
    /// Return type
    ReturnType: MLIRType
    /// Blocks within the function (first is entry)
    Blocks: MLIRBlock list
    /// Attributes (e.g., visibility)
    Attributes: string list
    /// Whether the function has internal linkage (default true for lambdas)
    IsInternal: bool
}

/// Global declarations (string literals, extern funcs, buffers)
type MLIRGlobal =
    | StringLiteral of name: string * content: string * length: int
    | ExternFunc of name: string * signature: string
    | StaticBuffer of name: string * size: int
    /// Module-level mutable variable: name, MLIR type, initial value ops
    | MutableGlobal of name: string * mlirType: string

/// A complete MLIR module
type MLIRModule = {
    /// Global declarations
    Globals: MLIRGlobal list
    /// Function definitions
    Functions: MLIRFunc list
}

// ═══════════════════════════════════════════════════════════════════
// SCF Region Tracking - For structured control flow emission
// ═══════════════════════════════════════════════════════════════════

/// Kind of SCF region for control flow operations (mirrors fsnative's RegionKind)
type SCFRegionKind =
    /// Guard/condition region (while condition, if condition)
    | GuardRegion
    /// Body region (while body, for body)
    | BodyRegion
    /// Then branch region (if-then)
    | ThenRegion
    /// Else branch region (if-then-else)
    | ElseRegion
    /// Start expression region (for loop start bound)
    | StartExprRegion
    /// End expression region (for loop end bound)
    | EndExprRegion
    /// Match case body region (index: 0-based case index)
    | MatchCaseRegion of index: int

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Focus - Where we are in the MLIR structure
// ═══════════════════════════════════════════════════════════════════

/// Current focus within MLIR structure being built
type MLIRFocus =
    | AtModule
    | InFunction of funcName: string
    | InBlock of funcName: string * blockLabel: string

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Path - Breadcrumbs back to module level
// ═══════════════════════════════════════════════════════════════════

/// Path back through MLIR structure (breadcrumbs for navigation)
type MLIRPath =
    | Top
    /// EnteredFunction preserves parent function's ops, var bindings, node SSAs, and SSA counter for proper scoping
    /// Each function has its own SSA namespace - values from outer scopes are NOT valid in nested functions
    /// parentSSACounter is CRITICAL: allows parent function to continue numbering from where it left off
    | EnteredFunction of parent: MLIRPath * funcName: string * funcParams: (string * MLIRType) list * funcReturnType: MLIRType * isInternal: bool * completedFuncs: MLIRFunc list * parentOps: MLIROp list * parentVarBindings: Map<string, string * string> * parentNodeSSAs: Map<string, string * string> * parentSSACounter: int
    | EnteredBlock of parent: MLIRPath * funcName: string * blockLabel: string * completedBlocks: MLIRBlock list * currentParams: (string * MLIRType) list * currentReturnType: MLIRType * isInternal: bool

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper State - Functional state threaded through traversal
// ═══════════════════════════════════════════════════════════════════

/// Immutable state for MLIR composition (symmetric to EmissionState in PSGZipper)
type MLIRState = {
    /// SSA counter for unique value names
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Lambda counter for unique lambda function names
    LambdaCounter: int
    /// Entry point Lambda node IDs (these become "main")
    /// Set during initialization from SemanticGraph.EntryPoints
    EntryPointLambdaIds: Set<int>
    /// Map from PSG NodeId to (SSA value name, MLIR type)
    /// Records what SSA value was produced for each PSG node
    NodeSSA: Map<string, string * string>
    /// Map from variable name to (SSA value name, MLIR type)
    /// Records SSA bindings for Lambda parameters and let bindings
    VarBindings: Map<string, string * string>
    /// String literals: content -> global name (for deduplication)
    StringLiterals: Map<string, string>
    /// External functions declared
    ExternalFuncs: Set<string>
    /// Current function name (for context)
    CurrentFunction: string option
    /// Map from function name to its MLIR return type string
    /// Used to get correct return type at call sites
    FuncReturnTypes: Map<string, string>
    /// Map from function name to its parameter count
    /// Used to filter out unit arguments at call sites
    FuncParamCounts: Map<string, int>
    /// Map from function name to its parameter types (MLIR type strings)
    /// Used to generate correct call signatures for both externs and internal functions
    FuncParamTypes: Map<string, string list>
    /// Stack of operation indices marking the start of SCF regions
    /// When we enter a control flow node (while, for, if), we push the current ops count
    /// When we exit, we pop and extract ops since that index as the region content
    RegionStack: int list
    /// Pending SCF regions: NodeId -> RegionKind -> accumulated ops text
    /// Stores captured operations for each region of each control flow node
    /// Used to build SCF operations after all children are processed
    PendingRegions: Map<string, Map<SCFRegionKind, string list>>
    /// Current SCF region being captured: (parentNodeId, regionKind)
    /// When set, new operations are being accumulated for this region
    CurrentSCFRegion: (string * SCFRegionKind) option
    /// Snapshot of VarBindings at SCF loop entry: loopNodeId -> varName -> (ssa, ty)
    /// Used to determine iter_args by comparing before/after body
    SCFVarSnapshots: Map<string, Map<string, string * string>>
    /// Pre-analyzed iter_args for SCF loops: loopNodeId -> list of (varName, initSSA, argSSA, type)
    /// Set up BEFORE traversing guard/body, so ops use correct iter_arg SSAs from the start
    SCFIterArgs: Map<string, (string * string * string * string) list>
    /// Set of NodeIds of mutable bindings whose address is taken
    /// These need alloca-based memory, not pure SSA
    AddressedMutables: Set<int>
    /// Map from mutable binding NodeId to its alloca SSA pointer
    /// Used by VarRef/Set/AddressOf to access the stack slot
    MutableAllocas: Map<int, string * string>  // NodeId -> (allocaSSA, elementType)
    /// Pre-computed map from loop body NodeId to list of modified variable names
    /// Used for SCF iter_args generation - computed ONCE before transfer begins
    ModifiedVarsInLoopBodies: Map<int, string list>
    /// Pre-computed SSA assignments (coeffect from Preprocessing)
    /// SSA names are assigned BEFORE transfer, not computed during emission
    PrecomputedSSA: SSACoeffect.SSAAssignment option
    /// Module-level mutable bindings that need LLVM globals
    /// Map from binding name -> (bindingNodeId, globalName)
    /// These cannot use SSA (function-scoped), need addressof + load/store
    ModuleLevelMutables: Map<string, int * string>
}

module MLIRState =
    /// Create initial state with no entry point knowledge
    let create () : MLIRState = {
        SSACounter = 0
        LabelCounter = 0
        LambdaCounter = 0
        EntryPointLambdaIds = Set.empty
        NodeSSA = Map.empty
        VarBindings = Map.empty
        StringLiterals = Map.empty
        ExternalFuncs = Set.empty
        CurrentFunction = None
        FuncReturnTypes = Map.empty
        FuncParamCounts = Map.empty
        FuncParamTypes = Map.empty
        RegionStack = []
        PendingRegions = Map.empty
        CurrentSCFRegion = None
        SCFVarSnapshots = Map.empty
        SCFIterArgs = Map.empty
        AddressedMutables = Set.empty
        MutableAllocas = Map.empty
        ModifiedVarsInLoopBodies = Map.empty
        PrecomputedSSA = None
        ModuleLevelMutables = Map.empty
    }

    /// Create state with entry point Lambda IDs
    /// These Lambdas will be named "main" instead of "lambda_N"
    let createWithEntryPoints (entryPointLambdaIds: Set<int>) : MLIRState = {
        SSACounter = 0
        LabelCounter = 0
        LambdaCounter = 0
        EntryPointLambdaIds = entryPointLambdaIds
        NodeSSA = Map.empty
        VarBindings = Map.empty
        StringLiterals = Map.empty
        ExternalFuncs = Set.empty
        CurrentFunction = None
        FuncReturnTypes = Map.empty
        FuncParamCounts = Map.empty
        FuncParamTypes = Map.empty
        RegionStack = []
        PendingRegions = Map.empty
        CurrentSCFRegion = None
        SCFVarSnapshots = Map.empty
        SCFIterArgs = Map.empty
        AddressedMutables = Set.empty
        MutableAllocas = Map.empty
        ModifiedVarsInLoopBodies = Map.empty
        PrecomputedSSA = None
        ModuleLevelMutables = Map.empty
    }

    /// Create state with entry points and full mutability analysis result
    let createWithAnalysis
        (entryPointLambdaIds: Set<int>)
        (addressedMutables: Set<int>)
        (modifiedVarsInLoopBodies: Map<int, string list>)
        (moduleLevelMutables: MutAnalysis.ModuleLevelMutable list)
        (ssaAssignment: SSACoeffect.SSAAssignment)
        : MLIRState =
        // Build map from variable name to (bindingId, globalName)
        let moduleLevelMap =
            moduleLevelMutables
            |> List.fold (fun acc mlm ->
                let globalName = sprintf "__mutable_%s" mlm.Name
                Map.add mlm.Name (mlm.BindingId, globalName) acc
            ) Map.empty
        {
            SSACounter = 0
            LabelCounter = 0
            LambdaCounter = 0
            EntryPointLambdaIds = entryPointLambdaIds
            NodeSSA = Map.empty
            VarBindings = Map.empty
            StringLiterals = Map.empty
            ExternalFuncs = Set.empty
            CurrentFunction = None
            FuncReturnTypes = Map.empty
            FuncParamCounts = Map.empty
            FuncParamTypes = Map.empty
            RegionStack = []
            PendingRegions = Map.empty
            CurrentSCFRegion = None
            SCFVarSnapshots = Map.empty
            SCFIterArgs = Map.empty
            AddressedMutables = addressedMutables
            MutableAllocas = Map.empty
            ModifiedVarsInLoopBodies = modifiedVarsInLoopBodies
            PrecomputedSSA = Some ssaAssignment
            ModuleLevelMutables = moduleLevelMap
        }

    /// Look up pre-computed SSA name for a node (from coeffect)
    let lookupPrecomputedSSA (nodeId: int) (state: MLIRState) : string option =
        state.PrecomputedSSA |> Option.bind (fun ssa -> SSACoeffect.lookupSSA (FSharp.Native.Compiler.Checking.Native.SemanticGraph.NodeId nodeId) ssa)

    /// Check if a mutable binding needs alloca (its address is taken)
    let isAddressedMutable (bindingNodeId: int) (state: MLIRState) : bool =
        Set.contains bindingNodeId state.AddressedMutables

    /// Record an alloca for an addressed mutable binding
    let recordMutableAlloca (bindingNodeId: int) (allocaSSA: string) (elementType: string) (state: MLIRState) : MLIRState =
        { state with MutableAllocas = Map.add bindingNodeId (allocaSSA, elementType) state.MutableAllocas }

    /// Look up alloca for an addressed mutable binding
    let lookupMutableAlloca (bindingNodeId: int) (state: MLIRState) : (string * string) option =
        Map.tryFind bindingNodeId state.MutableAllocas

    /// Check if a variable name is a module-level mutable
    let isModuleLevelMutable (varName: string) (state: MLIRState) : bool =
        Map.containsKey varName state.ModuleLevelMutables

    /// Look up the global name for a module-level mutable
    /// Returns (bindingNodeId, globalName) if found
    let lookupModuleLevelMutable (varName: string) (state: MLIRState) : (int * string) option =
        Map.tryFind varName state.ModuleLevelMutables

    /// Look up pre-computed modified variables for a loop body
    /// Returns the list of variable names that are modified in the given loop body
    let lookupModifiedVarsInLoop (bodyNodeId: int) (state: MLIRState) : string list =
        Map.tryFind bodyNodeId state.ModifiedVarsInLoopBodies
        |> Option.defaultValue []

    /// Store pre-analyzed iter_args for a loop (called BEFORE guard/body traversal)
    let storeIterArgs (loopNodeId: string) (iterArgs: (string * string * string * string) list) (state: MLIRState) : MLIRState =
        { state with SCFIterArgs = Map.add loopNodeId iterArgs state.SCFIterArgs }

    /// Get pre-analyzed iter_args for a loop
    let getIterArgs (loopNodeId: string) (state: MLIRState) : (string * string * string * string) list option =
        Map.tryFind loopNodeId state.SCFIterArgs

    /// Clear iter_args for a loop (after SCF op is emitted)
    let clearIterArgs (loopNodeId: string) (state: MLIRState) : MLIRState =
        { state with SCFIterArgs = Map.remove loopNodeId state.SCFIterArgs }

    /// Snapshot VarBindings at SCF loop entry
    let snapshotVarsForLoop (loopNodeId: string) (state: MLIRState) : MLIRState =
        { state with SCFVarSnapshots = Map.add loopNodeId state.VarBindings state.SCFVarSnapshots }

    /// Get VarBindings snapshot for a loop
    let getVarSnapshot (loopNodeId: string) (state: MLIRState) : Map<string, string * string> option =
        Map.tryFind loopNodeId state.SCFVarSnapshots

    /// Clear VarBindings snapshot for a loop
    let clearVarSnapshot (loopNodeId: string) (state: MLIRState) : MLIRState =
        { state with SCFVarSnapshots = Map.remove loopNodeId state.SCFVarSnapshots }
    
    /// Record a function's return type
    let recordFuncReturnType (funcName: string) (returnType: string) (state: MLIRState) : MLIRState =
        { state with FuncReturnTypes = Map.add funcName returnType state.FuncReturnTypes }

    /// Look up a function's return type
    let lookupFuncReturnType (funcName: string) (state: MLIRState) : string option =
        Map.tryFind funcName state.FuncReturnTypes

    /// Record a function's parameter count
    let recordFuncParamCount (funcName: string) (paramCount: int) (state: MLIRState) : MLIRState =
        { state with FuncParamCounts = Map.add funcName paramCount state.FuncParamCounts }

    /// Look up a function's parameter count
    let lookupFuncParamCount (funcName: string) (state: MLIRState) : int option =
        Map.tryFind funcName state.FuncParamCounts

    /// Record a function's parameter types
    let recordFuncParamTypes (funcName: string) (paramTypes: string list) (state: MLIRState) : MLIRState =
        { state with FuncParamTypes = Map.add funcName paramTypes state.FuncParamTypes }

    /// Look up a function's parameter types
    let lookupFuncParamTypes (funcName: string) (state: MLIRState) : string list option =
        Map.tryFind funcName state.FuncParamTypes

    /// Check if a function is an extern
    let isExtern (funcName: string) (state: MLIRState) : bool =
        Set.contains funcName state.ExternalFuncs

    /// Push a region start marker onto the stack
    /// The marker is the current ops count (will be used to extract ops since this point)
    let pushRegion (currentOpsCount: int) (state: MLIRState) : MLIRState =
        { state with RegionStack = currentOpsCount :: state.RegionStack }

    /// Pop and return the region start marker from the stack
    let popRegion (state: MLIRState) : int option * MLIRState =
        match state.RegionStack with
        | [] -> None, state
        | startIndex :: rest -> Some startIndex, { state with RegionStack = rest }

    // ─────────────────────────────────────────────────────────────────
    // SCF Region Tracking - For structured control flow witnessing
    // ─────────────────────────────────────────────────────────────────

    /// Begin tracking an SCF region - marks the start and sets current region
    /// Called before processing a region's children (guard, body, then, else)
    let beginSCFRegionTracking (parentNodeId: string) (regionKind: SCFRegionKind) (currentOpsCount: int) (state: MLIRState) : MLIRState =
        { state with
            RegionStack = currentOpsCount :: state.RegionStack
            CurrentSCFRegion = Some (parentNodeId, regionKind) }

    /// End tracking an SCF region - extracts ops and stores in PendingRegions
    /// Called after processing a region's children
    /// Returns the start index (for ops extraction) and updated state
    let endSCFRegionTracking (parentNodeId: string) (regionKind: SCFRegionKind) (state: MLIRState) : int option * MLIRState =
        match state.RegionStack with
        | [] -> None, state
        | startIndex :: restStack ->
            Some startIndex,
            { state with
                RegionStack = restStack
                CurrentSCFRegion = None }

    /// Store captured region operations in PendingRegions
    let storeRegionOps (parentNodeId: string) (regionKind: SCFRegionKind) (ops: string list) (state: MLIRState) : MLIRState =
        let existingRegions = Map.tryFind parentNodeId state.PendingRegions |> Option.defaultValue Map.empty
        let updatedRegions = Map.add regionKind ops existingRegions
        { state with PendingRegions = Map.add parentNodeId updatedRegions state.PendingRegions }

    /// Get pending regions for a control flow node
    let getPendingRegions (parentNodeId: string) (state: MLIRState) : Map<SCFRegionKind, string list> option =
        Map.tryFind parentNodeId state.PendingRegions

    /// Clear pending regions for a control flow node (after SCF op is emitted)
    let clearPendingRegions (parentNodeId: string) (state: MLIRState) : MLIRState =
        { state with PendingRegions = Map.remove parentNodeId state.PendingRegions }

    /// Parse an MLIR signature string and extract param types and return type
    /// e.g., "(i32, i64, i32) -> i32" -> (["i32"; "i64"; "i32"], "i32")
    let parseSignature (signature: string) : string list * string =
        // Find "->" to split params from return type
        match signature.IndexOf("->") with
        | -1 -> [], signature.Trim()
        | arrowIdx ->
            let paramsStr = signature.Substring(0, arrowIdx).Trim()
            let retStr = signature.Substring(arrowIdx + 2).Trim()
            // Extract params from "(param1, param2, ...)"
            let paramTypes =
                if paramsStr.StartsWith("(") && paramsStr.EndsWith(")") then
                    let inner = paramsStr.Substring(1, paramsStr.Length - 2).Trim()
                    if inner.Length = 0 then []
                    else inner.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
                else []
            paramTypes, retStr

    /// Yield next SSA name on demand (codata yield)
    let yieldSSA (state: MLIRState) : string * MLIRState =
        let name = sprintf "%%v%d" state.SSACounter
        name, { state with SSACounter = state.SSACounter + 1 }

    /// Yield next block label on demand (codata yield)
    let yieldLabel (state: MLIRState) : string * MLIRState =
        let name = sprintf "bb%d" state.LabelCounter
        name, { state with LabelCounter = state.LabelCounter + 1 }

    /// Yield next lambda function name on demand (codata yield)
    /// Entry point Lambdas become "main", others get unique names
    let yieldLambdaName (state: MLIRState) : string * MLIRState =
        let name = sprintf "lambda_%d" state.LambdaCounter
        name, { state with LambdaCounter = state.LambdaCounter + 1 }

    /// Yield lambda name for a specific node ID
    /// Entry point Lambdas (in EntryPointLambdaIds) become "main"
    let yieldLambdaNameForNode (nodeId: int) (state: MLIRState) : string * MLIRState =
        if Set.contains nodeId state.EntryPointLambdaIds then
            // This Lambda is an entry point - name it main
            "main", state  // Don't increment counter for main
        else
            let name = sprintf "lambda_%d" state.LambdaCounter
            name, { state with LambdaCounter = state.LambdaCounter + 1 }

    /// Bind SSA observation to a PSG node identity
    let bindNodeSSA (nodeId: string) (ssaName: string) (mlirType: string) (state: MLIRState) : MLIRState =
        { state with NodeSSA = Map.add nodeId (ssaName, mlirType) state.NodeSSA }

    /// Recall a prior SSA observation for a PSG node
    let recallNodeSSA (nodeId: string) (state: MLIRState) : (string * string) option =
        Map.tryFind nodeId state.NodeSSA

    /// Bind a variable name to an SSA value (for Lambda parameters)
    let bindVar (varName: string) (ssaName: string) (mlirType: string) (state: MLIRState) : MLIRState =
        { state with VarBindings = Map.add varName (ssaName, mlirType) state.VarBindings }

    /// Recall SSA value for a variable name
    let recallVar (varName: string) (state: MLIRState) : (string * string) option =
        Map.tryFind varName state.VarBindings

    /// Observe a string literal requirement (coeffect - context need)
    let observeStringLiteral (content: string) (state: MLIRState) : string * MLIRState =
        match Map.tryFind content state.StringLiterals with
        | Some name -> name, state
        | None ->
            let name = sprintf "str%d" state.StringLiterals.Count
            name, { state with StringLiterals = Map.add content name state.StringLiterals }

    /// Observe an external function requirement with signature (coeffect - context need)
    /// Parses the signature to extract param types and return type for later use
    let observeExternFuncWithSignature (name: string) (signature: string) (state: MLIRState) : MLIRState =
        let paramTypes, retType = parseSignature signature
        let state' = { state with ExternalFuncs = Set.add name state.ExternalFuncs }
        let state'' = recordFuncParamTypes name paramTypes state'
        let state''' = recordFuncParamCount name (List.length paramTypes) state''
        recordFuncReturnType name retType state'''

    /// Observe an external function requirement (coeffect - context need)
    /// Simple version that just tracks the function name
    let observeExternFunc (name: string) (state: MLIRState) : MLIRState =
        { state with ExternalFuncs = Set.add name state.ExternalFuncs }

    /// Set current function context
    let setCurrentFunction (name: string option) (state: MLIRState) : MLIRState =
        { state with CurrentFunction = name }

// ═══════════════════════════════════════════════════════════════════
// The MLIRZipper Type - Symmetric to PSGZipper
// ═══════════════════════════════════════════════════════════════════

/// The MLIR Composition Zipper - symmetric counterpart to PSGZipper
///
/// Where PSGZipper navigates the INPUT (PSG structure),
/// MLIRZipper composes the OUTPUT (MLIR compute graph).
///
/// Key insight: The zipper pattern works for both READING and WRITING trees.
/// - PSGZipper: Focus on current PSG node, Path to get back
/// - MLIRZipper: Focus on current MLIR position, Path with accumulated content
type MLIRZipper = {
    /// Current focus in MLIR structure
    Focus: MLIRFocus
    /// Path back to module level (with accumulated content)
    Path: MLIRPath
    /// Operations accumulated at current focus (in reverse order for efficiency)
    CurrentOps: MLIROp list
    /// Functional state (SSA counters, node mappings, literals)
    State: MLIRState
    /// Completed globals (accumulated during traversal)
    Globals: MLIRGlobal list
    /// Completed lambda functions (accumulated during traversal)
    CompletedFunctions: MLIRFunc list
    /// Platform helper functions (emitted once at module level)
    /// These are func.func definitions that implement Parse, Format, etc.
    PlatformHelpers: Set<string>
}

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Operations - Navigation and Emission
// ═══════════════════════════════════════════════════════════════════

module MLIRZipper =

    // ─────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────

    /// Create a new MLIRZipper at module level
    let create () : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = MLIRState.create ()
        Globals = []
        CompletedFunctions = []
        PlatformHelpers = Set.empty
    }

    /// Create with initial state (for continuing from previous context)
    let createWithState (state: MLIRState) : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = state
        Globals = []
        CompletedFunctions = []
        PlatformHelpers = Set.empty
    }

    /// Create with entry point Lambda IDs
    /// These Lambdas will be named "main" instead of "lambda_N"
    let createWithEntryPoints (entryPointLambdaIds: Set<int>) : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = MLIRState.createWithEntryPoints entryPointLambdaIds
        Globals = []
        CompletedFunctions = []
        PlatformHelpers = Set.empty
    }

    /// Create with entry points and full mutability analysis result
    let createWithAnalysis
        (entryPointLambdaIds: Set<int>)
        (addressedMutables: Set<int>)
        (modifiedVarsInLoopBodies: Map<int, string list>)
        (moduleLevelMutables: MutAnalysis.ModuleLevelMutable list)
        (ssaAssignment: SSACoeffect.SSAAssignment)
        : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = MLIRState.createWithAnalysis entryPointLambdaIds addressedMutables modifiedVarsInLoopBodies moduleLevelMutables ssaAssignment
        Globals = []
        CompletedFunctions = []
        PlatformHelpers = Set.empty
    }

    // ─────────────────────────────────────────────────────────────────
    // State Operations (functional threading with codata vocabulary)
    // ─────────────────────────────────────────────────────────────────

    /// Map over the state
    let mapState (f: MLIRState -> MLIRState) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = f zipper.State }

    /// Replace the state
    let withState (state: MLIRState) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = state }

    /// Yield next SSA name on demand (codata yield)
    let yieldSSA (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldSSA zipper.State
        name, { zipper with State = newState }

    /// Yield next block label on demand (codata yield)
    let yieldLabel (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLabel zipper.State
        name, { zipper with State = newState }

    /// Bind SSA observation to a PSG node identity
    let bindNodeSSA (nodeId: string) (ssaName: string) (mlirType: string) (zipper: MLIRZipper) : MLIRZipper =
        mapState (MLIRState.bindNodeSSA nodeId ssaName mlirType) zipper

    /// Recall a prior SSA observation for a PSG node
    let recallNodeSSA (nodeId: string) (zipper: MLIRZipper) : (string * string) option =
        MLIRState.recallNodeSSA nodeId zipper.State

    /// Bind a variable name to an SSA value (for Lambda parameters)
    let bindVar (varName: string) (ssaName: string) (mlirType: string) (zipper: MLIRZipper) : MLIRZipper =
        mapState (MLIRState.bindVar varName ssaName mlirType) zipper

    /// Recall SSA value for a variable name
    let recallVar (varName: string) (zipper: MLIRZipper) : (string * string) option =
        MLIRState.recallVar varName zipper.State

    /// Look up a function's tracked return type
    let lookupFuncReturnType (funcName: string) (zipper: MLIRZipper) : string option =
        MLIRState.lookupFuncReturnType funcName zipper.State

    /// Look up a function's parameter count
    let lookupFuncParamCount (funcName: string) (zipper: MLIRZipper) : int option =
        MLIRState.lookupFuncParamCount funcName zipper.State

    /// Look up a function's parameter types
    let lookupFuncParamTypes (funcName: string) (zipper: MLIRZipper) : string list option =
        MLIRState.lookupFuncParamTypes funcName zipper.State

    /// Check if a function is an extern
    let isExtern (funcName: string) (zipper: MLIRZipper) : bool =
        MLIRState.isExtern funcName zipper.State

    /// Observe a string literal requirement (coeffect - context need)
    let observeStringLiteral (content: string) (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.observeStringLiteral content zipper.State
        // Also add to globals
        // CRITICAL: Use UTF-8 byte count, not .NET character count
        // Multi-byte UTF-8 sequences (emoji, non-ASCII) have different byte counts
        let len = System.Text.Encoding.UTF8.GetByteCount(content) + 1  // +1 for null terminator
        let strGlobal = StringLiteral (name, content, len)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | StringLiteral (n, _, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else strGlobal :: zipper.Globals
        name, { zipper with State = newState; Globals = newGlobals }

    /// Observe an external function requirement (coeffect - context need)
    /// Also parses and stores the signature for correct call-site generation
    let observeExternFunc (name: string) (signature: string) (zipper: MLIRZipper) : MLIRZipper =
        let newState = MLIRState.observeExternFuncWithSignature name signature zipper.State
        let extGlobal = ExternFunc (name, signature)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | ExternFunc (n, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else extGlobal :: zipper.Globals
        { zipper with State = newState; Globals = newGlobals }

    /// Observe a static buffer requirement (coeffect - for ReadLine etc.)
    let observeStaticBuffer (name: string) (size: int) (zipper: MLIRZipper) : MLIRZipper =
        let bufGlobal = StaticBuffer (name, size)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | StaticBuffer (n, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else bufGlobal :: zipper.Globals
        { zipper with Globals = newGlobals }

    /// Observe a module-level mutable variable requirement
    /// Emits an LLVM global for the mutable binding
    let observeMutableGlobal (globalName: string) (mlirType: string) (zipper: MLIRZipper) : MLIRZipper =
        let mutGlobal = MutableGlobal (globalName, mlirType)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | MutableGlobal (n, _) -> n = globalName
                | _ -> false) zipper.Globals
            then zipper.Globals
            else mutGlobal :: zipper.Globals
        { zipper with Globals = newGlobals }

    // ─────────────────────────────────────────────────────────────────
    // Navigation - Enter/Exit structures
    // ─────────────────────────────────────────────────────────────────

    /// Enter a function definition with explicit visibility control
    let enterFunctionWithVisibility (name: string) (parameters: (string * MLIRType) list) (returnType: MLIRType) (isInternal: bool) (zipper: MLIRZipper) : MLIRZipper =
        // Record param types and return type for this function
        let paramTypes = parameters |> List.map (snd >> Serialize.mlirType)
        let stateWithParamTypes =
            zipper.State
            |> MLIRState.recordFuncParamTypes name paramTypes
            |> MLIRState.recordFuncParamCount name (List.length parameters)
            |> MLIRState.recordFuncReturnType name (Serialize.mlirType returnType)
        match zipper.Focus with
        | AtModule ->
            // Enter from module level - preserve NodeSSA (module-level bindings should remain visible)
            // Save NodeSSA and SSACounter in path so they're restored when we exit back to module level
            // Reset SSA counter for function-local numbering
            { zipper with
                Focus = InFunction name
                Path = EnteredFunction (zipper.Path, name, parameters, returnType, isInternal, [], [], Map.empty, stateWithParamTypes.NodeSSA, stateWithParamTypes.SSACounter)
                CurrentOps = []
                State = { stateWithParamTypes with
                            VarBindings = Map.empty
                            CurrentFunction = Some name
                            SSACounter = 0 } }  // Reset SSA counter for function-local numbering
        | InFunction _ ->
            // Already in a function - nested lambda
            // CRITICAL: Save current ops, var bindings, node SSAs, AND SSA counter in path so they're restored when we exit
            // Filter NodeSSA to only keep function references (@xxx) and pipe markers ($pipe:) - value SSAs are function-local
            // Also reset SSA counter so nested functions have their own %v0, %v1, etc.
            let globalOnlyNodeSSA =
                stateWithParamTypes.NodeSSA
                |> Map.filter (fun _ (ssa, _) -> ssa.StartsWith("@") || ssa.StartsWith("$pipe:"))
            { zipper with
                Focus = InFunction name
                Path = EnteredFunction (zipper.Path, name, parameters, returnType, isInternal, [], zipper.CurrentOps, stateWithParamTypes.VarBindings, stateWithParamTypes.NodeSSA, stateWithParamTypes.SSACounter)
                CurrentOps = []
                State = { stateWithParamTypes with
                            VarBindings = Map.empty
                            NodeSSA = globalOnlyNodeSSA
                            CurrentFunction = Some name
                            SSACounter = 0 } }  // Reset SSA counter for nested function
        | _ ->
            // Can only enter function from module level or inside another function
            failwithf "Cannot enter function '%s' - invalid focus: %A" name zipper.Focus

    /// Enter a function definition (internal by default for lambdas)
    let enterFunction (name: string) (parameters: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        enterFunctionWithVisibility name parameters returnType true zipper

    /// Enter a basic block within current function
    let enterBlock (label: string) (arguments: (string * MLIRType) list) (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, _, funcParams, retTy, isInternal, completedFuncs, _parentOps, _parentVarBindings, _parentNodeSSAs, _parentSSACounter) ->
            // Save current ops as we enter a new block
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (zipper.Path, funcName, label, [], funcParams, retTy, isInternal)
                CurrentOps = [] }
        | InBlock (funcName, prevLabel), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy, isInternal) ->
            // Finish previous block, enter new one
            let prevBlock = {
                Label = prevLabel
                Arguments = []
                Operations = List.rev zipper.CurrentOps
            }
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (parentPath, fn, label, prevBlock :: completedBlocks, funcParams, retTy, isInternal)
                CurrentOps = [] }
        | _ ->
            failwithf "Cannot enter block '%s' - not in a function" label

    /// Exit current block, returning to function level
    /// Note: Uses 'rec' to enable mutual recursion with exitFunction
    let rec exitBlock (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InBlock (funcName, label), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy, _isInternal) ->
            // Complete current block
            let block = {
                Label = label
                Arguments = []
                Operations = List.rev zipper.CurrentOps
            }
            { zipper with
                Focus = InFunction funcName
                Path = parentPath
                CurrentOps = [] }  // Blocks are stored in path, ops reset
        | _ ->
            failwithf "Cannot exit block - not in a block (focus: %A)" zipper.Focus

    /// Exit current function, returning to module level or parent function
    and exitFunction (zipper: MLIRZipper) : MLIRZipper * MLIRFunc =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, name, funcParams, funcReturnType, isInternal, completedFuncs, parentOps, parentVarBindings, parentNodeSSAs, parentSSACounter) ->
            // Create function with accumulated ops as single entry block
            let entryBlock = {
                Label = "entry"
                Arguments = []
                Operations = List.rev zipper.CurrentOps  // Reverse to get correct order
            }
            let func = {
                Name = name
                Parameters = funcParams
                ReturnType = funcReturnType
                Blocks = [entryBlock]
                Attributes = []
                IsInternal = isInternal  // Use observed visibility
            }
            // CRITICAL FIX: Merge new function references from nested scope back to parent
            // This preserves bindings like "@lambda_X" and "_lambdaName" markers that were
            // created during nested function processing (e.g., writeStrOut defined inside
            // Console.WriteLine). Without this, function references are lost and appear as
            // undefined extern symbols.
            let mergedNodeSSAs =
                let newFuncRefs =
                    zipper.State.NodeSSA
                    |> Map.filter (fun key (ssa, _) ->
                        // Keep function references (@xxx), pipe markers ($pipe:),
                        // partial apps ($partial:), platform markers ($platform:),
                        // and lambda name markers (xxx_lambdaName)
                        ssa.StartsWith("@") ||
                        ssa.StartsWith("$pipe:") ||
                        ssa.StartsWith("$partial:") ||
                        ssa.StartsWith("$platform:") ||
                        key.EndsWith("_lambdaName"))
                // Merge new refs into parent's NodeSSA (new refs take precedence)
                Map.fold (fun acc k v -> Map.add k v acc) parentNodeSSAs newFuncRefs
            // Determine new focus based on parentPath
            // If parentPath is Top, we return to AtModule
            // If parentPath is EnteredFunction, we return to InFunction (and restore CurrentFunction AND SSACounter!)
            let newFocus, newState =
                match parentPath with
                | Top ->
                    AtModule, { MLIRState.setCurrentFunction None zipper.State with VarBindings = parentVarBindings; NodeSSA = mergedNodeSSAs; SSACounter = parentSSACounter }
                | EnteredFunction (_, parentFuncName, _, _, _, _, _, _, _, _) ->
                    InFunction parentFuncName, { MLIRState.setCurrentFunction (Some parentFuncName) zipper.State with VarBindings = parentVarBindings; NodeSSA = mergedNodeSSAs; SSACounter = parentSSACounter }
                | EnteredBlock (_, parentFuncName, _, _, _, _, _) ->
                    InFunction parentFuncName, { MLIRState.setCurrentFunction (Some parentFuncName) zipper.State with VarBindings = parentVarBindings; NodeSSA = mergedNodeSSAs; SSACounter = parentSSACounter }
            { zipper with
                Focus = newFocus
                Path = parentPath
                CurrentOps = parentOps  // CRITICAL: Restore parent function's ops
                State = newState },      // CRITICAL: State now includes restored VarBindings, merged NodeSSAs, AND SSACounter
            func
        | InBlock (funcName, label), _ ->
            // Exit block first, then function
            let zipper' = exitBlock zipper
            exitFunction zipper'
        | _ ->
            failwithf "Cannot exit function - not in a function (focus: %A)" zipper.Focus

    // ─────────────────────────────────────────────────────────────────
    // Witnessing - Recording observations of computation structure
    // The fold WITNESSES nodes; these functions RECORD what is observed
    // ─────────────────────────────────────────────────────────────────

    /// Witness an operation (record observation at current focus)
    let witnessOp (text: string) (results: (string * MLIRType) list) (zipper: MLIRZipper) : MLIRZipper =
        let op = { Text = text; Results = results }
        { zipper with CurrentOps = op :: zipper.CurrentOps }

    /// Witness an operation with single result
    let witnessOpWithResult (text: string) (resultSSA: string) (resultType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        witnessOp text [(resultSSA, resultType)] zipper

    /// Witness a void operation (observation with no yielded value)
    let witnessVoidOp (text: string) (zipper: MLIRZipper) : MLIRZipper =
        witnessOp text [] zipper

    // ─────────────────────────────────────────────────────────────────
    // SCF Region Tracking - For structured control flow emission
    // ─────────────────────────────────────────────────────────────────

    /// Begin capturing a region - marks current ops count for later extraction
    /// Call this BEFORE processing control flow children (in pre-hook)
    let beginRegion (zipper: MLIRZipper) : MLIRZipper =
        let opsCount = List.length zipper.CurrentOps
        { zipper with State = MLIRState.pushRegion opsCount zipper.State }

    /// End region capture and extract operations since the marked point
    /// Returns the extracted operations (as text list) and updated zipper with those ops removed
    let endRegion (zipper: MLIRZipper) : string list * MLIRZipper =
        match MLIRState.popRegion zipper.State with
        | None, _ ->
            // No region marker - return empty and unchanged
            [], zipper
        | Some startIndex, newState ->
            // CurrentOps is in reverse order (newest first)
            // We need ops from startIndex to end (the region content)
            let totalOps = List.length zipper.CurrentOps
            let regionOpsCount = totalOps - startIndex

            // Take the region ops (they're at the front of CurrentOps since it's reversed)
            let regionOps = zipper.CurrentOps |> List.take regionOpsCount |> List.rev
            let remainingOps = zipper.CurrentOps |> List.skip regionOpsCount

            // Extract operation text
            let regionText = regionOps |> List.map (fun op -> op.Text)

            regionText, { zipper with CurrentOps = remainingOps; State = newState }

    /// Begin tracking an SCF region - for use with foldWithSCFRegions hook
    /// Marks current ops count and sets tracking state
    let beginSCFRegion (parentNodeId: string) (regionKind: SCFRegionKind) (zipper: MLIRZipper) : MLIRZipper =
        let opsCount = List.length zipper.CurrentOps
        { zipper with State = MLIRState.beginSCFRegionTracking parentNodeId regionKind opsCount zipper.State }

    /// End tracking an SCF region - extracts ops and stores in PendingRegions
    /// Called by foldWithSCFRegions hook after each region's children are processed
    let endSCFRegion (parentNodeId: string) (regionKind: SCFRegionKind) (zipper: MLIRZipper) : MLIRZipper =
        match MLIRState.endSCFRegionTracking parentNodeId regionKind zipper.State with
        | None, _ ->
            // No region marker - return unchanged
            zipper
        | Some startIndex, newState ->
            // CurrentOps is in reverse order (newest first)
            let totalOps = List.length zipper.CurrentOps
            let regionOpsCount = totalOps - startIndex

            // Take the region ops (they're at the front of CurrentOps since it's reversed)
            let regionOps = zipper.CurrentOps |> List.take regionOpsCount |> List.rev
            let remainingOps = zipper.CurrentOps |> List.skip regionOpsCount

            // Extract operation text
            let regionText = regionOps |> List.map (fun op -> op.Text)

            // Store in PendingRegions
            let finalState = MLIRState.storeRegionOps parentNodeId regionKind regionText newState

            { zipper with CurrentOps = remainingOps; State = finalState }

    /// Get pending regions for a control flow node (convenience wrapper)
    let getPendingRegions (parentNodeId: string) (zipper: MLIRZipper) : Map<SCFRegionKind, string list> option =
        MLIRState.getPendingRegions parentNodeId zipper.State

    /// Clear pending regions after SCF op is emitted (convenience wrapper)
    let clearPendingRegions (parentNodeId: string) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.clearPendingRegions parentNodeId zipper.State }

    /// Snapshot VarBindings at loop entry for iter_args detection
    let snapshotVarsForLoop (loopNodeId: string) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.snapshotVarsForLoop loopNodeId zipper.State }

    /// Get VarBindings snapshot for a loop
    let getVarSnapshot (loopNodeId: string) (zipper: MLIRZipper) : Map<string, string * string> option =
        MLIRState.getVarSnapshot loopNodeId zipper.State

    /// Get current VarBindings
    let getVarBindings (zipper: MLIRZipper) : Map<string, string * string> =
        zipper.State.VarBindings

    /// Clear VarBindings snapshot for a loop
    let clearVarSnapshot (loopNodeId: string) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.clearVarSnapshot loopNodeId zipper.State }

    /// Store pre-analyzed iter_args for a loop (called BEFORE guard/body traversal)
    let storeIterArgs (loopNodeId: string) (iterArgs: (string * string * string * string) list) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.storeIterArgs loopNodeId iterArgs zipper.State }

    /// Get pre-analyzed iter_args for a loop
    let getIterArgs (loopNodeId: string) (zipper: MLIRZipper) : (string * string * string * string) list option =
        MLIRState.getIterArgs loopNodeId zipper.State

    /// Clear iter_args for a loop (after SCF op is emitted)
    let clearIterArgs (loopNodeId: string) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.clearIterArgs loopNodeId zipper.State }

    // ─────────────────────────────────────────────────────────────────
    // Addressed Mutable Bindings - For variables whose address is taken
    // ─────────────────────────────────────────────────────────────────

    /// Check if a mutable binding needs alloca (its address is taken)
    let isAddressedMutable (bindingNodeId: int) (zipper: MLIRZipper) : bool =
        MLIRState.isAddressedMutable bindingNodeId zipper.State

    /// Record an alloca for an addressed mutable binding
    let recordMutableAlloca (bindingNodeId: int) (allocaSSA: string) (elementType: string) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = MLIRState.recordMutableAlloca bindingNodeId allocaSSA elementType zipper.State }

    /// Look up alloca for an addressed mutable binding
    let lookupMutableAlloca (bindingNodeId: int) (zipper: MLIRZipper) : (string * string) option =
        MLIRState.lookupMutableAlloca bindingNodeId zipper.State

    /// Check if a variable name is a module-level mutable
    let isModuleLevelMutable (varName: string) (zipper: MLIRZipper) : bool =
        MLIRState.isModuleLevelMutable varName zipper.State

    /// Look up the global name for a module-level mutable
    /// Returns (bindingNodeId, globalName) if found
    let lookupModuleLevelMutable (varName: string) (zipper: MLIRZipper) : (int * string) option =
        MLIRState.lookupModuleLevelMutable varName zipper.State

    /// Look up pre-computed modified variables for a loop body
    /// Returns the list of variable names that are modified in the given loop body
    let lookupModifiedVarsInLoop (bodyNodeId: int) (zipper: MLIRZipper) : string list =
        MLIRState.lookupModifiedVarsInLoop bodyNodeId zipper.State

    /// Witness an alloca with string type (for when type is already serialized)
    /// Emits: %c1 = arith.constant 1 : i64
    ///        %ptr = llvm.alloca %c1 x elementType : (i64) -> !llvm.ptr
    let witnessAllocaStr (elementTypeStr: string) (zipper: MLIRZipper) : string * MLIRZipper =
        // First emit constant 1 for the count
        let countSSA, zipper1 = yieldSSA zipper
        let countParams : ConstantParams = { Result = countSSA; Value = "1"; Type = "i64" }
        let countOp = render ArithTemplates.Quot.Constant.intConst countParams
        let zipper2 = witnessOpWithResult countOp countSSA (Integer I64) zipper1
        // Then emit the alloca
        let allocaSSA, zipper3 = yieldSSA zipper2
        let allocaParams : AllocaParams = { Result = allocaSSA; Count = countSSA; ElementType = elementTypeStr }
        let allocaOp = render LLVMTemplates.Quot.Memory.alloca allocaParams
        allocaSSA, witnessOpWithResult allocaOp allocaSSA Pointer zipper3

    /// Witness a store to an alloca
    /// Emits: llvm.store %value, %ptr : type, !llvm.ptr
    let witnessStore (valueSSA: string) (valueType: string) (ptrSSA: string) (zipper: MLIRZipper) : MLIRZipper =
        let storeParams : StoreParams = { Value = valueSSA; Pointer = ptrSSA; Type = valueType }
        let op = render LLVMTemplates.Quot.Memory.store storeParams
        witnessVoidOp op zipper

    /// Witness a load with string type (for when type is already serialized)
    /// Emits: %value = llvm.load %ptr : !llvm.ptr -> type
    let witnessLoadStr (ptrSSA: string) (resultTypeStr: string) (zipper: MLIRZipper) : string * MLIRZipper =
        let resultSSA, zipper1 = yieldSSA zipper
        let loadParams : LoadParams = { Result = resultSSA; Pointer = ptrSSA; Type = resultTypeStr }
        let op = render LLVMTemplates.Quot.Memory.load loadParams
        // Parse the type string back to MLIRType for witnessOpWithResult
        let resultType = Serialize.deserializeType resultTypeStr
        resultSSA, witnessOpWithResult op resultSSA resultType zipper1

    // ─────────────────────────────────────────────────────────────────
    // Template Primitives - Global Access
    // These are the INFRASTRUCTURE OF TRANSFORM, composable building blocks
    // Note: witnessAddressOf is defined later in the file using templates
    // ─────────────────────────────────────────────────────────────────

    /// Store to a global (addressof ∘ store)
    /// Uses the existing witnessAddressOf primitive defined below
    let witnessGlobalStore (globalName: string) (valueSSA: string) (valueType: string) (zipper: MLIRZipper) : MLIRZipper =
        // Inline addressof here since witnessAddressOf is defined later
        let ptrSSA, zipper1 = yieldSSA zipper
        let addrText = render LLVMTemplates.Quot.Memory.addressof {| Result = ptrSSA; GlobalName = globalName |}
        let zipper2 = witnessOpWithResult addrText ptrSSA Pointer zipper1
        witnessStore valueSSA valueType ptrSSA zipper2

    /// Load from a global (addressof ∘ load)
    let witnessGlobalLoad (globalName: string) (elemType: string) (zipper: MLIRZipper) : string * MLIRZipper =
        // Inline addressof here since witnessAddressOf is defined later
        let ptrSSA, zipper1 = yieldSSA zipper
        let addrText = render LLVMTemplates.Quot.Memory.addressof {| Result = ptrSSA; GlobalName = globalName |}
        let zipper2 = witnessOpWithResult addrText ptrSSA Pointer zipper1
        witnessLoadStr ptrSSA elemType zipper2

    // ─────────────────────────────────────────────────────────────────
    // SCF Witness Functions - Emit structured control flow operations
    // ─────────────────────────────────────────────────────────────────

    /// Witness an scf.if operation with optional else branch
    /// Returns the result SSA (if resultType is Some) and updated zipper
    /// thenResultSSA/elseResultSSA are the SSA values to yield from each branch
    let witnessSCFIf
            (condSSA: string)
            (thenOps: string list)
            (thenResultSSA: string option)
            (elseOps: string list option)
            (elseResultSSA: string option)
            (resultType: MLIRType option)
            (zipper: MLIRZipper) : string option * MLIRZipper =
        match resultType with
        | None ->
            // Statement form - no result, just scf.yield at end of each branch
            let indent = "      "
            let thenRegion =
                let bodyOps = thenOps |> List.map (fun op -> indent + op) |> String.concat "\n"
                if bodyOps = "" then
                    sprintf "%sscf.yield" indent
                else
                    sprintf "%s\n%sscf.yield" bodyOps indent
            let elseRegion =
                match elseOps with
                | Some ops when not (List.isEmpty ops) ->
                    let elseContent = ops |> List.map (fun op -> indent + op) |> String.concat "\n"
                    sprintf " else {\n%s\n%sscf.yield\n    }" elseContent indent
                | Some _ ->
                    sprintf " else {\n%sscf.yield\n    }" indent
                | None -> ""
            let text = sprintf "scf.if %s {\n%s\n    }%s" condSSA thenRegion elseRegion
            None, witnessVoidOp text zipper

        | Some ty ->
            // Expression form - yields a value from each branch
            let ssaName, zipper' = yieldSSA zipper
            let tyStr = Serialize.mlirType ty
            let indent = "      "
            
            // Get the yield values - use provided SSAs or generate a default
            let thenYieldVal = thenResultSSA |> Option.defaultValue "%undef_then"
            let elseYieldVal = elseResultSSA |> Option.defaultValue "%undef_else"
            
            let thenRegion =
                let bodyOps = thenOps |> List.map (fun op -> indent + op) |> String.concat "\n"
                if bodyOps = "" then
                    sprintf "%sscf.yield %s : %s" indent thenYieldVal tyStr
                else
                    sprintf "%s\n%sscf.yield %s : %s" bodyOps indent thenYieldVal tyStr
            let elseRegion =
                match elseOps with
                | Some ops when not (List.isEmpty ops) ->
                    let bodyOps = ops |> List.map (fun op -> indent + op) |> String.concat "\n"
                    sprintf " else {\n%s\n%sscf.yield %s : %s\n    }" bodyOps indent elseYieldVal tyStr
                | Some _ ->
                    sprintf " else {\n%sscf.yield %s : %s\n    }" indent elseYieldVal tyStr
                | None ->
                    // No else branch but expression form - need a default else
                    sprintf " else {\n%sscf.yield %s : %s\n    }" indent elseYieldVal tyStr
            let text = sprintf "%s = scf.if %s -> (%s) {\n%s\n    }%s" ssaName condSSA tyStr thenRegion elseRegion
            Some ssaName, witnessOpWithResult text ssaName ty zipper'

    /// Witness an scf.for operation with iteration variable and optional iter_args
    /// Returns the result SSAs (from iter_args) and updated zipper
    let witnessSCFFor
            (loopVar: string)
            (loopVarTy: MLIRType)
            (startSSA: string)
            (endSSA: string)
            (stepSSA: string)
            (bodyOps: string list)
            (iterArgs: (string * string * MLIRType) list)  // (argName, initSSA, type)
            (zipper: MLIRZipper) : string list * MLIRZipper =
        let loopVarTyStr = Serialize.mlirType loopVarTy
        let indent = "      "

        // Build iter_args clause if present
        let iterArgsClause, resultTypes, yieldVals =
            if List.isEmpty iterArgs then
                "", "", ""
            else
                let argsList =
                    iterArgs
                    |> List.map (fun (name, initSSA, ty) -> sprintf "%%%s = %s" name initSSA)
                    |> String.concat ", "
                let typesList =
                    iterArgs
                    |> List.map (fun (_, _, ty) -> Serialize.mlirType ty)
                    |> String.concat ", "
                let yieldList =
                    iterArgs
                    |> List.map (fun (name, _, ty) -> sprintf "%%%s_next" name)
                    |> String.concat ", "
                sprintf " iter_args(%s) -> (%s)" argsList typesList,
                sprintf " -> (%s)" typesList,
                yieldList

        // Build body region
        let bodyContent =
            bodyOps
            |> List.map (fun op -> indent + op)
            |> String.concat "\n"

        // Add scf.yield if we have iter_args
        let yieldOp =
            if List.isEmpty iterArgs then ""
            else sprintf "\n%sscf.yield %s : %s" indent yieldVals (iterArgs |> List.map (fun (_, _, ty) -> Serialize.mlirType ty) |> String.concat ", ")

        let header = sprintf "scf.for %%%s = %s to %s step %s%s" loopVar startSSA endSSA stepSSA iterArgsClause
        let text = sprintf "%s {\n%s%s\n    }" header bodyContent yieldOp

        // If we have iter_args, we need result SSAs
        if List.isEmpty iterArgs then
            [], witnessVoidOp text zipper
        else
            // Generate result SSAs for each iter_arg
            let resultSSAs, zipper' =
                iterArgs
                |> List.fold (fun (accSSAs, z) (name, _, _) ->
                    let ssa, z' = yieldSSA z
                    (accSSAs @ [ssa], z')
                ) ([], zipper)
            resultSSAs, witnessVoidOp text zipper'

    /// Witness an scf.while operation with iter_args support.
    /// iterArgsWithNext: list of (argName, initSSA, nextSSA, type)
    ///   - argName: the iter_arg name used in guard/body (e.g., "i_arg")
    ///   - initSSA: the initial value SSA (before loop)
    ///   - nextSSA: the next iteration value SSA (from body, current VarBinding)
    ///   - type: MLIR type
    let witnessSCFWhile
            (condOps: string list)
            (condSSA: string)
            (bodyOps: string list)
            (iterArgsWithNext: (string * string * string * MLIRType) list)  // (argName, initSSA, nextSSA, type)
            (zipper: MLIRZipper) : string list * MLIRZipper =
        let indent = "      "

        // Build iter_args initialization
        let initList, typesList =
            if List.isEmpty iterArgsWithNext then
                "", ""
            else
                let inits =
                    iterArgsWithNext
                    |> List.map (fun (name, initSSA, _, _) -> sprintf "%%%s = %s" name initSSA)
                    |> String.concat ", "
                let types =
                    iterArgsWithNext
                    |> List.map (fun (_, _, _, ty) -> Serialize.mlirType ty)
                    |> String.concat ", "
                inits, types

        // Build condition region
        let condContent =
            condOps
            |> List.map (fun op -> indent + op)
            |> String.concat "\n"

        // Condition yields the iter_args values along with the condition
        let condYield =
            if List.isEmpty iterArgsWithNext then
                sprintf "%sscf.condition(%s)" indent condSSA
            else
                let yieldArgs =
                    iterArgsWithNext
                    |> List.map (fun (name, _, _, _) -> sprintf "%%%s" name)
                    |> String.concat ", "
                sprintf "%sscf.condition(%s) %s : %s" indent condSSA yieldArgs typesList

        // Build body region
        let bodyContent =
            bodyOps
            |> List.map (fun op -> indent + op)
            |> String.concat "\n"

        // Body yields next iteration values (current binding after body ops)
        // SCF while always requires scf.yield in body, even with no iter_args
        let bodyYield =
            if List.isEmpty iterArgsWithNext then
                sprintf "\n%sscf.yield" indent
            else
                let yieldArgs =
                    iterArgsWithNext
                    |> List.map (fun (_, _, nextSSA, _) -> nextSSA)
                    |> String.concat ", "
                sprintf "\n%sscf.yield %s : %s" indent yieldArgs typesList

        // Build the full operation
        // SCF while always requires type annotation, even with no iter_args
        let header =
            if List.isEmpty iterArgsWithNext then
                "scf.while : () -> ()"
            else
                sprintf "scf.while (%s) : (%s) -> (%s)" initList typesList typesList

        let text = sprintf "%s {\n%s\n%s\n    } do {\n%s%s\n    }" header condContent condYield bodyContent bodyYield

        // Return result SSAs
        if List.isEmpty iterArgsWithNext then
            [], witnessVoidOp text zipper
        else
            let resultSSAs, zipper' =
                iterArgsWithNext
                |> List.fold (fun (accSSAs, z) (name, _, _, _) ->
                    let ssa, z' = yieldSSA z
                    (accSSAs @ [ssa], z')
                ) ([], zipper)
            resultSSAs, witnessVoidOp text zipper'

    /// Witness a constant, yielding an SSA name
    let witnessConstant (value: int64) (ty: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let tyStr = Serialize.integerBitWidth ty
        let constParams : ConstantParams = { Result = ssaName; Value = string value; Type = tyStr }
        let text = render ArithTemplates.Quot.Constant.intConst constParams
        ssaName, witnessOpWithResult text ssaName (Integer ty) zipper'

    /// Witness addressof for a string literal, yielding pointer SSA
    let witnessAddressOf (globalName: string) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let text = render LLVMTemplates.Quot.Memory.addressof {| Result = ssaName; GlobalName = globalName |}
        ssaName, witnessOpWithResult text ssaName Pointer zipper'

    /// Witness binary arithmetic operation
    let witnessArith (op: string) (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let tyStr = Serialize.mlirType ty
        let text = sprintf "%s = %s %s, %s : %s" ssaName op lhs rhs tyStr
        ssaName, witnessOpWithResult text ssaName ty zipper'

    /// Witness function call
    let witnessCall (funcName: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper

        // Check tracked parameter count to filter out extra unit arguments
        let actualArgs, actualArgTypes =
            match MLIRState.lookupFuncParamCount funcName zipper'.State with
            | Some paramCount when paramCount < List.length args ->
                // Function has fewer parameters than arguments - truncate to match
                // This handles cases where PSG has unit arguments but function was defined without them
                List.take paramCount args, List.take paramCount argTypes
            | _ ->
                args, argTypes

        // Check if we need to convert argument types to match function declaration
        // This handles polymorphic functions where TVar -> Pointer but call site has concrete types
        let argsStr, typesStr, zipper'' =
            match MLIRState.lookupFuncParamTypes funcName zipper'.State with
            | Some funcParamTypes when List.length funcParamTypes = List.length actualArgs ->
                // Have declared param types - check for mismatches and insert conversions
                let convertedArgs, convertedTypes, z =
                    List.zip3 actualArgs actualArgTypes funcParamTypes
                    |> List.fold (fun (argsAcc, typesAcc, z) (arg, argType, declaredTypeStr) ->
                        let argTypeStr = Serialize.mlirType argType
                        if argTypeStr = declaredTypeStr then
                            // Types match - no conversion needed
                            argsAcc @ [arg], typesAcc @ [declaredTypeStr], z
                        elif declaredTypeStr = "!llvm.ptr" && argTypeStr.StartsWith("i") then
                            // Function expects ptr but we have integer - insert inttoptr
                            let convSSA, z' = yieldSSA z
                            let convText = sprintf "%s = llvm.inttoptr %s : %s to !llvm.ptr" convSSA arg argTypeStr
                            let z'' = witnessOpWithResult convText convSSA Pointer z'
                            argsAcc @ [convSSA], typesAcc @ ["!llvm.ptr"], z''
                        elif argTypeStr = "!llvm.ptr" && declaredTypeStr.StartsWith("i") then
                            // Function expects integer but we have ptr - insert ptrtoint
                            let convSSA, z' = yieldSSA z
                            let convText = sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to %s" convSSA arg declaredTypeStr
                            let z'' = witnessOpWithResult convText convSSA (Serialize.deserializeType declaredTypeStr) z'
                            argsAcc @ [convSSA], typesAcc @ [declaredTypeStr], z''
                        else
                            // Type mismatch with no known conversion - this is a compiler bug
                            failwithf "Type mismatch in call to '%s': argument has type '%s' but function expects '%s'. No conversion available." funcName argTypeStr declaredTypeStr
                    ) ([], [], zipper')
                String.concat ", " convertedArgs, String.concat ", " convertedTypes, z
            | _ ->
                // No declared param types - use actual types
                String.concat ", " actualArgs,
                actualArgTypes |> List.map Serialize.mlirType |> String.concat ", ",
                zipper'

        // Use tracked return type if available (for correct lowered types)
        // Otherwise fall back to the PSG-derived resultType
        let actualRetType, actualRetStr =
            match MLIRState.lookupFuncReturnType funcName zipper''.State with
            | Some trackedRetStr ->
                // Parse the tracked type back to MLIRType for witnessOpWithResult
                Serialize.deserializeType trackedRetStr, trackedRetStr
            | None ->
                resultType, Serialize.mlirType resultType
        let text = sprintf "%s = llvm.call @%s(%s) : (%s) -> %s" ssaName funcName argsStr typesStr actualRetStr
        ssaName, witnessOpWithResult text ssaName actualRetType zipper''

    /// Witness indirect function call (call through function pointer)
    let witnessIndirectCall (funcPtr: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        // First, emit addressof for any arguments that are function symbols (@name)
        // These need to be converted to SSA values before use
        let zipper', argsResolved =
            args
            |> List.fold (fun (z, acc) arg ->
                if arg.StartsWith("@") then
                    // Function symbol - emit addressof to get SSA value
                    let addrSSA, z' = yieldSSA z
                    let addrText = sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" addrSSA arg
                    let z'' = witnessOpWithResult addrText addrSSA Pointer z'
                    z'', addrSSA :: acc
                else
                    z, arg :: acc
            ) (zipper, [])
            |> fun (z, acc) -> z, List.rev acc

        let ssaName, zipper'' = yieldSSA zipper'
        let argsStr = String.concat ", " argsResolved
        let typesStr = argTypes |> List.map Serialize.mlirType |> String.concat ", "
        let retStr = Serialize.mlirType resultType
        // Indirect call uses the function pointer SSA value, not a symbol name
        let text = sprintf "%s = llvm.call %s(%s) : !llvm.ptr, (%s) -> %s" ssaName funcPtr argsStr typesStr retStr
        ssaName, witnessOpWithResult text ssaName resultType zipper''

    /// Witness syscall via inline assembly
    /// sysNum: SSA value for syscall number
    /// args: list of (ssaName, mlirTypeString) pairs for arguments
    /// resultType: MLIRType for return value
    let witnessSyscall (sysNum: string) (args: (string * string) list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let argSSAs = args |> List.map fst
        let argTypes = args |> List.map snd
        let allSSAs = sysNum :: argSSAs
        let allTypes = "i64" :: argTypes  // syscall number is always i64
        let argsStr = String.concat ", " allSSAs
        // Linux x86-64 syscall: rax = syscall number, rdi/rsi/rdx = args
        let constraints =
            match List.length args with
            | 0 -> "={rax},{rax},~{rcx},~{r11},~{memory}"
            | 1 -> "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"
            | 2 -> "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
            | 3 -> "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
            | _ -> "={rax},{rax},{rdi},{rsi},{rdx},{r10},~{rcx},~{r11},~{memory}"
        let typesStr = String.concat ", " allTypes
        let retStr = Serialize.mlirType resultType
        let text = sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"%s\" %s : (%s) -> %s"
                       ssaName constraints argsStr typesStr retStr
        ssaName, witnessOpWithResult text ssaName resultType zipper'

    /// Witness unreachable (for exit syscall paths)
    let witnessUnreachable (zipper: MLIRZipper) : MLIRZipper =
        witnessVoidOp "llvm.unreachable" zipper

    // ─────────────────────────────────────────────────────────────────
    // Lambda Support
    // ─────────────────────────────────────────────────────────────────

    /// Yield next lambda function name on demand (codata yield)
    let yieldLambdaName (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLambdaName zipper.State
        name, { zipper with State = newState }

    /// Yield lambda name for a specific node ID
    /// Entry point Lambdas (in EntryPointLambdaIds) become "main"
    let yieldLambdaNameForNode (nodeId: int) (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLambdaNameForNode nodeId zipper.State
        name, { zipper with State = newState }

    /// Add a completed lambda function to the zipper
    /// Also records the function's return type and parameter count for call site generation
    let addCompletedFunction (func: MLIRFunc) (zipper: MLIRZipper) : MLIRZipper =
        let returnTypeStr = Serialize.mlirType func.ReturnType
        let paramCount = List.length func.Parameters
        let state1 = MLIRState.recordFuncReturnType func.Name returnTypeStr zipper.State
        let state2 = MLIRState.recordFuncParamCount func.Name paramCount state1
        { zipper with
            CompletedFunctions = func :: zipper.CompletedFunctions
            State = state2 }

    // ─────────────────────────────────────────────────────────────────
    // Freestanding Entry Point Wrapper
    // For freestanding binaries, we need a _start function that calls main
    // and then calls the exit syscall (since there's no libc to do this)
    // ─────────────────────────────────────────────────────────────────

    /// Add a _start wrapper for freestanding binaries
    /// _start calls main(argc, argv) and then calls exit(result)
    /// This is necessary because freestanding binaries can't return from main
    let addFreestandingEntryPoint (zipper: MLIRZipper) : MLIRZipper =
        // Generate _start function that:
        // 1. Calls main(argc, argv)
        // 2. Calls exit syscall with main's return value
        //
        // For Linux x86_64:
        //   At _start entry, the stack contains: argc, argv[0], argv[1], ..., NULL, envp...
        //   We load argc from (%rsp) and compute argv as %rsp+8
        let startOps = [
            // Load argc from stack: argc is at (%rsp)
            { Text = "%sp = llvm.inline_asm \"mov %rsp, $0\", \"=r,~{memory}\" : () -> i64"; Results = ["%sp", Pointer] }
            { Text = "%argc_ptr = llvm.inttoptr %sp : i64 to !llvm.ptr"; Results = ["%argc_ptr", Pointer] }
            { Text = "%argc_64 = llvm.load %argc_ptr : !llvm.ptr -> i64"; Results = ["%argc_64", Integer I64] }
            { Text = "%argc = arith.trunci %argc_64 : i64 to i32"; Results = ["%argc", Integer I32] }
            // Compute argv = %rsp + 8
            { Text = "%eight = arith.constant 8 : i64"; Results = ["%eight", Integer I64] }
            { Text = "%argv_addr = arith.addi %sp, %eight : i64"; Results = ["%argv_addr", Integer I64] }
            { Text = "%argv = llvm.inttoptr %argv_addr : i64 to !llvm.ptr"; Results = ["%argv", Pointer] }
            // Call main(argc, argv)
            { Text = "%result = llvm.call @main(%argc, %argv) : (i32, !llvm.ptr) -> i32"; Results = ["%result", Integer I32] }
            // Extend result to i64 for syscall
            { Text = "%result_64 = arith.extsi %result : i32 to i64"; Results = ["%result_64", Integer I64] }
            // Call exit syscall (60 on Linux x86_64)
            { Text = "%syscall_num = arith.constant 60 : i64"; Results = ["%syscall_num", Integer I64] }
            { Text = "%ignored = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %syscall_num, %result_64 : (i64, i64) -> i64"; Results = ["%ignored", Integer I64] }
            // Unreachable - exit never returns
            { Text = "llvm.unreachable"; Results = [] }
        ]

        let startFunc: MLIRFunc = {
            Name = "_start"
            Parameters = []  // _start takes no arguments (reads from stack)
            ReturnType = Unit  // _start never returns (calls exit syscall)
            Blocks = [{
                Label = "entry"
                Arguments = []
                Operations = startOps
            }]
            Attributes = []
            IsInternal = false  // _start must be exported as the entry point
        }

        { zipper with CompletedFunctions = startFunc :: zipper.CompletedFunctions }

    // ─────────────────────────────────────────────────────────────────
    // Platform Helper Registration
    // ─────────────────────────────────────────────────────────────────

    /// Register a platform helper function to be emitted at module level.
    /// Helpers are only emitted once (tracked by name in Set).
    let registerPlatformHelper (helperName: string) (helperBody: string) (zipper: MLIRZipper) : MLIRZipper =
        if Set.contains helperName zipper.PlatformHelpers then
            zipper  // Already registered
        else
            { zipper with PlatformHelpers = Set.add helperName zipper.PlatformHelpers }

    /// Platform helper definitions (MLIR func.func bodies)
    /// These are stored separately from the Set of names for actual emission
    let mutable private helperBodies : Map<string, string> = Map.empty

    /// Register a helper with its body (call once per helper name)
    let registerPlatformHelperWithBody (helperName: string) (helperBody: string) (zipper: MLIRZipper) : MLIRZipper =
        if not (Map.containsKey helperName helperBodies) then
            helperBodies <- Map.add helperName helperBody helperBodies
        registerPlatformHelper helperName helperBody zipper

    /// Get the body for a registered helper
    let getPlatformHelperBody (helperName: string) : string option =
        Map.tryFind helperName helperBodies

    // ─────────────────────────────────────────────────────────────────
    // Extraction - Comonad extract: collapse accumulated context to final value
    // This is the ONLY place where MLIR text is actually produced
    // ─────────────────────────────────────────────────────────────────

    // Use Serialize.escape from MLIRTypes for string escaping

    /// Extract: comonad operation that collapses zipper to complete MLIR text
    /// This is the final observation - the accumulated context becomes a value
    let extract (zipper: MLIRZipper) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine("module {") |> ignore

        // Emit globals (string literals first, then externs, then buffers)
        for glb in List.rev zipper.Globals do
            match glb with
            | StringLiteral (name, content, len) ->
                let escaped = Serialize.escape content
                sb.AppendLine(sprintf "  llvm.mlir.global internal constant @%s(\"%s\\00\") : !llvm.array<%d x i8>"
                    name escaped len) |> ignore
            | ExternFunc (name, signature) ->
                sb.AppendLine(sprintf "  llvm.func @%s%s attributes {sym_visibility = \"private\"}"
                    name signature) |> ignore
            | StaticBuffer (name, size) ->
                // Zero-initialized buffer - no initializer needed
                sb.AppendLine(sprintf "  llvm.mlir.global internal @%s() : !llvm.array<%d x i8>"
                    name size) |> ignore
            | MutableGlobal (name, mlirType) ->
                // Module-level mutable - zero-initialized global variable
                // Use internal linkage (not exported)
                sb.AppendLine(sprintf "  llvm.mlir.global internal @%s() : %s"
                    name mlirType) |> ignore

        // Emit platform helper functions (func.func dialect, will be lowered to LLVM)
        for helperName in zipper.PlatformHelpers do
            match getPlatformHelperBody helperName with
            | Some body ->
                sb.AppendLine("") |> ignore
                sb.AppendLine(body) |> ignore
            | None -> ()

        // Emit completed lambda functions
        for func in List.rev zipper.CompletedFunctions do
            let paramsStr = 
                func.Parameters 
                |> List.mapi (fun i (name, ty) -> sprintf "%%arg%d: %s" i (Serialize.mlirType ty))
                |> String.concat ", "
            let retStr = Serialize.mlirType func.ReturnType
            let attrs = 
                if List.isEmpty func.Attributes then ""
                else sprintf " attributes {%s}" (String.concat ", " (List.map (fun a -> sprintf "sym_visibility = \"%s\"" a) func.Attributes))
            let visibility = if func.IsInternal then "internal " else ""
            sb.AppendLine(sprintf "  llvm.func %s@%s(%s) -> %s%s {" visibility func.Name paramsStr retStr attrs) |> ignore
            // For MLIR LLVM dialect, entry block should not have explicit label when function has parameters
            let hasParams = not (List.isEmpty func.Parameters)
            for i, block in List.indexed func.Blocks do
                // Skip block label for entry block (first block) when function has parameters
                if i = 0 && hasParams then
                    () // No block label - parameters serve as implicit entry block arguments
                else
                    sb.AppendLine(sprintf "  ^%s:" block.Label) |> ignore
                for op in block.Operations do
                    sb.AppendLine(sprintf "    %s" op.Text) |> ignore
            sb.AppendLine("  }") |> ignore

        // Emit current operations (if any)
        for op in List.rev zipper.CurrentOps do
            sb.AppendLine(sprintf "  %s" op.Text) |> ignore

        sb.AppendLine("}") |> ignore

        sb.ToString()

    /// Get all emitted operations as text (without module wrapper)
    let getOperationsText (zipper: MLIRZipper) : string =
        zipper.CurrentOps
        |> List.rev
        |> List.map (fun op -> op.Text)
        |> String.concat "\n"

// ═══════════════════════════════════════════════════════════════════
// Dual-Zipper Transfer Type
// ═══════════════════════════════════════════════════════════════════

/// Result of transferring a PSG node to MLIR via zipper
type TransferResult =
    /// Produced an SSA value with type
    | TRValue of ssa: string * mlirType: string
    /// Produced no value (unit/void)
    | TRVoid
    /// Transfer failed with error
    | TRError of message: string
    /// Built-in operator/function (handled inline at application sites)
    | TRBuiltin of name: string

module TransferResult =
    let isValue = function TRValue _ -> true | _ -> false
    let isVoid = function TRVoid -> true | _ -> false
    let isError = function TRError _ -> true | _ -> false
    let isBuiltin = function TRBuiltin _ -> true | _ -> false

    let getSSA = function TRValue (ssa, _) -> Some ssa | _ -> None
    let getType = function TRValue (_, t) -> Some t | _ -> None
    let getBuiltinName = function TRBuiltin name -> Some name | _ -> None
