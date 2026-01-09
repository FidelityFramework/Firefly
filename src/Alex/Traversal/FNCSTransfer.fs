/// FNCSTransfer - Witness-based transfer from FNCS SemanticGraph to MLIR
///
/// ARCHITECTURAL FOUNDATION (Coeffects & Codata):
/// This module witnesses the FNCS SemanticGraph structure and produces MLIR
/// via the MLIRZipper codata accumulator.
///
/// KEY DESIGN PRINCIPLES:
/// 1. Post-order traversal: Children before parents (SSAs available when parent visited)
/// 2. Dispatch on SemanticKind ONLY - no pattern matching on symbol names
/// 3. Platform bindings via SemanticKind.PlatformBinding marker → PlatformDispatch
/// 4. Codata vocabulary: witness, observe, yield, bind, recall, extract
///
/// This replaces the deleted FNCSEmitter.fs antipattern.
module Alex.Traversal.FNCSTransfer

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
module LitWitness = Alex.Witnesses.LiteralWitness
module BindWitness = Alex.Witnesses.BindingWitness
module AppWitness = Alex.Witnesses.ApplicationWitness
module CFWitness = Alex.Witnesses.ControlFlowWitness
module MemWitness = Alex.Witnesses.MemoryWitness
module LambdaWitness = Alex.Witnesses.LambdaWitness
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis
module SSAAssign = Alex.Preprocessing.SSAAssignment
// NOTE: SatApps removed - application saturation now happens in FNCS during type checking
open Alex.Patterns.SemanticPatterns

// ═══════════════════════════════════════════════════════════════════
// Type Mapping: NativeType → MLIRType (delegated to TypeMapping module)
// ═══════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════
// Main Transfer Fold
// ═══════════════════════════════════════════════════════════════════

/// Witness a single node based on its SemanticKind
/// Dispatch is ONLY on SemanticKind - no symbol name pattern matching
let witnessNode (graph: SemanticGraph) (node: SemanticNode) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match node.Kind with
    // Literals
    | SemanticKind.Literal lit ->
        let zipper', result = LitWitness.witness lit zipper
        // Bind result to this node for future reference
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Variable references
    | SemanticKind.VarRef (name, defId) ->
        let zipper', result = BindWitness.witnessVarRef name defId graph zipper
        // CRITICAL: Bind result to this node for future reference
        // This is needed when VarRef is used as value in a Binding or TypeAnnotation
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | TRBuiltin opName ->
            // Built-in operator - bind a marker so TypeAnnotation can forward it
            let marker = "$builtin:" + opName
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) marker "func" zipper'
            zipper'', result
        | _ -> zipper', result

    // Platform bindings (the ONLY place platform calls are recognized)
    | SemanticKind.PlatformBinding entryPoint ->
        // Platform binding node itself doesn't produce a value
        // The Application that uses it will call witnessPlatformBinding
        zipper, TRVoid

    // Compiler intrinsics (e.g., NativePtr.toNativeInt)
    | SemanticKind.Intrinsic intrinsicInfo ->
        // Intrinsic node itself produces a function value
        // The Application will handle generating actual MLIR code
        // Bind a marker so Application can recognize and handle it
        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("$intrinsic:" + intrinsicInfo.FullName) "func" zipper
        zipper', TRValue ("$intrinsic:" + intrinsicInfo.FullName, "func")

    // Function applications
    | SemanticKind.Application (funcId, argIds) ->
        let zipper', result = AppWitness.witness funcId argIds node.Type graph zipper
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Sequential expressions
    | SemanticKind.Sequential nodeIds ->
        let zipper', result = CFWitness.witnessSequential nodeIds zipper
        // Bind result to this node for TypeAnnotation to recall
        // But only if we have a non-empty SSA value
        match result with
        | TRValue (ssa, ty) when ssa <> "" ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Bindings
    | SemanticKind.Binding (name, isMutable, _isRecursive, _isEntryPoint) ->
        // Get the value node (first child)
        match node.Children with
        | valueId :: _ ->
            BindWitness.witness name isMutable valueId node zipper
        | [] ->
            zipper, TRError "Binding has no value child"

    // Module definitions (container - children already processed)
    | SemanticKind.ModuleDef (name, _members) ->
        zipper, TRVoid

    // Type definitions (don't generate runtime code)
    | SemanticKind.TypeDef (_name, _kind, _members) ->
        zipper, TRVoid

    // Record expressions - construct a record value
    | SemanticKind.RecordExpr (fields, copyFrom) ->
        MemWitness.witnessRecordExpr fields copyFrom node zipper

    // Lambdas - delegate to LambdaWitness
    | SemanticKind.Lambda (params', bodyId) ->
        LambdaWitness.witness params' bodyId node zipper

    // Type annotations - pass through the inner expression's value
    | SemanticKind.TypeAnnotation (exprId, _annotatedType) ->
        // Type annotation doesn't generate code - just forward the inner expression's value
        // First, check if the inner expression is unit-typed (void calls don't have SSA values)
        match SemanticGraph.tryGetNode exprId graph with
        | Some innerNode ->
            let innerType = mapType innerNode.Type
            match innerType with
            | Unit ->
                // Unit-typed expression produces no SSA value - this is valid
                zipper, TRVoid
            | _ ->
                // Non-unit expression should have an SSA value
                match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
                | Some (ssa, ty) ->
                    let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
                    zipper1, TRValue (ssa, ty)
                | None ->
                    // Try pre-computed SSA from coeffect
                    match MLIRState.lookupPrecomputedSSA (NodeId.value exprId) zipper.State with
                    | Some ssaName ->
                        let tyStr = Serialize.mlirType innerType
                        let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssaName tyStr zipper
                        zipper1, TRValue (ssaName, tyStr)
                    | None ->
                        zipper, TRError (sprintf "TypeAnnotation inner expr %A not computed (non-unit)" exprId)
        | None ->
            zipper, TRError (sprintf "TypeAnnotation inner expr %A not found in graph" exprId)

    // Mutable set
    | SemanticKind.Set (targetId, valueId) ->
        match SemanticGraph.tryGetNode targetId graph with
        | Some targetNode ->
            match targetNode.Kind with
            | SemanticKind.VarRef (name, _) ->
                // Get the value SSA first
                match MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
                | Some (valueSSA, valueType) ->
                    // Check if this is a module-level mutable
                    match MLIRZipper.lookupModuleLevelMutable name zipper with
                    | Some (_bindingId, globalName) ->
                        // Module-level mutable: use global store template primitive
                        let zipper' = MLIRZipper.witnessGlobalStore globalName valueSSA valueType zipper
                        zipper', TRVoid
                    | None ->
                        // Local mutable: pure SSA rebinding (for SCF iter_args)
                        let zipper' = MLIRZipper.bindVar name valueSSA valueType zipper
                        zipper', TRVoid
                | None ->
                    zipper, TRError (sprintf "Set: value for '%s' not computed" name)
            | _ ->
                // Non-variable target (field set, array set, etc.) - use store
                match MLIRZipper.recallNodeSSA (string (NodeId.value targetId)) zipper,
                      MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
                | Some (targetSSA, _), Some (valueSSA, valueType) ->
                    let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA targetSSA valueType
                    let zipper' = MLIRZipper.witnessVoidOp storeText zipper
                    zipper', TRVoid
                | _ ->
                    zipper, TRError "Set: target or value not computed"
        | None ->
            zipper, TRError "Set: target node not found"

    // Control flow - delegates to ControlFlowWitness module
    | SemanticKind.IfThenElse (guardId, thenId, elseIdOpt) ->
        CFWitness.witnessIfThenElse guardId thenId elseIdOpt node zipper

    | SemanticKind.WhileLoop (guardId, bodyId) ->
        CFWitness.witnessWhileLoop guardId bodyId node zipper

    | SemanticKind.ForLoop (varName, startId, finishId, isUp, bodyId) ->
        CFWitness.witnessForLoop varName startId finishId isUp bodyId node zipper

    | SemanticKind.Match (scrutineeId, cases) ->
        CFWitness.witnessMatch scrutineeId cases node graph zipper

    // Interpolated strings
    | SemanticKind.InterpolatedString parts ->
        // TODO: Implement string interpolation lowering
        zipper, TRError "InterpolatedString not yet implemented"

    // Array/collection indexing
    | SemanticKind.IndexGet (collectionId, indexId) ->
        MemWitness.witnessIndexGet collectionId indexId node zipper

    | SemanticKind.IndexSet (collectionId, indexId, valueId) ->
        MemWitness.witnessIndexSet collectionId indexId valueId zipper

    // Address-of operator
    | SemanticKind.AddressOf (exprId, isMutable) ->
        MemWitness.witnessAddressOf exprId isMutable node graph zipper

    // Tuple expressions - construct a tuple value
    | SemanticKind.TupleExpr elementIds ->
        MemWitness.witnessTupleExpr elementIds node zipper

    // TraitCall - SRTP member resolution
    | SemanticKind.TraitCall (memberName, typeArgs, argId) ->
        MemWitness.witnessTraitCall memberName typeArgs argId node zipper

    // Array expressions - construct an array
    | SemanticKind.ArrayExpr elementIds ->
        MemWitness.witnessArrayExpr elementIds node zipper

    // List expressions - construct a list
    | SemanticKind.ListExpr elementIds ->
        MemWitness.witnessListExpr elementIds node zipper

    // Field access: expr.fieldName
    | SemanticKind.FieldGet (exprId, fieldName) ->
        MemWitness.witnessFieldGet exprId fieldName node graph zipper

    // Field set: expr.fieldName <- value
    | SemanticKind.FieldSet (exprId, fieldName, valueId) ->
        MemWitness.witnessFieldSet exprId fieldName valueId graph zipper

    // Pattern bindings (variables introduced by match patterns)
    | SemanticKind.PatternBinding name ->
        // PatternBinding represents a variable bound by a pattern.
        // The binding was set up in BeforeRegion for MatchCaseRegion via emitPatternBindings.
        // Here we just look up the bound SSA and forward it.
        match Map.tryFind name zipper.State.VarBindings with
        | Some (ssa, ty) ->
            let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper', TRValue (ssa, ty)
        | None ->
            // Variable not yet bound - this might happen for patterns not yet implemented
            zipper, TRError (sprintf "PatternBinding '%s' not found in bindings" name)

    // Union case construction (discriminated union value creation)
    | SemanticKind.UnionCase (caseName, caseIndex, payloadOpt) ->
        let zipper', result = MemWitness.witnessUnionCase caseName caseIndex payloadOpt node zipper
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Error nodes
    | SemanticKind.Error msg ->
        zipper, TRError msg

    // Catch-all for unimplemented kinds
    | kind ->
        zipper, TRError (sprintf "SemanticKind not yet implemented: %A" kind)






/// Core transfer implementation with optional error collection
let private transferGraphCore
    (graph: SemanticGraph)
    (isFreestanding: bool)
    (collectErrors: bool)
    : string * string list =

    // NOTE: Application saturation (pipes, curried apps) now happens in FNCS during type checking.
    // The SemanticGraph received here is already fully saturated.
    // See: ~/repos/fsnative/src/Compiler/Checking.Native/CheckExpressions.fs (SynExpr.App handling)

    // Initialize platform bindings
    Alex.Bindings.Console.ConsoleBindings.registerBindings()
    Alex.Bindings.Console.ConsoleBindings.registerConsoleIntrinsics()
    Alex.Bindings.Process.ProcessBindings.registerBindings()
    Alex.Bindings.Time.TimeBindings.registerBindings()
    Alex.Bindings.WebView.WebViewBindings.registerBindings()
    Alex.Bindings.DynamicLib.DynamicLibBindings.registerBindings()
    Alex.Bindings.GTK.GTKBindings.registerBindings()
    Alex.Bindings.WebKit.WebKitBindings.registerBindings()

    // Pre-analyze graph (ONCE, before transfer begins)
    // This adheres to the photographer principle: observe structure, don't compute during transfer
    let entryPointLambdaIds = MutAnalysis.findEntryPointLambdaIds graph
    let analysisResult = MutAnalysis.analyze graph
    // SSA assignment is a coeffect - computed BEFORE transfer, not during
    let ssaAssignment = SSAAssign.assignSSA graph
    let initialZipper =
        MLIRZipper.createWithAnalysis
            entryPointLambdaIds
            analysisResult.AddressedMutableBindings
            analysisResult.ModifiedVarsInLoopBodies
            analysisResult.ModuleLevelMutableBindings
            ssaAssignment
    let scfHook = CFWitness.createSCFRegionHook graph

    // Error accumulator (only used if collectErrors)
    let mutable errors = []

    // Traverse with SCF region hooks
    let traversedZipper =
        Traversal.foldWithSCFRegions
            LambdaWitness.preBindParams
            (Some scfHook)
            (fun zipper node ->
                let zipper', result = witnessNode graph node zipper
                if collectErrors then
                    match result with
                    | TRError msg -> errors <- msg :: errors
                    | _ -> ()
                zipper')
            initialZipper
            graph

    // Add freestanding entry point if needed
    let finalZipper =
        if isFreestanding then MLIRZipper.addFreestandingEntryPoint traversedZipper
        else traversedZipper

    MLIRZipper.extract finalZipper, List.rev errors

/// Transfer an entire SemanticGraph to MLIR
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    fst (transferGraphCore graph isFreestanding false)

/// Transfer a graph and return both MLIR text and any errors
let transferGraphWithDiagnostics (graph: SemanticGraph) (isFreestanding: bool) : string * string list =
    transferGraphCore graph isFreestanding true
