/// MLIRTransfer - Canonical Transfer from SemanticGraph to MLIR
///
/// CANONICAL ARCHITECTURE (January 2026):
/// See: mlir_transfer_canonical_architecture memory
///
/// The Three Concerns:
/// - PSGZipper: Pure navigation (Focus, Path, Graph)
/// - TransferCoeffects: Pre-computed, immutable coeffects
/// - MLIRAccumulator: Mutable fold state
///
/// Transfer is a FOLD: witnesses RETURN codata, the fold accumulates.
/// NO push-model emission. NO mutable coeffects. NO state in zipper.
module Alex.Traversal.MLIRTransfer

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.Arith.Templates
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.Bindings.PlatformTypes
open Alex.Patterns.SemanticPatterns
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices
module PlatformRes = PSGElaboration.PlatformBindingResolution

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Local versions for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/// Get single pre-assigned SSA for a node (coeffects version)
let private requireSSAFromCoeffs (nodeId: NodeId) (coeffs: TransferCoeffects) : SSA =
    match SSAAssign.lookupSSA nodeId coeffs.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get all pre-assigned SSAs for a node (coeffects version)
let private requireSSAsFromCoeffs (nodeId: NodeId) (coeffs: TransferCoeffects) : SSA list =
    match SSAAssign.lookupSSAs nodeId coeffs.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Derive byte length of string in UTF-8 encoding
let deriveStringByteLength (s: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(s)

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Compute all coeffects from graph (called ONCE at start)
let computeCoeffects (graph: SemanticGraph) (isFreestanding: bool) : TransferCoeffects =
    let hostPlatform = TargetPlatform.detectHost()
    let runtimeMode = if isFreestanding then Freestanding else Console
    {
        SSA = SSAAssign.assignSSA hostPlatform.Arch graph
        Platform = PlatformRes.analyze graph runtimeMode hostPlatform.OS hostPlatform.Arch
        Mutability = MutAnalysis.analyze graph
        PatternBindings = PatternAnalysis.analyze graph
        Strings = StringCollect.collect graph
        YieldStates = YieldStateIndices.run graph
        EntryPointLambdaIds = MutAnalysis.findEntryPointLambdaIds graph
    }

// ═══════════════════════════════════════════════════════════════════════════
// RECURSIVE TRAVERSAL WITH WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Visit a node and its children, returning comprehensive WitnessOutput
///
/// ARCHITECTURAL PRINCIPLE: This is the FOLD. It:
/// 1. Navigates via zipper (Huet-style, purely positional)
/// 2. Classifies via SemanticKind match (semantic lens)
/// 3. Calls witnesses which RETURN WitnessOutput (codata)
/// 4. ACCUMULATES the returned codata (single point of accumulation)
///
/// The fold is the only place that adds to TopLevelOps.
/// Witnesses return; the fold accumulates.
let rec private visitNode
    (ctx: WitnessContext)
    (z: PSGZipper)
    (nodeId: NodeId)
    : WitnessOutput =

    let nodeIdVal = NodeId.value nodeId
    let acc = ctx.Accumulator

    // DAG handling: if already visited, recall the cached result
    if MLIRAccumulator.isVisited nodeIdVal acc then
        match MLIRAccumulator.recallNode nodeIdVal acc with
        | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
        | None -> WitnessOutput.empty
    else
        MLIRAccumulator.markVisited nodeIdVal acc

        match SemanticGraph.tryGetNode nodeId ctx.Graph with
        | None ->
            WitnessOutput.error (sprintf "Node %d not found" nodeIdVal)
        | Some node ->
            // Classify and witness the node
            let output = classifyAndWitness ctx z node

            // ACCUMULATE: The fold adds top-level ops (witnesses just return them)
            MLIRAccumulator.addTopLevelOps output.TopLevelOps acc

            // Record the result for DAG handling
            recordResult nodeIdVal output.Result acc

            // Return with TopLevelOps cleared (they've been accumulated)
            { output with TopLevelOps = [] }

/// Classify node by SemanticKind and witness it
/// Returns WitnessOutput with all codata (inline ops, top-level ops, result)
///
/// NOTE: Witnesses currently return (MLIROp list * TransferResult).
/// This function bridges to WitnessOutput. As witnesses are updated,
/// they will return WitnessOutput directly.
and private classifyAndWitness
    (ctx: WitnessContext)
    (z: PSGZipper)
    (node: SemanticNode)
    : WitnessOutput =

    let acc = ctx.Accumulator

    match node.Kind with
    // ─────────────────────────────────────────────────────────────────────
    // LITERALS
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Literal lit ->
        let ops, result = Alex.Witnesses.LiteralWitness.witness ctx.Coeffects.SSA ctx.Coeffects.Platform.TargetArch node.Id lit
        WitnessOutput.inline' ops result

    // ─────────────────────────────────────────────────────────────────────
    // VARIABLE REFERENCES - Direct pattern + template handling
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.VarRef (name, defIdOpt) ->
        let mutability = ctx.Coeffects.Mutability
        let arch = ctx.Coeffects.Platform.TargetArch
        
        // Helper: get type from definition node
        let getDefType nodeId =
            match SemanticGraph.tryGetNode nodeId ctx.Graph with
            | Some n -> mapNativeTypeWithGraphForArch arch ctx.Graph n.Type
            | None -> MLIRTypes.i32
        
        // 1. Module-level mutable: addressof + load
        match mutability with
        | ModuleLevelMutableRef name globalName ->
            let ssas = requireSSAs node.Id ctx
            let ptrSSA, loadSSA = ssas.[0], ssas.[1]
            let elemType = defIdOpt |> Option.map getDefType |> Option.defaultValue MLIRTypes.i32
            let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GFunc globalName))
            let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
            WitnessOutput.inline' [addrOp; loadOp] (TRValue { SSA = loadSSA; Type = elemType })
        | _ ->
        
        match defIdOpt with
        | None ->
            // No defId - lookup by name in accumulator
            match MLIRAccumulator.recallVar name acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None -> WitnessOutput.error (sprintf "Variable '%s' has no definition" name)
        | Some defId ->
            let defIdVal = NodeId.value defId
            
            // 2. Addressed mutable: load from alloca
            match mutability with
            | AddressedMutableRef defIdVal () ->
                match SSAAssign.lookupSSAs defId ctx.Coeffects.SSA with
                | Some ssas when ssas.Length >= 2 ->
                    let allocaSSA = ssas.[1]
                    let loadSSA = requireSSA node.Id ctx
                    let elemType = getDefType defId
                    let loadOp = MLIROp.LLVMOp (Load (loadSSA, allocaSSA, elemType, NotAtomic))
                    WitnessOutput.inline' [loadOp] (TRValue { SSA = loadSSA; Type = elemType })
                | _ -> WitnessOutput.error (sprintf "No alloca SSA for addressed mutable '%s'" name)
            | _ ->
            
            // 3. Captured variable
            if MLIRAccumulator.isCapturedVariable name acc then
                match MLIRAccumulator.recallVar name acc with
                | Some (ptrSSA, _) when MLIRAccumulator.isCapturedMutable name acc ->
                    let elemType = getDefType defId
                    let loadSSA = requireSSA node.Id ctx
                    let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
                    WitnessOutput.inline' [loadOp] (TRValue { SSA = loadSSA; Type = elemType })
                | Some (ssa, ty) ->
                    WitnessOutput.value { SSA = ssa; Type = ty }
                | None ->
                    WitnessOutput.error (sprintf "Captured variable '%s' not in scope" name)
            else
            
            // 4. Function parameter or let-bound value: lookup in accumulator/coeffects
            match MLIRAccumulator.recallVar name acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None ->
            match MLIRAccumulator.recallNode defIdVal acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None ->
            match SSAAssign.lookupSSA defId ctx.Coeffects.SSA with
            | Some ssa -> WitnessOutput.value { SSA = ssa; Type = getDefType defId }
            | None ->
            // Function reference marker
            WitnessOutput.inline' [] (TRBuiltin (name, []))

    // ─────────────────────────────────────────────────────────────────────
    // MODULE - Visit children, no ops produced
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.ModuleDef (_name, members) ->
        visitChildren ctx z members

    // ─────────────────────────────────────────────────────────────────────
    // SEQUENTIAL - Visit children in order, return last result
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Sequential exprs ->
        visitChildren ctx z exprs

    // ─────────────────────────────────────────────────────────────────────
    // LET EXPRESSION - Direct pattern + template handling
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Binding (name, isMut, _isRec, _isEntry) ->
        match node.Children with
        | [valueId] ->
            let valueOutput = visitNode ctx z valueId
            let nodeIdVal = NodeId.value node.Id
            let mutability = ctx.Coeffects.Mutability
            let arch = ctx.Coeffects.Platform.TargetArch
            
            // 1. Module-level mutable: emit global definition
            match mutability with
            | ModuleLevelMutable nodeIdVal globalName when isMut ->
                let mlirType = mapNativeTypeWithGraphForArch arch ctx.Graph node.Type
                let globalOp = MLIROp.LLVMOp (GlobalDef (globalName, "zeroinitializer", mlirType, false))
                { InlineOps = valueOutput.InlineOps
                  TopLevelOps = [globalOp] @ valueOutput.TopLevelOps
                  Result = TRVoid }
            | _ ->
            
            // 2. Addressed mutable: alloca + store
            match mutability with
            | AddressedMutable nodeIdVal () when isMut ->
                match valueOutput.Result with
                | TRValue valueVal ->
                    match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                    | Some ssas when ssas.Length >= 2 ->
                        let oneSSA = ssas.[0]
                        let allocaSSA = ssas.[1]
                        let elemType = valueVal.Type
                        let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
                        let allocaOp = MLIROp.LLVMOp (Alloca (allocaSSA, oneSSA, elemType, None))
                        let storeOp = MLIROp.LLVMOp (Store (valueVal.SSA, allocaSSA, elemType, NotAtomic))
                        // Record the alloca pointer for later VarRef lookups
                        MLIRAccumulator.bindVar name allocaSSA MLIRTypes.ptr acc
                        MLIRAccumulator.bindNode nodeIdVal allocaSSA MLIRTypes.ptr acc
                        { InlineOps = valueOutput.InlineOps @ [oneOp; allocaOp; storeOp]
                          TopLevelOps = valueOutput.TopLevelOps
                          Result = TRValue { SSA = allocaSSA; Type = MLIRTypes.ptr } }
                    | _ ->
                        WitnessOutput.error (sprintf "No SSAs for mutable '%s'" name)
                | TRError msg ->
                    { valueOutput with Result = TRError msg }
                | _ ->
                    WitnessOutput.error (sprintf "Mutable '%s' has void value" name)
            | _ ->
            
            // 3. Regular binding: value flows through, record association
            match valueOutput.Result with
            | TRValue v ->
                MLIRAccumulator.bindVar name v.SSA v.Type acc
                MLIRAccumulator.bindNode nodeIdVal v.SSA v.Type acc
                valueOutput
            | TRVoid ->
                valueOutput
            | TRError _ ->
                valueOutput
            | TRBuiltin _ ->
                valueOutput
        | _ ->
            WitnessOutput.error (sprintf "Let expression '%s' has wrong number of children" name)

    // ─────────────────────────────────────────────────────────────────────
    // SET (Mutable assignment) - Coeffect-driven classification
    // The target is a NodeId - we look at the target node to classify it
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Set (targetId, valueId) ->
        let valueOutput = visitNode ctx z valueId
        let mutability = ctx.Coeffects.Mutability
        
        // Look up the target node to understand what we're assigning to
        match SemanticGraph.tryGetNode targetId ctx.Graph with
        | None ->
            WitnessOutput.error (sprintf "Set target node %d not found" (NodeId.value targetId))
        | Some targetNode ->
            match valueOutput.Result with
            | TRValue v ->
                // Classify the target via coeffects
                match targetNode.Kind with
                | SemanticKind.VarRef (name, defIdOpt) ->
                    // 1. Module-level mutable: addressof + store
                    match mutability with
                    | ModuleLevelMutableRef name globalName ->
                        let addrSSA = requireSSA node.Id ctx
                        let addrOp = MLIROp.LLVMOp (AddressOf (addrSSA, GFunc globalName))
                        let storeOp = MLIROp.LLVMOp (Store (v.SSA, addrSSA, v.Type, NotAtomic))
                        { valueOutput with
                            InlineOps = valueOutput.InlineOps @ [addrOp; storeOp]
                            Result = TRVoid }
                    | _ ->
                    
                    // 2. Captured mutable: store through captured pointer
                    if MLIRAccumulator.isCapturedMutable name acc then
                        match MLIRAccumulator.recallVar name acc with
                        | Some (ptrSSA, _) ->
                            let storeOp = MLIROp.LLVMOp (Store (v.SSA, ptrSSA, v.Type, NotAtomic))
                            { valueOutput with
                                InlineOps = valueOutput.InlineOps @ [storeOp]
                                Result = TRVoid }
                        | None ->
                            { valueOutput with Result = TRError (sprintf "Captured mutable '%s' not in scope" name) }
                    else
                    
                    // 3. Addressed mutable: store to alloca
                    match defIdOpt with
                    | Some defId ->
                        let defIdVal = NodeId.value defId
                        match mutability with
                        | AddressedMutableRef defIdVal () ->
                            match SSAAssign.lookupSSAs defId ctx.Coeffects.SSA with
                            | Some ssas when ssas.Length >= 2 ->
                                let allocaSSA = ssas.[1]
                                let storeOp = MLIROp.LLVMOp (Store (v.SSA, allocaSSA, v.Type, NotAtomic))
                                { valueOutput with
                                    InlineOps = valueOutput.InlineOps @ [storeOp]
                                    Result = TRVoid }
                            | _ ->
                                { valueOutput with Result = TRError (sprintf "No alloca for '%s'" name) }
                        | _ ->
                            { valueOutput with Result = TRError (sprintf "Set target '%s' is not mutable" name) }
                    | None ->
                        { valueOutput with Result = TRError (sprintf "Set target '%s' has no definition" name) }
                | _ ->
                    WitnessOutput.error (sprintf "Set target is not a VarRef: %A" targetNode.Kind)
            | TRError msg ->
                { valueOutput with Result = TRError msg }
            | _ ->
                { valueOutput with Result = TRVoid }

    // ─────────────────────────────────────────────────────────────────────
    // APPLICATION
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Application (funcId, argIds) ->
        // Visit function and arguments
        let funcOutput = visitNode ctx z funcId
        let argOutputs = argIds |> List.map (visitNode ctx z)
        let combinedArgs = WitnessOutput.combineAll argOutputs

        // Witness the application
        let appOps, appResult = Alex.Witnesses.CallDispatch.witness ctx node

        { InlineOps = funcOutput.InlineOps @ combinedArgs.InlineOps @ appOps
          TopLevelOps = funcOutput.TopLevelOps @ combinedArgs.TopLevelOps
          Result = appResult }

    // ─────────────────────────────────────────────────────────────────────
    // LAMBDA - Pre-bind params, visit body, witness
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Lambda (params', bodyId, _captures, _enclosing, _context) ->
        // Pre-bind parameters (pushes scope, returns MLIR params)
        let funcParams = Alex.Witnesses.LambdaWitness.preBindParams ctx node

        // For entry point lambdas, emit ModuleInit bindings first (MainPrologue strategy)
        // Per fsnative-spec/program-structure-and-execution.md: "static initializers...
        // are executed in compilation order before the entry point function is called"
        let moduleInitOutput =
            if ctx.Coeffects.EntryPointLambdaIds.Contains (NodeId.value node.Id) then
                // This Lambda is an entry point. Find its parent Binding.
                match node.Parent with
                | Some parentBindingId ->
                    // Find the module whose EntryPoint matches this Binding
                    let moduleOpt =
                        ctx.Graph.ModuleClassifications.Value
                        |> Map.tryPick (fun _moduleId mc ->
                            match mc.EntryPoint with
                            | Some epId when epId = parentBindingId -> Some mc
                            | _ -> None)

                    match moduleOpt with
                    | Some mc ->
                        // Only emit THIS module's ModuleInit bindings
                        let outputs = mc.ModuleInit |> List.map (visitNode ctx z)
                        WitnessOutput.combineAll outputs
                    | None ->
                        WitnessOutput.empty
                | None ->
                    WitnessOutput.empty
            else
                WitnessOutput.empty

        // Visit body
        let bodyOutput = visitNode ctx z bodyId

        // Create the body callback for witness (pull model)
        // Include moduleInit ops BEFORE body ops (static initialization order)
        let combinedBodyOps = moduleInitOutput.InlineOps @ bodyOutput.InlineOps
        let witnessBody = fun (_ctx: WitnessContext) -> (combinedBodyOps, bodyOutput.Result)

        // Witness the lambda - returns (funcDefOpt, closureOps, result)
        let funcDefOpt, closureOps, result =
            Alex.Witnesses.LambdaWitness.witness params' bodyId node funcParams witnessBody ctx

        // Function definition goes to TopLevelOps (returned, not mutated)
        let topLevelOps =
            match funcDefOpt with
            | Some funcDef -> funcDef :: (moduleInitOutput.TopLevelOps @ bodyOutput.TopLevelOps)
            | None -> moduleInitOutput.TopLevelOps @ bodyOutput.TopLevelOps

        { InlineOps = closureOps
          TopLevelOps = topLevelOps
          Result = result }

    // ─────────────────────────────────────────────────────────────────────
    // IF-THEN-ELSE
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.IfThenElse (condId, thenId, elseIdOpt) ->
        let condOutput = visitNode ctx z condId
        let condSSA = match condOutput.Result with TRValue v -> v.SSA | _ -> V 0

        let thenOutput = visitNode ctx z thenId
        let thenSSA = match thenOutput.Result with TRValue v -> Some v.SSA | _ -> None

        let elseOutput, elseSSA =
            match elseIdOpt with
            | Some elseId ->
                let o = visitNode ctx z elseId
                let ssa = match o.Result with TRValue v -> Some v.SSA | _ -> None
                Some o, ssa
            | None -> None, None

        let resultType = match thenOutput.Result with TRValue v -> Some v.Type | _ -> None

        let elseOpsOpt = elseOutput |> Option.map (fun o -> o.InlineOps)
        let ifOps, ifResult =
            Alex.Witnesses.ControlFlowWitness.witnessIfThenElse node.Id ctx condSSA thenOutput.InlineOps thenSSA elseOpsOpt elseSSA resultType

        let allTopLevel =
            condOutput.TopLevelOps @ thenOutput.TopLevelOps @
            (elseOutput |> Option.map (fun o -> o.TopLevelOps) |> Option.defaultValue [])

        { InlineOps = condOutput.InlineOps @ ifOps
          TopLevelOps = allTopLevel
          Result = ifResult }

    // ─────────────────────────────────────────────────────────────────────
    // WHILE LOOP
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.WhileLoop (condId, bodyId) ->
        let condOutput = visitNode ctx z condId
        let condSSA = match condOutput.Result with TRValue v -> v.SSA | _ -> V 0
        let bodyOutput = visitNode ctx z bodyId

        let whileOps, whileResult =
            Alex.Witnesses.ControlFlowWitness.witnessWhileLoop node.Id ctx condOutput.InlineOps condSSA bodyOutput.InlineOps []

        { InlineOps = whileOps
          TopLevelOps = condOutput.TopLevelOps @ bodyOutput.TopLevelOps
          Result = whileResult }

    // ─────────────────────────────────────────────────────────────────────
    // TYPE ANNOTATION - Transparent, pass through
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.TypeAnnotation (innerExpr, _) ->
        visitNode ctx z innerExpr

    // ─────────────────────────────────────────────────────────────────────
    // INTRINSIC / PLATFORM - Handled via Application when called
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Intrinsic _
    | SemanticKind.PlatformBinding _ ->
        WitnessOutput.empty

    // ─────────────────────────────────────────────────────────────────────
    // PATTERN (Parameter) - handled by Lambda preBindParams
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.PatternBinding _ ->
        WitnessOutput.empty

    // ─────────────────────────────────────────────────────────────────────
    // FIELD GET - struct.field or record.field access
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.FieldGet (structId, fieldName) ->
        let structOutput = visitNode ctx z structId
        match structOutput.Result with
        | TRValue v ->
            match SemanticGraph.tryGetNode structId ctx.Graph with
            | Some structNode ->
                let structNativeType = structNode.Type
                let fieldMlirType = mapNativeType node.Type

                // Delegate field resolution to witness
                let fieldOps, fieldResult =
                    Alex.Witnesses.MemoryWitness.witnessFieldGet node.Id ctx v.SSA structNativeType fieldName fieldMlirType

                // Record the result for later recall
                match fieldResult with
                | TRValue fv -> MLIRAccumulator.bindNode (NodeId.value node.Id) fv.SSA fv.Type acc
                | _ -> ()

                { InlineOps = structOutput.InlineOps @ fieldOps
                  TopLevelOps = structOutput.TopLevelOps
                  Result = fieldResult }
            | None ->
                WitnessOutput.error (sprintf "FieldGet: struct node %d not found" (NodeId.value structId))
        | TRError msg ->
            { structOutput with Result = TRError msg }
        | _ ->
            WitnessOutput.error (sprintf "FieldGet: struct expression has no value")

    // ─────────────────────────────────────────────────────────────────────
    // EXPLICIT NOT-YET-IMPLEMENTED CASES
    // Each semantic kind must be explicitly listed for proper diagnostics
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Match _ ->
        // Match expressions should be elaborated away by Baker into IfThenElse chains
        WitnessOutput.error (sprintf "Match expression should have been elaborated by Baker (node %d)" (NodeId.value node.Id))
    | SemanticKind.ForLoop _ ->
        WitnessOutput.error (sprintf "ForLoop not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.ForEach _ ->
        WitnessOutput.error (sprintf "ForEach not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TryWith _ ->
        WitnessOutput.error (sprintf "TryWith not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TryFinally _ ->
        WitnessOutput.error (sprintf "TryFinally not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.RecordExpr _ ->
        WitnessOutput.error (sprintf "RecordExpr not yet implemented (node %d)" (NodeId.value node.Id))

    // ─────────────────────────────────────────────────────────────────────
    // DISCRIMINATED UNION OPERATIONS (January 2026)
    // Pointer-based DUs with case eliminators for type-safe payload extraction
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.DUGetTag (duValueId, duType) ->
        // Extract tag from DU pointer
        let duOutput = visitNode ctx z duValueId
        match duOutput.Result with
        | TRValue duVal ->
            let tagOps, tagResult =
                Alex.Witnesses.MemoryWitness.witnessDUGetTag node.Id ctx duVal.SSA duType
            { InlineOps = duOutput.InlineOps @ tagOps
              TopLevelOps = duOutput.TopLevelOps
              Result = tagResult }
        | TRError msg -> { duOutput with Result = TRError msg }
        | _ -> WitnessOutput.error "DUGetTag: DU expression has no value"

    | SemanticKind.DUEliminate (duValueId, caseIndex, caseName, payloadType) ->
        // Type-safe payload extraction via case eliminator
        let duOutput = visitNode ctx z duValueId
        match duOutput.Result with
        | TRValue duVal ->
            let payloadMlirType = mapNativeType payloadType
            // Pass the ACTUAL DU type for extraction, plus the desired payload type
            // The DU type determines the extractvalue struct annotation;
            // if payload types differ, a bitcast is needed
            let elimOps, elimResult =
                Alex.Witnesses.MemoryWitness.witnessDUEliminate node.Id ctx duVal.SSA duVal.Type caseIndex caseName payloadMlirType
            match elimResult with
            | TRValue ev -> MLIRAccumulator.bindNode (NodeId.value node.Id) ev.SSA ev.Type acc
            | _ -> ()
            { InlineOps = duOutput.InlineOps @ elimOps
              TopLevelOps = duOutput.TopLevelOps
              Result = elimResult }
        | TRError msg -> { duOutput with Result = TRError msg }
        | _ -> WitnessOutput.error "DUEliminate: DU expression has no value"

    | SemanticKind.DUConstruct (caseName, caseIndex, payloadOpt, _arenaHintOpt) ->
        // Construct DU value - for now, use inline struct approach
        // Full arena allocation will be added when arena infrastructure is complete
        let payloadOutput =
            match payloadOpt with
            | Some payloadId -> visitNode ctx z payloadId
            | None -> WitnessOutput.empty
        match payloadOutput.Result with
        | TRValue pv ->
            let constructOps, constructResult =
                Alex.Witnesses.MemoryWitness.witnessDUConstruct node.Id ctx caseName caseIndex (Some pv) (mapNativeType node.Type)
            { InlineOps = payloadOutput.InlineOps @ constructOps
              TopLevelOps = payloadOutput.TopLevelOps
              Result = constructResult }
        | TRVoid ->
            // Nullary case (no payload)
            let constructOps, constructResult =
                Alex.Witnesses.MemoryWitness.witnessDUConstruct node.Id ctx caseName caseIndex None (mapNativeType node.Type)
            { InlineOps = constructOps
              TopLevelOps = []
              Result = constructResult }
        | TRError msg -> { payloadOutput with Result = TRError msg }
        | TRBuiltin _ -> WitnessOutput.error "DUConstruct: payload expression is a builtin (unexpected)"

    | SemanticKind.UnionCase _ ->
        WitnessOutput.error (sprintf "UnionCase not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TupleExpr _ ->
        WitnessOutput.error (sprintf "TupleExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.ArrayExpr _ ->
        WitnessOutput.error (sprintf "ArrayExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.ListExpr _ ->
        WitnessOutput.error (sprintf "ListExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.FieldSet _ ->
        WitnessOutput.error (sprintf "FieldSet not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.IndexGet _ ->
        WitnessOutput.error (sprintf "IndexGet not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.IndexSet _ ->
        WitnessOutput.error (sprintf "IndexSet not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.NamedIndexedPropertySet _ ->
        WitnessOutput.error (sprintf "NamedIndexedPropertySet not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.Upcast _ ->
        WitnessOutput.error (sprintf "Upcast not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.Downcast _ ->
        WitnessOutput.error (sprintf "Downcast not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TypeTest _ ->
        WitnessOutput.error (sprintf "TypeTest not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.AddressOf (targetId, _isByref) ->
        // AddressOf takes the address of a variable/expression
        // For mutable locals, this returns the alloca pointer
        // For immutable locals, this might require creating a temp alloca
        match SemanticGraph.tryGetNode targetId ctx.Graph with
        | Some targetNode ->
            match targetNode.Kind with
            | SemanticKind.VarRef (name, defIdOpt) ->
                // Check if this is a mutable variable with an alloca
                match MLIRAccumulator.recallVar name acc with
                | Some (ssa, ty) ->
                    // The SSA might be the value or the alloca pointer
                    // For mutable variables, the accumulator stores the alloca pointer
                    // Return it directly as the address
                    WitnessOutput.value { SSA = ssa; Type = MLIRTypes.ptr }
                | None ->
                    // Try looking up by defId
                    match defIdOpt with
                    | Some defId ->
                        match MLIRAccumulator.recallNode (NodeId.value defId) acc with
                        | Some (ssa, _ty) ->
                            // Return the binding's SSA as the address
                            WitnessOutput.value { SSA = ssa; Type = MLIRTypes.ptr }
                        | None ->
                            WitnessOutput.error (sprintf "AddressOf: variable '%s' not found in scope (node %d)" name (NodeId.value node.Id))
                    | None ->
                        WitnessOutput.error (sprintf "AddressOf: variable '%s' has no definition (node %d)" name (NodeId.value node.Id))
            | _ ->
                // Taking address of a non-VarRef expression
                // This might require visiting the child first to compute the value
                let childOutput = visitChildren ctx z targetNode.Children
                match childOutput.Result with
                | TRValue v ->
                    // For a computed value, we'd need to alloca and store
                    // For now, assume it's already a pointer
                    WitnessOutput.value { SSA = v.SSA; Type = MLIRTypes.ptr }
                | TRVoid ->
                    WitnessOutput.error (sprintf "AddressOf: target expression has no value (node %d)" (NodeId.value node.Id))
                | TRError msg ->
                    WitnessOutput.error (sprintf "AddressOf: target error: %s (node %d)" msg (NodeId.value node.Id))
        | None ->
            WitnessOutput.error (sprintf "AddressOf: target node %d not found (node %d)" (NodeId.value targetId) (NodeId.value node.Id))
    | SemanticKind.Deref _ ->
        WitnessOutput.error (sprintf "Deref not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TraitCall _ ->
        WitnessOutput.error (sprintf "TraitCall not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.Quote _ ->
        WitnessOutput.error (sprintf "Quote not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.ObjectExpr _ ->
        WitnessOutput.error (sprintf "ObjectExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TypeDef _ ->
        WitnessOutput.error (sprintf "TypeDef not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.MemberDef _ ->
        WitnessOutput.error (sprintf "MemberDef not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.InterpolatedString _ ->
        WitnessOutput.error (sprintf "InterpolatedString not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.LazyExpr _ ->
        WitnessOutput.error (sprintf "LazyExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.LazyForce _ ->
        WitnessOutput.error (sprintf "LazyForce not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.SeqExpr _ ->
        WitnessOutput.error (sprintf "SeqExpr not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.Yield _ ->
        WitnessOutput.error (sprintf "Yield not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.YieldBang _ ->
        WitnessOutput.error (sprintf "YieldBang not yet implemented (node %d)" (NodeId.value node.Id))
    | SemanticKind.TupleGet (tupleId, index) ->
        // Extract element from tuple using llvm.extractvalue
        let tupleOutput = visitNode ctx z tupleId
        match tupleOutput.Result with
        | TRValue tupleVal ->
            let elemSSA = requireSSAFromCoeffs node.Id ctx.Coeffects
            let elemType = mapNativeType node.Type
            let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (elemSSA, tupleVal.SSA, [index], tupleVal.Type))
            MLIRAccumulator.bindNode (NodeId.value node.Id) elemSSA elemType acc
            { InlineOps = tupleOutput.InlineOps @ [extractOp]
              TopLevelOps = tupleOutput.TopLevelOps
              Result = TRValue { SSA = elemSSA; Type = elemType } }
        | TRError msg -> { tupleOutput with Result = TRError msg }
        | _ -> failwithf "TupleGet: tuple expression has no value (node %d)" (NodeId.value node.Id)
    | SemanticKind.Error msg ->
        WitnessOutput.error (sprintf "PSG Error node: %s (node %d)" msg (NodeId.value node.Id))

/// Record a result in the accumulator (for DAG handling)
and private recordResult (nodeIdVal: int) (result: TransferResult) (acc: MLIRAccumulator) : unit =
    match result with
    | TRValue v -> MLIRAccumulator.bindNode nodeIdVal v.SSA v.Type acc
    | _ -> ()

/// Visit children and combine their outputs
and private visitChildren
    (ctx: WitnessContext)
    (z: PSGZipper)
    (childIds: NodeId list)
    : WitnessOutput =
    childIds
    |> List.map (visitNode ctx z)
    |> WitnessOutput.combineAll

// ═══════════════════════════════════════════════════════════════════════════
// MLIR GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MLIR module wrapper with string table and operations
let private wrapInModule (ops: MLIROp list) (stringTable: StringCollect.StringTable) : string =
    let sb = System.Text.StringBuilder()

    // Module header
    sb.AppendLine("module {") |> ignore

    // String constants from StringTable coeffect
    for KeyValue(_hash, entry) in stringTable do
        let escaped = entry.Content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n")
        sb.AppendLine(sprintf "  llvm.mlir.global private constant %s(\"%s\\00\") : !llvm.array<%d x i8>"
            entry.GlobalName escaped (entry.ByteLength + 1)) |> ignore

    // Closure heap globals for escaping closures
    // @closure_pos: current allocation position (bump pointer)
    // @closure_heap: pre-allocated buffer for closure environments
    // TODO: Only emit when closures are actually used
    let closureHeapSize = 65536  // 64KB - sufficient for most programs
    sb.AppendLine(sprintf "  llvm.mlir.global internal @closure_pos(0 : i64) : i64") |> ignore
    // MLIR requires initializer region with llvm.mlir.zero for zero-initialized arrays
    sb.AppendLine(sprintf "  llvm.mlir.global internal @closure_heap() : !llvm.array<%d x i8> {" closureHeapSize) |> ignore
    sb.AppendLine(sprintf "    %%0 = llvm.mlir.zero : !llvm.array<%d x i8>" closureHeapSize) |> ignore
    sb.AppendLine(sprintf "    llvm.return %%0 : !llvm.array<%d x i8>" closureHeapSize) |> ignore
    sb.AppendLine("  }") |> ignore

    // Operations (reversed since we accumulated in reverse order)
    for op in List.rev ops do
        sb.AppendLine(sprintf "  %s" (Alex.Dialects.Core.Serialize.opToString op)) |> ignore

    // Module footer
    sb.AppendLine("}") |> ignore

    sb.ToString()

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer graph to MLIR with diagnostics
/// This is the main entry point called by CompilationOrchestrator
let transferGraphWithDiagnostics
    (graph: SemanticGraph)
    (isFreestanding: bool)
    (intermediatesDir: string option)
    : string * string list =

    // Compute all coeffects ONCE before traversal
    let coeffects = computeCoeffects graph isFreestanding

    // Serialize ALL coeffects if requested (debugging)
    // "Pierce the Veil" - complete visibility into nanopass infrastructure
    match intermediatesDir with
    | Some dir ->
        PSGElaboration.PreprocessingSerializer.serializeAll
            dir
            coeffects.SSA
            coeffects.Mutability
            coeffects.YieldStates
            coeffects.PatternBindings
            coeffects.Strings
            coeffects.EntryPointLambdaIds
            graph
    | None -> ()

    // Create accumulator
    let acc = MLIRAccumulator.empty()

    // Create witness context
    let ctx: WitnessContext = {
        Coeffects = coeffects
        Accumulator = acc
        Graph = graph
    }

    // Collect all reachable function definitions from all modules
    // This ensures library functions (like Console.write) get emitted, not just entry points
    let allFunctionBindings =
        graph.ModuleClassifications.Value
        |> Map.toList
        |> List.collect (fun (_moduleId, classification) ->
            classification.Definitions
            |> List.filter (fun defId ->
                // Only process reachable bindings that are functions (have Lambda child)
                match SemanticGraph.tryGetNode defId graph with
                | Some node when node.IsReachable ->
                    match node.Kind with
                    | SemanticKind.Binding _ ->
                        // Check if this binding has a Lambda child (is a function)
                        node.Children
                        |> List.exists (fun childId ->
                            match SemanticGraph.tryGetNode childId graph with
                            | Some child ->
                                match child.Kind with
                                | SemanticKind.Lambda _ -> true
                                | _ -> false
                            | None -> false)
                    | _ -> false
                | _ -> false))
        |> List.distinct

    // Visit all function bindings (this emits their func.func definitions)
    for bindingId in allFunctionBindings do
        match PSGZipper.create graph bindingId with
        | Some z ->
            let _output = visitNode ctx z bindingId
            ()
        | None ->
            MLIRAccumulator.addError (sprintf "Could not create zipper at function binding %A" bindingId) acc

    // Visit entry points - these are module-level constructs that call the functions above
    for entryId in graph.EntryPoints do
        match PSGZipper.create graph entryId with
        | Some z ->
            // The zipper provides positional attention
            // visitNode returns WitnessOutput with all codata
            // TopLevelOps are accumulated by the fold (in visitNode)
            let _output = visitNode ctx z entryId
            // Note: TopLevelOps already accumulated by visitNode
            // InlineOps from entry point are typically empty (module-level)
            ()
        | None ->
            MLIRAccumulator.addError (sprintf "Could not create zipper at entry point %A" entryId) acc

    // Generate MLIR from accumulated top-level ops
    let mlir = wrapInModule acc.TopLevelOps coeffects.Strings

    (mlir, List.rev acc.Errors)

/// Transfer graph to MLIR (no diagnostics)
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    fst (transferGraphWithDiagnostics graph isFreestanding None)
