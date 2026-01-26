let witness
    (params': (string * NativeType * NodeId) list)
    (bodyId: NodeId)
    (node: SemanticNode)
    (funcParams: (SSA * MLIRType) list)
    (witnessBody: WitnessContext -> MLIROp list * TransferResult)
    (ctx: WitnessContext)
    : MLIROp option * MLIROp list * TransferResult =

    // Get lambda name directly from SSAAssignment coeffects (NOT from mutable Focus state)
    // Focus can be corrupted by nested lambda processing; coeffects are immutable truth
    let lambdaName =
        match lookupLambdaName node.Id ctx.Coeffects.SSA with
        | Some name -> name
        | None -> sprintf "lambda_%d" (NodeId.value node.Id)

    // Look up ClosureLayout coeffect - determines if this is a closing lambda
    // For closing lambdas, SSAAssignment has pre-computed all layout information
    let closureLayoutOpt = lookupClosureLayout node.Id ctx.Coeffects.SSA

    witnessInFunctionScope lambdaName node bodyId witnessBody funcParams closureLayoutOpt ctx

/// Check if a type is unit
let private isUnitType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp(tc, _) ->
        match tc.NTUKind with
        | Some NTUKind.NTUunit -> true
        | _ -> tc.Name = "unit"
    | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// Pre-order Lambda Parameter Binding
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-bind Lambda parameters to SSA names BEFORE body is processed
/// This enters function scope so body operations are captured
/// For entry point (main), uses C-style signature: (argc: i32, argv: ptr) -> i32
///
/// CLOSURE HANDLING:
/// For closing lambdas (with captures):
/// - Arg 0 is the env_ptr (all other args shift by 1)
/// - Captured variables are bound by emitting GEP+Load from env struct at function entry
/// - These entry ops are prepended to body ops later
let preBindParams (ctx: WitnessContext) (node: SemanticNode) : (SSA * MLIRType) list =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId, captures, _, _) ->
        let nodeIdVal = NodeId.value node.Id
        // ARCHITECTURAL FIX: Use SSAAssignment coeffect for lambda names
        // This respects binding names when Lambda is bound via `let name = ...`
        let lambdaName =
            match PSGElaboration.SSAAssignment.lookupLambdaName node.Id ctx.Coeffects.SSA with
            | Some name -> name
            | None -> sprintf "lambda_%d" nodeIdVal  // fallback for anonymous

        // ARCHITECTURAL PRINCIPLE (January 2026):
        // Alex emits what the PSG says. No platform-specific signature overrides.
        // If main needs C-style signature (console mode), FNCS handles that transformation.
        // If main needs F#-style signature (freestanding mode with _start), PSG already has it.

        let mlirParams, paramBindings, needsArgvConversion, isUnitParam =
            // Use node.Type for parameter types - trust the PSG
            let rec extractParamTypes ty count =
                if count <= 0 then []
                else
                    match ty with
                    | NativeType.TFun(paramTy, resultTy) ->
                        paramTy :: extractParamTypes resultTy (count - 1)
                    | _ -> []

            // NTU PRINCIPLE: Unit has size 0, so unit parameters are elided
            // Check if the function domain type is unit - if so, nullary at native level
            // Conservative: only elide when we have positive proof of NTUunit
            let isUnitParam =
                match node.Type with
                | NativeType.TFun(NativeType.TApp(tc, _), _) ->
                    tc.NTUKind = Some NTUKind.NTUunit || tc.Name = "unit"
                | NativeType.TFun _ ->
                    false  // TVar, TFun, TTuple domains - not unit, don't elide
                | _nonFunType ->
                    // Lambda should have TFun type. If not, conservative fallback:
                    // don't elide parameters (safe behavior, just potentially suboptimal)
                    false

            // CLOSURE HANDLING:
            // 1. Escaping closures (with ClosureLayout): Arg 0 is env_ptr, params shift by 1
            // 2. Nested functions with captures (no ClosureLayout): captures become explicit params
            // 3. Simple lambdas (no captures): no offset
            let closureLayoutOpt = lookupClosureLayout node.Id ctx.Coeffects.SSA
            let hasEscapingClosure = Option.isSome closureLayoutOpt
            let hasNestedCaptures = not (List.isEmpty captures) && Option.isNone closureLayoutOpt

            // Use graph-aware type mapping for record types
            let mapTypeWithGraph = mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph

            // Build capture params for nested functions
            let captureParams, captureBindings =
                if hasNestedCaptures then
                    let capPs = captures |> List.mapi (fun i cap ->
                        (Arg i, mapTypeWithGraph cap.Type))
                    let capBs = captures |> List.mapi (fun i cap ->
                        (cap.Name, Some (Arg i, mapTypeWithGraph cap.Type)))
                    capPs, capBs
                else
                    [], []

            let captureCount = List.length captureParams
            let argOffset =
                if hasEscapingClosure then 1  // env_ptr at Arg 0
                elif hasNestedCaptures then captureCount  // captures at Arg 0..N-1
                else 0

            let mlirPs, bindings =
                if isUnitParam then
                    // NTUunit has size 0 - no parameter generated
                    // A function taking unit is nullary at the native level
                    // This aligns with call sites which also elide unit arguments
                    captureParams, captureBindings  // Still include capture params if nested
                else
                    let nodeParamTypes = extractParamTypes node.Type (List.length params')

                    // Build structured params: (Arg i+offset, MLIRType)
                    // For closing lambdas, params start at Arg 1 (Arg 0 is env_ptr)
                    // For nested functions with captures, params start at Arg N (captures at Arg 0..N-1)
                    let ps = params' |> List.mapi (fun i (_name, nativeTy, _nodeId) ->
                        let actualType =
                            if i < List.length nodeParamTypes then nodeParamTypes.[i]
                            else nativeTy
                        (Arg (i + argOffset), mapTypeWithGraph actualType))

                    // Build bindings: (paramName, Some (Arg i+offset, MLIRType))
                    let bs = params' |> List.mapi (fun i (paramName, paramType, _nodeId) ->
                        let actualType =
                            if i < List.length nodeParamTypes then nodeParamTypes.[i]
                            else paramType
                        (paramName, Some (Arg (i + argOffset), mapTypeWithGraph actualType)))

                    captureParams @ ps, captureBindings @ bs

            mlirPs, bindings, false, isUnitParam

        // Determine captured variables and mutables for scope creation
        let closureLayoutOpt = lookupClosureLayout node.Id ctx.Coeffects.SSA
        let capturedVarNames, capturedMutNames =
            match closureLayoutOpt with
            | Some layout ->
                let varNames = layout.Captures |> List.map (fun slot -> slot.Name) |> Set.ofList
                let mutNames = layout.Captures
                               |> List.filter (fun slot -> slot.Mode = ByRef)
                               |> List.map (fun slot -> slot.Name)
                               |> Set.ofList
                varNames, mutNames
            | None ->
                Set.empty, Set.empty

        // Create and push a new scope for this function with captured var/mut info
        let functionScope = {
            VarAssoc = Map.empty
            NodeAssoc = Map.empty
            CapturedVars = capturedVarNames
            CapturedMuts = capturedMutNames
        }
        MLIRAccumulator.pushScope functionScope ctx.Accumulator

        // Bind parameters in the accumulator
        if needsArgvConversion && not (List.isEmpty paramBindings) then
            let paramName = fst (List.head paramBindings)
            bindArgvParameters paramName ctx
        else
            paramBindings
            |> List.iter (fun (paramName, bindingOpt) ->
                match bindingOpt with
                | Some (argSSA, mlirType) ->
                    let paramNodeIdOpt =
                        params'
                        |> List.tryFind (fun (n, _, _) -> n = paramName)
                        |> Option.map (fun (_, _, id) -> id)
                    MLIRAccumulator.bindVar paramName argSSA mlirType ctx.Accumulator
                    match paramNodeIdOpt with
                    | Some id -> MLIRAccumulator.bindNode (NodeId.value id) argSSA mlirType ctx.Accumulator
                    | None -> ()
                | None -> ()
            )

        // For closing lambdas: bind captured variables
        // TRUE FLAT CLOSURE: Captures are extracted from closure struct (Arg 0) at indices 1, 2, ...
        // We bind SSAs here; actual extractvalue ops are emitted by witness
        match closureLayoutOpt with
        | Some layout ->
            // Bind each captured variable name to its extraction SSA
            // Extraction SSAs are derived from SlotIndex: v0, v1, ..., v(N-1)
            // CapturedVars/CapturedMuts are already set in the scope above
            layout.Captures
            |> List.iter (fun slot ->
                let extractSSA = V slot.SlotIndex  // Derived from PSG structure
                MLIRAccumulator.bindVar slot.Name extractSSA slot.SlotType ctx.Accumulator
            )
        | None -> ()

        // Return the computed MLIR params
        mlirParams
    | _ -> []
