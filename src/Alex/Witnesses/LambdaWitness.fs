/// Lambda Witness - Witness Lambda nodes to MLIR functions
///
/// ARCHITECTURAL PRINCIPLES (Four Pillars):
/// 1. Coeffects: SSA assignment is pre-computed, lookup via context
/// 2. Active Patterns: Match on semantic meaning, not strings
/// 3. Zipper: Navigate and accumulate structured ops
/// 4. Templates: Return structured MLIROp types, no sprintf
///
/// Functions are built as structured data:
/// - Accumulate body ops during child traversal
/// - Create FuncDef with body region
/// - Emit to top-level
///
/// WITNESS PATTERN: Returns (MLIROp list * TransferResult)
/// Witnesses OBSERVE and RETURN. They do NOT emit directly.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let private mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Argv Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Bind entry point parameters for C-style main
/// At OS entry point: %arg0: i32 = argc, %arg1: !llvm.ptr = argv
/// For F# string[] parameter, pattern matching handles conversion at use site
let private bindArgvParameters (paramName: string) (z: PSGZipper) : PSGZipper =
    // Bind C-style argc/argv under well-known names using STRUCTURED types
    let z1 = bindVarSSA "__argc" (Arg 0) MLIRTypes.i32 z
    let z2 = bindVarSSA "__argv" (Arg 1) MLIRTypes.ptr z1
    // Bind F# parameter name to argv pointer
    bindVarSSA paramName (Arg 1) MLIRTypes.ptr z2

// ═══════════════════════════════════════════════════════════════════════════
// Function Body Building
// ═══════════════════════════════════════════════════════════════════════════

/// Create a default return value when body doesn't produce one
/// NOTE: This should rarely happen - well-formed functions produce results
/// For unit functions, body produces i32 0. For divergent functions, use unreachable.
let private createDefaultReturn (nodeId: NodeId) (z: PSGZipper) (declaredRetType: MLIRType) : MLIROp list * SSA =
    // Use Lambda node's pre-assigned SSA for return value synthesis
    let resultSSA = requireNodeSSA nodeId z
    match declaredRetType with
    | TPtr ->
        // Null pointer for pointer returns
        let op = MLIROp.LLVMOp (LLVMOp.NullPtr resultSSA)
        [op], resultSSA
    | _ ->
        // Zero constant for integer returns
        let op = MLIROp.ArithOp (ArithOp.ConstI (resultSSA, 0L, declaredRetType))
        [op], resultSSA

/// Create return instruction with type
let private createReturnOp (valueSSA: SSA option) (valueTy: MLIRType option) : MLIROp =
    MLIROp.LLVMOp (LLVMOp.Return (valueSSA, valueTy))

/// Create unreachable terminator (for panic paths)
let private createUnreachable () : MLIROp =
    MLIROp.LLVMOp LLVMOp.Unreachable

/// Handle return type mismatch between declared and computed types
/// Uses Lambda node's pre-assigned SSAs for type reconciliation ops
let private reconcileReturnType
    (nodeId: NodeId)
    (z: PSGZipper)
    (bodySSA: SSA)
    (bodyType: MLIRType)
    (declaredRetType: MLIRType)
    : MLIROp list * SSA =
    match declaredRetType, bodyType with
    | TInt I32, TPtr ->
        // Unit function with pointer body - return 0
        // Use Lambda node's SSA for the synthesized constant
        let resultSSA = requireNodeSSA nodeId z
        let op = MLIROp.ArithOp (ArithOp.ConstI (resultSSA, 0L, TInt I32))
        [op], resultSSA
    | TPtr, TInt _ ->
        // Pointer return but integer body - return null
        let resultSSA = requireNodeSSA nodeId z
        let op = MLIROp.LLVMOp (LLVMOp.NullPtr resultSSA)
        [op], resultSSA
    | _, _ ->
        // Types match or compatible - use body result
        [], bodySSA

/// Extract final return type by peeling TFun layers
let private extractFinalReturnType (ty: NativeType) (paramCount: int) : NativeType =
    let rec peel ty count =
        match ty with
        | NativeType.TFun(_, resultTy) when count > 0 ->
            peel resultTy (count - 1)
        | NativeType.TFun(_, resultTy) when paramCount = 0 ->
            // Unit lambda: peel the unit->result layer
            resultTy
        | _ -> ty
    peel ty paramCount

// ═══════════════════════════════════════════════════════════════════════════
// Lambda Witnessing
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Lambda node when in function scope
/// Body operations were accumulated during child traversal
/// Returns: (ops to emit, result) - following witness pattern
let private witnessInFunctionScope
    (lambdaName: string)
    (node: SemanticNode)
    (bodyId: NodeId)
    (bodyOps: MLIROp list)
    (funcParams: (SSA * MLIRType) list)
    (z: PSGZipper)
    : MLIROp option * MLIROp list * TransferResult =

    // Determine declared return type (use graph-aware mapping for record types)
    let paramCount =
        match node.Kind with
        | SemanticKind.Lambda (params', _bodyId) -> List.length params'
        | _ -> 0
    let finalRetType = extractFinalReturnType node.Type paramCount
    let declaredRetType = mapNativeTypeWithGraph z.Graph finalRetType

    // Look up body's SSA result (already processed in post-order)
    // Using structured recallNodeResult which returns (SSA * MLIRType) option
    let bodyResult =
        recallNodeResult (NodeId.value bodyId) z

    // Build return/terminator ops
    let returnOps, terminator =
        match bodyResult with
        | Some (ssa, bodyType) ->
            // Normal return - may need type reconciliation
            let reconcileOps, finalSSA = reconcileReturnType node.Id z ssa bodyType declaredRetType
            reconcileOps, createReturnOp (Some finalSSA) (Some declaredRetType)
        | None ->
            // No body result - create default (uses Lambda node's pre-assigned SSA)
            let defaultOps, defaultSSA = createDefaultReturn node.Id z declaredRetType
            defaultOps, createReturnOp (Some defaultSSA) (Some declaredRetType)

    // Build complete body ops: accumulated + return + terminator
    let completeBodyOps = bodyOps @ returnOps @ [terminator]

    // Determine visibility
    let visibility =
        if isFuncInternal lambdaName z then FuncVisibility.Private
        else FuncVisibility.Public

    // Build function body as single entry block
    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []
        Ops = completeBodyOps
    }
    let bodyRegion: Region = { Blocks = [entryBlock] }

    // Create function definition
    let funcDef = FuncOp.FuncDef (lambdaName, funcParams, declaredRetType, bodyRegion, visibility)

    // ARCHITECTURAL NOTE: For bound lambdas (let name = lambda...),
    // the call site uses `func.call @name` directly - no addressof needed.
    // First-class function values would need `func.constant` (not llvm.mlir.addressof).
    // For now, return just the function definition with no local ops.

    // Return: (funcDef for TopLevel, no localOps, void result)
    // Photographer Principle: OBSERVE and RETURN, do not EMIT
    Some (MLIROp.FuncOp funcDef), [], TRVoid

/// Witness a Lambda node - main entry point
/// Returns: (funcDef option * localOps * TransferResult)
/// - funcDef: The function definition to add to TopLevel
/// - localOps: Local operations (addressof) to add to CurrentOps
/// - result: The TransferResult for this node
/// 
/// PHOTOGRAPHER PRINCIPLE: This witness OBSERVES and RETURNS.
/// It does NOT emit directly. The caller (FNCSTransfer) handles accumulation.
let witness
    (params': (string * NativeType * NodeId) list)
    (bodyId: NodeId)
    (node: SemanticNode)
    (bodyOps: MLIROp list)
    (z: PSGZipper)
    : MLIROp option * MLIROp list * TransferResult =

    // Get function parameters from state (set by preBindParams)
    let funcParams =
        match z.State.CurrentFuncParams with
        | Some ps -> ps
        | None -> []

    // Get lambda name directly from SSAAssignment coeffects (NOT from mutable Focus state)
    // Focus can be corrupted by nested lambda processing; coeffects are immutable truth
    let lambdaName =
        match Alex.Preprocessing.SSAAssignment.lookupLambdaName node.Id z.State.SSAAssignment with
        | Some name -> name
        | None -> sprintf "lambda_%d" (NodeId.value node.Id)
    
    witnessInFunctionScope lambdaName node bodyId bodyOps funcParams z

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
let preBindParams (z: PSGZipper) (node: SemanticNode) : PSGZipper =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId) ->
        let nodeIdVal = NodeId.value node.Id
        // ARCHITECTURAL FIX: Use SSAAssignment coeffect for lambda names
        // This respects binding names when Lambda is bound via `let name = ...`
        let lambdaName =
            match Alex.Preprocessing.SSAAssignment.lookupLambdaName node.Id z.State.SSAAssignment with
            | Some name -> name
            | None -> yieldLambdaNameForNode nodeIdVal z  // fallback for anonymous

        // For main, use C-style signature
        let isMain = (lambdaName = "main")

        let mlirParams, paramBindings, needsArgvConversion, isUnitParam =
            if isMain then
                // C-style main: (int argc, char** argv) -> int
                // Parameters use structured SSA: Arg 0, Arg 1
                let cParams = [(Arg 0, TInt I32); (Arg 1, TPtr)]
                let bindings, needsConv =
                    match params' with
                    | [(paramName, paramType, _nodeId)] when paramName <> "_" ->
                        match paramType with
                        | NativeType.TApp({ Name = "array" }, [NativeType.TApp({ Name = "string" }, [])]) ->
                            // string[] parameter - needs argv conversion
                            [(paramName, None)], true
                        | _ ->
                            // Other parameter type - bind to argv pointer
                            [(paramName, Some (Arg 1, TPtr))], false
                    | [(_, _, _)] -> [], false  // Discarded parameter
                    | [] -> [], false  // Unit parameter
                    | _ ->
                        // Multiple parameters - bind each to sequential args
                        params' |> List.mapi (fun i (name, _ty, _nodeId) ->
                            if name = "_" then (name, None)
                            else (name, Some (Arg (i + 1), TPtr))), false
                cParams, bindings, needsConv, false
            else
                // Regular lambda - use node.Type for parameter types
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

                let mlirPs, bindings =
                    if isUnitParam then
                        // NTUunit has size 0 - no parameter generated
                        // A function taking unit is nullary at the native level
                        // This aligns with call sites which also elide unit arguments
                        [], []
                    else
                        let nodeParamTypes = extractParamTypes node.Type (List.length params')

                        // Use graph-aware type mapping for record types
                        let mapTypeWithGraph = mapNativeTypeWithGraph z.Graph

                        // Build structured params: (Arg i, MLIRType)
                        let ps = params' |> List.mapi (fun i (_name, nativeTy, _nodeId) ->
                            let actualType =
                                if i < List.length nodeParamTypes then nodeParamTypes.[i]
                                else nativeTy
                            (Arg i, mapTypeWithGraph actualType))

                        // Build bindings: (paramName, Some (Arg i, MLIRType))
                        let bs = params' |> List.mapi (fun i (paramName, paramType, _nodeId) ->
                            let actualType =
                                if i < List.length nodeParamTypes then nodeParamTypes.[i]
                                else paramType
                            (paramName, Some (Arg i, mapTypeWithGraph actualType)))
                        ps, bs

                mlirPs, bindings, false, isUnitParam

        // Use graph-aware type mapping for return type (may be record type)
        let mapTypeWithGraph = mapNativeTypeWithGraph z.Graph

        // Return type
        let returnType =
            if isMain then TInt I32
            else
                // If unit param, count is 1 (the unit arg) for return type peeling
                let paramCount = if List.isEmpty params' && (match node.Type with NativeType.TFun(d, _) -> isUnitType d | _ -> false) then 1 else List.length params'
                let finalRetTy = extractFinalReturnType node.Type paramCount
                mapTypeWithGraph finalRetTy

        // Determine visibility
        let visibility = if isMain then FuncVisibility.Public else FuncVisibility.Private

        // Enter function scope - this sets State.Focus = InFunction lambdaName
        let z1 = enterFunction lambdaName mlirParams returnType visibility z

        // Handle argv conversion or standard parameter bindings
        let z2 =
            if needsArgvConversion && not (List.isEmpty paramBindings) then
                let paramName = fst (List.head paramBindings)
                bindArgvParameters paramName z1
            else
                paramBindings
                |> List.fold (fun acc (paramName, bindingOpt) ->
                    match bindingOpt with
                    | Some (ssa, mlirType) -> bindVarSSA paramName ssa mlirType acc
                    | None -> acc
                ) z1

        z2
    | _ -> z
