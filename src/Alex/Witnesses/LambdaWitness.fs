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
/// CLOSURE ARCHITECTURE (MLKit-style flat closures):
/// - Simple Lambda (no captures): Emit func.func, call via func.call @name
/// - Closing Lambda (with captures):
///   - Emit func.func with env_ptr as first parameter
///   - In parent scope, emit closure construction:
///     1. Alloca env struct
///     2. GEP + Store for each captured value
///     3. Build closure struct {code_ptr, env_ptr}
///
/// WITNESS PATTERN: Returns (MLIROp list * TransferResult)
/// Witnesses OBSERVE and RETURN. They do NOT emit directly.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping
open Alex.Preprocessing.SSAAssignment

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let private mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE CONSTRUCTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build closure construction ops for a Lambda with captures
/// Uses pre-computed ClosureLayout from SSAAssignment (coeffect pattern)
///
/// TRUE FLAT CLOSURE: Captures are inlined directly in the closure struct.
/// Struct layout: {code_ptr, capture_0, capture_1, ...}
/// No alloca, no env_ptr - captures are copied by value into the struct.
///
/// ALL SSAs come from SSAAssignment - witnesses OBSERVE, do NOT compute.
///
/// Emits (in parent scope):
///   %code_addr = llvm.mlir.addressof @lambda_name
///   %undef = llvm.mlir.undef : !llvm.struct<(ptr, T0, T1, ...)>
///   %with_code = llvm.insertvalue %undef, %code_addr[0] : ...
///   %with_cap0 = llvm.insertvalue %with_code, %captured0[1] : ...
///   %closure = llvm.insertvalue %with_capN-1, %capturedN[N] : ...
let private buildClosureConstruction
    (layout: ClosureLayout)
    (lambdaName: string)
    (z: PSGZipper)
    : MLIROp list =

    // 1. Get address of the lambda function (SSA from SSAAssignment)
    let codeAddrOp = MLIROp.LLVMOp (AddressOf (layout.CodeAddrSSA, GFunc lambdaName))

    // 2. Create undef closure struct (SSA from SSAAssignment)
    let undefOp = MLIROp.LLVMOp (Undef (layout.ClosureUndefSSA, layout.ClosureStructType))

    // 3. Insert code_ptr at index 0 (SSA from SSAAssignment)
    let withCodeOp = MLIROp.LLVMOp (InsertValue (layout.ClosureWithCodeSSA, layout.ClosureUndefSSA,
        layout.CodeAddrSSA, [0], layout.ClosureStructType))

    // 4. Insert each captured value at index (1 + slotIndex)
    // Use pre-computed CaptureInsertSSAs from SSAAssignment
    // Each insertvalue uses the PREVIOUS insertvalue result as input
    let captureOps =
        layout.Captures
        |> List.mapi (fun i slot ->
            // Get the captured value's SSA from the source binding
            let capturedSSA =
                match slot.SourceNodeId with
                | Some srcId ->
                    match recallNodeResult (NodeId.value srcId) z with
                    | Some (ssa, _ty) -> ssa
                    | None ->
                        match lookupSSA srcId z.State.SSAAssignment with
                        | Some ssa -> ssa
                        | None -> V 0
                | None -> V 0

            // Previous SSA: for i=0, use ClosureWithCodeSSA; otherwise use previous CaptureInsertSSA
            let prevSSA =
                if i = 0 then layout.ClosureWithCodeSSA
                else layout.CaptureInsertSSAs.[i - 1]

            // Result SSA from pre-computed CaptureInsertSSAs
            let resultSSA = layout.CaptureInsertSSAs.[i]

            // Insert at index (1 + slotIndex) since index 0 is code_ptr
            let insertIndex = 1 + slot.SlotIndex
            MLIROp.LLVMOp (InsertValue (resultSSA, prevSSA, capturedSSA, [insertIndex], layout.ClosureStructType))
        )

    // The final CaptureInsertSSA equals ClosureResultSSA (per SSAAssignment)
    [codeAddrOp; undefOp; withCodeOp] @ captureOps

// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE EXTRACTION (Inside Closure Function)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate ops to extract captured values from closure struct at function entry
/// These ops are prepended to the function body
///
/// TRUE FLAT CLOSURE: Arg 0 is the closure struct itself (passed by value).
/// Layout: {code_ptr, capture_0, capture_1, ...}
///
/// For each capture slot:
///   %extracted = llvm.extractvalue %closure_arg[1 + slotIndex]
///
/// The extracted SSA is bound to the capture name in preBindParams
let private buildCaptureExtractionOps
    (layout: ClosureLayout)
    (_z: PSGZipper)
    : MLIROp list =

    // For true flat closures, Arg 0 is the closure struct itself (not a pointer to env)
    let closureArgSSA = Arg 0

    // Extract each captured value directly from the closure struct
    // Captures are at indices 1, 2, ... (index 0 is code_ptr)
    let extractOps =
        layout.Captures
        |> List.map (fun slot ->
            // slot.GepSSA is the SSA we bound in preBindParams - use it for the extracted value
            // Extract from index (1 + slotIndex) since index 0 is code_ptr
            let extractIndex = 1 + slot.SlotIndex
            MLIROp.LLVMOp (ExtractValue (slot.GepSSA, closureArgSSA, [extractIndex], layout.ClosureStructType))
        )

    extractOps

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
let private createDefaultReturn (z: PSGZipper) (declaredRetType: MLIRType) : MLIROp list * SSA =
    // Use synthetic SSA for return value synthesis (not pre-assigned)
    let resultSSA = freshSynthSSA z
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
/// Uses synthetic SSAs for type reconciliation ops (not pre-assigned)
let private reconcileReturnType
    (z: PSGZipper)
    (bodySSA: SSA)
    (bodyType: MLIRType)
    (declaredRetType: MLIRType)
    : MLIROp list * SSA =
    match declaredRetType, bodyType with
    | TInt I32, TPtr ->
        // Unit function with pointer body - return 0
        let resultSSA = freshSynthSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstI (resultSSA, 0L, TInt I32))
        [op], resultSSA
    | TPtr, TInt _ ->
        // Pointer return but integer body - return null
        let resultSSA = freshSynthSSA z
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
///
/// CLOSURE HANDLING:
/// - Simple Lambda (no captures): Emit func.func, return TRVoid
/// - Closing Lambda (with captures):
///   - Emit func.func with env_ptr as first parameter (Arg 0)
///   - Return closure construction ops and TRValue with closure struct
let private witnessInFunctionScope
    (lambdaName: string)
    (node: SemanticNode)
    (bodyId: NodeId)
    (bodyOps: MLIROp list)
    (funcParams: (SSA * MLIRType) list)
    (closureLayoutOpt: ClosureLayout option)
    (z: PSGZipper)
    : MLIROp option * MLIROp list * TransferResult =

    // Determine declared return type (use graph-aware mapping for record types)
    let paramCount =
        match node.Kind with
        | SemanticKind.Lambda (params', _bodyId, _captures) -> List.length params'
        | _ -> 0
    let finalRetType = extractFinalReturnType node.Type paramCount
    let declaredRetType = mapNativeTypeWithGraph z.Graph finalRetType

    // Look up body's SSA result (already processed in post-order)
    // Using structured recallNodeResult which returns (SSA * MLIRType) option
    let bodyResult =
        recallNodeResult (NodeId.value bodyId) z

    // Build return/terminator ops
    // TRUE FLAT CLOSURE: When body produces a closure struct, use its ACTUAL type
    // (not the mapped TFun -> {ptr,ptr} type from declaredRetType)
    let returnOps, terminator, actualRetType =
        match bodyResult with
        | Some (ssa, bodyType) ->
            // Check if body produces a closure struct (TStruct starting with TPtr)
            // If so, use bodyType as the actual return type
            let effectiveRetType =
                match bodyType, declaredRetType with
                | TStruct (TPtr :: _), TStruct [TPtr; TPtr] ->
                    // Body is a flat closure struct, declaredRetType is old {ptr,ptr} model
                    // Use the actual closure struct type
                    bodyType
                | _ ->
                    declaredRetType
            // Normal return - may need type reconciliation
            let reconcileOps, finalSSA = reconcileReturnType z ssa bodyType effectiveRetType
            reconcileOps, createReturnOp (Some finalSSA) (Some effectiveRetType), effectiveRetType
        | None ->
            // No body result - create default (uses synthetic SSA)
            let defaultOps, defaultSSA = createDefaultReturn z declaredRetType
            defaultOps, createReturnOp (Some defaultSSA) (Some declaredRetType), declaredRetType

    // For closing lambdas: prepend capture extraction ops to body
    // These ops extract captured values from env struct at function entry
    let captureExtractionOps =
        match closureLayoutOpt with
        | Some layout -> buildCaptureExtractionOps layout z
        | None -> []

    // Build complete body ops: capture extraction + accumulated + return + terminator
    let completeBodyOps = captureExtractionOps @ bodyOps @ returnOps @ [terminator]

    // funcParams already includes env_ptr if this is a closing lambda (from preBindParams)
    // No additional modification needed here
    let finalFuncParams = funcParams

    // Build function body as single entry block
    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []
        Ops = completeBodyOps
    }
    let bodyRegion: Region = { Blocks = [entryBlock] }

    // For closing lambdas: emit closure construction ops and return TRValue
    // For simple lambdas: just the funcDef, TRVoid
    match closureLayoutOpt with
    | Some layout ->
        // CLOSING LAMBDA: Use llvm.func because we take its address with llvm.mlir.addressof
        // llvm.mlir.addressof can only reference llvm.func or llvm.mlir.global
        let llvmFuncDef = LLVMOp.LLVMFuncDef (lambdaName, finalFuncParams, actualRetType, bodyRegion, LLVMPrivate)
        // Build closure construction ops (in parent scope)
        let closureOps = buildClosureConstruction layout lambdaName z
        // Return: llvmFuncDef for TopLevel, closureOps for CurrentOps, TRValue with closure
        Some (MLIROp.LLVMOp llvmFuncDef), closureOps, TRValue { SSA = layout.ClosureResultSSA; Type = layout.ClosureStructType }
    | None ->
        // Simple lambda: use func.func, no address taken
        let visibility =
            if isFuncInternal lambdaName z then FuncVisibility.Private
            else FuncVisibility.Public
        let funcDef = FuncOp.FuncDef (lambdaName, finalFuncParams, actualRetType, bodyRegion, visibility)
        // Photographer Principle: OBSERVE and RETURN, do not EMIT
        Some (MLIROp.FuncOp funcDef), [], TRVoid

/// Witness a Lambda node - main entry point
/// Returns: (funcDef option * localOps * TransferResult)
/// - funcDef: The function definition to add to TopLevel
/// - localOps: Local operations (closure construction for captures)
/// - result: The TransferResult for this node (TRVoid for simple, TRValue for closing)
///
/// PHOTOGRAPHER PRINCIPLE: This witness OBSERVES and RETURNS.
/// It does NOT emit directly. The caller (FNCSTransfer) handles accumulation.
///
/// CLOSURE HANDLING:
/// - Simple Lambda (no captures): Return funcDef, [], TRVoid
/// - Closing Lambda (with captures): Return funcDef, closureOps, TRValue {closure_struct}
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
        match lookupLambdaName node.Id z.State.SSAAssignment with
        | Some name -> name
        | None -> sprintf "lambda_%d" (NodeId.value node.Id)

    // Look up ClosureLayout coeffect - determines if this is a closing lambda
    // For closing lambdas, SSAAssignment has pre-computed all layout information
    let closureLayoutOpt = lookupClosureLayout node.Id z.State.SSAAssignment

    witnessInFunctionScope lambdaName node bodyId bodyOps funcParams closureLayoutOpt z

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
let preBindParams (z: PSGZipper) (node: SemanticNode) : PSGZipper =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId, captures) ->
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

                // CLOSURE HANDLING: If this lambda has captures, Arg 0 is env_ptr
                // and all regular parameters shift by 1
                let hasCapturesClosure = not (List.isEmpty captures)
                let argOffset = if hasCapturesClosure then 1 else 0

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

                        // Build structured params: (Arg i+offset, MLIRType)
                        // For closing lambdas, params start at Arg 1 (Arg 0 is env_ptr)
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

        // Check if this is a closing lambda (has captures)
        let closureLayoutOpt = lookupClosureLayout node.Id z.State.SSAAssignment
        let hasCapturesClosure = Option.isSome closureLayoutOpt

        // For TRUE FLAT CLOSURES: first parameter is the closure struct itself (not env_ptr)
        // Layout: {code_ptr, capture_0, capture_1, ...}
        // The closure struct is passed BY VALUE, enabling safe stack returns
        let finalMlirParams =
            match closureLayoutOpt with
            | Some layout ->
                // Prepend closure struct (Arg 0) - the full closure with inline captures
                (Arg 0, layout.ClosureStructType) :: mlirParams
            | None ->
                mlirParams

        // Enter function scope - this sets State.Focus = InFunction lambdaName
        let z1 = enterFunction lambdaName finalMlirParams returnType visibility z

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

        // For closing lambdas: bind captured variables
        // TRUE FLAT CLOSURE: Captures are extracted from closure struct (Arg 0) at indices 1, 2, ...
        // We bind SSAs here; actual extractvalue ops are emitted by witness
        let z3 =
            match closureLayoutOpt with
            | Some layout ->
                // Bind each captured variable name to its slot's GEP SSA
                // For ByRef captures: SSA is a pointer, mark as captured mutable
                // For ByValue captures: SSA is the value itself
                layout.Captures
                |> List.fold (fun acc slot ->
                    let acc' = bindVarSSA slot.Name slot.GepSSA slot.SlotType acc
                    // Mark ByRef captures so VarRef/Set know to load/store through pointer
                    match slot.Mode with
                    | ByRef -> markCapturedMutable slot.Name acc'
                    | ByValue -> acc'
                ) z2
            | None -> z2

        z3
    | _ -> z
