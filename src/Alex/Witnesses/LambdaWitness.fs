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

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.Coeffects
open PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL ARENA ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Allocate memory in the global 'closure_heap' bump allocator
/// Uses pre-computed SSAs from ClosureLayout (coeffect pattern)
/// Returns (ops, resultPtrSSA)
let private allocateInClosureArena (layout: ClosureLayout) : MLIROp list * SSA =
    let ops = [
        // Load current position
        MLIROp.LLVMOp (AddressOf (layout.HeapPosPtrSSA, GFunc "closure_pos"))
        MLIROp.LLVMOp (Load (layout.HeapPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))

        // Compute result pointer: heap_base + pos
        MLIROp.LLVMOp (AddressOf (layout.HeapBaseSSA, GFunc "closure_heap"))
        MLIROp.LLVMOp (GEP (layout.HeapResultPtrSSA, layout.HeapBaseSSA, [(layout.HeapPosSSA, MLIRTypes.i64)], MLIRTypes.i8))

        // Update position: pos + size (sizeSSA from layout)
        MLIROp.ArithOp (ArithOp.AddI (layout.HeapNewPosSSA, layout.HeapPosSSA, layout.SizeSSA, MLIRTypes.i64))
        MLIROp.LLVMOp (Store (layout.HeapNewPosSSA, layout.HeapPosPtrSSA, MLIRTypes.i64, NotAtomic))
    ]
    ops, layout.HeapResultPtrSSA

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE CONSTRUCTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build closure construction ops for a Lambda with captures
/// Uses pre-computed ClosureLayout from SSAAssignment (coeffect pattern)
///
/// TRUE FLAT CLOSURE (HEAP ALLOCATED):
/// 1. Construct the environment struct (captures) by value.
/// 2. Allocate space in the global closure arena.
/// 3. Store the environment struct to the arena.
/// 4. Return Uniform Pair: {code_ptr, env_ptr}
let private buildClosureConstruction
    (layout: ClosureLayout)
    (lambdaName: string)
    (ctx: WitnessContext)
    : MLIROp list =

    // 1. Get address of the lambda function (SSA from ClosureLayout)
    let codeAddrOp = MLIROp.LLVMOp (AddressOf (layout.CodeAddrSSA, GFunc lambdaName))

    // 2. Create the Flat Environment struct (contains captures)
    // All SSAs from ClosureLayout - no synthesis during emission
    let flatUndefOp = MLIROp.LLVMOp (Undef (layout.ClosureUndefSSA, layout.ClosureStructType))

    // Insert code_ptr at index 0
    let flatWithCodeOp = MLIROp.LLVMOp (InsertValue (layout.ClosureWithCodeSSA, layout.ClosureUndefSSA,
        layout.CodeAddrSSA, [0], layout.ClosureStructType))

    // Insert each captured value using pre-computed CaptureInsertSSAs
    let mutable currentFlatSSA = layout.ClosureWithCodeSSA
    let flatOps =
        layout.Captures
        |> List.mapi (fun i slot ->
            let capturedSSA =
                // Look up variable in PARENT scope (current scope is inner lambda)
                let recallParentVarSSA name =
                    match ctx.Accumulator.ScopeStack with
                    | parentScope :: _ -> Map.tryFind name parentScope.VarAssoc
                    | [] -> None

                match recallParentVarSSA slot.Name with
                | Some (ssa, _) -> ssa
                | None ->
                    match slot.SourceNodeId with
                    | Some srcId ->
                        match MLIRAccumulator.recallNode (NodeId.value srcId) ctx.Accumulator with
                        | Some (ssa, _ty) -> ssa
                        | None ->
                            match lookupSSA srcId ctx.Coeffects.SSA with
                            | Some s -> s
                            | None ->
                                failwithf "LambdaWitness: No SSA for captured variable '%s' (sourceNodeId %d)"
                                    slot.Name (NodeId.value srcId)
                    | None ->
                        failwithf "LambdaWitness: No source node for captured variable '%s'" slot.Name

            // Use pre-computed SSA from CaptureInsertSSAs
            let nextSSA = layout.CaptureInsertSSAs.[i]
            let insertIndex = 1 + slot.SlotIndex
            let op = MLIROp.LLVMOp (InsertValue (nextSSA, currentFlatSSA, capturedSSA, [insertIndex], layout.ClosureStructType))
            currentFlatSSA <- nextSSA
            [op]
        )
        |> List.concat

    // 3. Allocate in Global Arena (avoiding stack lifetime issues)
    // Calculate size using GEP null trick - all SSAs from ClosureLayout
    let nullOp = MLIROp.LLVMOp (NullPtr layout.SizeNullPtrSSA)
    // Generate constant 1 for GEP index
    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (layout.SizeOneSSA, 1L, MLIRTypes.i32))
    // GEP null[1] gives pointer to address == sizeof(type)
    let sizeGepOp = MLIROp.LLVMOp (GEP (layout.SizeGepSSA, layout.SizeNullPtrSSA, [(layout.SizeOneSSA, MLIRTypes.i32)], layout.ClosureStructType))
    let ptrToIntOp = MLIROp.LLVMOp (PtrToInt (layout.SizeSSA, layout.SizeGepSSA, MLIRTypes.i64))

    // Allocate using ClosureLayout SSAs
    let allocOps, envPtrSSA = allocateInClosureArena layout

    // Store the environment struct to the arena
    let storeOp = MLIROp.LLVMOp (Store (currentFlatSSA, envPtrSSA, layout.ClosureStructType, NotAtomic))

    // 4. Build the uniform {ptr, ptr} Function Value pair
    // Use layout SSAs for pair construction
    let pairTy = TStruct [TPtr; TPtr]

    let buildPairOps = [
        MLIROp.LLVMOp (Undef (layout.PairUndefSSA, pairTy))
        MLIROp.LLVMOp (InsertValue (layout.PairWithCodeSSA, layout.PairUndefSSA, layout.CodeAddrSSA, [0], pairTy))
        MLIROp.LLVMOp (InsertValue (layout.ClosureResultSSA, layout.PairWithCodeSSA, envPtrSSA, [1], pairTy))
    ]

    [codeAddrOp; flatUndefOp; flatWithCodeOp] @ flatOps @
    [nullOp; oneOp; sizeGepOp; ptrToIntOp] @ allocOps @ [storeOp] @ buildPairOps

// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE EXTRACTION (Inside Closure Function)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate ops to extract captured values from closure struct at function entry
/// These ops are prepended to the function body
///
/// CLOSURE CONVENTION: Closing Lambdas take env_ptr (Arg 0) which points to the struct.
/// The struct type and extraction indices depend on context:
/// - Regular closure: Load {code_ptr, cap0, cap1, ...}, extract at indices 1, 2, ...
/// - Lazy thunk: Load {computed, value, code_ptr, cap0, cap1, ...}, extract at indices 3, 4, ...
///
/// January 2026: Compositional layout via ClosureLayout coeffect
/// SSA layout: v0..v(N-1) = extraction, vN = struct load, v(N+1)+ = body
let private buildCaptureExtractionOps
    (layout: ClosureLayout)
    (_ctx: WitnessContext)
    : MLIROp list =

    // Arg 0 is the environment pointer (passed as TPtr)
    let envPtrSSA = Arg 0

    // Get struct type and extraction base from context (compositional layout)
    let loadStructType = closureLoadStructType layout
    let extractionBase = closureExtractionBaseIndex layout

    // Load the struct from the environment pointer
    // Use pre-computed StructLoadSSA from ClosureLayout (coeffect pattern)
    let loadOp = MLIROp.LLVMOp (Load (layout.StructLoadSSA, envPtrSSA, loadStructType, NotAtomic))

    // Extract each captured value from the loaded struct
    // Extraction index = baseIndex + slotIndex (context-dependent)
    // Extraction SSAs are derived from SlotIndex: v0, v1, ..., v(N-1)
    // This is the child function's SSA namespace - body SSAs start at N+1
    let extractOps =
        layout.Captures
        |> List.map (fun slot ->
            let extractSSA = V slot.SlotIndex  // Derived from PSG structure
            let extractIndex = extractionBase + slot.SlotIndex
            MLIROp.LLVMOp (ExtractValue (extractSSA, layout.StructLoadSSA, [extractIndex], loadStructType))
        )

    loadOp :: extractOps

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Argv Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Bind entry point parameters for C-style main
/// At OS entry point: %arg0: i32 = argc, %arg1: !llvm.ptr = argv
/// For F# string[] parameter, pattern matching handles conversion at use site
let private bindArgvParameters (paramName: string) (ctx: WitnessContext) : unit =
    // Bind C-style argc/argv under well-known names using STRUCTURED types
    MLIRAccumulator.bindVar "__argc" (Arg 0) MLIRTypes.i32 ctx.Accumulator
    MLIRAccumulator.bindVar "__argv" (Arg 1) MLIRTypes.ptr ctx.Accumulator
    // Bind F# parameter name to argv pointer
    MLIRAccumulator.bindVar paramName (Arg 1) MLIRTypes.ptr ctx.Accumulator

// ═══════════════════════════════════════════════════════════════════════════
// Function Body Building
// ═══════════════════════════════════════════════════════════════════════════

let private createReturnOp (valueSSA: SSA) (valueTy: MLIRType) (isClosure: bool) : MLIROp =
    if isClosure then
        MLIROp.LLVMOp (LLVMOp.Return (Some valueSSA, Some valueTy))
    else
        MLIROp.FuncOp (FuncOp.FuncReturn [(valueSSA, valueTy)])

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

// ... (Lambda Witnessing) ...

/// Check if function is internal (private visibility)
/// Main is public, all others are private
let private isFuncInternal (name: string) : bool =
    name <> "main"

let private witnessInFunctionScope
    (lambdaName: string)
    (node: SemanticNode)
    (bodyId: NodeId)
    (witnessBody: WitnessContext -> MLIROp list * TransferResult)
    (funcParams: (SSA * MLIRType) list)
    (closureLayoutOpt: ClosureLayout option)
    (ctx: WitnessContext)
    : MLIROp option * MLIROp list * TransferResult =

    let paramCount =
        match node.Kind with
        | SemanticKind.Lambda (params', _bodyId, _captures, _, _) -> List.length params'
        | _ -> 0
    let finalRetType = extractFinalReturnType node.Type paramCount
    let declaredRetType = mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph finalRetType

    // FLAT CLOSURE PATTERN (January 2026):
    // Closures (address taken via llvm.mlir.addressof) need llvm.func + llvm.return
    // Non-closures (called by name) use func.func + func.return
    //
    // PRD-14 FIX (January 2026):
    // Lazy thunks ALWAYS have their address taken (even without captures),
    // so they must use llvm.func. Check the Lambda's context.
    let isLazyThunk =
        match node.Kind with
        | SemanticKind.Lambda (_, _, _, _, LambdaContext.LazyThunk) -> true
        | _ -> false
    let isClosure = closureLayoutOpt.IsSome || isLazyThunk

    let bodyOps, bodyRes = witnessBody ctx
    let terminator, actualRetType =
        match bodyRes with
        | TRValue { SSA = ssa; Type = bodyType } ->
            let effectiveRetType =
                match bodyType, declaredRetType with
                | TStruct (TInt I1 :: valTy :: TPtr :: caps), TStruct [TInt I1; declValTy; TPtr]
                    when not (List.isEmpty caps) && valTy = declValTy -> bodyType
                | TStruct (TInt I32 :: valTy :: TPtr :: caps), TStruct [TInt I32; declValTy; TPtr]
                    when not (List.isEmpty caps) && valTy = declValTy -> bodyType
                | TStruct (TPtr :: caps), TStruct [TPtr] when not (List.isEmpty caps) -> bodyType
                | TStruct (TPtr :: _), TStruct [TPtr; TPtr] -> bodyType
                | _ -> declaredRetType
            createReturnOp ssa effectiveRetType isClosure, effectiveRetType
        | TRVoid ->
            // Unit-returning function - emit void return
            if isClosure then
                MLIROp.LLVMOp (LLVMOp.Return (None, None)), MLIRTypes.unit
            else
                MLIROp.FuncOp (FuncOp.FuncReturn []), MLIRTypes.unit
        | TRError msg -> failwithf "Lambda body error: %s" msg
        | TRBuiltin (name, _) -> failwithf "Lambda body cannot be builtin: %s" name

    let captureExtractionOps =
        match closureLayoutOpt with
        | Some layout -> buildCaptureExtractionOps layout ctx
        | None -> []

    let completeBodyOps = captureExtractionOps @ bodyOps @ [terminator]

    // Check if this is a closing lambda (has captures)
    // For TRUE FLAT CLOSURES: first parameter is the environment pointer
    let finalMlirParams =
        match closureLayoutOpt with
        | Some _layout ->
            // Prepend env pointer (Arg 0) - passed as opaque ptr
            (Arg 0, MLIRTypes.ptr) :: funcParams
        | None ->
            funcParams

    // Build function body as single entry block
    let entryBlock: Block = {
        Label = BlockRef "entry"
        Args = []
        Ops = completeBodyOps
    }
    let bodyRegion: Region = { Blocks = [entryBlock] }

    // FLAT CLOSURE PATTERN (January 2026):
    // Closures (address taken) use llvm.func because llvm.mlir.addressof requires it
    // Non-closures (called by name) use func.func for MLIR portability
    // See fsnative-spec/spec/drafts/backend-lowering-architecture.md
    match closureLayoutOpt, isLazyThunk with
    | Some layout, _ ->
        // CLOSING LAMBDA: use llvm.func (address will be taken)
        let llvmVisibility = if isFuncInternal lambdaName then LLVMPrivate else LLVMPrivate // closures always private
        let llvmFuncDef = LLVMOp.LLVMFuncDef (lambdaName, finalMlirParams, actualRetType, bodyRegion, llvmVisibility)
        let closureOps = buildClosureConstruction layout lambdaName ctx
        // Return: funcDef for TopLevel, closureOps for CurrentOps, TRValue with closure
        // FIX: The result of buildClosureConstruction is now {ptr, ptr} (generic closure),
        // NOT layout.ClosureStructType (flat struct). The flat struct is hidden in the arena.
        Some (MLIROp.LLVMOp llvmFuncDef), closureOps, TRValue { SSA = layout.ClosureResultSSA; Type = TStruct [TPtr; TPtr] }
    | None, true ->
        // PRD-14: Lazy thunk without captures - use llvm.func (address will be taken)
        // but no closure construction needed
        let llvmFuncDef = LLVMOp.LLVMFuncDef (lambdaName, finalMlirParams, actualRetType, bodyRegion, LLVMPrivate)
        Some (MLIROp.LLVMOp llvmFuncDef), [], TRVoid
    | None, false ->
        // Simple lambda: no closure construction, use func.func (MLIR portable)
        let visibility = if isFuncInternal lambdaName then Private else Public
        let funcDef = FuncOp.FuncDef (lambdaName, finalMlirParams, actualRetType, bodyRegion, visibility)
        Some (MLIROp.FuncOp funcDef), [], TRVoid

/// Witness a Lambda node - main entry point
/// Returns: (funcDef option * localOps * TransferResult)
/// - funcDef: The function definition to add to TopLevel
/// - localOps: Local operations (closure construction for captures)
/// - result: The TransferResult for this node (TRVoid for simple, TRValue for closing)
///
/// PHOTOGRAPHER PRINCIPLE: This witness OBSERVES and RETURNS.
/// It does NOT emit directly. The caller (MLIRTransfer) handles accumulation.
///
/// CLOSURE HANDLING:
/// - Simple Lambda (no captures): Return funcDef, [], TRVoid
/// - Closing Lambda (with captures): Return funcDef, closureOps, TRValue {closure_struct}
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
