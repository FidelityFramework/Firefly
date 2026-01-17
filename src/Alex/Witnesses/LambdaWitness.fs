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



// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL ARENA ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Allocate memory in the global 'closure_heap' bump allocator
/// Returns (ops, resultPtrSSA)
let private allocateInClosureArena (z: PSGZipper) (sizeSSA: SSA) : MLIROp list * SSA =
    let posPtrSSA = freshSynthSSA z
    let posSSA = freshSynthSSA z
    let heapBaseSSA = freshSynthSSA z
    let resultPtrSSA = freshSynthSSA z
    let newPosSSA = freshSynthSSA z
    
    let ops = [
        // Load current position
        MLIROp.LLVMOp (AddressOf (posPtrSSA, GFunc "closure_pos"))
        MLIROp.LLVMOp (Load (posSSA, posPtrSSA, MLIRTypes.i64, NotAtomic))
        
        // Compute result pointer: heap_base + pos
        MLIROp.LLVMOp (AddressOf (heapBaseSSA, GFunc "closure_heap"))
        MLIROp.LLVMOp (GEP (resultPtrSSA, heapBaseSSA, [(posSSA, MLIRTypes.i64)], MLIRTypes.i8))
        
        // Update position: pos + size
        MLIROp.ArithOp (ArithOp.AddI (newPosSSA, posSSA, sizeSSA, MLIRTypes.i64))
        MLIROp.LLVMOp (Store (newPosSSA, posPtrSSA, MLIRTypes.i64, NotAtomic))
    ]
    ops, resultPtrSSA

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
    (z: PSGZipper)
    : MLIROp list =

    // 1. Get address of the lambda function (SSA from SSAAssignment)
    let codeAddrOp = MLIROp.LLVMOp (AddressOf (layout.CodeAddrSSA, GFunc lambdaName))

    // 2. Create the Flat Environment struct (contains captures)
    // We use synthetic SSAs for the intermediate construction
    let flatUndefSSA = freshSynthSSA z
    let flatUndefOp = MLIROp.LLVMOp (Undef (flatUndefSSA, layout.ClosureStructType))
    
    // Insert code_ptr at index 0
    let flatWithCodeSSA = freshSynthSSA z
    let flatWithCodeOp = MLIROp.LLVMOp (InsertValue (flatWithCodeSSA, flatUndefSSA,
        layout.CodeAddrSSA, [0], layout.ClosureStructType))
        
    // Insert each captured value
    let mutable currentFlatSSA = flatWithCodeSSA
    let flatOps = 
        layout.Captures
        |> List.mapi (fun i slot ->
            let capturedSSA, funcOps =
                // Helper to look up variable in PARENT scope (since z is in inner scope)
                let recallParentVarSSA name =
                    match z.State.VarBindingsStack with
                    | parentBindings :: _ -> Map.tryFind name parentBindings
                    | [] -> None
                
                match recallParentVarSSA slot.Name with
                | Some (ssa, _) -> ssa, []
                | None ->
                    match slot.SourceNodeId with
                    | Some srcId ->
                        match recallNodeResult (NodeId.value srcId) z with
                        | Some (ssa, _ty) -> ssa, []
                        | None ->
                            match lookupLambdaName srcId z.State.SSAAssignment with
                            | Some funcName ->
                                // Found a global function: create {ptr, null}
                                // This happens when capturing a top-level function like 'write'
                                let funcPtrSSA = freshSynthSSA z
                                let addrOp = MLIROp.LLVMOp (AddressOf (funcPtrSSA, GFunc funcName))
                                
                                let closureStructSSA = freshSynthSSA z
                                let undefSSA = freshSynthSSA z
                                let withPtrSSA = freshSynthSSA z
                                let nullEnvSSA = freshSynthSSA z
                                let closureTy = TStruct [TPtr; TPtr]
                                
                                let ops = [
                                    addrOp
                                    MLIROp.LLVMOp (Undef (undefSSA, closureTy))
                                    MLIROp.LLVMOp (InsertValue (withPtrSSA, undefSSA, funcPtrSSA, [0], closureTy))
                                    MLIROp.LLVMOp (NullPtr nullEnvSSA)
                                    MLIROp.LLVMOp (InsertValue (closureStructSSA, withPtrSSA, nullEnvSSA, [1], closureTy))
                                ]
                                closureStructSSA, ops
                            | None ->
                                let ssa = match lookupSSA srcId z.State.SSAAssignment with Some s -> s | None -> V 0
                                ssa, []
                    | None -> V 0, []
            
            let nextSSA = freshSynthSSA z
            let insertIndex = 1 + slot.SlotIndex
            let op = MLIROp.LLVMOp (InsertValue (nextSSA, currentFlatSSA, capturedSSA, [insertIndex], layout.ClosureStructType))
            currentFlatSSA <- nextSSA
            funcOps @ [op]
        )
        |> List.concat
        
    // 3. Allocate in Global Arena (avoiding stack lifetime issues)
    // Calculate size using GEP null trick
    let nullPtrSSA = freshSynthSSA z
    let sizePtrSSA = freshSynthSSA z
    let sizeSSA = freshSynthSSA z
    let oneSSA = freshSynthSSA z
    
    let nullOp = MLIROp.LLVMOp (NullPtr nullPtrSSA)
    // Generate constant 1 for GEP index
    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i32))
    // GEP null[1] gives pointer to address == sizeof(type)
    let sizeGepOp = MLIROp.LLVMOp (GEP (sizePtrSSA, nullPtrSSA, [(oneSSA, MLIRTypes.i32)], layout.ClosureStructType))
    let ptrToIntOp = MLIROp.LLVMOp (PtrToInt (sizeSSA, sizePtrSSA, MLIRTypes.i64))
    
    // Allocate
    let allocOps, envPtrSSA = allocateInClosureArena z sizeSSA
    
    // Store the environment struct to the arena
    let storeOp = MLIROp.LLVMOp (Store (currentFlatSSA, envPtrSSA, layout.ClosureStructType, NotAtomic))
    
    // 4. Build the uniform {ptr, ptr} Function Value pair
    // Use layout.ClosureResultSSA for the FINAL result (aligns with AppWitness/SSAAssignment)
    let pairTy = TStruct [TPtr; TPtr]
    let pairUndefSSA = freshSynthSSA z
    let pairWithCodeSSA = freshSynthSSA z
    
    let buildPairOps = [
        MLIROp.LLVMOp (Undef (pairUndefSSA, pairTy))
        MLIROp.LLVMOp (InsertValue (pairWithCodeSSA, pairUndefSSA, layout.CodeAddrSSA, [0], pairTy))
        MLIROp.LLVMOp (InsertValue (layout.ClosureResultSSA, pairWithCodeSSA, envPtrSSA, [1], pairTy))
    ]

    [codeAddrOp; flatUndefOp; flatWithCodeOp] @ flatOps @ 
    [nullOp; oneOp; sizeGepOp; ptrToIntOp] @ allocOps @ [storeOp] @ buildPairOps

// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE EXTRACTION (Inside Closure Function)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate ops to extract captured values from closure struct at function entry
/// These ops are prepended to the function body
///
/// CLOSURE CONVENTION: Closing Lambdas take env_ptr (Arg 0) which points to the closure struct.
/// We load the struct from the pointer, then extract values.
///
/// For each capture slot:
///   %struct = load %env_ptr
///   %extracted = llvm.extractvalue %struct[1 + slotIndex]
let private buildCaptureExtractionOps
    (layout: ClosureLayout)
    (z: PSGZipper)
    : MLIROp list =

    // Arg 0 is the environment pointer (passed as TPtr)
    let envPtrSSA = Arg 0
    
    // Load the closure struct from the environment pointer
    let structSSA = freshSynthSSA z
    let loadOp = MLIROp.LLVMOp (Load (structSSA, envPtrSSA, layout.ClosureStructType, NotAtomic))

    // Extract each captured value from the loaded struct
    // Captures are at indices 1, 2, ... (index 0 is code_ptr)
    let extractOps =
        layout.Captures
        |> List.map (fun slot ->
            // slot.GepSSA is the SSA we bound in preBindParams - use it for the extracted value
            let extractIndex = 1 + slot.SlotIndex
            MLIROp.LLVMOp (ExtractValue (slot.GepSSA, structSSA, [extractIndex], layout.ClosureStructType))
        )

    loadOp :: extractOps

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
    | TStruct _ ->
        // Undef for struct returns (including closures)
        let op = MLIROp.LLVMOp (LLVMOp.Undef (resultSSA, declaredRetType))
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

// ... (Lambda Witnessing) ...

let private witnessInFunctionScope
    (lambdaName: string)
    (node: SemanticNode)
    (bodyId: NodeId)
    (witnessBody: PSGZipper -> MLIROp list * TransferResult)
    (funcParams: (SSA * MLIRType) list)
    (closureLayoutOpt: ClosureLayout option)
    (z: PSGZipper)
    : MLIROp option * MLIROp list * TransferResult =

    // ... (paramCount logic) ...
    let paramCount =
        match node.Kind with
        | SemanticKind.Lambda (params', _bodyId, _captures) -> List.length params'
        | _ -> 0
    let finalRetType = extractFinalReturnType node.Type paramCount
    let declaredRetType = mapNativeTypeWithGraph z.Graph finalRetType

    // ... (bodyResult logic) ...
    let bodyResult = recallNodeResult (NodeId.value bodyId) z

    // ... (returnOps logic) ...
    let returnOps, terminator, actualRetType =
        match bodyResult with
        | Some (ssa, bodyType) ->
            let effectiveRetType =
                match bodyType, declaredRetType with
                | TStruct (TPtr :: _), TStruct [TPtr; TPtr] -> bodyType
                | _ -> declaredRetType
            let reconcileOps, finalSSA = reconcileReturnType z ssa bodyType effectiveRetType
            reconcileOps, createReturnOp (Some finalSSA) (Some effectiveRetType), effectiveRetType
        | None ->
            let defaultOps, defaultSSA = createDefaultReturn z declaredRetType
            defaultOps, createReturnOp (Some defaultSSA) (Some declaredRetType), declaredRetType

    // ... (bodyRes logic) ...
    let bodyOps, bodyRes = witnessBody z
    let returnOps, terminator, actualRetType =
        match bodyRes with
        | TRValue { SSA = ssa; Type = bodyType } ->
            let effectiveRetType =
                match bodyType, declaredRetType with
                | TStruct (TPtr :: _), TStruct [TPtr; TPtr] -> bodyType
                | _ -> declaredRetType
            let reconcileOps, finalSSA = reconcileReturnType z ssa bodyType effectiveRetType
            reconcileOps, createReturnOp (Some finalSSA) (Some effectiveRetType), effectiveRetType
        | _ ->
            let defaultOps, defaultSSA = createDefaultReturn z declaredRetType
            defaultOps, createReturnOp (Some defaultSSA) (Some declaredRetType), declaredRetType

    // For closing lambdas: bind captured variables (already done in preBindParams)
    let captureExtractionOps =
        match closureLayoutOpt with
        | Some layout -> buildCaptureExtractionOps layout z
        | None -> []

    // Build complete body ops
    let completeBodyOps = captureExtractionOps @ bodyOps @ returnOps @ [terminator]

    // Check if this is a closing lambda (has captures)
    // For TRUE FLAT CLOSURES: first parameter is the environment pointer
    let finalMlirParams =
        match closureLayoutOpt with
        | Some layout ->
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

    // ... (emit logic) ...
    
    // Fix linkage mapping:
    let linkage = if isFuncInternal lambdaName z then LLVMPrivate else LLVMExternal

    let llvmFuncDef = LLVMOp.LLVMFuncDef (lambdaName, finalMlirParams, actualRetType, bodyRegion, linkage)

    match closureLayoutOpt with
    | Some layout ->
        // CLOSING LAMBDA: Emit closure construction ops (in parent scope)
        let closureOps = buildClosureConstruction layout lambdaName z
        // Return: llvmFuncDef for TopLevel, closureOps for CurrentOps, TRValue with closure
        // FIX: The result of buildClosureConstruction is now {ptr, ptr} (generic closure),
        // NOT layout.ClosureStructType (flat struct). The flat struct is hidden in the arena.
        Some (MLIROp.LLVMOp llvmFuncDef), closureOps, TRValue { SSA = layout.ClosureResultSSA; Type = TStruct [TPtr; TPtr] }
    | None ->
        // Simple lambda: no closure construction
        // Return: llvmFuncDef for TopLevel, no local ops, TRVoid
        Some (MLIROp.LLVMOp llvmFuncDef), [], TRVoid

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
    (witnessBody: PSGZipper -> MLIROp list * TransferResult)
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

    witnessInFunctionScope lambdaName node bodyId witnessBody funcParams closureLayoutOpt z

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

        // For TRUE FLAT CLOSURES (Arena): parameters start at Arg 1.
        // Arg 0 (env_ptr) is prepended by witnessInFunctionScope.
        // We pass only user params here.
        let finalMlirParams = mlirParams

        // Enter function scope - this sets State.Focus = InFunction lambdaName
        let z1 = enterFunction lambdaName finalMlirParams returnType visibility z

        // Handle argv conversion or standard parameter bindings
        // ARCHITECTURAL FIX: If a parameter is AddressedMutable (captured ByRef or mutated),
        // we MUST allocate stack space and store the argument value.
        // The binding should then point to the ALLOCA, not the argument value.
        let z2 =
            if needsArgvConversion && not (List.isEmpty paramBindings) then
                let paramName = fst (List.head paramBindings)
                bindArgvParameters paramName z1
            else
                paramBindings
                |> List.fold (fun acc (paramName, bindingOpt) ->
                    match bindingOpt with
                    | Some (argSSA, mlirType) ->
                        // Check if this parameter needs alloca (captured ByRef or mutated)
                        // Note: We don't have the NodeId of the parameter binding here easily
                        // But we can check by NAME if we trust unique names in function scope
                        // Or relying on MutabilityAnalysis to track by name if NodeId isn't available
                        // Wait, params' has NodeId!
                        let paramNodeId =
                            params'
                            |> List.tryFind (fun (n, _, _) -> n = paramName)
                            |> Option.map (fun (_, _, id) -> NodeId.value id)
                            
                        let needsStack =
                            match paramNodeId with
                            | Some id -> needsAlloca id paramName acc
                            | None -> false // Should not happen

                        if needsStack then
                            // Allocate stack slot
                            let allocaSSA = freshSynthSSA acc
                            let allocaOp = MLIROp.LLVMOp (LLVMOp.Alloca (allocaSSA, Arg -1, mlirType, None)) // Arg -1 is unused for count=1
                            // Store argument value
                            let storeOp = MLIROp.LLVMOp (LLVMOp.Store (argSSA, allocaSSA, mlirType, AtomicOrdering.NotAtomic))
                            
                            // Emit ops to function prologue
                            emit allocaOp acc
                            emit storeOp acc
                            
                            // Bind name to ALLOCA (as pointer)
                            // VarRef witness handles loading from pointers for mutable bindings
                            // But we need to make sure the type recorded is the VALUE type or PTR type?
                            // bindVarSSA takes the type of the SSA. allocaSSA is TPtr.
                            // But checking logic expects to know the underlying type?
                            // VarRef witness uses the type from NodeBindings/VarBindings.
                            // If we bind (allocaSSA, TPtr), then VarRef will see TPtr.
                            // Standard VarRef logic: if it's mutable/addressed, it expects TPtr and loads it.
                            bindVarSSA paramName allocaSSA TPtr acc
                        else
                            // Immutable parameter - bind directly to argument value
                            bindVarSSA paramName argSSA mlirType acc
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
                    // Mark ALL captures so VarRef knows to use VarBindings (not NodeBindings)
                    let acc'' = markCapturedVariable slot.Name acc'
                    // Additionally mark ByRef captures so VarRef/Set know to load/store through pointer
                    match slot.Mode with
                    | ByRef -> markCapturedMutable slot.Name acc''
                    | ByValue -> acc''
                ) z2
            | None -> z2

        z3
    | _ -> z
