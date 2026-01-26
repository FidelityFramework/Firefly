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
