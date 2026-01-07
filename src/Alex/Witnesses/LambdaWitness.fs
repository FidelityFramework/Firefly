/// Lambda Witness - Witness Lambda nodes to MLIR functions
///
/// Handles both cases:
/// 1. InFunction scope - Lambda body was captured by preBindLambdaParams
/// 2. Fallback - Create function directly (old behavior)
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Templates.TemplateTypes

module ArithTemplates = Alex.Templates.ArithTemplates
module LLVMTemplates = Alex.Templates.LLVMTemplates

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let private mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

/// Helper: extract first element of 3-tuple
let private fst3 (a, _, _) = a

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Argv Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MLIR prologue to convert C argv (char**) to F# string[] (fat pointers)
/// 
/// At OS entry point, we receive:
///   %arg0: i32 = argc (count including program name)
///   %arg1: !llvm.ptr = argv (pointer to null-terminated C strings)
///
/// F# entry point expects:
///   argv: string[] = array of fat pointer strings (ptr, len)
///
/// This function generates the conversion, binding the F# parameter name to
/// a pointer to the constructed F# string array (as a fat pointer with count).
///
/// The principled approach:
///   1. Compute count = argc - 1 (excluding program name)
///   2. For each argv[i] where i >= 1:
///      - Load the char* pointer
///      - Compute strlen via inline null byte scan
///      - Construct fat pointer {ptr, len}
///   3. Bind F# parameter to array info (pointer to first converted string, count)
///
/// For the current use case (pattern matching [|prefix|]), we generate inline
/// conversion at the pattern match site. This is the transitional approach
/// while we build out the full conversion infrastructure.
///
/// ARCHITECTURAL NOTE: Eventually, this should generate a proper loop that
/// pre-converts ALL argv strings to F# fat pointers. The pattern matching
/// code would then ONLY work with F# semantics.
let private generateArgvConversionPrologue (paramName: string) (zipper: MLIRZipper) : MLIRZipper =
    // Bind argc and argv under well-known names for pattern matching access
    // Pattern matching code uses these to:
    // 1. Check array length (argc - 1)
    // 2. Convert individual C strings to F# fat pointers at extraction time
    //
    // This is a transitional approach. The principled fix would be to generate
    // the full conversion loop here so pattern matching only sees F# strings.
    // That requires:
    // 1. Inline strlen (null byte scan loop) - pure MLIR, no libc
    // 2. Array allocation for fat pointers
    // 3. Population loop
    //
    // For now, the pattern matching code in ControlFlowWitness handles the
    // C-to-F# string conversion at extraction time using inline strlen.
    
    let z1 = MLIRZipper.bindVar "__argc" "%arg0" "i32" zipper
    let z2 = MLIRZipper.bindVar "__argv" "%arg1" "!llvm.ptr" z1
    
    // Bind the F# parameter name to the argv pointer
    // The pattern matching code knows to treat this specially
    MLIRZipper.bindVar paramName "%arg1" "!llvm.ptr" z2

// ═══════════════════════════════════════════════════════════════════════════
// Lambda Witnessing
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Lambda node when already in function scope
/// Body operations were captured inside function scope by preBindLambdaParams
let private witnessInFunctionScope
    (funcName: string)
    (node: SemanticNode)
    (bodyId: NodeId)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =
    
    // Get the lambda name that was stored during pre-bind
    let lambdaName = 
        match MLIRZipper.recallNodeSSA (string (NodeId.value node.Id) + "_lambdaName") zipper with
        | Some (name, _) -> name
        | None -> funcName  // Use the current function name as fallback
    
    // Determine the declared return type from the Lambda's F# type
    // For curried functions, peel all TFun layers to get the final return type
    let declaredRetType =
        // Get param count from the lambda's parameters
        let paramCount =
            match node.Kind with
            | SemanticKind.Lambda (params', _) -> List.length params'
            | _ -> 0

        // Peel TFun layers based on param count (or peel one for unit lambdas)
        let rec extractFinalReturnType (ty: NativeType) (count: int) : NativeType =
            match ty with
            | NativeType.TFun(_, resultTy) when count > 0 ->
                extractFinalReturnType resultTy (count - 1)
            | NativeType.TFun(_, resultTy) when paramCount = 0 ->
                // Unit lambda: peel the unit->result layer
                resultTy
            | _ -> ty

        let finalRetTy = extractFinalReturnType node.Type paramCount
        Serialize.mlirType (mapType finalRetTy)
    
    // Look up the body's SSA result (already processed in post-order, inside function scope)
    // Thread the zipper through in case we need to generate a default value
    let bodySSA, bodyType, zipperWithBody = 
        match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper with
        | Some (ssa, ty) when ssa <> "" -> ssa, ty, zipper
        | _ -> 
            // Body didn't produce a value (or produced empty SSA) - generate appropriate default
            let zeroSSA, z = MLIRZipper.yieldSSA zipper
            if declaredRetType = "!llvm.ptr" then
                // For pointer returns, use llvm.mlir.zero for null pointer
                let zeroParams = {| Result = zeroSSA; Type = "!llvm.ptr" |}
                let zeroText = render LLVMTemplates.Quot.Global.zeroInit zeroParams
                let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA Pointer z
                zeroSSA, "!llvm.ptr", z'
            else
                // For integer returns, use arith.constant 0
                let constParams : ConstantParams = { Result = zeroSSA; Value = "0"; Type = declaredRetType }
                let zeroText = render ArithTemplates.Quot.Constant.intConst constParams
                let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
                zeroSSA, declaredRetType, z'
    
    // Check for return type mismatch
    // Case 1: declared i32 (unit) but body produces ptr - return 0:i32
    // Case 2: declared ptr but body produces i32 - return null ptr
    let returnSSA, returnType, zipperForReturn =
        if declaredRetType = "i32" && bodyType = "!llvm.ptr" then
            // Unit function with side-effecting body - ignore result, return 0
            let zeroSSA, z = MLIRZipper.yieldSSA zipperWithBody
            let constParams : ConstantParams = { Result = zeroSSA; Value = "0"; Type = "i32" }
            let zeroText = render ArithTemplates.Quot.Constant.intConst constParams
            let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
            zeroSSA, "i32", z'
        elif declaredRetType = "!llvm.ptr" && (bodyType = "i32" || bodyType.StartsWith("i")) then
            // Function returns ptr but body computed an integer - return null ptr
            let nullSSA, z = MLIRZipper.yieldSSA zipperWithBody
            let nullParams = {| Result = nullSSA; Type = "!llvm.ptr" |}
            let nullText = render LLVMTemplates.Quot.Global.zeroInit nullParams
            let z' = MLIRZipper.witnessOpWithResult nullText nullSSA Pointer z
            nullSSA, "!llvm.ptr", z'
        else
            bodySSA, bodyType, zipperWithBody
    
    // Add return instruction or panic terminator to end the function body
    let zipper1 =
        if bodySSA = "$panic" then
            MLIRZipper.witnessUnreachable zipperWithBody
        else
            let retParams = {| Value = returnSSA; Type = returnType |}
            let returnText = render LLVMTemplates.Quot.Control.retValue retParams
            MLIRZipper.witnessVoidOp returnText zipperForReturn
    
    // Exit function scope - this creates the MLIRFunc with all accumulated body ops
    let zipper2, func = MLIRZipper.exitFunction zipper1
    
    // Add the function to completed functions
    let zipper3 = MLIRZipper.addCompletedFunction func zipper2
    
    // After exitFunction, we're at module level - we can't emit addressof here
    // (MLIR doesn't allow operations at module level)
    // Instead, bind the function name as a marker for later use
    // The addressof will be emitted when the lambda is actually called
    let zipper4 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper3
    zipper4, TRValue ("@" + lambdaName, "!llvm.ptr")

/// Witness a Lambda node when NOT in function scope (fallback)
/// Creates function directly without captured body operations
let private witnessFallback
    (params': (string * NativeType) list)
    (bodyId: NodeId)
    (node: SemanticNode)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =
    
    let lambdaName, zipper1 = MLIRZipper.yieldLambdaName zipper
    
    let mlirParams = params' |> List.mapi (fun i (_name, nativeTy) ->
        (sprintf "arg%d" i, mapType nativeTy))
    
    let returnType = 
        match node.Type with
        | NativeType.TFun (_, retTy) -> mapType retTy
        | _ -> mapType node.Type
    
    // Look up body SSA
    let bodySSA, bodyType = 
        match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper1 with
        | Some (ssa, ty) -> ssa, ty
        | None -> 
            let undefSSA, _z = MLIRZipper.yieldSSA zipper1
            undefSSA, Serialize.mlirType returnType
    
    // Create return operation
    let retParams = {| Value = bodySSA; Type = bodyType |}
    let retOp: MLIROp = {
        Text = render LLVMTemplates.Quot.Control.retValue retParams
        Results = []
    }
    
    // Create function with just the return op (body ops weren't captured)
    let func: MLIRFunc = {
        Name = lambdaName
        Parameters = mlirParams
        ReturnType = returnType
        Blocks = [{
            Label = "entry"
            Arguments = []
            Operations = [retOp]
        }]
        Attributes = []
        IsInternal = true
    }
    
    let zipper2 = MLIRZipper.addCompletedFunction func zipper1
    
    // At module level - can't emit addressof here
    // Bind the function name as a marker for later use
    let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper2
    zipper3, TRValue ("@" + lambdaName, "!llvm.ptr")

/// Witness a Lambda node - main entry point
/// Handles both InFunction scope (body captured) and fallback (direct creation)
let witness
    (params': (string * NativeType) list)
    (bodyId: NodeId)
    (node: SemanticNode)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =
    
    // Check if we're in function scope (preBindLambdaParams should have entered it)
    match zipper.Focus with
    | InFunction funcName ->
        witnessInFunctionScope funcName node bodyId zipper
    | _ ->
        witnessFallback params' bodyId node zipper

// ═══════════════════════════════════════════════════════════════════════════
// Pre-order Lambda Parameter Binding
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-bind Lambda parameters to SSA names BEFORE body is processed
/// This is called in pre-order for Lambda nodes only
/// CRITICAL: Also enters function scope so body operations are captured
/// NOTE: For entry point (main), uses C-style signature: (argc: i32, argv: ptr) -> i32
let preBindParams (zipper: MLIRZipper) (node: SemanticNode) : MLIRZipper =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId) ->
        // Generate lambda function name - entry point Lambdas become "main"
        let nodeIdVal = NodeId.value node.Id
        let lambdaName, zipper1 = MLIRZipper.yieldLambdaNameForNode nodeIdVal zipper

        // For main, use C-style signature regardless of F# signature
        // C main: (int argc, char** argv) -> int
        // F# entry: (argv: string[]) -> int  OR  (unit) -> int
        let isMain = (lambdaName = "main")

        let mlirParams, paramBindings, needsArgvConversion =
            if isMain then
                // C-style main signature
                let cParams = [("arg0", Integer I32); ("arg1", Pointer)]
                // Check if F# parameter is an array of strings (needs conversion)
                let bindings, needsConv =
                    match params' with
                    | [(paramName, paramType)] ->
                        // Single F# parameter - check what kind it is
                        if paramName = "_" then
                            // Discarded parameter - no binding needed
                            // C main signature (argc, argv) is still used, just not bound to F#
                            [], false
                        else
                            match paramType with
                            | NativeType.TApp({ Name = "array" }, [NativeType.TApp({ Name = "string" }, [])]) ->
                                // F# `argv: string[]` - needs conversion from C char**
                                // Don't bind yet - we'll generate conversion code and bind after
                                [(paramName, "", "")], true
                            | _ ->
                                // Named parameter - bind to argv pointer (C main convention)
                                // Use C type (!llvm.ptr), not F# type (which may be unresolved)
                                [(paramName, "%arg1", "!llvm.ptr")], false
                    | [] ->
                        // Unit parameter - nothing to bind
                        [], false
                    | _ ->
                        // Multiple parameters (unusual for entry point) - bind to C args
                        params' |> List.mapi (fun i (name, _ty) ->
                            if name = "_" then (name, "", "")  // Skip discarded
                            else (name, sprintf "%%arg%d" (i + 1), "!llvm.ptr")), false
                cParams, bindings, needsConv
            else
                // Regular lambda - use node.Type for parameter types (has instantiated generics)
                let rec extractParamTypesFromFun (ty: NativeType) (count: int) : NativeType list =
                    if count <= 0 then []
                    else
                        match ty with
                        | NativeType.TFun(paramTy, resultTy) ->
                            paramTy :: extractParamTypesFromFun resultTy (count - 1)
                        | _ -> []

                let nodeParamTypes = extractParamTypesFromFun node.Type (List.length params')

                let mlirPs = params' |> List.mapi (fun i (_name, nativeTy) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else nativeTy
                    (sprintf "arg%d" i, mapType actualType))
                let bindings = params' |> List.mapi (fun i (paramName, paramType) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else paramType
                    (paramName, sprintf "%%arg%d" i, Serialize.mlirType (mapType actualType)))
                mlirPs, bindings, false

        // Return type: main always returns i32
        // For curried functions, we need to peel off ALL parameter layers to get the final return type
        let returnType =
            if isMain then Integer I32
            else
                // Extract final return type by peeling off one TFun layer per parameter
                // Special case: unit lambdas (params' = []) need to peel the unit->result layer
                let rec extractFinalReturnType (ty: NativeType) (paramCount: int) : NativeType =
                    match ty with
                    | NativeType.TFun(_, resultTy) when paramCount > 0 ->
                        extractFinalReturnType resultTy (paramCount - 1)
                    | NativeType.TFun(_, resultTy) when List.isEmpty params' ->
                        // Unit lambda: params' is empty but we need to peel the outer TFun
                        resultTy
                    | _ -> ty
                let finalRetTy = extractFinalReturnType node.Type (List.length params')
                mapType finalRetTy

        // Enter function scope - main is NOT internal (must be exported)
        let zipper2 =
            if isMain then
                MLIRZipper.enterFunctionWithVisibility lambdaName mlirParams returnType false zipper1
            else
                MLIRZipper.enterFunction lambdaName mlirParams returnType zipper1

        // For main with string[] argv, generate conversion prologue
        let zipper3 =
            if needsArgvConversion && not (List.isEmpty paramBindings) then
                let paramName = fst3 (List.head paramBindings)
                // Generate prologue to convert C argv to F# string array
                // We'll use %arg0 (argc) and %arg1 (argv) to build proper fat pointer strings
                generateArgvConversionPrologue paramName zipper2
            else
                // Standard bindings
                paramBindings
                |> List.fold (fun z (paramName, ssaName, mlirType) ->
                    if ssaName <> "" then MLIRZipper.bindVar paramName ssaName mlirType z
                    else z
                ) zipper2

        // Also bind the lambda name to the node for later retrieval
        MLIRZipper.bindNodeSSA (string (NodeId.value node.Id) + "_lambdaName") lambdaName "func" zipper3
    | _ -> zipper
