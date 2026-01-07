/// WebViewBindings - Platform-specific webview bindings (witness-based)
///
/// ARCHITECTURAL FOUNDATION:
/// Uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
module Alex.Bindings.WebView.WebViewBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ===================================================================
// Helper Witness Functions
// ===================================================================

/// Extract pointer from fat string (struct<ptr, i64>) for C string interop
/// Fidelity strings are fat pointers; C APIs expect just the pointer
let private extractStringPointer (strSSA: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    match ty with
    | Struct _ ->
        // Fat string - extract the pointer (element 0)
        let ptrSSA, z = MLIRZipper.yieldSSA zipper
        let extractText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let z' = MLIRZipper.witnessOpWithResult extractText ptrSSA Pointer z
        ptrSSA, z'
    | NativeStrType ->
        // Fat string (NativeStr alias) - extract the pointer (element 0)
        let ptrSSA, z = MLIRZipper.yieldSSA zipper
        let extractText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let z' = MLIRZipper.witnessOpWithResult extractText ptrSSA Pointer z
        ptrSSA, z'
    | Pointer ->
        // Already a pointer (e.g., from NativePtr)
        strSSA, zipper
    | _ ->
        // Unknown type - pass through and hope for the best
        strSSA, zipper

/// Convert a value to pointer if needed (for nativeint args)
let private ensurePointer (ssa: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    match ty with
    | Pointer -> ssa, zipper
    | Integer _ ->
        let ssaName, z = MLIRZipper.yieldSSA zipper
        let convText = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" ssaName ssa
        let z' = MLIRZipper.witnessOpWithResult convText ssaName Pointer z
        ssaName, z'
    | _ -> ssa, zipper

/// Witness a call to a webview library function
/// Returns: (resultSSA, updatedZipper)
let witnessWebViewCall (funcName: string) (args: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // 1. Observe the external function requirement (coeffect)
    // We need to construct the signature for the extern declaration
    let argTypes = args |> List.map snd
    let argTypeStrs = argTypes |> List.map Serialize.mlirType
    let retTypeStr = Serialize.mlirType returnType
    let signature = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) retTypeStr
    
    let zipper1 = MLIRZipper.observeExternFunc funcName signature zipper

    // 2. Witness the call operation
    let argSSAs = args |> List.map fst
    // Use witnessCall which handles yielding SSA and emitting the op
    MLIRZipper.witnessCall funcName argSSAs argTypes returnType zipper1

// ===================================================================
// Platform Primitive Bindings (Witness-Based)
// ===================================================================

let witnessCreateWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(debugSSA, debugTy); (windowSSA, windowTy)] ->
        // Convert debug to i32 if needed
        let debug32SSA, zipper1 =
            match debugTy with
            | Integer I32 -> debugSSA, zipper
            | Integer I64 ->
                let ssaName, z = MLIRZipper.yieldSSA zipper
                let truncText = sprintf "%s = arith.trunci %s : i64 to i32" ssaName debugSSA
                let z' = MLIRZipper.witnessOpWithResult truncText ssaName (Integer I32) z
                ssaName, z'
            | _ -> debugSSA, zipper
        // Convert window nativeint to pointer if needed
        let windowPtrSSA, zipper2 =
            match windowTy with
            | Pointer -> windowSSA, zipper1
            | Integer _ ->
                let ssaName, z = MLIRZipper.yieldSSA zipper1
                let convText = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" ssaName windowSSA
                let z' = MLIRZipper.witnessOpWithResult convText ssaName Pointer z
                ssaName, z'
            | _ -> windowSSA, zipper1
        let args = [ (debug32SSA, Integer I32); (windowPtrSSA, Pointer) ]
        let resultSSA, zipper3 = witnessWebViewCall "webview_create" args Pointer zipper2
        zipper3, WitnessedValue (resultSSA, Pointer)
    | _ -> zipper, NotSupported "createWebview requires (debug: int, window: nativeint)"

let witnessDestroyWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let args = [ (wPtrSSA, Pointer) ]
        let _, zipper2 = witnessWebViewCall "webview_destroy" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "destroyWebview requires (webview: nativeint)"

let witnessRunWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let args = [ (wPtrSSA, Pointer) ]
        let _, zipper2 = witnessWebViewCall "webview_run" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "runWebview requires (webview: nativeint)"

let witnessTerminateWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let args = [ (wPtrSSA, Pointer) ]
        let _, zipper2 = witnessWebViewCall "webview_terminate" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "terminateWebview requires (webview: nativeint)"

let witnessSetWebViewTitle (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (titleSSA, titleTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let titlePtrSSA, zipper2 = extractStringPointer titleSSA titleTy zipper1
        let args = [ (wPtrSSA, Pointer); (titlePtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_set_title" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "setWebviewTitle requires (webview: nativeint, title: string)"

let witnessSetWebViewSize (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (widthSSA, _); (heightSSA, _); (hintsSSA, _)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let args = [
            (wPtrSSA, Pointer);
            (widthSSA, Integer I32);
            (heightSSA, Integer I32);
            (hintsSSA, Integer I32)
        ]
        let _, zipper2 = witnessWebViewCall "webview_set_size" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "setWebviewSize requires (webview, width, height, hints)"

let witnessNavigateWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (urlSSA, urlTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let urlPtrSSA, zipper2 = extractStringPointer urlSSA urlTy zipper1
        let args = [ (wPtrSSA, Pointer); (urlPtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_navigate" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "navigateWebview requires (webview, url)"

let witnessSetWebViewHtml (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (htmlSSA, htmlTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let htmlPtrSSA, zipper2 = extractStringPointer htmlSSA htmlTy zipper1
        let args = [ (wPtrSSA, Pointer); (htmlPtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_set_html" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "setWebviewHtml requires (webview, html)"

let witnessInitWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (jsSSA, jsTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let jsPtrSSA, zipper2 = extractStringPointer jsSSA jsTy zipper1
        let args = [ (wPtrSSA, Pointer); (jsPtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_init" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "initWebview requires (webview, js)"

let witnessEvalWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (jsSSA, jsTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let jsPtrSSA, zipper2 = extractStringPointer jsSSA jsTy zipper1
        let args = [ (wPtrSSA, Pointer); (jsPtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_eval" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "evalWebview requires (webview, js)"

let witnessBindWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (nameSSA, nameTy)] -> 
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let namePtrSSA, zipper2 = extractStringPointer nameSSA nameTy zipper1
        let args = [ (wPtrSSA, Pointer); (namePtrSSA, Pointer) ]
        let _, zipper3 = witnessWebViewCall "webview_bind" args Unit zipper2
        zipper3, WitnessedVoid
    | _ -> zipper, NotSupported "bindWebview arguments mismatch"

let witnessReturnWebView (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(wSSA, wTy); (idSSA, idTy); (statusSSA, _); (resultSSA, resultTy)] ->
        let wPtrSSA, zipper1 = ensurePointer wSSA wTy zipper
        let idPtrSSA, zipper2 = extractStringPointer idSSA idTy zipper1
        let resultPtrSSA, zipper3 = extractStringPointer resultSSA resultTy zipper2
        let args = [ 
            (wPtrSSA, Pointer); 
            (idPtrSSA, Pointer); 
            (statusSSA, Integer I32); 
            (resultPtrSSA, Pointer) 
        ]
        let _, zipper4 = witnessWebViewCall "webview_return" args Unit zipper3
        zipper4, WitnessedVoid
    | _ -> zipper, NotSupported "returnWebview requires (webview, id, status, result)"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register for Linux x86_64
    // We register against the base entry point name (e.g. "createWebview")
    // because CheckExpressions.fs extracts only the last part of the name.
    let reg binding name = 
        PlatformDispatch.register Linux X86_64 name binding
    
    reg witnessCreateWebView "createWebview"
    reg witnessDestroyWebView "destroyWebview"
    reg witnessRunWebView "runWebview"
    reg witnessTerminateWebView "terminateWebview"
    reg witnessSetWebViewTitle "setWebviewTitle"
    reg witnessSetWebViewSize "setWebviewSize"
    reg witnessNavigateWebView "navigateWebview"
    reg witnessSetWebViewHtml "setWebviewHtml"
    reg witnessInitWebView "initWebview"
    reg witnessEvalWebView "evalWebview"
    reg witnessBindWebView "bindWebview"
    reg witnessReturnWebView "returnWebview"
