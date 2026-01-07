/// WebKitBindings - Platform bindings for WebKitGTK on Linux
///
/// ARCHITECTURAL FOUNDATION:
/// WebKitGTK is dynamically linked at load time - the system's
/// libwebkit2gtk is already installed on Linux desktops.
module Alex.Bindings.WebKit.WebKitBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ===================================================================
// Helper Functions
// ===================================================================

/// Extract pointer from fat string for C string interop
let private extractStringPointer (strSSA: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    match ty with
    | Struct _ ->
        let ptrSSA, z = MLIRZipper.yieldSSA zipper
        let extractText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let z' = MLIRZipper.witnessOpWithResult extractText ptrSSA Pointer z
        ptrSSA, z'
    | Pointer -> strSSA, zipper
    | _ -> strSSA, zipper

/// Witness a call to a WebKit function
let witnessWebKitCall (funcName: string) (args: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    let argTypes = args |> List.map snd
    let argTypeStrs = argTypes |> List.map Serialize.mlirType
    let retTypeStr = Serialize.mlirType returnType
    let signature = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) retTypeStr

    let zipper1 = MLIRZipper.observeExternFunc funcName signature zipper
    let argSSAs = args |> List.map fst
    MLIRZipper.witnessCall funcName argSSAs argTypes returnType zipper1

// ===================================================================
// WebKitGTK Platform Bindings
// ===================================================================

/// webkit_web_view_new - Create a new WebView widget
let witnessWebViewNew (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    let resultSSA, zipper1 = witnessWebKitCall "webkit_web_view_new" [] Pointer zipper
    zipper1, WitnessedValue (resultSSA, Pointer)

/// webkit_web_view_load_uri - Load a URL
let witnessWebViewLoadUri (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(webviewSSA, _); (uriSSA, uriTy)] ->
        let uriPtrSSA, zipper1 = extractStringPointer uriSSA uriTy zipper
        let args = [ (webviewSSA, Pointer); (uriPtrSSA, Pointer) ]
        let _, zipper2 = witnessWebKitCall "webkit_web_view_load_uri" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "webkit_web_view_load_uri requires (webview, uri)"

/// webkit_web_view_load_html - Load HTML content
let witnessWebViewLoadHtml (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(webviewSSA, _); (htmlSSA, htmlTy); (baseUriSSA, baseUriTy)] ->
        let htmlPtrSSA, zipper1 = extractStringPointer htmlSSA htmlTy zipper
        let baseUriPtrSSA, zipper2 = extractStringPointer baseUriSSA baseUriTy zipper1
        let args = [ (webviewSSA, Pointer); (htmlPtrSSA, Pointer); (baseUriPtrSSA, Pointer) ]
        let _, zipper3 = witnessWebKitCall "webkit_web_view_load_html" args Unit zipper2
        zipper3, WitnessedVoid
    | [(webviewSSA, _); (htmlSSA, htmlTy)] ->
        // Overload with no base URI - pass NULL
        let htmlPtrSSA, zipper1 = extractStringPointer htmlSSA htmlTy zipper
        let nullSSA, zipper2 = MLIRZipper.witnessConstant 0L I64 zipper1
        let nullPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let convText = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" nullPtrSSA nullSSA
        let zipper4 = MLIRZipper.witnessOpWithResult convText nullPtrSSA Pointer zipper3
        let args = [ (webviewSSA, Pointer); (htmlPtrSSA, Pointer); (nullPtrSSA, Pointer) ]
        let _, zipper5 = witnessWebKitCall "webkit_web_view_load_html" args Unit zipper4
        zipper5, WitnessedVoid
    | _ -> zipper, NotSupported "webkit_web_view_load_html requires (webview, html[, baseUri])"

/// webkit_web_view_run_javascript - Execute JavaScript
let witnessWebViewRunJavaScript (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(webviewSSA, _); (scriptSSA, scriptTy)] ->
        let scriptPtrSSA, zipper1 = extractStringPointer scriptSSA scriptTy zipper
        // webkit_web_view_run_javascript(webview, script, NULL, NULL, NULL)
        // The last 3 args are: cancellable, callback, user_data
        let nullSSA, zipper2 = MLIRZipper.witnessConstant 0L I64 zipper1
        let nullPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let convText = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" nullPtrSSA nullSSA
        let zipper4 = MLIRZipper.witnessOpWithResult convText nullPtrSSA Pointer zipper3
        let args = [
            (webviewSSA, Pointer)
            (scriptPtrSSA, Pointer)
            (nullPtrSSA, Pointer)  // cancellable
            (nullPtrSSA, Pointer)  // callback
            (nullPtrSSA, Pointer)  // user_data
        ]
        let _, zipper5 = witnessWebKitCall "webkit_web_view_run_javascript" args Unit zipper4
        zipper5, WitnessedVoid
    | _ -> zipper, NotSupported "webkit_web_view_run_javascript requires (webview, script)"

/// webkit_web_view_get_settings - Get WebView settings
let witnessWebViewGetSettings (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(webviewSSA, _)] ->
        let args = [ (webviewSSA, Pointer) ]
        let resultSSA, zipper1 = witnessWebKitCall "webkit_web_view_get_settings" args Pointer zipper
        zipper1, WitnessedValue (resultSSA, Pointer)
    | _ -> zipper, NotSupported "webkit_web_view_get_settings requires (webview)"

/// webkit_settings_set_enable_developer_extras - Enable dev tools
let witnessSettingsSetEnableDeveloperExtras (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(settingsSSA, _); (enableSSA, _)] ->
        let args = [ (settingsSSA, Pointer); (enableSSA, Integer I1) ]
        let _, zipper1 = witnessWebKitCall "webkit_settings_set_enable_developer_extras" args Unit zipper
        zipper1, WitnessedVoid
    | _ -> zipper, NotSupported "webkit_settings_set_enable_developer_extras requires (settings, enable)"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register WebKit bindings for Linux
    PlatformDispatch.register Linux X86_64 "webViewNew" witnessWebViewNew
    PlatformDispatch.register Linux X86_64 "webViewLoadUri" witnessWebViewLoadUri
    PlatformDispatch.register Linux X86_64 "webViewLoadHtml" witnessWebViewLoadHtml
    PlatformDispatch.register Linux X86_64 "webViewRunJavaScript" witnessWebViewRunJavaScript
    PlatformDispatch.register Linux X86_64 "webViewGetSettings" witnessWebViewGetSettings
    PlatformDispatch.register Linux X86_64 "settingsSetEnableDeveloperExtras" witnessSettingsSetEnableDeveloperExtras
