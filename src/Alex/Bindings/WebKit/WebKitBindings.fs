/// WebKitBindings - Platform bindings for WebKitGTK on Linux
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.WebKit.WebKitBindings

open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ===================================================================
// Type Constants
// ===================================================================

/// Fat string type: { ptr, i64 }
let fatStringType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

// ===================================================================
// Helper Functions
// ===================================================================

/// Create a Val from SSA and type
let inline val' ssa ty : Val = { SSA = ssa; Type = ty }

/// Extract pointer from fat string for C string interop
let extractStringPointer (z: PSGZipper) (strSSA: SSA) (ty: MLIRType) : SSA * MLIROp list =
    match ty with
    | TStruct _ ->
        let ptrSSA = freshSynthSSA z
        let op = MLIROp.LLVMOp (extractValueAt ptrSSA strSSA 0 fatStringType)
        ptrSSA, [op]
    | TPtr _ ->
        strSSA, []
    | _ ->
        strSSA, []

/// Convert integer to pointer if needed
let ensurePointer (z: PSGZipper) (ssa: SSA) (ty: MLIRType) : SSA * MLIROp list =
    match ty with
    | TPtr -> ssa, []
    | TInt _ ->
        let ptrSSA = freshSynthSSA z
        let op = MLIROp.LLVMOp (intToPtr ptrSSA ssa MLIRTypes.i64)
        ptrSSA, [op]
    | _ -> ssa, []

/// Create null pointer constant
let nullPointer (z: PSGZipper) : SSA * MLIROp list =
    let nullSSA = freshSynthSSA z
    let op = MLIROp.LLVMOp (NullPtr nullSSA)
    nullSSA, [op]

// ===================================================================
// WebKitGTK Platform Bindings
// ===================================================================

/// webViewNew - Create a new WebView widget
let bindWebViewNew (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let resultSSA = freshSynthSSA z
    let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "webkit_web_view_new" [] MLIRTypes.ptr)
    BoundOps ([callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })

/// webViewLoadUri - Load a URL
let bindWebViewLoadUri (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [webview; uri] ->
        let webviewPtr, webviewOps = ensurePointer z webview.SSA webview.Type
        let uriPtr, uriOps = extractStringPointer z uri.SSA uri.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webkit_web_view_load_uri"
            [val' webviewPtr MLIRTypes.ptr; val' uriPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (webviewOps @ uriOps @ [callOp], None)
    | _ ->
        NotSupported "webViewLoadUri requires (webview, uri)"

/// webViewLoadHtml - Load HTML content
let bindWebViewLoadHtml (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [webview; html; baseUri] ->
        let webviewPtr, webviewOps = ensurePointer z webview.SSA webview.Type
        let htmlPtr, htmlOps = extractStringPointer z html.SSA html.Type
        let baseUriPtr, baseUriOps = extractStringPointer z baseUri.SSA baseUri.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webkit_web_view_load_html"
            [val' webviewPtr MLIRTypes.ptr; val' htmlPtr MLIRTypes.ptr; val' baseUriPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (webviewOps @ htmlOps @ baseUriOps @ [callOp], None)
    | [webview; html] ->
        // Overload with no base URI - pass NULL
        let webviewPtr, webviewOps = ensurePointer z webview.SSA webview.Type
        let htmlPtr, htmlOps = extractStringPointer z html.SSA html.Type
        let nullSSA, nullOps = nullPointer z
        let callOp = MLIROp.LLVMOp (callFunc None "webkit_web_view_load_html"
            [val' webviewPtr MLIRTypes.ptr; val' htmlPtr MLIRTypes.ptr; val' nullSSA MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (webviewOps @ htmlOps @ nullOps @ [callOp], None)
    | _ ->
        NotSupported "webViewLoadHtml requires (webview, html[, baseUri])"

/// webViewRunJavaScript - Execute JavaScript
let bindWebViewRunJavaScript (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [webview; script] ->
        let webviewPtr, webviewOps = ensurePointer z webview.SSA webview.Type
        let scriptPtr, scriptOps = extractStringPointer z script.SSA script.Type
        // webkit_web_view_run_javascript(webview, script, NULL, NULL, NULL)
        // Last 3 args: cancellable, callback, user_data
        let nullSSA, nullOps = nullPointer z
        let callOp = MLIROp.LLVMOp (callFunc None "webkit_web_view_run_javascript"
            [val' webviewPtr MLIRTypes.ptr
             val' scriptPtr MLIRTypes.ptr
             val' nullSSA MLIRTypes.ptr   // cancellable
             val' nullSSA MLIRTypes.ptr   // callback
             val' nullSSA MLIRTypes.ptr]  // user_data
            MLIRTypes.unit)
        BoundOps (webviewOps @ scriptOps @ nullOps @ [callOp], None)
    | _ ->
        NotSupported "webViewRunJavaScript requires (webview, script)"

/// webViewGetSettings - Get WebView settings
let bindWebViewGetSettings (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [webview] ->
        let webviewPtr, webviewOps = ensurePointer z webview.SSA webview.Type
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "webkit_web_view_get_settings"
            [val' webviewPtr MLIRTypes.ptr] MLIRTypes.ptr)
        BoundOps (webviewOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })
    | _ ->
        NotSupported "webViewGetSettings requires (webview)"

/// settingsSetEnableDeveloperExtras - Enable dev tools
let bindSettingsSetEnableDeveloperExtras (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [settings; enable] ->
        let settingsPtr, settingsOps = ensurePointer z settings.SSA settings.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webkit_settings_set_enable_developer_extras"
            [val' settingsPtr MLIRTypes.ptr; val' enable.SSA MLIRTypes.i1] MLIRTypes.unit)
        BoundOps (settingsOps @ [callOp], None)
    | _ ->
        NotSupported "settingsSetEnableDeveloperExtras requires (settings, enable)"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register WebKit bindings for Linux x86_64
    PlatformDispatch.register Linux X86_64 "webViewNew" bindWebViewNew
    PlatformDispatch.register Linux X86_64 "webViewLoadUri" bindWebViewLoadUri
    PlatformDispatch.register Linux X86_64 "webViewLoadHtml" bindWebViewLoadHtml
    PlatformDispatch.register Linux X86_64 "webViewRunJavaScript" bindWebViewRunJavaScript
    PlatformDispatch.register Linux X86_64 "webViewGetSettings" bindWebViewGetSettings
    PlatformDispatch.register Linux X86_64 "settingsSetEnableDeveloperExtras" bindSettingsSetEnableDeveloperExtras

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "webViewNew" bindWebViewNew
    PlatformDispatch.register Linux ARM64 "webViewLoadUri" bindWebViewLoadUri
    PlatformDispatch.register Linux ARM64 "webViewLoadHtml" bindWebViewLoadHtml
    PlatformDispatch.register Linux ARM64 "webViewRunJavaScript" bindWebViewRunJavaScript
    PlatformDispatch.register Linux ARM64 "webViewGetSettings" bindWebViewGetSettings
    PlatformDispatch.register Linux ARM64 "settingsSetEnableDeveloperExtras" bindSettingsSetEnableDeveloperExtras
