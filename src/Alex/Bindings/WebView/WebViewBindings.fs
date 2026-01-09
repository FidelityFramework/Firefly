/// WebViewBindings - Platform-specific webview bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.WebView.WebViewBindings

open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
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

/// Extract pointer from fat string (struct<ptr, i64>) for C string interop
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

/// Convert integer to pointer if needed (for nativeint args)
let ensurePointer (z: PSGZipper) (ssa: SSA) (ty: MLIRType) : SSA * MLIROp list =
    match ty with
    | TPtr _ -> ssa, []
    | TInt _ ->
        let ptrSSA = freshSynthSSA z
        let op = MLIROp.LLVMOp (intToPtr ptrSSA ssa MLIRTypes.i64)
        ptrSSA, [op]
    | _ -> ssa, []

/// Truncate i64 to i32 if needed
let truncToI32 (z: PSGZipper) (ssa: SSA) (ty: MLIRType) : SSA * MLIROp list =
    match ty with
    | TInt I32 -> ssa, []
    | TInt I64 ->
        let i32SSA = freshSynthSSA z
        let op = MLIROp.ArithOp (truncI i32SSA ssa MLIRTypes.i64 MLIRTypes.i32)
        i32SSA, [op]
    | TInt _ ->
        let i32SSA = freshSynthSSA z
        let op = MLIROp.ArithOp (truncI i32SSA ssa ty MLIRTypes.i32)
        i32SSA, [op]
    | _ -> ssa, []

// ===================================================================
// Platform Primitive Bindings
// ===================================================================

/// createWebview(debug: int, window: nativeint) -> nativeint
let bindCreateWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [debug; window] ->
        let debug32, debugOps = truncToI32 z debug.SSA debug.Type
        let windowPtr, windowOps = ensurePointer z window.SSA window.Type
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "webview_create"
            [val' debug32 MLIRTypes.i32; val' windowPtr MLIRTypes.ptr] MLIRTypes.ptr)
        let allOps = debugOps @ windowOps @ [callOp]
        BoundOps (allOps, Some { SSA = resultSSA; Type = MLIRTypes.ptr })
    | _ ->
        NotSupported "createWebview requires (debug: int, window: nativeint)"

/// destroyWebview(webview: nativeint) -> unit
let bindDestroyWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_destroy"
            [val' wPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ [callOp], None)
    | _ ->
        NotSupported "destroyWebview requires (webview: nativeint)"

/// runWebview(webview: nativeint) -> unit
let bindRunWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_run"
            [val' wPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ [callOp], None)
    | _ ->
        NotSupported "runWebview requires (webview: nativeint)"

/// terminateWebview(webview: nativeint) -> unit
let bindTerminateWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_terminate"
            [val' wPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ [callOp], None)
    | _ ->
        NotSupported "terminateWebview requires (webview: nativeint)"

/// setWebviewTitle(webview: nativeint, title: string) -> unit
let bindSetWebViewTitle (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; title] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let titlePtr, titleOps = extractStringPointer z title.SSA title.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_set_title"
            [val' wPtr MLIRTypes.ptr; val' titlePtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ titleOps @ [callOp], None)
    | _ ->
        NotSupported "setWebviewTitle requires (webview: nativeint, title: string)"

/// setWebviewSize(webview: nativeint, width: int, height: int, hints: int) -> unit
let bindSetWebViewSize (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; width; height; hints] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let width32, wOps = truncToI32 z width.SSA width.Type
        let height32, hOps = truncToI32 z height.SSA height.Type
        let hints32, hintsOps = truncToI32 z hints.SSA hints.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_set_size"
            [val' wPtr MLIRTypes.ptr; val' width32 MLIRTypes.i32; val' height32 MLIRTypes.i32; val' hints32 MLIRTypes.i32] MLIRTypes.unit)
        BoundOps (ptrOps @ wOps @ hOps @ hintsOps @ [callOp], None)
    | _ ->
        NotSupported "setWebviewSize requires (webview, width, height, hints)"

/// navigateWebview(webview: nativeint, url: string) -> unit
let bindNavigateWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; url] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let urlPtr, urlOps = extractStringPointer z url.SSA url.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_navigate"
            [val' wPtr MLIRTypes.ptr; val' urlPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ urlOps @ [callOp], None)
    | _ ->
        NotSupported "navigateWebview requires (webview, url)"

/// setWebviewHtml(webview: nativeint, html: string) -> unit
let bindSetWebViewHtml (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; html] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let htmlPtr, htmlOps = extractStringPointer z html.SSA html.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_set_html"
            [val' wPtr MLIRTypes.ptr; val' htmlPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ htmlOps @ [callOp], None)
    | _ ->
        NotSupported "setWebviewHtml requires (webview, html)"

/// initWebview(webview: nativeint, js: string) -> unit
let bindInitWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; js] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let jsPtr, jsOps = extractStringPointer z js.SSA js.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_init"
            [val' wPtr MLIRTypes.ptr; val' jsPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ jsOps @ [callOp], None)
    | _ ->
        NotSupported "initWebview requires (webview, js)"

/// evalWebview(webview: nativeint, js: string) -> unit
let bindEvalWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; js] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let jsPtr, jsOps = extractStringPointer z js.SSA js.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_eval"
            [val' wPtr MLIRTypes.ptr; val' jsPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ jsOps @ [callOp], None)
    | _ ->
        NotSupported "evalWebview requires (webview, js)"

/// bindWebview(webview: nativeint, name: string) -> unit
let bindBindWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; name] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let namePtr, nameOps = extractStringPointer z name.SSA name.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_bind"
            [val' wPtr MLIRTypes.ptr; val' namePtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ nameOps @ [callOp], None)
    | _ ->
        NotSupported "bindWebview arguments mismatch"

/// returnWebview(webview: nativeint, id: string, status: int, result: string) -> unit
let bindReturnWebView (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [w; id; status; result] ->
        let wPtr, ptrOps = ensurePointer z w.SSA w.Type
        let idPtr, idOps = extractStringPointer z id.SSA id.Type
        let status32, statusOps = truncToI32 z status.SSA status.Type
        let resultPtr, resultOps = extractStringPointer z result.SSA result.Type
        let callOp = MLIROp.LLVMOp (callFunc None "webview_return"
            [val' wPtr MLIRTypes.ptr; val' idPtr MLIRTypes.ptr; val' status32 MLIRTypes.i32; val' resultPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (ptrOps @ idOps @ statusOps @ resultOps @ [callOp], None)
    | _ ->
        NotSupported "returnWebview requires (webview, id, status, result)"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    let register name binding =
        PlatformDispatch.register Linux X86_64 name binding

    register "createWebview" bindCreateWebView
    register "destroyWebview" bindDestroyWebView
    register "runWebview" bindRunWebView
    register "terminateWebview" bindTerminateWebView
    register "setWebviewTitle" bindSetWebViewTitle
    register "setWebviewSize" bindSetWebViewSize
    register "navigateWebview" bindNavigateWebView
    register "setWebviewHtml" bindSetWebViewHtml
    register "initWebview" bindInitWebView
    register "evalWebview" bindEvalWebView
    register "bindWebview" bindBindWebView
    register "returnWebview" bindReturnWebView
