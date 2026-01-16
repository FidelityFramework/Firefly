# PRD-26: WebKitGTK WebView

> **Sample**: `26_WebViewBasic` | **Status**: Planned | **Depends On**: PRD-25 (GTKWindow)

## 1. Executive Summary

This PRD adds WebKitGTK WebView to GTK windows - enabling hybrid web/native applications. This is the foundation for the WREN (WebView + Region + Elmish + Native) stack vision.

**Key Insight**: WebKitGTK provides a C API similar to GTK. The same FFI patterns from PRD-25 apply. JavaScript<->F# bridging is future work (PRD beyond scope).

## 2. Language Feature Specification

### 2.1 WebView Creation

```fsharp
[<DllImport("libwebkitgtk-6.0.so.4")>]
extern nativeptr<byte> webkit_web_view_new()

let webview = webkit_web_view_new()
```

### 2.2 Loading Content

```fsharp
// Load URL
[<DllImport("libwebkitgtk-6.0.so.4")>]
extern void webkit_web_view_load_uri(view: nativeptr<byte>, uri: string)

webkit_web_view_load_uri(webview, "https://fsharp.org")

// Load HTML
[<DllImport("libwebkitgtk-6.0.so.4")>]
extern void webkit_web_view_load_html(view: nativeptr<byte>, html: string, baseUri: string)

webkit_web_view_load_html(webview, "<h1>Hello WREN!</h1>", "about:blank")
```

### 2.3 WebView in GTK Window

```fsharp
let window = gtk_window_new()
let webview = webkit_web_view_new()
gtk_window_set_child(window, webview)
webkit_web_view_load_html(webview, html, "about:blank")
gtk_widget_show(window)
```

## 3. FNCS Layer Implementation

### 3.1 WebKit Extern Declarations

Same pattern as GTK - extern functions with DllImport:

```fsharp
[<DllImport("libwebkitgtk-6.0.so.4")>]
extern nativeptr<byte> webkit_web_view_new()

[<DllImport("libwebkitgtk-6.0.so.4")>]
extern void webkit_web_view_load_uri(view: nativeptr<byte>, uri: string)

[<DllImport("libwebkitgtk-6.0.so.4")>]
extern void webkit_web_view_load_html(view: nativeptr<byte>, content: string, baseUri: string)
```

No new FNCS machinery needed beyond PRD-25.

## 4. Firefly/Alex Layer Implementation

### 4.1 Same FFI Patterns

WebKit uses the same patterns as GTK:
- External function declarations
- Opaque pointer types
- String parameters

No new Alex machinery needed beyond PRD-25.

## 5. MLIR Output Specification

### 5.1 WebKit Function Declarations

```mlir
llvm.func @webkit_web_view_new() -> !llvm.ptr
    attributes { "link" = "webkitgtk-6.0" }

llvm.func @webkit_web_view_load_uri(!llvm.ptr, !llvm.ptr)
    attributes { "link" = "webkitgtk-6.0" }

llvm.func @webkit_web_view_load_html(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    attributes { "link" = "webkitgtk-6.0" }
```

### 5.2 WebView Creation and Loading

```mlir
// let webview = webkit_web_view_new()
%webview = llvm.call @webkit_web_view_new() : () -> !llvm.ptr

// webkit_web_view_load_html(webview, html, "about:blank")
%html = llvm.mlir.addressof @html_content : !llvm.ptr
%base = llvm.mlir.addressof @about_blank : !llvm.ptr
llvm.call @webkit_web_view_load_html(%webview, %html, %base)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module WebViewBasicSample

// GTK imports (from PRD-25)
[<DllImport("libgtk-4.so.1")>]
extern void gtk_init()

[<DllImport("libgtk-4.so.1")>]
extern nativeptr<byte> gtk_window_new()

[<DllImport("libgtk-4.so.1")>]
extern void gtk_window_set_title(window: nativeptr<byte>, title: string)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_window_set_default_size(window: nativeptr<byte>, width: int, height: int)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_window_set_child(window: nativeptr<byte>, child: nativeptr<byte>)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_widget_show(widget: nativeptr<byte>)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_main()

// WebKit imports
[<DllImport("libwebkitgtk-6.0.so.4")>]
extern nativeptr<byte> webkit_web_view_new()

[<DllImport("libwebkitgtk-6.0.so.4")>]
extern void webkit_web_view_load_html(view: nativeptr<byte>, content: string, baseUri: string)

let htmlContent = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: system-ui;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        h1 { font-size: 3em; }
    </style>
</head>
<body>
    <h1>Hello WREN Stack!</h1>
</body>
</html>
"""

[<EntryPoint>]
let main _ =
    Console.writeln "=== WebView Basic Test ==="

    gtk_init()

    let window = gtk_window_new()
    gtk_window_set_title(window, "WREN WebView")
    gtk_window_set_default_size(window, 800, 600)

    let webview = webkit_web_view_new()
    webkit_web_view_load_html(webview, htmlContent, "about:blank")

    gtk_window_set_child(window, webview)
    gtk_widget_show(window)

    Console.writeln "WebView shown, entering main loop..."
    gtk_main()

    0
```

### 6.2 Expected Behavior

- Window appears titled "WREN WebView"
- WebView displays styled "Hello WREN Stack!" message
- Closing window exits program

## 7. Files to Create/Modify

No new files beyond PRD-25. Sample code demonstrates WebKit usage.

## 8. Implementation Checklist

### Phase 1: Verify PRD-25 Complete
- [ ] GTK FFI working
- [ ] External calls working
- [ ] Library linking working

### Phase 2: WebKit Integration
- [ ] Add WebKit library linking
- [ ] Verify WebView creates
- [ ] Verify HTML loading

### Phase 3: Validation
- [ ] Sample 26 compiles
- [ ] WebView displays content
- [ ] Window closes correctly

## 9. Future: JavaScript Bridge

Full WREN stack needs JavaScript<->F# communication:

```fsharp
// Future API
webkit_web_view_run_javascript(webview, "document.title = 'Updated'")

// Callback from JS to F#
webkit_web_view_register_script_message_handler(webview, "fsharp", onMessage)
```

This is beyond the scope of Sample 26 but establishes the foundation.

## 10. Linking

```bash
# Link against both GTK and WebKit
gcc app.o -o app $(pkg-config --libs gtk4 webkitgtk-6.0)
```

## 11. Related PRDs

- **PRD-25**: GTKWindow - GTK foundation
- **PRD-29-31**: MailboxProcessor - UI event handling
- (Future): Elmish architecture, JS bridge
