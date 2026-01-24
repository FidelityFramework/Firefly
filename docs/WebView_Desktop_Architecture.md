# WebView-Based Desktop UI Architecture

> **Working Title**: Fidelity WebView-Based Desktop UI
>
> **Status**: Architectural design phase
>
> **Related Documents**:
> - [Architecture_Canonical.md](./Architecture_Canonical.md) - Layer separation principle
> - [Demo_UI_Stretch_Goal.md](./Demo_UI_Stretch_Goal.md) - FidelityUI long-term vision

---

## Executive Summary

This document describes a desktop application framework built on the Fidelity ecosystem:

- **Single native executable** with embedded web UI
- **Partas.Solid** component model compiled to JavaScript via Fable
- **Firefly** orchestrates the entire build pipeline
- **Platform-agnostic Fidelity.Platform bindings**, platform-specific Alex implementations
- **System webview** (WebKitGTK/WebView2/WKWebView) renders the UI

The architecture follows the same model as SAFE Stack for web applications, but targets native desktop instead of web servers.

---

## For Web Developers: What This Means For You

If you're coming from a web development background with HTML, CSS, and JavaScript experience, here's how this framework fits into what you already know.

### The Webview IS a Browser

The desktop application's UI runs in an actual browser engine:
- **Linux**: WebKitGTK (same engine as Safari, GNOME Web)
- **Windows**: WebView2 (Microsoft Edge/Chromium)
- **macOS**: WKWebView (Safari's engine)

Your CSS works. Your JavaScript works. Your DOM knowledge applies directly. The difference is that instead of a web server providing the HTML, the native application embeds it directly.

### SolidJS, Not React

The UI layer uses [SolidJS](https://www.solidjs.com/), a reactive JavaScript framework. If you know React, here's the key difference:

| React | SolidJS |
|-------|---------|
| Virtual DOM diffing | Fine-grained reactivity |
| Components re-render entirely | Only changed DOM nodes update |
| `useState` returns `[value, setter]` | `createSignal` returns `[getter, setter]` |
| JSX compiles to `React.createElement` | JSX compiles to real DOM operations |

SolidJS is faster because it doesn't diff - it knows exactly what changed and updates only that.

### Partas.Solid: F# Syntax for SolidJS

Instead of writing JSX in JavaScript:

```jsx
// JavaScript/SolidJS
function Button(props) {
    return <button class={props.class} onClick={props.onClick}>
        {props.children}
    </button>;
}
```

You write the equivalent in F# using Partas.Solid:

```fsharp
// F#/Partas.Solid
[<SolidTypeComponent>]
type Button() =
    inherit button()

    [<Erase>]
    member val onClick: (unit -> unit) = jsNative with get, set

    member props.constructor =
        button(class' = props.class, onClick = fun _ -> props.onClick()) {
            props.children
        }
```

Fable compiles this F# code to the JavaScript you'd write by hand. The output is idiomatic SolidJS.

### The "Backend" Is Native Code

In a traditional web app, your frontend talks to a server over HTTP. In this architecture:

- There is no server
- The "backend" is compiled native code (via Firefly)
- Frontend and backend run in the **same process**
- Communication uses direct function binding, not HTTP

```
Traditional Web:
  Browser ──HTTP──► Server (Node.js, .NET, etc.)

This Architecture:
  WebView ──bind()──► Native Code (same process)
```

---

## For .NET Developers: F# Deep Features Explained

If you're a C# or VB.NET developer encountering F# for the first time, some features may seem unfamiliar. This section explains the three key language features that power the Fidelity architecture.

### Quotations: Code as Data

In C#, you may have used `Expression<Func<T>>` with LINQ:

```csharp
// C# - Expression trees
Expression<Func<int, bool>> isEven = x => x % 2 == 0;
// The lambda is captured as a data structure, not compiled to IL
```

F# quotations are similar but more powerful:

```fsharp
// F# - Quotations
let isEven = <@ fun x -> x % 2 = 0 @>
// Captured as Expr<int -> bool> - inspectable at compile time
```

**Why this matters for Fidelity:**

Quotations allow the compiler to inspect code structure at compile time. When Farscape generates hardware bindings, it uses quotations to carry semantic information:

```fsharp
let gpioDescriptor: Expr<PeripheralDescriptor> = <@
    { Name = "GPIO"
      BaseAddress = 0x48000000un
      MemoryRegion = Peripheral }
@>
```

The Firefly compiler inspects this quotation during PSG construction, extracting the memory region classification. This information flows through the pipeline, informing Alex that volatile loads are required for peripheral access.

Unlike runtime reflection, quotations are **compile-time artifacts**. They require no runtime support, impose no overhead, and introduce no BCL dependencies.

### Active Patterns: Beyond Pattern Matching

C# has pattern matching with `switch` expressions:

```csharp
// C# - Pattern matching
var result = shape switch
{
    Circle c => c.Radius * c.Radius * Math.PI,
    Rectangle r => r.Width * r.Height,
    _ => 0
};
```

F# has similar `match` expressions, but active patterns add a crucial capability: **compositional matching**.

```fsharp
// F# - Active pattern definition
let (|PeripheralAccess|_|) (node: PSGNode) =
    match node with
    | CallToExtern name args when isPeripheralBinding name ->
        Some (extractPeripheralInfo args)
    | _ -> None

// Usage - composes with other patterns
match currentNode with
| PeripheralAccess info -> emitVolatileAccess info
| SRTPDispatch srtp -> emitResolvedCall srtp
| _ -> emitDefault node
```

**Why this matters for Fidelity:**

Active patterns are how the typed tree zipper and Alex traversal recognize PSG structures. They:
- **Compose** with `&` (and) and `|` (or)
- **Encapsulate** recognition logic
- **Decouple** pattern definition from usage

Compare to the visitor pattern in C#, which spreads classification across multiple methods. Active patterns keep the structure visible and the logic local.

### Computation Expressions: Continuations Made Easy

If you've used `async`/`await` in C#:

```csharp
// C# - async/await
async Task<int> ProcessAsync()
{
    var data = await FetchDataAsync();
    var result = await ComputeAsync(data);
    return result;
}
```

F# has computation expressions, which are more general:

```fsharp
// F# - async computation expression
let processAsync = async {
    let! data = fetchDataAsync()
    let! result = computeAsync data
    return result
}
```

**The key insight**: Every `let!` is syntactic sugar for continuation capture.

```fsharp
// What async { let! x = someAsync; ... } actually means:
builder.Bind(someAsync, fun x ->
    // Everything after let! is captured as a continuation
    ...
)
```

The nested lambdas are continuations. This observation has profound implications for native compilation.

**Why this matters for Fidelity:**

Computation expressions already express the control flow patterns that the DCont (delimited continuations) dialect needs:

| Pattern | Compilation Target |
|---------|-------------------|
| Sequential effects (`async`, state) | DCont dialect - preserves continuations |
| Parallel pure (`validated`, reader) | Inet dialect - compiles to data flow |
| Mixed | Both dialects with analysis-driven splitting |

The MLIR builder itself is a computation expression:

```fsharp
let emitFunction (node: PSGNode) : MLIR<Val> = mlir {
    let! funcType = deriveType node
    let! entry = createBlock "entry"
    do! setInsertionPoint entry
    let! result = emitBody node.Body
    do! emitReturn result
    return result
}
```

The compiler's internal structure mirrors the patterns it compiles.

---

## For SolidJS Developers: Partas.Solid Patterns

If you're familiar with SolidJS, Partas.Solid provides the same reactive primitives with F# syntax.

### Component Definitions

The `[<SolidTypeComponent>]` attribute marks a type member as a component constructor:

```fsharp
[<Erase>]
type Counter() =
    inherit div()

    [<Erase>]
    member val initialCount: int = 0 with get, set

    [<SolidTypeComponent>]
    member props.constructor =
        let count, setCount = createSignal props.initialCount

        div(class' = "counter") {
            button(onClick = fun _ -> setCount (count() - 1)) { "-" }
            span() { count() }
            button(onClick = fun _ -> setCount (count() + 1)) { "+" }
        }
```

This compiles to standard SolidJS:

```javascript
export function Counter(props) {
    props = mergeProps({ initialCount: 0 }, props);
    const [count, setCount] = createSignal(props.initialCount);

    return <div class="counter">
        <button onClick={() => setCount(count() - 1)}>-</button>
        <span>{count()}</span>
        <button onClick={() => setCount(count() + 1)}>+</button>
    </div>;
}
```

### Reactive Primitives

The same SolidJS primitives are available:

| SolidJS | Partas.Solid |
|---------|--------------|
| `createSignal(value)` | `createSignal value` |
| `createEffect(() => ...)` | `createEffect (fun () -> ...)` |
| `createMemo(() => ...)` | `createMemo (fun () -> ...)` |
| `<Show when={...}>` | `Show(when' = ...) { ... }` |
| `<For each={...}>` | `For(each = ...) { ... }` |

### Inspecting the Output

The Fable compilation produces JavaScript files. You can inspect them to verify the output matches your expectations. The Partas.Solid plugin is aggressive about producing idiomatic JS - the output should look like what you'd write by hand.

---

## The Stack Model

This architecture follows the same conceptual model as other F# full-stack solutions:

| Stack | Frontend | Backend | Transport |
|-------|----------|---------|-----------|
| **SAFE** | Fable/React | Saturn/.NET | HTTP |
| **SPEC** | Partas.Solid | CloudflareFS | HTTP/Workers |
| **This Stack** | Partas.Solid | Firefly/Native | webview_bind |

The key difference: this stack targets **native desktop** with the frontend and backend in the **same process**.

### Comparison to Electron

| Aspect | Electron | This Stack |
|--------|----------|------------|
| Runtime | Bundled Chromium (~150MB) | System webview (~0MB added) |
| Backend | Node.js (interpreted) | Native code (compiled) |
| Binary size | 150-300MB | 5-20MB |
| Startup | Slow (load Chromium) | Fast (native + load HTML) |
| Memory | High (Chromium overhead) | Low (shared system webview) |

This is **not** Electron. The browser engine is the system's webview, not bundled. The backend is compiled native code, not interpreted JavaScript.

---

## The Split Model (Critical Architecture)

The Fidelity framework enforces strict layer separation between platform-agnostic and platform-specific code.

### Fidelity.Platform: Platform-Agnostic Bindings

Fidelity.Platform defines the **interface** for platform bindings without any platform-specific code:

```fsharp
// Fidelity.Platform/Platform.fs
module Platform.Bindings =
    // Alex provides implementation for each platform
    let createWebview (debug: int) (window: nativeint) : nativeint =
        Unchecked.defaultof<nativeint>

    let setWebviewHtml (webview: nativeint) (html: nativeint) : int =
        Unchecked.defaultof<int>

    // ... etc
```

The `Unchecked.defaultof<T>` is a **conduit marker** - it signals that Alex provides the actual implementation.

**There is NO platform-specific code in Fidelity.Platform:**
- No `#if LINUX`
- No `DllImport` (that's BCL!)
- No Linux/, Windows/, macOS/ directories

### Alex: Platform-Specific MLIR Emission

Alex makes all platform decisions during code generation:

```fsharp
// Alex/Bindings/Webview/WebviewBindings.fs
let bindCreateWebview (platform: TargetPlatform) (prim: PlatformPrimitive) = mlir {
    match platform.OS with
    | Linux ->
        // WebKitGTK: call webkit_web_view_new()
        let! result = llvm.call "@webview_create" [...]
        return Emitted result
    | Windows ->
        // WebView2: CreateCoreWebView2Controller()
        let! result = llvm.call "@webview_create" [...]
        return Emitted result
    | MacOS ->
        // WKWebView via Objective-C runtime
        let! result = llvm.call "@webview_create" [...]
        return Emitted result
}
```

The same Fidelity.Platform code compiles to different platform-specific MLIR depending on the `--target` flag.

### Why This Matters

This separation enables:
- **Cross-compilation**: Build for any platform from any platform
- **Single codebase**: No conditional compilation in application code
- **Type safety**: Compiler verifies the interface, Alex provides implementation
- **Extensibility**: Add new platforms without touching Fidelity.Platform

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single Native Executable                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Embedded Frontend                │  Native Backend             │
│  ┌────────────────────────┐       │  ┌────────────────────────┐ │
│  │                        │       │  │                        │ │
│  │  Partas.Solid          │       │  │  Application Logic     │ │
│  │  (F# → Fable → JS)     │       │  │  (F# → Firefly → MLIR) │ │
│  │                        │       │  │                        │ │
│  │  SolidJS Reactivity    │◄─────►│  │  Platform.Bindings     │ │
│  │  (fine-grained DOM)    │  IPC  │  │  (webview conduits)    │ │
│  │                        │       │  │                        │ │
│  │  Vite Bundle           │       │  │  Alex → LLVM → Binary  │ │
│  │  (HTML + JS + CSS)     │       │  │                        │ │
│  └────────────────────────┘       │  └────────────────────────┘ │
│                                   │                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                   │                              │
│  WebView Runtime                  │  System Library (dynamic)   │
│  ┌────────────────────────┐       │  ┌────────────────────────┐ │
│  │ Linux: WebKitGTK       │       │  │ libwebkit2gtk-4.0.so   │ │
│  │ Windows: WebView2      │───────┼──│ WebView2Loader.dll     │ │
│  │ macOS: WKWebView       │       │  │ WebKit.framework       │ │
│  └────────────────────────┘       │  └────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Build time**: Firefly orchestrates Fable → Vite → native compilation
2. **Embedding**: Bundled HTML/JS/CSS becomes a string constant in the binary
3. **Runtime**: Native code creates webview, loads embedded HTML via `setHtml()`
4. **Interaction**: JS calls native functions via `webview_bind`, native responds via `webview_return`

---

## Build Pipeline

Firefly acts as the unified build orchestrator, similar to how `dotnet` coordinates builds in the SAFE Stack:

```
firefly build MyApp.fidproj
    │
    ├─► Phase 1: Frontend Compilation
    │   │
    │   ├─► dotnet fable src/Frontend -o build/fable
    │   │   (Partas.Solid F# → SolidJS JavaScript)
    │   │
    │   └─► npm run build (Vite)
    │       (Bundle → build/dist/index.html with inlined JS/CSS)
    │
    ├─► Phase 2: Asset Embedding
    │   │
    │   └─► Generate EmbeddedAssets.fs
    │       (HTML file → F# string constant)
    │
    ├─► Phase 3: Native Compilation
    │   │
    │   └─► Firefly compile src/Backend + EmbeddedAssets.fs
    │       (F# → PSG → Alex → MLIR → LLVM → native)
    │
    └─► Output: MyApp (single executable)
```

### Cross-Platform Builds

```bash
# Build for current platform
firefly build

# Cross-compile
firefly build --target linux-x64
firefly build --target windows-x64
firefly build --target macos-arm64
```

Each target produces a native executable for that platform, with platform-specific webview bindings compiled in.

---

## IPC Model

Communication between the JavaScript frontend and native backend uses webview's binding mechanism.

### Function Binding

Native code registers functions that JavaScript can call:

```fsharp
// Native side
webview_bind w "calculateTotal" (fun id args ->
    let items = decode args  // Parse arguments
    let total = calculate items
    webview_return w id 0 (encode total)  // Return result
)
```

```javascript
// JavaScript side
const total = await window.calculateTotal([
    { name: "Item 1", price: 10.00 },
    { name: "Item 2", price: 25.50 }
]);
```

### IPC Format: BAREWire over Base64

To avoid JSON (per project preference), IPC uses BAREWire binary encoding:

```
JavaScript                              Native
    │                                      │
    │  payload = bareEncode(data)          │
    │  base64 = btoa(payload)              │
    │                                      │
    │  myFunc(base64) ─────────────────►   │
    │                                      │  bytes = base64Decode(req)
    │                                      │  data = BAREWire.decode(bytes)
    │                                      │
    │                                      │  result = process(data)
    │                                      │
    │  ◄───────────────── webview_return   │  encoded = BAREWire.encode(result)
    │                                      │  webview_return(w, id, 0, base64(encoded))
    │                                      │
    │  response = bareDecode(atob(result)) │
```

This provides:
- Binary efficiency (smaller payloads than JSON)
- Type safety (schema-defined encoding)
- No JSON dependency

---

## Platform Matrix

| Platform | Webview Runtime | Library | Link Type |
|----------|-----------------|---------|-----------|
| Linux x64 | WebKitGTK | libwebkit2gtk-4.0.so | Dynamic |
| Linux ARM64 | WebKitGTK | libwebkit2gtk-4.0.so | Dynamic |
| Windows x64 | WebView2 | WebView2Loader.dll | Dynamic |
| macOS x64 | WKWebView | WebKit.framework | System |
| macOS ARM64 | WKWebView | WebKit.framework | System |

### Runtime Requirements

- **Linux**: WebKitGTK package installed (virtually always present on desktop distros)
- **Windows**: WebView2 runtime (ships with Windows 11, installer available for Windows 10)
- **macOS**: WKWebView (always present - part of the OS)

---

## Relationship to FidelityUI

This webview-based approach is a **pragmatic interim solution** for desktop applications. The long-term vision described in [Demo_UI_Stretch_Goal.md](./Demo_UI_Stretch_Goal.md) includes:

- **FidelityUI**: Native widget toolkit using LVGL (embedded) and GTK4/Skia (desktop)
- **Compile-time widget transformation**: F# UI definitions compiled directly to native rendering
- **Zero runtime overhead**: No JavaScript, no browser engine

The webview approach provides:
- **Faster time to demo**: Leverage existing Partas.Solid/SolidJS ecosystem
- **Web skills transfer**: HTML/CSS/JS knowledge applies directly
- **Bridge to native**: Same MVU patterns will transfer to FidelityUI

Both approaches share the architectural principles:
- Platform-agnostic definitions in Fidelity.Platform
- Platform-specific implementations in Alex
- Firefly as the unified build orchestrator

---

## Cross-References

- **[Architecture_Canonical.md](./Architecture_Canonical.md)** - Layer separation principle, the split model
- **[Demo_UI_Stretch_Goal.md](./Demo_UI_Stretch_Goal.md)** - FidelityUI long-term vision
- **[WebView_Build_Integration.md](./WebView_Build_Integration.md)** - Build system details

---

## Glossary

| Term | Definition |
|------|------------|
| **Fidelity.Platform** | Native F# standard library - platform-agnostic bindings |
| **Alex** | Firefly's multi-dimensional targeting layer - generates platform-specific MLIR |
| **Conduit** | A Platform.Bindings function where Alex provides the implementation |
| **DCont** | Delimited continuations MLIR dialect |
| **Fable** | F# to JavaScript compiler |
| **Partas.Solid** | F# DSL for SolidJS components |
| **Platform.Bindings** | Module convention for platform-provided functions |
| **PSG** | Program Semantic Graph - Firefly's intermediate representation |
| **Quotation** | F# feature: `<@ code @>` captures code as inspectable data |
| **SolidJS** | Reactive JavaScript UI framework (alternative to React) |
| **WebView** | System browser component embedded in native applications |
| **XParsec** | Parser combinator library for PSG pattern matching |
| **Zipper** | Bidirectional traversal structure for trees/graphs |
