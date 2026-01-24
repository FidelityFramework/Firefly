# WebView Desktop Design Notes

> **Type**: Design document (mechanical implementation details)
>
> **Related Architecture**: [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md)

---

## Callback Handling Approaches

The webview C API uses function pointers for callbacks:

```c
typedef void (*webview_bind_fn)(const char *id, const char *req, void *arg);

webview_error_t webview_bind(
    webview_t w,
    const char *name,
    webview_bind_fn fn,
    void *arg
);
```

Firefly-compiled code must provide a function matching this signature. Two approaches are viable:

### Approach A: C Shim (Pragmatic)

A small C wrapper provides the function pointer, dispatching to Fidelity code:

```c
// webview_fidelity_shim.c

// Fidelity exports this function
extern void fidelity_dispatch(const char *name, const char *id, const char *req);

// Trampoline that webview calls
static void callback_trampoline(const char *id, const char *req, void *arg) {
    const char *name = (const char *)arg;
    fidelity_dispatch(name, id, req);
}

// Wrapper that Fidelity calls instead of webview_bind directly
void webview_fidelity_bind(webview_t w, const char *name) {
    webview_bind(w, name, callback_trampoline, (void *)name);
}
```

**Fidelity side:**
```fsharp
// Platform.Bindings - uses the shim
let bindWebview (webview: nativeint) (name: nativeint) : int =
    Unchecked.defaultof<int>  // Maps to webview_fidelity_bind

// Fidelity exports fidelity_dispatch
let fidelity_dispatch (name: nativeint) (id: nativeint) (req: nativeint) : unit =
    // Look up handler by name, invoke with id and req
    ()
```

**Pros:**
- Works with current Firefly capabilities
- Minimal C code (~20 lines)
- Well-understood pattern

**Cons:**
- Requires C compilation step
- Additional linkage complexity
- Not pure Fidelity

### Approach B: Firefly Function Export

Extend Firefly to export functions with C calling convention:

```fsharp
// Hypothetical syntax
[<CExport("my_callback")>]
let myCallback (id: nativeint) (req: nativeint) (arg: nativeint) : unit =
    // Handle callback
    ()
```

Firefly generates:
```llvm
define void @my_callback(ptr %id, ptr %req, ptr %arg) {
    ; ... compiled F# body ...
}
```

Then Platform.Bindings.bindWebview can pass `@my_callback` directly to webview_bind.

**Pros:**
- Pure Fidelity solution
- No C code required
- Cleaner distribution

**Cons:**
- Requires Firefly enhancement (export with C linkage)
- Must handle calling convention correctly
- More implementation work

### Decision

**Deferred** - Document both approaches. Implement based on timeline and complexity assessment during Milestone 3.

---

## BAREWire-over-Webview IPC Design

Webview's IPC is string-based. BAREWire produces binary. Bridge via base64.

### Encoding Flow

```
JavaScript                              Native
    │                                      │
    │  // Encode to binary                 │
    │  const bytes = bareEncode(schema, data)
    │                                      │
    │  // Binary to base64 string          │
    │  const b64 = btoa(String.fromCharCode(...bytes))
    │                                      │
    │  // Call native function             │
    │  window.myFunc(b64) ─────────────────►
    │                                      │
    │                  // Base64 to bytes  │
    │                  let bytes = base64Decode(b64)
    │                                      │
    │                  // Decode with BAREWire
    │                  let data = BAREWire.decode schema bytes
    │                                      │
    │                  // Process          │
    │                  let result = process data
    │                                      │
    │                  // Encode result    │
    │                  let resultBytes = BAREWire.encode schema result
    │                                      │
    │                  // Return as base64 │
    │  ◄─────────────── webview_return(w, id, 0, base64Encode(resultBytes))
    │                                      │
    │  // Decode response                  │
    │  const resultBytes = Uint8Array.from(atob(response), c => c.charCodeAt(0))
    │  const result = bareDecode(schema, resultBytes)
```

### JavaScript BAREWire Implementation

Need a JS implementation of BARE encoding. Options:

1. **@aspect/bare** - Existing npm package for BARE protocol
2. **Custom minimal encoder** - Generate from schema
3. **Fable-compiled BAREWire** - Compile BAREWire to JS via Fable

Option 3 is most consistent with the stack - same encoding logic on both sides.

### Schema Sharing

Shared schema defined in F#:

```fsharp
// Shared/Messages.fs
module MyApp.Messages

open BAREWire

type Request =
    | QueryItems of filter: string
    | UpdateItem of id: int * data: ItemData

type Response =
    | Items of Item list
    | Updated of success: bool
    | Error of message: string

let requestSchema = BAREWire.schema<Request>
let responseSchema = BAREWire.schema<Response>
```

Frontend (Fable-compiled) and backend (Firefly-compiled) use the same types and schemas.

### Base64 Utilities

**Native side (Fidelity.Platform):**
```fsharp
module Fidelity.Platform.Encoding.Base64

let encode (bytes: Span<byte>) : string = ...
let decode (str: string) : byte array = ...
```

**JavaScript side:**
```javascript
function toBase64(bytes) {
    return btoa(String.fromCharCode(...bytes));
}

function fromBase64(str) {
    return Uint8Array.from(atob(str), c => c.charCodeAt(0));
}
```

### Message Wrapper

To support multiple message types over single binding:

```fsharp
type Envelope = {
    Type: string      // Message type discriminator
    Payload: byte[]   // BAREWire-encoded payload
}
```

Or use BARE's union encoding directly if schema supports it.

---

## Webview C API Reference

For implementation, here's the complete webview C API:

```c
// Types
typedef void *webview_t;
typedef enum { WEBVIEW_HINT_NONE, WEBVIEW_HINT_MIN, WEBVIEW_HINT_MAX, WEBVIEW_HINT_FIXED } webview_hint_t;
typedef enum { WEBVIEW_ERROR_OK, ... } webview_error_t;

// Lifecycle
webview_t webview_create(int debug, void *window);
webview_error_t webview_destroy(webview_t w);
webview_error_t webview_run(webview_t w);
webview_error_t webview_terminate(webview_t w);

// Window
webview_error_t webview_set_title(webview_t w, const char *title);
webview_error_t webview_set_size(webview_t w, int width, int height, webview_hint_t hints);

// Navigation
webview_error_t webview_navigate(webview_t w, const char *url);
webview_error_t webview_set_html(webview_t w, const char *html);

// JavaScript
webview_error_t webview_init(webview_t w, const char *js);
webview_error_t webview_eval(webview_t w, const char *js);

// Binding
webview_error_t webview_bind(webview_t w, const char *name,
    void (*fn)(const char *id, const char *req, void *arg), void *arg);
webview_error_t webview_unbind(webview_t w, const char *name);
webview_error_t webview_return(webview_t w, const char *id, int status, const char *result);

// Dispatch (thread-safe)
webview_error_t webview_dispatch(webview_t w,
    void (*fn)(webview_t w, void *arg), void *arg);
```

### Platform Libraries

| Platform | Library | Package |
|----------|---------|---------|
| Linux | libwebkit2gtk-4.1.so | webkit2gtk |
| Windows | WebView2Loader.dll | WebView2 SDK |
| macOS | WebKit.framework | System |

### Linking

**Linux:**
```bash
# Dynamic linking
-lwebkit2gtk-4.1 -lgtk-4 -lglib-2.0

# Or via pkg-config
$(pkg-config --libs webkit2gtk-4.1)
```

**Windows:**
```
WebView2Loader.dll (runtime)
WebView2LoaderStatic.lib (static linking option)
```

**macOS:**
```bash
-framework WebKit -framework Cocoa
```

---

## Platform.Bindings.Webview Mapping

Mapping from Fidelity.Platform bindings to webview C API:

| Fidelity.Platform Binding | C Function | Notes |
|---------------|------------|-------|
| `createWebview` | `webview_create` | |
| `destroyWebview` | `webview_destroy` | |
| `runWebview` | `webview_run` | Blocks until terminated |
| `terminateWebview` | `webview_terminate` | |
| `setWebviewTitle` | `webview_set_title` | |
| `setWebviewSize` | `webview_set_size` | hints param = 0 (NONE) |
| `navigateWebview` | `webview_navigate` | |
| `setWebviewHtml` | `webview_set_html` | |
| `evalWebview` | `webview_eval` | |
| `initWebview` | `webview_init` | |
| `returnWebview` | `webview_return` | |
| `bindWebview` | See callback section | Requires shim or export |

---

## Alex Binding Implementation Notes

### LLVM External Declaration

For each webview function:

```llvm
declare ptr @webview_create(i32 %debug, ptr %window)
declare i32 @webview_destroy(ptr %w)
declare i32 @webview_run(ptr %w)
; ... etc
```

### Dynamic Library Loading

Linux example:
```llvm
; Load library at startup
@webview_lib = external global ptr

define internal void @init_webview_lib() {
    %lib = call ptr @dlopen(ptr @.str.libwebkit, i32 1)  ; RTLD_LAZY
    store ptr %lib, ptr @webview_lib
    ; Resolve symbols...
    ret void
}
```

Or rely on standard dynamic linking (simpler, library must be installed).

### Syscall vs Library Call

Unlike console bindings (which use syscalls), webview bindings call library functions:

```fsharp
// Console binding - syscall
let emitWriteSyscall = mlir {
    let! result = llvm.inline_asm "syscall" [syscallNum; fd; buf; count]
    return result
}

// Webview binding - library call
let emitWebviewCreate = mlir {
    let! result = llvm.call "@webview_create" [debug; window] (!llvm.ptr)
    return result
}
```

No inline assembly needed - just standard LLVM function calls.

---

## Threading Considerations

Webview is single-threaded from the application's perspective:
- `webview_run()` blocks on the GUI event loop
- Callbacks fire on the GUI thread
- `webview_dispatch()` allows posting work from other threads

For long-running native operations:

```fsharp
let handleExpensiveRequest id req =
    // Spawn worker (future: Fidelity async)
    async {
        let! result = expensiveComputation req
        // Post result back to GUI thread
        webview_dispatch w (fun _ _ ->
            webview_return w id 0 (encode result)
        ) 0n
    } |> Async.Start
```

This pattern avoids blocking the GUI thread during computation.

---

## Memory Management

### String Lifetimes

Webview expects null-terminated C strings. With string (native semantics):

```fsharp
let setTitle w title =
    // string is null-terminated (UTF-8 fat pointer)
    // Pass pointer directly, valid for duration of call
    setWebviewTitle w (NativePtr.toNativeInt title.Pointer)
```

For callback responses:
```fsharp
let respondToCallback w id result =
    // Result string must remain valid until webview_return completes
    use resultStr = String.withNullTerminator result
    returnWebview w id 0 (NativePtr.toNativeInt resultStr.Pointer)
    // resultStr deallocated after call completes
```

### Webview Handle Lifetime

The webview handle (`webview_t`) is valid from `create` to `destroy`:

```fsharp
let main () =
    let w = createWebview 1 0n  // Allocates webview

    // ... use w ...

    destroyWebview w |> ignore  // Must be called
    0
```

Failing to call `destroy` leaks the webview and associated resources.
