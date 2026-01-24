# D-01: GTK Window (FFI Foundation)

> **Sample**: `25_GTKWindow` | **Status**: Planned | **Depends On**: C-01 to A-06

## 1. Executive Summary

This PRD introduces GTK FFI - calling into native GTK libraries to create desktop GUI applications. This validates Fidelity's ability to interoperate with existing C libraries.

**Key Insight**: GTK uses a C API with function pointers for callbacks. Fidelity's closure representation (C-01) is compatible with C function pointer calling conventions.

## 2. Language Feature Specification

### 2.1 GTK Initialization

```fsharp
[<DllImport("libgtk-4.so.1")>]
extern void gtk_init()

[<DllImport("libgtk-4.so.1")>]
extern nativeptr<GtkWidget> gtk_window_new()
```

### 2.2 Window Creation

```fsharp
let window = gtk_window_new()
gtk_window_set_title(window, "Hello GTK")
gtk_window_set_default_size(window, 400, 300)
```

### 2.3 Signal Connection

```fsharp
// Connect callback for window close
g_signal_connect(window, "destroy", (fun _ -> gtk_main_quit()), null)
```

### 2.4 Main Loop

```fsharp
gtk_widget_show(window)
gtk_main()  // Blocks until quit
```

## 3. FNCS Layer Implementation

### 3.1 DllImport Attribute

```fsharp
type DllImportAttribute(library: string) =
    inherit System.Attribute()
    member _.Library = library
```

FNCS recognizes `[<DllImport>]` and marks the function as external:

```fsharp
type SemanticKind =
    | ExternFunction of
        name: string *
        library: string *
        paramTypes: NativeType list *
        returnType: NativeType
```

### 3.2 GtkWidget as Opaque Pointer

```fsharp
// GTK types are opaque pointers
type GtkWidget = nativeptr<byte>
type GtkWindow = nativeptr<byte>
type GtkApplication = nativeptr<byte>
```

No need to model GTK struct internals - just pass pointers.

### 3.3 Callback Type

GTK callbacks are `void (*)(GtkWidget*, gpointer)`:

```fsharp
// Callback signature
type GtkCallback = nativeptr<byte> -> nativeptr<byte> -> unit
```

Fidelity closures with no captures can be passed as C function pointers.

## 4. Firefly/Alex Layer Implementation

### 4.1 External Function Declaration

```fsharp
let witnessExternFunction z name library paramTypes returnType =
    // Emit LLVM function declaration
    let paramStr = paramTypes |> List.map toLLVMType |> String.concat ", "
    emit $"llvm.func @{name}({paramStr}) -> {toLLVMType returnType}"
    emit $"  attributes {{ \"link\" = \"{library}\" }}"
```

### 4.2 External Call

```fsharp
let witnessExternCall z funcName args =
    let argStr = args |> List.map (sprintf "%%%s") |> String.concat ", "
    let resultSSA = freshSSA ()
    emit $"  %%{resultSSA} = llvm.call @{funcName}({argStr})"
    TRValue { SSA = resultSSA; Type = returnType }
```

### 4.3 Closure to C Function Pointer

For callbacks, extract the code pointer from closure:

```fsharp
let witnessClosureToFuncPtr z closureSSA =
    // If closure has no captures, code pointer is directly usable
    let codePtr = freshSSA ()
    emit $"  %%{codePtr} = llvm.extractvalue %%{closureSSA}[0]"
    TRValue { SSA = codePtr; Type = TFunctionPointer }
```

**Limitation**: Only capture-free closures can be converted to C function pointers. Closures with captures would need a trampoline (future work).

## 5. MLIR Output Specification

### 5.1 External Function Declarations

```mlir
// GTK function declarations
llvm.func @gtk_init() attributes { "link" = "gtk-4" }
llvm.func @gtk_window_new() -> !llvm.ptr attributes { "link" = "gtk-4" }
llvm.func @gtk_window_set_title(!llvm.ptr, !llvm.ptr) attributes { "link" = "gtk-4" }
llvm.func @gtk_window_set_default_size(!llvm.ptr, i32, i32) attributes { "link" = "gtk-4" }
llvm.func @gtk_widget_show(!llvm.ptr) attributes { "link" = "gtk-4" }
llvm.func @gtk_main() attributes { "link" = "gtk-4" }
llvm.func @gtk_main_quit() attributes { "link" = "gtk-4" }
llvm.func @g_signal_connect_data(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i64
    attributes { "link" = "gobject-2.0" }
```

### 5.2 Window Creation

```mlir
// gtk_init()
llvm.call @gtk_init() : () -> ()

// let window = gtk_window_new()
%window = llvm.call @gtk_window_new() : () -> !llvm.ptr

// gtk_window_set_title(window, "Hello GTK")
%title = llvm.mlir.addressof @str_hello_gtk : !llvm.ptr
llvm.call @gtk_window_set_title(%window, %title) : (!llvm.ptr, !llvm.ptr) -> ()

// gtk_window_set_default_size(window, 400, 300)
llvm.call @gtk_window_set_default_size(%window, %c400, %c300) : (!llvm.ptr, i32, i32) -> ()
```

### 5.3 Signal Connection

```mlir
// g_signal_connect(window, "destroy", quit_callback, null)
%destroy_str = llvm.mlir.addressof @str_destroy : !llvm.ptr
%callback = llvm.mlir.addressof @on_destroy : !llvm.ptr
%null = llvm.mlir.null : !llvm.ptr
llvm.call @g_signal_connect_data(%window, %destroy_str, %callback, %null, %null, %c0)

// Callback function
llvm.func @on_destroy(%widget: !llvm.ptr, %data: !llvm.ptr) {
    llvm.call @gtk_main_quit() : () -> ()
    llvm.return
}
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module GTKWindowSample

[<DllImport("libgtk-4.so.1")>]
extern void gtk_init()

[<DllImport("libgtk-4.so.1")>]
extern nativeptr<byte> gtk_window_new()

[<DllImport("libgtk-4.so.1")>]
extern void gtk_window_set_title(window: nativeptr<byte>, title: string)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_window_set_default_size(window: nativeptr<byte>, width: int, height: int)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_widget_show(widget: nativeptr<byte>)

[<DllImport("libgtk-4.so.1")>]
extern void gtk_main()

[<DllImport("libgtk-4.so.1")>]
extern void gtk_main_quit()

[<DllImport("libgobject-2.0.so.0")>]
extern int64 g_signal_connect_data(
    instance: nativeptr<byte>,
    signal: string,
    handler: nativeptr<byte>,
    data: nativeptr<byte>,
    destroy: nativeptr<byte>,
    flags: int)

let onDestroy (_widget: nativeptr<byte>) (_data: nativeptr<byte>) : unit =
    gtk_main_quit()

[<EntryPoint>]
let main _ =
    Console.writeln "=== GTK Window Test ==="

    gtk_init()

    let window = gtk_window_new()
    gtk_window_set_title(window, "Fidelity GTK")
    gtk_window_set_default_size(window, 400, 300)

    // Connect destroy signal
    g_signal_connect_data(window, "destroy", NativePtr.toVoidPtr onDestroy, NativePtr.nullPtr, NativePtr.nullPtr, 0)

    gtk_widget_show(window)

    Console.writeln "Window shown, entering main loop..."
    gtk_main()

    Console.writeln "Main loop exited"
    0
```

### 6.2 Expected Behavior

- Window appears titled "Fidelity GTK"
- Window is 400x300 pixels
- Closing window exits the program

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `Attributes.fs` | CREATE | DllImport attribute definition |
| `CheckExpressions.fs` | MODIFY | Handle extern function declarations |
| `SemanticGraph.fs` | MODIFY | Add ExternFunction SemanticKind |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/ExternWitness.fs` | CREATE | External function call emission |
| `src/Alex/CodeGeneration/LinkAttributes.fs` | CREATE | Library linking metadata |

## 8. Implementation Checklist

### Phase 1: FNCS FFI Support
- [ ] Implement DllImport attribute parsing
- [ ] Add ExternFunction to SemanticKind
- [ ] Type-check extern declarations

### Phase 2: Alex FFI Emission
- [ ] Emit external function declarations
- [ ] Emit external calls
- [ ] Handle library linking attributes

### Phase 3: GTK Integration
- [ ] Verify GTK libraries load
- [ ] Verify callback works
- [ ] Verify main loop runs

### Phase 4: Validation
- [ ] Sample 25 compiles
- [ ] Window appears
- [ ] Window closes correctly

## 9. Linking

The generated binary needs to link against GTK libraries:

```bash
# During LLVM compilation
llc ... -filetype=obj -o window.o
gcc window.o -o window $(pkg-config --libs gtk4)
```

Or use `lld` with appropriate library paths.

## 10. Related PRDs

- **C-01**: Closures - Callback functions
- **D-02**: WebViewBasic - WebKit in GTK
- **T-03 to T-05**: MailboxProcessor - GUI event handling
