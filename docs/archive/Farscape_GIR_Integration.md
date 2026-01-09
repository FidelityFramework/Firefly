# Farscape GIR Integration: Native Desktop UI Bindings

This document outlines the roadmap for extending Farscape to consume GObject Introspection (GIR) files, enabling automatic generation of Fidelity-compatible F# bindings for GTK4 and other GObject-based libraries. This represents a key evolution in Farscape's binding generation capabilities, moving beyond simple C header parsing to leverage rich metadata that captures memory ownership, type hierarchies, and callback signatures.

## The Binding Challenge

As discussed in "The Farscape Bridge" and "Binding F# to C++ in Farscape," generating high-quality foreign function bindings requires more than just extracting function signatures from headers. The critical questions for any binding are:

- **Who owns the memory?** Does the caller allocate? Does the callee free?
- **What can be null?** Which parameters accept null, which return values might be null?
- **How do callbacks work?** What's the signature? When is it called? Who owns the closure?

Traditional C header parsing can extract the function signatures, but these semantic questions require additional annotation or documentation. This is where GObject Introspection shines—it provides machine-readable answers to all of these questions.

## GObject Introspection: Metadata-Rich Binding Source

GObject Introspection (GIR) is the GNOME project's solution to the binding generation problem. Rather than parsing headers and guessing at semantics, GIR files are **generated from annotated source code** with explicit metadata about ownership, nullability, array lengths, and more.

Consider how a simple constructor appears in GIR:

```xml
<constructor name="new_with_label" c:identifier="gtk_button_new_with_label">
  <return-value transfer-ownership="none">
    <!-- Caller does NOT own - GTK manages widget lifecycle -->
    <type name="Widget" c:type="GtkWidget*"/>
  </return-value>
  <parameters>
    <parameter name="label" transfer-ownership="none">
      <!-- GTK copies the string - we can pass stack-allocated -->
      <type name="utf8" c:type="const char*"/>
    </parameter>
  </parameters>
</constructor>
```

This tells Farscape everything needed for safe Fidelity bindings:
- The returned widget pointer is managed by GTK (no cleanup needed)
- The label string is copied by GTK (safe to pass stack-allocated `string`)

This metadata is exactly what Fidelity requires for deterministic memory management without a garbage collector.

## Why GIR for Fidelity?

The "Memory Management by Choice" and "F# Memory Management Goes Native" blog posts establish Fidelity's memory philosophy: the compiler knows allocation lifetimes, enabling stack and arena allocation without GC. But this only works if bindings accurately describe ownership semantics.

GIR provides:

| Metadata | What It Tells Farscape |
|----------|----------------------|
| `transfer-ownership="none"` | Safe to pass stack-allocated, no cleanup needed |
| `transfer-ownership="full"` | Caller must free, or explicitly transfer ownership |
| `nullable="1"` | Parameter or return can be null—generate `Option` type |
| `array length="2"` | Parameter 2 contains the array length |
| `glib:signal` | Event with specific callback signature |
| `parent="Widget"` | Inheritance hierarchy for type checking |

This rich metadata enables Farscape to generate bindings that preserve Fidelity's memory safety guarantees.

## GIR File Ecosystem

On Linux systems with GTK4 development packages installed, GIR files live in `/usr/share/gir-1.0/`:

```
/usr/share/gir-1.0/
├── Gtk-4.0.gir          # Main GTK4 API (~3MB of metadata)
├── Gdk-4.0.gir          # Low-level graphics/input
├── Gsk-4.0.gir          # GTK Scene Kit (rendering)
├── GLib-2.0.gir         # Core utilities
├── GObject-2.0.gir      # Object system fundamentals
├── Gio-2.0.gir          # I/O and application framework
├── Pango-1.0.gir        # Text rendering
├── cairo-1.0.gir        # 2D graphics
├── Adw-1.gir            # libadwaita (modern GNOME widgets)
├── GStreamer-1.0.gir    # Multimedia framework
└── ...hundreds more
```

This ecosystem means a single Farscape GIR parser unlocks bindings for the entire GNOME platform stack—not just GTK4, but also GStreamer for multimedia, WebKitGTK for web views, libsecret for credential storage, and more.

## The Three-Layer Binding Architecture

As established in "Farscape's Modular Entry Points," Farscape generates bindings in three layers. This architecture applies equally to GIR-based generation:

### Layer 1: Platform Bindings

Platform.Bindings modules extracted from GIR's `c:identifier` attributes. Alex provides MLIR emission that links against GTK4:

```fsharp
// Auto-generated from Gtk-4.0.gir - Platform.Bindings pattern (BCL-free)
module Platform.Bindings.GTK.Button =
    /// Create a new button widget
    let create () : nativeint =
        Unchecked.defaultof<nativeint>

    /// Create a button with a text label
    let createWithLabel (label: nativeint) : nativeint =
        Unchecked.defaultof<nativeint>

    /// Set the button's label text
    let setLabel (button: nativeint) (label: nativeint) : unit =
        ()
```

### Layer 2: Type Definitions

Opaque handle wrappers that provide type safety without exposing internal structure:

```fsharp
module Fidelity.GTK4.Types

/// Opaque handle to a GtkButton
[<Struct>]
type Button = { Handle: nativeptr<GtkButton> }

/// Opaque handle to a GtkWindow
[<Struct>]
type Window = { Handle: nativeptr<GtkWindow> }

// Signal callback types (from glib:signal definitions)
type ClickedHandler = unit -> unit
```

### Layer 3: Functional Wrappers

The idiomatic F# API that developers actually use. This layer transforms imperative C patterns into functional, pipe-friendly operations:

```fsharp
module Fidelity.GTK4.Button

/// Create a button with a text label
let withLabel (text: string) : Button =
    { Handle = gtk_button_new_with_label(text.Pointer) |> NativePtr.cast }

/// Set the button's label text (returns button for chaining)
let setLabel (text: string) (button: Button) : Button =
    gtk_button_set_label(button.Handle, text.Pointer)
    button

/// Connect a handler to the clicked signal
let onClicked (handler: unit -> unit) (button: Button) : Button =
    let callback = NativeCallback.createStatic handler
    g_signal_connect(button.Handle |> NativePtr.cast, "clicked"n.Pointer, callback, NativePtr.nullPtr)
    |> ignore
    button
```

This layered approach, discussed in "The Farscape Bridge," allows developers to drop to lower layers when needed while providing a clean high-level API for typical usage.

## Functional Wrappers: The Farscape Innovation

The key insight from "Binding F# to C++ in Farscape" is that **Layer 3 wrappers are what make library code expressible in the PSG**. Raw extern calls appear as opaque function invocations—the semantic structure is lost. Functional wrappers decompose operations into patterns that Alex can recognize and optimize.

Consider how this GTK4 application code looks with proper wrappers:

```fsharp
open Fidelity.GTK4

let app =
    Application.create "com.example.demo"
    |> Application.onActivate (fun app ->
        Window.create app
        |> Window.title "Hello GTK4"n
        |> Window.defaultSize 400 300
        |> Window.child (
            Box.vertical 12
            |> Box.append (
                Button.withLabel "Click Me"n
                |> Button.onClicked (fun () -> printfn "Clicked!")
            )
            |> Box.append (
                Button.withLabel "Quit"n
                |> Button.onClicked (fun () -> Application.quit app)
            )
        )
        |> Window.present
    )
    |> Application.run 0 [||]
```

The pipe chain creates PSG structure that shows the widget hierarchy, event connections, and data flow. This is the same principle that makes Alloy's functional wrappers essential for compilation—the decomposed structure gives Alex visibility into intent.

## GObject Signals: Event Binding Without Closures

GObject's signal system presents a particular challenge for Fidelity: signals expect callbacks, but Fidelity doesn't support heap-allocated closures. The solution, aligned with principles in "Building User Interfaces with the Fidelity Framework," is static function pointers.

```fsharp
module NativeCallback =
    /// Create a static function pointer from a handler
    /// Handler must not capture variables (closure-free)
    let createStatic (handler: unit -> unit) : nativeptr<unit -> unit> =
        // Firefly compiler ensures this is a static function pointer
        NativePtr.ofFunction handler
```

The Firefly compiler enforces that signal handlers don't capture local variables. If state is needed, it must be passed through the signal's user data parameter or stored in application-level state accessible via the MVU model.

This constraint actually aligns well with the MVU pattern described in "Building User Interfaces with the Fidelity Framework"—event handlers dispatch messages rather than mutating state directly.

## GIR Schema Deep Dive

Understanding the GIR schema helps clarify what Farscape's parser must handle.

### Namespace Declaration

```xml
<namespace name="Gtk"
           version="4.0"
           shared-library="libgtk-4.so.1"
           c:identifier-prefixes="Gtk"
           c:symbol-prefixes="gtk">
```

This provides everything needed for the Platform.Bindings module structure.

### Class Definitions

```xml
<class name="Button"
       c:type="GtkButton"
       parent="Widget"
       glib:type-name="GtkButton">
  <implements name="Accessible"/>
  <implements name="Actionable"/>
  <!-- constructors, methods, properties, signals -->
</class>
```

The `parent` attribute establishes inheritance (Button extends Widget), while `implements` lists interface conformance.

### Transfer Ownership

| Value | Meaning | Fidelity Implication |
|-------|---------|---------------------|
| `none` | No ownership transfer | Safe to pass stack-allocated, no cleanup needed |
| `full` | Caller receives ownership | Must free/unref, or transfer to another owner |
| `container` | Own container but not elements | Rare, complex lifetime |

### Signal Definitions

```xml
<glib:signal name="clicked" when="first" action="1">
  <return-value transfer-ownership="none">
    <type name="none" c:type="void"/>
  </return-value>
</glib:signal>
```

Signals with parameters include `<parameters>` elements describing the callback signature.

## Farscape GIR Parser Architecture

Extending Farscape requires new components:

```
farscape/
├── src/
│   ├── Parsers/
│   │   ├── CHeaderParser.fs      # Existing C header parser
│   │   ├── GirParser.fs          # NEW: GIR XML parser
│   │   └── GirTypes.fs           # NEW: GIR schema types
│   ├── Generators/
│   │   ├── FSharpGenerator.fs    # Existing generator
│   │   ├── GirToFSharp.fs        # NEW: GIR-specific generation
│   │   └── FunctionalWrapper.fs  # NEW: Layer 3 wrapper generation
│   └── ...
```

### CLI Extension

```bash
farscape generate-gir \
    --gir /usr/share/gir-1.0/Gtk-4.0.gir \
    --include-gir /usr/share/gir-1.0/GLib-2.0.gir \
    --include-gir /usr/share/gir-1.0/GObject-2.0.gir \
    --output ./Fidelity.GTK4 \
    --namespace Fidelity.GTK4
```

The `--include-gir` flags handle type dependencies—GTK4 references GLib and GObject types that must be resolved.

## Cross-Platform Binding Roadmap

GIR establishes a pattern that extends to other platforms. Each major platform provides introspection metadata in different forms:

### Linux: GIR → Fidelity.GTK4

```
GIR files (XML) → Farscape GIR parser → F# bindings → libgtk-4.so.1
```

This is the primary focus, providing native Linux desktop UI.

### macOS: Objective-C Runtime → Fidelity.AppKit

Apple's Objective-C runtime provides introspection capabilities, and bridging headers expose API metadata:

```
Objective-C headers + runtime introspection → Farscape ObjC parser → F# bindings → AppKit.framework
```

Key differences from GIR:
- Message dispatch instead of direct function calls
- ARC memory management semantics
- Selector-based method invocation

### Windows: WinRT Metadata → Fidelity.WinUI

Windows provides type information via WinMD files (Windows Runtime metadata):

```
WinMD files → Farscape WinRT parser → F# bindings → Windows.UI.* APIs
```

### The Common Pattern

| Platform | Introspection Source | Native API | Binding Library |
|----------|---------------------|------------|-----------------|
| Linux | GIR (XML) | GTK4, GLib | `libgtk-4.so.1` |
| macOS | ObjC headers + runtime | AppKit, UIKit | System frameworks |
| Windows | WinMD, TLB | WinUI, WPF | System DLLs |

All three follow Farscape's three-layer pattern:
1. **Layer 1**: Extern declarations
2. **Layer 2**: Type mappings and struct layouts
3. **Layer 3**: Functional F# wrappers

This unified approach means learning to write FidelityUI code once enables targeting any platform.

## Relation to FidelityUI

As described in "Building User Interfaces with the Fidelity Framework" and "A Window Layout System for Fidelity," FidelityUI provides a unified MVU API that abstracts over rendering backends. GIR-generated `Fidelity.GTK4` becomes one such backend:

```
┌─────────────────────────────────────────────────────────────┐
│ F# Application Code (Fabulous-style MVU API)               │
│   view model = VStack { Button "Click"; Label "Hello" }    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FidelityUI Abstraction Layer                                │
│   Widget descriptions, layout, events                       │
└─────────────────────────────────────────────────────────────┘
          ↓                    ↓                    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ LVGL Backend    │  │ GTK4 Backend    │  │ AppKit Backend  │
│ (Embedded)      │  │ (Linux Desktop) │  │ (macOS Desktop) │
│ Farscape C hdrs │  │ Farscape GIR    │  │ Farscape ObjC   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

The same MVU F# code renders to:
- LVGL widgets on embedded Linux (Keystation)
- Native GTK4 widgets on Linux desktop
- Native AppKit on macOS (future)

This is the true Fidelity promise: **F# is the universal syntax, the binding target is an implementation detail.**

## Implementation Phases

### Phase 1: GIR Parser Core

- Parse GIR XML schema into internal type model
- Handle `<include>` dependencies between GIR files
- Resolve type references across namespaces
- Test with `GLib-2.0.gir` (simpler than full GTK4)

### Phase 2: Basic Code Generation

- Generate Layer 1 extern declarations
- Generate Layer 2 type wrappers (opaque handles)
- Handle basic types: primitives, strings, objects
- Test with subset of GTK4 (Window, Button, Label)

### Phase 3: Signal and Callback Support

- Parse `<glib:signal>` definitions
- Generate signal connection helpers
- Handle signals with parameters
- Test with Button.clicked, Entry.changed

### Phase 4: Functional Wrappers (Layer 3)

- Generate fluent/pipe-friendly API
- Builder pattern for widget construction
- Property setters return `self` for chaining
- Idiomatic F# naming conventions

### Phase 5: Full GTK4 Coverage

- All widget classes
- Layout containers (Box, Grid, Stack)
- Input widgets (Entry, TextView, ComboBox)
- Dialogs and application framework
- Generate API documentation from GIR `<doc>` elements

### Phase 6: Platform Expansion

- macOS Objective-C binding prototype
- Windows WinRT binding prototype
- Shared abstraction layer (FidelityUI backend interface)

## References

### SpeakEZ Blog Posts

These posts provide the conceptual foundation for this work:

- "The Farscape Bridge" - Core binding architecture and layer separation
- "Binding F# to C++ in Farscape" - Functional wrapper design principles
- "Farscape's Modular Entry Points" - Three-layer binding pattern
- "Building User Interfaces with the Fidelity Framework" - FidelityUI MVU architecture
- "A Window Layout System for Fidelity" - Layout engine and widget abstraction
- "Memory Management by Choice" - Fidelity's ownership model
- "F# Memory Management Goes Native" - Stack and arena allocation

### External Resources

- GObject Introspection Documentation: https://gi.readthedocs.io/
- GIR Format Reference: https://gi.readthedocs.io/en/latest/annotations/giannotations.html
- gir.core (C# bindings using GIR): https://github.com/gircore/gir.core
- GTK4 API Reference: https://docs.gtk.org/gtk4/
- Gtk4DotNet (functional C# GTK4 bindings): https://github.com/uriegel/Gtk4DotNet

### Related Firefly Documentation

- `docs/Hardware_Showcase_Roadmap.md` - Embedded targets and LVGL path
- `docs/Demo_UI_Stretch_Goal.md` - Demo day UI plans with multi-backend strategy
