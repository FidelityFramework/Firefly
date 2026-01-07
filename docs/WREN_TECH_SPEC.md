# WREN Stack: Technical Specification

The WREN stack (Webview, Reactive, Embedded, Native) achieves high-performance desktop experiences by "welding" a Fable frontend to a Firefly backend inside a single native binary.

## 1. The Threading Model ("The Hook")

WREN applications utilize a dual-thread architecture to ensure UI responsiveness while performing heavy native computations.

### The Main Thread (UI)
*   **Owner**: The system's WebView event loop (Thread 0).
*   **Responsibility**: Rendering Partas.Solid components, processing DaisyUI styles, and managing DOM events.
*   **Constraint**: Must never be blocked by long-running tasks.

### The Logic Thread (Native)
*   **Owner**: Firefly-compiled native logic running on OS threads.
*   **Responsibility**: Hardware interaction, intensive data processing, and BAREWire encoding.
*   **Spawn**: Triggered via `NativeThread.spawn` (or equivalent) before the WebView starts.

## 2. Communication: BAREWire-over-WebView

Communication between the UI and Native Logic avoids JSON overhead by using **BAREWire** binary messages, bridged via Base64 for WebView compatibility.

### IPC Workflow
1.  **Frontend**: Calls `Fidelity.WebView.Fable.callNative`. BAREWire encodes an F# record into a `Uint8Array`.
2.  **Bridge**: The binary is Base64-encoded and passed through the WebView's `webview_bind` conduit.
3.  **Backend**: The native handler receives the Base64, decodes the BAREWire payload directly into a native F# record (targeting zero-copy where possible), and executes.
4.  **Return**: Results are encoded, Base64-wrapped, and dispatched back to the UI thread via `WebView.returnResult`.

## 3. The Unified Build Track

Firefly orchestrates the "Weld" by running two concurrent workers:

### Phase A: The Face (Fable/Vite)
*   Firefly invokes `dotnet fable` on the `src/Frontend` project.
*   Vite bundles the output, inlining all JS/CSS into a single `index.html`.

### Phase B: The Brain (Firefly/Alex)
*   Firefly performs reachability analysis on `src/Backend`.
*   The `index.html` string is captured as an F# `[<Literal>]`.
*   Alex (MLIR backend) places the HTML string into the `.rodata` section and the logic into the `.text` section of the binary.

## 4. Logical Boundaries

| Component | Library | Environment |
| :--- | :--- | :--- |
| **Surface** | `Fidelity.Platform.WebView` | Native (MLIR) |
| **Bridge** | `Fidelity.WebView` | Dual (Native/JS) |
| **Contract** | `BAREWire` | Shared (F#) |
| **Pattern** | `Partas.Solid` | Fable (JS) |