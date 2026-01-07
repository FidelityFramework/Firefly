# The WREN Stack: Reactive Native Fusion

The **WREN** stack is a design pattern for building ultra-thin, high-performance desktop applications by fusing modern web technologies with F# Native. It treats the UI and the logic as two perspectives of a single, unified compute unit.

## The Acronym

*   **W**ebview: The system surface (WebKitGTK/WebView2) used for rendering.
*   **R**eactive: Partas.Solid + SolidJS for fine-grained, declarative UI.
*   **E**mbedded: Assets are inlined and compiled directly into the binary's `.rodata`.
*   **N**ative: Firefly + Alex + BAREWire for close-to-the-metal performance.

## Core Concept: The "Weld"

The WREN stack rejects the idea of the "Frontend" and "Backend" as separate entities. Instead, it uses **Firefly** to "weld" them into a single binary.

1.  **Fable Worker**: Firefly triggers a Fable build worker to compile Partas.Solid to JS.
2.  **Vite Unification**: The JS, CSS (DaisyUI), and HTML are inlined into a single `index.html`.
3.  **Alex Injection**: Firefly reads this file and embeds it as a static string literal in the native executable.
4.  **Instant Wake**: On execution, the native binary spawns the system WebView and loads the UI from its own RAM—no disk I/O, no network.

## The BAREWire Nervous System

Communication in a WREN application happens via **BAREWire**. 
*   **Shared Contract**: A shared F# project defines the BAREWire schemas.
*   **Dual-Targeting**: BAREWire compiles for both Fable (JS) and Firefly (Native).
*   **Zero-Copy Intent**: Since both threads share the same process memory, WREN aims for zero-copy binary communication between the background logic and the UI surface.

## Threading Architecture

WREN enforces a clean separation of concerns:
*   **The Main Thread**: Owned by the WebView event loop (Thread 0). Handles all rendering and DaisyUI animations.
*   **The Logic Thread(s)**: Native F# code running on OS threads. Handles hardware polling, heavy math, and BAREWire processing.
*   **The Dispatcher**: A thread-safe bridge that allows native logic to safely update the reactive UI signals.

## Marketability

*   **Tiny Footprint**: Megabytes, not hundreds of megabytes.
*   **Native Speed**: True OS threads and native machine code for the "Brain."
*   **Web Beauty**: Full access to the Tailwind/DaisyUI ecosystem for the "Face."
*   **One File**: Delivered as a single, standalone native executable.

---
*WREN: Small, native, and incredibly fast. The fusion of high-level design and low-level power.*
