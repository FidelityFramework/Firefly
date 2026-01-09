/// GTKBindings - Platform bindings for GTK3 on Linux
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.GTK.GTKBindings

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

/// Convert integer to pointer if needed (for nativeint args)
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
// GTK Platform Bindings
// ===================================================================

/// gtkInit - Initialize GTK with NULL args
let bindGtkInit (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let nullSSA, nullOps = nullPointer z
    let callOp = MLIROp.LLVMOp (callFunc None "gtk_init"
        [val' nullSSA MLIRTypes.ptr; val' nullSSA MLIRTypes.ptr] MLIRTypes.unit)
    BoundOps (nullOps @ [callOp], None)

/// gtkWindowNew - Create a new toplevel window
let bindGtkWindowNew (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    // GTK_WINDOW_TOPLEVEL = 0
    let zeroSSA = freshSynthSSA z
    let constOp = MLIROp.ArithOp (constI zeroSSA 0L MLIRTypes.i32)
    let resultSSA = freshSynthSSA z
    let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "gtk_window_new"
        [val' zeroSSA MLIRTypes.i32] MLIRTypes.ptr)
    BoundOps ([constOp; callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })

/// gtkWindowSetTitle - Set window title
let bindGtkWindowSetTitle (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [window; title] ->
        let windowPtr, windowOps = ensurePointer z window.SSA window.Type
        let titlePtr, titleOps = extractStringPointer z title.SSA title.Type
        let callOp = MLIROp.LLVMOp (callFunc None "gtk_window_set_title"
            [val' windowPtr MLIRTypes.ptr; val' titlePtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (windowOps @ titleOps @ [callOp], None)
    | _ ->
        NotSupported "gtkWindowSetTitle requires (window, title)"

/// gtkWindowSetDefaultSize - Set default window size
let bindGtkWindowSetDefaultSize (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [window; width; height] ->
        let windowPtr, windowOps = ensurePointer z window.SSA window.Type
        let callOp = MLIROp.LLVMOp (callFunc None "gtk_window_set_default_size"
            [val' windowPtr MLIRTypes.ptr; val' width.SSA MLIRTypes.i32; val' height.SSA MLIRTypes.i32] MLIRTypes.unit)
        BoundOps (windowOps @ [callOp], None)
    | _ ->
        NotSupported "gtkWindowSetDefaultSize requires (window, width, height)"

/// gtkContainerAdd - Add widget to container
let bindGtkContainerAdd (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [container; widget] ->
        let containerPtr, containerOps = ensurePointer z container.SSA container.Type
        let widgetPtr, widgetOps = ensurePointer z widget.SSA widget.Type
        let callOp = MLIROp.LLVMOp (callFunc None "gtk_container_add"
            [val' containerPtr MLIRTypes.ptr; val' widgetPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (containerOps @ widgetOps @ [callOp], None)
    | _ ->
        NotSupported "gtkContainerAdd requires (container, widget)"

/// gtkWidgetShowAll - Show widget and children
let bindGtkWidgetShowAll (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [widget] ->
        let widgetPtr, widgetOps = ensurePointer z widget.SSA widget.Type
        let callOp = MLIROp.LLVMOp (callFunc None "gtk_widget_show_all"
            [val' widgetPtr MLIRTypes.ptr] MLIRTypes.unit)
        BoundOps (widgetOps @ [callOp], None)
    | _ ->
        NotSupported "gtkWidgetShowAll requires (widget)"

/// gtkMain - Run GTK main loop
let bindGtkMain (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let callOp = MLIROp.LLVMOp (callFunc None "gtk_main" [] MLIRTypes.unit)
    BoundOps ([callOp], None)

/// gtkMainQuit - Quit GTK main loop
let bindGtkMainQuit (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    let callOp = MLIROp.LLVMOp (callFunc None "gtk_main_quit" [] MLIRTypes.unit)
    BoundOps ([callOp], None)

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register GTK bindings for Linux x86_64
    PlatformDispatch.register Linux X86_64 "gtkInit" bindGtkInit
    PlatformDispatch.register Linux X86_64 "gtkWindowNew" bindGtkWindowNew
    PlatformDispatch.register Linux X86_64 "gtkWindowSetTitle" bindGtkWindowSetTitle
    PlatformDispatch.register Linux X86_64 "gtkWindowSetDefaultSize" bindGtkWindowSetDefaultSize
    PlatformDispatch.register Linux X86_64 "gtkContainerAdd" bindGtkContainerAdd
    PlatformDispatch.register Linux X86_64 "gtkWidgetShowAll" bindGtkWidgetShowAll
    PlatformDispatch.register Linux X86_64 "gtkMain" bindGtkMain
    PlatformDispatch.register Linux X86_64 "gtkMainQuit" bindGtkMainQuit

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "gtkInit" bindGtkInit
    PlatformDispatch.register Linux ARM64 "gtkWindowNew" bindGtkWindowNew
    PlatformDispatch.register Linux ARM64 "gtkWindowSetTitle" bindGtkWindowSetTitle
    PlatformDispatch.register Linux ARM64 "gtkWindowSetDefaultSize" bindGtkWindowSetDefaultSize
    PlatformDispatch.register Linux ARM64 "gtkContainerAdd" bindGtkContainerAdd
    PlatformDispatch.register Linux ARM64 "gtkWidgetShowAll" bindGtkWidgetShowAll
    PlatformDispatch.register Linux ARM64 "gtkMain" bindGtkMain
    PlatformDispatch.register Linux ARM64 "gtkMainQuit" bindGtkMainQuit
