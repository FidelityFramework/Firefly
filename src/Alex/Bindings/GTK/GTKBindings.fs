/// GTKBindings - Platform bindings for GTK3 on Linux
///
/// ARCHITECTURAL FOUNDATION:
/// GTK is dynamically linked at load time - the system's libgtk-3.so
/// is already installed on Linux desktops. We declare external functions
/// and let the loader resolve them.
module Alex.Bindings.GTK.GTKBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ===================================================================
// Helper Functions
// ===================================================================

/// Extract pointer from fat string for C string interop
let private extractStringPointer (strSSA: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    match ty with
    | Struct _ ->
        let ptrSSA, z = MLIRZipper.yieldSSA zipper
        let extractText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let z' = MLIRZipper.witnessOpWithResult extractText ptrSSA Pointer z
        ptrSSA, z'
    | Pointer -> strSSA, zipper
    | _ -> strSSA, zipper

/// Witness a call to a GTK function
let witnessGtkCall (funcName: string) (args: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    let argTypes = args |> List.map snd
    let argTypeStrs = argTypes |> List.map Serialize.mlirType
    let retTypeStr = Serialize.mlirType returnType
    let signature = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) retTypeStr

    let zipper1 = MLIRZipper.observeExternFunc funcName signature zipper
    let argSSAs = args |> List.map fst
    MLIRZipper.witnessCall funcName argSSAs argTypes returnType zipper1

// ===================================================================
// GTK Platform Bindings
// ===================================================================

/// gtk_init - Initialize GTK
let witnessGtkInit (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    // gtk_init(NULL, NULL) - we pass null pointers
    let nullSSA, zipper1 = MLIRZipper.witnessConstant 0L I64 zipper
    let nullPtrSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let convText = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" nullPtrSSA nullSSA
    let zipper3 = MLIRZipper.witnessOpWithResult convText nullPtrSSA Pointer zipper2

    let args = [ (nullPtrSSA, Pointer); (nullPtrSSA, Pointer) ]
    let _, zipper4 = witnessGtkCall "gtk_init" args Unit zipper3
    zipper4, WitnessedVoid

/// gtk_window_new - Create a new window
let witnessGtkWindowNew (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    // GTK_WINDOW_TOPLEVEL = 0
    let zeroSSA, zipper1 = MLIRZipper.witnessConstant 0L I32 zipper
    let args = [ (zeroSSA, Integer I32) ]
    let resultSSA, zipper2 = witnessGtkCall "gtk_window_new" args Pointer zipper1
    zipper2, WitnessedValue (resultSSA, Pointer)

/// gtk_window_set_title - Set window title
let witnessGtkWindowSetTitle (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(windowSSA, _); (titleSSA, titleTy)] ->
        let titlePtrSSA, zipper1 = extractStringPointer titleSSA titleTy zipper
        let args = [ (windowSSA, Pointer); (titlePtrSSA, Pointer) ]
        let _, zipper2 = witnessGtkCall "gtk_window_set_title" args Unit zipper1
        zipper2, WitnessedVoid
    | _ -> zipper, NotSupported "gtk_window_set_title requires (window, title)"

/// gtk_window_set_default_size - Set default window size
let witnessGtkWindowSetDefaultSize (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(windowSSA, _); (widthSSA, _); (heightSSA, _)] ->
        let args = [ (windowSSA, Pointer); (widthSSA, Integer I32); (heightSSA, Integer I32) ]
        let _, zipper1 = witnessGtkCall "gtk_window_set_default_size" args Unit zipper
        zipper1, WitnessedVoid
    | _ -> zipper, NotSupported "gtk_window_set_default_size requires (window, width, height)"

/// gtk_container_add - Add widget to container
let witnessGtkContainerAdd (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(containerSSA, _); (widgetSSA, _)] ->
        let args = [ (containerSSA, Pointer); (widgetSSA, Pointer) ]
        let _, zipper1 = witnessGtkCall "gtk_container_add" args Unit zipper
        zipper1, WitnessedVoid
    | _ -> zipper, NotSupported "gtk_container_add requires (container, widget)"

/// gtk_widget_show_all - Show widget and children
let witnessGtkWidgetShowAll (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(widgetSSA, _)] ->
        let args = [ (widgetSSA, Pointer) ]
        let _, zipper1 = witnessGtkCall "gtk_widget_show_all" args Unit zipper
        zipper1, WitnessedVoid
    | _ -> zipper, NotSupported "gtk_widget_show_all requires (widget)"

/// gtk_main - Run GTK main loop
let witnessGtkMain (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    let _, zipper1 = witnessGtkCall "gtk_main" [] Unit zipper
    zipper1, WitnessedVoid

/// gtk_main_quit - Quit GTK main loop
let witnessGtkMainQuit (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    let _, zipper1 = witnessGtkCall "gtk_main_quit" [] Unit zipper
    zipper1, WitnessedVoid

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register GTK bindings for Linux
    // Entry point names match the Bindings.xxx function names
    PlatformDispatch.register Linux X86_64 "gtkInit" witnessGtkInit
    PlatformDispatch.register Linux X86_64 "gtkWindowNew" witnessGtkWindowNew
    PlatformDispatch.register Linux X86_64 "gtkWindowSetTitle" witnessGtkWindowSetTitle
    PlatformDispatch.register Linux X86_64 "gtkWindowSetDefaultSize" witnessGtkWindowSetDefaultSize
    PlatformDispatch.register Linux X86_64 "gtkContainerAdd" witnessGtkContainerAdd
    PlatformDispatch.register Linux X86_64 "gtkWidgetShowAll" witnessGtkWidgetShowAll
    PlatformDispatch.register Linux X86_64 "gtkMain" witnessGtkMain
    PlatformDispatch.register Linux X86_64 "gtkMainQuit" witnessGtkMainQuit
