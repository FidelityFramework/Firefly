/// DynamicLibBindings - Platform bindings for dynamic library loading (dlopen/dlsym)
///
/// ARCHITECTURAL FOUNDATION:
/// These are libc functions for runtime library loading, enabling the
/// dynamic binding pattern where libraries are loaded at runtime rather
/// than linked at compile time.
module Alex.Bindings.DynamicLib.DynamicLibBindings

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
    | Pointer ->
        strSSA, zipper
    | _ ->
        strSSA, zipper

/// Witness a call to a libc function
let witnessLibcCall (funcName: string) (args: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    let argTypes = args |> List.map snd
    let argTypeStrs = argTypes |> List.map Serialize.mlirType
    let retTypeStr = Serialize.mlirType returnType
    let signature = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) retTypeStr

    let zipper1 = MLIRZipper.observeExternFunc funcName signature zipper
    let argSSAs = args |> List.map fst
    MLIRZipper.witnessCall funcName argSSAs argTypes returnType zipper1

// ===================================================================
// Platform Bindings
// ===================================================================

/// dlopen - Open a dynamic library
/// Signature: (path: string, flags: int) -> nativeint (handle)
let witnessDlopen (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(pathSSA, pathTy); (flagsSSA, _)] ->
        // Extract C string pointer from fat string
        let pathPtrSSA, zipper1 = extractStringPointer pathSSA pathTy zipper
        let args = [ (pathPtrSSA, Pointer); (flagsSSA, Integer I32) ]
        let resultSSA, zipper2 = witnessLibcCall "dlopen" args Pointer zipper1
        zipper2, WitnessedValue (resultSSA, Pointer)
    | _ -> zipper, NotSupported "dlopen requires (path: string, flags: int)"

/// dlsym - Look up a symbol in a library
/// Signature: (handle: nativeint, symbol: string) -> nativeint (function pointer)
let witnessDlsym (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(handleSSA, _); (symbolSSA, symbolTy)] ->
        let symbolPtrSSA, zipper1 = extractStringPointer symbolSSA symbolTy zipper
        let args = [ (handleSSA, Pointer); (symbolPtrSSA, Pointer) ]
        let resultSSA, zipper2 = witnessLibcCall "dlsym" args Pointer zipper1
        zipper2, WitnessedValue (resultSSA, Pointer)
    | _ -> zipper, NotSupported "dlsym requires (handle: nativeint, symbol: string)"

/// dlclose - Close a dynamic library
/// Signature: (handle: nativeint) -> int
let witnessDlclose (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(handleSSA, _)] ->
        let args = [ (handleSSA, Pointer) ]
        let resultSSA, zipper1 = witnessLibcCall "dlclose" args (Integer I32) zipper
        zipper1, WitnessedValue (resultSSA, Integer I32)
    | _ -> zipper, NotSupported "dlclose requires (handle: nativeint)"

/// dlerror - Get last error message
/// Signature: () -> nativeint (pointer to error string)
let witnessDlerror (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [] | [(_, Unit)] | [(_, Integer I32)] ->
        let args = []
        let resultSSA, zipper1 = witnessLibcCall "dlerror" args Pointer zipper
        zipper1, WitnessedValue (resultSSA, Pointer)
    | _ -> zipper, NotSupported "dlerror takes no arguments"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register for Linux x86_64
    PlatformDispatch.register Linux X86_64 "dlopen" witnessDlopen
    PlatformDispatch.register Linux X86_64 "dlsym" witnessDlsym
    PlatformDispatch.register Linux X86_64 "dlclose" witnessDlclose
    PlatformDispatch.register Linux X86_64 "dlerror" witnessDlerror

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "dlopen" witnessDlopen
    PlatformDispatch.register Linux ARM64 "dlsym" witnessDlsym
    PlatformDispatch.register Linux ARM64 "dlclose" witnessDlclose
    PlatformDispatch.register Linux ARM64 "dlerror" witnessDlerror
