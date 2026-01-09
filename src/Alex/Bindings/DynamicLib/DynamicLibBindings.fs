/// DynamicLibBindings - Platform bindings for dynamic library loading (dlopen/dlsym)
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.DynamicLib.DynamicLibBindings

open Alex.Dialects.Core.Types
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

// ===================================================================
// Platform Bindings
// ===================================================================

/// dlopen - Open a dynamic library
/// Signature: (path: string, flags: int) -> nativeint (handle)
let bindDlopen (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [path; flags] ->
        let pathPtr, pathOps = extractStringPointer z path.SSA path.Type
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "dlopen"
            [val' pathPtr MLIRTypes.ptr; val' flags.SSA MLIRTypes.i32] MLIRTypes.ptr)
        BoundOps (pathOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })
    | _ ->
        NotSupported "dlopen requires (path: string, flags: int)"

/// dlsym - Look up a symbol in a library
/// Signature: (handle: nativeint, symbol: string) -> nativeint (function pointer)
let bindDlsym (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [handle; symbol] ->
        let symbolPtr, symbolOps = extractStringPointer z symbol.SSA symbol.Type
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "dlsym"
            [val' handle.SSA MLIRTypes.ptr; val' symbolPtr MLIRTypes.ptr] MLIRTypes.ptr)
        BoundOps (symbolOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })
    | _ ->
        NotSupported "dlsym requires (handle: nativeint, symbol: string)"

/// dlclose - Close a dynamic library
/// Signature: (handle: nativeint) -> int
let bindDlclose (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [handle] ->
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "dlclose"
            [val' handle.SSA MLIRTypes.ptr] MLIRTypes.i32)
        BoundOps ([callOp], Some { SSA = resultSSA; Type = MLIRTypes.i32 })
    | _ ->
        NotSupported "dlclose requires (handle: nativeint)"

/// dlerror - Get last error message
/// Signature: () -> nativeint (pointer to error string)
let bindDlerror (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [] | [_] ->  // May receive unit arg
        let resultSSA = freshSynthSSA z
        let callOp = MLIROp.LLVMOp (callFunc (Some resultSSA) "dlerror" [] MLIRTypes.ptr)
        BoundOps ([callOp], Some { SSA = resultSSA; Type = MLIRTypes.ptr })
    | _ ->
        NotSupported "dlerror takes no arguments"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Register for Linux x86_64
    PlatformDispatch.register Linux X86_64 "dlopen" bindDlopen
    PlatformDispatch.register Linux X86_64 "dlsym" bindDlsym
    PlatformDispatch.register Linux X86_64 "dlclose" bindDlclose
    PlatformDispatch.register Linux X86_64 "dlerror" bindDlerror

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "dlopen" bindDlopen
    PlatformDispatch.register Linux ARM64 "dlsym" bindDlsym
    PlatformDispatch.register Linux ARM64 "dlclose" bindDlclose
    PlatformDispatch.register Linux ARM64 "dlerror" bindDlerror
