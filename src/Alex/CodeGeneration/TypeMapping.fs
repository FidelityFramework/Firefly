/// TypeMapping - FNCS NativeType to MLIR type conversion
///
/// Maps FNCS native types to their MLIR representations.
/// Handles primitives, functions, tuples, options, lists, and arrays.
///
/// FNCS-native: Uses NativeType from FSharp.Native.Compiler.Checking.Native
///
/// NTU Collection Architecture:
/// - FatPointer: ptr + length, both platform-word sized
/// - NTUCompound: n × platform-word sized components
/// - These are resolved via platform context (quotations)
module Alex.CodeGeneration.TypeMapping

open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes

//-------------------------------------------------------------------------
// Platform Context for NTU Layout Resolution
//-------------------------------------------------------------------------

/// Platform context carries quotation-resolved values
/// Currently defaults to x86_64 (64-bit words)
type PlatformContext = {
    /// Word size in bits (32 or 64)
    WordSize: int
    /// Pointer alignment in bytes
    PointerAlign: int
}

/// Default platform context for x86_64
let defaultPlatform : PlatformContext = {
    WordSize = 64
    PointerAlign = 8
}

/// Resolve a TypeLayout to concrete size and alignment
/// Uses platform quotations for NTU types
let resolveLayout (ctx: PlatformContext) (layout: TypeLayout) : (int * int) option =
    match layout with
    | TypeLayout.Inline(size, align) when size >= 0 && align > 0 ->
        Some (size, align)
    | TypeLayout.Inline _ ->
        // Unknown size - cannot resolve statically
        None
    | TypeLayout.PlatformWord ->
        // Platform word: wordSize bits, wordSize/8 alignment
        let bytes = ctx.WordSize / 8
        Some (bytes, bytes)
    | TypeLayout.FatPointer ->
        // Fat pointer: ptr + length, both platform-word sized
        // Total = 2 × wordSize, aligned to word
        let wordBytes = ctx.WordSize / 8
        Some (wordBytes * 2, wordBytes)
    | TypeLayout.NTUCompound n ->
        // NTU compound: n × platform-word components
        let wordBytes = ctx.WordSize / 8
        Some (wordBytes * n, wordBytes)
    | TypeLayout.Reference _ ->
        // Reference types are pointer-sized
        let bytes = ctx.WordSize / 8
        Some (bytes, bytes)
    | TypeLayout.Opaque ->
        // Cannot resolve opaque layouts
        None

/// Get MLIR integer type for platform word
let platformWordType (ctx: PlatformContext) : string =
    match ctx.WordSize with
    | 32 -> "i32"
    | 64 -> "i64"
    | n -> sprintf "i%d" n

/// Get MLIR struct type for fat pointer (ptr + length)
let fatPointerType (ctx: PlatformContext) : string =
    let wordType = platformWordType ctx
    sprintf "!llvm.struct<(ptr, %s)>" wordType

/// Convert a FNCS NativeType to its MLIR representation
let rec nativeTypeToMLIR (ty: NativeType) : string =
    match ty with
    // Function types: domain -> range
    | NativeType.TFun(domain, range) ->
        let paramType = nativeTypeToMLIR domain
        let retType = nativeTypeToMLIR range
        sprintf "(%s) -> %s" paramType retType

    // Tuple types
    | NativeType.TTuple(elements, _isStruct) ->
        let elemTypes =
            elements
            |> List.map nativeTypeToMLIR
            |> String.concat ", "
        sprintf "tuple<%s>" elemTypes

    // Byref types (pointer-like)
    | NativeType.TByref(_, _) ->
        "!llvm.ptr"

    // Native pointers
    | NativeType.TNativePtr _ ->
        "!llvm.ptr"

    // Type applications (e.g., int, string, option<'T>, list<'T>)
    | NativeType.TApp(conRef, args) ->
        mapTypeApp conRef args

    // Type variables (generic parameters)
    | NativeType.TVar _ ->
        // For now, assume generic types are pointers (common in Alloy)
        "!llvm.ptr"

    // Forall types (polymorphic)
    | NativeType.TForall(_, body) ->
        // Instantiate body (generic erased at runtime)
        nativeTypeToMLIR body

    // Measure types (used for UMX phantom types)
    | NativeType.TMeasure _ ->
        // Measures are erased at runtime, use the underlying type
        "i32"

    // Anonymous record types
    | NativeType.TAnon(fields, _isStruct) ->
        let fieldTypes =
            fields
            |> List.map (fun (_, ty) -> nativeTypeToMLIR ty)
            |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldTypes

    // Record types
    | NativeType.TRecord(_, fields) ->
        let fieldTypes =
            fields
            |> List.map (fun (_, ty) -> nativeTypeToMLIR ty)
            |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldTypes

    // Union types (discriminated unions)
    | NativeType.TUnion(_, _cases) ->
        // For now, represent as tagged union (tag + max payload)
        "!llvm.struct<(i32, i64)>"

    // Error types (shouldn't reach code generation)
    | NativeType.TError _ ->
        "i32"

/// Map a type constructor application to MLIR
/// Uses default platform context (x86_64) for NTU type resolution
and mapTypeApp (conRef: TypeConRef) (args: NativeType list) : string =
    mapTypeAppWithContext defaultPlatform conRef args

/// Map a type constructor with explicit platform context
/// Prefers NTUKind discrimination over string matching for type resolution
and mapTypeAppWithContext (ctx: PlatformContext) (conRef: TypeConRef) (args: NativeType list) : string =
    // First, check if we have NTUKind metadata - this is the authoritative source
    match conRef.NTUKind with
    | Some kind -> mapNTUKind ctx kind args
    | None -> mapByName ctx conRef args

/// Map NTUKind to MLIR type - authoritative NTU resolution
and mapNTUKind (ctx: PlatformContext) (kind: NTUKind) (args: NativeType list) : string =
    match kind with
    // Platform-word sized integers (resolved via platform quotations)
    | NTUKind.NTUint -> platformWordType ctx   // Signed platform word
    | NTUKind.NTUuint -> platformWordType ctx  // Unsigned platform word
    | NTUKind.NTUnint -> platformWordType ctx  // Native int (pointer-sized)
    | NTUKind.NTUunint -> platformWordType ctx // Native uint

    // Size types (like size_t, ptrdiff_t)
    | NTUKind.NTUsize -> platformWordType ctx  // size_t equivalent
    | NTUKind.NTUdiff -> platformWordType ctx  // ptrdiff_t equivalent

    // Fixed-width integers
    | NTUKind.NTUint8 -> "i8"
    | NTUKind.NTUint16 -> "i16"
    | NTUKind.NTUint32 -> "i32"
    | NTUKind.NTUint64 -> "i64"
    | NTUKind.NTUuint8 -> "i8"
    | NTUKind.NTUuint16 -> "i16"
    | NTUKind.NTUuint32 -> "i32"
    | NTUKind.NTUuint64 -> "i64"

    // Floating point
    | NTUKind.NTUfloat32 -> "f32"
    | NTUKind.NTUfloat64 -> "f64"

    // Pointer type (no payload - union case takes no data)
    | NTUKind.NTUptr -> "!llvm.ptr"

    // String type (fat pointer: ptr + length)
    | NTUKind.NTUstring -> fatPointerType ctx

    // Boolean
    | NTUKind.NTUbool -> "i1"

    // Unit (void)
    | NTUKind.NTUunit -> "()"

    // Character (Unicode codepoint, 32-bit)
    | NTUKind.NTUchar -> "i32"

    // Decimal (128-bit, represented as struct of two i64)
    | NTUKind.NTUdecimal -> "!llvm.struct<(i64, i64)>"

    // Compound value types
    | NTUKind.NTUuuid -> "!llvm.struct<(i64, i64)>"  // 128-bit UUID as two i64
    | NTUKind.NTUdatetime -> "i64"  // 64-bit ticks
    | NTUKind.NTUtimespan -> "i64"  // 64-bit duration in ticks

    // Other types - fallback to pointer (most non-primitive types are reference-like)
    | NTUKind.NTUother -> "!llvm.ptr"

/// Fallback mapping by name for non-NTU types
and mapByName (ctx: PlatformContext) (conRef: TypeConRef) (args: NativeType list) : string =
    match conRef.Name with
    // Legacy fixed-width integral types (for backwards compat)
    | "int32" | "Int32" -> "i32"
    | "int64" | "Int64" -> "i64"
    | "int16" | "Int16" -> "i16"
    | "int8" | "sbyte" | "SByte" -> "i8"
    | "uint32" | "UInt32" -> "i32"
    | "uint64" | "UInt64" -> "i64"
    | "uint16" | "UInt16" -> "i16"
    | "uint8" | "byte" | "Byte" -> "i8"

    // Boolean
    | "bool" | "Boolean" -> "i1"

    // Floating point
    | "float32" | "Single" -> "f32"
    | "float" | "float64" | "Double" -> "f64"

    // Strings - fallback for types without NTUKind
    | "string" | "String" -> fatPointerType ctx

    // Native pointers
    | "nativeint" | "IntPtr" -> "!llvm.ptr"
    | "unativeint" | "UIntPtr" -> "!llvm.ptr"
    | "nativeptr" | "Ptr" -> "!llvm.ptr"

    // Char (Unicode codepoint)
    | "char" | "Char" -> "i32"

    // Unit / Void
    | "unit" | "Unit" | "Void" -> "()"

    // Option type - value type in native (tag + payload)
    | "option" | "Option" | "voption" | "ValueOption" ->
        let innerType =
            match args with
            | [arg] -> nativeTypeToMLIR arg
            | _ -> "i32"
        sprintf "!llvm.struct<(i1, %s)>" innerType  // tag + payload

    // Result type
    | "Result" ->
        "!llvm.struct<(i32, i64, i64)>"  // tag + ok_payload + error_payload

    // List type
    | "list" | "List" ->
        "!llvm.ptr"  // Lists are pointers to cons cells

    // Array type - fat pointer (ptr + length, both platform-word sized)
    // Uses FatPointer layout, resolved via platform context
    | "array" | "Array" ->
        // Array is a fat pointer regardless of element type
        // The element type is used for pointer arithmetic, not storage
        fatPointerType ctx

    // Span/ReadOnlySpan - also fat pointers (non-owning views)
    | "Span" | "ReadOnlySpan" ->
        fatPointerType ctx

    // StackBuffer - stack-allocated array, fat pointer representation
    | "StackBuffer" ->
        fatPointerType ctx

    // Default fallback
    | _ ->
        // Check if it's a pointer-like type by name
        if conRef.Name.Contains("ptr") || conRef.Name.Contains("Ptr") ||
           conRef.Name.Contains("Buffer") || conRef.Name.Contains("Span") then
            "!llvm.ptr"
        else
            "i32"

/// Extract return type from a function type
/// Walks through curried function type to get the final return type
let getReturnType (ty: NativeType) : string =
    let rec getReturn t =
        match t with
        | NativeType.TFun(_, range) -> getReturn range
        | _ -> t
    nativeTypeToMLIR (getReturn ty)

/// Extract parameter types from a function type
/// Returns list of MLIR types for each curried parameter
let getParamTypes (ty: NativeType) : string list =
    let rec extractParams funcType acc =
        match funcType with
        | NativeType.TFun(domain, range) ->
            let paramType = nativeTypeToMLIR domain
            extractParams range (paramType :: acc)
        | _ ->
            List.rev acc
    extractParams ty []

/// Check if a type is a primitive MLIR type (i32, i64, f32, etc.)
let isPrimitive (mlirType: string) : bool =
    match mlirType with
    | "i1" | "i8" | "i16" | "i32" | "i64" -> true
    | "f32" | "f64" -> true
    | "()" -> true
    | _ -> false

/// Check if a type is an integer type
let isInteger (mlirType: string) : bool =
    match mlirType with
    | "i1" | "i8" | "i16" | "i32" | "i64" -> true
    | _ -> false

/// Check if a type is a floating-point type
let isFloat (mlirType: string) : bool =
    match mlirType with
    | "f32" | "f64" -> true
    | _ -> false

/// Check if a type is a pointer type
let isPointer (mlirType: string) : bool =
    mlirType = "!llvm.ptr" || mlirType.StartsWith("!llvm.ptr")

/// Get the bit width of an integer type
let integerBitWidth (mlirType: string) : int option =
    match mlirType with
    | "i1" -> Some 1
    | "i8" -> Some 8
    | "i16" -> Some 16
    | "i32" -> Some 32
    | "i64" -> Some 64
    | _ -> None

/// Get the appropriate zero constant for a type
let zeroConstant (mlirType: string) : string =
    match mlirType with
    | "i1" -> "0"
    | "i8" | "i16" | "i32" | "i64" -> "0"
    | "f32" -> "0.0"
    | "f64" -> "0.0"
    | _ -> "0"

/// Get the appropriate one constant for a type
let oneConstant (mlirType: string) : string =
    match mlirType with
    | "i1" -> "1"
    | "i8" | "i16" | "i32" | "i64" -> "1"
    | "f32" -> "1.0"
    | "f64" -> "1.0"
    | _ -> "1"

// ═══════════════════════════════════════════════════════════════════════════
// NativeType to MLIRType Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type
/// This is the canonical conversion used throughout Alex
let rec mapNativeType (ty: NativeType) : MLIRType =
    match ty with
    // Applied types - match on type constructor name and layout
    | NativeType.TApp(tycon, _args) ->
        match tycon.Name with
        | "unit" -> Unit
        | "bool" -> Integer I1
        | "int8" | "sbyte" -> Integer I8
        | "uint8" | "byte" -> Integer I8
        | "int16" -> Integer I16
        | "uint16" -> Integer I16
        | "int" | "int32" -> Integer I32
        | "uint" | "uint32" -> Integer I32
        | "int64" -> Integer I64
        | "uint64" -> Integer I64
        | "nativeint" -> Integer I64  // Platform dependent
        | "unativeint" -> Integer I64
        | "float32" | "single" -> Float F32
        | "float" | "double" -> Float F64
        | "char" -> Integer I32  // Unicode codepoint
        | "string" -> NativeStrType  // Fat pointer {ptr: *u8, len: i64}
        | "Ptr" | "nativeptr" -> Pointer
        | "array" -> Pointer
        | "list" -> Pointer
        | "option" | "voption" -> Pointer  // TODO: Value type layout
        | _ ->
            // Check if this is a user-defined discriminated union
            // DUs have Layout.Inline with size > word size (tag + payload)
            // For now, detect by Inline layout pattern (9, 8) = 1-byte tag + 8-byte payload
            match tycon.Layout with
            | TypeLayout.Inline (size, _align) when size > 8 ->
                // User-defined DU - use tagged union struct
                // We use i32 for tag (for easier comparison) and i64 for payload
                Struct [Integer I32; Integer I64]
            | _ -> Pointer  // Default to pointer for other unknown types

    // Function types
    | NativeType.TFun _ -> Pointer  // Function pointer + closure

    // Tuple types
    | NativeType.TTuple _ -> Pointer  // TODO: Proper struct layout

    // Type variables (erased)
    | NativeType.TVar _ -> Pointer

    // Byref types
    | NativeType.TByref _ -> Pointer

    // Native pointers
    | NativeType.TNativePtr _ -> Pointer

    // Forall types - look at body
    | NativeType.TForall(_, body) -> mapNativeType body

    // Record types
    | NativeType.TRecord _ -> Pointer

    // Union types - tagged union struct (tag: i32, payload: i64)
    | NativeType.TUnion _ -> Struct [Integer I32; Integer I64]

    // Anonymous record types
    | NativeType.TAnon _ -> Pointer

    // Measure types - strip measure
    | NativeType.TMeasure _ -> Integer I32  // Usually applied to a numeric type

    // Error type
    | NativeType.TError _ -> Pointer
