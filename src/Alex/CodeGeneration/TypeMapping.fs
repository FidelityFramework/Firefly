/// TypeMapping - FNCS NativeType to MLIR type conversion
///
/// Maps FNCS native types to their MLIR representations.
/// Uses structured MLIRType from Alex.Dialects.Core.Types.
///
/// FNCS-native: Uses NativeType from FSharp.Native.Compiler.Checking.Native
module Alex.CodeGeneration.TypeMapping

open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.UnionFind
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Native string type: fat pointer = {ptr, len: i64}
let NativeStrType = TStruct [TPtr; TInt I64]

// ═══════════════════════════════════════════════════════════════════════════
// MAIN TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to structured MLIRType
/// This is the canonical conversion used throughout Alex
let rec mapNativeType (ty: NativeType) : MLIRType =
    match ty with
    | NativeType.TApp(tycon, args) ->
        match tycon.Name with
        | "unit" -> TInt I32  // Unit represented as i32 (returns 0)
        | "bool" -> TInt I1
        | "int8" | "sbyte" -> TInt I8
        | "uint8" | "byte" -> TInt I8
        | "int16" -> TInt I16
        | "uint16" -> TInt I16
        | "int" | "int32" -> TInt I32
        | "uint" | "uint32" -> TInt I32
        | "int64" -> TInt I64
        | "uint64" -> TInt I64
        | "nativeint" -> TPtr
        | "unativeint" -> TPtr
        | "float32" | "single" -> TFloat F32
        | "float" | "double" -> TFloat F64
        | "char" -> TInt I32  // Unicode codepoint
        | "string" -> NativeStrType
        | "Ptr" | "nativeptr" -> TPtr
        | "array" -> NativeStrType  // Fat pointer {ptr, len}
        | "list" -> failwithf "list type not yet implemented: %A" ty
        | "option" ->
            match args with
            | [innerTy] -> TStruct [TInt I1; mapNativeType innerTy]
            | _ -> failwithf "option type requires exactly one type argument: %A" ty
        | "voption" ->
            match args with
            | [innerTy] -> TStruct [TInt I1; mapNativeType innerTy]
            | _ -> failwithf "voption type requires exactly one type argument: %A" ty
        | _ ->
            // Check FieldCount for record types
            if tycon.FieldCount > 0 then
                match tycon.Layout with
                | TypeLayout.Inline (size, align) when size > 0 && align > 0 ->
                    if size = 16 && align = 8 && tycon.FieldCount = 2 then
                        TStruct [TPtr; TPtr]
                    else
                        failwithf "Record '%s' with %d fields needs SemanticGraph for exact types (layout: %d, %d)"
                            tycon.Name tycon.FieldCount size align
                | TypeLayout.NTUCompound n when n = tycon.FieldCount ->
                    TStruct (List.replicate n TPtr)
                | _ ->
                    failwithf "Record '%s' has unsupported layout: %A" tycon.Name tycon.Layout
            else
                match tycon.Layout with
                | TypeLayout.Inline (size, _) when size > 8 ->
                    TStruct [TInt I32; TInt I64]  // Tagged union
                | TypeLayout.Inline (size, align) when size > 0 ->
                    failwithf "TApp with unknown Inline layout (%d, %d): %s" size align tycon.Name
                | TypeLayout.FatPointer ->
                    NativeStrType
                | TypeLayout.PlatformWord ->
                    match tycon.NTUKind with
                    | Some NTUKind.NTUptr | Some NTUKind.NTUfnptr -> TPtr
                    | Some NTUKind.NTUint | Some NTUKind.NTUuint
                    | Some NTUKind.NTUnint | Some NTUKind.NTUunint
                    | Some NTUKind.NTUsize | Some NTUKind.NTUdiff -> TIndex
                    | None -> TIndex
                    | Some kind -> failwithf "Unexpected NTUKind %A with PlatformWord layout: %s" kind tycon.Name
                | TypeLayout.Opaque ->
                    failwithf "TApp with Opaque layout - FNCS must resolve type '%s'" tycon.Name
                | TypeLayout.Reference _ ->
                    failwithf "Reference type not yet implemented: %s" tycon.Name
                | TypeLayout.NTUCompound n ->
                    if n = 2 then TStruct [TPtr; TPtr]
                    elif n = 1 then TPtr
                    else failwithf "NTUCompound(%d) not yet implemented for %s" n tycon.Name
                | TypeLayout.Inline _ ->
                    failwithf "Unknown inline type '%s' with no fields" tycon.Name

    | NativeType.TFun _ -> TStruct [TPtr; TPtr]  // Function pointer + closure

    | NativeType.TTuple(elements, _) ->
        TStruct (elements |> List.map mapNativeType)

    | NativeType.TVar tvar ->
        // Use Union-Find to resolve type variable chains
        match find tvar with
        | (_, Some boundTy) -> mapNativeType boundTy
        | (root, None) ->
            failwithf "Unbound type variable '%s' - type inference incomplete" root.Name

    | NativeType.TByref _ -> TPtr
    | NativeType.TNativePtr _ -> TPtr
    | NativeType.TForall(_, body) -> mapNativeType body

    | NativeType.TRecord(tc, _) ->
        failwithf "Unexpected TRecord '%s' - use TApp with tycon.Fields" tc.Name

    | NativeType.TUnion _ -> TStruct [TInt I32; TInt I64]  // Tagged union

    | NativeType.TAnon(fields, _) ->
        TStruct (fields |> List.map (fun (_, ty) -> mapNativeType ty))

    | NativeType.TMeasure _ ->
        failwith "Measure type should have been stripped - this is an FNCS issue"

    | NativeType.TError msg ->
        failwithf "NativeType.TError: %s" msg

// ═══════════════════════════════════════════════════════════════════════════
// STRING-BASED TYPE MAPPING (for legacy code)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert FNCS NativeType to MLIR type string
/// Uses Serialize module for structured type → string conversion
let nativeTypeToMLIR (ty: NativeType) : string =
    let mlirType = mapNativeType ty
    Alex.Dialects.Core.Serialize.typeToString mlirType

/// Map a type constructor application to MLIR string
let mapTypeApp (conRef: TypeConRef) (args: NativeType list) : string =
    nativeTypeToMLIR (NativeType.TApp(conRef, args))

/// Extract element type from a pointer type (nativeptr<T> → T)
/// Returns the MLIR type of the element, or None if not a pointer type
let extractPtrElementType (ty: NativeType) : MLIRType option =
    match ty with
    | NativeType.TApp(tycon, [elemTy]) when tycon.Name = "nativeptr" || tycon.Name = "Ptr" ->
        Some (mapNativeType elemTy)
    | NativeType.TNativePtr elemTy ->
        Some (mapNativeType elemTy)
    | _ -> None

/// Extract return type from a function type as string
let getReturnType (ty: NativeType) : string =
    let rec getReturn t =
        match t with
        | NativeType.TFun(_, range) -> getReturn range
        | _ -> t
    nativeTypeToMLIR (getReturn ty)

/// Extract parameter types from a function type as strings
let getParamTypes (ty: NativeType) : string list =
    let rec extractParams funcType acc =
        match funcType with
        | NativeType.TFun(domain, range) ->
            let paramType = nativeTypeToMLIR domain
            extractParams range (paramType :: acc)
        | _ ->
            List.rev acc
    extractParams ty []

// ═══════════════════════════════════════════════════════════════════════════
// TYPE PREDICATES
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type is a primitive MLIR type
let isPrimitive (mlirType: string) : bool =
    match mlirType with
    | "i1" | "i8" | "i16" | "i32" | "i64" -> true
    | "f32" | "f64" -> true
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
