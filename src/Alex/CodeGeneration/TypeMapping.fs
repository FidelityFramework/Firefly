/// TypeMapping - FNCS NativeType to MLIR type conversion
///
/// Maps FNCS native types to their MLIR representations.
/// Uses structured MLIRType from Alex.Dialects.Core.Types.
///
/// FNCS-native: Uses NativeType from FSharp.Native.Compiler.Checking.Native
module Alex.CodeGeneration.TypeMapping

open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.UnionFind
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Bindings.PlatformTypes

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Native string type: fat pointer = {ptr, len: i64}
let NativeStrType = TStruct [TPtr; TInt I64]

// ═══════════════════════════════════════════════════════════════════════════
// MAIN TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to structured MLIRType with architecture awareness.
/// This is the canonical conversion used throughout Alex.
/// Uses NTU layout information for platform-aware type mapping.
///
/// PRINCIPLED DESIGN (January 2026):
/// PlatformWord types (int, uint, nativeint, size_t, ptrdiff_t) resolve to
/// the actual word size of the target architecture. This is NOT hardcoded to i64!
/// - 64-bit targets (x86_64, ARM64, RISCV64): i64
/// - 32-bit targets (ARM32, RISCV32, WASM32): i32
///
/// The architecture is passed explicitly to ensure correct codegen for all targets.
let rec mapNativeTypeForArch (arch: Architecture) (ty: NativeType) : MLIRType =
    let wordWidth = platformWordWidth arch
    match ty with
    | NativeType.TApp(tycon, args) ->
        // FIRST: Check NTU layout for types that have it - this is the authoritative source
        // for platform-dependent types like int (PlatformWord)
        match tycon.Layout, tycon.NTUKind with
        // Zero-size unit type
        | TypeLayout.Inline (0, 1), Some NTUKind.NTUunit -> TInt I32
        // Boolean: 1-bit
        | TypeLayout.Inline (1, 1), Some NTUKind.NTUbool -> TInt I1
        // Fixed-width integers by NTUKind
        | _, Some NTUKind.NTUint8 -> TInt I8
        | _, Some NTUKind.NTUuint8 -> TInt I8
        | _, Some NTUKind.NTUint16 -> TInt I16
        | _, Some NTUKind.NTUuint16 -> TInt I16
        | _, Some NTUKind.NTUint32 -> TInt I32
        | _, Some NTUKind.NTUuint32 -> TInt I32
        | _, Some NTUKind.NTUint64 -> TInt I64
        | _, Some NTUKind.NTUuint64 -> TInt I64
        // Platform-word integers (int, uint, nativeint, size_t, ptrdiff_t)
        // Size depends on target architecture via platformWordWidth coeffect
        | TypeLayout.PlatformWord, Some NTUKind.NTUint
        | TypeLayout.PlatformWord, Some NTUKind.NTUuint
        | TypeLayout.PlatformWord, Some NTUKind.NTUnint
        | TypeLayout.PlatformWord, Some NTUKind.NTUunint
        | TypeLayout.PlatformWord, Some NTUKind.NTUsize
        | TypeLayout.PlatformWord, Some NTUKind.NTUdiff
        | TypeLayout.PlatformWord, None -> TInt wordWidth  // Platform word resolved per architecture
        // Pointers
        | TypeLayout.PlatformWord, Some NTUKind.NTUptr
        | TypeLayout.PlatformWord, Some NTUKind.NTUfnptr -> TPtr
        | _, Some NTUKind.NTUptr -> TPtr
        // Floats
        | _, Some NTUKind.NTUfloat32 -> TFloat F32
        | _, Some NTUKind.NTUfloat64 -> TFloat F64
        // Char (Unicode codepoint)
        | _, Some NTUKind.NTUchar -> TInt I32
        // String (fat pointer)
        | TypeLayout.FatPointer, Some NTUKind.NTUstring -> NativeStrType
        // SECOND: Name-based fallback for types without proper NTU metadata
        // Note: Arrays have FatPointer layout but no specific NTUKind, handled in fallback
        | _ ->
            match tycon.Name with
            // Byref types: all variants map to pointers
            | "byref" | "inref" | "outref" -> TPtr
            | "Ptr" | "nativeptr" -> TPtr
            | "option" ->
                match args with
                | [innerTy] -> TStruct [TInt I1; mapNativeTypeForArch arch innerTy]
                | _ -> failwithf "option type requires exactly one type argument: %A" ty
            | "voption" ->
                match args with
                | [innerTy] -> TStruct [TInt I1; mapNativeTypeForArch arch innerTy]
                | _ -> failwithf "voption type requires exactly one type argument: %A" ty
            | "result" ->
                // Result<'T, 'E> is a DU with Ok and Error cases
                match args with
                | [okTy; errorTy] -> TStruct [TInt I8; mapNativeTypeForArch arch okTy; mapNativeTypeForArch arch errorTy]
                | _ -> failwithf "result type requires exactly two type arguments: %A" ty
            | "list" -> failwithf "list type not yet implemented: %A" ty
            | "array" | "Array" ->
                // Array<T> is a fat pointer {ptr, len} - same layout as string
                NativeStrType
            | _ ->
                // Check FieldCount for record types
                if tycon.FieldCount > 0 then
                    match tycon.Layout with
                    | TypeLayout.Inline (size, align) when size > 0 && align > 0 ->
                        // Infer struct layout from size/align/fieldCount
                        match size, align, tycon.FieldCount with
                        | 16, 8, 2 -> TStruct [TPtr; TPtr]  // Two pointers
                        | 24, 8, 2 -> TStruct [TPtr; TInt I64; TInt I64]  // Fat pointer (string) + int
                        | 32, 8, 2 -> TStruct [TPtr; TInt I64; TPtr; TInt I64]  // Two fat pointers (strings)
                        | 32, 8, 3 -> TStruct [TPtr; TInt I64; TInt I64]  // Fat pointer + int + padding or ptr
                        | 40, 8, 3 -> TStruct [TPtr; TInt I64; TPtr; TInt I64; TInt I64]  // 3 fat ptrs (24+16)
                        | 48, 8, 3 -> TStruct [TPtr; TInt I64; TPtr; TInt I64; TPtr; TInt I64]  // 3 fat pointers
                        | 56, 8, 3 -> TStruct [TPtr; TInt I64; TPtr; TInt I64; TPtr; TInt I64; TInt I64]  // 3 strings + int
                        | _ ->
                            // Fallback: estimate fields based on word-aligned size
                            let numWords = size / 8
                            TStruct (List.replicate numWords TPtr)
                    | TypeLayout.NTUCompound n when n = tycon.FieldCount ->
                        TStruct (List.replicate n TPtr)
                    | _ ->
                        failwithf "Record '%s' has unsupported layout: %A" tycon.Name tycon.Layout
                else
                    match tycon.Layout with
                    | TypeLayout.Inline (size, align) when size > 8 ->
                        // DU layout: tag + payload, computed from FNCS
                        // Tag size inferred from (size mod align): 1=i8, 2=i16
                        // Payload fills remaining space up to alignment
                        let tagSize = size % align
                        let tagType =
                            match tagSize with
                            | 1 -> TInt I8
                            | 2 -> TInt I16
                            | _ -> TInt I8  // Default to i8 for unusual layouts
                        let payloadSize = size - tagSize
                        let payloadType =
                            match payloadSize with
                            | 1 -> TInt I8
                            | 2 -> TInt I16
                            | 4 -> TInt I32
                            | 8 -> TInt I64
                            | n -> TArray (n, TInt I8)  // Larger payloads as byte array
                        TStruct [tagType; payloadType]
                    | TypeLayout.Inline (size, align) when size > 0 ->
                        failwithf "TApp with unknown Inline layout (%d, %d): %s" size align tycon.Name
                    | TypeLayout.FatPointer ->
                        NativeStrType
                    | TypeLayout.PlatformWord ->
                        // Should have been handled by NTU-aware code at top; this is a fallback
                        TInt wordWidth
                    | TypeLayout.Opaque ->
                        failwithf "TApp with Opaque layout - FNCS must resolve type '%s'" tycon.Name
                    | TypeLayout.Reference _ ->
                        failwithf "Reference type not yet implemented: %s" tycon.Name
                    | TypeLayout.NTUCompound n ->
                        // Arena<'lifetime> and similar compound types: N platform words
                        if n = 1 then TPtr
                        else TStruct (List.replicate n TPtr)
                    | TypeLayout.Inline _ ->
                        failwithf "Unknown inline type '%s' with no fields" tycon.Name

    | NativeType.TFun _ -> TStruct [TPtr; TPtr]  // Function pointer + closure

    | NativeType.TTuple(elements, _) ->
        TStruct (elements |> List.map (mapNativeTypeForArch arch))

    | NativeType.TVar tvar ->
        // Use Union-Find to resolve type variable chains
        match find tvar with
        | (_, Some boundTy) -> mapNativeTypeForArch arch boundTy
        | (root, None) ->
            failwithf "Unbound type variable '%s' - type inference incomplete" root.Name

    | NativeType.TByref _ -> TPtr
    | NativeType.TNativePtr _ -> TPtr
    | NativeType.TForall(_, body) -> mapNativeTypeForArch arch body

    // PRD-14: Lazy<T> - FLAT CLOSURE: { computed: i1, value: T, code_ptr: ptr }
    // Captures are added dynamically at witness time, not in type mapping
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        TStruct [TInt I1; elemMlir; TPtr]  // Flat: just code_ptr, no env_ptr

    // Named records are TApp with FieldCount > 0 - handled in TApp case above

    | NativeType.TUnion (tycon, _cases) ->
        // DU layout should come from NTU via tycon.Layout
        // PLACEHOLDER: This needs proper integration with NTU layout computation
        // The layout (tag size, payload size/alignment) should be pre-computed in FNCS
        // based on platform bindings, not hardcoded here
        //
        // TODO: tycon.Layout should provide the computed struct layout
        // For now, using a placeholder that will cause type errors if mismatched
        failwithf "TUnion '%s' layout not yet integrated with NTU - needs FNCS enhancement" tycon.Name

    | NativeType.TAnon(fields, _) ->
        TStruct (fields |> List.map (fun (_, ty) -> mapNativeTypeForArch arch ty))

    | NativeType.TMeasure _ ->
        failwith "Measure type should have been stripped - this is an FNCS issue"

    | NativeType.TError msg ->
        failwithf "NativeType.TError: %s" msg

/// BACKWARD COMPATIBLE: mapNativeType without explicit architecture
/// Defaults to X86_64 for host compilation. For cross-compilation,
/// callers should use mapNativeTypeForArch explicitly.
let mapNativeType (ty: NativeType) : MLIRType =
    mapNativeTypeForArch X86_64 ty

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH-AWARE TYPE MAPPING (for record types)
// ═══════════════════════════════════════════════════════════════════════════

/// Map a NativeType to MLIRType with architecture awareness, using graph lookup for record field types.
/// This is the principled approach per spec type-representation-architecture.md:
/// record fields are looked up via tryGetRecordFields, not guessed from layout.
/// RECURSIVE: nested record types also use graph lookup.
///
/// PRINCIPLED DESIGN (January 2026):
/// Takes Architecture explicitly to ensure PlatformWord types resolve correctly.
let rec mapNativeTypeWithGraphForArch (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    match ty with
    | NativeType.TApp(tycon, args) when tycon.FieldCount > 0 ->
        // Record type: look up field types from TypeDef
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Map each field type to MLIR RECURSIVELY (nested records also use graph lookup)
            TStruct (fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy))
        | None ->
            // Fallback: shouldn't happen if TypeDef nodes are properly created
            failwithf "Record type '%s' not found in TypeDef nodes - FNCS must create TypeDef for records" tycon.Name
    | NativeType.TApp(tycon, args) ->
        // Non-record TApp (FieldCount = 0) - but check if it might be a record by name lookup
        // This handles cases where FieldCount wasn't preserved in type extraction
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Found record definition - use graph lookup
            TStruct (fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy))
        | None ->
            // Not a record - use standard mapping with architecture
            mapNativeTypeForArch arch ty
    | NativeType.TTuple(elements, _) ->
        // Tuples also need recursive mapping for nested records
        TStruct (elements |> List.map (mapNativeTypeWithGraphForArch arch graph))
    | NativeType.TAnon(fields, _) ->
        // Anonymous records need recursive mapping
        TStruct (fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy))
    // PRD-14: Lazy<T> - FLAT CLOSURE, need recursive mapping in case T is a record
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeWithGraphForArch arch graph elemTy
        TStruct [TInt I1; elemMlir; TPtr]  // Flat: just code_ptr, captures added at witness
    | _ ->
        // Non-record types: use standard mapping with architecture
        mapNativeTypeForArch arch ty

/// BACKWARD COMPATIBLE: mapNativeTypeWithGraph without explicit architecture
/// Defaults to X86_64 for host compilation.
let mapNativeTypeWithGraph (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    mapNativeTypeWithGraphForArch X86_64 graph ty

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
