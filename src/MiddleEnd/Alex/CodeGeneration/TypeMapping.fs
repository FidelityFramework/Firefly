/// TypeMapping - FNCS NativeType to MLIR type conversion
///
/// Maps FNCS native types to their MLIR representations.
/// Uses structured MLIRType from Alex.Dialects.Core.Types.
///
/// FNCS-native: Uses NativeType from FSharp.Native.Compiler.NativeTypedTree
module Alex.CodeGeneration.TypeMapping

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.NativeTypedTree.UnionFind
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// TYPE SIZE COMPUTATION (for DU slot sizing)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute byte size of an MLIRType (for DU payload slot sizing)
/// This is used to determine which payload type is larger for heterogeneous DUs
let rec mlirTypeSize (ty: MLIRType) : int =
    match ty with
    | TInt I1 -> 1
    | TInt I8 -> 1
    | TInt I16 -> 2
    | TInt I32 -> 4
    | TInt I64 -> 8
    | TFloat F32 -> 4
    | TFloat F64 -> 8
    | TFunc _ -> 16  // Function pointer + closure = 2 words
    | TMemRef _ -> 8  // Pointer-sized (dynamic memref)
    | TMemRefStatic _ -> 8  // Pointer-sized (static memref)
    | TMemRefScalar _ -> 8  // Pointer-sized (scalar memref)
    | TVector (_, elemTy) -> mlirTypeSize elemTy  // Vector element size (shape is complex)
    | TIndex -> 8  // Platform word
    | TUnit -> 0
    | TError _ -> 0  // Error types have no runtime size

/// Compute max payload size in bytes for heterogeneous DUs
let maxPayloadBytes (ty1: MLIRType) (ty2: MLIRType) : int =
    max (mlirTypeSize ty1) (mlirTypeSize ty2)

/// Architecture-aware type size computation
/// Uses platform word width for pointer-sized types (TIndex, TIndex, memrefs)
let rec mlirTypeSizeForArch (arch: Architecture) (ty: MLIRType) : int =
    let wordBytes =
        match platformWordWidth arch with
        | I32 -> 4  // 32-bit platforms (ARM32, WASM32, etc.)
        | I64 -> 8  // 64-bit platforms (x64, ARM64, WASM64, etc.)
        | _ -> failwith "Unexpected platform word width for size calculation"
    match ty with
    | TInt I1 -> 1
    | TInt I8 -> 1
    | TInt I16 -> 2
    | TInt I32 -> 4
    | TInt I64 -> 8
    | TFloat F32 -> 4
    | TFloat F64 -> 8
    | TIndex -> wordBytes            // Platform-sized
    | TFunc _ -> 2 * wordBytes       // Function pointer + closure = 2 words
    | TMemRef _ -> wordBytes         // Pointer-sized (dynamic memref)
    | TMemRefStatic _ -> wordBytes   // Pointer-sized (static memref)
    | TMemRefScalar _ -> wordBytes   // Pointer-sized (scalar memref)
    | TVector (_, elemTy) -> mlirTypeSizeForArch arch elemTy
    | TUnit -> 0
    | TError _ -> 0

// ═══════════════════════════════════════════════════════════════════════════
// NTUKind DIRECT MAPPING (for literals)
// ═══════════════════════════════════════════════════════════════════════════

/// Map NTUKind directly to MLIRType with architecture awareness.
/// Used for NativeLiteral where we have the kind without a full NativeType.
///
/// PRINCIPLED DESIGN (January 2026):
/// The NTUKind in a literal IS the type. Platform-dependent kinds
/// (NTUnativeint, NTUint, etc.) resolve based on target architecture.
let mapNTUKindToMLIRType (arch: Architecture) (kind: NTUKind) : MLIRType =
    let wordWidth = platformWordWidth arch
    match kind with
    // Fixed-width signed integers
    | NTUKind.NTUint8 -> TInt I8
    | NTUKind.NTUint16 -> TInt I16
    | NTUKind.NTUint32 -> TInt I32
    | NTUKind.NTUint64 -> TInt I64
    // Fixed-width unsigned integers (same MLIR type, signedness is in ops)
    | NTUKind.NTUuint8 -> TInt I8
    | NTUKind.NTUuint16 -> TInt I16
    | NTUKind.NTUuint32 -> TInt I32
    | NTUKind.NTUuint64 -> TInt I64
    // Platform-word integers - size depends on architecture
    | NTUKind.NTUint      // F# int (platform word)
    | NTUKind.NTUuint     // F# uint (platform word)
        -> TInt wordWidth
    // Native pointer-sized types - map to MLIR index for memref operations
    | NTUKind.NTUnint     // nativeint - used for buffer sizes/offsets
    | NTUKind.NTUunint    // unativeint
    | NTUKind.NTUsize     // size_t
    | NTUKind.NTUdiff     // ptrdiff_t
        -> TIndex
    // Floating point
    | NTUKind.NTUfloat32 -> TFloat F32
    | NTUKind.NTUfloat64 -> TFloat F64
    // Boolean
    | NTUKind.NTUbool -> TInt I1
    // Character (Unicode codepoint = i32)
    | NTUKind.NTUchar -> TInt I32
    // Unit
    | NTUKind.NTUunit -> TInt I32  // Unit represented as i32 0
    // Pointers
    | NTUKind.NTUptr | NTUKind.NTUfnptr -> TIndex
    // String as memref (portable MLIR type, not LLVM struct)
    // memref<?xi8> represents a dynamic-sized buffer with length tracked in descriptor
    | NTUKind.NTUstring -> TMemRef (TInt I8)
    // Composite/complex types - representation comes from platform tier, not here
    | kind -> failwithf "NTUKind %A requires platform-tier resolution, not scalar mapping" kind

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
        // Platform-word integers (int, uint) - size depends on architecture
        | TypeLayout.PlatformWord, Some NTUKind.NTUint
        | TypeLayout.PlatformWord, Some NTUKind.NTUuint
        | TypeLayout.PlatformWord, None -> TInt wordWidth  // Platform word resolved per architecture
        // Native pointer-sized types (nativeint, size_t, etc.) - map to index for memref
        | TypeLayout.PlatformWord, Some NTUKind.NTUnint
        | TypeLayout.PlatformWord, Some NTUKind.NTUunint
        | TypeLayout.PlatformWord, Some NTUKind.NTUsize
        | TypeLayout.PlatformWord, Some NTUKind.NTUdiff
            -> TIndex
        // Pointers
        | TypeLayout.PlatformWord, Some NTUKind.NTUptr
        | TypeLayout.PlatformWord, Some NTUKind.NTUfnptr -> TIndex
        | _, Some NTUKind.NTUptr -> TIndex
        // Floats
        | _, Some NTUKind.NTUfloat32 -> TFloat F32
        | _, Some NTUKind.NTUfloat64 -> TFloat F64
        // Char (Unicode codepoint)
        | _, Some NTUKind.NTUchar -> TInt I32
        // String as memref (portable MLIR type, not LLVM struct)
        // At F# level: string has .Pointer/.Length accessors (FNCS synthetic members)
        // At MLIR level: memref<?xi8> (dynamic buffer)
        // Descriptor (ptr+size) is MLIR's concern, not explicitly modeled here
        | TypeLayout.FatPointer, Some NTUKind.NTUstring -> TMemRef (TInt I8)
        // String with Opaque layout (memref transition - January 2026)
        // After FNCS memref transition, strings use TypeLayout.Opaque instead of FatPointer
        // Both layouts map to the same MLIR type: memref<?xi8>
        | TypeLayout.Opaque, Some NTUKind.NTUstring -> TMemRef (TInt I8)
        // SECOND: Name-based fallback for types without proper NTU metadata
        // Note: Arrays have FatPointer layout but no specific NTUKind, handled in fallback
        | _ ->
            match tycon.Name with
            // Byref types: all variants map to pointers
            | "byref" | "inref" | "outref" -> TIndex
            | "Ptr" | "nativeptr" -> TIndex
            | "option" ->
                // Option is a DU with 2 cases (None, Some) - tag must be i8, not i1
                // DU tags are ALWAYS i8 (or i16 for >256 cases), never boolean
                match args with
                | [innerTy] ->
                    let innerMlir = mapNativeTypeForArch arch innerTy
                    let totalBytes = 1 + mlirTypeSizeForArch arch innerMlir
                    TMemRefStatic(totalBytes, TInt I8)
                | _ -> failwithf "option type requires exactly one type argument: %A" ty
            | "voption" ->
                // ValueOption is a DU with 2 cases (ValueNone, ValueSome) - tag must be i8
                match args with
                | [innerTy] ->
                    let innerMlir = mapNativeTypeForArch arch innerTy
                    let totalBytes = 1 + mlirTypeSizeForArch arch innerMlir
                    TMemRefStatic(totalBytes, TInt I8)
                | _ -> failwithf "voption type requires exactly one type argument: %A" ty
            | "result" ->
                // Result<'T, 'E> is a pointer to arena-allocated case-specific storage
                // Per architecture: "DU values are pointers"
                TIndex
            | "list" ->
                // PRD-13a: list<'T> is a pointer to cons cell (linked list)
                TIndex
            | "array" | "Array" ->
                // Array<T>: Following string migration pattern - use memref descriptor (ptr + len implicit)
                // Phase 2: memref<?xT> represents array with runtime length
                match args with
                | [elemTy] -> TMemRef (mapNativeTypeForArch arch elemTy)
                | _ -> failwithf "array<'T> requires exactly one type argument, got %d" args.Length
            | _ ->
                // Check FieldCount for record types
                if tycon.FieldCount > 0 then
                    // Record types MUST be resolved via graph-aware mapping (mapNativeTypeWithGraphForArch)
                    // If we reach here, it means FNCS didn't provide proper TypeDef metadata
                    // FAIL LOUDLY - no guessing struct layout from size/align heuristics
                    failwithf "Record type '%s' (fields=%d, layout=%A) lacks proper TypeDef metadata - use mapNativeTypeWithGraphForArch or fix FNCS"
                        tycon.Name tycon.FieldCount tycon.Layout
                else
                    match tycon.Layout with
                    | TypeLayout.Inline (size, align) when size > 8 ->
                        // DU layout: FNCS provides size & align - type uses size, allocation uses align
                        // Heterogeneous struct → TMemRefStatic (size, TInt I8)
                        // This is the CORRECT portable representation for WASM and other backends
                        TMemRefStatic (size, TInt I8)
                    | TypeLayout.Inline (size, align) when size > 0 ->
                        failwithf "TApp with unknown Inline layout (%d, %d): %s" size align tycon.Name
                    | TypeLayout.FatPointer ->
                        // FatPointer types should have been handled earlier by NTUKind or name
                        // Strings: TypeLayout.FatPointer + NTUKind.NTUstring → TMemRef (line 151)
                        // Arrays: Name match "array"|"Array" → TMemRef (line 207)
                        // If we reach here, check if it's a string by name (defensive)
                        if tycon.Name.ToLowerInvariant().Contains("string") then
                            // String without proper NTUKind - use memref but warn
                            printfn "WARNING: String type '%s' lacks NTUKind.NTUstring - fix FNCS intrinsic definition" tycon.Name
                            TMemRef <| TInt I8
                        else
                            // Unknown FatPointer type - fail loudly
                            failwithf "FatPointer type '%s' lacks proper NTUKind or name match - fix FNCS metadata" tycon.Name
                    | TypeLayout.PlatformWord ->
                        // Should have been handled by NTU-aware code at top; this is a fallback
                        TInt wordWidth
                    | TypeLayout.Opaque ->
                        failwithf "TApp with Opaque layout - FNCS must resolve type '%s'" tycon.Name
                    | TypeLayout.Reference _ ->
                        failwithf "Reference type not yet implemented: %s" tycon.Name
                    | TypeLayout.NTUCompound n ->
                        // Arena<'lifetime> and similar compound types: N platform words
                        // Phase 2: Use memref array for multiple pointer fields (homogeneous)
                        if n = 1 then TIndex
                        else TMemRefStatic (n, TIndex)  // Array of N indices (portable)
                    | TypeLayout.Inline _ ->
                        failwithf "Unknown inline type '%s' with no fields" tycon.Name

    | NativeType.TFun _ ->
        // Closures: {codePtr: ptr, envPtr: ptr} - homogeneous, use memref array
        // Phase 2: Memref-backed pattern - array of 2 indices (portable, platform-sized)
        // Use TIndex (not TPtr) because index can be memref element type
        TMemRefStatic (2, TIndex)

    | NativeType.TTuple(elements, _) ->
        // Tuple: Product of element types - convert to byte-level memref
        let elemTypes = elements |> List.map (mapNativeTypeForArch arch)
        let totalSize = elemTypes |> List.sumBy (mlirTypeSizeForArch arch)
        TMemRefStatic (totalSize, TInt I8)

    | NativeType.TVar tvar ->
        // Use Union-Find to resolve type variable chains
        match find tvar with
        | (_, Some boundTy) -> mapNativeTypeForArch arch boundTy
        | (root, None) ->
            failwithf "Unbound type variable '%s' - type inference incomplete" root.Name

    | NativeType.TByref _ -> TIndex
    | NativeType.TNativePtr _ -> TIndex
    | NativeType.TForall(_, body) -> mapNativeTypeForArch arch body

    // PRD-14: Lazy<T> - FLAT CLOSURE: { computed: i1, value: T, code_ptr: ptr }
    // Captures are added dynamically at witness time, not in type mapping
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Base layout: i1 + T + ptr - convert to byte-level memref
        let totalSize = mlirTypeSizeForArch arch (TInt I1) + mlirTypeSizeForArch arch elemMlir + mlirTypeSizeForArch arch TIndex
        TMemRefStatic (totalSize, TInt I8)

    // PRD-15: Seq<T> - FLAT CLOSURE: { state: i32, current: T, moveNext_ptr: ptr }
    // Captures are added dynamically at witness time, not in type mapping
    | NativeType.TSeq elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Base layout: i32 + T + ptr - convert to byte-level memref
        let totalSize = mlirTypeSizeForArch arch (TInt I32) + mlirTypeSizeForArch arch elemMlir + mlirTypeSizeForArch arch TIndex
        TMemRefStatic (totalSize, TInt I8)

    // PRD-15/16: SeqEnumerator<T> - mutable iteration state over a seq
    // { seq_ptr: ptr, state: i32, current: T, hasValue: i1 }
    | NativeType.TSeqEnumerator elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Layout: ptr + i32 + T + i1 - convert to byte-level memref
        let totalSize = mlirTypeSizeForArch arch TIndex + mlirTypeSizeForArch arch (TInt I32) + mlirTypeSizeForArch arch elemMlir + mlirTypeSizeForArch arch (TInt I1)
        TMemRefStatic (totalSize, TInt I8)

    // PRD-13a: Immutable collection types - all are reference types (pointer to nodes)
    | NativeType.TList _ -> TIndex  // Pointer to cons cell
    | NativeType.TMap _ -> TIndex   // Pointer to tree root
    | NativeType.TSet _ -> TIndex   // Pointer to tree root

    // Named records are TApp with FieldCount > 0 - handled in TApp case above

    | NativeType.TUnion (tycon, cases) ->
        // DU layout: (tag, payload) where payload accommodates all cases
        // Tag type: i8 for ≤256 cases, i16 for more
        let tagType = if List.length cases <= 256 then TInt I8 else TInt I16

        // Compute max payload size from case field types
        // Each case can have multiple fields (tuple payload) or single field
        let casePayloadTypes =
            cases
            |> List.map (fun case ->
                match case.Fields with
                | [] -> None  // No payload (e.g., None case)
                | [(_, ty)] -> Some (mapNativeTypeForArch arch ty)  // Single field
                | fields ->  // Multiple fields = tuple payload
                    let fieldTypes = fields |> List.map (fun (_, ty) -> mapNativeTypeForArch arch ty)
                    let totalBytes = fieldTypes |> List.sumBy (mlirTypeSizeForArch arch)
                    Some (TMemRefStatic(totalBytes, TInt I8)))

        // Find the "largest" payload type for union storage
        // For now, use the first non-None case's type (proper size comparison would need layout info)
        let payloadType =
            casePayloadTypes
            |> List.choose id
            |> List.tryHead
            |> Option.defaultValue (TInt I8)  // Empty union fallback

        // Convert to byte-level memref: tag + payload
        let totalSize = mlirTypeSizeForArch arch tagType + mlirTypeSizeForArch arch payloadType
        TMemRefStatic (totalSize, TInt I8)

    | NativeType.TAnon(fields, _) ->
        // Anonymous records - convert to byte-level memref
        let fieldTypes = fields |> List.map (fun (_, ty) -> mapNativeTypeForArch arch ty)
        let totalSize = fieldTypes |> List.sumBy (mlirTypeSizeForArch arch)
        TMemRefStatic (totalSize, TInt I8)

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
// FIELD OFFSET CALCULATION (for byte-level memref field access)
// ═══════════════════════════════════════════════════════════════════════════

/// Calculate byte offset for a field within a struct
/// Uses FNCS-provided type structure and arch-aware size computation
let calculateFieldOffsetForArch (arch: Architecture) (nativeType: NativeType) (fieldIndex: int) : int =
    match nativeType with
    | NativeType.TTuple(elements, _) ->
        // Offset = sum of sizes of all fields before fieldIndex
        elements
        |> List.take fieldIndex
        |> List.map (mapNativeTypeForArch arch >> mlirTypeSizeForArch arch)
        |> List.sum

    | NativeType.TAnon(fields, _) ->
        // Offset = sum of sizes of all fields before fieldIndex
        fields
        |> List.take fieldIndex
        |> List.map (snd >> mapNativeTypeForArch arch >> mlirTypeSizeForArch arch)
        |> List.sum

    | NativeType.TLazy elemTy ->
        // Layout: evaluated (I1) | value (elemTy) | thunk (TIndex)
        match fieldIndex with
        | 0 -> 0  // evaluated flag
        | 1 -> mlirTypeSizeForArch arch (TInt I1)  // value after flag
        | 2 -> mlirTypeSizeForArch arch (TInt I1) + mlirTypeSizeForArch arch (mapNativeTypeForArch arch elemTy)  // thunk after value
        | _ -> failwith $"Invalid field index {fieldIndex} for TLazy"

    | NativeType.TSeq elemTy ->
        // Layout: state (I32) | current (elemTy) | moveNext (TIndex)
        match fieldIndex with
        | 0 -> 0  // state
        | 1 -> mlirTypeSizeForArch arch (TInt I32)  // current after state
        | 2 -> mlirTypeSizeForArch arch (TInt I32) + mlirTypeSizeForArch arch (mapNativeTypeForArch arch elemTy)  // moveNext after current
        | _ -> failwith $"Invalid field index {fieldIndex} for TSeq"

    | NativeType.TSeqEnumerator elemTy ->
        // Layout: source (TIndex) | index (I32) | current (elemTy) | hasValue (I1)
        match fieldIndex with
        | 0 -> 0  // source
        | 1 -> mlirTypeSizeForArch arch TIndex  // index after source
        | 2 -> mlirTypeSizeForArch arch TIndex + mlirTypeSizeForArch arch (TInt I32)  // current after index
        | 3 -> mlirTypeSizeForArch arch TIndex + mlirTypeSizeForArch arch (TInt I32) + mlirTypeSizeForArch arch (mapNativeTypeForArch arch elemTy)  // hasValue after current
        | _ -> failwith $"Invalid field index {fieldIndex} for TSeqEnumerator"

    | NativeType.TUnion (_, cases) ->
        // Layout: tag | payload (max size of all cases)
        match fieldIndex with
        | 0 -> 0  // tag at offset 0
        | 1 ->
            // Payload offset = tag size
            let tagType = if List.length cases <= 256 then TInt I8 else TInt I16
            mlirTypeSizeForArch arch tagType
        | _ -> failwith $"Invalid field index {fieldIndex} for TUnion"

    | NativeType.TApp ({ Name = name }, _) when name = "Closure" || name = "FunctionPointer" ->
        // Layout: codePtr (TIndex) | closure (TIndex)
        match fieldIndex with
        | 0 -> 0  // codePtr
        | 1 -> mlirTypeSizeForArch arch TIndex  // closure after codePtr
        | _ -> failwith $"Invalid field index {fieldIndex} for {name}"

    | NativeType.TFun _ ->
        // TFun is closures: {codePtr, envPtr} - same as Closure
        match fieldIndex with
        | 0 -> 0  // codePtr
        | 1 -> mlirTypeSizeForArch arch TIndex  // envPtr after codePtr
        | _ -> failwith $"Invalid field index {fieldIndex} for TFun"

    | _ -> failwith $"Cannot calculate field offset for type {nativeType} - not a struct type"

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
            let fieldTypes = fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy)
            let totalBytes = fieldTypes |> List.sumBy (mlirTypeSizeForArch arch)
            TMemRefStatic(totalBytes, TInt I8)
        | None ->
            // Fallback: shouldn't happen if TypeDef nodes are properly created
            failwithf "Record type '%s' not found in TypeDef nodes - FNCS must create TypeDef for records" tycon.Name
    | NativeType.TApp(tycon, args) ->
        // Non-record TApp (FieldCount = 0) - but check if it might be a record by name lookup
        // This handles cases where FieldCount wasn't preserved in type extraction
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Found record definition - use graph lookup
            let fieldTypes = fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy)
            let totalBytes = fieldTypes |> List.sumBy (mlirTypeSizeForArch arch)
            TMemRefStatic(totalBytes, TInt I8)
        | None ->
            // Not a record - use standard mapping with architecture
            mapNativeTypeForArch arch ty
    | NativeType.TTuple(elements, _) ->
        // Tuples also need recursive mapping for nested records
        let elementTypes = elements |> List.map (mapNativeTypeWithGraphForArch arch graph)
        let totalBytes = elementTypes |> List.sumBy (mlirTypeSizeForArch arch)
        TMemRefStatic(totalBytes, TInt I8)
    | NativeType.TAnon(fields, _) ->
        // Anonymous records need recursive mapping
        let fieldTypes = fields |> List.map (fun (_, fieldTy) -> mapNativeTypeWithGraphForArch arch graph fieldTy)
        let totalBytes = fieldTypes |> List.sumBy (mlirTypeSizeForArch arch)
        TMemRefStatic(totalBytes, TInt I8)
    // PRD-14: Lazy<T> - FLAT CLOSURE, need recursive mapping in case T is a record
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeWithGraphForArch arch graph elemTy
        let totalBytes = 1 + mlirTypeSizeForArch arch elemMlir + mlirTypeSizeForArch arch TIndex
        TMemRefStatic(totalBytes, TInt I8)  // Flat: just code_ptr, captures added at witness
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
/// DEPRECATED: Use extractPtrElementTypeWithGraph for correct record handling
let extractPtrElementType (ty: NativeType) : MLIRType option =
    match ty with
    | NativeType.TApp(tycon, [elemTy]) when tycon.Name = "nativeptr" || tycon.Name = "Ptr" ->
        Some (mapNativeType elemTy)
    | NativeType.TNativePtr elemTy ->
        Some (mapNativeType elemTy)
    | _ -> None

/// Extract element type from a pointer type with graph-aware mapping
/// CANONICAL: Use this version for correct record type handling
let extractPtrElementTypeWithGraph (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType option =
    match ty with
    | NativeType.TApp(tycon, [elemTy]) when tycon.Name = "nativeptr" || tycon.Name = "Ptr" ->
        Some (mapNativeTypeWithGraphForArch arch graph elemTy)
    | NativeType.TNativePtr elemTy ->
        Some (mapNativeTypeWithGraphForArch arch graph elemTy)
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

/// Check if a type is an index type (platform-sized pointer/address)
let isIndex (mlirType: string) : bool =
    mlirType = "index"

/// Get the bit width of an integer type
let integerBitWidth (mlirType: string) : int option =
    match mlirType with
    | "i1" -> Some 1
    | "i8" -> Some 8
    | "i16" -> Some 16
    | "i32" -> Some 32
    | "i64" -> Some 64
    | _ -> None
