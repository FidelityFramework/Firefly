/// TypeSizing - Compute byte sizes from MLIR type strings
///
/// This module computes the byte size of types from their MLIR representation.
/// Used by witnesses that need to allocate stack space or pass size parameters.
///
/// Architecture Note:
/// This is a temporary solution using direct NTU → MLIR type → size mapping.
/// When BAREWire is refactored to be NTU-native (Phase 3), type sizing will
/// migrate to BAREWire as the canonical source for type layout information.
///
/// The mapping here mirrors NTU type layouts:
/// - Fixed integers: i8=1, i16=2, i32=4, i64=8
/// - Floats: f32=4, f64=8
/// - Pointers: 8 bytes (x86_64)
/// - Fat pointers (string, array): 16 bytes (ptr + length)
/// - Structs: sum of field sizes (no padding in MLIR packed structs)
module Alex.CodeGeneration.TypeSizing

open System

/// Platform word size in bytes (x86_64 = 8)
let private wordSize = 8L

/// Compute byte size from MLIR type string
/// Returns the size in bytes for stack allocation and memcpy operations
let rec computeSize (mlirType: string) : int64 =
    let trimmed = mlirType.Trim()
    match trimmed with
    // Boolean - 1 byte
    | "i1" -> 1L

    // Fixed-width integers
    | "i8" -> 1L
    | "i16" -> 2L
    | "i32" -> 4L
    | "i64" -> 8L

    // Floating point
    | "f32" -> 4L
    | "f64" -> 8L

    // Pointers - platform word size
    | "!llvm.ptr" -> wordSize
    | s when s.StartsWith("!llvm.ptr<") -> wordSize

    // Unit/void - zero size
    | "()" -> 0L

    // Struct types - sum of field sizes
    | s when s.StartsWith("!llvm.struct<(") && s.EndsWith(")>") ->
        parseStructSize s

    // Array types - count * element size
    | s when s.StartsWith("!llvm.array<") && s.EndsWith(">") ->
        parseArraySize s

    // Tuple types (from nativeTypeToMLIR)
    | s when s.StartsWith("tuple<") && s.EndsWith(">") ->
        parseTupleSize s

    // Unknown type - fail explicitly (no silent fallback)
    | _ ->
        failwithf "TypeSizing.computeSize: Unknown MLIR type '%s'" mlirType

/// Parse struct size from "!llvm.struct<(type1, type2, ...)>"
/// Sums the sizes of all fields
and parseStructSize (s: string) : int64 =
    // Extract content between "!llvm.struct<(" and ")>"
    let prefix = "!llvm.struct<("
    let suffix = ")>"

    if not (s.StartsWith(prefix) && s.EndsWith(suffix)) then
        failwithf "TypeSizing.parseStructSize: Invalid struct format '%s'" s

    let content = s.Substring(prefix.Length, s.Length - prefix.Length - suffix.Length)

    if String.IsNullOrWhiteSpace(content) then
        0L  // Empty struct
    else
        // Split by comma, handling nested types
        let fields = splitTypeList content
        fields |> List.sumBy computeSize

/// Parse array size from "!llvm.array<N x elementType>"
and parseArraySize (s: string) : int64 =
    // Extract content between "!llvm.array<" and ">"
    let prefix = "!llvm.array<"
    let suffix = ">"

    if not (s.StartsWith(prefix) && s.EndsWith(suffix)) then
        failwithf "TypeSizing.parseArraySize: Invalid array format '%s'" s

    let content = s.Substring(prefix.Length, s.Length - prefix.Length - suffix.Length)

    // Format: "N x elementType"
    let parts = content.Split([|" x "|], StringSplitOptions.None)
    if parts.Length <> 2 then
        failwithf "TypeSizing.parseArraySize: Expected 'N x type' format, got '%s'" content

    let count = Int64.Parse(parts.[0].Trim())
    let elementType = parts.[1].Trim()
    let elementSize = computeSize elementType

    count * elementSize

/// Parse tuple size from "tuple<type1, type2, ...>"
and parseTupleSize (s: string) : int64 =
    let prefix = "tuple<"
    let suffix = ">"

    if not (s.StartsWith(prefix) && s.EndsWith(suffix)) then
        failwithf "TypeSizing.parseTupleSize: Invalid tuple format '%s'" s

    let content = s.Substring(prefix.Length, s.Length - prefix.Length - suffix.Length)

    if String.IsNullOrWhiteSpace(content) then
        0L
    else
        let elements = splitTypeList content
        elements |> List.sumBy computeSize

/// Split a comma-separated type list, respecting nested angle brackets
/// "!llvm.ptr, i64" → ["!llvm.ptr"; "i64"]
/// "!llvm.struct<(i32, i64)>, i8" → ["!llvm.struct<(i32, i64)>"; "i8"]
and splitTypeList (content: string) : string list =
    let mutable depth = 0
    let mutable current = System.Text.StringBuilder()
    let mutable results = []

    for c in content do
        match c with
        | '<' | '(' ->
            depth <- depth + 1
            current.Append(c) |> ignore
        | '>' | ')' ->
            depth <- depth - 1
            current.Append(c) |> ignore
        | ',' when depth = 0 ->
            let field = current.ToString().Trim()
            if not (String.IsNullOrEmpty(field)) then
                results <- field :: results
            current.Clear() |> ignore
        | _ ->
            current.Append(c) |> ignore

    // Don't forget the last field
    let lastField = current.ToString().Trim()
    if not (String.IsNullOrEmpty(lastField)) then
        results <- lastField :: results

    List.rev results

/// Common type sizes for quick reference
module CommonSizes =
    /// i32 size in bytes
    let i32 = 4L

    /// i64 size in bytes
    let i64 = 8L

    /// Pointer size in bytes (x86_64)
    let ptr = 8L

    /// Fat pointer size (ptr + length) - used for string, array
    let fatPointer = 16L

    /// String type size (fat pointer)
    let string = fatPointer

    /// Check if type is a fat pointer (string, array)
    let isFatPointer (mlirType: string) : bool =
        mlirType = "!llvm.struct<(ptr, i64)>" ||
        mlirType = "!llvm.struct<(!llvm.ptr, i64)>"

/// Get size as MLIR i64 constant expression
let sizeAsConstant (mlirType: string) : string =
    let size = computeSize mlirType
    sprintf "arith.constant %d : i64" size
