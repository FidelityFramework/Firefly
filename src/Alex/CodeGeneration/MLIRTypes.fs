/// MLIRTypes - Core MLIR type representations for Alex code generation
///
/// This module contains the essential type definitions used throughout
/// the MLIR generation pipeline. It is imported by MLIRZipper and bindings.
///
/// Extracted from legacy MLIRBuilder.fs during FNCS integration (December 2025).
/// The monad-based MLIR<'T> computation expression has been removed in favor
/// of the codata accumulator pattern in MLIRZipper.fs.
module Alex.CodeGeneration.MLIRTypes

// ═══════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════

/// MLIR integer bit widths
type IntegerBitWidth = I1 | I8 | I16 | I32 | I64

/// MLIR floating point precision
type FloatPrecision = F32 | F64

/// MLIR types - the structured representation used throughout Alex
/// String serialization happens only in Serialize module during MLIR text emission
///
/// ARCHITECTURAL PRINCIPLE: A compiler's PRIMARY job is to surface errors.
/// Silent failures are the most severe architectural violations.
/// TypeError MUST propagate and NEVER be silently ignored.
type MLIRType =
    | Integer of IntegerBitWidth
    | Float of FloatPrecision
    | Pointer
    | Struct of MLIRType list
    | Array of int * MLIRType
    | Function of MLIRType list * MLIRType
    | Unit
    | Index
    /// Type mapping error - MUST propagate up the pipeline
    /// Contains error message describing why type mapping failed
    | TypeError of string

/// An SSA value - the primary currency of MLIR operations
[<Struct>]
type SSA =
    | V of int
    | Arg of int
    member this.Name =
        match this with
        | V n -> sprintf "%%v%d" n
        | Arg n -> sprintf "%%arg%d" n

/// A typed SSA value
type Val = { SSA: SSA; Type: MLIRType }

/// Global symbol reference
type Global =
    | GFunc of string
    | GStr of uint32
    | GBytes of int

/// Integer comparison predicates
type ICmp = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge

// ═══════════════════════════════════════════════════════════════════
// Type Serialization
// ═══════════════════════════════════════════════════════════════════

module Serialize =
    let integerBitWidth = function
        | I1 -> "i1" | I8 -> "i8" | I16 -> "i16" | I32 -> "i32" | I64 -> "i64"

    let floatPrecision = function F32 -> "f32" | F64 -> "f64"

    let rec mlirType = function
        | Integer bitWidth -> integerBitWidth bitWidth
        | Float precision -> floatPrecision precision
        | Pointer -> "!llvm.ptr"
        | Struct fields ->
            let fieldTypes = fields |> List.map mlirType |> String.concat ", "
            sprintf "!llvm.struct<(%s)>" fieldTypes
        | Array (count, elementType) -> sprintf "!llvm.array<%d x %s>" count (mlirType elementType)
        | Function (parameterTypes, returnType) ->
            let parameterString = parameterTypes |> List.map mlirType |> String.concat ", "
            sprintf "(%s) -> %s" parameterString (mlirType returnType)
        | Unit -> "i32"  // Unit maps to i32 for compatibility (returns 0)
        | Index -> "index"
        | TypeError msg -> failwithf "COMPILER ERROR: Attempted to serialize TypeError to MLIR: %s" msg

    /// Deserialize an MLIR type string back to MLIRType
    /// This is the inverse of mlirType
    /// Returns TypeError for unknown types - NEVER falls back silently
    let deserializeType (s: string) : MLIRType =
        match s.Trim() with
        | "i1" -> Integer I1
        | "i8" -> Integer I8
        | "i16" -> Integer I16
        | "i32" -> Integer I32
        | "i64" -> Integer I64
        | "f32" -> Float F32
        | "f64" -> Float F64
        | "!llvm.ptr" -> Pointer
        | "index" -> Index
        // Native string type: fat pointer {ptr, len}
        | "!llvm.struct<(!llvm.ptr, i64)>" -> Struct [Pointer; Integer I64]
        // DU type: tag (i32) + payload (i64)
        | "!llvm.struct<(i32, i64)>" -> Struct [Integer I32; Integer I64]
        // Two-pointer struct (e.g., WebView handle)
        | "!llvm.struct<(!llvm.ptr, !llvm.ptr)>" -> Struct [Pointer; Pointer]
        | _ when s.StartsWith("!llvm.ptr") -> Pointer  // Handle opaque pointer variations
        | _ when s.StartsWith("!llvm.struct") ->
            // Generic struct - we MUST parse this properly, not fall back
            TypeError (sprintf "deserializeType: unparsed struct type '%s' - implement proper parsing" s)
        | _ -> TypeError (sprintf "deserializeType: unknown MLIR type '%s'" s)

    let ssa (s: SSA) = s.Name

    let global_ = function
        | GFunc name -> sprintf "@%s" name
        | GStr hash -> sprintf "@str_%u" hash
        | GBytes idx -> sprintf "@bytes_%d" idx

    let icmp = function
        | Eq -> "eq" | Ne -> "ne"
        | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
        | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

    /// Escape string content for MLIR literals
    let escape (s: string) =
        s.Replace("\\", "\\\\")
         .Replace("\"", "\\\"")
         .Replace("\n", "\\0A")
         .Replace("\r", "\\0D")
         .Replace("\t", "\\09")

// ═══════════════════════════════════════════════════════════════════
// Native Type Constants
// ═══════════════════════════════════════════════════════════════════

/// Native string type: fat pointer = {ptr: *u8, len: i64}
/// This is the canonical MLIR representation for F# string in native compilation.
/// See Serena memory: string_length_handling, fidelity_memory_model
let NativeStrType = Struct [Pointer; Integer I64]

/// Serialized form of native string type for pattern matching
let NativeStrTypeStr = "!llvm.struct<(!llvm.ptr, i64)>"

/// Check if a type string represents a native string (fat pointer)
let isNativeStrType (typeStr: string) =
    typeStr = NativeStrTypeStr || typeStr = "!llvm.struct<(!llvm.ptr, i64)>"
