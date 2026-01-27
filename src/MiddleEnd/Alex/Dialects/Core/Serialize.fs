/// MLIR Type Serialization
///
/// Converts structured MLIRType to MLIR text format strings.
module Alex.Dialects.Core.Serialize

open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// TYPE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert IntWidth to MLIR type string
let intWidthToString (width: IntWidth) : string =
    match width with
    | I1 -> "i1"
    | I8 -> "i8"
    | I16 -> "i16"
    | I32 -> "i32"
    | I64 -> "i64"

/// Convert FloatWidth to MLIR type string
let floatWidthToString (width: FloatWidth) : string =
    match width with
    | F32 -> "f32"
    | F64 -> "f64"

/// Convert MLIRType to MLIR text format string
let rec typeToString (ty: MLIRType) : string =
    match ty with
    | TInt width -> intWidthToString width
    | TFloat width -> floatWidthToString width
    | TPtr -> "!llvm.ptr"
    | TStruct fields ->
        let fieldStrs = fields |> List.map typeToString |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldStrs
    | TArray (count, elemTy) ->
        sprintf "!llvm.array<%d x %s>" count (typeToString elemTy)
    | TFunc (paramTypes, retType) ->
        let paramStrs = paramTypes |> List.map typeToString |> String.concat ", "
        sprintf "(%s) -> %s" paramStrs (typeToString retType)
    | TMemRef elemTy ->
        sprintf "memref<?x%s>" (typeToString elemTy)
    | TVector (count, elemTy) ->
        sprintf "vector<%dx%s>" count (typeToString elemTy)
    | TIndex -> "index"
    | TUnit -> "i32"  // Unit represented as i32 (value 0)
    | TError msg -> sprintf "<<ERROR: %s>>" msg
