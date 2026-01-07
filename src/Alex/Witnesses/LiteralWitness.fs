/// Literal Witness - Witness literal values to MLIR
///
/// Observes literal PSG nodes and generates corresponding MLIR constants.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Templates.TemplateTypes
open Alex.Templates.ArithTemplates
open Alex.Templates.MemoryTemplates

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a literal value and generate corresponding MLIR
let witness (lit: LiteralValue) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match lit with
    | LiteralValue.Unit ->
        let ssaName, zipper' = MLIRZipper.witnessConstant 0L I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Bool b ->
        let value = if b then 1L else 0L
        let ssaName, zipper' = MLIRZipper.witnessConstant value I1 zipper
        zipper', TRValue (ssaName, "i1")

    | LiteralValue.Int8 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.Int16 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.Int32 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int64 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant n I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.UInt8 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.UInt16 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt32 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.UInt64 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.NativeInt n ->
        let wordType = Alex.CodeGeneration.TypeMapping.platformWordType Alex.CodeGeneration.TypeMapping.defaultPlatform
        let width = if wordType = "i64" then I64 else I32
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) width zipper
        zipper', TRValue (ssaName, wordType)

    | LiteralValue.UNativeInt n ->
        let wordType = Alex.CodeGeneration.TypeMapping.platformWordType Alex.CodeGeneration.TypeMapping.defaultPlatform
        let width = if wordType = "i64" then I64 else I32
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) width zipper
        zipper', TRValue (ssaName, wordType)

    | LiteralValue.Char c ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 c) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Float32 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let constParams = { Result = ssaName; Value = sprintf "%e" (float f); Type = "f32" }
        let text = render Quot.Constant.floatConst constParams
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F32) zipper'
        zipper'', TRValue (ssaName, "f32")

    | LiteralValue.Float64 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let constParams = { Result = ssaName; Value = sprintf "%e" f; Type = "f64" }
        let text = render Quot.Constant.floatConst constParams
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F64) zipper'
        zipper'', TRValue (ssaName, "f64")

    | LiteralValue.String s ->
        // Native string: fat pointer struct {ptr: *u8, len: i64}
        let globalName, zipper1 = MLIRZipper.observeStringLiteral s zipper
        let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
        // CRITICAL: Use UTF-8 byte count, not .NET character count
        let byteLen = System.Text.Encoding.UTF8.GetByteCount(s)
        let lenSSA, zipper3 = MLIRZipper.witnessConstant (int64 byteLen) I64 zipper2
        
        // Build fat pointer struct using templates
        let undefSSA, zipper4 = MLIRZipper.yieldSSA zipper3
        let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
        let zipper5 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper4
        
        let withPtrSSA, zipper6 = MLIRZipper.yieldSSA zipper5
        let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = ptrSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
        let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
        let zipper7 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper6
        
        let fatPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let insertLenParams : Quot.Aggregate.InsertParams = { Result = fatPtrSSA; Value = lenSSA; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
        let insertLenText = render Quot.Aggregate.insertValue insertLenParams
        let zipper9 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper8
        
        zipper9, TRValue (fatPtrSSA, NativeStrTypeStr)

    | _ ->
        zipper, TRError (sprintf "Unsupported literal: %A" lit)
