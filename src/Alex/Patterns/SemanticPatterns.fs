/// Semantic Patterns - Active patterns for semantic classification
///
/// ARCHITECTURAL FOUNDATION:
/// This module provides typed active patterns that replace string matching
/// on operator and function names. Instead of matching on "op_Addition",
/// we pattern match on (|ArithAdd|_|) which is:
/// - Type-safe (compiler checks exhaustiveness)
/// - Self-documenting (patterns describe intent)
/// - Composable (patterns can be nested)
/// - Extensible (add new patterns without changing existing code)
module Alex.Patterns.SemanticPatterns

open Alex.Templates.TemplateTypes

// ═══════════════════════════════════════════════════════════════════════════
// Binary Arithmetic Operators
// ═══════════════════════════════════════════════════════════════════════════

// Note: BinaryArithOp and CompareOp are canonical in TemplateTypes
// This module re-exports them for pattern matching convenience

/// Active pattern to classify binary arithmetic by operator name
let (|BinaryArith|_|) (opName: string) =
    match opName with
    | "op_Addition" -> Some Add
    | "op_Subtraction" -> Some Sub
    | "op_Multiply" -> Some Mul
    | "op_Division" -> Some Div
    | "op_Modulus" -> Some Mod
    | "op_BitwiseAnd" -> Some BitAnd
    | "op_BitwiseOr" -> Some BitOr
    | "op_ExclusiveOr" -> Some BitXor
    | "op_LeftShift" -> Some ShiftLeft
    | "op_RightShift" -> Some ShiftRight
    | _ -> None

/// Active pattern to classify comparison by operator name
let (|Compare|_|) (opName: string) =
    match opName with
    | "op_LessThan" -> Some Lt
    | "op_LessThanOrEqual" -> Some Le
    | "op_GreaterThan" -> Some Gt
    | "op_GreaterThanOrEqual" -> Some Ge
    | "op_Equality" -> Some Eq
    | "op_Inequality" -> Some Ne
    | _ -> None

/// Active pattern for arithmetic that returns the appropriate op based on type
let (|ArithBinaryOp|_|) (opName: string, isInt: bool) =
    match opName, isInt with
    | "op_Addition", true -> Some ("arith.addi", false)
    | "op_Subtraction", true -> Some ("arith.subi", false)
    | "op_Multiply", true -> Some ("arith.muli", false)
    | "op_Division", true -> Some ("arith.divsi", false)
    | "op_Modulus", true -> Some ("arith.remsi", false)
    | "op_Addition", false -> Some ("arith.addf", false)
    | "op_Subtraction", false -> Some ("arith.subf", false)
    | "op_Multiply", false -> Some ("arith.mulf", false)
    | "op_Division", false -> Some ("arith.divf", false)
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Comparison Operators
// ═══════════════════════════════════════════════════════════════════════════

/// Integer comparison predicates (for arith.cmpi)
type IntCmpPred =
    | SLT   // signed less than
    | SLE   // signed less than or equal
    | SGT   // signed greater than
    | SGE   // signed greater than or equal
    | EQ    // equal
    | NE    // not equal

/// Float comparison predicates (for arith.cmpf)
type FloatCmpPred =
    | OLT   // ordered less than
    | OLE   // ordered less than or equal
    | OGT   // ordered greater than
    | OGE   // ordered greater than or equal
    | OEQ   // ordered equal
    | ONE   // ordered not equal

/// Active pattern for comparison operations
let (|CmpBinaryOp|_|) (opName: string, isInt: bool) =
    match opName, isInt with
    // Integer comparisons
    | "op_LessThan", true -> Some ("arith.cmpi", "slt")
    | "op_LessThanOrEqual", true -> Some ("arith.cmpi", "sle")
    | "op_GreaterThan", true -> Some ("arith.cmpi", "sgt")
    | "op_GreaterThanOrEqual", true -> Some ("arith.cmpi", "sge")
    | "op_Equality", true -> Some ("arith.cmpi", "eq")
    | "op_Inequality", true -> Some ("arith.cmpi", "ne")
    // Float comparisons
    | "op_LessThan", false -> Some ("arith.cmpf", "olt")
    | "op_LessThanOrEqual", false -> Some ("arith.cmpf", "ole")
    | "op_GreaterThan", false -> Some ("arith.cmpf", "ogt")
    | "op_GreaterThanOrEqual", false -> Some ("arith.cmpf", "oge")
    | "op_Equality", false -> Some ("arith.cmpf", "oeq")
    | "op_Inequality", false -> Some ("arith.cmpf", "one")
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Bitwise Operators
// ═══════════════════════════════════════════════════════════════════════════

/// Active pattern for bitwise binary operations
let (|BitwiseBinaryOp|_|) (opName: string) =
    match opName with
    | "op_BitwiseAnd" -> Some "arith.andi"
    | "op_BitwiseOr" -> Some "arith.ori"
    | "op_ExclusiveOr" -> Some "arith.xori"
    | "op_LeftShift" -> Some "arith.shli"
    | "op_RightShift" -> Some "arith.shrsi"  // signed right shift
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Unary Operators
// ═══════════════════════════════════════════════════════════════════════════

/// Unary operation kinds
type UnaryOpKind =
    | BoolNot       // Logical not (XOR with true)
    | IntNegate     // Integer negation (0 - x)
    | BitwiseNot    // Bitwise complement (XOR with -1)

/// Active pattern for unary operations
let (|UnaryOp|_|) (opName: string) =
    match opName with
    | "not" -> Some BoolNot
    | "op_UnaryNegation" -> Some IntNegate
    | "op_OnesComplement" -> Some BitwiseNot
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// NativePtr Operations
// ═══════════════════════════════════════════════════════════════════════════

open FSharp.Native.Compiler.Checking.Native.SemanticGraph

/// NativePtr operation kinds (type-safe dispatch)
type NativePtrOpKind =
    | PtrToNativeInt    // ptr -> nativeint (ptrtoint)
    | PtrOfNativeInt    // nativeint -> ptr (inttoptr)
    | PtrToVoidPtr      // 'a nativeptr -> voidptr (no-op cast)
    | PtrOfVoidPtr      // voidptr -> 'a nativeptr (no-op cast)
    | PtrGet            // indexed load: ptr[idx]
    | PtrSet            // indexed store: ptr[idx] <- value
    | PtrStackAlloc     // stack allocation
    | PtrCopy           // memcpy
    | PtrFill           // memset
    | PtrAdd            // pointer arithmetic

/// Active pattern for NativePtr operations with typed dispatch (using IntrinsicInfo)
let (|NativePtrOp|_|) (info: IntrinsicInfo) =
    if info.Module <> IntrinsicModule.NativePtr then None
    else
        match info.Operation with
        | "toNativeInt" -> Some PtrToNativeInt
        | "ofNativeInt" -> Some PtrOfNativeInt
        | "toVoidPtr" -> Some PtrToVoidPtr
        | "ofVoidPtr" -> Some PtrOfVoidPtr
        | "get" -> Some PtrGet
        | "set" -> Some PtrSet
        | "stackalloc" -> Some PtrStackAlloc
        | "copy" -> Some PtrCopy
        | "fill" -> Some PtrFill
        | "add" -> Some PtrAdd
        | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Operator Classification
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a name is a primitive operator (op_*)
let (|PrimitiveOp|_|) (name: string) =
    if name.StartsWith("op_") then Some name else None

/// Check if intrinsic is a Sys operation (using IntrinsicInfo)
let (|SysOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Sys then Some info.Operation else None

/// Check if intrinsic is a Console operation (using IntrinsicInfo)
let (|ConsoleOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Console then Some info.Operation else None

/// Platform intrinsic - unified pattern for Sys, Console, and other platform operations
/// Returns (module, operation) tuple for dispatch to platform bindings
let (|PlatformIntrinsic|_|) (info: IntrinsicInfo) =
    match info.Category with
    | IntrinsicCategory.Platform -> Some (info.Module, info.Operation)
    | _ -> None

/// Check if intrinsic is a NativeStr operation (using IntrinsicInfo)
let (|NativeStrOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.NativeStr then Some info.Operation else None

/// Check if intrinsic is a NativeDefault operation (using IntrinsicInfo)
let (|NativeDefaultOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.NativeDefault then Some info.Operation else None

/// Check if intrinsic is a String operation (using IntrinsicInfo)
let (|StringOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.String then Some info.Operation else None

/// Check if intrinsic is a Format operation (using IntrinsicInfo)
/// Format intrinsics convert values to strings: int->string, float->string, etc.
let (|FormatOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Format then Some info.Operation else None

/// Check if intrinsic is a Parse operation (using IntrinsicInfo)
/// Parse intrinsics convert strings to values: string->int, string->float, etc.
let (|ParseOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Parse then Some info.Operation else None

/// Check if intrinsic is a Convert operation (using IntrinsicInfo)
/// Convert intrinsics perform numeric type conversions: int->float, float->int, etc.
let (|ConvertOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Convert then Some info.Operation else None

/// Check if intrinsic is an Array operation (using IntrinsicInfo)
let (|ArrayOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Array then Some info.Operation else None

/// Check if intrinsic is an Unchecked operation (using IntrinsicInfo)
let (|UncheckedOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Unchecked then Some info.Operation else None

/// Check if intrinsic is a Crypto operation (using IntrinsicInfo)
/// Crypto intrinsics: sha1, base64Encode, base64Decode
let (|CryptoOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Crypto then Some info.Operation else None

/// Check if intrinsic is a Bits operation (using IntrinsicInfo)
/// Bits intrinsics: htons, ntohs, htonl, ntohl, float32ToInt32Bits, int32BitsToFloat32, etc.
let (|BitsOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Bits then Some info.Operation else None

// ═══════════════════════════════════════════════════════════════════════════
// Reactive Signals (SolidJS-inspired native signals)
// ═══════════════════════════════════════════════════════════════════════════

/// Check if intrinsic is a FnPtr operation (using IntrinsicInfo)
/// FnPtr intrinsics: ofFunction, invoke, isNull, null
let (|FnPtrOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.FnPtr then Some info.Operation else None

/// Check if intrinsic is a Signal operation (using IntrinsicInfo)
/// Signal intrinsics: create, get, set, update
let (|SignalOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Signal then Some info.Operation else None

/// Check if intrinsic is an Effect operation (using IntrinsicInfo)
/// Effect intrinsics: create, createWithCleanup, dispose
let (|EffectOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Effect then Some info.Operation else None

/// Check if intrinsic is a Memo operation (using IntrinsicInfo)
/// Memo intrinsics: create, get
let (|MemoOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Memo then Some info.Operation else None

/// Check if intrinsic is a Batch operation (using IntrinsicInfo)
/// Batch intrinsics: run
let (|BatchOp|_|) (info: IntrinsicInfo) =
    if info.Module = IntrinsicModule.Batch then Some info.Operation else None

/// Conversion function names
let private conversionFunctions = 
    set ["int"; "int8"; "int16"; "int32"; "int64";
         "byte"; "uint8"; "uint16"; "uint32"; "uint64";
         "float"; "float32"; "double"; "single"; "decimal";
         "nativeint"; "unativeint"; "char"; "string"]

/// Check if a name is a type conversion function
let (|ConversionOp|_|) (name: string) =
    if Set.contains name conversionFunctions then Some name else None

/// Option/ValueOption constructors
let (|OptionConstructor|_|) (name: string) =
    match name with
    | "Some" | "None" | "ValueSome" | "ValueNone" -> Some name
    | _ -> None

/// Box/unbox operations
let (|BoxOp|_|) (name: string) =
    match name with
    | "box" | "unbox" -> Some name
    | _ -> None

/// Printf family functions
let (|PrintfOp|_|) (name: string) =
    match name with
    | "printf" | "printfn" | "sprintf" | "failwith" | "failwithf" -> Some name
    | _ -> None

/// Array operation names (for name-based recognition in BuiltInOperator)
let (|ArrayOpName|_|) (name: string) =
    match name with
    | "Array.zeroCreate" | "Array.length" | "Array.get" | "Array.set" -> Some name
    | _ -> None

/// Other built-in functions
let (|OtherBuiltin|_|) (name: string) =
    match name with
    | "ignore" | "raise" | "reraise" | "typeof" | "sizeof" | "nameof" -> Some name
    | _ -> None

/// Master pattern: Is this identifier a built-in operator or function?
/// NOTE: This is NAME-based recognition for identifier resolution, distinct from
/// IntrinsicInfo-based patterns which classify structured intrinsic nodes.
let (|BuiltInOperator|_|) (name: string) =
    match name with
    | PrimitiveOp _ -> Some name        // op_Addition, op_Multiply, etc.
    | ConversionOp _ -> Some name       // int, float, byte, etc.
    | OptionConstructor _ -> Some name  // Some, None, etc.
    | BoxOp _ -> Some name              // box, unbox
    | PrintfOp _ -> Some name           // printf, printfn, sprintf, etc.
    | ArrayOpName _ -> Some name        // Array.zeroCreate, Array.length, etc.
    | OtherBuiltin _ -> Some name       // ignore, raise, typeof, etc.
    | "not" -> Some name                // Boolean not
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Type Classification Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type string represents an integer type
let (|IntegerType|FloatType|BoolType|OtherType|) (tyStr: string) =
    match tyStr with
    | "i8" | "i16" | "i32" | "i64" -> IntegerType
    | "f32" | "f64" -> FloatType
    | "i1" -> BoolType
    | _ -> OtherType

/// Check if type is integer
let isIntegerType (tyStr: string) =
    match tyStr with
    | IntegerType -> true
    | _ -> false

/// Check if type is float
let isFloatType (tyStr: string) =
    match tyStr with
    | FloatType -> true
    | _ -> false

/// Check if type is boolean
let isBoolType (tyStr: string) =
    match tyStr with
    | BoolType -> true
    | _ -> false
