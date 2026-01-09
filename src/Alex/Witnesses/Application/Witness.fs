/// Application/Witness - Witness function application semantics to MLIR
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This module handles APPLICATION node witnessing. It is NOT a central dispatcher.
/// It delegates to factored modules (Primitives, Format, Platform) based on SemanticKind.
///
/// The PSGZipper contains the Graph - no separate graph parameter needed.
/// Returns structured MLIROp list via the factored witness modules.
module Alex.Witnesses.Application.Witness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns

// Import factored witness modules
module Primitives = Alex.Witnesses.Application.Primitives
module Format = Alex.Witnesses.Application.Format
module Platform = Alex.Witnesses.Application.Platform

// ═══════════════════════════════════════════════════════════════════════════
// ARGUMENT RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve argument node IDs to Val (SSA + type)
let private resolveArgs (argNodeIds: NodeId list) (z: PSGZipper) : Val list =
    argNodeIds
    |> List.choose (fun nodeId ->
        match lookupNodeSSA nodeId z, SemanticGraph.tryGetNode nodeId z.Graph with
        | Some ssa, Some node ->
            Some { SSA = ssa; Type = mapNativeType node.Type }
        | _ -> None)

/// Resolve function node to its SemanticKind
let private resolveFuncKind (funcNodeId: NodeId) (z: PSGZipper) : SemanticKind option =
    match SemanticGraph.tryGetNode funcNodeId z.Graph with
    | Some node -> Some node.Kind
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// ZEROED DEFAULT
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativeDefault.zeroed<'T>
let private witnessZeroed (z: PSGZipper) (resultType: MLIRType) : (MLIROp list * TransferResult) option =
    let resultSSA = freshSynthSSA z

    let zeroOp =
        match resultType with
        | TInt _ ->
            MLIROp.ArithOp (ArithOp.ConstI (resultSSA, 0L, resultType))
        | TFloat F32 ->
            MLIROp.ArithOp (ArithOp.ConstF (resultSSA, 0.0, MLIRTypes.f32))
        | TFloat F64 ->
            MLIROp.ArithOp (ArithOp.ConstF (resultSSA, 0.0, MLIRTypes.f64))
        | TPtr ->
            MLIROp.LLVMOp (LLVMOp.ZeroInit (resultSSA, MLIRTypes.ptr))
        | TStruct _ ->
            MLIROp.LLVMOp (LLVMOp.Undef (resultSSA, resultType))
        | _ ->
            MLIROp.LLVMOp (LLVMOp.Undef (resultSSA, resultType))

    Some ([zeroOp], TRValue { SSA = resultSSA; Type = resultType })

// ═══════════════════════════════════════════════════════════════════════════
// STRING OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// String concatenation using LLVM memcpy
let private witnessStringConcat (z: PSGZipper) (str1: Val) (str2: Val) : (MLIROp list * TransferResult) option =
    // Extract ptr/len from both strings
    let ptr1SSA = freshSynthSSA z
    let len1SSA = freshSynthSSA z
    let ptr2SSA = freshSynthSSA z
    let len2SSA = freshSynthSSA z

    let extractOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptr1SSA, str1.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (len1SSA, str1.SSA, [1], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptr2SSA, str2.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (len2SSA, str2.SSA, [1], MLIRTypes.nativeStr))
    ]

    // Total length
    let totalLenSSA = freshSynthSSA z
    let addLenOp = MLIROp.ArithOp (ArithOp.AddI (totalLenSSA, len1SSA, len2SSA, MLIRTypes.i64))

    // Allocate buffer
    let bufSSA = freshSynthSSA z
    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, totalLenSSA, MLIRTypes.i8, None))

    // Copy first string
    let memcpy1Op = MLIROp.LLVMOp (LLVMOp.MemCpy (bufSSA, ptr1SSA, len1SSA, false))

    // GEP to offset for second string
    let offsetSSA = freshSynthSSA z
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (offsetSSA, bufSSA, [(len1SSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Copy second string
    let memcpy2Op = MLIROp.LLVMOp (LLVMOp.MemCpy (offsetSSA, ptr2SSA, len2SSA, false))

    // Build result fat string
    let undefSSA = freshSynthSSA z
    let withPtrSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z

    let buildStrOps = [
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, bufSSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, totalLenSSA, [1], MLIRTypes.nativeStr))
    ]

    let allOps = extractOps @ [addLenOp; allocOp; memcpy1Op; gepOp; memcpy2Op] @ buildStrOps
    Some (allOps, TRValue { SSA = resultSSA; Type = MLIRTypes.nativeStr })

/// Witness string operations
let private witnessStringOp
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (_resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match opName, args with
    | "concat2", [str1; str2] ->
        witnessStringConcat z str1 str2

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// CONVERSION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness numeric type conversions
let private witnessConversion
    (z: PSGZipper)
    (convName: string)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        let resultSSA = freshSynthSSA z

        let convOp =
            match convName, arg.Type, resultType with
            // Integer to float
            | "toFloat", TInt _, TFloat _ ->
                Some (ArithOp.SIToFP (resultSSA, arg.SSA, arg.Type, resultType))
            | "toFloat32", TInt _, TFloat F32 ->
                Some (ArithOp.SIToFP (resultSSA, arg.SSA, arg.Type, MLIRTypes.f32))

            // Float to integer
            | "toInt", TFloat _, TInt I32 ->
                Some (ArithOp.FPToSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))
            | "toInt64", TFloat _, TInt I64 ->
                Some (ArithOp.FPToSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))

            // Integer widening (signed)
            | "toInt64", TInt w, TInt I64 when w <> I64 ->
                Some (ArithOp.ExtSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))
            | "toInt", TInt I8, TInt I32 ->
                Some (ArithOp.ExtSI (resultSSA, arg.SSA, MLIRTypes.i8, MLIRTypes.i32))
            | "toInt", TInt I16, TInt I32 ->
                Some (ArithOp.ExtSI (resultSSA, arg.SSA, MLIRTypes.i16, MLIRTypes.i32))

            // Integer narrowing
            | "toInt", TInt I64, TInt I32 ->
                Some (ArithOp.TruncI (resultSSA, arg.SSA, MLIRTypes.i64, MLIRTypes.i32))
            | "toByte", TInt _, TInt I8 ->
                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i8))

            // Identity conversions
            | "toInt", TInt I32, TInt I32 ->
                None  // No-op, return arg directly
            | "toInt64", TInt I64, TInt I64 ->
                None
            | "toFloat", TFloat F64, TFloat F64 ->
                None
            | "toFloat32", TFloat F32, TFloat F32 ->
                None

            | _ -> None

        match convOp with
        | Some op ->
            Some ([MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = resultType })
        | None when convName = "toInt" && arg.Type = MLIRTypes.i32 ->
            Some ([], TRValue arg)  // Identity
        | None when convName = "toInt64" && arg.Type = MLIRTypes.i64 ->
            Some ([], TRValue arg)  // Identity
        | None ->
            None

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an intrinsic operation based on IntrinsicInfo
let private witnessIntrinsic
    (z: PSGZipper)
    (intrinsicInfo: IntrinsicInfo)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeType returnType

    match intrinsicInfo with
    // Platform operations - delegate to Platform module
    | SysOp opName ->
        Platform.witnessSysOp z opName args mlirReturnType

    | ConsoleOp opName ->
        Platform.witnessConsoleOp z opName args mlirReturnType

    // Format operations - delegate to Format module
    | FormatOp "int" ->
        match args with
        | [intVal] ->
            let ops, resultVal = Format.intToString z intVal
            Some (ops, TRValue resultVal)
        | _ -> None

    | FormatOp "float" ->
        match args with
        | [floatVal] ->
            let ops, resultVal = Format.floatToString z floatVal
            Some (ops, TRValue resultVal)
        | _ -> None

    // Parse operations - delegate to Format module
    | ParseOp "int" ->
        match args with
        | [strVal] ->
            let ops, resultVal = Format.stringToInt z strVal
            Some (ops, TRValue resultVal)
        | _ -> None

    // NativePtr operations - handled by MemoryWitness (not here)
    | NativePtrOp _ ->
        None  // Delegate to MemoryWitness

    // Convert operations - use Arith dialect directly
    | ConvertOp convName ->
        witnessConversion z convName args mlirReturnType

    // String operations
    | StringOp opName ->
        witnessStringOp z opName args mlirReturnType

    // Default zeroed value
    | NativeDefaultOp "zeroed" ->
        witnessZeroed z mlirReturnType

    // Unhandled intrinsics
    | _ ->
        None

// ═══════════════════════════════════════════════════════════════════════════
// BINARY/UNARY PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════

/// Try to witness a binary primitive operation
let private tryWitnessBinaryPrimitive
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [lhs; rhs] ->
        Primitives.tryWitnessBinaryOp z opName lhs rhs
    | _ -> None

/// Try to witness a unary primitive operation
let private tryWitnessUnaryPrimitive
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        Primitives.witnessUnary z opName arg
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a function application
/// This is called by FNCSTransfer for Application nodes
/// PHOTOGRAPHER PRINCIPLE: Returns ops, does not emit
let witness
    (funcNodeId: NodeId)
    (argNodeIds: NodeId list)
    (returnType: NativeType)
    (z: PSGZipper)
    : MLIROp list * TransferResult =

    // Resolve arguments to Val list
    let args = resolveArgs argNodeIds z
    let mlirReturnType = mapNativeType returnType

    // Get the function node's semantic kind
    match resolveFuncKind funcNodeId z with
    | None ->
        [], TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))

    | Some funcKind ->
        match funcKind with
        // Platform bindings - delegate to Platform module
        | SemanticKind.PlatformBinding entryPoint ->
            match Platform.witnessPlatformBinding z entryPoint args mlirReturnType with
            | Some (ops, result) ->
                ops, result
            | None ->
                [], TRError (sprintf "Platform binding '%s' failed" entryPoint)

        // Intrinsic operations
        | SemanticKind.Intrinsic intrinsicInfo ->
            match witnessIntrinsic z intrinsicInfo args returnType with
            | Some (ops, result) ->
                ops, result
            | None ->
                // Intrinsic not handled - return marker for deferred handling
                [], TRError (sprintf "Unhandled intrinsic: %s" intrinsicInfo.FullName)

        // Variable reference (function call or primitive)
        | SemanticKind.VarRef (name, _defId) ->
            // Try binary primitive first
            match tryWitnessBinaryPrimitive z name args with
            | Some (ops, result) ->
                ops, result
            | None ->
                // Try unary primitive
                match tryWitnessUnaryPrimitive z name args with
                | Some (ops, result) ->
                    ops, result
                | None ->
                    // Regular function call - needs call emission
                    // This will be handled by call witnessing
                    [], TRError (sprintf "Function call not yet implemented: %s" name)

        // Other kinds not handled here
        | _ ->
            [], TRError (sprintf "Unexpected function kind in application: %A" funcKind)

