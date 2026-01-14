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
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns

// Import factored witness modules
module Primitives = Alex.Witnesses.Application.Primitives
module Format = Alex.Witnesses.Application.Format
module Platform = Alex.Witnesses.Application.Platform

// ═══════════════════════════════════════════════════════════════════════════
// UNIT TYPE DETECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a NativeType is the unit type (void semantics)
let private isUnitType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> true
    | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// ARGUMENT RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve argument node IDs to Val (SSA + type)
/// CRITICAL: Use recallNodeResult to get ACTUAL emitted SSAs from NodeBindings,
/// not lookupNodeSSA which only gets pre-assigned SSAs from SSAAssignment.
/// For nested Applications, child results are in NodeBindings.
let private resolveArgs (argNodeIds: NodeId list) (z: PSGZipper) : Val list =
    argNodeIds
    |> List.choose (fun nodeId ->
        // Check if the argument node has Unit type - if so, drop it from the call
        // This aligns with FNCS behavior where unit parameters are omitted from Lambda definitions
        let isUnitArg =
            match SemanticGraph.tryGetNode nodeId z.Graph with
            | Some node -> isUnitType node.Type
            | None -> false

        if isUnitArg then None
        else
            // First try NodeBindings (actual emitted results from post-order traversal)
            match recallNodeResult (NodeId.value nodeId) z with
            | Some (ssa, ty) ->
                Some { SSA = ssa; Type = ty }
            | None ->
                // Fall back to pre-assigned SSA if not in NodeBindings
                // (shouldn't happen for children processed in post-order)
                match lookupNodeSSA nodeId z, SemanticGraph.tryGetNode nodeId z.Graph with
                | Some ssa, Some node ->
                    Some { SSA = ssa; Type = mapNativeType node.Type }
                | _ -> None)

/// Resolve function node to its SemanticKind
/// Looks through TypeAnnotation nodes to find the underlying function
let private resolveFuncKind (funcNodeId: NodeId) (z: PSGZipper) : SemanticKind option =
    let rec resolve nodeId =
        match SemanticGraph.tryGetNode nodeId z.Graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.TypeAnnotation (innerExpr, _) ->
                // TypeAnnotation is transparent - look at the inner expression
                resolve innerExpr
            | kind -> Some kind
        | None -> None
    resolve funcNodeId


// ═══════════════════════════════════════════════════════════════════════════
// ZEROED DEFAULT
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativeDefault.zeroed<'T>
let private witnessZeroed (appNodeId: NodeId) (z: PSGZipper) (resultType: MLIRType) : (MLIROp list * TransferResult) option =
    let resultSSA = requireNodeSSA appNodeId z

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
/// Requires 10 SSAs: 4 extract + 1 addLen + 1 alloca + 1 gep + 3 build
let private witnessStringConcat (appNodeId: NodeId) (z: PSGZipper) (str1: Val) (str2: Val) : (MLIROp list * TransferResult) option =
    // Get pre-assigned SSAs (10 total)
    let ssas = requireNodeSSAs appNodeId z
    let resultSSA = requireNodeSSA appNodeId z
    
    // Extract ptr/len from both strings
    let ptr1SSA = ssas.[0]
    let len1SSA = ssas.[1]
    let ptr2SSA = ssas.[2]
    let len2SSA = ssas.[3]

    let extractOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptr1SSA, str1.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (len1SSA, str1.SSA, [1], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptr2SSA, str2.SSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (len2SSA, str2.SSA, [1], MLIRTypes.nativeStr))
    ]

    // Total length
    let totalLenSSA = ssas.[4]
    let addLenOp = MLIROp.ArithOp (ArithOp.AddI (totalLenSSA, len1SSA, len2SSA, MLIRTypes.i64))

    // Allocate buffer
    let bufSSA = ssas.[5]
    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, totalLenSSA, MLIRTypes.i8, None))

    // Copy first string (dst, src, len using intrMemcpy via OpEnvelope)
    let bufVal = { SSA = bufSSA; Type = MLIRTypes.ptr }
    let ptr1Val = { SSA = ptr1SSA; Type = MLIRTypes.ptr }
    let len1Val = { SSA = len1SSA; Type = MLIRTypes.i64 }
    let memcpy1Op = intrMemcpy bufVal ptr1Val len1Val false

    // GEP to offset for second string
    let offsetSSA = ssas.[6]
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (offsetSSA, bufSSA, [(len1SSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Copy second string (dst, src, len using intrMemcpy via OpEnvelope)
    let offsetVal = { SSA = offsetSSA; Type = MLIRTypes.ptr }
    let ptr2Val = { SSA = ptr2SSA; Type = MLIRTypes.ptr }
    let len2Val = { SSA = len2SSA; Type = MLIRTypes.i64 }
    let memcpy2Op = intrMemcpy offsetVal ptr2Val len2Val false

    // Build result fat string
    let undefSSA = ssas.[7]
    let withPtrSSA = ssas.[8]

    let buildStrOps = [
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, bufSSA, [0], MLIRTypes.nativeStr))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, totalLenSSA, [1], MLIRTypes.nativeStr))
    ]

    let allOps = extractOps @ [addLenOp; allocOp; memcpy1Op; gepOp; memcpy2Op] @ buildStrOps
    Some (allOps, TRValue { SSA = resultSSA; Type = MLIRTypes.nativeStr })

/// Witness string operations
let private witnessStringOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (_resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match opName, args with
    | "concat2", [str1; str2] ->
        witnessStringConcat appNodeId z str1 str2

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativePtr intrinsic operations
let private witnessNativePtrOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opKind: NativePtrOpKind)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    let resultSSA = requireNodeSSA appNodeId z

    match opKind, args with
    | PtrRead, [ptr] ->
        // NativePtr.read: nativeptr<'T> -> 'T (direct load)
        let op = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, ptr.SSA, resultType, NotAtomic))
        Some ([op], TRValue { SSA = resultSSA; Type = resultType })

    | PtrWrite, [ptr; value] ->
        // NativePtr.write: nativeptr<'T> -> 'T -> unit (direct store)
        let op = MLIROp.LLVMOp (LLVMOp.Store (value.SSA, ptr.SSA, value.Type, NotAtomic))
        Some ([op], TRVoid)

    | PtrGet, [ptr; idx] ->
        // NativePtr.get: ptr -> int -> 'T (indexed load)
        // Needs 2 SSAs: gep intermediate + load result
        let ssas = requireNodeSSAs appNodeId z
        let gepSSA = ssas.[0]
        let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptr.SSA, [(idx.SSA, idx.Type)], resultType))
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, gepSSA, resultType, NotAtomic))
        Some ([gepOp; loadOp], TRValue { SSA = resultSSA; Type = resultType })

    | PtrSet, [ptr; idx; value] ->
        // NativePtr.set: ptr -> int -> 'T -> unit (indexed store)
        // Needs 1 SSA for gep intermediate
        let ssas = requireNodeSSAs appNodeId z
        let gepSSA = ssas.[0]
        let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptr.SSA, [(idx.SSA, idx.Type)], value.Type))
        let storeOp = MLIROp.LLVMOp (LLVMOp.Store (value.SSA, gepSSA, value.Type, NotAtomic))
        Some ([gepOp; storeOp], TRVoid)

    | PtrAdd, [ptr; offset] ->
        // NativePtr.add: ptr -> int -> ptr
        let op = MLIROp.LLVMOp (LLVMOp.GEP (resultSSA, ptr.SSA, [(offset.SSA, offset.Type)], TInt I8))
        Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })

    | PtrStackAlloc, [count] ->
        // NativePtr.stackalloc: int -> ptr
        // Alloca count must be i64
        if count.Type = MLIRTypes.i64 then
            let op = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, count.SSA, resultType, None))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })
        else
            // Extend count to i64 - needs 2 SSAs: extsi + alloca
            let ssas = requireNodeSSAs appNodeId z
            let extSSA = ssas.[0]
            let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, count.SSA, count.Type, MLIRTypes.i64))
            let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, extSSA, resultType, None))
            Some ([extOp; allocOp], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })

    | PtrToNativeInt, [ptr] ->
        // NativePtr.toNativeInt: ptr -> nativeint
        let op = MLIROp.LLVMOp (LLVMOp.PtrToInt (resultSSA, ptr.SSA, MLIRTypes.i64))
        Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 })

    | PtrOfNativeInt, [intVal] ->
        // NativePtr.ofNativeInt: nativeint -> ptr
        let op = MLIROp.LLVMOp (LLVMOp.IntToPtr (resultSSA, intVal.SSA, MLIRTypes.ptr))
        Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })

    | PtrToVoidPtr, [ptr] ->
        // NativePtr.toVoidPtr: nativeptr<'T> -> voidptr (no-op)
        Some ([], TRValue { SSA = ptr.SSA; Type = MLIRTypes.ptr })

    | PtrOfVoidPtr, [ptr] ->
        // NativePtr.ofVoidPtr: voidptr -> nativeptr<'T> (no-op)
        Some ([], TRValue { SSA = ptr.SSA; Type = MLIRTypes.ptr })

    | PtrCopy, [src; dst; count] ->
        // NativePtr.copy: src -> dst -> count -> unit
        let op = intrMemcpy dst src count false
        Some ([op], TRVoid)

    | PtrFill, [ptr; value; count] ->
        // NativePtr.fill: ptr -> byte -> count -> unit
        let op = intrMemset ptr value count false
        Some ([op], TRVoid)

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// CONVERSION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness numeric type conversions
let private witnessConversion
    (appNodeId: NodeId)
    (z: PSGZipper)
    (convName: string)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        let resultSSA = requireNodeSSA appNodeId z

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
    (appNodeId: NodeId)
    (z: PSGZipper)
    (intrinsicInfo: IntrinsicInfo)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeType returnType
    // Get pre-assigned result SSA for this Application node
    let resultSSA = requireNodeSSA appNodeId z

    match intrinsicInfo with
    // Platform operations - delegate to Platform module
    | SysOp opName ->
        Platform.witnessSysOp appNodeId z opName args mlirReturnType

    // NOTE: Console is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
    // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md

    // Format operations - delegate to Format module
    | FormatOp "int" ->
        match args with
        | [intVal] ->
            let ops, resultVal = Format.intToString appNodeId z intVal
            Some (ops, TRValue resultVal)
        | _ -> None

    | FormatOp "float" ->
        match args with
        | [floatVal] ->
            let ops, resultVal = Format.floatToString appNodeId z floatVal
            Some (ops, TRValue resultVal)
        | _ -> None

    // Parse operations - delegate to Format module
    | ParseOp "int" ->
        match args with
        | [strVal] ->
            let ops, resultVal = Format.stringToInt appNodeId z strVal
            Some (ops, TRValue resultVal)
        | _ -> None

    // NativePtr operations
    | NativePtrOp opKind ->
        // For stackalloc, we need the element type from the NativeType (not just ptr)
        match opKind with
        | PtrStackAlloc ->
            match extractPtrElementType returnType, args with
            | Some elemType, [count] ->
                if count.Type = MLIRTypes.i64 then
                    let op = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, count.SSA, elemType, None))
                    Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })
                else
                    // Extend count to i64 - use SSAs from pre-allocation
                    let ssas = requireNodeSSAs appNodeId z
                    let extSSA = ssas.[0]  // First SSA for intermediate
                    let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, count.SSA, count.Type, MLIRTypes.i64))
                    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, extSSA, elemType, None))
                    Some ([extOp; allocOp], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })
            | None, _ ->
                // Fall back if we can't extract element type
                witnessNativePtrOp appNodeId z opKind args mlirReturnType
            | _, _ ->
                witnessNativePtrOp appNodeId z opKind args mlirReturnType
        | _ ->
            witnessNativePtrOp appNodeId z opKind args mlirReturnType

    // Convert operations - use Arith dialect directly
    | ConvertOp convName ->
        witnessConversion appNodeId z convName args mlirReturnType

    // String operations
    | StringOp opName ->
        witnessStringOp appNodeId z opName args mlirReturnType

    // Default zeroed value
    | NativeDefaultOp "zeroed" ->
        witnessZeroed appNodeId z mlirReturnType

    // Operators module - delegate to Primitives
    // Handles arithmetic, comparison, bitwise, and unary operators (including "not")
    | _ when intrinsicInfo.Module = IntrinsicModule.Operators || intrinsicInfo.FullName.StartsWith("op_") || intrinsicInfo.FullName = "not" ->
        match args with
        | [lhs; rhs] ->
             Primitives.tryWitnessBinaryOp appNodeId z intrinsicInfo.Operation lhs rhs
        | [arg] ->
             Primitives.witnessUnary appNodeId z intrinsicInfo.Operation arg
        | _ -> None

    // NativeStr operations
    | NativeStrOp opName ->
        match opName, args with
        | "fromPointer", [ptr; len] ->
             // NativeStr.fromPointer: ptr -> i64 -> nativestr
             // Requires up to 4 SSAs: optional extsi + undef + withPtr + result
             let ssas = requireNodeSSAs appNodeId z
             
             // Ensure len is i64 (nativeStr layout requires i64 length)
             let lenSSA, lenOps, ssaOffset =
                 if len.Type = MLIRTypes.i64 then
                     len.SSA, [], 0
                 else
                     let extSSA = ssas.[0]
                     let extOp = ArithOp.ExtSI (extSSA, len.SSA, len.Type, MLIRTypes.i64)
                     extSSA, [MLIROp.ArithOp extOp], 1

             let undefSSA = ssas.[ssaOffset]
             let withPtrSSA = ssas.[ssaOffset + 1]

             let ops = lenOps @ [
                 MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))
                 MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, ptr.SSA, [0], MLIRTypes.nativeStr))
                 MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, lenSSA, [1], MLIRTypes.nativeStr))
             ]
             Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.nativeStr })
        | _ -> None

    // Unhandled intrinsics
    | _ ->
        None

// ═══════════════════════════════════════════════════════════════════════════
// BINARY/UNARY PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════

/// Try to witness a binary primitive operation
let private tryWitnessBinaryPrimitive
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [lhs; rhs] ->
        Primitives.tryWitnessBinaryOp appNodeId z opName lhs rhs
    | _ -> None

/// Try to witness a unary primitive operation
let private tryWitnessUnaryPrimitive
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        Primitives.witnessUnary appNodeId z opName arg
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a function application
/// This is called by FNCSTransfer for Application nodes
/// PHOTOGRAPHER PRINCIPLE: Returns ops, does not emit
let witness
    (appNodeId: NodeId)
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
            match Platform.witnessPlatformBinding appNodeId z entryPoint args mlirReturnType with
            | Some (ops, result) ->
                ops, result
            | None ->
                [], TRError (sprintf "Platform binding '%s' failed" entryPoint)

        // Intrinsic operations
        | SemanticKind.Intrinsic intrinsicInfo ->
            match witnessIntrinsic appNodeId z intrinsicInfo args returnType with
            | Some (ops, result) ->
                ops, result
            | None ->
                // Intrinsic not handled - return marker for deferred handling
                [], TRError (sprintf "Unhandled intrinsic: %s" intrinsicInfo.FullName)

        // Variable reference (function call or primitive)
        | SemanticKind.VarRef (name, defIdOpt) ->
            // Resolve actual function name from Lambda coeffects if this refs a Lambda
            let funcName =
                match defIdOpt with
                | Some defId ->
                    match Alex.Preprocessing.SSAAssignment.lookupLambdaName defId z.State.SSAAssignment with
                    | Some lambdaName -> lambdaName
                    | None -> name
                | None -> name  // No definition, use VarRef name
            // Try binary primitive first
            match tryWitnessBinaryPrimitive appNodeId z funcName args with
            | Some (ops, result) ->
                ops, result
            | None ->
                // Try unary primitive
                match tryWitnessUnaryPrimitive appNodeId z funcName args with
                | Some (ops, result) ->
                    ops, result
                | None ->
                    // Regular function call - emit func.call
                    // F# ≅ SSA: function application maps directly to func.call
                    // Check native type for unit (void semantics)
                    if isUnitType returnType then
                        // Void function - no result SSA needed
                        // Unit is represented as i32 in MLIR but call returns void
                        let callOp = MLIROp.FuncOp (FuncOp.FuncCall (None, funcName, args, mlirReturnType))
                        [callOp], TRVoid
                    else
                        // Function with return value - lookup pre-assigned SSA
                        match lookupNodeSSA appNodeId z with
                        | Some resultSSA ->
                            let callOp = MLIROp.FuncOp (FuncOp.FuncCall (Some resultSSA, funcName, args, mlirReturnType))
                            [callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                        | None ->
                            [], TRError (sprintf "No SSA assigned for function call result: %s" funcName)

        // Other kinds not handled here
        // NOTE: If we see SemanticKind.Application here, it means FNCS didn't flatten
        // curried calls. Per spec, FNCS should flatten fully-applied curried calls
        // to single Application nodes with all arguments.
        | _ ->
            [], TRError (sprintf "Unexpected function kind in application: %A" funcKind)

