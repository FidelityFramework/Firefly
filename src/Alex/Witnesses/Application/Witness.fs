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
module ArenaTemplates = Alex.Dialects.LLVM.Templates
module SCF = Alex.Dialects.SCF.Templates
module SSAAssignment = Alex.Preprocessing.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE RETURN TYPE LOOKUP
// ═══════════════════════════════════════════════════════════════════════════

/// For functions that return closures, find the actual ClosureStructType
/// by looking up the inner Lambda's ClosureLayout from SSAAssignment.
/// Returns None if the callee doesn't return a closure.
let rec private tryGetClosureReturnType (funcNodeId: NodeId) (z: PSGZipper) : MLIRType option =
    // Find the callee's definition (should be a Lambda or Binding to Lambda)
    match SemanticGraph.tryGetNode funcNodeId z.Graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.Lambda (_, bodyId, _) ->
            // The body of this Lambda might BE the returned closure
            // Or it might be a binding whose value is the closure
            // Traverse through the body to find the closure
            tryGetClosureFromBody bodyId z
        | SemanticKind.Binding (_, _, _, _) ->
            // Binding - check the children for the bound value
            match funcNode.Children with
            | [valueId] -> tryGetClosureReturnType valueId z
            | _ -> None
        | SemanticKind.VarRef (_, Some defId) ->
            // VarRef - recurse to the definition
            tryGetClosureReturnType defId z
        | SemanticKind.TypeAnnotation (innerExpr, _) ->
            // TypeAnnotation is transparent - look through it
            tryGetClosureReturnType innerExpr z
        | _ -> None
    | None -> None

/// Find a closure struct type from a body expression (the return value of a factory function)
and private tryGetClosureFromBody (nodeId: NodeId) (z: PSGZipper) : MLIRType option =
    match SemanticGraph.tryGetNode nodeId z.Graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Lambda (_, _, captures) when not (List.isEmpty captures) ->
            // The body IS the closure - get its ClosureLayout
            match SSAAssignment.lookupClosureLayout nodeId z.State.SSAAssignment with
            | Some layout -> Some layout.ClosureStructType
            | None -> None
        | SemanticKind.Sequential nodes when not (List.isEmpty nodes) ->
            // Sequential block - check the last expression (result)
            let resultId = List.last nodes
            tryGetClosureFromBody resultId z
        | SemanticKind.TypeAnnotation (innerExpr, _) ->
            // TypeAnnotation is transparent - look through it
            tryGetClosureFromBody innerExpr z
        | SemanticKind.Binding (_, _, _, _) ->
            // Let-in binding - the result is the body (second child usually)
            // But for closures, the body after the binding is what matters
            // Actually, for `let x = ... in <body>`, check <body>
            match node.Children with
            | [_valueId; bodyId] -> tryGetClosureFromBody bodyId z
            | _ -> None
        | _ -> None
    | None -> None

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

/// Witness String.contains - scan string for character
/// String.contains: string -> char -> bool
/// Generates a while loop that scans through the string
let private witnessStringContains
    (appNodeId: NodeId)
    (z: PSGZipper)
    (str: Val)
    (charVal: Val)
    : (MLIROp list * TransferResult) option =

    let ssas = requireNodeSSAs appNodeId z
    let mutable ssaIdx = 0
    let nextSSA () = let s = ssas.[ssaIdx] in ssaIdx <- ssaIdx + 1; s

    // Extract ptr and len from string struct
    let ptrSSA = nextSSA ()
    let lenSSA = nextSSA ()
    let extractPtrOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (ptrSSA, str.SSA, [0], str.Type))
    let extractLenOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, str.SSA, [1], str.Type))

    // Trunc char from i32 to i8 for byte comparison
    let charByteSSA = nextSSA ()
    let truncOp = MLIROp.ArithOp (ArithOp.TruncI (charByteSSA, charVal.SSA, charVal.Type, MLIRTypes.i8))

    // Constants
    let zeroSSA = nextSSA ()
    let falseSSA = nextSSA ()
    let oneSSA = nextSSA ()
    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i64))
    let falseOp = MLIROp.ArithOp (ArithOp.ConstI (falseSSA, 0L, MLIRTypes.i1))
    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))

    // Block args for while loop: (found: i1, idx: i64)
    let foundArgSSA = nextSSA ()
    let idxArgSSA = nextSSA ()
    let foundArg = { SSA = foundArgSSA; Type = MLIRTypes.i1 }
    let idxArg = { SSA = idxArgSSA; Type = MLIRTypes.i64 }

    // Condition region: continue while not found AND idx < len
    let notFoundSSA = nextSSA ()
    let inBoundsSSA = nextSSA ()
    let continueSSA = nextSSA ()
    let trueSSA = nextSSA ()
    let condOps = [
        MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.XOrI (notFoundSSA, foundArgSSA, trueSSA, MLIRTypes.i1))
        MLIROp.ArithOp (ArithOp.CmpI (inBoundsSSA, ICmpPred.Slt, idxArgSSA, lenSSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AndI (continueSSA, notFoundSSA, inBoundsSSA, MLIRTypes.i1))
        MLIROp.SCFOp (SCFOp.Condition (continueSSA, [foundArg; idxArg]))
    ]
    let condRegion = SCF.singleBlockRegion "" [foundArg; idxArg] condOps

    // Body region: load byte, compare, yield (match, idx+1)
    let bytePtrSSA = nextSSA ()
    let byteSSA = nextSSA ()
    let matchSSA = nextSSA ()
    let nextIdxSSA = nextSSA ()
    let bodyOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (bytePtrSSA, ptrSSA, [(idxArgSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Load (byteSSA, bytePtrSSA, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (matchSSA, ICmpPred.Eq, byteSSA, charByteSSA, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.AddI (nextIdxSSA, idxArgSSA, oneSSA, MLIRTypes.i64))
        MLIROp.SCFOp (SCFOp.Yield [{ SSA = matchSSA; Type = MLIRTypes.i1 }; { SSA = nextIdxSSA; Type = MLIRTypes.i64 }])
    ]
    let bodyRegion = SCF.singleBlockRegion "bb0" [foundArg; idxArg] bodyOps

    // While loop: returns (found, idx) starting at (false, 0)
    let resultFoundSSA = nextSSA ()
    let resultIdxSSA = nextSSA ()
    let iterArgs = [{ SSA = falseSSA; Type = MLIRTypes.i1 }; { SSA = zeroSSA; Type = MLIRTypes.i64 }]
    let whileOp = MLIROp.SCFOp (SCFOp.While ([resultFoundSSA; resultIdxSSA], condRegion, bodyRegion, iterArgs))

    let allOps = [extractPtrOp; extractLenOp; truncOp; zeroOp; falseOp; oneOp; whileOp]
    Some (allOps, TRValue { SSA = resultFoundSSA; Type = MLIRTypes.i1 })

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

    | "contains", [str; charVal] ->
        witnessStringContains appNodeId z str charVal

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
        // Source type is intVal.Type (i64 for nativeint), NOT ptr
        let op = MLIROp.LLVMOp (LLVMOp.IntToPtr (resultSSA, intVal.SSA, intVal.Type))
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

            // Unsigned integer conversions (same-size = identity/reinterpret)
            | "toUInt32", TInt I32, TInt I32 ->
                None  // Same size, just reinterpret
            | "toUInt16", TInt I16, TInt I16 ->
                None
            | "toUInt64", TInt I64, TInt I64 ->
                None
            // Unsigned widening (zero-extend)
            | "toUInt32", TInt w, TInt I32 when w < I32 ->
                Some (ArithOp.ExtUI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))
            | "toUInt64", TInt w, TInt I64 when w < I64 ->
                Some (ArithOp.ExtUI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))
            // Unsigned narrowing (truncate)
            | "toUInt16", TInt w, TInt I16 when w > I16 ->
                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i16))
            | "toUInt32", TInt w, TInt I32 when w > I32 ->
                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))

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
        | None when convName = "toUInt32" && arg.Type = MLIRTypes.i32 ->
            Some ([], TRValue arg)  // Identity (same bit representation)
        | None when convName = "toUInt64" && arg.Type = MLIRTypes.i64 ->
            Some ([], TRValue arg)  // Identity
        | None when convName = "toUInt16" && arg.Type = MLIRTypes.i16 ->
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

    | FormatOp "int64" ->
        match args with
        | [int64Val] ->
            // intToString handles i64 directly (no extension needed for i64 input)
            let ops, resultVal = Format.intToString appNodeId z int64Val
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

    | ParseOp "float" ->
        match args with
        | [strVal] ->
            let ops, resultVal = Format.stringToFloat appNodeId z strVal
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

    // Arena operations - deterministic bump allocation
    // Delegates to ArenaTemplates for MLIR construction
    | ArenaOp opName ->
        match opName, args with
        | "fromPointer", [baseArg; capacityArg] ->
            // Arena.fromPointer: nativeint -> int -> Arena<'lifetime>
            let ssas = requireNodeSSAs appNodeId z
            let mutable ssaIdx = 0
            let nextSSA () = let s = ssas.[ssaIdx] in ssaIdx <- ssaIdx + 1; s

            // Handle base: convert nativeint to ptr if needed
            let basePtrOps, basePtrSSA =
                if baseArg.Type = MLIRTypes.ptr then [], baseArg.SSA
                else
                    let convSSA = nextSSA()
                    [MLIROp.LLVMOp (LLVMOp.IntToPtr (convSSA, baseArg.SSA, baseArg.Type))], convSSA

            // Handle capacity: extend to i64 if needed
            let capExtOps, capI64SSA =
                if capacityArg.Type = MLIRTypes.i64 then [], capacityArg.SSA
                else
                    let extSSA = nextSSA()
                    [MLIROp.ArithOp (ArithOp.ExtSI (extSSA, capacityArg.SSA, capacityArg.Type, MLIRTypes.i64))], extSSA

            // SSAs for template
            let capPtrSSA = nextSSA()
            let zeroI64SSA = nextSSA()
            let zeroPtrSSA = nextSSA()
            let undefSSA = nextSSA()
            let withBaseSSA = nextSSA()
            let withCapSSA = nextSSA()

            // Zero constant + template
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroI64SSA, 0L, MLIRTypes.i64))
            let templateOps = ArenaTemplates.buildArena resultSSA basePtrSSA capI64SSA capPtrSSA zeroI64SSA zeroPtrSSA undefSSA withBaseSSA withCapSSA
                              |> List.map MLIROp.LLVMOp

            Some (basePtrOps @ capExtOps @ [zeroOp] @ templateOps, TRValue { SSA = resultSSA; Type = ArenaTemplates.arenaType })

        | "alloc", [arenaByref; sizeArg] ->
            // Arena.alloc: byref<Arena<'lifetime>> -> int -> nativeint
            let ssas = requireNodeSSAs appNodeId z
            let mutable ssaIdx = 0
            let nextSSA () = let s = ssas.[ssaIdx] in ssaIdx <- ssaIdx + 1; s

            let arenaSSA = nextSSA()
            let baseSSA = nextSSA()
            let posPtrSSA = nextSSA()
            let posI64SSA = nextSSA()

            // Handle size: extend to i64 if needed
            let sizeOps, sizeSSA =
                if sizeArg.Type = MLIRTypes.i64 then [], sizeArg.SSA
                else
                    let extSSA = nextSSA()
                    [MLIROp.ArithOp (ArithOp.ExtSI (extSSA, sizeArg.SSA, sizeArg.Type, MLIRTypes.i64))], extSSA

            let newPosI64SSA = nextSSA()
            let newPosPtrSSA = nextSSA()
            let resultPtrSSA = nextSSA()
            let newArenaSSA = nextSSA()
            let resultIntSSA = nextSSA()  // Final nativeint result

            // Use templates
            let extractOps = ArenaTemplates.extractArenaForAlloc arenaSSA arenaByref.SSA baseSSA posPtrSSA posI64SSA
                             |> List.map MLIROp.LLVMOp
            let (bumpLLVMOps, addOp) = ArenaTemplates.bumpArenaPosition arenaSSA arenaByref.SSA baseSSA posI64SSA sizeSSA newPosI64SSA newPosPtrSSA resultPtrSSA newArenaSSA
            let bumpOps = [MLIROp.ArithOp addOp] @ (bumpLLVMOps |> List.map MLIROp.LLVMOp)

            // Convert GEP result (ptr) to nativeint as per Arena.alloc signature
            let ptrToIntOp = MLIROp.LLVMOp (LLVMOp.PtrToInt (resultIntSSA, resultPtrSSA, MLIRTypes.i64))

            Some (extractOps @ sizeOps @ bumpOps @ [ptrToIntOp], TRValue { SSA = resultIntSSA; Type = MLIRTypes.i64 })

        | "remaining", [arena] ->
            // Arena.remaining: Arena<'lifetime> -> int
            let ssas = requireNodeSSAs appNodeId z
            let capPtrSSA, posPtrSSA, capI64SSA, posI64SSA, diffSSA = ssas.[0], ssas.[1], ssas.[2], ssas.[3], ssas.[4]

            let extractOps = ArenaTemplates.extractArenaForRemaining arena.SSA capPtrSSA posPtrSSA capI64SSA posI64SSA
                             |> List.map MLIROp.LLVMOp
            let mathOps = [
                MLIROp.ArithOp (ArithOp.SubI (diffSSA, capI64SSA, posI64SSA, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, diffSSA, MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (extractOps @ mathOps, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })

        | "reset", [arenaByref] ->
            // Arena.reset: byref<Arena<'lifetime>> -> unit
            let ssas = requireNodeSSAs appNodeId z
            let arenaSSA, zeroI64SSA, zeroPtrSSA, newArenaSSA = ssas.[0], ssas.[1], ssas.[2], ssas.[3]

            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroI64SSA, 0L, MLIRTypes.i64))
            let templateOps = ArenaTemplates.resetArenaPosition arenaSSA arenaByref.SSA zeroI64SSA zeroPtrSSA newArenaSSA
                              |> List.map MLIROp.LLVMOp
            Some ([zeroOp] @ templateOps, TRVoid)

        | _ -> None

    // DateTime operations
    | DateTimeOp opName ->
        match opName, args with
        | "now", [] | "utcNow", [] ->
            // Delegates to clock_gettime syscall
            Platform.witnessSysOp appNodeId z "clock_gettime" [] mlirReturnType
        | "hour", [msVal] ->
            // ms / 3600000 % 24
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 3600000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[1], 24L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[2], msVal.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[3], ssas.[2], ssas.[1], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[3], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "minute", [msVal] ->
            // ms / 60000 % 60
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 60000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[1], 60L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[2], msVal.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[3], ssas.[2], ssas.[1], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[3], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "second", [msVal] ->
            // ms / 1000 % 60
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[1], 60L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[2], msVal.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[3], ssas.[2], ssas.[1], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[3], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "millisecond", [msVal] ->
            // ms % 1000
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[1], msVal.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[1], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "utcOffset", [] ->
            // Get timezone offset via platform binding (uses libc localtime_r)
            Platform.witnessPlatformBinding appNodeId z "DateTime.utcOffset" [] mlirReturnType
        | "toLocal", [utcMs] ->
            // Convert UTC to local via platform binding
            Platform.witnessPlatformBinding appNodeId z "DateTime.toLocal" [utcMs] mlirReturnType
        | "toUtc", [localMs] ->
            // Convert local to UTC via platform binding
            Platform.witnessPlatformBinding appNodeId z "DateTime.toUtc" [localMs] mlirReturnType
        | _ ->
            // Other DateTime ops (formatting) - not yet implemented
            None

    // TimeSpan operations
    | TimeSpanOp opName ->
        match opName, args with
        | "fromMilliseconds", [ms] ->
            // Identity - TimeSpan is represented as int64 milliseconds
            Some ([], TRValue { SSA = ms.SSA; Type = ms.Type })
        | "fromSeconds", [sec] ->
            // sec * 1000
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.MulI (resultSSA, sec.SSA, ssas.[0], MLIRTypes.i64))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i64 })
        | "hours", [ts] ->
            // ts / 3600000
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 3600000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[1], ts.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[1], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "minutes", [ts] ->
            // (ts / 60000) % 60
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 60000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[1], 60L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[2], ts.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[3], ssas.[2], ssas.[1], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[3], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "seconds", [ts] ->
            // (ts / 1000) % 60
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[1], 60L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[2], ts.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[3], ssas.[2], ssas.[1], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[3], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "milliseconds", [ts] ->
            // ts % 1000
            let ssas = requireNodeSSAs appNodeId z
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[1], ts.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[1], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | _ -> None

    // Bits operations - byte swapping and bit casting
    | BitsOp opName ->
        match opName, args with
        | "htons", [val16] | "ntohs", [val16] ->
            // Byte swap uint16 using llvm.intr.bswap
            let op = MLIROp.LLVMOp (LLVMOp.Bswap (resultSSA, val16.SSA, MLIRTypes.i16))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.i16 })
        | "htonl", [val32] | "ntohl", [val32] ->
            // Byte swap uint32 using llvm.intr.bswap
            let op = MLIROp.LLVMOp (LLVMOp.Bswap (resultSSA, val32.SSA, MLIRTypes.i32))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "float32ToInt32Bits", [f32] ->
            // Bitcast float32 to int32
            let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, f32.SSA, MLIRTypes.f32, MLIRTypes.i32))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "int32BitsToFloat32", [i32] ->
            // Bitcast int32 to float32
            let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, i32.SSA, MLIRTypes.i32, MLIRTypes.f32))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.f32 })
        | "float64ToInt64Bits", [f64] ->
            // Bitcast float64 to int64
            let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, f64.SSA, MLIRTypes.f64, MLIRTypes.i64))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 })
        | "int64BitsToFloat64", [i64] ->
            // Bitcast int64 to float64
            let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, i64.SSA, MLIRTypes.i64, MLIRTypes.f64))
            Some ([op], TRValue { SSA = resultSSA; Type = MLIRTypes.f64 })
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
    // Use graph-aware mapping for record types
    let mlirReturnType = mapNativeTypeWithGraph z.Graph returnType

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

            // Check if this is a closure call (VarRef points to a closure value)
            // If so, we need indirect call through the closure struct
            // Closures can be:
            // 1. Lambda with captures (inline closure)
            // 2. Binding to Lambda with captures
            // 3. Binding to Application that returns a function type (closure factory result)
            let isClosureCall =
                match defIdOpt with
                | Some defId ->
                    match SemanticGraph.tryGetNode defId z.Graph with
                    | Some defNode ->
                        match defNode.Kind with
                        | SemanticKind.Lambda (_, _, captures) -> not (List.isEmpty captures)
                        | SemanticKind.Binding (_, _, _, _) ->
                            // Check the bound value
                            match defNode.Children with
                            | [childId] ->
                                match SemanticGraph.tryGetNode childId z.Graph with
                                | Some childNode ->
                                    match childNode.Kind with
                                    | SemanticKind.Lambda (_, _, captures) ->
                                        // Binding to Lambda with captures
                                        not (List.isEmpty captures)
                                    | SemanticKind.Application _ ->
                                        // Binding to Application - check if result is a function type
                                        // e.g., `let counter = makeCounter 0` where makeCounter returns a closure
                                        match childNode.Type with
                                        | NativeType.TFun _ -> true  // Returns function = closure factory result
                                        | _ -> false
                                    | _ -> false
                                | None -> false
                            | _ -> false
                        | _ -> false
                    | None -> false
                | None -> false

            if isClosureCall then
                // TRUE FLAT CLOSURE CALL:
                // Closure struct layout: {code_ptr: ptr, capture_0, capture_1, ...}
                // Extract code_ptr (index 0), then call with ENTIRE closure struct as first arg
                // The callee extracts captures from the closure struct

                // Get the closure struct SSA - check VarBindings first (params/captures),
                // then NodeBindings (local let bindings)
                let closureSSAOpt =
                    match recallVarSSA name z with
                    | Some (ssa, ty) -> Some (ssa, ty)
                    | None ->
                        // Not in VarBindings - check NodeBindings using defId
                        match defIdOpt with
                        | Some defId ->
                            match recallNodeResult (NodeId.value defId) z with
                            | Some (ssa, ty) -> Some (ssa, ty)
                            | None -> None
                        | None -> None
                match closureSSAOpt with
                | Some (closureSSA, closureType) ->
                    // Generate synthetic SSA for code_ptr extraction
                    let codePtrSSA = freshSynthSSA z

                    // Extract code_ptr (index 0)
                    let extractCodeOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, closureSSA, [0], closureType))

                    // Pass ENTIRE closure struct as first arg (not just env_ptr)
                    // The callee function receives it and extracts captures
                    let closureArg = { SSA = closureSSA; Type = closureType }
                    let callArgs = closureArg :: args

                    // Indirect call through code_ptr
                    if isUnitType returnType then
                        let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (None, codePtrSSA, callArgs, mlirReturnType))
                        [extractCodeOp; callOp], TRVoid
                    else
                        match lookupNodeSSA appNodeId z with
                        | Some resultSSA ->
                            let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, callArgs, mlirReturnType))
                            [extractCodeOp; callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                        | None ->
                            [], TRError (sprintf "No SSA assigned for closure call result: %s" name)
                | None ->
                    [], TRError (sprintf "Closure '%s' not bound in scope" name)
            else
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
                                // TRUE FLAT CLOSURE: If callee returns a closure, use actual closure struct type
                                // (not the generic {ptr, ptr} from mapNativeType(TFun))
                                let effectiveRetType =
                                    match defIdOpt with
                                    | Some defId ->
                                        match tryGetClosureReturnType defId z with
                                        | Some closureType -> closureType
                                        | None -> mlirReturnType
                                    | None -> mlirReturnType
                                let callOp = MLIROp.FuncOp (FuncOp.FuncCall (Some resultSSA, funcName, args, effectiveRetType))
                                [callOp], TRValue { SSA = resultSSA; Type = effectiveRetType }
                            | None ->
                                [], TRError (sprintf "No SSA assigned for function call result: %s" funcName)

        // Nested Application - the function is itself an Application result (closure)
        // This happens with curried functions: App(App(makeCounter, [0]), [_eta0])
        // The inner Application returns a closure struct; we call through it
        // TRUE FLAT CLOSURE: Pass entire closure struct to callee
        | SemanticKind.Application (_, _) ->
            // Look up the result of the inner Application from NodeBindings
            // Post-order traversal guarantees it's already processed
            match recallNodeResult (NodeId.value funcNodeId) z with
            | Some (closureSSA, closureType) ->
                // The result is a flat closure struct {code_ptr, capture_0, capture_1, ...}
                // Do an indirect call through code_ptr, passing entire closure as first arg
                let codePtrSSA = freshSynthSSA z

                // Extract code_ptr (index 0)
                let extractCodeOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, closureSSA, [0], closureType))

                // Pass ENTIRE closure struct as first arg
                let closureArg = { SSA = closureSSA; Type = closureType }
                let callArgs = closureArg :: args

                // Indirect call through code_ptr
                if isUnitType returnType then
                    let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (None, codePtrSSA, callArgs, mlirReturnType))
                    [extractCodeOp; callOp], TRVoid
                else
                    match lookupNodeSSA appNodeId z with
                    | Some resultSSA ->
                        let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, callArgs, mlirReturnType))
                        [extractCodeOp; callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                    | None ->
                        [], TRError (sprintf "No SSA assigned for nested closure call result")
            | None ->
                [], TRError (sprintf "Nested Application result not found in NodeBindings: %d" (NodeId.value funcNodeId))

        // Other kinds not handled
        | _ ->
            [], TRError (sprintf "Unexpected function kind in application: %A" funcKind)

