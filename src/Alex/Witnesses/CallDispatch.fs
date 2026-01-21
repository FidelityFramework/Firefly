/// CallDispatch - Witness function application semantics to MLIR
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This module handles APPLICATION node witnessing. It delegates to factored
/// modules (ArithOps, FormatOps, SyscallOps) based on SemanticKind.
///
/// Returns structured MLIROp list via the factored witness modules.
///
/// DIALECT BOUNDARY (January 2026):
/// This module respects the MLIR dialect hierarchy:
///
/// PORTABLE DIALECTS (func, arith, cf, scf):
///   - Direct function calls by name → FuncOp.FuncCall
///   - Arithmetic operations → ArithOp.*
///   - Control flow → CFOp.*, SCFOp.*
///
/// LLVM DIALECT (backend-target-intrinsic):
///   - Indirect calls through function pointers → LLVMOp.IndirectCall
///   - Struct manipulation (extractvalue, insertvalue, gep) → LLVMOp.*
///   - Functions whose address is taken → llvm.func (in LambdaWitness)
///
/// See fsnative-spec/spec/drafts/backend-lowering-architecture.md and
/// Serena memory `mlir_dialect_architecture` for the full specification.
module Alex.Witnesses.CallDispatch

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
// NOTE: LLVM.Templates imported for struct ops (extractvalue, etc.) and indirect calls.
// Do NOT use `callFunc` for direct calls - use FuncOp.FuncCall instead.
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns
open Alex.Bindings.PlatformTypes

// Import factored witness modules
module Primitives = Alex.Witnesses.ArithOps
module Format = Alex.Witnesses.FormatOps
module SyscallOps = Alex.Witnesses.SyscallOps
module ArenaTemplates = Alex.Dialects.LLVM.Templates
module SCF = Alex.Dialects.SCF.Templates
module SSAAssignment = PSGElaboration.SSAAssignment
module LazyWitness = Alex.Witnesses.LazyWitness
module SeqOpWitness = Alex.Witnesses.SeqOpWitness
// PRD-13a: Collection Witnesses
module ListWitness = Alex.Witnesses.ListWitness
module MapWitness = Alex.Witnesses.MapWitness
module SetWitness = Alex.Witnesses.SetWitness
module OptionWitness = Alex.Witnesses.OptionWitness

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS (use ctx: WitnessContext)
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSAs for a node, fail if not found
let private requireNodeSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssignment.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get result SSA for a node (the final SSA from its allocation)
/// Nodes may have multiple SSAs for intermediate values; this returns the Result
let private requireNodeSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssignment.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No result SSA for node %A" nodeId

/// Recall the result (SSA, type) for a previously-processed node
let private recallNodeResult (nodeId: NodeId) (ctx: WitnessContext) : (SSA * MLIRType) option =
    MLIRAccumulator.recallNode (NodeId.value nodeId) ctx.Accumulator

/// Lookup SSA for a node (returns option, doesn't fail)
let private lookupNodeSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA option =
    match SSAAssignment.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> Some ssa
    | None -> None

/// Recall the SSA for a variable (from variable bindings)
let private recallVarSSA (name: string) (ctx: WitnessContext) : (SSA * MLIRType) option =
    MLIRAccumulator.recallVar name ctx.Accumulator

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE RETURN TYPE LOOKUP
// ═══════════════════════════════════════════════════════════════════════════

/// For functions that return closures, find the actual ClosureStructType
/// by looking up the inner Lambda's ClosureLayout from SSAAssignment.
/// Returns None if the callee doesn't return a closure.
let rec private tryGetClosureReturnType (funcNodeId: NodeId) (ctx: WitnessContext) : MLIRType option =
    // Find the callee's definition (should be a Lambda or Binding to Lambda)
    match SemanticGraph.tryGetNode funcNodeId ctx.Graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.Lambda (_, bodyId, _, _, _) ->
            // The body of this Lambda might BE the returned closure
            // Or it might be a binding whose value is the closure
            // Traverse through the body to find the closure
            tryGetClosureFromBody bodyId ctx
        | SemanticKind.Binding (_, _, _, _) ->
            // Binding - check the children for the bound value
            match funcNode.Children with
            | [valueId] -> tryGetClosureReturnType valueId ctx
            | _ -> None
        | SemanticKind.VarRef (_, Some defId) ->
            // VarRef - recurse to the definition
            tryGetClosureReturnType defId ctx
        | SemanticKind.TypeAnnotation (innerExpr, _) ->
            // TypeAnnotation is transparent - look through it
            tryGetClosureReturnType innerExpr ctx
        | _ -> None
    | None -> None

/// Find a closure struct type from a body expression (the return value of a factory function)
and private tryGetClosureFromBody (nodeId: NodeId) (ctx: WitnessContext) : MLIRType option =
    match SemanticGraph.tryGetNode nodeId ctx.Graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Lambda (_, _, captures, _, _) when not (List.isEmpty captures) ->
            // The body IS the closure - get its ClosureLayout
            match SSAAssignment.lookupClosureLayout nodeId ctx.Coeffects.SSA with
            | Some layout -> Some layout.ClosureStructType
            | None -> None
        | SemanticKind.Sequential nodes when not (List.isEmpty nodes) ->
            // Sequential block - check the last expression (result)
            let resultId = List.last nodes
            tryGetClosureFromBody resultId ctx
        | SemanticKind.TypeAnnotation (innerExpr, _) ->
            // TypeAnnotation is transparent - look through it
            tryGetClosureFromBody innerExpr ctx
        | SemanticKind.Binding (_, _, _, _) ->
            // Let-in binding - the result is the body (second child usually)
            // But for closures, the body after the binding is what matters
            // Actually, for `let x = ... in <body>`, check <body>
            match node.Children with
            | [_valueId; bodyId] -> tryGetClosureFromBody bodyId ctx
            | _ -> None
        | _ -> None
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// ARGUMENT RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

let private resolveArgs (argNodeIds: NodeId list) (ctx: WitnessContext) : Val list =
    argNodeIds
    |> List.choose (fun nodeId ->
        let isUnitArg =
            match SemanticGraph.tryGetNode nodeId ctx.Graph with
            | Some node ->
                match node.Type with
                | NativeType.TApp(tc, _) when tc.Name = "unit" -> true
                | _ -> false
            | None -> false
        if isUnitArg then None
        else
            match recallNodeResult nodeId ctx with
            | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
            | None -> failwithf "No result for arg %d" (NodeId.value nodeId))

/// Resolve function node to its SemanticKind
/// Looks through TypeAnnotation nodes to find the underlying function
let private resolveFuncKind (funcNodeId: NodeId) (ctx: WitnessContext) : SemanticKind option =
    let rec resolve nodeId =
        match SemanticGraph.tryGetNode nodeId ctx.Graph with
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
let private witnessZeroed (appNodeId: NodeId) (ctx: WitnessContext) (resultType: MLIRType) : (MLIROp list * TransferResult) option =
    let resultSSA = requireNodeSSA appNodeId ctx

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
let private witnessStringConcat (appNodeId: NodeId) (ctx: WitnessContext) (str1: Val) (str2: Val) : (MLIROp list * TransferResult) option =
    // Get pre-assigned SSAs (10 total)
    let ssas = requireNodeSSAs appNodeId ctx
    let resultSSA = requireNodeSSA appNodeId ctx
    
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
    (ctx: WitnessContext)
    (str: Val)
    (charVal: Val)
    : (MLIROp list * TransferResult) option =

    let ssas = requireNodeSSAs appNodeId ctx
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
    (ctx: WitnessContext)
    (opName: string)
    (args: Val list)
    (_resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match opName, args with
    | "concat2", [str1; str2] ->
        witnessStringConcat appNodeId ctx str1 str2

    | "contains", [str; charVal] ->
        witnessStringContains appNodeId ctx str charVal

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativePtr intrinsic operations
let private witnessNativePtrOp
    (appNodeId: NodeId)
    (ctx: WitnessContext)
    (opKind: NativePtrOpKind)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    let resultSSA = requireNodeSSA appNodeId ctx

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
        let ssas = requireNodeSSAs appNodeId ctx
        let gepSSA = ssas.[0]
        let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptr.SSA, [(idx.SSA, idx.Type)], resultType))
        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, gepSSA, resultType, NotAtomic))
        Some ([gepOp; loadOp], TRValue { SSA = resultSSA; Type = resultType })

    | PtrSet, [ptr; idx; value] ->
        // NativePtr.set: ptr -> int -> 'T -> unit (indexed store)
        // Needs 1 SSA for gep intermediate
        let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
    (ctx: WitnessContext)
    (convName: string)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        let resultSSA = requireNodeSSA appNodeId ctx

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
    (ctx: WitnessContext)
    (intrinsicInfo: IntrinsicInfo)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch returnType
    // Get pre-assigned result SSA for this Application node
    let resultSSA = requireNodeSSA appNodeId ctx

    match intrinsicInfo with
    // Platform operations - delegate to Platform module
    | SysOp opName ->
        match SyscallOps.witnessSysOp appNodeId ctx.Coeffects.SSA opName args mlirReturnType with
        | Some (inlineOps, topLevelOps, result) -> Some (inlineOps @ topLevelOps, result)
        | None -> None

    // NOTE: Console is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
    // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md

    // Format operations - delegate to Format module
    | FormatOp "int" ->
        match args with
        | [intVal] ->
            let ops, resultVal = Format.intToString appNodeId ctx.Coeffects.SSA intVal
            Some (ops, TRValue resultVal)
        | _ -> None

    | FormatOp "int64" ->
        match args with
        | [int64Val] ->
            // intToString handles i64 directly (no extension needed for i64 input)
            let ops, resultVal = Format.intToString appNodeId ctx.Coeffects.SSA int64Val
            Some (ops, TRValue resultVal)
        | _ -> None

    | FormatOp "float" ->
        match args with
        | [floatVal] ->
            let ops, resultVal = Format.floatToString appNodeId ctx.Coeffects.SSA floatVal
            Some (ops, TRValue resultVal)
        | _ -> None

    // Parse operations - delegate to Format module
    | ParseOp "int" ->
        match args with
        | [strVal] ->
            let ops, resultVal = Format.stringToInt appNodeId ctx.Coeffects.SSA strVal
            Some (ops, TRValue resultVal)
        | _ -> None

    | ParseOp "float" ->
        match args with
        | [strVal] ->
            let ops, resultVal = Format.stringToFloat appNodeId ctx.Coeffects.SSA strVal
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
                    let ssas = requireNodeSSAs appNodeId ctx
                    let extSSA = ssas.[0]  // First SSA for intermediate
                    let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, count.SSA, count.Type, MLIRTypes.i64))
                    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, extSSA, elemType, None))
                    Some ([extOp; allocOp], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr })
            | None, _ ->
                // Fall back if we can't extract element type
                witnessNativePtrOp appNodeId ctx opKind args mlirReturnType
            | _, _ ->
                witnessNativePtrOp appNodeId ctx opKind args mlirReturnType
        | _ ->
            witnessNativePtrOp appNodeId ctx opKind args mlirReturnType

    // Convert operations - use Arith dialect directly
    | ConvertOp convName ->
        witnessConversion appNodeId ctx convName args mlirReturnType

    // String operations
    | StringOp opName ->
        witnessStringOp appNodeId ctx opName args mlirReturnType

    // Default zeroed value
    | NativeDefaultOp "zeroed" ->
        witnessZeroed appNodeId ctx mlirReturnType

    // Platform introspection - compile-time constants based on target architecture
    | PlatformModuleOp opName ->
        match opName with
        | "wordSize" ->
            // Platform.wordSize : int - returns word size in bytes
            // 8 on 64-bit (x86_64, ARM64, RISCV64), 4 on 32-bit (ARM32, RISCV32, WASM32)
            let wordBytes =
                match ctx.Coeffects.Platform.TargetArch with
                | Architecture.X86_64 | Architecture.ARM64 | Architecture.RISCV64 -> 8L
                | Architecture.ARM32_Thumb | Architecture.RISCV32 | Architecture.WASM32 -> 4L
            let op = MLIROp.ArithOp (ArithOp.ConstI (resultSSA, wordBytes, mlirReturnType))
            Some ([op], TRValue { SSA = resultSSA; Type = mlirReturnType })
        | "sizeof" ->
            // Platform.sizeof<'T> : int - returns size of type 'T in bytes
            // The type argument is resolved from the Application node's type instantiation
            // For now, we resolve based on the return type context (mlirReturnType should be int)
            // The actual type to measure comes from the type parameter - we need to extract it
            // from the node's type instantiation
            match SemanticGraph.tryGetNode appNodeId ctx.Graph with
            | Some node ->
                // The sizeof call should have a type argument - extract it from the Application
                // For sizeof<int>, the instantiation includes the 'int' type
                let sizeBytes =
                    // Get the type instantiation from the application
                    // This is stored in node metadata - for now, use return type mapping as proxy
                    // TODO: Extract actual type parameter from type instantiation
                    match mlirReturnType with
                    | TInt I8 -> 1L
                    | TInt I16 -> 2L
                    | TInt I32 -> 4L
                    | TInt I64 -> 8L
                    | TFloat F32 -> 4L
                    | TFloat F64 -> 8L
                    | TPtr ->
                        match ctx.Coeffects.Platform.TargetArch with
                        | Architecture.X86_64 | Architecture.ARM64 | Architecture.RISCV64 -> 8L
                        | Architecture.ARM32_Thumb | Architecture.RISCV32 | Architecture.WASM32 -> 4L
                    | _ -> 8L  // Default to word size for unknown types
                let op = MLIROp.ArithOp (ArithOp.ConstI (resultSSA, sizeBytes, mlirReturnType))
                Some ([op], TRValue { SSA = resultSSA; Type = mlirReturnType })
            | None -> None
        | _ -> None

    // Operators module - delegate to Primitives
    // Handles arithmetic, comparison, bitwise, and unary operators (including "not")
    | _ when intrinsicInfo.Module = IntrinsicModule.Operators || intrinsicInfo.FullName.StartsWith("op_") || intrinsicInfo.FullName = "not" ->
        match args with
        | [lhs; rhs] ->
             Primitives.tryWitnessBinaryOp appNodeId ctx.Coeffects.SSA intrinsicInfo.Operation lhs rhs
        | [arg] ->
             Primitives.witnessUnary appNodeId ctx.Coeffects.SSA intrinsicInfo.Operation arg
        | _ -> None

    // NativeStr operations
    | NativeStrOp opName ->
        match opName, args with
        | "fromPointer", [ptr; len] ->
             // NativeStr.fromPointer: ptr -> i64 -> nativestr
             // Requires up to 4 SSAs: optional extsi + undef + withPtr + result
             let ssas = requireNodeSSAs appNodeId ctx
             
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            match SyscallOps.witnessSysOp appNodeId ctx.Coeffects.SSA "clock_gettime" [] mlirReturnType with
            | Some (inlineOps, topLevelOps, result) -> Some (inlineOps @ topLevelOps, result)
            | None -> None
        | "hour", [msVal] ->
            // ms / 3600000 % 24
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.RemSI (ssas.[1], msVal.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[1], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "utcOffset", [] ->
            // Get timezone offset via platform binding (uses libc localtime_r)
            match SyscallOps.witnessPlatformBinding appNodeId ctx.Coeffects.SSA "DateTime.utcOffset" [] mlirReturnType with
            | Some (inlineOps, topLevelOps, result) -> Some (inlineOps @ topLevelOps, result)
            | None -> None
        | "toLocal", [utcMs] ->
            // Convert UTC to local via platform binding
            match SyscallOps.witnessPlatformBinding appNodeId ctx.Coeffects.SSA "DateTime.toLocal" [utcMs] mlirReturnType with
            | Some (inlineOps, topLevelOps, result) -> Some (inlineOps @ topLevelOps, result)
            | None -> None
        | "toUtc", [localMs] ->
            // Convert local to UTC via platform binding
            match SyscallOps.witnessPlatformBinding appNodeId ctx.Coeffects.SSA "DateTime.toUtc" [localMs] mlirReturnType with
            | Some (inlineOps, topLevelOps, result) -> Some (inlineOps @ topLevelOps, result)
            | None -> None
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
            let ssas = requireNodeSSAs appNodeId ctx
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 1000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.MulI (resultSSA, sec.SSA, ssas.[0], MLIRTypes.i64))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i64 })
        | "hours", [ts] ->
            // ts / 3600000
            let ssas = requireNodeSSAs appNodeId ctx
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 3600000L, MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.DivSI (ssas.[1], ts.SSA, ssas.[0], MLIRTypes.i64))
                MLIROp.ArithOp (ArithOp.TruncI (resultSSA, ssas.[1], MLIRTypes.i64, MLIRTypes.i32))
            ]
            Some (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i32 })
        | "minutes", [ts] ->
            // (ts / 60000) % 60
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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
            let ssas = requireNodeSSAs appNodeId ctx
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

    // PRD-14: Lazy operations handled via SemanticKind.LazyExpr and LazyForce
    // in FNCSTransfer.fs, not through intrinsic pattern matching.

    // PRD-16: Seq operations (map, filter, take, fold, collect)
    // SeqOpWitness returns 3-tuples: (inlineOps, topLevelOps, result)
    // We concatenate topLevelOps with inlineOps - serializer handles function definitions
    | SeqOp opName ->
        match opName, args with
        | "map", [mapper; innerSeq] ->
            // Seq.map : ('T -> 'U) -> seq<'T> -> seq<'U>
            let outputElementType =
                match returnType with
                | NativeType.TSeq elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64  // Fallback
            let inputElementType =
                match innerSeq.Type with
                | TStruct (TInt I32 :: elemTy :: _) -> elemTy  // Extract from Seq struct
                | _ -> MLIRTypes.i64  // Fallback
            let (inlineOps, topLevelOps, result) = SeqOpWitness.witnessSeqMap appNodeId ctx.Coeffects.SSA mapper innerSeq inputElementType outputElementType
            Some (topLevelOps @ inlineOps, result)

        | "filter", [predicate; innerSeq] ->
            // Seq.filter : ('T -> bool) -> seq<'T> -> seq<'T>
            let elementType =
                match returnType with
                | NativeType.TSeq elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (inlineOps, topLevelOps, result) = SeqOpWitness.witnessSeqFilter appNodeId ctx.Coeffects.SSA predicate innerSeq elementType
            Some (topLevelOps @ inlineOps, result)

        | "take", [count; innerSeq] ->
            // Seq.take : int -> seq<'T> -> seq<'T>
            let elementType =
                match returnType with
                | NativeType.TSeq elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (inlineOps, topLevelOps, result) = SeqOpWitness.witnessSeqTake appNodeId ctx.Coeffects.SSA count innerSeq elementType
            Some (topLevelOps @ inlineOps, result)

        | "fold", [folder; initial; seq] ->
            // Seq.fold : ('S -> 'T -> 'S) -> 'S -> seq<'T> -> 'S
            let accType = mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch returnType
            let elementType =
                match seq.Type with
                | TStruct (TInt I32 :: elemTy :: _) -> elemTy
                | _ -> MLIRTypes.i64
            let (inlineOps, topLevelOps, result) = SeqOpWitness.witnessSeqFold appNodeId ctx.Coeffects.SSA folder initial seq accType elementType
            Some (topLevelOps @ inlineOps, result)

        | "collect", [mapper; outerSeq] ->
            // Seq.collect : ('T -> seq<'U>) -> seq<'T> -> seq<'U>
            let outputElementType =
                match returnType with
                | NativeType.TSeq elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (inlineOps, topLevelOps, result) = SeqOpWitness.witnessSeqCollect appNodeId ctx.Coeffects.SSA mapper outerSeq outputElementType
            Some (topLevelOps @ inlineOps, result)

        | _ ->
            // Other Seq operations not yet implemented
            None

    // PRD-13a: List operations
    | ListOp opName ->
        match opName, args with
        | "empty", [] ->
            let (ops, result) = ListWitness.witnessEmpty appNodeId ctx.Coeffects.SSA
            Some (ops, result)
        
        | "isEmpty", [listVal] ->
            let (ops, result) = ListWitness.witnessIsEmpty appNodeId ctx.Coeffects.SSA listVal
            Some (ops, result)
        
        | "head", [listVal] ->
            let elementType =
                match returnType with
                | NativeType.TList elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (ops, result) = ListWitness.witnessHead appNodeId ctx.Coeffects.SSA listVal elementType
            Some (ops, result)
        
        | "tail", [listVal] ->
            let elementType =
                match listVal.Type with
                | TStruct [elemTy; TPtr] -> elemTy
                | _ -> MLIRTypes.i64
            let (ops, result) = ListWitness.witnessTail appNodeId ctx.Coeffects.SSA listVal elementType
            Some (ops, result)
        
        | "cons", [headVal; tailVal] ->
            let elementType = headVal.Type
            let (ops, result) = ListWitness.witnessCons appNodeId ctx.Coeffects.SSA headVal tailVal elementType
            Some (ops, result)
        
        | _ ->
            // Other List operations (map, filter, fold, etc.) - require functional decomposition
            None

    // PRD-13a: Map operations
    | MapOp opName ->
        match opName, args with
        | "empty", [] ->
            let (ops, result) = MapWitness.witnessEmpty appNodeId ctx.Coeffects.SSA
            Some (ops, result)
        
        | "isEmpty", [mapVal] ->
            let (ops, result) = MapWitness.witnessIsEmpty appNodeId ctx.Coeffects.SSA mapVal
            Some (ops, result)
        
        | "tryFind", [keyVal; mapVal] ->
            let keyType = keyVal.Type
            // Option is represented as TUnion, extract value type from return type if possible
            let valueType = mlirReturnType  // Use the mapped return type
            let (ops, result) = MapWitness.witnessTryFind appNodeId ctx.Coeffects.SSA keyVal mapVal keyType valueType
            Some (ops, result)
        
        | "add", [keyVal; valueVal; mapVal] ->
            let keyType = keyVal.Type
            let valueType = valueVal.Type
            let (ops, result) = MapWitness.witnessAdd appNodeId ctx.Coeffects.SSA keyVal valueVal mapVal keyType valueType
            Some (ops, result)
        
        | "containsKey", [keyVal; mapVal] ->
            let keyType = keyVal.Type
            let (ops, result) = MapWitness.witnessContainsKey appNodeId ctx.Coeffects.SSA keyVal mapVal keyType
            Some (ops, result)
        
        | "values", [mapVal] ->
            let valueType =
                match returnType with
                | NativeType.TList valueTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch valueTy
                | _ -> MLIRTypes.i64
            let (ops, result) = MapWitness.witnessValues appNodeId ctx.Coeffects.SSA mapVal valueType
            Some (ops, result)
        
        | "keys", [mapVal] ->
            let keyType =
                match returnType with
                | NativeType.TList keyTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch keyTy
                | _ -> MLIRTypes.i64
            let (ops, result) = MapWitness.witnessKeys appNodeId ctx.Coeffects.SSA mapVal keyType
            Some (ops, result)
        
        | _ ->
            // Other Map operations - require functional decomposition or complex tree traversal
            None

    // PRD-13a: Set operations
    | SetOp opName ->
        match opName, args with
        | "empty", [] ->
            let (ops, result) = SetWitness.witnessEmpty appNodeId ctx.Coeffects.SSA
            Some (ops, result)
        
        | "isEmpty", [setVal] ->
            let (ops, result) = SetWitness.witnessIsEmpty appNodeId ctx.Coeffects.SSA setVal
            Some (ops, result)
        
        | "contains", [valueVal; setVal] ->
            let elementType = valueVal.Type
            let (ops, result) = SetWitness.witnessContains appNodeId ctx.Coeffects.SSA valueVal setVal elementType
            Some (ops, result)
        
        | "add", [valueVal; setVal] ->
            let elementType = valueVal.Type
            let (ops, result) = SetWitness.witnessAdd appNodeId ctx.Coeffects.SSA valueVal setVal elementType
            Some (ops, result)
        
        | "remove", [valueVal; setVal] ->
            let elementType = valueVal.Type
            let (ops, result) = SetWitness.witnessRemove appNodeId ctx.Coeffects.SSA valueVal setVal elementType
            Some (ops, result)
        
        | "union", [set1Val; set2Val] ->
            let elementType =
                match returnType with
                | NativeType.TSet elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (ops, result) = SetWitness.witnessUnion appNodeId ctx.Coeffects.SSA set1Val set2Val elementType
            Some (ops, result)
        
        | "intersect", [set1Val; set2Val] ->
            let elementType =
                match returnType with
                | NativeType.TSet elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (ops, result) = SetWitness.witnessIntersect appNodeId ctx.Coeffects.SSA set1Val set2Val elementType
            Some (ops, result)
        
        | "difference", [set1Val; set2Val] ->
            let elementType =
                match returnType with
                | NativeType.TSet elemTy -> mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch elemTy
                | _ -> MLIRTypes.i64
            let (ops, result) = SetWitness.witnessDifference appNodeId ctx.Coeffects.SSA set1Val set2Val elementType
            Some (ops, result)
        
        | _ ->
            // Other Set operations - require functional decomposition
            None

    // PRD-13a: Option operations
    | OptionOp opName ->
        match opName, args with
        | "isSome", [optionVal] ->
            let valueType =
                match optionVal.Type with
                | TStruct [TInt I8; valTy] -> valTy
                | _ -> MLIRTypes.i64
            let (ops, result) = OptionWitness.witnessIsSome appNodeId ctx.Coeffects.SSA optionVal valueType
            Some (ops, result)
        
        | "isNone", [optionVal] ->
            let valueType =
                match optionVal.Type with
                | TStruct [TInt I8; valTy] -> valTy
                | _ -> MLIRTypes.i64
            let (ops, result) = OptionWitness.witnessIsNone appNodeId ctx.Coeffects.SSA optionVal valueType
            Some (ops, result)
        
        | "get", [optionVal] ->
            // Option.get returns the unwrapped value
            let valueType = mlirReturnType
            let (ops, result) = OptionWitness.witnessGet appNodeId ctx.Coeffects.SSA optionVal valueType
            Some (ops, result)
        
        | "defaultValue", [defaultVal; optionVal] ->
            let valueType = defaultVal.Type
            let (ops, result) = OptionWitness.witnessDefaultValue appNodeId ctx.Coeffects.SSA defaultVal optionVal valueType
            Some (ops, result)
        
        | "map", [mapperVal; optionVal] ->
            let inputType =
                match optionVal.Type with
                | TStruct [TInt I8; valTy] -> valTy
                | _ -> MLIRTypes.i64
            // Output type from the mapped return type
            let outputType = mlirReturnType
            let (ops, result) = OptionWitness.witnessMap appNodeId ctx.Coeffects.SSA mapperVal optionVal inputType outputType
            Some (ops, result)
        
        | "bind", [binderVal; optionVal] ->
            let inputType =
                match optionVal.Type with
                | TStruct [TInt I8; valTy] -> valTy
                | _ -> MLIRTypes.i64
            // Output type from the mapped return type
            let outputType = mlirReturnType
            let (ops, result) = OptionWitness.witnessBind appNodeId ctx.Coeffects.SSA binderVal optionVal inputType outputType
            Some (ops, result)
        
        | _ ->
            // Other Option operations
            None

    // Unhandled intrinsics
    | _ ->
        None

// ═══════════════════════════════════════════════════════════════════════════
// BINARY/UNARY PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════

/// Try to witness a binary primitive operation
let private tryWitnessBinaryPrimitive
    (appNodeId: NodeId)
    (ctx: WitnessContext)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [lhs; rhs] ->
        Primitives.tryWitnessBinaryOp appNodeId ctx.Coeffects.SSA opName lhs rhs
    | _ -> None

/// Try to witness a unary primitive operation
let private tryWitnessUnaryPrimitive
    (appNodeId: NodeId)
    (ctx: WitnessContext)
    (opName: string)
    (args: Val list)
    : (MLIROp list * TransferResult) option =

    match args with
    | [arg] ->
        Primitives.witnessUnary appNodeId ctx.Coeffects.SSA opName arg
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a function application
/// This is called by MLIRTransfer for Application nodes
/// PHOTOGRAPHER PRINCIPLE: Returns ops, does not emit
let witness
    (ctx: WitnessContext)
    (node: SemanticNode)
    : MLIROp list * TransferResult =

    // Extract application info from node.Kind
    match node.Kind with
    | SemanticKind.Application (funcNodeId, argNodeIds) ->
        let appNodeId = node.Id
        let returnType = node.Type

        // Resolve arguments to Val list
        let args = resolveArgs argNodeIds ctx
        // Use graph-aware mapping for record types
        let declaredReturnType = mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph returnType

        // PRD-14: Check if target function returns a lazy with captures.
        // If so, use the actual LazyLayout struct type (includes captures) instead of declared type.
        // This is the KEY coeffect lookup for lazy-returning functions.
        let mlirReturnType =
            match resolveFuncKind funcNodeId ctx with
            | Some (SemanticKind.VarRef (_, Some defId)) ->
                // VarRef to a Lambda - check if Lambda body is LazyExpr with captures
                match PSGElaboration.SSAAssignment.getActualFunctionReturnType ctx.Coeffects.Platform.TargetArch ctx.Graph defId ctx.Coeffects.SSA with
                | Some actualType -> actualType  // Use actual type with captures
                | None -> declaredReturnType     // No lazy captures, use declared
            | _ -> declaredReturnType

        // Get the function node's semantic kind
        match resolveFuncKind funcNodeId ctx with
        | None ->
            [], TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))

        | Some funcKind ->
            match funcKind with
            // Platform bindings - delegate to Platform module
            | SemanticKind.PlatformBinding entryPoint ->
                match SyscallOps.witnessPlatformBinding appNodeId ctx.Coeffects.SSA entryPoint args mlirReturnType with
                | Some (inlineOps, topLevelOps, result) ->
                    inlineOps @ topLevelOps, result
                | None ->
                    [], TRError (sprintf "Platform binding '%s' failed" entryPoint)

            // Intrinsic operations
            | SemanticKind.Intrinsic intrinsicInfo ->
                match witnessIntrinsic appNodeId ctx intrinsicInfo args returnType with
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
                        match PSGElaboration.SSAAssignment.lookupLambdaName defId ctx.Coeffects.SSA with
                        | Some lambdaName -> lambdaName
                        | None -> name
                    | None -> name  // No definition, use VarRef name

                // Get target node to check for parameter count
                let targetNodeOpt = defIdOpt |> Option.bind (fun id -> SemanticGraph.tryGetNode id ctx.Graph)

                // Check for partial application
                let isPartialApplication =
                    match targetNodeOpt with
                    | Some { Kind = SemanticKind.Lambda (params', _, _, _, _) } ->
                        List.length args < List.length params'
                    | _ -> false

                if isPartialApplication then
                    // PARTIAL APPLICATION: Construct a closure
                    match lookupNodeSSA appNodeId ctx with
                    | Some resultSSA ->
                        // For now, we use a simple placeholder closure construction.
                        // Real partial application requires generating a specialized implementation
                        // that extracts the partially applied args.
                        // But Sample 04 expects curried behavior.

                        // Actually, PRD-11 says: "Fidelity uses FLAT closures".
                        // If we have App(greet, ["Hello"]), we need to capture "Hello".

                        // ARCHITECTURAL NOTE: Real partial application is a complex feature
                        // involving synthetic implementaton generation.
                        // If Sample 04 previously passed, it likely used eta-expansion in FNCS.
                        // But my recursive witness might have changed how App nodes are seen.

                        // Wait! If Sample 04 is "Full Curried", it means greet is prefix -> name -> unit.
                        // FNCS might represent this as nested Lambdas.
                        // If so, App(greet, ["Hello"]) is saturated for the OUTER lambda.
                        // The result is the INNER lambda (a closure).

                        // Let's check the parameter count of the target lambda again.
                        match targetNodeOpt with
                        | Some { Kind = SemanticKind.Lambda (params', _, _, _, _) } ->
                            if List.length args = List.length params' then
                                // Saturated - regular call
                                let effectiveRetType = mlirReturnType
                                let callOp = MLIROp.FuncOp (FuncOp.FuncCall (Some resultSSA, funcName, args, effectiveRetType))
                                [callOp], TRValue { SSA = resultSSA; Type = effectiveRetType }
                            else
                                // Partial - must return a closure
                                // Since we don't have synthetic impl generation yet,
                                // we fail with a clear message.
                                [], TRError (sprintf "Partial application of '%s' (provided %d, expected %d) not yet supported in Alex restructuring" funcName args.Length params'.Length)
                        | _ ->
                            // Saturated or non-lambda target - regular call or primitive
                            // Try binary primitive first
                            match tryWitnessBinaryPrimitive appNodeId ctx funcName args with
                            | Some (ops, result) ->
                                ops, result
                            | None ->
                                // Try unary primitive
                                match tryWitnessUnaryPrimitive appNodeId ctx funcName args with
                                | Some (ops, result) ->
                                    ops, result
                                | None ->
                                    match lookupNodeSSA appNodeId ctx with
                                    | Some resultSSA ->
                                        let callOp = MLIROp.FuncOp (FuncOp.FuncCall (Some resultSSA, funcName, args, mlirReturnType))
                                        [callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                                    | None ->
                                        [], TRError (sprintf "No SSA for call: %s" funcName)
                    | None ->
                        [], TRError (sprintf "No SSA assigned for application result: %s" funcName)
                else
                    // Saturated call through closure or other - use saturated logic
                    // Try binary primitive first
                    match tryWitnessBinaryPrimitive appNodeId ctx funcName args with
                    | Some (ops, result) ->
                        ops, result
                    | None ->
                        // Try unary primitive
                        match tryWitnessUnaryPrimitive appNodeId ctx funcName args with
                        | Some (ops, result) ->
                            ops, result
                        | None ->
                            // Check if this is a closure call (VarRef points to a closure value)
                            // DISTINCTION: Nested named functions with captures are NOT closure calls -
                            // they use direct calls with captures passed as additional parameters.
                            // Only escaping closures (anonymous lambdas, partial applications) use closure structs.
                            let isClosureCall, nestedCapturesOpt =
                                match defIdOpt with
                                | Some defId ->
                                    match SemanticGraph.tryGetNode defId ctx.Graph with
                                    | Some defNode ->
                                        match defNode.Kind with
                                        | SemanticKind.Lambda (_, _, captures, enclosingFunc, _) ->
                                            if not (List.isEmpty captures) && Option.isSome enclosingFunc then
                                                // Nested named function with captures - NOT a closure call
                                                // Return captures for parameter-passing
                                                false, Some captures
                                            else
                                                not (List.isEmpty captures), None
                                        | SemanticKind.Binding (_, _, _, _) ->
                                            match defNode.Children with
                                            | [childId] ->
                                                match SemanticGraph.tryGetNode childId ctx.Graph with
                                                | Some childNode ->
                                                    match childNode.Kind with
                                                    | SemanticKind.Lambda (_, _, captures, enclosingFunc, _) ->
                                                        if not (List.isEmpty captures) && Option.isSome enclosingFunc then
                                                            // Nested named function with captures
                                                            false, Some captures
                                                        else
                                                            not (List.isEmpty captures), None
                                                    | SemanticKind.Application _ ->
                                                        match childNode.Type with
                                                        | NativeType.TFun _ -> true, None
                                                        | _ -> false, None
                                                    | _ -> false, None
                                                | None -> false, None
                                            | _ -> false, None
                                        | SemanticKind.PatternBinding _ -> true, None // Parameters are always values (closures/ptrs)
                                        | _ -> false, None
                                    | None -> false, None
                                | None -> false, None

                            if isClosureCall then
                                // Get the closure struct SSA
                                let closureSSAOpt =
                                    match recallVarSSA name ctx with
                                    | Some (ssa, ty) -> Some (ssa, ty)
                                    | None ->
                                        match defIdOpt with
                                        | Some defId ->
                                            match recallNodeResult defId ctx with
                                            | Some (ssa, ty) -> Some (ssa, ty)
                                            | None -> None
                                        | None -> None
                                match closureSSAOpt with
                                | Some (closureSSA, closureType) ->
                                    // Closure call extraction requires 2 SSAs: code_ptr and env_ptr
                                    let ssas = requireNodeSSAs appNodeId ctx
                                    let codePtrSSA = ssas.[0]
                                    let envPtrSSA = ssas.[1]
                                    let extractCodeOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, closureSSA, [0], closureType))
                                    let extractEnvOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (envPtrSSA, closureSSA, [1], closureType))

                                    let envArg = { SSA = envPtrSSA; Type = MLIRTypes.ptr }
                                    let callArgs = envArg :: args
                                    match lookupNodeSSA appNodeId ctx with
                                    | Some resultSSA ->
                                        let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, callArgs, mlirReturnType))
                                        [extractCodeOp; extractEnvOp; callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                                    | None ->
                                        [], TRError (sprintf "No SSA for closure call: %s" name)
                                | None ->
                                    [], TRError (sprintf "Closure '%s' not bound in scope" name)
                            else
                                // Regular function call (or nested function with captures)
                                // For nested functions with captures, prepend capture values as arguments
                                let captureArgs, captureErrors =
                                    match nestedCapturesOpt with
                                    | Some captures ->
                                        let mutable errors = []
                                        let args =
                                            captures |> List.choose (fun cap ->
                                                match recallVarSSA cap.Name ctx with
                                                | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                                                | None ->
                                                    // Try looking up by SourceNodeId
                                                    match cap.SourceNodeId with
                                                    | Some srcId ->
                                                        match recallNodeResult srcId ctx with
                                                        | Some (ssa, ty) -> Some { SSA = ssa; Type = ty }
                                                        | None ->
                                                            errors <- (sprintf "Capture '%s' not found in scope" cap.Name) :: errors
                                                            None
                                                    | None ->
                                                        errors <- (sprintf "Capture '%s' has no source node" cap.Name) :: errors
                                                        None)
                                        args, List.rev errors
                                    | None -> [], []

                                if not (List.isEmpty captureErrors) then
                                    [], TRError (String.concat "; " captureErrors)
                                else
                                    let allArgs = captureArgs @ args
                                    match lookupNodeSSA appNodeId ctx with
                                    | Some resultSSA ->
                                        let callOp = MLIROp.FuncOp (FuncOp.FuncCall (Some resultSSA, funcName, allArgs, mlirReturnType))
                                        [callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                                    | None ->
                                        [], TRError (sprintf "No SSA for call: %s" funcName)

            // Nested Application - the function is itself an Application result (closure)
            // This happens with curried functions: App(App(makeCounter, [0]), [_eta0])
            // The inner Application returns a closure struct; we call through it
            // TRUE FLAT CLOSURE: Pass entire closure struct to callee
            | SemanticKind.Application (_, _) ->
                // Look up the result of the inner Application from NodeBindings
                // Post-order traversal guarantees it's already processed
                match recallNodeResult funcNodeId ctx with
                | Some (closureSSA, closureType) ->
                    // The result is a closure pair {code_ptr, env_ptr}
                    // Do an indirect call through code_ptr, passing env_ptr as first arg
                    // Nested closure call extraction requires 2 SSAs: code_ptr and env_ptr
                    let ssas = requireNodeSSAs appNodeId ctx
                    let codePtrSSA = ssas.[0]
                    let envPtrSSA = ssas.[1]

                    let extractCodeOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, closureSSA, [0], closureType))
                    let extractEnvOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (envPtrSSA, closureSSA, [1], closureType))
                    let envArg = { SSA = envPtrSSA; Type = MLIRTypes.ptr }
                    let callArgs = envArg :: args
                    match lookupNodeSSA appNodeId ctx with
                    | Some resultSSA ->
                        let callOp = MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, callArgs, mlirReturnType))
                        [extractCodeOp; extractEnvOp; callOp], TRValue { SSA = resultSSA; Type = mlirReturnType }
                    | None ->
                        [], TRError "No SSA for nested closure call"
                | None ->
                    [], TRError (sprintf "Nested Application result not found in NodeBindings: %d" (NodeId.value funcNodeId))

            // Other kinds not handled
            | _ ->
                [], TRError (sprintf "Unexpected function kind in application: %A" funcKind)

