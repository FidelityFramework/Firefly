/// PtrIntrinsics - Witness NativePtr intrinsic operations
///
/// SCOPE: Handle ONLY NativePtr intrinsics from FNCS.
/// DOES NOT: Implement transform logic, fill FNCS gaps, route to other witnesses.
///
/// NativePtr operations provide low-level pointer manipulation:
/// - read/write: direct load/store
/// - get/set: indexed load/store (GEP + load/store)
/// - add: pointer arithmetic (GEP)
/// - stackalloc: stack allocation (alloca)
/// - toNativeInt/ofNativeInt: pointer/integer conversion
/// - toVoidPtr/ofVoidPtr: type-level conversions (no-op at MLIR)
/// - copy: memcpy wrapper
/// - fill: memset wrapper
module Alex.Witnesses.PtrIntrinsics

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.Arith.Templates
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get pre-assigned SSAs for a node
let private requireSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssign.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get result SSA for a node
let private requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No result SSA for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness NativePtr intrinsic operations
let witness
    (ctx: WitnessContext)
    (node: SemanticNode)
    (args: Val list)
    (returnType: NativeType)
    : MLIROp list * TransferResult =

    let appNodeId = node.Id
    let resultType = mapType returnType ctx
    let resultSSA = requireSSA appNodeId ctx

    // Extract intrinsic info
    match node.Kind with
    | SemanticKind.Application (funcNodeId, _) ->
        match SemanticGraph.tryGetNode funcNodeId ctx.Graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic intrinsicInfo ->
                match intrinsicInfo with
                | NativePtrOp opKind ->
                    match opKind, args with
                    | PtrRead, [ptr] ->
                        // NativePtr.read: nativeptr<'T> -> 'T (direct load)
                        let op = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, ptr.SSA, resultType, NotAtomic))
                        [op], TRValue { SSA = resultSSA; Type = resultType }

                    | PtrWrite, [ptr; value] ->
                        // NativePtr.write: nativeptr<'T> -> 'T -> unit (direct store)
                        let op = MLIROp.LLVMOp (LLVMOp.Store (value.SSA, ptr.SSA, value.Type, NotAtomic))
                        [op], TRVoid

                    | PtrGet, [ptr; idx] ->
                        // NativePtr.get: ptr -> int -> 'T (indexed load)
                        // Needs 2 SSAs: gep intermediate + load result
                        let ssas = requireSSAs appNodeId ctx
                        let gepSSA = ssas.[0]
                        let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptr.SSA, [(idx.SSA, idx.Type)], resultType))
                        let loadOp = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, gepSSA, resultType, NotAtomic))
                        [gepOp; loadOp], TRValue { SSA = resultSSA; Type = resultType }

                    | PtrSet, [ptr; idx; value] ->
                        // NativePtr.set: ptr -> int -> 'T -> unit (indexed store)
                        // Needs 1 SSA for gep intermediate
                        let ssas = requireSSAs appNodeId ctx
                        let gepSSA = ssas.[0]
                        let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptr.SSA, [(idx.SSA, idx.Type)], value.Type))
                        let storeOp = MLIROp.LLVMOp (LLVMOp.Store (value.SSA, gepSSA, value.Type, NotAtomic))
                        [gepOp; storeOp], TRVoid

                    | PtrAdd, [ptr; offset] ->
                        // NativePtr.add: ptr -> int -> ptr
                        let op = MLIROp.LLVMOp (LLVMOp.GEP (resultSSA, ptr.SSA, [(offset.SSA, offset.Type)], TInt I8))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr }

                    | PtrStackAlloc, [count] ->
                        // NativePtr.stackalloc: int -> ptr
                        // Alloca count must be i64
                        if count.Type = MLIRTypes.i64 then
                            let op = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, count.SSA, resultType, None))
                            [op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr }
                        else
                            // Extend count to i64 - needs 2 SSAs: extsi + alloca
                            let ssas = requireSSAs appNodeId ctx
                            let extSSA = ssas.[0]
                            let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, count.SSA, count.Type, MLIRTypes.i64))
                            let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (resultSSA, extSSA, resultType, None))
                            [extOp; allocOp], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr }

                    | PtrToNativeInt, [ptr] ->
                        // NativePtr.toNativeInt: ptr -> nativeint
                        let op = MLIROp.LLVMOp (LLVMOp.PtrToInt (resultSSA, ptr.SSA, MLIRTypes.i64))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 }

                    | PtrOfNativeInt, [intVal] ->
                        // NativePtr.ofNativeInt: nativeint -> ptr
                        // Source type is intVal.Type (i64 for nativeint), NOT ptr
                        let op = MLIROp.LLVMOp (LLVMOp.IntToPtr (resultSSA, intVal.SSA, intVal.Type))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.ptr }

                    | PtrToVoidPtr, [ptr] ->
                        // NativePtr.toVoidPtr: nativeptr<'T> -> voidptr (no-op)
                        [], TRValue { SSA = ptr.SSA; Type = MLIRTypes.ptr }

                    | PtrOfVoidPtr, [ptr] ->
                        // NativePtr.ofVoidPtr: voidptr -> nativeptr<'T> (no-op)
                        [], TRValue { SSA = ptr.SSA; Type = MLIRTypes.ptr }

                    | PtrCopy, [src; dst; count] ->
                        // NativePtr.copy: src -> dst -> count -> unit
                        let op = intrMemcpy dst src count false
                        [op], TRVoid

                    | PtrFill, [ptr; value; count] ->
                        // NativePtr.fill: ptr -> byte -> count -> unit
                        let op = intrMemset ptr value count false
                        [op], TRVoid

                    | _ ->
                        [], TRError (sprintf "Unhandled NativePtr operation with %d args" args.Length)

                | _ ->
                    [], TRError "Not a NativePtr intrinsic"
            | _ ->
                [], TRError "Function node is not an intrinsic"
        | None ->
            [], TRError "Function node not found"
    | _ ->
        [], TRError "Not an Application node"
