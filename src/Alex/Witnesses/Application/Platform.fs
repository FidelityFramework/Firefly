/// Application/Platform - Witness platform operations to MLIR via bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This is a THIN ADAPTER layer. Platform-specific logic lives in Bindings/*.
/// This module simply:
/// 1. Builds PlatformPrimitive from witness context
/// 2. Calls PlatformDispatch.dispatch (which returns structured MLIROp)
/// 3. Converts BindingResult to (MLIROp list * TransferResult)
///
/// NO sprintf. NO platform-specific logic. Just dispatch coordination.
module Alex.Witnesses.Application.Platform

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Bindings.BindingTypes
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.Checking.Native.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// RESULT CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert BindingResult to witness return format
let private bindingResultToTransfer (result: BindingResult) : (MLIROp list * TransferResult) option =
    match result with
    | BoundOps (ops, Some resultVal) ->
        Some (ops, TRValue resultVal)
    | BoundOps (ops, None) ->
        Some (ops, TRVoid)
    | NotSupported _ ->
        None

/// Convert BindingResult with error propagation
let private bindingResultToTransferWithError (result: BindingResult) : MLIROp list * TransferResult =
    match result with
    | BoundOps (ops, Some resultVal) ->
        ops, TRValue resultVal
    | BoundOps (ops, None) ->
        ops, TRVoid
    | NotSupported reason ->
        [], TRError reason

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM BINDING DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a platform binding operation
/// Entry point for SemanticKind.PlatformBinding nodes
/// Uses pre-assigned SSAs from Application node
let witnessPlatformBinding
    (appNodeId: NodeId)
    (z: PSGZipper)
    (entryPoint: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * TransferResult) option =

    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = returnType
        BindingStrategy = Static
    }

    let result = PlatformDispatch.dispatch appNodeId z prim
    bindingResultToTransfer result

/// Witness a platform binding, returning error on failure
/// Uses pre-assigned SSAs from Application node
let witnessPlatformBindingRequired
    (appNodeId: NodeId)
    (z: PSGZipper)
    (entryPoint: string)
    (args: Val list)
    (returnType: MLIRType)
    : MLIROp list * TransferResult =

    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = returnType
        BindingStrategy = Static
    }

    let result = PlatformDispatch.dispatch appNodeId z prim
    bindingResultToTransferWithError result

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC DISPATCH HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Sys.* intrinsic operation
/// These are TRUE PRIMITIVES - directly generate inline syscalls
/// Requires 5 pre-assigned SSAs: syscallNum[0], resultSSA[1], truncResult[2], fdExt[3], lenExt[4]
let witnessSysOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * TransferResult) option =

    // Get pre-assigned SSAs for syscall operations
    let ssas = requireNodeSSAs appNodeId z

    match opName, args with
    // Sys.write: fd:int -> ptr:nativeptr<byte> -> len:int -> int
    | "write", [fdVal; ptrVal; lenVal] ->
        // Linux x86_64: syscall 1 = write(fd, buf, count)
        // SSAs: syscallNum[0], resultSSA[1], truncResult[2], fdExt[3], lenExt[4]
        let syscallNum = ssas.[0]
        let resultSSA = ssas.[1]
        let truncResult = ssas.[2]
        
        // Build fd extension ops (if needed)
        let fdOps, fdFinal =
            if fdVal.Type = MLIRTypes.i64 then
                [], fdVal.SSA
            else
                let fdExt = ssas.[3]
                [MLIROp.ArithOp (ArithOp.ExtSI (fdExt, fdVal.SSA, fdVal.Type, MLIRTypes.i64))], fdExt
        
        // Build len extension ops (if needed)
        let lenOps, lenFinal =
            if lenVal.Type = MLIRTypes.i64 then
                [], lenVal.SSA
            else
                let lenExt = ssas.[4]
                [MLIROp.ArithOp (ArithOp.ExtSI (lenExt, lenVal.SSA, lenVal.Type, MLIRTypes.i64))], lenExt
        
        let ops =
            [MLIROp.ArithOp (ArithOp.ConstI (syscallNum, 1L, MLIRTypes.i64))]
            @ fdOps
            @ lenOps
            @ [
                MLIROp.LLVMOp (LLVMOp.InlineAsm (
                    Some resultSSA,
                    "syscall",
                    "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}",
                    [(syscallNum, MLIRTypes.i64)
                     (fdFinal, MLIRTypes.i64)
                     (ptrVal.SSA, MLIRTypes.ptr)
                     (lenFinal, MLIRTypes.i64)],
                    Some MLIRTypes.i64,
                    true,
                    false))
                MLIROp.ArithOp (ArithOp.TruncI (truncResult, resultSSA, MLIRTypes.i64, MLIRTypes.i32))
            ]
        Some (ops, TRValue { SSA = truncResult; Type = MLIRTypes.i32 })

    // Sys.read: fd:int -> ptr:nativeptr<byte> -> len:int -> int
    | "read", [fdVal; ptrVal; lenVal] ->
        // Linux x86_64: syscall 0 = read(fd, buf, count)
        // Uses same SSA indices: syscallNum[0], resultSSA[1], truncResult[2], fdExt[3], lenExt[4]
        let syscallNum = ssas.[0]
        let resultSSA = ssas.[1]
        let truncResult = ssas.[2]
        let fdExt = ssas.[3]
        let lenExt = ssas.[4]
        
        let ops = [
            // Syscall number 0 for read
            MLIROp.ArithOp (ArithOp.ConstI (syscallNum, 0L, MLIRTypes.i64))
            // Extend fd to i64 (use actual type)
            MLIROp.ArithOp (ArithOp.ExtSI (fdExt, fdVal.SSA, fdVal.Type, MLIRTypes.i64))
            // Extend len to i64 (use actual type)
            MLIROp.ArithOp (ArithOp.ExtSI (lenExt, lenVal.SSA, lenVal.Type, MLIRTypes.i64))
            // Inline syscall
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                Some resultSSA,
                "syscall",
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}",
                [(syscallNum, MLIRTypes.i64)
                 (fdExt, MLIRTypes.i64)
                 (ptrVal.SSA, MLIRTypes.ptr)
                 (lenExt, MLIRTypes.i64)],
                Some MLIRTypes.i64,
                true,
                false))
            // Truncate result to i32
            MLIROp.ArithOp (ArithOp.TruncI (truncResult, resultSSA, MLIRTypes.i64, MLIRTypes.i32))
        ]
        Some (ops, TRValue { SSA = truncResult; Type = MLIRTypes.i32 })

    | _ -> None

// NOTE: witnessConsoleOp removed - Console is NOT an intrinsic
// It's Layer 3 user code in Fidelity.Platform that uses Sys.* intrinsics.
// See fsnative-spec/spec/platform-bindings.md

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING (uses TypeMapping.mapNativeType)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness with NativeType conversion
/// Uses TypeMapping.mapNativeType for authoritative type mapping
/// Uses pre-assigned SSAs from Application node
let witnessPlatformBindingNative
    (appNodeId: NodeId)
    (z: PSGZipper)
    (entryPoint: string)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeType returnType
    witnessPlatformBinding appNodeId z entryPoint args mlirReturnType

