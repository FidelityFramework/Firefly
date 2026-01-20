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

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
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
// SYS.* INTRINSIC DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Sys.* intrinsic operation
/// Dispatches to platform-specific bindings in SyscallBindings.fs
/// This is a THIN ADAPTER - syscall logic lives in Bindings/SyscallBindings.fs
let witnessSysOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * TransferResult) option =

    // Build entry point as "Sys.{opName}" for dispatch lookup
    let entryPoint = $"Sys.{opName}"

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

    let mlirReturnType = mapNativeTypeForArch z.State.Platform.TargetArch returnType
    witnessPlatformBinding appNodeId z entryPoint args mlirReturnType

