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
let witnessPlatformBinding
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

    let result = PlatformDispatch.dispatch z prim
    bindingResultToTransfer result

/// Witness a platform binding, returning error on failure
let witnessPlatformBindingRequired
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

    let result = PlatformDispatch.dispatch z prim
    bindingResultToTransferWithError result

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC DISPATCH HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Sys.* intrinsic operation
/// Maps FNCS SysOp intrinsics to platform bindings
let witnessSysOp
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * TransferResult) option =

    // Sys intrinsics map directly to platform entry points
    // e.g., "write" -> "Sys.write", "read" -> "Sys.read"
    let entryPoint = sprintf "Sys.%s" opName
    witnessPlatformBinding z entryPoint args returnType

/// Witness a Console.* intrinsic operation
/// Maps FNCS ConsoleOp intrinsics to platform bindings
let witnessConsoleOp
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * TransferResult) option =

    // Console intrinsics map to Console.* entry points
    let entryPoint = sprintf "Console.%s" opName
    witnessPlatformBinding z entryPoint args returnType

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING (uses TypeMapping.mapNativeType)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness with NativeType conversion
/// Uses TypeMapping.mapNativeType for authoritative type mapping
let witnessPlatformBindingNative
    (z: PSGZipper)
    (entryPoint: string)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeType returnType
    witnessPlatformBinding z entryPoint args mlirReturnType

