/// BindingTypes - Platform binding types for witness-based MLIR generation
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// The fold accumulates what bindings return via withOps.
module Alex.Bindings.BindingTypes

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes

// ===================================================================
// Binding Strategy
// ===================================================================

type BindingStrategy =
    | Static
    | Dynamic

// ===================================================================
// Binding Result - What bindings RETURN
// ===================================================================

/// Result of a binding: operations to emit and optional result value
type BindingResult =
    | BoundOps of ops: MLIROp list * result: Val option
    | NotSupported of reason: string

// ===================================================================
// Platform Primitive Types
// ===================================================================

/// Platform primitive call (uses typed Val instead of string tuples)
type PlatformPrimitive = {
    EntryPoint: string
    Library: string
    CallingConvention: string
    Args: Val list
    ReturnType: MLIRType
    BindingStrategy: BindingStrategy
}

/// External function declaration
type ExternalDeclaration = {
    Name: string
    Signature: string
    Library: string option
}

// ===================================================================
// Binding Signature - RETURNS ops, does not emit
// ===================================================================

/// A binding takes nodeId (for pre-assigned SSAs), zipper, and primitive, RETURNS ops
/// SSAs are pre-allocated during SSAAssignment pass (coeffects pattern)
type Binding = NodeId -> PSGZipper -> PlatformPrimitive -> BindingResult

// ===================================================================
// Platform Dispatch Registry
// ===================================================================

module PlatformDispatch =
    let mutable private bindings: Map<(OSFamily * Architecture * string), Binding> = Map.empty
    let mutable private currentPlatform: TargetPlatform option = None

    let register (os: OSFamily) (arch: Architecture) (entryPoint: string) (binding: Binding) =
        let key = (os, arch, entryPoint)
        bindings <- Map.add key binding bindings

    let registerForOS (os: OSFamily) (entryPoint: string) (binding: Binding) =
        register os X86_64 entryPoint binding
        register os ARM64 entryPoint binding

    let setTargetPlatform (platform: TargetPlatform) =
        currentPlatform <- Some platform

    let getTargetPlatform () =
        currentPlatform |> Option.defaultValue (TargetPlatform.detectHost())

    /// Dispatch: Returns BindingResult (ops + result or NotSupported)
    /// nodeId is used to get pre-assigned SSAs from coeffects
    let dispatch (nodeId: NodeId) (zipper: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, prim.EntryPoint)
        match Map.tryFind key bindings with
        | Some binding -> binding nodeId zipper prim
        | None ->
            let fallbackKey = (platform.OS, X86_64, prim.EntryPoint)
            match Map.tryFind fallbackKey bindings with
            | Some binding -> binding nodeId zipper prim
            | None -> NotSupported $"No binding for {prim.EntryPoint} on {platform.OS}/{platform.Arch}"

    let hasBinding (entryPoint: string) : bool =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, entryPoint)
        Map.containsKey key bindings ||
        Map.containsKey (platform.OS, X86_64, entryPoint) bindings

    let clear () =
        bindings <- Map.empty
        currentPlatform <- None

    let getRegisteredEntryPoints () : string list =
        let platform = getTargetPlatform()
        bindings
        |> Map.toList
        |> List.filter (fun ((os, arch, _), _) -> os = platform.OS && (arch = platform.Arch || arch = X86_64))
        |> List.map (fun ((_, _, ep), _) -> ep)
        |> List.distinct
