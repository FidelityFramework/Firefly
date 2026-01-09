/// Coeffects - Read-only context from nanopasses
///
/// ARCHITECTURAL PRINCIPLE:
/// Coeffects are pre-computed by nanopasses BEFORE witnessing.
/// Witnesses LOOK UP coeffects - they never compute or mutate them.
/// "FNCSTransfer is not allowed to ask 'What if?' It is only allowed to ask 'What is?'"
module Alex.Traversal.Coeffects

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Preprocessing.SSAAssignment
open Alex.Preprocessing.MutabilityAnalysis
open Alex.Preprocessing.PlatformConfig
open Alex.Bindings.PlatformTypes

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECTS RECORD
// ═══════════════════════════════════════════════════════════════════════════

/// All pre-computed analysis results that witnesses can look up.
/// This is the "mise-en-place" - everything is prepared before witnessing begins.
type Coeffects = {
    /// SSA assignments for all PSG nodes (from SSAAssignment nanopass)
    SSA: SSAAssignment

    /// Mutability analysis (which bindings need alloca)
    Mutability: MutabilityAnalysisResult

    /// Platform binding resolutions (syscall vs libc, _start wrapper)
    Platform: PlatformResolutionResult
}

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT LOOKUPS
// These are the "What is?" questions witnesses can ask.
// ═══════════════════════════════════════════════════════════════════════════

/// Look up the result SSA for a node
let lookupSSA (nodeId: NodeId) (c: Coeffects) : SSA option =
    Alex.Preprocessing.SSAAssignment.lookupSSA nodeId c.SSA

/// Look up all SSAs for a node (for multi-SSA expansion)
let lookupSSAs (nodeId: NodeId) (c: Coeffects) : SSA list option =
    Alex.Preprocessing.SSAAssignment.lookupSSAs nodeId c.SSA

/// Look up the function name for a Lambda
let lookupLambdaName (nodeId: NodeId) (c: Coeffects) : string option =
    Alex.Preprocessing.SSAAssignment.lookupLambdaName nodeId c.SSA

/// Check if a Lambda is an entry point
let isEntryPoint (nodeId: NodeId) (c: Coeffects) : bool =
    Alex.Preprocessing.SSAAssignment.isEntryPoint nodeId c.SSA

/// Check if a binding needs alloca (addressed mutable)
let isAddressedMutable (nodeId: int) (c: Coeffects) : bool =
    Set.contains nodeId c.Mutability.AddressedMutableBindings

/// Check if a variable is modified in any loop body
let isModifiedInLoop (varName: string) (c: Coeffects) : bool =
    c.Mutability.ModifiedVarsInLoopBodies
    |> Map.exists (fun _ names -> List.contains varName names)

/// Check if a binding needs alloca (either addressed or loop-modified)
let needsAlloca (nodeId: int) (varName: string) (c: Coeffects) : bool =
    isAddressedMutable nodeId c || isModifiedInLoop varName c

/// Look up a platform binding resolution
let lookupPlatformBinding (nodeId: int) (c: Coeffects) : BindingResolution option =
    Alex.Preprocessing.PlatformConfig.lookupBinding nodeId c.Platform

/// Check if we're in freestanding mode
let isFreestanding (c: Coeffects) : bool =
    c.Platform.RuntimeMode = Freestanding

/// Get the runtime mode
let getRuntimeMode (c: Coeffects) : RuntimeMode =
    c.Platform.RuntimeMode

/// Get target OS
let getTargetOS (c: Coeffects) : OSFamily =
    c.Platform.TargetOS

/// Get target architecture
let getTargetArch (c: Coeffects) : Architecture =
    c.Platform.TargetArch

/// Get _start wrapper ops if needed (freestanding mode)
let getStartWrapperOps (c: Coeffects) : MLIROp list option =
    c.Platform.StartWrapperOps

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING
// "What IS the MLIR type for this NativeType on this platform?"
// ═══════════════════════════════════════════════════════════════════════════

/// Get platform word size in bits
let private platformWordBits (arch: Architecture) : int =
    match arch with
    | X86_64 | ARM64 | RISCV64 -> 64
    | ARM32_Thumb | RISCV32 | WASM32 -> 32

/// Map NTUKind to MLIRType given platform architecture
let private mapNTUKind (kind: NTUKind) (arch: Architecture) : MLIRType =
    let wordBits = platformWordBits arch
    match kind with
    // Platform-dependent integer types
    | NTUKind.NTUint | NTUKind.NTUuint | NTUKind.NTUnint | NTUKind.NTUunint ->
        if wordBits = 64 then MLIRTypes.i64 else MLIRTypes.i32
    | NTUKind.NTUsize | NTUKind.NTUdiff ->
        if wordBits = 64 then MLIRTypes.i64 else MLIRTypes.i32
    | NTUKind.NTUptr | NTUKind.NTUfnptr ->
        MLIRTypes.ptr

    // Fixed-width signed integers
    | NTUKind.NTUint8 -> MLIRTypes.i8
    | NTUKind.NTUint16 -> MLIRTypes.i16
    | NTUKind.NTUint32 -> MLIRTypes.i32
    | NTUKind.NTUint64 -> MLIRTypes.i64

    // Fixed-width unsigned integers (same LLVM types)
    | NTUKind.NTUuint8 -> MLIRTypes.i8
    | NTUKind.NTUuint16 -> MLIRTypes.i16
    | NTUKind.NTUuint32 -> MLIRTypes.i32
    | NTUKind.NTUuint64 -> MLIRTypes.i64

    // Floating point
    | NTUKind.NTUfloat32 -> MLIRTypes.f32
    | NTUKind.NTUfloat64 -> MLIRTypes.f64

    // Special types
    | NTUKind.NTUbool -> MLIRTypes.i1
    | NTUKind.NTUchar -> MLIRTypes.i32  // UTF-32 codepoint
    | NTUKind.NTUunit -> MLIRTypes.i32  // Represented as i32 (could be void in some contexts)
    | NTUKind.NTUstring -> TStruct [MLIRTypes.ptr; MLIRTypes.i64]  // Fat pointer: ptr + length

    // Compound value types (128-bit represented as struct of two i64)
    | NTUKind.NTUdecimal -> TStruct [MLIRTypes.i64; MLIRTypes.i64]
    | NTUKind.NTUuuid -> TStruct [MLIRTypes.i64; MLIRTypes.i64]
    | NTUKind.NTUdatetime -> MLIRTypes.i64  // Ticks since epoch
    | NTUKind.NTUtimespan -> MLIRTypes.i64  // Duration in ticks

    // Other/user-defined - default to pointer-sized
    | NTUKind.NTUother -> if wordBits = 64 then MLIRTypes.i64 else MLIRTypes.i32

/// Map a NativeType to MLIRType using platform context from coeffects
let rec mapType (ty: NativeType) (c: Coeffects) : MLIRType =
    let arch = c.Platform.TargetArch
    match ty with
    | NativeType.TApp (tycon, _args) ->
        match tycon.NTUKind with
        | Some kind -> mapNTUKind kind arch
        | None ->
            // User-defined type - check layout
            match tycon.Layout with
            | TypeLayout.PlatformWord -> if platformWordBits arch = 64 then MLIRTypes.i64 else MLIRTypes.i32
            | TypeLayout.Inline (size, _align) when size > 0 ->
                // Inline struct - approximate by size
                match size with
                | 1 -> MLIRTypes.i8
                | 2 -> MLIRTypes.i16
                | 4 -> MLIRTypes.i32
                | 8 -> MLIRTypes.i64
                | 16 -> TStruct [MLIRTypes.i64; MLIRTypes.i64]  // 128-bit as two i64
                | _ -> TStruct []  // Generic struct
            | _ -> MLIRTypes.ptr  // Reference type

    | NativeType.TNativePtr _ -> MLIRTypes.ptr
    | NativeType.TByref _ -> MLIRTypes.ptr
    | NativeType.TFun _ -> MLIRTypes.ptr  // Function pointer

    | NativeType.TTuple (elements, isStruct) ->
        let elemTypes = elements |> List.map (fun e -> mapType e c)
        if isStruct then TStruct elemTypes else MLIRTypes.ptr

    | NativeType.TRecord (tycon, _fields) ->
        match tycon.NTUKind with
        | Some kind -> mapNTUKind kind arch
        | None -> MLIRTypes.ptr  // Reference to record

    | NativeType.TUnion (tycon, _cases) ->
        match tycon.NTUKind with
        | Some kind -> mapNTUKind kind arch
        | None -> MLIRTypes.ptr  // Reference to union

    | NativeType.TVar _ -> MLIRTypes.ptr  // Type variable - assume reference
    | NativeType.TForall (_, body) -> mapType body c
    | NativeType.TAnon (_, isStruct) -> if isStruct then TStruct [] else MLIRTypes.ptr
    | NativeType.TMeasure _ -> MLIRTypes.i32  // Phantom - zero-sized but represented
    | NativeType.TError _ -> MLIRTypes.i64  // Error fallback

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECTS CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Create coeffects from nanopass results
let create
    (ssa: SSAAssignment)
    (mutability: MutabilityAnalysisResult)
    (platform: PlatformResolutionResult)
    : Coeffects =
    {
        SSA = ssa
        Mutability = mutability
        Platform = platform
    }
