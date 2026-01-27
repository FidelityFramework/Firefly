/// Transfer Types - Core types for MLIR Transfer
///
/// CANONICAL ARCHITECTURE (January 2026):
/// This file defines the types that witnesses receive. It compiles BEFORE
/// witnesses so they can elegantly take `ctx: WitnessContext` rather than
/// explicit parameter threading.
///
/// The Three Concerns:
/// - PSGZipper: Pure navigation (Focus, Path, Graph) - defined in PSGZipper.fs
/// - TransferCoeffects: Pre-computed, immutable coeffects
/// - MLIRAccumulator: Mutable fold state
///
/// See: mlir_transfer_canonical_architecture memory
module Alex.Traversal.TransferTypes

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES (for type definitions)
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER COEFFECTS (Pre-computed, Immutable)
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed coeffects - computed ONCE before traversal, NEVER modified
type TransferCoeffects = {
    SSA: SSAAssign.SSAAssignment
    Platform: PlatformResolutionResult
    Mutability: MutAnalysis.MutabilityAnalysisResult
    PatternBindings: PatternAnalysis.PatternBindingAnalysisResult
    Strings: StringCollect.StringTable
    YieldStates: YieldStateIndices.YieldStateCoeffect
    EntryPointLambdaIds: Set<int>
}

// ═══════════════════════════════════════════════════════════════════════════
// MLIR ACCUMULATOR (Mutable Fold State)
// ═══════════════════════════════════════════════════════════════════════════

/// Scope for nested regions
type AccumulatorScope = {
    VarAssoc: Map<string, SSA * MLIRType>
    NodeAssoc: Map<int, SSA * MLIRType>
    CapturedVars: Set<string>
    CapturedMuts: Set<string>
}

/// Mutable accumulator - the `acc` in the fold
type MLIRAccumulator = {
    mutable TopLevelOps: MLIROp list
    mutable Errors: string list
    mutable Visited: Set<int>
    mutable ScopeStack: AccumulatorScope list
    mutable CurrentScope: AccumulatorScope
}

module MLIRAccumulator =
    let empty () : MLIRAccumulator =
        let globalScope = {
            VarAssoc = Map.empty
            NodeAssoc = Map.empty
            CapturedVars = Set.empty
            CapturedMuts = Set.empty
        }
        {
            TopLevelOps = []
            Errors = []
            Visited = Set.empty
            ScopeStack = []
            CurrentScope = globalScope
        }

    let addTopLevelOp (op: MLIROp) (acc: MLIRAccumulator) =
        acc.TopLevelOps <- op :: acc.TopLevelOps

    let addTopLevelOps (ops: MLIROp list) (acc: MLIRAccumulator) =
        acc.TopLevelOps <- List.rev ops @ acc.TopLevelOps

    let addError (err: string) (acc: MLIRAccumulator) =
        acc.Errors <- err :: acc.Errors

    let markVisited (nodeIdVal: int) (acc: MLIRAccumulator) =
        acc.Visited <- Set.add nodeIdVal acc.Visited

    let isVisited (nodeIdVal: int) (acc: MLIRAccumulator) =
        Set.contains nodeIdVal acc.Visited

    let bindVar (name: string) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.CurrentScope <- { acc.CurrentScope with VarAssoc = Map.add name (ssa, ty) acc.CurrentScope.VarAssoc }

    let recallVar (name: string) (acc: MLIRAccumulator) =
        Map.tryFind name acc.CurrentScope.VarAssoc

    let bindNode (nodeIdVal: int) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.CurrentScope <- { acc.CurrentScope with NodeAssoc = Map.add nodeIdVal (ssa, ty) acc.CurrentScope.NodeAssoc }

    let recallNode (nodeIdVal: int) (acc: MLIRAccumulator) =
        Map.tryFind nodeIdVal acc.CurrentScope.NodeAssoc

    let isCapturedVariable (name: string) (acc: MLIRAccumulator) =
        Set.contains name acc.CurrentScope.CapturedVars

    let isCapturedMutable (name: string) (acc: MLIRAccumulator) =
        Set.contains name acc.CurrentScope.CapturedMuts

    let pushScope (scope: AccumulatorScope) (acc: MLIRAccumulator) =
        acc.ScopeStack <- acc.CurrentScope :: acc.ScopeStack
        acc.CurrentScope <- scope

    let popScope (acc: MLIRAccumulator) =
        match acc.ScopeStack with
        | prev :: rest ->
            acc.CurrentScope <- prev
            acc.ScopeStack <- rest
        | [] -> ()

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER RESULT (Result of witnessing a node)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of witnessing a PSG node
type TransferResult =
    | TRValue of Val                    // Produces a value (SSA + type)
    | TRVoid                             // Produces no value (effect only)
    | TRError of string                  // Error (gap in coverage)

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS OUTPUT (What witnesses return)
// ═══════════════════════════════════════════════════════════════════════════

/// Codata returned by witnesses
type WitnessOutput = {
    InlineOps: MLIROp list
    TopLevelOps: MLIROp list
    Result: TransferResult
}

module WitnessOutput =
    let empty = { InlineOps = []; TopLevelOps = []; Result = TRVoid }
    let inline' ops result = { InlineOps = ops; TopLevelOps = []; Result = result }
    let value v = { InlineOps = []; TopLevelOps = []; Result = TRValue v }
    let error msg = { InlineOps = []; TopLevelOps = []; Result = TRError msg }
    let withTopLevel topOps output = { output with TopLevelOps = topOps @ output.TopLevelOps }
    let combine (a: WitnessOutput) (b: WitnessOutput) =
        { InlineOps = a.InlineOps @ b.InlineOps
          TopLevelOps = a.TopLevelOps @ b.TopLevelOps
          Result = b.Result }
    let combineAll outputs = outputs |> List.fold combine empty

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS CONTEXT (What witnesses receive)
// ═══════════════════════════════════════════════════════════════════════════

/// Context passed to witnesses - the elegant single parameter
type WitnessContext = {
    Coeffects: TransferCoeffects
    Accumulator: MLIRAccumulator
    Graph: SemanticGraph
}

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Convenience functions)
// ═══════════════════════════════════════════════════════════════════════════

/// Get single pre-assigned SSA for a node
let requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get all pre-assigned SSAs for a node
let requireSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssign.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get target architecture from coeffects
let targetArch (ctx: WitnessContext) : Architecture =
    ctx.Coeffects.Platform.TargetArch

/// CANONICAL TYPE MAPPING - the ONLY way to map NativeType to MLIRType
/// Uses graph-aware mapping that correctly handles records by looking up
/// field types from TypeDef nodes. ALL type mapping should go through this.
let mapType (ty: NativeType) (ctx: WitnessContext) : MLIRType =
    mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph ty

/// Get platform-aware word width for string length, array length, etc.
let wordWidth (ctx: WitnessContext) : IntBitWidth =
    platformWordWidth ctx.Coeffects.Platform.TargetArch
