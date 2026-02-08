/// BindingWitness - Witness binding nodes (let bindings)
///
/// Bindings don't emit MLIR - they forward the bound value's SSA.
/// The binding name is tracked in the accumulator for VarRef lookup.
///
/// FUNCTION BINDINGS: Bindings whose child is a Lambda (function) are SKIPPED.
/// LambdaWitness handles generating FuncDefs for module-level functions.
///
/// NANOPASS: This witness handles ONLY Binding nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.BindingWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness binding nodes - forwards bound value's SSA (immutable) or emits memref.alloca (mutable)
let private witnessBinding (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBinding ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, isMut, _isRec, isEntry), _) ->
        // Binding has one child - the value being bound
        match node.Children with
        | [valueId] ->
            // Check for underscore discard pattern
            if name = "_" then
                // Discard pattern - value is witnessed for side effects, but no SSA binding created
                // Post-order ensures child is already witnessed
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
            else
                // Check if the child is a Lambda (function binding)
                match SemanticGraph.tryGetNode valueId ctx.Graph with
                | Some valueNode when valueNode.Kind.ToString().StartsWith("Lambda") ->
                    // Function binding - Lambda child already generated FuncDef (post-order)
                    // Binding node is structural (name association), return TRVoid per Domain Responsibility Principle
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | _ ->
                    // Check if this binding holds a partial application (curry flattening)
                    if Set.contains node.Id ctx.Coeffects.CurryFlattening.PartialAppBindings then
                        // Partial application binding - no MLIR emitted
                        // Saturated call sites use the coeffect to emit direct calls
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    // Check if binding is mutable
                    elif isMut then
                        // MUTABLE BINDING: Allocate memref and initialize
                        match MLIRAccumulator.recallNode valueId ctx.Accumulator with
                        | Some (initSSA, elemType) ->
                            let (NodeId nodeIdInt) = node.Id
                            match tryMatch (pBuildMutableBinding nodeIdInt elemType initSSA)
                                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((ops, result), _) ->
                                { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None ->
                                WitnessOutput.error $"Mutable binding '{name}': Pattern emission failed"
                        | None ->
                            WitnessOutput.error $"Mutable binding '{name}': Initial value not yet witnessed"
                    else
                        // IMMUTABLE BINDING: Forward child's SSA (existing logic)
                        match MLIRAccumulator.recallNode valueId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // Forward the value's SSA - binding doesn't emit ops
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                        | None ->
                            // Check if the child was witnessed but returned TRVoid
                            // This happens for entry point Lambdas (function definitions) and other module-level declarations
                            if isEntry then
                                // Entry point binding - child is a function definition, not a value
                                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                            else
                                WitnessOutput.error $"Binding '{name}': Value not yet witnessed (non-entry binding without SSA)"
        | _ ->
            WitnessOutput.error $"Binding '{name}': Expected 1 child, got {node.Children.Length}"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Binding nanopass - witnesses let bindings
let nanopass : Nanopass = {
    Name = "Binding"
    Witness = witnessBinding
}
