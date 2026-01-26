/// Loop Witness - Witness loop control flow to MLIR (SCF dialect)
///
/// SCOPE: Handle while loops and for loops.
/// DOES NOT: Implement conditionals, pattern matching (separate witnesses).
///
/// Uses SCF dialect for structured loop constructs.
module Alex.Witnesses.ControlFlow.LoopWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module SCF = Alex.Dialects.SCF.Templates
module SSAAssignment = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Get all pre-assigned SSAs for a node
let private requireNodeSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssignment.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// WHILE LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a while loop using SCF dialect
/// condOps: operations to compute the condition
/// condResultSSA: the boolean result of the condition
/// bodyOps: operations in the loop body
/// iterArgs: loop-carried values
let witnessWhileLoop
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (condOps: MLIROp list)
    (condResultSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build condition region with scf.condition terminator
    let condTerminator = MLIROp.SCFOp (SCF.scfCondition condResultSSA iterArgs)
    let condRegion = SCF.singleBlockRegion "" [] (condOps @ [condTerminator])

    // Build body region with scf.yield terminator
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "" [] (bodyOps @ [bodyTerminator])

    // Result SSAs (one per iter arg) - use pre-assigned SSAs when iterArgs is non-empty
    let resultSSAs =
        if List.isEmpty iterArgs then []
        else
            let ssas = requireNodeSSAs nodeId ctx
            iterArgs |> List.mapi (fun i _ -> ssas.[i])

    let whileOp = SCFOp.While (resultSSAs, condRegion, bodyRegion, iterArgs)

    // While loops are typically void in F# semantics
    [MLIROp.SCFOp whileOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// FOR LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a for loop using SCF dialect
/// start, stop, step: loop bounds
/// bodyOps: operations for the loop body
/// iterArgs: additional iteration arguments beyond the induction variable
/// Note: ivSSA and stepSSA come from ForLoop node's pre-assigned SSAs
let witnessForLoop
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (ivSSA: SSA)
    (startSSA: SSA)
    (stopSSA: SSA)
    (stepSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build body region with scf.yield terminator
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "" [] (bodyOps @ [bodyTerminator])

    // Result SSAs - use pre-assigned SSAs when iterArgs is non-empty
    // Note: indices 0,1 are used for ivSSA,stepSSA, so iterArgs start at index 2
    let resultSSAs =
        if List.isEmpty iterArgs then []
        else
            let ssas = requireNodeSSAs nodeId ctx
            iterArgs |> List.mapi (fun i _ -> ssas.[2 + i])

    let forOp = SCFOp.For (resultSSAs, ivSSA, startSSA, stopSSA, stepSSA, bodyRegion, iterArgs)

    [MLIROp.SCFOp forOp], TRVoid
