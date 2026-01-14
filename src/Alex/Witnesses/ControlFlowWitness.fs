/// Control Flow Witness - Witness control flow constructs to MLIR (SCF dialect)
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Witnesses OBSERVE and RETURN structured MLIROp lists.
/// They do NOT emit strings. ZERO SPRINTF for MLIR generation.
/// All SSAs come from pre-computed SSAAssignment coeffect.
///
/// Uses:
/// - requireNodeSSA/requireNodeSSAs for pre-assigned SSAs (from coeffects)
/// - SCFTemplates for structured SCF operations
/// - ArithTemplates for arithmetic operations
/// - LLVMTemplates for LLVM operations
module Alex.Witnesses.ControlFlowWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

module SCF = Alex.Dialects.SCF.Templates
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Map NativeType to MLIRType (delegated to TypeMapping)
let private mapType = mapNativeType

/// Check if a type is unit/void
let private isUnitType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> true
    | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENTIAL EXPRESSION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a sequential expression (children already processed in post-order)
/// Returns the result of the last expression.
/// NOTE: Sequential does NOT produce its own SSA - it passes through the last child's result.
/// We use recallNodeResult (from NodeBindings) because that's the actual emitted SSA.
let witnessSequential (z: PSGZipper) (nodeIds: NodeId list) : MLIROp list * TransferResult =
    match List.tryLast nodeIds with
    | Some lastId ->
        // Use recallNodeResult to get the ACTUAL emitted SSA (from NodeBindings),
        // not lookupNodeSSA which gets the pre-assigned SSA.
        match recallNodeResult (NodeId.value lastId) z with
        | Some (ssa, ty) ->
            // Pass through the last child's result
            [], TRValue { SSA = ssa; Type = ty }
        | None ->
            // No result bound - might be void or not yet processed
            [], TRVoid
    | None ->
        [], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// IF-THEN-ELSE
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an if-then-else expression using SCF dialect
/// thenOps and elseOps are the pre-witnessed operations from the regions
/// Uses pre-assigned SSAs: result[0], thenZero[1], elseZero[2]
let witnessIfThenElse
    (nodeId: NodeId)
    (z: PSGZipper)
    (condSSA: SSA)
    (thenOps: MLIROp list)
    (thenResultSSA: SSA option)
    (elseOps: MLIROp list option)
    (elseResultSSA: SSA option)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs for IfThenElse: result[0], thenZero[1], elseZero[2]
    let ssas = requireNodeSSAs nodeId z

    // Build then region with yield
    let thenYieldOps =
        match thenResultSSA, resultType with
        | Some ssa, Some ty -> thenOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
        | None, Some ty ->
            // Body is void/unit but if-then-else expects result (e.g. i32 for unit)
            let zeroSSA = ssas.[1]  // thenZero
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
            thenOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
        | _ -> thenOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]

    let thenRegion = SCF.singleBlockRegion "then" [] thenYieldOps

    // Build else region if present
    let elseRegionOpt =
        match elseOps with
        | Some ops ->
            let elseYieldOps =
                match elseResultSSA, resultType with
                | Some ssa, Some ty -> ops @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
                | None, Some ty ->
                    let zeroSSA = ssas.[2]  // elseZero
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                    ops @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
                | _ -> ops @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | None -> None

    // Build SCF if operation
    match resultType with
    | Some ty ->
        let resultSSA = ssas.[0]  // result
        let ifOp = SCFOp.If ([resultSSA], condSSA, thenRegion, elseRegionOpt, [ty])
        [MLIROp.SCFOp ifOp], TRValue { SSA = resultSSA; Type = ty }
    | None ->
        let ifOp = SCFOp.If ([], condSSA, thenRegion, elseRegionOpt, [])
        [MLIROp.SCFOp ifOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// WHILE LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a while loop using SCF dialect
/// condOps: operations for the guard condition (should produce a boolean)
/// bodyOps: operations for the loop body
/// iterArgs: iteration arguments (mutable variables carried through loop)
/// Note: When iterArgs is implemented via coeffects, SSAs for result will be pre-assigned
let witnessWhileLoop
    (nodeId: NodeId)
    (z: PSGZipper)
    (condOps: MLIROp list)
    (condResultSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build condition region with scf.condition terminator
    let condBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let condTerminator = MLIROp.SCFOp (SCF.scfCondition condResultSSA iterArgs)
    let condRegion = SCF.singleBlockRegion "before" condBlockArgs (condOps @ [condTerminator])

    // Build body region with scf.yield terminator
    let bodyBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "do" bodyBlockArgs (bodyOps @ [bodyTerminator])

    // Result SSAs (one per iter arg) - use pre-assigned SSAs when iterArgs is non-empty
    let resultSSAs =
        if List.isEmpty iterArgs then []
        else
            let ssas = requireNodeSSAs nodeId z
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
    (z: PSGZipper)
    (ivSSA: SSA)
    (startSSA: SSA)
    (stopSSA: SSA)
    (stepSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build body region
    // Block args: induction variable + iter args
    let ivArg = SCF.blockArg ivSSA MLIRTypes.index
    let iterBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let allBlockArgs = ivArg :: iterBlockArgs

    // Body with yield
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "body" allBlockArgs (bodyOps @ [bodyTerminator])

    // Result SSAs - use pre-assigned SSAs when iterArgs is non-empty
    // Note: indices 0,1 are used for ivSSA,stepSSA, so iterArgs start at index 2
    let resultSSAs =
        if List.isEmpty iterArgs then []
        else
            let ssas = requireNodeSSAs nodeId z
            iterArgs |> List.mapi (fun i _ -> ssas.[2 + i])

    let forOp = SCFOp.For (resultSSAs, ivSSA, startSSA, stopSSA, stepSSA, bodyRegion, iterArgs)

    [MLIROp.SCFOp forOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN MATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a pattern match expression
/// Lowered to a chain of scf.if operations
/// Each case becomes an if-then-else checking the discriminator
/// Uses pre-assigned SSAs: discrim[0], then tag/cmp/zero/result for each case
let witnessMatch
    (nodeId: NodeId)
    (z: PSGZipper)
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (cases: (int * MLIROp list * SSA option) list)  // (tag, ops, resultSSA)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs for match
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // For DU matching, extract discriminator (first field of struct)
    let discrimSSA = nextSSA ()
    let extractDiscrim = MLIROp.LLVMOp (LLVMOp.ExtractValue (discrimSSA, scrutineeSSA, [0], scrutineeType))

    // Build nested if-else chain
    let rec buildIfChain (cases: (int * MLIROp list * SSA option) list) : MLIROp list * SSA option =
        match cases with
        | [] ->
            // Should not happen - match should be exhaustive
            [], None
        | [(_, caseOps, resultSSA)] ->
            // Last case - no condition check needed
            match resultSSA, resultType with
            | Some ssa, Some ty -> 
                caseOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])], Some ssa
            | None, Some ty ->
                let zeroSSA = nextSSA ()
                let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                caseOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])], Some zeroSSA
            | _ -> 
                caseOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)], None
        | (tag, caseOps, resultSSA) :: rest ->
            // Check if discriminator matches this tag
            let tagSSA = nextSSA ()
            let cmpSSA = nextSSA ()

            let checkOps = [
                MLIROp.ArithOp (ArithOp.ConstI (tagSSA, int64 tag, MLIRTypes.i32))
                MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, discrimSSA, tagSSA, MLIRTypes.i32))
            ]

            // Then branch: this case
            let thenOps =
                match resultSSA, resultType with
                | Some ssa, Some ty -> caseOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
                | None, Some ty ->
                    let zeroSSA = nextSSA ()
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                    caseOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
                | _ -> caseOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
            let thenRegion = SCF.singleBlockRegion "then" [] thenOps

            // Else branch: remaining cases
            let elseOps, _elseResultSSA = buildIfChain rest
            let elseRegion = SCF.singleBlockRegion "else" [] elseOps

            let resultSSAOpt = resultType |> Option.map (fun _ -> nextSSA ())
            let resultSSAs = resultSSAOpt |> Option.toList
            let resultTypes = resultType |> Option.toList

            let ifOp = SCFOp.If (resultSSAs, cmpSSA, thenRegion, Some elseRegion, resultTypes)

            (checkOps @ [MLIROp.SCFOp ifOp]), resultSSAOpt

    let chainOps, finalResultSSA = buildIfChain cases
    let matchOps = [extractDiscrim] @ chainOps

    match finalResultSSA, resultType with
    | Some ssa, Some ty ->
        matchOps, TRValue { SSA = ssa; Type = ty }
    | _ ->
        matchOps, TRVoid

