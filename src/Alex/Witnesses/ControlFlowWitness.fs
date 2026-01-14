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
    // NOTE: SCF regions use anonymous entry blocks - block args are implicitly defined by the operation
    let condTerminator = MLIROp.SCFOp (SCF.scfCondition condResultSSA iterArgs)
    let condRegion = SCF.singleBlockRegion "" [] (condOps @ [condTerminator])

    // Build body region with scf.yield terminator
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "" [] (bodyOps @ [bodyTerminator])

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
    // NOTE: SCF regions use anonymous entry blocks - block args are implicitly defined by the operation
    let bodyTerminator = MLIROp.SCFOp (SCF.scfYield iterArgs)
    let bodyRegion = SCF.singleBlockRegion "" [] (bodyOps @ [bodyTerminator])

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
/// Generically handles single DU and tuple patterns by extracting tags from scrutinee
/// and comparing against expected tag values from each case's Pattern
/// Uses pre-assigned SSAs for all intermediates
let witnessMatch
    (nodeId: NodeId)
    (z: PSGZipper)
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (cases: (int list * MLIROp list * SSA option) list)  // (tags, ops, resultSSA)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs for match
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Determine number of tags to check based on first case's pattern
    let numTags = match cases with | (tags, _, _) :: _ -> List.length tags | [] -> 1

    // Extract tags from scrutinee - handles both single DU and tuple of DUs
    let extractOps, extractedTags =
        if numTags <= 1 then
            // Single DU pattern - extract tag from field 0
            let tagType = match scrutineeType with | TStruct (t::_) -> t | _ -> TInt I8
            let tagSSA = nextSSA ()
            let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, scrutineeSSA, [0], scrutineeType))
            [extractOp], [(tagSSA, tagType)]
        else
            // Tuple pattern - extract each element then its tag
            match scrutineeType with
            | TStruct fields ->
                let mutable ops = []
                let mutable tags = []
                for i, fieldType in List.indexed fields do
                    // Extract tuple element
                    let elemSSA = nextSSA ()
                    ops <- ops @ [MLIROp.LLVMOp (LLVMOp.ExtractValue (elemSSA, scrutineeSSA, [i], scrutineeType))]
                    // Extract tag from DU element
                    let tagType = match fieldType with | TStruct (t::_) -> t | _ -> TInt I8
                    let tagSSA = nextSSA ()
                    ops <- ops @ [MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, elemSSA, [0], fieldType))]
                    tags <- tags @ [(tagSSA, tagType)]
                ops, tags
            | _ ->
                // Fallback - single tag
                let tagSSA = nextSSA ()
                [MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, scrutineeSSA, [0], scrutineeType))],
                [(tagSSA, TInt I8)]

    // Generate comparison ops for a list of (expected tag, extracted SSA, type) triples
    // Returns the ops and the final combined condition SSA
    let buildTagComparison (expectedTags: int list) : MLIROp list * SSA =
        match List.zip expectedTags extractedTags with
        | [] ->
            // No tags - always true
            let trueSSA = nextSSA ()
            [MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, TInt I1))], trueSSA
        | [(expected, (actualSSA, tagType))] ->
            // Single tag - simple comparison
            let expectedSSA = nextSSA ()
            let cmpSSA = nextSSA ()
            let ops = [
                MLIROp.ArithOp (ArithOp.ConstI (expectedSSA, int64 expected, tagType))
                MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, actualSSA, expectedSSA, tagType))
            ]
            ops, cmpSSA
        | pairs ->
            // Multiple tags - compare each and AND together
            let mutable ops = []
            let mutable cmpSSAs = []
            for (expected, (actualSSA, tagType)) in pairs do
                let expectedSSA = nextSSA ()
                let cmpSSA = nextSSA ()
                ops <- ops @ [
                    MLIROp.ArithOp (ArithOp.ConstI (expectedSSA, int64 expected, tagType))
                    MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, actualSSA, expectedSSA, tagType))
                ]
                cmpSSAs <- cmpSSAs @ [cmpSSA]
            // AND all comparisons together
            let rec andChain (ssas: SSA list) : MLIROp list * SSA =
                match ssas with
                | [] -> [], nextSSA ()  // Shouldn't happen
                | [single] -> [], single
                | first :: second :: rest ->
                    let andSSA = nextSSA ()
                    let andOp = MLIROp.ArithOp (ArithOp.AndI (andSSA, first, second, TInt I1))
                    let restOps, finalSSA = andChain (andSSA :: rest)
                    [andOp] @ restOps, finalSSA
            let andOps, finalSSA = andChain cmpSSAs
            ops @ andOps, finalSSA

    // Build nested if-else chain
    // Returns (ops, resultSSA) - the ops do NOT include a final yield; caller adds that
    let rec buildIfChain (cases: (int list * MLIROp list * SSA option) list) : MLIROp list * SSA option =
        match cases with
        | [] ->
            // Should not happen - match should be exhaustive
            [], None
        | [(_, caseOps, resultSSA)] ->
            // Last case - no condition check, just return case ops
            // Caller will add the yield
            match resultSSA, resultType with
            | Some ssa, _ ->
                caseOps, Some ssa
            | None, Some ty ->
                // Need to create a zero constant for void case
                let zeroSSA = nextSSA ()
                let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                caseOps @ [zeroOp], Some zeroSSA
            | None, None ->
                caseOps, None
        | (tags, caseOps, resultSSA) :: rest ->
            // Build comparison for ALL tags in the pattern
            let checkOps, cmpSSA = buildTagComparison tags

            // Then branch: this case - add yield at end
            let thenOps =
                match resultSSA, resultType with
                | Some ssa, Some ty -> caseOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
                | None, Some ty ->
                    let zeroSSA = nextSSA ()
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                    caseOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
                | _ -> caseOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
            let thenRegion = SCF.singleBlockRegion "then" [] thenOps

            // Else branch: remaining cases - add yield at end
            let elseOps, elseResultSSA = buildIfChain rest
            let elseOpsWithYield =
                match elseResultSSA, resultType with
                | Some ssa, Some ty ->
                    elseOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
                | _ ->
                    elseOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
            let elseRegion = SCF.singleBlockRegion "else" [] elseOpsWithYield

            let resultSSAOpt = resultType |> Option.map (fun _ -> nextSSA ())
            let resultSSAs = resultSSAOpt |> Option.toList
            let resultTypes = resultType |> Option.toList

            let ifOp = SCFOp.If (resultSSAs, cmpSSA, thenRegion, Some elseRegion, resultTypes)

            // Return the if op and its result - no yield here, caller handles it
            (checkOps @ [MLIROp.SCFOp ifOp]), resultSSAOpt

    let chainOps, finalResultSSA = buildIfChain cases
    let matchOps = extractOps @ chainOps

    match finalResultSSA, resultType with
    | Some ssa, Some ty ->
        matchOps, TRValue { SSA = ssa; Type = ty }
    | _ ->
        matchOps, TRVoid

