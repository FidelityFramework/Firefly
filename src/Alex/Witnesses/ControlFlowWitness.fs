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

open FSharp.Native.Compiler.PSG.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

module SCF = Alex.Dialects.SCF.Templates
module MutAnalysis = PSGElaboration.MutabilityAnalysis

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

/// Extract tag values from Pattern (for DU patterns)
/// Returns empty list for non-DU patterns (Record, Wildcard, etc.)
let private extractTagsFromPattern (pattern: Pattern) : int list =
    let rec extract p =
        match p with
        | Pattern.Union (_caseName, tagIndex, _payload, _unionType) ->
            [tagIndex]
        | Pattern.Tuple elements ->
            elements |> List.collect extract
        | Pattern.Var _ | Pattern.Wildcard | Pattern.Record _ -> []
        | Pattern.Or (p1, _) -> extract p1  // Both branches have same structure
        | Pattern.And (p1, p2) -> extract p1 @ extract p2
        | Pattern.As (inner, _) -> extract inner
        | _ -> []
    extract pattern

/// Check if pattern requires tag-based comparison (DU patterns)
/// vs guard-only matching (Record, Wildcard patterns)
let private requiresTagComparison (pattern: Pattern) : bool =
    let rec check p =
        match p with
        | Pattern.Union _ -> true
        | Pattern.Tuple elements -> elements |> List.exists check
        | Pattern.Or (p1, p2) -> check p1 || check p2
        | Pattern.And (p1, p2) -> check p1 || check p2
        | Pattern.As (inner, _) -> check inner
        | Pattern.Record _ | Pattern.Wildcard | Pattern.Var _ -> false
        | _ -> false
    check pattern

/// Witness a pattern match expression
/// Derives match strategy from Pattern type (Four Pillars: Pattern IS the classification)
/// - DU patterns: tag extraction and comparison
/// - Record/Wildcard patterns: guards only (pattern always matches structurally)
/// - Guards (when clauses): evaluated and combined with pattern match
/// Uses pre-assigned SSAs for all intermediates
let witnessMatch
    (nodeId: NodeId)
    (z: PSGZipper)
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (cases: (Pattern * SSA option * MLIROp list * SSA option) list)  // (pattern, guardSSA, ops, resultSSA)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs for match
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Determine if we need tag-based comparison based on patterns
    let needsTagComparison = cases |> List.exists (fun (p, _, _, _) -> requiresTagComparison p)

    // Extract tags from scrutinee (only if needed for DU patterns)
    let extractOps, extractedTags =
        if not needsTagComparison then
            // Record/Wildcard patterns - no tag extraction needed
            [], []
        else
            // DU patterns - extract tags based on pattern structure
            let numTags =
                cases
                |> List.tryPick (fun (p, _, _, _) ->
                    let tags = extractTagsFromPattern p
                    if List.isEmpty tags then None else Some (List.length tags))
                |> Option.defaultValue 1

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
                        let elemSSA = nextSSA ()
                        ops <- ops @ [MLIROp.LLVMOp (LLVMOp.ExtractValue (elemSSA, scrutineeSSA, [i], scrutineeType))]
                        let tagType = match fieldType with | TStruct (t::_) -> t | _ -> TInt I8
                        let tagSSA = nextSSA ()
                        ops <- ops @ [MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, elemSSA, [0], fieldType))]
                        tags <- tags @ [(tagSSA, tagType)]
                    ops, tags
                | _ ->
                    let tagSSA = nextSSA ()
                    [MLIROp.LLVMOp (LLVMOp.ExtractValue (tagSSA, scrutineeSSA, [0], scrutineeType))],
                    [(tagSSA, TInt I8)]

    // Generate comparison ops for DU tag matching
    let buildTagComparison (expectedTags: int list) : MLIROp list * SSA =
        if List.isEmpty expectedTags || List.isEmpty extractedTags then
            // No tags - always true (record/wildcard pattern)
            let trueSSA = nextSSA ()
            [MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, TInt I1))], trueSSA
        else
            match List.zip expectedTags extractedTags with
            | [(expected, (actualSSA, tagType))] ->
                let expectedSSA = nextSSA ()
                let cmpSSA = nextSSA ()
                let ops = [
                    MLIROp.ArithOp (ArithOp.ConstI (expectedSSA, int64 expected, tagType))
                    MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, actualSSA, expectedSSA, tagType))
                ]
                ops, cmpSSA
            | pairs ->
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
                let rec andChain (ssas: SSA list) : MLIROp list * SSA =
                    match ssas with
                    | [] -> [], nextSSA ()
                    | [single] -> [], single
                    | first :: second :: rest ->
                        let andSSA = nextSSA ()
                        let andOp = MLIROp.ArithOp (ArithOp.AndI (andSSA, first, second, TInt I1))
                        let restOps, finalSSA = andChain (andSSA :: rest)
                        [andOp] @ restOps, finalSSA
                let andOps, finalSSA = andChain cmpSSAs
                ops @ andOps, finalSSA

    // Helper: Find the index of the op that defines a given SSA
    // Returns the index (0-based) of the op that produces this SSA as its result
    let findDefiningOpIndex (ops: MLIROp list) (ssa: SSA) : int option =
        ops |> List.tryFindIndex (fun op ->
            match op with
            | MLIROp.ArithOp (ArithOp.CmpI (r, _, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.ConstI (r, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.AndI (r, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.OrI (r, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.XOrI (r, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.AddI (r, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.SubI (r, _, _, _)) -> r = ssa
            | MLIROp.ArithOp (ArithOp.MulI (r, _, _, _)) -> r = ssa
            | MLIROp.LLVMOp (LLVMOp.ExtractValue (r, _, _, _)) -> r = ssa
            | MLIROp.LLVMOp (LLVMOp.InsertValue (r, _, _, _, _)) -> r = ssa
            | _ -> false)

    // Helper: Split ops into guardOps (ops needed to compute guard) and bodyOps (the rest)
    // guardOps are emitted BEFORE the if, bodyOps go INSIDE the then block
    let splitOpsAtGuard (ops: MLIROp list) (guardSSA: SSA option) : MLIROp list * MLIROp list =
        match guardSSA with
        | None -> [], ops  // No guard - all ops are body ops
        | Some gSSA ->
            match findDefiningOpIndex ops gSSA with
            | Some idx ->
                // Split: ops[0..idx] are guard ops, ops[idx+1..] are body ops
                let guardOps = ops |> List.take (idx + 1)
                let bodyOps = ops |> List.skip (idx + 1)
                guardOps, bodyOps
            | None ->
                // Guard SSA not found in ops - assume it was computed elsewhere
                [], ops

    // Build nested if-else chain (or direct execution for record/wildcard patterns)
    // Now handles guards (when clauses) for record patterns
    let rec buildIfChain (cases: (Pattern * SSA option * MLIROp list * SSA option) list) : MLIROp list * SSA option =
        match cases with
        | [] ->
            [], None
        | [(_, guardSSA, caseOps, resultSSA)] ->
            // Last case - unconditional execution (wildcard/fallthrough)
            // Even last case might have guard ops that need to be emitted
            let guardOps, bodyOps = splitOpsAtGuard caseOps guardSSA
            match resultSSA, resultType with
            | Some ssa, _ ->
                guardOps @ bodyOps, Some ssa
            | None, Some ty ->
                let zeroSSA = nextSSA ()
                let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                guardOps @ bodyOps @ [zeroOp], Some zeroSSA
            | None, None ->
                guardOps @ bodyOps, None
        | (pattern, guardSSA, caseOps, resultSSA) :: rest ->
            // For record/wildcard patterns without guards, this case always matches
            // For DU patterns, compare tags
            // For patterns WITH guards, must check the guard
            let tags = extractTagsFromPattern pattern
            let hasTagCheck = not (List.isEmpty tags) || requiresTagComparison pattern
            let hasGuard = guardSSA.IsSome

            // Split caseOps: guard ops go BEFORE the if, body ops go INSIDE the then
            let guardOps, bodyOps = splitOpsAtGuard caseOps guardSSA

            // Compute the condition:
            // - No tag, no guard: unconditional (take this case)
            // - Tag only: tag comparison
            // - Guard only: guard value
            // - Both: tag AND guard
            let isUnconditional = not hasTagCheck && not hasGuard

            if isUnconditional then
                // Record/Wildcard without guard: pattern always matches, take this case directly
                match resultSSA, resultType with
                | Some ssa, _ -> caseOps, Some ssa
                | None, Some ty ->
                    let zeroSSA = nextSSA ()
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                    caseOps @ [zeroOp], Some zeroSSA
                | None, None -> caseOps, None
            else
                // Conditional case: compute the combined condition
                // guardOps are emitted BEFORE the if (outside)
                let checkOps, cmpSSA =
                    match hasTagCheck, guardSSA with
                    | false, Some gSSA ->
                        // Guard only (record pattern with when clause)
                        // guardOps already includes the ops that compute gSSA
                        guardOps, gSSA
                    | true, None ->
                        // Tag only (DU pattern without when clause)
                        buildTagComparison tags
                    | true, Some gSSA ->
                        // Both tag and guard: AND them together
                        let tagOps, tagSSA = buildTagComparison tags
                        let andSSA = nextSSA ()
                        let andOp = MLIROp.ArithOp (ArithOp.AndI (andSSA, tagSSA, gSSA, TInt I1))
                        guardOps @ tagOps @ [andOp], andSSA
                    | false, None ->
                        // Should not happen (handled by isUnconditional)
                        let trueSSA = nextSSA ()
                        [MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, TInt I1))], trueSSA

                // Only body ops go inside the then block (guard ops are outside)
                let thenOps =
                    match resultSSA, resultType with
                    | Some ssa, Some ty -> bodyOps @ [MLIROp.SCFOp (SCF.scfYield [{ SSA = ssa; Type = ty }])]
                    | None, Some ty ->
                        let zeroSSA = nextSSA ()
                        let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                        bodyOps @ [zeroOp; MLIROp.SCFOp (SCF.scfYield [{ SSA = zeroSSA; Type = ty }])]
                    | _ -> bodyOps @ [MLIROp.SCFOp (SCF.scfYieldVoid)]
                let thenRegion = SCF.singleBlockRegion "then" [] thenOps

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
                (checkOps @ [MLIROp.SCFOp ifOp]), resultSSAOpt

    let chainOps, finalResultSSA = buildIfChain cases
    let matchOps = extractOps @ chainOps

    match finalResultSSA, resultType with
    | Some ssa, Some ty ->
        matchOps, TRValue { SSA = ssa; Type = ty }
    | _ ->
        matchOps, TRVoid

