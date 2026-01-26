/// Match Witness - Witness pattern matching to MLIR (SCF dialect)
///
/// SCOPE: Handle pattern match expressions with DU/Record/Wildcard patterns and guards.
/// DOES NOT: Implement loops, conditionals (separate witnesses).
///
/// Pattern matching strategies:
/// - DU patterns: Tag extraction and comparison
/// - Record/Wildcard patterns: Guards only (pattern always matches structurally)
/// - Guards (when clauses): Evaluated and combined with pattern match
module Alex.Witnesses.ControlFlow.MatchWitness

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
// PATTERN HELPERS
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

/// Find the index of the op that defines a given SSA
let private findDefiningOpIndex (ops: MLIROp list) (ssa: SSA) : int option =
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

/// Split ops into guardOps (needed to compute guard) and bodyOps (the rest)
let private splitOpsAtGuard (ops: MLIROp list) (guardSSA: SSA option) : MLIROp list * MLIROp list =
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

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN MATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a pattern match expression
/// Derives match strategy from Pattern type (Four Pillars: Pattern IS the classification)
let witnessMatch
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (cases: (Pattern * SSA option * MLIROp list * SSA option) list)  // (pattern, guardSSA, ops, resultSSA)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Get pre-assigned SSAs for match
    let ssas = requireNodeSSAs nodeId ctx
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

    // Build nested if-else chain (handles guards for record patterns)
    let rec buildIfChain (cases: (Pattern * SSA option * MLIROp list * SSA option) list) : MLIROp list * SSA option =
        match cases with
        | [] ->
            [], None
        | [(_, guardSSA, caseOps, resultSSA)] ->
            // Last case - unconditional execution
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
            let tags = extractTagsFromPattern pattern
            let hasTagCheck = not (List.isEmpty tags) || requiresTagComparison pattern
            let hasGuard = guardSSA.IsSome

            // Split caseOps: guard ops go BEFORE the if, body ops go INSIDE the then
            let guardOps, bodyOps = splitOpsAtGuard caseOps guardSSA

            // Compute the condition
            let isUnconditional = not hasTagCheck && not hasGuard

            if isUnconditional then
                // Record/Wildcard without guard: pattern always matches
                match resultSSA, resultType with
                | Some ssa, _ -> caseOps, Some ssa
                | None, Some ty ->
                    let zeroSSA = nextSSA ()
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
                    caseOps @ [zeroOp], Some zeroSSA
                | None, None -> caseOps, None
            else
                // Conditional case: compute the combined condition
                let checkOps, cmpSSA =
                    match hasTagCheck, guardSSA with
                    | false, Some gSSA ->
                        // Guard only (record pattern with when clause)
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

                // Only body ops go inside the then block
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
