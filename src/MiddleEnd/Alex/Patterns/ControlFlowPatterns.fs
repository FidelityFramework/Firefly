/// ControlFlowPatterns - Structured control flow constructions
///
/// PUBLIC: Witnesses use these to emit control flow operations (If, While, For, Switch).
/// All control flow constructions compose SCFElements and CFElements.
module Alex.Patterns.ControlFlowPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.SCFElements  // pSCFIf, pSCFWhile, pSCFFor
open Alex.Elements.CFElements   // pSwitch
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// SEQ FOREACH LOOP
// ═══════════════════════════════════════════════════════════

/// ForEach loop over seq (MoveNext-based iteration)
/// MoveNext should: extract code_ptr[2], alloca seq, store seq, call code_ptr(seq_ptr) -> i1
let pBuildForEachLoop (collectionSSA: SSA) (bodyOps: MLIROp list)
                      (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        // ForEach is a while loop structure:
        // 1. Extract code_ptr from seq struct at [2]
        // 2. Alloca space for seq, store seq to get pointer
        // 3. Call code_ptr(seq_ptr) -> i1 (returns true if has next)
        // 4. If true, extract current element, execute body, loop
        // 5. If false, exit loop

        // Implementation: Use SCF.While with:
        //   - Condition region: call MoveNext, extract result
        //   - Body region: extract current, execute bodyOps, yield
        return! fail (Message "ForEach MoveNext implementation gap - needs: extract code_ptr[2], alloca/store seq, indirect call")
    }

// ═══════════════════════════════════════════════════════════
// STRUCTURED CONTROL FLOW (SCF)
// ═══════════════════════════════════════════════════════════

/// If/then/else via SCF.If
let pBuildIfThenElse (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list option) : PSGParser<MLIROp list> =
    parser {
        let! ifOp = pSCFIf cond thenOps elseOps
        return [ifOp]
    }

/// While loop via SCF.While
let pBuildWhileLoop (condOps: MLIROp list) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! whileOp = pSCFWhile condOps bodyOps
        return [whileOp]
    }

/// For loop via SCF.For
let pBuildForLoop (lower: SSA) (upper: SSA) (step: SSA) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! forOp = pSCFFor lower upper step bodyOps
        return [forOp]
    }

// ═══════════════════════════════════════════════════════════
// CONTROL FLOW (CF)
// ═══════════════════════════════════════════════════════════

/// Switch statement via CF.Switch
let pSwitch (flag: SSA) (flagTy: MLIRType) (defaultOps: MLIROp list)
            (cases: (int64 * MLIROp list) list) : PSGParser<MLIROp list> =
    parser {
        // Convert case ops to block refs (would need actual block construction)
        // For now, placeholder structure
        let defaultBlock = BlockRef "default"
        let caseBlocks = cases |> List.map (fun (value, _) ->
            (value, BlockRef $"case_{value}", []))

        let! switchOp = Alex.Elements.CFElements.pSwitch flag flagTy defaultBlock [] caseBlocks
        return [switchOp]
    }
