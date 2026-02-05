/// ApplicationPatterns - Function application and invocation patterns
///
/// PUBLIC: Witnesses use these to emit function calls (direct and indirect).
/// Application patterns handle calling conventions and argument passing.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.ApplicationPatterns

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements  // pFuncCall, pFuncCallIndirect
open Alex.Elements.ArithElements // Arithmetic elements for wrapper patterns

// ═══════════════════════════════════════════════════════════
// APPLICATION PATTERNS (Function Calls)
// ═══════════════════════════════════════════════════════════

/// Build function application (indirect call via function pointer)
/// For known function names, use pDirectCall instead (future optimization)
/// SSA extracted from coeffects via nodeId: [0] = result
let pApplicationCall (nodeId: NodeId) (funcSSA: SSA) (args: (SSA * MLIRType) list) (retType: MLIRType)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pApplicationCall: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Emit indirect call via function pointer
        let argVals = args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let! callOp = pFuncCallIndirect (Some resultSSA) funcSSA argVals retType
        return ([callOp], TRValue { SSA = resultSSA; Type = retType })
    }

/// Build direct function call (for known function names - portable)
/// Uses func.call (portable) instead of llvm.call (backend-specific)
/// SSA extracted from coeffects via nodeId: [0] = result
let pDirectCall (nodeId: NodeId) (funcName: string) (args: (SSA * MLIRType) list) (retType: MLIRType)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pDirectCall: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Convert (SSA * MLIRType) list to Val list for pFuncCall
        let vals = args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let! callOp = pFuncCall (Some resultSSA) funcName vals retType
        return ([callOp], TRValue { SSA = resultSSA; Type = retType })
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC WRAPPER PATTERNS
// ═══════════════════════════════════════════════════════════
// These patterns wrap arithmetic Elements to maintain Element/Pattern/Witness firewall.
// Witnesses call patterns (not Elements directly), patterns extract SSAs monadically.

/// Generic binary arithmetic pattern wrapper
/// Takes operation name, internally dispatches to correct Element (maintains firewall)
/// SSA extracted from coeffects via nodeId: [0] = result
let pBinaryArithOp (nodeId: NodeId) (operation: string)
                   (lhsSSA: SSA) (rhsSSA: SSA) (resultType: MLIRType)
                   : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pBinaryArithOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Dispatch to correct Element based on operation name
        let! op =
            match operation with
            // Integer arithmetic
            | "addi" -> pAddI resultSSA lhsSSA rhsSSA
            | "subi" -> pSubI resultSSA lhsSSA rhsSSA
            | "muli" -> pMulI resultSSA lhsSSA rhsSSA
            | "divsi" -> pDivSI resultSSA lhsSSA rhsSSA
            | "divui" -> pDivUI resultSSA lhsSSA rhsSSA
            | "remsi" -> pRemSI resultSSA lhsSSA rhsSSA
            | "remui" -> pRemUI resultSSA lhsSSA rhsSSA
            // Float arithmetic
            | "addf" -> pAddF resultSSA lhsSSA rhsSSA
            | "subf" -> pSubF resultSSA lhsSSA rhsSSA
            | "mulf" -> pMulF resultSSA lhsSSA rhsSSA
            | "divf" -> pDivF resultSSA lhsSSA rhsSSA
            // Bitwise
            | "andi" -> pAndI resultSSA lhsSSA rhsSSA
            | "ori" -> pOrI resultSSA lhsSSA rhsSSA
            | "xori" -> pXorI resultSSA lhsSSA rhsSSA
            | "shli" -> pShLI resultSSA lhsSSA rhsSSA
            | "shrui" -> pShRUI resultSSA lhsSSA rhsSSA
            | "shrsi" -> pShRSI resultSSA lhsSSA rhsSSA
            | _ -> fail (Message $"Unknown binary arithmetic operation: {operation}")

        return ([op], TRValue { SSA = resultSSA; Type = resultType })
    }

/// Generic comparison pattern wrapper
/// SSA extracted from coeffects via nodeId: [0] = result
let pComparisonOp (nodeId: NodeId)
                  (pred: ICmpPred)
                  (lhsSSA: SSA) (rhsSSA: SSA) (operandType: MLIRType)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pComparisonOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let! op = pCmpI resultSSA pred lhsSSA rhsSSA operandType
        return ([op], TRValue { SSA = resultSSA; Type = TInt I1 })  // Comparisons always return i1
    }

/// Unary NOT pattern (xori with constant 1)
/// SSA extracted from coeffects via nodeId: [0] = constant, [1] = result
let pUnaryNot (nodeId: NodeId) (operandSSA: SSA) (resultType: MLIRType)
              : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pUnaryNot: Expected 2 SSAs, got {ssas.Length}"
        let constSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // Emit constant 1 for boolean NOT
        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, 1L, TInt I1))
        // Emit XOR operation
        let! xorOp = pXorI resultSSA operandSSA constSSA
        return ([constOp; xorOp], TRValue { SSA = resultSSA; Type = resultType })
    }
