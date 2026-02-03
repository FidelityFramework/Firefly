/// ArithmeticPatterns - High-level arithmetic dispatchers
///
/// PUBLIC: Witnesses use these to emit arithmetic operations.
/// These patterns dispatch to appropriate Element operations based on atomic operation classification.
module Alex.Patterns.ArithmeticPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.ArithElements  // pAddI, pSubI, pMulI, pDivSI, pDivUI, pRemSI, pRemUI, pAddF, pSubF, pMulF, pDivF, pCmpI
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types  // IntrinsicInfo

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

/// Create parser failure with error message
let pfail msg : PSGParser<'a> = fail (Message msg)

// ═══════════════════════════════════════════════════════════
// ATOMIC OPERATION CLASSIFICATION (STUB - Needs nanopass implementation)
// ═══════════════════════════════════════════════════════════

/// Atomic operation category classification
/// TODO: This should be provided by a nanopass that analyzes intrinsic operations
type AtomicOpCategory =
    | BinaryArith of string  // addi, subi, muli, divsi, divui, etc.
    | Comparison of string   // slt, sle, sgt, sge, eq, ne, etc.
    | Bitwise of string      // and, or, xor, shl, shr
    | Unary of string        // neg, not

/// Pattern to extract classified atomic operation from current node
/// STUB: This parser accesses the AtomicOperations coeffect to retrieve the classification
/// Full implementation requires AtomicOperations nanopass
let pClassifiedAtomicOp : PSGParser<IntrinsicInfo * AtomicOpCategory> =
    parser {
        let! state = getUserState
        // Access AtomicOperations coeffect (computed in nanopass)
        // For now, return error - full implementation requires AtomicOperations coeffect
        return! pfail "pClassifiedAtomicOp: Requires AtomicOperations coeffect (not yet implemented)"
    }

// ═══════════════════════════════════════════════════════════
// BINARY ARITHMETIC DISPATCHERS
// ═══════════════════════════════════════════════════════════

/// Binary arithmetic operations (+, -, *, /, %)
/// Takes: result SSA, LHS SSA, RHS SSA, architecture
/// Matches atomic operation classification and emits appropriate operation
let pBuildBinaryArith (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, category) = pClassifiedAtomicOp

        let! op =
            parser {
                match category with
                | BinaryArith "addi" -> return! pAddI resultSSA lhsSSA rhsSSA
                | BinaryArith "subi" -> return! pSubI resultSSA lhsSSA rhsSSA
                | BinaryArith "muli" -> return! pMulI resultSSA lhsSSA rhsSSA
                | BinaryArith "divsi" -> return! pDivSI resultSSA lhsSSA rhsSSA
                | BinaryArith "divui" -> return! pDivUI resultSSA lhsSSA rhsSSA
                | BinaryArith "remsi" -> return! pRemSI resultSSA lhsSSA rhsSSA
                | BinaryArith "remui" -> return! pRemUI resultSSA lhsSSA rhsSSA
                | BinaryArith "addf" -> return! pAddF resultSSA lhsSSA rhsSSA
                | BinaryArith "subf" -> return! pSubF resultSSA lhsSSA rhsSSA
                | BinaryArith "mulf" -> return! pMulF resultSSA lhsSSA rhsSSA
                | BinaryArith "divf" -> return! pDivF resultSSA lhsSSA rhsSSA
                | _ -> return! pfail $"Unsupported binary arithmetic atomic operation: {info.FullName}"
            }

        let! state = getUserState
        let ty = mapNativeTypeForArch arch state.Current.Type

        return ([op], TRValue { SSA = resultSSA; Type = ty })
    }

// ═══════════════════════════════════════════════════════════
// COMPARISON DISPATCHERS
// ═══════════════════════════════════════════════════════════

/// Comparison operations (<, <=, >, >=, ==, !=)
/// Takes: result SSA, LHS SSA, RHS SSA, architecture
/// Emits CmpI or CmpF based on operand types
let pBuildComparison (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, category) = pClassifiedAtomicOp

        let! predicate =
            parser {
                match category with
                | Comparison "slt" -> return ICmpPred.Slt
                | Comparison "sle" -> return ICmpPred.Sle
                | Comparison "sgt" -> return ICmpPred.Sgt
                | Comparison "sge" -> return ICmpPred.Sge
                | Comparison "eq" -> return ICmpPred.Eq
                | Comparison "ne" -> return ICmpPred.Ne
                | Comparison "ult" -> return ICmpPred.Ult
                | Comparison "ule" -> return ICmpPred.Ule
                | Comparison "ugt" -> return ICmpPred.Ugt
                | Comparison "uge" -> return ICmpPred.Uge
                | _ -> return! pfail $"Unsupported comparison atomic operation: {info.FullName}"
            }

        let! cmpOp = pCmpI resultSSA predicate lhsSSA rhsSSA

        // Comparison always returns i1 (boolean)
        let resultType = mapNTUKindToMLIRType arch NTUKind.NTUbool

        return ([cmpOp], TRValue { SSA = resultSSA; Type = resultType })
    }

// ═══════════════════════════════════════════════════════════
// BITWISE AND UNARY (STUBS - Need FNCS elaboration)
// ═══════════════════════════════════════════════════════════

/// Bitwise operations (&, |, ^, <<, >>)
/// Note: These are NOT currently in FNCS as atomic operations - need to add them
/// For now, return error indicating FNCS elaboration needed
let pBuildBitwise (resultSSA: SSA) (lhsSSA: SSA) (rhsSSA: SSA) (arch: Architecture)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        return! pfail "Bitwise operations not yet in FNCS as atomic operations - need elaboration (AND, OR, XOR, SHL, SHR)"
    }

/// Unary operations (-, not, ~)
/// Note: Unary minus is typically represented as 0 - x
/// Logical not is typically xor with true (all 1s)
/// For now, return error indicating these need special handling
let pBuildUnary (resultSSA: SSA) (operandSSA: SSA) (arch: Architecture)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        return! pfail "Unary operations need special handling - negation as (0 - x), not as (xor x, -1)"
    }
