/// StringPatterns - String operation elision patterns
///
/// Provides composable patterns for string operations using XParsec.
/// Strings are represented as fat pointers: {ptr: index, length: int}
module Alex.Patterns.StringPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.MLIRAtomics
open Alex.Elements.IndexElements
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// MEMORY COPY PATTERN
// ═══════════════════════════════════════════════════════════

/// Call external memcpy(dest, src, len) -> void*
/// External memcpy declaration will be added at module level if needed
/// resultSSA: The SSA assigned to the memcpy result (from coeffects analysis)
let pMemCopy (resultSSA: SSA) (destSSA: SSA) (srcSSA: SSA) (lenSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Get platform word type (pointers are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Build memcpy call: void* memcpy(void* dest, const void* src, size_t n)
        let args = [
            { SSA = destSSA; Type = platformWordTy }   // dest
            { SSA = srcSSA; Type = platformWordTy }    // src
            { SSA = lenSSA; Type = platformWordTy }    // len
        ]

        // Call external memcpy - uses result SSA from coeffects analysis
        let! call = pFuncCall (Some resultSSA) "memcpy" args platformWordTy

        return [call]
    }

// ═══════════════════════════════════════════════════════════
// STRING CONCATENATION PATTERN
// ═══════════════════════════════════════════════════════════

/// String.concat2: concatenate two strings
/// Result is a new string allocated on stack with combined length
///
/// Fat pointer representation: {ptr: index, length: int}
///
/// Pattern:
/// 1. Extract ptr and length from both input strings
/// 2. Compute combined length
/// 3. Allocate result buffer
/// 4. memcpy(result, str1.ptr, len1)
/// 5. memcpy(result + len1, str2.ptr, len2)
/// 6. Build result fat pointer {result_ptr, combined_length}
let pStringConcat2 (resultSSA: SSA) (str1SSA: SSA) (str2SSA: SSA) (str1Type: MLIRType) (str2Type: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let platformWordTy = state.Platform.PlatformWordType
        let intTy = mapNativeTypeForArch arch Types.intType

        // Allocate temporary SSAs
        // TODO BACKFILL: ALL intermediate SSAs should come from coeffects (nodeExpansionCost)
        let str1PtrSSA = failwith "StringPatterns.pStringConcat2: str1PtrSSA must come from coeffects (removed SSA.V 999970)"
        let str1LenSSA = failwith "StringPatterns.pStringConcat2: str1LenSSA must come from coeffects (removed SSA.V 999971)"
        let str2PtrSSA = failwith "StringPatterns.pStringConcat2: str2PtrSSA must come from coeffects (removed SSA.V 999972)"
        let str2LenSSA = failwith "StringPatterns.pStringConcat2: str2LenSSA must come from coeffects (removed SSA.V 999973)"
        let combinedLenSSA = failwith "StringPatterns.pStringConcat2: combinedLenSSA must come from coeffects (removed SSA.V 999974)"
        let resultBufferSSA = failwith "StringPatterns.pStringConcat2: resultBufferSSA must come from coeffects (removed SSA.V 999975)"
        let resultPtrSSA = failwith "StringPatterns.pStringConcat2: resultPtrSSA must come from coeffects (removed SSA.V 999976)"
        let offsetPtrSSA = failwith "StringPatterns.pStringConcat2: offsetPtrSSA must come from coeffects (removed SSA.V 999977)"
        let resultStructSSA = failwith "StringPatterns.pStringConcat2: resultStructSSA must come from coeffects (removed SSA.V 999978)"

        // 1. Extract components from str1: {ptr[0], length[1]}
        let! extract1Ptr = pExtractValue str1PtrSSA str1SSA [0] TIndex
        let! extract1Len = pExtractValue str1LenSSA str1SSA [1] intTy

        // 2. Extract components from str2: {ptr[0], length[1]}
        let! extract2Ptr = pExtractValue str2PtrSSA str2SSA [0] TIndex
        let! extract2Len = pExtractValue str2LenSSA str2SSA [1] intTy

        // 3. Compute combined length: len1 + len2
        let! addLen = pAddI combinedLenSSA str1LenSSA str2LenSSA

        // 4. Allocate result buffer (combined length bytes)
        let resultTy = TMemRef (TInt I8)
        let! allocOp = pAlloca resultBufferSSA (TInt I8) None

        // 5. Extract result buffer pointer for memcpy
        let! extractResult = pExtractBasePtr resultPtrSSA resultBufferSSA resultTy

        // 6. Cast pointers to platform words for memcpy
        // TODO BACKFILL: Cast SSAs should come from coeffects (nodeExpansionCost)
        let str1PtrWord = failwith "StringPatterns.pStringConcat2: str1PtrWord must come from coeffects (removed SSA.V 999981)"
        let str2PtrWord = failwith "StringPatterns.pStringConcat2: str2PtrWord must come from coeffects (removed SSA.V 999982)"
        let resultPtrWord = failwith "StringPatterns.pStringConcat2: resultPtrWord must come from coeffects (removed SSA.V 999983)"
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA platformWordTy

        // 7. Cast lengths to platform words for memcpy
        // TODO BACKFILL: Length cast SSAs should come from coeffects (nodeExpansionCost)
        let len1Word = failwith "StringPatterns.pStringConcat2: len1Word must come from coeffects (removed SSA.V 999984)"
        let len2Word = failwith "StringPatterns.pStringConcat2: len2Word must come from coeffects (removed SSA.V 999985)"
        let! castLen1 = pIndexCastS len1Word str1LenSSA platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenSSA platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        // TODO BACKFILL: memcpy result SSAs should come from coeffects (nodeExpansionCost)
        let memcpy1ResultSSA = failwith "StringPatterns.pStringConcat2: memcpy1ResultSSA must come from coeffects (removed SSA.V 999986)"
        let! copy1Ops = pMemCopy memcpy1ResultSSA resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1
        let! addOffset = pIndexAdd offsetPtrSSA resultPtrSSA str1LenSSA
        // TODO BACKFILL: offsetPtrWord should come from coeffects (nodeExpansionCost)
        let offsetPtrWord = failwith "StringPatterns.pStringConcat2: offsetPtrWord must come from coeffects (removed SSA.V 999987)"
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        // TODO BACKFILL: memcpy2ResultSSA should come from coeffects (nodeExpansionCost)
        let memcpy2ResultSSA = failwith "StringPatterns.pStringConcat2: memcpy2ResultSSA must come from coeffects (removed SSA.V 999988)"
        let! copy2Ops = pMemCopy memcpy2ResultSSA offsetPtrWord str2PtrWord len2Word

        // 11. Build result string fat pointer {result_ptr, combined_length}
        let totalBytes = mlirTypeSize TIndex + mlirTypeSize intTy
        let stringTy = TMemRefStatic(totalBytes, TInt I8)
        // TODO BACKFILL: struct construction SSAs should come from coeffects (nodeExpansionCost)
        let undefSSA = failwith "StringPatterns.pStringConcat2: undefSSA must come from coeffects (removed SSA.V 999989)"
        let insertPtrSSA = failwith "StringPatterns.pStringConcat2: insertPtrSSA must come from coeffects (removed SSA.V 999990)"
        let! undefOp = pUndef undefSSA stringTy
        let! insertPtr = pInsertValue insertPtrSSA undefSSA resultPtrSSA [0] stringTy
        let! insertLen = pInsertValue resultSSA insertPtrSSA combinedLenSSA [1] stringTy

        // Collect all operations
        let ops =
            [ extract1Ptr; extract1Len
              extract2Ptr; extract2Len
              addLen
              allocOp; extractResult
              cast1; cast2; cast3
              castLen1; castLen2 ]
            @ copy1Ops
            @ [ addOffset; castOffset ]
            @ copy2Ops
            @ [ undefOp; insertPtr; insertLen ]

        return (ops, TRValue { SSA = resultSSA; Type = stringTy })
    }
