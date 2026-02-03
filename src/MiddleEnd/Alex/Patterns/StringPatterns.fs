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
let pMemCopy (destSSA: SSA) (srcSSA: SSA) (lenSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Get platform word type (pointers are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Allocate result SSA for memcpy return value (returns dest pointer)
        let resultSSA = SSA.V 999980

        // Build memcpy call: void* memcpy(void* dest, const void* src, size_t n)
        let args = [
            { SSA = destSSA; Type = platformWordTy }   // dest
            { SSA = srcSSA; Type = platformWordTy }    // src
            { SSA = lenSSA; Type = platformWordTy }    // len
        ]

        // Call external memcpy (returns pointer, but we ignore it)
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
        let str1PtrSSA = SSA.V 999970
        let str1LenSSA = SSA.V 999971
        let str2PtrSSA = SSA.V 999972
        let str2LenSSA = SSA.V 999973
        let combinedLenSSA = SSA.V 999974
        let resultBufferSSA = SSA.V 999975
        let resultPtrSSA = SSA.V 999976
        let offsetPtrSSA = SSA.V 999977
        let resultStructSSA = SSA.V 999978

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
        let str1PtrWord = SSA.V 999981
        let str2PtrWord = SSA.V 999982
        let resultPtrWord = SSA.V 999983
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA platformWordTy

        // 7. Cast lengths to platform words for memcpy
        let len1Word = SSA.V 999984
        let len2Word = SSA.V 999985
        let! castLen1 = pIndexCastS len1Word str1LenSSA platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenSSA platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        let! copy1Ops = pMemCopy resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1
        let! addOffset = pIndexAdd offsetPtrSSA resultPtrSSA str1LenSSA
        let offsetPtrWord = SSA.V 999986
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        let! copy2Ops = pMemCopy offsetPtrWord str2PtrWord len2Word

        // 11. Build result string fat pointer {result_ptr, combined_length}
        let totalBytes = mlirTypeSize TIndex + mlirTypeSize intTy
        let stringTy = TMemRefStatic(totalBytes, TInt I8)
        let undefSSA = SSA.V 999987
        let insertPtrSSA = SSA.V 999988
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
