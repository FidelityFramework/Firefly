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
// STRING CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// Convert static buffer to dynamic string (memref<?xi8>)
/// Takes buffer (memref<Nxi8>), casts to memref<?xi8>
/// Length is intrinsic to memref descriptor, no separate parameter needed
let pStringFromBuffer (resultSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Cast static memref to dynamic for function boundary
        // memref<Nxi8> -> memref<?xi8>
        let stringType = TMemRef (TInt I8)
        let! castOp = pMemRefCast resultSSA bufferSSA bufferType stringType
        return ([castOp], TRValue { SSA = resultSSA; Type = stringType })
    }

/// Get string length via memref.dim
/// Strings ARE memrefs (memref<?xi8>), length is intrinsic to descriptor
/// SSA layout:
///   dimConstSSA: dimension constant (0 for 1D arrays)
///   lenIndexSSA: memref.dim result (index type)
///   resultSSA: cast to int result
let pStringLength (resultSSA: SSA) (stringSSA: SSA) (stringType: MLIRType) (dimConstSSA: SSA) (lenIndexSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let intTy = mapNativeTypeForArch state.Platform.TargetArch Types.intType

        // Get string dimension (dimension 0 for 1D memref)
        let! dimConstOp = pConstI dimConstSSA 0L TIndex
        let! dimOp = pMemRefDim lenIndexSSA stringSSA dimConstSSA stringType

        // Cast index to int
        let! castOp = pIndexCastS resultSSA lenIndexSSA intTy

        return ([dimConstOp; dimOp; castOp], TRValue { SSA = resultSSA; Type = intTy })
    }

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

/// String.concat2: concatenate two strings using pure memref operations
/// SSA layout (17 total - pure memref with index arithmetic, NO i64 round-trip):
///   [0] = dim const for str1 (dimension 0)
///   [1] = str1 len (memref.dim result, index type)
///   [2] = dim const for str2 (dimension 0)
///   [3] = str2 len (memref.dim result, index type)
///   [4] = combined length (index type - NO cast to i64!)
///   [5] = result buffer (memref)
///   [6] = str1 ptr (memref.extract_aligned_pointer_as_index)
///   [7] = str2 ptr (memref.extract_aligned_pointer_as_index)
///   [8] = result ptr (memref.extract_aligned_pointer_as_index)
///   [9] = str1 ptr word cast
///   [10] = str2 ptr word cast
///   [11] = result ptr word cast
///   [12] = str1 len word cast
///   [13] = str2 len word cast
///   [14] = memcpy1 result
///   [15] = offset ptr (result + len1, index type)
///   [16] = offset ptr word cast
///   [17] = memcpy2 result
let pStringConcat2 (ssas: SSA list) (str1SSA: SSA) (str2SSA: SSA) (str1Type: MLIRType) (str2Type: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        do! ensure (ssas.Length >= 18) $"pStringConcat2: Expected 18 SSAs, got {ssas.Length}"

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        let dimConst1SSA = ssas.[0]
        let str1LenSSA = ssas.[1]
        let dimConst2SSA = ssas.[2]
        let str2LenSSA = ssas.[3]
        let combinedLenSSA = ssas.[4]
        let resultBufferSSA = ssas.[5]
        let str1PtrSSA = ssas.[6]
        let str2PtrSSA = ssas.[7]
        let resultPtrSSA = ssas.[8]
        let str1PtrWord = ssas.[9]
        let str2PtrWord = ssas.[10]
        let resultPtrWord = ssas.[11]
        let len1Word = ssas.[12]
        let len2Word = ssas.[13]
        let memcpy1ResultSSA = ssas.[14]
        let offsetPtrSSA = ssas.[15]
        let offsetPtrWord = ssas.[16]
        let memcpy2ResultSSA = ssas.[17]

        // 1. Get str1 length via memref.dim (strings ARE memrefs) → index
        let! dimConst1Op = pConstI dimConst1SSA 0L TIndex
        let! dim1Op = pMemRefDim str1LenSSA str1SSA dimConst1SSA str1Type

        // 2. Get str2 length via memref.dim → index
        let! dimConst2Op = pConstI dimConst2SSA 0L TIndex
        let! dim2Op = pMemRefDim str2LenSSA str2SSA dimConst2SSA str2Type

        // 3. Compute combined length: len1 + len2 (index arithmetic via arith.addi, NO i64!)
        let addLenOp = ArithOp (AddI (combinedLenSSA, str1LenSSA, str2LenSSA, TIndex))

        // 4. Allocate result buffer with runtime size (index type, NO conversion!)
        let resultTy = TMemRef (TInt I8)
        let! allocOp = pAlloc resultBufferSSA combinedLenSSA (TInt I8)

        // 5. Extract pointers from memrefs for memcpy (FFI boundary)
        let! extractStr1Ptr = pExtractBasePtr str1PtrSSA str1SSA str1Type
        let! extractStr2Ptr = pExtractBasePtr str2PtrSSA str2SSA str2Type
        let! extractResultPtr = pExtractBasePtr resultPtrSSA resultBufferSSA resultTy

        // 6. Cast pointers to platform words for memcpy
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA platformWordTy

        // 7. Cast lengths to platform words for memcpy (index → platform word)
        let! castLen1 = pIndexCastS len1Word str1LenSSA platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenSSA platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        let! copy1Ops = pMemCopy memcpy1ResultSSA resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1 (index arithmetic via arith.addi)
        let addOffset = ArithOp (AddI (offsetPtrSSA, resultPtrSSA, str1LenSSA, TIndex))
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        let! copy2Ops = pMemCopy memcpy2ResultSSA offsetPtrWord str2PtrWord len2Word

        // 11. Return result buffer as memref (NO fat pointer struct construction)
        // In MLIR: string IS memref<?xi8> directly

        // Collect all operations (pure index arithmetic - NO i64 conversions!)
        let ops =
            [dimConst1Op; dim1Op] @
            [dimConst2Op; dim2Op] @
            [addLenOp; allocOp] @
            [extractStr1Ptr; extractStr2Ptr; extractResultPtr] @
            [cast1; cast2; cast3; castLen1; castLen2] @
            copy1Ops @
            [addOffset; castOffset] @
            copy2Ops

        return (ops, TRValue { SSA = resultBufferSSA; Type = resultTy })
    }
