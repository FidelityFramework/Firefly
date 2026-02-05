/// PlatformPatterns - Platform syscall operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for platform I/O operations.
/// Patterns compose Elements into platform-specific syscall sequences.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.PlatformPatterns

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers     // preturn
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.ArithElements
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements
open Alex.Elements.MLIRAtomics

// ═══════════════════════════════════════════════════════════
// PLATFORM I/O SYSCALLS
// ═══════════════════════════════════════════════════════════

/// Build Sys.write syscall pattern (portable)
/// Uses func.call (portable) for direct syscall
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (memref or ptr depending on source)
/// - bufferType: Actual MLIR type of buffer (TMemRefScalar, TMemRef, or TIndex)
/// - countSSA: Number of bytes to write SSA
/// Build Sys.write syscall pattern with FFI pointer extraction
/// Uses MLIR standard dialects (memref.extract_aligned_pointer_as_index + index.casts)
/// to extract pointers from memrefs at FFI boundaries.
///
/// Buffers are ALWAYS memrefs at syscall boundaries - we ALWAYS extract pointers.
/// Length is extracted via memref.dim (NO explicit count parameter).
///
/// SSA layout (6 SSAs):
///   [0] = buf_ptr_index (memref.extract_aligned_pointer_as_index)
///   [1] = buf_ptr_i64 (index.casts)
///   [2] = dim_index_const (constant 0 for dimension)
///   [3] = count_index (memref.dim result, index type)
///   [4] = count_i64 (count cast to platform word)
///   [5] = result (func.call return value)
///
/// Parameters:
/// - nodeId: NodeId for extracting SSAs from coeffects (6 SSAs allocated)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (ALWAYS memref)
/// - bufferType: MLIR type of buffer (ALWAYS TMemRef or TMemRefStatic)
let pSysWrite (nodeId: NodeId) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Buffer is ALWAYS a memref at syscall boundary
        // We ALWAYS need FFI pointer extraction (no conditionals)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 6) $"pSysWrite: Expected 6 SSAs, got {ssas.Length}"

        let buf_ptr_index = ssas.[0]
        let buf_ptr_i64 = ssas.[1]
        let dim_index_const = ssas.[2]
        let count_index = ssas.[3]
        let count_i64 = ssas.[4]
        let resultSSA = ssas.[5]

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Extract pointer from memref: memref.extract_aligned_pointer_as_index
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType

        // Cast index to i64: index.casts
        let! castOp = pIndexCastS buf_ptr_i64 buf_ptr_index platformWordTy

        // Extract length via memref.dim (dimension 0)
        let! dimConstOp = pConstI dim_index_const 0L TIndex
        let! dimOp = pMemRefDim count_index bufferSSA dim_index_const bufferType

        // Cast count to platform word for syscall
        let! countCastOp = pIndexCastS count_i64 count_index platformWordTy

        // Syscall with extracted i64 pointer and length
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = count_i64; Type = platformWordTy }    // Length from memref.dim
        ]
        let! writeCall = pFuncCall (Some resultSSA) "write" vals platformWordTy

        return ([extractOp; castOp; dimConstOp; dimOp; countCastOp; writeCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build Sys.read syscall pattern with FFI pointer extraction
/// Uses MLIR standard dialects (memref.extract_aligned_pointer_as_index + index.casts)
/// to extract pointers from memrefs at FFI boundaries.
///
/// Buffers are ALWAYS memrefs at syscall boundaries - we ALWAYS extract pointers.
/// Buffer capacity (maxCount) is extracted via memref.dim (NO explicit count parameter).
///
/// SSA layout (6 SSAs):
///   [0] = buf_ptr_index (memref.extract_aligned_pointer_as_index)
///   [1] = buf_ptr_i64 (index.casts)
///   [2] = dim_index_const (constant 0 for dimension)
///   [3] = capacity_index (memref.dim result, index type)
///   [4] = capacity_i64 (capacity cast to platform word)
///   [5] = result (func.call return value)
///
/// Parameters:
/// - nodeId: NodeId for extracting SSAs from coeffects (6 SSAs allocated)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Buffer SSA (ALWAYS memref)
/// - bufferType: MLIR type of buffer (ALWAYS TMemRef or TMemRefStatic)
let pSysRead (nodeId: NodeId) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Buffer is ALWAYS a memref at syscall boundary
        // We ALWAYS need FFI pointer extraction (no conditionals)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 6) $"pSysRead: Expected 6 SSAs, got {ssas.Length}"

        let buf_ptr_index = ssas.[0]
        let buf_ptr_i64 = ssas.[1]
        let dim_index_const = ssas.[2]
        let capacity_index = ssas.[3]
        let capacity_i64 = ssas.[4]
        let resultSSA = ssas.[5]

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Extract pointer from memref: memref.extract_aligned_pointer_as_index
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType

        // Cast index to i64: index.casts
        let! castOp = pIndexCastS buf_ptr_i64 buf_ptr_index platformWordTy

        // Extract buffer capacity via memref.dim (dimension 0)
        let! dimConstOp = pConstI dim_index_const 0L TIndex
        let! dimOp = pMemRefDim capacity_index bufferSSA dim_index_const bufferType

        // Cast capacity to platform word for syscall
        let! capacityCastOp = pIndexCastS capacity_i64 capacity_index platformWordTy

        // Syscall with extracted i64 pointer and capacity
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = capacity_i64; Type = platformWordTy } // Capacity from memref.dim
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy

        return ([extractOp; castOp; dimConstOp; dimOp; capacityCastOp; readCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }
