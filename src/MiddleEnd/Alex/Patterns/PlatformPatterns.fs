/// PlatformPatterns - Platform syscall operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for platform I/O operations.
/// Patterns compose Elements into platform-specific syscall sequences.
module Alex.Patterns.PlatformPatterns

open XParsec
open XParsec.Parsers     // preturn
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.ArithElements
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements

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
/// Build Sys.write syscall pattern
/// Memrefs stay as memrefs - LLVM handles lowering to pointers
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (memref or ptr)
/// - bufferType: Actual MLIR type of buffer (from witness observation)
/// - countSSA: Number of bytes to write SSA
let pSysWrite (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Get platform word type (all syscall parameters and return values are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Pass buffer as-is (memref or ptr) - LLVM lowers to platform ABI
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = bufferSSA; Type = bufferType }
            { SSA = countSSA; Type = platformWordTy }
        ]
        let! writeCall = pFuncCall (Some resultSSA) "write" vals platformWordTy
        return ([writeCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build Sys.read syscall pattern
/// Memrefs stay as memrefs - LLVM handles lowering to pointers
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes read)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Buffer SSA (memref or ptr)
/// - bufferType: Actual MLIR type of buffer (from witness observation)
/// - countSSA: Maximum bytes to read SSA
let pSysRead (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Get platform word type (all syscall parameters and return values are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Pass buffer as-is (memref or ptr) - LLVM lowers to platform ABI
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = bufferSSA; Type = bufferType }
            { SSA = countSSA; Type = platformWordTy }
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy
        return ([readCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }
