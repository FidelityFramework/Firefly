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

// ═══════════════════════════════════════════════════════════
// PLATFORM I/O SYSCALLS
// ═══════════════════════════════════════════════════════════

/// Build Sys.write syscall pattern (portable)
/// Uses func.call (portable) for direct syscall
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Pointer to buffer SSA
/// - countSSA: Number of bytes to write SSA
let pSysWrite (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Portable direct call to write syscall: write(fd: i64, buffer: ptr, count: i64) -> i64
        let vals = [
            { SSA = fdSSA; Type = TInt I64 }
            { SSA = bufferSSA; Type = TPtr }
            { SSA = countSSA; Type = TInt I64 }
        ]
        let! writeCall = pFuncCall (Some resultSSA) "write" vals (TInt I64)

        return ([writeCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }

/// Build Sys.read syscall pattern (portable)
/// Uses func.call (portable) for direct syscall
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes read)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Pointer to buffer SSA
/// - countSSA: Maximum bytes to read SSA
let pSysRead (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Portable direct call to read syscall: read(fd: i64, buffer: ptr, count: i64) -> i64
        let vals = [
            { SSA = fdSSA; Type = TInt I64 }
            { SSA = bufferSSA; Type = TPtr }
            { SSA = countSSA; Type = TInt I64 }
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals (TInt I64)

        return ([readCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }
