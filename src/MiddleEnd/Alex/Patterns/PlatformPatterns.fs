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
/// Build Sys.write syscall pattern with FFI boundary normalization
/// Normalizes memref→ptr at FFI boundaries (Patterns handle FFI concern)
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (memref or ptr depending on source)
/// - bufferType: Actual MLIR type of buffer (TMemRefScalar, TMemRef, or TIndex)
/// - countSSA: Number of bytes to write SSA
/// - conversionSSA: Optional SSA for memref→ptr conversion (witness allocates if needed)
let pSysWriteTyped (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) (countSSA: SSA) (conversionSSA: SSA option) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Convert memref to pointer for FFI boundary (syscalls expect integer pointer)
        match bufferType with
        | TMemRefScalar _ | TMemRef _ ->
            // Buffer is memref - extract pointer as portable integer
            match conversionSSA with
            | Some convSSA ->
                // Extract pointer as index, then cast to platform word (i64/i32)
                let! state = getUserState
                let platformWordTy = state.Platform.PlatformWordType

                let indexSSA = SSA.V 999993  // Temp for index result
                let! extractOp = pExtractBasePtr indexSSA bufferSSA bufferType
                let! castOp = pIndexCastS convSSA indexSSA platformWordTy

                let vals = [
                    { SSA = fdSSA; Type = TInt I64 }
                    { SSA = convSSA; Type = platformWordTy }  // Use converted pointer (i64/i32)
                    { SSA = countSSA; Type = TInt I64 }
                ]
                let! writeCall = pFuncCall (Some resultSSA) "write" vals (TInt I64)

                return ([extractOp; castOp; writeCall], TRValue { SSA = resultSSA; Type = TInt I64 })
            | None ->
                return failwith "pSysWriteTyped: conversionSSA required for memref buffer"
        | _ ->
            // Buffer is already a pointer/integer type - use directly
            let vals = [
                { SSA = fdSSA; Type = TInt I64 }
                { SSA = bufferSSA; Type = bufferType }
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
            { SSA = bufferSSA; Type = TIndex }
            { SSA = countSSA; Type = TInt I64 }
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals (TInt I64)

        return ([readCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }
