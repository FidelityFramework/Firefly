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
        // Get platform word type (all syscall parameters and return values are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Convert memref to pointer for FFI boundary (syscalls expect integer pointer)
        match bufferType with
        | TMemRefScalar _ | TMemRef _ ->
            // Buffer is memref - extract pointer as portable integer
            match conversionSSA with
            | Some convSSA ->
                // Extract pointer as index, then cast to platform word (i64/i32)
                let indexSSA = SSA.V 999993  // Temp for index result
                let! extractOp = pExtractBasePtr indexSSA bufferSSA bufferType
                let! castOp = pIndexCastS convSSA indexSSA platformWordTy

                let vals = [
                    { SSA = fdSSA; Type = platformWordTy }
                    { SSA = convSSA; Type = platformWordTy }
                    { SSA = countSSA; Type = platformWordTy }
                ]
                let! writeCall = pFuncCall (Some resultSSA) "write" vals platformWordTy

                return ([extractOp; castOp; writeCall], TRValue { SSA = resultSSA; Type = platformWordTy })
            | None ->
                return failwith "pSysWriteTyped: conversionSSA required for memref buffer"
        | _ ->
            // Buffer is already a pointer/integer type - use directly
            let vals = [
                { SSA = fdSSA; Type = platformWordTy }
                { SSA = bufferSSA; Type = bufferType }
                { SSA = countSSA; Type = platformWordTy }
            ]
            let! writeCall = pFuncCall (Some resultSSA) "write" vals platformWordTy
            return ([writeCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build Sys.read syscall pattern with FFI boundary normalization
/// Normalizes memref→ptr at FFI boundaries (Patterns handle FFI concern)
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes read)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Buffer SSA (memref or ptr depending on source)
/// - bufferType: Actual MLIR type of buffer (from witness observation)
/// - countSSA: Maximum bytes to read SSA
let pSysRead (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Get platform word type (all syscall parameters and return values are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Convert memref to pointer for FFI boundary (syscalls expect integer pointer)
        match bufferType with
        | TMemRefScalar _ | TMemRef _ | TMemRefStatic _ ->
            // Buffer is memref - extract pointer as index, then cast to platform word
            let indexSSA = SSA.V 999995  // Temp for index result
            let convSSA = SSA.V 999996   // Temp for platform word result

            // Extract pointer as portable index
            let! extractOp = pExtractBasePtr indexSSA bufferSSA bufferType

            // Cast index to platform word for FFI boundary
            let! castOp = pIndexCastS convSSA indexSSA platformWordTy

            // Emit syscall with converted pointer
            let vals = [
                { SSA = fdSSA; Type = platformWordTy }
                { SSA = convSSA; Type = platformWordTy }
                { SSA = countSSA; Type = platformWordTy }
            ]
            let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy

            return ([extractOp; castOp; readCall], TRValue { SSA = resultSSA; Type = platformWordTy })

        | TIndex ->
            // Already index - cast directly to platform word
            let convSSA = SSA.V 999996

            let! castOp = pIndexCastS convSSA bufferSSA platformWordTy

            let vals = [
                { SSA = fdSSA; Type = platformWordTy }
                { SSA = convSSA; Type = platformWordTy }
                { SSA = countSSA; Type = platformWordTy }
            ]
            let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy

            return ([castOp; readCall], TRValue { SSA = resultSSA; Type = platformWordTy })

        | _ ->
            // Scalar type - pass through directly (already platform-compatible)
            let vals = [
                { SSA = fdSSA; Type = platformWordTy }
                { SSA = bufferSSA; Type = bufferType }
                { SSA = countSSA; Type = platformWordTy }
            ]
            let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy

            return ([readCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }
