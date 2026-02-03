/// LiteralPatterns - Literal value emission and string handling
///
/// PUBLIC: Witnesses use these to emit literal constants and string operations.
/// Literal patterns map NativeLiteral values to MLIR constants.
/// String patterns handle string literal globals and pointer/length extraction for FFI.
module Alex.Patterns.LiteralPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRElements  // pConstI, pConstF, pUndef, pInsertValue
open Alex.Elements.MemRefElements  // pMemRefGetGlobal, pExtractBasePtr, pMemRefDim
open Alex.Elements.IndexElements  // pIndexCastS
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// LITERAL PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build literal: Match literal from PSG and emit constant MLIR
let pBuildLiteral (lit: NativeLiteral) (ssa: SSA) (arch: Architecture) : PSGParser<MLIROp list * TransferResult> =
    parser {
        match lit with
        | NativeLiteral.Unit ->
            let ty = mapNTUKindToMLIRType arch NTUKind.NTUunit
            let! op = pConstI ssa 0L ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Bool b ->
            let value = if b then 1L else 0L
            let ty = mapNTUKindToMLIRType arch NTUKind.NTUbool
            let! op = pConstI ssa value ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Int (n, kind) ->
            let ty = mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa n ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.UInt (n, kind) ->
            let ty = mapNTUKindToMLIRType arch kind
            let! op = pConstI ssa (int64 n) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Char c ->
            let ty = mapNTUKindToMLIRType arch NTUKind.NTUchar
            let! op = pConstI ssa (int64 c) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Float (f, kind) ->
            let ty = mapNTUKindToMLIRType arch kind
            let! op = pConstF ssa f ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.String _ ->
            // String literals require witness-level handling with multiple SSAs
            // Use pBuildStringLiteral pattern instead
            return! fail (Message "String literals require pBuildStringLiteral pattern with SSA list")

        | _ ->
            return! fail (Message $"Unsupported literal: {lit}")
    }

// ═══════════════════════════════════════════════════════════
// STRING PATTERNS
// ═══════════════════════════════════════════════════════════

/// Derive global reference name from string content (pure)
let deriveGlobalRef (content: string) : string =
    let hash = content.GetHashCode()
    sprintf "str_%d" (abs hash)

/// Derive byte length from string content (pure)
let deriveByteLength (content: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(content)

/// Build string literal: get reference to global memref (portable MLIR)
/// SSAs: [0-3] = unused (legacy), [4] = memref.get_global result
/// Returns: ((ops, globalName, content, byteLength), result)
/// NOTE: Witness must emit memref.global to TopLevelOps separately
let pBuildStringLiteral (content: string) (ssas: SSA list) (arch: Architecture)
                         : PSGParser<(MLIROp list * string * string * int) * TransferResult> =
    parser {
        do! emitTrace "pBuildStringLiteral.entry" (sprintf "content='%s', ssas=%A, arch=%A" content ssas arch)

        // Declarative guard - no imperative if statements
        do! ensure (ssas.Length >= 5) $"pBuildStringLiteral: Expected 5 SSAs, got {ssas.Length}"

        do! emitTrace "pBuildStringLiteral.ssa_validated" (sprintf "SSA count OK: %d" ssas.Length)

        // Use StringCollection pure derivation (coeffect model)
        let globalName = deriveGlobalRef content
        let byteLength = deriveByteLength content

        do! emitTrace "pBuildStringLiteral.derived" (sprintf "globalName=%s, byteLength=%d" globalName byteLength)

        // String type: memref<Nxi8> where N is byte length (static-sized memref for literals)
        let stringTy = TMemRefStatic (byteLength, TInt I8)

        do! emitTrace "pBuildStringLiteral.types" (sprintf "stringTy=%A" stringTy)

        // InlineOps: Get reference to global memref
        let resultSSA = ssas.[4]  // Use last SSA for result (others unused for now)

        do! emitTrace "pBuildStringLiteral.ssas_extracted" (sprintf "result=%A" resultSSA)

        // memref.get_global @globalName : memref<?xi8>
        do! emitTrace "pBuildStringLiteral.calling_pMemRefGetGlobal" (sprintf "resultSSA=%A, globalName=%s, stringTy=%A" resultSSA globalName stringTy)
        let! getGlobalOp = pMemRefGetGlobal resultSSA globalName stringTy

        do! emitTrace "pBuildStringLiteral.elements_complete" "memref.get_global succeeded"

        let inlineOps = [getGlobalOp]
        let result = TRValue { SSA = resultSSA; Type = stringTy }

        do! emitTrace "pBuildStringLiteral.returning" (sprintf "Returning %d ops" (List.length inlineOps))

        // Return ops + (globalName, content, byteLength) for witness to emit memref.global
        return ((inlineOps, globalName, content, byteLength), result)
    }

/// Extract pointer from memref (for FFI/syscalls like Sys.write)
/// String is now memref<?xi8>, extract base pointer as index then cast to target type
let pStringGetPtr (stringSSA: SSA) (ptrSSA: SSA) (ptrTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let memrefTy = TMemRef (TInt I8)

        // Check if we need to cast index → ptrTy
        match ptrTy with
        | TIndex ->
            // No cast needed - result is index
            let! extractOp = pExtractBasePtr ptrSSA stringSSA memrefTy
            return [extractOp]
        | _ ->
            // Extract as index, then cast to target type (e.g., i64 for x86-64, i32 for ARM32)
            let indexSSA = SSA.V 999994  // Temporary SSA for index result
            let! extractOp = pExtractBasePtr indexSSA stringSSA memrefTy
            let! castOp = pIndexCastS ptrSSA indexSSA ptrTy
            return [extractOp; castOp]
    }

/// Extract length from memref descriptor (for FFI/syscalls)
/// Uses memref.dim to get the dynamic size
let pStringGetLength (stringSSA: SSA) (lengthSSA: SSA) (lengthTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        // memref.dim %memref, %c0 : memref<?xi8>
        // Need constant 0 for dimension index (0th dimension = length)
        let dimIndexSSA = SSA.V 999999  // Temporary SSA for constant
        let memrefTy = TMemRef (TInt I8)
        let! constOp = pConstI dimIndexSSA 0L TIndex
        let! dimOp = pMemRefDim lengthSSA stringSSA dimIndexSSA memrefTy
        return [constOp; dimOp]
    }

/// Construct string from pointer and length
let pStringConstruct (ptrTy: MLIRType) (lengthTy: MLIRType) (ptr: SSA) (length: SSA) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 3) $"pStringConstruct: Expected at least 3 SSAs, got {ssas.Length}"

        // String type is {ptr: nativeptr<byte>, length: int}
        let stringTy = TStruct [ptrTy; lengthTy]
        let! undefOp = pUndef ssas.[0] stringTy
        let! insertPtrOp = pInsertValue ssas.[1] ssas.[0] ptr [0] stringTy
        let! insertLenOp = pInsertValue ssas.[2] ssas.[1] length [1] stringTy
        return [undefOp; insertPtrOp; insertLenOp]
    }
