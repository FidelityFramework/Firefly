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
open Alex.Elements.MLIRAtomics  // pConstI, pConstF, pUndef, pInsertValue
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
/// TypeLayout.Opaque: Single SSA for memref.get_global
/// SSA layout: [0] = memref.get_global result
/// Returns: ((ops, globalName, content, byteLength), result)
/// NOTE: Witness must emit memref.global to TopLevelOps separately
let pBuildStringLiteral (content: string) (ssas: SSA list) (arch: Architecture)
                         : PSGParser<(MLIROp list * string * string * int) * TransferResult> =
    parser {
        do! emitTrace "pBuildStringLiteral.entry" (sprintf "content='%s', ssas=%A, arch=%A" content ssas arch)

        // Declarative guard - no imperative if statements
        do! ensure (ssas.Length >= 1) $"pBuildStringLiteral: Expected 1 SSA, got {ssas.Length}"

        do! emitTrace "pBuildStringLiteral.ssa_validated" (sprintf "SSA count OK: %d" ssas.Length)

        // Use StringCollection pure derivation (coeffect model)
        let globalName = deriveGlobalRef content
        let byteLength = deriveByteLength content

        do! emitTrace "pBuildStringLiteral.derived" (sprintf "globalName=%s, byteLength=%d" globalName byteLength)

        // String type: memref<Nxi8> where N is byte length (static-sized memref for literals)
        let stringTy = TMemRefStatic (byteLength, TInt I8)

        do! emitTrace "pBuildStringLiteral.types" (sprintf "stringTy=%A" stringTy)

        // InlineOps: Get reference to global memref (TypeLayout.Opaque)
        let resultSSA = ssas.[0]  // Single SSA for memref.get_global

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

// DEAD CODE DELETED: pStringGetPtr and pStringGetLength were unused
// Pointer extraction happens inline in PlatformPatterns.pSysWrite
// Length extraction happens inline in PlatformPatterns.pSysWrite via memref.dim
