/// MemRefPatterns - Composable patterns for memref operations
///
/// PUBLIC: Witnesses import these patterns to emit memref operations.
/// Patterns compose Element primitives into domain-specific operations.
///
/// ARCHITECTURE: Patterns are PUBLIC, Elements are module internal.
/// This enforces composition - witnesses cannot directly emit Element operations.
module Alex.Patterns.MemRefPatterns

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Traversal.TransferTypes
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements  // pIndexConst
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// MUTABLE VARIABLE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build complete mutable binding: alloca + initialize with store
/// For F# `let mutable x = initialValue`
///
/// Emits:
///   %xRef = memref.alloca() : memref<1x{elemType}>
///   memref.store %initialValue, %xRef[%c0] : {elemType}, memref<1x{elemType}>
///
/// Returns: memref SSA name (for VarRef to recall)
let pBuildMutableBinding (nodeId: int) (elemType: MLIRType) (initSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSAs from Coeffects
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 2) $"pBuildMutableBinding: Expected at least 2 SSAs, got {ssas.Length}"
        let memrefSSA = ssas.[0]  // For memref allocation result
        let zeroSSA = ssas.[1]    // For zero constant index

        // Step 2: Allocate memref (rank-1 with single element for scalar)
        let! allocOp = pAlloca memrefSSA 1 elemType None

        // Step 3: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 4: Store initial value to memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore initSSA memrefSSA [zeroSSA] elemType memrefType

        let ops = [allocOp; zeroOp; storeOp]
        let result = TRValue { SSA = memrefSSA; Type = TMemRef elemType }
        return (ops, result)
    }

/// Load value from mutable variable
/// For F# VarRef to mutable binding
///
/// Emits:
///   %c0 = arith.constant 0 : index
///   %value = memref.load %memrefSSA[%c0] : memref<1x{elemType}>
///
/// Returns: loaded value SSA (type = elemType, not TMemRef)
let pLoadMutableVariable (nodeId: int) (memrefSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSAs from Coeffects
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 2) $"pLoadMutableVariable: Expected at least 2 SSAs, got {ssas.Length}"
        let zeroSSA = ssas.[0]    // For zero constant index
        let valueSSA = ssas.[1]   // For loaded value result

        // Step 2: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 3: Load value from memref
        let! loadOp = pLoad valueSSA memrefSSA [zeroSSA]

        let ops = [zeroOp; loadOp]
        let result = TRValue { SSA = valueSSA; Type = elemType }
        return (ops, result)
    }

/// Store value to mutable variable
/// For F# `x <- newValue` (SemanticKind.Set)
///
/// Emits:
///   %c0 = arith.constant 0 : index
///   memref.store %newValue, %memrefSSA[%c0] : {elemType}, memref<1x{elemType}>
///
/// Returns: TRVoid (stores don't produce values)
let pStoreMutableVariable (nodeId: int) (memrefSSA: SSA) (valueSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSA from Coeffects for zero constant
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 1) $"pStoreMutableVariable: Expected at least 1 SSA, got {ssas.Length}"
        let zeroSSA = ssas.[0]  // For zero constant index

        // Step 2: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 3: Store value to memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore valueSSA memrefSSA [zeroSSA] elemType memrefType

        let ops = [zeroOp; storeOp]
        return (ops, TRVoid)
    }
