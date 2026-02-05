/// CollectionPatterns - Immutable collection patterns via DU construction
///
/// PUBLIC: Witnesses use these to emit Option, List, Map, Set, Result operations.
/// All collections are implemented as discriminated unions with tag-based dispatch.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Patterns access SSAs monadically via getUserState.
/// Witnesses pass NodeIds, NOT SSAs. Patterns return TransferResult with the result SSA.
/// This eliminates ALL imperative lookups in witnesses - pure XParsec monadic observation.
module Alex.Patterns.CollectionPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes  // TransferResult, TRValue, TRVoid
open Alex.Elements.MLIRAtomics  // pExtractValue, pConstI
open Alex.Elements.ArithElements  // pCmpI
open Alex.Patterns.MemoryPatterns  // pDUCase - DU construction foundation
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// ═══════════════════════════════════════════════════════════
// OPTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Option.Some: tag=1 + value
/// Returns: (ops, result) where result is the constructed Option<'T> value
let pOptionSome (nodeId: NodeId) (value: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 1L [value] ty

/// Option.None: tag=0
/// Returns: (ops, result) where result is the constructed Option<'T> value
let pOptionNone (nodeId: NodeId) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 0L [] ty

/// Option.IsSome: extract tag, compare with 1
/// Returns: (ops, result) where result is i1 boolean
let pOptionIsSome (nodeId: NodeId) (optionSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pOptionIsSome: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let tagSSA = ssas.[1]
        let oneSSA = ssas.[2]
        let resultSSA = ssas.[3]

        let tagTy = TInt I8
        let! extractTagOps = pExtractValue tagSSA optionSSA 0 offsetSSA tagTy
        let! oneConstOp = pConstI oneSSA 1L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA oneSSA tagTy
        let ops = extractTagOps @ [oneConstOp; cmpOp]
        return (ops, TRValue { SSA = resultSSA; Type = TInt I1 })
    }

/// Option.IsNone: extract tag, compare with 0
/// Returns: (ops, result) where result is i1 boolean
let pOptionIsNone (nodeId: NodeId) (optionSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pOptionIsNone: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let tagSSA = ssas.[1]
        let zeroSSA = ssas.[2]
        let resultSSA = ssas.[3]

        let tagTy = TInt I8
        let! extractTagOps = pExtractValue tagSSA optionSSA 0 offsetSSA tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA tagTy
        let ops = extractTagOps @ [zeroConstOp; cmpOp]
        return (ops, TRValue { SSA = resultSSA; Type = TInt I1 })
    }

/// Option.Get: extract value field (index 1)
/// Returns: (ops, result) where result is the extracted value
let pOptionGet (nodeId: NodeId) (optionSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pOptionGet: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA optionSSA 1 offsetSSA valueTy
        return (ops, TRValue { SSA = resultSSA; Type = valueTy })
    }

// ═══════════════════════════════════════════════════════════
// LIST PATTERNS
// ═══════════════════════════════════════════════════════════

/// List.Cons: tag=1 + head + tail
/// Returns: (ops, result) where result is the constructed List<'T> value
let pListCons (nodeId: NodeId) (head: Val) (tail: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 1L [head; tail] ty

/// List.Empty: tag=0
/// Returns: (ops, result) where result is the constructed List<'T> value
let pListEmpty (nodeId: NodeId) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 0L [] ty

/// List.IsEmpty: extract tag, compare with 0
/// Returns: (ops, result) where result is i1 boolean
let pListIsEmpty (nodeId: NodeId) (listSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pListIsEmpty: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let tagSSA = ssas.[1]
        let zeroSSA = ssas.[2]
        let resultSSA = ssas.[3]

        let tagTy = TInt I8
        let! extractTagOps = pExtractValue tagSSA listSSA 0 offsetSSA tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA tagTy
        let ops = extractTagOps @ [zeroConstOp; cmpOp]
        return (ops, TRValue { SSA = resultSSA; Type = TInt I1 })
    }

/// List.Head: extract first element (index 1)
/// Returns: (ops, result) where result is the head value
let pListHead (nodeId: NodeId) (listSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pListHead: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA listSSA 1 offsetSSA valueTy
        return (ops, TRValue { SSA = resultSSA; Type = valueTy })
    }

/// List.Tail: extract tail list (index 2)
/// Returns: (ops, result) where result is the tail list
let pListTail (nodeId: NodeId) (listSSA: SSA) (tailTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pListTail: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA listSSA 2 offsetSSA tailTy
        return (ops, TRValue { SSA = resultSSA; Type = tailTy })
    }

// ═══════════════════════════════════════════════════════════
// MAP PATTERNS
// ═══════════════════════════════════════════════════════════

/// Map.Empty: empty tree structure (tag=0 for leaf node)
/// Returns: (ops, result) where result is the constructed Map<'K,'V> value
let pMapEmpty (nodeId: NodeId) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 0L [] ty

/// Map.Add: create new tree node with key, value, left, right subtrees
/// Returns: (ops, result) where result is the constructed Map<'K,'V> value
let pMapAdd (nodeId: NodeId) (key: Val) (value: Val) (left: Val) (right: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 1L [key; value; left; right] ty

/// Map.Key: extract key field (index 1)
/// Returns: (ops, result) where result is the extracted key
let pMapKey (nodeId: NodeId) (mapNodeSSA: SSA) (keyTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pMapKey: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA mapNodeSSA 1 offsetSSA keyTy
        return (ops, TRValue { SSA = resultSSA; Type = keyTy })
    }

/// Map.Value: extract value field (index 2)
/// Returns: (ops, result) where result is the extracted value
let pMapValue (nodeId: NodeId) (mapNodeSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pMapValue: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA mapNodeSSA 2 offsetSSA valueTy
        return (ops, TRValue { SSA = resultSSA; Type = valueTy })
    }

/// Map.Left: extract left subtree (index 3)
/// Returns: (ops, result) where result is the left subtree
let pMapLeft (nodeId: NodeId) (mapNodeSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pMapLeft: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA mapNodeSSA 3 offsetSSA subtreeTy
        return (ops, TRValue { SSA = resultSSA; Type = subtreeTy })
    }

/// Map.Right: extract right subtree (index 4)
/// Returns: (ops, result) where result is the right subtree
let pMapRight (nodeId: NodeId) (mapNodeSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pMapRight: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA mapNodeSSA 4 offsetSSA subtreeTy
        return (ops, TRValue { SSA = resultSSA; Type = subtreeTy })
    }

/// Map.Height: extract height field (index 5)
/// Returns: (ops, result) where result is the height value
let pMapHeight (nodeId: NodeId) (mapNodeSSA: SSA) (heightTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pMapHeight: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA mapNodeSSA 5 offsetSSA heightTy
        return (ops, TRValue { SSA = resultSSA; Type = heightTy })
    }

// ═══════════════════════════════════════════════════════════
// SET PATTERNS
// ═══════════════════════════════════════════════════════════

/// Set.Empty: empty tree structure (tag=0 for leaf node)
/// Returns: (ops, result) where result is the constructed Set<'T> value
let pSetEmpty (nodeId: NodeId) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 0L [] ty

/// Set.Add: create new tree node with element, left, right subtrees
/// Returns: (ops, result) where result is the constructed Set<'T> value
let pSetAdd (nodeId: NodeId) (element: Val) (left: Val) (right: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 1L [element; left; right] ty

/// Set.Value: extract element field (index 1)
/// Returns: (ops, result) where result is the extracted element
let pSetValue (nodeId: NodeId) (setNodeSSA: SSA) (elemTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pSetValue: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA setNodeSSA 1 offsetSSA elemTy
        return (ops, TRValue { SSA = resultSSA; Type = elemTy })
    }

/// Set.Left: extract left subtree (index 2)
/// Returns: (ops, result) where result is the left subtree
let pSetLeft (nodeId: NodeId) (setNodeSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pSetLeft: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA setNodeSSA 2 offsetSSA subtreeTy
        return (ops, TRValue { SSA = resultSSA; Type = subtreeTy })
    }

/// Set.Right: extract right subtree (index 3)
/// Returns: (ops, result) where result is the right subtree
let pSetRight (nodeId: NodeId) (setNodeSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pSetRight: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA setNodeSSA 3 offsetSSA subtreeTy
        return (ops, TRValue { SSA = resultSSA; Type = subtreeTy })
    }

/// Set.Height: extract height field (index 4)
/// Returns: (ops, result) where result is the height value
let pSetHeight (nodeId: NodeId) (setNodeSSA: SSA) (heightTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pSetHeight: Expected 2 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! ops = pExtractValue resultSSA setNodeSSA 4 offsetSSA heightTy
        return (ops, TRValue { SSA = resultSSA; Type = heightTy })
    }

// ═══════════════════════════════════════════════════════════
// RESULT PATTERNS
// ═══════════════════════════════════════════════════════════

/// Result.Ok: tag=0 + value
/// Returns: (ops, result) where result is the constructed Result<'T,'E> value
let pResultOk (nodeId: NodeId) (value: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 0L [value] ty

/// Result.Error: tag=1 + error
/// Returns: (ops, result) where result is the constructed Result<'T,'E> value
let pResultError (nodeId: NodeId) (error: Val) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    pDUCase nodeId 1L [error] ty
