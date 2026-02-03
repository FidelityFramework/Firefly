/// CollectionPatterns - Immutable collection patterns via DU construction
///
/// PUBLIC: Witnesses use these to emit Option, List, Map, Set, Result operations.
/// All collections are implemented as discriminated unions with tag-based dispatch.
module Alex.Patterns.CollectionPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Elements.MLIRAtomics  // pExtractValue, pConstI
open Alex.Elements.ArithElements  // pCmpI
open Alex.Patterns.MemoryPatterns  // pDUCase - DU construction foundation
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// ═══════════════════════════════════════════════════════════
// OPTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Option.Some: tag=1 + value
let pOptionSome (value: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [value] ssas ty

/// Option.None: tag=0
let pOptionNone (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Option.IsSome: extract tag, compare with 1
let pOptionIsSome (optionSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA optionSSA [0] tagTy
        let! oneConstOp = pConstI oneSSA 1L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA oneSSA
        return [extractTagOp; oneConstOp; cmpOp]
    }

/// Option.IsNone: extract tag, compare with 0
let pOptionIsNone (optionSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA optionSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Option.Get: extract value field (index 1)
let pOptionGet (optionSSA: SSA) (resultSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        let! extractOp = pExtractValue resultSSA optionSSA [1] valueTy
        return [extractOp]
    }

// ═══════════════════════════════════════════════════════════
// LIST PATTERNS
// ═══════════════════════════════════════════════════════════

/// List.Cons: tag=1 + head + tail
let pListCons (head: Val) (tail: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [head; tail] ssas ty

/// List.Empty: tag=0
let pListEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// List.IsEmpty: extract tag, compare with 0
let pListIsEmpty (listSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA listSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

// ═══════════════════════════════════════════════════════════
// MAP PATTERNS
// ═══════════════════════════════════════════════════════════

/// Map.Empty: empty tree structure (tag=0 for leaf node)
let pMapEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Map.Add: create new tree node with key, value, left, right subtrees
/// Structure: tag=1, key, value, left_tree, right_tree
let pMapAdd (key: Val) (value: Val) (left: Val) (right: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [key; value; left; right] ssas ty

/// Map.ContainsKey: tree traversal comparing keys
/// This is a composite operation that would be implemented as a recursive function
/// The pattern here just shows the key comparison at each node
let pMapKeyCompare (mapNodeSSA: SSA) (searchKey: SSA) (keySSA: SSA) (keyTy: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract key from node (index 1, after tag at index 0)
        let! extractKeyOp = pExtractValue keySSA mapNodeSSA [1] keyTy
        // Compare search key with node key
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq searchKey keySSA
        return [extractKeyOp; cmpOp]
    }

/// Map.TryFind: similar to ContainsKey but returns Option<value>
let pMapExtractValue (mapNodeSSA: SSA) (valueSSA: SSA) (valueTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        // Extract value from node (index 2, after tag and key)
        let! extractValueOp = pExtractValue valueSSA mapNodeSSA [2] valueTy
        return [extractValueOp]
    }

// ═══════════════════════════════════════════════════════════
// SET PATTERNS
// ═══════════════════════════════════════════════════════════

/// Set.Empty: empty tree structure (tag=0 for leaf node)
let pSetEmpty (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [] ssas ty

/// Set.Add: create new tree node with element, left, right subtrees
/// Structure: tag=1, element, left_tree, right_tree
let pSetAdd (element: Val) (left: Val) (right: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [element; left; right] ssas ty

/// Set.Contains: tree traversal comparing elements
let pSetElementCompare (setNodeSSA: SSA) (searchElem: SSA) (elemSSA: SSA) (elemTy: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Extract element from node (index 1, after tag at index 0)
        let! extractElemOp = pExtractValue elemSSA setNodeSSA [1] elemTy
        // Compare search element with node element
        let! cmpOp = pCmpI resultSSA ICmpPred.Eq searchElem elemSSA
        return [extractElemOp; cmpOp]
    }

/// Set.Union: combines two sets (implemented as tree merge operation)
/// Pattern shows extraction of subtrees for recursive union
let pSetExtractSubtrees (setNodeSSA: SSA) (leftSSA: SSA) (rightSSA: SSA) (subtreeTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        // Extract left subtree (index 2)
        let! extractLeftOp = pExtractValue leftSSA setNodeSSA [2] subtreeTy
        // Extract right subtree (index 3)
        let! extractRightOp = pExtractValue rightSSA setNodeSSA [3] subtreeTy
        return [extractLeftOp; extractRightOp]
    }

// ═══════════════════════════════════════════════════════════
// RESULT PATTERNS
// ═══════════════════════════════════════════════════════════

/// Result.Ok: tag=0 + value
let pResultOk (value: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 0L [value] ssas ty

/// Result.Error: tag=1 + error
let pResultError (error: Val) (ssas: SSA list) (ty: MLIRType) : PSGParser<MLIROp list> =
    pDUCase 1L [error] ssas ty

/// Result.IsOk: extract tag, compare with 0
let pResultIsOk (resultSSA: SSA) (tagSSA: SSA) (zeroSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA resultSSA [0] tagTy
        let! zeroConstOp = pConstI zeroSSA 0L tagTy
        let! cmpOp = pCmpI cmpSSA ICmpPred.Eq tagSSA zeroSSA
        return [extractTagOp; zeroConstOp; cmpOp]
    }

/// Result.IsError: extract tag, compare with 1
let pResultIsError (resultSSA: SSA) (tagSSA: SSA) (oneSSA: SSA) (cmpSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let tagTy = TInt I8  // DU tags are always i8
        let! extractTagOp = pExtractValue tagSSA resultSSA [0] tagTy
        let! oneConstOp = pConstI oneSSA 1L tagTy
        let! cmpOp = pCmpI cmpSSA ICmpPred.Eq tagSSA oneSSA
        return [extractTagOp; oneConstOp; cmpOp]
    }
