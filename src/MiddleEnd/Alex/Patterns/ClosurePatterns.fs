/// ClosurePatterns - Closure and lambda operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for lambda and closure operations.
/// Patterns compose Elements into semantic closure/lambda operations.
module Alex.Patterns.ClosurePatterns

open XParsec
open XParsec.Parsers     // preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Elements.MLIRElements
open Alex.Elements.MemRefElements
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements
open Alex.Elements.FuncElements

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// ARENA ALLOCATION
// ═══════════════════════════════════════════════════════════

/// Allocate in global closure heap arena (bump allocator)
/// SSAs: [0] = heap_pos_ptr, [1] = heap_pos, [2] = heap_base, [3] = result_ptr, [4] = new_pos
/// Returns ops and the result pointer SSA
let pAllocateInArena (sizeSSA: SSA) (ssas: SSA list) : PSGParser<MLIROp list * SSA> =
    parser {
        do! ensure (ssas.Length >= 5) $"pAllocateInArena: Expected 5 SSAs, got {ssas.Length}"

        let heapPosPtrSSA = ssas.[0]
        let heapPosSSA = ssas.[1]
        let heapBaseSSA = ssas.[2]
        let resultPtrSSA = ssas.[3]
        let newPosSSA = ssas.[4]

        // Load current position
        let! addressOfPosOp = pLoad heapPosPtrSSA heapPosPtrSSA  // Placeholder - need AddressOf
        let! loadPosOp = pLoad heapPosSSA heapPosPtrSSA

        // Compute result pointer: heap_base + pos
        let! addressOfBaseOp = pLoad heapBaseSSA heapBaseSSA  // Placeholder - need AddressOf
        let! gepOp = pGEP resultPtrSSA heapBaseSSA [(heapPosSSA, TInt I64)]

        // Update position: pos + size
        let! addOp = pAddI newPosSSA heapPosSSA sizeSSA
        let! storePosOp = pStore newPosSSA heapPosPtrSSA [] (TInt I64)

        return ([addressOfPosOp; loadPosOp; addressOfBaseOp; gepOp; addOp; storePosOp], resultPtrSSA)
    }

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

/// Create function definition (func.func for named calls, llvm.func for closures)
/// isClosure: true for llvm.func (address taken), false for func.func (named calls)
let pFunctionDef (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                 (body: MLIROp list) (isClosure: bool) : PSGParser<MLIROp> =
    parser {
        // For now, always use func.func
        // The distinction between func.func and llvm.func will be handled by the witness
        let visibility = if name = "main" then FuncVisibility.Public else FuncVisibility.Private
        return! pFuncDef name params' retTy body visibility
    }

// ═══════════════════════════════════════════════════════════
// CAPTURE EXTRACTION
// ═══════════════════════════════════════════════════════════

/// Extract captures from closure struct at function entry
/// Arg 0 is env_ptr, load struct, extract captures at baseIndex + slotIndex
/// SSAs: [0] = struct load, [1..N] = extracted captures
let pExtractCaptures (baseIndex: int) (captureTypes: MLIRType list) (structType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let captureCount = captureTypes.Length
        do! ensure (ssas.Length >= captureCount + 1) $"pExtractCaptures: Expected {captureCount + 1} SSAs, got {ssas.Length}"

        let structLoadSSA = ssas.[0]
        let envPtrSSA = Arg 0  // First argument is always env_ptr for closures

        // Load struct from env_ptr
        let! loadOp = pLoad structLoadSSA envPtrSSA

        // Extract each capture
        let! extractOps =
            captureTypes
            |> List.mapi (fun i capTy ->
                parser {
                    let extractSSA = ssas.[i + 1]
                    let extractIndex = baseIndex + i
                    let! extractOp = pExtractValue extractSSA structLoadSSA [extractIndex] capTy
                    return extractOp
                })
            |> sequence

        return loadOp :: extractOps
    }

// ═══════════════════════════════════════════════════════════
// LAMBDA CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// Build simple lambda (no captures) - just creates function definition
/// Returns: (topLevelOps, inlineOps, resultSSA)
/// - topLevelOps: function definition
/// - inlineOps: empty (no struct to construct)
/// - resultSSA: function reference
let pBuildSimpleLambda (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                       (bodyOps: MLIROp list) (funcRefSSA: SSA) : PSGParser<MLIROp list * MLIROp list * SSA> =
    parser {
        let! funcDefOp = pFunctionDef name params' retTy bodyOps false
        return [funcDefOp], [], funcRefSSA
    }

/// Build closure lambda (with captures) - creates function definition + closure struct
/// Returns: (topLevelOps, inlineOps, resultSSA)
/// - topLevelOps: function definition
/// - inlineOps: closure struct construction (via pFlatClosure)
/// - resultSSA: closure struct value
let pBuildClosureLambda (name: string) (params': (SSA * MLIRType) list) (retTy: MLIRType)
                        (bodyOps: MLIROp list) (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list)
                        (ssas: SSA list) : PSGParser<MLIROp list * MLIROp list * SSA> =
    parser {
        // Create function definition (with env_ptr as first parameter for captures)
        let envPtrParam = Arg 0, TPtr
        let allParams = envPtrParam :: params'
        let! funcDefOp = pFunctionDef name allParams retTy bodyOps true

        // Create closure struct (delegates to pFlatClosure from ElisionPatterns)
        let! structOps = ElisionPatterns.pFlatClosure codePtr codePtrTy captures ssas

        return [funcDefOp], structOps, List.last ssas
    }
