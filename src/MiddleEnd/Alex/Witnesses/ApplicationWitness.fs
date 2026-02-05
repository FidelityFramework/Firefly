/// ApplicationWitness - Witness function application nodes
///
/// Application nodes emit MLIR function calls (direct or indirect).
/// Post-order traversal ensures function and arguments are already witnessed.
///
/// NANOPASS: This witness handles ONLY Application nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Pure XParsec monadic observation.
/// ALL SSA extractions via patterns (NodeId-based API) - ZERO imperative lookups.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns
open Alex.Patterns.PlatformPatterns
open Alex.Patterns.ApplicationPatterns
open Alex.Patterns.LiteralPatterns
open Alex.Patterns.StringPatterns
open Alex.CodeGeneration.TypeMapping

// NOTE: Elements are module internal - witnesses delegate to Patterns, NOT Elements

// ═══════════════════════════════════════════════════════════
// MUTABLE VARIABLE LOAD HELPER
// ═══════════════════════════════════════════════════════════

/// Load value from TMemRef argument if needed
/// Returns (newSSA, valueType, loadOps) where:
/// - TMemRef: emits memref.load, returns (loadedSSA, elemType, [loadOp])
/// - Other: returns (originalSSA, originalType, [])
let private loadIfMemRef (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) (ctx: WitnessContext) : (SSA * MLIRType * MLIROp list) =
    match ty with
    | TMemRef elemType ->
        // Emit memref.load to get the value
        let (NodeId nodeIdInt) = nodeId
        match tryMatch (Alex.Patterns.MemRefPatterns.pLoadMutableVariable nodeIdInt ssa elemType)
                      ctx.Graph (SemanticGraph.tryGetNode nodeId ctx.Graph |> Option.get)
                      ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((ops, TRValue result), _) ->
            (result.SSA, result.Type, ops)
        | Some ((ops, _), _) ->
            // Shouldn't happen - pLoadMutableVariable always returns TRValue
            (ssa, ty, ops)
        | None ->
            // Load pattern failed - return original (error will propagate)
            (ssa, ty, [])
    | _ ->
        // Not a memref - return as-is
        (ssa, ty, [])

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Helper: Navigate to actual function node (unwrap TypeAnnotation if present)
let private resolveFunctionNode funcId graph =
    match SemanticGraph.tryGetNode funcId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.TypeAnnotation (innerFuncId, _) ->
            // Unwrap TypeAnnotation to get actual function
            SemanticGraph.tryGetNode innerFuncId graph
        | _ -> Some funcNode
    | None -> None

/// Witness application nodes - emits function calls
let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((funcId, argIds), _) ->
        // Resolve actual function node (unwrap TypeAnnotation if present)
        match resolveFunctionNode funcId ctx.Graph with
        | Some funcNode when funcNode.Kind.ToString().StartsWith("Intrinsic") ->
            // Atomic operation marked as Intrinsic in FNCS - dispatch based on module and operation
            match funcNode.Kind with
            | SemanticKind.Intrinsic info ->
                // Recall argument SSAs
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    let unwitnessedArgs =
                        List.zip argIds argsResult
                        |> List.filter (fun (_, result) -> Option.isNone result)
                        |> List.map fst
                    WitnessOutput.error $"Application node {node.Id}: Atomic operation {info.FullName} arguments not yet witnessed: {unwitnessedArgs}"
                else
                    let argSSAs = argsResult |> List.choose id |> List.map fst
                    let argTypes = argsResult |> List.choose id |> List.map snd  // Extract types for comparisons

                    // Dispatch based on intrinsic module and operation
                    match info.Module, info.Operation, argSSAs with
                    // MemRef operations (MLIR memref semantics)
                    // These are the TARGET operations created by Baker's NativePtr transformation.
                    // Pure memref semantics - NO NativePtr, NO i64↔index casts, NO LLVM cruft.

                    | IntrinsicModule.MemRef, "alloca", [countSSA] ->
                        // MemRef.alloca: nativeint -> memref<?x'T>
                        // Baker has already transformed NativePtr.stackalloc → MemRef.alloca
                        let countNodeId = argIds.[0]
                        match SemanticGraph.tryGetNode countNodeId ctx.Graph with
                        | Some countNode ->
                            match countNode.Kind with
                            | SemanticKind.Literal (NativeLiteral.Int (value, _)) ->
                                let count = int value
                                // Pattern extracts result SSA monadically via node.Id
                                match tryMatch (pNativePtrStackAlloc node.Id count) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "MemRef.alloca pattern failed"
                            | _ -> WitnessOutput.error $"MemRef.alloca: count must be a literal (node {countNodeId})"
                        | None -> WitnessOutput.error $"MemRef.alloca: could not resolve count argument (node {countNodeId})"

                    | IntrinsicModule.MemRef, "store", [valueSSA; ptrSSA; indexSSA] ->
                        // MemRef.store: 'T -> memref<?x'T> -> nativeint -> unit
                        // Baker has already transformed NativePtr.write → MemRef.store
                        // PATTERN: Detect and unwrap MemRef.add(base, offset) to extract base memref and offset
                        let (memrefSSA, offsetSSA, memrefType, offsetLoadOps) =
                            let ptrArgNodeId = argIds.[1]  // Second argument (ptr position)
                            match SemanticGraph.tryGetNode ptrArgNodeId ctx.Graph with
                            | Some ptrNode ->
                                match ptrNode.Kind with
                                | SemanticKind.Application (funcId, addArgIds) ->
                                    // Check if this is calling MemRef.add
                                    match SemanticGraph.tryGetNode funcId ctx.Graph with
                                    | Some funcNode ->
                                        match funcNode.Kind with
                                        | SemanticKind.Intrinsic info when info.Module = IntrinsicModule.MemRef && info.Operation = "add" ->
                                            // Pattern matched: MemRef.add(base, offset)
                                            // Extract base memref and offset from MemRef.add arguments
                                            match MLIRAccumulator.recallNode addArgIds.[0] ctx.Accumulator,
                                                  MLIRAccumulator.recallNode addArgIds.[1] ctx.Accumulator with
                                            | Some (baseSSA, baseTy), Some (offsetSSA, offsetTy) ->
                                                // Check if offset is TMemRef (mutable variable) and needs loading
                                                match offsetTy with
                                                | TMemRef elemType ->
                                                    // Offset is mutable variable - emit load operations
                                                    // VarRef node (addArgIds.[1]) has 2 SSAs: [0] = index const, [1] = load result
                                                    let offsetNodeIdValue = NodeId.value addArgIds.[1]
                                                    match ctx.Coeffects.SSA.NodeSSA.TryFind(offsetNodeIdValue) with
                                                    | Some alloc when alloc.SSAs.Length >= 2 ->
                                                        let ssas = alloc.SSAs
                                                        let zeroSSA = ssas.[0]
                                                        let loadedOffsetSSA = ssas.[1]

                                                        // Compose load operations (same pattern as pRecallArgWithLoad)
                                                        let zeroOp = MLIROp.IndexOp (IndexOp.IndexConst (zeroSSA, 0L))
                                                        let loadOp = MLIROp.MemRefOp (MemRefOp.Load (loadedOffsetSSA, offsetSSA, [zeroSSA], elemType))
                                                        let loadOps = [zeroOp; loadOp]

                                                        // Use loaded value as offset
                                                        (baseSSA, loadedOffsetSSA, baseTy, loadOps)
                                                    | _ ->
                                                        // SSAs not found or insufficient - use offsetSSA as-is
                                                        // This handles non-VarRef cases (literals, computed values)
                                                        (baseSSA, offsetSSA, baseTy, [])
                                                | _ ->
                                                    // Offset is not TMemRef - use as-is (immutable or literal)
                                                    (baseSSA, offsetSSA, baseTy, [])
                                            | _ ->
                                                // Recall failed, fallback to original values
                                                (ptrSSA, indexSSA, argTypes.[1], [])
                                        | _ ->
                                            // Not MemRef.add, use original values
                                            (ptrSSA, indexSSA, argTypes.[1], [])
                                    | None -> (ptrSSA, indexSSA, argTypes.[1], [])
                                | _ ->
                                    // Not an Application, use original values
                                    (ptrSSA, indexSSA, argTypes.[1], [])
                            | None -> (ptrSSA, indexSSA, argTypes.[1], [])

                        let arch = ctx.Coeffects.Platform.TargetArch
                        let elemType =
                            match memrefType with
                            | TMemRef elemTy -> elemTy
                            | _ -> TError "Expected memref type for MemRef.store"
                        match tryMatch (pMemRefStoreIndexed memrefSSA valueSSA offsetSSA elemType memrefType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((storeOps, result), _) ->
                            // Prepend offset load operations (if any) to store operations
                            { InlineOps = offsetLoadOps @ storeOps; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "MemRef.store pattern failed"

                    | IntrinsicModule.MemRef, "load", [memrefSSA; indexSSA] ->
                        // MemRef.load: memref<?x'T> -> index -> 'T
                        // Baker has already transformed NativePtr.read → MemRef.load
                        // Uses MLIR memref.load operation (NOT LLVM pointer load)
                        match tryMatch (pMemRefLoad node.Id memrefSSA indexSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "MemRef.load pattern failed"

                    | IntrinsicModule.MemRef, "copy", [destSSA; srcSSA; countSSA] ->
                        // MemRef.copy: memref -> memref -> nativeint -> unit
                        // Baker has already transformed NativePtr.copy → MemRef.copy
                        // Emits call to memcpy library function (no result SSA needed - returns TRVoid)
                        match tryMatch (pMemCopy destSSA srcSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "MemRef.copy pattern failed"

                    | IntrinsicModule.MemRef, "add", [_basePtrSSA; offsetSSA] ->
                        // MemRef.add is a MARKER operation from Baker transformation
                        // MLIR memref ops take indices directly (no getelementptr equivalent)
                        // This marker returns the OFFSET/INDEX for downstream memref.store/load
                        // Base pointer is discarded; offset flows as index parameter
                        { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = offsetSSA; Type = TIndex } }

                    // Arena operations (F-02: Arena Allocation)
                    | IntrinsicModule.Arena, "create", [sizeSSA] ->
                        // Arena.create: int -> Arena<'lifetime>
                        // Creates stack-allocated byte buffer (memref<N x i8>)
                        let sizeNodeId = argIds.[0]
                        match SemanticGraph.tryGetNode sizeNodeId ctx.Graph with
                        | Some sizeNode ->
                            match sizeNode.Kind with
                            | SemanticKind.Literal (NativeLiteral.Int (value, _)) ->
                                let sizeBytes = int value
                                // Pattern extracts result SSA monadically via node.Id
                                match tryMatch (pArenaCreate node.Id sizeBytes) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "Arena.create pattern failed"
                            | _ -> WitnessOutput.error $"Arena.create: size must be a literal int (node {sizeNodeId})"
                        | None -> WitnessOutput.error $"Arena.create: could not resolve size argument (node {sizeNodeId})"

                    | IntrinsicModule.Arena, "alloc", [arenaSSA; sizeSSA] ->
                        // Arena.alloc: Arena<'lifetime> byref -> int -> nativeint
                        // Allocates from arena (simplified: returns arena memref)
                        // Get arena type from argument
                        let arenaType = argTypes.[0]
                        // Pattern extracts result SSA monadically via node.Id
                        match tryMatch (pArenaAlloc node.Id arenaSSA sizeSSA arenaType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Arena.alloc pattern failed"

                    | IntrinsicModule.Sys, "write", [fdSSA; bufferSSA] ->
                        // Witness observes buffer type
                        // Pattern extracts 6 SSAs monadically for FFI pointer + length extraction
                        let bufferNodeId = argIds.[1]
                        match MLIRAccumulator.recallNode bufferNodeId ctx.Accumulator with
                        | Some (_, bufferType) ->
                            // Pattern extracts SSAs monadically via node.Id (6 SSAs)
                            match tryMatch (pSysWrite node.Id fdSSA bufferSSA bufferType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "Sys.write pattern failed"
                        | None -> WitnessOutput.error "Sys.write: buffer argument not yet witnessed"

                    | IntrinsicModule.Sys, "read", [fdSSA; bufferSSA] ->
                        // Witness observes buffer type
                        // Pattern extracts 6 SSAs monadically for FFI pointer + capacity extraction
                        let bufferNodeId = argIds.[1]
                        match MLIRAccumulator.recallNode bufferNodeId ctx.Accumulator with
                        | Some (_, bufferType) ->
                            // Pattern extracts SSAs monadically via node.Id (6 SSAs)
                            match tryMatch (pSysRead node.Id fdSSA bufferSSA bufferType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "Sys.read pattern failed"
                        | None -> WitnessOutput.error "Sys.read: buffer argument not yet witnessed"

                    | IntrinsicModule.NativeStr, "fromPointer", [bufferSSA; lengthSSA] ->
                        // FNCS contract (Intrinsics.fs:372): "creates a new memref<?xi8> with specified length"
                        // This requires allocate + memcpy to create substring, not just cast
                        // Pattern: Check if length is TMemRef (mutable variable) and load if needed
                        let bufferType = argTypes.[0]
                        let lengthType = argTypes.[1]

                        // Check if length needs loading (TMemRef case - mutable variable)
                        let (actualLengthSSA, lengthLoadOps) =
                            match lengthType with
                            | TMemRef elemType ->
                                // Length is mutable variable - compose load operations
                                // VarRef node has 2 SSAs: [0] = index const, [1] = load result
                                let lengthArgNodeId = argIds.[1]
                                let lengthNodeIdValue = NodeId.value lengthArgNodeId
                                match ctx.Coeffects.SSA.NodeSSA.TryFind(lengthNodeIdValue) with
                                | Some alloc when alloc.SSAs.Length >= 2 ->
                                    let ssas = alloc.SSAs
                                    let zeroSSA = ssas.[0]
                                    let loadedLengthSSA = ssas.[1]

                                    // Compose load operations (same pattern as MemRef.store offset)
                                    let zeroOp = MLIROp.IndexOp (IndexOp.IndexConst (zeroSSA, 0L))
                                    let loadOp = MLIROp.MemRefOp (MemRefOp.Load (loadedLengthSSA, lengthSSA, [zeroSSA], elemType))
                                    let loadOps = [zeroOp; loadOp]

                                    (loadedLengthSSA, loadOps)
                                | _ ->
                                    // SSAs not found - use lengthSSA as-is
                                    (lengthSSA, [])
                            | _ ->
                                // Length is not TMemRef - use as-is
                                (lengthSSA, [])

                        // Call pattern with loaded (or direct) length - honors FNCS contract
                        match tryMatch (pStringFromPointerWithLength node.Id bufferSSA actualLengthSSA bufferType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) ->
                            // Prepend length load operations (if any) to string construction operations
                            { InlineOps = lengthLoadOps @ ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "NativeStr.fromPointer: substring construction failed"

                    // String atomic intrinsics (not decomposed by Baker)
                    | IntrinsicModule.String, "length", [stringSSA] ->
                        // String.length: memref.dim to get string dimension
                        // Strings ARE memrefs, length is intrinsic to descriptor
                        // Pattern extracts 3 SSAs monadically via node.Id
                        let stringType = argTypes.[0]
                        match tryMatch (pStringLength node.Id stringSSA stringType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "String.length pattern failed"

                    | IntrinsicModule.String, "concat2", [str1SSA; str2SSA] ->
                        // String.concat2: pure memref operations with index arithmetic
                        // Generates: memref.dim + arith.addi (index) + memref.alloc + memcpy
                        // NO i64 round-trip - pure index throughout
                        // Pattern extracts 18 SSAs monadically via node.Id
                        let str1Type = argTypes.[0]
                        let str2Type = argTypes.[1]
                        match tryMatch (pStringConcat2 node.Id str1SSA str2SSA str1Type str2Type) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "String.concat2 pattern failed"

                    // Other complex string operations handled by platform libraries

                    // Binary arithmetic intrinsics (+, -, *, /, %)
                    | IntrinsicModule.Operators, _, [lhsSSA; rhsSSA] ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let resultType = mapNativeTypeForArch arch node.Type

                        // For comparisons, we need operand type (not result type which is always i1)
                        let operandType =
                            match argTypes with
                            | operandTy :: _ -> operandTy  // Type of first operand
                            | [] -> resultType  // Fallback (shouldn't happen for binary ops)

                        // Use classification from PSGCombinators for principled operator dispatch
                        let category = classifyAtomicOp info
                        let opResult =
                            match category with
                            // Binary arithmetic (PULL model) - witness passes minimal selector, pattern pulls args
                            | BinaryArith "addi" -> tryMatch (pBinaryArithOp node.Id "addi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "subi" -> tryMatch (pBinaryArithOp node.Id "subi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "muli" -> tryMatch (pBinaryArithOp node.Id "muli") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "divsi" -> tryMatch (pBinaryArithOp node.Id "divsi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "divui" -> tryMatch (pBinaryArithOp node.Id "divui") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "remsi" -> tryMatch (pBinaryArithOp node.Id "remsi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "remui" -> tryMatch (pBinaryArithOp node.Id "remui") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                            // Floating-point arithmetic (PULL model)
                            | BinaryArith "addf" -> tryMatch (pBinaryArithOp node.Id "addf") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "subf" -> tryMatch (pBinaryArithOp node.Id "subf") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "mulf" -> tryMatch (pBinaryArithOp node.Id "mulf") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "divf" -> tryMatch (pBinaryArithOp node.Id "divf") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                            // Bitwise operations (PULL model)
                            | BinaryArith "andi" -> tryMatch (pBinaryArithOp node.Id "andi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "ori" -> tryMatch (pBinaryArithOp node.Id "ori") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "xori" -> tryMatch (pBinaryArithOp node.Id "xori") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "shli" -> tryMatch (pBinaryArithOp node.Id "shli") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "shrui" -> tryMatch (pBinaryArithOp node.Id "shrui") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | BinaryArith "shrsi" -> tryMatch (pBinaryArithOp node.Id "shrsi") ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                            // Comparison operations (PULL model) - pattern extracts operands and types
                            | Comparison "eq" -> tryMatch (pComparisonOp node.Id ICmpPred.Eq) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "ne" -> tryMatch (pComparisonOp node.Id ICmpPred.Ne) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "slt" -> tryMatch (pComparisonOp node.Id ICmpPred.Slt) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "sle" -> tryMatch (pComparisonOp node.Id ICmpPred.Sle) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "sgt" -> tryMatch (pComparisonOp node.Id ICmpPred.Sgt) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "sge" -> tryMatch (pComparisonOp node.Id ICmpPred.Sge) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "ult" -> tryMatch (pComparisonOp node.Id ICmpPred.Ult) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "ule" -> tryMatch (pComparisonOp node.Id ICmpPred.Ule) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "ugt" -> tryMatch (pComparisonOp node.Id ICmpPred.Ugt) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                            | Comparison "uge" -> tryMatch (pComparisonOp node.Id ICmpPred.Uge) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                            | _ -> None

                        match opResult with
                        | Some ((ops, result), _) ->
                            { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None ->
                            WitnessOutput.error $"Unknown or unsupported binary operator: {info.Operation} (category: {category})"

                    // Unary operators (single operand) - use wrapper pattern
                    | IntrinsicModule.Operators, _, [operandSSA] ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let resultType = mapNativeTypeForArch arch node.Type

                        // Use classification from PSGCombinators for principled operator dispatch
                        let category = classifyAtomicOp info
                        match category with
                        // Unary operations (PULL model) - witness passes minimal selector
                        | UnaryArith "xori" ->
                            // Boolean NOT: xori %operand, 1
                            // Pattern extracts operand monadically and pulls from accumulator
                            // Use tryMatchWithDiagnostics to surface actual pattern failure reasons
                            match tryMatchWithDiagnostics (pUnaryNot node.Id) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Result.Ok ((ops, result), _) ->
                                { InlineOps = ops; TopLevelOps = []; Result = result }
                            | Result.Error msg ->
                                // Surface architectural error from pattern (e.g., "Operand is TMemRef")
                                WitnessOutput.error $"UnaryArith 'xori' (not): {msg}"

                        | _ ->
                            WitnessOutput.error $"Unknown or unsupported unary operator: {info.Operation} (category: {category})"

                    | _ -> WitnessOutput.error $"Atomic operation not yet implemented: {info.FullName} with {argSSAs.Length} args"

            | _ -> WitnessOutput.error "Expected SemanticKind.Intrinsic from FNCS"

        | Some funcNode when funcNode.Kind.ToString().StartsWith("VarRef") ->
            // Extract qualified function name using PSG resolution (not string parsing!)
            // VarRef has: (localName, Some definitionNodeId) where definitionNodeId points to the Binding
            // We follow the resolution to get the fully qualified name (e.g., "Console.write")
            let funcName =
                match funcNode.Kind with
                | SemanticKind.VarRef (localName, Some defId) ->
                    // Follow resolution to binding node
                    match SemanticGraph.tryGetNode defId ctx.Graph with
                    | Some bindingNode ->
                        match bindingNode.Kind with
                        | SemanticKind.Binding (bindName, _, _, _) ->
                            // Check if binding has a module parent (ModuleDef)
                            match bindingNode.Parent with
                            | Some parentId ->
                                match SemanticGraph.tryGetNode parentId ctx.Graph with
                                | Some parentNode ->
                                    match parentNode.Kind with
                                    | SemanticKind.ModuleDef (moduleName, _) ->
                                        // Qualified name: Module.Function
                                        sprintf "%s.%s" moduleName bindName
                                    | _ -> bindName  // No module parent, use binding name
                                | None -> bindName
                            | None -> bindName
                        | _ -> localName  // Not a binding, use local name
                    | None -> localName  // Resolution failed, use local name
                | SemanticKind.VarRef (localName, None) ->
                    localName  // Unresolved reference, use local name
                | _ -> "unknown_func"  // Not a VarRef (shouldn't happen given guard above)

            // Recall argument SSAs with types
            let argsResult =
                argIds
                |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

            // Ensure all arguments were witnessed
            let allWitnessed = argsResult |> List.forall Option.isSome
            if not allWitnessed then
                WitnessOutput.error "Application: Some arguments not yet witnessed"
            else
                // Keep both SSA and type for each argument
                let args = argsResult |> List.choose id

                // Get result type
                let arch = ctx.Coeffects.Platform.TargetArch
                let retType = mapNativeTypeForArch arch node.Type

                // Emit direct function call by name (no declaration coordination)
                // Declaration will be collected and emitted by MLIR Declaration Collection Pass
                // Pattern extracts result SSA monadically via node.Id
                match tryMatch (pDirectCall node.Id funcName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "Direct function call pattern emission failed"

        | _ ->
            // Function is an SSA value (indirect call)
            match MLIRAccumulator.recallNode funcId ctx.Accumulator with
            | None -> WitnessOutput.error "Application: Function not yet witnessed"
            | Some (funcSSA, funcTy) ->
                // Recall argument SSAs with types
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                // Ensure all arguments were witnessed
                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    WitnessOutput.error "Application: Some arguments not yet witnessed"
                else
                    // Keep both SSA and type for each argument
                    let args = argsResult |> List.choose id

                    // Get result type
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type

                    // Emit indirect call
                    // Pattern extracts result SSA monadically via node.Id
                    match tryMatch (pApplicationCall node.Id funcSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Application pattern emission failed"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Application nanopass - witnesses function applications
let nanopass : Nanopass = {
    Name = "Application"
    Witness = witnessApplication
}
