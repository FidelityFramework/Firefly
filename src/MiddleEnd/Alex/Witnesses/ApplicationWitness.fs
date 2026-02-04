/// ApplicationWitness - Witness function application nodes
///
/// Application nodes emit MLIR function calls (direct or indirect).
/// Post-order traversal ensures function and arguments are already witnessed.
///
/// NANOPASS: This witness handles ONLY Application nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
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
open Alex.Elements.ArithElements
open Alex.CodeGeneration.TypeMapping

module SSAAssign = PSGElaboration.SSAAssignment

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

                    // Get result SSA
                    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error $"Application: No SSA for atomic operation {info.FullName}"
                    | Some resultSSA ->
                        // Dispatch based on intrinsic module and operation
                        match info.Module, info.Operation, argSSAs with
                        | IntrinsicModule.NativePtr, "stackalloc", [countSSA] ->
                            match tryMatch (pNativePtrStackAlloc resultSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.stackalloc pattern failed"

                        | IntrinsicModule.NativePtr, "write", [ptrSSA; valueSSA] ->
                            // NativePtr.write: ptr -> value -> unit
                            // Emit memref.store with empty indices (scalar write)
                            let ptrNodeId = argIds.[0]
                            match SemanticGraph.tryGetNode ptrNodeId ctx.Graph with
                            | Some ptrNode ->
                                match ptrNode.Type with
                                | NativeType.TNativePtr innerTy ->
                                    let arch = ctx.Coeffects.Platform.TargetArch
                                    let elemType = mapNativeTypeForArch arch innerTy
                                    match tryMatch (pMemRefStore valueSSA ptrSSA elemType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                    | None -> WitnessOutput.error "NativePtr.write: memref.store failed"
                                | otherTy ->
                                    WitnessOutput.error (sprintf "NativePtr.write: Expected nativeptr<'T>, got %A" otherTy)
                            | None -> WitnessOutput.error "NativePtr.write: Could not resolve pointer argument node"

                        | IntrinsicModule.NativePtr, "add", [basePtrSSA; offsetSSA] ->
                            // NativePtr.add(ptr, offset) returns offset as index for later use with write/read
                            // We just return the offset SSA - NativePtr.write/read will use it as the index
                            // Record the result as the offset (for compound pattern with write/read)
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = offsetSSA; Type = TIndex } }

                        | IntrinsicModule.NativePtr, "copy", [destSSA; srcSSA; countSSA] ->
                            // NativePtr.copy(dest, src, count) - memory copy operation
                            // Witness the coeffect-assigned resultSSA for the memcpy MLIR operation
                            match tryMatch (pMemCopy resultSSA destSSA srcSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some (ops, _) ->
                                // memcpy returns dest pointer, but NativePtr.copy returns unit
                                // So we return void result
                                { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
                            | None -> WitnessOutput.error "NativePtr.copy pattern failed"

                        | IntrinsicModule.NativePtr, "read", [ptrSSA] ->
                            // NativePtr.read needs an index SSA for memref.load - get additional SSA
                            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                            | Some ssas when ssas.Length >= 2 ->
                                let indexZeroSSA = ssas.[0]
                                let readResultSSA = ssas.[1]
                                match tryMatch (pNativePtrRead readResultSSA ptrSSA indexZeroSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "NativePtr.read pattern failed"
                            | _ -> WitnessOutput.error "NativePtr.read: Need 2 SSAs (index, result)"

                        | IntrinsicModule.Sys, "write", [fdSSA; bufferSSA; countSSA] ->
                            // Witness observes actual buffer type (memref stays as memref)
                            let bufferNodeId = argIds.[1]
                            match MLIRAccumulator.recallNode bufferNodeId ctx.Accumulator with
                            | Some (_, bufferType) ->
                                // Pass buffer as-is - LLVM handles memref lowering
                                match tryMatch (pSysWrite resultSSA fdSSA bufferSSA bufferType countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "Sys.write pattern failed"
                            | None -> WitnessOutput.error "Sys.write: buffer argument not yet witnessed"

                        | IntrinsicModule.Sys, "read", [fdSSA; bufferSSA; countSSA] ->
                            // Witness observes actual buffer type (memref stays as memref)
                            let bufferNodeId = argIds.[1]
                            match MLIRAccumulator.recallNode bufferNodeId ctx.Accumulator with
                            | Some (_, bufferType) ->
                                // Pass buffer as-is - LLVM handles memref lowering
                                match tryMatch (pSysRead resultSSA fdSSA bufferSSA bufferType countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "Sys.read pattern failed"
                            | None -> WitnessOutput.error "Sys.read: buffer argument not yet witnessed"

                        | IntrinsicModule.NativeStr, "fromPointer", [ptrSSA; lengthSSA] ->
                            // NativeStr.fromPointer(ptr, len) constructs a string fat pointer
                            // String is struct {ptr: nativeptr<byte>, length: int}
                            let arch = ctx.Coeffects.Platform.TargetArch
                            let ptrTy = TIndex  // Pointer represented as index
                            let lengthTy = mapNativeTypeForArch arch Types.intType  // Use platform int

                            // Witness SSAs for struct construction (undef, insertPtr, result)
                            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                            | Some ssas when ssas.Length >= 3 ->
                                match tryMatch (pStringConstruct ptrTy lengthTy ptrSSA lengthSSA ssas) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                | Some (ops, _) ->
                                    // String is represented as a fat pointer struct (index + i32)
                                    let stringSize = if arch = Architecture.X86_64 then 16 else 12
                                    let stringTy = TMemRefStatic(stringSize, TInt I8)
                                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = stringTy } }
                                | None -> WitnessOutput.error "NativeStr.fromPointer pattern failed"
                            | _ -> WitnessOutput.error "NativeStr.fromPointer: Need 3 SSAs (undef, insertPtr, result)"

                        // String operations (IntrinsicModule.String)
                        | IntrinsicModule.String, "concat2", [str1SSA; str2SSA] ->
                            // Get string types from arguments
                            let str1Type = argTypes.[0]
                            let str2Type = argTypes.[1]

                            match tryMatch (pStringConcat2 resultSSA str1SSA str2SSA str1Type str2Type) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "String.concat2 pattern failed"

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
                                // Binary arithmetic (signed integer)
                                | BinaryArith "addi" -> tryMatch (pAddI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "subi" -> tryMatch (pSubI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "muli" -> tryMatch (pMulI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "divsi" -> tryMatch (pDivSI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "divui" -> tryMatch (pDivUI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "remsi" -> tryMatch (pRemSI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "remui" -> tryMatch (pRemUI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                                // Floating-point arithmetic
                                | BinaryArith "addf" -> tryMatch (pAddF resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "subf" -> tryMatch (pSubF resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "mulf" -> tryMatch (pMulF resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "divf" -> tryMatch (pDivF resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                                // Bitwise operations
                                | BinaryArith "andi" -> tryMatch (pAndI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "ori" -> tryMatch (pOrI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "xori" -> tryMatch (pXorI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "shli" -> tryMatch (pShLI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "shrui" -> tryMatch (pShRUI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | BinaryArith "shrsi" -> tryMatch (pShRSI resultSSA lhsSSA rhsSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                                // Comparison operations - pass OPERAND type, not result type
                                | Comparison "eq" -> tryMatch (pCmpI resultSSA ICmpPred.Eq lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "ne" -> tryMatch (pCmpI resultSSA ICmpPred.Ne lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "slt" -> tryMatch (pCmpI resultSSA ICmpPred.Slt lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "sle" -> tryMatch (pCmpI resultSSA ICmpPred.Sle lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "sgt" -> tryMatch (pCmpI resultSSA ICmpPred.Sgt lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "sge" -> tryMatch (pCmpI resultSSA ICmpPred.Sge lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "ult" -> tryMatch (pCmpI resultSSA ICmpPred.Ult lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "ule" -> tryMatch (pCmpI resultSSA ICmpPred.Ule lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "ugt" -> tryMatch (pCmpI resultSSA ICmpPred.Ugt lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
                                | Comparison "uge" -> tryMatch (pCmpI resultSSA ICmpPred.Uge lhsSSA rhsSSA operandType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator

                                | _ -> None

                            match opResult with
                            | Some (op, _) ->
                                { InlineOps = [op]; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = resultType } }
                            | None ->
                                WitnessOutput.error $"Unknown or unsupported binary operator: {info.Operation} (category: {category})"

                        // Unary operators (single operand)
                        | IntrinsicModule.Operators, _, [operandSSA] ->
                            let arch = ctx.Coeffects.Platform.TargetArch
                            let resultType = mapNativeTypeForArch arch node.Type

                            // Witness pre-assigned SSAs (Operators get 5 SSAs from SSAAssignment)
                            match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                            | Some ssas when ssas.Length >= 2 ->
                                // Use classification from PSGCombinators for principled operator dispatch
                                let category = classifyAtomicOp info
                                let opResult =
                                    match category with
                                    // Unary operations
                                    | UnaryArith "xori" ->
                                        // Boolean NOT: xori %operand, 1
                                        // Witness pre-assigned SSAs: ssas.[0] = constant, ssas.[1] = result
                                        let constSSA = ssas.[0]
                                        // Emit constant 1 (for i1 boolean type)
                                        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, 1L, TInt I1))
                                        // Emit XOR operation
                                        match tryMatch (pXorI resultSSA operandSSA constSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                        | Some (xorOp, _) -> Some [constOp; xorOp]
                                        | None -> None

                                    | _ -> None

                                match opResult with
                                | Some ops ->
                                    { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = resultSSA; Type = resultType } }
                                | None ->
                                    WitnessOutput.error $"Unknown or unsupported unary operator: {info.Operation} (category: {category})"
                            | _ -> WitnessOutput.error "UnaryArith: Need 2 SSAs (constant, result)"

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

                // Get result SSA and return type
                match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                | None -> WitnessOutput.error "Application: No SSA assigned to result"
                | Some resultSSA ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type

                    // Emit direct function call by name (no declaration coordination)
                    // Declaration will be collected and emitted by MLIR Declaration Collection Pass
                    match tryMatch (pDirectCall resultSSA funcName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
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

                    // Get result SSA and return type
                    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error "Application: No SSA assigned to result"
                    | Some resultSSA ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let retType = mapNativeTypeForArch arch node.Type

                        // Emit indirect call
                        match tryMatch (pApplicationCall resultSSA funcSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
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
