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
open Alex.Patterns.ElisionPatterns
open Alex.Patterns.MemoryPatterns
open Alex.Patterns.PlatformPatterns
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
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
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

                    // Get result SSA
                    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error $"Application: No SSA for atomic operation {info.FullName}"
                    | Some resultSSA ->
                        // Dispatch based on intrinsic module and operation
                        match info.Module, info.Operation, argSSAs with
                        | IntrinsicModule.NativePtr, "stackalloc", [countSSA] ->
                            match tryMatch (pNativePtrStackAlloc resultSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.stackalloc pattern failed"

                        | IntrinsicModule.NativePtr, "write", [ptrSSA; valueSSA] ->
                            // Extract element type from pointer argument (nativeptr<'T>)
                            let ptrNodeId = argIds.[0]
                            match SemanticGraph.tryGetNode ptrNodeId ctx.Graph with
                            | Some ptrNode ->
                                let elemType =
                                    match ptrNode.Type with
                                    | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "nativeptr" ->
                                        mapNativeTypeForArch ctx.Coeffects.Platform.TargetArch innerTy
                                    | _ ->
                                        // Fallback to i8 if type extraction fails
                                        TInt I8

                                // Allocate temporary SSA for index constant
                                let indexSSA = SSA.V 999990  // Temporary SSA for index 0
                                match tryMatch (pNativePtrWrite valueSSA ptrSSA elemType indexSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "NativePtr.write pattern failed"
                            | None -> WitnessOutput.error "NativePtr.write: Could not resolve pointer argument node"

                        | IntrinsicModule.NativePtr, "read", [ptrSSA] ->
                            match tryMatch (pNativePtrRead resultSSA ptrSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.read pattern failed"

                        | IntrinsicModule.Sys, "write", [fdSSA; bufferSSA; countSSA] ->
                            // Witness observes actual buffer type (no declaration coordination)
                            // Recall buffer argument to get its actual type (memref or ptr)
                            let bufferNodeId = argIds.[1]
                            match MLIRAccumulator.recallNode bufferNodeId ctx.Accumulator with
                            | Some (_, bufferType) ->
                                // Allocate conversion SSA if buffer is memref (Pattern will handle conversion)
                                let conversionSSA =
                                    match bufferType with
                                    | TMemRefScalar _ | TMemRef _ -> Some (MLIRAccumulator.freshMLIRTemp ctx.Accumulator)
                                    | _ -> None
                                
                                // Emit call with actual buffer type
                                // Declaration will be collected and emitted by MLIR Declaration Collection Pass
                                // Pattern will normalize memref→ptr at FFI boundary if needed
                                match tryMatch (pSysWriteTyped resultSSA fdSSA bufferSSA bufferType countSSA conversionSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                                | None -> WitnessOutput.error "Sys.write pattern failed"
                            | None -> WitnessOutput.error "Sys.write: buffer argument not yet witnessed"

                        | IntrinsicModule.Sys, "read", [fdSSA; bufferSSA; countSSA] ->
                            match tryMatch (pSysRead resultSSA fdSSA bufferSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "Sys.read pattern failed"

                        // Binary arithmetic intrinsics (+, -, *, /, %)
                        | IntrinsicModule.Operators, _, [lhsSSA; rhsSSA] ->
                            let arch = ctx.Coeffects.Platform.TargetArch
                            match tryMatch (pBuildBinaryArith resultSSA lhsSSA rhsSSA arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error $"Binary arithmetic pattern failed for {info.Operation}"

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
                    match tryMatch (pDirectCall resultSSA funcName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
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
                        match tryMatch (pApplicationCall resultSSA funcSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
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
