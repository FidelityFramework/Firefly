/// BitsOps - Witness bit-level operations
///
/// SCOPE: Handle ONLY Bits module intrinsics from FNCS.
/// DOES NOT: Implement transform logic, fill FNCS gaps.
///
/// Bits operations provide:
/// - Byte swapping: htons, htonl, ntohs, ntohl (network/host endianness)
/// - Bitcasting: float↔int conversions without value change
module Alex.Witnesses.BitsOps

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.SemanticPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// SSA HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get result SSA for a node
let private requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No result SSA for node %A" nodeId

// ═══════════════════════════════════════════════════════════════════════════
// BITS OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Bits intrinsic operations
let witness
    (ctx: WitnessContext)
    (node: SemanticNode)
    (args: Val list)
    (returnType: NativeType)
    : MLIROp list * TransferResult =

    let appNodeId = node.Id
    let resultSSA = requireSSA appNodeId ctx

    // Extract intrinsic info
    match node.Kind with
    | SemanticKind.Application (funcNodeId, _) ->
        match SemanticGraph.tryGetNode funcNodeId ctx.Graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic intrinsicInfo ->
                match intrinsicInfo with
                | BitsOp opName ->
                    match opName, args with
                    | "htons", [val16] | "ntohs", [val16] ->
                        // Byte swap uint16 using llvm.intr.bswap - use arg's actual type
                        let op = MLIROp.LLVMOp (LLVMOp.Bswap (resultSSA, val16.SSA, val16.Type))
                        [op], TRValue { SSA = resultSSA; Type = val16.Type }

                    | "htonl", [val] | "ntohl", [val] ->
                        // Byte swap using llvm.intr.bswap - use arg's actual type (platform word)
                        let op = MLIROp.LLVMOp (LLVMOp.Bswap (resultSSA, val.SSA, val.Type))
                        [op], TRValue { SSA = resultSSA; Type = val.Type }

                    | "float32ToInt32Bits", [f32] ->
                        // Bitcast float32 to int32
                        let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, f32.SSA, MLIRTypes.f32, MLIRTypes.i32))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.i32 }

                    | "int32BitsToFloat32", [i32] ->
                        // Bitcast int32 to float32
                        let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, i32.SSA, MLIRTypes.i32, MLIRTypes.f32))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.f32 }

                    | "float64ToInt64Bits", [f64] ->
                        // Bitcast float64 to int64
                        let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, f64.SSA, MLIRTypes.f64, MLIRTypes.i64))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.i64 }

                    | "int64BitsToFloat64", [i64] ->
                        // Bitcast int64 to float64
                        let op = MLIROp.LLVMOp (LLVMOp.Bitcast (resultSSA, i64.SSA, MLIRTypes.i64, MLIRTypes.f64))
                        [op], TRValue { SSA = resultSSA; Type = MLIRTypes.f64 }

                    | _ ->
                        [], TRError (sprintf "Unhandled Bits operation: %s (with %d args)" opName args.Length)

                | _ ->
                    [], TRError "Not a Bits intrinsic"
            | _ ->
                [], TRError "Function node is not an intrinsic"
        | None ->
            [], TRError "Function node not found"
    | _ ->
        [], TRError "Not an Application node"
