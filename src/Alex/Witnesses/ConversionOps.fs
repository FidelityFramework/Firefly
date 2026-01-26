/// ConversionOps - Witness type conversion operations
///
/// SCOPE: Handle ONLY Convert module intrinsics from FNCS.
/// DOES NOT: Implement transform logic, fill FNCS gaps.
///
/// Conversion operations provide:
/// - Integer ↔ Float: SIToFP, FPToSI
/// - Integer widening: ExtSI (signed), ExtUI (unsigned)
/// - Integer narrowing: TruncI
/// - Identity conversions: No-op when source = target type
module Alex.Witnesses.ConversionOps

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
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
// CONVERSION OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Convert intrinsic operations
let witness
    (ctx: WitnessContext)
    (node: SemanticNode)
    (args: Val list)
    (returnType: NativeType)
    : MLIROp list * TransferResult =

    let appNodeId = node.Id
    let resultType = mapType returnType ctx
    let resultSSA = requireSSA appNodeId ctx

    // Extract intrinsic info
    match node.Kind with
    | SemanticKind.Application (funcNodeId, _) ->
        match SemanticGraph.tryGetNode funcNodeId ctx.Graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic intrinsicInfo ->
                match intrinsicInfo with
                | ConvertOp convName ->
                    match args with
                    | [arg] ->
                        let convOp =
                            match convName, arg.Type, resultType with
                            // Integer to float
                            | "toFloat", TInt _, TFloat _ ->
                                Some (ArithOp.SIToFP (resultSSA, arg.SSA, arg.Type, resultType))
                            | "toFloat32", TInt _, TFloat F32 ->
                                Some (ArithOp.SIToFP (resultSSA, arg.SSA, arg.Type, MLIRTypes.f32))

                            // Float to integer
                            | "toInt", TFloat _, TInt I32 ->
                                Some (ArithOp.FPToSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))
                            | "toInt64", TFloat _, TInt I64 ->
                                Some (ArithOp.FPToSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))

                            // Integer widening (signed)
                            | "toInt64", TInt w, TInt I64 when w <> I64 ->
                                Some (ArithOp.ExtSI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))
                            | "toInt", TInt I8, TInt I32 ->
                                Some (ArithOp.ExtSI (resultSSA, arg.SSA, MLIRTypes.i8, MLIRTypes.i32))
                            | "toInt", TInt I16, TInt I32 ->
                                Some (ArithOp.ExtSI (resultSSA, arg.SSA, MLIRTypes.i16, MLIRTypes.i32))

                            // Integer narrowing
                            | "toInt", TInt I64, TInt I32 ->
                                Some (ArithOp.TruncI (resultSSA, arg.SSA, MLIRTypes.i64, MLIRTypes.i32))
                            | "toByte", TInt _, TInt I8 ->
                                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i8))

                            // Unsigned integer conversions (same-size = identity/reinterpret)
                            | "toUInt32", TInt I32, TInt I32 ->
                                None  // Same size, just reinterpret
                            | "toUInt16", TInt I16, TInt I16 ->
                                None
                            | "toUInt64", TInt I64, TInt I64 ->
                                None
                            // Unsigned widening (zero-extend)
                            | "toUInt32", TInt w, TInt I32 when w < I32 ->
                                Some (ArithOp.ExtUI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))
                            | "toUInt64", TInt w, TInt I64 when w < I64 ->
                                Some (ArithOp.ExtUI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i64))
                            // Unsigned narrowing (truncate)
                            | "toUInt16", TInt w, TInt I16 when w > I16 ->
                                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i16))
                            | "toUInt32", TInt w, TInt I32 when w > I32 ->
                                Some (ArithOp.TruncI (resultSSA, arg.SSA, arg.Type, MLIRTypes.i32))

                            // Identity conversions
                            | "toInt", TInt I32, TInt I32 ->
                                None  // No-op, return arg directly
                            | "toInt64", TInt I64, TInt I64 ->
                                None
                            | "toFloat", TFloat F64, TFloat F64 ->
                                None
                            | "toFloat32", TFloat F32, TFloat F32 ->
                                None

                            | _ -> None

                        match convOp with
                        | Some op ->
                            [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = resultType }
                        | None when convName = "toInt" && arg.Type = MLIRTypes.i32 ->
                            [], TRValue arg  // Identity
                        | None when convName = "toInt64" && arg.Type = MLIRTypes.i64 ->
                            [], TRValue arg  // Identity
                        | None when convName = "toUInt32" && arg.Type = MLIRTypes.i32 ->
                            [], TRValue arg  // Identity (same bit representation)
                        | None when convName = "toUInt64" && arg.Type = MLIRTypes.i64 ->
                            [], TRValue arg  // Identity
                        | None when convName = "toUInt16" && arg.Type = MLIRTypes.i16 ->
                            [], TRValue arg  // Identity
                        | None ->
                            [], TRError (sprintf "Unhandled conversion: %s from %A to %A" convName arg.Type resultType)

                    | _ ->
                        [], TRError (sprintf "Convert.%s expects 1 argument, got %d" convName args.Length)

                | _ ->
                    [], TRError "Not a Convert intrinsic"
            | _ ->
                [], TRError "Function node is not an intrinsic"
        | None ->
            [], TRError "Function node not found"
    | _ ->
        [], TRError "Not an Application node"
