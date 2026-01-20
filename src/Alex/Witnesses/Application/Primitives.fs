/// Application/Primitives - Witness primitive operations to MLIR
///
/// Handles binary (arithmetic, comparison, bitwise) and unary operations.
/// Returns structured MLIROp, uses coeffects from WitnessContext.
module Alex.Witnesses.Application.Primitives

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Patterns.SemanticPatterns

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Check if types are compatible for binary operations
let private typesMatch (ty1: MLIRType) (ty2: MLIRType) : bool =
    ty1 = ty2

/// Get integer bit width from MLIR type
/// TIndex is platform-word integer (i64 on 64-bit), treated as I64 for operations
let private getIntBitWidth (ty: MLIRType) : IntBitWidth option =
    match ty with
    | TInt w -> Some w
    | TIndex -> Some I64  // Platform-word integer
    | _ -> None

/// Get float width from MLIR type
let private getFloatBitWidth (ty: MLIRType) : FloatBitWidth option =
    match ty with
    | TFloat w -> Some w
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// BINARY ARITHMETIC OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a binary arithmetic operation using structured MLIROp
let witnessBinaryArith
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (lhs: Val)
    (rhs: Val)
    : (MLIROp list * TransferResult) option =

    if not (typesMatch lhs.Type rhs.Type) then None
    else
        let resultSSA = requireNodeSSA appNodeId z
        let ty = lhs.Type
        
        match getIntBitWidth ty, getFloatBitWidth ty with
        | Some _, None ->
            // Integer operations
            let arithOp =
                match opName with
                | "op_Addition" -> Some (ArithOp.AddI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Subtraction" -> Some (ArithOp.SubI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Multiply" -> Some (ArithOp.MulI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Division" -> Some (ArithOp.DivSI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Modulus" -> Some (ArithOp.RemSI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | _ -> None
            arithOp |> Option.map (fun op ->
                [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = ty })
        
        | None, Some _ ->
            // Float operations
            let arithOp =
                match opName with
                | "op_Addition" -> Some (ArithOp.AddF (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Subtraction" -> Some (ArithOp.SubF (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Multiply" -> Some (ArithOp.MulF (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_Division" -> Some (ArithOp.DivF (resultSSA, lhs.SSA, rhs.SSA, ty))
                | _ -> None
            arithOp |> Option.map (fun op ->
                [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = ty })
        
        | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a comparison operation using structured MLIROp
let witnessComparison
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (lhs: Val)
    (rhs: Val)
    : (MLIROp list * TransferResult) option =

    if not (typesMatch lhs.Type rhs.Type) then None
    else
        let resultSSA = requireNodeSSA appNodeId z
        let ty = lhs.Type
        let resultType = MLIRTypes.bool  // Comparisons always return i1
        
        match getIntBitWidth ty, getFloatBitWidth ty with
        | Some _, None ->
            // Integer comparisons
            let pred =
                match opName with
                | "op_LessThan" -> Some ICmpPred.Slt
                | "op_LessThanOrEqual" -> Some ICmpPred.Sle
                | "op_GreaterThan" -> Some ICmpPred.Sgt
                | "op_GreaterThanOrEqual" -> Some ICmpPred.Sge
                | "op_Equality" -> Some ICmpPred.Eq
                | "op_Inequality" -> Some ICmpPred.Ne
                | _ -> None
            pred |> Option.map (fun p ->
                let op = ArithOp.CmpI (resultSSA, p, lhs.SSA, rhs.SSA, ty)
                [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = resultType })
        
        | None, Some _ ->
            // Float comparisons (ordered)
            let pred =
                match opName with
                | "op_LessThan" -> Some FCmpPred.OLt
                | "op_LessThanOrEqual" -> Some FCmpPred.OLe
                | "op_GreaterThan" -> Some FCmpPred.OGt
                | "op_GreaterThanOrEqual" -> Some FCmpPred.OGe
                | "op_Equality" -> Some FCmpPred.OEq
                | "op_Inequality" -> Some FCmpPred.ONe
                | _ -> None
            pred |> Option.map (fun p ->
                let op = ArithOp.CmpF (resultSSA, p, lhs.SSA, rhs.SSA, ty)
                [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = resultType })
        
        | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a bitwise binary operation using structured MLIROp
let witnessBitwise
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (lhs: Val)
    (rhs: Val)
    : (MLIROp list * TransferResult) option =

    if not (typesMatch lhs.Type rhs.Type) then None
    else
        match getIntBitWidth lhs.Type with
        | None -> None
        | Some _ ->
            let resultSSA = requireNodeSSA appNodeId z
            let ty = lhs.Type
            
            let arithOp =
                match opName with
                | "op_BitwiseAnd" -> Some (ArithOp.AndI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_BitwiseOr" -> Some (ArithOp.OrI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_ExclusiveOr" -> Some (ArithOp.XOrI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_LeftShift" -> Some (ArithOp.ShLI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | "op_RightShift" -> Some (ArithOp.ShRSI (resultSSA, lhs.SSA, rhs.SSA, ty))
                | _ -> None
            
            arithOp |> Option.map (fun op ->
                [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = ty })

// ═══════════════════════════════════════════════════════════════════════════
// BOOLEAN OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a boolean binary operation (AND/OR on i1)
let witnessBooleanBinary
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (lhs: Val)
    (rhs: Val)
    : (MLIROp list * TransferResult) option =

    match lhs.Type, rhs.Type with
    | TInt I1, TInt I1 ->
        let resultSSA = requireNodeSSA appNodeId z
        let ty = MLIRTypes.bool
        
        let arithOp =
            match opName with
            | "op_BooleanAnd" -> Some (ArithOp.AndI (resultSSA, lhs.SSA, rhs.SSA, ty))
            | "op_BooleanOr" -> Some (ArithOp.OrI (resultSSA, lhs.SSA, rhs.SSA, ty))
            | _ -> None
        
        arithOp |> Option.map (fun op ->
            [MLIROp.ArithOp op], TRValue { SSA = resultSSA; Type = ty })
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// UNARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a unary operation
let witnessUnary
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (arg: Val)
    : (MLIROp list * TransferResult) option =

    // Get pre-allocated SSAs - need 2 for most unary ops (const + result)
    let ssas = requireNodeSSAs appNodeId z
    let resultSSA = requireNodeSSA appNodeId z

    match opName with
    | "not" when arg.Type = MLIRTypes.bool ->
        // Boolean NOT: XOR with true (1) - needs 2 SSAs
        let trueSSA = ssas.[0]
        let ty = MLIRTypes.bool
        let trueOp = MLIROp.ArithOp (ArithOp.ConstI (trueSSA, 1L, ty))
        let xorOp = MLIROp.ArithOp (ArithOp.XOrI (resultSSA, arg.SSA, trueSSA, ty))
        Some ([trueOp; xorOp], TRValue { SSA = resultSSA; Type = ty })

    | "op_UnaryNegation" ->
        match getIntBitWidth arg.Type with
        | Some _ ->
            // Integer negation: 0 - x - needs 2 SSAs
            let zeroSSA = ssas.[0]
            let ty = arg.Type
            let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, ty))
            let subOp = MLIROp.ArithOp (ArithOp.SubI (resultSSA, zeroSSA, arg.SSA, ty))
            Some ([zeroOp; subOp], TRValue { SSA = resultSSA; Type = ty })
        | None ->
            match getFloatBitWidth arg.Type with
            | Some _ ->
                // Float negation: negf - needs 1 SSA
                let ty = arg.Type
                let negOp = MLIROp.ArithOp (ArithOp.NegF (resultSSA, arg.SSA, ty))
                Some ([negOp], TRValue { SSA = resultSSA; Type = ty })
            | None -> None

    | "op_OnesComplement" ->
        match getIntBitWidth arg.Type with
        | Some _ ->
            // Bitwise NOT: XOR with -1 (all ones) - needs 2 SSAs
            let onesSSA = ssas.[0]
            let ty = arg.Type
            let onesOp = MLIROp.ArithOp (ArithOp.ConstI (onesSSA, -1L, ty))
            let xorOp = MLIROp.ArithOp (ArithOp.XOrI (resultSSA, arg.SSA, onesSSA, ty))
            Some ([onesOp; xorOp], TRValue { SSA = resultSSA; Type = ty })
        | None -> None

    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED BINARY DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Try to witness any binary primitive operation
let tryWitnessBinaryOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (lhs: Val)
    (rhs: Val)
    : (MLIROp list * TransferResult) option =

    // Try each category in order
    witnessBinaryArith appNodeId z opName lhs rhs
    |> Option.orElseWith (fun () -> witnessComparison appNodeId z opName lhs rhs)
    |> Option.orElseWith (fun () -> witnessBitwise appNodeId z opName lhs rhs)
    |> Option.orElseWith (fun () -> witnessBooleanBinary appNodeId z opName lhs rhs)
