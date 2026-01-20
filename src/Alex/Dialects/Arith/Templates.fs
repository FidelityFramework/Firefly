/// Arith Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → ArithOp
/// NO sprintf. NO string formatting. Just data construction.
module Alex.Dialects.Arith.Templates

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer addition: arith.addi
let addI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.AddI (result, lhs, rhs, ty)

/// Integer subtraction: arith.subi
let subI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.SubI (result, lhs, rhs, ty)

/// Integer multiplication: arith.muli
let mulI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.MulI (result, lhs, rhs, ty)

/// Signed integer division: arith.divsi
let divSI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.DivSI (result, lhs, rhs, ty)

/// Unsigned integer division: arith.divui
let divUI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.DivUI (result, lhs, rhs, ty)

/// Signed integer remainder: arith.remsi
let remSI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.RemSI (result, lhs, rhs, ty)

/// Unsigned integer remainder: arith.remui
let remUI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.RemUI (result, lhs, rhs, ty)

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Bitwise AND: arith.andi
let andI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.AndI (result, lhs, rhs, ty)

/// Bitwise OR: arith.ori
let orI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.OrI (result, lhs, rhs, ty)

/// Bitwise XOR: arith.xori
let xorI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.XOrI (result, lhs, rhs, ty)

/// Shift left: arith.shli
let shlI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.ShLI (result, lhs, rhs, ty)

/// Arithmetic shift right (signed): arith.shrsi
let shrSI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.ShRSI (result, lhs, rhs, ty)

/// Logical shift right (unsigned): arith.shrui
let shrUI (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.ShRUI (result, lhs, rhs, ty)

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Float addition: arith.addf
let addF (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.AddF (result, lhs, rhs, ty)

/// Float subtraction: arith.subf
let subF (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.SubF (result, lhs, rhs, ty)

/// Float multiplication: arith.mulf
let mulF (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.MulF (result, lhs, rhs, ty)

/// Float division: arith.divf
let divF (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.DivF (result, lhs, rhs, ty)

/// Float remainder: arith.remf
let remF (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.RemF (result, lhs, rhs, ty)

/// Float negation: arith.negf
let negF (result: SSA) (operand: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.NegF (result, operand, ty)

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer comparison: arith.cmpi
let cmpI (result: SSA) (pred: ICmpPred) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.CmpI (result, pred, lhs, rhs, ty)

/// Float comparison: arith.cmpf
let cmpF (result: SSA) (pred: FCmpPred) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.CmpF (result, pred, lhs, rhs, ty)

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer constant: arith.constant
let constI (result: SSA) (value: int64) (ty: MLIRType) : ArithOp =
    ArithOp.ConstI (result, value, ty)

/// Float constant: arith.constant
let constF (result: SSA) (value: float) (ty: MLIRType) : ArithOp =
    ArithOp.ConstF (result, value, ty)

/// Boolean true constant
let constTrue (result: SSA) : ArithOp =
    ArithOp.ConstI (result, 1L, MLIRTypes.i1)

/// Boolean false constant
let constFalse (result: SSA) : ArithOp =
    ArithOp.ConstI (result, 0L, MLIRTypes.i1)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Sign extend: arith.extsi
let extSI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.ExtSI (result, operand, fromTy, toTy)

/// Zero extend: arith.extui
let extUI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.ExtUI (result, operand, fromTy, toTy)

/// Truncate: arith.trunci
let truncI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.TruncI (result, operand, fromTy, toTy)

/// Signed int to float: arith.sitofp
let siToFP (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.SIToFP (result, operand, fromTy, toTy)

/// Unsigned int to float: arith.uitofp
let uiToFP (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.UIToFP (result, operand, fromTy, toTy)

/// Float to signed int: arith.fptosi
let fpToSI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.FPToSI (result, operand, fromTy, toTy)

/// Float to unsigned int: arith.fptoui
let fpToUI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.FPToUI (result, operand, fromTy, toTy)

/// Float extend: arith.extf
let extF (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.ExtF (result, operand, fromTy, toTy)

/// Float truncate: arith.truncf
let truncF (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.TruncF (result, operand, fromTy, toTy)

/// Index cast (signed): arith.index_cast
let indexCast (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.IndexCast (result, operand, fromTy, toTy)

/// Index cast (unsigned): arith.index_castui
let indexCastUI (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : ArithOp =
    ArithOp.IndexCastUI (result, operand, fromTy, toTy)

// ═══════════════════════════════════════════════════════════════════════════
// SELECT
// ═══════════════════════════════════════════════════════════════════════════

/// Select: arith.select (ternary conditional)
let select (result: SSA) (cond: SSA) (trueVal: SSA) (falseVal: SSA) (ty: MLIRType) : ArithOp =
    ArithOp.Select (result, cond, trueVal, falseVal, ty)

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCH HELPERS - Select template based on operation kind
// ═══════════════════════════════════════════════════════════════════════════

/// Binary arithmetic operation kinds
type BinaryArithKind =
    | Add | Sub | Mul | Div | Rem
    | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight

/// Comparison operation kinds (prefixed to avoid collision with ICmpPred)
type CompareKind =
    | CmpLt | CmpLe | CmpGt | CmpGe | CmpEq | CmpNe

/// Select integer binary template from kind
let intBinary (kind: BinaryArithKind) (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    match kind with
    | Add -> addI result lhs rhs ty
    | Sub -> subI result lhs rhs ty
    | Mul -> mulI result lhs rhs ty
    | Div -> divSI result lhs rhs ty
    | Rem -> remSI result lhs rhs ty
    | BitAnd -> andI result lhs rhs ty
    | BitOr -> orI result lhs rhs ty
    | BitXor -> xorI result lhs rhs ty
    | ShiftLeft -> shlI result lhs rhs ty
    | ShiftRight -> shrSI result lhs rhs ty

/// Select float binary template from kind
let floatBinary (kind: BinaryArithKind) (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    match kind with
    | Add -> addF result lhs rhs ty
    | Sub -> subF result lhs rhs ty
    | Mul -> mulF result lhs rhs ty
    | Div -> divF result lhs rhs ty
    | Rem -> remF result lhs rhs ty
    | _ -> failwith "Bitwise operations not supported for float"

/// Map compare kind to signed integer predicate
let signedPred (kind: CompareKind) : ICmpPred =
    match kind with
    | CmpLt -> Slt
    | CmpLe -> Sle
    | CmpGt -> Sgt
    | CmpGe -> Sge
    | CmpEq -> Eq
    | CmpNe -> Ne

/// Map compare kind to unsigned integer predicate
let unsignedPred (kind: CompareKind) : ICmpPred =
    match kind with
    | CmpLt -> Ult
    | CmpLe -> Ule
    | CmpGt -> Ugt
    | CmpGe -> Uge
    | CmpEq -> Eq
    | CmpNe -> Ne

/// Map compare kind to ordered float predicate
let orderedPred (kind: CompareKind) : FCmpPred =
    match kind with
    | CmpLt -> OLt
    | CmpLe -> OLe
    | CmpGt -> OGt
    | CmpGe -> OGe
    | CmpEq -> OEq
    | CmpNe -> ONe

/// Select integer comparison template from kind (signed)
let intCompare (kind: CompareKind) (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    cmpI result (signedPred kind) lhs rhs ty

/// Select float comparison template from kind (ordered)
let floatCompare (kind: CompareKind) (result: SSA) (lhs: SSA) (rhs: SSA) (ty: MLIRType) : ArithOp =
    cmpF result (orderedPred kind) lhs rhs ty

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap ArithOp in MLIROp
let wrap (op: ArithOp) : MLIROp = MLIROp.ArithOp op
