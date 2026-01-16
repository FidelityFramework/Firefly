/// Core MLIR Serialization - StringBuilder-based emission
///
/// ARCHITECTURAL PRINCIPLE: ZERO SPRINTF
/// All serialization uses StringBuilder.Append chains.
/// This is the ONLY place where structured types become text.
///
/// Pattern: Each emit function takes (thing, StringBuilder) -> unit
/// The StringBuilder accumulates the MLIR text output.
module Alex.Dialects.Core.Serialize

open System.Text
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit SSA value reference
let ssa (s: SSA) (sb: StringBuilder) : unit =
    match s with
    | V n -> sb.Append("%v").Append(n) |> ignore
    | Arg n -> sb.Append("%arg").Append(n) |> ignore

/// Emit SSA value reference and return the StringBuilder (for chaining)
let ssaC (s: SSA) (sb: StringBuilder) : StringBuilder =
    match s with
    | V n -> sb.Append("%v").Append(n)
    | Arg n -> sb.Append("%arg").Append(n)

/// Emit block reference
let blockRef (BlockRef label) (sb: StringBuilder) : unit =
    sb.Append("^").Append(label) |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// TYPE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit integer bit width
let intBitWidth (bw: IntBitWidth) (sb: StringBuilder) : unit =
    match bw with
    | I1 -> sb.Append("i1") |> ignore
    | I8 -> sb.Append("i8") |> ignore
    | I16 -> sb.Append("i16") |> ignore
    | I32 -> sb.Append("i32") |> ignore
    | I64 -> sb.Append("i64") |> ignore

/// Emit float bit width
let floatBitWidth (bw: FloatBitWidth) (sb: StringBuilder) : unit =
    match bw with
    | F16 -> sb.Append("f16") |> ignore
    | BF16 -> sb.Append("bf16") |> ignore
    | F32 -> sb.Append("f32") |> ignore
    | F64 -> sb.Append("f64") |> ignore

/// Emit MLIR type
let rec mlirType (t: MLIRType) (sb: StringBuilder) : unit =
    match t with
    | TInt bw -> intBitWidth bw sb
    | TFloat bw -> floatBitWidth bw sb
    | TIndex -> sb.Append("index") |> ignore
    | TPtr -> sb.Append("!llvm.ptr") |> ignore
    | TStruct fields ->
        // LLVM structs delegate to llvmType for proper index -> i64 conversion
        llvmStruct fields sb
    | TArray (count, elem) ->
        sb.Append("!llvm.array<").Append(count).Append(" x ") |> ignore
        mlirType elem sb
        sb.Append(">") |> ignore
    | TFunc (args, ret) ->
        sb.Append("(") |> ignore
        args |> List.iteri (fun i a ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType a sb)
        sb.Append(") -> ") |> ignore
        mlirType ret sb
    | TVector (shape, elem) ->
        sb.Append("vector<") |> ignore
        shape.Dims |> List.iteri (fun i dim ->
            if i > 0 then sb.Append("x") |> ignore
            if shape.Scalable then sb.Append("[").Append(dim).Append("]") |> ignore
            else sb.Append(dim) |> ignore)
        sb.Append("x") |> ignore
        mlirType elem sb
        sb.Append(">") |> ignore
    | TMemRef (shape, elem) ->
        sb.Append("memref<") |> ignore
        shape |> List.iteri (fun i dim ->
            sb.Append(dim).Append("x") |> ignore)
        mlirType elem sb
        sb.Append(">") |> ignore
    | TUnit -> sb.Append("i32") |> ignore  // Unit maps to i32 for ABI
    | TError msg -> failwithf "COMPILER ERROR: Attempted to serialize TError: %s" msg

/// Emit LLVM struct type with proper field type conversion
/// Separate from mlirType to break circular dependency
and llvmStruct (fields: MLIRType list) (sb: StringBuilder) : unit =
    sb.Append("!llvm.struct<(") |> ignore
    fields |> List.iteri (fun i f ->
        if i > 0 then sb.Append(", ") |> ignore
        llvmType f sb)  // Struct fields must use LLVM types (index -> i64)
    sb.Append(")>") |> ignore

/// Emit MLIR type for LLVM dialect (Index -> i64 on 64-bit)
and llvmType (t: MLIRType) (sb: StringBuilder) : unit =
    match t with
    | TIndex -> sb.Append("i64") |> ignore  // Platform word on 64-bit
    | TStruct fields -> llvmStruct fields sb  // Handle struct recursively
    | TArray (count, elem) ->
        // LLVM arrays also need llvmType for elements
        sb.Append("!llvm.array<").Append(count).Append(" x ") |> ignore
        llvmType elem sb
        sb.Append(">") |> ignore
    | _ -> mlirType t sb

// ═══════════════════════════════════════════════════════════════════════════
// PREDICATE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit integer comparison predicate
let icmpPred (p: ICmpPred) (sb: StringBuilder) : unit =
    match p with
    | Eq -> sb.Append("eq") |> ignore
    | Ne -> sb.Append("ne") |> ignore
    | Slt -> sb.Append("slt") |> ignore
    | Sle -> sb.Append("sle") |> ignore
    | Sgt -> sb.Append("sgt") |> ignore
    | Sge -> sb.Append("sge") |> ignore
    | Ult -> sb.Append("ult") |> ignore
    | Ule -> sb.Append("ule") |> ignore
    | Ugt -> sb.Append("ugt") |> ignore
    | Uge -> sb.Append("uge") |> ignore

/// Emit float comparison predicate
let fcmpPred (p: FCmpPred) (sb: StringBuilder) : unit =
    match p with
    | OEq -> sb.Append("oeq") |> ignore
    | ONe -> sb.Append("one") |> ignore
    | OLt -> sb.Append("olt") |> ignore
    | OLe -> sb.Append("ole") |> ignore
    | OGt -> sb.Append("ogt") |> ignore
    | OGe -> sb.Append("oge") |> ignore
    | UEq -> sb.Append("ueq") |> ignore
    | UNe -> sb.Append("une") |> ignore
    | ULt -> sb.Append("ult") |> ignore
    | ULe -> sb.Append("ule") |> ignore
    | UGt -> sb.Append("ugt") |> ignore
    | UGe -> sb.Append("uge") |> ignore
    | Ord -> sb.Append("ord") |> ignore
    | Uno -> sb.Append("uno") |> ignore
    | AlwaysFalse -> sb.Append("false") |> ignore
    | AlwaysTrue -> sb.Append("true") |> ignore

/// Emit index comparison predicate
let indexCmpPred (p: IndexCmpPred) (sb: StringBuilder) : unit =
    match p with
    | IEq -> sb.Append("eq") |> ignore
    | INe -> sb.Append("ne") |> ignore
    | ISlt -> sb.Append("slt") |> ignore
    | ISle -> sb.Append("sle") |> ignore
    | ISgt -> sb.Append("sgt") |> ignore
    | ISge -> sb.Append("sge") |> ignore
    | IUlt -> sb.Append("ult") |> ignore
    | IUle -> sb.Append("ule") |> ignore
    | IUgt -> sb.Append("ugt") |> ignore
    | IUge -> sb.Append("uge") |> ignore

/// Emit atomic ordering
let atomicOrdering (o: AtomicOrdering) (sb: StringBuilder) : unit =
    match o with
    | NotAtomic -> sb.Append("not_atomic") |> ignore
    | Unordered -> sb.Append("unordered") |> ignore
    | Monotonic -> sb.Append("monotonic") |> ignore
    | Acquire -> sb.Append("acquire") |> ignore
    | Release -> sb.Append("release") |> ignore
    | AcqRel -> sb.Append("acq_rel") |> ignore
    | SeqCst -> sb.Append("seq_cst") |> ignore

/// Emit atomic RMW kind
let atomicRMWKind (k: AtomicRMWKind) (sb: StringBuilder) : unit =
    match k with
    | Xchg -> sb.Append("xchg") |> ignore
    | Add -> sb.Append("add") |> ignore
    | Sub -> sb.Append("sub") |> ignore
    | And -> sb.Append("_and") |> ignore
    | Nand -> sb.Append("nand") |> ignore
    | Or -> sb.Append("_or") |> ignore
    | Xor -> sb.Append("_xor") |> ignore
    | Max -> sb.Append("max") |> ignore
    | Min -> sb.Append("min") |> ignore
    | UMax -> sb.Append("umax") |> ignore
    | UMin -> sb.Append("umin") |> ignore
    | FAdd -> sb.Append("fadd") |> ignore
    | FSub -> sb.Append("fsub") |> ignore
    | FMax -> sb.Append("fmax") |> ignore
    | FMin -> sb.Append("fmin") |> ignore
    | UIncWrap -> sb.Append("uinc_wrap") |> ignore
    | UDecWrap -> sb.Append("udec_wrap") |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL REFERENCE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit global reference
let globalRef (g: GlobalRef) (sb: StringBuilder) : unit =
    match g with
    | GFunc name -> sb.Append("@").Append(name) |> ignore
    | GString hash -> sb.Append("@str_").Append(hash) |> ignore
    | GBytes idx -> sb.Append("@bytes_").Append(idx) |> ignore
    | GNamed name -> sb.Append("@").Append(name) |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// VISIBILITY SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit function visibility
let funcVisibility (v: FuncVisibility) (sb: StringBuilder) : unit =
    match v with
    | Public -> ()  // default, no attribute
    | Private -> sb.Append("private ") |> ignore
    | Nested -> sb.Append("nested ") |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// STRING ESCAPING
// ═══════════════════════════════════════════════════════════════════════════

/// Escape a string for MLIR string literals
/// ZERO sprintf - character-by-character emission
let escapeString (s: string) (sb: StringBuilder) : unit =
    for c in s do
        match c with
        | '\\' -> sb.Append("\\\\") |> ignore
        | '"' -> sb.Append("\\\"") |> ignore
        | '\n' -> sb.Append("\\0A") |> ignore
        | '\r' -> sb.Append("\\0D") |> ignore
        | '\t' -> sb.Append("\\09") |> ignore
        | c when int c < 32 || int c > 126 ->
            // Emit as hex escape
            sb.Append("\\").Append((int c).ToString("X2")) |> ignore
        | c -> sb.Append(c) |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// ATTRIBUTE VALUE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit attribute value (for OpEnvelope)
let rec attrValue (av: AttrValue) (sb: StringBuilder) : unit =
    match av with
    | IntAttr (v, ty) ->
        sb.Append(v).Append(" : ") |> ignore
        mlirType ty sb
    | FloatAttr (v, ty) ->
        if System.Double.IsNaN(v) then sb.Append("0x7FC00000") |> ignore
        elif System.Double.IsPositiveInfinity(v) then sb.Append("0x7F800000") |> ignore
        elif System.Double.IsNegativeInfinity(v) then sb.Append("0xFF800000") |> ignore
        else sb.Append(v.ToString("G17")) |> ignore
        sb.Append(" : ") |> ignore
        mlirType ty sb
    | StringAttr s ->
        sb.Append("\"") |> ignore
        escapeString s sb
        sb.Append("\"") |> ignore
    | BoolAttr b ->
        sb.Append(if b then "true" else "false") |> ignore
    | TypeAttr ty ->
        mlirType ty sb
    | ArrayAttr values ->
        sb.Append("[") |> ignore
        values |> List.iteri (fun i v ->
            if i > 0 then sb.Append(", ") |> ignore
            attrValue v sb)
        sb.Append("]") |> ignore
    | DictAttr dict ->
        sb.Append("{") |> ignore
        dict |> Map.toList |> List.iteri (fun i (k, v) ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(k).Append(" = ") |> ignore
            attrValue v sb)
        sb.Append("}") |> ignore
    | UnitAttr ->
        () // Unit attributes have no value representation

// ═══════════════════════════════════════════════════════════════════════════
// ARITH DIALECT SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Helper for binary arith ops (reduces duplication)
let private binaryArith (opName: string) (r: SSA) (l: SSA) (rh: SSA) (t: MLIRType) (sb: StringBuilder) : unit =
    ssa r sb; sb.Append(" = arith.").Append(opName).Append(" ") |> ignore
    ssa l sb; sb.Append(", ") |> ignore
    ssa rh sb; sb.Append(" : ") |> ignore
    mlirType t sb

/// Helper for conversion arith ops
let private conversionArith (opName: string) (r: SSA) (op: SSA) (ft: MLIRType) (tt: MLIRType) (sb: StringBuilder) : unit =
    ssa r sb; sb.Append(" = arith.").Append(opName).Append(" ") |> ignore
    ssa op sb; sb.Append(" : ") |> ignore
    mlirType ft sb; sb.Append(" to ") |> ignore
    mlirType tt sb

/// Emit arith dialect operation - EXHAUSTIVE from ArithOps.td
let arithOp (op: ArithOp) (sb: StringBuilder) : unit =
    match op with
    // === INTEGER BINARY OPERATIONS ===
    | AddI (r, l, rh, t) -> binaryArith "addi" r l rh t sb
    | SubI (r, l, rh, t) -> binaryArith "subi" r l rh t sb
    | MulI (r, l, rh, t) -> binaryArith "muli" r l rh t sb
    | DivSI (r, l, rh, t) -> binaryArith "divsi" r l rh t sb
    | DivUI (r, l, rh, t) -> binaryArith "divui" r l rh t sb
    | CeilDivSI (r, l, rh, t) -> binaryArith "ceildivsi" r l rh t sb
    | CeilDivUI (r, l, rh, t) -> binaryArith "ceildivui" r l rh t sb
    | FloorDivSI (r, l, rh, t) -> binaryArith "floordivsi" r l rh t sb
    | RemSI (r, l, rh, t) -> binaryArith "remsi" r l rh t sb
    | RemUI (r, l, rh, t) -> binaryArith "remui" r l rh t sb

    // === INTEGER EXTENDED OPERATIONS ===
    | AddUIExtended (r, o, l, rh, t) ->
        ssa r sb; sb.Append(", ") |> ignore
        ssa o sb; sb.Append(" = arith.addui_extended ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(" : ") |> ignore
        mlirType t sb; sb.Append(", i1") |> ignore
    | MulSIExtended (rl, rh', l, rh, t) ->
        ssa rl sb; sb.Append(", ") |> ignore
        ssa rh' sb; sb.Append(" = arith.mulsi_extended ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(" : ") |> ignore
        mlirType t sb
    | MulUIExtended (rl, rh', l, rh, t) ->
        ssa rl sb; sb.Append(", ") |> ignore
        ssa rh' sb; sb.Append(" = arith.mului_extended ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(" : ") |> ignore
        mlirType t sb

    // === INTEGER BITWISE OPERATIONS ===
    | AndI (r, l, rh, t) -> binaryArith "andi" r l rh t sb
    | OrI (r, l, rh, t) -> binaryArith "ori" r l rh t sb
    | XOrI (r, l, rh, t) -> binaryArith "xori" r l rh t sb
    | ShLI (r, l, rh, t) -> binaryArith "shli" r l rh t sb
    | ShRSI (r, l, rh, t) -> binaryArith "shrsi" r l rh t sb
    | ShRUI (r, l, rh, t) -> binaryArith "shrui" r l rh t sb

    // === INTEGER MIN/MAX ===
    | MaxSI (r, l, rh, t) -> binaryArith "maxsi" r l rh t sb
    | MaxUI (r, l, rh, t) -> binaryArith "maxui" r l rh t sb
    | MinSI (r, l, rh, t) -> binaryArith "minsi" r l rh t sb
    | MinUI (r, l, rh, t) -> binaryArith "minui" r l rh t sb

    // === FLOAT BINARY OPERATIONS ===
    | AddF (r, l, rh, t) -> binaryArith "addf" r l rh t sb
    | SubF (r, l, rh, t) -> binaryArith "subf" r l rh t sb
    | MulF (r, l, rh, t) -> binaryArith "mulf" r l rh t sb
    | DivF (r, l, rh, t) -> binaryArith "divf" r l rh t sb
    | RemF (r, l, rh, t) -> binaryArith "remf" r l rh t sb
    | NegF (r, op, t) ->
        ssa r sb; sb.Append(" = arith.negf ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        mlirType t sb

    // === FLOAT MIN/MAX ===
    | MaximumF (r, l, rh, t) -> binaryArith "maximumf" r l rh t sb
    | MaxNumF (r, l, rh, t) -> binaryArith "maxnumf" r l rh t sb
    | MinimumF (r, l, rh, t) -> binaryArith "minimumf" r l rh t sb
    | MinNumF (r, l, rh, t) -> binaryArith "minnumf" r l rh t sb

    // === COMPARISON OPERATIONS ===
    | CmpI (r, pred, l, rh, t) ->
        ssa r sb; sb.Append(" = arith.cmpi ") |> ignore
        icmpPred pred sb; sb.Append(", ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(" : ") |> ignore
        mlirType t sb
    | CmpF (r, pred, l, rh, t) ->
        ssa r sb; sb.Append(" = arith.cmpf ") |> ignore
        fcmpPred pred sb; sb.Append(", ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(" : ") |> ignore
        mlirType t sb

    // === CONSTANTS ===
    | ConstI (r, v, t) ->
        ssa r sb; sb.Append(" = arith.constant ").Append(v).Append(" : ") |> ignore
        mlirType t sb
    | ConstF (r, v, t) ->
        ssa r sb
        sb.Append(" = arith.constant ") |> ignore
        // Exact IEEE 754 bit representation - type-preserving, no string manipulation
        match t with
        | TFloat F32 ->
            let bits = System.BitConverter.SingleToInt32Bits(float32 v)
            sb.AppendFormat("0x{0:X8}", bits) |> ignore
        | _ ->
            let bits = System.BitConverter.DoubleToInt64Bits(v)
            sb.AppendFormat("0x{0:X16}", bits) |> ignore
        sb.Append(" : ") |> ignore
        mlirType t sb

    // === INTEGER CONVERSIONS ===
    | ExtSI (r, op, ft, tt) -> conversionArith "extsi" r op ft tt sb
    | ExtUI (r, op, ft, tt) -> conversionArith "extui" r op ft tt sb
    | TruncI (r, op, ft, tt) -> conversionArith "trunci" r op ft tt sb

    // === FLOAT CONVERSIONS ===
    | ArithOp.SIToFP (r, op, ft, tt) -> conversionArith "sitofp" r op ft tt sb
    | ArithOp.UIToFP (r, op, ft, tt) -> conversionArith "uitofp" r op ft tt sb
    | ArithOp.FPToSI (r, op, ft, tt) -> conversionArith "fptosi" r op ft tt sb
    | ArithOp.FPToUI (r, op, ft, tt) -> conversionArith "fptoui" r op ft tt sb
    | ExtF (r, op, ft, tt) -> conversionArith "extf" r op ft tt sb
    | TruncF (r, op, ft, tt) -> conversionArith "truncf" r op ft tt sb

    // === SCALING FLOAT CONVERSIONS ===
    | ScalingExtF (r, op, scale, ft, tt) ->
        ssa r sb; sb.Append(" = arith.scaling_extf ") |> ignore
        ssa op sb; sb.Append(", ") |> ignore
        ssa scale sb; sb.Append(" : ") |> ignore
        mlirType ft sb; sb.Append(", ") |> ignore
        mlirType (TInt I8) sb; sb.Append(" to ") |> ignore
        mlirType tt sb
    | ScalingTruncF (r, op, scale, ft, tt) ->
        ssa r sb; sb.Append(" = arith.scaling_truncf ") |> ignore
        ssa op sb; sb.Append(", ") |> ignore
        ssa scale sb; sb.Append(" : ") |> ignore
        mlirType ft sb; sb.Append(", ") |> ignore
        mlirType (TInt I8) sb; sb.Append(" to ") |> ignore
        mlirType tt sb

    // === INDEX CONVERSIONS ===
    | IndexCast (r, op, ft, tt) -> conversionArith "index_cast" r op ft tt sb
    | IndexCastUI (r, op, ft, tt) -> conversionArith "index_castui" r op ft tt sb

    // === BITCAST ===
    | ArithOp.Bitcast (r, op, ft, tt) -> conversionArith "bitcast" r op ft tt sb

    // === SELECT ===
    | Select (r, c, tv, fv, t) ->
        ssa r sb; sb.Append(" = arith.select ") |> ignore
        ssa c sb; sb.Append(", ") |> ignore
        ssa tv sb; sb.Append(", ") |> ignore
        ssa fv sb; sb.Append(" : ") |> ignore
        mlirType t sb

// ═══════════════════════════════════════════════════════════════════════════
// LLVM DIALECT SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit LLVM dialect operation - EXHAUSTIVE from LLVMOps.td
let llvmOp (op: LLVMOp) (sb: StringBuilder) : unit =
    match op with
    // === MEMORY OPERATIONS ===
    | Alloca (r, count, elemTy, align) ->
        ssa r sb; sb.Append(" = llvm.alloca ") |> ignore
        ssa count sb; sb.Append(" x ") |> ignore
        llvmType elemTy sb
        match align with
        | Some a -> sb.Append(" {alignment = ").Append(a).Append("}") |> ignore
        | None -> ()
        sb.Append(" : (i64) -> !llvm.ptr") |> ignore
    | LLVMOp.Load (r, ptr, ty, ordering) ->
        ssa r sb; sb.Append(" = llvm.load ") |> ignore
        match ordering with
        | NotAtomic -> ()
        | _ -> sb.Append("atomic ") |> ignore; atomicOrdering ordering sb; sb.Append(" ") |> ignore
        ssa ptr sb; sb.Append(" : !llvm.ptr -> ") |> ignore
        llvmType ty sb
    | LLVMOp.Store (v, ptr, valueTy, ordering) ->
        sb.Append("llvm.store ") |> ignore
        match ordering with
        | NotAtomic -> ()
        | _ -> sb.Append("atomic ") |> ignore; atomicOrdering ordering sb; sb.Append(" ") |> ignore
        ssa v sb; sb.Append(", ") |> ignore
        ssa ptr sb; sb.Append(" : ") |> ignore
        llvmType valueTy sb; sb.Append(", !llvm.ptr") |> ignore
    | GEP (r, base', indices, elemTy) ->
        ssa r sb; sb.Append(" = llvm.getelementptr ") |> ignore
        ssa base' sb
        for (idx, _) in indices do
            sb.Append("[") |> ignore
            ssa idx sb
            sb.Append("]") |> ignore
        sb.Append(" : (!llvm.ptr") |> ignore
        for (_, idxTy) in indices do
            sb.Append(", ") |> ignore
            mlirType idxTy sb
        sb.Append(") -> !llvm.ptr, ") |> ignore
        llvmType elemTy sb
    | StructGEP (r, base', fieldIndex, structTy) ->
        // Emit: %r = llvm.getelementptr inbounds %base[0, fieldIndex] : (!llvm.ptr) -> !llvm.ptr, structTy
        ssa r sb; sb.Append(" = llvm.getelementptr inbounds ") |> ignore
        ssa base' sb
        sb.Append("[0, ") |> ignore
        sb.Append(fieldIndex) |> ignore
        sb.Append("] : (!llvm.ptr) -> !llvm.ptr, ") |> ignore
        llvmType structTy sb
    | MemCpy (dst, src, len, isVolatile) ->
        sb.Append("llvm.intr.memcpy ") |> ignore
        ssa dst sb; sb.Append(", ") |> ignore
        ssa src sb; sb.Append(", ") |> ignore
        ssa len sb
        if isVolatile then sb.Append(" volatile") |> ignore
        sb.Append(" : !llvm.ptr, !llvm.ptr, i64") |> ignore
    | MemMove (dst, src, len, isVolatile) ->
        sb.Append("llvm.intr.memmove ") |> ignore
        ssa dst sb; sb.Append(", ") |> ignore
        ssa src sb; sb.Append(", ") |> ignore
        ssa len sb
        if isVolatile then sb.Append(" volatile") |> ignore
        sb.Append(" : !llvm.ptr, !llvm.ptr, i64") |> ignore
    | MemSet (dst, value, len, isVolatile) ->
        sb.Append("llvm.intr.memset ") |> ignore
        ssa dst sb; sb.Append(", ") |> ignore
        ssa value sb; sb.Append(", ") |> ignore
        ssa len sb
        if isVolatile then sb.Append(" volatile") |> ignore
        sb.Append(" : !llvm.ptr, i8, i64") |> ignore

    // === GLOBAL OPERATIONS ===
    | AddressOf (r, g) ->
        ssa r sb; sb.Append(" = llvm.mlir.addressof ") |> ignore
        globalRef g sb; sb.Append(" : !llvm.ptr") |> ignore
    | GlobalDef (name, value, ty, isConst) ->
        sb.Append("llvm.mlir.global ") |> ignore
        if isConst then sb.Append("constant ") |> ignore
        else sb.Append("internal ") |> ignore
        sb.Append("@").Append(name).Append("(").Append(value).Append(") : ") |> ignore
        llvmType ty sb
    | GlobalString (name, content, len) ->
        sb.Append("llvm.mlir.global internal constant @").Append(name).Append("(\"") |> ignore
        escapeString content sb
        sb.Append("\") : !llvm.array<").Append(len).Append(" x i8>") |> ignore
    | NullPtr r ->
        ssa r sb; sb.Append(" = llvm.mlir.zero : !llvm.ptr") |> ignore

    // === CALL OPERATIONS ===
    | Call (result, func, args, retTy) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("llvm.call ") |> ignore
        globalRef func sb
        sb.Append("(") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa arg.SSA sb)
        sb.Append(") : (") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            llvmType arg.Type sb)
        sb.Append(") -> ") |> ignore
        llvmType retTy sb
    | IndirectCall (result, funcPtr, args, retTy) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("llvm.call ") |> ignore
        ssa funcPtr sb
        sb.Append("(") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa arg.SSA sb)
        sb.Append(") : !llvm.ptr, (") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            llvmType arg.Type sb)
        sb.Append(") -> ") |> ignore
        llvmType retTy sb
    | Invoke (result, func, args, retTy, normalDest, unwindDest) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("llvm.invoke ") |> ignore
        globalRef func sb
        sb.Append("(") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa arg.SSA sb)
        sb.Append(") to ") |> ignore
        blockRef normalDest sb
        sb.Append(" unwind ") |> ignore
        blockRef unwindDest sb
        sb.Append(" : (") |> ignore
        args |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            llvmType arg.Type sb)
        sb.Append(") -> ") |> ignore
        llvmType retTy sb
    | Landingpad (r, ty, isCleanup) ->
        ssa r sb; sb.Append(" = llvm.landingpad ") |> ignore
        if isCleanup then sb.Append("cleanup ") |> ignore
        sb.Append(": ") |> ignore
        llvmType ty sb

    // === STRUCT/AGGREGATE OPERATIONS ===
    | ExtractValue (r, agg, indices, aggTy) ->
        ssa r sb; sb.Append(" = llvm.extractvalue ") |> ignore
        ssa agg sb
        sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(idx) |> ignore)
        sb.Append("]") |> ignore
        sb.Append(" : ") |> ignore
        llvmType aggTy sb
    | InsertValue (r, agg, v, indices, aggTy) ->
        ssa r sb; sb.Append(" = llvm.insertvalue ") |> ignore
        ssa v sb; sb.Append(", ") |> ignore
        ssa agg sb
        sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(idx) |> ignore)
        sb.Append("]") |> ignore
        sb.Append(" : ") |> ignore
        llvmType aggTy sb
    | Undef (r, ty) ->
        ssa r sb; sb.Append(" = llvm.mlir.undef : ") |> ignore
        llvmType ty sb
    | Poison (r, ty) ->
        ssa r sb; sb.Append(" = llvm.mlir.poison : ") |> ignore
        llvmType ty sb
    | ZeroInit (r, ty) ->
        ssa r sb; sb.Append(" = llvm.mlir.zero : ") |> ignore
        llvmType ty sb

    // === VECTOR OPERATIONS ===
    | ExtractElement (r, vec, idx) ->
        ssa r sb; sb.Append(" = llvm.extractelement ") |> ignore
        ssa vec sb; sb.Append("[") |> ignore
        ssa idx sb; sb.Append("]") |> ignore
    | InsertElement (r, vec, v, idx) ->
        ssa r sb; sb.Append(" = llvm.insertelement ") |> ignore
        ssa v sb; sb.Append(", ") |> ignore
        ssa vec sb; sb.Append("[") |> ignore
        ssa idx sb; sb.Append("]") |> ignore
    | ShuffleVector (r, v1, v2, mask) ->
        ssa r sb; sb.Append(" = llvm.shufflevector ") |> ignore
        ssa v1 sb; sb.Append(", ") |> ignore
        ssa v2 sb; sb.Append(" [") |> ignore
        mask |> List.iteri (fun i m ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(m) |> ignore)
        sb.Append("]") |> ignore

    // === CONVERSION OPERATIONS ===
    | LLVMOp.Bitcast (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.bitcast ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | Bswap (r, op, ty) ->
        // llvm.intr.bswap : (i16) -> i16  (or i32, i64)
        ssa r sb; sb.Append(" = llvm.intr.bswap(") |> ignore
        ssa op sb; sb.Append(") : (") |> ignore
        llvmType ty sb; sb.Append(") -> ") |> ignore
        llvmType ty sb
    | IntToPtr (r, op, ft) ->
        ssa r sb; sb.Append(" = llvm.inttoptr ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to !llvm.ptr") |> ignore
    | PtrToInt (r, op, tt) ->
        ssa r sb; sb.Append(" = llvm.ptrtoint ") |> ignore
        ssa op sb; sb.Append(" : !llvm.ptr to ") |> ignore
        llvmType tt sb
    | AddrSpaceCast (r, op, fromAS, toAS) ->
        ssa r sb; sb.Append(" = llvm.addrspacecast ") |> ignore
        ssa op sb; sb.Append(" : !llvm.ptr<").Append(fromAS).Append("> to !llvm.ptr<").Append(toAS).Append(">") |> ignore
    | ZExt (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.zext ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | SExt (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.sext ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | Trunc (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.trunc ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | FPExt (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.fpext ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | FPTrunc (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.fptrunc ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | LLVMOp.FPToSI (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.fptosi ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | LLVMOp.FPToUI (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.fptoui ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | LLVMOp.SIToFP (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.sitofp ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb
    | LLVMOp.UIToFP (r, op, ft, tt) ->
        ssa r sb; sb.Append(" = llvm.uitofp ") |> ignore
        ssa op sb; sb.Append(" : ") |> ignore
        llvmType ft sb; sb.Append(" to ") |> ignore
        llvmType tt sb

    // === INTEGER ARITHMETIC (LLVM dialect with flags) ===
    | LLVMAdd (r, l, rh, nsw, nuw) ->
        ssa r sb; sb.Append(" = llvm.add ") |> ignore
        if nsw then sb.Append("nsw ") |> ignore
        if nuw then sb.Append("nuw ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMSub (r, l, rh, nsw, nuw) ->
        ssa r sb; sb.Append(" = llvm.sub ") |> ignore
        if nsw then sb.Append("nsw ") |> ignore
        if nuw then sb.Append("nuw ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMMul (r, l, rh, nsw, nuw) ->
        ssa r sb; sb.Append(" = llvm.mul ") |> ignore
        if nsw then sb.Append("nsw ") |> ignore
        if nuw then sb.Append("nuw ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMUDiv (r, l, rh, exact) ->
        ssa r sb; sb.Append(" = llvm.udiv ") |> ignore
        if exact then sb.Append("exact ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMSDiv (r, l, rh, exact) ->
        ssa r sb; sb.Append(" = llvm.sdiv ") |> ignore
        if exact then sb.Append("exact ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMURem (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.urem ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMSRem (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.srem ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMAnd (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.and ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMOr (r, l, rh, disjoint) ->
        ssa r sb; sb.Append(" = llvm.or ") |> ignore
        if disjoint then sb.Append("disjoint ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMXor (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.xor ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMShl (r, l, rh, nsw, nuw) ->
        ssa r sb; sb.Append(" = llvm.shl ") |> ignore
        if nsw then sb.Append("nsw ") |> ignore
        if nuw then sb.Append("nuw ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMLShr (r, l, rh, exact) ->
        ssa r sb; sb.Append(" = llvm.lshr ") |> ignore
        if exact then sb.Append("exact ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMAShr (r, l, rh, exact) ->
        ssa r sb; sb.Append(" = llvm.ashr ") |> ignore
        if exact then sb.Append("exact ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb

    // === FLOAT ARITHMETIC ===
    | LLVMFAdd (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.fadd ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMFSub (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.fsub ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMFMul (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.fmul ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMFDiv (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.fdiv ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMFRem (r, l, rh) ->
        ssa r sb; sb.Append(" = llvm.frem ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMFNeg (r, op) ->
        ssa r sb; sb.Append(" = llvm.fneg ") |> ignore
        ssa op sb

    // === COMPARISON ===
    | ICmp (r, pred, l, rh) ->
        ssa r sb; sb.Append(" = llvm.icmp \"") |> ignore
        icmpPred pred sb; sb.Append("\" ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | FCmp (r, pred, l, rh) ->
        ssa r sb; sb.Append(" = llvm.fcmp \"") |> ignore
        fcmpPred pred sb; sb.Append("\" ") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb
    | LLVMSelect (r, c, tv, fv) ->
        ssa r sb; sb.Append(" = llvm.select ") |> ignore
        ssa c sb; sb.Append(", ") |> ignore
        ssa tv sb; sb.Append(", ") |> ignore
        ssa fv sb

    // === ATOMIC OPERATIONS ===
    | AtomicRMW (r, op', ptr, v, ordering) ->
        ssa r sb; sb.Append(" = llvm.atomicrmw ") |> ignore
        atomicRMWKind op' sb; sb.Append(" ") |> ignore
        ssa ptr sb; sb.Append(", ") |> ignore
        ssa v sb; sb.Append(" ") |> ignore
        atomicOrdering ordering sb
    | AtomicCmpXchg (rv, rs, ptr, expected, desired, successOrd, failOrd) ->
        ssa rv sb; sb.Append(", ") |> ignore
        ssa rs sb; sb.Append(" = llvm.cmpxchg ") |> ignore
        ssa ptr sb; sb.Append(", ") |> ignore
        ssa expected sb; sb.Append(", ") |> ignore
        ssa desired sb; sb.Append(" ") |> ignore
        atomicOrdering successOrd sb; sb.Append(" ") |> ignore
        atomicOrdering failOrd sb
    | Fence (ordering, scope) ->
        sb.Append("llvm.fence ") |> ignore
        atomicOrdering ordering sb
        match scope with
        | Some s -> sb.Append(" syncscope(\"").Append(s).Append("\")") |> ignore
        | None -> ()

    // === CONTROL FLOW (LLVM terminators) ===
    | LLVMBr dest ->
        sb.Append("llvm.br ") |> ignore
        blockRef dest sb
    | LLVMCondBr (c, trueDest, falseDest) ->
        sb.Append("llvm.cond_br ") |> ignore
        ssa c sb; sb.Append(", ") |> ignore
        blockRef trueDest sb; sb.Append(", ") |> ignore
        blockRef falseDest sb
    | LLVMSwitch (v, defaultDest, cases) ->
        sb.Append("llvm.switch ") |> ignore
        ssa v sb; sb.Append(" : i64, ") |> ignore
        blockRef defaultDest sb; sb.Append(" [") |> ignore
        cases |> List.iteri (fun i (caseVal, dest) ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(caseVal).Append(": ") |> ignore
            blockRef dest sb)
        sb.Append("]") |> ignore
    | Unreachable ->
        sb.Append("llvm.unreachable") |> ignore
    | Resume v ->
        sb.Append("llvm.resume ") |> ignore
        ssa v sb

    // === PHI NODE ===
    | Phi (r, ty, incomings) ->
        ssa r sb; sb.Append(" = llvm.mlir.phi (") |> ignore
        incomings |> List.iteri (fun i (v, blk) ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa v sb; sb.Append(" : ") |> ignore
            blockRef blk sb)
        sb.Append(") : ") |> ignore
        llvmType ty sb

    // === INLINE ASM ===
    | InlineAsm (result, asm, constraints, args, retTy, hasSideEffects, isAlignStack) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("llvm.inline_asm ") |> ignore
        if hasSideEffects then sb.Append("has_side_effects ") |> ignore
        if isAlignStack then sb.Append("is_align_stack ") |> ignore
        sb.Append("\"").Append(asm).Append("\", \"").Append(constraints).Append("\" ") |> ignore
        // Emit SSA values
        args |> List.iteri (fun i (ssaVal, _) ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa ssaVal sb)
        // Emit function type signature: (argTypes) -> retType
        sb.Append(" : (") |> ignore
        args |> List.iteri (fun i (_, argTy) ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType argTy sb)
        sb.Append(") -> ") |> ignore
        match retTy with
        | Some t -> mlirType t sb
        | None -> sb.Append("()") |> ignore

    // === VA ARG ===
    | VaArg (r, vaList, ty) ->
        ssa r sb; sb.Append(" = llvm.va_arg ") |> ignore
        ssa vaList sb; sb.Append(" : !llvm.ptr, ") |> ignore
        llvmType ty sb

    // === RETURN ===
    | Return (value, valueTy) ->
        sb.Append("llvm.return") |> ignore
        match value, valueTy with
        | Some v, Some ty ->
            sb.Append(" ") |> ignore
            ssa v sb
            sb.Append(" : ") |> ignore
            mlirType ty sb
        | Some v, None ->
            sb.Append(" ") |> ignore
            ssa v sb
        | None, _ -> ()

    // LLVMFuncDef is handled in the recursive group (needs access to `region`)
    | LLVMFuncDef _ -> failwith "LLVMFuncDef should be serialized via llvmFuncDef in recursive group"

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS FOR DIALECT SERIALIZATION (before recursive group)
// ═══════════════════════════════════════════════════════════════════════════

/// Helper for binary index ops
let private binaryIndex (opName: string) (r: SSA) (l: SSA) (rh: SSA) (sb: StringBuilder) : unit =
    ssa r sb; sb.Append(" = index.").Append(opName).Append(" ") |> ignore
    ssa l sb; sb.Append(", ") |> ignore
    ssa rh sb

// ═══════════════════════════════════════════════════════════════════════════
// SCF DIALECT SERIALIZATION (recursive group - needs access to `region`)
// ═══════════════════════════════════════════════════════════════════════════

/// Emit LLVM function definition (in recursive group because it needs `region`)
/// llvm.func @name(%arg0: type, ...) -> retType { ... }
let rec llvmFuncDef (name: string) (args: (SSA * MLIRType) list) (retTy: MLIRType) (body: Region) (linkage: LLVMLinkage) (sb: StringBuilder) : unit =
    sb.Append("llvm.func ") |> ignore
    match linkage with
    | LLVMPrivate -> sb.Append("private ") |> ignore
    | LLVMInternal -> sb.Append("internal ") |> ignore
    | LLVMExternal -> ()
    sb.Append("@").Append(name).Append("(") |> ignore
    args |> List.iteri (fun i (a, t) ->
        if i > 0 then sb.Append(", ") |> ignore
        ssa a sb; sb.Append(": ") |> ignore
        llvmType t sb)
    sb.Append(") -> ") |> ignore
    llvmType retTy sb
    region body 2 sb

/// Emit region
/// For single-block regions with no block args, emit ops directly (MLIR doesn't need labels)
/// For multi-block or blocks with args, emit full block structure
/// Empty label (BlockRef "") means implicit entry block - no label emitted
and region (r: Region) (indent: int) (sb: StringBuilder) : unit =
    sb.Append(" {") |> ignore
    for i, blk in r.Blocks |> List.indexed do
        sb.AppendLine() |> ignore
        // Check if this block has an explicit label (non-empty)
        let hasExplicitLabel = match blk.Label with BlockRef s -> s <> ""
        // Only emit block label if: has explicit label AND (multiple blocks, or block has args, or not first block)
        let needsLabel = hasExplicitLabel && (List.length r.Blocks > 1 || not (List.isEmpty blk.Args))
        if needsLabel then
            for _ in 1..indent do sb.Append("  ") |> ignore
            blockRef blk.Label sb
            if not (List.isEmpty blk.Args) then
                sb.Append("(") |> ignore
                blk.Args |> List.iteri (fun j arg ->
                    if j > 0 then sb.Append(", ") |> ignore
                    ssa arg.SSA sb; sb.Append(" : ") |> ignore
                    mlirType arg.Type sb)
                sb.Append(")") |> ignore
            sb.Append(":") |> ignore
            sb.AppendLine() |> ignore
        for op in blk.Ops do
            for _ in 1..(indent) do sb.Append("  ") |> ignore
            mlirOp op sb
            sb.AppendLine() |> ignore
    for _ in 1..(indent-1) do sb.Append("  ") |> ignore
    sb.Append("}") |> ignore

/// Emit SCF dialect operation - EXHAUSTIVE from SCFOps.td
and scfOp (op: SCFOp) (indent: int) (sb: StringBuilder) : unit =
    match op with
    // === CONDITIONALS ===
    | If (results, cond, thenR, elseR, resultTypes) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.if ") |> ignore
        ssa cond sb
        if not (List.isEmpty resultTypes) then
            sb.Append(" -> (") |> ignore
            resultTypes |> List.iteri (fun i t ->
                if i > 0 then sb.Append(", ") |> ignore
                mlirType t sb)
            sb.Append(")") |> ignore
        region thenR (indent + 1) sb
        match elseR with
        | Some er ->
            sb.Append(" else") |> ignore
            region er (indent + 1) sb
        | None -> ()

    // === LOOPS ===
    | While (results, condR, bodyR, iterArgs) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.while") |> ignore
        if not (List.isEmpty iterArgs) then
            sb.Append(" (") |> ignore
            // Use the region's block arg SSAs for bindings (they're what ops inside reference)
            let blockArgs = match condR.Blocks with [b] -> b.Args | _ -> []
            iterArgs |> List.iteri (fun i arg ->
                if i > 0 then sb.Append(", ") |> ignore
                // Use block arg SSA if available, otherwise fall back to iter prefix
                if i < List.length blockArgs then
                    ssa blockArgs.[i].SSA sb
                else
                    sb.Append("%iter").Append(i.ToString()) |> ignore
                sb.Append(" = ") |> ignore
                ssa arg.SSA sb)
            sb.Append(")") |> ignore
        sb.Append(" : (") |> ignore
        iterArgs |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType arg.Type sb)
        sb.Append(") -> (") |> ignore
        iterArgs |> List.iteri (fun i arg ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType arg.Type sb)
        sb.Append(")") |> ignore
        region condR (indent + 1) sb
        sb.Append(" do") |> ignore
        region bodyR (indent + 1) sb

    | For (results, iv, start, stop, step, bodyR, iterArgs) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.for ") |> ignore
        ssa iv sb; sb.Append(" = ") |> ignore
        ssa start sb; sb.Append(" to ") |> ignore
        ssa stop sb; sb.Append(" step ") |> ignore
        ssa step sb
        if not (List.isEmpty iterArgs) then
            sb.Append(" iter_args(") |> ignore
            iterArgs |> List.iteri (fun i arg ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa arg.SSA sb; sb.Append(" = ...") |> ignore)
            sb.Append(")") |> ignore
        region bodyR (indent + 1) sb

    | Forall (results, lbs, ubs, steps, outputs, bodyR) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.forall (") |> ignore
        lbs |> List.iteri (fun i lb ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa lb sb)
        sb.Append(") in (") |> ignore
        ubs |> List.iteri (fun i ub ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa ub sb)
        sb.Append(")") |> ignore
        if not (List.isEmpty outputs) then
            sb.Append(" shared_outs(") |> ignore
            outputs |> List.iteri (fun i o ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa o.SSA sb)
            sb.Append(")") |> ignore
        region bodyR (indent + 1) sb

    | Parallel (results, lbs, ubs, steps, initVals, bodyR) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.parallel (") |> ignore
        // Parallel takes lower, upper, step for each dimension
        lbs |> List.iteri (fun i lb ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa lb sb)
        sb.Append(") = (") |> ignore
        lbs |> List.iteri (fun i _ ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append("...") |> ignore)  // lower bounds
        sb.Append(") to (") |> ignore
        ubs |> List.iteri (fun i ub ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa ub sb)
        sb.Append(") step (") |> ignore
        steps |> List.iteri (fun i s ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa s sb)
        sb.Append(")") |> ignore
        if not (List.isEmpty initVals) then
            sb.Append(" init (") |> ignore
            initVals |> List.iteri (fun i v ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa v.SSA sb)
            sb.Append(")") |> ignore
        region bodyR (indent + 1) sb

    // === SWITCH ===
    | IndexSwitch (results, arg, cases, caseRegions, defaultR, resultTypes) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.index_switch ") |> ignore
        ssa arg sb
        List.iter2 (fun (c: int64) r -> 
            sb.Append(" case ").Append(c) |> ignore
            region r (indent + 1) sb) cases caseRegions
        sb.Append(" default") |> ignore
        region defaultR (indent + 1) sb
        if not (List.isEmpty resultTypes) then
            sb.Append(" -> (") |> ignore
            resultTypes |> List.iteri (fun i t ->
                if i > 0 then sb.Append(", ") |> ignore
                mlirType t sb)
            sb.Append(")") |> ignore

    // === EXECUTE REGION ===
    | ExecuteRegion (results, bodyR, resultTypes) ->
        if not (List.isEmpty results) then
            results |> List.iteri (fun i r ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa r sb)
            sb.Append(" = ") |> ignore
        sb.Append("scf.execute_region") |> ignore
        region bodyR (indent + 1) sb
        if not (List.isEmpty resultTypes) then
            sb.Append(" -> (") |> ignore
            resultTypes |> List.iteri (fun i t ->
                if i > 0 then sb.Append(", ") |> ignore
                mlirType t sb)
            sb.Append(")") |> ignore

    // === REGION TERMINATORS ===
    | Yield values ->
        sb.Append("scf.yield") |> ignore
        if not (List.isEmpty values) then
            // SSA values
            sb.Append(" ") |> ignore
            values |> List.iteri (fun i v ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa v.SSA sb)
            // Type annotations
            sb.Append(" : ") |> ignore
            values |> List.iteri (fun i v ->
                if i > 0 then sb.Append(", ") |> ignore
                mlirType v.Type sb)

    | Condition (cond, args) ->
        sb.Append("scf.condition(") |> ignore
        ssa cond sb
        sb.Append(")") |> ignore
        if not (List.isEmpty args) then
            // SSA values
            sb.Append(" ") |> ignore
            args |> List.iteri (fun i a ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa a.SSA sb)
            // Type annotations
            sb.Append(" : ") |> ignore
            args |> List.iteri (fun i a ->
                if i > 0 then sb.Append(", ") |> ignore
                mlirType a.Type sb)

    | Reduce (operands, reductionRegions) ->
        sb.Append("scf.reduce(") |> ignore
        operands |> List.iteri (fun i op ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa op.SSA sb)
        sb.Append(")") |> ignore
        reductionRegions |> List.iter (fun r -> region r (indent + 1) sb)

    | ReduceReturn r ->
        sb.Append("scf.reduce.return ") |> ignore
        ssa r sb

    | InParallel bodyR ->
        sb.Append("scf.forall.in_parallel") |> ignore
        region bodyR (indent + 1) sb

// ═══════════════════════════════════════════════════════════════════════════
// FUNC DIALECT SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit func dialect operation - EXHAUSTIVE from FuncOps.td
and funcOp (op: FuncOp) (indent: int) (sb: StringBuilder) : unit =
    match op with
    | FuncDef (name, args, retTy, body, vis) ->
        // MLIR syntax: func.func private @name(...) for private functions
        sb.Append("func.func ") |> ignore
        funcVisibility vis sb
        sb.Append("@").Append(name).Append("(") |> ignore
        args |> List.iteri (fun i (a, t) ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa a sb; sb.Append(": ") |> ignore
            mlirType t sb)
        sb.Append(") -> ") |> ignore
        mlirType retTy sb
        region body (indent + 1) sb

    | FuncDecl (name, argTypes, retTy, vis) ->
        // MLIR syntax: func.func private @name(...) for private functions
        sb.Append("func.func ") |> ignore
        funcVisibility vis sb
        sb.Append("@").Append(name).Append("(") |> ignore
        argTypes |> List.iteri (fun i t ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType t sb)
        sb.Append(") -> ") |> ignore
        mlirType retTy sb

    | FuncCall (result, func, args, retTy) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("func.call @").Append(func).Append("(") |> ignore
        args |> List.iteri (fun i a ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa a.SSA sb)
        sb.Append(") : (") |> ignore
        args |> List.iteri (fun i a ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType a.Type sb)
        sb.Append(") -> ") |> ignore
        mlirType retTy sb

    | FuncCallIndirect (result, callee, args, retTy) ->
        match result with
        | Some r -> ssa r sb; sb.Append(" = ") |> ignore
        | None -> ()
        sb.Append("func.call_indirect ") |> ignore
        ssa callee sb; sb.Append("(") |> ignore
        args |> List.iteri (fun i a ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa a.SSA sb)
        sb.Append(") : (") |> ignore
        args |> List.iteri (fun i a ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType a.Type sb)
        sb.Append(") -> ") |> ignore
        mlirType retTy sb

    | FuncConstant (r, funcName, funcTy) ->
        ssa r sb; sb.Append(" = func.constant @").Append(funcName).Append(" : ") |> ignore
        mlirType funcTy sb

    | FuncReturn values ->
        sb.Append("func.return") |> ignore
        if not (List.isEmpty values) then
            sb.Append(" ") |> ignore
            values |> List.iteri (fun i v ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa v sb)

// ═══════════════════════════════════════════════════════════════════════════
// INDEX DIALECT SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Emit index dialect operation - EXHAUSTIVE from IndexOps.td
and indexOp (op: IndexOp) (sb: StringBuilder) : unit =
    match op with
    // === CONSTANTS ===
    | IndexConst (r, v) ->
        ssa r sb; sb.Append(" = index.constant ").Append(v) |> ignore
    | IndexBoolConst (r, v) ->
        ssa r sb; sb.Append(" = index.bool.constant ") |> ignore
        sb.Append(if v then "true" else "false") |> ignore

    // === ARITHMETIC ===
    | IndexAdd (r, l, rh) -> binaryIndex "add" r l rh sb
    | IndexSub (r, l, rh) -> binaryIndex "sub" r l rh sb
    | IndexMul (r, l, rh) -> binaryIndex "mul" r l rh sb
    | IndexDivS (r, l, rh) -> binaryIndex "divs" r l rh sb
    | IndexDivU (r, l, rh) -> binaryIndex "divu" r l rh sb
    | IndexCeilDivS (r, l, rh) -> binaryIndex "ceildivs" r l rh sb
    | IndexCeilDivU (r, l, rh) -> binaryIndex "ceildivu" r l rh sb
    | IndexFloorDivS (r, l, rh) -> binaryIndex "floordivs" r l rh sb
    | IndexRemS (r, l, rh) -> binaryIndex "rems" r l rh sb
    | IndexRemU (r, l, rh) -> binaryIndex "remu" r l rh sb

    // === MIN/MAX ===
    | IndexMaxS (r, l, rh) -> binaryIndex "maxs" r l rh sb
    | IndexMaxU (r, l, rh) -> binaryIndex "maxu" r l rh sb
    | IndexMinS (r, l, rh) -> binaryIndex "mins" r l rh sb
    | IndexMinU (r, l, rh) -> binaryIndex "minu" r l rh sb

    // === BITWISE ===
    | IndexShl (r, l, rh) -> binaryIndex "shl" r l rh sb
    | IndexShrS (r, l, rh) -> binaryIndex "shrs" r l rh sb
    | IndexShrU (r, l, rh) -> binaryIndex "shru" r l rh sb
    | IndexAnd (r, l, rh) -> binaryIndex "and" r l rh sb
    | IndexOr (r, l, rh) -> binaryIndex "or" r l rh sb
    | IndexXor (r, l, rh) -> binaryIndex "xor" r l rh sb

    // === COMPARISON ===
    | IndexCmp (r, pred, l, rh) ->
        ssa r sb; sb.Append(" = index.cmp ") |> ignore
        indexCmpPred pred sb; sb.Append("(") |> ignore
        ssa l sb; sb.Append(", ") |> ignore
        ssa rh sb; sb.Append(")") |> ignore

    // === CASTS ===
    | IndexCastS (r, op, tt) ->
        ssa r sb; sb.Append(" = index.casts ") |> ignore
        ssa op sb; sb.Append(" : index to ") |> ignore
        mlirType tt sb
    | IndexCastU (r, op, tt) ->
        ssa r sb; sb.Append(" = index.castu ") |> ignore
        ssa op sb; sb.Append(" : index to ") |> ignore
        mlirType tt sb

    // === SIZE OF ===
    | IndexSizeOf (r, ty) ->
        ssa r sb; sb.Append(" = index.sizeof ") |> ignore
        mlirType ty sb

/// Emit CF (Control Flow) dialect operation - EXHAUSTIVE from ControlFlowOps.td
and cfOp (op: CFOp) (sb: StringBuilder) : unit =
    match op with
    | Assert (cond, msg) ->
        sb.Append("cf.assert ") |> ignore
        ssa cond sb; sb.Append(", \"") |> ignore
        escapeString msg sb
        sb.Append("\"") |> ignore

    | Br (dest, destOps) ->
        sb.Append("cf.br ") |> ignore
        blockRef dest sb
        if not (List.isEmpty destOps) then
            sb.Append("(") |> ignore
            destOps |> List.iteri (fun i op ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa op.SSA sb; sb.Append(" : ") |> ignore
                mlirType op.Type sb)
            sb.Append(")") |> ignore

    | CondBr (cond, trueDest, trueOps, falseDest, falseOps, weights) ->
        sb.Append("cf.cond_br ") |> ignore
        ssa cond sb; sb.Append(", ") |> ignore
        blockRef trueDest sb
        if not (List.isEmpty trueOps) then
            sb.Append("(") |> ignore
            trueOps |> List.iteri (fun i op ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa op.SSA sb)
            sb.Append(")") |> ignore
        sb.Append(", ") |> ignore
        blockRef falseDest sb
        if not (List.isEmpty falseOps) then
            sb.Append("(") |> ignore
            falseOps |> List.iteri (fun i op ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa op.SSA sb)
            sb.Append(")") |> ignore
        match weights with
        | Some (tw, fw) -> sb.Append(" {branch_weights = [").Append(tw).Append(", ").Append(fw).Append("]}") |> ignore
        | None -> ()

    | Switch (flag, flagTy, defaultDest, defaultOps, cases) ->
        sb.Append("cf.switch ") |> ignore
        ssa flag sb; sb.Append(" : ") |> ignore
        mlirType flagTy sb; sb.Append(", [") |> ignore
        sb.Append("default: ") |> ignore
        blockRef defaultDest sb
        if not (List.isEmpty defaultOps) then
            sb.Append("(") |> ignore
            defaultOps |> List.iteri (fun i op ->
                if i > 0 then sb.Append(", ") |> ignore
                ssa op.SSA sb)
            sb.Append(")") |> ignore
        for (caseVal, caseDest, caseOps) in cases do
            sb.Append(", ").Append(caseVal).Append(": ") |> ignore
            blockRef caseDest sb
            if not (List.isEmpty caseOps) then
                sb.Append("(") |> ignore
                caseOps |> List.iteri (fun i op ->
                    if i > 0 then sb.Append(", ") |> ignore
                    ssa op.SSA sb)
                sb.Append(")") |> ignore
        sb.Append("]") |> ignore

/// Emit Vector dialect operation - CORE from VectorOps.td
and vectorOp (op: VectorOp) (sb: StringBuilder) : unit =
    match op with
    // === BROADCAST ===
    | Broadcast (r, src, resultTy) ->
        ssa r sb; sb.Append(" = vector.broadcast ") |> ignore
        ssa src sb; sb.Append(" : ") |> ignore
        mlirType resultTy sb

    // === EXTRACT/INSERT ===
    | Extract (r, vec, pos) ->
        ssa r sb; sb.Append(" = vector.extract ") |> ignore
        ssa vec sb; sb.Append("[") |> ignore
        pos |> List.iteri (fun i p ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(p) |> ignore)
        sb.Append("]") |> ignore
    | Insert (r, src, dest, pos) ->
        ssa r sb; sb.Append(" = vector.insert ") |> ignore
        ssa src sb; sb.Append(", ") |> ignore
        ssa dest sb; sb.Append("[") |> ignore
        pos |> List.iteri (fun i p ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(p) |> ignore)
        sb.Append("]") |> ignore
    | ExtractStrided (r, vec, offsets, sizes, strides) ->
        ssa r sb; sb.Append(" = vector.extract_strided_slice ") |> ignore
        ssa vec sb; sb.Append(" {offsets = [") |> ignore
        offsets |> List.iteri (fun i o -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(o) |> ignore)
        sb.Append("], sizes = [") |> ignore
        sizes |> List.iteri (fun i s -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(s) |> ignore)
        sb.Append("], strides = [") |> ignore
        strides |> List.iteri (fun i s -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(s) |> ignore)
        sb.Append("]}") |> ignore
    | InsertStrided (r, src, dest, offsets, strides) ->
        ssa r sb; sb.Append(" = vector.insert_strided_slice ") |> ignore
        ssa src sb; sb.Append(", ") |> ignore
        ssa dest sb; sb.Append(" {offsets = [") |> ignore
        offsets |> List.iteri (fun i o -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(o) |> ignore)
        sb.Append("], strides = [") |> ignore
        strides |> List.iteri (fun i s -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(s) |> ignore)
        sb.Append("]}") |> ignore

    // === SHAPE OPERATIONS ===
    | ShapeCast (r, src, resultTy) ->
        ssa r sb; sb.Append(" = vector.shape_cast ") |> ignore
        ssa src sb; sb.Append(" : ... to ") |> ignore
        mlirType resultTy sb
    | Transpose (r, vec, transp) ->
        ssa r sb; sb.Append(" = vector.transpose ") |> ignore
        ssa vec sb; sb.Append(", [") |> ignore
        transp |> List.iteri (fun i t -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(t) |> ignore)
        sb.Append("]") |> ignore
    | FlattenTranspose (r, vec) ->
        ssa r sb; sb.Append(" = vector.flat_transpose ") |> ignore
        ssa vec sb

    // === REDUCTION ===
    | ReductionAdd (r, vec, acc) ->
        ssa r sb; sb.Append(" = vector.reduction <add>, ") |> ignore
        ssa vec sb
        match acc with Some a -> sb.Append(", ") |> ignore; ssa a sb | None -> ()
    | ReductionMul (r, vec, acc) ->
        ssa r sb; sb.Append(" = vector.reduction <mul>, ") |> ignore
        ssa vec sb
        match acc with Some a -> sb.Append(", ") |> ignore; ssa a sb | None -> ()
    | ReductionAnd (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <and>, ") |> ignore
        ssa vec sb
    | ReductionOr (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <or>, ") |> ignore
        ssa vec sb
    | ReductionXor (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <xor>, ") |> ignore
        ssa vec sb
    | ReductionMinSI (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <minsi>, ") |> ignore
        ssa vec sb
    | ReductionMinUI (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <minui>, ") |> ignore
        ssa vec sb
    | ReductionMaxSI (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <maxsi>, ") |> ignore
        ssa vec sb
    | ReductionMaxUI (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <maxui>, ") |> ignore
        ssa vec sb
    | ReductionMinF (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <minimumf>, ") |> ignore
        ssa vec sb
    | ReductionMaxF (r, vec) ->
        ssa r sb; sb.Append(" = vector.reduction <maximumf>, ") |> ignore
        ssa vec sb

    // === FMA ===
    | FMA (r, lhs, rhs, acc) ->
        ssa r sb; sb.Append(" = vector.fma ") |> ignore
        ssa lhs sb; sb.Append(", ") |> ignore
        ssa rhs sb; sb.Append(", ") |> ignore
        ssa acc sb

    // === SPLAT ===
    | Splat (r, v, resultTy) ->
        ssa r sb; sb.Append(" = vector.splat ") |> ignore
        ssa v sb; sb.Append(" : ") |> ignore
        mlirType resultTy sb

    // === LOAD/STORE ===
    | VectorLoad (r, base', indices) ->
        ssa r sb; sb.Append(" = vector.load ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa idx sb)
        sb.Append("]") |> ignore
    | VectorStore (v, base', indices) ->
        sb.Append("vector.store ") |> ignore
        ssa v sb; sb.Append(", ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa idx sb)
        sb.Append("]") |> ignore
    | MaskedLoad (r, base', indices, mask, passthru) ->
        ssa r sb; sb.Append(" = vector.maskedload ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa idx sb)
        sb.Append("], ") |> ignore
        ssa mask sb; sb.Append(", ") |> ignore
        ssa passthru sb
    | MaskedStore (v, base', indices, mask) ->
        sb.Append("vector.maskedstore ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        indices |> List.iteri (fun i idx ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa idx sb)
        sb.Append("], ") |> ignore
        ssa mask sb; sb.Append(", ") |> ignore
        ssa v sb
    | Gather (r, base', indices, indexVec, mask, passthru) ->
        ssa r sb; sb.Append(" = vector.gather ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        ssa indices sb; sb.Append("], ") |> ignore
        ssa indexVec sb; sb.Append(", ") |> ignore
        ssa mask sb; sb.Append(", ") |> ignore
        ssa passthru sb
    | Scatter (v, base', indices, indexVec, mask) ->
        sb.Append("vector.scatter ") |> ignore
        ssa base' sb; sb.Append("[") |> ignore
        ssa indices sb; sb.Append("], ") |> ignore
        ssa indexVec sb; sb.Append(", ") |> ignore
        ssa mask sb; sb.Append(", ") |> ignore
        ssa v sb

    // === MASK OPERATIONS ===
    | CreateMask (r, operands) ->
        ssa r sb; sb.Append(" = vector.create_mask ") |> ignore
        operands |> List.iteri (fun i op ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa op sb)
    | ConstantMask (r, dims) ->
        ssa r sb; sb.Append(" = vector.constant_mask [") |> ignore
        dims |> List.iteri (fun i d -> if i > 0 then sb.Append(", ") |> ignore; sb.Append(d) |> ignore)
        sb.Append("]") |> ignore

    // === PRINT ===
    | Print (src, punct) ->
        sb.Append("vector.print ") |> ignore
        ssa src sb
        match punct with
        | Some p -> sb.Append(" punctuation ").Append(p) |> ignore
        | None -> ()

// ═══════════════════════════════════════════════════════════════════════════
// MAIN DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Emit any MLIR operation (dispatches to dialect-specific serializer)
and mlirOp (op: MLIROp) (sb: StringBuilder) : unit =
    match op with
    // === UNIVERSAL SUPER-STRUCTURE ===
    | Envelope env -> opEnvelope env 1 sb
    // === DIALECT-SPECIFIC ===
    | ArithOp a -> arithOp a sb
    | LLVMOp (LLVMFuncDef (name, args, retTy, body, linkage)) ->
        // LLVMFuncDef needs special handling (uses region from recursive group)
        llvmFuncDef name args retTy body linkage sb
    | LLVMOp l -> llvmOp l sb
    | SCFOp s -> scfOp s 1 sb
    | CFOp c -> cfOp c sb
    | FuncOp f -> funcOp f 1 sb
    | IndexOp i -> indexOp i sb
    | VectorOp v -> vectorOp v sb

/// Emit OpEnvelope - the universal MLIR operation representation
/// Format: %r = "dialect.op"(%a, %b) <{attr = val}> ({ regions }) : (T1, T2) -> (R)
and opEnvelope (env: OpEnvelope) (indent: int) (sb: StringBuilder) : unit =
    // Results (if any)
    if not (List.isEmpty env.Results) then
        env.Results |> List.iteri (fun i r ->
            if i > 0 then sb.Append(", ") |> ignore
            ssa r.SSA sb)
        sb.Append(" = ") |> ignore

    // Operation name in quotes: "dialect.operation"
    sb.Append("\"").Append(env.Dialect).Append(".").Append(env.Operation).Append("\"") |> ignore

    // Operands: (%a, %b, ...)
    sb.Append("(") |> ignore
    env.Operands |> List.iteri (fun i op ->
        if i > 0 then sb.Append(", ") |> ignore
        ssa op.SSA sb)
    sb.Append(")") |> ignore

    // Attributes: <{name = value, ...}> (only if non-empty)
    if not (Map.isEmpty env.Attributes) then
        sb.Append(" <{") |> ignore
        env.Attributes |> Map.toList |> List.iteri (fun i (k, v) ->
            if i > 0 then sb.Append(", ") |> ignore
            sb.Append(k).Append(" = ") |> ignore
            attrValue v sb)
        sb.Append("}>") |> ignore

    // Regions: ({ ... }) (only if non-empty)
    if not (List.isEmpty env.Regions) then
        sb.Append(" (") |> ignore
        env.Regions |> List.iteri (fun i r ->
            if i > 0 then sb.Append(", ") |> ignore
            region r (indent + 1) sb)
        sb.Append(")") |> ignore

    // Type signature: : (operand types) -> (result types)
    sb.Append(" : (") |> ignore
    env.Operands |> List.iteri (fun i op ->
        if i > 0 then sb.Append(", ") |> ignore
        mlirType op.Type sb)
    sb.Append(") -> ") |> ignore
    if List.isEmpty env.Results then
        sb.Append("()") |> ignore
    elif List.length env.Results = 1 then
        mlirType (List.head env.Results).Type sb
    else
        sb.Append("(") |> ignore
        env.Results |> List.iteri (fun i r ->
            if i > 0 then sb.Append(", ") |> ignore
            mlirType r.Type sb)
        sb.Append(")") |> ignore

/// Emit operation with indentation
let mlirOpIndented (indent: int) (op: MLIROp) (sb: StringBuilder) : unit =
    for _ in 1..indent do sb.Append("  ") |> ignore
    mlirOp op sb
    sb.AppendLine() |> ignore

// ═══════════════════════════════════════════════════════════════════════════
// CONVENIENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an operation to string (for debugging/output)
/// This is the ONLY place where structured types become strings
let opToString (op: MLIROp) : string =
    let sb = StringBuilder()
    mlirOp op sb
    sb.ToString()

/// Convert a list of operations to string
let opsToString (ops: MLIROp list) : string =
    let sb = StringBuilder()
    for op in ops do
        sb.Append("    ") |> ignore
        mlirOp op sb
        sb.AppendLine() |> ignore
    sb.ToString()

/// Convert a type to string
let typeToString (t: MLIRType) : string =
    let sb = StringBuilder()
    mlirType t sb
    sb.ToString()

/// Convert an SSA to string
let ssaToString (s: SSA) : string =
    let sb = StringBuilder()
    ssa s sb
    sb.ToString()
