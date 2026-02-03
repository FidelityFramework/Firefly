/// MLIR Serialization
///
/// Converts structured MLIR types and operations to MLIR text format.
/// This is the ONLY place where sprintf is used for MLIR text generation.
/// All upstream code (witnesses, patterns, elements) works with structured MLIROp.
module Alex.Dialects.Core.Serialize

open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// TYPE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert IntWidth to MLIR type string
let intWidthToString (width: IntWidth) : string =
    match width with
    | I1 -> "i1"
    | I8 -> "i8"
    | I16 -> "i16"
    | I32 -> "i32"
    | I64 -> "i64"

/// Convert FloatWidth to MLIR type string
let floatWidthToString (width: FloatWidth) : string =
    match width with
    | F32 -> "f32"
    | F64 -> "f64"

/// Convert MLIRType to MLIR text format string
let rec typeToString (ty: MLIRType) : string =
    match ty with
    | TInt width -> intWidthToString width
    | TFloat width -> floatWidthToString width
    | TFunc (paramTypes, retType) ->
        let paramStrs = paramTypes |> List.map typeToString |> String.concat ", "
        sprintf "(%s) -> %s" paramStrs (typeToString retType)
    | TMemRef elemTy ->
        sprintf "memref<?x%s>" (typeToString elemTy)
    | TMemRefStatic (size, elemTy) ->
        sprintf "memref<%dx%s>" size (typeToString elemTy)
    | TMemRefScalar elemTy ->
        // Scalar memref (0D) is represented as 1-element static memref in MLIR
        sprintf "memref<1x%s>" (typeToString elemTy)
    | TVector (count, elemTy) ->
        sprintf "vector<%dx%s>" count (typeToString elemTy)
    | TIndex -> "index"
    | TUnit -> "i32"  // Unit represented as i32 (value 0)
    | TError msg -> sprintf "<<ERROR: %s>>" msg

// ═══════════════════════════════════════════════════════════════════════════
// SSA SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert SSA to MLIR SSA value string
let ssaToString (ssa: SSA) : string =
    match ssa with
    | V n -> sprintf "%%v%d" n
    | Arg n -> sprintf "%%arg%d" n

/// Convert Val (SSA + type) to typed SSA value string
let valToString (v: Val) : string =
    sprintf "%s : %s" (ssaToString v.SSA) (typeToString v.Type)

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert ICmpPred to MLIR predicate string
let icmpPredToString (pred: ICmpPred) : string =
    match pred with
    | ICmpPred.Eq -> "eq"
    | ICmpPred.Ne -> "ne"
    | ICmpPred.Slt -> "slt"
    | ICmpPred.Sle -> "sle"
    | ICmpPred.Sgt -> "sgt"
    | ICmpPred.Sge -> "sge"
    | ICmpPred.Ult -> "ult"
    | ICmpPred.Ule -> "ule"
    | ICmpPred.Ugt -> "ugt"
    | ICmpPred.Uge -> "uge"

/// Convert FCmpPred to MLIR predicate string
let fcmpPredToString (pred: FCmpPred) : string =
    match pred with
    | OEq -> "oeq"
    | OGt -> "ogt"
    | OGe -> "oge"
    | OLt -> "olt"
    | OLe -> "ole"
    | ONe -> "one"
    | Ord -> "ord"
    | UEq -> "ueq"
    | UGt -> "ugt"
    | UGe -> "uge"
    | ULt -> "ult"
    | ULe -> "ule"
    | UNe -> "une"
    | Uno -> "uno"
    | AlwaysFalse -> "false"
    | AlwaysTrue -> "true"

/// Serialize ArithOp to MLIR text
let arithOpToString (op: ArithOp) : string =
    match op with
    | ConstI (result, value, ty) ->
        sprintf "%s = arith.constant %d : %s" (ssaToString result) value (typeToString ty)
    | ConstF (result, value, ty) ->
        sprintf "%s = arith.constant %f : %s" (ssaToString result) value (typeToString ty)
    | AddI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.addi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | SubI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.subi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | MulI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.muli %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | DivSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | DivUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | RemSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.remsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | RemUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.remui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | AddF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.addf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | SubF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.subf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | MulF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.mulf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | DivF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | CmpI (result, pred, lhs, rhs, ty) ->
        sprintf "%s = arith.cmpi %s, %s, %s : %s" (ssaToString result) (icmpPredToString pred) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | CmpF (result, pred, lhs, rhs, ty) ->
        sprintf "%s = arith.cmpf %s, %s, %s : %s" (ssaToString result) (fcmpPredToString pred) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | ExtSI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.extsi %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString srcTy) (typeToString destTy)
    | ExtUI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.extui %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString srcTy) (typeToString destTy)
    | TruncI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.trunci %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString srcTy) (typeToString destTy)
    | SIToFP (result, value, srcTy, destTy) ->
        sprintf "%s = arith.sitofp %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString srcTy) (typeToString destTy)
    | FPToSI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.fptosi %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString srcTy) (typeToString destTy)
    | Select (result, cond, trueVal, falseVal, ty) ->
        sprintf "%s = arith.select %s, %s, %s : %s"
            (ssaToString result) (ssaToString cond) (ssaToString trueVal) (ssaToString falseVal) (typeToString ty)
    // Bitwise operations (migrated from LLVM dialect)
    | AndI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.andi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | OrI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.ori %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | XorI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.xori %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | ShLI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shli %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | ShRUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shrui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)
    | ShRSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shrsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString ty)

/// Serialize MemRefOp to MLIR text
let memrefOpToString (op: MemRefOp) : string =
    match op with
    | MemRefOp.Load (result, memref, indices, ty) ->
        // Build indices string
        let indicesStr = if List.isEmpty indices then "" else sprintf "[%s]" (indices |> List.map ssaToString |> String.concat ", ")
        // ty is element type - construct proper memref type for serialization
        // Single index → 1-element static memref (matches pAlloca pattern)
        // Multiple indices → dynamic memref
        let memrefType =
            match indices.Length with
            | 0 | 1 -> TMemRefStatic (1, ty)
            | _ -> TMemRef ty
        sprintf "%s = memref.load %s%s : %s"
            (ssaToString result) (ssaToString memref) indicesStr (typeToString memrefType)
    | MemRefOp.Store (value, memref, indices, ty) ->
        // Build indices string
        let indicesStr = if List.isEmpty indices then "" else sprintf "[%s]" (indices |> List.map ssaToString |> String.concat ", ")
        // ty is element type - construct proper memref type for serialization
        // Single index → 1-element static memref (matches pAlloca pattern)
        // Multiple indices → dynamic memref
        let memrefType =
            match indices.Length with
            | 0 | 1 -> TMemRefStatic (1, ty)
            | _ -> TMemRef ty
        sprintf "memref.store %s, %s%s : %s"
            (ssaToString value) (ssaToString memref) indicesStr (typeToString memrefType)
    | MemRefOp.Alloca (result, memrefType, alignmentOpt) ->
        match alignmentOpt with
        | Some alignment ->
            sprintf "%s = memref.alloca() {alignment = %d : i64} : %s"
                (ssaToString result) alignment (typeToString memrefType)
        | None ->
            sprintf "%s = memref.alloca() : %s" (ssaToString result) (typeToString memrefType)
    | MemRefOp.SubView (result, source, offsets, resultType) ->
        let offsetsStr = offsets |> List.map ssaToString |> String.concat ", "
        sprintf "%s = memref.subview %s[%s] : %s"
            (ssaToString result) (ssaToString source) offsetsStr (typeToString resultType)
    | MemRefOp.ExtractBasePtr (result, memref, ty) ->
        // Extract pointer as platform word (index type) - PORTABLE!
        // This replaces the old LLVM-specific unrealized_conversion_cast
        // Returns index (platform word size), caller must cast to target type if needed
        sprintf "%s = memref.extract_aligned_pointer_as_index %s : %s -> index"
            (ssaToString result) (ssaToString memref) (typeToString ty)
    | MemRefOp.GetGlobal (result, globalName, memrefType) ->
        // memref.get_global @symbol_name : memref<...>
        sprintf "%s = memref.get_global @%s : %s"
            (ssaToString result) globalName (typeToString memrefType)
    | MemRefOp.Dim (result, memref, index, memrefType) ->
        // memref.dim %memref, %index : memref<...>
        sprintf "%s = memref.dim %s, %s : %s"
            (ssaToString result) (ssaToString memref) (ssaToString index) (typeToString memrefType)
    | MemRefOp.Cast (result, source, srcType, destType) ->
        // memref.cast %source : srcType to destType
        sprintf "%s = memref.cast %s : %s to %s"
            (ssaToString result) (ssaToString source) (typeToString srcType) (typeToString destType)

/// Serialize top-level MLIROp to MLIR text
let rec opToString (op: MLIROp) : string =
    match op with
    | MLIROp.ArithOp aop -> arithOpToString aop
    | MLIROp.MemRefOp mop -> memrefOpToString mop
    | MLIROp.FuncOp fop ->
        match fop with
        | FuncDef (name, args, retTy, body, _visibility) ->
            printfn "[DEBUG] Serializing FuncDef %s with %d body ops" name (List.length body)
            let argsStr = args |> List.map (fun (ssa, ty) -> sprintf "%s: %s" (ssaToString ssa) (typeToString ty)) |> String.concat ", "
            let bodyStr = body |> List.map opToString |> String.concat "\n    "
            sprintf "func.func @%s(%s) -> %s {\n    %s\n}" name argsStr (typeToString retTy) bodyStr
        | FuncDecl (name, paramTypes, retTy, _visibility) ->
            let paramsStr = paramTypes |> List.map typeToString |> String.concat ", "
            sprintf "func.func private @%s(%s) -> %s" name paramsStr (typeToString retTy)
        | ExternDecl (name, paramTypes, retTy) ->
            let paramsStr = paramTypes |> List.map typeToString |> String.concat ", "
            sprintf "func.func private @%s(%s) -> %s" name paramsStr (typeToString retTy)
        | FuncCall (resultOpt, funcName, args, retTy) ->
            let argSSAs = args |> List.map (fun v -> ssaToString v.SSA) |> String.concat ", "
            let argTypes = args |> List.map (fun v -> typeToString v.Type) |> String.concat ", "
            match resultOpt with
            | Some result -> sprintf "%s = func.call @%s(%s) : (%s) -> %s" (ssaToString result) funcName argSSAs argTypes (typeToString retTy)
            | None -> sprintf "func.call @%s(%s) : (%s) -> %s" funcName argSSAs argTypes (typeToString retTy)
        | FuncCallIndirect (resultOpt, callee, args, retTy) ->
            let argSSAs = args |> List.map (fun v -> ssaToString v.SSA) |> String.concat ", "
            let argTypes = args |> List.map (fun v -> typeToString v.Type) |> String.concat ", "
            match resultOpt with
            | Some result -> sprintf "%s = func.call_indirect %s(%s) : (%s) -> %s" (ssaToString result) (ssaToString callee) argSSAs argTypes (typeToString retTy)
            | None -> sprintf "func.call_indirect %s(%s) : (%s) -> %s" (ssaToString callee) argSSAs argTypes (typeToString retTy)
        | FuncConstant (result, funcName, funcTy) ->
            sprintf "%s = func.constant @%s : %s" (ssaToString result) funcName (typeToString funcTy)
        | Return (valueOpt, tyOpt) ->
            match valueOpt, tyOpt with
            | Some value, Some ty -> sprintf "func.return %s : %s" (ssaToString value) (typeToString ty)
            | Some value, None -> sprintf "func.return %s" (ssaToString value)
            | None, _ -> "func.return"
    | MLIROp.GlobalString (name, content, byteLength) ->
        // Escape the string content for MLIR
        let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\t", "\\t")
        // Add @ prefix for MLIR global symbol naming convention
        // Emit memref.global (portable MLIR) instead of llvm.mlir.global
        // Use dense<...> for array literal initialization
        let bytes = System.Text.Encoding.UTF8.GetBytes(content)
        let denseStr = bytes |> Array.map (sprintf "%d") |> String.concat ", "
        sprintf "memref.global \"private\" constant @%s : memref<%dxi8> = dense<[%s]>" name byteLength denseStr
    | MLIROp.IndexOp iop ->
        match iop with
        | IndexOp.IndexCastS (result, operand, destTy) ->
            sprintf "%s = index.casts %s : index to %s" (ssaToString result) (ssaToString operand) (typeToString destTy)
        | IndexOp.IndexCastU (result, operand, destTy) ->
            sprintf "%s = index.castu %s : index to %s" (ssaToString result) (ssaToString operand) (typeToString destTy)
        | _ ->
            sprintf "// TODO: Serialize IndexOp %A" iop
    | MLIROp.SCFOp scfOp ->
        match scfOp with
        | SCFOp.While (condOps, bodyOps) ->
            // scf.while with condition and body regions
            let condStr = condOps |> List.map opToString |> String.concat "\n      "
            let bodyStr = bodyOps |> List.map opToString |> String.concat "\n      "
            sprintf "scf.while : () -> () {\n      %s\n    } do {\n      %s\n    }" condStr bodyStr
        | SCFOp.If (cond, thenOps, elseOpsOpt) ->
            let thenStr = thenOps |> List.map opToString |> String.concat "\n      "
            match elseOpsOpt with
            | Some elseOps ->
                let elseStr = elseOps |> List.map opToString |> String.concat "\n      "
                sprintf "scf.if %s {\n      %s\n    } else {\n      %s\n    }" (ssaToString cond) thenStr elseStr
            | None ->
                sprintf "scf.if %s {\n      %s\n    }" (ssaToString cond) thenStr
        | SCFOp.For (lower, upper, step, bodyOps) ->
            let bodyStr = bodyOps |> List.map opToString |> String.concat "\n      "
            sprintf "scf.for %s = %s to %s step %s {\n      %s\n    }" 
                (ssaToString lower) (ssaToString upper) (ssaToString step) (ssaToString step) bodyStr
        | SCFOp.Yield ssas ->
            let ssaStr = ssas |> List.map ssaToString |> String.concat ", "
            sprintf "scf.yield %s" ssaStr
        | SCFOp.Condition (cond, args) ->
            let argsStr = args |> List.map ssaToString |> String.concat ", "
            sprintf "scf.condition(%s) %s" (ssaToString cond) argsStr
    | _ ->
        // For now, return placeholder for unimplemented operations (CFOp, VectorOp, Block, Region)
        sprintf "// TODO: Serialize %A" op

/// Serialize a list of operations with proper indentation
let opsToString (ops: MLIROp list) (indent: string) : string =
    ops
    |> List.map opToString
    |> List.map (fun line -> indent + line)
    |> String.concat "\n"

/// Serialize a complete MLIR module
let moduleToString (moduleName: string) (ops: MLIROp list) : string =
    let opsText = opsToString ops "  "
    sprintf "module @%s {\n%s\n}" moduleName opsText
