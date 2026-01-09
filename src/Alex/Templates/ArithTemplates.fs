/// Arithmetic Templates - arith dialect operations
///
/// Templates for MLIR arith dialect operations:
/// - Binary integer/float arithmetic
/// - Comparison operations
/// - Unary operations
/// - Constants
module Alex.Templates.ArithTemplates

open Alex.Templates.TemplateTypes

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer addition: arith.addi
let addI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.addi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Integer subtraction: arith.subi
let subI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.subi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Integer multiplication: arith.muli
let mulI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.muli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Signed integer division: arith.divsi
let divSI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.divsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Signed integer remainder: arith.remsi
let remSI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.remsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Bitwise AND: arith.andi
let andI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.andi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Bitwise OR: arith.ori
let orI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.ori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Bitwise XOR: arith.xori
let xorI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.xori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Shift left: arith.shli
let shlI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.shli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Arithmetic shift right (signed): arith.shrsi
let shrSI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.shrsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Comparison predicate type
type CmpIPredicate = 
    | Eq | Ne | Slt | Sle | Sgt | Sge
    | Ult | Ule | Ugt | Uge

/// Convert predicate to MLIR string
let cmpIPredicateStr = function
    | Eq -> "eq" | Ne -> "ne"
    | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
    | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

/// Integer comparison: arith.cmpi
let cmpI predicate = simple "arith" "compare" (fun (p: CompareParams) ->
    sprintf "%s = arith.cmpi %s, %s, %s : %s" p.Result (cmpIPredicateStr predicate) p.Lhs p.Rhs p.Type)

/// Map CompareOp to signed predicate
let signedPredicate (op: CompareOp) : CmpIPredicate =
    match op with
    | CompareOp.Lt -> Slt 
    | CompareOp.Le -> Sle 
    | CompareOp.Gt -> Sgt 
    | CompareOp.Ge -> Sge 
    | CompareOp.Eq -> CmpIPredicate.Eq 
    | CompareOp.Ne -> CmpIPredicate.Ne

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Float addition: arith.addf
let addF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.addf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float subtraction: arith.subf
let subF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.subf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float multiplication: arith.mulf
let mulF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.mulf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float division: arith.divf
let divF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.divf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Float comparison predicate (ordered)
type CmpFPredicate =
    | OEq | ONE | OLt | OLe | OGt | OGe
    | UEq | UNE | ULt | ULe | UGt | UGe
    | Ord | Uno  // ordered / unordered

/// Convert predicate to MLIR string
let cmpFPredicateStr = function
    | OEq -> "oeq" | ONE -> "one"
    | OLt -> "olt" | OLe -> "ole" | OGt -> "ogt" | OGe -> "oge"
    | UEq -> "ueq" | UNE -> "une"
    | ULt -> "ult" | ULe -> "ule" | UGt -> "ugt" | UGe -> "uge"
    | Ord -> "ord" | Uno -> "uno"

/// Float comparison: arith.cmpf
let cmpF predicate = simple "arith" "compare" (fun (p: CompareParams) ->
    sprintf "%s = arith.cmpf %s, %s, %s : %s" p.Result (cmpFPredicateStr predicate) p.Lhs p.Rhs p.Type)

/// Map CompareOp to ordered float predicate
let orderedPredicate (op: CompareOp) : CmpFPredicate =
    match op with
    | CompareOp.Lt -> OLt 
    | CompareOp.Le -> OLe 
    | CompareOp.Gt -> OGt 
    | CompareOp.Ge -> OGe 
    | CompareOp.Eq -> OEq 
    | CompareOp.Ne -> ONE

// ═══════════════════════════════════════════════════════════════════════════
// UNARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer negation (0 - x)
/// Note: Requires constant 0 first, then subi
let negI = simple "arith" "unary" (fun (p: UnaryOpParams) ->
    // This returns just the subi; caller must emit the constant 0 separately
    sprintf "%s = arith.subi %%zero, %s : %s" p.Result p.Operand p.Type)

/// Float negation: arith.negf
let negF = simple "arith" "unary" (fun (p: UnaryOpParams) ->
    sprintf "%s = arith.negf %s : %s" p.Result p.Operand p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer constant: arith.constant
let constantI = simple "arith" "constant" (fun (p: ConstantParams) ->
    sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type)

/// Float constant: arith.constant
let constantF = simple "arith" "constant" (fun (p: ConstantParams) ->
    sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type)

/// Boolean true constant
let constantTrue = simple "arith" "constant" (fun result ->
    sprintf "%s = arith.constant true" result)

/// Boolean false constant
let constantFalse = simple "arith" "constant" (fun result ->
    sprintf "%s = arith.constant false" result)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Sign extend: arith.extsi
let extSI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extsi %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Zero extend: arith.extui
let extUI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extui %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Truncate: arith.trunci
let truncI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.trunci %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float to signed int: arith.fptosi
let fpToSI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.fptosi %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Signed int to float: arith.sitofp
let siToFP = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.sitofp %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float extend: arith.extf
let extF = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extf %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float truncate: arith.truncf
let truncF = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.truncf %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCH HELPERS - Select template based on witness
// ═══════════════════════════════════════════════════════════════════════════

/// Select integer binary template from BinaryArithOp witness
let intBinaryTemplate (op: BinaryArithOp) =
    match op with
    | Add -> addI
    | Sub -> subI
    | Mul -> mulI
    | Div -> divSI
    | Mod -> remSI
    | BitAnd -> andI
    | BitOr -> orI
    | BitXor -> xorI
    | ShiftLeft -> shlI
    | ShiftRight -> shrSI

/// Select float binary template from BinaryArithOp witness
let floatBinaryTemplate (op: BinaryArithOp) =
    match op with
    | Add -> addF
    | Sub -> subF
    | Mul -> mulF
    | Div -> divF
    | _ -> failwith "Unsupported float operation"

/// Select integer comparison template from CompareOp witness
let intCompareTemplate op = cmpI (signedPredicate op)

/// Select float comparison template from CompareOp witness
let floatCompareTemplate op = cmpF (orderedPredicate op)


// ═══════════════════════════════════════════════════════════════════════════
// QUOTATION-BASED TEMPLATES (Phase 5: Multi-Target Generation)
// ═══════════════════════════════════════════════════════════════════════════
//
// These MLIRTemplate definitions use F# quotations for inspectability.
// The quotation structure can be analyzed for:
// - MLIR text generation (current)
// - TableGen definitions (future)
// - C++ code generation (future)
// - Documentation extraction (future)

/// Quotation-based templates organized by category
module Quot =
    
    /// Integer binary operations with full metadata
    module IntBinary =
        let addI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.addi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "addi"
            IsTerminator = false
            Category = "binary"
        }
        
        let subI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.subi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "subi"
            IsTerminator = false
            Category = "binary"
        }
        
        let mulI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.muli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "muli"
            IsTerminator = false
            Category = "binary"
        }
        
        let divSI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.divsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "divsi"
            IsTerminator = false
            Category = "binary"
        }
        
        let remSI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.remsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "remsi"
            IsTerminator = false
            Category = "binary"
        }
    
    /// Integer bitwise operations
    module IntBitwise =
        let andI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.andi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "andi"
            IsTerminator = false
            Category = "bitwise"
        }
        
        let orI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.ori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "ori"
            IsTerminator = false
            Category = "bitwise"
        }
        
        let xorI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.xori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "xori"
            IsTerminator = false
            Category = "bitwise"
        }
        
        let shlI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.shli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "shli"
            IsTerminator = false
            Category = "bitwise"
        }
        
        let shrSI : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.shrsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "shrsi"
            IsTerminator = false
            Category = "bitwise"
        }
    
    /// Type conversion operations
    module Conversion =
        let extSI : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.extsi %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "extsi"
            IsTerminator = false
            Category = "conversion"
        }
        
        let extUI : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.extui %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "extui"
            IsTerminator = false
            Category = "conversion"
        }
        
        let truncI : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.trunci %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "trunci"
            IsTerminator = false
            Category = "conversion"
        }
        
        /// Extend float precision (f32 to f64)
        let extF : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.extf %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "extf"
            IsTerminator = false
            Category = "conversion"
        }
        
        /// Truncate float precision (f64 to f32)
        let truncF : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.truncf %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "truncf"
            IsTerminator = false
            Category = "conversion"
        }
        
        /// Signed integer to float (int -> float)
        let siToFP : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.sitofp %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "sitofp"
            IsTerminator = false
            Category = "conversion"
        }
        
        /// Float to signed integer (float -> int)
        let fpToSI : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.fptosi %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "fptosi"
            IsTerminator = false
            Category = "conversion"
        }

        /// Unsigned integer to float (uint -> float)
        let uiToFP : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.uitofp %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "uitofp"
            IsTerminator = false
            Category = "conversion"
        }

        /// Float to unsigned integer (float -> uint)
        let fpToUI : MLIRTemplate<ConversionParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.fptoui %s : %s to %s" p.Result p.Operand p.FromType p.ToType @>
            Dialect = "arith"
            OpName = "fptoui"
            IsTerminator = false
            Category = "conversion"
        }

    /// Constants
    module Constant =
        let intConst : MLIRTemplate<ConstantParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type @>
            Dialect = "arith"
            OpName = "constant"
            IsTerminator = false
            Category = "constant"
        }
        
        /// Float constant uses the same format as intConst but semantically typed
        let floatConst : MLIRTemplate<ConstantParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type @>
            Dialect = "arith"
            OpName = "constant"
            IsTerminator = false
            Category = "constant"
        }
    
    /// Float binary operations
    module FloatBinary =
        let addF : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.addf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "addf"
            IsTerminator = false
            Category = "binary"
        }
        
        let subF : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.subf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "subf"
            IsTerminator = false
            Category = "binary"
        }
        
        let mulF : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.mulf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "mulf"
            IsTerminator = false
            Category = "binary"
        }
        
        let divF : MLIRTemplate<BinaryOpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.divf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "divf"
            IsTerminator = false
            Category = "binary"
        }
    
    /// Comparison operations (both integer and float)
    module Compare =
        /// Parameters for comparison operations (result is always i1)
        type CmpParams = {
            Result: string
            Predicate: string  // "slt", "sle", "olt", "oeq", etc.
            Lhs: string
            Rhs: string
            Type: string
        }
        
        /// Integer comparison
        let cmpI : MLIRTemplate<CmpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.cmpi %s, %s, %s : %s" p.Result p.Predicate p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "cmpi"
            IsTerminator = false
            Category = "compare"
        }
        
        /// Float comparison
        let cmpF : MLIRTemplate<CmpParams> = {
            Quotation = <@ fun p -> sprintf "%s = arith.cmpf %s, %s, %s : %s" p.Result p.Predicate p.Lhs p.Rhs p.Type @>
            Dialect = "arith"
            OpName = "cmpf"
            IsTerminator = false
            Category = "compare"
        }
