/// Application Witness - Witness function applications to MLIR
///
/// Observes application PSG nodes and generates corresponding MLIR calls.
/// Handles intrinsics, platform bindings, primitive ops, and curried calls.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.NativeGlobals
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
open Alex.Patterns.SemanticPatterns
open Alex.Templates.TemplateTypes
open Alex.Templates.ArithTemplates
open Alex.Templates.MemoryTemplates

module LLVMTemplates = Alex.Templates.LLVMTemplates

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE OPERATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Try to emit a binary primitive operation (arithmetic, comparison, bitwise)
/// Uses quotation-based templates for principled MLIR generation.
let tryEmitPrimitiveBinaryOp (opName: string) (arg1SSA: string) (arg1Type: string) (arg2SSA: string) (arg2Type: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    if arg1Type <> arg2Type then None
    elif not (isIntegerType arg1Type || isFloatType arg1Type) then None
    else
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let isInt = isIntegerType arg1Type
        
        // Helper to emit binary ops via templates
        let emitBinaryOp (template: MLIRTemplate<BinaryOpParams>) resultType =
            let binaryParams: BinaryOpParams = { Result = resultSSA; Lhs = arg1SSA; Rhs = arg2SSA; Type = arg1Type }
            let opText = render template binaryParams
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Serialize.deserializeType resultType) zipper'
            Some (resultSSA, resultType, zipper'')
        
        // Helper to emit comparison ops via templates
        let emitCmpOp pred isCmpF =
            let cmpParams: Quot.Compare.CmpParams = { Result = resultSSA; Predicate = pred; Lhs = arg1SSA; Rhs = arg2SSA; Type = arg1Type }
            let template = if isCmpF then Quot.Compare.cmpF else Quot.Compare.cmpI
            let opText = render template cmpParams
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Integer I1) zipper'
            Some (resultSSA, "i1", zipper'')
        
        // Select template based on operation and type
        match opName, isInt with
        // Integer arithmetic operations
        | "op_Addition", true -> emitBinaryOp Quot.IntBinary.addI arg1Type
        | "op_Subtraction", true -> emitBinaryOp Quot.IntBinary.subI arg1Type
        | "op_Multiply", true -> emitBinaryOp Quot.IntBinary.mulI arg1Type
        | "op_Division", true -> emitBinaryOp Quot.IntBinary.divSI arg1Type
        | "op_Modulus", true -> emitBinaryOp Quot.IntBinary.remSI arg1Type
        // Float arithmetic operations
        | "op_Addition", false -> emitBinaryOp Quot.FloatBinary.addF arg1Type
        | "op_Subtraction", false -> emitBinaryOp Quot.FloatBinary.subF arg1Type
        | "op_Multiply", false -> emitBinaryOp Quot.FloatBinary.mulF arg1Type
        | "op_Division", false -> emitBinaryOp Quot.FloatBinary.divF arg1Type
        // Integer comparisons
        | "op_LessThan", true -> emitCmpOp "slt" false
        | "op_LessThanOrEqual", true -> emitCmpOp "sle" false
        | "op_GreaterThan", true -> emitCmpOp "sgt" false
        | "op_GreaterThanOrEqual", true -> emitCmpOp "sge" false
        | "op_Equality", true -> emitCmpOp "eq" false
        | "op_Inequality", true -> emitCmpOp "ne" false
        // Float comparisons
        | "op_LessThan", false -> emitCmpOp "olt" true
        | "op_LessThanOrEqual", false -> emitCmpOp "ole" true
        | "op_GreaterThan", false -> emitCmpOp "ogt" true
        | "op_GreaterThanOrEqual", false -> emitCmpOp "oge" true
        | "op_Equality", false -> emitCmpOp "oeq" true
        | "op_Inequality", false -> emitCmpOp "one" true
        // Bitwise operations (int only)
        | "op_BitwiseAnd", true -> emitBinaryOp Quot.IntBitwise.andI arg1Type
        | "op_BitwiseOr", true -> emitBinaryOp Quot.IntBitwise.orI arg1Type
        | "op_ExclusiveOr", true -> emitBinaryOp Quot.IntBitwise.xorI arg1Type
        | "op_LeftShift", true -> emitBinaryOp Quot.IntBitwise.shlI arg1Type
        | "op_RightShift", true -> emitBinaryOp Quot.IntBitwise.shrSI arg1Type
        | _ -> None

/// Try to emit a unary primitive operation
/// Uses quotation-based templates for principled MLIR generation.
let tryEmitPrimitiveUnaryOp (opName: string) (argSSA: string) (argType: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    
    // Use active pattern for type-safe unary operation classification
    match opName with
    | UnaryOp BoolNot when isBoolType argType ->
        // Boolean NOT: XOR with true
        let trueSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = trueSSA; Value = "true"; Type = "i1" }
        let trueOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult trueOp trueSSA (Integer I1) zipper''
        let xorParams: BinaryOpParams = { Result = resultSSA; Lhs = argSSA; Rhs = trueSSA; Type = "i1" }
        let notOp = render Quot.IntBitwise.xorI xorParams
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Integer I1) zipper'''
        Some (resultSSA, "i1", zipper4)
        
    | UnaryOp IntNegate when isIntegerType argType ->
        // Integer negation: 0 - x
        let zeroSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = zeroSSA; Value = "0"; Type = argType }
        let zeroOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult zeroOp zeroSSA (Serialize.deserializeType argType) zipper''
        let subParams: BinaryOpParams = { Result = resultSSA; Lhs = zeroSSA; Rhs = argSSA; Type = argType }
        let negOp = render Quot.IntBinary.subI subParams
        let zipper4 = MLIRZipper.witnessOpWithResult negOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
        
    | UnaryOp BitwiseNot when isIntegerType argType ->
        // Bitwise NOT: XOR with -1 (all ones)
        let onesSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = onesSSA; Value = "-1"; Type = argType }
        let onesOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult onesOp onesSSA (Serialize.deserializeType argType) zipper''
        let xorParams: BinaryOpParams = { Result = resultSSA; Lhs = argSSA; Rhs = onesSSA; Type = argType }
        let notOp = render Quot.IntBitwise.xorI xorParams
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
        
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// FORMAT INTRINSIC HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit inline integer-to-string conversion using scf.while loop
/// Converts an integer value to a fat string (!llvm.struct<(ptr, i64)>)
/// Algorithm: extract digits via % 10, store backwards in buffer, handle sign
/// Note: Handles i32 input by extending to i64 first
let emitIntToString (intSSA: string) (intType: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // If input is i32, extend to i64 first
    let int64SSA, z0 =
        if intType = "i32" then
            let extSSA, z = MLIRZipper.yieldSSA zipper
            let extParams : ConversionParams = { Result = extSSA; Operand = intSSA; FromType = "i32"; ToType = "i64" }
            let extText = render Quot.Conversion.extSI extParams
            let z' = MLIRZipper.witnessOpWithResult extText extSSA (Integer I64) z
            (extSSA, z')
        else
            (intSSA, zipper)

    // Constants
    let zeroSSA, z1 = MLIRZipper.yieldSSA z0
    let zeroText = sprintf "%s = arith.constant 0 : i64" zeroSSA
    let z2 = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I64) z1

    let oneSSA, z3 = MLIRZipper.yieldSSA z2
    let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
    let z4 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) z3

    let tenSSA, z5 = MLIRZipper.yieldSSA z4
    let tenText = sprintf "%s = arith.constant 10 : i64" tenSSA
    let z6 = MLIRZipper.witnessOpWithResult tenText tenSSA (Integer I64) z5

    let asciiZeroSSA, z7 = MLIRZipper.yieldSSA z6
    let asciiZeroText = sprintf "%s = arith.constant 48 : i8" asciiZeroSSA  // '0' = 48
    let z8 = MLIRZipper.witnessOpWithResult asciiZeroText asciiZeroSSA (Integer I8) z7

    let bufSizeSSA, z9 = MLIRZipper.yieldSSA z8
    let bufSizeText = sprintf "%s = arith.constant 21 : i64" bufSizeSSA  // Max i64 digits + sign
    let z10 = MLIRZipper.witnessOpWithResult bufSizeText bufSizeSSA (Integer I64) z9

    let minusCharSSA, z11 = MLIRZipper.yieldSSA z10
    let minusCharText = sprintf "%s = arith.constant 45 : i8" minusCharSSA  // '-' = 45
    let z12 = MLIRZipper.witnessOpWithResult minusCharText minusCharSSA (Integer I8) z11

    // Allocate buffer
    let bufSSA, z13 = MLIRZipper.yieldSSA z12
    let allocaParams : AllocaParams = { Result = bufSSA; Count = bufSizeSSA; ElementType = "i8" }
    let allocaText = render Quot.Core.alloca allocaParams
    let z14 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer z13

    // Check if negative
    let isNegSSA, z15 = MLIRZipper.yieldSSA z14
    let isNegText = sprintf "%s = arith.cmpi slt, %s, %s : i64" isNegSSA int64SSA zeroSSA
    let z16 = MLIRZipper.witnessOpWithResult isNegText isNegSSA (Integer I1) z15

    // Get absolute value: abs = select(isNeg, -n, n)
    let negatedSSA, z17 = MLIRZipper.yieldSSA z16
    let negatedText = sprintf "%s = arith.subi %s, %s : i64" negatedSSA zeroSSA int64SSA
    let z18 = MLIRZipper.witnessOpWithResult negatedText negatedSSA (Integer I64) z17

    let absSSA, z19 = MLIRZipper.yieldSSA z18
    let absText = sprintf "%s = arith.select %s, %s, %s : i64" absSSA isNegSSA negatedSSA int64SSA
    let z20 = MLIRZipper.witnessOpWithResult absText absSSA (Integer I64) z19

    // Start position at end of buffer (index 20, will write backwards)
    let startPosSSA, z21 = MLIRZipper.yieldSSA z20
    let startPosText = sprintf "%s = arith.constant 20 : i64" startPosSSA
    let z22 = MLIRZipper.witnessOpWithResult startPosText startPosSSA (Integer I64) z21

    // Digit extraction loop using scf.while
    // State: (current_number: i64, current_pos: i64)
    // Guard: number > 0
    // Body: digit = n % 10, store '0' + digit at buf[pos], pos--, n = n / 10
    let resultSSA, z23 = MLIRZipper.yieldSSA z22
    let nArg = sprintf "%s_n" (resultSSA.TrimStart('%'))
    let posArg = sprintf "%s_pos" (resultSSA.TrimStart('%'))

    // Build loop body operations as strings
    let digitSSA, _ = MLIRZipper.yieldSSA z23
    let digitText = sprintf "%s = arith.remsi %%%s, %s : i64" digitSSA nArg tenSSA

    let digit8SSA = sprintf "%s_8" (digitSSA.TrimStart('%'))
    let digit8Text = sprintf "%%%s = arith.trunci %s : i64 to i8" digit8SSA digitSSA

    let charSSA = sprintf "%s_char" (digitSSA.TrimStart('%'))
    let charText = sprintf "%%%s = arith.addi %%%s, %s : i8" charSSA digit8SSA asciiZeroSSA

    let gepSSA = sprintf "%s_gep" (digitSSA.TrimStart('%'))
    let gepText = sprintf "%%%s = llvm.getelementptr %s[%%%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" gepSSA bufSSA posArg

    let storeText = sprintf "llvm.store %%%s, %%%s : i8, !llvm.ptr" charSSA gepSSA

    let newPosSSA = sprintf "%s_newpos" (digitSSA.TrimStart('%'))
    let newPosText = sprintf "%%%s = arith.subi %%%s, %s : i64" newPosSSA posArg oneSSA

    let newNSSA = sprintf "%s_newn" (digitSSA.TrimStart('%'))
    let newNText = sprintf "%%%s = arith.divsi %%%s, %s : i64" newNSSA nArg tenSSA

    // Guard: n > 0
    let condSSA = sprintf "%s_cond" (digitSSA.TrimStart('%'))
    let condText = sprintf "%%%s = arith.cmpi sgt, %%%s, %s : i64" condSSA nArg zeroSSA

    // Emit the scf.while loop
    let whileText =
        sprintf "%s:2 = scf.while (%%%s = %s, %%%s = %s) : (i64, i64) -> (i64, i64) {\n  %s\n  scf.condition(%%%s) %%%s, %%%s : i64, i64\n} do {\n^bb0(%%%s: i64, %%%s: i64):\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  scf.yield %%%s, %%%s : i64, i64\n}"
            resultSSA nArg absSSA posArg startPosSSA
            condText condSSA nArg posArg
            nArg posArg
            digitText digit8Text charText gepText storeText newPosText newNText
            newNSSA newPosSSA

    let z24 = MLIRZipper.witnessOpWithResult whileText resultSSA (Integer I64) z23

    // Get final position from the loop result (second element of tuple)
    let finalPosSSA, z25 = MLIRZipper.yieldSSA z24
    let finalPosText = sprintf "%s = arith.addi %s#1, %s : i64" finalPosSSA resultSSA oneSSA
    let z26 = MLIRZipper.witnessOpWithResult finalPosText finalPosSSA (Integer I64) z25

    // Handle special case: input was 0 (loop didn't execute)
    let wasZeroSSA, z27 = MLIRZipper.yieldSSA z26
    let wasZeroText = sprintf "%s = arith.cmpi eq, %s, %s : i64" wasZeroSSA absSSA zeroSSA
    let z28 = MLIRZipper.witnessOpWithResult wasZeroText wasZeroSSA (Integer I1) z27

    // If zero, write '0' at position 20
    let zeroCharSSA, z29 = MLIRZipper.yieldSSA z28
    let zeroCharText = sprintf "%s = arith.constant 48 : i8" zeroCharSSA  // '0'
    let z30 = MLIRZipper.witnessOpWithResult zeroCharText zeroCharSSA (Integer I8) z29

    // scf.if for zero handling
    let ifZeroText = sprintf "scf.if %s {\n  %%gep_zero = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n  llvm.store %s, %%gep_zero : i8, !llvm.ptr\n}" wasZeroSSA bufSSA startPosSSA zeroCharSSA
    let z31 = MLIRZipper.witnessVoidOp ifZeroText z30

    // Adjust position for zero case
    let adjPosSSA, z32 = MLIRZipper.yieldSSA z31
    let adjPosText = sprintf "%s = arith.select %s, %s, %s : i64" adjPosSSA wasZeroSSA startPosSSA finalPosSSA
    let z33 = MLIRZipper.witnessOpWithResult adjPosText adjPosSSA (Integer I64) z32

    // Handle negative: write '-' at pos-1 if negative, adjust pos
    let negPosSSA, z34 = MLIRZipper.yieldSSA z33
    let negPosText = sprintf "%s = arith.subi %s, %s : i64" negPosSSA adjPosSSA oneSSA
    let z35 = MLIRZipper.witnessOpWithResult negPosText negPosSSA (Integer I64) z34

    let ifNegText = sprintf "scf.if %s {\n  %%gep_neg = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n  llvm.store %s, %%gep_neg : i8, !llvm.ptr\n}" isNegSSA bufSSA negPosSSA minusCharSSA
    let z36 = MLIRZipper.witnessVoidOp ifNegText z35

    let startPtrPosSSA, z37 = MLIRZipper.yieldSSA z36
    let startPtrPosText = sprintf "%s = arith.select %s, %s, %s : i64" startPtrPosSSA isNegSSA negPosSSA adjPosSSA
    let z38 = MLIRZipper.witnessOpWithResult startPtrPosText startPtrPosSSA (Integer I64) z37

    // Get pointer to start of string
    let strPtrSSA, z39 = MLIRZipper.yieldSSA z38
    let strPtrText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" strPtrSSA bufSSA startPtrPosSSA
    let z40 = MLIRZipper.witnessOpWithResult strPtrText strPtrSSA Pointer z39

    // Calculate length: 21 - startPos
    let strLenSSA, z41 = MLIRZipper.yieldSSA z40
    let strLenText = sprintf "%s = arith.subi %s, %s : i64" strLenSSA bufSizeSSA startPtrPosSSA
    let z42 = MLIRZipper.witnessOpWithResult strLenText strLenSSA (Integer I64) z41

    // Build fat string struct
    let undefSSA, z43 = MLIRZipper.yieldSSA z42
    let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
    let z44 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType z43

    let withPtrSSA, z45 = MLIRZipper.yieldSSA z44
    let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = strPtrSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
    let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
    let z46 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType z45

    let fatStrSSA, z47 = MLIRZipper.yieldSSA z46
    let insertLenParams : Quot.Aggregate.InsertParams = { Result = fatStrSSA; Value = strLenSSA; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
    let insertLenText = render Quot.Aggregate.insertValue insertLenParams
    let z48 = MLIRZipper.witnessOpWithResult insertLenText fatStrSSA NativeStrType z47

    (fatStrSSA, z48)

/// Emit inline float-to-string conversion
/// Simplified implementation: converts to integer part + "." + fractional digits
/// Note: If the input is i64 (from union extraction), bitcasts it to f64 first
let emitFloatToString (floatSSA: string) (floatType: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // If the input is i64 (from union extraction), bitcast to f64 first
    let actualFloatSSA, z0 =
        if floatType = "i64" then
            let castSSA, z = MLIRZipper.yieldSSA zipper
            let castText = sprintf "%s = llvm.bitcast %s : i64 to f64" castSSA floatSSA
            let z' = MLIRZipper.witnessOpWithResult castText castSSA (Float F64) z
            (castSSA, z')
        else
            (floatSSA, zipper)

    // Convert float to i64 for integer part
    let intPartSSA, z1 = MLIRZipper.yieldSSA z0
    let intPartText = sprintf "%s = arith.fptosi %s : f64 to i64" intPartSSA actualFloatSSA
    let z2 = MLIRZipper.witnessOpWithResult intPartText intPartSSA (Integer I64) z1

    // Convert integer part back to float
    let intAsFloatSSA, z3 = MLIRZipper.yieldSSA z2
    let intAsFloatText = sprintf "%s = arith.sitofp %s : i64 to f64" intAsFloatSSA intPartSSA
    let z4 = MLIRZipper.witnessOpWithResult intAsFloatText intAsFloatSSA (Float F64) z3

    // Get fractional part: float - intAsFloat
    let fracSSA, z5 = MLIRZipper.yieldSSA z4
    let fracText = sprintf "%s = arith.subf %s, %s : f64" fracSSA actualFloatSSA intAsFloatSSA
    let z6 = MLIRZipper.witnessOpWithResult fracText fracSSA (Float F64) z5

    // Multiply fractional by 1000000 to get 6 decimal digits
    let scaleSSA, z7 = MLIRZipper.yieldSSA z6
    let scaleText = sprintf "%s = arith.constant 1000000.0 : f64" scaleSSA
    let z8 = MLIRZipper.witnessOpWithResult scaleText scaleSSA (Float F64) z7

    let scaledFracSSA, z9 = MLIRZipper.yieldSSA z8
    let scaledFracText = sprintf "%s = arith.mulf %s, %s : f64" scaledFracSSA fracSSA scaleSSA
    let z10 = MLIRZipper.witnessOpWithResult scaledFracText scaledFracSSA (Float F64) z9

    // Get absolute value of fractional (for negative numbers)
    let zeroFloatSSA, z11 = MLIRZipper.yieldSSA z10
    let zeroFloatText = sprintf "%s = arith.constant 0.0 : f64" zeroFloatSSA
    let z12 = MLIRZipper.witnessOpWithResult zeroFloatText zeroFloatSSA (Float F64) z11

    let fracNegSSA, z13 = MLIRZipper.yieldSSA z12
    let fracNegText = sprintf "%s = arith.cmpf olt, %s, %s : f64" fracNegSSA scaledFracSSA zeroFloatSSA
    let z14 = MLIRZipper.witnessOpWithResult fracNegText fracNegSSA (Integer I1) z13

    let negScaledSSA, z15 = MLIRZipper.yieldSSA z14
    let negScaledText = sprintf "%s = arith.negf %s : f64" negScaledSSA scaledFracSSA
    let z16 = MLIRZipper.witnessOpWithResult negScaledText negScaledSSA (Float F64) z15

    let absFracSSA, z17 = MLIRZipper.yieldSSA z16
    let absFracText = sprintf "%s = arith.select %s, %s, %s : f64" absFracSSA fracNegSSA negScaledSSA scaledFracSSA
    let z18 = MLIRZipper.witnessOpWithResult absFracText absFracSSA (Float F64) z17

    let fracIntSSA, z19 = MLIRZipper.yieldSSA z18
    let fracIntText = sprintf "%s = arith.fptosi %s : f64 to i64" fracIntSSA absFracSSA
    let z20 = MLIRZipper.witnessOpWithResult fracIntText fracIntSSA (Integer I64) z19

    // Convert integer part to string
    let intStrSSA, z21 = emitIntToString intPartSSA "i64" z20

    // Create decimal point string
    let dotBufSSA, z22 = MLIRZipper.yieldSSA z21
    let oneSSA, z23 = MLIRZipper.yieldSSA z22
    let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
    let z24 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) z23

    let dotAllocaParams : AllocaParams = { Result = dotBufSSA; Count = oneSSA; ElementType = "i8" }
    let dotAllocaText = render Quot.Core.alloca dotAllocaParams
    let z25 = MLIRZipper.witnessOpWithResult dotAllocaText dotBufSSA Pointer z24

    let dotCharSSA, z26 = MLIRZipper.yieldSSA z25
    let dotCharText = sprintf "%s = arith.constant 46 : i8" dotCharSSA  // '.' = 46
    let z27 = MLIRZipper.witnessOpWithResult dotCharText dotCharSSA (Integer I8) z26

    let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" dotCharSSA dotBufSSA
    let z28 = MLIRZipper.witnessVoidOp storeText z27

    // Build dot fat string
    let dotUndefSSA, z29 = MLIRZipper.yieldSSA z28
    let dotUndefText = render Quot.Aggregate.undef {| Result = dotUndefSSA; Type = NativeStrTypeStr |}
    let z30 = MLIRZipper.witnessOpWithResult dotUndefText dotUndefSSA NativeStrType z29

    let dotWithPtrSSA, z31 = MLIRZipper.yieldSSA z30
    let dotInsertPtrParams : Quot.Aggregate.InsertParams = { Result = dotWithPtrSSA; Value = dotBufSSA; Aggregate = dotUndefSSA; Index = 0; AggType = NativeStrTypeStr }
    let dotInsertPtrText = render Quot.Aggregate.insertValue dotInsertPtrParams
    let z32 = MLIRZipper.witnessOpWithResult dotInsertPtrText dotWithPtrSSA NativeStrType z31

    let dotStrSSA, z33 = MLIRZipper.yieldSSA z32
    let dotInsertLenParams : Quot.Aggregate.InsertParams = { Result = dotStrSSA; Value = oneSSA; Aggregate = dotWithPtrSSA; Index = 1; AggType = NativeStrTypeStr }
    let dotInsertLenText = render Quot.Aggregate.insertValue dotInsertLenParams
    let z34 = MLIRZipper.witnessOpWithResult dotInsertLenText dotStrSSA NativeStrType z33

    // Convert fractional part to string
    let fracStrSSA, z35 = emitIntToString fracIntSSA "i64" z34

    // Concatenate: intStr + "." + fracStr
    // First: intStr + "."
    let ptr1SSA, z36 = MLIRZipper.yieldSSA z35
    let extractPtr1Params : Quot.Aggregate.ExtractParams = { Result = ptr1SSA; Aggregate = intStrSSA; Index = 0; AggType = NativeStrTypeStr }
    let extractPtr1 = render Quot.Aggregate.extractValue extractPtr1Params
    let z37 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer z36

    let len1SSA, z38 = MLIRZipper.yieldSSA z37
    let extractLen1Params : Quot.Aggregate.ExtractParams = { Result = len1SSA; Aggregate = intStrSSA; Index = 1; AggType = NativeStrTypeStr }
    let extractLen1 = render Quot.Aggregate.extractValue extractLen1Params
    let z39 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) z38

    let ptr2SSA, z40 = MLIRZipper.yieldSSA z39
    let extractPtr2Params : Quot.Aggregate.ExtractParams = { Result = ptr2SSA; Aggregate = dotStrSSA; Index = 0; AggType = NativeStrTypeStr }
    let extractPtr2 = render Quot.Aggregate.extractValue extractPtr2Params
    let z41 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer z40

    let ptr3SSA, z42 = MLIRZipper.yieldSSA z41
    let extractPtr3Params : Quot.Aggregate.ExtractParams = { Result = ptr3SSA; Aggregate = fracStrSSA; Index = 0; AggType = NativeStrTypeStr }
    let extractPtr3 = render Quot.Aggregate.extractValue extractPtr3Params
    let z43 = MLIRZipper.witnessOpWithResult extractPtr3 ptr3SSA Pointer z42

    let len3SSA, z44 = MLIRZipper.yieldSSA z43
    let extractLen3Params : Quot.Aggregate.ExtractParams = { Result = len3SSA; Aggregate = fracStrSSA; Index = 1; AggType = NativeStrTypeStr }
    let extractLen3 = render Quot.Aggregate.extractValue extractLen3Params
    let z45 = MLIRZipper.witnessOpWithResult extractLen3 len3SSA (Integer I64) z44

    // Total length = len1 + 1 + len3
    let len12SSA, z46 = MLIRZipper.yieldSSA z45
    let len12Params : BinaryOpParams = { Result = len12SSA; Lhs = len1SSA; Rhs = oneSSA; Type = "i64" }
    let len12Text = render Quot.IntBinary.addI len12Params
    let z47 = MLIRZipper.witnessOpWithResult len12Text len12SSA (Integer I64) z46

    let totalLenSSA, z48 = MLIRZipper.yieldSSA z47
    let totalLenParams : BinaryOpParams = { Result = totalLenSSA; Lhs = len12SSA; Rhs = len3SSA; Type = "i64" }
    let totalLenText = render Quot.IntBinary.addI totalLenParams
    let z49 = MLIRZipper.witnessOpWithResult totalLenText totalLenSSA (Integer I64) z48

    // Allocate result buffer
    let resultBufSSA, z50 = MLIRZipper.yieldSSA z49
    let resultAllocaParams : AllocaParams = { Result = resultBufSSA; Count = totalLenSSA; ElementType = "i8" }
    let resultAllocaText = render Quot.Core.alloca resultAllocaParams
    let z51 = MLIRZipper.witnessOpWithResult resultAllocaText resultBufSSA Pointer z50

    // Copy integer part
    let memcpy1Params : Quot.Intrinsic.MemCopyParams = { Dest = resultBufSSA; Src = ptr1SSA; Len = len1SSA }
    let memcpy1 = render Quot.Intrinsic.memcpy memcpy1Params
    let z52 = MLIRZipper.witnessVoidOp memcpy1 z51

    // GEP to dot position
    let dotPosSSA, z53 = MLIRZipper.yieldSSA z52
    let dotGepParams : GepParams = { Result = dotPosSSA; Base = resultBufSSA; Offset = len1SSA; ElementType = "i8" }
    let dotGepText = render Quot.Gep.i64 dotGepParams
    let z54 = MLIRZipper.witnessOpWithResult dotGepText dotPosSSA Pointer z53

    // Copy dot
    let memcpy2Params : Quot.Intrinsic.MemCopyParams = { Dest = dotPosSSA; Src = ptr2SSA; Len = oneSSA }
    let memcpy2 = render Quot.Intrinsic.memcpy memcpy2Params
    let z55 = MLIRZipper.witnessVoidOp memcpy2 z54

    // GEP to frac position
    let fracPosSSA, z56 = MLIRZipper.yieldSSA z55
    let fracGepParams : GepParams = { Result = fracPosSSA; Base = resultBufSSA; Offset = len12SSA; ElementType = "i8" }
    let fracGepText = render Quot.Gep.i64 fracGepParams
    let z57 = MLIRZipper.witnessOpWithResult fracGepText fracPosSSA Pointer z56

    // Copy fractional part
    let memcpy3Params : Quot.Intrinsic.MemCopyParams = { Dest = fracPosSSA; Src = ptr3SSA; Len = len3SSA }
    let memcpy3 = render Quot.Intrinsic.memcpy memcpy3Params
    let z58 = MLIRZipper.witnessVoidOp memcpy3 z57

    // Build result fat string
    let resultUndefSSA, z59 = MLIRZipper.yieldSSA z58
    let resultUndefText = render Quot.Aggregate.undef {| Result = resultUndefSSA; Type = NativeStrTypeStr |}
    let z60 = MLIRZipper.witnessOpWithResult resultUndefText resultUndefSSA NativeStrType z59

    let resultWithPtrSSA, z61 = MLIRZipper.yieldSSA z60
    let resultInsertPtrParams : Quot.Aggregate.InsertParams = { Result = resultWithPtrSSA; Value = resultBufSSA; Aggregate = resultUndefSSA; Index = 0; AggType = NativeStrTypeStr }
    let resultInsertPtrText = render Quot.Aggregate.insertValue resultInsertPtrParams
    let z62 = MLIRZipper.witnessOpWithResult resultInsertPtrText resultWithPtrSSA NativeStrType z61

    let finalStrSSA, z63 = MLIRZipper.yieldSSA z62
    let finalInsertLenParams : Quot.Aggregate.InsertParams = { Result = finalStrSSA; Value = totalLenSSA; Aggregate = resultWithPtrSSA; Index = 1; AggType = NativeStrTypeStr }
    let finalInsertLenText = render Quot.Aggregate.insertValue finalInsertLenParams
    let z64 = MLIRZipper.witnessOpWithResult finalInsertLenText finalStrSSA NativeStrType z63

    (finalStrSSA, z64)

// ═══════════════════════════════════════════════════════════════════════════
// PARSE INTRINSIC HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit inline string-to-integer conversion using scf.while loop
/// Converts a fat string (!llvm.struct<(ptr, i64)>) to an integer
/// Algorithm: iterate through chars, accumulate result = result * 10 + (char - '0'), handle sign
let emitStringToInt (strSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Extract ptr and len from string struct
    let ptrSSA, z1 = MLIRZipper.yieldSSA zipper
    let extractPtrParams : Quot.Aggregate.ExtractParams = { Result = ptrSSA; Aggregate = strSSA; Index = 0; AggType = NativeStrTypeStr }
    let extractPtr = render Quot.Aggregate.extractValue extractPtrParams
    let z2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer z1

    let lenSSA, z3 = MLIRZipper.yieldSSA z2
    let extractLenParams : Quot.Aggregate.ExtractParams = { Result = lenSSA; Aggregate = strSSA; Index = 1; AggType = NativeStrTypeStr }
    let extractLen = render Quot.Aggregate.extractValue extractLenParams
    let z4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) z3

    // Constants
    let zeroSSA, z5 = MLIRZipper.yieldSSA z4
    let zeroText = sprintf "%s = arith.constant 0 : i64" zeroSSA
    let z6 = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I64) z5

    let oneSSA, z7 = MLIRZipper.yieldSSA z6
    let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
    let z8 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) z7

    let tenSSA, z9 = MLIRZipper.yieldSSA z8
    let tenText = sprintf "%s = arith.constant 10 : i64" tenSSA
    let z10 = MLIRZipper.witnessOpWithResult tenText tenSSA (Integer I64) z9

    let asciiZeroSSA, z11 = MLIRZipper.yieldSSA z10
    let asciiZeroText = sprintf "%s = arith.constant 48 : i64" asciiZeroSSA  // '0' = 48
    let z12 = MLIRZipper.witnessOpWithResult asciiZeroText asciiZeroSSA (Integer I64) z11

    let minusCharSSA, z13 = MLIRZipper.yieldSSA z12
    let minusCharText = sprintf "%s = arith.constant 45 : i8" minusCharSSA  // '-' = 45
    let z14 = MLIRZipper.witnessOpWithResult minusCharText minusCharSSA (Integer I8) z13

    // Check if first char is '-'
    let firstCharSSA, z15 = MLIRZipper.yieldSSA z14
    let firstCharText = sprintf "%s = llvm.load %s : !llvm.ptr -> i8" firstCharSSA ptrSSA
    let z16 = MLIRZipper.witnessOpWithResult firstCharText firstCharSSA (Integer I8) z15

    let isNegSSA, z17 = MLIRZipper.yieldSSA z16
    let isNegText = sprintf "%s = arith.cmpi eq, %s, %s : i8" isNegSSA firstCharSSA minusCharSSA
    let z18 = MLIRZipper.witnessOpWithResult isNegText isNegSSA (Integer I1) z17

    // Start index: 1 if negative, 0 otherwise
    let startIdxSSA, z19 = MLIRZipper.yieldSSA z18
    let startIdxText = sprintf "%s = arith.select %s, %s, %s : i64" startIdxSSA isNegSSA oneSSA zeroSSA
    let z20 = MLIRZipper.witnessOpWithResult startIdxText startIdxSSA (Integer I64) z19

    // Digit parsing loop using scf.while
    // State: (result: i64, index: i64)
    // Guard: index < len
    // Body: result = result * 10 + (char - '0'), index++
    let loopResultSSA, z21 = MLIRZipper.yieldSSA z20
    let resArg = sprintf "%s_res" (loopResultSSA.TrimStart('%'))
    let idxArg = sprintf "%s_idx" (loopResultSSA.TrimStart('%'))

    // Build loop operations as strings
    let condSSA = sprintf "%s_cond" (loopResultSSA.TrimStart('%'))
    let condText = sprintf "%%%s = arith.cmpi slt, %%%s, %s : i64" condSSA idxArg lenSSA

    let gepSSA = sprintf "%s_gep" (loopResultSSA.TrimStart('%'))
    let gepText = sprintf "%%%s = llvm.getelementptr %s[%%%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" gepSSA ptrSSA idxArg

    let charSSA = sprintf "%s_char" (loopResultSSA.TrimStart('%'))
    let charText = sprintf "%%%s = llvm.load %%%s : !llvm.ptr -> i8" charSSA gepSSA

    let char64SSA = sprintf "%s_char64" (loopResultSSA.TrimStart('%'))
    let char64Text = sprintf "%%%s = arith.extui %%%s : i8 to i64" char64SSA charSSA

    let digitSSA = sprintf "%s_digit" (loopResultSSA.TrimStart('%'))
    let digitText = sprintf "%%%s = arith.subi %%%s, %s : i64" digitSSA char64SSA asciiZeroSSA

    let mulSSA = sprintf "%s_mul" (loopResultSSA.TrimStart('%'))
    let mulText = sprintf "%%%s = arith.muli %%%s, %s : i64" mulSSA resArg tenSSA

    let newResSSA = sprintf "%s_newres" (loopResultSSA.TrimStart('%'))
    let newResText = sprintf "%%%s = arith.addi %%%s, %%%s : i64" newResSSA mulSSA digitSSA

    let newIdxSSA = sprintf "%s_newidx" (loopResultSSA.TrimStart('%'))
    let newIdxText = sprintf "%%%s = arith.addi %%%s, %s : i64" newIdxSSA idxArg oneSSA

    // Emit the scf.while loop
    let whileText =
        sprintf "%s:2 = scf.while (%%%s = %s, %%%s = %s) : (i64, i64) -> (i64, i64) {\n  %s\n  scf.condition(%%%s) %%%s, %%%s : i64, i64\n} do {\n^bb0(%%%s: i64, %%%s: i64):\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  scf.yield %%%s, %%%s : i64, i64\n}"
            loopResultSSA resArg zeroSSA idxArg startIdxSSA
            condText condSSA resArg idxArg
            resArg idxArg
            gepText charText char64Text digitText mulText newResText newIdxText
            newResSSA newIdxSSA

    let z22 = MLIRZipper.witnessOpWithResult whileText loopResultSSA (Integer I64) z21

    // Get final result from loop (first element)
    let parsedSSA, z23 = MLIRZipper.yieldSSA z22
    let parsedText = sprintf "%s = arith.addi %s#0, %s : i64" parsedSSA loopResultSSA zeroSSA  // Just copy result[0]
    let z24 = MLIRZipper.witnessOpWithResult parsedText parsedSSA (Integer I64) z23

    // Apply negative sign if needed: result = select(isNeg, -result, result)
    let negatedSSA, z25 = MLIRZipper.yieldSSA z24
    let negatedText = sprintf "%s = arith.subi %s, %s : i64" negatedSSA zeroSSA parsedSSA
    let z26 = MLIRZipper.witnessOpWithResult negatedText negatedSSA (Integer I64) z25

    let finalSSA, z27 = MLIRZipper.yieldSSA z26
    let finalText = sprintf "%s = arith.select %s, %s, %s : i64" finalSSA isNegSSA negatedSSA parsedSSA
    let z28 = MLIRZipper.witnessOpWithResult finalText finalSSA (Integer I64) z27

    (finalSSA, z28)

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM BINDING DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a platform binding call
let witnessPlatformBinding (entryPoint: string) (argSSAs: (string * MLIRType) list) (returnType: NativeType) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = argSSAs
        ReturnType = mapType returnType
        BindingStrategy = Static
    }

    let zipper', result = PlatformDispatch.dispatch prim zipper

    match result with
    | WitnessedValue (ssa, ty) ->
        zipper', TRValue (ssa, Serialize.mlirType ty)
    | WitnessedVoid ->
        zipper', TRVoid
    | NotSupported reason ->
        zipper', TRError (sprintf "Platform binding '%s' not supported: %s" entryPoint reason)

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an application and generate corresponding MLIR
let witness (funcNodeId: NodeId) (argNodeIds: NodeId list) (returnType: NativeType) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match SemanticGraph.tryGetNode funcNodeId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.PlatformBinding entryPoint ->
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    match SemanticGraph.tryGetNode nodeId graph with
                    | Some argNode ->
                        match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
                        | Some (ssa, _) -> Some (ssa, mapType argNode.Type)
                        | None -> None
                    | None -> None)
            
            let expectedParamCount =
                match entryPoint with
                | "writeBytes" | "readBytes" -> 3
                | "getCurrentTicks" -> 0
                | "sleep" -> 1
                // WebView bindings
                | "createWebview" -> 2
                | "destroyWebview" | "runWebview" | "terminateWebview" -> 1
                | "setWebviewTitle" | "navigateWebview" | "setWebviewHtml"
                | "initWebview" | "evalWebview" | "bindWebview" -> 2
                | "setWebviewSize" | "returnWebview" -> 4
                | _ -> 0

            if List.length argSSAs < expectedParamCount then
                let argsEncoded = 
                    argSSAs 
                    |> List.collect (fun (ssa, ty) -> [ssa; Serialize.mlirType ty]) 
                    |> String.concat ":"
                let marker = 
                    if argsEncoded.Length > 0 then sprintf "$platform:%s:%s" entryPoint argsEncoded
                    else sprintf "$platform:%s" entryPoint
                ()
                zipper, TRValue (marker, "func")
            else
                witnessPlatformBinding entryPoint argSSAs returnType zipper

        | SemanticKind.Intrinsic intrinsicInfo ->
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)

            match intrinsicInfo, argSSAs with
            // NativePtr operations - type-safe dispatch via NativePtrOpKind
            // Uses quotation-based templates for principled MLIR generation.
            | NativePtrOp op, argSSAs ->
                match op, argSSAs with
                | PtrToNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let convParams: Quot.Conversion.PtrIntParams = { Result = ssaName; Operand = argSSA; IntType = "i64" }
                    let text = render Quot.Conversion.ptrToInt convParams
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Integer I64) zipper'
                    zipper'', TRValue (ssaName, "i64")
                | PtrOfNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let convParams: Quot.Conversion.PtrIntParams = { Result = ssaName; Operand = argSSA; IntType = "i64" }
                    let text = render Quot.Conversion.intToPtr convParams
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
                    zipper'', TRValue (ssaName, "!llvm.ptr")
                | PtrToVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrOfVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrGet, [(ptrSSA, _); (idxSSA, _)] ->
                    let elemType = Serialize.mlirType (mapType returnType)
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = gepSSA; Base = ptrSSA; Offset = idxSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
                    let loadParams: LoadParams = { Result = loadSSA; Pointer = gepSSA; Type = elemType }
                    let loadText = render Quot.Core.load loadParams
                    let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType returnType) zipper'''
                    zipper4, TRValue (loadSSA, elemType)
                | PtrSet, [(ptrSSA, _); (idxSSA, _); (valSSA, _)] ->
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = gepSSA; Base = ptrSSA; Offset = idxSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let storeParams: StoreParams = { Value = valSSA; Pointer = gepSSA; Type = "i8" }
                    let storeText = render Quot.Core.store storeParams
                    let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
                    zipper''', TRVoid
                | PtrStackAlloc, [(countSSA, _)] ->
                    let elemType =
                        match returnType with
                        | NativeType.TNativePtr elemTy -> Serialize.mlirType (mapType elemTy)
                        | _ -> "i8"
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let countSSA64, zipper'' = MLIRZipper.yieldSSA zipper'
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper''' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper''
                    let allocaParams: AllocaParams = { Result = ssaName; Count = countSSA64; ElementType = elemType }
                    let allocaText = render Quot.Core.alloca allocaParams
                    let zipper4 = MLIRZipper.witnessOpWithResult allocaText ssaName Pointer zipper'''
                    zipper4, TRValue (ssaName, "!llvm.ptr")
                | PtrCopy, [(destSSA, _); (srcSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memcpyParams: Quot.Intrinsic.MemCopyParams = { Dest = destSSA; Src = srcSSA; Len = countSSA64 }
                    let memcpyText = render Quot.Intrinsic.memcpy memcpyParams
                    let zipper''' = MLIRZipper.witnessVoidOp memcpyText zipper''
                    zipper''', TRVoid
                | PtrFill, [(destSSA, _); (valueSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memsetParams: Quot.Intrinsic.MemSetParams = { Dest = destSSA; Value = valueSSA; Len = countSSA64 }
                    let memsetText = render Quot.Intrinsic.memset memsetParams
                    let zipper''' = MLIRZipper.witnessVoidOp memsetText zipper''
                    zipper''', TRVoid
                | PtrAdd, [(ptrSSA, _); (offsetSSA, _)] ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = resultSSA; Base = ptrSSA; Offset = offsetSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText resultSSA Pointer zipper'
                    zipper'', TRValue (resultSSA, "!llvm.ptr")
                | _, _ ->
                    zipper, TRError (sprintf "NativePtr operation arity mismatch: %A" op)

            // Sys intrinsics - direct syscall dispatch
            | SysOp opName, argSSAs ->
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                witnessPlatformBinding opName argSSAsWithTypes returnType zipper

            // Console intrinsics - higher-level I/O that lowers to Sys calls
            | ConsoleOp "write", [(strSSA, _)] ->
                // Console.write: extract pointer/length from fat string, call Sys.write(1, ptr, len)
                let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtrParams : Quot.Aggregate.ExtractParams = { Result = ptrSSA; Aggregate = strSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr = render Quot.Aggregate.extractValue extractPtrParams
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLenParams : Quot.Aggregate.ExtractParams = { Result = lenSSA; Aggregate = strSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen = render Quot.Aggregate.extractValue extractLenParams
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdParams : ConstantParams = { Result = fdSSA; Value = "1"; Type = "i32" }
                let fdText = render Quot.Constant.intConst fdParams
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Call Sys.write via platform binding
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" argSSAsWithTypes returnType zipper6

            | ConsoleOp "writeln", [(strSSA, _)] ->
                // Console.writeln: write the string, then write newline
                let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtrParams : Quot.Aggregate.ExtractParams = { Result = ptrSSA; Aggregate = strSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr = render Quot.Aggregate.extractValue extractPtrParams
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLenParams : Quot.Aggregate.ExtractParams = { Result = lenSSA; Aggregate = strSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen = render Quot.Aggregate.extractValue extractLenParams
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdParams : ConstantParams = { Result = fdSSA; Value = "1"; Type = "i32" }
                let fdText = render Quot.Constant.intConst fdParams
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Write the string
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                // Sys.write returns bytes written (i32 after platform binding truncation)
                let zipper7, _ = witnessPlatformBinding "Sys.write" argSSAsWithTypes Types.int32Type zipper6

                // Write newline: allocate newline char on stack, write it
                let nlSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let nlParams : ConstantParams = { Result = nlSSA; Value = "10"; Type = "i8" }  // '\n' = 10
                let nlText = render Quot.Constant.intConst nlParams
                let zipper9 = MLIRZipper.witnessOpWithResult nlText nlSSA (Integer I8) zipper8

                let oneSSA, zipper10 = MLIRZipper.yieldSSA zipper9
                let oneParams : ConstantParams = { Result = oneSSA; Value = "1"; Type = "i64" }
                let oneText = render Quot.Constant.intConst oneParams
                let zipper11 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) zipper10

                let nlBufSSA, zipper12 = MLIRZipper.yieldSSA zipper11
                let allocaParams : AllocaParams = { Result = nlBufSSA; Count = oneSSA; ElementType = "i8" }
                let allocaText = render Quot.Core.alloca allocaParams
                let zipper13 = MLIRZipper.witnessOpWithResult allocaText nlBufSSA Pointer zipper12

                let storeParams : StoreParams = { Value = nlSSA; Pointer = nlBufSSA; Type = "i8" }
                let storeText = render Quot.Core.store storeParams
                let zipper14 = MLIRZipper.witnessVoidOp storeText zipper13

                // Write the newline
                let nlArgSSAs = [(fdSSA, Integer I32); (nlBufSSA, Pointer); (oneSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" nlArgSSAs returnType zipper14

            | ConsoleOp "readln", ([] | [_]) ->  // Takes unit arg (or elided)
                // Dispatch to platform binding (character-by-character reading in ConsoleBindings)
                witnessPlatformBinding "Console.readln" [] Types.stringType zipper

            | NativeStrOp "fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = ptrSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
                let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extParams : ConversionParams = { Result = lenSSA64; Operand = lenSSA; FromType = "i32"; ToType = "i64" }
                let extText = render Quot.Conversion.extSI extParams
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenParams : Quot.Aggregate.InsertParams = { Result = fatPtrSSA; Value = lenSSA64; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
                let insertLenText = render Quot.Aggregate.insertValue insertLenParams
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            | NativeDefaultOp "zeroed", ([] | [_]) ->
                let zeroSSA, zipper' = MLIRZipper.yieldSSA zipper
                let mlirRetType = mapType returnType
                let mlirTypeStr = Serialize.mlirType mlirRetType
                let zeroText =
                    match mlirRetType with
                    | Integer _ ->
                        let constParams : ConstantParams = { Result = zeroSSA; Value = "0"; Type = mlirTypeStr }
                        render Quot.Constant.intConst constParams
                    | Float F32 ->
                        let floatParams : ConstantParams = { Result = zeroSSA; Value = "0.0"; Type = "f32" }
                        render Quot.Constant.floatConst floatParams
                    | Float F64 ->
                        let floatParams : ConstantParams = { Result = zeroSSA; Value = "0.0"; Type = "f64" }
                        render Quot.Constant.floatConst floatParams
                    | Pointer ->
                        render LLVMTemplates.Quot.Global.zeroInit {| Result = zeroSSA; Type = "!llvm.ptr" |}
                    | Struct _ when mlirTypeStr = NativeStrTypeStr ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = NativeStrTypeStr |}
                    | Struct _ ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = mlirTypeStr |}
                    | _ ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = mlirTypeStr |}
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            // Format intrinsics - convert values to strings
            | FormatOp "int", [(intSSA, intType)] ->
                let resultSSA, zipper' = emitIntToString intSSA intType zipper
                zipper', TRValue (resultSSA, NativeStrTypeStr)

            | FormatOp "float", [(floatSSA, floatType)] ->
                let resultSSA, zipper' = emitFloatToString floatSSA floatType zipper
                zipper', TRValue (resultSSA, NativeStrTypeStr)

            // Parse intrinsics - convert strings to values
            // Uses platform helpers (emitted once at module level)
            | ParseOp "int", [(strSSA, _)] ->
                let resultSSA, zipper1 = Alex.Bindings.PlatformHelpers.emitParseIntCall strSSA zipper
                zipper1, TRValue (resultSSA, "i64")

            | ParseOp "float", [(strSSA, _)] ->
                let resultSSA, zipper1 = Alex.Bindings.PlatformHelpers.emitParseFloatCall strSSA zipper
                zipper1, TRValue (resultSSA, "f64")

            // Convert intrinsics - numeric type conversions
            | ConvertOp "toFloat", [(argSSA, argType)] ->
                // int -> float conversion (sitofp)
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let fromType = if argType = "i32" then "i32" else "i64"  // Support both i32 and i64 input
                let convParams : ConversionParams = { Result = resultSSA; Operand = argSSA; FromType = fromType; ToType = "f64" }
                let convText = render Quot.Conversion.siToFP convParams
                let zipper'' = MLIRZipper.witnessOpWithResult convText resultSSA (Float F64) zipper'
                zipper'', TRValue (resultSSA, "f64")

            | ConvertOp "toInt", [(argSSA, argType)] ->
                // float -> int conversion (fptosi)
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let fromType = if argType = "f32" then "f32" else "f64"  // Support both f32 and f64 input
                let convParams : ConversionParams = { Result = resultSSA; Operand = argSSA; FromType = fromType; ToType = "i64" }
                let convText = render Quot.Conversion.fpToSI convParams
                let zipper'' = MLIRZipper.witnessOpWithResult convText resultSSA (Integer I64) zipper'
                zipper'', TRValue (resultSSA, "i64")

            | StringOp "contains", [(strSSA, _); (charSSA, charType)] ->
                // String.contains : string -> char -> bool
                // char is i32 (Unicode), helper expects i8 - truncate for ASCII chars
                let char8SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let truncText = sprintf "%s = arith.trunci %s : i32 to i8" char8SSA charSSA
                let zipper2 = MLIRZipper.witnessOpWithResult truncText char8SSA (Integer I8) zipper1
                let resultSSA, zipper3 = Alex.Bindings.PlatformHelpers.emitStringContainsCharCall strSSA char8SSA zipper2
                zipper3, TRValue (resultSSA, "i1")

            | StringOp "concat2", [(str1SSA, _); (str2SSA, _)] ->
                // Extract ptr and len from first string
                let ptr1SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr1Params : Quot.Aggregate.ExtractParams = { Result = ptr1SSA; Aggregate = str1SSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr1 = render Quot.Aggregate.extractValue extractPtr1Params
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer zipper1

                let len1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen1Params : Quot.Aggregate.ExtractParams = { Result = len1SSA; Aggregate = str1SSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen1 = render Quot.Aggregate.extractValue extractLen1Params
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) zipper3

                // Extract ptr and len from second string
                let ptr2SSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let extractPtr2Params : Quot.Aggregate.ExtractParams = { Result = ptr2SSA; Aggregate = str2SSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr2 = render Quot.Aggregate.extractValue extractPtr2Params
                let zipper6 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer zipper5

                let len2SSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let extractLen2Params : Quot.Aggregate.ExtractParams = { Result = len2SSA; Aggregate = str2SSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen2 = render Quot.Aggregate.extractValue extractLen2Params
                let zipper8 = MLIRZipper.witnessOpWithResult extractLen2 len2SSA (Integer I64) zipper7

                // Compute total length
                let totalLenSSA, zipper9 = MLIRZipper.yieldSSA zipper8
                let addParams : BinaryOpParams = { Result = totalLenSSA; Lhs = len1SSA; Rhs = len2SSA; Type = "i64" }
                let addLenText = render Quot.IntBinary.addI addParams
                let zipper10 = MLIRZipper.witnessOpWithResult addLenText totalLenSSA (Integer I64) zipper9

                // Allocate buffer for concatenated string
                let bufSSA, zipper11 = MLIRZipper.yieldSSA zipper10
                let allocaParams : AllocaParams = { Result = bufSSA; Count = totalLenSSA; ElementType = "i8" }
                let allocaText = render Quot.Core.alloca allocaParams
                let zipper12 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper11

                // Copy first string
                let memcpy1Params : Quot.Intrinsic.MemCopyParams = { Dest = bufSSA; Src = ptr1SSA; Len = len1SSA }
                let memcpy1 = render Quot.Intrinsic.memcpy memcpy1Params
                let zipper13 = MLIRZipper.witnessVoidOp memcpy1 zipper12

                // GEP to offset position for second string
                let offsetSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let gepParams : GepParams = { Result = offsetSSA; Base = bufSSA; Offset = len1SSA; ElementType = "i8" }
                let gepText = render Quot.Gep.i64 gepParams
                let zipper15 = MLIRZipper.witnessOpWithResult gepText offsetSSA Pointer zipper14

                // Copy second string
                let memcpy2Params : Quot.Intrinsic.MemCopyParams = { Dest = offsetSSA; Src = ptr2SSA; Len = len2SSA }
                let memcpy2 = render Quot.Intrinsic.memcpy memcpy2Params
                let zipper16 = MLIRZipper.witnessVoidOp memcpy2 zipper15

                // Build result fat string struct
                let undefSSA, zipper17 = MLIRZipper.yieldSSA zipper16
                let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
                let zipper18 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper17

                let withPtrSSA, zipper19 = MLIRZipper.yieldSSA zipper18
                let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = bufSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
                let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
                let zipper20 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper19

                let resultSSA, zipper21 = MLIRZipper.yieldSSA zipper20
                let insertLenParams : Quot.Aggregate.InsertParams = { Result = resultSSA; Value = totalLenSSA; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
                let insertLenText = render Quot.Aggregate.insertValue insertLenParams
                let zipper22 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType zipper21

                zipper22, TRValue (resultSSA, NativeStrTypeStr)

            | info, [] ->
                zipper, TRValue ("$intrinsic:" + info.FullName, "func")
            | info, _ ->
                zipper, TRError (sprintf "Unknown intrinsic: %s with %d args" info.FullName (List.length argSSAs))

        | SemanticKind.VarRef (name, defId) ->
            let argSSAsAndTypes =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
            let argSSAs = argSSAsAndTypes |> List.map fst
            let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

            if name = "op_PipeRight" || name = "op_PipeLeft" then
                match argSSAs with
                | [argSSA] ->
                    let (argType: string) = argSSAsAndTypes |> List.head |> snd
                    let marker = sprintf "$pipe:%s:%s" argSSA argType
                    zipper, TRValue (marker, "func")
                | _ ->
                    zipper, TRError (sprintf "Pipe operator '%s' expects 1 argument, got %d" name (List.length argSSAs))
            elif name = "ignore" then
                // ZERO-COST INTRINSIC: ignore produces no code and just returns unit
                let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                let unitParams : ConstantParams = { Result = unitSSA; Value = "0"; Type = "i32" }
                let unitText = render Quot.Constant.intConst unitParams
                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                zipper'', TRValue (unitSSA, "i32")
            elif name = "failwith" || name = "failwithf" then
                // NATIVE PANIC: failwith triggers SYS_exit(1) for now
                let sysExitSSA, z1 = MLIRZipper.witnessConstant 60L I64 zipper
                let errorCodeSSA, z2 = MLIRZipper.witnessConstant 1L I64 z1
                let _, z3 = MLIRZipper.witnessSyscall sysExitSSA [errorCodeSSA, "i64"] (Integer I64) z2
                // Return a marker that signals a panic occurred (terminates the block)
                z3, TRValue ("$panic", "i32")
            else

            match defId with
            | Some defNodeId ->
                match MLIRZipper.recallNodeSSA (string (NodeId.value defNodeId)) zipper with
                | Some (funcSSA, _funcType) ->
                    if funcSSA.StartsWith("@") then
                        let funcName = funcSSA.Substring(1)
                        
                        let expectedParams = MLIRZipper.lookupFuncParamCount funcName zipper
                        match expectedParams with
                        | Some paramCount when paramCount > List.length argSSAs ->
                            let argPairs = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName argSSAs argTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | None ->
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                let argSSAsWithTypes = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                
                match argSSAsWithTypes with
                | [(arg1SSA, arg1Type); (arg2SSA, arg2Type)] ->
                    match tryEmitPrimitiveBinaryOp name arg1SSA arg1Type arg2SSA arg2Type zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | [(argSSA, argType)] ->
                    match tryEmitPrimitiveUnaryOp name argSSA argType zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | _ ->
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))

        | SemanticKind.Lambda _ ->
            zipper, TRError "Lambda application not yet supported"

        | SemanticKind.Application (innerFuncId, innerArgIds) ->
            match MLIRZipper.recallNodeSSA (string (NodeId.value funcNodeId)) zipper with
            | Some (funcSSA, _funcType) ->
                let argSSAsAndTypes =
                    argNodeIds
                    |> List.choose (fun nodeId ->
                        MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
                let argSSAs = argSSAsAndTypes |> List.map fst
                let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

                if funcSSA.StartsWith("$pipe:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 3 then
                        let pipedArgSSA = parts.[1]
                        let pipedArgType = parts.[2]
                        match argSSAs with
                        | [fSSA] ->
                            let retTypeStr = Serialize.mlirType (mapType returnType)
                            if retTypeStr = "i32" && pipedArgType = "i32" then
                                let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                let unitParams : ConstantParams = { Result = unitSSA; Value = "0"; Type = "i32" }
                                let unitText = render Quot.Constant.intConst unitParams
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1)
                                    if funcName = "ignore" then
                                        let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                        let unitParams : ConstantParams = { Result = unitSSA; Value = "0"; Type = "i32" }
                                        let unitText = render Quot.Constant.intConst unitParams
                                        let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                        zipper'', TRValue (unitSSA, "i32")
                                    else
                                        let ssaName, zipper' = MLIRZipper.witnessCall funcName [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                                else
                                    let ssaName, zipper' = MLIRZipper.witnessIndirectCall fSSA [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                        | _ ->
                            zipper, TRError (sprintf "Pipe application expected 1 function arg, got %d" (List.length argSSAs))
                    else
                        zipper, TRError (sprintf "Invalid pipe marker: %s" funcSSA)
                elif funcSSA.StartsWith("$partial:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let funcName = parts.[1]
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, ty)
                                | _ -> None)
                            |> Array.toList
                        let allArgSSAs = (appliedArgs |> List.map fst) @ argSSAs
                        let allArgTypes = (appliedArgs |> List.map (snd >> Serialize.deserializeType)) @ argTypes
                        
                        match MLIRZipper.lookupFuncParamCount funcName zipper with
                        | Some paramCount when paramCount > List.length allArgSSAs ->
                            let argPairs = List.zip allArgSSAs (allArgTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName allArgSSAs allArgTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        zipper, TRError (sprintf "Invalid partial marker: %s" funcSSA)
                elif funcSSA.StartsWith("$platform:") then
                    ()
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let entryPoint = parts.[1]
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, Serialize.deserializeType ty)
                                | _ -> None)
                            |> Array.toList
                        let allArgSSAs = appliedArgs @ (List.zip argSSAs argTypes)
                        
                        let expectedParamCount =
                            match entryPoint with
                            | "writeBytes" | "readBytes" -> 3
                            | "getCurrentTicks" -> 0
                            | "sleep" -> 1
                            // WebView bindings
                            | "createWebview" -> 2
                            | "destroyWebview" | "runWebview" | "terminateWebview" -> 1
                            | "setWebviewTitle" | "navigateWebview" | "setWebviewHtml"
                            | "initWebview" | "evalWebview" | "bindWebview" -> 2
                            | "setWebviewSize" | "returnWebview" -> 4
                            | _ -> List.length allArgSSAs

                        if List.length allArgSSAs < expectedParamCount then
                            let argsEncoded = allArgSSAs |> List.collect (fun (a, t) -> [a; Serialize.mlirType t]) |> String.concat ":"
                            let marker = sprintf "$platform:%s:%s" entryPoint argsEncoded
                            zipper, TRValue (marker, "func")
                        else
                            witnessPlatformBinding entryPoint allArgSSAs returnType zipper
                    else
                        zipper, TRError (sprintf "Invalid platform marker: %s" funcSSA)
                else

                let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                zipper, TRError (sprintf "Curried function application not computed: %A" (innerFuncId, innerArgIds))

        | _ ->
            zipper, TRError (sprintf "Unexpected function node kind: %A" funcNode.Kind)

    | None ->
        zipper, TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))
