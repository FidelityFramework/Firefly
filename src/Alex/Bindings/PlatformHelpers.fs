/// Platform Helper Functions - Structured MLIR function definitions
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// All operations expressed as structured MLIROp types.
/// ZERO sprintf. Function bodies are Regions containing structured ops.
module Alex.Bindings.PlatformHelpers

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.SCF.Templates
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════════════
// Helper Names
// ═══════════════════════════════════════════════════════════════════════════

[<Literal>]
let ParseIntHelper = "fidelity_parse_int"

[<Literal>]
let ParseFloatHelper = "fidelity_parse_float"

[<Literal>]
let StringContainsCharHelper = "fidelity_string_contains_char"

[<Literal>]
let Base64EncodeHelper = "fidelity_base64_encode"

[<Literal>]
let Base64DecodeHelper = "fidelity_base64_decode"

[<Literal>]
let Sha1Helper = "fidelity_sha1"

// ═══════════════════════════════════════════════════════════════════════════
// Type Constants
// ═══════════════════════════════════════════════════════════════════════════

let fatStringType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

// ═══════════════════════════════════════════════════════════════════════════
// SSA Counter for building function bodies
// ═══════════════════════════════════════════════════════════════════════════

/// Mutable counter for generating SSA values within a function body
type SSACounter() =
    let mutable n = 0
    member _.Next() =
        let v = V n
        n <- n + 1
        v
    member _.Reset() = n <- 0

// ═══════════════════════════════════════════════════════════════════════════
// PARSE INT HELPER
// Parses a fat string to i64
// ═══════════════════════════════════════════════════════════════════════════

/// Build the fidelity_parse_int function as structured MLIROp
let buildParseIntFunc () : FuncOp =
    let ssa = SSACounter()

    // Function argument: %arg0 = fat string
    let strArg = Arg 0

    // Extract pointer and length from fat string
    let ptrSSA = ssa.Next()
    let lenSSA = ssa.Next()
    let extractPtr = MLIROp.LLVMOp (ExtractValue (ptrSSA, strArg, [0], fatStringType))
    let extractLen = MLIROp.LLVMOp (ExtractValue (lenSSA, strArg, [1], fatStringType))

    // Constants
    let c0 = ssa.Next()
    let c1 = ssa.Next()
    let c10 = ssa.Next()
    let c48 = ssa.Next()
    let c45_i8 = ssa.Next()
    let constOps = [
        MLIROp.ArithOp (ConstI (c0, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c1, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c10, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c48, 48L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c45_i8, 45L, MLIRTypes.i8))
    ]

    // Check if first char is '-'
    let firstChar = ssa.Next()
    let isNeg = ssa.Next()
    let loadFirst = MLIROp.LLVMOp (Load (firstChar, ptrSSA, MLIRTypes.i8, NotAtomic))
    let cmpNeg = MLIROp.ArithOp (CmpI (isNeg, Eq, firstChar, c45_i8, MLIRTypes.i8))

    // Starting position: 1 if negative, 0 if positive
    let startPos = ssa.Next()
    let selectStart = MLIROp.ArithOp (Select (startPos, isNeg, c1, c0, MLIRTypes.i64))

    // Build the while loop body for parsing digits
    // Block arguments as typed vals
    let valArg = { SSA = Arg 0; Type = MLIRTypes.i64 }    // accumulated value
    let posArg = { SSA = Arg 1; Type = MLIRTypes.i64 }    // current position

    // This is the condition region
    let condInBounds = ssa.Next()
    let condOps = [
        MLIROp.ArithOp (CmpI (condInBounds, Slt, posArg.SSA, lenSSA, MLIRTypes.i64))  // pos < len
    ]
    let condRegion = {
        Blocks = [{
            Label = BlockRef "cond"
            Args = []
            Ops = condOps @ [MLIROp.SCFOp (scfCondition condInBounds [valArg; posArg])]
        }]
    }

    // Body region: parse one digit
    let charPtr = ssa.Next()
    let char' = ssa.Next()
    let charI64 = ssa.Next()
    let digit = ssa.Next()
    let valTimes10 = ssa.Next()
    let newVal = ssa.Next()
    let newPos = ssa.Next()
    let bodyOps = [
        MLIROp.LLVMOp (GEP (charPtr, ptrSSA, [(posArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (Load (char', charPtr, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (ExtUI (charI64, char', MLIRTypes.i8, MLIRTypes.i64))
        MLIROp.ArithOp (SubI (digit, charI64, c48, MLIRTypes.i64))
        MLIROp.ArithOp (MulI (valTimes10, valArg.SSA, c10, MLIRTypes.i64))
        MLIROp.ArithOp (AddI (newVal, valTimes10, digit, MLIRTypes.i64))
        MLIROp.ArithOp (AddI (newPos, posArg.SSA, c1, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [{ SSA = newVal; Type = MLIRTypes.i64 }; { SSA = newPos; Type = MLIRTypes.i64 }])
    ]
    let bodyRegion = {
        Blocks = [{
            Label = BlockRef "body"
            Args = [valArg; posArg]
            Ops = bodyOps
        }]
    }

    // The while loop
    let result0 = ssa.Next()
    let result1 = ssa.Next()
    let whileOp = MLIROp.SCFOp (While (
        [result0; result1],
        condRegion,
        bodyRegion,
        [{SSA = c0; Type = MLIRTypes.i64}; {SSA = startPos; Type = MLIRTypes.i64}]
    ))

    // Apply sign: if negative, negate result
    let negated = ssa.Next()
    let final' = ssa.Next()
    let negateOp = MLIROp.ArithOp (SubI (negated, c0, result0, MLIRTypes.i64))
    let selectFinal = MLIROp.ArithOp (Select (final', isNeg, negated, result0, MLIRTypes.i64))

    // Return
    let retOp = MLIROp.FuncOp (FuncReturn [final'])

    // Build the complete function body
    let allOps =
        [extractPtr; extractLen] @
        constOps @
        [loadFirst; cmpNeg; selectStart; whileOp; negateOp; selectFinal; retOp]

    let bodyRegion = {
        Blocks = [{
            Label = BlockRef "entry"
            Args = []
            Ops = allOps
        }]
    }

    FuncOp.FuncDef (
        ParseIntHelper,
        [(Arg 0, fatStringType)],
        MLIRTypes.i64,
        bodyRegion,
        Private
    )

// ═══════════════════════════════════════════════════════════════════════════
// PARSE FLOAT HELPER
// Parses a fat string to f64
// ═══════════════════════════════════════════════════════════════════════════

/// Build the fidelity_parse_float function as structured MLIROp
let buildParseFloatFunc () : FuncOp =
    let ssa = SSACounter()

    let strArg = Arg 0

    // Extract pointer and length
    let ptrSSA = ssa.Next()
    let lenSSA = ssa.Next()
    let extractPtr = MLIROp.LLVMOp (ExtractValue (ptrSSA, strArg, [0], fatStringType))
    let extractLen = MLIROp.LLVMOp (ExtractValue (lenSSA, strArg, [1], fatStringType))

    // Constants
    let c0_i64 = ssa.Next()
    let c1_i64 = ssa.Next()
    let c10_i64 = ssa.Next()
    let c48 = ssa.Next()
    let c45_i8 = ssa.Next()
    let c46_i8 = ssa.Next()
    let c0_f64 = ssa.Next()
    let c1_f64 = ssa.Next()
    let c10_f64 = ssa.Next()
    let constOps = [
        MLIROp.ArithOp (ConstI (c0_i64, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c1_i64, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c10_i64, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c48, 48L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c45_i8, 45L, MLIRTypes.i8))
        MLIROp.ArithOp (ConstI (c46_i8, 46L, MLIRTypes.i8))
        MLIROp.ArithOp (ConstF (c0_f64, 0.0, MLIRTypes.f64))
        MLIROp.ArithOp (ConstF (c1_f64, 1.0, MLIRTypes.f64))
        MLIROp.ArithOp (ConstF (c10_f64, 10.0, MLIRTypes.f64))
    ]

    // Check if first char is '-'
    let firstChar = ssa.Next()
    let isNeg = ssa.Next()
    let startPos = ssa.Next()
    let signOps = [
        MLIROp.LLVMOp (Load (firstChar, ptrSSA, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (CmpI (isNeg, Eq, firstChar, c45_i8, MLIRTypes.i8))
        MLIROp.ArithOp (Select (startPos, isNeg, c1_i64, c0_i64, MLIRTypes.i64))
    ]

    // Integer part parsing while loop (until decimal point)
    // Block arguments as typed vals
    let intValArg = { SSA = Arg 0; Type = MLIRTypes.f64 }    // accumulated value
    let intPosArg = { SSA = Arg 1; Type = MLIRTypes.i64 }    // current position

    // Condition: pos < len AND char != '.'
    let intCondCharPtr = ssa.Next()
    let intCondChar = ssa.Next()
    let intCondInBounds = ssa.Next()
    let intCondNotDot = ssa.Next()
    let intCondContinue = ssa.Next()
    let intCondOps = [
        MLIROp.ArithOp (CmpI (intCondInBounds, Slt, intPosArg.SSA, lenSSA, MLIRTypes.i64))
        MLIROp.LLVMOp (GEP (intCondCharPtr, ptrSSA, [(intPosArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (Load (intCondChar, intCondCharPtr, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (CmpI (intCondNotDot, Ne, intCondChar, c46_i8, MLIRTypes.i8))
        MLIROp.ArithOp (AndI (intCondContinue, intCondInBounds, intCondNotDot, MLIRTypes.i1))
        MLIROp.SCFOp (scfCondition intCondContinue [intValArg; intPosArg])
    ]
    let intCondRegion = { Blocks = [{ Label = BlockRef "intcond"; Args = []; Ops = intCondOps }] }

    // Integer part body
    let intCharPtr = ssa.Next()
    let intChar = ssa.Next()
    let intCharI64 = ssa.Next()
    let intDigit = ssa.Next()
    let intValTimes10 = ssa.Next()
    let intNewVal = ssa.Next()
    let intNewPos = ssa.Next()
    let intBodyOps = [
        MLIROp.LLVMOp (GEP (intCharPtr, ptrSSA, [(intPosArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (Load (intChar, intCharPtr, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (ExtUI (intCharI64, intChar, MLIRTypes.i8, MLIRTypes.i64))
        MLIROp.ArithOp (SubI (intDigit, intCharI64, c48, MLIRTypes.i64))
        MLIROp.ArithOp (MulI (intValTimes10, intValArg.SSA, c10_i64, MLIRTypes.i64))
        MLIROp.ArithOp (AddI (intNewVal, intValTimes10, intDigit, MLIRTypes.i64))
        MLIROp.ArithOp (AddI (intNewPos, intPosArg.SSA, c1_i64, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [{ SSA = intNewVal; Type = MLIRTypes.i64 }; { SSA = intNewPos; Type = MLIRTypes.i64 }])
    ]
    let intBodyRegion = {
        Blocks = [{
            Label = BlockRef "intbody"
            Args = [intValArg; intPosArg]
            Ops = intBodyOps
        }]
    }

    let intResult0 = ssa.Next()
    let intResult1 = ssa.Next()
    let intWhileOp = MLIROp.SCFOp (While (
        [intResult0; intResult1],
        intCondRegion,
        intBodyRegion,
        [{SSA = c0_i64; Type = MLIRTypes.i64}; {SSA = startPos; Type = MLIRTypes.i64}]
    ))

    // Convert integer part to float
    let intF64 = ssa.Next()
    let intToFloat = MLIROp.ArithOp (ArithOp.SIToFP (intF64, intResult0, MLIRTypes.i64, MLIRTypes.f64))

    // Check for decimal point
    let hasDecimal = ssa.Next()
    let fracStart = ssa.Next()
    let decimalOps = [
        MLIROp.ArithOp (CmpI (hasDecimal, Slt, intResult1, lenSSA, MLIRTypes.i64))
        MLIROp.ArithOp (AddI (fracStart, intResult1, c1_i64, MLIRTypes.i64))
    ]

    // Fractional part while loop
    // Block arguments as typed vals
    let fracFracArg = { SSA = Arg 0; Type = MLIRTypes.f64 }    // accumulated fraction
    let fracDivArg = { SSA = Arg 1; Type = MLIRTypes.f64 }     // divisor
    let fracPosArg = { SSA = Arg 2; Type = MLIRTypes.i64 }     // current position

    let fracCondInBounds = ssa.Next()
    let fracCondContinue = ssa.Next()
    let fracCondOps = [
        MLIROp.ArithOp (CmpI (fracCondInBounds, Slt, fracPosArg.SSA, lenSSA, MLIRTypes.i64))
        MLIROp.ArithOp (AndI (fracCondContinue, hasDecimal, fracCondInBounds, MLIRTypes.i1))
        MLIROp.SCFOp (scfCondition fracCondContinue [fracFracArg; fracDivArg; fracPosArg])
    ]
    let fracCondRegion = { Blocks = [{ Label = BlockRef "fraccond"; Args = []; Ops = fracCondOps }] }

    let fracCharPtr = ssa.Next()
    let fracChar = ssa.Next()
    let fracCharI64 = ssa.Next()
    let fracDigitI64 = ssa.Next()
    let fracDigitF64 = ssa.Next()
    let fracNewDiv = ssa.Next()
    let fracScaledDigit = ssa.Next()
    let fracNewFrac = ssa.Next()
    let fracNewPos = ssa.Next()
    let fracBodyOps = [
        MLIROp.LLVMOp (GEP (fracCharPtr, ptrSSA, [(fracPosArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (Load (fracChar, fracCharPtr, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (ExtUI (fracCharI64, fracChar, MLIRTypes.i8, MLIRTypes.i64))
        MLIROp.ArithOp (SubI (fracDigitI64, fracCharI64, c48, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.SIToFP (fracDigitF64, fracDigitI64, MLIRTypes.i64, MLIRTypes.f64))
        MLIROp.ArithOp (MulF (fracNewDiv, fracDivArg.SSA, c10_f64, MLIRTypes.f64))
        MLIROp.ArithOp (DivF (fracScaledDigit, fracDigitF64, fracNewDiv, MLIRTypes.f64))
        MLIROp.ArithOp (AddF (fracNewFrac, fracFracArg.SSA, fracScaledDigit, MLIRTypes.f64))
        MLIROp.ArithOp (AddI (fracNewPos, fracPosArg.SSA, c1_i64, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [{ SSA = fracNewFrac; Type = MLIRTypes.f64 }; { SSA = fracNewDiv; Type = MLIRTypes.f64 }; { SSA = fracNewPos; Type = MLIRTypes.i64 }])
    ]
    let fracBodyRegion = {
        Blocks = [{
            Label = BlockRef "fracbody"
            Args = [fracFracArg; fracDivArg; fracPosArg]
            Ops = fracBodyOps
        }]
    }

    let fracResult0 = ssa.Next()
    let fracResult1 = ssa.Next()
    let fracResult2 = ssa.Next()
    let fracWhileOp = MLIROp.SCFOp (While (
        [fracResult0; fracResult1; fracResult2],
        fracCondRegion,
        fracBodyRegion,
        [
            {SSA = c0_f64; Type = MLIRTypes.f64}
            {SSA = c1_f64; Type = MLIRTypes.f64}
            {SSA = fracStart; Type = MLIRTypes.i64}
        ]
    ))

    // Combine integer and fractional parts
    let combined = ssa.Next()
    let negated = ssa.Next()
    let final' = ssa.Next()
    let combineOps = [
        MLIROp.ArithOp (AddF (combined, intF64, fracResult0, MLIRTypes.f64))
        MLIROp.ArithOp (NegF (negated, combined, MLIRTypes.f64))
        MLIROp.ArithOp (Select (final', isNeg, negated, combined, MLIRTypes.f64))
    ]

    let retOp = MLIROp.FuncOp (FuncReturn [final'])

    let allOps =
        [extractPtr; extractLen] @
        constOps @
        signOps @
        [intWhileOp; intToFloat] @
        decimalOps @
        [fracWhileOp] @
        combineOps @
        [retOp]

    let bodyRegion = { Blocks = [{ Label = BlockRef "entry"; Args = []; Ops = allOps }] }

    FuncOp.FuncDef (
        ParseFloatHelper,
        [(Arg 0, fatStringType)],
        MLIRTypes.f64,
        bodyRegion,
        Private
    )

// ═══════════════════════════════════════════════════════════════════════════
// STRING CONTAINS CHAR HELPER
// ═══════════════════════════════════════════════════════════════════════════

let buildStringContainsCharFunc () : FuncOp =
    let ssa = SSACounter()

    let strArg = Arg 0
    let targetArg = Arg 1

    let ptrSSA = ssa.Next()
    let lenSSA = ssa.Next()
    let extractPtr = MLIROp.LLVMOp (ExtractValue (ptrSSA, strArg, [0], fatStringType))
    let extractLen = MLIROp.LLVMOp (ExtractValue (lenSSA, strArg, [1], fatStringType))

    let c0 = ssa.Next()
    let c1 = ssa.Next()
    let cFalse = ssa.Next()
    let constOps = [
        MLIROp.ArithOp (ConstI (c0, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (c1, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ConstI (cFalse, 0L, MLIRTypes.i1))
    ]

    // Condition: in_bounds AND not_found
    // Block arguments as typed vals
    let foundArg = { SSA = Arg 0; Type = MLIRTypes.i1 }    // found flag
    let posArg = { SSA = Arg 1; Type = MLIRTypes.i64 }     // current position

    let condInBounds = ssa.Next()
    let condNotFound = ssa.Next()
    let condContinue = ssa.Next()
    let condOps = [
        MLIROp.ArithOp (CmpI (condInBounds, Slt, posArg.SSA, lenSSA, MLIRTypes.i64))
        MLIROp.ArithOp (CmpI (condNotFound, Eq, foundArg.SSA, cFalse, MLIRTypes.i1))
        MLIROp.ArithOp (AndI (condContinue, condInBounds, condNotFound, MLIRTypes.i1))
        MLIROp.SCFOp (scfCondition condContinue [foundArg; posArg])
    ]
    let condRegion = { Blocks = [{ Label = BlockRef "cond"; Args = []; Ops = condOps }] }

    let charPtr = ssa.Next()
    let char' = ssa.Next()
    let isMatch = ssa.Next()
    let newPos = ssa.Next()
    let bodyOps = [
        MLIROp.LLVMOp (GEP (charPtr, ptrSSA, [(posArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (Load (char', charPtr, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (CmpI (isMatch, Eq, char', targetArg, MLIRTypes.i8))
        MLIROp.ArithOp (AddI (newPos, posArg.SSA, c1, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [{ SSA = isMatch; Type = MLIRTypes.i1 }; { SSA = newPos; Type = MLIRTypes.i64 }])
    ]
    let bodyRegion = {
        Blocks = [{
            Label = BlockRef "body"
            Args = [foundArg; posArg]
            Ops = bodyOps
        }]
    }

    let result0 = ssa.Next()
    let result1 = ssa.Next()
    let whileOp = MLIROp.SCFOp (While (
        [result0; result1],
        condRegion,
        bodyRegion,
        [{SSA = cFalse; Type = MLIRTypes.i1}; {SSA = c0; Type = MLIRTypes.i64}]
    ))

    let retOp = MLIROp.FuncOp (FuncReturn [result0])

    let allOps = [extractPtr; extractLen] @ constOps @ [whileOp; retOp]
    let bodyRegion = { Blocks = [{ Label = BlockRef "entry"; Args = []; Ops = allOps }] }

    FuncOp.FuncDef (
        StringContainsCharHelper,
        [(Arg 0, fatStringType); (Arg 1, MLIRTypes.i8)],
        MLIRTypes.i1,
        bodyRegion,
        Private
    )

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTION CACHE
// Track which helpers have been registered
// ═══════════════════════════════════════════════════════════════════════════

let mutable private registeredHelpers : Set<string> = Set.empty

let private ensureHelper (name: string) (builder: unit -> FuncOp) (z: PSGZipper) : MLIROp list =
    if registeredHelpers.Contains name then
        []
    else
        registeredHelpers <- registeredHelpers.Add name
        [MLIROp.FuncOp (builder ())]

// ═══════════════════════════════════════════════════════════════════════════
// BINDINGS FOR HELPER CALLS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a Val from SSA and type
let inline val' ssa ty : Val = { SSA = ssa; Type = ty }

/// Emit call to fidelity_parse_int
/// Uses pre-assigned SSA from Application node
let bindParseInt (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] ->
        let helperOps = ensureHelper ParseIntHelper buildParseIntFunc z
        let resultSSA = requireNodeSSA appNodeId z
        let callOp = MLIROp.FuncOp (FuncCall (Some resultSSA, ParseIntHelper, [str], MLIRTypes.i64))
        BoundOps (helperOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.i64 })
    | _ ->
        NotSupported "parseint requires (string)"

/// Emit call to fidelity_parse_float
/// Uses pre-assigned SSA from Application node
let bindParseFloat (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] ->
        let helperOps = ensureHelper ParseFloatHelper buildParseFloatFunc z
        let resultSSA = requireNodeSSA appNodeId z
        let callOp = MLIROp.FuncOp (FuncCall (Some resultSSA, ParseFloatHelper, [str], MLIRTypes.f64))
        BoundOps (helperOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.f64 })
    | _ ->
        NotSupported "parsefloat requires (string)"

/// Emit call to fidelity_string_contains_char
/// Uses pre-assigned SSA from Application node
let bindStringContainsChar (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str; char'] ->
        let helperOps = ensureHelper StringContainsCharHelper buildStringContainsCharFunc z
        let resultSSA = requireNodeSSA appNodeId z
        let callOp = MLIROp.FuncOp (FuncCall (Some resultSSA, StringContainsCharHelper, [str; char'], MLIRTypes.i1))
        BoundOps (helperOps @ [callOp], Some { SSA = resultSSA; Type = MLIRTypes.i1 })
    | _ ->
        NotSupported "stringContainsChar requires (string, char)"

// ═══════════════════════════════════════════════════════════════════════════
// BASE64 AND SHA1 - TODO: Implement as structured ops
// These are more complex and will be added incrementally
// ═══════════════════════════════════════════════════════════════════════════

let bindBase64Encode (_appNodeId: NodeId) (_z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    NotSupported "base64Encode not yet implemented with structured ops"

let bindBase64Decode (_appNodeId: NodeId) (_z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    NotSupported "base64Decode not yet implemented with structured ops"

let bindSha1 (_appNodeId: NodeId) (_z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    NotSupported "sha1 not yet implemented with structured ops"

// ═══════════════════════════════════════════════════════════════════════════
// REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════

let registerBindings () =
    // Reset helper cache
    registeredHelpers <- Set.empty

    // Register for all platforms (these are pure compute, platform-independent)
    for os in [Linux; MacOS] do
        for arch in [X86_64; ARM64] do
            PlatformDispatch.register os arch "parseInt" bindParseInt
            PlatformDispatch.register os arch "parseFloat" bindParseFloat
            PlatformDispatch.register os arch "stringContainsChar" bindStringContainsChar
            PlatformDispatch.register os arch "base64Encode" bindBase64Encode
            PlatformDispatch.register os arch "base64Decode" bindBase64Decode
            PlatformDispatch.register os arch "sha1" bindSha1
