/// Platform Helper Functions - Structured MLIR function definitions
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// All operations expressed as structured MLIROp types.
/// ZERO sprintf. Function bodies are Regions containing structured ops.
module Alex.Bindings.PlatformHelpers

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
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

    // Return (with type for MLIR syntax)
    let retOp = MLIROp.FuncOp (FuncReturn [(final', MLIRTypes.i64)])

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

    let retOp = MLIROp.FuncOp (FuncReturn [(final', MLIRTypes.f64)])

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

    let retOp = MLIROp.FuncOp (FuncReturn [(result0, MLIRTypes.i1)])

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
let mutable private registeredExterns : Set<string> = Set.empty

let private ensureHelper (name: string) (builder: unit -> FuncOp) (z: PSGZipper) : MLIROp list =
    if registeredHelpers.Contains name then
        []
    else
        registeredHelpers <- registeredHelpers.Add name
        [MLIROp.FuncOp (builder ())]

/// Ensure an extern function declaration is emitted to TopLevel (once)
let private ensureExternDecl (name: string) (argTypes: MLIRType list) (retTy: MLIRType) (z: PSGZipper) : unit =
    if not (registeredExterns.Contains name) then
        registeredExterns <- registeredExterns.Add name
        emitTopLevel (MLIROp.FuncOp (FuncOp.FuncDecl (name, argTypes, retTy, Private))) z

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
// DATETIME BINDINGS
// DateTime operations using milliseconds since Unix epoch
// ═══════════════════════════════════════════════════════════════════════════

/// DateTime.now - delegates to Sys.clock_gettime (returns ms since epoch)
let bindDateTimeNow (appNodeId: NodeId) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    // DateTime.now is identical to Sys.clock_gettime - just dispatch to it
    let sysPrim: PlatformPrimitive = {
        EntryPoint = "Sys.clock_gettime"
        Library = "platform"
        CallingConvention = "ccc"
        Args = []
        ReturnType = MLIRTypes.i64
        BindingStrategy = Static
    }
    PlatformDispatch.dispatch appNodeId z sysPrim

/// DateTime.hour - extract hour component (0-23) from ms since epoch
let bindDateTimeHour (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [msVal] ->
        let ssas = requireNodeSSAs appNodeId z
        // ms / 3600000 % 24
        let c3600000 = ssas.[0]
        let c24 = ssas.[1]
        let hoursFull = ssas.[2]
        let hoursDay = ssas.[3]
        let hoursTrunc = ssas.[4]
        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c3600000, 3600000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.ConstI (c24, 24L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.DivSI (hoursFull, msVal.SSA, c3600000, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.RemSI (hoursDay, hoursFull, c24, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.TruncI (hoursTrunc, hoursDay, MLIRTypes.i64, MLIRTypes.i32))
        ]
        BoundOps (ops, Some { SSA = hoursTrunc; Type = MLIRTypes.i32 })
    | _ -> NotSupported "DateTime.hour requires 1 argument"

/// DateTime.minute - extract minute component (0-59) from ms since epoch
let bindDateTimeMinute (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [msVal] ->
        let ssas = requireNodeSSAs appNodeId z
        // ms / 60000 % 60
        let c60000 = ssas.[0]
        let c60 = ssas.[1]
        let minsFull = ssas.[2]
        let minsHour = ssas.[3]
        let minsTrunc = ssas.[4]
        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c60000, 60000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.ConstI (c60, 60L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.DivSI (minsFull, msVal.SSA, c60000, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.RemSI (minsHour, minsFull, c60, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.TruncI (minsTrunc, minsHour, MLIRTypes.i64, MLIRTypes.i32))
        ]
        BoundOps (ops, Some { SSA = minsTrunc; Type = MLIRTypes.i32 })
    | _ -> NotSupported "DateTime.minute requires 1 argument"

/// DateTime.second - extract second component (0-59) from ms since epoch
let bindDateTimeSecond (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [msVal] ->
        let ssas = requireNodeSSAs appNodeId z
        // ms / 1000 % 60
        let c1000 = ssas.[0]
        let c60 = ssas.[1]
        let secsFull = ssas.[2]
        let secsMin = ssas.[3]
        let secsTrunc = ssas.[4]
        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c1000, 1000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.ConstI (c60, 60L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.DivSI (secsFull, msVal.SSA, c1000, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.RemSI (secsMin, secsFull, c60, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.TruncI (secsTrunc, secsMin, MLIRTypes.i64, MLIRTypes.i32))
        ]
        BoundOps (ops, Some { SSA = secsTrunc; Type = MLIRTypes.i32 })
    | _ -> NotSupported "DateTime.second requires 1 argument"

/// DateTime.millisecond - extract millisecond component (0-999) from ms since epoch
let bindDateTimeMillisecond (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [msVal] ->
        let ssas = requireNodeSSAs appNodeId z
        // ms % 1000
        let c1000 = ssas.[0]
        let msRem = ssas.[1]
        let msTrunc = ssas.[2]
        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c1000, 1000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.RemSI (msRem, msVal.SSA, c1000, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.TruncI (msTrunc, msRem, MLIRTypes.i64, MLIRTypes.i32))
        ]
        BoundOps (ops, Some { SSA = msTrunc; Type = MLIRTypes.i32 })
    | _ -> NotSupported "DateTime.millisecond requires 1 argument"

/// DateTime.utcOffset - get local timezone offset in seconds from UTC
/// Uses libc localtime_r() to get tm_gmtoff from struct tm
/// struct tm layout on Linux x86_64:
///   - 9 ints (36 bytes) + 4 padding = offset 40 for tm_gmtoff (long, 8 bytes)
let bindDateTimeUtcOffset (appNodeId: NodeId) (z: PSGZipper) (_prim: PlatformPrimitive) : BindingResult =
    // Ensure localtime_r is declared at module level: ptr localtime_r(ptr, ptr)
    ensureExternDecl "localtime_r" [MLIRTypes.ptr; MLIRTypes.ptr] MLIRTypes.ptr z
    let ssas = requireNodeSSAs appNodeId z
    // SSA allocation:
    // 0: const 1 for alloca
    // 1: time_t alloca (8 bytes)
    // 2: tm struct alloca (56 bytes)
    // 3: const 56 for struct tm size
    // 4: current seconds (from clock_gettime)
    // 5: const 1000 for ms->sec conversion
    // 6: const 0 for clock id
    // 7: syscall 228 for clock_gettime
    // 8: timespec alloca
    // 9: const 0 for idx 0
    // 10: tv_sec ptr
    // 11: tv_sec loaded
    // 12: time_t ptr (for call)
    // 13: tm_gmtoff ptr
    // 14: const 40 for tm_gmtoff offset
    // 15: tm_gmtoff value (long)
    // 16: result (i32)
    let c1 = ssas.[0]
    let timeAllocaSSA = ssas.[1]
    let tmAllocaSSA = ssas.[2]
    let c56 = ssas.[3]
    let secFromClock = ssas.[4]
    let c1000 = ssas.[5]
    let clockId = ssas.[6]
    let syscallNum = ssas.[7]
    let timespecAlloca = ssas.[8]
    let idx0 = ssas.[9]
    let tvSecPtr = ssas.[10]
    let tvSecVal = ssas.[11]
    let timePtr = ssas.[12]
    let gmtoffPtr = ssas.[13]
    let c40 = ssas.[14]
    let gmtoffVal = ssas.[15]
    let resultSSA = ssas.[16]

    // Types
    let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

    let ops = [
        // Allocate structs
        MLIROp.ArithOp (ArithOp.ConstI (c1, 1L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (timespecAlloca, c1, timespecType, None))
        MLIROp.LLVMOp (LLVMOp.Alloca (timeAllocaSSA, c1, MLIRTypes.i64, None))  // time_t
        MLIROp.ArithOp (ArithOp.ConstI (c56, 56L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (tmAllocaSSA, c56, MLIRTypes.i8, None))  // struct tm (56 bytes)

        // Call clock_gettime to get current time
        MLIROp.ArithOp (ArithOp.ConstI (clockId, 0L, MLIRTypes.i64))  // CLOCK_REALTIME
        MLIROp.ArithOp (ArithOp.ConstI (syscallNum, 228L, MLIRTypes.i64))  // clock_gettime syscall
        MLIROp.LLVMOp (LLVMOp.InlineAsm (
            Some secFromClock,
            "syscall",
            "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
            [(syscallNum, MLIRTypes.i64); (clockId, MLIRTypes.i64); (timespecAlloca, MLIRTypes.ptr)],
            Some MLIRTypes.i64,
            true,
            false))

        // Load tv_sec from timespec
        MLIROp.ArithOp (ArithOp.ConstI (idx0, 0L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.GEP (tvSecPtr, timespecAlloca, [(idx0, MLIRTypes.i64)], timespecType))
        MLIROp.LLVMOp (LLVMOp.Load (tvSecVal, tvSecPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))

        // Store seconds to time_t variable for localtime_r
        MLIROp.LLVMOp (LLVMOp.Store (tvSecVal, timeAllocaSSA, MLIRTypes.i64, AtomicOrdering.NotAtomic))

        // Call localtime_r(time_t*, struct tm*) - libc function
        // Returns pointer to struct tm (same as second arg)
        MLIROp.FuncOp (FuncOp.FuncCall (
            Some timePtr,
            "localtime_r",
            [{ SSA = timeAllocaSSA; Type = MLIRTypes.ptr }; { SSA = tmAllocaSSA; Type = MLIRTypes.ptr }],
            MLIRTypes.ptr))

        // Extract tm_gmtoff at offset 40
        MLIROp.ArithOp (ArithOp.ConstI (c40, 40L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.GEP (gmtoffPtr, tmAllocaSSA, [(c40, MLIRTypes.i64)], MLIRTypes.i8))
        // Cast to i64 pointer and load
        MLIROp.LLVMOp (LLVMOp.Load (gmtoffVal, gmtoffPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))

        // Truncate to i32 for return
        MLIROp.ArithOp (ArithOp.TruncI (resultSSA, gmtoffVal, MLIRTypes.i64, MLIRTypes.i32))
    ]
    BoundOps (ops, Some { SSA = resultSSA; Type = MLIRTypes.i32 })

/// DateTime.toLocal - convert UTC milliseconds to local milliseconds
/// Adds the UTC offset (in seconds) * 1000 to the input
let bindDateTimeToLocal (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    // Ensure localtime_r is declared at module level
    ensureExternDecl "localtime_r" [MLIRTypes.ptr; MLIRTypes.ptr] MLIRTypes.ptr z
    match prim.Args with
    | [utcMs] ->
        // Get UTC offset via the same mechanism as utcOffset
        // Then: localMs = utcMs + (offsetSeconds * 1000)
        let ssas = requireNodeSSAs appNodeId z

        // First get the offset (reuse utcOffset logic but inline it)
        let c1 = ssas.[0]
        let timespecAlloca = ssas.[1]
        let timeAllocaSSA = ssas.[2]
        let c56 = ssas.[3]
        let tmAllocaSSA = ssas.[4]
        let clockId = ssas.[5]
        let syscallNum = ssas.[6]
        let _ = ssas.[7]  // syscall result (unused)
        let idx0 = ssas.[8]
        let tvSecPtr = ssas.[9]
        let tvSecVal = ssas.[10]
        let timePtr = ssas.[11]
        let c40 = ssas.[12]
        let gmtoffPtr = ssas.[13]
        let gmtoffVal = ssas.[14]
        let c1000 = ssas.[15]
        let offsetMs = ssas.[16]
        let resultSSA = ssas.[17]

        let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c1, 1L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (timespecAlloca, c1, timespecType, None))
            MLIROp.LLVMOp (LLVMOp.Alloca (timeAllocaSSA, c1, MLIRTypes.i64, None))
            MLIROp.ArithOp (ArithOp.ConstI (c56, 56L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (tmAllocaSSA, c56, MLIRTypes.i8, None))

            MLIROp.ArithOp (ArithOp.ConstI (clockId, 0L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.ConstI (syscallNum, 228L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                None,  // Don't need result
                "syscall",
                "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
                [(syscallNum, MLIRTypes.i64); (clockId, MLIRTypes.i64); (timespecAlloca, MLIRTypes.ptr)],
                Some MLIRTypes.i64,
                true,
                false))

            MLIROp.ArithOp (ArithOp.ConstI (idx0, 0L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.GEP (tvSecPtr, timespecAlloca, [(idx0, MLIRTypes.i64)], timespecType))
            MLIROp.LLVMOp (LLVMOp.Load (tvSecVal, tvSecPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))
            MLIROp.LLVMOp (LLVMOp.Store (tvSecVal, timeAllocaSSA, MLIRTypes.i64, AtomicOrdering.NotAtomic))

            MLIROp.FuncOp (FuncOp.FuncCall (
                Some timePtr,
                "localtime_r",
                [{ SSA = timeAllocaSSA; Type = MLIRTypes.ptr }; { SSA = tmAllocaSSA; Type = MLIRTypes.ptr }],
                MLIRTypes.ptr))

            MLIROp.ArithOp (ArithOp.ConstI (c40, 40L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.GEP (gmtoffPtr, tmAllocaSSA, [(c40, MLIRTypes.i64)], MLIRTypes.i8))
            MLIROp.LLVMOp (LLVMOp.Load (gmtoffVal, gmtoffPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))

            // Convert offset from seconds to milliseconds
            MLIROp.ArithOp (ArithOp.ConstI (c1000, 1000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.MulI (offsetMs, gmtoffVal, c1000, MLIRTypes.i64))

            // Add offset to input
            MLIROp.ArithOp (ArithOp.AddI (resultSSA, utcMs.SSA, offsetMs, MLIRTypes.i64))
        ]
        BoundOps (ops, Some { SSA = resultSSA; Type = MLIRTypes.i64 })
    | _ -> NotSupported "DateTime.toLocal requires 1 argument"

/// DateTime.toUtc - convert local milliseconds to UTC milliseconds
/// Subtracts the UTC offset (in seconds) * 1000 from the input
let bindDateTimeToUtc (appNodeId: NodeId) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    // Ensure localtime_r is declared at module level
    ensureExternDecl "localtime_r" [MLIRTypes.ptr; MLIRTypes.ptr] MLIRTypes.ptr z
    match prim.Args with
    | [localMs] ->
        let ssas = requireNodeSSAs appNodeId z

        let c1 = ssas.[0]
        let timespecAlloca = ssas.[1]
        let timeAllocaSSA = ssas.[2]
        let c56 = ssas.[3]
        let tmAllocaSSA = ssas.[4]
        let clockId = ssas.[5]
        let syscallNum = ssas.[6]
        let idx0 = ssas.[7]
        let tvSecPtr = ssas.[8]
        let tvSecVal = ssas.[9]
        let timePtr = ssas.[10]
        let c40 = ssas.[11]
        let gmtoffPtr = ssas.[12]
        let gmtoffVal = ssas.[13]
        let c1000 = ssas.[14]
        let offsetMs = ssas.[15]
        let resultSSA = ssas.[16]

        let timespecType = TStruct [MLIRTypes.i64; MLIRTypes.i64]

        let ops = [
            MLIROp.ArithOp (ArithOp.ConstI (c1, 1L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (timespecAlloca, c1, timespecType, None))
            MLIROp.LLVMOp (LLVMOp.Alloca (timeAllocaSSA, c1, MLIRTypes.i64, None))
            MLIROp.ArithOp (ArithOp.ConstI (c56, 56L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.Alloca (tmAllocaSSA, c56, MLIRTypes.i8, None))

            MLIROp.ArithOp (ArithOp.ConstI (clockId, 0L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.ConstI (syscallNum, 228L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.InlineAsm (
                None,
                "syscall",
                "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}",
                [(syscallNum, MLIRTypes.i64); (clockId, MLIRTypes.i64); (timespecAlloca, MLIRTypes.ptr)],
                Some MLIRTypes.i64,
                true,
                false))

            MLIROp.ArithOp (ArithOp.ConstI (idx0, 0L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.GEP (tvSecPtr, timespecAlloca, [(idx0, MLIRTypes.i64)], timespecType))
            MLIROp.LLVMOp (LLVMOp.Load (tvSecVal, tvSecPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))
            MLIROp.LLVMOp (LLVMOp.Store (tvSecVal, timeAllocaSSA, MLIRTypes.i64, AtomicOrdering.NotAtomic))

            MLIROp.FuncOp (FuncOp.FuncCall (
                Some timePtr,
                "localtime_r",
                [{ SSA = timeAllocaSSA; Type = MLIRTypes.ptr }; { SSA = tmAllocaSSA; Type = MLIRTypes.ptr }],
                MLIRTypes.ptr))

            MLIROp.ArithOp (ArithOp.ConstI (c40, 40L, MLIRTypes.i64))
            MLIROp.LLVMOp (LLVMOp.GEP (gmtoffPtr, tmAllocaSSA, [(c40, MLIRTypes.i64)], MLIRTypes.i8))
            MLIROp.LLVMOp (LLVMOp.Load (gmtoffVal, gmtoffPtr, MLIRTypes.i64, AtomicOrdering.NotAtomic))

            MLIROp.ArithOp (ArithOp.ConstI (c1000, 1000L, MLIRTypes.i64))
            MLIROp.ArithOp (ArithOp.MulI (offsetMs, gmtoffVal, c1000, MLIRTypes.i64))

            // Subtract offset from input
            MLIROp.ArithOp (ArithOp.SubI (resultSSA, localMs.SSA, offsetMs, MLIRTypes.i64))
        ]
        BoundOps (ops, Some { SSA = resultSSA; Type = MLIRTypes.i64 })
    | _ -> NotSupported "DateTime.toUtc requires 1 argument"

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
            // DateTime operations
            PlatformDispatch.register os arch "DateTime.now" bindDateTimeNow
            PlatformDispatch.register os arch "DateTime.utcNow" bindDateTimeNow  // Same as now for now
            PlatformDispatch.register os arch "DateTime.hour" bindDateTimeHour
            PlatformDispatch.register os arch "DateTime.minute" bindDateTimeMinute
            PlatformDispatch.register os arch "DateTime.second" bindDateTimeSecond
            PlatformDispatch.register os arch "DateTime.millisecond" bindDateTimeMillisecond
            // Timezone operations (require libc localtime_r)
            PlatformDispatch.register os arch "DateTime.utcOffset" bindDateTimeUtcOffset
            PlatformDispatch.register os arch "DateTime.toLocal" bindDateTimeToLocal
            PlatformDispatch.register os arch "DateTime.toUtc" bindDateTimeToUtc
