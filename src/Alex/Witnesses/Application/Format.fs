/// Application/Format - Value to String Conversions
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Witnesses OBSERVE and RETURN structured MLIROp lists.
/// All SSAs come from pre-computed SSAAssignment coeffect.
/// No freshSynthSSA - all SSAs are pre-assigned.
///
/// Migrated from ApplicationWitness.fs emitIntToString, emitFloatToString, emitStringToInt
module Alex.Witnesses.Application.Format

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Dialects.SCF.Templates
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

// Uses MLIRTypes.nativeStr for fat pointer type

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER TO STRING CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an integer value to a string (fat pointer)
/// Uses a while loop to extract digits, handles sign, zero case
/// Uses ~40 pre-assigned SSAs
let intToString (nodeId: NodeId) (z: PSGZipper) (intVal: Val) : MLIROp list * Val =
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // If input is i32, extend to i64 first
    let int64SSA, extOps =
        match intVal.Type with
        | TInt I32 ->
            let extSSA = nextSSA ()
            let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, intVal.SSA, MLIRTypes.i32, MLIRTypes.i64))
            (extSSA, [extOp])
        | _ -> (intVal.SSA, [])

    // Constants
    let zeroSSA = nextSSA ()
    let oneSSA = nextSSA ()
    let tenSSA = nextSSA ()
    let asciiZeroSSA = nextSSA ()
    let bufSizeSSA = nextSSA ()
    let minusCharSSA = nextSSA ()
    let startPosSSA = nextSSA ()

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (tenSSA, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (asciiZeroSSA, 48L, MLIRTypes.i8))  // '0' = 48
        MLIROp.ArithOp (ArithOp.ConstI (bufSizeSSA, 21L, MLIRTypes.i64))   // Max i64 digits + sign
        MLIROp.ArithOp (ArithOp.ConstI (minusCharSSA, 45L, MLIRTypes.i8))  // '-' = 45
        MLIROp.ArithOp (ArithOp.ConstI (startPosSSA, 20L, MLIRTypes.i64))  // Start at end of buffer
    ]

    // Allocate buffer
    let bufSSA = nextSSA ()
    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, bufSizeSSA, MLIRTypes.i8, None))

    // Check if negative
    let isNegSSA = nextSSA ()
    let isNegOp = MLIROp.ArithOp (ArithOp.CmpI (isNegSSA, ICmpPred.Slt, int64SSA, zeroSSA, MLIRTypes.i64))

    // Get absolute value: abs = select(isNeg, -n, n)
    let negatedSSA = nextSSA ()
    let absSSA = nextSSA ()
    let absOps = [
        MLIROp.ArithOp (ArithOp.SubI (negatedSSA, zeroSSA, int64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.Select (absSSA, isNegSSA, negatedSSA, int64SSA, MLIRTypes.i64))
    ]

    // Build while loop for digit extraction
    // State: (current_number: i64, current_pos: i64)
    // Guard: number > 0
    // Body: digit = n % 10, store '0' + digit at buf[pos], pos--, n = n / 10

    // Block arguments for the loop (typed vals used consistently)
    let nArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }      // Current number
    let posArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }    // Current position
    let nArgSSA = nArg.SSA
    let posArgSSA = posArg.SSA

    // Condition region: check if n > 0
    let condSSA = nextSSA ()
    let condOps = [
        MLIROp.ArithOp (ArithOp.CmpI (condSSA, ICmpPred.Sgt, nArgSSA, zeroSSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfCondition condSSA [nArg; posArg])
    ]
    // Empty label but keep args - args define SSAs available inside region
    let condRegion = singleBlockRegion "" [nArg; posArg] condOps

    // Body region: extract digit, store, decrement
    let digitSSA = nextSSA ()
    let digit8SSA = nextSSA ()
    let charSSA = nextSSA ()
    let gepSSA = nextSSA ()
    let newPosSSA = nextSSA ()
    let newNSSA = nextSSA ()

    let bodyOps = [
        // digit = n % 10
        MLIROp.ArithOp (ArithOp.RemSI (digitSSA, nArgSSA, tenSSA, MLIRTypes.i64))
        // digit8 = trunc digit to i8
        MLIROp.ArithOp (ArithOp.TruncI (digit8SSA, digitSSA, MLIRTypes.i64, MLIRTypes.i8))
        // char = '0' + digit
        MLIROp.ArithOp (ArithOp.AddI (charSSA, digit8SSA, asciiZeroSSA, MLIRTypes.i8))
        // gep = buf[pos]
        MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, bufSSA, [(posArgSSA, MLIRTypes.i64)], MLIRTypes.i8))
        // store char at gep
        MLIROp.LLVMOp (LLVMOp.Store (charSSA, gepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        // newPos = pos - 1
        MLIROp.ArithOp (ArithOp.SubI (newPosSSA, posArgSSA, oneSSA, MLIRTypes.i64))
        // newN = n / 10
        MLIROp.ArithOp (ArithOp.DivSI (newNSSA, nArgSSA, tenSSA, MLIRTypes.i64))
        // yield newN, newPos (same types as block args)
        MLIROp.SCFOp (scfYield [{ SSA = newNSSA; Type = MLIRTypes.i64 }; { SSA = newPosSSA; Type = MLIRTypes.i64 }])
    ]
    // "do" region needs explicit block args - use ^bb0 to trigger emission
    let bodyRegion = singleBlockRegion "bb0" [nArg; posArg] bodyOps

    // The while loop itself
    let loopResultSSA = nextSSA ()
    let loopResult2SSA = nextSSA ()
    let whileOp = MLIROp.SCFOp (SCFOp.While (
        [loopResultSSA; loopResult2SSA],
        condRegion,
        bodyRegion,
        [{ SSA = absSSA; Type = MLIRTypes.i64 }; { SSA = startPosSSA; Type = MLIRTypes.i64 }]
    ))

    // Get final position (second element of tuple + 1)
    let finalPosSSA = nextSSA ()
    let finalPosOp = MLIROp.ArithOp (ArithOp.AddI (finalPosSSA, loopResult2SSA, oneSSA, MLIRTypes.i64))

    // Handle special case: input was 0 (loop didn't execute)
    let wasZeroSSA = nextSSA ()
    let wasZeroOp = MLIROp.ArithOp (ArithOp.CmpI (wasZeroSSA, ICmpPred.Eq, absSSA, zeroSSA, MLIRTypes.i64))

    // If zero, write '0' at position 20
    let zeroCharSSA = nextSSA ()
    let zeroCharOp = MLIROp.ArithOp (ArithOp.ConstI (zeroCharSSA, 48L, MLIRTypes.i8))

    // Build scf.if for zero case
    let gepZeroSSA = nextSSA ()
    let zeroIfOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (gepZeroSSA, bufSSA, [(startPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (zeroCharSSA, gepZeroSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let zeroIfRegion = singleBlockRegion "then" [] zeroIfOps
    let zeroIfOp = MLIROp.SCFOp (SCFOp.If ([], wasZeroSSA, zeroIfRegion, None, []))

    // Adjust position for zero case
    let adjPosSSA = nextSSA ()
    let adjPosOp = MLIROp.ArithOp (ArithOp.Select (adjPosSSA, wasZeroSSA, startPosSSA, finalPosSSA, MLIRTypes.i64))

    // Handle negative: write '-' at pos-1 if negative
    let negPosSSA = nextSSA ()
    let negPosOp = MLIROp.ArithOp (ArithOp.SubI (negPosSSA, adjPosSSA, oneSSA, MLIRTypes.i64))

    // Build scf.if for negative case
    let gepNegSSA = nextSSA ()
    let negIfOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (gepNegSSA, bufSSA, [(negPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (minusCharSSA, gepNegSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let negIfRegion = singleBlockRegion "then" [] negIfOps
    let negIfOp = MLIROp.SCFOp (SCFOp.If ([], isNegSSA, negIfRegion, None, []))

    // Select start position based on sign
    let startPtrPosSSA = nextSSA ()
    let startPtrPosOp = MLIROp.ArithOp (ArithOp.Select (startPtrPosSSA, isNegSSA, negPosSSA, adjPosSSA, MLIRTypes.i64))

    // Get pointer to start of string
    let strPtrSSA = nextSSA ()
    let strPtrOp = MLIROp.LLVMOp (LLVMOp.GEP (strPtrSSA, bufSSA, [(startPtrPosSSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Calculate length: 21 - startPos
    let strLenSSA = nextSSA ()
    let strLenOp = MLIROp.ArithOp (ArithOp.SubI (strLenSSA, bufSizeSSA, startPtrPosSSA, MLIRTypes.i64))

    // Build fat string struct
    let undefSSA = nextSSA ()
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))

    let withPtrSSA = nextSSA ()
    let insertPtrOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, strPtrSSA, [0], MLIRTypes.nativeStr))

    let fatStrSSA = nextSSA ()
    let insertLenOp = MLIROp.LLVMOp (LLVMOp.InsertValue (fatStrSSA, withPtrSSA, strLenSSA, [1], MLIRTypes.nativeStr))

    // Combine all operations
    let allOps =
        extOps @
        constOps @
        [allocOp; isNegOp] @
        absOps @
        [whileOp; finalPosOp; wasZeroOp; zeroCharOp; zeroIfOp; adjPosOp; negPosOp; negIfOp] @
        [startPtrPosOp; strPtrOp; strLenOp; undefOp; insertPtrOp; insertLenOp]

    (allOps, { SSA = fatStrSSA; Type = MLIRTypes.nativeStr })

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT TO STRING CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a float value to a string (fat pointer)
/// Format: [-]digits.digits (6 decimal places)
/// Uses ~70 pre-assigned SSAs
let floatToString (nodeId: NodeId) (z: PSGZipper) (floatVal: Val) : MLIROp list * Val =
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Determine float type (f32 or f64)
    let floatType = floatVal.Type

    // ─────────────────────────────────────────────────────────────────────────
    // CONSTANTS
    // ─────────────────────────────────────────────────────────────────────────
    let zeroI64SSA = nextSSA ()
    let oneI64SSA = nextSSA ()
    let tenI64SSA = nextSSA ()
    let asciiZeroSSA = nextSSA ()
    let dotCharSSA = nextSSA ()
    let minusCharSSA = nextSSA ()
    let bufSizeSSA = nextSSA ()
    let precisionSSA = nextSSA ()       // 1000000 for 6 decimal places
    let sixSSA = nextSSA ()             // Number of fractional digits

    let zeroFSSA = nextSSA ()
    let oneFSSA = nextSSA ()
    let precisionFSSA = nextSSA ()      // 1000000.0

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroI64SSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneI64SSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (tenI64SSA, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (asciiZeroSSA, 48L, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.ConstI (dotCharSSA, 46L, MLIRTypes.i8))      // '.'
        MLIROp.ArithOp (ArithOp.ConstI (minusCharSSA, 45L, MLIRTypes.i8))    // '-'
        MLIROp.ArithOp (ArithOp.ConstI (bufSizeSSA, 32L, MLIRTypes.i64))     // Max buffer size
        MLIROp.ArithOp (ArithOp.ConstI (precisionSSA, 1000000L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (sixSSA, 6L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstF (zeroFSSA, 0.0, floatType))
        MLIROp.ArithOp (ArithOp.ConstF (oneFSSA, 1.0, floatType))
        MLIROp.ArithOp (ArithOp.ConstF (precisionFSSA, 1000000.0, floatType))
    ]

    // ─────────────────────────────────────────────────────────────────────────
    // BUFFER ALLOCATION
    // ─────────────────────────────────────────────────────────────────────────
    let bufSSA = nextSSA ()
    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, bufSizeSSA, MLIRTypes.i8, None))

    // ─────────────────────────────────────────────────────────────────────────
    // SIGN HANDLING
    // ─────────────────────────────────────────────────────────────────────────
    let isNegSSA = nextSSA ()
    let negatedFSSA = nextSSA ()
    let absFSSA = nextSSA ()

    let signOps = [
        // Check if negative
        MLIROp.ArithOp (ArithOp.CmpF (isNegSSA, FCmpPred.OLt, floatVal.SSA, zeroFSSA, floatType))
        // Negate: -f
        MLIROp.ArithOp (ArithOp.NegF (negatedFSSA, floatVal.SSA, floatType))
        // Select absolute value (Select is type-polymorphic)
        MLIROp.ArithOp (ArithOp.Select (absFSSA, isNegSSA, negatedFSSA, floatVal.SSA, floatType))
    ]

    // ─────────────────────────────────────────────────────────────────────────
    // SPLIT INTO INTEGER AND FRACTIONAL PARTS
    // ─────────────────────────────────────────────────────────────────────────
    let intPartI64SSA = nextSSA ()      // Integer part as i64
    let intPartFSSA = nextSSA ()        // Integer part back to float
    let fracPartFSSA = nextSSA ()       // Fractional part as float
    let fracScaledFSSA = nextSSA ()     // Fractional * 1000000
    let fracPartI64SSA = nextSSA ()     // Fractional part as i64

    let splitOps = [
        // Integer part = trunc(abs)
        MLIROp.ArithOp (ArithOp.FPToSI (intPartI64SSA, absFSSA, floatType, MLIRTypes.i64))
        // Convert back to float for subtraction
        MLIROp.ArithOp (ArithOp.SIToFP (intPartFSSA, intPartI64SSA, MLIRTypes.i64, floatType))
        // Fractional = abs - intPart
        MLIROp.ArithOp (ArithOp.SubF (fracPartFSSA, absFSSA, intPartFSSA, floatType))
        // Scale fractional: frac * 1000000
        MLIROp.ArithOp (ArithOp.MulF (fracScaledFSSA, fracPartFSSA, precisionFSSA, floatType))
        // Convert scaled frac to i64
        MLIROp.ArithOp (ArithOp.FPToSI (fracPartI64SSA, fracScaledFSSA, floatType, MLIRTypes.i64))
    ]

    // ─────────────────────────────────────────────────────────────────────────
    // WRITE FRACTIONAL DIGITS (RIGHT TO LEFT, 6 DIGITS WITH LEADING ZEROS)
    // Position 31..26 in buffer (6 digits)
    // ─────────────────────────────────────────────────────────────────────────
    let startFracPosSSA = nextSSA ()
    let startFracPosOp = MLIROp.ArithOp (ArithOp.ConstI (startFracPosSSA, 31L, MLIRTypes.i64))

    // Fractional digit loop - always writes 6 digits
    let fracNArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let fracPosArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let fracCountArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }

    let fracCondSSA = nextSSA ()
    let fracCondOps = [
        MLIROp.ArithOp (ArithOp.CmpI (fracCondSSA, ICmpPred.Sgt, fracCountArg.SSA, zeroI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfCondition fracCondSSA [fracNArg; fracPosArg; fracCountArg])
    ]
    let fracCondRegion = singleBlockRegion "" [fracNArg; fracPosArg; fracCountArg] fracCondOps

    let fracDigitSSA = nextSSA ()
    let fracDigit8SSA = nextSSA ()
    let fracCharSSA = nextSSA ()
    let fracGepSSA = nextSSA ()
    let fracNewPosSSA = nextSSA ()
    let fracNewNSSA = nextSSA ()
    let fracNewCountSSA = nextSSA ()

    let fracBodyOps = [
        MLIROp.ArithOp (ArithOp.RemSI (fracDigitSSA, fracNArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.TruncI (fracDigit8SSA, fracDigitSSA, MLIRTypes.i64, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.AddI (fracCharSSA, fracDigit8SSA, asciiZeroSSA, MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.GEP (fracGepSSA, bufSSA, [(fracPosArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (fracCharSSA, fracGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.SubI (fracNewPosSSA, fracPosArg.SSA, oneI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.DivSI (fracNewNSSA, fracNArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.SubI (fracNewCountSSA, fracCountArg.SSA, oneI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [
            { SSA = fracNewNSSA; Type = MLIRTypes.i64 }
            { SSA = fracNewPosSSA; Type = MLIRTypes.i64 }
            { SSA = fracNewCountSSA; Type = MLIRTypes.i64 }
        ])
    ]
    let fracBodyRegion = singleBlockRegion "bb0" [fracNArg; fracPosArg; fracCountArg] fracBodyOps

    let fracLoopResult1SSA = nextSSA ()
    let fracLoopResult2SSA = nextSSA ()
    let fracLoopResult3SSA = nextSSA ()
    let fracWhileOp = MLIROp.SCFOp (SCFOp.While (
        [fracLoopResult1SSA; fracLoopResult2SSA; fracLoopResult3SSA],
        fracCondRegion,
        fracBodyRegion,
        [
            { SSA = fracPartI64SSA; Type = MLIRTypes.i64 }
            { SSA = startFracPosSSA; Type = MLIRTypes.i64 }
            { SSA = sixSSA; Type = MLIRTypes.i64 }
        ]
    ))

    // ─────────────────────────────────────────────────────────────────────────
    // WRITE DECIMAL POINT AT POSITION 25
    // ─────────────────────────────────────────────────────────────────────────
    let dotPosSSA = nextSSA ()
    let dotGepSSA = nextSSA ()
    let dotOps = [
        MLIROp.ArithOp (ArithOp.ConstI (dotPosSSA, 25L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.GEP (dotGepSSA, bufSSA, [(dotPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (dotCharSSA, dotGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]

    // ─────────────────────────────────────────────────────────────────────────
    // WRITE INTEGER PART (RIGHT TO LEFT FROM POSITION 24)
    // ─────────────────────────────────────────────────────────────────────────
    let startIntPosSSA = nextSSA ()
    let startIntPosOp = MLIROp.ArithOp (ArithOp.ConstI (startIntPosSSA, 24L, MLIRTypes.i64))

    let intNArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let intPosArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }

    let intCondSSA = nextSSA ()
    let intCondOps = [
        MLIROp.ArithOp (ArithOp.CmpI (intCondSSA, ICmpPred.Sgt, intNArg.SSA, zeroI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfCondition intCondSSA [intNArg; intPosArg])
    ]
    let intCondRegion = singleBlockRegion "" [intNArg; intPosArg] intCondOps

    let intDigitSSA = nextSSA ()
    let intDigit8SSA = nextSSA ()
    let intCharSSA = nextSSA ()
    let intGepSSA = nextSSA ()
    let intNewPosSSA = nextSSA ()
    let intNewNSSA = nextSSA ()

    let intBodyOps = [
        MLIROp.ArithOp (ArithOp.RemSI (intDigitSSA, intNArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.TruncI (intDigit8SSA, intDigitSSA, MLIRTypes.i64, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.AddI (intCharSSA, intDigit8SSA, asciiZeroSSA, MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.GEP (intGepSSA, bufSSA, [(intPosArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (intCharSSA, intGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.SubI (intNewPosSSA, intPosArg.SSA, oneI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.DivSI (intNewNSSA, intNArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [
            { SSA = intNewNSSA; Type = MLIRTypes.i64 }
            { SSA = intNewPosSSA; Type = MLIRTypes.i64 }
        ])
    ]
    let intBodyRegion = singleBlockRegion "bb0" [intNArg; intPosArg] intBodyOps

    let intLoopResult1SSA = nextSSA ()
    let intLoopResult2SSA = nextSSA ()
    let intWhileOp = MLIROp.SCFOp (SCFOp.While (
        [intLoopResult1SSA; intLoopResult2SSA],
        intCondRegion,
        intBodyRegion,
        [
            { SSA = intPartI64SSA; Type = MLIRTypes.i64 }
            { SSA = startIntPosSSA; Type = MLIRTypes.i64 }
        ]
    ))

    // Get final position after integer loop
    let finalIntPosSSA = nextSSA ()
    let finalIntPosOp = MLIROp.ArithOp (ArithOp.AddI (finalIntPosSSA, intLoopResult2SSA, oneI64SSA, MLIRTypes.i64))

    // ─────────────────────────────────────────────────────────────────────────
    // HANDLE ZERO INTEGER PART (write '0' if intPart was 0)
    // ─────────────────────────────────────────────────────────────────────────
    let wasZeroSSA = nextSSA ()
    let wasZeroOp = MLIROp.ArithOp (ArithOp.CmpI (wasZeroSSA, ICmpPred.Eq, intPartI64SSA, zeroI64SSA, MLIRTypes.i64))

    let zeroCharConstSSA = nextSSA ()
    let zeroGepSSA = nextSSA ()
    let zeroIfOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroCharConstSSA, 48L, MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.GEP (zeroGepSSA, bufSSA, [(startIntPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (zeroCharConstSSA, zeroGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let zeroIfRegion = singleBlockRegion "then" [] zeroIfOps
    let zeroIfOp = MLIROp.SCFOp (SCFOp.If ([], wasZeroSSA, zeroIfRegion, None, []))

    // Adjust position for zero case
    let adjIntPosSSA = nextSSA ()
    let adjIntPosOp = MLIROp.ArithOp (ArithOp.Select (adjIntPosSSA, wasZeroSSA, startIntPosSSA, finalIntPosSSA, MLIRTypes.i64))

    // ─────────────────────────────────────────────────────────────────────────
    // HANDLE NEGATIVE SIGN
    // ─────────────────────────────────────────────────────────────────────────
    let negPosSSA = nextSSA ()
    let negPosOp = MLIROp.ArithOp (ArithOp.SubI (negPosSSA, adjIntPosSSA, oneI64SSA, MLIRTypes.i64))

    let negGepSSA = nextSSA ()
    let negIfOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (negGepSSA, bufSSA, [(negPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (minusCharSSA, negGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let negIfRegion = singleBlockRegion "then" [] negIfOps
    let negIfOp = MLIROp.SCFOp (SCFOp.If ([], isNegSSA, negIfRegion, None, []))

    // Select start position based on sign
    let startPtrPosSSA = nextSSA ()
    let startPtrPosOp = MLIROp.ArithOp (ArithOp.Select (startPtrPosSSA, isNegSSA, negPosSSA, adjIntPosSSA, MLIRTypes.i64))

    // ─────────────────────────────────────────────────────────────────────────
    // BUILD FAT STRING
    // ─────────────────────────────────────────────────────────────────────────
    let strPtrSSA = nextSSA ()
    let strPtrOp = MLIROp.LLVMOp (LLVMOp.GEP (strPtrSSA, bufSSA, [(startPtrPosSSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Length = 32 - startPos (covers integer + '.' + 6 fractional digits)
    let strLenSSA = nextSSA ()
    let strLenOp = MLIROp.ArithOp (ArithOp.SubI (strLenSSA, bufSizeSSA, startPtrPosSSA, MLIRTypes.i64))

    let undefSSA = nextSSA ()
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))

    let withPtrSSA = nextSSA ()
    let insertPtrOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, strPtrSSA, [0], MLIRTypes.nativeStr))

    let fatStrSSA = nextSSA ()
    let insertLenOp = MLIROp.LLVMOp (LLVMOp.InsertValue (fatStrSSA, withPtrSSA, strLenSSA, [1], MLIRTypes.nativeStr))

    // ─────────────────────────────────────────────────────────────────────────
    // COMBINE ALL OPERATIONS
    // ─────────────────────────────────────────────────────────────────────────
    let allOps =
        constOps @
        [allocOp] @
        signOps @
        splitOps @
        [startFracPosOp; fracWhileOp] @
        dotOps @
        [startIntPosOp; intWhileOp; finalIntPosOp; wasZeroOp; zeroIfOp; adjIntPosOp] @
        [negPosOp; negIfOp; startPtrPosOp] @
        [strPtrOp; strLenOp; undefOp; insertPtrOp; insertLenOp]

    (allOps, { SSA = fatStrSSA; Type = MLIRTypes.nativeStr })

// ═══════════════════════════════════════════════════════════════════════════
// STRING TO INTEGER CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a string to an integer
/// Handles sign prefix and digit characters
/// Uses ~30 pre-assigned SSAs
let stringToInt (nodeId: NodeId) (z: PSGZipper) (strVal: Val) : MLIROp list * Val =
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Constants
    let zeroSSA = nextSSA ()
    let oneSSA = nextSSA ()
    let tenSSA = nextSSA ()
    let asciiZeroSSA = nextSSA ()
    let minusCharSSA = nextSSA ()

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (tenSSA, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (asciiZeroSSA, 48L, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.ConstI (minusCharSSA, 45L, MLIRTypes.i8))
    ]

    // Extract pointer and length from fat string
    let ptrSSA = nextSSA ()
    let lenSSA = nextSSA ()
    let extractOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptrSSA, strVal.SSA, [0], strVal.Type))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, strVal.SSA, [1], strVal.Type))
    ]

    // Check first character for minus sign
    let firstCharSSA = nextSSA ()
    let isNegSSA = nextSSA ()
    let signCheckOps = [
        MLIROp.LLVMOp (LLVMOp.Load (firstCharSSA, ptrSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (isNegSSA, ICmpPred.Eq, firstCharSSA, minusCharSSA, MLIRTypes.i8))
    ]

    // Starting index: 1 if negative, 0 otherwise
    let zeroIdxSSA = nextSSA ()
    let oneIdxSSA = nextSSA ()
    let startIdxSSA = nextSSA ()
    let idxOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroIdxSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneIdxSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.Select (startIdxSSA, isNegSSA, oneIdxSSA, zeroIdxSSA, MLIRTypes.i64))
    ]

    // While loop: accumulate digits
    // State: (accumulator: i64, index: i64)
    // Guard: index < length
    // Body: acc = acc * 10 + (char - '0'), index++

    let accArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let idxArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let accArgSSA = accArg.SSA
    let idxArgSSA = idxArg.SSA

    // Condition region
    let condSSA = nextSSA ()
    let condOps = [
        MLIROp.ArithOp (ArithOp.CmpI (condSSA, ICmpPred.Slt, idxArgSSA, lenSSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfCondition condSSA [accArg; idxArg])
    ]
    let condRegion = singleBlockRegion "" [accArg; idxArg] condOps

    // Body region
    let gepSSA = nextSSA ()
    let charSSA = nextSSA ()
    let char64SSA = nextSSA ()
    let asciiZero64SSA = nextSSA ()
    let digitSSA = nextSSA ()
    let acc10SSA = nextSSA ()
    let newAccSSA = nextSSA ()
    let newIdxSSA = nextSSA ()

    let bodyOps = [
        // gep = ptr[idx]
        MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptrSSA, [(idxArgSSA, MLIRTypes.i64)], MLIRTypes.i8))
        // char = load gep
        MLIROp.LLVMOp (LLVMOp.Load (charSSA, gepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        // char64 = ext char to i64
        MLIROp.ArithOp (ArithOp.ExtUI (char64SSA, charSSA, MLIRTypes.i8, MLIRTypes.i64))
        // asciiZero64
        MLIROp.ArithOp (ArithOp.ConstI (asciiZero64SSA, 48L, MLIRTypes.i64))
        // digit = char64 - asciiZero64
        MLIROp.ArithOp (ArithOp.SubI (digitSSA, char64SSA, asciiZero64SSA, MLIRTypes.i64))
        // acc10 = acc * 10
        MLIROp.ArithOp (ArithOp.MulI (acc10SSA, accArgSSA, tenSSA, MLIRTypes.i64))
        // newAcc = acc10 + digit
        MLIROp.ArithOp (ArithOp.AddI (newAccSSA, acc10SSA, digitSSA, MLIRTypes.i64))
        // newIdx = idx + 1
        MLIROp.ArithOp (ArithOp.AddI (newIdxSSA, idxArgSSA, oneSSA, MLIRTypes.i64))
        // yield (same types as block args)
        MLIROp.SCFOp (scfYield [{ SSA = newAccSSA; Type = MLIRTypes.i64 }; { SSA = newIdxSSA; Type = MLIRTypes.i64 }])
    ]
    let bodyRegion = singleBlockRegion "bb0" [accArg; idxArg] bodyOps

    // While loop
    let loopResultSSA = nextSSA ()
    let loopResult2SSA = nextSSA ()
    let whileOp = MLIROp.SCFOp (SCFOp.While (
        [loopResultSSA; loopResult2SSA],
        condRegion,
        bodyRegion,
        [{ SSA = zeroSSA; Type = MLIRTypes.i64 }; { SSA = startIdxSSA; Type = MLIRTypes.i64 }]
    ))

    // Apply sign: result = select(isNeg, -acc, acc)
    let negatedSSA = nextSSA ()
    let resultSSA = nextSSA ()
    let signOps = [
        MLIROp.ArithOp (ArithOp.SubI (negatedSSA, zeroSSA, loopResultSSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.Select (resultSSA, isNegSSA, negatedSSA, loopResultSSA, MLIRTypes.i64))
    ]

    let allOps =
        constOps @
        extractOps @
        signCheckOps @
        idxOps @
        [whileOp] @
        signOps

    (allOps, { SSA = resultSSA; Type = MLIRTypes.i64 })

/// Convert string to f64: Parse.float
/// Algorithm: Two-pass parsing (integer part, then fractional part)
/// Handles: [sign] digits [. digits]
let stringToFloat (nodeId: NodeId) (z: PSGZipper) (strVal: Val) : MLIROp list * Val =
    let ssas = requireNodeSSAs nodeId z
    let mutable ssaIdx = 0
    let nextSSA () =
        let ssa = ssas.[ssaIdx]
        ssaIdx <- ssaIdx + 1
        ssa

    // Constants
    let zeroI64SSA = nextSSA ()
    let oneI64SSA = nextSSA ()
    let tenI64SSA = nextSSA ()
    let dotCharSSA = nextSSA ()
    let minusCharSSA = nextSSA ()
    let zeroF64SSA = nextSSA ()
    let tenF64SSA = nextSSA ()

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroI64SSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneI64SSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (tenI64SSA, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (dotCharSSA, 46L, MLIRTypes.i8))  // '.'
        MLIROp.ArithOp (ArithOp.ConstI (minusCharSSA, 45L, MLIRTypes.i8))  // '-'
        MLIROp.ArithOp (ArithOp.ConstF (zeroF64SSA, 0.0, MLIRTypes.f64))
        MLIROp.ArithOp (ArithOp.ConstF (tenF64SSA, 10.0, MLIRTypes.f64))
    ]

    // Extract pointer and length from fat string
    let ptrSSA = nextSSA ()
    let lenSSA = nextSSA ()
    let extractOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptrSSA, strVal.SSA, [0], strVal.Type))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, strVal.SSA, [1], strVal.Type))
    ]

    // Check first character for minus sign
    let firstCharSSA = nextSSA ()
    let isNegSSA = nextSSA ()
    let signCheckOps = [
        MLIROp.LLVMOp (LLVMOp.Load (firstCharSSA, ptrSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (isNegSSA, ICmpPred.Eq, firstCharSSA, minusCharSSA, MLIRTypes.i8))
    ]

    // Starting index: 1 if negative, 0 otherwise
    let startIdxSSA = nextSSA ()
    let idxOps = [
        MLIROp.ArithOp (ArithOp.Select (startIdxSSA, isNegSSA, oneI64SSA, zeroI64SSA, MLIRTypes.i64))
    ]

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 1: Parse integer part (until '.' or end)
    // ═══════════════════════════════════════════════════════════════════════════

    // While loop state: (acc: i64, idx: i64)
    let intAccArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let intIdxArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }

    // Condition: idx < len AND char != '.'
    let intCondGepSSA = nextSSA ()
    let intCondCharSSA = nextSSA ()
    let intInBoundsSSA = nextSSA ()
    let intNotDotSSA = nextSSA ()
    let intContinueSSA = nextSSA ()

    let intCondOps = [
        MLIROp.ArithOp (ArithOp.CmpI (intInBoundsSSA, ICmpPred.Slt, intIdxArg.SSA, lenSSA, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.GEP (intCondGepSSA, ptrSSA, [(intIdxArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Load (intCondCharSSA, intCondGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (intNotDotSSA, ICmpPred.Ne, intCondCharSSA, dotCharSSA, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.AndI (intContinueSSA, intInBoundsSSA, intNotDotSSA, MLIRTypes.i1))
        MLIROp.SCFOp (scfCondition intContinueSSA [intAccArg; intIdxArg])
    ]
    let intCondRegion = singleBlockRegion "" [intAccArg; intIdxArg] intCondOps

    // Body: acc = acc * 10 + digit, idx++
    let intBodyGepSSA = nextSSA ()
    let intBodyCharSSA = nextSSA ()
    let intBodyChar64SSA = nextSSA ()
    let intBodyDigitSSA = nextSSA ()
    let intBodyAcc10SSA = nextSSA ()
    let intBodyNewAccSSA = nextSSA ()
    let intBodyNewIdxSSA = nextSSA ()
    let asciiZero64SSA = nextSSA ()

    let intBodyOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (intBodyGepSSA, ptrSSA, [(intIdxArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Load (intBodyCharSSA, intBodyGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.ExtUI (intBodyChar64SSA, intBodyCharSSA, MLIRTypes.i8, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (asciiZero64SSA, 48L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.SubI (intBodyDigitSSA, intBodyChar64SSA, asciiZero64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.MulI (intBodyAcc10SSA, intAccArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AddI (intBodyNewAccSSA, intBodyAcc10SSA, intBodyDigitSSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AddI (intBodyNewIdxSSA, intIdxArg.SSA, oneI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [{ SSA = intBodyNewAccSSA; Type = MLIRTypes.i64 }; { SSA = intBodyNewIdxSSA; Type = MLIRTypes.i64 }])
    ]
    let intBodyRegion = singleBlockRegion "bb0" [intAccArg; intIdxArg] intBodyOps

    let intLoopResultSSA = nextSSA ()
    let intLoopIdxSSA = nextSSA ()
    let intWhileOp = MLIROp.SCFOp (SCFOp.While (
        [intLoopResultSSA; intLoopIdxSSA],
        intCondRegion,
        intBodyRegion,
        [{ SSA = zeroI64SSA; Type = MLIRTypes.i64 }; { SSA = startIdxSSA; Type = MLIRTypes.i64 }]
    ))

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 2: Parse fractional part (after '.')
    // ═══════════════════════════════════════════════════════════════════════════

    // Skip the '.' - fracStartIdx = intLoopIdx + 1
    let fracStartIdxSSA = nextSSA ()
    let skipDotOp = MLIROp.ArithOp (ArithOp.AddI (fracStartIdxSSA, intLoopIdxSSA, oneI64SSA, MLIRTypes.i64))

    // While loop state: (frac: i64, divisor: i64, idx: i64)
    let fracAccArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let fracDivArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }
    let fracIdxArg = { SSA = nextSSA (); Type = MLIRTypes.i64 }

    // Condition: idx < len
    let fracInBoundsSSA = nextSSA ()
    let fracCondOps = [
        MLIROp.ArithOp (ArithOp.CmpI (fracInBoundsSSA, ICmpPred.Slt, fracIdxArg.SSA, lenSSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfCondition fracInBoundsSSA [fracAccArg; fracDivArg; fracIdxArg])
    ]
    let fracCondRegion = singleBlockRegion "" [fracAccArg; fracDivArg; fracIdxArg] fracCondOps

    // Body: frac = frac * 10 + digit, divisor *= 10, idx++
    let fracBodyGepSSA = nextSSA ()
    let fracBodyCharSSA = nextSSA ()
    let fracBodyChar64SSA = nextSSA ()
    let fracBodyAsciiZeroSSA = nextSSA ()
    let fracBodyDigitSSA = nextSSA ()
    let fracBodyAcc10SSA = nextSSA ()
    let fracBodyNewAccSSA = nextSSA ()
    let fracBodyNewDivSSA = nextSSA ()
    let fracBodyNewIdxSSA = nextSSA ()

    let fracBodyOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (fracBodyGepSSA, ptrSSA, [(fracIdxArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Load (fracBodyCharSSA, fracBodyGepSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.ExtUI (fracBodyChar64SSA, fracBodyCharSSA, MLIRTypes.i8, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (fracBodyAsciiZeroSSA, 48L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.SubI (fracBodyDigitSSA, fracBodyChar64SSA, fracBodyAsciiZeroSSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.MulI (fracBodyAcc10SSA, fracAccArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AddI (fracBodyNewAccSSA, fracBodyAcc10SSA, fracBodyDigitSSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.MulI (fracBodyNewDivSSA, fracDivArg.SSA, tenI64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.AddI (fracBodyNewIdxSSA, fracIdxArg.SSA, oneI64SSA, MLIRTypes.i64))
        MLIROp.SCFOp (scfYield [
            { SSA = fracBodyNewAccSSA; Type = MLIRTypes.i64 }
            { SSA = fracBodyNewDivSSA; Type = MLIRTypes.i64 }
            { SSA = fracBodyNewIdxSSA; Type = MLIRTypes.i64 }
        ])
    ]
    let fracBodyRegion = singleBlockRegion "bb0" [fracAccArg; fracDivArg; fracIdxArg] fracBodyOps

    let fracLoopResultSSA = nextSSA ()
    let fracLoopDivSSA = nextSSA ()
    let fracLoopIdxSSA = nextSSA ()
    let fracWhileOp = MLIROp.SCFOp (SCFOp.While (
        [fracLoopResultSSA; fracLoopDivSSA; fracLoopIdxSSA],
        fracCondRegion,
        fracBodyRegion,
        [
            { SSA = zeroI64SSA; Type = MLIRTypes.i64 }     // frac starts at 0
            { SSA = oneI64SSA; Type = MLIRTypes.i64 }      // divisor starts at 1
            { SSA = fracStartIdxSSA; Type = MLIRTypes.i64 }  // idx starts after '.'
        ]
    ))

    // ═══════════════════════════════════════════════════════════════════════════
    // COMBINE: result = intPart + fracPart / divisor
    // ═══════════════════════════════════════════════════════════════════════════

    let intPartF64SSA = nextSSA ()
    let fracPartF64SSA = nextSSA ()
    let divF64SSA = nextSSA ()
    let fracValueSSA = nextSSA ()
    let combinedSSA = nextSSA ()

    let combineOps = [
        // Convert integer part to f64
        MLIROp.ArithOp (ArithOp.SIToFP (intPartF64SSA, intLoopResultSSA, MLIRTypes.i64, MLIRTypes.f64))
        // Convert fractional accumulator to f64
        MLIROp.ArithOp (ArithOp.SIToFP (fracPartF64SSA, fracLoopResultSSA, MLIRTypes.i64, MLIRTypes.f64))
        // Convert divisor to f64
        MLIROp.ArithOp (ArithOp.SIToFP (divF64SSA, fracLoopDivSSA, MLIRTypes.i64, MLIRTypes.f64))
        // fracValue = fracPart / divisor
        MLIROp.ArithOp (ArithOp.DivF (fracValueSSA, fracPartF64SSA, divF64SSA, MLIRTypes.f64))
        // combined = intPart + fracValue
        MLIROp.ArithOp (ArithOp.AddF (combinedSSA, intPartF64SSA, fracValueSSA, MLIRTypes.f64))
    ]

    // Apply sign: result = select(isNeg, -combined, combined)
    let negatedSSA = nextSSA ()
    let resultSSA = nextSSA ()
    let signOps = [
        MLIROp.ArithOp (ArithOp.NegF (negatedSSA, combinedSSA, MLIRTypes.f64))
        MLIROp.ArithOp (ArithOp.Select (resultSSA, isNegSSA, negatedSSA, combinedSSA, MLIRTypes.f64))
    ]

    let allOps =
        constOps @
        extractOps @
        signCheckOps @
        idxOps @
        [intWhileOp] @
        [skipDotOp] @
        [fracWhileOp] @
        combineOps @
        signOps

    (allOps, { SSA = resultSSA; Type = MLIRTypes.f64 })
