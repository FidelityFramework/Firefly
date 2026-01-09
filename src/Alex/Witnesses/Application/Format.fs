/// Application/Format - Value to String Conversions
///
/// ARCHITECTURAL PRINCIPLE: Returns STRUCTURED MLIROp lists, no sprintf.
/// Uses PSGZipper for SSA allocation, builds SCF regions for loops.
///
/// Migrated from ApplicationWitness.fs emitIntToString, emitFloatToString, emitStringToInt
module Alex.Witnesses.Application.Format

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
let intToString (z: PSGZipper) (intVal: Val) : MLIROp list * Val =
    // If input is i32, extend to i64 first
    let int64SSA, extOps =
        match intVal.Type with
        | TInt I32 ->
            let extSSA = freshSynthSSA z
            let extOp = MLIROp.ArithOp (ArithOp.ExtSI (extSSA, intVal.SSA, MLIRTypes.i32, MLIRTypes.i64))
            (extSSA, [extOp])
        | _ -> (intVal.SSA, [])

    // Constants
    let zeroSSA = freshSynthSSA z
    let oneSSA = freshSynthSSA z
    let tenSSA = freshSynthSSA z
    let asciiZeroSSA = freshSynthSSA z
    let bufSizeSSA = freshSynthSSA z
    let minusCharSSA = freshSynthSSA z
    let startPosSSA = freshSynthSSA z

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
    let bufSSA = freshSynthSSA z
    let allocOp = MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, bufSizeSSA, MLIRTypes.i8, None))

    // Check if negative
    let isNegSSA = freshSynthSSA z
    let isNegOp = MLIROp.ArithOp (ArithOp.CmpI (isNegSSA, ICmpPred.Slt, int64SSA, zeroSSA, MLIRTypes.i64))

    // Get absolute value: abs = select(isNeg, -n, n)
    let negatedSSA = freshSynthSSA z
    let absSSA = freshSynthSSA z
    let absOps = [
        MLIROp.ArithOp (ArithOp.SubI (negatedSSA, zeroSSA, int64SSA, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.Select (absSSA, isNegSSA, negatedSSA, int64SSA, MLIRTypes.i64))
    ]

    // Build while loop for digit extraction
    // State: (current_number: i64, current_pos: i64)
    // Guard: number > 0
    // Body: digit = n % 10, store '0' + digit at buf[pos], pos--, n = n / 10

    // Block arguments for the loop
    let nArgSSA = freshSynthSSA z      // Current number
    let posArgSSA = freshSynthSSA z    // Current position

    // Condition region: check if n > 0
    let condSSA = freshSynthSSA z
    let condOps = [
        MLIROp.ArithOp (ArithOp.CmpI (condSSA, ICmpPred.Sgt, nArgSSA, zeroSSA, MLIRTypes.i64))
        MLIROp.SCFOp (SCFOp.Condition (condSSA, [nArgSSA; posArgSSA]))
    ]
    let condRegion = singleBlockRegion "before" [blockArg nArgSSA MLIRTypes.i64; blockArg posArgSSA MLIRTypes.i64] condOps

    // Body region: extract digit, store, decrement
    let digitSSA = freshSynthSSA z
    let digit8SSA = freshSynthSSA z
    let charSSA = freshSynthSSA z
    let gepSSA = freshSynthSSA z
    let newPosSSA = freshSynthSSA z
    let newNSSA = freshSynthSSA z

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
        // yield newN, newPos
        MLIROp.SCFOp (SCFOp.Yield [newNSSA; newPosSSA])
    ]
    let bodyRegion = singleBlockRegion "do" [blockArg nArgSSA MLIRTypes.i64; blockArg posArgSSA MLIRTypes.i64] bodyOps

    // The while loop itself
    let loopResultSSA = freshSynthSSA z
    let loopResult2SSA = freshSynthSSA z
    let whileOp = MLIROp.SCFOp (SCFOp.While (
        [loopResultSSA; loopResult2SSA],
        condRegion,
        bodyRegion,
        [{ SSA = absSSA; Type = MLIRTypes.i64 }; { SSA = startPosSSA; Type = MLIRTypes.i64 }]
    ))

    // Get final position (second element of tuple + 1)
    let finalPosSSA = freshSynthSSA z
    let finalPosOp = MLIROp.ArithOp (ArithOp.AddI (finalPosSSA, loopResult2SSA, oneSSA, MLIRTypes.i64))

    // Handle special case: input was 0 (loop didn't execute)
    let wasZeroSSA = freshSynthSSA z
    let wasZeroOp = MLIROp.ArithOp (ArithOp.CmpI (wasZeroSSA, ICmpPred.Eq, absSSA, zeroSSA, MLIRTypes.i64))

    // If zero, write '0' at position 20
    let zeroCharSSA = freshSynthSSA z
    let zeroCharOp = MLIROp.ArithOp (ArithOp.ConstI (zeroCharSSA, 48L, MLIRTypes.i8))

    // Build scf.if for zero case
    let gepZeroSSA = freshSynthSSA z
    let zeroIfOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (gepZeroSSA, bufSSA, [(startPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (zeroCharSSA, gepZeroSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let zeroIfRegion = singleBlockRegion "then" [] zeroIfOps
    let zeroIfOp = MLIROp.SCFOp (SCFOp.If ([], wasZeroSSA, zeroIfRegion, None, []))

    // Adjust position for zero case
    let adjPosSSA = freshSynthSSA z
    let adjPosOp = MLIROp.ArithOp (ArithOp.Select (adjPosSSA, wasZeroSSA, startPosSSA, finalPosSSA, MLIRTypes.i64))

    // Handle negative: write '-' at pos-1 if negative
    let negPosSSA = freshSynthSSA z
    let negPosOp = MLIROp.ArithOp (ArithOp.SubI (negPosSSA, adjPosSSA, oneSSA, MLIRTypes.i64))

    // Build scf.if for negative case
    let gepNegSSA = freshSynthSSA z
    let negIfOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (gepNegSSA, bufSSA, [(negPosSSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Store (minusCharSSA, gepNegSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
    ]
    let negIfRegion = singleBlockRegion "then" [] negIfOps
    let negIfOp = MLIROp.SCFOp (SCFOp.If ([], isNegSSA, negIfRegion, None, []))

    // Select start position based on sign
    let startPtrPosSSA = freshSynthSSA z
    let startPtrPosOp = MLIROp.ArithOp (ArithOp.Select (startPtrPosSSA, isNegSSA, negPosSSA, adjPosSSA, MLIRTypes.i64))

    // Get pointer to start of string
    let strPtrSSA = freshSynthSSA z
    let strPtrOp = MLIROp.LLVMOp (LLVMOp.GEP (strPtrSSA, bufSSA, [(startPtrPosSSA, MLIRTypes.i64)], MLIRTypes.i8))

    // Calculate length: 21 - startPos
    let strLenSSA = freshSynthSSA z
    let strLenOp = MLIROp.ArithOp (ArithOp.SubI (strLenSSA, bufSizeSSA, startPtrPosSSA, MLIRTypes.i64))

    // Build fat string struct
    let undefSSA = freshSynthSSA z
    let undefOp = MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, MLIRTypes.nativeStr))

    let withPtrSSA = freshSynthSSA z
    let insertPtrOp = MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, strPtrSSA, [0], MLIRTypes.nativeStr))

    let fatStrSSA = freshSynthSSA z
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

/// Convert a float value to a string
/// Handles integer part, decimal point, and fractional part
let floatToString (z: PSGZipper) (floatVal: Val) : MLIROp list * Val =
    // For now, this is a simplified implementation
    // TODO: Implement full float-to-string with proper formatting
    // The logic follows the same pattern as intToString but with:
    // 1. Split into integer and fractional parts
    // 2. Convert integer part
    // 3. Add decimal point
    // 4. Convert fractional part (fixed precision)

    // Allocate buffer for "0.000000" style output (simplified)
    let bufSizeSSA = freshSynthSSA z
    let bufSSA = freshSynthSSA z

    let allocOps = [
        MLIROp.ArithOp (ArithOp.ConstI (bufSizeSSA, 32L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Alloca (bufSSA, bufSizeSSA, MLIRTypes.i8, None))
    ]

    // Convert float to integer part
    let intPartSSA = freshSynthSSA z
    let intPartOp = MLIROp.ArithOp (ArithOp.FPToSI (intPartSSA, floatVal.SSA, floatVal.Type, MLIRTypes.i64))

    // Convert integer part to string (reuse intToString logic via recursion conceptually)
    // For now, emit a simplified version - just the integer part
    let intVal = { SSA = intPartSSA; Type = MLIRTypes.i64 }
    let intOps, intResult = intToString z intVal

    // Return the integer part as a string (simplified)
    (allocOps @ [intPartOp] @ intOps, intResult)

// ═══════════════════════════════════════════════════════════════════════════
// STRING TO INTEGER CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a string to an integer
/// Handles sign prefix and digit characters
let stringToInt (z: PSGZipper) (strVal: Val) : MLIROp list * Val =
    // Constants
    let zeroSSA = freshSynthSSA z
    let oneSSA = freshSynthSSA z
    let tenSSA = freshSynthSSA z
    let asciiZeroSSA = freshSynthSSA z
    let minusCharSSA = freshSynthSSA z

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (tenSSA, 10L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (asciiZeroSSA, 48L, MLIRTypes.i8))
        MLIROp.ArithOp (ArithOp.ConstI (minusCharSSA, 45L, MLIRTypes.i8))
    ]

    // Extract pointer and length from fat string
    let ptrSSA = freshSynthSSA z
    let lenSSA = freshSynthSSA z
    let extractOps = [
        MLIROp.LLVMOp (LLVMOp.ExtractValue (ptrSSA, strVal.SSA, [0], strVal.Type))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (lenSSA, strVal.SSA, [1], strVal.Type))
    ]

    // Check first character for minus sign
    let firstCharSSA = freshSynthSSA z
    let isNegSSA = freshSynthSSA z
    let signCheckOps = [
        MLIROp.LLVMOp (LLVMOp.Load (firstCharSSA, ptrSSA, MLIRTypes.i8, AtomicOrdering.NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (isNegSSA, ICmpPred.Eq, firstCharSSA, minusCharSSA, MLIRTypes.i8))
    ]

    // Starting index: 1 if negative, 0 otherwise
    let zeroIdxSSA = freshSynthSSA z
    let oneIdxSSA = freshSynthSSA z
    let startIdxSSA = freshSynthSSA z
    let idxOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroIdxSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneIdxSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.Select (startIdxSSA, isNegSSA, oneIdxSSA, zeroIdxSSA, MLIRTypes.i64))
    ]

    // While loop: accumulate digits
    // State: (accumulator: i64, index: i64)
    // Guard: index < length
    // Body: acc = acc * 10 + (char - '0'), index++

    let accArgSSA = freshSynthSSA z
    let idxArgSSA = freshSynthSSA z

    // Condition region
    let condSSA = freshSynthSSA z
    let condOps = [
        MLIROp.ArithOp (ArithOp.CmpI (condSSA, ICmpPred.Slt, idxArgSSA, lenSSA, MLIRTypes.i64))
        MLIROp.SCFOp (SCFOp.Condition (condSSA, [accArgSSA; idxArgSSA]))
    ]
    let condRegion = singleBlockRegion "before" [blockArg accArgSSA MLIRTypes.i64; blockArg idxArgSSA MLIRTypes.i64] condOps

    // Body region
    let gepSSA = freshSynthSSA z
    let charSSA = freshSynthSSA z
    let char64SSA = freshSynthSSA z
    let asciiZero64SSA = freshSynthSSA z
    let digitSSA = freshSynthSSA z
    let acc10SSA = freshSynthSSA z
    let newAccSSA = freshSynthSSA z
    let newIdxSSA = freshSynthSSA z

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
        // yield
        MLIROp.SCFOp (SCFOp.Yield [newAccSSA; newIdxSSA])
    ]
    let bodyRegion = singleBlockRegion "do" [blockArg accArgSSA MLIRTypes.i64; blockArg idxArgSSA MLIRTypes.i64] bodyOps

    // While loop
    let loopResultSSA = freshSynthSSA z
    let loopResult2SSA = freshSynthSSA z
    let whileOp = MLIROp.SCFOp (SCFOp.While (
        [loopResultSSA; loopResult2SSA],
        condRegion,
        bodyRegion,
        [{ SSA = zeroSSA; Type = MLIRTypes.i64 }; { SSA = startIdxSSA; Type = MLIRTypes.i64 }]
    ))

    // Apply sign: result = select(isNeg, -acc, acc)
    let negatedSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z
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
