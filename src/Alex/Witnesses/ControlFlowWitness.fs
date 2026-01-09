/// Control Flow Witness - Witness control flow constructs to MLIR (SCF dialect)
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Witnesses OBSERVE and RETURN structured MLIROp lists.
/// They do NOT emit strings. ZERO SPRINTF for MLIR generation.
///
/// Uses:
/// - freshSynthSSA for synthesized values (constants, temporaries)
/// - lookupNodeSSA for PSG node SSAs (from coeffects)
/// - SCFTemplates for structured SCF operations
/// - ArithTemplates for arithmetic operations
/// - LLVMTemplates for LLVM operations
module Alex.Witnesses.ControlFlowWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

module SCF = Alex.Dialects.SCF.Templates
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis

// ═══════════════════════════════════════════════════════════════════════════
// TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Map NativeType to MLIRType (delegated to TypeMapping)
let private mapType = mapNativeType

/// Check if a type is unit/void
let private isUnitType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> true
    | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// INLINE STRLEN (Entry Point Boundary Only)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate inline strlen using scf.while loop
/// Used ONLY at entry point boundary to convert C strings to F# fat pointers
/// Returns: (ops, lengthSSA)
let inlineStrlen (z: PSGZipper) (ptrSSA: SSA) : MLIROp list * SSA =
    // Constants
    let zeroSSA = freshSynthSSA z
    let oneSSA = freshSynthSSA z
    let nullByteSSA = freshSynthSSA z

    let constOps = [
        MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
        MLIROp.ArithOp (ArithOp.ConstI (nullByteSSA, 0L, MLIRTypes.i8))
    ]

    // Loop iteration argument: current index
    let iterArg: Val = { SSA = zeroSSA; Type = MLIRTypes.i64 }

    // Condition region: load byte, compare to null
    let gepSSA = freshSynthSSA z
    let byteSSA = freshSynthSSA z
    let condSSA = freshSynthSSA z

    // Block argument for condition region (receives iter arg)
    let condBlockArg = SCF.blockArg (V 0) MLIRTypes.i64  // %arg0 in condition block

    let condOps = [
        MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, ptrSSA, [(condBlockArg.SSA, MLIRTypes.i64)], MLIRTypes.i8))
        MLIROp.LLVMOp (LLVMOp.Load (byteSSA, gepSSA, MLIRTypes.i8, NotAtomic))
        MLIROp.ArithOp (ArithOp.CmpI (condSSA, ICmpPred.Ne, byteSSA, nullByteSSA, MLIRTypes.i8))
        MLIROp.SCFOp (SCFOp.Condition (condSSA, [condBlockArg.SSA]))
    ]

    // Body region: increment counter
    let nextSSA = freshSynthSSA z
    let bodyBlockArg = SCF.blockArg (V 0) MLIRTypes.i64  // %arg0 in body block

    let bodyOps = [
        MLIROp.ArithOp (ArithOp.AddI (nextSSA, bodyBlockArg.SSA, oneSSA, MLIRTypes.i64))
        MLIROp.SCFOp (SCFOp.Yield [nextSSA])
    ]

    // Build regions
    let condRegion = SCF.singleBlockRegion "before" [condBlockArg] condOps
    let bodyRegion = SCF.singleBlockRegion "do" [bodyBlockArg] bodyOps

    // Result SSA
    let lenSSA = freshSynthSSA z

    let whileOp = MLIROp.SCFOp (SCFOp.While ([lenSSA], condRegion, bodyRegion, [iterArg]))

    (constOps @ [whileOp], lenSSA)

// ═══════════════════════════════════════════════════════════════════════════
// C STRING TO FAT POINTER CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a C string (char*) to F# fat pointer {ptr, len}
/// Used at entry point boundary for argv conversion
let cStringToFatPointer (z: PSGZipper) (cStrPtr: SSA) : MLIROp list * Val =
    // Get string length via inline strlen
    let strlenOps, lenSSA = inlineStrlen z cStrPtr

    // Build fat pointer struct {ptr, len}
    let undefSSA = freshSynthSSA z
    let withPtrSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z

    let fatStrType = MLIRTypes.nativeStr

    let buildOps = [
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, fatStrType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, cStrPtr, [0], fatStrType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, lenSSA, [1], fatStrType))
    ]

    (strlenOps @ buildOps, { SSA = resultSSA; Type = fatStrType })

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENTIAL EXPRESSION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a sequential expression (children already processed in post-order)
/// Returns the result of the last expression.
/// NOTE: Sequential does NOT produce its own SSA - it passes through the last child's result.
/// We use recallNodeResult (from NodeBindings) because that's the actual emitted SSA.
let witnessSequential (z: PSGZipper) (nodeIds: NodeId list) : MLIROp list * TransferResult =
    match List.tryLast nodeIds with
    | Some lastId ->
        // Use recallNodeResult to get the ACTUAL emitted SSA (from NodeBindings),
        // not lookupNodeSSA which gets the pre-assigned SSA.
        match recallNodeResult (NodeId.value lastId) z with
        | Some (ssa, ty) ->
            // Pass through the last child's result
            [], TRValue { SSA = ssa; Type = ty }
        | None ->
            // No result bound - might be void or not yet processed
            [], TRVoid
    | None ->
        [], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// IF-THEN-ELSE
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an if-then-else expression using SCF dialect
/// thenOps and elseOps are the pre-witnessed operations from the regions
let witnessIfThenElse
    (z: PSGZipper)
    (condSSA: SSA)
    (thenOps: MLIROp list)
    (thenResultSSA: SSA option)
    (elseOps: MLIROp list option)
    (elseResultSSA: SSA option)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // Build then region with yield
    let thenYieldOps =
        match thenResultSSA with
        | Some ssa -> thenOps @ [MLIROp.SCFOp (SCFOp.Yield [ssa])]
        | None -> thenOps @ [MLIROp.SCFOp (SCFOp.Yield [])]

    let thenRegion = SCF.singleBlockRegion "then" [] thenYieldOps

    // Build else region if present
    let elseRegionOpt =
        match elseOps with
        | Some ops ->
            let elseYieldOps =
                match elseResultSSA with
                | Some ssa -> ops @ [MLIROp.SCFOp (SCFOp.Yield [ssa])]
                | None -> ops @ [MLIROp.SCFOp (SCFOp.Yield [])]
            Some (SCF.singleBlockRegion "else" [] elseYieldOps)
        | None -> None

    // Build SCF if operation
    match resultType with
    | Some ty ->
        let resultSSA = freshSynthSSA z
        let ifOp = SCFOp.If ([resultSSA], condSSA, thenRegion, elseRegionOpt, [ty])
        [MLIROp.SCFOp ifOp], TRValue { SSA = resultSSA; Type = ty }
    | None ->
        let ifOp = SCFOp.If ([], condSSA, thenRegion, elseRegionOpt, [])
        [MLIROp.SCFOp ifOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// WHILE LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a while loop using SCF dialect
/// condOps: operations for the guard condition (should produce a boolean)
/// bodyOps: operations for the loop body
/// iterArgs: iteration arguments (mutable variables carried through loop)
let witnessWhileLoop
    (z: PSGZipper)
    (condOps: MLIROp list)
    (condResultSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build condition region with scf.condition terminator
    let condBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let condTerminator = MLIROp.SCFOp (SCFOp.Condition (condResultSSA, iterArgs |> List.map (fun v -> v.SSA)))
    let condRegion = SCF.singleBlockRegion "before" condBlockArgs (condOps @ [condTerminator])

    // Build body region with scf.yield terminator
    let bodyBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let bodyTerminator = MLIROp.SCFOp (SCFOp.Yield (iterArgs |> List.map (fun v -> v.SSA)))
    let bodyRegion = SCF.singleBlockRegion "do" bodyBlockArgs (bodyOps @ [bodyTerminator])

    // Result SSAs (one per iter arg)
    let resultSSAs = iterArgs |> List.map (fun _ -> freshSynthSSA z)

    let whileOp = SCFOp.While (resultSSAs, condRegion, bodyRegion, iterArgs)

    // While loops are typically void in F# semantics
    [MLIROp.SCFOp whileOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// FOR LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a for loop using SCF dialect
/// start, stop, step: loop bounds
/// bodyOps: operations for the loop body
/// iterArgs: additional iteration arguments beyond the induction variable
let witnessForLoop
    (z: PSGZipper)
    (ivSSA: SSA)
    (startSSA: SSA)
    (stopSSA: SSA)
    (stepSSA: SSA)
    (bodyOps: MLIROp list)
    (iterArgs: Val list)
    : MLIROp list * TransferResult =

    // Build body region
    // Block args: induction variable + iter args
    let ivArg = SCF.blockArg ivSSA MLIRTypes.index
    let iterBlockArgs = iterArgs |> List.map (fun v -> SCF.blockArg v.SSA v.Type)
    let allBlockArgs = ivArg :: iterBlockArgs

    // Body with yield
    let bodyTerminator = MLIROp.SCFOp (SCFOp.Yield (iterArgs |> List.map (fun v -> v.SSA)))
    let bodyRegion = SCF.singleBlockRegion "body" allBlockArgs (bodyOps @ [bodyTerminator])

    // Result SSAs
    let resultSSAs = iterArgs |> List.map (fun _ -> freshSynthSSA z)

    let forOp = SCFOp.For (resultSSAs, ivSSA, startSSA, stopSSA, stepSSA, bodyRegion, iterArgs)

    [MLIROp.SCFOp forOp], TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN MATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a pattern match expression
/// Lowered to a chain of scf.if operations
/// Each case becomes an if-then-else checking the discriminator
let witnessMatch
    (z: PSGZipper)
    (scrutineeSSA: SSA)
    (scrutineeType: MLIRType)
    (cases: (int * MLIROp list * SSA option) list)  // (tag, ops, resultSSA)
    (resultType: MLIRType option)
    : MLIROp list * TransferResult =

    // For DU matching, extract discriminator (first field of struct)
    let discrimSSA = freshSynthSSA z
    let extractDiscrim = MLIROp.LLVMOp (LLVMOp.ExtractValue (discrimSSA, scrutineeSSA, [0], scrutineeType))

    // Build nested if-else chain
    let rec buildIfChain (cases: (int * MLIROp list * SSA option) list) : MLIROp list =
        match cases with
        | [] ->
            // Should not happen - match should be exhaustive
            []
        | [(_, caseOps, resultSSA)] ->
            // Last case - no condition check needed
            match resultSSA with
            | Some ssa -> caseOps @ [MLIROp.SCFOp (SCFOp.Yield [ssa])]
            | None -> caseOps @ [MLIROp.SCFOp (SCFOp.Yield [])]
        | (tag, caseOps, resultSSA) :: rest ->
            // Check if discriminator matches this tag
            let tagSSA = freshSynthSSA z
            let cmpSSA = freshSynthSSA z

            let checkOps = [
                MLIROp.ArithOp (ArithOp.ConstI (tagSSA, int64 tag, MLIRTypes.i32))
                MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, discrimSSA, tagSSA, MLIRTypes.i32))
            ]

            // Then branch: this case
            let thenOps =
                match resultSSA with
                | Some ssa -> caseOps @ [MLIROp.SCFOp (SCFOp.Yield [ssa])]
                | None -> caseOps @ [MLIROp.SCFOp (SCFOp.Yield [])]
            let thenRegion = SCF.singleBlockRegion "then" [] thenOps

            // Else branch: remaining cases
            let elseOps = buildIfChain rest
            let elseRegion = SCF.singleBlockRegion "else" [] elseOps

            let resultSSAs =
                match resultType with
                | Some _ -> [freshSynthSSA z]
                | None -> []
            let resultTypes =
                match resultType with
                | Some ty -> [ty]
                | None -> []

            let ifOp = SCFOp.If (resultSSAs, cmpSSA, thenRegion, Some elseRegion, resultTypes)

            checkOps @ [MLIROp.SCFOp ifOp]

    let matchOps = [extractDiscrim] @ buildIfChain cases

    match resultType with
    | Some ty ->
        let resultSSA = freshSynthSSA z
        matchOps, TRValue { SSA = resultSSA; Type = ty }
    | None ->
        matchOps, TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN BINDING EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a value from a DU payload and bind it
/// For single-field DUs, extracts field 1 (field 0 is discriminator)
let extractDUPayload
    (z: PSGZipper)
    (duSSA: SSA)
    (duType: MLIRType)
    (payloadType: MLIRType)
    : MLIROp list * Val =

    // Extract payload from field index 1 (after discriminator at index 0)
    let payloadSSA = freshSynthSSA z
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (payloadSSA, duSSA, [1], duType))

    [extractOp], { SSA = payloadSSA; Type = payloadType }

/// Extract a field from a tuple/struct
let extractTupleField
    (z: PSGZipper)
    (tupleSSA: SSA)
    (tupleType: MLIRType)
    (fieldIndex: int)
    (fieldType: MLIRType)
    : MLIROp list * Val =

    let fieldSSA = freshSynthSSA z
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, tupleSSA, [fieldIndex], tupleType))

    [extractOp], { SSA = fieldSSA; Type = fieldType }

// ═══════════════════════════════════════════════════════════════════════════
// ARGV CONVERSION (Entry Point)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert C-style argv (char**) to F# string array
/// argc: count including program name
/// argv: pointer to null-terminated C strings
/// Returns ops to build F# string[] (fat pointer array)
let convertArgv
    (z: PSGZipper)
    (argcSSA: SSA)
    (argvSSA: SSA)
    : MLIROp list * Val =

    // Compute actual arg count (argc - 1, skip program name)
    let oneSSA = freshSynthSSA z
    let countSSA = freshSynthSSA z

    let countOps = [
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i32))
        MLIROp.ArithOp (ArithOp.SubI (countSSA, argcSSA, oneSSA, MLIRTypes.i32))
    ]

    // For now, return a placeholder - full implementation would use scf.for
    // to iterate over argv and convert each C string to fat pointer
    // This is a simplified version that handles the common single-arg case

    // Get pointer to argv[1] (first real argument)
    let idxSSA = freshSynthSSA z
    let gepSSA = freshSynthSSA z
    let argPtrSSA = freshSynthSSA z

    let getArgOps = [
        MLIROp.ArithOp (ArithOp.ConstI (idxSSA, 1L, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.GEP (gepSSA, argvSSA, [(idxSSA, MLIRTypes.i64)], MLIRTypes.ptr))
        MLIROp.LLVMOp (LLVMOp.Load (argPtrSSA, gepSSA, MLIRTypes.ptr, NotAtomic))
    ]

    // Convert to fat pointer
    let convOps, fatStrVal = cStringToFatPointer z argPtrSSA

    // Build array struct { ptr, count }
    let undefSSA = freshSynthSSA z
    let withPtrSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z
    let count64SSA = freshSynthSSA z

    let arrayType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

    let buildArrayOps = [
        MLIROp.ArithOp (ArithOp.ExtSI (count64SSA, countSSA, MLIRTypes.i32, MLIRTypes.i64))
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, fatStrVal.SSA, [0], arrayType))
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withPtrSSA, count64SSA, [1], arrayType))
    ]

    (countOps @ getArgOps @ convOps @ buildArrayOps, { SSA = resultSSA; Type = arrayType })
