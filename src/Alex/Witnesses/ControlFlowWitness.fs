/// Control Flow Witness - Witness control flow constructs (SCF dialect)
///
/// ARCHITECTURAL FOUNDATION:
/// This module witnesses control flow PSG nodes using MLIR's SCF dialect.
/// Uses post-order traversal with captured regions from BeforeRegion/AfterRegion hooks.
///
/// CONTROL FLOW CONSTRUCTS:
/// - Sequential: Chain of expressions, returns last
/// - IfThenElse: scf.if with then/else regions
/// - WhileLoop: scf.while with guard/body regions and iter_args
/// - ForLoop: scf.for with induction variable
/// - Match: Pattern matching lowered to scf.if chains
module Alex.Witnesses.ControlFlowWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
// RegionKind is available via SemanticGraph
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Templates.TemplateTypes
open Alex.Templates.MemoryTemplates
module MutAnalysis = Alex.Preprocessing.MutabilityAnalysis
module PatternAnalysis = Alex.Preprocessing.PatternBindingAnalysis
module ArithTemplates = Alex.Templates.ArithTemplates
module LLVMTemplates = Alex.Templates.LLVMTemplates
module ControlFlowTemplates = Alex.Templates.ControlFlowTemplates

// ═══════════════════════════════════════════════════════════════════════════
// Type Mapping Helper (delegated to TypeMapping module)
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// Inline Strlen Implementation (for C string to F# fat pointer conversion)
// ═══════════════════════════════════════════════════════════════════════════

/// Emit inline strlen using scf.while loop
/// This is used ONLY at the entry point boundary to convert C strings to F# fat pointers
/// After this conversion, all string operations use the native F# {ptr, len} representation
///
/// Generated MLIR:
///   %zero = arith.constant 0 : i64
///   %one = arith.constant 1 : i64
///   %null = arith.constant 0 : i8
///   %len = scf.while (%i = %zero) : (i64) -> i64 {
///       %cur_ptr = llvm.getelementptr %ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
///       %byte = llvm.load %cur_ptr : !llvm.ptr -> i8
///       %cond = arith.cmpi ne, %byte, %null : i8
///       scf.condition(%cond) %i : i64
///   } do {
///       ^bb0(%i_arg: i64):
///       %next = arith.addi %i_arg, %one : i64
///       scf.yield %next : i64
///   }
let emitInlineStrlen (ptrSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Generate constants for the loop
    let zeroSSA, z1 = MLIRZipper.yieldSSA zipper
    let zeroText = sprintf "%s = arith.constant 0 : i64" zeroSSA
    let z2 = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I64) z1

    let oneSSA, z3 = MLIRZipper.yieldSSA z2
    let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
    let z4 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) z3

    let nullByteSSA, z5 = MLIRZipper.yieldSSA z4
    let nullByteText = sprintf "%s = arith.constant 0 : i8" nullByteSSA
    let z6 = MLIRZipper.witnessOpWithResult nullByteText nullByteSSA (Integer I8) z5

    // Generate SSA names for loop variables
    let lenSSA, z7 = MLIRZipper.yieldSSA z6
    let iArgSSA = sprintf "%s_arg" (lenSSA.TrimStart('%'))  // Iter arg name (without %)

    // Build the guard region ops (load byte, compare to null)
    let gepSSA, z8 = MLIRZipper.yieldSSA z7
    let gepText = sprintf "%s = llvm.getelementptr %s[%%%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" gepSSA ptrSSA iArgSSA

    let byteSSA, z9 = MLIRZipper.yieldSSA z8
    let byteText = sprintf "%s = llvm.load %s : !llvm.ptr -> i8" byteSSA gepSSA

    let condSSA, z10 = MLIRZipper.yieldSSA z9
    let condText = sprintf "%s = arith.cmpi ne, %s, %s : i8" condSSA byteSSA nullByteSSA

    // Build the body region op (increment counter)
    let nextSSA, z11 = MLIRZipper.yieldSSA z10
    let nextText = sprintf "%s = arith.addi %%%s, %s : i64" nextSSA iArgSSA oneSSA

    // Emit the full scf.while construct
    let whileText =
        sprintf "%s = scf.while (%%%s = %s) : (i64) -> i64 {\n  %s\n  %s\n  %s\n  scf.condition(%s) %%%s : i64\n} do {\n^bb0(%%%s: i64):\n  %s\n  scf.yield %s : i64\n}"
            lenSSA iArgSSA zeroSSA
            gepText byteText condText condSSA iArgSSA
            iArgSSA nextText nextSSA

    let z12 = MLIRZipper.witnessOpWithResult whileText lenSSA (Integer I64) z11

    (lenSSA, z12)

// ═══════════════════════════════════════════════════════════════════════════
// Pattern Binding Emission
// NOTE: Pattern binding ANALYSIS is in PatternAnalysis module (preprocessing)
// This section handles EMISSION of bindings to MLIR
// ═══════════════════════════════════════════════════════════════════════════

/// Emit pattern bindings for a match case
/// For array pattern [|x|], extracts array[0] and binds to x
/// Returns updated zipper with bindings in place
let rec emitPatternBindings
    (pattern: Pattern)
    (scrutineeSSA: string)
    (_scrutineeType: NativeType)
    (_graph: SemanticGraph)
    (zipper: MLIRZipper) : MLIRZipper =
    
    match pattern with
    | Pattern.Array [Pattern.Var (name, varTy)] ->
        // Single-element array pattern: [|prefix|]
        // Extract argv[1] and convert C-string to fat pointer
        //
        // argv is char** in C calling convention
        // argv[0] = program name (skip this!)
        // argv[1] = first command line argument (this is what F# wants)
        // Each argv[i] is a char* (null-terminated C string)
        // F# string is a fat pointer: {ptr, len}
        //
        // 1. Generate constant index 1 (to skip program name at argv[0])
        let idxSSA, z0 = MLIRZipper.yieldSSA zipper
        let idxText = sprintf "%s = arith.constant 1 : i64" idxSSA
        let z1 = MLIRZipper.witnessOpWithResult idxText idxSSA (Integer I64) z0

        // 2. GEP to get pointer to argv[1] (first argument)
        let gepSSA, z1a = MLIRZipper.yieldSSA z1
        let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr" gepSSA scrutineeSSA idxSSA
        let z2 = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer z1a

        // 3. Load the char* (raw C string pointer)
        let rawPtrSSA, z3 = MLIRZipper.yieldSSA z2
        let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> !llvm.ptr" rawPtrSSA gepSSA
        let z4 = MLIRZipper.witnessOpWithResult loadText rawPtrSSA Pointer z3

        // 4. Compute string length via inline strlen loop
        // Scan for null terminator using scf.while - no libc dependency
        // This is the ONLY place where we need strlen: at the OS/entry point boundary
        // After this conversion, all F# code uses the native fat pointer {ptr, len} representation
        let lenSSA, z6 = emitInlineStrlen rawPtrSSA z4

        // 5. Construct fat pointer struct {ptr, len}
        let undefSSA, z7 = MLIRZipper.yieldSSA z6
        let undefText = sprintf "%s = llvm.mlir.undef : !llvm.struct<(!llvm.ptr, i64)>" undefSSA
        let z8 = MLIRZipper.witnessOpWithResult undefText undefSSA (Struct [Pointer; Integer I64]) z7

        let struct1SSA, z9 = MLIRZipper.yieldSSA z8
        let ins1Text = sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(!llvm.ptr, i64)>" struct1SSA rawPtrSSA undefSSA
        let z10 = MLIRZipper.witnessOpWithResult ins1Text struct1SSA (Struct [Pointer; Integer I64]) z9

        let struct2SSA, z11 = MLIRZipper.yieldSSA z10
        let ins2Text = sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(!llvm.ptr, i64)>" struct2SSA lenSSA struct1SSA
        let z12 = MLIRZipper.witnessOpWithResult ins2Text struct2SSA (Struct [Pointer; Integer I64]) z11

        // 6. Bind the variable name to the fat pointer
        let elemTy = "!llvm.struct<(!llvm.ptr, i64)>"
        MLIRZipper.bindVar name struct2SSA elemTy z12
    
    | Pattern.Var (name, varTy) ->
        // Direct variable pattern - bind scrutinee directly
        let ty = Serialize.mlirType (mapType varTy)
        MLIRZipper.bindVar name scrutineeSSA ty zipper
    
    | Pattern.Wildcard ->
        // Wildcard - no bindings needed
        zipper
    
    | Pattern.Const _ ->
        // Constant pattern - no bindings needed
        zipper

    | Pattern.Union (_caseName, payloadOpt, _unionType) ->
        // DU pattern: IntVal x or FloatVal x
        // FNCS wraps single-field payloads in a Tuple: Union(caseName, Some(Tuple [Var(name, ty)]))
        // 1. Extract payload from union struct at index 1
        // 2. Convert to appropriate type based on pattern variable type
        // 3. Bind to pattern variable
        
        // Helper to bind a single variable from i64 payload
        let bindSingleVar name varTy z =
            // Extract payload i64 from struct
            let payload64SSA, z1 = MLIRZipper.yieldSSA z
            let extractParams : Quot.Aggregate.ExtractParams = { Result = payload64SSA; Aggregate = scrutineeSSA; Index = 1; AggType = "!llvm.struct<(i32, i64)>" }
            let extractText = render Quot.Aggregate.extractValue extractParams
            let z2 = MLIRZipper.witnessOpWithResult extractText payload64SSA (Integer I64) z1

            // Convert payload to target type based on pattern variable type
            let mlirTy = mapType varTy
            let payloadSSA, z3 =
                match mlirTy with
                | Integer I32 ->
                    // i64 -> i32: truncate
                    let truncSSA, z = MLIRZipper.yieldSSA z2
                    let truncParams : ConversionParams = { Result = truncSSA; Operand = payload64SSA; FromType = "i64"; ToType = "i32" }
                    let truncText = render ArithTemplates.Quot.Conversion.truncI truncParams
                    let z' = MLIRZipper.witnessOpWithResult truncText truncSSA (Integer I32) z
                    truncSSA, z'
                | Float F64 ->
                    // i64 -> f64: bitcast
                    let castSSA, z = MLIRZipper.yieldSSA z2
                    let castParams : Quot.Conversion.BitcastParams = { Result = castSSA; Operand = payload64SSA; FromType = "i64"; ToType = "f64" }
                    let castText = render Quot.Conversion.bitcast castParams
                    let z' = MLIRZipper.witnessOpWithResult castText castSSA (Float F64) z
                    castSSA, z'
                | _ ->
                    // Keep as i64
                    payload64SSA, z2

            // Bind the variable
            let ty = Serialize.mlirType mlirTy
            MLIRZipper.bindVar name payloadSSA ty z3
        
        match payloadOpt with
        | Some (Pattern.Tuple [Pattern.Var (name, varTy)]) ->
            // Single-field case wrapped in Tuple (common case from FNCS)
            bindSingleVar name varTy zipper
        | Some (Pattern.Var (name, varTy)) ->
            // Direct var (shouldn't happen but handle anyway)
            bindSingleVar name varTy zipper
        | Some (Pattern.Tuple _patterns) ->
            // Multi-field case - TODO
            zipper
        | None ->
            // No payload (like None in Option)
            zipper
        | _ ->
            zipper

    | Pattern.Tuple elements ->
        // Tuple pattern: extract each element from scrutinee tuple and bind sub-patterns
        // Scrutinee is a tuple struct: !llvm.struct<(Elem1Type, Elem2Type, ...)>
        // For a tuple of DUs, each element is !llvm.struct<(i32, i64)>

        // Get the scrutinee type to determine element types
        let elemTypes =
            match _scrutineeType with
            | NativeType.TTuple (types, _) -> types
            | _ -> List.replicate (List.length elements) _scrutineeType  // Fallback: assume uniform type

        // Build the aggregate type string for the tuple
        let elemTypesMLIR = elemTypes |> List.map (fun t -> Serialize.mlirType (mapType t))
        let aggTypeStr = sprintf "!llvm.struct<(%s)>" (String.concat ", " elemTypesMLIR)

        // Recursively emit bindings for each element
        elements
        |> List.mapi (fun idx subPattern -> (idx, subPattern))
        |> List.fold (fun (z: MLIRZipper) (idx, subPattern) ->
            // Skip wildcards - no extraction needed
            match subPattern with
            | Pattern.Wildcard -> z
            | _ ->
                // Get element type
                let elemType = if idx < List.length elemTypes then elemTypes.[idx] else _scrutineeType

                // Extract element from tuple struct
                let elemSSA, z1 = MLIRZipper.yieldSSA z
                let extractParams : Quot.Aggregate.ExtractParams = { Result = elemSSA; Aggregate = scrutineeSSA; Index = idx; AggType = aggTypeStr }
                let extractText = render Quot.Aggregate.extractValue extractParams
                let z2 = MLIRZipper.witnessOpWithResult extractText elemSSA (mapType elemType) z1

                // Recursively emit pattern bindings for this element
                emitPatternBindings subPattern elemSSA elemType _graph z2
        ) zipper

    | _ ->
        // Other patterns not yet supported
        zipper

// ═══════════════════════════════════════════════════════════════════════════
// Sequential Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a sequential expression (children already processed in post-order)
/// Returns the result of the last expression
let witnessSequential (nodeIds: NodeId list) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match List.tryLast nodeIds with
    | Some lastId ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value lastId)) zipper with
        | Some (ssa, ty) -> zipper, TRValue (ssa, ty)
        | None -> zipper, TRVoid
    | None ->
        zipper, TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// If-Then-Else
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an if-then-else expression using SCF dialect
let witnessIfThenElse 
    (guardId: NodeId) 
    (thenId: NodeId) 
    (elseIdOpt: NodeId option) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let nodeIdStr = string (NodeId.value node.Id)
    
    // Recall guard's SSA (boolean condition)
    match MLIRZipper.recallNodeSSA (string (NodeId.value guardId)) zipper with
    | Some (condSSA, _) ->
        // Check if we have captured regions from the SCF hook
        match MLIRZipper.getPendingRegions nodeIdStr zipper with
        | Some regions ->
            // Get then and optionally else region ops
            let thenOps = 
                match Map.tryFind SCFRegionKind.ThenRegion regions with
                | Some ops -> ops
                | None -> []
            let elseOps = 
                match Map.tryFind SCFRegionKind.ElseRegion regions with
                | Some ops -> Some ops
                | None -> None
            
            // Determine result type (None for void/unit)
            let resultType = 
                match node.Type with
                | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> None
                | ty -> Some (mapType ty)
            
            // Get the result SSA values from the then/else branch expressions
            let thenResultSSA =
                match MLIRZipper.recallNodeSSA (string (NodeId.value thenId)) zipper with
                | Some (ssa, _) -> Some ssa
                | None -> None
            let elseResultSSA =
                match elseIdOpt with
                | Some elseId ->
                    match MLIRZipper.recallNodeSSA (string (NodeId.value elseId)) zipper with
                    | Some (ssa, _) -> Some ssa
                    | None -> None
                | None -> None
            
            // Witness the SCF if operation with yield values
            let resultSSAOpt, zipper' = MLIRZipper.witnessSCFIf condSSA thenOps thenResultSSA elseOps elseResultSSA resultType zipper
            
            // Clear pending regions
            let zipper'' = MLIRZipper.clearPendingRegions nodeIdStr zipper'
            
            match resultSSAOpt with
            | Some resultSSA ->
                let resultTy = Serialize.mlirType (Option.get resultType)
                let zipper''' = MLIRZipper.bindNodeSSA nodeIdStr resultSSA resultTy zipper''
                zipper''', TRValue (resultSSA, resultTy)
            | None ->
                zipper'', TRVoid
        | None ->
            // No captured regions - fallback to simple behavior (shouldn't happen with SCF hook)
            match MLIRZipper.recallNodeSSA (string (NodeId.value thenId)) zipper with
            | Some (thenSSA, thenType) ->
                let zipper' = MLIRZipper.bindNodeSSA nodeIdStr thenSSA thenType zipper
                zipper', TRValue (thenSSA, thenType)
            | None ->
                zipper, TRVoid
    | None ->
        zipper, TRError "IfThenElse: guard condition not computed"

// ═══════════════════════════════════════════════════════════════════════════
// While Loop
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a while loop using SCF dialect with iter_args
let witnessWhileLoop 
    (guardId: NodeId) 
    (bodyId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let nodeIdStr = string (NodeId.value node.Id)
    
    match MLIRZipper.getPendingRegions nodeIdStr zipper with
    | Some regions ->
        // Get guard and body region ops (already using correct SSA names)
        let guardOps =
            match Map.tryFind SCFRegionKind.GuardRegion regions with
            | Some ops -> ops
            | None -> []
        let bodyOps =
            match Map.tryFind SCFRegionKind.BodyRegion regions with
            | Some ops -> ops
            | None -> []

        // Get the condition SSA (already uses iter_arg names if applicable)
        let condSSA =
            match MLIRZipper.recallNodeSSA (string (NodeId.value guardId)) zipper with
            | Some (ssa, _) -> ssa
            | None -> "%cond_missing"  // Error case

        // Get pre-analyzed iter_args: (varName, initSSA, argSSA, tyStr)
        let iterArgsInfo = MLIRZipper.getIterArgs nodeIdStr zipper |> Option.defaultValue []

        // Build iter_args with next values by looking up current VarBindings
        let currentBindings = MLIRZipper.getVarBindings zipper
        let iterArgsWithNext =
            iterArgsInfo
            |> List.map (fun (varName, initSSA, argSSA, tyStr) ->
                // Parse type from string
                let mlirTy =
                    if tyStr = "i32" then Integer I32
                    elif tyStr = "i64" then Integer I64
                    elif tyStr = "i1" then Integer I1
                    elif tyStr = "i8" then Integer I8
                    elif tyStr = "!llvm.ptr" then Pointer
                    else Integer I32  // Default fallback
                // Get current (next) SSA value from VarBindings
                let nextSSA =
                    match Map.tryFind varName currentBindings with
                    | Some (ssa, _) -> ssa
                    | None -> argSSA  // Fallback to arg if not modified
                // Use argSSA (without %) for the iter_arg name in scf.while header
                let argName = argSSA.TrimStart('%')
                (argName, initSSA, nextSSA, mlirTy))

        // Witness the SCF while operation
        let resultSSAs, zipper' = MLIRZipper.witnessSCFWhile guardOps condSSA bodyOps iterArgsWithNext zipper

        // Clear pending regions and iter_args
        let zipper'' = MLIRZipper.clearPendingRegions nodeIdStr zipper'
        let zipper''' = MLIRZipper.clearIterArgs nodeIdStr zipper''

        // Update VarBindings with final loop values (for code after the loop)
        let zipperFinal =
            resultSSAs
            |> List.zip (iterArgsInfo |> List.map (fun (name, _, _, tyStr) ->
                let mlirTy =
                    if tyStr = "i32" then Integer I32
                    elif tyStr = "i64" then Integer I64
                    elif tyStr = "i1" then Integer I1
                    elif tyStr = "i8" then Integer I8
                    elif tyStr = "!llvm.ptr" then Pointer
                    else Integer I32
                (name, mlirTy)))
            |> List.fold (fun z ((name, ty), resultSSA) ->
                MLIRZipper.bindVar name resultSSA (Serialize.mlirType ty) z) zipper'''

        // While loops typically return unit in F#
        zipperFinal, TRVoid
    | None ->
        // No captured regions - fallback (shouldn't happen with SCF hook)
        zipper, TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// For Loop
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a for loop using SCF dialect
let witnessForLoop 
    (varName: string) 
    (startId: NodeId) 
    (finishId: NodeId) 
    (isUp: bool) 
    (bodyId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let nodeIdStr = string (NodeId.value node.Id)
    
    match MLIRZipper.getPendingRegions nodeIdStr zipper with
    | Some regions ->
        // Get body region ops
        let bodyOps = 
            match Map.tryFind SCFRegionKind.BodyRegion regions with
            | Some ops -> ops
            | None -> []
        
        // Get start and end SSAs
        let startSSA, endSSA =
            match MLIRZipper.recallNodeSSA (string (NodeId.value startId)) zipper,
                  MLIRZipper.recallNodeSSA (string (NodeId.value finishId)) zipper with
            | Some (s, _), Some (e, _) -> s, e
            | _ -> "%start_missing", "%end_missing"  // Error case
        
        // Create step constant (1 or -1 based on direction)
        let stepValue = if isUp then 1L else -1L
        let stepSSA, zipper1 = MLIRZipper.witnessConstant stepValue I32 zipper
        
        // Loop variable type (typically i32 for F# int)
        let loopVarTy = Integer I32
        
        // For now, no iter_args beyond the loop variable itself
        let iterArgs: (string * string * MLIRType) list = []
        
        // Witness the SCF for operation
        let resultSSAs, zipper2 = MLIRZipper.witnessSCFFor varName loopVarTy startSSA endSSA stepSSA bodyOps iterArgs zipper1
        
        // Clear pending regions
        let zipper3 = MLIRZipper.clearPendingRegions nodeIdStr zipper2
        
        // For loops return unit in F#
        zipper3, TRVoid
    | None ->
        // No captured regions - fallback
        zipper, TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// Pattern Match
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a pattern match expression
/// Currently handles 2-case matches with wildcard fallback using scf.if
let witnessMatch 
    (scrutineeId: NodeId) 
    (cases: MatchCase list) 
    (node: SemanticNode) 
    (graph: SemanticGraph) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let nodeIdStr = string (NodeId.value node.Id)
    
    match cases with
    | [case0; case1] when (match case1.Pattern with Pattern.Wildcard -> true | _ -> false) ->
        // Two cases with wildcard fallback - use scf.if
        match MLIRZipper.recallNodeSSA (string (NodeId.value scrutineeId)) zipper with
        | Some (scrutineeSSA, _) ->
            // Generate condition based on case0 pattern
            let condSSA, zipper1 =
                match case0.Pattern with
                | Pattern.Array [_] ->
                    // Single-element array pattern on argv
                    // For entry point: check if argc == 2 (program name + 1 arg)
                    // argc is %arg0 in main function
                    // Note: scrutineeSSA is argv (%arg1), we need argc (%arg0)

                    // Generate constant 2 for comparison (program + 1 arg)
                    let twoSSA, z1 = MLIRZipper.witnessConstant 2L I32 zipper

                    // Compare: argc == 2
                    // %arg0 is argc (hardcoded for entry point pattern)
                    let cmpSSA, z2 = MLIRZipper.yieldSSA z1
                    let cmpText = sprintf "%s = arith.cmpi eq, %%arg0, %s : i32" cmpSSA twoSSA
                    let z3 = MLIRZipper.witnessOpWithResult cmpText cmpSSA (Integer I1) z2
                    (cmpSSA, z3)
                | Pattern.Tuple [_] ->
                    // Single-element tuple pattern - check if has >= 1 elements
                    let lenSSA, z1 = MLIRZipper.yieldSSA zipper
                    let lenText = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenSSA scrutineeSSA
                    let z2 = MLIRZipper.witnessOpWithResult lenText lenSSA (Integer I64) z1

                    // Generate constant 1 for comparison
                    let oneSSA, z3 = MLIRZipper.witnessConstant 1L I64 z2

                    // Compare: len >= 1
                    let cmpSSA, z4 = MLIRZipper.yieldSSA z3
                    let cmpText = sprintf "%s = arith.cmpi sge, %s, %s : i64" cmpSSA lenSSA oneSSA
                    let z5 = MLIRZipper.witnessOpWithResult cmpText cmpSSA (Integer I1) z4
                    (cmpSSA, z5)
                | _ ->
                    // Default: generate true constant (always matches first case)
                    let trueSSA, z1 = MLIRZipper.witnessConstant 1L I1 zipper
                    (trueSSA, z1)
            
            // Get captured regions for case bodies
            match MLIRZipper.getPendingRegions nodeIdStr zipper1 with
            | Some regions ->
                // Get case 0 ops (then branch)
                let case0Ops =
                    match Map.tryFind (SCFRegionKind.MatchCaseRegion 0) regions with
                    | Some ops -> ops
                    | None -> []
                
                // Get case 1 ops (else branch)
                let case1Ops =
                    match Map.tryFind (SCFRegionKind.MatchCaseRegion 1) regions with
                    | Some ops -> Some ops
                    | None -> Some []
                
                // Get result type
                let resultType =
                    match node.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> None
                    | ty -> Some (mapType ty)
                
                // Get case body result SSAs
                let case0ResultSSA =
                    match MLIRZipper.recallNodeSSA (string (NodeId.value case0.Body)) zipper1 with
                    | Some (ssa, _) -> Some ssa
                    | None -> None
                
                let case1ResultSSA =
                    match MLIRZipper.recallNodeSSA (string (NodeId.value case1.Body)) zipper1 with
                    | Some (ssa, _) -> Some ssa
                    | None -> None
                
                // Emit scf.if with the case bodies
                let resultSSAOpt, zipper2 = MLIRZipper.witnessSCFIf condSSA case0Ops case0ResultSSA case1Ops case1ResultSSA resultType zipper1
                
                // Clear pending regions
                let zipper3 = MLIRZipper.clearPendingRegions nodeIdStr zipper2
                
                match resultSSAOpt with
                | Some resultSSA ->
                    let resultTy = Serialize.mlirType (Option.get resultType)
                    let zipper4 = MLIRZipper.bindNodeSSA nodeIdStr resultSSA resultTy zipper3
                    zipper4, TRValue (resultSSA, resultTy)
                | None ->
                    zipper3, TRVoid
            | None ->
                // No captured regions - fallback
                zipper1, TRError "Match: no captured regions"
        | None ->
            zipper, TRError "Match: scrutinee not computed"

    // Two-case DU match: both cases are Union patterns
    | [case0; case1] when
        (match case0.Pattern, case1.Pattern with
         | Pattern.Union _, Pattern.Union _ -> true
         | _ -> false) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value scrutineeId)) zipper with
        | Some (scrutineeSSA, _) ->
            // Extract tag from scrutinee union struct at index 0
            let tagSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let extractTagParams : Quot.Aggregate.ExtractParams = { Result = tagSSA; Aggregate = scrutineeSSA; Index = 0; AggType = "!llvm.struct<(i32, i64)>" }
            let extractTagText = render Quot.Aggregate.extractValue extractTagParams
            let zipper2 = MLIRZipper.witnessOpWithResult extractTagText tagSSA (Integer I32) zipper1

            // Generate constant 0 for case 0's tag (first case index)
            let case0TagSSA, zipper3 = MLIRZipper.witnessConstant 0L I32 zipper2

            // Compare: tag == 0 (matches case 0)
            let condSSA, zipper4 = MLIRZipper.yieldSSA zipper3
            let cmpParams : ArithTemplates.Quot.Compare.CmpParams = { Result = condSSA; Predicate = "eq"; Lhs = tagSSA; Rhs = case0TagSSA; Type = "i32" }
            let cmpText = render ArithTemplates.Quot.Compare.cmpI cmpParams
            let zipper5 = MLIRZipper.witnessOpWithResult cmpText condSSA (Integer I1) zipper4

            // Get captured regions for case bodies
            match MLIRZipper.getPendingRegions nodeIdStr zipper5 with
            | Some regions ->
                // Get case 0 ops (then branch)
                let case0Ops =
                    match Map.tryFind (SCFRegionKind.MatchCaseRegion 0) regions with
                    | Some ops -> ops
                    | None -> []

                // Get case 1 ops (else branch)
                let case1Ops =
                    match Map.tryFind (SCFRegionKind.MatchCaseRegion 1) regions with
                    | Some ops -> Some ops
                    | None -> Some []

                // Get result type
                let resultType =
                    match node.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> None
                    | ty -> Some (mapType ty)

                // Get case body result SSAs
                let case0ResultSSA =
                    match MLIRZipper.recallNodeSSA (string (NodeId.value case0.Body)) zipper5 with
                    | Some (ssa, _) -> Some ssa
                    | None -> None

                let case1ResultSSA =
                    match MLIRZipper.recallNodeSSA (string (NodeId.value case1.Body)) zipper5 with
                    | Some (ssa, _) -> Some ssa
                    | None -> None

                // Emit scf.if with the case bodies
                let resultSSAOpt, zipper6 = MLIRZipper.witnessSCFIf condSSA case0Ops case0ResultSSA case1Ops case1ResultSSA resultType zipper5

                // Clear pending regions
                let zipper7 = MLIRZipper.clearPendingRegions nodeIdStr zipper6

                match resultSSAOpt with
                | Some resultSSA ->
                    let resultTy = Serialize.mlirType (Option.get resultType)
                    let zipper8 = MLIRZipper.bindNodeSSA nodeIdStr resultSSA resultTy zipper7
                    zipper8, TRValue (resultSSA, resultTy)
                | None ->
                    zipper7, TRVoid
            | None ->
                zipper5, TRError "Match: no captured regions for DU match"
        | None ->
            zipper, TRError "Match: scrutinee not computed for DU match"

    // Four-case tuple match: all cases are Tuple of two Union patterns
    // This handles: match a, b with | IntVal x, IntVal y -> ... | IntVal x, FloatVal y -> ... | FloatVal x, IntVal y -> ... | FloatVal x, FloatVal y -> ...
    | [case0; case1; case2; case3] when
        (List.forall (fun (c: MatchCase) ->
            match c.Pattern with
            | Pattern.Tuple [Pattern.Union _; Pattern.Union _] -> true
            | _ -> false) [case0; case1; case2; case3]) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value scrutineeId)) zipper with
        | Some (scrutineeSSA, _) ->
            // Get scrutinee type for proper element extraction
            let scrutineeType =
                match SemanticGraph.tryGetNode scrutineeId graph with
                | Some n -> n.Type
                | None -> NativeType.TApp({ Name = "unit"; Module = []; ParamKinds = []; Layout = TypeLayout.Inline(0, 1); NTUKind = None; FieldCount = 0 }, [])

            // Build tuple aggregate type string for 2 DUs: !llvm.struct<(!llvm.struct<(i32, i64)>, !llvm.struct<(i32, i64)>)>
            let elemTypes =
                match scrutineeType with
                | NativeType.TTuple (types, _) -> types
                | _ -> [scrutineeType; scrutineeType]  // Fallback

            let duStructType = "!llvm.struct<(i32, i64)>"
            let tupleAggType = sprintf "!llvm.struct<(%s, %s)>" duStructType duStructType

            // Extract both DU elements from the tuple
            let elem0SSA, zipper1 = MLIRZipper.yieldSSA zipper
            let extract0Params : Quot.Aggregate.ExtractParams = { Result = elem0SSA; Aggregate = scrutineeSSA; Index = 0; AggType = tupleAggType }
            let extract0Text = render Quot.Aggregate.extractValue extract0Params
            let zipper2 = MLIRZipper.witnessOpWithResult extract0Text elem0SSA (Struct [Integer I32; Integer I64]) zipper1

            let elem1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
            let extract1Params : Quot.Aggregate.ExtractParams = { Result = elem1SSA; Aggregate = scrutineeSSA; Index = 1; AggType = tupleAggType }
            let extract1Text = render Quot.Aggregate.extractValue extract1Params
            let zipper4 = MLIRZipper.witnessOpWithResult extract1Text elem1SSA (Struct [Integer I32; Integer I64]) zipper3

            // Extract tags from both DUs
            let tag0SSA, zipper5 = MLIRZipper.yieldSSA zipper4
            let tagExtract0Params : Quot.Aggregate.ExtractParams = { Result = tag0SSA; Aggregate = elem0SSA; Index = 0; AggType = duStructType }
            let tagExtract0Text = render Quot.Aggregate.extractValue tagExtract0Params
            let zipper6 = MLIRZipper.witnessOpWithResult tagExtract0Text tag0SSA (Integer I32) zipper5

            let tag1SSA, zipper7 = MLIRZipper.yieldSSA zipper6
            let tagExtract1Params : Quot.Aggregate.ExtractParams = { Result = tag1SSA; Aggregate = elem1SSA; Index = 0; AggType = duStructType }
            let tagExtract1Text = render Quot.Aggregate.extractValue tagExtract1Params
            let zipper8 = MLIRZipper.witnessOpWithResult tagExtract1Text tag1SSA (Integer I32) zipper7

            // Generate tag constants for comparison
            let zeroSSA, zipper9 = MLIRZipper.witnessConstant 0L I32 zipper8

            // Compare tag0 == 0 (first element is case index 0, e.g., IntVal)
            let cond0SSA, zipper10 = MLIRZipper.yieldSSA zipper9
            let cmp0Params : ArithTemplates.Quot.Compare.CmpParams = { Result = cond0SSA; Predicate = "eq"; Lhs = tag0SSA; Rhs = zeroSSA; Type = "i32" }
            let cmp0Text = render ArithTemplates.Quot.Compare.cmpI cmp0Params
            let zipper11 = MLIRZipper.witnessOpWithResult cmp0Text cond0SSA (Integer I1) zipper10

            // Compare tag1 == 0 (second element is case index 0)
            let cond1SSA, zipper12 = MLIRZipper.yieldSSA zipper11
            let cmp1Params : ArithTemplates.Quot.Compare.CmpParams = { Result = cond1SSA; Predicate = "eq"; Lhs = tag1SSA; Rhs = zeroSSA; Type = "i32" }
            let cmp1Text = render ArithTemplates.Quot.Compare.cmpI cmp1Params
            let zipper13 = MLIRZipper.witnessOpWithResult cmp1Text cond1SSA (Integer I1) zipper12

            // Get captured regions for all 4 case bodies
            match MLIRZipper.getPendingRegions nodeIdStr zipper13 with
            | Some regions ->
                let getCaseOps idx =
                    match Map.tryFind (SCFRegionKind.MatchCaseRegion idx) regions with
                    | Some ops -> ops
                    | None -> []

                let case0Ops = getCaseOps 0  // IntVal, IntVal
                let case1Ops = getCaseOps 1  // IntVal, FloatVal
                let case2Ops = getCaseOps 2  // FloatVal, IntVal
                let case3Ops = getCaseOps 3  // FloatVal, FloatVal

                // Get result type (should be Number for this sample)
                let resultType =
                    match node.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "unit" -> None
                    | ty -> Some (mapType ty)

                // Get case body result SSAs
                let getBodySSA (c: MatchCase) =
                    match MLIRZipper.recallNodeSSA (string (NodeId.value c.Body)) zipper13 with
                    | Some (ssa, _) -> Some ssa
                    | None -> None

                let case0ResultSSA = getBodySSA case0
                let case1ResultSSA = getBodySSA case1
                let case2ResultSSA = getBodySSA case2
                let case3ResultSSA = getBodySSA case3

                // Generate nested scf.if structure:
                // if tag0 == 0:
                //     if tag1 == 0: case0 (IntVal, IntVal)
                //     else: case1 (IntVal, FloatVal)
                // else:
                //     if tag1 == 0: case2 (FloatVal, IntVal)
                //     else: case3 (FloatVal, FloatVal)

                // Inner if for "tag0 == 0" branch (cases 0 and 1)
                let innerTrue01SSAOpt, zipper14 = MLIRZipper.witnessSCFIf cond1SSA case0Ops case0ResultSSA (Some case1Ops) case1ResultSSA resultType zipper13

                // Inner if for "tag0 != 0" branch (cases 2 and 3)
                let innerFalse23SSAOpt, zipper15 = MLIRZipper.witnessSCFIf cond1SSA case2Ops case2ResultSSA (Some case3Ops) case3ResultSSA resultType zipper14

                // Outer if based on tag0 == 0
                // The inner ifs produced SSA values (or not for void). We need to yield from outer if.
                match resultType, innerTrue01SSAOpt, innerFalse23SSAOpt with
                | Some resTy, Some innerTrueSSA, Some innerFalseSSA ->
                    // Result-bearing match - create outer scf.if that yields inner results
                    let outerResultSSA, zipper16 = MLIRZipper.yieldSSA zipper15
                    let outerIfText = sprintf "%s = scf.if %s -> %s {\n  scf.yield %s : %s\n} else {\n  scf.yield %s : %s\n}"
                                        outerResultSSA cond0SSA (Serialize.mlirType resTy)
                                        innerTrueSSA (Serialize.mlirType resTy)
                                        innerFalseSSA (Serialize.mlirType resTy)
                    let zipper17 = MLIRZipper.witnessOpWithResult outerIfText outerResultSSA resTy zipper16

                    // Clear pending regions
                    let zipper18 = MLIRZipper.clearPendingRegions nodeIdStr zipper17

                    let resultTy = Serialize.mlirType resTy
                    let zipper19 = MLIRZipper.bindNodeSSA nodeIdStr outerResultSSA resultTy zipper18
                    zipper19, TRValue (outerResultSSA, resultTy)
                | _ ->
                    // Void match - just emit the outer scf.if without yield
                    let outerIfText = sprintf "scf.if %s {\n} else {\n}" cond0SSA
                    let zipper16 = MLIRZipper.witnessVoidOp outerIfText zipper15
                    let zipper17 = MLIRZipper.clearPendingRegions nodeIdStr zipper16
                    zipper17, TRVoid
            | None ->
                zipper13, TRError "Match: no captured regions for 4-case tuple match"
        | None ->
            zipper, TRError "Match: scrutinee not computed for 4-case tuple match"

    | _ ->
        // Handle other match patterns (more cases, guards, etc.)
        zipper, TRError (sprintf "Match with %d cases not yet implemented" (List.length cases))


// ═══════════════════════════════════════════════════════════════════════════
// SCF Region Hook for Control Flow Tracking
// ═══════════════════════════════════════════════════════════════════════════

/// Map from fsnative's RegionKind to MLIRZipper's SCFRegionKind
let mapRegionKind (rk: RegionKind) : SCFRegionKind =
    match rk with
    | RegionKind.GuardRegion -> SCFRegionKind.GuardRegion
    | RegionKind.BodyRegion -> SCFRegionKind.BodyRegion
    | RegionKind.ThenRegion -> SCFRegionKind.ThenRegion
    | RegionKind.ElseRegion -> SCFRegionKind.ElseRegion
    | RegionKind.StartExprRegion -> SCFRegionKind.StartExprRegion
    | RegionKind.EndExprRegion -> SCFRegionKind.EndExprRegion
    | RegionKind.MatchCaseRegion idx -> SCFRegionKind.MatchCaseRegion idx

/// Create SCF Region Hook that tracks operations and sets up iter_args bindings.
/// This is a function (not a value) because it needs access to the SemanticGraph.
///
/// KEY ARCHITECTURE: Analyze then Witness (NOT Capture-and-Substitute)
/// - In BeforeRegion for GuardRegion: analyze body subtree, rebind vars to iter_args
/// - Both guard and body then use correct SSA names from the start
/// - No string substitution needed in witnessWhileLoop
let createSCFRegionHook (graph: SemanticGraph) : SCFRegionHook<MLIRZipper> = {
    BeforeRegion = fun zipper nodeId regionKind ->
        let parentIdStr = string (NodeId.value nodeId)
        let scfKind = mapRegionKind regionKind

        // For GuardRegion of a WhileLoop/ForLoop, analyze body and set up iter_args BEFORE traversal
        let zipper' =
            match regionKind with
            | RegionKind.GuardRegion ->
                // Get the parent node to check its kind and extract body NodeId
                match SemanticGraph.tryGetNode nodeId graph with
                | Some parentNode ->
                    match parentNode.Kind with
                    | SemanticKind.WhileLoop (_, bodyId) ->
                        // Look up PRE-COMPUTED modified vars from analysis
                        // (Photographer principle: observe, don't compute during transfer)
                        let modifiedVarNames = MLIRZipper.lookupModifiedVarsInLoop (NodeId.value bodyId) zipper

                        // For each modified var, set up iter_arg bindings
                        let iterArgsWithZipper =
                            modifiedVarNames
                            |> List.fold (fun (accIterArgs, accZipper) varName ->
                                // Look up current SSA binding for this var
                                match Map.tryFind varName accZipper.State.VarBindings with
                                | Some (initSSA, tyStr) ->
                                    // Generate iter_arg SSA name
                                    let argSSA = sprintf "%%%s_arg" varName
                                    // Rebind the var to iter_arg SSA (so guard/body ops use it)
                                    let reboundZipper = MLIRZipper.bindVar varName argSSA tyStr accZipper
                                    // Collect iter_arg info: (varName, initSSA, argSSA, type)
                                    ((varName, initSSA, argSSA, tyStr) :: accIterArgs, reboundZipper)
                                | None ->
                                    // Var not found in bindings - skip (might be defined inside loop)
                                    (accIterArgs, accZipper)
                            ) ([], zipper)

                        let iterArgs, zipperWithRebindings = iterArgsWithZipper

                        // Store iter_args info for later use in witnessWhileLoop
                        if not (List.isEmpty iterArgs) then
                            MLIRZipper.storeIterArgs parentIdStr (List.rev iterArgs) zipperWithRebindings
                        else
                            zipperWithRebindings

                    | SemanticKind.ForLoop (_, _, _, _, bodyId) ->
                        // Look up PRE-COMPUTED modified vars from analysis
                        let modifiedVarNames = MLIRZipper.lookupModifiedVarsInLoop (NodeId.value bodyId) zipper
                        let iterArgsWithZipper =
                            modifiedVarNames
                            |> List.fold (fun (accIterArgs, accZipper) varName ->
                                match Map.tryFind varName accZipper.State.VarBindings with
                                | Some (initSSA, tyStr) ->
                                    let argSSA = sprintf "%%%s_arg" varName
                                    let reboundZipper = MLIRZipper.bindVar varName argSSA tyStr accZipper
                                    ((varName, initSSA, argSSA, tyStr) :: accIterArgs, reboundZipper)
                                | None ->
                                    (accIterArgs, accZipper)
                            ) ([], zipper)
                        let iterArgs, zipperWithRebindings = iterArgsWithZipper
                        if not (List.isEmpty iterArgs) then
                            MLIRZipper.storeIterArgs parentIdStr (List.rev iterArgs) zipperWithRebindings
                        else
                            zipperWithRebindings
                    | _ -> zipper
                | None -> zipper
            
            | RegionKind.MatchCaseRegion idx ->
                // For MatchCaseRegion: START region tracking FIRST, THEN emit pattern bindings
                // This ensures argv extraction and strlen are INSIDE the conditional branch,
                // not executed unconditionally before the argc check.
                //
                // ARCHITECTURAL PRINCIPLE: Pattern binding extraction must be GUARDED by the
                // pattern match condition. We cannot safely access argv[1] until we've verified
                // argc >= 2. By starting region tracking first, the ops go inside scf.if branch.
                let zipperWithRegion = MLIRZipper.beginSCFRegion parentIdStr scfKind zipper

                match SemanticGraph.tryGetNode nodeId graph with
                | Some parentNode ->
                    match parentNode.Kind with
                    | SemanticKind.Match (scrutineeId, cases) ->
                        // Get the scrutinee SSA (already computed in post-order)
                        match MLIRZipper.recallNodeSSA (string (NodeId.value scrutineeId)) zipperWithRegion with
                        | Some (scrutineeSSA, _) ->
                            // Get the case at this index
                            if idx < List.length cases then
                                let case = cases.[idx]
                                // Get scrutinee type for proper element extraction
                                match SemanticGraph.tryGetNode scrutineeId graph with
                                | Some scrutineeNode ->
                                    // Emit pattern bindings (extracts elements, binds variables)
                                    // These ops are now captured INSIDE the region
                                    emitPatternBindings case.Pattern scrutineeSSA scrutineeNode.Type graph zipperWithRegion
                                | None -> zipperWithRegion
                            else zipperWithRegion
                        | None -> zipperWithRegion
                    | _ -> zipperWithRegion
                | None -> zipperWithRegion

            | _ -> zipper

        // Begin region tracking
        // MatchCaseRegion already started tracking above and returned with region active
        // Other region kinds need to start tracking here
        match regionKind with
        | RegionKind.MatchCaseRegion _ -> zipper'  // Already handled - just return
        | _ -> MLIRZipper.beginSCFRegion parentIdStr scfKind zipper'

    AfterRegion = fun zipper nodeId regionKind ->
        let parentIdStr = string (NodeId.value nodeId)
        let scfKind = mapRegionKind regionKind
        // End region tracking (captures ops)
        MLIRZipper.endSCFRegion parentIdStr scfKind zipper
}
