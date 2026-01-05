/// FNCSTransfer - Witness-based transfer from FNCS SemanticGraph to MLIR
///
/// ARCHITECTURAL FOUNDATION (Coeffects & Codata):
/// This module witnesses the FNCS SemanticGraph structure and produces MLIR
/// via the MLIRZipper codata accumulator.
///
/// KEY DESIGN PRINCIPLES:
/// 1. Post-order traversal: Children before parents (SSAs available when parent visited)
/// 2. Dispatch on SemanticKind ONLY - no pattern matching on symbol names
/// 3. Platform bindings via SemanticKind.PlatformBinding marker → PlatformDispatch
/// 4. Codata vocabulary: witness, observe, yield, bind, recall, extract
///
/// This replaces the deleted FNCSEmitter.fs antipattern.
module Alex.Traversal.FNCSTransfer

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════
// Type Mapping: NativeType → MLIRType
// ═══════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type
/// NativeType uses TApp(tycon, args) for applied types
let rec mapType (ty: NativeType) : MLIRType =
    match ty with
    // Applied types - match on type constructor name
    | NativeType.TApp(tycon, _args) ->
        match tycon.Name with
        | "unit" -> Unit
        | "bool" -> Integer I1
        | "int8" | "sbyte" -> Integer I8
        | "uint8" | "byte" -> Integer I8
        | "int16" -> Integer I16
        | "uint16" -> Integer I16
        | "int" | "int32" -> Integer I32
        | "uint" | "uint32" -> Integer I32
        | "int64" -> Integer I64
        | "uint64" -> Integer I64
        | "nativeint" -> Integer I64  // Platform dependent
        | "unativeint" -> Integer I64
        | "float32" | "single" -> Float F32
        | "float" | "double" -> Float F64
        | "char" -> Integer I32  // Unicode codepoint
        | "string" -> NativeStrType  // Fat pointer {ptr: *u8, len: i64}
        | "Ptr" | "nativeptr" -> Pointer
        | "array" -> Pointer
        | "list" -> Pointer
        | "option" | "voption" -> Pointer  // TODO: Value type layout
        | _ -> Pointer  // Default to pointer for unknown applied types

    // Function types
    | NativeType.TFun _ -> Pointer  // Function pointer + closure

    // Tuple types
    | NativeType.TTuple _ -> Pointer  // TODO: Proper struct layout

    // Type variables (erased)
    | NativeType.TVar _ -> Pointer

    // Byref types
    | NativeType.TByref _ -> Pointer

    // Native pointers
    | NativeType.TNativePtr _ -> Pointer

    // Forall types - look at body
    | NativeType.TForall(_, body) -> mapType body

    // Other types
    | NativeType.TRecord _ -> Pointer
    | NativeType.TUnion _ -> Pointer
    | NativeType.TAnon _ -> Pointer
    | NativeType.TMeasure _ -> Unit  // Phantom type, no runtime representation
    | NativeType.TError msg -> failwithf "Type error in semantic graph: %s" msg

// ═══════════════════════════════════════════════════════════════════
// Witness Functions for SemanticKind
// ═══════════════════════════════════════════════════════════════════

/// Witness a literal value
let witnessLiteral (lit: LiteralValue) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match lit with
    | LiteralValue.Unit ->
        // Unit needs a concrete representation for SSA references
        // Generate a zero constant (LLVM doesn't have true unit type)
        let ssaName, zipper' = MLIRZipper.witnessConstant 0L I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Bool b ->
        let value = if b then 1L else 0L
        let ssaName, zipper' = MLIRZipper.witnessConstant value I1 zipper
        zipper', TRValue (ssaName, "i1")

    | LiteralValue.Int32 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int64 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant n I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.String s ->
        // NATIVE STRING: Fat pointer struct {ptr: *u8, len: i64}
        // See Serena memories: string_length_handling, fidelity_memory_model
        //
        // Step 1: Observe string literal (creates global constant)
        let globalName, zipper1 = MLIRZipper.observeStringLiteral s zipper
        
        // Step 2: Get address of character data
        let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
        
        // Step 3: Create length constant (string length without null terminator)
        let len = s.Length
        let lenSSA, zipper3 = MLIRZipper.witnessConstant (int64 len) I64 zipper2
        
        // Step 4: Build fat pointer struct using llvm.mlir.undef + insertvalue
        // This follows the Fidelity memory model: F# types dictate MLIR layout
        let undefSSA, zipper4 = MLIRZipper.yieldSSA zipper3
        let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
        let zipper5 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper4
        
        // Step 5: Insert pointer at index 0
        let withPtrSSA, zipper6 = MLIRZipper.yieldSSA zipper5
        let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
        let zipper7 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper6
        
        // Step 6: Insert length at index 1
        let fatPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA withPtrSSA NativeStrTypeStr
        let zipper9 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper8
        
        zipper9, TRValue (fatPtrSSA, NativeStrTypeStr)

    | LiteralValue.Float32 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f32" ssaName (float f)
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F32) zipper'
        zipper'', TRValue (ssaName, "f32")

    | LiteralValue.Float64 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f64" ssaName f
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F64) zipper'
        zipper'', TRValue (ssaName, "f64")

    | LiteralValue.Char c ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 c) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int8 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.UInt8 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.Int16 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt16 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt32 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.UInt64 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I64 zipper
        zipper', TRValue (ssaName, "i64")

    | _ ->
        // Unsupported literal type
        zipper, TRError (sprintf "Unsupported literal: %A" lit)

/// Check if a name is a built-in operator or function
let isBuiltInOperator (name: string) =
    name.StartsWith("op_") ||
    name.StartsWith("NativePtr.") ||
    name.StartsWith("Sys.") ||
    name.StartsWith("Unchecked.") ||
    List.contains name ["not"; "int"; "int8"; "int16"; "int32"; "int64";
                        "byte"; "uint8"; "uint16"; "uint32"; "uint64";
                        "float"; "float32"; "double"; "single"; "decimal";
                        "nativeint"; "unativeint"; "char"; "string";
                        "box"; "unbox"; "Some"; "None"; "ValueSome"; "ValueNone";
                        "printf"; "printfn"; "sprintf"; "failwith"; "failwithf";
                        "Array.zeroCreate"; "Array.length"; "Array.get"; "Array.set";
                        "ignore"; "raise"; "reraise"; "typeof"; "sizeof"; "nameof"]


/// Emit a primitive binary operator as arith operation
/// Returns Some (resultSSA, resultType, zipper') if the operator is recognized
/// Returns None if this is not a primitive binary operator
let tryEmitPrimitiveBinaryOp (opName: string) (arg1SSA: string) (arg1Type: string) (arg2SSA: string) (arg2Type: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    // Only handle integer operands for now
    let isIntType t = t = "i8" || t = "i16" || t = "i32" || t = "i64"
    let isFloatType t = t = "f32" || t = "f64"
    
    // Both arguments must be same type for arith ops
    if arg1Type <> arg2Type then None
    elif not (isIntType arg1Type || isFloatType arg1Type) then None
    else
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        
        // Map operator name to arith operation
        let arithOp, resultType =
            match opName, isIntType arg1Type with
            // Integer arithmetic
            | "op_Addition", true -> Some (sprintf "%s = arith.addi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Subtraction", true -> Some (sprintf "%s = arith.subi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Multiply", true -> Some (sprintf "%s = arith.muli %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Division", true -> Some (sprintf "%s = arith.divsi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Modulus", true -> Some (sprintf "%s = arith.remsi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            // Integer comparison (result is i1 = bool)
            | "op_LessThan", true -> Some (sprintf "%s = arith.cmpi slt, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_LessThanOrEqual", true -> Some (sprintf "%s = arith.cmpi sle, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_GreaterThan", true -> Some (sprintf "%s = arith.cmpi sgt, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_GreaterThanOrEqual", true -> Some (sprintf "%s = arith.cmpi sge, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_Equality", true -> Some (sprintf "%s = arith.cmpi eq, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_Inequality", true -> Some (sprintf "%s = arith.cmpi ne, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            // Bitwise operations
            | "op_BitwiseAnd", true -> Some (sprintf "%s = arith.andi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_BitwiseOr", true -> Some (sprintf "%s = arith.ori %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_ExclusiveOr", true -> Some (sprintf "%s = arith.xori %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_LeftShift", true -> Some (sprintf "%s = arith.shli %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_RightShift", true -> Some (sprintf "%s = arith.shrsi %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            // Float arithmetic  
            | "op_Addition", false -> Some (sprintf "%s = arith.addf %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Subtraction", false -> Some (sprintf "%s = arith.subf %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Multiply", false -> Some (sprintf "%s = arith.mulf %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            | "op_Division", false -> Some (sprintf "%s = arith.divf %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), arg1Type
            // Float comparison
            | "op_LessThan", false -> Some (sprintf "%s = arith.cmpf olt, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_LessThanOrEqual", false -> Some (sprintf "%s = arith.cmpf ole, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_GreaterThan", false -> Some (sprintf "%s = arith.cmpf ogt, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_GreaterThanOrEqual", false -> Some (sprintf "%s = arith.cmpf oge, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_Equality", false -> Some (sprintf "%s = arith.cmpf oeq, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | "op_Inequality", false -> Some (sprintf "%s = arith.cmpf one, %s, %s : %s" resultSSA arg1SSA arg2SSA arg1Type), "i1"
            | _ -> None, ""
        
        match arithOp with
        | Some op ->
            let zipper'' = MLIRZipper.witnessOpWithResult op resultSSA (Serialize.deserializeType resultType) zipper'
            Some (resultSSA, resultType, zipper'')
        | None -> None

/// Emit a primitive unary operator as arith operation
let tryEmitPrimitiveUnaryOp (opName: string) (argSSA: string) (argType: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    let isIntType t = t = "i8" || t = "i16" || t = "i32" || t = "i64"
    let isBoolType t = t = "i1"
    
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    
    match opName, isIntType argType, isBoolType argType with
    // Boolean not
    | "not", _, true -> 
        // XOR with true (1) to flip the bit
        let trueSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let trueOp = sprintf "%s = arith.constant true" trueSSA
        let zipper''' = MLIRZipper.witnessOpWithResult trueOp trueSSA (Integer I1) zipper''
        let notOp = sprintf "%s = arith.xori %s, %s : i1" resultSSA argSSA trueSSA
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Integer I1) zipper'''
        Some (resultSSA, "i1", zipper4)
    // Integer negation
    | "op_UnaryNegation", true, _ ->
        // Subtract from 0
        let zeroSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let zeroOp = sprintf "%s = arith.constant 0 : %s" zeroSSA argType
        let zipper''' = MLIRZipper.witnessOpWithResult zeroOp zeroSSA (Serialize.deserializeType argType) zipper''
        let negOp = sprintf "%s = arith.subi %s, %s : %s" resultSSA zeroSSA argSSA argType
        let zipper4 = MLIRZipper.witnessOpWithResult negOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
    // Bitwise not
    | "op_OnesComplement", true, _ ->
        // XOR with all 1s (-1 in two's complement)
        let onesSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let onesOp = sprintf "%s = arith.constant -1 : %s" onesSSA argType
        let zipper''' = MLIRZipper.witnessOpWithResult onesOp onesSSA (Serialize.deserializeType argType) zipper''
        let notOp = sprintf "%s = arith.xori %s, %s : %s" resultSSA argSSA onesSSA argType
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
    | _ -> None

/// Witness a variable reference (recall its SSA from prior observation)
/// INVARIANT: FNCS guarantees definitions are traversed before uses
/// If this fails, it's an FNCS graph construction bug - hard stop with diagnostic
let witnessVarRef (name: string) (defId: NodeId option) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Check if this references an addressed mutable - if so, we need to load from alloca
    let isAddressedMut =
        match defId with
        | Some nodeId -> MLIRZipper.isAddressedMutable (NodeId.value nodeId) zipper
        | None -> false

    // First try to look up by variable name (for Lambda parameters)
    match MLIRZipper.recallVar name zipper with
    | Some (ssaName, mlirType) ->
        if isAddressedMut then
            // Addressed mutable: ssaName is the alloca pointer, we need to load the value
            // Look up the element type from the mutable alloca record
            match defId with
            | Some nodeId ->
                match MLIRZipper.lookupMutableAlloca (NodeId.value nodeId) zipper with
                | Some (_, elementType) ->
                    // Load the value from the alloca
                    let loadedSSA, zipper' = MLIRZipper.witnessLoadStr ssaName elementType zipper
                    zipper', TRValue (loadedSSA, elementType)
                | None ->
                    // Fallback: shouldn't happen if analysis is correct
                    zipper, TRValue (ssaName, mlirType)
            | None ->
                zipper, TRValue (ssaName, mlirType)
        else
            // Regular var: return as-is
            zipper, TRValue (ssaName, mlirType)
    | None ->
        // Not a bound variable - try definition node
        match defId with
        | Some nodeId ->
            // Recall the SSA value bound to this definition
            match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
            | Some (ssaName, mlirType) ->
                zipper, TRValue (ssaName, mlirType)
            | None ->
                // Definition node wasn't traversed - check if it's a module-level binding
                // Module-level bindings may be siblings of entry points, not children
                match SemanticGraph.tryGetNode nodeId graph with
                | None ->
                    failwithf "FNCS GRAPH ERROR: Variable '%s' references node %d which is NOT in graph. Tree-shaking must include all referenced definitions." name (NodeId.value nodeId)
                | Some defNode ->
                    // If it's a Binding, try to find its value (first child) SSA
                    match defNode.Kind, defNode.Children with
                    | SemanticKind.Binding (_, _, _, _), valueNodeId :: _ ->
                        // Check if the binding's value (Lambda) was traversed
                        match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
                        | Some (lambdaSSA, lambdaType) ->
                            // Found the value's SSA - use it
                            zipper, TRValue (lambdaSSA, lambdaType)
                        | None ->
                            // Value wasn't traversed - try looking for its function name marker
                            match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId) + "_lambdaName") zipper with
                            | Some (lambdaName, _) ->
                                // Found the lambda name marker - construct function reference
                                zipper, TRValue ("@" + lambdaName, "!llvm.ptr")
                            | None ->
                                // DEFERRED RESOLUTION: The binding's value wasn't traversed yet
                                // Check if the value is a Literal (constant) or Lambda (function)
                                match SemanticGraph.tryGetNode valueNodeId graph with
                                | Some valueNode ->
                                    match valueNode.Kind with
                                    | SemanticKind.Literal lit ->
                                        // It's a constant - emit the literal value directly
                                        witnessLiteral lit zipper
                                    | SemanticKind.Lambda _ ->
                                        // It's a function - use its actual type from the graph
                                        let funcName = name
                                        let signature =
                                            match valueNode.Type with
                                            | NativeType.TFun(paramTy, retTy) ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType paramTy))
                                                    (Serialize.mlirType (mapType retTy))
                                            | _ ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType valueNode.Type))
                                                    "!llvm.ptr"
                                        let retType =
                                            match valueNode.Type with
                                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                            | _ -> "!llvm.ptr"
                                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                        zipper', TRValue ("@" + funcName, retType)
                                    | _ ->
                                        // Unknown value type - use its actual type
                                        let funcName = name
                                        let signature =
                                            match valueNode.Type with
                                            | NativeType.TFun(paramTy, retTy) ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType paramTy))
                                                    (Serialize.mlirType (mapType retTy))
                                            | _ -> sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType valueNode.Type))
                                                    "!llvm.ptr"
                                        let retType =
                                            match valueNode.Type with
                                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                            | _ -> "!llvm.ptr"
                                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                        zipper', TRValue ("@" + funcName, retType)
                                | None ->
                                    // Value node doesn't exist - use binding's type
                                    let funcName = name
                                    let signature =
                                        match defNode.Type with
                                        | NativeType.TFun(paramTy, retTy) ->
                                            sprintf "(%s) -> %s"
                                                (Serialize.mlirType (mapType paramTy))
                                                (Serialize.mlirType (mapType retTy))
                                        | _ -> sprintf "(%s) -> %s"
                                                (Serialize.mlirType (mapType defNode.Type))
                                                "!llvm.ptr"
                                    let retType =
                                        match defNode.Type with
                                        | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                        | _ -> "!llvm.ptr"
                                    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                    zipper', TRValue ("@" + funcName, retType)
                    | _ ->
                        // Non-binding reference - use its actual type
                        let funcName = name
                        let signature =
                            match defNode.Type with
                            | NativeType.TFun(paramTy, retTy) ->
                                sprintf "(%s) -> %s"
                                    (Serialize.mlirType (mapType paramTy))
                                    (Serialize.mlirType (mapType retTy))
                            | _ -> sprintf "(%s) -> %s"
                                    (Serialize.mlirType (mapType defNode.Type))
                                    "!llvm.ptr"
                        let retType =
                            match defNode.Type with
                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                            | _ -> "!llvm.ptr"
                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                        zipper', TRValue ("@" + funcName, retType)
        | None ->
            // No definition node - check if it's a built-in
            if isBuiltInOperator name then
                zipper, TRBuiltin name
            else
                zipper, TRError (sprintf "Variable '%s' has no definition node" name)

/// Witness a platform binding call
let witnessPlatformBinding (entryPoint: string) (argSSAs: (string * MLIRType) list) (returnType: NativeType) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Create PlatformPrimitive for dispatch
    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = argSSAs
        ReturnType = mapType returnType
        BindingStrategy = Static
    }

    // Dispatch to platform-specific implementation
    let zipper', result = PlatformDispatch.dispatch prim zipper

    match result with
    | WitnessedValue (ssa, ty) ->
        zipper', TRValue (ssa, Serialize.mlirType ty)
    | WitnessedVoid ->
        zipper', TRVoid
    | NotSupported reason ->
        zipper', TRError (sprintf "Platform binding '%s' not supported: %s" entryPoint reason)

/// Witness a function application
let witnessApplication (funcNodeId: NodeId) (argNodeIds: NodeId list) (returnType: NativeType) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Get the function node to understand what kind of call this is
    match SemanticGraph.tryGetNode funcNodeId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.PlatformBinding entryPoint ->
            // This is a platform binding call - get argument SSAs
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    match SemanticGraph.tryGetNode nodeId graph with
                    | Some argNode ->
                        match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
                        | Some (ssa, _) -> Some (ssa, mapType argNode.Type)
                        | None -> None
                    | None -> None)
            
            // Platform bindings have TVar types from FNCS (types aren't constrained).
            // Use known parameter counts for platform bindings.
            let expectedParamCount =
                match entryPoint with
                | "writeBytes" | "readBytes" -> 3   // fd, buffer, count
                | "getCurrentTicks" -> 0
                | "sleep" -> 1
                | _ -> 0  // Unknown - assume fully applied
            // Debug: %s has %d/%d args - use intermediates for inspection
            
            if List.length argSSAs < expectedParamCount then
                // Partial application - return a marker for later
                // Format: $platform:entryPoint:arg0:type0:arg1:type1:...
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
                // All arguments present - call the platform binding
                witnessPlatformBinding entryPoint argSSAs returnType zipper

        | SemanticKind.Intrinsic intrinsicName ->
            // Handle compiler intrinsics like NativePtr.toNativeInt
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)

            match intrinsicName, argSSAs with
            | "NativePtr.toNativeInt", [(argSSA, _)] ->
                // nativeptr<'T> -> nativeint (pointer to integer conversion)
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                // Use ptrtoint to convert pointer to integer
                let text = sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to i64" ssaName argSSA
                let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Integer I64) zipper'
                zipper'', TRValue (ssaName, "i64")
            | "NativePtr.ofNativeInt", [(argSSA, _)] ->
                // nativeint -> nativeptr<'T> (integer to pointer conversion)
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                let text = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" ssaName argSSA
                let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
                zipper'', TRValue (ssaName, "!llvm.ptr")
            | "NativePtr.toVoidPtr", [(argSSA, _)] ->
                // nativeptr<'T> -> voidptr (just pass through, same type at MLIR level)
                zipper, TRValue (argSSA, "!llvm.ptr")
            | "NativePtr.ofVoidPtr", [(argSSA, _)] ->
                // voidptr -> nativeptr<'T> (just pass through, same type at MLIR level)
                zipper, TRValue (argSSA, "!llvm.ptr")
            | "NativePtr.get", [(ptrSSA, _); (idxSSA, _)] ->
                // nativeptr<'T> -> int -> 'T (indexed load)
                let elemType = Serialize.mlirType (mapType returnType)
                let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
                let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA gepSSA elemType
                let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType returnType) zipper'''
                zipper4, TRValue (loadSSA, elemType)
            | "NativePtr.set", [(ptrSSA, _); (idxSSA, _); (valSSA, _)] ->
                // nativeptr<'T> -> int -> 'T -> unit (indexed store)
                let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" valSSA gepSSA
                let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
                zipper''', TRVoid
            | "NativePtr.stackalloc", [(countSSA, _)] ->
                // int -> nativeptr<'T> (stack allocation)
                // Allocate 'count' elements of the element type on the stack
                // The return type tells us the element type: nativeptr<'T> -> T
                let elemType =
                    match returnType with
                    | NativeType.TNativePtr elemTy -> Serialize.mlirType (mapType elemTy)
                    | _ -> "i8"  // Default to byte if type unknown
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                // Convert count from i32 to i64 for llvm.alloca
                let countSSA64, zipper'' = MLIRZipper.yieldSSA zipper'
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                let zipper''' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper''
                // Emit llvm.alloca
                let allocaText = sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ssaName countSSA64 elemType
                let zipper4 = MLIRZipper.witnessOpWithResult allocaText ssaName Pointer zipper'''
                zipper4, TRValue (ssaName, "!llvm.ptr")
            | "NativePtr.copy", [(destSSA, _); (srcSSA, _); (countSSA, _)] ->
                // dest:nativeptr<'T> -> src:nativeptr<'T> -> count:int -> unit
                // Maps to llvm.memcpy intrinsic
                // Convert count from i32 to i64 for llvm.intr.memcpy
                let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                // Emit llvm.intr.memcpy: dest, src, len, isVolatile
                let memcpyText = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" destSSA srcSSA countSSA64
                let zipper''' = MLIRZipper.witnessVoidOp memcpyText zipper''
                zipper''', TRVoid
            | "NativePtr.fill", [(destSSA, _); (valueSSA, _); (countSSA, _)] ->
                // dest:nativeptr<'T> -> value:'T -> count:int -> unit
                // Maps to llvm.memset intrinsic (when 'T = byte, fills count bytes)
                // Convert count from i32 to i64 for llvm.intr.memset
                let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                // Emit llvm.intr.memset: dest, value, len, isVolatile
                let memsetText = sprintf "\"llvm.intr.memset\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()" destSSA valueSSA countSSA64
                let zipper''' = MLIRZipper.witnessVoidOp memsetText zipper''
                zipper''', TRVoid
            | "NativePtr.add", [(ptrSSA, _); (offsetSSA, _)] ->
                // nativeptr<'T> -> int -> nativeptr<'T> (pointer arithmetic)
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" resultSSA ptrSSA offsetSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText resultSSA Pointer zipper'
                zipper'', TRValue (resultSSA, "!llvm.ptr")

            // ═══════════════════════════════════════════════════════════════════════════
            // Sys intrinsics - Dispatch to platform bindings layer
            // Platform-specific MLIR (syscall numbers, ABIs) is DATA in Bindings
            // ═══════════════════════════════════════════════════════════════════════════
            | intrinsicName, argSSAs when intrinsicName.StartsWith("Sys.") ->
                // Extract the entry point (e.g., "Sys.write" -> "Sys.write")
                // Map argument SSAs to (ssa, type) pairs for PlatformPrimitive
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                // Dispatch to platform bindings - all platform-specific logic is there
                witnessPlatformBinding intrinsicName argSSAsWithTypes returnType zipper

            // ═══════════════════════════════════════════════════════════════════════════
            // NativeStr intrinsics - Native string (fat pointer) construction
            // ═══════════════════════════════════════════════════════════════════════════
            | "NativeStr.fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                // Construct fat pointer struct from pointer and length
                // Same pattern as string literal emission: undef + insertvalue
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                // Insert pointer at index 0
                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                // Insert length at index 1 (need to extend to i64 if it's i32)
                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" lenSSA64 lenSSA
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA64 withPtrSSA NativeStrTypeStr
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            // ═══════════════════════════════════════════════════════════════════════════
            // NativeDefault intrinsics - Zero-initialized values
            // ═══════════════════════════════════════════════════════════════════════════
            | "NativeDefault.zeroed", [] ->
                // Zero value for the return type
                // The return type tells us what kind of zero to emit
                let zeroSSA, zipper' = MLIRZipper.yieldSSA zipper
                let mlirRetType = mapType returnType
                let mlirTypeStr = Serialize.mlirType mlirRetType
                let zeroText =
                    match mlirRetType with
                    | Integer _ -> sprintf "%s = arith.constant 0 : %s" zeroSSA mlirTypeStr
                    | Float F32 -> sprintf "%s = arith.constant 0.0 : f32" zeroSSA
                    | Float F64 -> sprintf "%s = arith.constant 0.0 : f64" zeroSSA
                    | Pointer -> sprintf "%s = llvm.mlir.zero : !llvm.ptr" zeroSSA
                    | Struct _ when mlirTypeStr = NativeStrTypeStr ->
                        // Zero string: undef (ptr=null, len=0 effectively)
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA NativeStrTypeStr
                    | Struct _ ->
                        // For other structs, use undef (zero-initialized)
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                    | _ ->
                        // Default: use undef
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            // ═══════════════════════════════════════════════════════════════════════════
            // String intrinsics - Native string operations (post-Alloy absorption)
            // ═══════════════════════════════════════════════════════════════════════════
            | "String.concat2", [(str1SSA, _); (str2SSA, _)] ->
                // string -> string -> string (concatenate two fat pointer strings)
                // For now, allocate a new buffer and copy both strings
                // TODO: This is a simplified implementation - real impl needs heap allocation

                // Extract ptr and len from first string
                let ptr1SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr1 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr1SSA str1SSA
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer zipper1

                let len1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen1 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len1SSA str1SSA
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) zipper3

                // Extract ptr and len from second string
                let ptr2SSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let extractPtr2 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr2SSA str2SSA
                let zipper6 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer zipper5

                let len2SSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let extractLen2 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len2SSA str2SSA
                let zipper8 = MLIRZipper.witnessOpWithResult extractLen2 len2SSA (Integer I64) zipper7

                // Calculate total length
                let totalLenSSA, zipper9 = MLIRZipper.yieldSSA zipper8
                let addLenText = sprintf "%s = arith.addi %s, %s : i64" totalLenSSA len1SSA len2SSA
                let zipper10 = MLIRZipper.witnessOpWithResult addLenText totalLenSSA (Integer I64) zipper9

                // Allocate buffer on stack for result
                let bufSSA, zipper11 = MLIRZipper.yieldSSA zipper10
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufSSA totalLenSSA
                let zipper12 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper11

                // Copy first string to buffer
                let memcpy1 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" bufSSA ptr1SSA len1SSA
                let zipper13 = MLIRZipper.witnessVoidOp memcpy1 zipper12

                // Calculate offset for second string (buf + len1)
                let offsetSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" offsetSSA bufSSA len1SSA
                let zipper15 = MLIRZipper.witnessOpWithResult gepText offsetSSA Pointer zipper14

                // Copy second string
                let memcpy2 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" offsetSSA ptr2SSA len2SSA
                let zipper16 = MLIRZipper.witnessVoidOp memcpy2 zipper15

                // Construct result fat pointer
                let undefSSA, zipper17 = MLIRZipper.yieldSSA zipper16
                let undefText = sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" undefSSA
                let zipper18 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper17

                let withPtrSSA, zipper19 = MLIRZipper.yieldSSA zipper18
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" withPtrSSA bufSSA undefSSA
                let zipper20 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper19

                let resultSSA, zipper21 = MLIRZipper.yieldSSA zipper20
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" resultSSA totalLenSSA withPtrSSA
                let zipper22 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType zipper21

                zipper22, TRValue (resultSSA, NativeStrTypeStr)

            // ═══════════════════════════════════════════════════════════════════════════
            // Console intrinsics - Dispatch to platform bindings layer
            // These are FNCS intrinsics (post-Alloy absorption) that wrap Sys.* operations
            // Platform-specific MLIR is generated via PlatformDispatch
            // ═══════════════════════════════════════════════════════════════════════════
            | intrinsicName, argSSAs when intrinsicName.StartsWith("Console.") ->
                // Console intrinsics dispatch to platform bindings just like Sys.*
                // Map argument SSAs to (ssa, type) pairs for PlatformPrimitive
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                // Dispatch to platform bindings - Console.write/writeln/readln/error/errorln
                witnessPlatformBinding intrinsicName argSSAsWithTypes returnType zipper

            | intrinsicName, [] ->
                // TypeApp or partial application - return marker for subsequent application
                // This handles polymorphic intrinsics (e.g., stackalloc<byte>) before value args are applied
                zipper, TRValue ("$intrinsic:" + intrinsicName, "func")
            | _ ->
                // Unknown intrinsic with args - report error
                zipper, TRError (sprintf "Unknown intrinsic: %s with %d args" intrinsicName (List.length argSSAs))

        | SemanticKind.VarRef (name, defId) ->
            // Function call to a named function - follow the definition to find the Lambda's SSA
            // Get both SSA values AND their actual types (not PSG declared types!)
            let argSSAsAndTypes =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
            let argSSAs = argSSAsAndTypes |> List.map fst
            // Use actual SSA types for the call signature
            let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

            // Special handling for built-in operators like pipe (|>)
            // op_PipeRight(x) returns a partial application marker; op_PipeRight(x)(f) becomes f(x)
            if name = "op_PipeRight" || name = "op_PipeLeft" then
                match argSSAs with
                | [argSSA] ->
                    // Partial application of pipe: |> x
                    // Return a marker that includes the argument so we can inline when called
                    let (argType: string) = argSSAsAndTypes |> List.head |> snd
                    let marker = sprintf "$pipe:%s:%s" argSSA argType
                    zipper, TRValue (marker, "func")
                | _ ->
                    zipper, TRError (sprintf "Pipe operator '%s' expects 1 argument, got %d" name (List.length argSSAs))
            else

            // Try to recall the Lambda's SSA from its definition node
            // This is critical: VarRef points to a Binding, whose child is the Lambda
            // The Lambda's SSA is bound to its node ID (e.g., "@lambda_6")
            match defId with
            | Some defNodeId ->
                // Try to find the Lambda's SSA via the definition binding
                match MLIRZipper.recallNodeSSA (string (NodeId.value defNodeId)) zipper with
                | Some (funcSSA, _funcType) ->
                    // Got the Lambda's SSA (like "@lambda_6") - use it for the call
                    // Check if this is a function reference (starts with @)
                    if funcSSA.StartsWith("@") then
                        // Direct function call
                        let funcName = funcSSA.Substring(1) // Remove leading @
                        
                        // Check for partial application: fewer arguments than function parameters
                        let expectedParams = MLIRZipper.lookupFuncParamCount funcName zipper
                        match expectedParams with
                        | Some paramCount when paramCount > List.length argSSAs ->
                            // Partial application - return a curried function marker
                            // Format: $partial:funcName:arg0:type0:arg1:type1:...
                            let argPairs = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            // Full application - call the function
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName argSSAs argTypes (mapType returnType) zipper
                            // Use tracked return type if available (for correct lowered types)
                            // This is critical: PSG may have TVar which maps to !llvm.ptr, but the
                            // actual function may return i32 - we need the ACTUAL type for correct SSA binding
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        // Function value in SSA register - indirect call
                        let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | None ->
                    // Definition wasn't traversed - fall back to name (may be undefined)
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                // No definition ID - check if this is a primitive operator
                // Primitive operators like +, -, *, / on integers/floats compile to arith ops
                let argSSAsWithTypes = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                
                // Try binary operator first (most common case)
                match argSSAsWithTypes with
                | [(arg1SSA, arg1Type); (arg2SSA, arg2Type)] ->
                    match tryEmitPrimitiveBinaryOp name arg1SSA arg1Type arg2SSA arg2Type zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        // Not a recognized primitive binary op - fall back to external call
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | [(argSSA, argType)] ->
                    // Try unary operator
                    match tryEmitPrimitiveUnaryOp name argSSA argType zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        // Not a recognized primitive unary op - fall back to external call
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | _ ->
                    // Not a binary or unary operator pattern - fall back to external call
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))

        | SemanticKind.Lambda _ ->
            // Lambda application - for now, treat as error (needs closure support)
            zipper, TRError "Lambda application not yet supported"

        | SemanticKind.Application (innerFuncId, innerArgIds) ->
            // Curried application: the function is itself a partial application
            // The inner application was already processed (post-order), so recall its result
            match MLIRZipper.recallNodeSSA (string (NodeId.value funcNodeId)) zipper with
            | Some (funcSSA, _funcType) ->
                // Get argument SSAs AND their actual types (not PSG declared types!)
                let argSSAsAndTypes =
                    argNodeIds
                    |> List.choose (fun nodeId ->
                        MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
                let argSSAs = argSSAsAndTypes |> List.map fst
                // Use actual SSA types for the call signature
                let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

                // Check if this is a pipe operator application: $pipe:argSSA:argType
                // If so, inline it: (|> x)(f) becomes f(x)
                if funcSSA.StartsWith("$pipe:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 3 then
                        let pipedArgSSA = parts.[1]
                        let pipedArgType = parts.[2]
                        // The function f is the first argument to this application
                        match argSSAs with
                        | [fSSA] ->
                            // Check return type - if it's unit (i32) and piped arg is also unit,
                            // this is likely `unit |> ignore` pattern - just return unit
                            let retTypeStr = Serialize.mlirType (mapType returnType)
                            if retTypeStr = "i32" && pipedArgType = "i32" then
                                // Unit piped to unit-returning function (like ignore)
                                // Just return unit directly to avoid type mismatch
                                let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                // Apply f to the piped argument: f(x)
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                // Check if f is a direct function reference (starts with @)
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1) // Remove leading @
                                    // Special handling for ignore - just discard the argument and return unit
                                    if funcName = "ignore" then
                                        let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                        let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
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
                // Handle partial application: $partial:funcName:arg0:type0:arg1:type1:...
                elif funcSSA.StartsWith("$partial:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let funcName = parts.[1]
                        // Collect already-applied arguments from the marker
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, ty)
                                | _ -> None)
                            |> Array.toList
                        // Combine with new arguments
                        let allArgSSAs = (appliedArgs |> List.map fst) @ argSSAs
                        let allArgTypes = (appliedArgs |> List.map (snd >> Serialize.deserializeType)) @ argTypes
                        
                        // Check if we now have enough arguments
                        match MLIRZipper.lookupFuncParamCount funcName zipper with
                        | Some paramCount when paramCount > List.length allArgSSAs ->
                            // Still partial - return updated marker
                            let argPairs = List.zip allArgSSAs (allArgTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            // Full application - call the function
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName allArgSSAs allArgTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        zipper, TRError (sprintf "Invalid partial marker: %s" funcSSA)
                // Handle partial platform binding: $platform:entryPoint:arg0:type0:arg1:type1:...
                elif funcSSA.StartsWith("$platform:") then
                    ()
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let entryPoint = parts.[1]
                        // Collect already-applied arguments from the marker
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, Serialize.deserializeType ty)
                                | _ -> None)
                            |> Array.toList
                        // Combine with new arguments
                        let allArgSSAs = appliedArgs @ (List.zip argSSAs argTypes)
                        
                        // Check expected param count from platform bindings (3 for writeBytes, etc.)
                        // For now, use simple heuristic: writeBytes/readBytes have 3 params
                        let expectedParamCount = 
                            match entryPoint with
                            | "writeBytes" | "readBytes" -> 3
                            | "getCurrentTicks" -> 0
                            | "sleep" -> 1
                            | _ -> List.length allArgSSAs  // Assume we have all
                        
                        if List.length allArgSSAs < expectedParamCount then
                            // Still partial - return updated marker
                            let argsEncoded = allArgSSAs |> List.collect (fun (a, t) -> [a; Serialize.mlirType t]) |> String.concat ":"
                            let marker = sprintf "$platform:%s:%s" entryPoint argsEncoded
                            zipper, TRValue (marker, "func")
                        else
                            // All arguments present - call the platform binding
                            witnessPlatformBinding entryPoint allArgSSAs returnType zipper
                    else
                        zipper, TRError (sprintf "Invalid platform marker: %s" funcSSA)
                else

                let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                // Inner application wasn't computed - this shouldn't happen in post-order
                zipper, TRError (sprintf "Curried function application not computed: %A" (innerFuncId, innerArgIds))

        | _ ->
            zipper, TRError (sprintf "Unexpected function node kind: %A" funcNode.Kind)

    | None ->
        zipper, TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))

/// Witness a sequential expression (children already processed in post-order)
let witnessSequential (nodeIds: NodeId list) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // In post-order, children are already witnessed
    // Return the result of the last expression
    match List.tryLast nodeIds with
    | Some lastId ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value lastId)) zipper with
        | Some (ssa, ty) -> zipper, TRValue (ssa, ty)
        | None -> zipper, TRVoid
    | None ->
        zipper, TRVoid

/// Witness a binding (let x = value)
let witnessBinding (name: string) (isMutable: bool) (valueNodeId: NodeId) (node: SemanticNode) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // The value has already been witnessed (post-order)
    match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
    | Some (ssa, ty) ->
        let nodeIdVal = NodeId.value node.Id
        
        // Check if this is an addressed mutable (needs alloca)
        if isMutable && MLIRZipper.isAddressedMutable nodeIdVal zipper then
            // Addressed mutable: use alloca + store instead of pure SSA
            // 1. Emit alloca for the element type (ty is already a string)
            let allocaSSA, zipper1 = MLIRZipper.witnessAllocaStr ty zipper
            // 2. Store initial value into alloca
            let zipper2 = MLIRZipper.witnessStore ssa ty allocaSSA zipper1
            // 3. Record alloca for this binding (store element type for later loads)
            let zipper3 = MLIRZipper.recordMutableAlloca nodeIdVal allocaSSA ty zipper2
            // 4. Bind the *pointer* to the node (so AddressOf can find it)
            let zipper4 = MLIRZipper.bindNodeSSA (string nodeIdVal) allocaSSA "!llvm.ptr" zipper3
            // 5. Bind var name to the alloca pointer for VarRef to find
            let zipper5 = MLIRZipper.bindVar name allocaSSA "!llvm.ptr" zipper4
            zipper5, TRValue (allocaSSA, "!llvm.ptr")
        else
            // Immutable or non-addressed mutable: use pure SSA
            // For non-addressed mutable vars, Set operations will rebind to new SSA values
            // This enables SCF iter_args for loops (no alloca/load/store)
            let zipper' = MLIRZipper.bindNodeSSA (string nodeIdVal) ssa ty zipper
            let zipper'' = MLIRZipper.bindVar name ssa ty zipper'
            zipper'', TRValue (ssa, ty)
    | None ->
        // HARD STOP: Binding's value expression didn't produce an SSA
        // This means either:
        // 1. The value is genuinely unit (OK)
        // 2. The value expression isn't implemented (BUG - should fail at source)
        // Check the node's type to distinguish
        match node.Type with
        | NativeType.TApp (tycon, _) when tycon.Name = "unit" ->
            // Unit binding - no SSA needed, but still bind the node for consistency
            zipper, TRVoid
        | _ ->
            // Non-unit value didn't produce SSA - this is an implementation gap
            failwithf "CODEGEN ERROR: Binding '%s' (node %d) has type %A but value expression produced no SSA. The value expression is not yet implemented for native compilation." name (NodeId.value node.Id) node.Type

// ═══════════════════════════════════════════════════════════════════
// Main Transfer Fold
// ═══════════════════════════════════════════════════════════════════

/// Witness a single node based on its SemanticKind
/// Dispatch is ONLY on SemanticKind - no symbol name pattern matching
let witnessNode (graph: SemanticGraph) (node: SemanticNode) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match node.Kind with
    // Literals
    | SemanticKind.Literal lit ->
        let zipper', result = witnessLiteral lit zipper
        // Bind result to this node for future reference
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Variable references
    | SemanticKind.VarRef (name, defId) ->
        let zipper', result = witnessVarRef name defId graph zipper
        // CRITICAL: Bind result to this node for future reference
        // This is needed when VarRef is used as value in a Binding or TypeAnnotation
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | TRBuiltin opName ->
            // Built-in operator - bind a marker so TypeAnnotation can forward it
            let marker = "$builtin:" + opName
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) marker "func" zipper'
            zipper'', result
        | _ -> zipper', result

    // Platform bindings (the ONLY place platform calls are recognized)
    | SemanticKind.PlatformBinding entryPoint ->
        // Platform binding node itself doesn't produce a value
        // The Application that uses it will call witnessPlatformBinding
        zipper, TRVoid

    // Compiler intrinsics (e.g., NativePtr.toNativeInt)
    | SemanticKind.Intrinsic intrinsicName ->
        // Intrinsic node itself produces a function value
        // The Application will handle generating actual MLIR code
        // Bind a marker so Application can recognize and handle it
        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("$intrinsic:" + intrinsicName) "func" zipper
        zipper', TRValue ("$intrinsic:" + intrinsicName, "func")

    // Function applications
    | SemanticKind.Application (funcId, argIds) ->
        // DEBUG: Trace Application processing  
        ()
        let zipper', result = witnessApplication funcId argIds node.Type graph zipper
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Sequential expressions
    | SemanticKind.Sequential nodeIds ->
        // DEBUG: Trace Sequential processing
        let nodeIdStrs = nodeIds |> List.map (fun nid -> string (NodeId.value nid)) |> String.concat ", "
        ()
        let zipper', result = witnessSequential nodeIds zipper
        // Bind result to this node for TypeAnnotation to recall
        // But only if we have a non-empty SSA value
        match result with
        | TRValue (ssa, ty) when ssa <> "" ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Bindings
    | SemanticKind.Binding (name, isMutable, _isRecursive, _isEntryPoint) ->
        // Get the value node (first child)
        match node.Children with
        | valueId :: _ ->
            witnessBinding name isMutable valueId node zipper
        | [] ->
            zipper, TRError "Binding has no value child"

    // Module definitions (container - children already processed)
    | SemanticKind.ModuleDef (name, _members) ->
        zipper, TRVoid

    // Type definitions (don't generate runtime code)
    | SemanticKind.TypeDef (_name, _kind, _members) ->
        zipper, TRVoid

    // Record expressions - construct a record value
    | SemanticKind.RecordExpr (fields, _copyFrom) ->
        // For records, we need to allocate space and set fields
        // For now, treat struct records as tuples
        let fieldSSAs =
            fields
            |> List.choose (fun (_fieldName, valueId) ->
                MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper)

        if List.length fieldSSAs <> List.length fields then
            // Some fields weren't computed
            zipper, TRError "Record fields not all computed"
        else
            // For a single-field record, just return that field's value
            match fieldSSAs with
            | [(ssa, ty)] ->
                let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
                zipper1, TRValue (ssa, ty)
            | _ ->
                // Multi-field record - for now, return unit and handle properly later
                zipper, TRVoid

    // Lambdas - non-capturing lambdas become named functions
    // Body operations were captured inside function scope by preBindLambdaParams
    | SemanticKind.Lambda (params', bodyId) ->
        // Check if we're in function scope (preBindLambdaParams should have entered it)
        match zipper.Focus with
        | InFunction funcName ->
            // Get the lambda name that was stored during pre-bind
            let lambdaName = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value node.Id) + "_lambdaName") zipper with
                | Some (name, _) -> name
                | None -> funcName  // Use the current function name as fallback
            
            // Determine the declared return type from the Lambda's F# type
            let declaredRetType = 
                match node.Type with
                | NativeType.TFun (_, retTy) -> Serialize.mlirType (mapType retTy)
                | _ -> "i32"
            
            // Look up the body's SSA result (already processed in post-order, inside function scope)
            // Thread the zipper through in case we need to generate a default value
            let bodySSA, bodyType, zipperWithBody = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper with
                | Some (ssa, ty) when ssa <> "" -> ssa, ty, zipper
                | _ -> 
                    // Body didn't produce a value (or produced empty SSA) - generate appropriate default
                    let zeroSSA, z = MLIRZipper.yieldSSA zipper
                    if declaredRetType = "!llvm.ptr" then
                        // For pointer returns, use llvm.mlir.zero for null pointer
                        let zeroText = sprintf "%s = llvm.mlir.zero : !llvm.ptr" zeroSSA
                        let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA Pointer z
                        zeroSSA, "!llvm.ptr", z'
                    else
                        // For integer returns, use arith.constant 0
                        let zeroText = sprintf "%s = arith.constant 0 : %s" zeroSSA declaredRetType
                        let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
                        zeroSSA, declaredRetType, z'
            
            // Check for return type mismatch
            // Case 1: declared i32 (unit) but body produces ptr - return 0:i32
            // Case 2: declared ptr but body produces i32 - return null ptr
            let returnSSA, returnType, zipperForReturn =
                if declaredRetType = "i32" && bodyType = "!llvm.ptr" then
                    // Unit function with side-effecting body - ignore result, return 0
                    let zeroSSA, z = MLIRZipper.yieldSSA zipperWithBody
                    let zeroText = sprintf "%s = arith.constant 0 : i32" zeroSSA
                    let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
                    zeroSSA, "i32", z'
                elif declaredRetType = "!llvm.ptr" && (bodyType = "i32" || bodyType.StartsWith("i")) then
                    // Function returns ptr but body computed an integer - return null ptr
                    let nullSSA, z = MLIRZipper.yieldSSA zipperWithBody
                    let nullText = sprintf "%s = llvm.mlir.zero : !llvm.ptr" nullSSA
                    let z' = MLIRZipper.witnessOpWithResult nullText nullSSA Pointer z
                    nullSSA, "!llvm.ptr", z'
                else
                    bodySSA, bodyType, zipperWithBody
            
            // Add return instruction to end the function body
            let returnText = sprintf "llvm.return %s : %s" returnSSA returnType
            let zipper1 = MLIRZipper.witnessVoidOp returnText zipperForReturn
            
            // Exit function scope - this creates the MLIRFunc with all accumulated body ops
            let zipper2, func = MLIRZipper.exitFunction zipper1
            
            // Add the function to completed functions
            let zipper3 = MLIRZipper.addCompletedFunction func zipper2
            
            // After exitFunction, we're at module level - we can't emit addressof here
            // (MLIR doesn't allow operations at module level)
            // Instead, bind the function name as a marker for later use
            // The addressof will be emitted when the lambda is actually called
            let zipper4 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper3
            zipper4, TRValue ("@" + lambdaName, "!llvm.ptr")
        
        | _ ->
            // Not in function scope - preBindLambdaParams may not have run for this Lambda
            // Fall back to creating function directly (old behavior)
            let lambdaName, zipper1 = MLIRZipper.yieldLambdaName zipper
            
            let mlirParams = params' |> List.mapi (fun i (_name, nativeTy) ->
                (sprintf "arg%d" i, mapType nativeTy))
            
            let returnType = 
                match node.Type with
                | NativeType.TFun (_, retTy) -> mapType retTy
                | _ -> mapType node.Type
            
            // Look up body SSA
            let bodySSA, bodyType = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper1 with
                | Some (ssa, ty) -> ssa, ty
                | None -> 
                    let undefSSA, z = MLIRZipper.yieldSSA zipper1
                    undefSSA, Serialize.mlirType returnType
            
            // Create return operation
            let retOp: MLIROp = {
                Text = sprintf "llvm.return %s : %s" bodySSA bodyType
                Results = []
            }
            
            // Create function with just the return op (body ops weren't captured)
            let func: MLIRFunc = {
                Name = lambdaName
                Parameters = mlirParams
                ReturnType = returnType
                Blocks = [{
                    Label = "entry"
                    Arguments = []
                    Operations = [retOp]
                }]
                Attributes = []
                IsInternal = true
            }
            
            let zipper2 = MLIRZipper.addCompletedFunction func zipper1
            
            // At module level - can't emit addressof here
            // Bind the function name as a marker for later use
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper2
            zipper3, TRValue ("@" + lambdaName, "!llvm.ptr")

    // Type annotations - pass through the inner expression's value
    | SemanticKind.TypeAnnotation (exprId, _annotatedType) ->
        // DEBUG: Trace TypeAnnotation processing
        ()
        // Type annotation doesn't generate code - just forward the inner expression's value
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
        | Some (ssa, ty) ->
            let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper1, TRValue (ssa, ty)
        | None ->
            // Inner expression not yet computed - shouldn't happen in post-order
            zipper, TRError (sprintf "TypeAnnotation inner expr %A not computed" exprId)

    // Mutable set
    | SemanticKind.Set (targetId, valueId) ->
        // Pure SSA rebinding for mutable variables (no store)
        // This enables SCF iter_args for loops
        match SemanticGraph.tryGetNode targetId graph with
        | Some targetNode ->
            match targetNode.Kind with
            | SemanticKind.VarRef (name, _) ->
                // Mutable variable assignment - rebind to new SSA value
                match MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
                | Some (valueSSA, valueType) ->
                    // Rebind the variable to the new SSA value
                    let zipper' = MLIRZipper.bindVar name valueSSA valueType zipper
                    zipper', TRVoid
                | None ->
                    zipper, TRError (sprintf "Set: value for '%s' not computed" name)
            | _ ->
                // Non-variable target (field set, array set, etc.) - use store
                match MLIRZipper.recallNodeSSA (string (NodeId.value targetId)) zipper,
                      MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
                | Some (targetSSA, _), Some (valueSSA, valueType) ->
                    let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA targetSSA valueType
                    let zipper' = MLIRZipper.witnessVoidOp storeText zipper
                    zipper', TRVoid
                | _ ->
                    zipper, TRError "Set: target or value not computed"
        | None ->
            zipper, TRError "Set: target node not found"

    // Control flow - uses SCF dialect with PendingRegions from region hooks
    | SemanticKind.IfThenElse (guardId, thenId, elseIdOpt) ->
        // Recall guard's SSA (boolean condition)
        let nodeIdStr = string (NodeId.value node.Id)
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
                // These are the values to yield from each branch
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

    | SemanticKind.WhileLoop (guardId, bodyId) ->
        // While loops use SCF with PendingRegions from hooks
        // ARCHITECTURE: Analyze then Witness - iter_args were set up in BeforeRegion
        // Guard and body ops already use correct iter_arg SSA names (no substitution needed)
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
            // These were computed and stored in BeforeRegion for GuardRegion
            let iterArgsInfo = MLIRZipper.getIterArgs nodeIdStr zipper |> Option.defaultValue []

            // Build iter_args with next values by looking up current VarBindings
            // After body traversal, each var's current binding is its "next" value for scf.yield
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
                        | None -> argSSA  // Fallback to arg if not modified (shouldn't happen)
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

    | SemanticKind.ForLoop (varName, startId, finishId, isUp, bodyId) ->
        // For loops use SCF with PendingRegions from hooks
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

    | SemanticKind.Match (scrutineeId, cases) ->
        // TODO: Implement pattern matching
        zipper, TRError "Match not yet implemented"

    // Interpolated strings
    | SemanticKind.InterpolatedString parts ->
        // TODO: Implement string interpolation lowering
        zipper, TRError "InterpolatedString not yet implemented"

    // Array/collection indexing
    | SemanticKind.IndexGet (collectionId, indexId) ->
        // Get the collection and index SSAs
        match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper with
        | Some (collSSA, collType), Some (indexSSA, _) ->
            // Generate GEP (getelementptr) for array access
            let ssaName, zipper' = MLIRZipper.yieldSSA zipper
            let elemType = Serialize.mlirType (mapType node.Type)
            let text = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" ssaName collSSA indexSSA
            let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
            // Load the element
            let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
            let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA ssaName elemType
            let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType node.Type) zipper'''
            let zipper5 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) loadSSA elemType zipper4
            zipper5, TRValue (loadSSA, elemType)
        | _ ->
            zipper, TRError "IndexGet: collection or index not computed"

    | SemanticKind.IndexSet (collectionId, indexId, valueId) ->
        // Get the collection, index, and value SSAs
        match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
        | Some (collSSA, _), Some (indexSSA, _), Some (valueSSA, valueType) ->
            // Generate GEP for array access
            let ptrSSA, zipper' = MLIRZipper.yieldSSA zipper
            let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" ptrSSA collSSA indexSSA
            let zipper'' = MLIRZipper.witnessOpWithResult gepText ptrSSA Pointer zipper'
            // Store the value
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA ptrSSA valueType
            let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
            zipper''', TRVoid
        | _ ->
            zipper, TRError "IndexSet: collection, index, or value not computed"

    // Address-of operator
    | SemanticKind.AddressOf (exprId, isMutable) ->
        // Check if the expression is a VarRef to an addressed mutable binding
        // If so, we return the alloca pointer directly (VarRef already emitted a load)
        match SemanticGraph.tryGetNode exprId graph with
        | Some exprNode ->
            match exprNode.Kind with
            | SemanticKind.VarRef (_, Some targetBindingId) ->
                let bindingIdVal = NodeId.value targetBindingId
                if MLIRZipper.isAddressedMutable bindingIdVal zipper then
                    // Addressed mutable: get the alloca pointer directly
                    match MLIRZipper.lookupMutableAlloca bindingIdVal zipper with
                    | Some (allocaSSA, _) ->
                        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) allocaSSA "!llvm.ptr" zipper
                        zipper', TRValue (allocaSSA, "!llvm.ptr")
                    | None ->
                        zipper, TRError "AddressOf: addressed mutable has no alloca"
                else
                    // Non-addressed VarRef - use the VarRef's SSA (may be an existing pointer)
                    match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
                    | Some (exprSSA, _) ->
                        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) exprSSA "!llvm.ptr" zipper
                        zipper', TRValue (exprSSA, "!llvm.ptr")
                    | None ->
                        zipper, TRError "AddressOf: VarRef expression not computed"
            | _ ->
                // Not a VarRef - use the expression's SSA (may be an existing pointer)
                match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
                | Some (exprSSA, _) ->
                    let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) exprSSA "!llvm.ptr" zipper
                    zipper', TRValue (exprSSA, "!llvm.ptr")
                | None ->
                    zipper, TRError "AddressOf: expression not computed"
        | None ->
            zipper, TRError "AddressOf: expression node not found in graph"

    // Tuple expressions - construct a tuple value
    | SemanticKind.TupleExpr elementIds ->
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            // Some elements weren't computed
            zipper, TRError "TupleExpr: not all elements computed"
        else
            match elementSSAs with
            | [] ->
                // Empty tuple is unit
                zipper, TRVoid
            | [(ssa, ty)] ->
                // Single element - just return it
                let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
                zipper', TRValue (ssa, ty)
            | elements ->
                // Multi-element tuple - for now, allocate struct and store elements
                // TODO: Proper tuple lowering based on ABI
                let tupleSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" tupleSSA (List.length elements)
                let zipper2 = MLIRZipper.witnessOp allocaText [(tupleSSA, Pointer)] zipper1
                // Store each element at its offset (simplified: just use pointer for now)
                let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) tupleSSA "!llvm.ptr" zipper2
                zipper3, TRValue (tupleSSA, "!llvm.ptr")

    // TraitCall - SRTP member resolution
    | SemanticKind.TraitCall (memberName, typeArgs, argId) ->
        // For SRTP, the trait call resolves at compile time to a specific member
        // We need to look up the resolved member and call it
        match MLIRZipper.recallNodeSSA (string (NodeId.value argId)) zipper with
        | Some (argSSA, argType) ->
            // For now, emit a call to the trait member name
            // TODO: Proper SRTP resolution from Baker/type checker
            let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
            let callText = sprintf "%s = llvm.call @%s(%s) : (%s) -> !llvm.ptr" resultSSA memberName argSSA argType
            let zipper'' = MLIRZipper.witnessOpWithResult callText resultSSA Pointer zipper'
            let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
            zipper''', TRValue (resultSSA, "!llvm.ptr")
        | None ->
            zipper, TRError (sprintf "TraitCall '%s': argument not computed" memberName)

    // Array expressions - construct an array
    | SemanticKind.ArrayExpr elementIds ->
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            zipper, TRError "ArrayExpr: not all elements computed"
        else
            match elementSSAs with
            | [] ->
                // Empty array - allocate empty array (just a pointer)
                let arrSSA, zipper' = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x 0 : (i64) -> !llvm.ptr" arrSSA
                let zipper'' = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper''
                zipper''', TRValue (arrSSA, "!llvm.ptr")
            | elements ->
                // Non-empty array - allocate and store elements
                let arrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" arrSSA (List.length elements)
                let zipper2 = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper1
                // Store each element (simplified - not considering element types)
                let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper2
                zipper3, TRValue (arrSSA, "!llvm.ptr")

    // List expressions - construct a list
    | SemanticKind.ListExpr elementIds ->
        // Lists are similar to arrays in our runtime - just linked list of pointers
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            zipper, TRError "ListExpr: not all elements computed"
        else
            // Allocate space for list nodes (simplified)
            let listSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" listSSA (List.length elementSSAs)
            let zipper2 = MLIRZipper.witnessOpWithResult allocaText listSSA Pointer zipper1
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) listSSA "!llvm.ptr" zipper2
            zipper3, TRValue (listSSA, "!llvm.ptr")

    // Field access: expr.fieldName
    | SemanticKind.FieldGet (exprId, fieldName) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
        | Some (exprSSA, exprType) ->
            // STRING INTRINSIC MEMBERS: Native string is fat pointer {ptr, len}
            // .Pointer → extractvalue at index 0 → !llvm.ptr
            // .Length → extractvalue at index 1 → i64
            // See Serena memories: string_length_handling, fidelity_memory_model
            if isNativeStrType exprType then
                match fieldName with
                | "Pointer" ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let extractText = sprintf "%s = llvm.extractvalue %s[0] : %s" resultSSA exprSSA NativeStrTypeStr
                    let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA Pointer zipper'
                    let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                    zipper''', TRValue (resultSSA, "!llvm.ptr")
                | "Length" ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let extractText = sprintf "%s = llvm.extractvalue %s[1] : %s" resultSSA exprSSA NativeStrTypeStr
                    let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA (Integer I64) zipper'
                    let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "i64" zipper''
                    zipper''', TRValue (resultSSA, "i64")
                | _ ->
                    zipper, TRError (sprintf "Unknown string field: %s (expected Pointer or Length)" fieldName)
            else
                // Generic field access for other struct types
                // TODO: Proper field offset calculation from record type
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> !llvm.ptr" resultSSA exprSSA
                let zipper'' = MLIRZipper.witnessOpWithResult loadText resultSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                zipper''', TRValue (resultSSA, "!llvm.ptr")
        | None ->
            zipper, TRError (sprintf "FieldGet '%s': expression not computed" fieldName)

    // Field set: expr.fieldName <- value
    | SemanticKind.FieldSet (exprId, fieldName, valueId) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
        | Some (exprSSA, _), Some (valueSSA, valueType) ->
            // Store value at field offset
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA exprSSA valueType
            let zipper' = MLIRZipper.witnessVoidOp storeText zipper
            zipper', TRVoid
        | _ ->
            zipper, TRError (sprintf "FieldSet '%s': expression or value not computed" fieldName)

    // Error nodes
    | SemanticKind.Error msg ->
        zipper, TRError msg

    // Catch-all for unimplemented kinds
    | kind ->
        zipper, TRError (sprintf "SemanticKind not yet implemented: %A" kind)

// ═══════════════════════════════════════════════════════════════════
// PSG Analysis Functions (for SCF iter_args detection)
// ═══════════════════════════════════════════════════════════════════

/// Analyze a subtree to find all Set nodes and extract their target variable names.
/// This is used BEFORE traversing a loop body to know which mutable vars need iter_args.
/// Uses the SemanticGraph structure directly (NOT the main fold) - this is pure analysis.
let findModifiedVarsInSubtree (graph: SemanticGraph) (rootNodeId: NodeId) : string list =
    let rec walk (nodeId: NodeId) (acc: Set<string>) : Set<string> =
        match SemanticGraph.tryGetNode nodeId graph with
        | None -> acc
        | Some node ->
            let acc =
                match node.Kind with
                | SemanticKind.Set (targetId, _) ->
                    // Get target node to extract variable name
                    match SemanticGraph.tryGetNode targetId graph with
                    | Some targetNode ->
                        match targetNode.Kind with
                        | SemanticKind.VarRef (name, _) -> Set.add name acc
                        | _ -> acc
                    | None -> acc
                | _ -> acc
            // Walk children
            node.Children |> List.fold (fun a c -> walk c a) acc
    walk rootNodeId Set.empty |> Set.toList

/// Find all mutable bindings whose address is taken (&& operator)
/// These need alloca instead of pure SSA
/// Returns a set of NodeIds of the mutable Binding nodes
let findAddressedMutableBindings (graph: SemanticGraph) : Set<int> =
    let mutableBindingIds = System.Collections.Generic.HashSet<int>()
    
    // Find all AddressOf nodes and check if they reference VarRef to mutable bindings
    for KeyValue(nodeId, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.AddressOf (exprId, _) ->
            // Check if exprId is a VarRef to a mutable binding
            match SemanticGraph.tryGetNode exprId graph with
            | Some exprNode ->
                match exprNode.Kind with
                | SemanticKind.VarRef (_, Some targetBindingId) ->
                    // Check if target is a mutable binding
                    match SemanticGraph.tryGetNode targetBindingId graph with
                    | Some bindingNode ->
                        match bindingNode.Kind with
                        | SemanticKind.Binding (_, isMutable, _, _) when isMutable ->
                            mutableBindingIds.Add(NodeId.value targetBindingId) |> ignore
                        | _ -> ()
                    | None -> ()
                | _ -> ()
            | None -> ()
        | _ -> ()
    
    mutableBindingIds |> Set.ofSeq

/// Pre-bind Lambda parameters to SSA names BEFORE body is processed
/// This is called in pre-order for Lambda nodes only
/// CRITICAL: Also enters function scope so body operations are captured
/// NOTE: For entry point (main), uses C-style signature: (argc: i32, argv: ptr) -> i32
let preBindLambdaParams (zipper: MLIRZipper) (node: SemanticNode) : MLIRZipper =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId) ->
        // Generate lambda function name - entry point Lambdas become "main"
        let nodeIdVal = NodeId.value node.Id
        let lambdaName, zipper1 = MLIRZipper.yieldLambdaNameForNode nodeIdVal zipper

        // For main, use C-style signature regardless of F# signature
        // C main: (int argc, char** argv) -> int
        // F# entry: (argv: string[]) -> int  OR  (unit) -> int
        let isMain = (lambdaName = "main")

        let mlirParams, paramBindings =
            if isMain then
                // C-style main signature
                let cParams = [("arg0", Integer I32); ("arg1", Pointer)]
                // F# parameter binds to %arg1 (argv), argc (%arg0) typically unused
                let bindings =
                    match params' with
                    | [(paramName, paramType)] ->
                        // Single F# parameter maps to argv (%arg1)
                        [(paramName, "%arg1", Serialize.mlirType (mapType paramType))]
                    | [] ->
                        // Unit parameter - nothing to bind
                        []
                    | _ ->
                        // Multiple parameters (unusual for entry point)
                        params' |> List.mapi (fun i (name, ty) ->
                            (name, sprintf "%%arg%d" (i + 1), Serialize.mlirType (mapType ty)))
                cParams, bindings
            else
                // Regular lambda - use node.Type for parameter types (has instantiated generics)
                // Extract parameter types from curried TFun chain
                let rec extractParamTypesFromFun (ty: NativeType) (count: int) : NativeType list =
                    if count <= 0 then []
                    else
                        match ty with
                        | NativeType.TFun(paramTy, resultTy) ->
                            paramTy :: extractParamTypesFromFun resultTy (count - 1)
                        | _ -> []

                let nodeParamTypes = extractParamTypesFromFun node.Type (List.length params')

                // Use instantiated types from node.Type if available, else fall back to params'
                let mlirPs = params' |> List.mapi (fun i (_name, nativeTy) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else nativeTy
                    (sprintf "arg%d" i, mapType actualType))
                let bindings = params' |> List.mapi (fun i (paramName, paramType) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else paramType
                    (paramName, sprintf "%%arg%d" i, Serialize.mlirType (mapType actualType)))
                mlirPs, bindings

        // Return type: main always returns i32
        let returnType =
            if isMain then Integer I32
            else
                match node.Type with
                | NativeType.TFun (_, retTy) -> mapType retTy
                | _ -> mapType node.Type

        // Enter function scope - main is NOT internal (must be exported)
        let zipper2 =
            if isMain then
                MLIRZipper.enterFunctionWithVisibility lambdaName mlirParams returnType false zipper1
            else
                MLIRZipper.enterFunction lambdaName mlirParams returnType zipper1

        // Bind F# parameter names to their SSA positions
        let zipper3 =
            paramBindings
            |> List.fold (fun z (paramName, ssaName, mlirType) ->
                MLIRZipper.bindVar paramName ssaName mlirType z
            ) zipper2

        // Also bind the lambda name to the node for later retrieval
        MLIRZipper.bindNodeSSA (string (NodeId.value node.Id) + "_lambdaName") lambdaName "func" zipper3
    | _ -> zipper

/// Find entry point Lambda IDs from the semantic graph
/// Entry point Bindings have Lambda children - those are the entry point Lambdas
let findEntryPointLambdaIds (graph: SemanticGraph) : Set<int> =
    graph.EntryPoints
    |> List.collect (fun epId ->
        match SemanticGraph.tryGetNode epId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding (_, _, _, _) ->
                // Entry point Binding - its children include the Lambda
                node.Children
                |> List.choose (fun childId ->
                    match SemanticGraph.tryGetNode childId graph with
                    | Some child when (match child.Kind with SemanticKind.Lambda _ -> true | _ -> false) ->
                        Some (NodeId.value childId)
                    | _ -> None)
            | SemanticKind.Lambda _ ->
                // Entry point is directly a Lambda
                [NodeId.value epId]
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Check module members for main binding
                memberIds
                |> List.collect (fun memberId ->
                    match SemanticGraph.tryGetNode memberId graph with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding (name, _, _, _) when name = "main" ->
                            memberNode.Children
                            |> List.choose (fun childId ->
                                match SemanticGraph.tryGetNode childId graph with
                                | Some child ->
                                    match child.Kind with
                                    | SemanticKind.Lambda _ -> Some (NodeId.value childId)
                                    | _ -> None
                                | None -> None)
                        | _ -> []
                    | None -> [])
            | _ -> []
        | None -> [])
    |> Set.ofList

// ═══════════════════════════════════════════════════════════════════
// SCF Region Hook for Control Flow Tracking
// ═══════════════════════════════════════════════════════════════════

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
                        // Analyze body subtree to find which mutable vars are Set
                        let modifiedVarNames = findModifiedVarsInSubtree graph bodyId

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
                        // Similar logic for ForLoop
                        let modifiedVarNames = findModifiedVarsInSubtree graph bodyId
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
            | _ -> zipper

        // Begin region tracking
        MLIRZipper.beginSCFRegion parentIdStr scfKind zipper'

    AfterRegion = fun zipper nodeId regionKind ->
        let parentIdStr = string (NodeId.value nodeId)
        let scfKind = mapRegionKind regionKind
        // End region tracking (captures ops)
        MLIRZipper.endSCFRegion parentIdStr scfKind zipper
}

/// Transfer an entire SemanticGraph to MLIR
/// Uses post-order traversal so children are witnessed before parents
/// Lambda parameters are pre-bound before their bodies are processed
/// SCF region hooks capture operations for control flow constructs
/// isFreestanding: if true, adds a _start wrapper that calls main and then exit
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    // Initialize platform bindings
    Alex.Bindings.Console.ConsoleBindings.registerBindings()
    Alex.Bindings.Console.ConsoleBindings.registerConsoleIntrinsics()
    Alex.Bindings.Process.ProcessBindings.registerBindings()
    Alex.Bindings.Time.TimeBindings.registerBindings()

    // Find entry point Lambda IDs - these will be named "main"
    let entryPointLambdaIds = findEntryPointLambdaIds graph

    // Pre-analyze: find mutable bindings whose address is taken
    let addressedMutables = findAddressedMutableBindings graph

    // Create initial zipper with entry point and addressed mutables knowledge
    let initialZipper = MLIRZipper.createWithAnalysis entryPointLambdaIds addressedMutables

    // Create SCF region hook with graph access for iter_args analysis
    let scfHook = createSCFRegionHook graph

    // Use foldWithSCFRegions to bind params before body processing
    // AND track SCF region boundaries for control flow constructs
    let traversedZipper =
        Traversal.foldWithSCFRegions
            preBindLambdaParams
            (Some scfHook)
            (fun zipper node ->
                let zipper', _result = witnessNode graph node zipper
                zipper')
            initialZipper
            graph

    // For freestanding binaries, add a _start wrapper that calls main and exit syscall
    // This is necessary because freestanding binaries can't return from main
    let finalZipper =
        if isFreestanding then
            MLIRZipper.addFreestandingEntryPoint traversedZipper
        else
            traversedZipper

    // Extract final MLIR - entry point Lambda is named main
    MLIRZipper.extract finalZipper

/// Transfer a graph and return both MLIR text and any errors
/// isFreestanding: if true, adds a _start wrapper that calls main and then exit
let transferGraphWithDiagnostics (graph: SemanticGraph) (isFreestanding: bool) : string * string list =
    // Initialize platform bindings
    Alex.Bindings.Console.ConsoleBindings.registerBindings()
    Alex.Bindings.Console.ConsoleBindings.registerConsoleIntrinsics()
    Alex.Bindings.Process.ProcessBindings.registerBindings()
    Alex.Bindings.Time.TimeBindings.registerBindings()

    // Find entry point Lambda IDs - these will be named "main"
    let entryPointLambdaIds = findEntryPointLambdaIds graph

    // Pre-analyze: find mutable bindings whose address is taken
    let addressedMutables = findAddressedMutableBindings graph

    // Create initial zipper with entry point and addressed mutables knowledge
    let initialZipper = MLIRZipper.createWithAnalysis entryPointLambdaIds addressedMutables

    // Create SCF region hook with graph access for iter_args analysis
    let scfHook = createSCFRegionHook graph

    // Track errors during traversal
    let mutable errors = []

    // Use foldWithSCFRegions to bind params before body processing
    // AND track SCF region boundaries for control flow constructs
    let traversedZipper =
        Traversal.foldWithSCFRegions
            preBindLambdaParams
            (Some scfHook)
            (fun zipper node ->
                let zipper', result = witnessNode graph node zipper
                match result with
                | TRError msg ->
                    errors <- msg :: errors
                    zipper'
                | _ ->
                    zipper')
            initialZipper
            graph

    // For freestanding binaries, add a _start wrapper that calls main and exit syscall
    let finalZipper =
        if isFreestanding then
            MLIRZipper.addFreestandingEntryPoint traversedZipper
        else
            traversedZipper

    // Extract final MLIR - entry point Lambda is named main
    MLIRZipper.extract finalZipper, List.rev errors
