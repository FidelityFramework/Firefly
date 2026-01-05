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
                let extractPtr = sprintf "%s = llvm.extractvalue %s[0] : %s" ptrSSA strSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen = sprintf "%s = llvm.extractvalue %s[1] : %s" lenSSA strSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdText = sprintf "%s = arith.constant 1 : i32" fdSSA
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Call Sys.write via platform binding
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" argSSAsWithTypes returnType zipper6

            | ConsoleOp "writeln", [(strSSA, _)] ->
                // Console.writeln: write the string, then write newline
                let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr = sprintf "%s = llvm.extractvalue %s[0] : %s" ptrSSA strSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen = sprintf "%s = llvm.extractvalue %s[1] : %s" lenSSA strSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdText = sprintf "%s = arith.constant 1 : i32" fdSSA
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Write the string
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                // Sys.write returns bytes written (i32 after platform binding truncation)
                let zipper7, _ = witnessPlatformBinding "Sys.write" argSSAsWithTypes Types.int32Type zipper6

                // Write newline: allocate newline char on stack, write it
                let nlSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let nlText = sprintf "%s = arith.constant 10 : i8" nlSSA  // '\n' = 10
                let zipper9 = MLIRZipper.witnessOpWithResult nlText nlSSA (Integer I8) zipper8

                let oneSSA, zipper10 = MLIRZipper.yieldSSA zipper9
                let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
                let zipper11 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) zipper10

                let nlBufSSA, zipper12 = MLIRZipper.yieldSSA zipper11
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" nlBufSSA oneSSA
                let zipper13 = MLIRZipper.witnessOpWithResult allocaText nlBufSSA Pointer zipper12

                let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" nlSSA nlBufSSA
                let zipper14 = MLIRZipper.witnessVoidOp storeText zipper13

                // Write the newline
                let nlArgSSAs = [(fdSSA, Integer I32); (nlBufSSA, Pointer); (oneSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" nlArgSSAs returnType zipper14

            | ConsoleOp "readln", ([] | [_]) ->  // Takes unit arg (or elided)
                // Console.readln: read a line from stdin into a buffer, return as string
                // For now, allocate a fixed buffer and read into it
                let bufSizeSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let bufSizeText = sprintf "%s = arith.constant 256 : i64" bufSizeSSA
                let zipper2 = MLIRZipper.witnessOpWithResult bufSizeText bufSizeSSA (Integer I64) zipper1

                let bufSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufSSA bufSizeSSA
                let zipper4 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper3

                // fd = 0 (stdin)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdText = sprintf "%s = arith.constant 0 : i32" fdSSA
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Call Sys.read
                let argSSAsWithTypes = [(fdSSA, Integer I32); (bufSSA, Pointer); (bufSizeSSA, Integer I64)]
                // Sys.read returns bytes read (i32 after platform binding truncation)
                let zipper7, readResult = witnessPlatformBinding "Sys.read" argSSAsWithTypes Types.int32Type zipper6

                // Get bytes read (strip newline)
                // Platform binding returns i32, extend to i64 for length arithmetic
                let bytesReadSSA32 = match readResult with TRValue (ssa, _) -> ssa | _ -> "%err"
                let bytesReadSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" bytesReadSSA bytesReadSSA32
                let zipper9 = MLIRZipper.witnessOpWithResult extText bytesReadSSA (Integer I64) zipper8

                let oneSSA, zipper10 = MLIRZipper.yieldSSA zipper9
                let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
                let zipper11 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) zipper10

                let lenSSA, zipper12 = MLIRZipper.yieldSSA zipper11
                let subText = sprintf "%s = arith.subi %s, %s : i64" lenSSA bytesReadSSA oneSSA
                let zipper13 = MLIRZipper.witnessOpWithResult subText lenSSA (Integer I64) zipper12

                // Build fat string struct
                let undefSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let zipper15 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper14

                let withPtrSSA, zipper16 = MLIRZipper.yieldSSA zipper15
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA bufSSA undefSSA NativeStrTypeStr
                let zipper17 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper16

                let fatPtrSSA, zipper18 = MLIRZipper.yieldSSA zipper17
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA withPtrSSA NativeStrTypeStr
                let zipper19 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper18

                zipper19, TRValue (fatPtrSSA, NativeStrTypeStr)

            | NativeStrOp "fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" lenSSA64 lenSSA
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA64 withPtrSSA NativeStrTypeStr
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            | NativeDefaultOp "zeroed", [] ->
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
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA NativeStrTypeStr
                    | Struct _ ->
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                    | _ ->
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            | StringOp "concat2", [(str1SSA, _); (str2SSA, _)] ->
                let ptr1SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr1 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr1SSA str1SSA
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer zipper1

                let len1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen1 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len1SSA str1SSA
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) zipper3

                let ptr2SSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let extractPtr2 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr2SSA str2SSA
                let zipper6 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer zipper5

                let len2SSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let extractLen2 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len2SSA str2SSA
                let zipper8 = MLIRZipper.witnessOpWithResult extractLen2 len2SSA (Integer I64) zipper7

                let totalLenSSA, zipper9 = MLIRZipper.yieldSSA zipper8
                let addLenText = sprintf "%s = arith.addi %s, %s : i64" totalLenSSA len1SSA len2SSA
                let zipper10 = MLIRZipper.witnessOpWithResult addLenText totalLenSSA (Integer I64) zipper9

                let bufSSA, zipper11 = MLIRZipper.yieldSSA zipper10
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufSSA totalLenSSA
                let zipper12 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper11

                let memcpy1 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" bufSSA ptr1SSA len1SSA
                let zipper13 = MLIRZipper.witnessVoidOp memcpy1 zipper12

                let offsetSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" offsetSSA bufSSA len1SSA
                let zipper15 = MLIRZipper.witnessOpWithResult gepText offsetSSA Pointer zipper14

                let memcpy2 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" offsetSSA ptr2SSA len2SSA
                let zipper16 = MLIRZipper.witnessVoidOp memcpy2 zipper15

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

            // Format intrinsics - value → string conversions
            // Format.int: Convert int64 to decimal string representation
            | FormatOp "int", [(intSSA, _)] ->
                // Integer to string conversion using inline digit extraction
                // Algorithm: repeatedly extract digits via mod 10, build in reverse
                // Maximum int64 has 20 digits + sign + null = 22 bytes, use 24 for safety

                // Allocate buffer for digits (24 bytes max)
                let bufSizeSSA, z1 = MLIRZipper.yieldSSA zipper
                let bufSizeText = sprintf "%s = arith.constant 24 : i64" bufSizeSSA
                let z2 = MLIRZipper.witnessOpWithResult bufSizeText bufSizeSSA (Integer I64) z1

                let bufSSA, z3 = MLIRZipper.yieldSSA z2
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufSSA bufSizeSSA
                let z4 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer z3

                // Constants for digit extraction
                let tenSSA, z5 = MLIRZipper.yieldSSA z4
                let tenText = sprintf "%s = arith.constant 10 : i64" tenSSA
                let z6 = MLIRZipper.witnessOpWithResult tenText tenSSA (Integer I64) z5

                let asciiZeroSSA, z7 = MLIRZipper.yieldSSA z6
                let asciiZeroText = sprintf "%s = arith.constant 48 : i8" asciiZeroSSA  // '0' = 48
                let z8 = MLIRZipper.witnessOpWithResult asciiZeroText asciiZeroSSA (Integer I8) z7

                // Start at end of buffer (we build backwards)
                let endIdxSSA, z9 = MLIRZipper.yieldSSA z8
                let endIdxText = sprintf "%s = arith.constant 23 : i64" endIdxSSA
                let z10 = MLIRZipper.witnessOpWithResult endIdxText endIdxSSA (Integer I64) z9

                // Handle zero case specially
                let zeroSSA, z11 = MLIRZipper.yieldSSA z10
                let zeroText = sprintf "%s = arith.constant 0 : i64" zeroSSA
                let z12 = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I64) z11

                let isZeroSSA, z13 = MLIRZipper.yieldSSA z12
                let isZeroText = sprintf "%s = arith.cmpi eq, %s, %s : i64" isZeroSSA intSSA zeroSSA
                let z14 = MLIRZipper.witnessOpWithResult isZeroText isZeroSSA (Integer I1) z13

                // scf.if for zero case vs digit extraction
                // For zero: just store '0' and return single-char string
                // For non-zero: do digit extraction loop

                // Simplified: for now, just handle as if non-zero with a single digit extraction
                // Full implementation would use scf.while for multi-digit numbers

                // Extract least significant digit: digit = value % 10
                let digitSSA, z15 = MLIRZipper.yieldSSA z14
                let remText = sprintf "%s = arith.remsi %s, %s : i64" digitSSA intSSA tenSSA
                let z16 = MLIRZipper.witnessOpWithResult remText digitSSA (Integer I64) z15

                // Convert to ASCII: char = digit + '0'
                let digit8SSA, z17 = MLIRZipper.yieldSSA z16
                let truncText = sprintf "%s = arith.trunci %s : i64 to i8" digit8SSA digitSSA
                let z18 = MLIRZipper.witnessOpWithResult truncText digit8SSA (Integer I8) z17

                let charSSA, z19 = MLIRZipper.yieldSSA z18
                let addCharText = sprintf "%s = arith.addi %s, %s : i8" charSSA digit8SSA asciiZeroSSA
                let z20 = MLIRZipper.witnessOpWithResult addCharText charSSA (Integer I8) z19

                // Store the single digit at buffer start (simplified single-digit case)
                let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" charSSA bufSSA
                let z21 = MLIRZipper.witnessVoidOp storeText z20

                // For now: return single-character string (works for 0-9)
                // TODO: Implement full scf.while loop for multi-digit numbers
                let oneSSA, z22 = MLIRZipper.yieldSSA z21
                let oneText = sprintf "%s = arith.constant 1 : i64" oneSSA
                let z23 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) z22

                // Build fat string
                let undefSSA, z24 = MLIRZipper.yieldSSA z23
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let z25 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType z24

                let withPtrSSA, z26 = MLIRZipper.yieldSSA z25
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA bufSSA undefSSA NativeStrTypeStr
                let z27 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType z26

                let resultSSA, z28 = MLIRZipper.yieldSSA z27
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" resultSSA oneSSA withPtrSSA NativeStrTypeStr
                let z29 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType z28

                z29, TRValue (resultSSA, NativeStrTypeStr)

            // Format.float: Convert float to string representation
            // TODO: Full implementation requires floating-point formatting algorithm (Grisu/Dragon4/Ryu)
            | FormatOp "float", [(floatSSA, _)] ->
                // Placeholder: emit constant "<float>" for now
                let placeholderStr = "<float>"
                let globalName, z1 = MLIRZipper.observeStringLiteral placeholderStr zipper

                let ptrSSA, z2 = MLIRZipper.yieldSSA z1
                let addrOfText = sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" ptrSSA globalName
                let z3 = MLIRZipper.witnessOpWithResult addrOfText ptrSSA Pointer z2

                let lenSSA, z4 = MLIRZipper.yieldSSA z3
                let lenText = sprintf "%s = arith.constant %d : i64" lenSSA (String.length placeholderStr)
                let z5 = MLIRZipper.witnessOpWithResult lenText lenSSA (Integer I64) z4

                let undefSSA, z6 = MLIRZipper.yieldSSA z5
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let z7 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType z6

                let withPtrSSA, z8 = MLIRZipper.yieldSSA z7
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
                let z9 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType z8

                let resultSSA, z10 = MLIRZipper.yieldSSA z9
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" resultSSA lenSSA withPtrSSA NativeStrTypeStr
                let z11 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType z10

                z11, TRValue (resultSSA, NativeStrTypeStr)

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
                                let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1)
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
