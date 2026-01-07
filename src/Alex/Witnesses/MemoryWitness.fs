/// Memory Witness - Witness memory and data structure operations
///
/// ARCHITECTURAL FOUNDATION:
/// This module witnesses memory-related PSG nodes including:
/// - Array/collection indexing (IndexGet, IndexSet)
/// - Address-of operator (AddressOf)
/// - Tuple/Record/Array/List construction
/// - Field access (FieldGet, FieldSet)
/// - SRTP trait calls
module Alex.Witnesses.MemoryWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Templates.TemplateTypes
open Alex.Templates.MemoryTemplates

module ArithTemplates = Alex.Templates.ArithTemplates
module LLVMTemplates = Alex.Templates.LLVMTemplates

// ═══════════════════════════════════════════════════════════════════════════
// Type Mapping Helper (delegated to TypeMapping module)
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// String Type Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type string represents native string type
let isNativeStrType (tyStr: string) : bool =
    tyStr = NativeStrTypeStr || tyStr.Contains("struct<(ptr, i64)>")

// ═══════════════════════════════════════════════════════════════════════════
// Index Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array/collection index get
let witnessIndexGet 
    (collectionId: NodeId) 
    (indexId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper with
    | Some (collSSA, _collType), Some (indexSSA, _) ->
        // Generate GEP (getelementptr) for array access
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let elemType = Serialize.mlirType (mapType node.Type)
        let gepParams = { Result = ssaName; Base = collSSA; Offset = indexSSA; ElementType = "i8" }
        let text = render Quot.Gep.i64 gepParams
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
        // Load the element
        let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
        let loadParams = { Result = loadSSA; Pointer = ssaName; Type = elemType }
        let loadText = render Quot.Core.load loadParams
        let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType node.Type) zipper'''
        let zipper5 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) loadSSA elemType zipper4
        zipper5, TRValue (loadSSA, elemType)
    | _ ->
        zipper, TRError "IndexGet: collection or index not computed"

/// Witness array/collection index set
let witnessIndexSet 
    (collectionId: NodeId) 
    (indexId: NodeId) 
    (valueId: NodeId) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
    | Some (collSSA, _), Some (indexSSA, _), Some (valueSSA, valueType) ->
        // Generate GEP for array access
        let ptrSSA, zipper' = MLIRZipper.yieldSSA zipper
        let gepParams = { Result = ptrSSA; Base = collSSA; Offset = indexSSA; ElementType = "i8" }
        let gepText = render Quot.Gep.i64 gepParams
        let zipper'' = MLIRZipper.witnessOpWithResult gepText ptrSSA Pointer zipper'
        // Store the value
        let storeParams = { Value = valueSSA; Pointer = ptrSSA; Type = valueType }
        let storeText = render Quot.Core.store storeParams
        let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
        zipper''', TRVoid
    | _ ->
        zipper, TRError "IndexSet: collection, index, or value not computed"

// ═══════════════════════════════════════════════════════════════════════════
// Address-Of Operator
// ═══════════════════════════════════════════════════════════════════════════

/// Witness address-of operator
let witnessAddressOf 
    (exprId: NodeId) 
    (isMutable: bool) 
    (node: SemanticNode) 
    (graph: SemanticGraph) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
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

// ═══════════════════════════════════════════════════════════════════════════
// Tuple Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness tuple construction
let witnessTupleExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let elementSSAs =
        elementIds
        |> List.choose (fun elemId ->
            MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

    if List.length elementSSAs <> List.length elementIds then
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
            // Multi-element tuple - build struct value using undef + insertvalue
            // Get element types from node.Type (should be TTuple)
            let elemTypeStrs =
                elements |> List.map (fun (_, tyStr) -> tyStr)
            let tupleTypeStr = sprintf "!llvm.struct<(%s)>" (String.concat ", " elemTypeStrs)

            // Start with undef struct
            let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA tupleTypeStr
            let zipper2 = MLIRZipper.witnessOp undefText [(undefSSA, Struct [])] zipper1

            // Insert each element into the struct
            let finalSSA, finalZipper =
                elements
                |> List.mapi (fun i (ssa, _) -> (i, ssa))
                |> List.fold (fun (accSSA, z) (idx, elemSSA) ->
                    let newSSA, z1 = MLIRZipper.yieldSSA z
                    let insertText = sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" newSSA elemSSA accSSA idx tupleTypeStr
                    let z2 = MLIRZipper.witnessOp insertText [(newSSA, Struct [])] z1
                    (newSSA, z2)
                ) (undefSSA, zipper2)

            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) finalSSA tupleTypeStr finalZipper
            zipper3, TRValue (finalSSA, tupleTypeStr)

// ═══════════════════════════════════════════════════════════════════════════
// Record Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness record construction
let witnessRecordExpr 
    (fields: (string * NodeId) list) 
    (copyFrom: NodeId option) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let fieldSSAs =
        fields
        |> List.choose (fun (_fieldName, valueId) ->
            MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper)

    if List.length fieldSSAs <> List.length fields then
        zipper, TRError "Record fields not all computed"
    else
        // For a single-field record, just return that field's value
        match fieldSSAs with
        | [(ssa, ty)] ->
            let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper1, TRValue (ssa, ty)
        | _ ->
            // Multi-field record - construct LLVM struct
            // 1. Create struct type from field types
            let fieldTypeStrs = fieldSSAs |> List.map snd
            let structTypeStr = sprintf "!llvm.struct<(%s)>" (String.concat ", " fieldTypeStrs)

            // 2. Create undef struct
            let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA structTypeStr
            let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA (Struct []) zipper1

            // 3. Insert each field value
            let zipper3, finalSSA =
                fieldSSAs
                |> List.indexed
                |> List.fold (fun (z, prevSSA) (idx, (fieldSSA, _fieldTy)) ->
                    let insertSSA, z1 = MLIRZipper.yieldSSA z
                    let insertText = sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" insertSSA fieldSSA prevSSA idx structTypeStr
                    let z2 = MLIRZipper.witnessOpWithResult insertText insertSSA (Struct []) z1
                    z2, insertSSA
                ) (zipper2, undefSSA)

            let zipper4 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) finalSSA structTypeStr zipper3
            zipper4, TRValue (finalSSA, structTypeStr)

// ═══════════════════════════════════════════════════════════════════════════
// Array Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array construction
let witnessArrayExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
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
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper2
            zipper3, TRValue (arrSSA, "!llvm.ptr")

// ═══════════════════════════════════════════════════════════════════════════
// List Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness list construction
let witnessListExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
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

// ═══════════════════════════════════════════════════════════════════════════
// Field Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Witness field get (expr.fieldName)
/// Uses the expression's NativeType to determine field index for extractvalue
let witnessFieldGet
    (exprId: NodeId)
    (fieldName: string)
    (node: SemanticNode)
    (graph: SemanticGraph)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =

    match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
    | Some (exprSSA, exprType) ->
        // STRING INTRINSIC MEMBERS: Native string is fat pointer {ptr, len}
        // .Pointer → extractvalue at index 0 → !llvm.ptr
        // .Length → extractvalue at index 1 → i64
        if isNativeStrType exprType then
            match fieldName with
            | "Pointer" ->
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let extractParams : Quot.Aggregate.ExtractParams = { Result = resultSSA; Aggregate = exprSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractText = render Quot.Aggregate.extractValue extractParams
                let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                zipper''', TRValue (resultSSA, "!llvm.ptr")
            | "Length" ->
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let extractParams : Quot.Aggregate.ExtractParams = { Result = resultSSA; Aggregate = exprSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractText = render Quot.Aggregate.extractValue extractParams
                let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA (Integer I64) zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "i64" zipper''
                zipper''', TRValue (resultSSA, "i64")
            | _ ->
                zipper, TRError (sprintf "Unknown string field: %s (expected Pointer or Length)" fieldName)
        else
            // Look up expression node to get its NativeType
            match SemanticGraph.tryGetNode exprId graph with
            | Some exprNode ->
                match exprNode.Type with
                | NativeType.TApp(tycon, _) ->
                    // FCS pattern: TyconRef.Deref for field info - look up via SemanticGraph.Types
                    // TypeConRef.Name already contains the fully qualified name matching TypeDef
                    let typeName = tycon.Name
                    match SemanticGraph.tryGetRecordFields typeName graph with
                    | Some fields ->
                        // Find field index by name
                        match fields |> List.tryFindIndex (fun (name, _) -> name = fieldName) with
                        | Some fieldIndex ->
                            // Get field type for result
                            let _, fieldType = fields.[fieldIndex]
                            let fieldMLIRType = mapType fieldType
                            let fieldTypeStr = Serialize.mlirType fieldMLIRType

                            let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                            let extractParams : Quot.Aggregate.ExtractParams = { Result = resultSSA; Aggregate = exprSSA; Index = fieldIndex; AggType = exprType }
                            let extractText = render Quot.Aggregate.extractValue extractParams
                            let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA fieldMLIRType zipper'
                            let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA fieldTypeStr zipper''
                            zipper''', TRValue (resultSSA, fieldTypeStr)
                        | None ->
                            zipper, TRError (sprintf "Field '%s' not found in record type '%s'" fieldName typeName)
                    | None ->
                        zipper, TRError (sprintf "Type '%s' not found in SemanticGraph.Types or is not a record" typeName)
                | _ ->
                    // Non-TApp type - this shouldn't happen for FieldGet on records
                    zipper, TRError (sprintf "FieldGet on non-TApp type: %A" exprNode.Type)
            | None ->
                zipper, TRError (sprintf "FieldGet: expression node %A not found in graph" exprId)
    | None ->
        zipper, TRError (sprintf "FieldGet '%s': expression not computed" fieldName)

/// Witness field set (expr.fieldName <- value)
/// For mutable record fields - uses insertvalue for struct values
let witnessFieldSet
    (exprId: NodeId)
    (fieldName: string)
    (valueId: NodeId)
    (graph: SemanticGraph)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =

    match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
    | Some (exprSSA, exprType), Some (valueSSA, valueType) ->
        // Look up expression node to get its NativeType
        match SemanticGraph.tryGetNode exprId graph with
        | Some exprNode ->
            match exprNode.Type with
            | NativeType.TApp(tycon, _) ->
                // FCS pattern: TyconRef.Deref for field info - look up via SemanticGraph.Types
                // TypeConRef.Name already contains the fully qualified name matching TypeDef
                let typeName = tycon.Name
                match SemanticGraph.tryGetRecordFields typeName graph with
                | Some fields ->
                    // Find field index by name
                    match fields |> List.tryFindIndex (fun (name, _) -> name = fieldName) with
                    | Some fieldIndex ->
                        // Use insertvalue to create a new struct with the updated field
                        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                        let insertParams : Quot.Aggregate.InsertParams = { Result = resultSSA; Value = valueSSA; Aggregate = exprSSA; Index = fieldIndex; AggType = exprType }
                        let insertText = render Quot.Aggregate.insertValue insertParams
                        let zipper'' = MLIRZipper.witnessOpWithResult insertText resultSSA (mapType exprNode.Type) zipper'
                        // Bind the updated struct (for chained operations)
                        let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value exprId)) resultSSA exprType zipper''
                        zipper''', TRVoid
                    | None ->
                        zipper, TRError (sprintf "FieldSet: field '%s' not found in record type '%s'" fieldName typeName)
                | None ->
                    // Not a record type - fall back to store (pointer semantics)
                    let storeParams = { Value = valueSSA; Pointer = exprSSA; Type = valueType }
                    let storeText = render Quot.Core.store storeParams
                    let zipper' = MLIRZipper.witnessVoidOp storeText zipper
                    zipper', TRVoid
            | _ ->
                // Non-TApp type - fall back to store (pointer semantics)
                let storeParams = { Value = valueSSA; Pointer = exprSSA; Type = valueType }
                let storeText = render Quot.Core.store storeParams
                let zipper' = MLIRZipper.witnessVoidOp storeText zipper
                zipper', TRVoid
        | None ->
            zipper, TRError (sprintf "FieldSet: expression node %A not found in graph" exprId)
    | _ ->
        zipper, TRError (sprintf "FieldSet '%s': expression or value not computed" fieldName)

// ═══════════════════════════════════════════════════════════════════════════
// TraitCall (SRTP)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness SRTP trait call
let witnessTraitCall
    (memberName: string)
    (typeArgs: NativeType list)
    (argId: NodeId)
    (node: SemanticNode)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =

    match MLIRZipper.recallNodeSSA (string (NodeId.value argId)) zipper with
    | Some (argSSA, argType) ->
        // For now, emit a call to the trait member name
        // TODO: Proper SRTP resolution from Baker/type checker
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let callParams : LLVMTemplates.Quot.Control.CallParams = {
            Result = resultSSA
            Callee = memberName
            Args = argSSA
            ArgTypes = argType
            ReturnType = "!llvm.ptr"
        }
        let callText = render LLVMTemplates.Quot.Control.call callParams
        let zipper'' = MLIRZipper.witnessOpWithResult callText resultSSA Pointer zipper'
        let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
        zipper''', TRValue (resultSSA, "!llvm.ptr")
    | None ->
        zipper, TRError (sprintf "TraitCall '%s': argument not computed" memberName)

// ═══════════════════════════════════════════════════════════════════════════
// Union Case Construction
// ═══════════════════════════════════════════════════════════════════════════

/// MLIR type for discriminated unions (tag + max payload)
let unionTypeStr = "!llvm.struct<(i32, i64)>"

/// Witness union case construction: IntVal 42, FloatVal 3.14, etc.
///
/// DU construction pattern:
/// 1. Create undef struct: %undef = llvm.mlir.undef : !llvm.struct<(i32, i64)>
/// 2. Insert tag: %with_tag = llvm.insertvalue %tag_val, %undef[0] : !llvm.struct<(i32, i64)>
/// 3. Insert payload (if any): %result = llvm.insertvalue %payload, %with_tag[1] : !llvm.struct<(i32, i64)>
let witnessUnionCase
    (caseName: string)
    (caseIndex: int)
    (payloadOpt: NodeId option)
    (node: SemanticNode)
    (zipper: MLIRZipper)
    : MLIRZipper * TransferResult =
        // Step 1: Create undef struct
        let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let undefParams : {| Result: string; Type: string |} = {| Result = undefSSA; Type = unionTypeStr |}
        let undefText = render Quot.Aggregate.undef undefParams
        let unionMLIRType = Struct [Integer I32; Integer I64]
        let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA unionMLIRType zipper1

        // Step 2: Create tag constant and insert at index 0
        let tagSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let tagConstParams : ConstantParams = { Result = tagSSA; Value = string caseIndex; Type = "i32" }
        let tagConstText = render ArithTemplates.Quot.Constant.intConst tagConstParams
        let zipper4 = MLIRZipper.witnessOpWithResult tagConstText tagSSA (Integer I32) zipper3

        let withTagSSA, zipper5 = MLIRZipper.yieldSSA zipper4
        let insertTagParams : Quot.Aggregate.InsertParams = {
            Result = withTagSSA
            Value = tagSSA
            Aggregate = undefSSA
            Index = 0
            AggType = unionTypeStr
        }
        let insertTagText = render Quot.Aggregate.insertValue insertTagParams
        let zipper6 = MLIRZipper.witnessOpWithResult insertTagText withTagSSA unionMLIRType zipper5

        // Step 3: Insert payload (if present)
        match payloadOpt with
        | None ->
            // No payload - just bind the result with tag only
            let zipper7 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) withTagSSA unionTypeStr zipper6
            zipper7, TRValue (withTagSSA, unionTypeStr)
        | Some payloadId ->
            match MLIRZipper.recallNodeSSA (string (NodeId.value payloadId)) zipper6 with
            | Some (payloadSSA, payloadType) ->
                // Convert payload to i64 for storage in union
                // For integers, use extsi/extui; for floats use bitcast
                let payload64SSA, zipper7 =
                    match payloadType with
                    | "i32" | "i64" ->
                        // Integer - extend to i64 if needed
                        if payloadType = "i64" then
                            payloadSSA, zipper6
                        else
                            let extSSA, z = MLIRZipper.yieldSSA zipper6
                            let extParams : ConversionParams = { Result = extSSA; Operand = payloadSSA; FromType = "i32"; ToType = "i64" }
                            let extText = render ArithTemplates.Quot.Conversion.extSI extParams
                            let z' = MLIRZipper.witnessOpWithResult extText extSSA (Integer I64) z
                            extSSA, z'
                    | "f64" ->
                        // Double - bitcast to i64
                        let castSSA, z = MLIRZipper.yieldSSA zipper6
                        let castParams : Quot.Conversion.BitcastParams = { Result = castSSA; Operand = payloadSSA; FromType = "f64"; ToType = "i64" }
                        let castText = render Quot.Conversion.bitcast castParams
                        let z' = MLIRZipper.witnessOpWithResult castText castSSA (Integer I64) z
                        castSSA, z'
                    | "f32" ->
                        // Float - extend to f64 then bitcast
                        let extSSA, z1 = MLIRZipper.yieldSSA zipper6
                        let extParams : ConversionParams = { Result = extSSA; Operand = payloadSSA; FromType = "f32"; ToType = "f64" }
                        let extText = render ArithTemplates.Quot.Conversion.extF extParams
                        let z2 = MLIRZipper.witnessOpWithResult extText extSSA (Float F64) z1
                        let castSSA, z3 = MLIRZipper.yieldSSA z2
                        let castParams : Quot.Conversion.BitcastParams = { Result = castSSA; Operand = extSSA; FromType = "f64"; ToType = "i64" }
                        let castText = render Quot.Conversion.bitcast castParams
                        let z4 = MLIRZipper.witnessOpWithResult castText castSSA (Integer I64) z3
                        castSSA, z4
                    | _ ->
                        // Other types - assume they fit in i64 or are already i64
                        payloadSSA, zipper6

                // Insert payload at index 1
                let resultSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let insertPayloadParams : Quot.Aggregate.InsertParams = {
                    Result = resultSSA
                    Value = payload64SSA
                    Aggregate = withTagSSA
                    Index = 1
                    AggType = unionTypeStr
                }
                let insertPayloadText = render Quot.Aggregate.insertValue insertPayloadParams
                let zipper9 = MLIRZipper.witnessOpWithResult insertPayloadText resultSSA unionMLIRType zipper8
                let zipper10 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA unionTypeStr zipper9
                zipper10, TRValue (resultSSA, unionTypeStr)
            | None ->
                zipper6, TRError (sprintf "UnionCase '%s': payload not computed" caseName)
