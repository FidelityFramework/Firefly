/// Memory Fields Witness - Record/struct field access
///
/// SCOPE: FieldGet, FieldSet, NestedFieldGet, field index resolution
/// DOES NOT: Indexing, aggregates, DUs (separate witnesses)
module Alex.Witnesses.Memory.Fields

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Witnesses.Memory.Indexing

let private resolveFieldIndex (structNativeType: NativeType) (fieldName: string) (ctx: WitnessContext) : int =
    match structNativeType with
    | NativeType.TApp (tc, _) ->
        match tc.Name with
        | "string" ->
            // String layout: (Pointer, Length)
            match fieldName with
            | "Pointer" -> 0
            | "Length" -> 1
            | _ -> failwithf "Unknown string field: %s" fieldName
        | _ ->
            // DU layout: (Tag, Item1, Item2, ...)
            // Record layout: fields in definition order
            match fieldName with
            | "Tag" -> 0
            | name when name.StartsWith("Item") ->
                match System.Int32.TryParse(name.Substring(4)) with
                | true, n -> n  // Item1 -> 1, Item2 -> 2
                | false, _ -> failwithf "Invalid DU field name: %s" name
            | _ ->
                // Record field - look up from TypeDef in SemanticGraph
                // TypeDefs use simple names (not module-qualified) per FNCS architecture
                match SemanticGraph.tryGetRecordFields tc.Name ctx.Graph with
                | Some fields ->
                    // Find field index by name
                    match fields |> List.tryFindIndex (fun (name, _) -> name = fieldName) with
                    | Some idx -> idx
                    | None -> failwithf "Record type '%s' has no field '%s'. Available fields: %A" tc.Name fieldName (fields |> List.map fst)
                | None ->
                    failwithf "Type '%s' not found in SemanticGraph or is not a record type" tc.Name
    | _ -> failwithf "FieldGet on non-TApp type: %A" structNativeType

/// Witness field get (struct.field or record.field)
/// Resolves field name to index based on type, then extracts
///
/// ARCHITECTURAL NOTE: This is TRANSLITERATION.
/// For DU payload extraction (Item1, Item2, etc.), we construct a case-specific
/// struct type using the KNOWN fieldType. LLVM doesn't see "a DU" as a single
/// construct - each match branch interprets memory with its own type.
/// In FloatVal branch: struct is (i8, f64). In IntVal branch: struct is (i8, i32).
/// The fieldType comes from PSG (baked in at Baker level) and is authoritative.
///
/// Uses 1 pre-assigned SSA: extract[0]
let witnessFieldGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structNativeType: NativeType)
    (fieldName: string)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldIndex = resolveFieldIndex structNativeType fieldName ctx

    // For DU payload extraction, construct case-specific struct type
    // using the KNOWN fieldType from PSG. This is transliteration - we state
    // the correct type for this specific branch.
    let structMlirType =
        match fieldName with
        | name when name.StartsWith("Item") ->
            // DU layout: (tag, payload) with case-specific payload type
            TStruct [TInt I8; fieldType]
        | _ ->
            // Tag extraction, string fields, records - use graph-aware mapping
            // that correctly handles nested record types
            mapType structNativeType ctx

    let fieldSSA = requireNodeSSA nodeId ctx
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, [fieldIndex], structMlirType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness nested field get (struct.field1.field2.etc)
/// Uses extractvalue with multiple indices for nested access
/// Uses 1 pre-assigned SSA: extract[0]
let witnessNestedFieldGet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structType: MLIRType)
    (indices: int list)
    (fieldType: MLIRType)
    : MLIROp list * TransferResult =

    let fieldSSA = requireNodeSSA nodeId ctx
    let extractOp = MLIROp.LLVMOp (LLVMOp.ExtractValue (fieldSSA, structSSA, indices, structType))

    [extractOp], TRValue { SSA = fieldSSA; Type = fieldType }

/// Witness field set (struct.field <- value)
/// Uses insertvalue for value types (returns new struct)
/// Uses 1 pre-assigned SSA: insert[0]
let witnessFieldSet
    (nodeId: NodeId)
    (ctx: WitnessContext)
    (structSSA: SSA)
    (structType: MLIRType)
    (fieldIndex: int)
    (valueSSA: SSA)
    : MLIROp list * TransferResult =

    let newStructSSA = requireNodeSSA nodeId ctx
    let insertOp = MLIROp.LLVMOp (LLVMOp.InsertValue (newStructSSA, structSSA, valueSSA, [fieldIndex], structType))

    [insertOp], TRValue { SSA = newStructSSA; Type = structType }

// ═══════════════════════════════════════════════════════════════════════════
// TUPLE CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness tuple construction
/// Builds a struct by inserting each element
/// Uses N+1 pre-assigned SSAs: undef[0], insert[1..N]
