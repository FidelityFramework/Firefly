# Firefly Discriminated Union Handling Guide

## Overview

Discriminated unions are fundamental to F# and critical for Firefly's design, especially for BAREWire schema preservation. This guide covers how to handle them throughout the compilation pipeline.

## The Challenge

F# Compiler Services (FCS) uses discriminated unions extensively in its AST representation. When serializing these for debug output, we encounter challenges because:

1. System.Text.Json doesn't support F# DUs by default
2. FCS types often wrap simple values in DUs for type safety
3. We need both human-readable debug output AND type-preserving internal representations

## Solutions by Context

### 1. Debug Output (JSON Serialization)

For debug assets, extract simple values:

```fsharp
// Extract string from QualifiedNameOfFile
let extractQualifiedName (qualName: QualifiedNameOfFile) =
    match qualName with
    | QualifiedNameOfFile(Ident(name, _)) -> name

// Extract identifier list from SynLongIdent
let extractIdentList (synLongIdent: SynLongIdent) =
    match synLongIdent with
    | SynLongIdent(identList, _, _) -> identList
```

### 2. Internal Processing (Type Preservation)

For compilation stages, preserve the full DU structure:

```fsharp
// Pattern matching preserves all information
match synPat with
| SynPat.Named(SynIdent(ident, trivia), isThis, access, range) ->
    // Process with full type information
    processNamedPattern ident trivia isThis access range
    
| SynPat.LongIdent(SynLongIdent(ids, dots, trivia), extraId, typarDecls, args, access, range) ->
    // Access all structural information
    processLongIdentPattern ids dots trivia extraId typarDecls args access range
```

### 3. BAREWire Schema Generation

For your BAREWire protocol, create explicit mappings:

```fsharp
/// Convert F# DU to BARE union schema
let toBareUnion (unionType: FSharpEntity) : BareUnionSchema =
    let cases = 
        unionType.UnionCases 
        |> Seq.map (fun case ->
            {| 
                Name = case.Name
                Tag = uint8 case.Tag  // BARE uses uint for tags
                Fields = 
                    case.Fields 
                    |> Seq.map (fun field -> toBareType field.FieldType)
                    |> Seq.toList
            |})
        |> Seq.toList
    
    {
        UnionName = unionType.DisplayName
        Cases = cases
        Encoding = if cases.Length <= 256 then U8Tag else U16Tag
    }
```

### 4. Memory Layout for DUs

Firefly must determine optimal layout for discriminated unions:

```fsharp
/// Analyze DU for stack allocation
let analyzeUnionLayout (union: FSharpEntity) =
    let cases = union.UnionCases |> Seq.toList
    
    match cases with
    | [] -> EmptyUnion
    | [single] when single.Fields.Count = 0 -> 
        // Enum-like union
        EnumUnion { Size = 1; Alignment = 1 }
    | _ ->
        // Calculate max size across all cases
        let maxSize = 
            cases 
            |> List.map (fun c -> 
                c.Fields |> Seq.sumBy (fun f -> sizeOf f.FieldType))
            |> List.max
        
        TaggedUnion {
            TagSize = if cases.Length <= 256 then 1 else 2
            MaxPayloadSize = maxSize
            Alignment = 8  // Platform specific
        }
```

## Best Practices for Firefly

### 1. Layer Your Representations

```fsharp
// Layer 1: FCS AST (as-is)
type FcsLayer = ParsedInput

// Layer 2: Simplified for analysis
type SimplifiedAst = 
    | Module of name: string * declarations: SimplifiedDecl list
    | Namespace of name: string * modules: SimplifiedAst list

// Layer 3: MLIR-ready representation
type MlirReadyAst =
    | MlirModule of name: string * ops: MlirOp list
```

### 2. Create Conversion Pipelines

```fsharp
// FCS -> Simplified
let simplifyAst (input: ParsedInput) : SimplifiedAst =
    match input with
    | ParsedInput.ImplFile(impl) -> convertImplFile impl
    | ParsedInput.SigFile(sig) -> convertSigFile sig

// Simplified -> MLIR-ready
let prepareForMlir (simplified: SimplifiedAst) : MlirReadyAst =
    // Apply Firefly-specific transformations
    simplified |> eliminateAllocations |> layoutMemory |> toMlirOps
```

### 3. Handle Option Types Specially

F#'s option type is a discriminated union that deserves special handling:

```fsharp
/// Optimize option types for stack allocation
let optimizeOption (optionType: FSharpType) =
    match optionType.GenericArguments with
    | [| valueType |] when isValueType valueType ->
        // Use nullable value type representation
        NullableValueRepresentation valueType
    | [| refType |] ->
        // Use null as None
        NullableReferenceRepresentation refType
    | _ -> 
        // Standard tagged union
        StandardOptionRepresentation
```

## Integration with BAREWire

Your BAREWire protocol can leverage F# DUs naturally:

```fsharp
// F# discriminated union
type NetworkMessage =
    | Heartbeat
    | Data of payload: byte[]
    | Error of code: int * message: string

// Generates BARE schema:
// type NetworkMessage union {
//   | Heartbeat
//   | Data data<byte>
//   | Error struct { code: int, message: string }
// }
```

## Debugging Tips

1. **Use JsonFSharpConverter** for initial prototyping
2. **Extract values** for human-readable debug output
3. **Preserve full types** in internal compiler stages
4. **Test with complex nested unions** early

## Example: Complete DU Handler

```fsharp
module DUHandler =
    open FSharp.Compiler.Syntax
    open System.Text.Json
    
    /// JSON-friendly representation
    type JsonFriendlyDU = {
        Kind: string
        Value: obj
        Metadata: Map<string, obj>
    }
    
    /// Convert any DU to JSON-friendly format
    let toJsonFriendly (du: 'T) : JsonFriendlyDU =
        let duType = du.GetType()
        let case, fields = FSharpValue.GetUnionFields(du, duType)
        
        {
            Kind = case.Name
            Value = 
                match fields with
                | [||] -> null
                | [|single|] -> single
                | multiple -> box multiple
            Metadata = 
                Map.ofList [
                    "Tag", box case.Tag
                    "DeclaringType", box case.DeclaringType.Name
                ]
        }
```

This approach gives you both the debugging visibility you need and the type preservation required for correct compilation.