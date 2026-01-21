/// Sample 13a_BAREWireCollections: Patterns Extracted from BAREWire
/// Tests the specific collection patterns that BAREWire uses.
///
/// Source analysis from:
/// - Schema/Validation.fs, Schema/Analysis.fs, Schema/Definition.fs
/// - Memory/View.fs, Memory/Region.fs
/// - Core/Uuid.fs
///
/// Key patterns:
/// - Map: empty, add, tryFind, containsKey, values, keys, isEmpty, toSeq, forall
/// - Set: empty, add, contains
/// - List: rev, exists, isEmpty, contains, map, sumBy, fold, forall, length, forall2, max
/// - Seq: collect, empty, map, fold, minBy, max, tryPick, toList, append
/// - Array: zeroCreate, init, blit, copy, collect
/// - Other: Option.map, fst, snd, max, min, String.concat
module BAREWireCollections

// ============================================================================
// Types mimicking BAREWire schema structures
// ============================================================================

type SchemaType =
    | Primitive of string
    | Optional of SchemaType
    | Array of SchemaType
    | Map of SchemaType * SchemaType
    | Struct of StructField list
    | Union of Map<int, SchemaType>
    | Enum of Map<string, int>
    | Named of string

and StructField = {
    Name: string
    FieldType: SchemaType
}

type Schema = {
    Types: Map<string, SchemaType>
    Root: string
}

type ValidationError =
    | EmptyEnum
    | EmptyUnion
    | CyclicReference of string list
    | MissingRoot

type PathElement =
    | StructField of string
    | UnionCase

// ============================================================================
// PATTERN 1: Map Operations (from Schema/Definition.fs, Validation.fs)
// ============================================================================

module SchemaOps =
    /// Create empty schema (uses Map.empty)
    let createSchema (rootTypeName: string) : Schema =
        { Types = Map.empty; Root = rootTypeName }

    /// Add type to schema (uses Map.add)
    let addType (name: string) (schemaType: SchemaType) (schema: Schema) : Schema =
        { schema with Types = Map.add name schemaType schema.Types }

    /// Lookup type (uses Map.tryFind)
    let tryFindType (name: string) (schema: Schema) : SchemaType option =
        Map.tryFind name schema.Types

    /// Check if type exists (uses Map.containsKey)
    let hasType (name: string) (schema: Schema) : bool =
        Map.containsKey name schema.Types

    /// Get all type names (uses Map.keys, Seq.toList)
    let getTypeNames (schema: Schema) : string list =
        schema.Types |> Map.keys |> Seq.toList

let testMapOperations () =
    Console.writeln "=== Map Operations (Schema Pattern) ==="

    let schema = SchemaOps.createSchema "Root"
                 |> SchemaOps.addType "Root" (Struct [])
                 |> SchemaOps.addType "Person" (Struct [{Name = "name"; FieldType = Primitive "string"}])
                 |> SchemaOps.addType "Address" (Struct [{Name = "city"; FieldType = Primitive "string"}])

    Console.write "Has 'Person': "
    Console.writeln (if SchemaOps.hasType "Person" schema then "true" else "false")

    Console.write "Has 'Unknown': "
    Console.writeln (if SchemaOps.hasType "Unknown" schema then "true" else "false")

    match SchemaOps.tryFindType "Person" schema with
    | Some _ -> Console.writeln "Found 'Person' type"
    | None -> Console.writeln "Did not find 'Person'"

    let names = SchemaOps.getTypeNames schema
    Console.write "Type count: "
    Console.writeln (Format.int (List.length names))

// ============================================================================
// PATTERN 2: Set Operations (from Schema/Validation.fs - cycle detection)
// ============================================================================

module CycleDetection =
    /// Visit a type, tracking visited set (uses Set.contains, Set.add, Set.empty)
    let rec visit (visited: Set<string>) (path: string list) (typeName: string) (schema: Schema) : string list option =
        if Set.contains typeName visited then
            Some (List.rev (typeName :: path))  // Cycle found
        else if List.contains typeName path then
            Some (List.rev (typeName :: path))  // Also cycle
        else
            let newVisited = Set.add typeName visited
            match Map.tryFind typeName schema.Types with
            | None -> None
            | Some typ ->
                // Check referenced types (simplified)
                match typ with
                | Named refName -> visit newVisited (typeName :: path) refName schema
                | _ -> None

    let detectCycle (schema: Schema) : string list option =
        Map.keys schema.Types
        |> Seq.tryPick (fun typeName -> visit Set.empty [] typeName schema)

let testSetOperations () =
    Console.writeln ""
    Console.writeln "=== Set Operations (Cycle Detection Pattern) ==="

    // Schema with no cycle
    let schema1 = SchemaOps.createSchema "Root"
                  |> SchemaOps.addType "Root" (Named "Person")
                  |> SchemaOps.addType "Person" (Primitive "string")

    match CycleDetection.detectCycle schema1 with
    | Some path ->
        Console.write "Unexpected cycle: "
        Console.writeln (String.concat " -> " path)
    | None ->
        Console.writeln "No cycle detected (correct)"

    // Schema with cycle: A -> B -> A
    let schema2 = SchemaOps.createSchema "A"
                  |> SchemaOps.addType "A" (Named "B")
                  |> SchemaOps.addType "B" (Named "A")

    match CycleDetection.detectCycle schema2 with
    | Some path ->
        Console.write "Cycle detected: "
        Console.writeln (String.concat " -> " path)
    | None ->
        Console.writeln "No cycle (unexpected)"

// ============================================================================
// PATTERN 3: List Operations (from Schema/Analysis.fs)
// ============================================================================

module SizeAnalysis =
    type SizeInfo = { Min: int; Max: int option; IsFixed: bool }

    /// Calculate struct size (uses List.map, List.sumBy, List.fold, List.forall)
    let calculateStructSize (fields: StructField list) : SizeInfo =
        // Simplified: each field is 4 bytes
        let fieldSizes = fields |> List.map (fun _ -> { Min = 4; Max = Some 4; IsFixed = true })
        let totalMin = fieldSizes |> List.sumBy (fun s -> s.Min)
        let totalMax =
            fieldSizes |> List.fold (fun acc size ->
                match acc, size.Max with
                | Some accMax, Some sizeMax -> Some (accMax + sizeMax)
                | _ -> None
            ) (Some 0)
        let isFixed = fieldSizes |> List.forall (fun s -> s.IsFixed)
        { Min = totalMin; Max = totalMax; IsFixed = isFixed }

    /// Check struct compatibility (uses List.length, List.forall2)
    let areStructsCompatible (fields1: StructField list) (fields2: StructField list) : bool =
        List.length fields1 = List.length fields2
        && List.forall2 (fun (f1: StructField) (f2: StructField) ->
            f1.Name = f2.Name
        ) fields1 fields2

let testListOperations () =
    Console.writeln ""
    Console.writeln "=== List Operations (Size Analysis Pattern) ==="

    let fields = [
        { Name = "id"; FieldType = Primitive "int" }
        { Name = "name"; FieldType = Primitive "string" }
        { Name = "age"; FieldType = Primitive "int" }
    ]

    let size = SizeAnalysis.calculateStructSize fields
    Console.write "Struct min size: "
    Console.writeln (Format.int size.Min)
    Console.write "Struct is fixed: "
    Console.writeln (if size.IsFixed then "true" else "false")

    // Test forall2
    let fields2 = [
        { Name = "id"; FieldType = Primitive "int" }
        { Name = "name"; FieldType = Primitive "string" }
        { Name = "age"; FieldType = Primitive "int" }
    ]
    let fields3 = [
        { Name = "id"; FieldType = Primitive "int" }
        { Name = "differentName"; FieldType = Primitive "string" }
    ]

    Console.write "fields1 compatible with fields2: "
    Console.writeln (if SizeAnalysis.areStructsCompatible fields fields2 then "true" else "false")

    Console.write "fields1 compatible with fields3: "
    Console.writeln (if SizeAnalysis.areStructsCompatible fields fields3 then "true" else "false")

    // Test List.exists and List.rev (from Validation.fs pathToString pattern)
    let path = [StructField "a"; UnionCase; StructField "b"]
    let hasUnionCase = List.exists (function UnionCase -> true | _ -> false) path
    Console.write "Path has UnionCase: "
    Console.writeln (if hasUnionCase then "true" else "false")

    let revPath = List.rev path
    Console.write "Reversed path length: "
    Console.writeln (Format.int (List.length revPath))

// ============================================================================
// PATTERN 4: Seq Operations (from Schema/Validation.fs, Analysis.fs)
// ============================================================================

module SeqPatterns =
    /// Get all referenced type names (uses Seq.collect, Seq.empty, Seq.append)
    let rec getReferencedTypes (schemaType: SchemaType) : seq<string> =
        match schemaType with
        | Primitive _ -> Seq.empty
        | Optional inner -> getReferencedTypes inner
        | Array inner -> getReferencedTypes inner
        | Map (k, v) -> Seq.append (getReferencedTypes k) (getReferencedTypes v)
        | Struct fields -> fields |> Seq.collect (fun f -> getReferencedTypes f.FieldType)
        | Union cases -> cases |> Map.values |> Seq.collect getReferencedTypes
        | Enum _ -> Seq.empty
        | Named name -> seq { yield name }

    /// Find max value size in union (uses Seq.map, Seq.max, Map.values)
    let getMaxUnionCaseSize (cases: Map<int, SchemaType>) : int =
        if Map.isEmpty cases then 0
        else
            cases
            |> Map.values
            |> Seq.map (fun _ -> 4)  // Simplified: each case is 4 bytes
            |> Seq.max

    /// Find min value in sequence (uses Seq.minBy)
    let getMinCaseSize (sizes: (string * int) list) : int =
        if List.isEmpty sizes then 0
        else
            sizes |> List.toSeq |> Seq.minBy snd |> snd

let testSeqOperations () =
    Console.writeln ""
    Console.writeln "=== Seq Operations (Validation Pattern) ==="

    let complexType = Struct [
        { Name = "person"; FieldType = Named "Person" }
        { Name = "address"; FieldType = Named "Address" }
        { Name = "tags"; FieldType = Array (Primitive "string") }
    ]

    let refs = SeqPatterns.getReferencedTypes complexType |> Seq.toList
    Console.write "Referenced types count: "
    Console.writeln (Format.int (List.length refs))

    // Test Map.values + Seq.max pattern
    let unionCases = Map.empty |> Map.add 0 (Primitive "int") |> Map.add 1 (Primitive "string")
    let maxSize = SeqPatterns.getMaxUnionCaseSize unionCases
    Console.write "Max union case size: "
    Console.writeln (Format.int maxSize)

    // Test Seq.minBy
    let sizes = [("a", 10); ("b", 5); ("c", 15)]
    let minSize = SeqPatterns.getMinCaseSize sizes
    Console.write "Min size: "
    Console.writeln (Format.int minSize)

// ============================================================================
// PATTERN 5: Array Operations (from Memory/Region.fs, Core/Uuid.fs)
// ============================================================================

module ArrayPatterns =
    /// Resize region (uses Array.zeroCreate, Array.blit)
    let resizeRegion (data: byte[]) (oldSize: int) (newSize: int) : byte[] =
        let newData = Array.zeroCreate newSize
        let copySize = min oldSize newSize
        Array.blit data 0 newData 0 copySize
        newData

    /// Initialize with pattern (uses Array.init)
    let initWithIndex (length: int) : int[] =
        Array.init length (fun i -> i * 2)

    /// Copy array (uses Array.copy)
    let copyData (data: byte[]) : byte[] =
        Array.copy data

let testArrayOperations () =
    Console.writeln ""
    Console.writeln "=== Array Operations (Memory Pattern) ==="

    // Test Array.zeroCreate + Array.blit (resize pattern)
    let original = [| 1uy; 2uy; 3uy; 4uy; 5uy |]
    let resized = ArrayPatterns.resizeRegion original 5 10
    Console.write "Resized array length: "
    Console.writeln (Format.int (Array.length resized))
    Console.write "Resized[0]: "
    Console.writeln (Format.int (int resized.[0]))
    Console.write "Resized[5] (should be 0): "
    Console.writeln (Format.int (int resized.[5]))

    // Test Array.init
    let indexed = ArrayPatterns.initWithIndex 5
    Console.write "Array.init result[2]: "
    Console.writeln (Format.int indexed.[2])

    // Test Array.copy
    let copied = ArrayPatterns.copyData original
    Console.write "Copied array length: "
    Console.writeln (Format.int (Array.length copied))

// ============================================================================
// PATTERN 6: Utility Functions (from Memory/View.fs)
// ============================================================================

module ViewPatterns =
    /// Get size and alignment (returns tuple, uses fst/snd)
    let getSizeAndAlignment (typ: SchemaType) : int * int =
        match typ with
        | Primitive "int" -> (4, 4)
        | Primitive "string" -> (8, 8)
        | _ -> (1, 1)

    /// Calculate max size from union cases (uses fst, snd, Seq.fold, max)
    let getUnionMaxSize (cases: Map<int, SchemaType>) : int * int =
        let maxSize =
            cases
            |> Map.values
            |> Seq.map (fun t -> fst (getSizeAndAlignment t))
            |> Seq.fold max 0
        let maxAlign =
            cases
            |> Map.values
            |> Seq.map (fun t -> snd (getSizeAndAlignment t))
            |> Seq.fold max 1
        (maxSize, maxAlign)

    /// Build field path (uses String.concat)
    let buildFieldPath (parts: string list) : string =
        String.concat "." parts

let testUtilityPatterns () =
    Console.writeln ""
    Console.writeln "=== Utility Patterns (View Pattern) ==="

    // Test fst/snd
    let sizeAlign = ViewPatterns.getSizeAndAlignment (Primitive "int")
    Console.write "Int size (fst): "
    Console.writeln (Format.int (fst sizeAlign))
    Console.write "Int align (snd): "
    Console.writeln (Format.int (snd sizeAlign))

    // Test max with Seq.fold
    let cases = Map.empty
                |> Map.add 0 (Primitive "int")
                |> Map.add 1 (Primitive "string")
    let (maxSz, maxAl) = ViewPatterns.getUnionMaxSize cases
    Console.write "Union max size: "
    Console.writeln (Format.int maxSz)
    Console.write "Union max align: "
    Console.writeln (Format.int maxAl)

    // Test String.concat
    let path = ViewPatterns.buildFieldPath ["root"; "person"; "name"]
    Console.write "Field path: "
    Console.writeln path

// ============================================================================
// PATTERN 7: Option.map (from Schema/Analysis.fs)
// ============================================================================

let testOptionMap () =
    Console.writeln ""
    Console.writeln "=== Option.map Pattern ==="

    // Pattern from Analysis.fs: innerSize.Max |> Option.map (fun m -> m + 1)
    let innerMax : int option = Some 10
    let adjusted = innerMax |> Option.map (fun m -> m + 1)

    match adjusted with
    | Some v ->
        Console.write "Some 10 |> Option.map (+1): Some "
        Console.writeln (Format.int v)
    | None ->
        Console.writeln "Unexpected None"

    let noneMax : int option = None
    let adjustedNone = noneMax |> Option.map (fun m -> m + 1)

    match adjustedNone with
    | Some _ -> Console.writeln "Unexpected Some"
    | None -> Console.writeln "None |> Option.map (+1): None (correct)"

// ============================================================================
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "================================================"
    Console.writeln "  PRD-13a: BAREWire Collection Patterns Test"
    Console.writeln "================================================"

    testMapOperations ()
    testSetOperations ()
    testListOperations ()
    testSeqOperations ()
    testArrayOperations ()
    testUtilityPatterns ()
    testOptionMap ()

    Console.writeln ""
    Console.writeln "================================================"
    Console.writeln "  All BAREWire patterns validated!"
    Console.writeln "================================================"
    0
