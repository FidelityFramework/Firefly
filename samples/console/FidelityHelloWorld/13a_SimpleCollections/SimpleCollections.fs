/// Sample 13a_SimpleCollections: Full PRD-13a Feature Coverage
/// Demonstrates all collection types and operations for regression testing.
///
/// Coverage:
/// - List: empty, cons, head, tail, length, map, filter, fold, exists, forall, rev, contains, isEmpty
/// - Map: empty, add, tryFind, containsKey, remove, values, keys, isEmpty, forall
/// - Set: empty, add, contains, remove, union, intersect
/// - Range expressions: [1..5], [1..2..10], [|1..5|]
/// - Utilities: Option.map, fst, snd, max, min
module SimpleCollections

// ============================================================================
// SECTION 1: List Operations
// ============================================================================

/// Test list creation and basic operations
let testListBasics () =
    Console.writeln "=== List Basics ==="

    // Empty list
    let empty : int list = []
    Console.write "Empty list length: "
    Console.writeln (Format.int (List.length empty))

    // List literal
    let nums = [1; 2; 3; 4; 5]
    Console.write "List [1;2;3;4;5] length: "
    Console.writeln (Format.int (List.length nums))

    // Head and tail
    let h = List.head nums
    Console.write "Head: "
    Console.writeln (Format.int h)

    let t = List.tail nums
    Console.write "Tail length: "
    Console.writeln (Format.int (List.length t))

    // Cons operator
    let extended = 0 :: nums
    Console.write "After cons 0: length = "
    Console.writeln (Format.int (List.length extended))

/// Test list transformations
let testListTransformations () =
    Console.writeln ""
    Console.writeln "=== List Transformations ==="

    let nums = [1; 2; 3; 4; 5]

    // Map
    let doubled = List.map (fun x -> x * 2) nums
    Console.write "Doubled first: "
    Console.writeln (Format.int (List.head doubled))

    // Filter
    let evens = List.filter (fun x -> x % 2 = 0) nums
    Console.write "Evens count: "
    Console.writeln (Format.int (List.length evens))

    // Fold
    let sum = List.fold (fun acc x -> acc + x) 0 nums
    Console.write "Sum via fold: "
    Console.writeln (Format.int sum)

    // Rev
    let reversed = List.rev nums
    Console.write "Reversed first: "
    Console.writeln (Format.int (List.head reversed))

/// Test list predicates
let testListPredicates () =
    Console.writeln ""
    Console.writeln "=== List Predicates ==="

    let nums = [1; 2; 3; 4; 5]
    let empty : int list = []

    // isEmpty
    Console.write "Empty isEmpty: "
    Console.writeln (if List.isEmpty empty then "true" else "false")

    Console.write "Nums isEmpty: "
    Console.writeln (if List.isEmpty nums then "true" else "false")

    // contains
    Console.write "Contains 3: "
    Console.writeln (if List.contains 3 nums then "true" else "false")

    Console.write "Contains 99: "
    Console.writeln (if List.contains 99 nums then "true" else "false")

    // exists
    Console.write "Exists >4: "
    Console.writeln (if List.exists (fun x -> x > 4) nums then "true" else "false")

    // forall
    Console.write "Forall >0: "
    Console.writeln (if List.forall (fun x -> x > 0) nums then "true" else "false")

// ============================================================================
// SECTION 2: Map Operations
// ============================================================================

/// Test map creation and basic operations
let testMapBasics () =
    Console.writeln ""
    Console.writeln "=== Map Basics ==="

    // Empty map
    let empty = Map.empty<string, int>
    Console.write "Empty map isEmpty: "
    Console.writeln (if Map.isEmpty empty then "true" else "false")

    // Add entries
    let m1 = Map.add "one" 1 empty
    let m2 = Map.add "two" 2 m1
    let m3 = Map.add "three" 3 m2

    Console.write "After 3 adds, isEmpty: "
    Console.writeln (if Map.isEmpty m3 then "true" else "false")

    // tryFind
    match Map.tryFind "two" m3 with
    | Some v ->
        Console.write "Found 'two': "
        Console.writeln (Format.int v)
    | None ->
        Console.writeln "Not found"

    match Map.tryFind "four" m3 with
    | Some _ -> Console.writeln "Unexpectedly found 'four'"
    | None -> Console.writeln "Correctly did not find 'four'"

    // containsKey
    Console.write "ContainsKey 'one': "
    Console.writeln (if Map.containsKey "one" m3 then "true" else "false")

    Console.write "ContainsKey 'zero': "
    Console.writeln (if Map.containsKey "zero" m3 then "true" else "false")

/// Test map iteration operations
let testMapIteration () =
    Console.writeln ""
    Console.writeln "=== Map Iteration ==="

    let m = Map.empty<string, int>
            |> Map.add "a" 1
            |> Map.add "b" 2
            |> Map.add "c" 3

    // values - returns seq, convert to list for testing
    let vals = Map.values m |> Seq.toList
    Console.write "Values count: "
    Console.writeln (Format.int (List.length vals))

    // keys - returns seq, convert to list for testing
    let ks = Map.keys m |> Seq.toList
    Console.write "Keys count: "
    Console.writeln (Format.int (List.length ks))

    // forall
    let allPositive = Map.forall (fun _ v -> v > 0) m
    Console.write "All values > 0: "
    Console.writeln (if allPositive then "true" else "false")

// ============================================================================
// SECTION 3: Set Operations
// ============================================================================

/// Test set creation and basic operations
let testSetBasics () =
    Console.writeln ""
    Console.writeln "=== Set Basics ==="

    // Empty set
    let empty = Set.empty<int>
    Console.write "Empty set count: "
    Console.writeln (Format.int (Set.count empty))

    // Add elements
    let s1 = Set.add 1 empty
    let s2 = Set.add 2 s1
    let s3 = Set.add 3 s2
    let s3dup = Set.add 2 s3  // Adding duplicate

    Console.write "After adding 1,2,3: count = "
    Console.writeln (Format.int (Set.count s3))

    Console.write "After adding dup 2: count = "
    Console.writeln (Format.int (Set.count s3dup))

    // contains
    Console.write "Contains 2: "
    Console.writeln (if Set.contains 2 s3 then "true" else "false")

    Console.write "Contains 99: "
    Console.writeln (if Set.contains 99 s3 then "true" else "false")

/// Test set operations (union, intersect)
let testSetOperations () =
    Console.writeln ""
    Console.writeln "=== Set Operations ==="

    let setA = Set.empty |> Set.add 1 |> Set.add 2 |> Set.add 3
    let setB = Set.empty |> Set.add 2 |> Set.add 3 |> Set.add 4

    // Union
    let unionSet = Set.union setA setB
    Console.write "Union {1,2,3} {2,3,4} count: "
    Console.writeln (Format.int (Set.count unionSet))

    // Intersect
    let intersectSet = Set.intersect setA setB
    Console.write "Intersect {1,2,3} {2,3,4} count: "
    Console.writeln (Format.int (Set.count intersectSet))

    // Remove
    let removed = Set.remove 2 setA
    Console.write "After removing 2 from {1,2,3}: count = "
    Console.writeln (Format.int (Set.count removed))

// ============================================================================
// SECTION 4: Range Expressions
// ============================================================================

/// Test range expressions
let testRangeExpressions () =
    Console.writeln ""
    Console.writeln "=== Range Expressions ==="

    // Simple list range [start..end]
    let range1 = [1..5]
    Console.write "[1..5] length: "
    Console.writeln (Format.int (List.length range1))
    Console.write "[1..5] first: "
    Console.writeln (Format.int (List.head range1))

    // List range with step [start..step..end]
    let range2 = [1..2..10]
    Console.write "[1..2..10] length: "
    Console.writeln (Format.int (List.length range2))
    Console.write "[1..2..10] first: "
    Console.writeln (Format.int (List.head range2))

    // Array range [|start..end|]
    let arrRange = [|1..5|]
    Console.write "[|1..5|] length: "
    Console.writeln (Format.int (Array.length arrRange))

    // Array range with step [|start..step..end|]
    let arrRange2 = [|0..10..50|]
    Console.write "[|0..10..50|] length: "
    Console.writeln (Format.int (Array.length arrRange2))

    // Descending range
    let descRange = [5..-1..1]
    Console.write "[5..-1..1] length: "
    Console.writeln (Format.int (List.length descRange))
    Console.write "[5..-1..1] first: "
    Console.writeln (Format.int (List.head descRange))

// ============================================================================
// SECTION 5: Utility Functions
// ============================================================================

/// Test tuple accessors and comparison functions
let testUtilities () =
    Console.writeln ""
    Console.writeln "=== Utilities ==="

    // fst and snd
    let pair = (42, "hello")
    Console.write "fst (42, \"hello\"): "
    Console.writeln (Format.int (fst pair))
    Console.write "snd (42, \"hello\"): "
    Console.writeln (snd pair)

    // max and min
    Console.write "max 10 20: "
    Console.writeln (Format.int (max 10 20))
    Console.write "min 10 20: "
    Console.writeln (Format.int (min 10 20))

    // Option.map
    let someVal = Some 5
    let mapped = Option.map (fun x -> x * 2) someVal
    match mapped with
    | Some v ->
        Console.write "Option.map (*2) (Some 5): Some "
        Console.writeln (Format.int v)
    | None ->
        Console.writeln "Unexpected None"

    let noneVal : int option = None
    let mappedNone = Option.map (fun x -> x * 2) noneVal
    match mappedNone with
    | Some _ -> Console.writeln "Unexpected Some"
    | None -> Console.writeln "Option.map on None: None (correct)"

// ============================================================================
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "============================================"
    Console.writeln "  PRD-13a: Simple Collections Test Suite"
    Console.writeln "============================================"

    testListBasics ()
    testListTransformations ()
    testListPredicates ()
    testMapBasics ()
    testMapIteration ()
    testSetBasics ()
    testSetOperations ()
    testRangeExpressions ()
    testUtilities ()

    Console.writeln ""
    Console.writeln "============================================"
    Console.writeln "  All tests completed!"
    Console.writeln "============================================"
    0
