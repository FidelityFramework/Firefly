/// Sample 08: Option Type
/// Demonstrates:
/// - Option<'T> type (Some / None)
/// - Pattern matching on Option
/// - Option.map, Option.bind
/// - Option.defaultValue
module OptionSample

/// Try to parse a string as an integer, returns None on failure
let tryParseInt (s: string) : int option =
    // Use Parse.tryInt which returns option
    Parse.tryInt s

/// Double a value if it exists
let doubleIfSome (opt: int option) : int option =
    match opt with
    | Some x -> Some (x * 2)
    | None -> None

/// Get value or default
let getOrDefault (defaultVal: int) (opt: int option) : int =
    match opt with
    | Some x -> x
    | None -> defaultVal

/// Chain operations using Option.map equivalent
let processNumber (input: string) : string =
    match tryParseInt input with
    | Some n ->
        let doubled = n * 2
        $"Doubled: {Format.int doubled}"
    | None ->
        "Invalid number"

[<EntryPoint>]
let main _ =
    Console.writeln "=== Option Type Test ==="

    // Test Some case
    let someVal = Some 42
    Console.write "Some 42 doubled: "
    match doubleIfSome someVal with
    | Some x -> Console.writeln (Format.int x)
    | None -> Console.writeln "None"

    // Test None case
    let noneVal : int option = None
    Console.write "None doubled: "
    match doubleIfSome noneVal with
    | Some x -> Console.writeln (Format.int x)
    | None -> Console.writeln "None"

    // Test getOrDefault
    Console.write "Some 10 or default 0: "
    Console.writeln (Format.int (getOrDefault 0 (Some 10)))
    Console.write "None or default 99: "
    Console.writeln (Format.int (getOrDefault 99 None))

    // Interactive parsing test
    Console.write "Enter a number: "
    let input = Console.readln ()
    Console.writeln (processNumber input)

    0
