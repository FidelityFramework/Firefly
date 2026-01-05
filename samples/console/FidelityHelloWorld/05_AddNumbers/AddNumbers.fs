/// Sample 05: Discriminated Union Pattern Matching
/// Demonstrates:
/// - Discriminated unions for runtime type representation
/// - Pattern matching for type-aware dispatch
module AddNumbers

// Discriminated union with two cases
type Number =
    | IntVal of int
    | FloatVal of float

// Simple 2-case pattern match
let formatNumber (n: Number) : string =
    match n with
    | IntVal x -> Format.int x
    | FloatVal x -> Format.float x

// Demo entry point
let runDemo () : int =
    Console.writeln "=== DU Pattern Match Test ==="

    // Test IntVal
    let a = IntVal 42
    Console.write "IntVal 42 -> "
    Console.writeln (formatNumber a)

    // Test FloatVal
    let b = FloatVal 3.14
    Console.write "FloatVal 3.14 -> "
    Console.writeln (formatNumber b)

    Console.writeln "Done!"
    0

[<EntryPoint>]
let main argv =
    runDemo ()
