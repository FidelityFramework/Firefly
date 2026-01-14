/// Sample 12: Closures
/// Demonstrates:
/// - Lambdas that capture variables from enclosing scope
/// - Mutable captured variables
/// - Counter pattern
/// - Closure as state encapsulation
module ClosuresSample

/// Create a counter that increments each time called
/// Returns a function that captures mutable state
let makeCounter (start: int) : (unit -> int) =
    let mutable count = start
    fun () ->
        count <- count + 1
        count

/// Create a greeting function that captures a name
let makeGreeter (name: string) : (string -> string) =
    fun greeting -> $"{greeting}, {name}!"

/// Create an accumulator that adds to running total
let makeAccumulator (initial: int) : (int -> int) =
    let mutable total = initial
    fun n ->
        total <- total + n
        total

/// Capture multiple values in a closure
let makeRangeChecker (min: int) (max: int) : (int -> bool) =
    fun x -> x >= min && x <= max

[<EntryPoint>]
let main _ =
    Console.writeln "=== Closures Test ==="

    // Counter closure
    Console.writeln "--- Counter ---"
    let counter = makeCounter 0
    Console.write "First call: "
    Console.writeln (Format.int (counter ()))  // 1
    Console.write "Second call: "
    Console.writeln (Format.int (counter ()))  // 2
    Console.write "Third call: "
    Console.writeln (Format.int (counter ()))  // 3

    // Greeter closure
    Console.writeln ""
    Console.writeln "--- Greeter ---"
    let greetAlice = makeGreeter "Alice"
    let greetBob = makeGreeter "Bob"
    Console.writeln (greetAlice "Hello")
    Console.writeln (greetAlice "Goodbye")
    Console.writeln (greetBob "Welcome")

    // Accumulator closure
    Console.writeln ""
    Console.writeln "--- Accumulator ---"
    let acc = makeAccumulator 100
    Console.write "Add 10: "
    Console.writeln (Format.int (acc 10))  // 110
    Console.write "Add 25: "
    Console.writeln (Format.int (acc 25))  // 135
    Console.write "Add 5: "
    Console.writeln (Format.int (acc 5))   // 140

    // Range checker closure
    Console.writeln ""
    Console.writeln "--- Range Checker ---"
    let inRange = makeRangeChecker 10 20
    Console.write "5 in range 10-20: "
    Console.writeln (if inRange 5 then "true" else "false")
    Console.write "15 in range 10-20: "
    Console.writeln (if inRange 15 then "true" else "false")
    Console.write "25 in range 10-20: "
    Console.writeln (if inRange 25 then "true" else "false")

    // Multiple independent closures from same factory
    Console.writeln ""
    Console.writeln "--- Independent Closures ---"
    let counter1 = makeCounter 0
    let counter2 = makeCounter 100
    Console.write "counter1: "
    Console.writeln (Format.int (counter1 ()))  // 1
    Console.write "counter2: "
    Console.writeln (Format.int (counter2 ()))  // 101
    Console.write "counter1: "
    Console.writeln (Format.int (counter1 ()))  // 2
    Console.write "counter2: "
    Console.writeln (Format.int (counter2 ()))  // 102

    0
