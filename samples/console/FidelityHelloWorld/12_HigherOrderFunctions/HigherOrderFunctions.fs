/// Sample 11: Higher-Order Functions
/// Demonstrates:
/// - Functions as first-class values
/// - Passing functions as arguments
/// - Returning functions from functions
/// - Function composition
/// - Common HOF patterns (map, filter, fold concepts)
module HigherOrderFunctionsSample

/// Apply a function twice to a value
let applyTwice (f: int -> int) (x: int) : int =
    f (f x)

/// Apply a function to both elements of a pair
let mapPair (f: int -> int) (a: int) (b: int) : int * int =
    (f a, f b)

/// Choose which function to apply based on condition
let chooseAndApply (condition: bool) (onTrue: int -> int) (onFalse: int -> int) (x: int) : int =
    if condition then onTrue x else onFalse x

/// Create an adder function (returns a function)
let makeAdder (n: int) : (int -> int) =
    fun x -> x + n

/// Create a multiplier function
let makeMultiplier (n: int) : (int -> int) =
    fun x -> x * n

/// Compose two functions: (f >> g) = fun x -> g (f x)
let compose (f: int -> int) (g: int -> int) : (int -> int) =
    fun x -> g (f x)

/// Simple increment
let increment (x: int) : int = x + 1

/// Simple timesTwo
let timesTwo (x: int) : int = x * 2

/// Simple square
let square (x: int) : int = x * x

[<EntryPoint>]
let main _ =
    Console.writeln "=== Higher-Order Functions Test ==="

    // Apply a function twice
    Console.write "increment twice on 5: "
    Console.writeln (Format.int (applyTwice increment 5))  // 7

    Console.write "timesTwo twice on 3: "
    Console.writeln (Format.int (applyTwice timesTwo 3))  // 12

    // Map over pair
    Console.writeln ""
    Console.write "timesTwo both (3, 7): "
    let resultPair = mapPair timesTwo 3 7
    match resultPair with
    | (a, b) ->
        Console.write (Format.int a)
        Console.write ", "
        Console.writeln (Format.int b)

    // Conditional function application
    Console.writeln ""
    Console.write "chooseAndApply true timesTwo square 5: "
    Console.writeln (Format.int (chooseAndApply true timesTwo square 5))  // 10
    Console.write "chooseAndApply false timesTwo square 5: "
    Console.writeln (Format.int (chooseAndApply false timesTwo square 5))  // 25

    // Function factories
    Console.writeln ""
    let add10 = makeAdder 10
    let mult3 = makeMultiplier 3
    Console.write "add10 applied to 5: "
    Console.writeln (Format.int (add10 5))  // 15
    Console.write "mult3 applied to 7: "
    Console.writeln (Format.int (mult3 7))  // 21

    // Function composition
    Console.writeln ""
    let doubleThenIncrement = compose timesTwo increment
    let incrementThenDouble = compose increment timesTwo
    Console.write "timesTwo then increment on 5: "
    Console.writeln (Format.int (doubleThenIncrement 5))  // 11
    Console.write "increment then timesTwo on 5: "
    Console.writeln (Format.int (incrementThenDouble 5))  // 12

    0
