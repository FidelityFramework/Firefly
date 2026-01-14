module Examples.HelloWorldFullCurried

/// Demonstrates CURRIED function patterns:
/// - Curried function with multiple parameters
/// - Direct curried call (all args at once)
/// - Sequential curried call (one arg at a time)
/// - Pipe operator: `x |> f`

/// Curried greeting function - takes prefix then name
let greet prefix name =
    Console.writeln $"{prefix}, {name}!"

/// Hello function reads name and pipes to greeter
let hello prefix =
    Console.write "Enter your name: "
    Console.readln()
    |> greet prefix

[<EntryPoint>]
let main argv =
    // Direct curried call with both arguments
    greet "Hello" "World"

    // Interactive: read name and pipe to greeter
    hello "Welcome"

    0
