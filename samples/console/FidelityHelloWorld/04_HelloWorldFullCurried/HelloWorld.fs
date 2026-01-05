module Examples.HelloWorldFullCurried

/// Demonstrates FULL-CURRIED patterns:
/// - Curried function with multiple parameters
/// - Pipe operator: `x |> f`
/// - Lambda: `fun name -> ...`
/// - Higher-order function composition
/// - Pattern matching on arrays

/// Curried greeting function - takes prefix then name
let greet prefix name =
    Console.writeln $"{prefix}, {name}!"

/// Hello function partially applies greet
let hello prefix =
    Console.write "Enter your name: "
    Console.readln()
    |> greet prefix

[<EntryPoint>]
let main argv =
    match argv with
    | [|prefix|] -> hello prefix
    | _ -> hello "Hello"
    0
