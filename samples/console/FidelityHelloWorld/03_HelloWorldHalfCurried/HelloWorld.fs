module Examples.HelloWorldHalfCurried

/// Demonstrates HALF-CURRIED patterns:
/// - Pipe operator: `x |> f`
/// - Function composition with pipes
/// Uses a helper function to format the greeting
let greet (name: string) : unit =
    Console.writeln $"Hello, {name}!"

let hello() =
    Console.write "Enter your name: "

    Console.readln()
    |> greet

[<EntryPoint>]
let main argv =
    hello()
    0
