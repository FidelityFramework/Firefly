module Examples.HelloWorldSaturated

/// Demonstrates SATURATED function calls - all arguments provided at once.
/// Uses FNCS Console intrinsics.
let hello() =
    Console.write "Enter your name: "
    let name = Console.readln()
    Console.writeln $"Hello, {name}!"

[<EntryPoint>]
let main argv =
    hello()
    0
