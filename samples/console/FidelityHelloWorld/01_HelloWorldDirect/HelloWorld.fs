/// HelloWorld - Direct Module Calls Pattern (Minimal Version)
/// Tests the basic compilation pipeline with static string output only
module Examples.HelloWorldDirect

[<EntryPoint>]
let main argv =
    // Simple static string output - no input, no variables
    // Uses FNCS Console intrinsics
    Console.write "Hello, World!"
    Console.writeln ""
    0
