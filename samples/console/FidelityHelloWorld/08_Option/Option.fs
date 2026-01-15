/// Sample 08: Option Type - Simplified test
module OptionSample

[<EntryPoint>]
let main argv =
    Console.writeln "=== Option Type Test ==="

    // Test Some case
    let someVal = Some 42
    Console.write "Some 42: "
    match someVal with
    | Some x -> Console.writeln (Format.int x)
    | None -> Console.writeln "None"

    Console.writeln "Done!"
    0
