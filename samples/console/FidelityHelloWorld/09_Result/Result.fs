/// Sample 09: Result Type
/// Demonstrates:
/// - Result<'T, 'E> type (Ok / Error)
/// - Pattern matching on Result
/// - Error propagation patterns
/// - Result.map, Result.bind equivalent patterns
module ResultSample

/// Validation error type
type ValidationError =
    | TooShort
    | TooLong
    | InvalidFormat

/// Format error for display
let formatError (err: ValidationError) : string =
    match err with
    | TooShort -> "Input too short"
    | TooLong -> "Input too long"
    | InvalidFormat -> "Invalid format"

/// Validate string length (3-10 characters)
let validateLength (s: string) : Result<string, ValidationError> =
    let len = String.length s
    if len < 3 then
        Error TooShort
    elif len > 10 then
        Error TooLong
    else
        Ok s

/// Validate string contains no spaces
let validateNoSpaces (s: string) : Result<string, ValidationError> =
    if String.contains s ' ' then
        Error InvalidFormat
    else
        Ok s

/// Chain validations - both must pass
let validateInput (input: string) : Result<string, ValidationError> =
    match validateLength input with
    | Error e -> Error e
    | Ok s ->
        match validateNoSpaces s with
        | Error e -> Error e
        | Ok validated -> Ok validated

/// Process valid input
let processValid (s: string) : string =
    $"Valid input: {s} (length: {Format.int (String.length s)})"

[<EntryPoint>]
let main _ =
    Console.writeln "=== Result Type Test ==="

    // Test Ok case
    let okVal : Result<int, string> = Ok 42
    Console.write "Ok 42: "
    match okVal with
    | Ok x -> Console.writeln (Format.int x)
    | Error e -> Console.writeln e

    // Test Error case
    let errVal : Result<int, string> = Error "Something went wrong"
    Console.write "Error case: "
    match errVal with
    | Ok x -> Console.writeln (Format.int x)
    | Error e -> Console.writeln e

    Console.writeln ""
    Console.writeln "=== Validation Chain Test ==="

    // Valid input
    Console.write "Testing 'hello': "
    match validateInput "hello" with
    | Ok s -> Console.writeln (processValid s)
    | Error e -> Console.writeln (formatError e)

    // Too short
    Console.write "Testing 'ab': "
    match validateInput "ab" with
    | Ok s -> Console.writeln (processValid s)
    | Error e -> Console.writeln (formatError e)

    // Too long
    Console.write "Testing 'verylongstring': "
    match validateInput "verylongstring" with
    | Ok s -> Console.writeln (processValid s)
    | Error e -> Console.writeln (formatError e)

    // Has spaces
    Console.write "Testing 'has space': "
    match validateInput "has space" with
    | Ok s -> Console.writeln (processValid s)
    | Error e -> Console.writeln (formatError e)

    // Interactive test
    Console.writeln ""
    Console.write "Enter a string to validate: "
    let input = Console.readln ()
    match validateInput input with
    | Ok s -> Console.writeln (processValid s)
    | Error e -> Console.writeln (formatError e)

    0
