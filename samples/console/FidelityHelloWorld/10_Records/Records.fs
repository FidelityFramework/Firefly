/// Sample 10: Records
/// Demonstrates:
/// - Record type definition
/// - Record construction
/// - Field access
/// - Record copy-and-update syntax (with)
/// - Pattern matching on records
module RecordsSample

/// Simple person record
type Person = {
    Name: string
    Age: int
}

/// Address record
type Address = {
    Street: string
    City: string
    Zip: string
}

/// Nested record
type Contact = {
    Person: Person
    Address: Address
    Email: string
}

/// Create a greeting for a person
let greetPerson (p: Person) : string =
    $"Hello, {p.Name}! You are {Format.int p.Age} years old."

/// Birthday - returns person with age + 1
let birthday (p: Person) : Person =
    { p with Age = p.Age + 1 }

/// Format full contact info
let formatContact (c: Contact) : string =
    $"{c.Person.Name}, {c.Address.City} - {c.Email}"

/// Match on person age category
let ageCategory (p: Person) : string =
    match p with
    | { Age = a } when a < 18 -> "Minor"
    | { Age = a } when a < 65 -> "Adult"
    | _ -> "Senior"

[<EntryPoint>]
let main _ =
    Console.writeln "=== Records Test ==="

    // Create a person
    let alice = { Name = "Alice"; Age = 30 }
    Console.writeln (greetPerson alice)

    // Copy-and-update
    let olderAlice = birthday alice
    Console.write "After birthday: "
    Console.writeln (greetPerson olderAlice)

    // Field access
    Console.write "Name field: "
    Console.writeln alice.Name
    Console.write "Age field: "
    Console.writeln (Format.int alice.Age)

    // Age category
    Console.write "Category: "
    Console.writeln (ageCategory alice)

    // Nested records
    let contact = {
        Person = { Name = "Bob"; Age = 25 }
        Address = { Street = "123 Main St"; City = "Springfield"; Zip = "12345" }
        Email = "bob@example.com"
    }
    Console.write "Contact: "
    Console.writeln (formatContact contact)

    // Test different age categories
    Console.writeln ""
    Console.writeln "=== Age Categories ==="
    let child = { Name = "Charlie"; Age = 10 }
    let adult = { Name = "Diana"; Age = 35 }
    let senior = { Name = "Edward"; Age = 70 }

    Console.write "Age 10: "
    Console.writeln (ageCategory child)
    Console.write "Age 35: "
    Console.writeln (ageCategory adult)
    Console.write "Age 70: "
    Console.writeln (ageCategory senior)

    0
