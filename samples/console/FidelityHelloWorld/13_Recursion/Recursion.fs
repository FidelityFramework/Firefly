/// Sample 13: Recursion
/// Demonstrates:
/// - Simple recursion (factorial)
/// - Tail recursion with accumulator
/// - Mutual recursion
/// - Recursive data structure traversal (conceptual)
module RecursionSample

/// Classic factorial - not tail recursive
let rec factorial (n: int) : int =
    if n <= 1 then 1
    else n * factorial (n - 1)

/// Tail-recursive factorial with accumulator
let factorialTail (n: int) : int =
    let rec loop acc n =
        if n <= 1 then acc
        else loop (acc * n) (n - 1)
    loop 1 n

/// Fibonacci - exponential time, demonstrates double recursion
let rec fibonacci (n: int) : int =
    if n <= 1 then n
    else fibonacci (n - 1) + fibonacci (n - 2)

/// Tail-recursive fibonacci
let fibonacciTail (n: int) : int =
    let rec loop a b count =
        if count <= 0 then a
        else loop b (a + b) (count - 1)
    loop 0 1 n

/// Sum from 1 to n - tail recursive
let sumTo (n: int) : int =
    let rec loop acc i =
        if i > n then acc
        else loop (acc + i) (i + 1)
    loop 0 1

/// Count digits in a number
let rec countDigits (n: int) : int =
    if n < 10 then 1
    else 1 + countDigits (n / 10)

/// GCD using Euclidean algorithm - naturally tail recursive
let rec gcd (a: int) (b: int) : int =
    if b = 0 then a
    else gcd b (a % b)

/// Power function - a^n
let rec power (a: int) (n: int) : int =
    if n = 0 then 1
    elif n = 1 then a
    else a * power a (n - 1)

/// Mutual recursion example: even/odd determination
let rec isEven (n: int) : bool =
    if n = 0 then true
    else isOdd (n - 1)
and isOdd (n: int) : bool =
    if n = 0 then false
    else isEven (n - 1)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Recursion Test ==="

    // Factorial
    Console.writeln "--- Factorial ---"
    Console.write "factorial 5: "
    Console.writeln (Format.int (factorial 5))  // 120
    Console.write "factorialTail 5: "
    Console.writeln (Format.int (factorialTail 5))  // 120
    Console.write "factorial 10: "
    Console.writeln (Format.int (factorial 10))  // 3628800

    // Fibonacci
    Console.writeln ""
    Console.writeln "--- Fibonacci ---"
    Console.write "fibonacci 10: "
    Console.writeln (Format.int (fibonacci 10))  // 55
    Console.write "fibonacciTail 10: "
    Console.writeln (Format.int (fibonacciTail 10))  // 55
    Console.write "fibonacciTail 20: "
    Console.writeln (Format.int (fibonacciTail 20))  // 6765

    // Sum
    Console.writeln ""
    Console.writeln "--- Sum ---"
    Console.write "sum 1 to 10: "
    Console.writeln (Format.int (sumTo 10))  // 55
    Console.write "sum 1 to 100: "
    Console.writeln (Format.int (sumTo 100))  // 5050

    // Count digits
    Console.writeln ""
    Console.writeln "--- Count Digits ---"
    Console.write "digits in 12345: "
    Console.writeln (Format.int (countDigits 12345))  // 5
    Console.write "digits in 7: "
    Console.writeln (Format.int (countDigits 7))  // 1

    // GCD
    Console.writeln ""
    Console.writeln "--- GCD ---"
    Console.write "gcd 48 18: "
    Console.writeln (Format.int (gcd 48 18))  // 6
    Console.write "gcd 100 35: "
    Console.writeln (Format.int (gcd 100 35))  // 5

    // Power
    Console.writeln ""
    Console.writeln "--- Power ---"
    Console.write "2^10: "
    Console.writeln (Format.int (power 2 10))  // 1024
    Console.write "3^5: "
    Console.writeln (Format.int (power 3 5))  // 243

    // Mutual recursion
    Console.writeln ""
    Console.writeln "--- Mutual Recursion (Even/Odd) ---"
    Console.write "isEven 10: "
    Console.writeln (if isEven 10 then "true" else "false")
    Console.write "isOdd 10: "
    Console.writeln (if isOdd 10 then "true" else "false")
    Console.write "isEven 7: "
    Console.writeln (if isEven 7 then "true" else "false")
    Console.write "isOdd 7: "
    Console.writeln (if isOdd 7 then "true" else "false")

    0
