/// Sample 11: Closures
/// Demonstrates:
/// - Lambdas that capture variables from enclosing scope
/// - Mutable captured variables (arena-allocated for escape safety)
/// - Counter pattern
/// - Closure as state encapsulation
///
/// ARCHITECTURAL NOTE (No Runtime):
/// Mutable captures that escape their defining scope need storage that
/// outlives that scope. With no runtime/GC, we use arena allocation:
/// - main creates an arena on its stack
/// - Factory functions receive the arena and allocate from it
/// - All closure state lives as long as main's stack frame
module ClosuresSample

/// Create a counter that increments each time called
/// Arena parameter allows mutable state to escape safely
let makeCounter (arena: byref<Arena<'a>>) (start: int) : (unit -> int) =
    // Allocate count in arena (not on makeCounter's stack)
    // This storage outlives makeCounter's return
    // Use Platform.wordSize() for platform-portable allocation (8 on 64-bit, 4 on 32-bit)
    let countPtr = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> countPtr) start
    fun () ->
        let ptr = NativePtr.ofNativeInt<int> countPtr
        let current = NativePtr.read ptr
        let next = current + 1
        NativePtr.write ptr next
        next

/// Create a greeting function that captures a name (immutable - no arena needed)
let makeGreeter (name: string) : (string -> string) =
    fun greeting -> $"{greeting}, {name}!"

/// Create an accumulator that adds to running total
/// Arena parameter for mutable state
let makeAccumulator (arena: byref<Arena<'a>>) (initial: int) : (int -> int) =
    // Use Platform.wordSize() for platform-portable allocation
    let totalPtr = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> totalPtr) initial
    fun n ->
        let ptr = NativePtr.ofNativeInt<int> totalPtr
        let current = NativePtr.read ptr
        let next = current + n
        NativePtr.write ptr next
        next

/// Capture multiple values in a closure (immutable - no arena needed)
let makeRangeChecker (min: int) (max: int) : (int -> bool) =
    fun x -> x >= min && x <= max

[<EntryPoint>]
let main _ =
    // Create arena on main's stack - all closure state lives here
    let arenaMem = NativePtr.stackalloc<byte> 4096
    let mutable arena = Arena.fromPointer (NativePtr.toNativeInt arenaMem) 4096

    Console.writeln "=== Closures Test ==="

    // Counter closure (uses arena for mutable state)
    Console.writeln "--- Counter ---"
    let counter = makeCounter &arena 0
    Console.write "First call: "
    Console.writeln (Format.int (counter ()))  // 1
    Console.write "Second call: "
    Console.writeln (Format.int (counter ()))  // 2
    Console.write "Third call: "
    Console.writeln (Format.int (counter ()))  // 3

    // Greeter closure (immutable capture - no arena)
    Console.writeln ""
    Console.writeln "--- Greeter ---"
    let greetAlice = makeGreeter "Alice"
    let greetBob = makeGreeter "Bob"
    Console.writeln (greetAlice "Hello")
    Console.writeln (greetAlice "Goodbye")
    Console.writeln (greetBob "Welcome")

    // Accumulator closure (uses arena for mutable state)
    Console.writeln ""
    Console.writeln "--- Accumulator ---"
    let acc = makeAccumulator &arena 100
    Console.write "Add 10: "
    Console.writeln (Format.int (acc 10))  // 110
    Console.write "Add 25: "
    Console.writeln (Format.int (acc 25))  // 135
    Console.write "Add 5: "
    Console.writeln (Format.int (acc 5))   // 140

    // Range checker closure (immutable capture - no arena)
    Console.writeln ""
    Console.writeln "--- Range Checker ---"
    let inRange = makeRangeChecker 10 20
    Console.write "5 in range 10-20: "
    Console.writeln (if inRange 5 then "true" else "false")
    Console.write "15 in range 10-20: "
    Console.writeln (if inRange 15 then "true" else "false")
    Console.write "25 in range 10-20: "
    Console.writeln (if inRange 25 then "true" else "false")

    // Multiple independent closures from same factory
    Console.writeln ""
    Console.writeln "--- Independent Closures ---"
    let counter1 = makeCounter &arena 0
    let counter2 = makeCounter &arena 100
    Console.write "counter1: "
    Console.writeln (Format.int (counter1 ()))  // 1
    Console.write "counter2: "
    Console.writeln (Format.int (counter2 ()))  // 101
    Console.write "counter1: "
    Console.writeln (Format.int (counter1 ()))  // 2
    Console.write "counter2: "
    Console.writeln (Format.int (counter2 ()))  // 102

    0
