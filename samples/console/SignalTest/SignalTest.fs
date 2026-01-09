/// SignalTest - Comprehensive validation of Fidelity.Signal reactive primitives
/// Tests: Signal, Effect, Memo, Batch - the full reactive signal infrastructure
///
/// Design: Per FNCS spec (reactive-signals.md Section 2.2), FnPtr.ofFunction only
/// supports top-level functions without closures. All signals that effect/memo
/// functions depend on must be at module level.
module Examples.SignalTest

// ============================================================================
// Test Utilities
// ============================================================================

let mutable passCount = 0
let mutable failCount = 0

let pass (name: string) =
    Console.write name
    Console.writeln " PASS"
    passCount <- passCount + 1

let fail (name: string) (expected: string) (actual: string) =
    Console.write name
    Console.write " FAIL - expected "
    Console.write expected
    Console.write ", got "
    Console.writeln actual
    failCount <- failCount + 1

let section (name: string) =
    Console.writeln ""
    Console.write "--- "
    Console.write name
    Console.writeln " ---"

// ============================================================================
// Signal Basic Tests
// ============================================================================

let testSignalBasics () =
    section "Signal Basics"

    // Test: Create signal with int, verify initial value
    let s1 = Signal.create 42
    let v1 = Signal.get s1
    if v1 = 42 then
        pass "Create int signal"
    else
        fail "Create int signal" "42" (Format.int v1)

    // Test: Set signal value
    Signal.set s1 100
    let v2 = Signal.get s1
    if v2 = 100 then
        pass "Set signal value"
    else
        fail "Set signal value" "100" (Format.int v2)

    // Test: Multiple signals are independent
    let s2 = Signal.create 10
    let s3 = Signal.create 20
    Signal.set s2 15
    let v3a = Signal.get s2
    let v3b = Signal.get s3
    if v3a = 15 && v3b = 20 then
        pass "Multiple signals independent"
    else
        fail "Multiple signals independent" "15,20" "mismatch"

    // Test: Same value doesn't change (for later effect testing)
    let beforeSet = Signal.get s1
    Signal.set s1 beforeSet  // Set to same value
    let afterSet = Signal.get s1
    if afterSet = beforeSet then
        pass "Same value preserved"
    else
        fail "Same value preserved" (Format.int beforeSet) (Format.int afterSet)

// ============================================================================
// Signal Type Tests
// ============================================================================

let testSignalTypes () =
    section "Signal Types"

    // Test: Int signal (4 bytes)
    let intSig = Signal.create 12345
    let intVal = Signal.get intSig
    if intVal = 12345 then
        pass "Int signal (i32)"
    else
        fail "Int signal (i32)" "12345" (Format.int intVal)

    // Test: Int64 signal (8 bytes)
    let int64Sig = Signal.create 9876543210L
    let int64Val = Signal.get int64Sig
    if int64Val = 9876543210L then
        pass "Int64 signal (i64)"
    else
        fail "Int64 signal (i64)" "9876543210" "mismatch"

    // Test: Bool signal
    let boolSig = Signal.create true
    let boolVal = Signal.get boolSig
    if boolVal then
        pass "Bool signal"
    else
        fail "Bool signal" "true" "false"

    // Test: Float signal
    let floatSig = Signal.create 3.14159
    let floatVal = Signal.get floatSig
    // Float comparison with tolerance
    if floatVal > 3.14 && floatVal < 3.15 then
        pass "Float signal (f64)"
    else
        fail "Float signal (f64)" "~3.14159" "mismatch"

// ============================================================================
// Effect Tests - Module Level State
// Per FNCS spec: Effect functions must be top-level (no closures)
// ============================================================================

let mutable effectRunCount = 0

// Module-level signal for effect test
// Note: Mutable so it becomes an LLVM global, accessible from top-level functions
// Signal<int> is internally an i32 (slot index), stored in global
let mutable effectCounter = Signal.create 0

// Top-level effect function (no closures - accesses module-level signal)
let incrementEffect () =
    let _ = Signal.get effectCounter  // Establish dependency
    effectRunCount <- effectRunCount + 1

let testEffects () =
    section "Effects"

    // Reset state
    effectRunCount <- 0
    Signal.set effectCounter 0

    // Test: Effect runs immediately on creation
    let eff = Effect.create (FnPtr.ofFunction incrementEffect)

    if effectRunCount = 1 then
        pass "Effect runs on creation"
    else
        fail "Effect runs on creation" "1" (Format.int effectRunCount)

    // Test: Effect re-runs when signal changes
    Signal.set effectCounter 1
    if effectRunCount = 2 then
        pass "Effect re-runs on signal change"
    else
        fail "Effect re-runs on signal change" "2" (Format.int effectRunCount)

    // Test: Effect runs again on another change
    Signal.set effectCounter 2
    if effectRunCount = 3 then
        pass "Effect tracks ongoing changes"
    else
        fail "Effect tracks ongoing changes" "3" (Format.int effectRunCount)

// ============================================================================
// Memo Tests - Module Level State
// Per FNCS spec: Memo functions must be top-level (no closures)
// ============================================================================

let mutable memoComputeCount = 0

// Module-level signal for memo test
// Mutable so it becomes an LLVM global accessible from top-level functions
let mutable memoSource = Signal.create 10

// Top-level memo computation function (no closures)
let doubleIt () =
    memoComputeCount <- memoComputeCount + 1
    let v = Signal.get memoSource
    v * 2

let testMemos () =
    section "Memos"

    // Reset state
    memoComputeCount <- 0
    Signal.set memoSource 10

    let doubled = Memo.create (FnPtr.ofFunction doubleIt)

    // Test: Memo computes initial value
    let m1 = Memo.get doubled
    if m1 = 20 && memoComputeCount = 1 then
        pass "Memo computes initial value"
    else
        fail "Memo computes initial value" "20, count=1" "mismatch"

    // Test: Memo caches (no recompute on second get)
    let m2 = Memo.get doubled
    if m2 = 20 && memoComputeCount = 1 then
        pass "Memo caches value"
    else
        fail "Memo caches value" "20, count=1" "mismatch"

    // Test: Memo recomputes when dependency changes
    Signal.set memoSource 25
    let m3 = Memo.get doubled
    if m3 = 50 && memoComputeCount = 2 then
        pass "Memo recomputes on dependency change"
    else
        fail "Memo recomputes on dependency change" "50, count=2" "mismatch"

// ============================================================================
// Batch Tests - Module Level State
// Per FNCS spec: Batch/Effect functions must be top-level (no closures)
// ============================================================================

let mutable batchEffectRunCount = 0

// Module-level signals for batch test
// Mutable so they become LLVM globals accessible from top-level functions
let mutable batchA = Signal.create 1
let mutable batchB = Signal.create 2

// Top-level effect function for batch test (no closures)
let sumEffect () =
    let va = Signal.get batchA
    let vb = Signal.get batchB
    batchEffectRunCount <- batchEffectRunCount + 1

// Top-level batch function (no closures)
let batchFn () =
    Signal.set batchA 10
    Signal.set batchB 20

let testBatch () =
    section "Batch"

    // Reset state
    batchEffectRunCount <- 0
    Signal.set batchA 1
    Signal.set batchB 2

    let eff = Effect.create (FnPtr.ofFunction sumEffect)

    // Effect ran once on creation
    let countAfterCreate = batchEffectRunCount

    // Test: Batch groups multiple updates
    Batch.run (FnPtr.ofFunction batchFn)

    // Should have run exactly once more (not twice)
    if batchEffectRunCount = countAfterCreate + 1 then
        pass "Batch groups updates (single effect run)"
    else
        fail "Batch groups updates" "2" (Format.int batchEffectRunCount)

// ============================================================================
// Main Entry Point
// ============================================================================

[<EntryPoint>]
let main argv =
    Console.writeln "========================================"
    Console.writeln "  Fidelity.Signal Comprehensive Test"
    Console.writeln "========================================"

    // Run all test suites
    testSignalBasics ()
    testSignalTypes ()
    testEffects ()
    testMemos ()
    testBatch ()

    // Summary
    Console.writeln ""
    Console.writeln "========================================"
    Console.write "Results: "
    Console.write (Format.int passCount)
    Console.write " passed, "
    Console.write (Format.int failCount)
    Console.writeln " failed"
    Console.writeln "========================================"

    if failCount = 0 then 0 else 1
