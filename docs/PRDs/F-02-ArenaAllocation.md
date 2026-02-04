# F-02: Arena Allocation

> **Sample**: `02_HelloWorldSaturated` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces deterministic memory management through arena allocation using MLIR memref semantics. String interpolation triggers runtime-sized allocation, demonstrating that dynamic strings in Fidelity use memref operations with explicit lifetime management.

**Key Achievement**: Established memref-based string operations (concat, length) with runtime-sized allocation via memref.alloc.

---

## 2. Surface Feature

```fsharp
module Examples.HelloWorldSaturated

let hello() =
    Console.write "Enter your name: "
    let name = Console.readln()
    Console.writeln $"Hello, {name}!"
```

**Expected Behavior**:
1. Print prompt (static string)
2. Read input (dynamic memref allocation)
3. Concatenate strings (runtime-sized memref.alloc)
4. Print greeting

---

## 3. Infrastructure Contributions

### 3.1 Strings as Memrefs

In MLIR memref semantics, strings **ARE** memrefs:

```
Static string literal:     memref<13 x i8>   (e.g., "Hello, World!")
Dynamic string (readln):   memref<?x i8>     (runtime-sized)
Concatenation result:      memref<?x i8>     (len1 + len2, computed at runtime)
```

**No fat pointers. No control structs. Just memrefs.**

### 3.2 String.Length

```fsharp
// F# signature
String.Length : string -> int
```

**MLIR implementation**:
```mlir
%len_index = memref.dim %str, %c0 : memref<?xi8>  // Returns index type
%len_int = index.casts %len_index : index to i64   // Cast to platform int for F#
```

**Key point**: `memref.dim` returns `index` type. Cast to platform int at F# boundary.

### 3.3 String.concat2

```fsharp
// Desugared from: $"Hello, {name}!"
String.concat2 "Hello, " name
```

**MLIR implementation**:
```mlir
// Get lengths (index type)
%c0 = arith.constant 0 : index
%len1 = memref.dim %str1, %c0 : memref<?xi8>
%len2 = memref.dim %str2, %c0 : memref<?xi8>

// Compute combined length (index arithmetic)
%total_len = arith.addi %len1, %len2 : index

// Allocate result (heap for now, arena later)
%result = memref.alloc(%total_len) : memref<?xi8>

// Copy both strings (FFI to libc memcpy)
%dest1 = memref.extract_aligned_pointer_as_index %result : memref<?xi8> -> index
%src1 = memref.extract_aligned_pointer_as_index %str1 : memref<?xi8> -> index
call @memcpy(%dest1, %src1, %len1)

%offset = arith.addi %dest1, %len1 : index
%src2 = memref.extract_aligned_pointer_as_index %str2 : memref<?xi8> -> index
call @memcpy(%offset, %src2, %len2)
```

**Critical**: All size arithmetic in `index` type, NOT i64.

### 3.4 Console Intrinsics

```fsharp
Console.readln : unit -> string              // Returns memref<?xi8>
Console.write  : string -> unit              // Takes memref<?xi8>
Console.writeln: string -> unit              // Takes memref<?xi8>
```

All operate on memrefs directly. No marshaling, no fat pointers.

---

## 4. PSG Representation

```
PSG Structure
├── ModuleOrNamespace: Examples.HelloWorldSaturated
│   └── Binding: hello
│       └── Lambda
│           ├── Application: Console.write
│           │   └── Literal: "Enter your name: "
│           ├── Binding: name
│           │   └── Application: Console.readln
│           └── Application: Console.writeln
│               └── Application: String.concat2
│                   ├── Literal: "Hello, "
│                   ├── VarRef: name
│                   └── Literal: "!"
```

Baker decomposes `String.concat2` into length extraction + memref.alloc + memcpy operations.

---

## 5. Coeffects Introduced

| Coeffect | Purpose |
|----------|---------|
| SSAAssignment | SSA values for all sub-expressions |
| PlatformArch | Target architecture (affects int size, NOT index) |

**No ArenaAffinity** - that's future work (A-04: Regions).

---

## 6. Memory Model

```
Stack Frame
┌──────────────────────────────────────────┐
│ name: memref<?xi8> descriptor           │
│   (points to heap-allocated buffer)      │
└──────────────────────────────────────────┘

Heap Memory (via memref.alloc)
┌──────────────────────────────────────────┐
│ "World\n" (user input, 6 bytes)         │
└──────────────────────────────────────────┘

Result of String.concat2
┌──────────────────────────────────────────┐
│ "Hello, World!\n!" (15 bytes)           │
└──────────────────────────────────────────┘
```

**Memref descriptors live on stack. Buffers allocated on heap (for now).**

---

## 7. MLIR Output Pattern

```mlir
module {
  // Static string literal
  memref.global "private" constant @str_794220374 : memref<18xi8> = dense<"Enter your name: \0A">

  func.func @HelloWorldSaturated.hello() {
    // Write prompt (static memref)
    %prompt = memref.get_global @str_794220374 : memref<18xi8>
    %prompt_cast = memref.cast %prompt : memref<18xi8> to memref<?xi8>
    call @Console.write(%prompt_cast) : (memref<?xi8>) -> ()

    // Read input (dynamic memref allocation inside Console.readln)
    %name = call @Console.readln() : () -> memref<?xi8>

    // String concatenation
    %c0 = arith.constant 0 : index

    // Get "Hello, " length
    %hello_lit = memref.get_global @str_hello : memref<7xi8>
    %hello = memref.cast %hello_lit : memref<7xi8> to memref<?xi8>
    %len1 = memref.dim %hello, %c0 : memref<?xi8>

    // Get name length
    %len2 = memref.dim %name, %c0 : memref<?xi8>

    // Get "!" length
    %exclaim_lit = memref.get_global @str_exclaim : memref<1xi8>
    %exclaim = memref.cast %exclaim_lit : memref<1xi8> to memref<?xi8>
    %len3 = memref.dim %exclaim, %c0 : memref<?xi8>

    // Compute total length (index arithmetic)
    %len_temp = arith.addi %len1, %len2 : index
    %total_len = arith.addi %len_temp, %len3 : index

    // Allocate result buffer
    %result = memref.alloc(%total_len) : memref<?xi8>

    // Copy parts (via FFI memcpy - takes index for sizes)
    call @memcpy_helper(%result, %hello, %len1)
    call @memcpy_helper_offset(%result, %len1, %name, %len2)
    %offset2 = arith.addi %len1, %len2 : index
    call @memcpy_helper_offset(%result, %offset2, %exclaim, %len3)

    // Write result
    call @Console.writeln(%result) : (memref<?xi8>) -> ()

    func.return
  }
}
```

**Pure memref dialect. No llvm.* operations. All size arithmetic in `index` type.**

---

## 8. Type Flow

```
F# Type System          FNCS NativeType              MLIR Type
─────────────────────────────────────────────────────────────────
string (literal)   →    NTUstring, Opaque      →     memref<N x i8>
string (dynamic)   →    NTUstring, Opaque      →     memref<?x i8>
int (length)       →    NTUint, PlatformWord   →     i64 (on x86_64)

Memref size/dim    →    (no NativeType)        →     index
```

**Critical distinction**:
- **F# `int`** (String.Length result) → platform word (i64 on x86_64)
- **MLIR `index`** (memref.dim result, size arithmetic) → MLIR index type
- Convert between them with `index.casts`

---

## 9. Validation

```bash
cd samples/console/FidelityHelloWorld/02_HelloWorldSaturated
/home/hhh/repos/Firefly/src/bin/Debug/net10.0/Firefly compile HelloWorld.fidproj -k
echo "World" | ./HelloWorld
# Expected output:
# Enter your name:
# Hello, World!
```

Check intermediate: `cat target/intermediates/07_output.mlir`
- Should see `memref.alloc(%size)` where `%size` is `index` type
- Should see `arith.addi` on `index` values for size arithmetic
- Should NOT see `llvm.*` operations
- Should NOT see `i64` in size arithmetic (only as result of `index.casts` for F# int)

---

## 10. Architectural Lessons

1. **Strings ARE memrefs**: No fat pointers, no control structs
2. **Index vs int**: MLIR `index` for sizes/dims, F# `int` (platform word) for user-visible values
3. **Runtime-sized allocation**: `memref.alloc(%size)` with runtime-computed size
4. **Heap bridge**: Using `memref.alloc` (heap) until arena infrastructure ready
5. **FFI boundary**: `memref.extract_aligned_pointer_as_index` for libc memcpy calls

---

## 11. Future: True Arena Allocation (A-04)

When proper arena support is added:

```mlir
// Stack-allocated arena buffer
%arena = memref.alloca(1024) : memref<1024xi8>

// Bump allocation via memref.subview
%offset = ... // Track current offset
%slice = memref.subview %arena[%offset][%size][1] : memref<1024xi8> to memref<?xi8>
```

**No control struct**. Arena IS the memref. Bump allocation = subview + offset tracking.

---

## 12. Downstream Dependencies

This sample's infrastructure enables:
- **F-03**: Pipe operators (function composition with string results)
- **F-04**: Full currying (partial application returning string functions)
- **A-04**: Region-based memory management (scoped memref.alloca)

---

## 13. Related Documents

- [F-01-HelloWorldDirect](F-01-HelloWorldDirect.md) - Static strings only
- [MLIR_Memref_Strings_Architecture.md](../MLIR_Memref_Strings_Architecture.md) - Memref-based string design
- [Memory_Management_Architecture.md](../Memory_Management_Architecture.md) - Overall memory model
