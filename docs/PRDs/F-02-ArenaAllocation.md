# F-02: Arena Allocation

> **Sample**: `02_HelloWorldSaturated` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces deterministic memory management through arena allocation. User input is read into an arena-backed buffer, demonstrating that dynamic allocation in Fidelity is explicit, scoped, and has predictable lifetime.

**Key Achievement**: Established the `Arena<'lifetime>` intrinsic type and byref parameter passing pattern for in-place allocation.

---

## 2. Surface Feature

```fsharp
module HelloWorld

let arena = Arena.create<'a> 1024
let name = Console.readlnFrom &arena
Console.writeln $"Hello, {name}!"
```

**Expected Behavior**:
1. Prompt user for name
2. Read input into arena-allocated buffer
3. Print personalized greeting

---

## 3. Infrastructure Contributions

### 3.1 Arena Type

The `Arena<'lifetime>` type is an FNCS intrinsic representing a bump allocator:

```
Arena Structure
┌─────────────────────────────────────┐
│ base: i8*   (allocation base)      │
│ offset: i64 (current bump pointer) │
│ size: i64   (total capacity)       │
└─────────────────────────────────────┘
```

**Key Properties**:
- Stack-allocated (the arena control struct)
- Bump allocation within (O(1) allocate, no free)
- Lifetime parameter tracks escape
- NTUKind: `NTUKind.Arena`

### 3.2 Arena.create Intrinsic

```fsharp
// In FNCS CheckExpressions.fs
| "Arena.create" ->
    // unit -> Arena<'lifetime>
    NativeType.TFun(env.Globals.UnitType, NativeType.TArena(lifetime))
```

Creates a stack-allocated arena with specified byte capacity.

### 3.3 byref Parameters

The `&arena` syntax passes the arena by reference, enabling in-place modification:

```fsharp
Console.readlnFrom : byref<Arena<'a>> -> string
```

**Representation**:
- `byref<T>` is a native pointer to T
- Enables mutation without copying
- Unifies with F#'s byref semantics

### 3.4 Console.readlnFrom Intrinsic

Reads a line from stdin, allocating the result in the provided arena:

```fsharp
| "Console.readlnFrom" ->
    // byref<Arena<'a>> -> string
    NativeType.TFun(
        NativeType.TByRef(NativeType.TArena(lifetime)),
        env.Globals.StringType
    )
```

**Implementation**:
1. Read bytes from stdin into temporary buffer
2. Bump-allocate string storage from arena
3. Copy bytes to arena storage
4. Return fat pointer to arena-allocated string

### 3.5 String Interpolation

The `$"Hello, {name}!"` syntax lowers to concatenation intrinsics:

```fsharp
// Desugared form
String.concat ["Hello, "; name; "!"]
```

No runtime formatting - purely compile-time string building.

---

## 4. PSG Representation

```
PSG Structure
├── ModuleOrNamespace: HelloWorld
│   └── StatementSequence
│       ├── LetBinding: arena
│       │   └── Application: Arena.create (Intrinsic)
│       ├── LetBinding: name
│       │   └── Application: Console.readlnFrom (Intrinsic)
│       │       └── AddressOf: arena
│       └── Application: Console.writeln
│           └── InterpolatedString
│               └── [literal, name, literal]
```

---

## 5. Coeffects Introduced

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA assignments for let bindings |
| ArenaAffinity | Track which arena each allocation uses |

The ArenaAffinity coeffect enables the compiler to verify that allocated values don't escape their arena's lifetime.

---

## 6. Memory Model

```
Stack Frame
┌─────────────────────────────────────┐
│ arena: Arena<'a>                   │
│   ├── base: ptr to 1024-byte block │
│   ├── offset: 0 initially          │
│   └── size: 1024                   │
├─────────────────────────────────────┤
│ name: string (fat pointer)         │
│   ├── ptr: points into arena       │
│   └── len: input length            │
└─────────────────────────────────────┘

Arena Memory Block (1024 bytes)
┌─────────────────────────────────────┐
│ "John\n\0" (user input + null)     │
│ [remaining space...]               │
└─────────────────────────────────────┘
```

---

## 7. MLIR Output Pattern

```mlir
module {
  llvm.func @main() -> i32 {
    // Allocate arena on stack
    %arena_mem = llvm.alloca 1 x !llvm.array<1024 x i8>
    %arena = llvm.alloca 1 x !llvm.struct<(ptr, i64, i64)>
    // Initialize arena struct
    %base_ptr = llvm.getelementptr %arena_mem[0, 0]
    llvm.store %base_ptr, %arena_base_slot
    llvm.store 0, %arena_offset_slot
    llvm.store 1024, %arena_size_slot

    // Read into arena
    %name = llvm.call @console_readln_from(%arena)

    // Interpolate and print
    // ...

    llvm.return %zero : i32
  }
}
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/02_HelloWorldSaturated
/path/to/Firefly compile HelloWorld.fidproj
echo "World" | ./HelloWorld
# Output: Hello, World!
```

---

## 9. Architectural Lessons

1. **Explicit Memory Management**: Arenas make allocation visible and deterministic
2. **byref for Mutation**: F#'s byref pattern maps directly to native pointers
3. **No GC Required**: Bump allocation + stack lifetimes eliminate runtime collection
4. **String Interning**: Interpolation is compile-time concatenation, not runtime formatting

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-09**: Result type (arena-allocated heterogeneous DUs)
- **C-01**: Closure environments (could be arena-allocated)
- **A-04**: Region-based memory management (extends arena concept)

---

## 11. Related Documents

- [F-01-HelloWorldDirect](F-01-HelloWorldDirect.md) - Basic compilation
- [Arena_Intrinsic_Architecture.md](../Arena_Intrinsic_Architecture.md) - Arena design
- [Memory_Management_Architecture.md](../Memory_Management_Architecture.md) - Overall memory model
