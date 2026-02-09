# F-01: HelloWorldDirect

> **Sample**: `01_HelloWorldDirect` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

The inaugural Firefly sample establishes the complete compilation pipeline from F# source to native binary. While the surface feature is trivial - printing "Hello, World!" - the underlying infrastructure represents the full FNCS→PSG→Alex→MLIR→LLVM chain.

**Key Achievement**: Proved that the compilation model works end-to-end, producing a freestanding binary with no runtime dependencies.

---

## 2. Surface Feature

```fsharp
module HelloWorld

Console.writeln "Hello, World!"
```

**Expected Output**:
```
Hello, World!
```

---

## 3. Infrastructure Contributions

### 3.1 String Representation

The `string` type is not .NET's `System.String`. It is an FNCS native type with a fat pointer representation:

```
UTF-8 Fat Pointer
┌─────────────────────────────────────┐
│ ptr: i8*  (pointer to UTF-8 data)  │
│ len: i64  (byte length, not chars) │
└─────────────────────────────────────┘
```

**Implementation Details**:
- String literals are placed in `.rodata` section
- Null-terminated for C interop compatibility
- Length stored separately for safe slicing
- NTUKind: `NTUKind.String`

### 3.2 Console Intrinsics

`Console.writeln` is an FNCS intrinsic, not a library function:

```fsharp
// In FNCS CheckExpressions.fs
| "Console.writeln" ->
    NativeType.TFun(env.Globals.StringType, env.Globals.UnitType)
```

**Lowering Path**:
1. FNCS recognizes `Console.writeln` as intrinsic
2. PSG marks node with `SemanticKind.Intrinsic`
3. Alex emits platform-specific syscall sequence via Bindings

### 3.3 Platform Syscall Binding

Linux x86_64 implementation:

```mlir
// Sys.write(fd=1, buffer=str, length=len)
%fd = arith.constant 1 : i32
%len = memref.dim %str, %c0 : memref<?xi8>
%written = func.call @Sys.write(%fd, %str) : (i32, memref<?xi8>) -> i64
```

The syscall is wrapped with newline handling for `writeln`.

### 3.4 Entry Point Generation

Alex generates the native entry point:

```mlir
func.func @main() -> i32 {
  // Module initialization
  // Call user's entry point (module-level statements)
  // Return 0
  %zero = arith.constant 0 : i32
  func.return %zero : i32
}
```

### 3.5 Static Data Placement

String literals are emitted as global constants:

```mlir
memref.global private constant @str_0 : memref<15xi8> = dense<[72,101,108,108,111,44,32,87,111,114,108,100,33,10,0]>
```

---

## 4. PSG Representation

The PSG for HelloWorldDirect contains:

```
PSG Structure
├── ModuleOrNamespace: HelloWorld
│   └── StatementSequence
│       └── Application
│           ├── Function: Console.writeln (Intrinsic)
│           └── Argument: "Hello, World!" (StringLiteral)
```

**Key Properties**:
- Single reachable entry point
- No captures (no closures)
- Intrinsic call marked for platform binding lookup

---

## 5. Coeffects Introduced

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | Pre-computed SSA assignments for all PSG nodes |

This sample establishes the fundamental coeffect pattern: metadata computed before emission, observed by witnesses during traversal.

---

## 6. MLIR Output Pattern

```mlir
module {
  memref.global private constant @str_0 : memref<14xi8> = dense<[72,101,108,108,111,44,32,87,111,114,108,100,33,10]>

  func.func @main() -> i32 {
    %str = memref.get_global @str_0 : memref<14xi8>
    %fd = arith.constant 1 : i32
    %written = func.call @Sys.write(%fd, %str) : (i32, memref<14xi8>) -> i64
    %zero = arith.constant 0 : i32
    func.return %zero : i32
  }

  func.func private @Sys.write(i32, memref<?xi8>) -> i64
}
```

---

## 7. Validation

```bash
cd samples/console/FidelityHelloWorld/01_HelloWorldDirect
/path/to/Firefly compile HelloWorld.fidproj
./HelloWorld
# Output: Hello, World!
```

---

## 8. Architectural Lessons

1. **Full Pipeline Required**: Even "hello world" needs the complete FNCS→PSG→Alex→MLIR chain
2. **Intrinsics, Not Libraries**: Platform operations are FNCS intrinsics, not linked library functions
3. **Fat Pointers for Strings**: UTF-8 representation with explicit length, not null-terminated C strings
4. **Coeffect Pattern Established**: Pre-computed metadata, not computed during emission

---

## 9. Downstream Dependencies

This sample's infrastructure enables:
- **F-02**: Arena allocation (builds on string representation)
- **All Samples**: Entry point generation, intrinsic recognition
- **C-01**: Closure infrastructure (extends function call patterns)

---

## 10. Related Documents

- [F-00-Synopsis](F-00-Synopsis.md) - Foundation Series overview
- [Architecture_Canonical.md](../Architecture_Canonical.md) - Two-layer architecture
- [FNCS_Architecture.md](../FNCS_Architecture.md) - FNCS intrinsic design
