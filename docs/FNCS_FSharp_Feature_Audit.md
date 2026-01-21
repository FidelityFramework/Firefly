# FNCS Feature Audit: F# Language Features vs Native Compiler Services

**Date**: January 2026
**Author**: Systematic audit of F# language specification against FNCS capabilities

---

## Executive Summary

This document catalogs every major F# language feature against FNCS (F# Native Compiler Services) support status. FNCS provides the native type universe for Fidelity/Firefly compilation to native binaries without .NET runtime dependencies.

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully supported in FNCS |
| 🚧 | Partial support (notes indicate limitations) |
| ❌ | Not supported (alternative provided if any) |
| ⚠️ | BCL-dependent (requires .NET runtime) |
| 🔮 | Planned for future PRD |

---

## 1. Primitive Types

### 1.1 Numeric Types

| F# Type | FNCS Status | NTUKind | Notes |
|---------|-------------|---------|-------|
| `int` | ✅ | `NTUint` | **Platform word** (64-bit on x86_64). Different from F#'s 32-bit `int`! |
| `int32` | ✅ | `NTUint32` | Fixed 32-bit signed integer |
| `int64` | ✅ | `NTUint64` | Fixed 64-bit signed integer |
| `int16` | ✅ | `NTUint16` | Fixed 16-bit signed integer |
| `int8` / `sbyte` | ✅ | `NTUint8` | Fixed 8-bit signed integer |
| `uint` | ✅ | `NTUuint` | Platform word unsigned |
| `uint32` | ✅ | `NTUuint32` | Fixed 32-bit unsigned |
| `uint64` | ✅ | `NTUuint64` | Fixed 64-bit unsigned |
| `uint16` | ✅ | `NTUuint16` | Fixed 16-bit unsigned |
| `uint8` / `byte` | ✅ | `NTUuint8` | Fixed 8-bit unsigned |
| `nativeint` | ✅ | `NTUnint` | Pointer-sized signed (System.IntPtr equivalent) |
| `unativeint` | ✅ | `NTUunint` | Pointer-sized unsigned (System.UIntPtr equivalent) |
| `float` / `double` | ✅ | `NTUfloat64` | 64-bit IEEE 754 |
| `float32` / `single` | ✅ | `NTUfloat32` | 32-bit IEEE 754 |
| `decimal` | ✅ | `NTUdecimal` | 128-bit decimal (16 bytes) |
| `bigint` | ⚠️ | `NTUother` | Requires `System.Numerics.BigInteger` - BCL dependent |

**Important Semantic Difference**: In FNCS/Fidelity, `int` follows ML/Rust semantics (platform word = 64-bit on x86_64), NOT .NET's 32-bit `System.Int32`. Use `int32` for explicit 32-bit integers.

### 1.2 Other Primitive Types

| F# Type | FNCS Status | NTUKind | Notes |
|---------|-------------|---------|-------|
| `bool` | ✅ | `NTUbool` | 1 byte |
| `char` | ✅ | `NTUchar` | UTF-32 code point (4 bytes), not UTF-16 like .NET |
| `string` | ✅ | `NTUstring` | UTF-8 fat pointer `{ptr, length}` - NOT `System.String` |
| `unit` | ✅ | `NTUunit` | Zero-sized type |
| `obj` | ❌ | N/A | **Eliminated** - no universal base type. Use SRTP for polymorphism |
| `exn` / `Exception` | 🚧 | Reference | Native exception type (limited, see Exceptions section) |

### 1.3 Special Numeric Literals

| Literal | FNCS Status | Notes |
|---------|-------------|-------|
| `3y` (sbyte) | ✅ | |
| `32uy` (byte) | ✅ | |
| `17s` (int16) | ✅ | |
| `99u` (uint32) | ✅ | |
| `99999999L` (int64) | ✅ | |
| `99999999I` (bigint) | ⚠️ | BCL-dependent |
| `1.0f` (float32) | ✅ | |
| `1.0` (float) | ✅ | |
| `99999999n` (nativeint) | ✅ | |
| Custom numeric literals (Q, R, Z, etc.) | ❌ | Requires FSharp.Core runtime |

---

## 2. Composite Types

### 2.1 Tuples

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Reference tuples `(a, b)` | ✅ | Struct in FNCS (value semantics) |
| Struct tuples `struct (a, b)` | ✅ | Explicit struct tuple |
| Large tuples (>7 elements) | ✅ | Flat struct, no `System.Tuple` nesting |
| Tuple deconstruction | ✅ | Pattern matching supported |
| `fst`, `snd` | ✅ | Built-in functions |

**Semantic Difference**: All tuples in FNCS are value types (struct semantics). No `System.Tuple`/`System.ValueTuple` runtime dependency.

### 2.2 Records

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Record definition | ✅ | Compiled to struct with computed layout |
| Record construction `{ field = value }` | ✅ | |
| Record copy-and-update `{ r with field = value }` | ✅ | |
| Mutable record fields | ✅ | With `mutable` keyword |
| Anonymous records `{| a = 1 |}` | ✅ | Both struct and reference forms |
| Struct records `[<Struct>]` | ✅ | All records are struct by default in FNCS |
| `[<CLIMutable>]` | ❌ | CLI interop attribute not applicable |
| Record equality/comparison | ✅ | Structural by default |

### 2.3 Discriminated Unions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Union definition | ✅ | |
| Single-case unions | ✅ | |
| Multi-case unions | ✅ | Tag + payload struct |
| Named union fields | ✅ | |
| Union pattern matching | ✅ | |
| `[<Struct>]` unions | ✅ | Value type unions |
| `option<'T>` | ✅ | Value type, `None` optimized to tag |
| `voption<'T>` (ValueOption) | ✅ | Explicit value option |
| `Result<'T, 'TError>` | ✅ | Value type result |
| `Choice<'T1, 'T2, ...>` | ✅ | Union with N cases |
| `[<RequireQualifiedAccess>]` | ✅ | Attribute supported |
| Union equality/comparison | ✅ | Structural by default |
| `UseNullAsTrueValue` compilation | 🚧 | Partial - None optimizations only |

### 2.4 Arrays

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Single-dimensional `'T[]` | ✅ | Fat pointer `{ptr, length}` |
| Array creation `[| 1; 2; 3 |]` | ✅ | |
| Array indexing `arr.[i]` | ✅ | |
| Array slicing `arr.[1..3]` | ✅ | |
| Multi-dimensional arrays `'T[,]` | 🚧 | Planned - PRD-17 |
| Jagged arrays `'T[][]` | ✅ | Array of arrays |
| Array module functions | 🚧 | Core operations only (map, fold, iter) |
| `Array.Parallel` | 🔮 | Future: vectorization via MLIR Vector dialect |

### 2.5 Lists

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| List type `'T list` | ✅ | Immutable singly-linked (arena-allocated) |
| List construction `[1; 2; 3]` | ✅ | |
| Cons operator `::` | ✅ | |
| List pattern matching | ✅ | |
| `List.empty` | ✅ | Primitive - null pointer |
| `List.head`, `List.tail` | ✅ | Primitives |
| `List.isEmpty` | ✅ | Primitive |
| `List.map`, `List.filter` | ✅ | Decompose to primitives |
| `List.fold`, `List.foldBack` | ✅ | Decompose to primitives |
| `List.concat`, `List.append` | ✅ | |
| `List.sort`, `List.sortBy` | 🚧 | Requires comparison constraints |

### 2.6 Sequences

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `seq { }` computation expression | ✅ | PRD-15: State machine struct |
| `yield` | ✅ | |
| `yield!` | ✅ | Sequence flattening |
| Sequence range `{1..10}` | ✅ | |
| Sequence range with step `{1..2..10}` | ✅ | |
| `Seq.map`, `Seq.filter` | ✅ | PRD-16: Composed state machines |
| `Seq.take`, `Seq.skip` | ✅ | |
| `Seq.fold` | ✅ | |
| `Seq.toList`, `Seq.toArray` | ✅ | |
| Lazy evaluation | ✅ | Pull-based via MoveNext |
| `IEnumerable<'T>` interop | ❌ | No BCL interface |

### 2.7 Maps and Sets

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `Map<'K, 'V>` | ✅ | Immutable AVL tree (PRD-13a) |
| `Set<'T>` | ✅ | Immutable AVL tree (PRD-13a) |
| `Map.empty`, `Set.empty` | ✅ | Primitives |
| `Map.add`, `Map.remove` | ✅ | Return new tree |
| `Map.find`, `Map.tryFind` | ✅ | |
| `Map.containsKey` | ✅ | |
| `Set.add`, `Set.remove` | ✅ | |
| `Set.contains` | ✅ | |
| Map/Set literals | ❌ | Use explicit construction |

### 2.8 Other Collection Types

| Type | FNCS Status | Notes |
|------|-------------|-------|
| `ResizeArray<'T>` (List<T>) | ❌ | Use `Array.create` + manual resize or arena-based growable |
| `Dictionary<'K, 'V>` | ❌ | Use `Map` or implement with arrays |
| `HashSet<'T>` | ❌ | Use `Set` or implement with arrays |
| `Queue<'T>`, `Stack<'T>` | ❌ | Implement with lists/arrays |
| `Span<'T>`, `ReadOnlySpan<'T>` | ✅ | Fat pointer with measures |
| `Memory<'T>` | ❌ | Use arena-allocated arrays |

---

## 3. Pointer and Reference Types

### 3.1 Pointers

| Type | FNCS Status | NTUKind | Notes |
|------|-------------|---------|-------|
| `nativeptr<'T>` | ✅ | `NTUptr` | Raw native pointer |
| `voidptr` | ✅ | `NTUptr` | Void pointer |
| `byref<'T>` | ✅ | `NTUptr` | Mutable reference |
| `inref<'T>` | ✅ | `NTUptr` | Read-only reference |
| `outref<'T>` | ✅ | `NTUptr` | Write-only reference |
| `FnPtr<'F>` | ✅ | `NTUfnptr` | Function pointer (no closures) |

### 3.2 Pointer Operations

| Operation | FNCS Status | Notes |
|-----------|-------------|-------|
| `NativePtr.stackalloc<'T> n` | ✅ | Stack allocation |
| `NativePtr.get`, `NativePtr.set` | ✅ | Index-based access |
| `NativePtr.read`, `NativePtr.write` | ✅ | Direct access |
| `NativePtr.add`, `NativePtr.sub` | ✅ | Pointer arithmetic |
| `NativePtr.toNativeInt` | ✅ | Convert to address |
| `NativePtr.ofNativeInt` | ✅ | Convert from address |
| `NativePtr.copy` | ✅ | Maps to `llvm.memcpy` |
| `NativePtr.fill` | ✅ | Maps to `llvm.memset` |
| `&&expr` (address-of) | ✅ | Byref generation |
| `fixed` expression | 🚧 | Partial support |

---

## 4. Functions and Closures

### 4.1 Function Definitions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `let f x = ...` | ✅ | |
| `let f x y = ...` (curried) | ✅ | |
| `let f (x, y) = ...` (tupled) | ✅ | |
| `let inline f x = ...` | ✅ | Important for escape analysis |
| `let rec f x = ...` | ✅ | Recursive functions |
| `let rec ... and ...` | ✅ | Mutual recursion |
| `let private f x = ...` | ✅ | Visibility |
| `let internal f x = ...` | ✅ | Visibility |
| Generic functions `let f<'T> x = ...` | ✅ | Monomorphization |

### 4.2 Lambda Expressions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `fun x -> ...` | ✅ | PRD-11: Flat closure model |
| `fun x y -> ...` (curried) | ✅ | |
| `fun (x, y) -> ...` (tupled) | ✅ | |
| Closures capturing variables | ✅ | Flat closure struct |
| Closures capturing mutable variables | ✅ | By-reference capture |
| `function` expression | ✅ | Matching function |

### 4.3 Function Application

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Function application `f x` | ✅ | |
| Partial application | ✅ | Creates closure |
| Pipe forward `x \|> f` | ✅ | Reduced during PSG construction |
| Pipe backward `f <\| x` | ✅ | |
| Composition `f >> g` | ✅ | |
| Composition `g << f` | ✅ | |
| High-precedence application `f(x)` | ✅ | |

---

## 5. Pattern Matching

### 5.1 Pattern Forms

| Pattern | FNCS Status | Notes |
|---------|-------------|-------|
| Constant patterns | ✅ | |
| Variable patterns | ✅ | |
| Wildcard `_` | ✅ | |
| As pattern `pat as ident` | ✅ | |
| Or pattern `pat \| pat` | ✅ | |
| And pattern `pat & pat` | ✅ | |
| Cons pattern `h :: t` | ✅ | |
| List pattern `[a; b; c]` | ✅ | |
| Array pattern `[\| a; b \|]` | ✅ | |
| Tuple pattern `(a, b)` | ✅ | |
| Record pattern `{ field = pat }` | ✅ | |
| Union case pattern | ✅ | |
| Type test pattern `:? Type` | 🚧 | Limited - no runtime reflection |
| Null pattern `null` | ❌ | No null in native types |
| When guards `when expr` | ✅ | |
| Active patterns | 🚧 | Simple active patterns only |

### 5.2 Match Expressions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `match expr with ...` | ✅ | |
| `function \| pat -> ...` | ✅ | |
| Pattern completeness checking | ✅ | |
| Pattern reachability warnings | ✅ | |

---

## 6. Control Flow

### 6.1 Conditionals

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `if expr then expr` | ✅ | |
| `if expr then expr else expr` | ✅ | |
| `elif` chains | ✅ | |

### 6.2 Loops

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `while expr do expr` | ✅ | |
| `for i = start to end do expr` | ✅ | |
| `for i = start downto end do expr` | ✅ | |
| `for pat in collection do expr` | ✅ | Uses Seq enumeration |
| `for pat in start..end do expr` | ✅ | |
| `break`, `continue` | ❌ | Use recursion or early return |

### 6.3 Sequential Execution

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `expr; expr` | ✅ | |
| `do expr` | ✅ | |

---

## 7. Exceptions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `exception` definition | 🚧 | Native exception type |
| `raise expr` | ✅ | |
| `failwith "message"` | ✅ | |
| `failwithf "format" args` | 🚧 | Limited format support |
| `invalidArg` | ✅ | |
| `try...with` | ✅ | Pattern-matching exception handler |
| `try...finally` | ✅ | Cleanup handler |
| `reraise()` | 🚧 | Partial support |
| Custom exception types | ✅ | |
| BCL exception types | ❌ | No System.* exceptions |
| Stack traces | ❌ | No runtime reflection for traces |

---

## 8. Type System Features

### 8.1 Type Parameters and Generics

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Generic types `'T` | ✅ | Monomorphized |
| Generic functions | ✅ | |
| Generic constraints `:>` | ✅ | Subtype constraints |
| Generic constraints `:` | 🚧 | Some constraints |
| `when` clause | ✅ | |
| Flexible types `#Type` | 🚧 | Limited |
| Anonymous type variables `_` | ✅ | |

### 8.2 Statically Resolved Type Parameters (SRTP)

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `^T` type parameters | ✅ | |
| Member constraints | ✅ | Resolved at compile time |
| `inline` requirement | ✅ | |
| Operator overloading via SRTP | ✅ | |
| SRTP arithmetic operators | ✅ | `+`, `-`, `*`, `/`, etc. |

### 8.3 Type Constraints

| Constraint | FNCS Status | Notes |
|------------|-------------|-------|
| `:> type` (subtype) | ✅ | |
| `: null` (nullness) | ❌ | No null in native type universe |
| `: struct` | ✅ | |
| `: not struct` | 🚧 | Reference types are arena-allocated |
| `: (new : unit -> 'T)` | ❌ | No default constructors |
| `: enum<underlying>` | ✅ | |
| `: delegate<args, ret>` | ❌ | No CLI delegates |
| `: unmanaged` | ✅ | For fixed-layout types |
| `: equality` | ✅ | |
| `: comparison` | ✅ | |
| Member constraint `(member ...)` | ✅ | SRTP |

### 8.4 Type Abbreviations

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `type Alias = ExistingType` | ✅ | |
| Generic abbreviations | ✅ | |
| `private` type abbreviations | ✅ | |

---

## 9. Units of Measure

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `[<Measure>] type m` | ✅ | Native measure definitions |
| Measure-annotated values `1.0<m>` | ✅ | |
| Measure-annotated types `float<m/s>` | ✅ | |
| Measure arithmetic | ✅ | Products, quotients, powers |
| Measure inference | ✅ | |
| Measure generics `'U` | ✅ | |
| Dimensionless `<1>` | ✅ | |
| **Measures on non-numeric types** | ✅ | Extended from F# - memory regions, access modes |

**FNCS Extension**: Unlike F# which restricts measures to numerics, FNCS supports measures on ANY type. This enables memory region tracking (`stack`, `arena`, `peripheral`) and access control (`ro`, `wo`, `rw`).

---

## 10. Object-Oriented Features

### 10.1 Classes

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Class definitions | 🚧 | Limited - prefer records/DUs |
| Primary constructors | 🚧 | |
| Additional constructors | ❌ | Use factory functions |
| `member` definitions | 🚧 | |
| `static member` | ✅ | Module functions preferred |
| Properties (get/set) | 🚧 | |
| Auto-properties | 🚧 | |
| `val` field declarations | 🚧 | |
| `inherit` | ❌ | No class inheritance |
| Object expressions | ❌ | |
| `this`/`self` identifier | 🚧 | |

### 10.2 Interfaces

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Interface definitions | 🚧 | Limited support |
| Interface implementation | 🚧 | |
| `interface ... with` | 🚧 | |
| Default interface members | ❌ | |
| Object expressions for interfaces | ❌ | |

### 10.3 Structs

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `[<Struct>]` attribute | ✅ | All value types |
| Struct members | ✅ | |
| Struct constructors | ✅ | |
| `DefaultValue` attribute | 🚧 | |

### 10.4 Inheritance and Polymorphism

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Class inheritance | ❌ | Use composition + SRTP |
| `base` keyword | ❌ | |
| `override` | ❌ | |
| `abstract` | ❌ | Use interfaces/SRTP |
| Virtual dispatch | ❌ | Use SRTP (compile-time) |
| `:>` upcast | 🚧 | Limited |
| `:?>` downcast | ❌ | No runtime type info |
| `:?` type test | ❌ | No runtime type info |

---

## 11. Modules and Namespaces

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `module Name` | ✅ | |
| `namespace Name` | ✅ | |
| Nested modules | ✅ | |
| `open` declarations | ✅ | |
| `[<AutoOpen>]` | ✅ | |
| `[<RequireQualifiedAccess>]` | ✅ | |
| Module functions | ✅ | Primary organization |
| Module values | ✅ | |
| Recursive modules `rec` | ✅ | |

---

## 12. Computation Expressions

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `seq { }` | ✅ | PRD-15 |
| `async { }` | 🔮 | PRD-17: LLVM coroutines planned |
| `task { }` | 🔮 | Future |
| Custom builders | 🚧 | Limited support |
| `let!` | ✅ | In seq/async |
| `do!` | ✅ | |
| `return` | ✅ | |
| `return!` | ✅ | |
| `yield` | ✅ | |
| `yield!` | ✅ | |
| `use` in CE | ✅ | |
| `use!` | 🔮 | |
| `match!` | 🔮 | Async pattern matching |
| `while` in CE | ✅ | |
| `for` in CE | ✅ | |
| `try...with` in CE | 🚧 | |
| `try...finally` in CE | 🚧 | |
| `and!` | ❌ | |

---

## 13. Lazy Values

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `lazy expr` | ✅ | PRD-14: Extended flat closure |
| `Lazy<'T>` type | ✅ | Struct with memoization state |
| `Lazy.force` | ✅ | |
| `lazyVal.Value` | ✅ | |
| `lazyVal.IsValueCreated` | ✅ | |
| Thread-safe initialization | ❌ | Single-threaded for now |

---

## 14. Quotations

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `<@ expr @>` typed quotations | ✅ | Used for platform bindings |
| `<@@ expr @@>` untyped | 🚧 | |
| `Expr<'T>` type | ✅ | |
| Quotation evaluation | ❌ | No runtime eval (compile-time only) |
| Quotation splicing `%expr` | ❌ | |
| `[<ReflectedDefinition>]` | 🚧 | For platform quotations |

---

## 15. Attributes

### 15.1 Supported Attributes

| Attribute | FNCS Status | Notes |
|-----------|-------------|-------|
| `[<Struct>]` | ✅ | |
| `[<Measure>]` | ✅ | |
| `[<Literal>]` | ✅ | |
| `[<RequireQualifiedAccess>]` | ✅ | |
| `[<AutoOpen>]` | ✅ | |
| `[<NoEquality>]` | ✅ | |
| `[<NoComparison>]` | ✅ | |
| `[<CustomEquality>]` | ✅ | |
| `[<CustomComparison>]` | ✅ | |
| `[<StructuralEquality>]` | ✅ | |
| `[<StructuralComparison>]` | ✅ | |
| `[<ReferenceEquality>]` | 🚧 | |
| `[<Obsolete>]` | ✅ | |

### 15.2 Unsupported Attributes (BCL-Dependent)

| Attribute | Status | Alternative |
|-----------|--------|-------------|
| `[<Serializable>]` | ❌ | Use BAREWire |
| `[<DllImport>]` | 🚧 | Use FNCS extern declarations |
| `[<MarshalAs>]` | ❌ | Native type layouts |
| `[<StructLayout>]` | 🚧 | `[<BAREStruct>]` for explicit |
| `[<FieldOffset>]` | 🚧 | BAREWire field attributes |
| `[<AllowNullLiteral>]` | ❌ | No null |

---

## 16. Interop Features

### 16.1 P/Invoke

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `[<DllImport>]` | ❌ | Use FNCS extern intrinsics |
| External function declarations | ✅ | Via FNCS Sys.* intrinsics |
| Platform syscalls | ✅ | Via platform bindings |

### 16.2 COM Interop

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| COM interfaces | ❌ | |
| COM objects | ❌ | |
| `ComImport` | ❌ | |

---

## 17. BCL-Dependent Features

These features inherently require the .NET Base Class Library and cannot be supported in freestanding mode:

| Feature | Status | Alternative |
|---------|--------|-------------|
| `System.String` methods | ❌ | FNCS String.* intrinsics |
| `System.DateTime` | ✅ | FNCS DateTime intrinsic (64-bit ticks) |
| `System.TimeSpan` | ✅ | FNCS TimeSpan intrinsic |
| `System.Guid` | ✅ | FNCS Uuid intrinsic |
| `System.Console` | ✅ | FNCS Console.* (Layer 3 in Fidelity.Platform) |
| `System.IO.*` | ❌ | Platform-specific file operations |
| `System.Net.*` | ❌ | Future: Farscape networking |
| `System.Collections.Generic.*` | ❌ | FNCS collections (List, Map, Set) |
| `System.Linq.*` | ❌ | Use Seq operations |
| `System.Threading.*` | ❌ | Future: Olivier actors |
| `System.Reflection.*` | ❌ | No runtime reflection |
| `System.Type` | ❌ | Compile-time only type info |
| `typeof<T>` | ❌ | No runtime type objects |
| `printf`/`sprintf` (full) | 🚧 | Limited format strings |

---

## 18. Runtime Features

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| Garbage collection | ❌ | Deterministic memory (arenas) |
| Runtime type checking | ❌ | Types erased at runtime |
| Reflection | ❌ | No System.Reflection |
| Dynamic typing | ❌ | No `dynamic`, no `obj` |
| Type providers | ❌ | Compile-time only analysis |
| Code quotation eval | ❌ | Quotations for platform bindings only |
| `box`/`unbox` | ❌ | No obj type |
| Null reference | ❌ | Option type instead |
| Default initialization | 🚧 | `NativeDefault.zeroed<'T>` |

---

## 19. String Features

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| String literals `"hello"` | ✅ | UTF-8 encoded |
| Verbatim strings `@"path"` | ✅ | |
| Triple-quoted `"""text"""` | ✅ | |
| String concatenation `+` | ✅ | String.concat intrinsic |
| String interpolation `$"x = {x}"` | 🚧 | **Limited** - simple cases only |
| `String.length` | ✅ | |
| `String.substring` | ✅ | |
| `String.concat` | ✅ | |
| String comparison | ✅ | UTF-8 byte comparison |
| String formatting `sprintf` | 🚧 | Limited format specifiers |
| Regular expressions | ❌ | No System.Text.RegularExpressions |
| `String.Split`, `String.Join` | 🚧 | Basic support |

**String Representation**: FNCS strings are UTF-8 fat pointers `{ptr: nativeptr<byte>, length: int}`, NOT `System.String`. This is more memory-efficient and compatible with native APIs.

---

## 20. Async/Concurrent Features

| Feature | FNCS Status | Notes |
|---------|-------------|-------|
| `async { }` | 🔮 | PRD-17: LLVM coroutines |
| `Async.RunSynchronously` | 🔮 | |
| `Async.Start` | 🔮 | |
| `Async.Parallel` | 🔮 | |
| `Async.AwaitTask` | ❌ | No Task interop |
| `task { }` | 🔮 | |
| `Task<'T>` | ❌ | Use native async |
| `MailboxProcessor<'T>` | 🔮 | Olivier actor model |
| `lock` expression | ❌ | Use actors for concurrency |
| `Interlocked.*` | 🔮 | Atomic operations planned |
| Thread-local storage | 🚧 | Actor-local instead |

---

## 21. Special Identifiers and Operators

| Identifier/Operator | FNCS Status | Notes |
|---------------------|-------------|-------|
| `ignore` | ✅ | |
| `id` | ✅ | Identity function |
| `not` | ✅ | Boolean negation |
| `raise` | ✅ | |
| `reraise` | 🚧 | |
| `sizeof<'T>` | ✅ | Compile-time size |
| `typeof<'T>` | ❌ | No runtime types |
| `nameof` | ✅ | Compile-time name |
| `||>` (pipeline 2-tuple) | ✅ | |
| `|||>` (pipeline 3-tuple) | ✅ | |
| `<||` | ✅ | |
| `<|||` | ✅ | |

---

## 22. Numeric Conversions

| Conversion | FNCS Status | Notes |
|------------|-------------|-------|
| `int`, `int32`, `int64`, etc. | ✅ | SRTP-based |
| `float`, `float32` | ✅ | |
| `byte`, `sbyte` | ✅ | |
| `decimal` | ✅ | |
| `char` (from int) | ✅ | |
| `string` (from any) | 🚧 | Limited ToString equivalent |
| `enum` | ✅ | |
| Checked conversions (`Checked.*`) | ✅ | Overflow detection |

---

## 23. Math Functions

| Function | FNCS Status | Notes |
|----------|-------------|-------|
| `abs` | ✅ | SRTP polymorphic |
| `sign` | ✅ | |
| `min`, `max` | ✅ | |
| `pown` | ✅ | Integer exponent |
| `sqrt` | ✅ | LLVM intrinsic |
| `sin`, `cos`, `tan` | ✅ | LLVM intrinsics |
| `asin`, `acos`, `atan`, `atan2` | ✅ | |
| `sinh`, `cosh`, `tanh` | ✅ | |
| `exp`, `log`, `log10` | ✅ | |
| `ceil`, `floor`, `round` | ✅ | |
| `infinity`, `nan` | ✅ | Float constants |
| `(**)`  power operator | ✅ | |

---

## Summary Tables

### Type Support Summary

| Category | Fully Supported | Partial | Not Supported |
|----------|-----------------|---------|---------------|
| Primitives | 15 | 1 | 1 |
| Collections | 6 | 2 | 4 |
| Composite Types | 6 | 1 | 0 |
| Functions | 10 | 0 | 0 |
| OOP Features | 2 | 8 | 6 |

### Key Gaps and Alternatives

| Gap | Alternative |
|-----|-------------|
| `obj` type | Use SRTP for polymorphism |
| `null` | Use `option<'T>` |
| `box`/`unbox` | Monomorphization |
| Runtime reflection | Compile-time analysis only |
| Class inheritance | Composition + SRTP |
| `ResizeArray` | Arena-allocated arrays |
| `async`/`task` | Future: LLVM coroutines (PRD-17) |
| GC | Arena-based memory management |
| String interpolation | Limited support (simple cases) |
| `System.*` types | FNCS intrinsic types |

---

## Appendix: PRD Roadmap for Missing Features

| PRD | Feature | Status |
|-----|---------|--------|
| PRD-13a | Core Collections (List, Map, Set) | Implemented |
| PRD-14 | Lazy | Implemented |
| PRD-15 | Simple Sequences | Implemented |
| PRD-16 | Sequence Operations | In Progress |
| PRD-17 | Async (LLVM coroutines) | Planned |
| PRD-18 | Multi-dimensional arrays | Planned |
| PRD-19 | Full computation expressions | Planned |
| PRD-20-22 | Arena and lifetime inference | Planned |
| Future | Olivier actor model | Planned |
| Future | Vector/SIMD operations | Planned |

---

## Document History

- 2026-01-20: Initial comprehensive audit
