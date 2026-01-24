# F-07: Bits Intrinsics

> **Sample**: `07_BitsTest` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces bit-level operations: network byte order conversion (htons/ntohl) and type reinterpretation (float↔int bits). These operations are essential for binary protocol handling and heterogeneous DU implementation.

**Key Achievement**: Established `Bits` intrinsic module with zero-cost type reinterpretation via LLVM bitcast.

---

## 2. Surface Feature

```fsharp
module BitsTest

// Network byte order
let port = 8080us
let networkPort = Bits.htons port

// Bit reinterpretation
let f = 3.14
let bits = Bits.float64ToInt64Bits f
let restored = Bits.int64BitsToFloat64 bits

Console.writeln $"Original: {f}"
Console.writeln $"Bits: {bits}"
Console.writeln $"Restored: {restored}"
```

---

## 3. Infrastructure Contributions

### 3.1 Network Byte Order Intrinsics

| Intrinsic | Type | Purpose |
|-----------|------|---------|
| `Bits.htons` | `uint16 -> uint16` | Host to network short |
| `Bits.ntohs` | `uint16 -> uint16` | Network to host short |
| `Bits.htonl` | `uint32 -> uint32` | Host to network long |
| `Bits.ntohl` | `uint32 -> uint32` | Network to host long |

**Implementation**: Byte swap on little-endian architectures (x86_64), no-op on big-endian.

```mlir
// htons implementation (x86_64)
%swapped = llvm.intr.bswap(%value) : i16
```

### 3.2 Bit Reinterpretation Intrinsics

| Intrinsic | Type | Purpose |
|-----------|------|---------|
| `Bits.float32ToInt32Bits` | `float32 -> int32` | Float to int bit pattern |
| `Bits.int32BitsToFloat32` | `int32 -> float32` | Int pattern to float |
| `Bits.float64ToInt64Bits` | `float64 -> int64` | Double to int64 pattern |
| `Bits.int64BitsToFloat64` | `int64 -> float64` | Int64 pattern to double |

**Implementation**: LLVM `bitcast` - zero cost, no computation.

```mlir
%bits = llvm.bitcast %float_val : f64 to i64
%restored = llvm.bitcast %bits : i64 to f64
```

### 3.3 FNCS Intrinsic Definitions

```fsharp
// In CheckExpressions.fs
| "Bits.htons" ->
    NativeType.TFun(env.Globals.UInt16Type, env.Globals.UInt16Type)

| "Bits.float64ToInt64Bits" ->
    NativeType.TFun(env.Globals.Float64Type, env.Globals.Int64Type)

| "Bits.int64BitsToFloat64" ->
    NativeType.TFun(env.Globals.Int64Type, env.Globals.Float64Type)
```

### 3.4 Use in DU Payload Storage

Bits intrinsics enable storing different-typed payloads in uniform DU slots:

```fsharp
type Number = IntVal of int | FloatVal of float
```

Both cases fit in 8 bytes:
- `IntVal`: Direct i64 storage
- `FloatVal`: `Bits.float64ToInt64Bits` for storage, `Bits.int64BitsToFloat64` for retrieval

---

## 4. PSG Representation

```
ModuleOrNamespace: BitsTest
├── LetBinding: port = 8080us
├── LetBinding: networkPort
│   └── Application
│       ├── Function: Bits.htons (Intrinsic)
│       └── Argument: port
│
├── LetBinding: f = 3.14
├── LetBinding: bits
│   └── Application
│       ├── Function: Bits.float64ToInt64Bits (Intrinsic)
│       └── Argument: f
│
├── LetBinding: restored
│   └── Application
│       ├── Function: Bits.int64BitsToFloat64 (Intrinsic)
│       └── Argument: bits
│
└── StatementSequence (Console.writeln calls)
```

---

## 5. MLIR Patterns

### 5.1 Byte Swap

```mlir
// Bits.htons
%port = llvm.mlir.constant(8080 : i16) : i16
%network_port = llvm.intr.bswap(%port) : i16
```

### 5.2 Bitcast

```mlir
// Bits.float64ToInt64Bits
%f = llvm.mlir.constant(3.14 : f64) : f64
%bits = llvm.bitcast %f : f64 to i64

// Bits.int64BitsToFloat64
%restored = llvm.bitcast %bits : i64 to f64
```

---

## 6. Round-Trip Guarantee

Bit reinterpretation is lossless:

```fsharp
Bits.int64BitsToFloat64 (Bits.float64ToInt64Bits x) = x  // Always true
Bits.int32BitsToFloat32 (Bits.float32ToInt32Bits x) = x  // Always true
```

This is not type conversion - it's viewing the same bits as different types.

---

## 7. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all bindings |

No new coeffects - Bits operations are pure computations.

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/07_BitsTest
/path/to/Firefly compile BitsTest.fidproj
./BitsTest
# Output:
# Original: 3.14
# Bits: 4614253070214989087
# Restored: 3.14
```

---

## 9. Architectural Lessons

1. **Zero-Cost Abstraction**: Bitcast has no runtime cost
2. **Type Safety Preserved**: Intrinsics are typed, no unsafe casts exposed
3. **Binary Protocol Ready**: Network byte order essential for sockets
4. **DU Enabler**: Uniform payload slots work across types

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **F-05**: Homogeneous DU payload storage (uses bit coercion)
- **F-09**: Heterogeneous DU (extends the pattern)
- **I-01**: Socket protocols (network byte order)

---

## 11. Related Documents

- [F-05-DiscriminatedUnions](F-05-DiscriminatedUnions.md) - Uses bits for payload storage
- [I-01-SocketBasics](I-01-SocketBasics.md) - Network byte order usage
- [FNCS_Architecture.md](../FNCS_Architecture.md) - Intrinsic design
