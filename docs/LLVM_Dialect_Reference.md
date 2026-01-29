# LLVM Dialect Reference - When to Use LLVM vs Standard MLIR

**Date:** January 29, 2026
**Purpose:** Guard rail document - defines when LLVM dialect is appropriate vs standard MLIR
**Audience:** Alex contributors, future maintainers

---

## GOLDEN RULE

> **LLVM dialect should ONLY be used when there is literally no other way to express the operation in standard MLIR.**

If you're about to add an LLVM dialect operation, ask:
1. Does MemRef dialect provide this? (memory operations)
2. Does Arith dialect provide this? (arithmetic, bitwise)
3. Does CF dialect provide this? (control flow)
4. Does Func dialect provide this? (functions)
5. Does Index dialect provide this? (indexing)

**Only proceed with LLVM if all answers are "no".**

---

## PART 1: APPROVED LLVM DIALECT OPERATIONS

These operations have **legitimate LLVM-specific semantics** and should use LLVM dialect:

### 1. Function Calls (C ABI)

**Why LLVM-specific:**
- C calling conventions (SysV AMD64, Win64, AAPCS, etc.)
- Varargs handling
- Function attributes (noreturn, nounwind, etc.)
- Linkage types (external, internal, weak, etc.)

**Operations:**
- `llvm.call` - Direct function call with C ABI
- `llvm.invoke` - Call with exception handling
- `llvm.call_intrinsic` - LLVM intrinsics (e.g., llvm.memcpy)

**Example:**
```mlir
%result = llvm.call @write(%fd, %buf, %count) : (i32, !llvm.ptr, i64) -> i64
```

**Standard alternative:** `func.call` exists but doesn't handle C ABI specifics, varargs, or LLVM attributes.

---

### 2. Function Returns

**Why LLVM-specific:**
- C ABI return value handling
- Struct return optimization (sret)
- Return attributes

**Operations:**
- `llvm.return` - Function return with C ABI

**Example:**
```mlir
llvm.return %value : i64
```

**Standard alternative:** `func.return` exists but doesn't handle C ABI sret or attributes.

---

### 3. Indirect Calls (Function Pointers)

**Why LLVM-specific:**
- Function pointer calls with C ABI
- Calling convention preservation

**Operations:**
- `llvm.call` (indirect) - Call through function pointer

**Example:**
```mlir
%result = llvm.call %funcPtr(%arg0, %arg1) : !llvm.ptr, (i64, i64) -> i64
```

**Standard alternative:** `func.call_indirect` doesn't handle C ABI or calling conventions.

---

### 4. Pointer Type Manipulation

**Why LLVM-specific:**
- LLVM pointer provenance semantics
- Type punning
- Low-level pointer arithmetic

**Operations:**
- `llvm.bitcast` - Type punning (reinterpret bits)
- `llvm.inttoptr` - Integer to pointer (address manipulation)
- `llvm.ptrtoint` - Pointer to integer (address inspection)
- `llvm.addrspacecast` - Address space conversion

**Example:**
```mlir
%ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
%bits = llvm.ptrtoint %ptr : !llvm.ptr to i64
%reinterpreted = llvm.bitcast %ptr : !llvm.ptr to !llvm.ptr<f64>
```

**Standard alternative:** None - MLIR standard dialects use opaque pointers, don't support these conversions.

---

### 5. Struct Layout (C Compatibility)

**Why LLVM-specific:**
- C struct layout rules (padding, alignment, packing)
- ABI compatibility with C libraries
- Struct-by-value semantics

**Operations:**
- `llvm.getelementptr` (for structs) - C struct field addressing with padding
- `llvm.extractvalue` - Extract from aggregate with C layout
- `llvm.insertvalue` - Insert into aggregate with C layout

**Example:**
```mlir
// C struct: struct { i8 a; i64 b; } has padding after 'a'
%field_ptr = llvm.getelementptr %struct_ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, i64)>
%field_val = llvm.extractvalue %struct[1] : !llvm.struct<(i8, i64)>
```

**Standard alternative:** MemRef dialect assumes dense layout, doesn't handle C struct padding.

---

### 6. SSA Phi Nodes

**Why LLVM-specific:**
- Traditional SSA phi nodes (vs block arguments)
- LLVM IR compatibility
- Legacy code generation patterns

**Operations:**
- `llvm.phi` - SSA phi node for control flow merge

**Example:**
```mlir
^bb3:
  %result = llvm.phi [%val1, ^bb1], [%val2, ^bb2] : i64
```

**Standard alternative:** CF dialect uses block arguments (more modern MLIR style):
```mlir
^bb3(%result: i64):
  // %result comes from block args, not phi
```

**Note:** Block arguments are preferred in new code, but phi nodes may be needed for LLVM IR compatibility.

---

### 7. Undefined Values

**Why LLVM-specific:**
- LLVM undefined value semantics
- Optimization opportunities (undef can be any value)
- Freezing undef to prevent poison propagation

**Operations:**
- `llvm.mlir.undef` - Undefined value
- `llvm.freeze` - Freeze undef to arbitrary concrete value

**Example:**
```mlir
%undef = llvm.mlir.undef : i64
%init_struct = llvm.insertvalue %undef, %value[0] : !llvm.struct<(i64, i64)>
```

**Standard alternative:** None - standard MLIR doesn't have undefined value semantics.

---

### 8. Inline Assembly

**Why LLVM-specific:**
- Direct assembly injection
- Platform-specific instructions
- Fine-grained control (atomics, special registers)

**Operations:**
- `llvm.inline_asm` - Inline assembly

**Example:**
```mlir
%result = llvm.inline_asm asm_dialect = att
  "syscall",
  "={rax},{rax},{rdi},{rsi},{rdx}"
  %rax_in, %rdi, %rsi, %rdx : (i64, i64, i64, i64) -> i64
```

**Standard alternative:** None - assembly is inherently target-specific.

---

### 9. Global Variables (C Linkage)

**Why LLVM-specific:**
- C linkage types (internal, external, weak, linkonce)
- Thread-local storage
- Constant vs mutable globals
- Alignment and section attributes

**Operations:**
- `llvm.mlir.global` - Global variable with C semantics

**Example:**
```mlir
llvm.mlir.global internal constant @str("Hello, World!") : !llvm.array<13 x i8>
llvm.mlir.global external thread_local @tls_var : i64
```

**Standard alternative:** `memref.global` exists but doesn't handle C linkage, TLS, or sections.

---

## PART 2: FORBIDDEN - Use Standard MLIR Instead

These operations should **NEVER** use LLVM dialect:

### Memory Operations → MemRef Dialect

| ❌ WRONG (LLVM) | ✅ RIGHT (MemRef) | Reason |
|----------------|-------------------|--------|
| `llvm.alloca` | `memref.alloca` | Stack allocation is universal |
| `llvm.load` | `memref.load` | Memory reads are universal |
| `llvm.store` | `memref.store` | Memory writes are universal |
| `llvm.getelementptr` (arrays) | `memref.subview` + index ops | Array addressing is universal |

**Example:**
```mlir
// WRONG
%ptr = llvm.alloca 1 x i64 : !llvm.ptr
%val = llvm.load %ptr : !llvm.ptr -> i64
llvm.store %new_val, %ptr : i64, !llvm.ptr

// RIGHT
%mem = memref.alloca() : memref<i64>
%val = memref.load %mem[] : memref<i64>
memref.store %new_val, %mem[] : memref<i64>
```

---

### Bitwise Operations → Arith Dialect

| ❌ WRONG (LLVM) | ✅ RIGHT (Arith) | Reason |
|----------------|------------------|--------|
| `llvm.and` | `arith.andi` | Bitwise AND is universal |
| `llvm.or` | `arith.ori` | Bitwise OR is universal |
| `llvm.xor` | `arith.xori` | Bitwise XOR is universal |
| `llvm.shl` | `arith.shli` | Shift left is universal |
| `llvm.lshr` | `arith.shrui` | Logical shift right is universal |
| `llvm.ashr` | `arith.shrsi` | Arithmetic shift right is universal |

**Example:**
```mlir
// WRONG
%result = llvm.and %lhs, %rhs : i64
%shifted = llvm.shl %val, %amount : i64

// RIGHT
%result = arith.andi %lhs, %rhs : i64
%shifted = arith.shli %val, %amount : i64
```

---

### Control Flow → CF Dialect

| ❌ WRONG (LLVM) | ✅ RIGHT (CF) | Reason |
|----------------|--------------|--------|
| `llvm.br` | `cf.br` | Unconditional branch is universal |
| `llvm.cond_br` | `cf.cond_br` | Conditional branch is universal |

**Example:**
```mlir
// WRONG
llvm.br ^bb1
llvm.cond_br %cond, ^bb1, ^bb2

// RIGHT
cf.br ^bb1
cf.cond_br %cond, ^bb1, ^bb2
```

---

### Arithmetic Operations → Arith Dialect

| ❌ WRONG (LLVM) | ✅ RIGHT (Arith) | Reason |
|----------------|------------------|--------|
| `llvm.add` | `arith.addi` | Addition is universal |
| `llvm.sub` | `arith.subi` | Subtraction is universal |
| `llvm.mul` | `arith.muli` | Multiplication is universal |
| `llvm.sdiv` | `arith.divsi` | Signed division is universal |
| `llvm.udiv` | `arith.divui` | Unsigned division is universal |
| `llvm.srem` | `arith.remsi` | Signed remainder is universal |
| `llvm.urem` | `arith.remui` | Unsigned remainder is universal |
| `llvm.icmp` | `arith.cmpi` | Integer comparison is universal |

---

## PART 3: DECISION FLOWCHART

```
Is this operation platform-specific?
│
├─ NO ──→ Use standard MLIR dialect
│         (MemRef, Arith, CF, Func, etc.)
│
└─ YES ──→ Does it involve:
           │
           ├─ C ABI calling conventions? ──→ LLVM (call, return)
           ├─ Pointer provenance/type punning? ──→ LLVM (bitcast, inttoptr, ptrtoint)
           ├─ C struct layout with padding? ──→ LLVM (getelementptr for structs, extractvalue, insertvalue)
           ├─ Inline assembly? ──→ LLVM (inline_asm)
           ├─ C linkage/TLS globals? ──→ LLVM (mlir.global)
           ├─ Undefined value semantics? ──→ LLVM (undef, freeze)
           │
           └─ None of the above ──→ RECONSIDER - probably should be standard MLIR
```

---

## PART 4: GUARD RAILS FOR CODE REVIEW

When reviewing new code that adds LLVM dialect operations, verify:

**✅ Checklist:**
- [ ] The operation is in the approved list (Part 1)
- [ ] There is NO standard MLIR equivalent (checked Part 2)
- [ ] The justification is documented in a comment
- [ ] The operation will ONLY be used when targeting LLVM backends

**❌ Red Flags:**
- Using `llvm.alloca` instead of `memref.alloca`
- Using `llvm.and/or/xor` instead of `arith.andi/ori/xori`
- Using `llvm.br/cond_br` instead of `cf.br/cond_br`
- Adding LLVM operations "because it works"
- Not checking if standard MLIR dialect provides the operation

---

## PART 5: REFERENCE - MLIR Standard Dialects

Quick reference for what standard dialects provide:

### Arith Dialect
- Integer arithmetic: `addi`, `subi`, `muli`, `divsi`, `divui`, `remsi`, `remui`
- Floating-point: `addf`, `subf`, `mulf`, `divf`
- Bitwise: `andi`, `ori`, `xori`, `shli`, `shrui`, `shrsi`
- Comparison: `cmpi`, `cmpf`
- Truncation/extension: `trunci`, `extsi`, `extui`, `sitofp`, `uitofp`, `fptosi`, `fptoui`
- **Reference:** https://mlir.llvm.org/docs/Dialects/ArithOps/

### MemRef Dialect
- Allocation: `alloca`, `alloc`, `dealloc`
- Access: `load`, `store`
- Indexing: `subview`, `view`, `reshape`
- Copying: `copy`, `dma_start`, `dma_wait`
- **Reference:** https://mlir.llvm.org/docs/Dialects/MemRef/

### CF Dialect
- Branching: `br`, `cond_br`
- Switch: `switch`
- Assert: `assert`
- **Reference:** https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/

### Func Dialect
- Functions: `func`, `return`, `call`, `call_indirect`
- Constants: `constant`
- **Reference:** https://mlir.llvm.org/docs/Dialects/Func/

### SCF Dialect (Structured Control Flow)
- Loops: `for`, `while`, `parallel`
- Conditionals: `if`, `else`
- **Reference:** https://mlir.llvm.org/docs/Dialects/SCFDialect/

### Index Dialect
- Index arithmetic: `add`, `sub`, `mul`, `divs`, `divu`
- Comparison: `cmp`
- Casts: `casts`, `castu`
- **Reference:** https://mlir.llvm.org/docs/Dialects/IndexOps/

---

## PART 6: EVOLUTION STRATEGY

As MLIR evolves, operations may migrate:

**Watch for:**
- New standard dialects that cover LLVM-specific operations
- Deprecation of LLVM dialect operations in favor of standard equivalents
- MLIR proposals for C ABI interop (may obviate some LLVM dialect usage)

**When this happens:**
1. Update this document
2. Create migration plan
3. Deprecate old LLVM usage
4. Update all references

**Example:** If MLIR adds a `cabi` dialect for C ABI calls, we could migrate:
```
llvm.call @func(...) : (...) -> ...
  ↓
cabi.call @func(...) convention("sysv_amd64") : (...) -> ...
```

---

## SUMMARY

**The 11 Approved LLVM Operations:**

1. `llvm.call` / `llvm.invoke` - C ABI function calls
2. `llvm.return` - C ABI returns
3. `llvm.bitcast` / `llvm.inttoptr` / `llvm.ptrtoint` - Pointer type manipulation
4. `llvm.getelementptr` (structs only) - C struct layout
5. `llvm.extractvalue` / `llvm.insertvalue` - C aggregate semantics
6. `llvm.phi` - SSA phi nodes (prefer CF block args in new code)
7. `llvm.mlir.undef` / `llvm.freeze` - Undefined value semantics
8. `llvm.inline_asm` - Inline assembly
9. `llvm.mlir.global` - C linkage globals

**Everything else uses standard MLIR dialects.**

---

**Last Updated:** January 29, 2026
**Maintainer:** Firefly Core Team
**Related:** `/home/hhh/.claude/plans/llvm-dialect-cleanup.md`
