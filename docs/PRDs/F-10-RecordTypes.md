# F-10: Record Types

> **Sample**: `10_Records` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces F# record types as structured data with named fields. Records are fundamental for closures, state machines, and typed IPC messages.

**Key Achievements**:
- Struct layout computation with GEP-based field access
- Copy-and-update semantics (`{ r with Field = value }`)
- Pattern matching: single-field, multi-field, nested, and wildcard patterns
- Guard expressions with pattern-bound variables

---

## 2. Surface Feature

```fsharp
module RecordsSample

type Person = { Name: string; Age: int }
type Address = { Street: string; City: string; Zip: string }
type Contact = { Person: Person; Address: Address; Email: string }

// Construction and field access
let alice = { Name = "Alice"; Age = 30 }
Console.writeln $"Hello, {alice.Name}! You are {alice.Age} years old."

// Copy-and-update
let olderAlice = { alice with Age = alice.Age + 1 }

// Single-field pattern with guard
let ageCategory (p: Person) =
    match p with
    | { Age = a } when a < 18 -> "Minor"
    | { Age = a } when a < 65 -> "Adult"
    | _ -> "Senior"

// Multi-field pattern extraction
let personSummary (p: Person) =
    match p with
    | { Name = n; Age = a } -> $"{n} is {Format.int a}"

// Nested record pattern
let getContactCity (c: Contact) =
    match c with
    | { Address = { City = city } } -> city

// Wildcard with partial extraction
let getPersonName (p: Person) =
    match p with
    | { Name = n; Age = _ } -> n
```

---

## 3. Infrastructure Contributions

### 3.1 Record Representation

Records are structs with named fields at computed offsets:

```
Person Layout
┌─────────────────────────────────────┐
│ Offset 0:  Name (string, 16 bytes) │
│            ├── ptr: i8*            │
│            └── len: i64            │
│ Offset 16: Age (int, 8 bytes)      │
│            └── value: i64          │
└─────────────────────────────────────┘
Total: 24 bytes
```

### 3.2 FNCS Type Definition

```fsharp
// Record type as NTU struct
type RecordDef = {
    Name: string
    Fields: (string * NativeType) list
}

// Person record
RecordDef {
    Name = "Person"
    Fields = [("Name", NTUKind.String); ("Age", NTUKind.Int64)]
}
```

### 3.3 Field Access

Field access uses memref operations (reinterpret_cast for same-type, view for different-type):

```fsharp
p.Name  // Access field at offset 0
p.Age   // Access field at offset 16
```

**MLIR**:
```mlir
// Record is memref<Nxi8> (flat byte buffer, like DU representation)
// p.Name — string field at offset 0
%name_ref = memref.view %person[%c0][] : memref<24xi8> to memref<?xi8>
%name = memref.load %name_ref[%c0] : memref<?xi8>

// p.Age — i64 field at offset 16
%age_ref = memref.view %person[%c16][] : memref<24xi8> to memref<1xi64>
%age = memref.load %age_ref[%c0] : memref<1xi64>
```

### 3.4 Record Construction

```fsharp
{ Name = "Alice"; Age = 30 }
```

Constructs a record in field order using a flat byte buffer:

```mlir
// Record is memref<Nxi8> — fields at known byte offsets
%person = memref.alloca() : memref<Nxi8>
// Store Name field (memref<?xi8>) at offset 0
%name_ref = memref.view %person[%c0][] : memref<Nxi8> to memref<1xindex>
memref.store %name, %name_ref[%c0] : memref<1xindex>
// Store Age field (i64) at offset after Name
%age_ref = memref.view %person[%name_end][] : memref<Nxi8> to memref<1xi64>
memref.store %age, %age_ref[%c0] : memref<1xi64>
```

### 3.5 Copy-and-Update

```fsharp
{ alice with Age = 31 }
```

Copies all fields except the updated ones:

```mlir
// Allocate new record
%new = memref.alloca() : memref<Nxi8>
// Copy Name from alice (read then write at same offset)
%alice_name_ref = memref.view %alice[%c0][] : memref<Nxi8> to memref<1xindex>
%name = memref.load %alice_name_ref[%c0] : memref<1xindex>
%new_name_ref = memref.view %new[%c0][] : memref<Nxi8> to memref<1xindex>
memref.store %name, %new_name_ref[%c0] : memref<1xindex>
// Insert new Age
%new_age = arith.constant 31 : i64
%new_age_ref = memref.view %new[%name_end][] : memref<Nxi8> to memref<1xi64>
memref.store %new_age, %new_age_ref[%c0] : memref<1xi64>
```

### 3.6 Nested Records

```fsharp
type Contact = {
    Person: Person
    Address: Address
}
```

**Layout**:
```
Contact Layout
┌─────────────────────────────────────┐
│ Offset 0:  Person (24 bytes)       │
│            └── embedded Person     │
│ Offset 24: Address (32 bytes)      │
│            └── embedded Address    │
└─────────────────────────────────────┘
Total: 56 bytes
```

### 3.7 Pattern Matching on Records

Record patterns support field extraction, nested patterns, and wildcards.

#### Single-Field Pattern with Guard

```fsharp
match person with
| { Age = a } when a >= 18 -> "Adult"
| _ -> "Minor"
```

Baker lowers to field extraction + guard:

```
IfThenElse
├── Let a = person.Age
├── Condition: a >= 18
├── Then: "Adult"
└── Else: "Minor"
```

#### Multi-Field Pattern Extraction

```fsharp
let personSummary (p: Person) : string =
    match p with
    | { Name = n; Age = a } -> $"{n} is {Format.int a}"
```

Multiple fields are extracted in sequence:

```
Let n = p.Name
Let a = p.Age
InterpolatedString [n; " is "; a]
```

#### Nested Record Patterns

```fsharp
let getContactCity (c: Contact) : string =
    match c with
    | { Address = { City = city } } -> city
```

Nested patterns recurse through field access:

```
Let addr = c.Address
Let city = addr.City
city
```

**MLIR**:
```mlir
// Access nested field: c.Address.City
// View Address field within Contact (at Address byte offset)
%addr_ref = memref.view %contact[%addr_offset][] : memref<Nxi8> to memref<Mxi8>
// View City field within Address (at City byte offset within Address)
%city_ref = memref.view %addr_ref[%city_offset][] : memref<Mxi8> to memref<1xindex>
%city = memref.load %city_ref[%c0] : memref<1xindex>
```

#### Wildcard with Partial Extraction

```fsharp
let getPersonName (p: Person) : string =
    match p with
    | { Name = n; Age = _ } -> n
```

Wildcards are optimized away - no `FieldGet` is emitted for ignored fields:

```
Let n = p.Name
n
```

**Architectural Note**: Wildcard handling is a principled optimization. Creating a `FieldGet` for a wildcard would produce an orphaned node (no binding consumes it). Baker detects `Pattern.Wildcard` and skips emission.

---

## 4. PSG Representation

```
ModuleOrNamespace: RecordTest
├── TypeDefinition: Person (Record)
│   ├── Field: Name (string)
│   └── Field: Age (int)
│
├── TypeDefinition: Address (Record)
├── TypeDefinition: Contact (Record)
│
├── LetBinding: alice
│   └── RecordConstruction
│       ├── Field: Name = "Alice"
│       └── Field: Age = 30
│
├── LetBinding: olderAlice
│   └── RecordCopyUpdate
│       ├── Source: alice
│       └── Updates: Age = 31
│
├── LetBinding: printPerson
│   └── Lambda: p ->
│       └── Application: Console.writeln
│           └── InterpolatedString
│               ├── FieldAccess: p.Name
│               └── FieldAccess: p.Age
│
├── Application: printPerson alice
└── Application: printPerson olderAlice
```

---

## 5. MLIR Type Definitions

```mlir
// All record/struct types are flat byte buffers (memref<Nxi8>).
// Fields are accessed at known byte offsets via memref.view.
// Strings are memref<?xi8>. Integers are i32/i64. Nested records are inlined.
//
// Person: memref<Nxi8> where N = sizeof(index) + sizeof(i64)
//   offset 0: Name (index — pointer to memref<?xi8>)
//   offset 8: Age (i64)
//
// Address: memref<Mxi8> where M = 2 * sizeof(index)
//   offset 0: Street (index)
//   offset 8: City (index)
//
// Contact: memref<Kxi8> where K = N + M (Person + Address inlined)
```

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for record values and field accesses |
| RecordLayout | Pre-computed field offsets and alignments |

---

## 7. Validation

```bash
cd samples/console/FidelityHelloWorld/10_Records
/path/to/Firefly compile Records.fidproj
./target/records
```

**Expected Output**:
```
=== Records Test ===
Hello, Alice! You are 30 years old.
After birthday: Hello, Alice! You are 31 years old.
Name field: Alice
Age field: 30
Category: Adult
Contact: Bob, Springfield - bob@example.com

=== Age Categories ===
Age 10: Minor
Age 35: Adult
Age 70: Senior

=== Multi-field Pattern ===
Summary: Alice is 30

=== Nested Pattern ===
Contact city: Springfield
Name only: Diana
```

---

## 8. Relationship to Closures

Records are the structural foundation for closures (C-01):

| Closure Aspect | Record Equivalent |
|----------------|-------------------|
| Environment | Record fields |
| Captures | Field values |
| Code pointer | Additional field |

A closure `{code_ptr, cap0, cap1, ...}` is essentially a record with a function pointer.

---

## 9. Architectural Lessons

1. **Struct Layout Computation**: Compiler determines field offsets, not runtime
2. **GEP for Access**: Named fields become numeric offsets
3. **Copy Semantics**: Records are values, copy-and-update is literal copying
4. **Composition Ready**: Records enable closures, state machines, messages

---

## 10. Downstream Dependencies

This sample's infrastructure enables:
- **C-01**: Closure environments (flat records with captures)
- **C-05, C-06**: State machine states (Lazy, Seq)
- **T-03, T-04, T-05**: Actor messages (typed records)

---

## 11. Related Documents

- [F-05-DiscriminatedUnions](F-05-DiscriminatedUnions.md) - Sum types complement records
- [C-01-Closures](C-01-Closures.md) - Closures use record-like layout
- [T-03-BasicActor](T-03-BasicActor.md) - Actors use record messages
