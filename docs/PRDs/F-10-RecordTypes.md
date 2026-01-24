# F-10: Record Types

> **Sample**: `10_Records` | **Status**: Retrospective | **Category**: Foundation

## 1. Executive Summary

This sample introduces F# record types as structured data with named fields. Records are fundamental for closures, state machines, and typed IPC messages.

**Key Achievement**: Established struct layout computation with GEP-based field access and copy-and-update semantics.

---

## 2. Surface Feature

```fsharp
module RecordTest

type Person = {
    Name: string
    Age: int
}

type Address = {
    Street: string
    City: string
}

type Contact = {
    Person: Person
    Address: Address
}

let alice = { Name = "Alice"; Age = 30 }
let olderAlice = { alice with Age = 31 }

let printPerson (p: Person) =
    Console.writeln $"{p.Name} is {p.Age} years old"

printPerson alice
printPerson olderAlice
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

Field access uses LLVM GEP (GetElementPtr):

```fsharp
p.Name  // Access field at offset 0
p.Age   // Access field at offset 16
```

**MLIR**:
```mlir
// p.Name
%name_ptr = llvm.getelementptr %person[0, 0] : !person_t
%name = llvm.load %name_ptr : !string_t

// p.Age
%age_ptr = llvm.getelementptr %person[0, 1] : !person_t
%age = llvm.load %age_ptr : i64
```

### 3.4 Record Construction

```fsharp
{ Name = "Alice"; Age = 30 }
```

Constructs a struct in field order:

```mlir
%person = llvm.mlir.undef : !person_t
%person1 = llvm.insertvalue %name, %person[0]
%person2 = llvm.insertvalue %age, %person1[1]
```

### 3.5 Copy-and-Update

```fsharp
{ alice with Age = 31 }
```

Copies all fields except the updated ones:

```mlir
// Copy alice
%new = llvm.mlir.undef : !person_t
// Copy Name from alice
%name = llvm.extractvalue %alice[0]
%new1 = llvm.insertvalue %name, %new[0]
// Insert new Age
%new_age = llvm.mlir.constant(31 : i64)
%new2 = llvm.insertvalue %new_age, %new1[1]
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
// String type
!string_t = !llvm.struct<(ptr, i64)>

// Person type
!person_t = !llvm.struct<(struct<(ptr, i64)>, i64)>

// Contact type (nested)
!address_t = !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
!contact_t = !llvm.struct<(struct<(struct<(ptr, i64)>, i64)>,
                           struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>)>
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
/path/to/Firefly compile RecordTest.fidproj
./RecordTest
# Output:
# Alice is 30 years old
# Alice is 31 years old
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
