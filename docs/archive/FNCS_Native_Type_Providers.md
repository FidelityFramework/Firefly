# Native Type Providers: A Future Direction for FNCS

## Current Status

Type providers are **disabled** in FNCS via `NO_TYPEPROVIDERS` define. This is the correct decision for native compilation because traditional type providers fundamentally depend on .NET runtime reflection:

- `Assembly.UnsafeLoadFrom` - Load provider assemblies at compile time
- `Activator.CreateInstance` - Instantiate provider implementations
- `System.Type` reflection - Introspect provided types and members
- `MaybeNull<'T>` patterns - Handle nullable BCL returns

These mechanisms are incompatible with native compilation and the native type universe.

## The Vision: Native Type Providers

The metaprogramming capability of type providers is valuable. A native-first approach could restore this capability without BCL reflection dependencies.

### Core Insight

Type providers in .NET use reflection because that's how .NET works. But the *purpose* of type providers is:

1. **Read external metadata** (database schemas, API specs, config files)
2. **Generate F# types** at compile time
3. **Provide IntelliSense** during development

None of these require runtime reflection if designed natively.

### Potential Approaches

#### 1. Static Metadata Files

Instead of loading assemblies, read static metadata formats:

```fsharp
// Native type provider using TOML/JSON schema
[<NativeTypeProvider("./schema.toml")>]
type Database = NativeProvider<"./schema.toml">

// schema.toml defines types statically
// [types.User]
// id = "int64"
// name = "string"
// email = "string voption"
```

Compile-time parsing generates types. No reflection needed.

#### 2. Quotation-Based Generators

Use F# quotations as the generation mechanism:

```fsharp
// Provider implemented as quotation transformer
module DatabaseProvider =
    let generateTypes (schemaPath: string) : Quotations.Expr<Type list> =
        let schema = parseSchema schemaPath
        <@@ [ for table in schema.tables ->
                { Name = table.name
                  Fields = [ for col in table.columns ->
                              { Name = col.name; Type = col.type } ] } ] @@>
```

The compiler evaluates quotations at compile time.

#### 3. Active Pattern Providers

Pattern-based type discovery:

```fsharp
// Active pattern that matches schema elements
let (|TableType|_|) (schemaPath: string) (name: string) =
    match parseTable schemaPath name with
    | Some table -> Some (generateRecord table)
    | None -> None

// Usage triggers compile-time generation
match "Users" with
| TableType "./schema.toml" record -> record
| _ -> failwith "Unknown table"
```

#### 4. Staged Metaprogramming (MetaOCaml-style)

Multi-stage programming where code generation is explicit:

```fsharp
// Stage 0: Read schema
let schema = .< parseSchema "./schema.toml" >.

// Stage 1: Generate types
let types = .< for table in ~schema.tables do
                  yield generateType table >.

// Stage 2: Compile generated code
let compiled = .~types
```

### Key Differences from .NET Type Providers

| Aspect | .NET Type Providers | Native Type Providers |
|--------|---------------------|----------------------|
| Metadata source | .NET assemblies | Static files (TOML, JSON, etc.) |
| Generation mechanism | Reflection + Activator | Quotations + compiler evaluation |
| Runtime dependency | Requires provider DLL | Zero runtime dependency |
| Type representation | System.Type | Native type descriptors |
| Nullability | MaybeNull patterns | voption (value option) |

### Implementation Considerations

1. **Schema formats**: Define standard schemas for common use cases
   - Database: Table/column definitions
   - API: OpenAPI/Swagger specs
   - Config: Structured configuration files

2. **Compiler integration**: FNCS would need new infrastructure
   - Schema parser (compile-time only)
   - Type generator from schema
   - IntelliSense integration

3. **Build system**: Schema files become build dependencies
   - Changes to schema trigger recompilation
   - No dynamic loading at runtime

### Use Cases in Fidelity

1. **Platform bindings**: Generate bindings from hardware description files
2. **Wire protocols**: BAREWire schema-driven type generation
3. **Configuration**: Typed config from TOML/JSON
4. **Embedded systems**: Memory-mapped register definitions

### Next Steps

1. Define schema format for common metadata patterns
2. Prototype quotation-based generator
3. Integrate with FNCS type checking pipeline
4. Evaluate performance vs. handwritten types

## References

- MetaOCaml: Multi-stage programming for OCaml
- Template Haskell: Compile-time metaprogramming
- Rust proc macros: Procedural macros for code generation
- C# Source Generators: Compile-time code generation

## Files

- `/home/hhh/repos/fsnative/src/Compiler/FSharpNative.Compiler.Service.fsproj` - NO_TYPEPROVIDERS define
- `/home/hhh/repos/fsnative/src/Compiler/TypedTree/TypeProviders.fs` - Existing TP infrastructure (guarded)

---

*This document captures the architectural decision to disable traditional type providers and outlines a native-first approach for future consideration. The core insight is that metaprogramming capability can be preserved without BCL reflection dependencies.*
