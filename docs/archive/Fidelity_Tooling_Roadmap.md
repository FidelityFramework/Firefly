# Fidelity Tooling Roadmap

## Introduction

If you've developed in F# with Visual Studio Code, you've used Ionide. You hover over a symbol and see its type. You press Ctrl+Click and jump to a definition. You see red squiggles under errors before you even save the file. This is the modern developer experience, and it's powered by a sophisticated stack of tooling that most developers never think about.

Fidelity will need its own tooling stack. We cannot simply reuse Ionide because Ionide is built on FSharp.Compiler.Service (FCS), which assumes .NET types and .NET semantics. When you hover over a `string` in Ionide, you see BCL semantics (UTF-16, heap-allocated, garbage collected). In Fidelity, that same `string` has native semantics: UTF-8 encoded, deterministically managed, fat pointer representation. The type *name* stays `string` - users shouldn't need to learn new type names - but the *semantics* are fundamentally different. The tooling must understand this difference.

This document explains what Fidelity tooling would need to provide, why existing tooling doesn't work directly, and how we plan to build a complete development environment for F# Native. Along the way, we explain the underlying concepts for developers who haven't needed to think about language server protocols or compiler services before.

This is a forward-looking design document. The implementation work lies ahead.

---

## Part 1: Understanding F# Tooling Today

Before we can build new tooling, we need to understand how existing F# tooling works.

### What Happens When You Open an F# File

When you open an F# file in VS Code with Ionide installed, a remarkable amount of machinery springs into action:

1. **Ionide Extension Activates**: VS Code loads the Ionide extension (JavaScript/TypeScript code running in VS Code's extension host)

2. **Language Server Starts**: Ionide spawns a separate process running FsAutoComplete (FSAC), a program written in F# that provides language intelligence

3. **Project Loading**: FSAC reads your `.fsproj` file using Ionide.ProjInfo, which invokes MSBuild to understand your project structure, references, and compilation options

4. **Initial Analysis**: FSAC uses FSharp.Compiler.Service to parse and type-check your files, building an in-memory representation of your code

5. **Communication Begins**: Ionide and FSAC communicate over the Language Server Protocol (LSP), a standardized JSON-RPC protocol for editor-language tool communication

6. **UI Updates**: As FSAC reports diagnostics, symbols, and type information, Ionide updates the VS Code UI. Squiggles appear, the outline view populates, hover information becomes available.

This happens in seconds, and then continues as you edit. Every keystroke can trigger re-analysis, new diagnostics, and updated completions.

### The Language Server Protocol

The Language Server Protocol (LSP) is a crucial piece of modern tooling infrastructure. Before LSP, every editor needed custom integration with every language. Vim had its own Rust plugin, VS Code had its own Rust plugin, Emacs had its own Rust plugin, and they all reimplemented similar logic.

LSP standardizes this. A language server is a separate process that receives notifications about file changes, responds to requests for hover information, completions, and definitions, sends diagnostics (errors and warnings) to the editor, and handles refactoring operations like rename.

Any editor that speaks LSP can use any language server. This is why you can use Ionide with VS Code, Vim (via Ionide-vim), Emacs, or any other LSP-capable editor.

The protocol looks like this (simplified):

```json
// Editor → Server: "What's the type at position 10:5 in Main.fs?"
{
    "method": "textDocument/hover",
    "params": {
        "textDocument": { "uri": "file:///path/to/Main.fs" },
        "position": { "line": 10, "character": 5 }
    }
}

// Server → Editor: "It's a function from string to int"
{
    "result": {
        "contents": {
            "kind": "markdown",
            "value": "```fsharp\nval myFunction: string -> int\n```"
        }
    }
}
```

### The F# Compiler Services (FCS)

At the heart of F# tooling is FSharp.Compiler.Service, a library form of the F# compiler. It provides parsing (converting F# source text to Abstract Syntax Trees), type checking (inferring types, resolving overloads, checking constraints), symbol resolution (finding definitions, references, implementations), and semantic information (what is this identifier, what are its members).

FCS is the same code that compiles your F# programs. This is important. It means the tooling and the compiler have identical understanding of your code. If FCS says a symbol is a string, the compiler will treat it as a string.

### The Tooling Stack

Here's the complete picture:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         F# Tooling Architecture                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          User Interface                              │   │
│  │  ┌─────────────────┐ ┌───────────────┐ ┌─────────────────────────┐  │   │
│  │  │ VS Code + Ionide │ │ Vim + LSP    │ │ Emacs + eglot          │  │   │
│  │  │                  │ │              │ │                        │  │   │
│  │  │ Syntax highlight │ │ Same features│ │ Same features          │  │   │
│  │  │ Hover info       │ │ via LSP      │ │ via LSP                │  │   │
│  │  │ Go to definition │ │              │ │                        │  │   │
│  │  │ Error squiggles  │ │              │ │                        │  │   │
│  │  └────────┬─────────┘ └──────┬───────┘ └───────────┬─────────────┘  │   │
│  │           │                  │                     │                │   │
│  │           └──────────────────┼─────────────────────┘                │   │
│  │                              │                                      │   │
│  │                              ▼ LSP (JSON-RPC)                       │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                       FsAutoComplete (FSAC)                          │   │
│  │                                                                      │   │
│  │  A long-running process that provides language intelligence          │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ LSP Server Implementation (Ionide.LanguageServerProtocol)      │ │   │
│  │  │ - Receives requests from editors                               │ │   │
│  │  │ - Routes to appropriate handlers                               │ │   │
│  │  │ - Formats and sends responses                                  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ Analysis Engine                                                 │ │   │
│  │  │                                                                 │ │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌───────────────────────┐  │ │   │
│  │  │  │ Ionide       │ │ FSharp       │ │ Additional Tools      │  │ │   │
│  │  │  │ .ProjInfo    │ │ .Compiler    │ │                       │  │ │   │
│  │  │  │              │ │ .Service     │ │ FSharpLint            │  │ │   │
│  │  │  │ Loads .fsproj│ │              │ │ Fantomas (formatting) │  │ │   │
│  │  │  │ via MSBuild  │ │ Type checks  │ │ Analyzers SDK         │  │ │   │
│  │  │  │              │ │ Resolves     │ │                       │  │ │   │
│  │  │  │ Returns file │ │ symbols      │ │                       │  │ │   │
│  │  │  │ list, refs   │ │              │ │                       │  │ │   │
│  │  │  └──────────────┘ └──────────────┘ └───────────────────────┘  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Why Fidelity Would Need Its Own Tooling

Given the sophistication of existing F# tooling, why can't we just use it? The answer lies in the fundamental differences between .NET F# and F# Native.

### The Type System Difference

Consider this simple code:

```fsharp
let greeting = "Hello, World!"
```

In standard F#, `greeting` has type `string`, which is `System.String`. This type is UTF-16 encoded, garbage collected, a reference type (lives on the heap), and immutable (contents can't change, but new strings can be created).

In Fidelity, `greeting` has type `string` with **native semantics**. This type is UTF-8 encoded, deterministically managed (no GC), a fat pointer (pointer + length), and lives wherever the memory model specifies (stack, arena, etc.). The type NAME is still `string` - we don't burden users with learning internal type names.

If you use Ionide with Fidelity code, it will tell you `greeting: string` with BCL semantics. Fidelity tooling would show `greeting: string` but understand the native semantics - UTF-8 representation, memory region, deterministic lifetime.

### The Option Type Example

The difference is even more significant with option types:

```fsharp
let maybeValue = Some 42
```

In standard F#, the type is `int option`. It's a reference type, allocated on the heap. `None` is a singleton object. `Some 42` allocates a heap object.

In Fidelity, the type is still written as `int option` with `Some`/`None` - same F# syntax users expect. But it has **native semantics**: a value type (like `voption`) that lives on the stack. `None` is a struct with a tag bit. `Some 42` is a struct with no allocation.

This isn't just a display difference. It affects how you think about the code. In Fidelity, creating a `Some` doesn't allocate. Returning `None` doesn't create a reference. The memory model is fundamentally different, but the syntax and type names remain familiar F#.

### SRTP Resolution

Statically Resolved Type Parameters (SRTP) behave the same syntactically but resolve differently:

```fsharp
let inline double x = x + x
```

In standard F#, `(+)` resolves against .NET method tables. The compiler finds `System.Int32.op_Addition` for integers, `System.String.Concat` for strings, and so on.

In Fidelity, `(+)` would resolve against Alloy's witness hierarchy. The compiler would find `Alloy.BasicOps.add` witnesses that provide native implementations.

When you hover over `double`, the tooling should show the constraint. But it would need to show resolution against Alloy witnesses, not .NET types.

### Project Format

Standard F# uses `.fsproj` files with MSBuild:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="Program.fs" />
  </ItemGroup>
  <ItemGroup>
    <PackageReference Include="FSharp.Core" Version="8.0.0" />
  </ItemGroup>
</Project>
```

Fidelity uses `.fidproj` files with TOML:

```toml
[package]
name = "my_app"
version = "1.0.0"

[compilation]
memory_model = "stack_only"
target = "native"

[dependencies]
alloy = { path = "/path/to/Alloy/src" }

[build]
sources = ["Main.fs"]
output = "my_app"
output_kind = "console"
```

Ionide.ProjInfo doesn't understand `.fidproj`. It expects MSBuild, SDK resolution, and NuGet packages. None of that applies to Fidelity.

### The Bottom Line

We would need tooling that understands Fidelity's native type semantics, shows types with correct semantic understanding (`string` as UTF-8 fat pointer, `option` as value type, etc.), resolves SRTP against Alloy witnesses, loads `.fidproj` project files, and reports Fidelity-specific diagnostics (FS8xxx codes).

The type NAMES stay familiar F# (`string`, `option`, `array`). The SEMANTICS are native. The tooling must understand this distinction.

This means building a new language server that uses fsnative (FNCS) instead of FCS.

---

## Part 3: The Proposed Fidelity Tooling Architecture

Here's what we anticipate building:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Fidelity Tooling Architecture                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          User Interface                              │   │
│  │  ┌─────────────────────┐ ┌───────────────┐ ┌──────────────────────┐ │   │
│  │  │ VS Code             │ │ Neovim        │ │ Other LSP Editors    │ │   │
│  │  │ + Fidelity Extension│ │ + Fidelity    │ │                      │ │   │
│  │  │                     │ │   plugin      │ │ Emacs, Helix, etc.   │ │   │
│  │  │ Reuses F# syntax    │ │               │ │                      │ │   │
│  │  │ (language unchanged)│ │ Uses built-in │ │ Standard LSP works   │ │   │
│  │  │                     │ │ LSP client    │ │                      │ │   │
│  │  └─────────┬───────────┘ └───────┬───────┘ └───────────┬──────────┘ │   │
│  │            │                     │                     │            │   │
│  │            └─────────────────────┼─────────────────────┘            │   │
│  │                                  │                                  │   │
│  │                                  ▼ LSP (JSON-RPC)                   │   │
│  └──────────────────────────────────┼──────────────────────────────────┘   │
│                                     │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │                  FsNativeAutoComplete (FSNAC)                        │   │
│  │                                                                      │   │
│  │  The language server for Fidelity development                        │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ LSP Server (Ionide.LanguageServerProtocol)                     │ │   │
│  │  │                                                                 │ │   │
│  │  │ We would reuse Ionide's LSP library directly. It's generic     │ │   │
│  │  │ infrastructure that handles the JSON-RPC protocol.             │ │   │
│  │  │ Only the handlers that call into the compiler would differ.    │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ FSNAC Core                                                      │ │   │
│  │  │                                                                 │ │   │
│  │  │  ┌──────────────────┐ ┌──────────────────────────────────────┐ │ │   │
│  │  │  │ Fidelity.ProjInfo│ │ fsnative (FNCS)                      │ │ │   │
│  │  │  │                  │ │                                      │ │ │   │
│  │  │  │ Loads .fidproj   │ │ Native type semantics                │ │ │   │
│  │  │  │ (TOML parsing)   │ │ string/option/array as native       │ │ │   │
│  │  │  │                  │ │                                      │ │ │   │
│  │  │  │ No MSBuild       │ │ SRTP against Alloy witnesses         │ │ │   │
│  │  │  │ No NuGet         │ │                                      │ │ │   │
│  │  │  │ Much simpler     │ │ Memory region tracking               │ │ │   │
│  │  │  │                  │ │ Access kind enforcement              │ │ │   │
│  │  │  └──────────────────┘ └──────────────────────────────────────┘ │ │   │
│  │  │                                                                 │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ Future: Fidelity.Analyzers.SDK                           │  │ │   │
│  │  │  │                                                          │  │ │   │
│  │  │  │ Custom analyzers for:                                    │  │ │   │
│  │  │  │ - Memory safety (lifetime violations)                    │  │ │   │
│  │  │  │ - Region misuse (stack escape, etc.)                     │  │ │   │
│  │  │  │ - Platform binding validation                            │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### What We Would Build New

**FsNativeAutoComplete (FSNAC)**: A new language server that uses FNCS instead of FCS. This would be the main effort.

**Fidelity.ProjInfo**: A simple TOML parser for `.fidproj` files. Much simpler than Ionide.ProjInfo because we don't have MSBuild complexity.

**Fidelity-vscode**: A VS Code extension that connects to FSNAC. Simpler than Ionide because we would delegate most work to the LSP server.

**Fidelity-vim**: A Neovim plugin that configures the built-in LSP client to use FSNAC.

### What We Would Reuse

**Ionide.LanguageServerProtocol**: The F# library that implements the LSP protocol. This is generic infrastructure. It handles JSON-RPC serialization, message routing, and the protocol mechanics. We expect to use it directly.

**ionide-fsgrammar**: The TextMate grammar for F# syntax highlighting. Since Fidelity uses F# syntax (just with different semantics), we could reuse syntax highlighting unchanged.

**Fantomas**: The F# code formatter. It operates on syntax, not semantics, so it should work with Fidelity code unchanged.

---

## Part 4: Why .NET-Hosted Tooling First

You might wonder: "If Fidelity compiles to native, shouldn't the tooling be native too?"

This is a reasonable question, and the answer is nuanced. We believe tooling should be .NET-hosted initially, with self-hosting as a long-term goal.

### The Bootstrap Problem

Consider what it takes to build a language server. You need to parse F# code, type-check it, respond to LSP requests, and handle file I/O, JSON serialization, and networking.

FNCS (the fsnative compiler services) is written in F# and runs on .NET. To build FSNAC, we would need to call FNCS APIs. If FSNAC must be native, we would need to either make FNCS itself native-compilable (a massive effort) or build a separate native implementation of type checking (duplicated effort).

Neither makes sense when we can just run FSNAC on .NET.

### The rust-analyzer Precedent

The Rust community faced this exact question with rust-analyzer (the Rust language server). rust-analyzer is written in Rust, which compiles to native code. But that's somewhat incidental. It's written in Rust because Rust is a good language for this kind of software, and because it can share code with the Rust compiler.

What we take from this: rust-analyzer being native isn't why it's good. It's good because it's well-designed, fast, and accurate. If rust-analyzer were written in Java but provided the same functionality at the same speed, users wouldn't care.

Similarly, FSNAC running on .NET shouldn't harm the developer experience. What would matter is that it provides accurate type information (using FNCS), responds quickly to requests, and understands Fidelity's semantics.

### The Self-Hosting Path

That said, self-hosting (building Fidelity tools with Fidelity) is a worthy long-term goal. Using Fidelity to build Fidelity tools would validate the platform. Native tooling would be easier to distribute (no .NET SDK required). And it would demonstrate that Fidelity can build real-world tools.

Here's how we envision the path:

```
Phase 1: .NET-Hosted (Current Target)
─────────────────────────────────────
FSNAC runs on .NET
Uses FNCS (which is .NET-based)
Editor plugins communicate via LSP
Firefly compiles user code to native, but tooling runs on .NET

Phase 2: Partial Self-Hosting (Future)
──────────────────────────────────────
Core analysis logic compiled by Firefly
Type resolution, SRTP solving done in native code
LSP layer still .NET (JSON-RPC is convenient in .NET)

Phase 3: Full Self-Hosting (Distant Future)
───────────────────────────────────────────
FSNAC is a Fidelity-compiled native binary
No .NET runtime required
Single executable distribution
```

What we take from this: Phase 1 would give developers a working environment. Phase 2 and 3 are optimization and validation.

---

## Part 5: Fidelity.ProjInfo and Why It Would Be Simple

One of the nice aspects of Fidelity tooling is how much simpler project loading could become.

### The MSBuild Complexity

Ionide.ProjInfo has to deal with MSBuild, which is extraordinarily complex. It must handle SDK resolution (finding the right .NET SDK version), project evaluation (running MSBuild's evaluation phase to expand properties and items), NuGet restoration (downloading packages, resolving dependencies), target framework negotiation (handling multi-targeting, conditional compilation), import resolution (processing `<Import>` statements, props/targets files), and reference resolution (turning package references into actual file paths).

This is necessary because `.fsproj` files are essentially programs written in a declarative configuration language. They can have conditions, imports, and computed values. They require execution to understand.

### The Fidelity Simplicity

`.fidproj` files are just data. TOML is a straightforward configuration format:

```toml
[package]
name = "my_app"
version = "1.0.0"

[compilation]
memory_model = "stack_only"
target = "native"

[dependencies]
alloy = { path = "/home/hhh/repos/Alloy/src" }

[build]
sources = ["Main.fs", "Helpers.fs"]
output = "my_app"
output_kind = "console"
```

Parsing this would be trivial. Read the file, parse TOML (standard library), extract the fields, done.

No SDK resolution (there's no SDK; Firefly is the compiler). No NuGet (dependencies are paths, or will use a simpler package format). No target framework (we target native). No complex import machinery.

```fsharp
module Fidelity.ProjInfo

open Tomlyn

type MemoryModel = StackOnly | ArenaOwned | Full
type OutputKind = Console | Freestanding | Library

type FidProject = {
    Name: string
    Version: string option
    MemoryModel: MemoryModel
    Target: string
    Dependencies: Map<string, DependencySpec>
    Sources: string list
    Output: string
    OutputKind: OutputKind
}

let load (path: string) : Result<FidProject, string> =
    let content = File.ReadAllText path
    let doc = Toml.Parse content

    // Extract fields directly - no evaluation needed
    let package = doc.["package"]
    let compilation = doc.["compilation"]
    let build = doc.["build"]

    Ok {
        Name = package.["name"].AsString
        Version = package.TryGet("version") |> Option.map (fun v -> v.AsString)
        MemoryModel = parseMemoryModel compilation.["memory_model"].AsString
        Target = compilation.["target"].AsString
        Dependencies = parseDependencies doc.["dependencies"]
        Sources = build.["sources"].AsArray |> Seq.map (fun s -> s.AsString) |> List.ofSeq
        Output = build.["output"].AsString
        OutputKind = parseOutputKind build.["output_kind"].AsString
    }
```

This could be the entire project loading logic. Compare to Ionide.ProjInfo's thousands of lines dealing with MSBuild's complexity.

---

## Part 6: FSNAC Design

FsNativeAutoComplete would be the heart of Fidelity tooling. Here's how we anticipate it would work.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            FSNAC Architecture                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         LSP Protocol Layer                             │ │
│  │                    (Ionide.LanguageServerProtocol)                     │ │
│  │                                                                        │ │
│  │  Receives:                          Sends:                             │ │
│  │  - textDocument/hover               - textDocument/publishDiagnostics  │ │
│  │  - textDocument/definition          - window/showMessage               │ │
│  │  - textDocument/completion          - etc.                             │ │
│  │  - textDocument/references                                             │ │
│  │  - etc.                                                                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          Request Handlers                              │ │
│  │                                                                        │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │ HoverHandler │ │ DefHandler   │ │ CompHandler  │ │ DiagHandler  │  │ │
│  │  │              │ │              │ │              │ │              │  │ │
│  │  │ Get symbol   │ │ Find def     │ │ Get options  │ │ Get errors   │  │ │
│  │  │ Format type  │ │ at position  │ │ at position  │ │ Format them  │  │ │
│  │  │ Return MD    │ │ Return loc   │ │ Return list  │ │ Publish      │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │                                                                        │ │
│  │  Handlers would call into FNCS Integration Layer                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      FNCS Integration Layer                            │ │
│  │                                                                        │ │
│  │  Wraps fsnative (FNCS) APIs in a convenient interface                 │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Project State                                                     │ │ │
│  │  │                                                                   │ │ │
│  │  │  Loaded projects: Map<projectPath, FidProject>                   │ │ │
│  │  │  File contents: Map<filePath, sourceText * version>              │ │ │
│  │  │  Checked files: Map<filePath, FNCSCheckResults>                  │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Analysis Functions                                                │ │ │
│  │  │                                                                   │ │ │
│  │  │  getTypeAtPosition: file * pos -> NativeType option              │ │ │
│  │  │  getSymbolAtPosition: file * pos -> FNCSSymbol option            │ │ │
│  │  │  getDefinition: symbol -> location option                        │ │ │
│  │  │  getReferences: symbol -> location list                          │ │ │
│  │  │  getDiagnostics: file -> FNCSError list                          │ │ │
│  │  │  getCompletions: file * pos -> CompletionItem list               │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         fsnative (FNCS)                                │ │
│  │                                                                        │ │
│  │  F# Native Compiler Services - the compiler as a library              │ │
│  │                                                                        │ │
│  │  - Parsing (same as FCS - F# syntax unchanged)                        │ │
│  │  - Type checking with native types                                    │ │
│  │  - SRTP resolution against Alloy witnesses                            │ │
│  │  - Symbol resolution and semantic information                         │ │
│  │  - Diagnostic generation (including FS8xxx native-specific)           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Expected Differences from FSAC

| Aspect | FSAC | FSNAC |
|--------|------|-------|
| **Compiler backend** | FSharp.Compiler.Service | fsnative (FNCS) |
| **Project files** | `.fsproj` (MSBuild) | `.fidproj` (TOML) |
| **Type display** | Types with BCL semantics | Same type names, native semantics |
| **String literal type** | `string` (UTF-16, heap) | `string` (UTF-8, fat pointer) |
| **Option type** | `'T option` (reference, heap) | `'T option` (value, stack) |
| **SRTP resolution** | .NET method tables | Alloy witness hierarchy |
| **Diagnostics** | FS0001-FS9999 | FS0001-FS7999 + FS8xxx (native-specific) |
| **Memory info** | N/A | Regions, access kinds |

### Native Type Display

When you hover over a value, FSNAC would format its type for display. This would include native-specific information:

```fsharp
module NativeTypeDisplay =

    /// Format type for hover display - uses standard F# type names
    let formatType (typ: NativeType) : string =
        match typ with
        | NativeType.Int32 -> "int"
        | NativeType.Int64 -> "int64"
        | NativeType.Float64 -> "float"
        | NativeType.String -> "string"  // Still shows "string" - native semantics are understood
        | NativeType.Option inner -> sprintf "%s option" (formatType inner)  // Same name, value semantics
        | NativeType.Array (elem, region) ->
            sprintf "%s[] <%s>" (formatType elem) (formatRegion region)
        | NativeType.Ptr (pointee, region, access) ->
            sprintf "Ptr<%s, %s, %s>" (formatType pointee) (formatRegion region) (formatAccess access)
        | NativeType.Function (args, ret) ->
            let argStr = args |> List.map formatType |> String.concat " -> "
            sprintf "%s -> %s" argStr (formatType ret)
        | NativeType.Record name -> name
        | NativeType.Union name -> name
        | _ -> typ.ToString()

    let formatRegion (region: MemoryRegion) : string =
        match region with
        | MemoryRegion.Stack -> "stack"
        | MemoryRegion.Heap -> "heap"
        | MemoryRegion.Arena name -> sprintf "arena:%s" name
        | MemoryRegion.Peripheral -> "peripheral"
        | MemoryRegion.Flash -> "flash"

    let formatAccess (access: AccessKind) : string =
        match access with
        | AccessKind.ReadOnly -> "ro"
        | AccessKind.WriteOnly -> "wo"
        | AccessKind.ReadWrite -> "rw"
```

When you hover over a pointer variable, you might see:

```
val gpioPtr: Ptr<uint32, peripheral, rw>
```

This would tell you it's a pointer to a 32-bit unsigned integer, in peripheral memory (memory-mapped I/O), with read-write access. This kind of information would be crucial for embedded development.

### Memory Region Information

FSNAC could provide additional information about memory layout:

```
type Person = {
    Name: string  // offset 0, size 16 (UTF-8 fat pointer: ptr + len)
    Age: int      // offset 16, size 4
}
// Total size: 24 bytes (with padding)
// Alignment: 8 bytes
```

This visibility into memory layout would help developers understand the native representation of their types.

---

## Part 7: Editor Extensions

### Fidelity-vscode

The VS Code extension would be relatively thin because most intelligence would come from the LSP server.

**What it would provide:**

1. **Syntax Highlighting**: Reuses ionide-fsgrammar (F# syntax unchanged)
2. **LSP Client**: Connects to FSNAC, routes requests/responses
3. **Commands**: Build with Firefly, run native binary
4. **Status Bar**: Shows project info, FSNAC status
5. **Configuration**: Extension settings for FSNAC path, Firefly path, etc.

**Structure:**

```
fidelity-vscode/
├── src/
│   ├── extension.ts           # Extension entry point
│   ├── client.ts              # LSP client configuration
│   ├── commands/
│   │   ├── build.ts           # Invoke Firefly
│   │   ├── run.ts             # Run native binary
│   │   └── clean.ts           # Clean build artifacts
│   └── views/
│       └── status.ts          # Status bar item
├── syntaxes/
│   └── fsharp.tmLanguage.json # Symlink to ionide-fsgrammar
├── package.json               # Extension manifest
└── tsconfig.json
```

**Example command (build):**

```typescript
import * as vscode from 'vscode';
import * as path from 'path';
import { spawn } from 'child_process';

export async function buildProject() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    // Find .fidproj file
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    if (!workspaceFolder) return;

    const fidprojFiles = await vscode.workspace.findFiles(
        new vscode.RelativePattern(workspaceFolder, '*.fidproj'),
        null, 1
    );

    if (fidprojFiles.length === 0) {
        vscode.window.showErrorMessage('No .fidproj file found');
        return;
    }

    // Run Firefly
    const fireflyPath = vscode.workspace.getConfiguration('fidelity').get<string>('fireflyPath', 'Firefly');
    const terminal = vscode.window.createTerminal('Fidelity Build');
    terminal.sendText(`${fireflyPath} compile "${fidprojFiles[0].fsPath}"`);
    terminal.show();
}
```

### Fidelity-vim

For Vim/Neovim users, we would provide a minimal plugin that configures LSP:

```lua
-- lua/fidelity/init.lua

local M = {}

function M.setup(opts)
    opts = opts or {}

    -- Path to FSNAC binary
    local fsnac_path = opts.fsnac_path or "fsnac"

    -- Configure LSP
    local lspconfig = require('lspconfig')
    local configs = require('lspconfig.configs')

    if not configs.fsnac then
        configs.fsnac = {
            default_config = {
                cmd = { fsnac_path },
                filetypes = { 'fsharp' },
                root_dir = function(fname)
                    return lspconfig.util.root_pattern('*.fidproj')(fname)
                        or lspconfig.util.find_git_ancestor(fname)
                end,
                settings = {},
            },
        }
    end

    lspconfig.fsnac.setup({
        on_attach = function(client, bufnr)
            -- Standard LSP keybindings
            local opts = { buffer = bufnr, noremap = true, silent = true }
            vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
            vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
            vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
            vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, opts)

            -- Fidelity-specific
            vim.keymap.set('n', '<leader>fb', ':!Firefly compile *.fidproj<CR>', opts)
        end,
    })
end

return M
```

Usage in user's Neovim config:

```lua
require('fidelity').setup({
    fsnac_path = '/home/user/.local/bin/fsnac'
})
```

Neovim's built-in LSP client handles the protocol; we would just configure it to use FSNAC.

---

## Part 8: Implementation Plan

### Phase 1: Minimal Viable FSNAC

**Goal**: Basic LSP server with hover and go-to-definition

**Tasks**:
1. Create fsnac repository with F# project structure
2. Add Ionide.LanguageServerProtocol dependency
3. Implement project loading (Fidelity.ProjInfo)
4. Create FNCS integration wrapper
5. Implement hover handler (show native types)
6. Implement go-to-definition handler

**Validation**:
- Use VS Code's generic LSP extension to connect to FSNAC
- Open a Fidelity project
- Hover over `let x = "hello"` and see `val x: string` (with native semantics understood by tooling)
- Ctrl+Click on a function call, jump to its definition

**Estimated effort**: This would be the largest phase because it establishes the foundation.

### Phase 2: VS Code Extension

**Goal**: Dedicated VS Code extension for Fidelity development

**Tasks**:
1. Create fidelity-vscode repository
2. Set up TypeScript extension project
3. Configure LSP client to use FSNAC
4. Reuse ionide-fsgrammar for syntax highlighting
5. Add "Build with Firefly" command
6. Add status bar item showing project/FSNAC status

**Validation**:
- Install extension in VS Code
- Full editing experience with correct type information
- Build project from VS Code command
- Status bar shows project name and build status

### Phase 3: Rich Features

**Goal**: Feature parity with core FSAC functionality

**Tasks**:
1. Find all references
2. Rename symbol (across project)
3. Document symbols (outline view)
4. Workspace symbols (search by name)
5. Signature help (function parameters)
6. Completion with type information

**Validation**:
- Navigate through FidelityHelloWorld samples using all features
- Rename a function, verify all call sites updated
- Use outline view to navigate large files

### Phase 4: Vim/Neovim Support

**Goal**: First-class Vim/Neovim experience

**Tasks**:
1. Create fidelity-vim repository
2. Create nvim-lspconfig integration
3. Add Vim 8+ support (ALE or CoC)
4. Write documentation and examples

**Validation**:
- Full LSP features in Neovim
- Hover, completion, go-to-definition all working
- Diagnostics shown inline

### Phase 5: Advanced Analysis

**Goal**: Fidelity-specific analysis beyond basic type checking

**Tasks**:
1. Design Fidelity.Analyzers.SDK API
2. Implement memory safety analyzer (detect potential lifetime issues)
3. Implement region analyzer (detect stack escapes)
4. Implement platform binding validator
5. Integrate analyzers with FSNAC diagnostic pipeline

**Validation**:
- Analyzer detects when a stack-allocated value might escape
- Analyzer warns about peripheral access from wrong context
- All warnings appear as LSP diagnostics with suggested fixes

---

## Part 9: Reusable Components

To summarize what we expect to reuse versus what we would build:

### Reuse Directly (No Changes Expected)

| Component | Source | Why It Should Work |
|-----------|--------|-------------------|
| **Ionide.LanguageServerProtocol** | ionide/LanguageServerProtocol | Generic LSP infrastructure, not F#-specific |
| **ionide-fsgrammar** | ionide/ionide-fsgrammar | F# syntax unchanged in Fidelity |
| **Fantomas** | fsprojects/fantomas | Code formatting is syntax-level |

### Build New

| Component | Purpose | Estimated Effort |
|-----------|---------|------------------|
| **fsnative-autocomplete (FSNAC)** | Language server using FNCS | High |
| **Fidelity.ProjInfo** | `.fidproj` parser | Low |
| **fidelity-vscode** | VS Code extension | Medium |
| **fidelity-vim** | Vim/Neovim plugin | Low |
| **Fidelity.Analyzers.SDK** | Custom analyzer framework | Medium (future) |

### Adapt Patterns

| Source | What We Would Learn |
|--------|---------------------|
| **FSAC architecture** | Handler patterns, caching strategies, incremental checking |
| **FSharpLint patterns** | How to integrate linting with LSP diagnostics |
| **Ionide-vscode-fsharp** | Extension structure, command patterns, UI integration |

---

## Conclusion

Fidelity tooling would require a new stack because Fidelity has different semantics than .NET F#. While type *names* stay the same (`string`, `option`, `array`), the *semantics* are fundamentally different (UTF-8 fat pointers, value-type options, deterministic memory). The tooling must understand these semantics. And we can't load `.fsproj` files when projects use `.fidproj`.

But the problem appears tractable. We would reuse generic infrastructure (LSP protocol, syntax grammar). We would build on FNCS (which already understands native types). We would start .NET-hosted (avoiding the bootstrap problem). And we would simplify where possible (TOML projects, no MSBuild).

The result we envision is a complete development environment for F# Native: type information, navigation, refactoring, diagnostics, all aware of native semantics, memory regions, and Fidelity's unique capabilities.

When you hover over a `string` and the tooling understands it's a UTF-8 fat pointer, you know you're writing native code. When you see a pointer annotated with its memory region and access kind, you understand the hardware you're targeting. The tooling would make Fidelity's power accessible - without burdening users with unfamiliar type names.

The implementation work lies ahead.
