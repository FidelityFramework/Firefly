# JellyJar: Solution Coherence for Fidelity

## The Metaphor

If you grew up catching fireflies on summer evenings, you may remember putting them in a jar and watching something remarkable happen. At first, each firefly flashes independently, on its own rhythm. But after a few minutes in the jar together, they begin to synchronize. Their flashing aligns. They pulse in unison.

Then you release them. They disperse across the yard, flying off to their own destinations. But for a while, they keep flashing together. They've internalized the shared rhythm. Even as independent agents, they remain in sync.

That's the idea behind JellyJar.

In the Fidelity framework, we anticipate scenarios where independent compilation targets (a microcontroller firmware, a Linux service, a Cloudflare Worker, a desktop application) would need to communicate. They would share data structures. They would pass messages. They would honor contracts. JellyJar is our concept for a mechanism that would bring these independent projects together, synchronize their understanding of shared interfaces, and then release them to compile independently while remaining compatible.

This document describes our current architectural thinking. The implementation work lies ahead.

## Why .NET Solutions May Not Be Enough

If you've worked with .NET, you know the `.sln` file. A solution groups related projects. You can build them together, share references, and Visual Studio understands the relationships.

But a .NET solution is relatively shallow. It's a container. Projects within a solution can reference each other, but the solution itself doesn't enforce much. You can have incompatible versions, mismatched interfaces, or runtime type mismatches that only surface when the code actually runs.

Fidelity would target a different world. When you compile F# to native code without a runtime, there would be no dynamic type checking. There would be no reflection to discover interfaces at runtime. If two components disagreed about the shape of a message, you wouldn't get an exception; you'd get memory corruption or silent data loss.

This concern becomes especially acute for distributed applications. Consider a hypothetical system with an embedded sensor (ARM Cortex-M), an edge aggregator (Linux x86_64), and a cloud processor (Cloudflare Workers via WebAssembly). All three would need to agree on the binary layout of sensor readings. All three would need to serialize and deserialize identically. There would be no common runtime to mediate.

We envision JellyJar as the component that would ensure they agree at compile time.

## What JellyJar Would Manage

### The Program Hypergraph

A single Fidelity project produces a Program Semantic Graph (PSG): the complete semantic structure of that compilation unit. We imagine JellyJar managing something larger: a Program Hypergraph that would span multiple projects.

The hypergraph would capture:

- **Shared type definitions**: Records, discriminated unions, and structs that appear in multiple projects
- **BAREWire contracts**: Binary encoding specifications that define wire formats
- **Platform binding surfaces**: Which platform bindings each project uses
- **Cross-project references**: Function calls, type references, and data flows across project boundaries
- **Target specifications**: The platform, architecture, and output format for each project

When building a JellyJar solution, the hypergraph would be constructed first. JellyJar would analyze the relationships, detect inconsistencies, and ensure all projects share compatible definitions before any individual compilation begins.

### BAREWire Contract Enforcement

BAREWire provides zero-copy binary encoding for Fidelity. A BAREWire schema defines the logical structure of messages: field names, types, ordering, and serialization rules. The schema serves as a contract that all participants agree to honor.

Here's the nuance: different targets may have different native representations. A big-endian ARM microcontroller and a little-endian x86_64 server store integers differently in memory. A 32-bit embedded target and a 64-bit cloud processor have different pointer sizes and alignment requirements. The byte-level representation of a structure in memory would not be identical across these targets.

But the BAREWire contract doesn't require identical memory layouts. It requires that when data crosses a boundary (network, IPC, file), all participants serialize and deserialize according to the same rules. The schema specifies the wire format. Each target's compiler generates architecture-appropriate code that correctly translates between native representation and wire format.

We expect JellyJar would track BAREWire contracts across the solution. It would verify that every project using a particular message type shares the same logical structure and references the same schema. If the edge aggregator referenced `SensorReading` and the firmware defined it, JellyJar would confirm they have corresponding BAREWire representations: the same fields, the same types, the same schema governing serialization. Each target would then compile serialization code appropriate for its architecture, but the wire format would be consistent.

This would happen at solution build time, before any individual project compiles. A structural mismatch (different fields, incompatible types, missing schema reference) would produce an error. The architecture-specific details of how each target implements the serialization would be handled by Firefly during individual project compilation.

### Cross-Target Type Compatibility

Different compilation targets have different type representations. An `int` on a 32-bit microcontroller is 32 bits; on a 64-bit Linux system, it's still 32 bits in F#, but pointer sizes differ. A `nativeint` varies by platform.

We anticipate JellyJar would track these differences. When a shared type flows across platform boundaries, JellyJar would verify that the representation is portable. If you were passing a structure containing `nativeint` between an ARM32 and x86_64 target, JellyJar would flag this as a potential problem.

For types that must be portable, JellyJar could enforce BAREWire encoding at the boundary. The internal representation might vary; the wire format would be fixed.

## Solution Structure

We're considering a `.fidsln` file format (Fidelity Solution) that would define the constituent projects and their relationships. One possibility:

```toml
[solution]
name = "DistributedSensorNetwork"

[projects]
firmware = { path = "./firmware/sensor.fidproj", target = "arm-cortex-m4" }
edge = { path = "./edge/aggregator.fidproj", target = "linux-x86_64" }
cloud = { path = "./cloud/processor.fidproj", target = "wasm32" }

[contracts]
sensor_protocol = { schema = "./schemas/sensor.bare", used_by = ["firmware", "edge"] }
aggregation_protocol = { schema = "./schemas/aggregation.bare", used_by = ["edge", "cloud"] }

[constraints]
# All projects must agree on these types
shared_types = ["SensorReading", "AggregatedBatch", "ProcessingResult"]
```

The `[contracts]` section would explicitly list BAREWire schemas and which projects use them. JellyJar would enforce that these projects have compatible implementations.

The `[constraints]` section would list types that must be identical across specified projects. JellyJar would verify structural equality, not just name equality.

This format is preliminary. We expect it to evolve as we better understand the requirements.

## External Contract Participants

Not every participant in a distributed system would be a Fidelity project. A Cloudflare Worker written in Rust might consume messages from a Fidelity edge service. A legacy C application might produce data that Fidelity firmware reads.

We imagine JellyJar could track these external participants through contract specifications:

```toml
[external]
legacy_sensor = { schema = "./schemas/legacy.bare", role = "producer" }
rust_worker = { schema = "./schemas/aggregation.bare", role = "consumer" }
```

JellyJar wouldn't compile these external components, but it would verify that Fidelity projects expecting to communicate with them use compatible schemas. The BAREWire schema would serve as the contract; JellyJar would ensure the Fidelity side honors it.

## The Synchronization Process

We envision the JellyJar build process proceeding in phases:

**Phase 1: Discovery**

JellyJar would read all `.fidproj` files in the solution. It would identify shared types, BAREWire schemas, and cross-project references. This would produce the initial hypergraph structure.

**Phase 2: Contract Resolution**

For each BAREWire contract, JellyJar would locate all projects that reference it. It would compute the expected binary representation based on the schema and the types each project defines.

**Phase 3: Compatibility Verification**

JellyJar would compare representations across projects. Structural mismatches (different field order, incompatible types, misaligned offsets) would produce errors with detailed diagnostics.

**Phase 4: Individual Compilation**

Once coherence is verified, each project would compile independently via Firefly. The PSG for each project would be built knowing that its shared types are compatible with sibling projects.

**Phase 5: Artifact Correlation**

After compilation, JellyJar could produce a manifest correlating the binaries. This manifest might record which binary corresponds to which target, which schemas are embedded, and how to deploy the complete system.

## Relationship to Other Fidelity Components

We see JellyJar sitting above Firefly in the tooling hierarchy:

| Component | Anticipated Role |
|-----------|------------------|
| **Firefly** | Would compile a single `.fidproj` to a native binary |
| **JellyJar** | Would coordinate multiple projects, ensure coherence, invoke Firefly for each |
| **BAREWire** | Would define binary encoding schemas that JellyJar enforces |
| **FSNAC** | Would provide IDE support; could integrate with JellyJar for solution-wide diagnostics |

For single-project workflows, you would use Firefly directly. For multi-project distributed systems, JellyJar would orchestrate the build.

## When JellyJar Would Be Valuable

We expect JellyJar to become valuable when:

- **Multiple targets share data**: An embedded device and a host application exchange binary messages
- **Distributed systems**: Microservices, edge computing, or IoT deployments where components run on different platforms
- **Mixed Fidelity/external**: Some components are Fidelity projects, others are Rust, C, or other languages, but all must agree on wire formats
- **Long-lived contracts**: The binary format must remain stable across versions for backward compatibility

If your project is a single binary for a single platform, Firefly alone would suffice. The moment you have two components that need to agree on something, JellyJar would help ensure they actually do.

## Comparison to .NET Solutions

| Aspect | .NET Solution | JellyJar Solution (Proposed) |
|--------|---------------|------------------------------|
| Project grouping | Yes | Yes |
| Shared references | Assembly references | Type and contract references |
| Cross-project type checking | Via shared assemblies | Via hypergraph verification |
| Binary compatibility | Runtime type checking | Compile-time BAREWire verification |
| Multi-target | Limited (portable libraries) | Native: each project could target different platform |
| External contract enforcement | None | BAREWire schema validation |
| Memory layout verification | None | Explicit layout checking |

## Current Status

JellyJar is in early design. The concepts described here represent our current thinking about how solution-level coherence could work in the Fidelity ecosystem. We have not yet implemented these ideas.

The priorities we're considering:

1. Define the `.fidsln` format and project relationship model
2. Implement hypergraph construction from multiple `.fidproj` files
3. Add BAREWire schema comparison and compatibility checking
4. Integrate with Firefly's build orchestration

As Fidelity matures and more projects require multi-target compilation, we expect JellyJar to become essential infrastructure. For now, it remains a design goal.

## The Release

In the metaphor, the fireflies eventually leave the jar. Each project would compile to its own binary, deploy to its own platform, run its own code. But they would carry with them the shared rhythm: the compatible types, the agreed-upon wire formats, the synchronized understanding of how to communicate.

JellyJar's job would be done when the builds complete successfully. The coherence it established would persist in the binaries themselves.
