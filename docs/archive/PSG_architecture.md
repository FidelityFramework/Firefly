# Program Semantic Graph (PSG) Architecture Decision Memo

> **Related Documents:**
> - `PSG_Nanopass_Architecture.md` - How PSG is constructed (true nanopass pipeline)
> - `TypedTree_Zipper_Design.md` - Typed tree overlay for SRTP resolution
> - `Architecture_Canonical.md` - Top-level architecture reference

## Executive Summary

This memo documents the architectural decisions for implementing the Program Semantic Graph (PSG) within the Firefly compiler. The PSG serves as the unified intermediate representation that correlates F# syntax with semantic type information, enabling precise reachability analysis and efficient MLIR generation while maintaining clear separation of concerns between the Core analysis phase and the Dabbit MLIR generation phase.

## Background and Requirements

The Firefly compiler requires a unified program representation that preserves both syntactic structure and rich type information throughout the compilation pipeline. This representation must support reachability analysis for dead code elimination while providing the semantic depth necessary for MLIR code generation. The PSG addresses the fundamental challenge that F# Compiler Services (FCS) provides syntax and semantic information in separate, uncorrelated structures.

## Key Architectural Decisions

### PSG as Single Source of Truth

The PSG will replace the need to work with separate syntax and semantic trees by providing a single traversable graph. This unified structure starts from entry points and contains all reachable code with preserved semantic information, eliminating the correlation burden from downstream compilation phases.

### FCS Correlation Strategy

The architecture leverages FCS built-in correlation mechanisms rather than implementing custom range-based mapping:

- **Primary Identity**: FSharpSymbol objects provide stable identity across syntax and semantic boundaries
- **Secondary Identity**: Range-based identification for syntax nodes without symbols (expressions, patterns)
- **Bridge Mechanism**: FSharpSymbolUse objects correlate symbol identity with source positions
- **Bulk Extraction**: GetAllUsesOfAllSymbols() provides comprehensive symbol usage data

### Node Identity and Graph Structure

PSG nodes will use FSharpSymbol identity hashes as primary keys where available, with range-based secondary keys for expressions and patterns. This approach aligns with FCS internal mechanisms while providing stable references for reachability analysis and MLIR generation.

### Reachability Analysis Approach

The implementation follows a fan-out strategy starting from EntryPoint functions:

- **Traversal Root**: Single EntryPoint function for proof of concept (multiple entry points deferred)
- **Execution Path Focus**: Include only functions and types on active execution paths
- **Conservative Analysis**: Handle F# language patterns (currying, partial application) through careful dependency tracking
- **Library Boundary Management**: Minimize FSharp.Core traversal depth while ensuring complete resolution

### Library Dependency Handling

The architecture distinguishes three categories of dependencies:

- **User Code**: Full AST and semantic information preserved
- **Alloy Library**: Complete analysis and inclusion for optimization opportunities
- **FSharp.Core**: Minimal inclusion limited to statically resolvable primitives and SRTP resolution targets

### Generic Type Resolution Strategy

A two-pass approach addresses the generic type resolution challenge:

- **Pass One**: Construct generic PSG with type parameters preserved (.gen.psg)
- **Pass Two**: Resolve all generics to concrete types (res.psg)
- **Separation of Concerns**: Complete resolution within Core phase before Dabbit MLIR generation

This approach maintains clean architectural boundaries while ensuring Dabbit receives fully resolved semantic information.

## MLIR Alignment Opportunities

The PSG structure naturally aligns with MLIR concepts:

- **MLIR Module** corresponds to the complete PSG (project + Alloy + required FSharp.Core)
- **MLIR Function** maps to PSG function nodes with reachability metadata
- **MLIR Operations** represent PSG expression nodes with preserved type information
- **MLIR Attributes** carry type signatures and semantic metadata

This alignment minimizes transformation complexity during MLIR generation.

## Implementation Considerations

### FSharp.Core Boundary Management

The depth of FSharp.Core traversal requires careful analysis. The architecture assumes most dependencies terminate at primitive operations (NativePtr, basic arithmetic, core collection operations) with minimal transitive dependencies. This assumption requires validation during implementation.

### Type Dependency Precision

Type references without function calls present a challenge for execution path analysis. The implementation must balance completeness against over-inclusion, particularly for complex F# patterns involving type-level computation and constraint resolution.

### Performance and Scalability

The fan-out approach provides natural performance characteristics, building only necessary portions of the semantic graph. The two-pass generic resolution may introduce additional complexity but maintains architectural clarity and debugging capability.

## Risk Assessment

### High Risk: F# Language Pattern Coverage

Complex F# patterns (currying, partial application, computation expressions) may introduce unexpected dependency paths. Mitigation requires comprehensive testing with representative F# codebases and careful analysis of FCS symbol relationship data.

### Medium Risk: FSharp.Core Dependency Depth

Unknown traversal depth into FSharp.Core could impact compilation performance and binary size. Mitigation involves empirical analysis of common F# code patterns and implementation of configurable depth limits.

### Low Risk: Generic Resolution Complexity

The two-pass approach adds implementation complexity but provides clear architectural benefits. The risk is manageable through careful phase separation and comprehensive intermediate representation validation.

## Next Steps

### Phase 1: PSG Foundation
- Implement FSharpSymbol-based node identity system
- Create PSG data structure with correlation mechanisms
- Implement basic fan-out traversal from EntryPoint

### Phase 2: Reachability Analysis
- Develop precise dependency tracking for F# language patterns
- Implement library boundary detection and management
- Validate FSharp.Core traversal depth assumptions

### Phase 2.5: Pruned Representation Generation
- Generate .pruned.ast files showing symbolic AST after tree-shaking
- Output only reachable portions of library modules in developer-readable format
- Provide transparency and validation for pruning decisions
- Enable developer review of library code inclusion scope

### Phase 3: Generic Resolution
- Implement two-pass generic resolution pipeline
- Validate complete type resolution before Dabbit handoff
- Establish debugging and validation capabilities for intermediate representations

### Phase 4: MLIR Integration
- Validate PSG to MLIR transformation efficiency
- Optimize PSG structure for downstream compilation performance
- Implement comprehensive testing across representative F# codebases

## Addendum on Coupling and Cohesion

The intersection of coupling analysis, MLIR compilation units, and LLVM caching presents several opportunities for architectural alignment between F# and MLIR and LLVM compilation caching tooling. This addendum explores why "coupling and cohesion" analysis in the Firefly is justified even in a "hello world" stage of compiler design.

## MLIR Compilation Boundary Opportunities

MLIR operates on module-level compilation units with well-defined interfaces. The coupling analysis in CouplingCohesion.fs could identify natural compilation boundaries by detecting code clusters with high internal cohesion and minimal external coupling. These boundaries align naturally with MLIR's module system, where each compilation unit maintains interface stability while allowing internal optimization.

MLIR supports incremental compilation through interface-based boundaries. When interfaces remain stable, downstream compilation units can reuse cached results. Firefly's coupling analysis could distinguish between interface dependencies and implementation dependencies, allowing the PSG to be used to track and identify which changes require recompilation versus which can leverage cached results.

## LLVM Link-Time Optimization Alignment

LLVM's LTO capabilities work most effectively when compilation units represent logical program components rather than arbitrary file boundaries. The cohesion analysis could identify code clusters that benefit from joint optimization, while coupling analysis identifies clean separation points where LTO can operate across cached boundaries without performance degradation.

The tiered compilation approach mentioned in the architecture documentation aligns well with LLVM's optimization pipeline. Hot code paths identified through coupling analysis could receive aggressive optimization during initial compilation, while cold code paths could use cached, lightly optimized versions until usage patterns justify deeper optimization.

## PSG Design Implications

Several PSG design decisions could support future caching strategies. The PSG should preserve module-level compilation boundaries as first-class entities, allowing partial graph reconstruction when only specific modules require recompilation. Interface nodes should be distinguished from implementation nodes, enabling interface stability tracking that supports incremental compilation.

Dependency edge classification becomes critical for caching effectiveness. The PSG should distinguish between strong dependencies that require joint compilation and weak dependencies that can span compilation boundaries. This classification enables the build system to determine optimal compilation unit boundaries based on actual program structure rather than source file organization.

## Early Decision Opportunities

The fan-out reachability analysis provides an opportunity to build caching awareness into the core architecture. As the analysis traverses from entry points, it could annotate traversal depth and coupling strength, creating natural compilation unit candidates. Functions and types with similar reachability patterns and strong coupling could be grouped into compilation units that benefit from joint optimization.

The two-pass generic resolution strategy aligns well with cached compilation. Generic definitions could be cached after the first pass, with only concrete instantiations requiring recompilation when usage patterns change. This approach leverages MLIR's template specialization capabilities while maintaining compilation efficiency.

The coupling analysis metrics could be extended to include interface stability indicators. Modules with stable public interfaces and high internal cohesion become excellent candidates for aggressive caching, while modules with volatile interfaces or low cohesion may require more frequent recompilation but benefit from smaller compilation unit boundaries.

These considerations suggest that the PSG should include compilation boundary metadata from the initial implementation. While the proof of concept will operate on single compilation units, the data structures and analysis algorithms should accommodate future segmentation without requiring fundamental architectural changes. This approach ensures that scaling decisions can leverage natural program structure rather than fighting against early architectural constraints.

## Conclusion

The PSG architecture provides a principled approach to unifying F# syntax and semantic information while maintaining clear architectural boundaries. The fan-out reachability analysis aligns with Firefly's zero-allocation goals, and the MLIR-aware design minimizes downstream transformation complexity. The two-pass generic resolution ensures complete type information while preserving separation of concerns between analysis and code generation phases.
