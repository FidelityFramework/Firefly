# Firefly Continuation Pipeline Prototyping Roadmap

## Overview: From PSG Morass to "DCont All The Way Down"

This roadmap transforms the current FCS-based `ProgramSemanticGraph` into a continuation-centric compilation pipeline targeting WAMI (WebAssembly with stack switching) primarily through MLIR's delimited continuation dialect. The focus remains on rapid prototyping with in-memory processing to achieve demonstrable results quickly using simple CLI proofs-of-concept.

### Understanding Continuations and CPS

Before diving into the pipeline, it's worth clarifying what we mean by continuations. In Continuation-Passing Style (CPS), instead of functions returning values directly, they receive an extra parameter - a continuation - that represents "what to do next" with the result:

```fsharp
// Direct style
let add x y = x + y
let result = add 3 4  // Returns 7

// CPS style
let addCPS x y k = k (x + y)
let result = addCPS 3 4 (fun sum -> sum)  // Passes 7 to continuation
```

Delimited continuations (DCont) improve on CPS by allowing us to capture and manipulate only *part* of the continuation - the part up to a delimiter. This is crucial for async/await:

```fsharp
// F# async is essentially delimited continuations
async {
    let! data = readAsync()    // Captures continuation from here...
    return process data        // ...to here
}

// Compiles to something like:
shift (fun k ->              // k is the delimited continuation
    readAsync (fun data ->   // When read completes...
        k (process data)))   // ...resume with processed data
```

The power of preserving continuations in our compilation pipeline is that async operations, generators, and effect handlers get first-class runtime support. The challenge is knowing when this flexibility is worth the overhead versus compiling directly to state machines.

## Critical Semantic Requirements: RAII Through Continuation Completion

The compilation pipeline implements deterministic resource management through continuation completion semantics. Resource cleanup occurs when continuations terminate, fail, or complete rather than at arbitrary textual boundaries. When a continuation suspends, resources remain accessible for resumption. When a continuation completes, associated resources receive automatic cleanup as part of the continuation termination sequence.

This approach aligns with the coeffects notion of resource tracking - resources are a form of context dependency that must be tracked through the computation. The coeffect system ensures that resource lifetimes are properly bounded:

```fsharp
type ResourceCoeffect =
    | NoResources                        // Pure computation
    | StackBounded of size: int          // Stack-allocated, auto-cleanup
    | ContinuationBounded of resources: Set<Resource>  // Cleanup at continuation completion
    | ExternalResources of handles: Set<Handle>        // Requires explicit management
```

This approach aligns resource lifetimes with computation lifetimes, ensuring that cleanup operations execute at natural completion points determined by the continuation control flow structure. Stack-allocated resources map to continuation frames that release automatically when continuations terminate, while system resources such as file handles trigger cleanup operations as part of continuation completion sequences.

## Phase 0: PSG Foundation Cleanup

**Goal**: Establish proper tombstone behavior and reachability analysis in existing `ProgramSemanticGraph`

### PSG Reachability Infrastructure

- [x] Consolidate any remaining vestigial 'intermediate' graphs into one cohesive PSG
- [x] Complete tombstone behavior implementation for soft-delete reachability analysis
- [x] Finalize `ReachabilityHelpers.markReachable` and `ReachabilityHelpers.markUnreachable` functions
- [x] Ensure `IsReachable`, `EliminationPass`, and `EliminationReason` fields work correctly
- [x] Validate reachability analysis produces consistent results before adding continuation layers
- [x] Add minimal context tracking to PSG nodes:
  ```fsharp
  type PSGNode = {
      // Existing fields...
      ContextRequirement: ContextRequirement option
  }
  
  type ContextRequirement =
      | Pure              // No external dependencies
      | AsyncBoundary     // Suspension point
      | ResourceAccess    // File/network access
  
  type ComputationPattern =
      | DataDriven        // Push-based, eager evaluation
      | DemandDriven      // Pull-based, lazy evaluation
  ```

**Success Criteria**: PSG reachability analysis works correctly with consistent tombstone behavior

## Phase 1: FCS Pipeline Enhancement  

**Goal**: Enhance FCS integration to extract continuation boundaries and preserve semantic information

### Enhanced FCS Processing

The FCS enhancement extends existing correlation systems to identify async boundaries, resource scopes, and effect boundaries within the F# source code. F# async blocks map to continuation delimiter metadata in PSG nodes, while use bindings and try-with constructs establish resource and exception boundaries that align with continuation completion points.

During FCS processing, identify contextual requirements and computation patterns:

- Pure computations require no special handling
- Async boundaries require continuation machinery
- Resource access requires cleanup tracking
- Lazy sequences and async enumerables indicate demand-driven patterns

```fsharp
let analyzeComputationPattern (expr: FSharpExpr) =
    match expr with
    | AsyncSeq _ | LazySeq _ -> DemandDriven
    | ListComprehension _ | ArrayInit _ -> DataDriven
    | _ -> DataDriven  // Default to eager
```

The processing preserves delimited continuation semantic information through the existing PSG.Types structure, ensuring that resource cleanup operations integrate naturally with continuation termination sequences. Validation occurs through the HelloWorldDirect example to confirm that continuation boundary detection functions correctly with realistic async patterns.

**Success Criteria**: FCS processing identifies and marks continuation boundaries in existing `ProgramSemanticGraph` structure with integrated resource management points

## Phase 2: Bidirectional Zipper Implementation

**Goal**: Implement bidirectional zipper for `ProgramSemanticGraph` traversal and transformation

### Bidirectional Zipper Core

```fsharp
// Bidirectional zipper operations for `ProgramSemanticGraph`
module PSGZipper =
    let moveDown: NodeId -> PSGZipper -> PSGZipper option           // Navigate to child
    let moveUp: PSGZipper -> PSGZipper option                       // Navigate to parent  
    let moveLeft: PSGZipper -> PSGZipper option                     // Navigate to previous sibling
    let moveRight: PSGZipper -> PSGZipper option                    // Navigate to next sibling
    let moveToRoot: PSGZipper -> PSGZipper                          // Navigate to root
    let moveToNode: NodeId -> PSGZipper -> PSGZipper option         // Direct navigation to specific node
    
    // Support both producer-driven and consumer-driven traversal
    let followDataFlow: PSGZipper -> PSGZipper option               // Forward flow
    let followDemandFlow: PSGZipper -> PSGZipper option             // Backward flow
```

### Context-Aware Transformations

- [ ] Implement zipper context preservation during transformations
- [ ] Add capability to transform focused node while maintaining graph invariants
- [ ] Support bulk transformations across zipper paths
- [ ] Integrate with existing `ChildrenStateHelpers` for consistent state management
- [ ] Track coeffect propagation during zipper traversal:
  ```fsharp
  type ZipperContext = {
      CurrentNode: PSGNode
      Path: NodeId list
      AccumulatedCoeffects: ContextRequirement list  // Track context along path
  }
  
  // Coeffect-aware zipper movement
  let moveWithCoeffects (direction: Direction) (zipper: PSGZipper) =
      let newZipper = move direction zipper
      match newZipper with
      | Some z -> 
          // Propagate coeffects based on node relationships
          let updatedCoeffects = propagateCoeffects z.CurrentNode z.AccumulatedCoeffects
          Some { z with AccumulatedCoeffects = updatedCoeffects }
      | None -> None
  ```
- [ ] Test zipper operations with simple async code transformations

**Success Criteria**: Bidirectional zipper enables complex PSG transformations while preserving context and graph integrity

## Phase 3: In-Memory Property List Generation

**Goal**: Generate comprehensive property lists for control and data flow analysis using in-memory structures

### Control Flow Property Generation

```fsharp
// Generate control flow properties for continuation boundaries identified in PSG
let analyzeControlFlow: ProgramSemanticGraph -> NodeId -> ControlFlowProperties
let findEntryPoints:    ProgramSemanticGraph -> NodeId list
let findExitPoints:     ProgramSemanticGraph -> NodeId list  
let findSuspendPoints:  ProgramSemanticGraph -> SuspendPoint list

// Coeffect-enhanced control flow analysis
type ControlFlowProperties = {
    EntryPoints: NodeId list
    ExitPoints: NodeId list
    SuspendPoints: SuspendPoint list
    ControlCoeffects: ControlFlowCoeffect list
}

type ControlFlowCoeffect =
    | PureSequential           // No suspension points, direct compilation possible
    | AsyncSuspension of points: NodeId list  // Must preserve continuation
    | ResourceBounded of acquire: NodeId * release: NodeId list  // RAII pattern
    | EffectfulRegion of effects: EffectType list  // External effects requiring preservation
```

### Data Flow Property Generation

```fsharp
// Track variable definitions, uses, and dependencies across continuation boundaries
let analyzeDataFlow:        ProgramSemanticGraph -> DataFlowProperties
let extractDefinitions:     PSGNode -> VariableDefinition list
let extractUses:           PSGNode -> VariableUse list
let computeLiveVariables:  ProgramSemanticGraph -> LivenessInfo

// Track both push and pull data flow patterns
type DataFlowEdge =
    | ProducerConsumer of source: NodeId * sink: NodeId      // Traditional data flow
    | ConsumerProducer of sink: NodeId * source: NodeId      // Demand-driven flow

// Enhanced with structural coeffects for per-variable tracking
type VariableUse = {
    Variable: string
    Location: range
    Context: VariableContext
    FlowPattern: DataFlowPattern
}

type DataFlowPattern =
    | Eager             // Computed immediately
    | Lazy              // Computed on first access
    | Streaming         // Computed on each demand

type VariableContext =
    | PureContext          // Variable used in pure computation
    | AsyncContext         // Variable crosses async boundary
    | ResourceContext      // Variable lifetime tied to resource
```

### In-Memory Property Storage

- [ ] Store control flow properties as efficient maps and data structures
- [ ] Store data flow dependencies as lists with graph relationships
- [ ] Create JSON serialization for property lists to enable debugging and inspection

### Effect Flow Properties

The effect flow analysis tracks async effects, resource acquisition and cleanup, and exception boundaries within the continuation structure. This directly implements the coeffects paper's notion of tracking "what code needs from its environment":

```fsharp
type EffectCoeffect =
    | Pure                              // No effects (can optimize aggressively)
    | Async of SuspensionPattern        // Requires continuation machinery
    | IO of IOPattern                   // External world interaction
    | ResourceEffect of ResourcePattern // Resource acquisition/release
    | Combined of EffectCoeffect list   // Multiple effects in region

let analyzeEffectFlow (graph: ProgramSemanticGraph) =
    // Track effect coeffects through the graph
    let effectMap = computeEffectCoeffects graph
    // Identify effect boundaries for compilation decisions
    let boundaries = findEffectBoundaries effectMap
    { Effects = effectMap; Boundaries = boundaries }
```

Effect dependencies are stored in memory-based graph structures that align with continuation completion points rather than arbitrary scope boundaries. This approach generates effect flow property lists that integrate naturally with MLIR effect dialect operations.

**Success Criteria**: Complete flow analysis generates actionable property lists stored in memory with fast lookup and JSON debugging capability

## Phase 4: MLIR Integration Pipeline

**Goal**: Compile `ProgramSemanticGraph` with continuation semantics to MLIR DCont dialect with integrated resource management

### MLIR Dialect Mapping Strategy

The MLIR dialect mapping transforms PSG nodes with continuation boundary metadata into appropriate MLIR DCont operations. Async boundaries map to `dcont.reset` and `dcont.shift` operations, while resource boundaries integrate cleanup operations with continuation completion sequences. Resource cleanup operations execute as part of continuation termination rather than as separate scope-based operations.

The compilation strategy selection leverages context requirements and computation patterns identified in earlier phases:

```fsharp
let selectCompilationStrategy (node: PSGNode) =
    match node.ContextRequirement, node.ComputationPattern with
    | Some Pure, DataDriven -> 
        // Pure eager computations compile to tight loops
        CompileDirect
    | Some AsyncBoundary, _ -> 
        // Async boundaries always need continuation machinery
        CompileToDCont
    | Some ResourceAccess, DemandDriven ->
        // Lazy resources benefit from continuation-based cleanup
        CompileToDContWithCleanup
    | _, DemandDriven ->
        // Demand-driven patterns naturally fit continuations
        CompileToDCont
    | _ -> 
        CompileDirect  // Default to direct compilation
```

The mapping preserves F# type information through the MLIR type system while ensuring that resource management aligns with continuation control flow. Validation occurs through HelloWorldDirect async patterns to confirm that resource management integrates correctly with continuation semantics.

### Type System Preservation

```fsharp
// Map F# types to MLIR types while preserving continuation structure
let deriveFunctionType:     PSGNode -> TypeMapping -> MLIRType
let generateTypeDefinitions: ProgramSemanticGraph -> MLIRModule -> TypeMapping  
let preserveContinuationTypes: ProgramSemanticGraph -> MLIRModule -> unit
```

### Code Generation Pipeline

```fsharp
// Core compilation functions mapping PSG to MLIR operations
let compilePSGToMLIR:         ProgramSemanticGraph -> MLIRModule
let compileContinuationNode:  PSGNode -> TypeMapping -> MLIRModule -> MLIROperation
let compileEffectfulNode:     PSGNode -> TypeMapping -> MLIRBlock -> MLIROperation list
```

**Success Criteria**: F# async code in PSG compiles to valid MLIR DCont dialect operations with preserved semantics and resource cleanup integrated with continuation completion

## Phase 5: WAMI Target Implementation

**Goal**: Configure MLIR optimization pipeline to lower DCont operations to WebAssembly with stack switching through opt_mlir

### MLIR Pass Pipeline Configuration

The implementation leverages existing MLIR infrastructure through opt_mlir for optimization and lowering, operating on the MLIR generated by the Phase 4 compilation pipeline. This approach delegates complex transformation work to proven MLIR optimization framework while maintaining oversight through telemetry and monitoring.

Pass selection considers the context requirements and computation patterns identified through analysis:

```fsharp
let configurePasses (module: MLIRModule) =
    let basePassses = ["canonicalize"; "cse"]
    
    // Context and pattern guided optimization selection
    let contextPasses = 
        module.AnalysisRegions
        |> List.collect (fun region ->
            match region.Context, region.Pattern with
            | Pure, DataDriven -> 
                // Eager pure regions get aggressive optimization
                ["inline"; "loop-invariant-code-motion"; "vectorize"; "mem2reg"]
            | AsyncSuspension _, _ ->
                // Async regions need continuation preservation
                ["dcont-to-wasm-suspender"; "async-runtime-lowering"]
            | _, DemandDriven ->
                // Demand-driven patterns benefit from lazy evaluation passes
                ["lazy-code-motion"; "dead-store-elimination"]
            | ResourceBounded _, _ ->
                // Resource regions need cleanup instrumentation
                ["resource-lifetime-analysis"; "cleanup-insertion"]
            | Combined contexts ->
                // Mixed regions get conservative treatment
                selectConservativePasses contexts
        )
        |> List.distinct
    
    basePasses @ contextPasses
```

The opt_mlir process receives the DCont dialect operations from Phase 4 and applies appropriate pass pipelines to lower them to WebAssembly with stack switching capabilities. The configuration specifies the necessary passes for DCont to WAMI transformation without requiring custom implementation of these complex optimization sequences.

### Arms-Length opt_mlir Process Management

The compilation pipeline manages opt_mlir execution through comprehensive process orchestration that maintains visibility into transformation progress while delegating optimization decisions to the mature MLIR infrastructure. Telemetry capture provides insight into pass execution and transformation results without requiring deep integration with MLIR internals.

Debug stream monitoring tracks DCont dialect transformations as they progress through the optimization pipeline, ensuring that continuation semantics preservation can be validated at each stage. Error handling and recovery mechanisms address opt_mlir process failures while maintaining compilation pipeline stability.

### WebAssembly Output Validation and Integration

The process wrapper implements comprehensive logging and performance monitoring for the complete compilation pipeline, from PSG transformation through MLIR optimization to final WebAssembly generation. Validation checkpoints ensure that continuation semantics and resource management guarantees survive the optimization process.

The generated WebAssembly binary receives validation to confirm that stack switching capabilities are present and that continuation flow can be traced through the runtime environment. Testing infrastructure validates continuation semantics in the target execution environment while establishing baseline measurements for performance characteristics.

### CLI Integration and Complete Pipeline Testing

The command-line interface orchestrates the complete compilation sequence from FCS processing through opt_mlir optimization to executable WebAssembly generation. The test harness validates HelloWorldDirect compilation through all pipeline stages while providing automated verification of execution results against expected continuation behavior.

Documentation covers telemetry interpretation and debugging procedures for both the custom compilation stages and the opt_mlir optimization process, ensuring that issues can be diagnosed across the complete pipeline.

**Success Criteria**: HelloWorldDirect F# source compiles through the complete pipeline to executable WebAssembly that preserves continuation semantics and runs correctly at command line with full telemetry visibility into opt_mlir transformations

## Implementation Priorities

### Foundation Phase

The PSG tombstone behavior forms the critical foundation for all subsequent analysis passes. Establishing proper reachability analysis with soft-delete semantics ensures that continuation boundary detection and flow analysis operate on reliable data structures.

The FCS enhancement phase builds directly on existing correlation systems to identify continuation boundaries, ensuring the pipeline can recognize and preserve F# async semantics from the earliest compilation stage. This phase validates the approach using the constrained HelloWorldDirect example.

### Core Compilation Pipeline

The bidirectional zipper implementation enables sophisticated PSG transformations while maintaining context, forming the basis for all code transformations required by the continuation compilation process. The zipper operations provide the navigation and transformation capabilities needed for the MLIR lowering process.

Property list generation converts PSG analysis into actionable intelligence that drives the compilation process. The in-memory approach eliminates external complexity while preserving the analytical depth required for effective continuation compilation.

### Target Implementation

MLIR DCont integration provides the bridge to modern compiler infrastructure while preserving high-level continuation semantics. The mapping from PSG continuation boundaries to MLIR delimited continuation operations represents the critical architectural transformation point.

WAMI stack switching implementation delivers the "dcont all the way down" vision, enabling functional programming patterns to run efficiently in WebAssembly environments while preserving continuation semantics throughout the execution pipeline.

## Risk Mitigation Strategy

### Technical Risk Management

The approach leverages existing MLIR infrastructure rather than building custom solutions, reducing implementation risk while providing access to the broader MLIR ecosystem. Starting with the constrained HelloWorldDirect example allows validation of core concepts before expanding to more complex scenarios.

The in-memory processing approach eliminates external dependencies during the prototyping phase while preserving the ability to add additional capabilities when the architecture matures. This strategy reduces technical risk while maintaining architectural flexibility.

### Validation Approach

Each phase builds meaningful value that can be validated independently, reducing cumulative risk while progressing toward the compilation pipeline goals. The HelloWorldDirect example provides a concrete validation target that exercises core language features while remaining simple enough for manual verification.

Coeffect tracking validation ensures compilation decisions are based on accurate context analysis:

- Verify that async boundaries are correctly identified and annotated
- Confirm resource coeffects align with actual resource usage patterns  
- Validate that compilation strategies match coeffect requirements
- Ensure coeffect propagation through the zipper maintains consistency

Success metrics focus on demonstrable compilation results rather than performance considerations, ensuring that architectural decisions are validated before complexity increases. The CLI execution target provides immediate feedback on compilation correctness.

## Success Metrics and Validation

### Foundation Success Indicators

The PSG tombstone behavior functions correctly with consistent reachability analysis results. FCS processing successfully identifies async boundaries, resource scopes, and effect boundaries in the HelloWorldDirect source code. The bidirectional zipper enables complex PSG transformations while maintaining graph integrity and context.

### Pipeline Success Indicators

Property list generation provides actionable compilation information stored in efficient in-memory data structures. MLIR integration produces valid DCont dialect operations from F# async constructs with preserved semantic information. The compilation pipeline processes the HelloWorldDirect example without errors through all transformation stages.

Curried functions and partial applications are naturally recognized as demand-driven patterns:

```fsharp
// Curried function creates suspension points
let add x y = x + y
let add5 = add 5  // Partial application = demand-driven

// Pipeline recognizes this as DemandDriven pattern
// May preserve as continuation for flexibility
```

### Target Success Indicators

WAMI WebAssembly successfully executes the HelloWorldDirect program with continuation semantics preserved throughout the compilation pipeline. The generated WebAssembly binary runs correctly at the command line and produces expected output. Debug tooling provides clear visibility into continuation flow from F# source through to WebAssembly execution.

## Integration Philosophy

This roadmap preserves the existing `ProgramSemanticGraph` foundation while extending it with continuation-aware analysis capabilities. The approach builds incrementally on proven FCS integration patterns while adding the sophisticated analysis capabilities needed for modern functional language compilation.

The coeffect-based analysis differs fundamentally from traditional effect systems: rather than tracking what code *does* to its environment (effects), we track what code *needs* from its environment (coeffects). This perspective shift is crucial for continuation compilation:
- Effects flow outward (what the code produces)
- Coeffects flow inward (what the code requires)
- Compilation decisions depend on requirements, not products

The bidirectional zipper provides the navigation and transformation capabilities needed for sophisticated compiler passes, while the property list approach separates analysis concerns from the core PSG structure. This separation enables the compiler to maintain multiple analysis perspectives on the same program structure while keeping the core representation clean and focused.

The in-memory processing approach enables rapid iteration and immediate feedback while maintaining architectural coherence. The HelloWorldDirect validation target ensures that all architectural decisions are grounded in concrete compilation requirements rather than theoretical considerations.

The ultimate goal remains unchanged: enabling F# functional programming patterns to compile efficiently to WebAssembly with continuation semantics preserved throughout the pipeline, delivering on the vision of "dcont all the way down" while maintaining the performance and reliability required for systems programming applications.