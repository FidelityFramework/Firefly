# XParsec and MLIR TableGen Generation: Architecture Memo

## Executive Summary

XParsec serves as the semantic bridge between F# source code and MLIR's transformation infrastructure. Rather than directly generating MLIR IR, XParsec analyzes F# programs to discover patterns and generates TableGen definitions that teach MLIR how to optimize these patterns. This memo describes the two-pass architecture for transforming F# through MLIR to LLVM IR, with particular focus on how XParsec drives the generation of transformation rules.

## Architectural Overview

The compilation pipeline follows a deliberate separation of concerns:

1. **Semantic Analysis Phase**: XParsec parses F# and identifies optimization opportunities
2. **Code Generation Phase**: XParsec generates TableGen patterns based on discovered semantics
3. **MLIR Integration Phase**: Generated TableGen is compiled into C++ transformation passes
4. **Transformation Phase**: MLIR applies these passes to lower high-level constructs to LLVM IR

This architecture enables F#-specific optimizations while leveraging MLIR's robust transformation infrastructure.

## Phase 1: Semantic Pattern Recognition

XParsec's primary role is to recognize semantic patterns in F# code that can benefit from specialized lowering strategies. Unlike traditional parsing that merely builds an AST, XParsec constructs a semantic understanding of the program.

### Pattern Discovery

Consider a simple F# function:

```fsharp
let processBuffer (data: byte[]) =
    data 
    |> Array.map (fun b -> b * 2uy)
    |> Array.filter (fun b -> b > 100uy)
    |> Array.sum
```

XParsec doesn't just see function calls - it recognizes a "map-filter-reduce" pattern that can be optimized as a single pass. The parser combinator might look like:

```fsharp
let mapFilterReducePattern =
    parser {
        let! mapper = captureMapping
        let! predicate = captureFilter  
        let! reducer = captureReduction
        
        return 
            match analyzeDataflow mapper predicate reducer with
            | CanFuse props -> FusedMapFilterReduce props
            | CannotFuse -> SeparateOperations
    }
```

### Semantic Properties

XParsec extracts properties beyond syntax:

- Data dependencies between operations
- Potential for operation fusion
- Memory access patterns
- Type-specific optimization opportunities

These properties inform the TableGen generation phase.

## Phase 2: TableGen Pattern Generation

Based on recognized patterns, XParsec generates TableGen definitions that encode transformation rules. This is where semantic understanding becomes actionable optimization strategy.

### Generated TableGen Structure

For the map-filter-reduce pattern above, XParsec might generate:

```tablegen
def MapFilterReducePattern : Pat<
    (Firefly_ReduceOp 
        (Firefly_FilterOp 
            (Firefly_MapOp $input, $mapper), 
            $predicate), 
        $reducer),
    (Firefly_FusedMapFilterReduceOp $input, $mapper, $predicate, $reducer),
    [(HasNoSideEffects $mapper),
     (HasNoSideEffects $predicate),
     (CanFuseOperations $mapper, $predicate, $reducer)]>;
```

### Constraint Generation

XParsec generates constraints based on its semantic analysis:

```fsharp
let generateConstraints pattern =
    match pattern with
    | FusedMapFilterReduce props ->
        [ sprintf "(HasNoSideEffects %s)" props.mapper
          sprintf "(IsElementwise %s)" props.mapper
          sprintf "(SinglePassPossible %s, %s)" props.mapper props.filter ]
    | _ -> []
```

These constraints ensure transformations only apply when semantically valid.

## Phase 3: MLIR Integration

The generated TableGen files are processed by MLIR's build system to create C++ transformation passes. This phase is largely automated but crucial for understanding the pipeline.

### Build Process

1. **TableGen Compilation**: `mlir-tblgen` processes the generated `.td` files
2. **C++ Generation**: Produces pattern matching and rewriting code
3. **Pass Registration**: Integrates with MLIR's pass infrastructure

### Generated C++ Structure

The TableGen definitions expand into C++ classes:

```cpp
struct MapFilterReducePatternRewriter : public OpRewritePattern<ReduceOp> {
    LogicalResult matchAndRewrite(ReduceOp op, 
                                  PatternRewriter &rewriter) const override {
        // Generated matching logic
        auto filterOp = op.getInput().getDefiningOp<FilterOp>();
        if (!filterOp) return failure();
        
        auto mapOp = filterOp.getInput().getDefiningOp<MapOp>();
        if (!mapOp) return failure();
        
        // Constraint checking (generated from TableGen)
        if (!hasNoSideEffects(mapOp.getMapper()))
            return failure();
            
        // Rewriting logic
        rewriter.replaceOpWithNewOp<FusedMapFilterReduceOp>(
            op, mapOp.getInput(), mapOp.getMapper(), 
            filterOp.getPredicate(), op.getReducer());
            
        return success();
    }
};
```

## Phase 4: Two-Pass Transformation Strategy

For the initial Hello World implementation, we employ a two-pass strategy that balances simplicity with effectiveness.

### Pass 1: High-Level Pattern Transformation

The first pass applies F#-specific optimizations identified by XParsec:

```mlir
// Before Pass 1
func.func @processBuffer(%arg0: !firefly.array<ui8>) -> ui8 {
  %0 = firefly.map %arg0 {
    ^bb0(%elem: ui8):
      %1 = arith.muli %elem, 2 : ui8
      firefly.yield %1 : ui8
  } : !firefly.array<ui8> -> !firefly.array<ui8>
  
  %2 = firefly.filter %0 {
    ^bb0(%elem: ui8):
      %3 = arith.cmpi ugt, %elem, 100 : ui8
      firefly.yield %3 : i1
  } : !firefly.array<ui8> -> !firefly.array<ui8>
  
  %4 = firefly.reduce %2, 0 : ui8 {
    ^bb0(%acc: ui8, %elem: ui8):
      %5 = arith.addi %acc, %elem : ui8
      firefly.yield %5 : ui8
  } : !firefly.array<ui8> -> ui8
  
  return %4 : ui8
}

// After Pass 1
func.func @processBuffer(%arg0: !firefly.array<ui8>) -> ui8 {
  %0 = firefly.fused_map_filter_reduce %arg0, 0 : ui8 {
    ^bb0(%elem: ui8, %acc: ui8):
      %1 = arith.muli %elem, 2 : ui8
      %2 = arith.cmpi ugt, %1, 100 : ui8
      %3 = scf.if %2 -> ui8 {
        %4 = arith.addi %acc, %1 : ui8
        scf.yield %4 : ui8
      } else {
        scf.yield %acc : ui8
      }
      firefly.yield %3 : ui8
  } : !firefly.array<ui8> -> ui8
  
  return %0 : ui8
}
```

### Pass 2: Lowering to LLVM Dialect

The second pass converts Firefly-specific operations to LLVM dialect:

```mlir
// After Pass 2
llvm.func @processBuffer(%arg0: !llvm.ptr<i8>) -> i8 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i8) : i8
  %c100 = llvm.mlir.constant(100 : i8) : i8
  %acc_init = llvm.mlir.constant(0 : i8) : i8
  
  // Get array length
  %len_ptr = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
  %len = llvm.load %len_ptr : !llvm.ptr<i64>
  
  // Get data pointer
  %data_ptr = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
  %data = llvm.load %data_ptr : !llvm.ptr<ptr<i8>>
  
  // Fused loop
  %result = llvm.br ^loop(%c0, %acc_init : i64, i8)
  
^loop(%i: i64, %acc: i8):
  %cond = llvm.icmp "ult" %i, %len : i64
  llvm.cond_br %cond, ^body, ^exit(%acc : i8)
  
^body:
  %elem_ptr = llvm.getelementptr %data[%i] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  %elem = llvm.load %elem_ptr : !llvm.ptr<i8>
  %doubled = llvm.mul %elem, %c2 : i8
  %gt_100 = llvm.icmp "ugt" %doubled, %c100 : i8
  %new_acc = llvm.select %gt_100, %doubled, %acc : i8
  %next_i = llvm.add %i, %c1 : i64
  llvm.br ^loop(%next_i, %new_acc : i64, i8)
  
^exit(%final: i8):
  llvm.return %final : i8
}
```

## XParsec Pattern Library

As the compiler evolves, XParsec builds a library of recognized patterns. Each pattern includes:

1. **Recognition Logic**: Parser combinators that identify the pattern
2. **Semantic Analysis**: Properties and constraints extraction
3. **TableGen Template**: Generated transformation rules
4. **Test Cases**: Verification of correct transformation

### Pattern Categories

For the initial implementation, focus on:

- **Collection Operations**: map, filter, fold, reduce
- **Immutable Updates**: record/DU updates that can be optimized
- **Tail Recursion**: Converting to loops
- **Pipeline Operations**: Function composition optimization

## Implementation Timeline

### Phase 1: Foundation

- Basic XParsec to TableGen generation
- Simple pattern recognition (map, filter)
- Manual verification of generated patterns

### Phase 2: Integration

- Automated build pipeline
- MLIR pass registration
- First working transformations

### Phase 3: Expansion (Ongoing)

- Additional pattern recognition
- Performance measurement
- Pattern library growth

## Key Design Decisions

### Why Generate TableGen?

1. **Separation of Concerns**: Pattern recognition logic stays in F#
2. **Leverage MLIR Infrastructure**: Reuse battle-tested transformation framework
3. **Maintainability**: TableGen patterns are declarative and reviewable
4. **Extensibility**: New patterns can be added without changing core infrastructure

### Why Two Passes?

1. **Clarity**: Each pass has a clear purpose
2. **Debuggability**: Can inspect intermediate representation
3. **Composability**: Passes can be reordered or skipped
4. **Testing**: Each pass can be tested independently

## Success Metrics

The effectiveness of this approach will be measured by:

1. **Correctness**: All transformations preserve program semantics
2. **Performance**: Generated code matches hand-optimized equivalents
3. **Coverage**: Percentage of F# patterns recognized and optimized
4. **Maintainability**: Ease of adding new patterns

## Conclusion

XParsec's role in the MLIR pipeline is to bridge the semantic gap between F#'s high-level constructs and LLVM's low-level representation. By generating TableGen patterns, we leverage MLIR's transformation infrastructure while maintaining F#-specific optimization knowledge. This architecture provides a solid foundation for evolving the Firefly compiler toward more sophisticated optimizations while keeping the implementation manageable and debuggable.
