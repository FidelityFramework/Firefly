# PRD-12: Higher-Order Functions

> **Sample**: `12_HigherOrderFunctions` | **Status**: Planned | **Depends On**: PRD-11 (Closures)

## 1. Executive Summary

Higher-order functions (HOFs) are functions that take functions as arguments or return functions as results. This PRD builds directly on the closure infrastructure from PRD-11 - HOFs are essentially the *use* of closures.

**Key Insight**: If closures work correctly, HOFs are largely "free" - they're just function values being passed around. The remaining work is ensuring function types propagate correctly through FNCS type inference and that Alex handles function-typed parameters/returns.

## 2. Language Feature Specification

### 2.1 HOF Patterns

```fsharp
// Function as parameter
let applyTwice (f: int -> int) (x: int) : int =
    f (f x)

// Function as return value
let makeAdder (n: int) : (int -> int) =
    fun x -> x + n

// Function composition
let compose (f: int -> int) (g: int -> int) : (int -> int) =
    fun x -> g (f x)
```

### 2.2 Type Representation

Function types `A -> B` are represented as:
- **No captures**: Simple function pointer (`ptr<fn(A) -> B>`)
- **With captures**: Closure struct (`{ ptr<fn(env, A) -> B>, ptr<env> }`)

The caller doesn't need to know which representation is used - both are called the same way (closure calling convention always passes env pointer, even if unused).

### 2.3 Calling Convention Unification

**All function values use closure calling convention**:
```
call(closure, args...) = extractCodePtr(closure)(extractEnv(closure), args...)
```

For functions without captures, the env pointer is a dummy (null or undefined) that the callee ignores.

## 3. FNCS Layer Implementation

### 3.1 Function Type in Parameters

FNCS already handles function types - verify that:

```fsharp
// In CheckExpressions.fs - parameter type checking
| SynType.Fun(argType, returnType, _) ->
    let argTy = checkType env argType
    let retTy = checkType env returnType
    NativeType.TFun(argTy, retTy)
```

### 3.2 Function Application to Function-Typed Arguments

When `f` is a parameter of type `A -> B`:
```fsharp
// checkApp in Applications.fs
match funcType with
| NativeType.TFun(paramTy, retTy) ->
    // f is callable, arg must match paramTy
    let argNode = checkExpr env builder arg
    unify argNode.Type paramTy
    // Result has type retTy
```

### 3.3 No New SemanticKinds Needed

HOF usage relies entirely on existing kinds:
- `Lambda` - for creating function values
- `App` - for applying functions
- `VarRef` - for referencing function-typed bindings
- `PatternBinding` - for binding function-typed parameters

## 4. Firefly/Alex Layer Implementation

### 4.1 Function-Typed Parameters

When a function parameter has function type, SSAAssignment treats it like any other parameter - it gets an SSA for the closure struct value:

```fsharp
// In lambda parameter processing
| paramType when isFunctionType paramType ->
    // Parameter is a closure struct
    let paramSSA = freshSSA ()
    bindParameter name paramSSA paramType
```

### 4.2 Function Invocation Through Parameter

When calling a function-typed parameter:

```fsharp
// witnessApp when callee is a VarRef to function parameter
let calleeSSA = lookupVarSSA calleeName z
let extractCode = freshSynthSSA z
let extractEnv = freshSynthSSA z

emit $"  %%{extractCode} = llvm.extractvalue %%{calleeSSA}[0]"
emit $"  %%{extractEnv} = llvm.extractvalue %%{calleeSSA}[1]"
emit $"  %%{resultSSA} = llvm.call %%{extractCode}(%%{extractEnv}, {argSSAs})"
```

### 4.3 Returning Function Values

When a function returns a function type, the return value is a closure struct:

```fsharp
// witnessReturn when return type is function
| returnType when isFunctionType returnType ->
    // valueSSA is already a closure struct
    emit $"  llvm.return %%{valueSSA}"
```

## 5. MLIR Output Specification

### 5.1 Function Taking Function Parameter

```mlir
// applyTwice : (int -> int) -> int -> int
llvm.func @applyTwice(%f: !closure_type, %x: i32) -> i32 {
    // First application: f x
    %code1 = llvm.extractvalue %f[0] : !closure_type -> !llvm.ptr
    %env1 = llvm.extractvalue %f[1] : !closure_type -> !llvm.ptr
    %r1 = llvm.call %code1(%env1, %x) : (!llvm.ptr, i32) -> i32

    // Second application: f (f x)
    %code2 = llvm.extractvalue %f[0] : !closure_type -> !llvm.ptr
    %env2 = llvm.extractvalue %f[1] : !closure_type -> !llvm.ptr
    %r2 = llvm.call %code2(%env2, %r1) : (!llvm.ptr, i32) -> i32

    llvm.return %r2 : i32
}
```

### 5.2 Function Returning Function

```mlir
// makeAdder : int -> (int -> int)
llvm.func @makeAdder(%n: i32) -> !closure_type {
    // Allocate environment for captured 'n'
    %env = llvm.alloca 1 x !llvm.struct<(i32)>
    %slot0 = llvm.getelementptr %env[0, 0]
    llvm.store %n, %slot0

    // Build closure struct
    %tmp = llvm.insertvalue undef : !closure_type[0], @adder_impl
    %closure = llvm.insertvalue %tmp[1], %env

    llvm.return %closure : !closure_type
}

llvm.func @adder_impl(%env: !llvm.ptr, %x: i32) -> i32 {
    %n_ptr = llvm.getelementptr %env[0, 0]
    %n = llvm.load %n_ptr : i32
    %result = arith.addi %x, %n : i32
    llvm.return %result : i32
}
```

## 6. Validation

### 6.1 Sample Code

**File**: `samples/console/FidelityHelloWorld/12_HigherOrderFunctions/HigherOrderFunctions.fs`

Key test cases:
- `applyTwice` - function as parameter, applied twice
- `mapPair` - function applied to multiple values
- `chooseAndApply` - conditional function selection
- `makeAdder/makeMultiplier` - function factories (closures)
- `compose` - function composition

### 6.2 Expected Output

```
=== Higher-Order Functions Test ===
increment twice on 5: 7
double twice on 3: 12

double both (3, 7): 6, 14

chooseAndApply true double square 5: 10
chooseAndApply false double square 5: 25

add10 applied to 5: 15
mult3 applied to 7: 21

double then increment on 5: 11
increment then double on 5: 12
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| None expected | - | HOF support should already exist if closures work |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Witnesses/LambdaWitness.fs` | VERIFY | Closure invocation handles function-typed callees |
| `src/Alex/Traversal/FNCSTransfer.fs` | VERIFY | App witness handles indirect calls |

## 8. Implementation Checklist

- [ ] Verify FNCS type-checks function-typed parameters
- [ ] Verify FNCS type-checks function return types
- [ ] Verify Alex handles function-typed parameter SSAs
- [ ] Verify Alex emits correct indirect call for function parameters
- [ ] Verify closure calling convention works uniformly
- [ ] Sample 12 compiles without errors
- [ ] Sample 12 produces correct output
- [ ] Samples 01-11 still pass

## 9. Risk Assessment

**Low Risk**: HOFs are primarily a type system feature. If PRD-11 (Closures) is complete, HOFs should largely "just work." The main verification is ensuring:
1. Function types flow correctly through inference
2. Indirect calls through function-typed bindings emit correct MLIR

## 10. Related PRDs

- **PRD-11**: Closures - Required foundation
- **PRD-14-15**: Sequences - Will use HOFs (`Seq.map`, `Seq.filter`)
- **PRD-29-31**: MailboxProcessor - Behavior functions are HOFs
