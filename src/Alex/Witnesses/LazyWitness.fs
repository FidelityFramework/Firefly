/// Lazy Witness - Witness Lazy<'T> values to MLIR
///
/// PRD-14: Deferred computation with memoization
///
/// ARCHITECTURAL PRINCIPLES (Four Pillars):
/// 1. Coeffects: SSA assignment is pre-computed, lookup via context
/// 2. Active Patterns: Match on semantic meaning (LazyExpr, LazyForce)
/// 3. Zipper: Navigate and accumulate structured ops
/// 4. Templates: Return structured MLIROp types, no sprintf
///
/// FLAT CLOSURE ARCHITECTURE (January 2026):
/// Lazy values are "extended flat closures" - captures are inlined directly.
/// NO env_ptr, NO nulls - following MLKit-style flat closure principles.
///
/// LAZY STRUCT LAYOUT:
/// !lazy_T = !llvm.struct<(i1, T, ptr, cap₀, cap₁, ...)>
///   - Field 0: Computed flag (i1)
///   - Field 1: Cached value (T) - valid when computed=true
///   - Field 2: Thunk function pointer
///   - Field 3+: Inlined captured values
///
/// THUNK CALLING CONVENTION (Option B - January 2026):
/// Thunk receives pointer to lazy struct, extracts its own captures:
///   llvm.func @thunk(%lazy_ptr: ptr) -> T {
///       %lazy = llvm.load %lazy_ptr : !lazy_struct_type
///       %cap0 = llvm.extractvalue %lazy[3] : !lazy_struct_type
///       %cap1 = llvm.extractvalue %lazy[4] : !lazy_struct_type
///       // ... compute with captures ...
///       llvm.return %result : T
///   }
///
/// FORCE IS UNIFORM:
/// Force doesn't need to know capture count - just passes pointer to thunk:
///   %code_ptr = llvm.extractvalue %lazy[2]
///   %lazy_ptr = llvm.alloca !lazy_struct_type
///   llvm.store %lazy, %lazy_ptr
///   %result = llvm.call %code_ptr(%lazy_ptr) : (ptr) -> T
///
/// OPERATIONS:
/// - witnessLazyCreate: Build lazy struct with thunk and captures
/// - witnessLazyForce: Uniform force - passes pointer to thunk
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// LAZY STRUCT TYPE (FLAT CLOSURE MODEL)
// ═══════════════════════════════════════════════════════════════════════════

/// Build the MLIR type for Lazy<T> with N captures
/// Layout: { computed: i1, value: T, code_ptr: ptr, cap₀, cap₁, ... }
let lazyStructType (elementType: MLIRType) (captureTypes: MLIRType list) : MLIRType =
    // {i1, T, ptr, cap₀, cap₁, ...}
    TStruct ([TInt I1; elementType; TPtr] @ captureTypes)

/// Lazy struct type with no captures (simplest case)
let lazyStructTypeNoCaptures (elementType: MLIRType) : MLIRType =
    // {i1, T, ptr}
    TStruct [TInt I1; elementType; TPtr]

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.CREATE - Build deferred computation (FLAT CLOSURE)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness lazy expression creation
///
/// Input:
///   - thunkName: Name of the thunk function
///   - elementType: The type T in Lazy<T>
///   - captureVals: Values captured by the lazy expression
///
/// Output:
///   - Lazy struct: {computed=false, value=undef, code_ptr, cap₀, cap₁, ...}
///
/// SSA cost: 5 + numCaptures
///   - 1: false constant
///   - 1: undef struct
///   - 1: insert computed flag
///   - 1: addressof code_ptr
///   - 1: insert code_ptr
///   - N: insert each capture
let witnessLazyCreate
    (appNodeId: NodeId)
    (z: PSGZipper)
    (thunkName: string)
    (elementType: MLIRType)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let captureTypes = captureVals |> List.map (fun v -> v.Type)
    let lazyType = lazyStructType elementType captureTypes
    let ssas = requireNodeSSAs appNodeId z

    // Pre-assigned SSAs (from SSAAssignment coeffect)
    let falseSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withComputedSSA = ssas.[2]
    let codePtrSSA = ssas.[3]
    let withCodePtrSSA = ssas.[4]
    // ssas.[5..] for captures

    let baseOps = [
        // Create false constant for computed flag
        MLIROp.ArithOp (ArithOp.ConstI (falseSSA, 0L, MLIRTypes.i1))

        // Create undef lazy struct
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, lazyType))

        // Insert computed=false at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (withComputedSSA, undefSSA, falseSSA, [0], lazyType))

        // Get thunk function address
        MLIROp.LLVMOp (LLVMOp.AddressOf (codePtrSSA, GFunc thunkName))

        // Insert code_ptr at index 2 (skip value at index 1)
        MLIROp.LLVMOp (LLVMOp.InsertValue (withCodePtrSSA, withComputedSSA, codePtrSSA, [2], lazyType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps, finalSSA =
        if captureVals.IsEmpty then
            [], withCodePtrSSA
        else
            let ops, lastSSA =
                captureVals
                |> List.indexed
                |> List.fold (fun (accOps, prevSSA) (i, capVal) ->
                    let nextSSA = ssas.[5 + i]
                    let captureIndex = 3 + i
                    let op = MLIROp.LLVMOp (LLVMOp.InsertValue (nextSSA, prevSSA, capVal.SSA, [captureIndex], lazyType))
                    (accOps @ [op], nextSSA)
                ) ([], withCodePtrSSA)
            ops, lastSSA

    (baseOps @ captureOps, TRValue { SSA = finalSSA; Type = lazyType })

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.FORCE - Evaluate thunk (OPTION B: UNIFORM CALLING CONVENTION)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy.force: Lazy<'T> -> 'T
///
/// OPTION B CALLING CONVENTION (January 2026):
/// Force is UNIFORM - doesn't need to know capture count.
/// 1. Create constant 1 for alloca count
/// 2. Extract code_ptr from lazy struct [2]
/// 3. Alloca space for lazy struct on stack
/// 4. Store lazy struct to alloca
/// 5. Call thunk with pointer to lazy struct
/// 6. Return result
///
/// The thunk receives the lazy struct pointer and extracts its own captures.
/// This provides clean separation - force site doesn't need capture info.
///
/// NOTE: This is pure thunk semantics - always evaluates.
/// True memoization (caching) requires mutation and will be added
/// after Arena PRDs (20-22) provide memory management support.
///
/// SSA cost: 4 (fixed, regardless of captures)
///   - 1: extract code_ptr
///   - 1: constant 1 for alloca count
///   - 1: alloca for lazy struct
///   - 1: indirect call result
let witnessLazyForce
    (appNodeId: NodeId)
    (z: PSGZipper)
    (lazyVal: Val)
    (elementType: MLIRType)
    : (MLIROp list * TransferResult) =

    let ssas = requireNodeSSAs appNodeId z
    let lazyStructType = lazyVal.Type

    // Pre-assigned SSAs (from SSAAssignment coeffect)
    // SSA cost is fixed: 4 SSAs (extract, const 1, alloca, call result)
    let codePtrSSA = ssas.[0]
    let oneSSA = ssas.[1]
    let allocaPtrSSA = ssas.[2]
    let resultSSA = ssas.[3]

    let ops = [
        // Extract code_ptr from lazy struct (index 2)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, lazyVal.SSA, [2], lazyStructType))

        // Create constant 1 for alloca count
        MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))

        // Alloca space for lazy struct on stack
        MLIROp.LLVMOp (LLVMOp.Alloca (allocaPtrSSA, oneSSA, lazyStructType, None))

        // Store lazy struct to alloca
        MLIROp.LLVMOp (LLVMOp.Store (lazyVal.SSA, allocaPtrSSA, lazyStructType, AtomicOrdering.NotAtomic))

        // Call thunk with pointer to lazy struct
        // Thunk signature: (ptr) -> T
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, [{ SSA = allocaPtrSSA; Type = TPtr }], elementType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = elementType })

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.ISVALUECREATED - Check if computed
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy.isValueCreated: Lazy<'T> -> bool
/// Returns the computed flag from the lazy struct
let witnessLazyIsValueCreated
    (appNodeId: NodeId)
    (z: PSGZipper)
    (lazyVal: Val)
    : (MLIROp list * TransferResult) =

    let resultSSA = requireNodeSSA appNodeId z

    let ops = [
        // Extract computed flag from lazy struct (index 0)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, lazyVal.SSA, [0], lazyVal.Type))
    ]

    (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 })
