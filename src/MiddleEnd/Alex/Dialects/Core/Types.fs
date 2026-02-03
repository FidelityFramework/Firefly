/// Alex MLIR Type System
///
/// Structured MLIR type representation for Alex code generation.
/// This is the canonical type system used throughout Alex.
module Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER AND FLOAT WIDTHS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer bit widths supported by MLIR
type IntWidth =
    | I1    // Boolean
    | I8    // Byte
    | I16   // Short
    | I32   // Int
    | I64   // Long

/// Floating-point widths supported by MLIR
type FloatWidth =
    | F32   // Single-precision
    | F64   // Double-precision

// ═══════════════════════════════════════════════════════════════════════════
// MLIR TYPE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Structured MLIR type representation
type MLIRType =
    | TInt of IntWidth
    | TFloat of FloatWidth
    | TPtr                                  // Opaque pointer (!llvm.ptr)
    | TStruct of MLIRType list              // Struct type
    | TArray of int * MLIRType              // Fixed-size array
    | TFunc of MLIRType list * MLIRType     // Function type (args, return)
    | TMemRef of MLIRType                   // MemRef type (dynamic 1D)
    | TMemRefStatic of int * MLIRType       // Static-sized MemRef type (1D with known size)
    | TMemRefScalar of MLIRType             // Scalar MemRef type (0D)
    | TVector of int * MLIRType             // Vector type (SIMD)
    | TIndex                                // Index type
    | TUnit                                 // Unit type (represented as i32 0)
    | TError of string                      // Error type

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Target operating system family
type OSFamily =
    | Linux
    | Windows
    | MacOS
    | FreeBSD

/// Target CPU architecture
type Architecture =
    | X86_64
    | ARM64
    | ARM32_Thumb
    | RISCV64
    | RISCV32
    | WASM32

/// Get the platform word width for a target architecture
/// This determines the size of int, nativeint, size_t, ptrdiff_t, etc.
let platformWordWidth (arch: Architecture) : IntWidth =
    match arch with
    | X86_64 | ARM64 | RISCV64 -> I64
    | ARM32_Thumb | RISCV32 | WASM32 -> I32

// ═══════════════════════════════════════════════════════════════════════════
// SSA VALUES AND BLOCK REFERENCES
// ═══════════════════════════════════════════════════════════════════════════

/// SSA value reference - the currency of MLIR operations
/// V = value from computation, Arg = function/block argument
[<Struct>]
type SSA =
    | V of int      // %v0, %v1, ...
    | Arg of int    // %arg0, %arg1, ...

/// Block label reference
[<Struct>]
type BlockRef = BlockRef of string

/// A typed SSA value (value + its type)
type Val = {
    SSA: SSA
    Type: MLIRType
}

/// Create a typed value
let val' ssa ty = { SSA = ssa; Type = ty }

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON PREDICATES
// ═══════════════════════════════════════════════════════════════════════════

/// Integer comparison predicates (for arith.cmpi)
type ICmpPred =
    | Eq | Ne                       // Equal, Not equal
    | Slt | Sle | Sgt | Sge         // Signed comparisons
    | Ult | Ule | Ugt | Uge         // Unsigned comparisons

/// Float comparison predicates (for arith.cmpf)
type FCmpPred =
    // Ordered comparisons (return false if either operand is NaN)
    | OEq | ONe | OLt | OLe | OGt | OGe
    // Unordered comparisons (return true if either operand is NaN)
    | UEq | UNe | ULt | ULe | UGt | UGe
    // Special predicates
    | Ord   // Ordered (neither operand is NaN)
    | Uno   // Unordered (either operand is NaN)
    // False/True (always)
    | AlwaysFalse | AlwaysTrue

/// Index comparison predicates (for index.cmp)
type IndexCmpPred =
    | Eq | Ne                       // Equal, Not equal
    | Slt | Sle | Sgt | Sge         // Signed comparisons
    | Ult | Ule | Ugt | Uge         // Unsigned comparisons

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY ORDERING AND VISIBILITY
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic ordering for memory operations (LLVM atomic semantics)
type AtomicOrdering =
    | NotAtomic                     // Non-atomic operation
    | Unordered                     // Lowest level of atomicity
    | Monotonic                     // Acquire/release semantics
    | Acquire                       // Acquire barrier
    | Release                       // Release barrier
    | AcquireRelease                // Both acquire and release
    | SequentiallyConsistent        // Strongest ordering

/// Function visibility for external linking
type FuncVisibility =
    | Public                        // Visible to all modules
    | Private                       // Visible only within module
    | Internal                      // Visible within compilation unit

// ═══════════════════════════════════════════════════════════════════════════
// MLIR OPERATIONS (for type-safe MLIR construction)
// ═══════════════════════════════════════════════════════════════════════════

/// MemRef dialect operations (standard MLIR memory operations)
type MemRefOp =
    | Load of SSA * SSA * SSA list * MLIRType                          // result, memref, indices, type
    | Store of SSA * SSA * SSA list * MLIRType                         // value, memref, indices, type
    | Alloca of SSA * MLIRType * int option                            // result, memrefType, alignment
    | SubView of SSA * SSA * SSA list * MLIRType                       // result, source, offsets, resultType
    | ExtractBasePtr of SSA * SSA * MLIRType                           // result, memref, memrefType → !llvm.ptr (for FFI)
    | GetGlobal of SSA * string * MLIRType                             // result, globalName, memrefType
    | Dim of SSA * SSA * SSA * MLIRType                                // result, memref, dimIndex, memrefType (returns index)
    | Cast of SSA * SSA * MLIRType * MLIRType                          // result, source, srcType, destType (memref type cast)

/// Arithmetic dialect operations
type ArithOp =
    // Constants
    | ConstI of SSA * int64 * MLIRType                      // result, value, type
    | ConstF of SSA * float * MLIRType                      // result, value, type
    // Integer arithmetic
    | AddI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | MulI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | DivUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | RemSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | RemUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | CmpI of SSA * ICmpPred * SSA * SSA * MLIRType         // result, predicate, lhs, rhs, type
    // Float arithmetic
    | AddF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | MulF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | CmpF of SSA * FCmpPred * SSA * SSA * MLIRType         // result, predicate, lhs, rhs, type
    // Type conversions
    | ExtSI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | ExtUI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | TruncI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | SIToFP of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | FPToSI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    // Bitwise operations (migrated from LLVM dialect)
    | AndI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | OrI of SSA * SSA * SSA * MLIRType                     // result, lhs, rhs, type
    | XorI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | ShLI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type (shift left)
    | ShRUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (logical shift right)
    | ShRSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (arithmetic shift right)
    // Other
    | Select of SSA * SSA * SSA * SSA * MLIRType            // result, cond, trueVal, falseVal, type

/// Index dialect operations
type IndexOp =
    | IndexConst of SSA * int64                       // result, value
    | IndexBoolConst of SSA * bool                    // result, value
    | IndexAdd of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexSub of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexMul of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexDivS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexDivU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexCeilDivS of SSA * SSA * SSA                // result, lhs, rhs (signed ceiling)
    | IndexCeilDivU of SSA * SSA * SSA                // result, lhs, rhs (unsigned ceiling)
    | IndexFloorDivS of SSA * SSA * SSA               // result, lhs, rhs (signed floor)
    | IndexRemS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexRemU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexMaxS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexMaxU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexMinS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexMinU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexShl of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexShrS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexShrU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexAnd of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexOr of SSA * SSA * SSA                      // result, lhs, rhs
    | IndexXor of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexCmp of SSA * IndexCmpPred * SSA * SSA      // result, predicate, lhs, rhs
    | IndexCastS of SSA * SSA * MLIRType              // result, operand, destType (signed)
    | IndexCastU of SSA * SSA * MLIRType              // result, operand, destType (unsigned)
    | IndexSizeOf of SSA * MLIRType                   // result, type

/// Vector dialect operations
type VectorOp =
    | Broadcast of SSA * SSA * MLIRType                                     // result, source, resultType
    | Extract of SSA * SSA * int list                                       // result, vector, position
    | Insert of SSA * SSA * SSA * int list                                  // result, source, dest, position
    | ExtractStrided of SSA * SSA * int list * int list * int list          // result, vector, offsets, sizes, strides
    | InsertStrided of SSA * SSA * SSA * int list * int list                // result, source, dest, offsets, strides
    | ShapeCast of SSA * SSA * MLIRType                                     // result, source, resultType
    | Transpose of SSA * SSA * int list                                     // result, vector, transp
    | FlattenTranspose of SSA * SSA                                         // result, vector
    | ReductionAdd of SSA * SSA * SSA option                                // result, vector, acc
    | ReductionMul of SSA * SSA * SSA option                                // result, vector, acc
    | ReductionAnd of SSA * SSA                                             // result, vector
    | ReductionOr of SSA * SSA                                              // result, vector
    | ReductionXor of SSA * SSA                                             // result, vector
    | ReductionMinSI of SSA * SSA                                           // result, vector
    | ReductionMinUI of SSA * SSA                                           // result, vector
    | ReductionMaxSI of SSA * SSA                                           // result, vector
    | ReductionMaxUI of SSA * SSA                                           // result, vector
    | ReductionMinF of SSA * SSA                                            // result, vector
    | ReductionMaxF of SSA * SSA                                            // result, vector
    | FMA of SSA * SSA * SSA * SSA                                          // result, lhs, rhs, acc
    | Splat of SSA * SSA * MLIRType                                         // result, value, resultType
    | VectorLoad of SSA * SSA * SSA list                                    // result, basePtr, indices
    | VectorStore of SSA * SSA * SSA list                                   // valueToStore, basePtr, indices
    | MaskedLoad of SSA * SSA * SSA list * SSA * SSA                        // result, basePtr, indices, mask, passthru
    | MaskedStore of SSA * SSA * SSA list * SSA                             // valueToStore, basePtr, indices, mask
    | Gather of SSA * SSA * SSA * SSA * SSA * SSA                           // result, basePtr, indices, indexVec, mask, passthru
    | Scatter of SSA * SSA * SSA * SSA * SSA                                // valueToStore, basePtr, indices, indexVec, mask
    | CreateMask of SSA * SSA list                                          // result, operands
    | ConstantMask of SSA * int list                                        // result, maskDimSizes
    | Print of SSA * string option                                          // source, punctuation

/// Top-level MLIR operation (all dialects)
/// Single-phase execution with nested accumulators - no scope markers needed
type MLIROp =
    | MemRefOp of MemRefOp
    | ArithOp of ArithOp
    | SCFOp of SCFOp
    | CFOp of CFOp
    | FuncOp of FuncOp
    | IndexOp of IndexOp
    | VectorOp of VectorOp
    | Block of string * MLIROp list                                 // label, ops
    | Region of MLIROp list                                         // blocks
    // Module-level declarations (backend-agnostic)
    | GlobalString of string * string * int                         // name, content, byteLength

/// Structured Control Flow (SCF) dialect operations
and SCFOp =
    | If of SSA * MLIROp list * MLIROp list option            // cond, thenOps, elseOps
    | While of MLIROp list * MLIROp list                      // condOps, bodyOps
    | For of SSA * SSA * SSA * MLIROp list                    // lower, upper, step, bodyOps
    | Yield of SSA list                                       // values
    | Condition of SSA * SSA list                             // cond, args

/// Control Flow (CF) dialect operations - unstructured control flow
and CFOp =
    | Assert of SSA * string                                                      // cond, msg
    | Br of BlockRef * Val list                                                   // dest, destOperands
    | CondBr of SSA * BlockRef * Val list * BlockRef * Val list * (int * int) option  // cond, trueDest, trueOps, falseDest, falseOps, weights
    | Switch of SSA * MLIRType * BlockRef * Val list * (int64 * BlockRef * Val list) list  // flag, flagTy, default, defaultOps, cases

/// Function dialect operations
and FuncOp =
    // Function definition/declaration
    | FuncDef of string * (SSA * MLIRType) list * MLIRType * MLIROp list * FuncVisibility  // name, args, retType, body, visibility
    | FuncDecl of string * MLIRType list * MLIRType * FuncVisibility                       // name, paramTypes, retType, visibility (external decl)
    | ExternDecl of string * MLIRType list * MLIRType                                      // name, paramTypes, retType (backward compat)
    // Function calls
    | FuncCall of SSA option * string * Val list * MLIRType                                // result, func, args, retType
    | FuncCallIndirect of SSA option * SSA * Val list * MLIRType                           // result, callee, args, retType
    | FuncConstant of SSA * string * MLIRType                                              // result, funcName, funcType
    // Return
    | Return of SSA option * MLIRType option                                               // value, type
