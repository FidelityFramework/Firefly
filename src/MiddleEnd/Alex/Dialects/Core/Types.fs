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
    | TMemRef of MLIRType                   // MemRef type
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

// ═══════════════════════════════════════════════════════════════════════════
// MLIR OPERATIONS (for type-safe MLIR construction)
// ═══════════════════════════════════════════════════════════════════════════

/// LLVM dialect operations
type LLVMOp =
    | Load of SSA * SSA * MLIRType                          // result, ptr, type
    | Store of SSA * SSA                                     // value, ptr
    | GEP of SSA * SSA * int list * MLIRType                // result, ptr, indices, type
    | Alloca of SSA * MLIRType * int                        // result, type, count
    | Call of SSA * string * (SSA * MLIRType) list * MLIRType  // result, callee, args, retType
    | ExtractValue of SSA * SSA * int list * MLIRType       // result, struct, indices, type
    | InsertValue of SSA * SSA * SSA * int list * MLIRType  // result, struct, value, indices, type
    | Undef of SSA * MLIRType                               // result, type
    | BitCast of SSA * SSA * MLIRType * MLIRType            // result, value, srcType, destType
    | IntToPtr of SSA * SSA * MLIRType                      // result, value, ptrType
    | PtrToInt of SSA * SSA * MLIRType                      // result, ptr, intType
    | Branch of string                                       // label
    | CondBranch of SSA * string * string                   // cond, trueLabel, falseLabel
    | Phi of SSA * (SSA * string) list * MLIRType           // result, predecessors, type
    | InlineAsm of string * string * bool * SSA * SSA list  // template, constraints, sideEffects, result, args

/// Arithmetic dialect operations
type ArithOp =
    | ConstI of SSA * int64 * MLIRType                      // result, value, type
    | ConstF of SSA * float * MLIRType                      // result, value, type
    | AddI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | MulI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | DivUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | RemSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | RemUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | CmpI of SSA * string * SSA * SSA * MLIRType           // result, predicate, lhs, rhs, type
    | AddF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | MulF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | CmpF of SSA * string * SSA * SSA * MLIRType           // result, predicate, lhs, rhs, type
    | ExtSI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | ExtUI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | TruncI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | SIToFP of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | FPToSI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | Select of SSA * SSA * SSA * SSA * MLIRType            // result, cond, trueVal, falseVal, type

/// Top-level MLIR operation (all dialects) - forward declaration for mutual recursion
type MLIROp =
    | LLVMOp of LLVMOp
    | ArithOp of ArithOp
    | SCFOp of SCFOp
    | CFOp of CFOp
    | FuncOp of FuncOp
    | Block of string * MLIROp list                                 // label, ops
    | Region of MLIROp list                                         // blocks

/// Structured Control Flow (SCF) dialect operations
and SCFOp =
    | If of SSA * SSA * MLIRType * MLIROp list * MLIROp list  // result, cond, type, thenOps, elseOps
    | While of SSA * MLIRType * MLIROp list * MLIROp list     // result, type, beforeOps, afterOps
    | For of SSA * SSA * SSA * SSA * MLIRType * MLIROp list   // result, lb, ub, step, type, bodyOps
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
    | FuncDecl of string * MLIRType list * MLIRType * MLIROp list  // name, paramTypes, retType, bodyOps
    | ExternDecl of string * MLIRType list * MLIRType              // name, paramTypes, retType
    | Return of SSA option * MLIRType option                       // value, type
