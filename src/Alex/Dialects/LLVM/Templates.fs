/// LLVM Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → LLVMOp
/// NO sprintf. NO string formatting. Just data construction.
module Alex.Dialects.LLVM.Templates

open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Allocate memory on stack: llvm.alloca
let alloca (result: SSA) (count: SSA) (elementTy: MLIRType) (alignment: int option) : LLVMOp =
    LLVMOp.Alloca (result, count, elementTy, alignment)

/// Load value from pointer: llvm.load
let load (result: SSA) (ptr: SSA) (ty: MLIRType) (ordering: AtomicOrdering) : LLVMOp =
    LLVMOp.Load (result, ptr, ty, ordering)

/// Load value from pointer (non-atomic): llvm.load
let loadNonAtomic (result: SSA) (ptr: SSA) (ty: MLIRType) : LLVMOp =
    LLVMOp.Load (result, ptr, ty, NotAtomic)

/// Store value to pointer: llvm.store
let store (value: SSA) (ptr: SSA) (valueTy: MLIRType) (ordering: AtomicOrdering) : LLVMOp =
    LLVMOp.Store (value, ptr, valueTy, ordering)

/// Store value to pointer (non-atomic): llvm.store
let storeNonAtomic (value: SSA) (ptr: SSA) (valueTy: MLIRType) : LLVMOp =
    LLVMOp.Store (value, ptr, valueTy, NotAtomic)

/// Get element pointer: llvm.getelementptr
let gep (result: SSA) (base': SSA) (indices: (SSA * MLIRType) list) (elemTy: MLIRType) : LLVMOp =
    LLVMOp.GEP (result, base', indices, elemTy)

/// Simplified GEP with single index
let gepSingle (result: SSA) (base': SSA) (index: SSA) (indexTy: MLIRType) (elemTy: MLIRType) : LLVMOp =
    LLVMOp.GEP (result, base', [(index, indexTy)], elemTy)

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Get address of global: llvm.mlir.addressof
let addressOf (result: SSA) (global': GlobalRef) : LLVMOp =
    LLVMOp.AddressOf (result, global')

/// Address of function
let addressOfFunc (result: SSA) (funcName: string) : LLVMOp =
    LLVMOp.AddressOf (result, GFunc funcName)

/// Address of string constant
let addressOfString (result: SSA) (hash: uint32) : LLVMOp =
    LLVMOp.AddressOf (result, GString hash)

/// Define a global variable
let globalDef (name: string) (value: string) (ty: MLIRType) (isConstant: bool) : LLVMOp =
    LLVMOp.GlobalDef (name, value, ty, isConstant)

/// Define a global string constant
let globalString (name: string) (content: string) (length: int) : LLVMOp =
    LLVMOp.GlobalString (name, content, length)

// ═══════════════════════════════════════════════════════════════════════════
// CALL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Direct function call: llvm.call
let call (result: SSA option) (func: GlobalRef) (args: Val list) (retTy: MLIRType) : LLVMOp =
    LLVMOp.Call (result, func, args, retTy)

/// Call function by name
let callFunc (result: SSA option) (funcName: string) (args: Val list) (retTy: MLIRType) : LLVMOp =
    LLVMOp.Call (result, GFunc funcName, args, retTy)

/// Indirect function call through pointer: llvm.call with pointer
let indirectCall (result: SSA option) (funcPtr: SSA) (args: Val list) (retTy: MLIRType) : LLVMOp =
    LLVMOp.IndirectCall (result, funcPtr, args, retTy)

// ═══════════════════════════════════════════════════════════════════════════
// STRUCT OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Extract field from struct: llvm.extractvalue
let extractValue (result: SSA) (aggregate: SSA) (indices: int list) (aggregateTy: MLIRType) : LLVMOp =
    LLVMOp.ExtractValue (result, aggregate, indices, aggregateTy)

/// Extract field from struct at single index
let extractValueAt (result: SSA) (aggregate: SSA) (index: int) (aggregateTy: MLIRType) : LLVMOp =
    LLVMOp.ExtractValue (result, aggregate, [index], aggregateTy)

/// Insert field into struct: llvm.insertvalue
let insertValue (result: SSA) (aggregate: SSA) (value: SSA) (indices: int list) (aggregateTy: MLIRType) : LLVMOp =
    LLVMOp.InsertValue (result, aggregate, value, indices, aggregateTy)

/// Insert field into struct at single index
let insertValueAt (result: SSA) (aggregate: SSA) (value: SSA) (index: int) (aggregateTy: MLIRType) : LLVMOp =
    LLVMOp.InsertValue (result, aggregate, value, [index], aggregateTy)

/// Create undefined value: llvm.mlir.undef
let undef (result: SSA) (ty: MLIRType) : LLVMOp =
    LLVMOp.Undef (result, ty)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Bitcast: llvm.bitcast
let bitcast (result: SSA) (operand: SSA) (fromTy: MLIRType) (toTy: MLIRType) : LLVMOp =
    LLVMOp.Bitcast (result, operand, fromTy, toTy)

/// Integer to pointer: llvm.inttoptr
let intToPtr (result: SSA) (operand: SSA) (fromTy: MLIRType) : LLVMOp =
    LLVMOp.IntToPtr (result, operand, fromTy)

/// Pointer to integer: llvm.ptrtoint
let ptrToInt (result: SSA) (operand: SSA) (toTy: MLIRType) : LLVMOp =
    LLVMOp.PtrToInt (result, operand, toTy)

// ═══════════════════════════════════════════════════════════════════════════
// INLINE ASSEMBLY
// ═══════════════════════════════════════════════════════════════════════════

/// Inline assembly: llvm.inline_asm
/// args is a list of (SSA, MLIRType) pairs for proper type signature emission
let inlineAsm (result: SSA option) (asm: string) (constraints: string) (args: (SSA * MLIRType) list) (retTy: MLIRType option) (hasSideEffects: bool) (isAlignStack: bool) : LLVMOp =
    LLVMOp.InlineAsm (result, asm, constraints, args, retTy, hasSideEffects, isAlignStack)

/// Inline assembly with side effects (common case for syscalls)
let inlineAsmWithSideEffects (result: SSA option) (asm: string) (constraints: string) (args: (SSA * MLIRType) list) (retTy: MLIRType option) : LLVMOp =
    LLVMOp.InlineAsm (result, asm, constraints, args, retTy, true, false)

// ═══════════════════════════════════════════════════════════════════════════
// TERMINATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Return from function: llvm.return
let ret (value: SSA option) (valueTy: MLIRType option) : LLVMOp =
    LLVMOp.Return (value, valueTy)

/// Return void
let retVoid : LLVMOp =
    LLVMOp.Return (None, None)

/// Return value with type
let retVal (value: SSA) (valueTy: MLIRType) : LLVMOp =
    LLVMOp.Return (Some value, Some valueTy)

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITE PATTERNS (common combinations)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a fat pointer (native string) from ptr and len
/// Returns operations to insert ptr at [0] and len at [1]
let buildFatPtr (resultSSA: SSA) (ptrSSA: SSA) (lenSSA: SSA) : LLVMOp list =
    let strTy = MLIRTypes.nativeStr
    [
        undef (V (-1)) strTy  // Placeholder - actual SSA assigned by caller
        insertValueAt (V (-2)) (V (-1)) ptrSSA 0 strTy
        insertValueAt resultSSA (V (-2)) lenSSA 1 strTy
    ]

/// Extract ptr and len from fat pointer (native string)
let extractFatPtr (aggregate: SSA) (ptrResult: SSA) (lenResult: SSA) : LLVMOp list =
    let strTy = MLIRTypes.nativeStr
    [
        extractValueAt ptrResult aggregate 0 strTy
        extractValueAt lenResult aggregate 1 strTy
    ]

// ═══════════════════════════════════════════════════════════════════════════
// SYSCALL PATTERNS (Linux x86_64)
// ═══════════════════════════════════════════════════════════════════════════

/// Build syscall with up to 6 arguments (Linux x86_64)
/// Returns inline_asm operation
/// args is a list of (SSA, MLIRType) pairs - typically (ssa, i64) for syscalls
let syscall (result: SSA) (sysNum: int) (args: (SSA * MLIRType) list) : LLVMOp =
    // Linux x86_64 syscall convention:
    // rax = syscall number
    // rdi, rsi, rdx, r10, r8, r9 = args[0..5]
    // Return in rax
    let constraints =
        match List.length args with
        | 0 -> "=r,{rax}"
        | 1 -> "=r,{rax},{rdi}"
        | 2 -> "=r,{rax},{rdi},{rsi}"
        | 3 -> "=r,{rax},{rdi},{rsi},{rdx}"
        | 4 -> "=r,{rax},{rdi},{rsi},{rdx},{r10}"
        | 5 -> "=r,{rax},{rdi},{rsi},{rdx},{r10},{r8}"
        | 6 -> "=r,{rax},{rdi},{rsi},{rdx},{r10},{r8},{r9}"
        | _ -> failwith "syscall: too many arguments (max 6)"

    // Create syscall number as constant first (handled by caller)
    // The args list should already have the syscall number prepended
    LLVMOp.InlineAsm (Some result, "syscall", constraints, args, Some MLIRTypes.i64, true, false)

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap LLVMOp in MLIROp
let wrap (op: LLVMOp) : MLIROp = MLIROp.LLVMOp op
