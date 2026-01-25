/// Core MLIR Types - Structured representations for type-safe MLIR generation
///
/// ARCHITECTURAL PRINCIPLE: ZERO SPRINTF
/// All MLIR representation is through structured types.
/// Serialization to text happens ONLY in Serialize.fs via StringBuilder.
///
/// EXHAUSTIVE COVERAGE: Every operation from upstream MLIR dialects.
/// Source: /usr/include/mlir/Dialect/*/IR/*.td (LLVM 18)
module Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA VALUES
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

// ═══════════════════════════════════════════════════════════════════════════
// MLIR TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Integer bit widths (standard MLIR integer types)
type IntBitWidth = I1 | I8 | I16 | I32 | I64

/// Floating point precision
type FloatBitWidth = F16 | BF16 | F32 | F64

/// Vector shape for SIMD operations
type VectorShape = { Dims: int list; Scalable: bool }

/// MLIR type - structured representation
/// Serialization happens in Serialize module, NOT here
type MLIRType =
    // Primitive types
    | TInt of IntBitWidth
    | TFloat of FloatBitWidth
    | TIndex
    | TPtr                                          // !llvm.ptr (opaque pointer)

    // Composite types
    | TStruct of fields: MLIRType list              // !llvm.struct<(...)>
    | TArray of count: int * element: MLIRType      // !llvm.array<N x T>
    | TFunc of args: MLIRType list * ret: MLIRType  // (T...) -> T
    | TVector of shape: VectorShape * element: MLIRType  // vector<NxT>

    // MemRef type (for memref dialect)
    | TMemRef of shape: int list * element: MLIRType

    // Unit type (maps to i32 returning 0 for ABI compatibility)
    | TUnit

    // Error type - MUST propagate, NEVER silently ignore
    | TError of message: string

/// Common type aliases for convenience
module MLIRTypes =
    let i1 = TInt I1
    let bool = TInt I1  // Semantic alias for boolean
    let i8 = TInt I8
    let i16 = TInt I16
    let i32 = TInt I32
    let i64 = TInt I64
    let f16 = TFloat F16
    let bf16 = TFloat BF16
    let f32 = TFloat F32
    let f64 = TFloat F64
    let ptr = TPtr
    let index = TIndex
    let unit = TUnit

    // NOTE: Native string type is NOT defined here - it's platform-dependent.
    // Use mapType Types.stringType ctx through TransferTypes for the canonical path.

    /// Check if type is an integer type
    let isInteger = function TInt _ -> true | _ -> false

    /// Check if type is a float type
    let isFloat = function TFloat _ -> true | _ -> false

    /// Check if type is a vector type
    let isVector = function TVector _ -> true | _ -> false

    /// Get bit width of integer type
    let intBitWidth = function
        | TInt I1 -> 1
        | TInt I8 -> 8
        | TInt I16 -> 16
        | TInt I32 -> 32
        | TInt I64 -> 64
        | _ -> 0

    /// Create fixed-size vector type
    let vector n elem = TVector ({ Dims = [n]; Scalable = false }, elem)

// ═══════════════════════════════════════════════════════════════════════════
// TYPED VALUES
// ═══════════════════════════════════════════════════════════════════════════

/// A typed SSA value (value + its type)
type Val = {
    SSA: SSA
    Type: MLIRType
}

/// Create a typed value
let val' ssa ty = { SSA = ssa; Type = ty }

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Result of transferring a PSG node to MLIR
/// - TRValue: Has an SSA result with type
/// - TRVoid: No result (unit operations, statements)
/// - TRError: Compilation error with message
/// - TRBuiltin: Handled specially (platform bindings defer to dispatch)
type TransferResult =
    | TRValue of Val
    | TRVoid
    | TRError of string
    | TRBuiltin of name: string * args: Val list

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
type IndexCmpPred = | IEq | INe | ISlt | ISle | ISgt | ISge | IUlt | IUle | IUgt | IUge

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL REFERENCES
// ═══════════════════════════════════════════════════════════════════════════

/// Global symbol reference
type GlobalRef =
    | GFunc of name: string                 // @funcname
    | GString of hash: uint32               // @str_HASH
    | GBytes of index: int                  // @bytes_N
    | GNamed of name: string                // @customname

// ═══════════════════════════════════════════════════════════════════════════
// ATTRIBUTE VALUES
// ═══════════════════════════════════════════════════════════════════════════

/// Attribute values for MLIR operations
/// Used in OpEnvelope for the universal operation representation
type AttrValue =
    | IntAttr of value: int64 * ty: MLIRType       // Integer with type (i32, i64, etc.)
    | FloatAttr of value: float * ty: MLIRType     // Float with type (f32, f64)
    | StringAttr of value: string                   // String attribute
    | BoolAttr of value: bool                       // Boolean attribute
    | TypeAttr of ty: MLIRType                      // Type attribute
    | ArrayAttr of values: AttrValue list           // Array of attributes
    | DictAttr of values: Map<string, AttrValue>   // Dictionary of attributes
    | UnitAttr                                      // Flag attribute (no value)

// ═══════════════════════════════════════════════════════════════════════════
// REGIONS AND BLOCKS
// ═══════════════════════════════════════════════════════════════════════════

/// A block argument is a typed SSA value
type BlockArg = Val

/// Region kind discriminator for SCF region collection
type RegionKind =
    | GuardRegion
    | BodyRegion
    | ThenRegion
    | ElseRegion
    | StartExprRegion
    | EndExprRegion
    | LambdaBodyRegion
    | MatchCaseRegion of idx: int

/// Hooks for SCF region emission (before/after region ops)
/// Generic over the zipper/state type 'Z
type SCFRegionHook<'Z> = {
    BeforeRegion: 'Z -> FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NodeId -> RegionKind -> 'Z
    AfterRegion: 'Z -> FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NodeId -> RegionKind -> 'Z
}

// ═══════════════════════════════════════════════════════════════════════════
// OPENVELOPE - THE SUPER-STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

/// OpEnvelope: The canonical super-structure for ANY MLIR operation.
///
/// This is NOT a "fallback" or "generic" form - it IS the fundamental
/// representation that all MLIR operations embody. Dialect-specific types
/// (ArithOp, LLVMOp, etc.) are type-safe construction helpers that produce
/// operations fitting this universal envelope.
///
/// Serializes to MLIR generic assembly form:
///   %r = "dialect.op"(%a, %b) <{attr = val}> ({ regions }) : (T1, T2) -> (R)
///
type OpEnvelope = {
    Dialect: string                         // "llvm", "arith", "scf", "func"
    Operation: string                       // "intr.memcpy", "addi", "if"
    Operands: Val list                      // Typed input operands
    Results: Val list                       // Typed results (empty for void)
    Attributes: Map<string, AttrValue>      // Named attributes
    Regions: Region list                    // Nested regions (for structured ops)
}

/// MLIROp: The operation type used throughout Alex.
/// Includes both dialect-specific constructors AND the universal OpEnvelope.
and MLIROp =
    // === UNIVERSAL SUPER-STRUCTURE ===
    | Envelope of OpEnvelope

    // === DIALECT-SPECIFIC CONSTRUCTORS ===
    // These provide type-safe, ergonomic construction for common operations
    | ArithOp of ArithOp
    | LLVMOp of LLVMOp
    | SCFOp of SCFOp
    | CFOp of CFOp              // Control Flow dialect (unstructured)
    | FuncOp of FuncOp
    | IndexOp of IndexOp
    | VectorOp of VectorOp      // Vector/SIMD operations

/// A basic block within a region
and Block = {
    Label: BlockRef
    Args: BlockArg list
    Ops: MLIROp list
}

/// A region containing blocks
and Region = {
    Blocks: Block list
}

// ═══════════════════════════════════════════════════════════════════════════
// ARITH DIALECT - EXHAUSTIVE (from ArithOps.td)
// 50 operations: arithmetic, comparison, conversion, constant
// ═══════════════════════════════════════════════════════════════════════════

and ArithOp =
    // === INTEGER BINARY OPERATIONS ===
    | AddI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | SubI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MulI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | DivSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | DivUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | CeilDivSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | CeilDivUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | FloorDivSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | RemSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | RemUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType

    // === INTEGER EXTENDED OPERATIONS ===
    | AddUIExtended of result: SSA * overflow: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MulSIExtended of resultLow: SSA * resultHigh: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MulUIExtended of resultLow: SSA * resultHigh: SSA * lhs: SSA * rhs: SSA * ty: MLIRType

    // === INTEGER BITWISE OPERATIONS ===
    | AndI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | OrI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | XOrI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | ShLI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | ShRSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | ShRUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType

    // === INTEGER MIN/MAX ===
    | MaxSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MaxUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MinSI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MinUI of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType

    // === FLOAT BINARY OPERATIONS ===
    | AddF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | SubF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | MulF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | DivF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | RemF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType
    | NegF of result: SSA * operand: SSA * ty: MLIRType

    // === FLOAT MIN/MAX ===
    | MaximumF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType  // IEEE 754 maximum
    | MaxNumF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType   // IEEE 754 maxNum
    | MinimumF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType  // IEEE 754 minimum
    | MinNumF of result: SSA * lhs: SSA * rhs: SSA * ty: MLIRType   // IEEE 754 minNum

    // === COMPARISON OPERATIONS ===
    | CmpI of result: SSA * pred: ICmpPred * lhs: SSA * rhs: SSA * ty: MLIRType
    | CmpF of result: SSA * pred: FCmpPred * lhs: SSA * rhs: SSA * ty: MLIRType

    // === CONSTANTS ===
    | ConstI of result: SSA * value: int64 * ty: MLIRType
    | ConstF of result: SSA * value: float * ty: MLIRType

    // === INTEGER CONVERSIONS ===
    | ExtSI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | ExtUI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | TruncI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType

    // === FLOAT CONVERSIONS ===
    | SIToFP of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | UIToFP of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPToSI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPToUI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | ExtF of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | TruncF of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType

    // === SCALING FLOAT CONVERSIONS ===
    | ScalingExtF of result: SSA * operand: SSA * scale: SSA * fromTy: MLIRType * toTy: MLIRType
    | ScalingTruncF of result: SSA * operand: SSA * scale: SSA * fromTy: MLIRType * toTy: MLIRType

    // === INDEX CONVERSIONS ===
    | IndexCast of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | IndexCastUI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType

    // === BITCAST ===
    | Bitcast of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType

    // === SELECT ===
    | Select of result: SSA * cond: SSA * trueVal: SSA * falseVal: SSA * ty: MLIRType

// ═══════════════════════════════════════════════════════════════════════════
// LLVM DIALECT - EXHAUSTIVE (from LLVMOps.td)
// Memory, globals, calls, conversions, control flow, atomics, vectors
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic ordering for memory operations
and AtomicOrdering =
    | NotAtomic | Unordered | Monotonic | Acquire | Release | AcqRel | SeqCst

/// Atomic RMW operation kind
and AtomicRMWKind =
    | Xchg | Add | Sub | And | Nand | Or | Xor | Max | Min | UMax | UMin
    | FAdd | FSub | FMax | FMin | UIncWrap | UDecWrap

and LLVMOp =
    // === MEMORY OPERATIONS ===
    | Alloca of result: SSA * count: SSA * elementTy: MLIRType * alignment: int option
    | Load of result: SSA * ptr: SSA * ty: MLIRType * ordering: AtomicOrdering
    | Store of value: SSA * ptr: SSA * valueTy: MLIRType * ordering: AtomicOrdering
    | GEP of result: SSA * base': SSA * indices: (SSA * MLIRType) list * elemTy: MLIRType
    /// GEP for struct field access with constant indices: getelementptr inbounds %ptr[0, fieldIndex] : ...
    | StructGEP of result: SSA * base': SSA * fieldIndex: int * structTy: MLIRType
    | MemCpy of dst: SSA * src: SSA * len: SSA * isVolatile: bool
    | MemMove of dst: SSA * src: SSA * len: SSA * isVolatile: bool
    | MemSet of dst: SSA * value: SSA * len: SSA * isVolatile: bool

    // === GLOBAL OPERATIONS ===
    | AddressOf of result: SSA * global': GlobalRef
    | GlobalDef of name: string * value: string * ty: MLIRType * isConstant: bool
    | GlobalString of name: string * content: string * length: int
    | NullPtr of result: SSA

    // === CALL OPERATIONS ===
    | Call of result: SSA option * func: GlobalRef * args: Val list * retTy: MLIRType
    | IndirectCall of result: SSA option * funcPtr: SSA * args: Val list * retTy: MLIRType
    | Invoke of result: SSA option * func: GlobalRef * args: Val list * retTy: MLIRType * normalDest: BlockRef * unwindDest: BlockRef
    | Landingpad of result: SSA * ty: MLIRType * isCleanup: bool

    // === STRUCT/AGGREGATE OPERATIONS ===
    | ExtractValue of result: SSA * aggregate: SSA * indices: int list * aggregateTy: MLIRType
    | InsertValue of result: SSA * aggregate: SSA * value: SSA * indices: int list * aggregateTy: MLIRType
    | Undef of result: SSA * ty: MLIRType
    | Poison of result: SSA * ty: MLIRType
    | ZeroInit of result: SSA * ty: MLIRType

    // === VECTOR OPERATIONS ===
    | ExtractElement of result: SSA * vector: SSA * index: SSA
    | InsertElement of result: SSA * vector: SSA * value: SSA * index: SSA
    | ShuffleVector of result: SSA * v1: SSA * v2: SSA * mask: int list

    // === CONVERSION OPERATIONS ===
    | Bitcast of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | Bswap of result: SSA * operand: SSA * ty: MLIRType  // llvm.intr.bswap - byte swap
    | IntToPtr of result: SSA * operand: SSA * fromTy: MLIRType
    | PtrToInt of result: SSA * operand: SSA * toTy: MLIRType
    | AddrSpaceCast of result: SSA * operand: SSA * fromAS: int * toAS: int
    | ZExt of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | SExt of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | Trunc of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPExt of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPTrunc of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPToSI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | FPToUI of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | SIToFP of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType
    | UIToFP of result: SSA * operand: SSA * fromTy: MLIRType * toTy: MLIRType

    // === INTEGER ARITHMETIC (LLVM dialect versions with flags) ===
    | LLVMAdd of result: SSA * lhs: SSA * rhs: SSA * nsw: bool * nuw: bool
    | LLVMSub of result: SSA * lhs: SSA * rhs: SSA * nsw: bool * nuw: bool
    | LLVMMul of result: SSA * lhs: SSA * rhs: SSA * nsw: bool * nuw: bool
    | LLVMUDiv of result: SSA * lhs: SSA * rhs: SSA * exact: bool
    | LLVMSDiv of result: SSA * lhs: SSA * rhs: SSA * exact: bool
    | LLVMURem of result: SSA * lhs: SSA * rhs: SSA
    | LLVMSRem of result: SSA * lhs: SSA * rhs: SSA
    | LLVMAnd of result: SSA * lhs: SSA * rhs: SSA
    | LLVMOr of result: SSA * lhs: SSA * rhs: SSA * disjoint: bool
    | LLVMXor of result: SSA * lhs: SSA * rhs: SSA
    | LLVMShl of result: SSA * lhs: SSA * rhs: SSA * nsw: bool * nuw: bool
    | LLVMLShr of result: SSA * lhs: SSA * rhs: SSA * exact: bool
    | LLVMAShr of result: SSA * lhs: SSA * rhs: SSA * exact: bool

    // === FLOAT ARITHMETIC ===
    | LLVMFAdd of result: SSA * lhs: SSA * rhs: SSA
    | LLVMFSub of result: SSA * lhs: SSA * rhs: SSA
    | LLVMFMul of result: SSA * lhs: SSA * rhs: SSA
    | LLVMFDiv of result: SSA * lhs: SSA * rhs: SSA
    | LLVMFRem of result: SSA * lhs: SSA * rhs: SSA
    | LLVMFNeg of result: SSA * operand: SSA

    // === COMPARISON ===
    | ICmp of result: SSA * pred: ICmpPred * lhs: SSA * rhs: SSA
    | FCmp of result: SSA * pred: FCmpPred * lhs: SSA * rhs: SSA
    | LLVMSelect of result: SSA * cond: SSA * trueVal: SSA * falseVal: SSA

    // === ATOMIC OPERATIONS ===
    | AtomicRMW of result: SSA * op: AtomicRMWKind * ptr: SSA * value: SSA * ordering: AtomicOrdering
    | AtomicCmpXchg of resultVal: SSA * resultSuccess: SSA * ptr: SSA * expected: SSA * desired: SSA * successOrdering: AtomicOrdering * failureOrdering: AtomicOrdering
    | Fence of ordering: AtomicOrdering * syncScope: string option

    // === CONTROL FLOW (LLVM terminators) ===
    | LLVMBr of dest: BlockRef
    | LLVMCondBr of cond: SSA * trueDest: BlockRef * falseDest: BlockRef
    | LLVMSwitch of value: SSA * defaultDest: BlockRef * cases: (int64 * BlockRef) list
    | Unreachable
    | Resume of value: SSA

    // === PHI NODE ===
    | Phi of result: SSA * ty: MLIRType * incomings: (SSA * BlockRef) list

    // === INLINE ASM ===
    | InlineAsm of result: SSA option * asm: string * constraints: string * args: (SSA * MLIRType) list * retTy: MLIRType option * hasSideEffects: bool * isAlignStack: bool

    // === VA ARG ===
    | VaArg of result: SSA * vaList: SSA * ty: MLIRType

    // === RETURN ===
    | Return of value: SSA option * valueTy: MLIRType option

    // === FUNCTION DEFINITION (llvm.func) ===
    // Used for functions whose address is taken (e.g., closures)
    // llvm.func @name(%arg0: type, ...) -> retType { ... }
    | LLVMFuncDef of name: string * args: (SSA * MLIRType) list * retTy: MLIRType * body: Region * linkage: LLVMLinkage

    // === FUNCTION DECLARATION (llvm.func without body) ===
    // Forward declaration for llvm.call to reference before definition
    // llvm.func @name(type, ...) -> retType
    | LLVMFuncDecl of name: string * argTypes: MLIRType list * retTy: MLIRType * linkage: LLVMLinkage

and LLVMLinkage =
    | LLVMPrivate
    | LLVMInternal
    | LLVMExternal

// ═══════════════════════════════════════════════════════════════════════════
// SCF DIALECT - EXHAUSTIVE (from SCFOps.td)
// Structured control flow: if, while, for, parallel, forall, reduce
// ═══════════════════════════════════════════════════════════════════════════

and SCFOp =
    // === CONDITIONALS ===
    | If of results: SSA list * cond: SSA * thenRegion: Region * elseRegion: Region option * resultTypes: MLIRType list

    // === LOOPS ===
    | While of results: SSA list * condRegion: Region * bodyRegion: Region * iterArgs: Val list
    | For of results: SSA list * iv: SSA * start: SSA * stop: SSA * step: SSA * bodyRegion: Region * iterArgs: Val list
    | Forall of results: SSA list * lowerBounds: SSA list * upperBounds: SSA list * steps: SSA list * outputs: Val list * bodyRegion: Region
    | Parallel of results: SSA list * lowerBounds: SSA list * upperBounds: SSA list * steps: SSA list * initVals: Val list * bodyRegion: Region

    // === SWITCH ===
    | IndexSwitch of results: SSA list * arg: SSA * cases: int64 list * caseRegions: Region list * defaultRegion: Region * resultTypes: MLIRType list

    // === EXECUTE REGION ===
    | ExecuteRegion of results: SSA list * bodyRegion: Region * resultTypes: MLIRType list

    // === REGION TERMINATORS ===
    | Yield of values: Val list
    | Condition of cond: SSA * args: Val list
    | Reduce of operands: Val list * reductionRegions: Region list
    | ReduceReturn of result: SSA
    | InParallel of bodyRegion: Region

// ═══════════════════════════════════════════════════════════════════════════
// CF DIALECT - EXHAUSTIVE (from ControlFlowOps.td)
// Unstructured control flow: branch, cond_br, switch, assert
// ═══════════════════════════════════════════════════════════════════════════

and CFOp =
    // === ASSERTIONS ===
    | Assert of cond: SSA * message: string

    // === UNCONDITIONAL BRANCH ===
    | Br of dest: BlockRef * destOperands: Val list

    // === CONDITIONAL BRANCH ===
    | CondBr of cond: SSA * trueDest: BlockRef * trueDestOperands: Val list * falseDest: BlockRef * falseDestOperands: Val list * branchWeights: (int * int) option

    // === SWITCH ===
    | Switch of flag: SSA * flagTy: MLIRType * defaultDest: BlockRef * defaultOperands: Val list * cases: (int64 * BlockRef * Val list) list

// ═══════════════════════════════════════════════════════════════════════════
// FUNC DIALECT - EXHAUSTIVE (from FuncOps.td)
// Function definition, declaration, calls, return
// ═══════════════════════════════════════════════════════════════════════════

and FuncVisibility = Public | Private | Nested

and FuncOp =
    // === FUNCTION DEFINITION ===
    | FuncDef of name: string * args: (SSA * MLIRType) list * retTy: MLIRType * body: Region * visibility: FuncVisibility

    // === FUNCTION DECLARATION (external) ===
    | FuncDecl of name: string * argTypes: MLIRType list * retTy: MLIRType * visibility: FuncVisibility

    // === DIRECT CALL ===
    | FuncCall of result: SSA option * func: string * args: Val list * retTy: MLIRType

    // === INDIRECT CALL ===
    | FuncCallIndirect of result: SSA option * callee: SSA * args: Val list * retTy: MLIRType

    // === FUNCTION CONSTANT (pointer to function) ===
    | FuncConstant of result: SSA * funcName: string * funcTy: MLIRType

    // === RETURN ===
    // func.return %val : type  (requires type information for MLIR syntax)
    | FuncReturn of values: (SSA * MLIRType) list

// ═══════════════════════════════════════════════════════════════════════════
// INDEX DIALECT - EXHAUSTIVE (from IndexOps.td)
// Operations on index type: arithmetic, comparisons, casts
// ═══════════════════════════════════════════════════════════════════════════

and IndexOp =
    // === CONSTANTS ===
    | IndexConst of result: SSA * value: int64
    | IndexBoolConst of result: SSA * value: bool

    // === ARITHMETIC ===
    | IndexAdd of result: SSA * lhs: SSA * rhs: SSA
    | IndexSub of result: SSA * lhs: SSA * rhs: SSA
    | IndexMul of result: SSA * lhs: SSA * rhs: SSA
    | IndexDivS of result: SSA * lhs: SSA * rhs: SSA
    | IndexDivU of result: SSA * lhs: SSA * rhs: SSA
    | IndexCeilDivS of result: SSA * lhs: SSA * rhs: SSA
    | IndexCeilDivU of result: SSA * lhs: SSA * rhs: SSA
    | IndexFloorDivS of result: SSA * lhs: SSA * rhs: SSA
    | IndexRemS of result: SSA * lhs: SSA * rhs: SSA
    | IndexRemU of result: SSA * lhs: SSA * rhs: SSA

    // === MIN/MAX ===
    | IndexMaxS of result: SSA * lhs: SSA * rhs: SSA
    | IndexMaxU of result: SSA * lhs: SSA * rhs: SSA
    | IndexMinS of result: SSA * lhs: SSA * rhs: SSA
    | IndexMinU of result: SSA * lhs: SSA * rhs: SSA

    // === BITWISE ===
    | IndexShl of result: SSA * lhs: SSA * rhs: SSA
    | IndexShrS of result: SSA * lhs: SSA * rhs: SSA
    | IndexShrU of result: SSA * lhs: SSA * rhs: SSA
    | IndexAnd of result: SSA * lhs: SSA * rhs: SSA
    | IndexOr of result: SSA * lhs: SSA * rhs: SSA
    | IndexXor of result: SSA * lhs: SSA * rhs: SSA

    // === COMPARISON ===
    | IndexCmp of result: SSA * pred: IndexCmpPred * lhs: SSA * rhs: SSA

    // === CASTS ===
    | IndexCastS of result: SSA * operand: SSA * toTy: MLIRType
    | IndexCastU of result: SSA * operand: SSA * toTy: MLIRType

    // === SIZE OF ===
    | IndexSizeOf of result: SSA * ty: MLIRType

// ═══════════════════════════════════════════════════════════════════════════
// VECTOR DIALECT - CORE OPERATIONS (from VectorOps.td)
// SIMD operations for vectorized code generation
// ═══════════════════════════════════════════════════════════════════════════

and VectorOp =
    // === BROADCAST ===
    | Broadcast of result: SSA * source: SSA * resultTy: MLIRType

    // === EXTRACT/INSERT ===
    | Extract of result: SSA * vector: SSA * position: int list
    | Insert of result: SSA * source: SSA * dest: SSA * position: int list
    | ExtractStrided of result: SSA * vector: SSA * offsets: int list * sizes: int list * strides: int list
    | InsertStrided of result: SSA * source: SSA * dest: SSA * offsets: int list * strides: int list

    // === SHAPE OPERATIONS ===
    | ShapeCast of result: SSA * source: SSA * resultTy: MLIRType
    | Transpose of result: SSA * vector: SSA * transp: int list
    | FlattenTranspose of result: SSA * vector: SSA

    // === REDUCTION ===
    | ReductionAdd of result: SSA * vector: SSA * acc: SSA option
    | ReductionMul of result: SSA * vector: SSA * acc: SSA option
    | ReductionAnd of result: SSA * vector: SSA
    | ReductionOr of result: SSA * vector: SSA
    | ReductionXor of result: SSA * vector: SSA
    | ReductionMinSI of result: SSA * vector: SSA
    | ReductionMinUI of result: SSA * vector: SSA
    | ReductionMaxSI of result: SSA * vector: SSA
    | ReductionMaxUI of result: SSA * vector: SSA
    | ReductionMinF of result: SSA * vector: SSA
    | ReductionMaxF of result: SSA * vector: SSA

    // === FMA ===
    | FMA of result: SSA * lhs: SSA * rhs: SSA * acc: SSA

    // === SPLAT ===
    | Splat of result: SSA * value: SSA * resultTy: MLIRType

    // === LOAD/STORE (prefixed to avoid collision with LLVMOp) ===
    | VectorLoad of result: SSA * basePtr: SSA * indices: SSA list
    | VectorStore of valueToStore: SSA * basePtr: SSA * indices: SSA list
    | MaskedLoad of result: SSA * basePtr: SSA * indices: SSA list * mask: SSA * passthru: SSA
    | MaskedStore of valueToStore: SSA * basePtr: SSA * indices: SSA list * mask: SSA
    | Gather of result: SSA * basePtr: SSA * indices: SSA * indexVec: SSA * mask: SSA * passthru: SSA
    | Scatter of valueToStore: SSA * basePtr: SSA * indices: SSA * indexVec: SSA * mask: SSA

    // === MASK OPERATIONS ===
    | CreateMask of result: SSA * operands: SSA list
    | ConstantMask of result: SSA * maskDimSizes: int list

    // === PRINT (debugging) ===
    | Print of source: SSA * punctuation: string option

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get the result SSA(s) from an operation
let opResults (op: MLIROp) : SSA list =
    match op with
    // === UNIVERSAL SUPER-STRUCTURE ===
    | Envelope env -> env.Results |> List.map (fun v -> v.SSA)

    // === DIALECT-SPECIFIC ===
    | ArithOp a ->
        match a with
        | AddI (r,_,_,_) | SubI (r,_,_,_) | MulI (r,_,_,_)
        | DivSI (r,_,_,_) | DivUI (r,_,_,_) | CeilDivSI (r,_,_,_) | CeilDivUI (r,_,_,_) | FloorDivSI (r,_,_,_)
        | RemSI (r,_,_,_) | RemUI (r,_,_,_)
        | AndI (r,_,_,_) | OrI (r,_,_,_) | XOrI (r,_,_,_)
        | ShLI (r,_,_,_) | ShRSI (r,_,_,_) | ShRUI (r,_,_,_)
        | MaxSI (r,_,_,_) | MaxUI (r,_,_,_) | MinSI (r,_,_,_) | MinUI (r,_,_,_)
        | AddF (r,_,_,_) | SubF (r,_,_,_) | MulF (r,_,_,_) | DivF (r,_,_,_) | RemF (r,_,_,_)
        | NegF (r,_,_)
        | MaximumF (r,_,_,_) | MaxNumF (r,_,_,_) | MinimumF (r,_,_,_) | MinNumF (r,_,_,_)
        | CmpI (r,_,_,_,_) | CmpF (r,_,_,_,_)
        | ConstI (r,_,_) | ConstF (r,_,_)
        | ExtSI (r,_,_,_) | ExtUI (r,_,_,_) | TruncI (r,_,_,_)
        | ArithOp.SIToFP (r,_,_,_) | ArithOp.UIToFP (r,_,_,_) | ArithOp.FPToSI (r,_,_,_) | ArithOp.FPToUI (r,_,_,_)
        | ExtF (r,_,_,_) | TruncF (r,_,_,_) | ScalingExtF (r,_,_,_,_) | ScalingTruncF (r,_,_,_,_)
        | IndexCast (r,_,_,_) | IndexCastUI (r,_,_,_) | ArithOp.Bitcast (r,_,_,_)
        | Select (r,_,_,_,_) -> [r]
        | AddUIExtended (r,o,_,_,_) -> [r; o]
        | MulSIExtended (rl,rh,_,_,_) | MulUIExtended (rl,rh,_,_,_) -> [rl; rh]

    | LLVMOp l ->
        match l with
        | Alloca (r,_,_,_) | LLVMOp.Load (r,_,_,_) | GEP (r,_,_,_) | StructGEP (r,_,_,_) | AddressOf (r,_)
        | ExtractValue (r,_,_,_) | InsertValue (r,_,_,_,_) | Undef (r,_) | Poison (r,_) | ZeroInit (r,_)
        | LLVMOp.Bitcast (r,_,_,_) | Bswap (r,_,_) | IntToPtr (r,_,_) | PtrToInt (r,_,_) | AddrSpaceCast (r,_,_,_)
        | ZExt (r,_,_,_) | SExt (r,_,_,_) | Trunc (r,_,_,_)
        | FPExt (r,_,_,_) | FPTrunc (r,_,_,_)
        | LLVMOp.FPToSI (r,_,_,_) | LLVMOp.FPToUI (r,_,_,_) | LLVMOp.SIToFP (r,_,_,_) | LLVMOp.UIToFP (r,_,_,_)
        | LLVMAdd (r,_,_,_,_) | LLVMSub (r,_,_,_,_) | LLVMMul (r,_,_,_,_)
        | LLVMUDiv (r,_,_,_) | LLVMSDiv (r,_,_,_) | LLVMURem (r,_,_) | LLVMSRem (r,_,_)
        | LLVMAnd (r,_,_) | LLVMOr (r,_,_,_) | LLVMXor (r,_,_)
        | LLVMShl (r,_,_,_,_) | LLVMLShr (r,_,_,_) | LLVMAShr (r,_,_,_)
        | LLVMFAdd (r,_,_) | LLVMFSub (r,_,_) | LLVMFMul (r,_,_) | LLVMFDiv (r,_,_) | LLVMFRem (r,_,_) | LLVMFNeg (r,_)
        | ICmp (r,_,_,_) | FCmp (r,_,_,_) | LLVMSelect (r,_,_,_)
        | AtomicRMW (r,_,_,_,_) | Landingpad (r,_,_)
        | ExtractElement (r,_,_) | InsertElement (r,_,_,_) | ShuffleVector (r,_,_,_)
        | Phi (r,_,_) | VaArg (r,_,_) | NullPtr r -> [r]
        | AtomicCmpXchg (rv,rs,_,_,_,_,_) -> [rv; rs]
        | Call (Some r,_,_,_) | IndirectCall (Some r,_,_,_) | Invoke (Some r,_,_,_,_,_) | InlineAsm (Some r,_,_,_,_,_,_) -> [r]
        | Call (None,_,_,_) | IndirectCall (None,_,_,_) | Invoke (None,_,_,_,_,_) | InlineAsm (None,_,_,_,_,_,_)
        | LLVMOp.Store _ | MemCpy _ | MemMove _ | MemSet _ | Fence _ | GlobalDef _ | GlobalString _
        | LLVMBr _ | LLVMCondBr _ | LLVMSwitch _ | Unreachable | Resume _ | Return _
        | LLVMFuncDef _ | LLVMFuncDecl _ -> []

    | SCFOp s ->
        match s with
        | If (rs,_,_,_,_) | While (rs,_,_,_) | For (rs,_,_,_,_,_,_) | Forall (rs,_,_,_,_,_) | Parallel (rs,_,_,_,_,_)
        | IndexSwitch (rs,_,_,_,_,_) | ExecuteRegion (rs,_,_) -> rs
        | Yield _ | Condition _ | Reduce _ | ReduceReturn _ | InParallel _ -> []

    | CFOp c ->
        match c with
        | Assert _ | Br _ | CondBr _ | Switch _ -> []

    | FuncOp f ->
        match f with
        | FuncCall (Some r,_,_,_) | FuncCallIndirect (Some r,_,_,_) | FuncConstant (r,_,_) -> [r]
        | FuncDef _ | FuncDecl _ | FuncCall (None,_,_,_) | FuncCallIndirect (None,_,_,_) | FuncReturn _ -> []

    | IndexOp i ->
        match i with
        | IndexConst (r,_) | IndexBoolConst (r,_)
        | IndexAdd (r,_,_) | IndexSub (r,_,_) | IndexMul (r,_,_)
        | IndexDivS (r,_,_) | IndexDivU (r,_,_) | IndexCeilDivS (r,_,_) | IndexCeilDivU (r,_,_) | IndexFloorDivS (r,_,_)
        | IndexRemS (r,_,_) | IndexRemU (r,_,_)
        | IndexMaxS (r,_,_) | IndexMaxU (r,_,_) | IndexMinS (r,_,_) | IndexMinU (r,_,_)
        | IndexShl (r,_,_) | IndexShrS (r,_,_) | IndexShrU (r,_,_)
        | IndexAnd (r,_,_) | IndexOr (r,_,_) | IndexXor (r,_,_)
        | IndexCmp (r,_,_,_) | IndexCastS (r,_,_) | IndexCastU (r,_,_) | IndexSizeOf (r,_) -> [r]

    | VectorOp v ->
        match v with
        | Broadcast (r,_,_) | Extract (r,_,_) | Insert (r,_,_,_) | ExtractStrided (r,_,_,_,_) | InsertStrided (r,_,_,_,_)
        | ShapeCast (r,_,_) | Transpose (r,_,_) | FlattenTranspose (r,_)
        | ReductionAdd (r,_,_) | ReductionMul (r,_,_) | ReductionAnd (r,_) | ReductionOr (r,_) | ReductionXor (r,_)
        | ReductionMinSI (r,_) | ReductionMinUI (r,_) | ReductionMaxSI (r,_) | ReductionMaxUI (r,_)
        | ReductionMinF (r,_) | ReductionMaxF (r,_)
        | FMA (r,_,_,_) | Splat (r,_,_)
        | VectorLoad (r,_,_) | MaskedLoad (r,_,_,_,_) | Gather (r,_,_,_,_,_)
        | CreateMask (r,_) | ConstantMask (r,_) -> [r]
        | VectorStore _ | MaskedStore _ | Scatter _ | Print _ -> []
