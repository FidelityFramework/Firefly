/// ConsoleBindings - Platform-specific console I/O bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// Uses dialect templates for all operations. ZERO sprintf.
module Alex.Bindings.Console.ConsoleBindings

open Alex.Dialects.Core.Types
open Alex.Dialects.Arith.Templates
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.SCF.Templates
open Alex.Traversal.PSGZipper
open Alex.Bindings.PlatformTypes
open Alex.Bindings.BindingTypes

// ===================================================================
// Syscall Numbers
// ===================================================================

module SyscallData =
    let linuxWrite = 1L
    let linuxRead = 0L
    let macosWrite = 0x2000004L
    let macosRead = 0x2000003L

    let getWriteSyscall (os: OSFamily) =
        match os with
        | Linux -> linuxWrite
        | MacOS -> macosWrite
        | _ -> linuxWrite

    let getReadSyscall (os: OSFamily) =
        match os with
        | Linux -> linuxRead
        | MacOS -> macosRead
        | _ -> linuxRead

// ===================================================================
// Helper: Build syscall via inline asm
// ===================================================================

/// Helper: wrap SSA as i64 typed arg (syscall convention)
let inline private i64Arg (ssa: SSA) = (ssa, MLIRTypes.i64)

/// Helper: wrap SSA as ptr typed arg
let inline private ptrArg (ssa: SSA) = (ssa, MLIRTypes.ptr)

/// Generate syscall with up to 3 args (fd, buf, count)
let buildSyscall3 (z: PSGZipper) (sysNum: int64) (fd: SSA) (buf: SSA) (count: SSA) : MLIROp list * SSA =
    let sysNumSSA = freshSynthSSA z
    let resultSSA = freshSynthSSA z
    let ops = [
        MLIROp.ArithOp (constI sysNumSSA sysNum MLIRTypes.i64)
        MLIROp.LLVMOp (syscall resultSSA (int sysNum) [i64Arg sysNumSSA; i64Arg fd; ptrArg buf; i64Arg count])
    ]
    ops, resultSSA

// ===================================================================
// Sys.write / writeBytes binding
// ===================================================================

/// writeBytes(fd, buf, count) -> bytes written
let bindWriteBytes (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [fd; buf; count] ->
        let syscallNum = SyscallData.getWriteSyscall os

        // Extend fd to i64 if needed
        let fdExt, fdOps =
            match fd.Type with
            | TInt I64 -> fd.SSA, []
            | TInt _ ->
                let ext = freshSynthSSA z
                ext, [MLIROp.ArithOp (extSI ext fd.SSA fd.Type MLIRTypes.i64)]
            | _ -> fd.SSA, []

        // Extend count to i64 if needed
        let countExt, countOps =
            match count.Type with
            | TInt I64 -> count.SSA, []
            | TInt _ ->
                let ext = freshSynthSSA z
                ext, [MLIROp.ArithOp (extSI ext count.SSA count.Type MLIRTypes.i64)]
            | _ -> count.SSA, []

        // Syscall
        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let syscallOps = [
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdExt; ptrArg buf.SSA; i64Arg countExt] (Some MLIRTypes.i64))
        ]

        // Truncate result to i32
        let truncSSA = freshSynthSSA z
        let truncOp = MLIROp.ArithOp (truncI truncSSA resultSSA MLIRTypes.i64 MLIRTypes.i32)

        let allOps = fdOps @ countOps @ syscallOps @ [truncOp]
        BoundOps (allOps, Some { SSA = truncSSA; Type = MLIRTypes.i32 })
    | _ ->
        NotSupported "writeBytes requires (fd, buf, count) arguments"

/// readBytes(fd, buf, maxCount) -> bytes read
let bindReadBytes (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [fd; buf; maxCount] ->
        let syscallNum = SyscallData.getReadSyscall os

        let fdExt, fdOps =
            match fd.Type with
            | TInt I64 -> fd.SSA, []
            | TInt _ ->
                let ext = freshSynthSSA z
                ext, [MLIROp.ArithOp (extSI ext fd.SSA fd.Type MLIRTypes.i64)]
            | _ -> fd.SSA, []

        let countExt, countOps =
            match maxCount.Type with
            | TInt I64 -> maxCount.SSA, []
            | TInt _ ->
                let ext = freshSynthSSA z
                ext, [MLIROp.ArithOp (extSI ext maxCount.SSA maxCount.Type MLIRTypes.i64)]
            | _ -> maxCount.SSA, []

        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let syscallOps = [
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdExt; ptrArg buf.SSA; i64Arg countExt] (Some MLIRTypes.i64))
        ]

        let truncSSA = freshSynthSSA z
        let truncOp = MLIROp.ArithOp (truncI truncSSA resultSSA MLIRTypes.i64 MLIRTypes.i32)

        let allOps = fdOps @ countOps @ syscallOps @ [truncOp]
        BoundOps (allOps, Some { SSA = truncSSA; Type = MLIRTypes.i32 })
    | _ ->
        NotSupported "readBytes requires (fd, buf, maxCount) arguments"

// ===================================================================
// Console.write / Console.writeln bindings
// ===================================================================

/// Console.write(str) - write string to stdout
let bindConsoleWrite (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] when str.Type = MLIRTypes.nativeStr ->
        // Extract ptr and len from fat pointer
        let ptrSSA = freshSynthSSA z
        let lenSSA = freshSynthSSA z
        let extractOps = [
            MLIROp.LLVMOp (extractValueAt ptrSSA str.SSA 0 MLIRTypes.nativeStr)
            MLIROp.LLVMOp (extractValueAt lenSSA str.SSA 1 MLIRTypes.nativeStr)
        ]

        // fd = 1 (stdout)
        let fdSSA = freshSynthSSA z
        let fdOp = MLIROp.ArithOp (constI fdSSA 1L MLIRTypes.i64)

        // Syscall
        let syscallNum = SyscallData.getWriteSyscall os
        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let syscallOps = [
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdSSA; ptrArg ptrSSA; i64Arg lenSSA] (Some MLIRTypes.i64))
        ]

        let allOps = extractOps @ [fdOp] @ syscallOps
        BoundOps (allOps, None)  // Console.write returns unit
    | _ ->
        NotSupported "Console.write requires (string) argument"

/// Console.writeln(str) - write string + newline to stdout
let bindConsoleWriteln (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] when str.Type = MLIRTypes.nativeStr ->
        // Extract ptr and len
        let ptrSSA = freshSynthSSA z
        let lenSSA = freshSynthSSA z
        let extractOps = [
            MLIROp.LLVMOp (extractValueAt ptrSSA str.SSA 0 MLIRTypes.nativeStr)
            MLIROp.LLVMOp (extractValueAt lenSSA str.SSA 1 MLIRTypes.nativeStr)
        ]

        // fd = 1 and syscall num
        let fdSSA = freshSynthSSA z
        let syscallNum = SyscallData.getWriteSyscall os
        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let writeStrOps = [
            MLIROp.ArithOp (constI fdSSA 1L MLIRTypes.i64)
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdSSA; ptrArg ptrSSA; i64Arg lenSSA] (Some MLIRTypes.i64))
        ]

        // Allocate and write newline
        let oneSSA = freshSynthSSA z
        let nlPtrSSA = freshSynthSSA z
        let nlCharSSA = freshSynthSSA z
        let nlOps = [
            MLIROp.ArithOp (constI oneSSA 1L MLIRTypes.i64)
            MLIROp.LLVMOp (alloca nlPtrSSA oneSSA MLIRTypes.i8 None)
            MLIROp.ArithOp (constI nlCharSSA 10L MLIRTypes.i8)  // '\n'
            MLIROp.LLVMOp (storeNonAtomic nlCharSSA nlPtrSSA MLIRTypes.i8)
        ]

        // Write newline
        let sysNumSSA2 = freshSynthSSA z
        let resultSSA2 = freshSynthSSA z
        let writeNlOps = [
            MLIROp.ArithOp (constI sysNumSSA2 syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA2) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA2; i64Arg fdSSA; ptrArg nlPtrSSA; i64Arg oneSSA] (Some MLIRTypes.i64))
        ]

        let allOps = extractOps @ writeStrOps @ nlOps @ writeNlOps
        BoundOps (allOps, None)
    | _ ->
        NotSupported "Console.writeln requires (string) argument"

/// Console.error(str) - write string to stderr
let bindConsoleError (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] when str.Type = MLIRTypes.nativeStr ->
        let ptrSSA = freshSynthSSA z
        let lenSSA = freshSynthSSA z
        let extractOps = [
            MLIROp.LLVMOp (extractValueAt ptrSSA str.SSA 0 MLIRTypes.nativeStr)
            MLIROp.LLVMOp (extractValueAt lenSSA str.SSA 1 MLIRTypes.nativeStr)
        ]

        // fd = 2 (stderr)
        let fdSSA = freshSynthSSA z
        let syscallNum = SyscallData.getWriteSyscall os
        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let syscallOps = [
            MLIROp.ArithOp (constI fdSSA 2L MLIRTypes.i64)
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdSSA; ptrArg ptrSSA; i64Arg lenSSA] (Some MLIRTypes.i64))
        ]

        let allOps = extractOps @ syscallOps
        BoundOps (allOps, None)
    | _ ->
        NotSupported "Console.error requires (string) argument"

/// Console.errorln(str) - write string + newline to stderr
let bindConsoleErrorln (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    match prim.Args with
    | [str] when str.Type = MLIRTypes.nativeStr ->
        let ptrSSA = freshSynthSSA z
        let lenSSA = freshSynthSSA z
        let extractOps = [
            MLIROp.LLVMOp (extractValueAt ptrSSA str.SSA 0 MLIRTypes.nativeStr)
            MLIROp.LLVMOp (extractValueAt lenSSA str.SSA 1 MLIRTypes.nativeStr)
        ]

        let fdSSA = freshSynthSSA z
        let syscallNum = SyscallData.getWriteSyscall os
        let sysNumSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let writeStrOps = [
            MLIROp.ArithOp (constI fdSSA 2L MLIRTypes.i64)  // stderr
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdSSA; ptrArg ptrSSA; i64Arg lenSSA] (Some MLIRTypes.i64))
        ]

        let oneSSA = freshSynthSSA z
        let nlPtrSSA = freshSynthSSA z
        let nlCharSSA = freshSynthSSA z
        let nlOps = [
            MLIROp.ArithOp (constI oneSSA 1L MLIRTypes.i64)
            MLIROp.LLVMOp (alloca nlPtrSSA oneSSA MLIRTypes.i8 None)
            MLIROp.ArithOp (constI nlCharSSA 10L MLIRTypes.i8)
            MLIROp.LLVMOp (storeNonAtomic nlCharSSA nlPtrSSA MLIRTypes.i8)
        ]

        let sysNumSSA2 = freshSynthSSA z
        let resultSSA2 = freshSynthSSA z
        let writeNlOps = [
            MLIROp.ArithOp (constI sysNumSSA2 syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some resultSSA2) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA2; i64Arg fdSSA; ptrArg nlPtrSSA; i64Arg oneSSA] (Some MLIRTypes.i64))
        ]

        let allOps = extractOps @ writeStrOps @ nlOps @ writeNlOps
        BoundOps (allOps, None)
    | _ ->
        NotSupported "Console.errorln requires (string) argument"

// ===================================================================
// Console.readln binding (complex: uses scf.while)
// ===================================================================

/// Console.readln() - read line from stdin, returns string
let bindConsoleReadln (os: OSFamily) (z: PSGZipper) (prim: PlatformPrimitive) : BindingResult =
    // Accept no args, unit arg, or pseudo-unit
    match prim.Args with
    | [] | [{ Type = TUnit }] | [{ Type = TInt I32 }] ->
        let syscallNum = SyscallData.getReadSyscall os

        // Allocate main buffer (256 bytes) and single-char buffer
        let bufSizeSSA = freshSynthSSA z
        let bufPtrSSA = freshSynthSSA z
        let oneSSA = freshSynthSSA z
        let charBufSSA = freshSynthSSA z
        let allocOps = [
            MLIROp.ArithOp (constI bufSizeSSA 256L MLIRTypes.i64)
            MLIROp.LLVMOp (alloca bufPtrSSA bufSizeSSA MLIRTypes.i8 None)
            MLIROp.ArithOp (constI oneSSA 1L MLIRTypes.i64)
            MLIROp.LLVMOp (alloca charBufSSA oneSSA MLIRTypes.i8 None)
        ]

        // Constants
        let zeroSSA = freshSynthSSA z
        let newlineSSA = freshSynthSSA z
        let fdSSA = freshSynthSSA z
        let maxPosSSA = freshSynthSSA z
        let constOps = [
            MLIROp.ArithOp (constI zeroSSA 0L MLIRTypes.i64)
            MLIROp.ArithOp (constI newlineSSA 10L MLIRTypes.i8)
            MLIROp.ArithOp (constI fdSSA 0L MLIRTypes.i64)  // stdin
            MLIROp.ArithOp (constI maxPosSSA 255L MLIRTypes.i64)
        ]

        // Build scf.while loop
        // iter_arg: position (i64), starting at 0
        // Guard: read byte, check (pos < 255 && bytes > 0 && char != '\n')
        // Body: store char at buffer[pos], yield pos+1

        let posArg = Arg 0
        let posVal = { SSA = posArg; Type = MLIRTypes.i64 }

        // Guard block ops
        let inBoundsSSA = freshSynthSSA z
        let sysNumSSA = freshSynthSSA z
        let bytesReadSSA = freshSynthSSA z
        let gotByteSSA = freshSynthSSA z
        let charSSA = freshSynthSSA z
        let notNewlineSSA = freshSynthSSA z
        let cont1SSA = freshSynthSSA z
        let cont2SSA = freshSynthSSA z

        let guardOps = [
            // Check pos < 255
            MLIROp.ArithOp (cmpI inBoundsSSA ICmpPred.Slt posArg maxPosSSA MLIRTypes.i64)
            // Read one byte
            MLIROp.ArithOp (constI sysNumSSA syscallNum MLIRTypes.i64)
            MLIROp.LLVMOp (inlineAsmWithSideEffects (Some bytesReadSSA) "syscall"
                "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                [i64Arg sysNumSSA; i64Arg fdSSA; ptrArg charBufSSA; i64Arg oneSSA] (Some MLIRTypes.i64))
            // Check bytes > 0
            MLIROp.ArithOp (cmpI gotByteSSA ICmpPred.Sgt bytesReadSSA zeroSSA MLIRTypes.i64)
            // Load char and check != '\n'
            MLIROp.LLVMOp (loadNonAtomic charSSA charBufSSA MLIRTypes.i8)
            MLIROp.ArithOp (cmpI notNewlineSSA ICmpPred.Ne charSSA newlineSSA MLIRTypes.i8)
            // AND conditions
            MLIROp.ArithOp (andI cont1SSA inBoundsSSA gotByteSSA MLIRTypes.i1)
            MLIROp.ArithOp (andI cont2SSA cont1SSA notNewlineSSA MLIRTypes.i1)
            // scf.condition
            MLIROp.SCFOp (scfCondition cont2SSA [posArg])
        ]

        // Body block ops
        let charToStoreSSA = freshSynthSSA z
        let storePtrSSA = freshSynthSSA z
        let oneBodySSA = freshSynthSSA z
        let nextPosSSA = freshSynthSSA z

        let bodyOps = [
            // Load the char (already read in guard)
            MLIROp.LLVMOp (loadNonAtomic charToStoreSSA charBufSSA MLIRTypes.i8)
            // GEP to buffer[pos]
            MLIROp.LLVMOp (gepSingle storePtrSSA bufPtrSSA posArg MLIRTypes.i64 MLIRTypes.i8)
            // Store char
            MLIROp.LLVMOp (storeNonAtomic charToStoreSSA storePtrSSA MLIRTypes.i8)
            // pos + 1
            MLIROp.ArithOp (constI oneBodySSA 1L MLIRTypes.i64)
            MLIROp.ArithOp (addI nextPosSSA posArg oneBodySSA MLIRTypes.i64)
            // Yield next pos
            MLIROp.SCFOp (scfYieldVal nextPosSSA)
        ]

        // Build regions
        let guardRegion = singleBlockRegion "before" [blockArg posArg MLIRTypes.i64] guardOps
        let bodyRegion = singleBlockRegion "do" [blockArg posArg MLIRTypes.i64] bodyOps

        // scf.while op
        let finalPosSSA = freshSynthSSA z
        let whileOp = MLIROp.SCFOp (scfWhile [finalPosSSA] guardRegion bodyRegion [posVal])

        // Build fat pointer result
        let undefSSA = freshSynthSSA z
        let withPtrSSA = freshSynthSSA z
        let resultSSA = freshSynthSSA z
        let buildStrOps = [
            MLIROp.LLVMOp (undef undefSSA MLIRTypes.nativeStr)
            MLIROp.LLVMOp (insertValueAt withPtrSSA undefSSA bufPtrSSA 0 MLIRTypes.nativeStr)
            MLIROp.LLVMOp (insertValueAt resultSSA withPtrSSA finalPosSSA 1 MLIRTypes.nativeStr)
        ]

        // Init value for loop (start at 0)
        let initOps = [MLIROp.ArithOp (constI (V -99) 0L MLIRTypes.i64)]  // placeholder, actual is zeroSSA

        let allOps = allocOps @ constOps @ [whileOp] @ buildStrOps
        BoundOps (allOps, Some { SSA = resultSSA; Type = MLIRTypes.nativeStr })
    | _ ->
        NotSupported "Console.readln takes no arguments"

// ===================================================================
// Registration
// ===================================================================

let registerBindings () =
    // Linux x86_64
    PlatformDispatch.register Linux X86_64 "writeBytes" (bindWriteBytes Linux)
    PlatformDispatch.register Linux X86_64 "readBytes" (bindReadBytes Linux)
    PlatformDispatch.register Linux X86_64 "Sys.write" (bindWriteBytes Linux)
    PlatformDispatch.register Linux X86_64 "Sys.read" (bindReadBytes Linux)

    // Linux ARM64
    PlatformDispatch.register Linux ARM64 "writeBytes" (bindWriteBytes Linux)
    PlatformDispatch.register Linux ARM64 "readBytes" (bindReadBytes Linux)
    PlatformDispatch.register Linux ARM64 "Sys.write" (bindWriteBytes Linux)
    PlatformDispatch.register Linux ARM64 "Sys.read" (bindReadBytes Linux)

    // macOS x86_64
    PlatformDispatch.register MacOS X86_64 "writeBytes" (bindWriteBytes MacOS)
    PlatformDispatch.register MacOS X86_64 "readBytes" (bindReadBytes MacOS)
    PlatformDispatch.register MacOS X86_64 "Sys.write" (bindWriteBytes MacOS)
    PlatformDispatch.register MacOS X86_64 "Sys.read" (bindReadBytes MacOS)

    // macOS ARM64
    PlatformDispatch.register MacOS ARM64 "writeBytes" (bindWriteBytes MacOS)
    PlatformDispatch.register MacOS ARM64 "readBytes" (bindReadBytes MacOS)
    PlatformDispatch.register MacOS ARM64 "Sys.write" (bindWriteBytes MacOS)
    PlatformDispatch.register MacOS ARM64 "Sys.read" (bindReadBytes MacOS)

let registerConsoleIntrinsics () =
    // Linux x86_64
    PlatformDispatch.register Linux X86_64 "Console.write" (bindConsoleWrite Linux)
    PlatformDispatch.register Linux X86_64 "Console.writeln" (bindConsoleWriteln Linux)
    PlatformDispatch.register Linux X86_64 "Console.error" (bindConsoleError Linux)
    PlatformDispatch.register Linux X86_64 "Console.errorln" (bindConsoleErrorln Linux)
    PlatformDispatch.register Linux X86_64 "Console.readln" (bindConsoleReadln Linux)

    // Linux ARM64
    PlatformDispatch.register Linux ARM64 "Console.write" (bindConsoleWrite Linux)
    PlatformDispatch.register Linux ARM64 "Console.writeln" (bindConsoleWriteln Linux)
    PlatformDispatch.register Linux ARM64 "Console.error" (bindConsoleError Linux)
    PlatformDispatch.register Linux ARM64 "Console.errorln" (bindConsoleErrorln Linux)
    PlatformDispatch.register Linux ARM64 "Console.readln" (bindConsoleReadln Linux)

    // macOS x86_64
    PlatformDispatch.register MacOS X86_64 "Console.write" (bindConsoleWrite MacOS)
    PlatformDispatch.register MacOS X86_64 "Console.writeln" (bindConsoleWriteln MacOS)
    PlatformDispatch.register MacOS X86_64 "Console.error" (bindConsoleError MacOS)
    PlatformDispatch.register MacOS X86_64 "Console.errorln" (bindConsoleErrorln MacOS)
    PlatformDispatch.register MacOS X86_64 "Console.readln" (bindConsoleReadln MacOS)

    // macOS ARM64
    PlatformDispatch.register MacOS ARM64 "Console.write" (bindConsoleWrite MacOS)
    PlatformDispatch.register MacOS ARM64 "Console.writeln" (bindConsoleWriteln MacOS)
    PlatformDispatch.register MacOS ARM64 "Console.error" (bindConsoleError MacOS)
    PlatformDispatch.register MacOS ARM64 "Console.errorln" (bindConsoleErrorln MacOS)
    PlatformDispatch.register MacOS ARM64 "Console.readln" (bindConsoleReadln MacOS)
