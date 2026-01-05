/// ConsoleBindings - Platform-specific console I/O bindings (witness-based)
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// Uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
module Alex.Bindings.Console.ConsoleBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data: Syscall numbers and conventions
// ===================================================================

module SyscallData =
    /// Linux x86-64 syscall numbers
    let linuxSyscalls = Map [
        "read", 0L
        "write", 1L
    ]

    /// macOS syscall numbers (with BSD 0x2000000 offset for x86-64)
    let macosSyscalls = Map [
        "read", 0x2000003L
        "write", 0x2000004L
    ]

    let getSyscallNumber (os: OSFamily) (name: string) : int64 option =
        match os with
        | Linux -> Map.tryFind name linuxSyscalls
        | MacOS -> Map.tryFind name macosSyscalls
        | _ -> None

// ===================================================================
// Helper Witness Functions
// ===================================================================

/// Witness sign extension (extsi) if needed
let witnessExtSIIfNeeded (ssaName: string) (fromType: MLIRType) (toWidth: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
    match fromType with
    | Integer fromWidth when fromWidth <> toWidth ->
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let fromStr = Serialize.integerBitWidth fromWidth
        let toStr = Serialize.integerBitWidth toWidth
        let text = sprintf "%s = arith.extsi %s : %s to %s" resultSSA ssaName fromStr toStr
        resultSSA, MLIRZipper.witnessOpWithResult text resultSSA (Integer toWidth) zipper'
    | _ ->
        // No extension needed
        ssaName, zipper

/// Witness truncation (trunci)
let witnessTruncI (ssaName: string) (fromType: MLIRType) (toWidth: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    let fromStr = Serialize.mlirType fromType
    let toStr = Serialize.integerBitWidth toWidth
    let text = sprintf "%s = arith.trunci %s : %s to %s" resultSSA ssaName fromStr toStr
    resultSSA, MLIRZipper.witnessOpWithResult text resultSSA (Integer toWidth) zipper'

// ===================================================================
// MLIR Generation for Console Primitives (Witness-Based)
// ===================================================================

/// Witness write syscall for Unix-like systems (Linux, macOS)
let witnessUnixWriteSyscall (syscallNum: int64) (fd: string) (fdType: MLIRType) (buf: string) (bufType: MLIRType) (count: string) (countType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // Extend fd to i64 if needed
    let fdExt, zipper1 = witnessExtSIIfNeeded fd fdType I64 zipper

    // Extend count to i64 if needed
    let countExt, zipper2 = witnessExtSIIfNeeded count countType I64 zipper1

    // Witness syscall number constant
    let sysNumSSA, zipper3 = MLIRZipper.witnessConstant syscallNum I64 zipper2

    // Witness syscall: write(fd, buf, count)
    // rax = syscall number, rdi = fd, rsi = buf, rdx = count
    // Buffer type can be !llvm.ptr or i64 (after ptrtoint conversion)
    let bufTypeStr = Serialize.mlirType bufType
    let args = [
        (fdExt, "i64")
        (buf, bufTypeStr)
        (countExt, "i64")
    ]
    MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper3

/// Witness read syscall for Unix-like systems (Linux, macOS)
let witnessUnixReadSyscall (syscallNum: int64) (fd: string) (fdType: MLIRType) (buf: string) (bufType: MLIRType) (maxCount: string) (countType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // Extend fd to i64 if needed
    let fdExt, zipper1 = witnessExtSIIfNeeded fd fdType I64 zipper

    // Extend count to i64 if needed
    let countExt, zipper2 = witnessExtSIIfNeeded maxCount countType I64 zipper1

    // Witness syscall number constant
    let sysNumSSA, zipper3 = MLIRZipper.witnessConstant syscallNum I64 zipper2

    // Witness syscall: read(fd, buf, count)
    // Buffer type can be !llvm.ptr or i64 (after ptrtoint conversion)
    let bufTypeStr = Serialize.mlirType bufType
    let args = [
        (fdExt, "i64")
        (buf, bufTypeStr)
        (countExt, "i64")
    ]
    MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper3

// ===================================================================
// Platform Bindings (Witness-Based Pattern)
// ===================================================================

/// writeBytes - write bytes to file descriptor
/// Witness binding from Alloy.Platform.Bindings.writeBytes
let witnessWriteBytes (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(fd, fdType); (buf, bufType); (count, countType)] ->
        match platform.OS with
        | Linux ->
            let resultSSA, zipper1 = witnessUnixWriteSyscall 1L fd fdType buf bufType count countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | MacOS ->
            let resultSSA, zipper1 = witnessUnixWriteSyscall 0x2000004L fd fdType buf bufType count countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | Windows ->
            zipper, NotSupported "Windows console not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Console not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "writeBytes requires (fd, buffer, count) arguments"

/// readBytes - read bytes from file descriptor
/// Witness binding from Alloy.Platform.Bindings.readBytes
let witnessReadBytes (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(fd, fdType); (buf, bufType); (maxCount, countType)] ->
        match platform.OS with
        | Linux ->
            let resultSSA, zipper1 = witnessUnixReadSyscall 0L fd fdType buf bufType maxCount countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | MacOS ->
            let resultSSA, zipper1 = witnessUnixReadSyscall 0x2000003L fd fdType buf bufType maxCount countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | Windows ->
            zipper, NotSupported "Windows console not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Console not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "readBytes requires (fd, buffer, maxCount) arguments"

// ===================================================================
// Registration (Witness-Based)
// ===================================================================

/// Register all console bindings for all platforms
/// Entry points match Platform.Bindings function names AND FNCS Sys intrinsics
let registerBindings () =
    // Register for Linux x86_64
    PlatformDispatch.register Linux X86_64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.linux_x86_64 prim zipper)
    // FNCS Sys intrinsics - same implementation as writeBytes/readBytes
    PlatformDispatch.register Linux X86_64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.linux_x86_64 prim zipper)

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "readBytes"
        (fun prim zipper -> witnessReadBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Sys.read"
        (fun prim zipper -> witnessReadBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)

    // Register for macOS x86_64
    PlatformDispatch.register MacOS X86_64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_x86_64 prim zipper)

    // Register for macOS ARM64
    PlatformDispatch.register MacOS ARM64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_arm64 prim zipper)

// ===================================================================
// Console Intrinsics (FNCS Post-Alloy Absorption)
// Console.write, Console.writeln, Console.error, Console.errorln, Console.readln
// ===================================================================

/// Witness Console.write - write string to stdout
/// Takes a string (fat pointer: ptr + length) and writes to fd=1
let witnessConsoleWrite (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(strSSA, Struct _)] ->
        // Extract pointer and length from fat pointer (string)
        let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let extractPtrText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let zipper2 = MLIRZipper.witnessOpWithResult extractPtrText ptrSSA Pointer zipper1

        let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let extractLenText = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenSSA strSSA
        let zipper4 = MLIRZipper.witnessOpWithResult extractLenText lenSSA (Integer I64) zipper3

        // Create fd=1 constant for stdout
        let fdSSA, zipper5 = MLIRZipper.witnessConstant 1L I64 zipper4

        // Call write syscall
        let syscallNum = match platform.OS with Linux -> 1L | MacOS -> 0x2000004L | _ -> 1L
        let resultSSA, zipper6 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) ptrSSA Pointer lenSSA (Integer I64) zipper5

        // Console.write returns unit, so just return unit
        zipper6, WitnessedValue ("", Unit)
    | _ ->
        zipper, NotSupported "Console.write requires (string) argument"

/// Witness Console.writeln - write string with newline to stdout
let witnessConsoleWriteln (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(strSSA, Struct _)] ->
        // First write the string
        let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let extractPtrText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let zipper2 = MLIRZipper.witnessOpWithResult extractPtrText ptrSSA Pointer zipper1

        let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let extractLenText = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenSSA strSSA
        let zipper4 = MLIRZipper.witnessOpWithResult extractLenText lenSSA (Integer I64) zipper3

        let fdSSA, zipper5 = MLIRZipper.witnessConstant 1L I64 zipper4

        let syscallNum = match platform.OS with Linux -> 1L | MacOS -> 0x2000004L | _ -> 1L
        let _, zipper6 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) ptrSSA Pointer lenSSA (Integer I64) zipper5

        // Now write the newline character
        // Create a stack-allocated byte for newline
        let oneSSA, zipper7 = MLIRZipper.witnessConstant 1L I64 zipper6
        let nlPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" nlPtrSSA oneSSA
        let zipper9 = MLIRZipper.witnessOpWithResult allocaText nlPtrSSA Pointer zipper8

        // Store newline character (10 = '\n')
        let nlCharSSA, zipper10 = MLIRZipper.witnessConstant 10L I8 zipper9
        let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" nlCharSSA nlPtrSSA
        let zipper11 = MLIRZipper.witnessOp storeText [] zipper10

        // Write the newline
        let _, zipper12 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) nlPtrSSA Pointer oneSSA (Integer I64) zipper11

        zipper12, WitnessedValue ("", Unit)
    | _ ->
        zipper, NotSupported "Console.writeln requires (string) argument"

/// Witness Console.error - write string to stderr
let witnessConsoleError (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(strSSA, Struct _)] ->
        let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let extractPtrText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let zipper2 = MLIRZipper.witnessOpWithResult extractPtrText ptrSSA Pointer zipper1

        let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let extractLenText = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenSSA strSSA
        let zipper4 = MLIRZipper.witnessOpWithResult extractLenText lenSSA (Integer I64) zipper3

        // fd=2 for stderr
        let fdSSA, zipper5 = MLIRZipper.witnessConstant 2L I64 zipper4

        let syscallNum = match platform.OS with Linux -> 1L | MacOS -> 0x2000004L | _ -> 1L
        let _, zipper6 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) ptrSSA Pointer lenSSA (Integer I64) zipper5

        zipper6, WitnessedValue ("", Unit)
    | _ ->
        zipper, NotSupported "Console.error requires (string) argument"

/// Witness Console.errorln - write string with newline to stderr
let witnessConsoleErrorln (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(strSSA, Struct _)] ->
        let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let extractPtrText = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrSSA strSSA
        let zipper2 = MLIRZipper.witnessOpWithResult extractPtrText ptrSSA Pointer zipper1

        let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
        let extractLenText = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenSSA strSSA
        let zipper4 = MLIRZipper.witnessOpWithResult extractLenText lenSSA (Integer I64) zipper3

        // fd=2 for stderr
        let fdSSA, zipper5 = MLIRZipper.witnessConstant 2L I64 zipper4

        let syscallNum = match platform.OS with Linux -> 1L | MacOS -> 0x2000004L | _ -> 1L
        let _, zipper6 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) ptrSSA Pointer lenSSA (Integer I64) zipper5

        // Write newline
        let oneSSA, zipper7 = MLIRZipper.witnessConstant 1L I64 zipper6
        let nlPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" nlPtrSSA oneSSA
        let zipper9 = MLIRZipper.witnessOpWithResult allocaText nlPtrSSA Pointer zipper8

        let nlCharSSA, zipper10 = MLIRZipper.witnessConstant 10L I8 zipper9
        let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" nlCharSSA nlPtrSSA
        let zipper11 = MLIRZipper.witnessOp storeText [] zipper10

        let _, zipper12 = witnessUnixWriteSyscall syscallNum fdSSA (Integer I64) nlPtrSSA Pointer oneSSA (Integer I64) zipper11

        zipper12, WitnessedValue ("", Unit)
    | _ ->
        zipper, NotSupported "Console.errorln requires (string) argument"

/// Witness Console.readln - read line from stdin
/// Returns a string (fat pointer). For now, uses a fixed buffer size.
let witnessConsoleReadln (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    // Match no args, unit arg (MLIRType.Unit), or pseudo-unit (i32 with value 0)
    // Unit literals are represented as "arith.constant 0 : i32" so we accept Integer I32 as well
    match prim.Args with
    | [] | [(_, Unit)] | [(_, Integer I32)] ->
        // Allocate buffer on stack (256 bytes for now)
        let bufSizeSSA, zipper1 = MLIRZipper.witnessConstant 256L I64 zipper
        let bufPtrSSA, zipper2 = MLIRZipper.yieldSSA zipper1
        let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufPtrSSA bufSizeSSA
        let zipper3 = MLIRZipper.witnessOpWithResult allocaText bufPtrSSA Pointer zipper2

        // fd=0 for stdin
        let fdSSA, zipper4 = MLIRZipper.witnessConstant 0L I64 zipper3

        // Read syscall
        let syscallNum = match platform.OS with Linux -> 0L | MacOS -> 0x2000003L | _ -> 0L
        let bytesReadSSA, zipper5 = witnessUnixReadSyscall syscallNum fdSSA (Integer I64) bufPtrSSA Pointer bufSizeSSA (Integer I64) zipper4

        // Strip trailing newline if present (adjust length by -1 if last char is \n)
        // For simplicity, just use bytesRead - 1 (assumes newline)
        let oneSSA, zipper6 = MLIRZipper.witnessConstant 1L I64 zipper5
        let adjLenSSA, zipper7 = MLIRZipper.yieldSSA zipper6
        let subText = sprintf "%s = arith.subi %s, %s : i64" adjLenSSA bytesReadSSA oneSSA
        let zipper8 = MLIRZipper.witnessOpWithResult subText adjLenSSA (Integer I64) zipper7

        // Construct fat pointer (string) from buffer and length
        let undefSSA, zipper9 = MLIRZipper.yieldSSA zipper8
        let undefText = sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" undefSSA
        let zipper10 = MLIRZipper.witnessOpWithResult undefText undefSSA (Struct [Pointer; Integer I64]) zipper9

        let withPtrSSA, zipper11 = MLIRZipper.yieldSSA zipper10
        let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" withPtrSSA bufPtrSSA undefSSA
        let zipper12 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA (Struct [Pointer; Integer I64]) zipper11

        let resultSSA, zipper13 = MLIRZipper.yieldSSA zipper12
        let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" resultSSA adjLenSSA withPtrSSA
        let zipper14 = MLIRZipper.witnessOpWithResult insertLenText resultSSA (Struct [Pointer; Integer I64]) zipper13

        zipper14, WitnessedValue (resultSSA, Struct [Pointer; Integer I64])
    | _ ->
        zipper, NotSupported "Console.readln takes no arguments"

/// Register Console intrinsic bindings
let registerConsoleIntrinsics () =
    // Linux x86_64
    PlatformDispatch.register Linux X86_64 "Console.write"
        (fun prim zipper -> witnessConsoleWrite TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Console.writeln"
        (fun prim zipper -> witnessConsoleWriteln TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Console.error"
        (fun prim zipper -> witnessConsoleError TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Console.errorln"
        (fun prim zipper -> witnessConsoleErrorln TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Console.readln"
        (fun prim zipper -> witnessConsoleReadln TargetPlatform.linux_x86_64 prim zipper)

    // Linux ARM64
    PlatformDispatch.register Linux ARM64 "Console.write"
        (fun prim zipper -> witnessConsoleWrite { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Console.writeln"
        (fun prim zipper -> witnessConsoleWriteln { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Console.error"
        (fun prim zipper -> witnessConsoleError { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Console.errorln"
        (fun prim zipper -> witnessConsoleErrorln { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Console.readln"
        (fun prim zipper -> witnessConsoleReadln { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)

    // macOS x86_64
    PlatformDispatch.register MacOS X86_64 "Console.write"
        (fun prim zipper -> witnessConsoleWrite TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Console.writeln"
        (fun prim zipper -> witnessConsoleWriteln TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Console.error"
        (fun prim zipper -> witnessConsoleError TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Console.errorln"
        (fun prim zipper -> witnessConsoleErrorln TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Console.readln"
        (fun prim zipper -> witnessConsoleReadln TargetPlatform.macos_x86_64 prim zipper)

    // macOS ARM64
    PlatformDispatch.register MacOS ARM64 "Console.write"
        (fun prim zipper -> witnessConsoleWrite TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Console.writeln"
        (fun prim zipper -> witnessConsoleWriteln TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Console.error"
        (fun prim zipper -> witnessConsoleError TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Console.errorln"
        (fun prim zipper -> witnessConsoleErrorln TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Console.readln"
        (fun prim zipper -> witnessConsoleReadln TargetPlatform.macos_arm64 prim zipper)
