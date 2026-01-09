/// Witness - The observation function for PSG nodes
///
/// ARCHITECTURAL PRINCIPLE:
/// The witness OBSERVES what's at the zipper focus.
/// It uses semantic patterns as LENSES to classify what it sees.
/// It RETURNS ops based on the observation.
///
/// "I am focused on a node. I observe its kind through my lens.
/// Based on what I see, I return the appropriate ops."
module Alex.Traversal.Witness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.Zipper
open Alex.Traversal.WitnessFold
open Alex.Traversal.Coeffects
open Alex.Patterns.SemanticPatterns
open Alex.Preprocessing.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// OBSERVATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get the result SSA from the focused node
let private focusResultSSA (z: Zipper) : SSA =
    match lookupSSA z.Focus.Id z.Coeffects with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" z.Focus.Id

/// Get all SSAs for the focused node (multi-SSA expansion)
let private focusAllSSAs (z: Zipper) : SSA list =
    match lookupSSAs z.Focus.Id z.Coeffects with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" z.Focus.Id

/// Get child result value by index
let private childValue (index: int) (childResults: WitnessResult list) : Val option =
    if index < List.length childResults then
        childResults.[index].Result
    else None

/// Require child result value
let private requireChildValue (index: int) (childResults: WitnessResult list) : Val =
    match childValue index childResults with
    | Some v -> v
    | None -> failwithf "No result from child %d" index

// ═══════════════════════════════════════════════════════════════════════════
// LITERAL OBSERVATION
// ═══════════════════════════════════════════════════════════════════════════

let private observeLiteral (z: Zipper) (lit: LiteralValue) : WitnessResult =
    match lit with
    | LiteralValue.Unit ->
        let ssa = focusResultSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, 0L, MLIRTypes.i32))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i32 }

    | LiteralValue.Bool b ->
        let ssa = focusResultSSA z
        let value = if b then 1L else 0L
        let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, value, MLIRTypes.i1))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i1 }

    | LiteralValue.Int32 n ->
        let ssa = focusResultSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, int64 n, MLIRTypes.i32))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i32 }

    | LiteralValue.Int64 n ->
        let ssa = focusResultSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, n, MLIRTypes.i64))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i64 }

    | LiteralValue.Float32 f ->
        let ssa = focusResultSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstF (ssa, float f, MLIRTypes.f32))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.f32 }

    | LiteralValue.Float64 f ->
        let ssa = focusResultSSA z
        let op = MLIROp.ArithOp (ArithOp.ConstF (ssa, f, MLIRTypes.f64))
        WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.f64 }

    | LiteralValue.String s ->
        // String literals need 5 SSAs: ptr, len, undef, withPtr, fatPtr
        let ssas = focusAllSSAs z
        match ssas with
        | [ptrSSA; lenSSA; undefSSA; withPtrSSA; fatPtrSSA] ->
            let hash = uint32 (s.GetHashCode())
            let len = s.Length
            let fatStringType = TStruct [MLIRTypes.ptr; MLIRTypes.i64]

            let ops = [
                MLIROp.LLVMOp (LLVMOp.AddressOf (ptrSSA, GString hash))
                MLIROp.ArithOp (ArithOp.ConstI (lenSSA, int64 len, MLIRTypes.i64))
                MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, fatStringType))
                MLIROp.LLVMOp (LLVMOp.InsertValue (withPtrSSA, undefSSA, ptrSSA, [0], fatStringType))
                MLIROp.LLVMOp (LLVMOp.InsertValue (fatPtrSSA, withPtrSSA, lenSSA, [1], fatStringType))
            ]
            WitnessResult.withResult ops { SSA = fatPtrSSA; Type = fatStringType }
        | _ ->
            failwithf "String literal needs 5 SSAs, got %d" (List.length ssas)

    | _ ->
        // Other literals - placeholder
        WitnessResult.empty

// ═══════════════════════════════════════════════════════════════════════════
// BINDING OBSERVATION
// ═══════════════════════════════════════════════════════════════════════════

let private observeBinding (z: Zipper) (childResults: WitnessResult list) (name: string) (isMutable: bool) : WitnessResult =
    // Binding: the child is the value being bound
    // We pass through the child's ops and result
    match childResults with
    | [childResult] ->
        // The binding's result is the bound value's result
        { Ops = childResult.Ops; Result = childResult.Result }
    | _ ->
        WitnessResult.empty

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENTIAL OBSERVATION
// ═══════════════════════════════════════════════════════════════════════════

let private observeSequential (z: Zipper) (childResults: WitnessResult list) : WitnessResult =
    // Sequential: combine all child ops, return last child's result
    let allOps = childResults |> List.collect (fun r -> r.Ops)
    let lastResult = childResults |> List.tryLast |> Option.bind (fun r -> r.Result)
    { Ops = allOps; Result = lastResult }

// ═══════════════════════════════════════════════════════════════════════════
// LAMBDA OBSERVATION
// ═══════════════════════════════════════════════════════════════════════════

let private observeLambda (z: Zipper) (childResults: WitnessResult list) (paramList: (string * NativeType) list) : WitnessResult =
    // Lambda: creates a function definition
    // The child result is the function body
    let funcName =
        match lookupLambdaName z.Focus.Id z.Coeffects with
        | Some name -> name
        | None -> sprintf "lambda_%d" (NodeId.value z.Focus.Id)

    // Map parameter types using coeffects
    let paramSSAsAndTypes =
        paramList
        |> List.mapi (fun i (_name, ty) ->
            let mlirType = mapType ty z.Coeffects
            (Arg i, mlirType))

    // Convert to block args (Val list)
    let blockArgs: BlockArg list =
        paramSSAsAndTypes
        |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })

    // Body ops from child - get the return type from the last child result
    let bodyOps, returnType =
        match childResults with
        | [bodyResult] ->
            let retTy =
                match bodyResult.Result with
                | Some v -> v.Type
                | None -> MLIRTypes.i32  // Unit represented as i32
            bodyResult.Ops, retTy
        | _ -> [], MLIRTypes.i32

    // Build function
    let visibility =
        if isEntryPoint z.Focus.Id z.Coeffects then FuncVisibility.Public
        else FuncVisibility.Private

    let bodyBlock: Block = {
        Label = BlockRef "entry"
        Args = blockArgs
        Ops = bodyOps
    }

    let funcOp = MLIROp.FuncOp (FuncOp.FuncDef (
        funcName,
        paramSSAsAndTypes,
        returnType,
        { Blocks = [bodyBlock] },
        visibility
    ))

    WitnessResult.ops [funcOp]

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC OBSERVATION (using semantic pattern lenses)
// ═══════════════════════════════════════════════════════════════════════════

let private observeIntrinsic (z: Zipper) (childResults: WitnessResult list) (info: IntrinsicInfo) : WitnessResult =
    // Look through semantic pattern lenses
    match info with
    | PlatformIntrinsic (module', op) ->
        // Platform intrinsic - lookup pre-resolved binding from coeffects
        let nodeIdVal = NodeId.value z.Focus.Id
        match lookupPlatformBinding nodeIdVal z.Coeffects with
        | Some resolution ->
            // Use the pre-resolved binding
            match resolution.Resolved with
            | Syscall (syscallNum, constraints) ->
                // Generate syscall inline asm
                let ssa = focusResultSSA z
                let args = childResults |> List.choose (fun r -> r.Result)
                let argPairs = args |> List.map (fun v -> (v.SSA, v.Type))
                let op = MLIROp.LLVMOp (LLVMOp.InlineAsm (
                    Some ssa, "syscall", constraints, argPairs, Some MLIRTypes.i64, true, false))
                WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i64 }
            | LibcCall funcName ->
                // Generate libc call
                let ssa = focusResultSSA z
                let args = childResults |> List.choose (fun r -> r.Result)
                let op = MLIROp.LLVMOp (LLVMOp.Call (Some ssa, GFunc funcName, args, MLIRTypes.i64))
                WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i64 }
            | InlineAsm (asm, constraints) ->
                let ssa = focusResultSSA z
                let args = childResults |> List.choose (fun r -> r.Result)
                let argPairs = args |> List.map (fun v -> (v.SSA, v.Type))
                let op = MLIROp.LLVMOp (LLVMOp.InlineAsm (Some ssa, asm, constraints, argPairs, Some MLIRTypes.i64, false, false))
                WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.i64 }
        | None ->
            // No pre-resolved binding - this is an error in the architecture
            // Platform intrinsics should always be resolved by the nanopass
            failwithf "Platform intrinsic %A.%s has no pre-resolved binding" module' op

    | NativePtrOp kind ->
        // NativePtr operations
        let ssa = focusResultSSA z
        match kind with
        | PtrAdd ->
            match childResults with
            | [_ptrResult; _idxResult] ->
                let ptr = requireChildValue 0 childResults
                let idx = requireChildValue 1 childResults
                // GEP: result, base, indices (with types), element type
                let op = MLIROp.LLVMOp (LLVMOp.GEP (ssa, ptr.SSA, [(idx.SSA, idx.Type)], MLIRTypes.i8))
                WitnessResult.withResult [op] { SSA = ssa; Type = MLIRTypes.ptr }
            | _ -> WitnessResult.empty
        | _ ->
            // Other NativePtr ops - placeholder
            WitnessResult.empty

    | _ ->
        // Other intrinsics - placeholder
        WitnessResult.empty

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// The main witness function - observes the focused node and returns ops.
/// This is the "photographer" - it sees what's there and captures it.
let witness (z: Zipper) (childResults: WitnessResult list) : WitnessResult =
    match z.Focus.Kind with
    | SemanticKind.Literal lit ->
        observeLiteral z lit

    | SemanticKind.Binding (name, isMutable, _ty, _isEntry) ->
        observeBinding z childResults name isMutable

    | SemanticKind.Sequential _ ->
        observeSequential z childResults

    | SemanticKind.Lambda (paramList, _bodyId) ->
        observeLambda z childResults paramList

    | SemanticKind.Intrinsic info ->
        observeIntrinsic z childResults info

    | SemanticKind.VarRef (_name, _definition) ->
        // Variable reference - lookup SSA and type from coeffects
        let ssa = focusResultSSA z
        // Map the node's type to MLIR type
        let mlirType = mapType z.Focus.Type z.Coeffects
        WitnessResult.value { SSA = ssa; Type = mlirType }

    | _ ->
        // Other node kinds - placeholder
        // Collect child ops and pass through
        let allOps = childResults |> List.collect (fun r -> r.Ops)
        { Ops = allOps; Result = None }
