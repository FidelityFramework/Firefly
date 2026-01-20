/// Func Dialect Templates - Structured operation constructors
///
/// ARCHITECTURAL PRINCIPLE: Templates return STRUCTURED TYPES, not strings.
/// These are the "lemmas" that XParsec composes into "proofs" (complete MLIR).
///
/// Each template is a pure function: inputs → FuncOp
/// NO sprintf. NO string formatting. Just data construction.
module Alex.Dialects.Func.Templates

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════════════════════

/// Define a function: func.func
let funcDef (name: string) (args: (SSA * MLIRType) list) (retTy: MLIRType) (body: Region) (visibility: FuncVisibility) : FuncOp =
    FuncOp.FuncDef (name, args, retTy, body, visibility)

/// Define a public function
let publicFunc (name: string) (args: (SSA * MLIRType) list) (retTy: MLIRType) (body: Region) : FuncOp =
    FuncOp.FuncDef (name, args, retTy, body, Public)

/// Define a private function
let privateFunc (name: string) (args: (SSA * MLIRType) list) (retTy: MLIRType) (body: Region) : FuncOp =
    FuncOp.FuncDef (name, args, retTy, body, Private)

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION DECLARATION (external functions)
// ═══════════════════════════════════════════════════════════════════════════

/// Declare external function: func.func (declaration only)
let funcDecl (name: string) (argTypes: MLIRType list) (retTy: MLIRType) (visibility: FuncVisibility) : FuncOp =
    FuncOp.FuncDecl (name, argTypes, retTy, visibility)

/// Declare private external function
let externFunc (name: string) (argTypes: MLIRType list) (retTy: MLIRType) : FuncOp =
    FuncOp.FuncDecl (name, argTypes, retTy, Private)

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION CALL
// ═══════════════════════════════════════════════════════════════════════════

/// Call function: func.call
let funcCall (result: SSA option) (func: string) (args: Val list) (retTy: MLIRType) : FuncOp =
    FuncOp.FuncCall (result, func, args, retTy)

/// Convenience: call with SSA list (creates untyped vals)
let funcCallSSA (result: SSA option) (func: string) (args: SSA list) (argTy: MLIRType) (retTy: MLIRType) : FuncOp =
    let vals : Val list = args |> List.map (fun s -> { SSA = s; Type = argTy })
    FuncOp.FuncCall (result, func, vals, retTy)

/// Call function with result (typed args)
let callWithResult (result: SSA) (func: string) (args: Val list) (retTy: MLIRType) : FuncOp =
    FuncOp.FuncCall (Some result, func, args, retTy)

/// Call void function (typed args)
let callVoid (func: string) (args: Val list) : FuncOp =
    FuncOp.FuncCall (None, func, args, TUnit)

/// Call function with SSA args (all same type)
let callWithResultSSA (result: SSA) (func: string) (args: SSA list) (argTy: MLIRType) (retTy: MLIRType) : FuncOp =
    let vals : Val list = args |> List.map (fun s -> { SSA = s; Type = argTy })
    FuncOp.FuncCall (Some result, func, vals, retTy)

/// Call void function with SSA args (all same type)
let callVoidSSA (func: string) (args: SSA list) (argTy: MLIRType) : FuncOp =
    let vals : Val list = args |> List.map (fun s -> { SSA = s; Type = argTy })
    FuncOp.FuncCall (None, func, vals, TUnit)

// ═══════════════════════════════════════════════════════════════════════════
// RETURN
// ═══════════════════════════════════════════════════════════════════════════

/// Return from function: func.return %val : type
let funcReturn (values: (SSA * MLIRType) list) : FuncOp =
    FuncOp.FuncReturn values

/// Convenience: return a single value with type
let funcReturnOne (value: SSA) (ty: MLIRType) : FuncOp =
    FuncOp.FuncReturn [(value, ty)]

/// Convenience: return nothing (void function)
let funcReturnVoid : FuncOp =
    FuncOp.FuncReturn []

/// Return value with type (alias for funcReturnOne)
let returnVal (value: SSA) (ty: MLIRType) : FuncOp =
    FuncOp.FuncReturn [(value, ty)]

/// Return void (alias for funcReturnVoid)
let returnVoid : FuncOp =
    FuncOp.FuncReturn []

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Build function arguments from SSA indices starting at 0
let buildArgs (types: MLIRType list) : (SSA * MLIRType) list =
    types |> List.mapi (fun i t -> (Arg i, t))

/// Build a simple function body region with one block
let singleBlockBody (args: (SSA * MLIRType) list) (ops: MLIROp list) : Region =
    {
        Blocks = [
            {
                Label = BlockRef "entry"
                Args = args |> List.map (fun (s, t) -> { SSA = s; Type = t })
                Ops = ops
            }
        ]
    }

// ═══════════════════════════════════════════════════════════════════════════
// WRAP TO MLIROp
// ═══════════════════════════════════════════════════════════════════════════

/// Wrap FuncOp in MLIROp
let wrap (op: FuncOp) : MLIROp = MLIROp.FuncOp op
