/// FuncElements - Atomic Func dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides ALL Func dialect operations from Types.fs.
module internal Alex.Elements.FuncElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// Note: FuncVisibility is assumed to be defined in Types.fs

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

let pFuncDef (name: string) (args: (SSA * MLIRType) list) (retTy: MLIRType)
                 (body: Region) (visibility: FuncVisibility) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncDef (name, args, retTy, body, visibility))
    }

// ═══════════════════════════════════════════════════════════
// FUNCTION DECLARATION (external)
// ═══════════════════════════════════════════════════════════

let pFuncDecl (name: string) (argTypes: MLIRType list) (retTy: MLIRType)
                  (visibility: FuncVisibility) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncDecl (name, argTypes, retTy, visibility))
    }

// ═══════════════════════════════════════════════════════════
// DIRECT CALL
// ═══════════════════════════════════════════════════════════

let pFuncCall (result: SSA option) (func: string) (args: Val list) (retTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncCall (result, func, args, retTy))
    }

// ═══════════════════════════════════════════════════════════
// INDIRECT CALL
// ═══════════════════════════════════════════════════════════

let pFuncCallIndirect (result: SSA option) (callee: SSA) (args: Val list) (retTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncCallIndirect (result, callee, args, retTy))
    }

// ═══════════════════════════════════════════════════════════
// FUNCTION CONSTANT (pointer to function)
// ═══════════════════════════════════════════════════════════

let pFuncConstant (result: SSA) (funcName: string) (funcTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncConstant (result, funcName, funcTy))
    }
