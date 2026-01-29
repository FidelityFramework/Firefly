/// CLI Diagnostics Types
///
/// Simple error reporting types for CLI diagnostics and doctor command.
/// Not related to parsing or XParsec.
module CLI.Diagnostics.Types

/// Diagnostic errors for toolchain verification
type DiagnosticError =
    | ConversionError of phase: string * source: string * target: string * message: string
    | SyntaxError of location: string * message: string * context: string list
    | TypeCheckError of construct: string * message: string * details: string option
    | InternalError of phase: string * message: string * details: string option
    | ParseError of location: string * message: string
    | DependencyResolutionError of symbol: string * message: string

/// Result type for diagnostic operations
type DiagnosticResult<'T> =
    | Success of 'T
    | Failure of DiagnosticError list
