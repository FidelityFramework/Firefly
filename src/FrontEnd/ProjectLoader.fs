/// ProjectLoader - FrontEnd entry point
///
/// Clean public API: .fidproj path → PSG + PlatformContext
/// Wraps FNCS project loading and type checking
module FrontEnd.ProjectLoader

open FSharp.Native.Compiler.Project

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Load and type-check a Firefly project
/// This is the single entry point for the FrontEnd
let load (projectPath: string) : Result<ProjectCheckResult, string> =
    ProjectChecker.checkProject projectPath
