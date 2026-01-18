#!/usr/bin/env dotnet fsi
// Runner.fsx - Firefly Regression Test Runner
// Usage: dotnet fsi Runner.fsx [options]

#r "/home/hhh/repos/Firefly/src/bin/Debug/net10.0/XParsec.dll"
#r "/home/hhh/repos/Fidelity.Toml/src/bin/Debug/net10.0/Fidelity.Toml.dll"

open System
open System.IO
open System.Diagnostics
open System.Text
open Fidelity.Toml

// =============================================================================
// Types
// =============================================================================

type ProcessResult =
    | Completed of exitCode: int * stdout: string * stderr: string
    | Timeout of timeoutMs: int
    | Failed of exn: Exception

type SampleDef = {
    Name: string
    ProjectFile: string
    BinaryName: string
    StdinFile: string option
    ExpectedOutput: string
    TimeoutSeconds: int
    Skip: bool
    SkipReason: string option
}

type CompileResult =
    | CompileSuccess of durationMs: int64
    | CompileFailed of exitCode: int * stderr: string * durationMs: int64
    | CompileTimeout of timeoutMs: int
    | CompileSkipped of reason: string

type RunResult =
    | RunSuccess of durationMs: int64
    | RunFailed of exitCode: int * stdout: string * stderr: string * durationMs: int64
    | OutputMismatch of expected: string * actual: string * durationMs: int64
    | RunTimeout of timeoutMs: int
    | RunSkipped of reason: string

type TestResult = { Sample: SampleDef; CompileResult: CompileResult; RunResult: RunResult option }
type TestConfig = { SamplesRoot: string; CompilerPath: string; DefaultTimeoutSeconds: int }
type TestReport = { RunId: string; ManifestPath: string; CompilerPath: string; StartTime: DateTime; EndTime: DateTime; Results: TestResult list }

type CliOptions = { ManifestPath: string; TargetSample: string option; Verbose: bool; TimeoutOverride: int option }

// =============================================================================
// Process Runner
// =============================================================================

let runWithTimeout cmd args workDir stdin (timeoutMs: int) =
    try
        use proc = new Process()
        proc.StartInfo <- ProcessStartInfo(
            FileName = cmd, Arguments = args, WorkingDirectory = workDir,
            RedirectStandardOutput = true, RedirectStandardError = true,
            RedirectStandardInput = Option.isSome stdin, UseShellExecute = false, CreateNoWindow = true)
        proc.Start() |> ignore
        match stdin with
        | Some input -> proc.StandardInput.Write(input: string); proc.StandardInput.Close()
        | None -> ()
        if not (proc.WaitForExit(timeoutMs)) then
            try proc.Kill() with _ -> ()
            Timeout timeoutMs
        else Completed(proc.ExitCode, proc.StandardOutput.ReadToEnd(), proc.StandardError.ReadToEnd())
    with ex -> Failed ex

let runWithTimeoutMeasured cmd args workDir stdin timeoutMs =
    let sw = Stopwatch.StartNew()
    let result = runWithTimeout cmd args workDir stdin timeoutMs
    sw.Stop()
    (result, sw.ElapsedMilliseconds)

let compileSample compilerPath projectDir projectFile timeoutMs =
    let (result, ms) = runWithTimeoutMeasured compilerPath $"compile {projectFile}" projectDir None timeoutMs
    match result with
    | Completed (0, _, _) -> CompileSuccess ms
    | Completed (code, _, stderr) -> CompileFailed (code, stderr, ms)
    | Timeout t -> CompileTimeout t
    | Failed ex -> CompileFailed (-1, ex.Message, ms)

let runBinary binaryPath workDir stdin timeoutMs =
    let (result, ms) = runWithTimeoutMeasured binaryPath "" workDir stdin timeoutMs
    match result with
    | Completed (0, stdout, _) -> (RunSuccess ms, stdout)
    | Completed (code, stdout, stderr) -> (RunFailed (code, stdout, stderr, ms), stdout)
    | Timeout t -> (RunTimeout t, "")
    | Failed ex -> (RunFailed (-1, "", ex.Message, ms), "")

// =============================================================================
// Output Verifier
// =============================================================================

let normalizeOutput (s: string) =
    s.Replace("\r\n", "\n").Split('\n')
    |> Array.map (fun line -> line.TrimEnd())
    |> String.concat "\n" |> fun s -> s.TrimEnd()

let outputMatches expected actual = normalizeOutput expected = normalizeOutput actual

let createDiffSummary expected actual maxLines =
    let expLines = (normalizeOutput expected).Split('\n')
    let actLines = (normalizeOutput actual).Split('\n')
    let sb = StringBuilder()
    let rec findDiff i =
        if i >= max expLines.Length actLines.Length then None
        else
            let e = if i < expLines.Length then expLines.[i] else "<end>"
            let a = if i < actLines.Length then actLines.[i] else "<end>"
            if e <> a then Some (i + 1, e, a) else findDiff (i + 1)
    match findDiff 0 with
    | Some (n, e, a) -> sb.AppendLine($"  First diff at line {n}:").AppendLine($"    Expected: {e}").AppendLine($"    Actual:   {a}") |> ignore
    | None -> sb.AppendLine("  No line diff found") |> ignore
    sb.AppendLine().AppendLine($"  Expected (first {maxLines} lines):") |> ignore
    expLines |> Array.truncate maxLines |> Array.iter (fun l -> sb.AppendLine($"    {l}") |> ignore)
    sb.AppendLine().AppendLine($"  Actual (first {maxLines} lines):") |> ignore
    actLines |> Array.truncate maxLines |> Array.iter (fun l -> sb.AppendLine($"    {l}") |> ignore)
    sb.ToString()

// =============================================================================
// Sample Discovery (Manifest Parsing)
// =============================================================================

let loadManifest manifestPath =
    let manifestDir = Path.GetDirectoryName(Path.GetFullPath(manifestPath))
    let doc = Toml.parseOrFail (File.ReadAllText(manifestPath))
    let resolve (path: string) = if Path.IsPathRooted(path) then path else Path.GetFullPath(Path.Combine(manifestDir, path))
    let config = {
        SamplesRoot = Toml.getString "config.samples_root" doc |> Option.defaultValue "" |> resolve
        CompilerPath = Toml.getString "config.compiler" doc |> Option.defaultValue "" |> resolve
        DefaultTimeoutSeconds = Toml.getInt "config.default_timeout_seconds" doc |> Option.map int |> Option.defaultValue 30
    }
    let samples =
        match Toml.getValue "samples" doc with
        | Some (TomlValue.Array items) ->
            items |> List.choose (function
                | TomlValue.Table tbl ->
                    let str k = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.String s -> Some s | _ -> None) |> Option.defaultValue ""
                    let strOpt k = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.String s -> Some s | _ -> None)
                    let intVal k d = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.Integer i -> Some (int i) | _ -> None) |> Option.defaultValue d
                    let boolVal k d = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.Boolean b -> Some b | _ -> None) |> Option.defaultValue d
                    Some { Name = str "name"; ProjectFile = str "project"; BinaryName = str "binary"
                           StdinFile = strOpt "stdin_file"; ExpectedOutput = str "expected_output"
                           TimeoutSeconds = intVal "timeout_seconds" config.DefaultTimeoutSeconds
                           Skip = boolVal "skip" false; SkipReason = strOpt "skip_reason" }
                | _ -> None)
        | _ -> []
    (config, samples)

let getSampleDir config sample = Path.Combine(config.SamplesRoot, sample.Name)
let getBinaryPath config sample = Path.Combine(getSampleDir config sample, sample.BinaryName)
let getStdinContent config sample =
    match sample.StdinFile with
    | Some file -> let p = Path.Combine(getSampleDir config sample, file) in if File.Exists(p) then Some (File.ReadAllText(p)) else None
    | None -> None

// =============================================================================
// Report Generator
// =============================================================================

let formatDuration ms = if ms < 1000L then $"{ms}ms" else $"{float ms / 1000.0:F2}s"
let compileStatusStr = function CompileSuccess _ -> "PASS" | CompileFailed _ -> "FAIL" | CompileTimeout _ -> "TIMEOUT" | CompileSkipped _ -> "SKIP"
let runStatusStr = function RunSuccess _ -> "PASS" | RunFailed _ -> "FAIL" | OutputMismatch _ -> "MISMATCH" | RunTimeout _ -> "TIMEOUT" | RunSkipped _ -> "SKIP"

let generateReport (report: TestReport) verbose =
    let sb = StringBuilder()
    sb.AppendLine("=== Firefly Regression Test ===").AppendLine($"Run ID: {report.RunId}")
      .AppendLine($"Manifest: {report.ManifestPath}").AppendLine($"Compiler: {report.CompilerPath}").AppendLine() |> ignore
    sb.AppendLine("=== Compilation Phase ===") |> ignore
    for r in report.Results do
        let dur = match r.CompileResult with CompileSuccess ms | CompileFailed (_,_,ms) -> formatDuration ms | CompileTimeout ms -> $">{ms}ms" | CompileSkipped _ -> "-"
        let extra = match r.CompileResult with CompileSkipped reason -> $" ({reason})" | _ -> ""
        sb.AppendLine($"[{compileStatusStr r.CompileResult}] {r.Sample.Name} ({dur}){extra}") |> ignore
    sb.AppendLine().AppendLine("=== Execution Phase ===") |> ignore
    for r in report.Results do
        match r.RunResult with
        | Some rr ->
            let dur = match rr with RunSuccess ms | RunFailed (_,_,_,ms) | OutputMismatch (_,_,ms) -> formatDuration ms | RunTimeout ms -> $">{ms}ms" | RunSkipped _ -> "-"
            let extra = match rr with RunSkipped reason -> $" ({reason})" | OutputMismatch (e,a,_) -> "\n" + createDiffSummary e a 5 | _ -> ""
            sb.AppendLine($"[{runStatusStr rr}] {r.Sample.Name} ({dur}){extra}") |> ignore
        | None -> sb.AppendLine($"[SKIP] {r.Sample.Name} (compile failed)") |> ignore
    sb.AppendLine().AppendLine("=== Summary ===") |> ignore
    let startStr = report.StartTime.ToString("yyyy-MM-ddTHH:mm:ss")
    let endStr = report.EndTime.ToString("yyyy-MM-ddTHH:mm:ss")
    sb.AppendLine($"Started: {startStr}").AppendLine($"Completed: {endStr}") |> ignore
    let dur = (report.EndTime - report.StartTime).TotalSeconds
    sb.AppendLine($"Duration: {dur:F1}s") |> ignore
    let cPass = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSuccess _ -> true | _ -> false) |> List.length
    let cFail = report.Results |> List.filter (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false) |> List.length
    let cSkip = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSkipped _ -> true | _ -> false) |> List.length
    sb.AppendLine($"Compilation: {cPass}/{report.Results.Length} passed, {cFail} failed, {cSkip} skipped") |> ignore
    let runs = report.Results |> List.choose (fun r -> r.RunResult)
    let rPass = runs |> List.filter (function RunSuccess _ -> true | _ -> false) |> List.length
    let rFail = runs |> List.filter (function RunFailed _ | OutputMismatch _ | RunTimeout _ -> true | _ -> false) |> List.length
    let rSkip = runs |> List.filter (function RunSkipped _ -> true | _ -> false) |> List.length
    sb.AppendLine($"Execution: {rPass}/{runs.Length} passed, {rFail} failed, {rSkip} skipped") |> ignore
    let status = if cFail = 0 && rFail = 0 then "PASSED" else "FAILED"
    sb.AppendLine($"Status: {status}").ToString()

let didPass report =
    let cFail = report.Results |> List.filter (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false) |> List.length
    let rFail = report.Results |> List.choose (fun r -> r.RunResult) |> List.filter (function RunFailed _ | OutputMismatch _ | RunTimeout _ -> true | _ -> false) |> List.length
    cFail = 0 && rFail = 0

// =============================================================================
// Test Execution
// =============================================================================

let runSampleTest config sample verbose =
    let sampleDir = getSampleDir config sample
    let binaryPath = getBinaryPath config sample
    let stdinContent = getStdinContent config sample
    let timeoutMs = sample.TimeoutSeconds * 1000
    if verbose then printfn "  Testing: %s" sample.Name
    if sample.Skip then
        let reason = defaultArg sample.SkipReason "No reason"
        if verbose then printfn "    Skipped: %s" reason
        { Sample = sample; CompileResult = CompileSkipped reason; RunResult = Some (RunSkipped reason) }
    else
        if verbose then printfn "    Compiling..."
        let compileResult = compileSample config.CompilerPath sampleDir sample.ProjectFile timeoutMs
        match compileResult with
        | CompileSuccess ms ->
            if verbose then printfn "    Compiled in %dms, executing..." ms
            let (runResult, actual) = runBinary binaryPath sampleDir stdinContent timeoutMs
            match runResult with
            | RunSuccess ms ->
                if outputMatches sample.ExpectedOutput actual then
                    if verbose then printfn "    Passed in %dms" ms
                    { Sample = sample; CompileResult = compileResult; RunResult = Some (RunSuccess ms) }
                else
                    if verbose then printfn "    Output mismatch!"
                    { Sample = sample; CompileResult = compileResult; RunResult = Some (OutputMismatch (sample.ExpectedOutput, actual, ms)) }
            | other -> { Sample = sample; CompileResult = compileResult; RunResult = Some other }
        | CompileFailed (code, stderr, ms) ->
            if verbose then printfn "    Compile failed (exit %d)" code
            { Sample = sample; CompileResult = compileResult; RunResult = None }
        | CompileTimeout _ ->
            if verbose then printfn "    Compile timed out"
            { Sample = sample; CompileResult = compileResult; RunResult = None }
        | CompileSkipped reason -> { Sample = sample; CompileResult = compileResult; RunResult = Some (RunSkipped reason) }

let buildCompiler compilerDir verbose =
    if verbose then printfn "Building compiler..."
    let (result, ms) = runWithTimeoutMeasured "dotnet" "build" compilerDir None 120000
    match result with
    | Completed (0, _, _) ->
        if verbose then printfn "Built in %dms" ms
        true
    | Completed (code, _, stderr) ->
        printfn "Build failed (exit %d): %s" code (stderr.Substring(0, min 500 stderr.Length))
        false
    | Timeout _ ->
        printfn "Build timed out"
        false
    | Failed ex ->
        printfn "Build exception: %s" ex.Message
        false

// =============================================================================
// CLI and Main
// =============================================================================

let defaultOptions = { ManifestPath = Path.Combine(__SOURCE_DIRECTORY__, "Manifest.toml"); TargetSample = None; Verbose = false; TimeoutOverride = None }

let rec parseArgs args opts =
    match args with
    | [] -> opts
    | "--sample" :: name :: rest -> parseArgs rest { opts with TargetSample = Some name }
    | "--verbose" :: rest -> parseArgs rest { opts with Verbose = true }
    | "--timeout" :: sec :: rest -> match Int32.TryParse(sec) with true, n -> parseArgs rest { opts with TimeoutOverride = Some n } | _ -> parseArgs rest opts
    | "--help" :: _ -> printfn "Usage: dotnet fsi Runner.fsx [--sample NAME] [--verbose] [--timeout SEC] [--help]"; exit 0
    | _ :: rest -> parseArgs rest opts

let main argv =
    let opts = parseArgs (Array.toList argv) defaultOptions
    printfn "=== Firefly Regression Test Runner ===\n"
    if not (File.Exists opts.ManifestPath) then printfn "ERROR: Manifest not found at %s" opts.ManifestPath; exit 1
    let (config, allSamples) = loadManifest opts.ManifestPath
    let samples = match opts.TimeoutOverride with Some t -> allSamples |> List.map (fun s -> { s with TimeoutSeconds = t }) | None -> allSamples
    let samplesToRun = match opts.TargetSample with
                       | Some name -> let f = samples |> List.filter (fun s -> s.Name = name)
                                      if f.IsEmpty then printfn "Sample '%s' not found" name; exit 1
                                      f
                       | None -> samples
    printfn "Manifest: %s\nCompiler: %s\nSamples: %d\n" opts.ManifestPath config.CompilerPath samplesToRun.Length
    let srcDir = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(config.CompilerPath), "..", "..", "..", ".."))
    if not (buildCompiler srcDir opts.Verbose) then exit 1
    printfn "\nRunning %d tests...\n" samplesToRun.Length
    let startTime = DateTime.Now
    let results = samplesToRun |> List.map (fun s -> runSampleTest config s opts.Verbose)
    let endTime = DateTime.Now
    let report = { RunId = startTime.ToString("yyyy-MM-ddTHH:mm:ss"); ManifestPath = opts.ManifestPath; CompilerPath = config.CompilerPath; StartTime = startTime; EndTime = endTime; Results = results }
    printfn "\n%s" (generateReport report opts.Verbose)
    if didPass report then 0 else 1

exit (main (Environment.GetCommandLineArgs() |> Array.skip 2))
