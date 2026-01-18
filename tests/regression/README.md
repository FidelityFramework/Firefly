# Firefly Regression Test Harness

A standalone F# script-based regression test infrastructure for validating Firefly compiler correctness across the FidelityHelloWorld sample suite.

## Quick Start

```bash
cd /home/hhh/repos/Firefly/tests/regression

# Run full test suite
dotnet fsi Runner.fsx

# Run full test suite in parallel (faster)
dotnet fsi Runner.fsx -- --parallel

# Run specific sample(s)
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect
dotnet fsi Runner.fsx -- --sample 07_BitsTest --sample 14_Lazy

# Run with verbose output
dotnet fsi Runner.fsx -- --verbose

# Run with custom timeout (seconds)
dotnet fsi Runner.fsx -- --timeout 60
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--sample NAME` | Run specific sample(s). Can be repeated for multiple samples. |
| `--verbose` | Show detailed output including compile errors and diff details. |
| `--parallel` | Run samples in parallel on separate threads. |
| `--timeout SEC` | Override the default timeout for all samples. |
| `--help` | Show help message. |

## How It Works

1. **Single Build**: The harness builds the Firefly compiler once at startup (which pulls in fsnative if needed).
2. **Sample Discovery**: Reads `Manifest.toml` for sample definitions and expected outputs.
3. **Compilation Phase**: Compiles each sample using the built compiler.
4. **Execution Phase**: Runs successfully compiled binaries and compares output.
5. **Reporting**: Generates a summary report with pass/fail status.

## Manifest Format

The `Manifest.toml` file defines all samples:

```toml
[config]
samples_root = "../../samples/console/FidelityHelloWorld"
compiler = "../../src/bin/Debug/net10.0/Firefly"
default_timeout_seconds = 30

[[samples]]
name = "01_HelloWorldDirect"
project = "HelloWorld.fidproj"
binary = "target/helloworld"
expected_output = """
Hello, World!
"""

[[samples]]
name = "02_HelloWorldSaturated"
project = "HelloWorld.fidproj"
binary = "target/helloworld"
stdin_file = "HelloWorld.stdin"    # Optional: provide input
expected_output = """
Enter your name: Hello, Houston!
"""

[[samples]]
name = "16_SeqOperations"
project = "SeqOperations.fidproj"
binary = "target/SeqOperations"
skip = true                        # Skip this sample
skip_reason = "PRD-16 not yet implemented"
expected_output = ""
```

## Output Format

```
=== Firefly Regression Test ===
Run ID: 2026-01-18T15:47:30
Manifest: /home/hhh/repos/Firefly/tests/regression/Manifest.toml
Compiler: /home/hhh/repos/Firefly/src/bin/Debug/net10.0/Firefly

=== Compilation Phase ===
[PASS] 01_HelloWorldDirect (922ms)
[PASS] 02_HelloWorldSaturated (942ms)
[FAIL] 07_BitsTest (725ms)
[SKIP] 16_SeqOperations (-) (PRD-16 not yet implemented)

=== Execution Phase ===
[PASS] 01_HelloWorldDirect (29ms)
[MISMATCH] 05_AddNumbers (28ms)
  First diff at line 3:
    Expected: FloatVal 3.14 -> 3.14
    Actual:   FloatVal 3.14 -> 3.140000
[SKIP] 07_BitsTest (compile failed)

=== Summary ===
Started: 2026-01-18T15:47:30
Completed: 2026-01-18T15:47:44
Duration: 14.3s
Compilation: 13/16 passed, 2 failed, 1 skipped
Execution: 13/14 passed, 0 failed, 1 skipped
Status: FAILED
```

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

## Files

| File | Purpose |
|------|---------|
| `Runner.fsx` | Main test runner script |
| `Manifest.toml` | Sample definitions and expected outputs |
| `README.md` | This documentation |

## Adding New Samples

1. Add the sample to `Manifest.toml` with:
   - `name`: Directory name under samples_root
   - `project`: The .fidproj file name
   - `binary`: Path to output binary (usually `target/<name>`)
   - `expected_output`: Expected stdout (use `"""` for multiline)
   - Optional: `stdin_file` for samples needing input
   - Optional: `timeout_seconds` for samples needing more time

2. If the sample needs stdin input, create a `.stdin` file in the sample directory.

## Troubleshooting

**"Manifest not found"**: Run from the `tests/regression/` directory.

**Build failures**: Check that fsnative is built and the compiler path in Manifest.toml is correct.

**Output mismatches**: Use `--verbose` to see the full expected vs actual diff.
