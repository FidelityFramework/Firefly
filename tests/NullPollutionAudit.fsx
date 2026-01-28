#!/usr/bin/env dotnet fsi

/// Null Pollution Audit Script
///
/// Validates that zero null pointer references remain in the codebase.
/// This is a ZERO TOLERANCE check - any null reference is a failure.
///
/// Usage: dotnet fsi tests/NullPollutionAudit.fsx
/// Exit code: 0 = clean (zero nulls), 1 = pollution found

open System.IO
open System.Text.RegularExpressions

let mutable nullCount = 0
let mutable findings = []

/// Check a file for null references
let checkFile (filePath: string) (patterns: string list) =
    let content = File.ReadAllText(filePath)
    let lines = File.ReadAllLines(filePath)

    for pattern in patterns do
        let regex = Regex(pattern, RegexOptions.IgnoreCase)
        let matches = regex.Matches(content)

        if matches.Count > 0 then
            for m in matches do
                // Find line number
                let lineNum =
                    lines
                    |> Array.findIndex (fun line -> line.Contains(m.Value))
                    |> (+) 1

                findings <- (filePath, lineNum, m.Value) :: findings
                nullCount <- nullCount + 1

printfn "═══════════════════════════════════════════════════════════"
printfn "   NULL POLLUTION AUDIT"
printfn "═══════════════════════════════════════════════════════════"
printfn ""

// ═══════════════════════════════════════════════════════════════════════════
// CHECK PRDs
// ═══════════════════════════════════════════════════════════════════════════

printfn "Checking PRDs for null references..."

let prdFiles = Directory.GetFiles("docs/PRDs", "*.md", SearchOption.AllDirectories)
let prdPatterns = [
    @"\bnull\s+pointer\b"
    @"\bNullPtr\b"
    @"\bptr\s+null\b"
    @"→\s*null\b"  // Table cells with null
]

for prd in prdFiles do
    // C-05-Lazy.md is allowed to have null in WRONG examples
    if not (prd.Contains("C-05-Lazy.md")) then
        checkFile prd prdPatterns

// ═══════════════════════════════════════════════════════════════════════════
// CHECK CODE
// ═══════════════════════════════════════════════════════════════════════════

printfn "Checking source code for NullPtr emissions..."

let codeFiles = Directory.GetFiles("src", "*.fs", SearchOption.AllDirectories)
let codePatterns = [
    @"\bNullPtr\b"
    @"\bnullPtr\b"
    @"SizeNullPtrSSA"
]

for codeFile in codeFiles do
    checkFile codeFile codePatterns

// ═══════════════════════════════════════════════════════════════════════════
// CHECK DOCUMENTATION
// ═══════════════════════════════════════════════════════════════════════════

printfn "Checking documentation for null references..."

let docFiles =
    Directory.GetFiles("docs", "*.md", SearchOption.AllDirectories)
    |> Array.filter (fun f -> not (f.Contains("PRDs")))  // PRDs already checked

let docPatterns = [
    @"\bnull\s+pointer\b"
    @"\bNullPtr\b"
]

for doc in docFiles do
    // C-05 is allowed, Architecture docs discussing "null-free" are allowed
    if not (doc.Contains("C-05")) then
        let content = File.ReadAllText(doc)
        // Only flag if it's NOT in the context of "null-free semantics"
        let lines = File.ReadAllLines(doc)
        for i, line in lines |> Array.indexed do
            if line.Contains("null pointer") && not (line.Contains("null-free")) && not (line.Contains("never null")) then
                findings <- (doc, i + 1, "null pointer") :: findings
                nullCount <- nullCount + 1

// ═══════════════════════════════════════════════════════════════════════════
// REPORT RESULTS
// ═══════════════════════════════════════════════════════════════════════════

printfn ""
printfn "═══════════════════════════════════════════════════════════"

if nullCount = 0 then
    printfn "✅ NULL ERADICATION COMPLETE"
    printfn ""
    printfn "   Zero null references found across:"
    printfn "   - %d PRD files" prdFiles.Length
    printfn "   - %d source files" codeFiles.Length
    printfn "   - %d documentation files" docFiles.Length
    printfn ""
    printfn "   Architecture is null-free. ✓"
    printfn "═══════════════════════════════════════════════════════════"
    exit 0
else
    printfn "❌ NULL POLLUTION FOUND"
    printfn ""
    printfn "   %d null references detected:" nullCount
    printfn ""

    // Group by file
    findings
    |> List.groupBy (fun (file, _, _) -> file)
    |> List.iter (fun (file, items) ->
        let shortPath = file.Replace("/home/hhh/repos/Firefly/", "")
        printfn "   %s:" shortPath
        items
        |> List.sortBy (fun (_, line, _) -> line)
        |> List.iter (fun (_, line, text) ->
            printfn "      Line %d: %s" line text
        )
        printfn ""
    )

    printfn "   Fix these references before proceeding."
    printfn "═══════════════════════════════════════════════════════════"
    exit 1
