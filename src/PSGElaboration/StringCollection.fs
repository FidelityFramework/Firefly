/// StringCollection - Pre-collect string literals from PSG
///
/// ARCHITECTURAL PRINCIPLE (Codata/Coeffect Model):
///
/// The StringTable is a COEFFECT - contextual information that flows INTO
/// witness computations. It is NOT queried or looked up - it is PRESENT.
///
/// String global names are DERIVED via pure functions, not looked up:
///   content → hash → @str_<hash>
///
/// The StringTable's purpose:
/// - Tells us WHICH strings exist in the program (for emitting globals)
/// - Carries pre-computed byte lengths (computed once during collection)
///
/// Witnesses use `deriveGlobalRef` (pure derivation), not "lookup".
/// The zipper is PASSIVE - it observes and produces, never mutates.
module PSGElaboration.StringCollection

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// STRING TABLE TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// A collected string literal
type StringEntry = {
    /// The string content
    Content: string
    /// UTF-8 byte length (for MLIR array sizing)
    ByteLength: int
    /// The global reference name (@str_<hash>)
    GlobalName: string
}

/// Pre-collected string table from PSG analysis
/// Key is hash (uint32), value is the string entry
type StringTable = Map<uint32, StringEntry>

// ═══════════════════════════════════════════════════════════════════════════
// PURE DERIVATION (deterministic - no state, no lookup)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute hash for a string literal
let private hashString (s: string) : uint32 =
    uint32 (s.GetHashCode())

/// Build global name from hash
let private globalNameFromHash (hash: uint32) : string =
    sprintf "@str_%u" hash

/// Derive the global reference name for a string literal
/// PURE FUNCTION: content → @str_<hash>
/// Witnesses use this directly - no lookup needed
let deriveGlobalRef (content: string) : string =
    globalNameFromHash (hashString content)

/// Derive the byte length for a string literal (UTF-8 encoding)
/// PURE FUNCTION: used by witnesses when emitting string operations
let deriveByteLength (content: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(content)

// ═══════════════════════════════════════════════════════════════════════════
// STRING COLLECTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Add a string to the table if not already present
let private addString (s: string) (table: StringTable) : StringTable =
    let hash = hashString s
    if Map.containsKey hash table then
        table
    else
        let byteLen = System.Text.Encoding.UTF8.GetByteCount(s)
        let entry = {
            Content = s
            ByteLength = byteLen
            GlobalName = globalNameFromHash hash
        }
        Map.add hash entry table

/// Collect strings from a single node
let private collectFromNode (node: SemanticNode) (table: StringTable) : StringTable =
    match node.Kind with
    | SemanticKind.Literal (LiteralValue.String s) ->
        addString s table
    
    | SemanticKind.InterpolatedString parts ->
        // Collect all StringPart entries
        parts
        |> List.fold (fun acc part ->
            match part with
            | InterpolatedPart.StringPart s -> addString s acc
            | InterpolatedPart.ExprPart _ -> acc) table
    
    | _ -> table

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Collect all string literals from the semantic graph
/// This is a preprocessing pass that runs ONCE before zipper traversal
let collect (graph: SemanticGraph) : StringTable =
    // Walk all nodes and collect strings
    graph.Nodes.Values
    |> Seq.fold (fun table node -> collectFromNode node table) Map.empty

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESS (StringTable tells us WHAT exists for global emission)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert table to list for global emission (hash, content, byteLength)
/// Used at serialization time to emit all string globals
/// The StringTable coeffect answers: "what strings need globals?"
let toList (table: StringTable) : (uint32 * string * int) list =
    table
    |> Map.toList
    |> List.map (fun (hash, entry) -> (hash, entry.Content, entry.ByteLength))
