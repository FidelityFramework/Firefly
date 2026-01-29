/// Extensions - XParsec combinator extensions for Alex-specific composition patterns
///
/// These combinators are built using only XParsec's public API.
/// They extend XParsec for parser composition use cases (vs text parsing).
/// Can be upstreamed to XParsec when appropriate.
module Alex.XParsec.Extensions

open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// LIST SEQUENCING
// ═══════════════════════════════════════════════════════════

/// Sequence a list of parsers into a parser of a list
///
/// This combinator is not in core XParsec because XParsec targets text/byte
/// stream parsing, not dynamic parser composition. We need it for composing
/// MLIR operation chains where the list length is known at pattern time.
///
/// Built using only XParsec public API (parser { }, preturn, pattern matching).
let rec sequence (parsers: Parser<'a, 'T, 'State, 'Input, 'InputSlice> list)
                 : Parser<'a list, 'T, 'State, 'Input, 'InputSlice> =
    match parsers with
    | [] -> preturn []
    | p :: ps ->
        parser {
            let! x = p
            let! xs = sequence ps
            return x :: xs
        }

/// Sequence and flatten results - for nested op lists
let sequenceConcat (parsers: Parser<'a list, 'T, 'State, 'Input, 'InputSlice> list)
                   : Parser<'a list, 'T, 'State, 'Input, 'InputSlice> =
    parser {
        let! results = sequence parsers
        return List.concat results
    }
