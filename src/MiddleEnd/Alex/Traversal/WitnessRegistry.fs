/// WitnessRegistry - Global registry of all witness nanopasses
///
/// Each witness module exports a `nanopass` value. This registry collects all
/// witnesses into a single registry for parallel execution.
///
/// MIGRATION STATUS: Witnesses are being migrated incrementally to nanopass pattern.
/// As each witness is migrated, uncomment its registration below.
module Alex.Traversal.WitnessRegistry

open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.TransferTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS MODULE IMPORTS
// ═══════════════════════════════════════════════════════════════════════════

// Import all witness modules
// As witnesses are migrated to export `nanopass`, they're imported here

// Priority 1: Simple Witnesses
module LiteralWitness = Alex.Witnesses.LiteralWitness
module ArithWitness = Alex.Witnesses.ArithWitness

// Priority 2: Collection Witnesses
// module OptionWitness = Alex.Witnesses.OptionWitness
// module ListWitness = Alex.Witnesses.ListWitness
// module MapWitness = Alex.Witnesses.MapWitness
// module SetWitness = Alex.Witnesses.SetWitness

// Priority 3: Control Flow (special - needs nanopass list for sub-graph traversal)
module ControlFlowWitness = Alex.Witnesses.ControlFlowWitness

// Priority 4: Memory & Lambda
// module MemoryWitness = Alex.Witnesses.MemoryWitness
// module LambdaWitness = Alex.Witnesses.LambdaWitness

// Priority 5: Advanced Features
module LazyWitness = Alex.Witnesses.LazyWitness
// module SeqWitness = Alex.Witnesses.SeqWitness

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

/// Global registry of all witness nanopasses
/// MIGRATION: Currently empty. As witnesses are migrated, register them here.
let mutable globalRegistry = NanopassRegistry.empty

/// Initialize the global registry
/// Called once at startup to populate the registry with all witness nanopasses
let initializeRegistry () =
    // First register leaf witnesses (literals, arithmetic, memory, etc.)
    let leafRegistry =
        NanopassRegistry.empty
        // Priority 1: Simple Witnesses
        |> NanopassRegistry.register LiteralWitness.nanopass
        |> NanopassRegistry.register ArithWitness.nanopass

        // Priority 2: Collection Witnesses
        // |> NanopassRegistry.register OptionWitness.nanopass
        // |> NanopassRegistry.register ListWitness.nanopass
        // |> NanopassRegistry.register MapWitness.nanopass
        // |> NanopassRegistry.register SetWitness.nanopass

        // Priority 4: Memory & Lambda
        // |> NanopassRegistry.register MemoryWitness.nanopass
        // |> NanopassRegistry.register LambdaWitness.nanopass

        // Priority 5: Advanced Features
        |> NanopassRegistry.register LazyWitness.nanopass
        // |> NanopassRegistry.register SeqWitness.nanopass

    // Now create ControlFlowWitness with the leaf nanopass list for sub-graph traversal
    // ControlFlowWitness needs access to ALL witnesses (including itself) for recursive sub-graphs
    let mutable finalRegistry = leafRegistry
    let controlFlowNanopass = ControlFlowWitness.createNanopass (leafRegistry.Nanopasses)
    finalRegistry <- NanopassRegistry.register controlFlowNanopass finalRegistry

    // Re-create ControlFlowWitness with FULL registry including itself (for recursive control flow)
    let controlFlowNanopassRecursive = ControlFlowWitness.createNanopass (finalRegistry.Nanopasses)
    finalRegistry <- { finalRegistry with
                        Nanopasses = finalRegistry.Nanopasses
                                     |> List.filter (fun np -> np.Name <> "ControlFlow")
                                     |> List.append [controlFlowNanopassRecursive] }

    globalRegistry <- finalRegistry

// ═══════════════════════════════════════════════════════════════════════════
// MIGRATION NOTES
// ═══════════════════════════════════════════════════════════════════════════

/// MIGRATION CHECKLIST:
///
/// For each witness file:
/// 1. Add category-selective witness function (match node.Kind, return skip for others)
/// 2. Export `let nanopass : Nanopass = { Name = "..."; Witness = witness... }`
/// 3. Uncomment the module import above
/// 4. Uncomment the registry registration in initializeRegistry()
/// 5. Test in isolation
/// 6. Verify parallel = sequential output
///
/// See: docs/Witness_Migration_Guide.md for full migration process
