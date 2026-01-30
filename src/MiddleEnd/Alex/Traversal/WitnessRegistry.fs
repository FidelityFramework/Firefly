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
module TypeAnnotationWitness = Alex.Witnesses.TypeAnnotationWitness
module PlatformWitness = Alex.Witnesses.PlatformWitness

// Structural Witnesses (January 2026 - parallel fan-out)
module BindingWitness = Alex.Witnesses.BindingWitness
module VarRefWitness = Alex.Witnesses.VarRefWitness
module SequentialWitness = Alex.Witnesses.SequentialWitness
module ApplicationWitness = Alex.Witnesses.ApplicationWitness

// Priority 2: Collection Witnesses
// module OptionWitness = Alex.Witnesses.OptionWitness
// module ListWitness = Alex.Witnesses.ListWitness
// module MapWitness = Alex.Witnesses.MapWitness
// module SetWitness = Alex.Witnesses.SetWitness

// Priority 3: Control Flow (special - needs nanopass list for sub-graph traversal)
module ControlFlowWitness = Alex.Witnesses.ControlFlowWitness

// Priority 4: Memory & Lambda (Lambda is special - needs nanopass list for body witnessing)
module MemoryWitness = Alex.Witnesses.MemoryWitness
module LambdaWitness = Alex.Witnesses.LambdaWitness

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
    // These witnesses don't need sub-graph traversal
    let leafRegistry =
        NanopassRegistry.empty
        // Priority 1: Simple Witnesses
        |> NanopassRegistry.register LiteralWitness.nanopass
        |> NanopassRegistry.register TypeAnnotationWitness.nanopass
        |> NanopassRegistry.register PlatformWitness.nanopass

        // Structural Witnesses (parallel fan-out)
        |> NanopassRegistry.register BindingWitness.nanopass
        |> NanopassRegistry.register VarRefWitness.nanopass
        |> NanopassRegistry.register SequentialWitness.nanopass
        |> NanopassRegistry.register ApplicationWitness.nanopass

        // Priority 2: Collection Witnesses
        // |> NanopassRegistry.register OptionWitness.nanopass
        // |> NanopassRegistry.register ListWitness.nanopass
        // |> NanopassRegistry.register MapWitness.nanopass
        // |> NanopassRegistry.register SetWitness.nanopass

        // Priority 4: Memory
        |> NanopassRegistry.register MemoryWitness.nanopass

        // Priority 5: Advanced Features
        |> NanopassRegistry.register LazyWitness.nanopass
        // |> NanopassRegistry.register SeqWitness.nanopass

    // Now create composite witnesses (Lambda, ControlFlow) that need sub-graph traversal
    // These witnesses need access to ALL other witnesses for witnessing sub-graphs
    let mutable workingRegistry = leafRegistry

    // Add LambdaWitness (needs to witness function bodies)
    let lambdaNanopass = LambdaWitness.createNanopass (workingRegistry.Nanopasses)
    workingRegistry <- NanopassRegistry.register lambdaNanopass workingRegistry

    // Add ControlFlowWitness (needs to witness branch bodies)
    let controlFlowNanopass = ControlFlowWitness.createNanopass (workingRegistry.Nanopasses)
    workingRegistry <- NanopassRegistry.register controlFlowNanopass workingRegistry

    // Re-create composite witnesses with FULL registry (including themselves for recursion)
    let lambdaNanopassRecursive = LambdaWitness.createNanopass (workingRegistry.Nanopasses)
    let controlFlowNanopassRecursive = ControlFlowWitness.createNanopass (workingRegistry.Nanopasses)

    let finalRegistry = { workingRegistry with
                            Nanopasses = workingRegistry.Nanopasses
                                         |> List.filter (fun np -> np.Name <> "Lambda" && np.Name <> "ControlFlow")
                                         |> List.append [lambdaNanopassRecursive; controlFlowNanopassRecursive] }

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
