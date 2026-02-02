/// SequentialWitness - DEPRECATED (January 2026)
///
/// ARCHITECTURAL DECISION: Sequential is NOT a witness.
///
/// Sequential nodes are structural scaffolding that organize the PSG tree.
/// They don't observe coeffects, emit MLIR, or produce values.
/// They only "forward" child results, which is composition/building, NOT witnessing.
///
/// VIOLATION: This pattern violated the codata photographer principle.
/// Witnesses should observe pre-computed data, not forward/build results.
///
/// FIX: LambdaWitness now traverses Sequential structure directly using
/// findLastValueNode to extract the last value-producing child.
///
/// This file is kept as a stub for documentation purposes.
/// Sequential is NOT registered in WitnessRegistry.
module Alex.Witnesses.SequentialWitness

open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture

/// DEPRECATED: Sequential witness removed - see module documentation
let nanopass : Nanopass = {
    Name = "Sequential-DEPRECATED"
    Witness = fun _ _ -> WitnessOutput.skip
}
