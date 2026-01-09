# Firefly Memory Audit - January 2026

## Executive Summary

Following the successful implementation of the Platform Binding Resolution nanopass, the Firefly codebase now has a principled architecture. This audit identifies which Serena memories are:
- **Canonical** - Core architectural truths to maintain
- **Redundant** - Overlapping with canonical docs
- **Historical** - Superseded by architectural evolution
- **Transitory** - Incremental progress docs no longer needed

## Architectural Reality (January 2026)

The Firefly compilation pipeline is now:

```
F# Source
    ↓
FNCS (types, intrinsics, PSG construction)
    ↓
Alex Preprocessing Nanopasses:
    - SSAAssignment
    - MutabilityAnalysis
    - PatternBindingAnalysis
    - PlatformBindingResolution  ← NEW (freestanding vs console)
    ↓
Coeffects (pre-computed, read-only)
    ↓
Zipper/Witness Traversal (observation only)
    ↓
MLIR → LLVM → Native Binary
```

## Memory Classification

### TIER 1: CANONICAL ARCHITECTURAL (7 memories - KEEP)

These document the current, correct architecture:

| Memory | Purpose | Status |
|--------|---------|--------|
| `architecture_principles` | Master architectural document | ✅ Current |
| `fncs_architecture` | FNCS design and intrinsics | ✅ Current |
| `ntu_architecture` | Native Type Universe | ✅ Current |
| `codata_photographer_principle` | Core emission pattern | ✅ Current |
| `four_pillars_of_transfer` | Coeffects, Templates, Zipper, XParsec | ✅ Current |
| `negative_examples` | Mistakes to avoid | ✅ Current (valuable) |
| `platform_binding_nanopass_architecture` | Platform resolution nanopass | ✅ NEW |

### TIER 2: ACTIVE PROJECT-SPECIFIC (2 memories - KEEP)

| Memory | Purpose | Status |
|--------|---------|--------|
| `wren_stack_architecture` | WREN desktop architecture | Keep if project active |
| `fidelity_signal_architecture` | Signal/event architecture | Keep if project active |

### TIER 3: REDUNDANT/SUPERSEDED (DELETE)

These overlap with canonical docs or are superseded:

| Memory | Superseded By | Action |
|--------|---------------|--------|
| `alex_zipper_architecture` | `architecture_principles` | DELETE |
| `binding_architecture_unified` | `platform_binding_nanopass_architecture` | DELETE |
| `platform_binding_witnessing` | `platform_binding_nanopass_architecture` | DELETE |
| `fncstransfer_architecture` | `architecture_principles` | DELETE |
| `fncstransfer_remediation` | Completed, superseded | DELETE |
| `baker_component` | FNCS absorbed Baker | DELETE |
| `baker_implementation_status` | FNCS absorbed Baker | DELETE |
| `nanopass_pipeline` | `architecture_principles` | DELETE |
| `static_dynamic_binding_strategy` | `platform_binding_nanopass_architecture` | DELETE |

### TIER 4: HISTORICAL ARCHIVE (DELETE)

Alloy was absorbed into FNCS (January 2026). These are historical:

| Memory | Reason | Action |
|--------|--------|--------|
| `alloy_architecture_and_role` | Already marked historical | DELETE |
| `alloy_integration` | Alloy absorbed | DELETE |
| `alloy_bcl_elimination_complete` | Alloy absorbed | DELETE |

### TIER 5: TRANSITORY/PROGRESS (DELETE)

These documented incremental progress, no longer needed:

| Memory | Reason |
|--------|--------|
| `compilation_timing_infrastructure` | Implementation detail |
| `pipeline_debugging` | Debugging notes |
| `task_completion_checklist` | Task management |
| `suggested_commands` | Operational, not architectural |
| `key_files` | Can be derived from code |
| `codebase_structure` | Outdated |
| `external_references` | Reference materials, not architecture |
| `reference_resources` | Reference materials |
| `fsnative_setup` | Setup notes |
| `style_and_conventions` | Operational |
| `binding_inspection_workflow` | Workflow notes |

### TIER 6: SPECIALIZED (REVIEW)

These may be valuable but need review:

| Memory | Notes |
|--------|-------|
| `delimited_continuations_architecture` | Future feature, keep if active |
| `farscape_maturation_plan` | Future feature, keep if active |
| `barewire_hardware_integration` | Future feature, keep if active |
| `fsharp_metaprogramming_patterns` | General patterns, may consolidate |
| `srtp_resolution_findings` | May be valuable for FNCS |

## Docs Folder Cleanup

### Current State: 40+ files

### Target State: ~10 essential documents

**KEEP (Essential):**
- `Architecture_Canonical.md` - Master architecture doc
- `FNCS_Architecture.md` - FNCS design
- `NTU_Architecture.md` - Native types
- `Platform_Binding_Model.md` - Platform bindings
- `WREN_STACK.md` / `WREN_TECH_SPEC.md` - If WREN active

**ARCHIVE (Move to docs/archive/):**
- Roadmap documents (outdated)
- Case studies (reference material)
- Integration guides for specific features
- YoshiPi hardware-specific docs
- QC_Demo directory
- WebView documents (if WREN inactive)

**DELETE (Superseded):**
- `PSG_to_MLIR_Emission_Design.md` - Superseded
- Duplicate architecture docs

## Recommended Actions

### Phase 1: Memory Cleanup (Firefly)
```bash
# Delete superseded memories
mcp__serena__delete_memory "alex_zipper_architecture"
mcp__serena__delete_memory "binding_architecture_unified"
mcp__serena__delete_memory "platform_binding_witnessing"
mcp__serena__delete_memory "fncstransfer_architecture"
# ... etc
```

### Phase 2: Docs Consolidation
1. Move historical docs to `docs/archive/`
2. Update `Architecture_Canonical.md` as master reference
3. Delete redundant/superseded docs

### Phase 3: fsnative Memory Audit
Similar classification for fsnative memories.

### Phase 4: Spec Alignment
Verify fsnative-spec reflects the implemented architecture.

## Memory Count Summary

| Category | Count | Action |
|----------|-------|--------|
| Canonical | 7 | Keep |
| Project-Specific | 2 | Keep if active |
| Redundant | 9 | Delete |
| Historical | 3 | Delete |
| Transitory | 11+ | Delete |
| Specialized | 5+ | Review |

**Target: ~15 focused memories (down from 67)**
