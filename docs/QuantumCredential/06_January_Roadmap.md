# Demo Day Roadmap: January 2025 (6 Weeks)

This document provides a comprehensive assessment of the Fidelity ecosystem and a realistic roadmap for the late January demo day.

## Current State Assessment

### Firefly Compiler (Core)

**Status: Functional for basic programs**

| Component | State | Notes |
|-----------|-------|-------|
| FCS Integration | ✅ Working | Type checking, symbol resolution functional |
| PSG Builder | ✅ Working | 11K+ nodes, 15K+ edges for HelloWorld |
| Nanopasses | ✅ Working | DefUse, ClassifyOps, ParameterAnnotation |
| Reachability | ✅ Working | 97%+ dead code elimination |
| Alex/Zipper | ✅ Working | PSG traversal and MLIR generation |
| MLIR Generation | ✅ Working | Generates valid MLIR |
| LLVM Lowering | ✅ Working | mlir-opt → llc → clang pipeline |
| Native Binary | ✅ Working | Freestanding executables run |

**Working Samples:**
- `01_HelloWorldDirect` - ✅ Compiles and runs "Hello, World!"
- `02_HelloWorldSaturated` - ✅ Compiles (reachability works, MLIR generation partial)
- `03_HelloWorldHalfCurried` - ⚠️ Function inlining works, but needs parameter binding, while loops, NativePtr ops
- `04_HelloWorldFullCurried` - ⚠️ Untested this session (requires Result.map, lambdas, sprintf)
- `TimeLoop` - ❌ PSG Builder error: "Unhandled module declaration type 'Expr'"

**Known Issues:**
- **Parameter binding for inlined functions** - When functions are inlined, parameters need to be bound to argument values
- **While loop control flow** - Need cf.cond_br and basic blocks
- **NativePtr operations** - stackalloc, set, get not yet implemented
- **FSharp.Core operators** - `not`, comparison operators have no SymbolUse edges to resolve
- Symbol correlation warnings for operators (op_Addition, etc.)
- Member property correlation issues (this.Year, etc.)
- TimeLoop sample fails due to module-level expressions

**Recent Progress (December 2024):**
- **Function call inlining WORKING** - PSGScribe now follows SymbolUse edges to inline functions
- **Deep inlining works** - Console.ReadLine → readln → readLine → readLineInto all inline
- **Comparison operations implemented** - arith.cmpi with Eq, Neq, Lt, Gt, Lte, Gte
- **Arithmetic operations implemented** - arith.addi, subi, muli, divsi, remsi
- **Binding value tracking** - SSA values recorded per binding, lookup follows SymbolUse edges
- PSGScribe properly delegates to Bindings via ExternDispatch
- SSA values sequential (%v0 through %v17+)
- XParsec wrappers added for pattern matching

---

### FNCS Intrinsics

**Status: Core primitives working, expanding**

| Module | State | Notes |
|--------|-------|-------|
| Core.fs | ✅ Working | Basic utilities, ignore, id |
| Console.fs | ✅ Working | Write, WriteLine via Primitives |
| Primitives.fs | ✅ Working | writeBytes, readBytes, sleep, time primitives |
| Memory.fs | ✅ Structured | Memory operations defined |
| Math.fs | ✅ Structured | Mathematical functions |
| Time.fs | ✅ Structured | DateTime, TimeSpan types |
| Bits.fs, Pointer.fs | ✅ Structured | Bit operations, pointer utilities |
| Result.fs | ✅ Structured | Result type and combinators |
| String.fs | ✅ Structured | String operations |
| Text.fs | ✅ Structured | Text encoding/decoding |
| Numerics.fs | ✅ Structured | Numeric type operations |

**Primitives Defined (via `__fidelity` extern):**
- `fidelity_write_bytes` - Write to file descriptor
- `fidelity_read_bytes` - Read from file descriptor
- `fidelity_get_current_ticks` - System time
- `fidelity_get_monotonic_ticks` - High-res timer
- `fidelity_get_tick_frequency` - Timer frequency
- `fidelity_sleep` - Sleep milliseconds

**Missing for Demo Day:**
- GPIO primitives (for hardware targets)
- ADC sampling primitives
- IR transceiver primitives
- Crypto primitives (or Farscape bindings to PQC libs)

---

### Farscape (Binding Generator)

**Status: Proof-of-concept, C header parsing only**

| Component | State | Notes |
|-----------|-------|-------|
| C Header Parsing | ✅ Working | Via CppSharp/LibClang |
| Code Generation | ✅ Basic | Generates P/Invoke declarations |
| Type Mapping | ⚠️ Partial | String handling needs work |
| GIR Parsing | ❌ Not started | Required for GTK4 bindings |
| Functional Wrappers | ❌ Not started | Layer 3 generation |

**Current Capabilities:**
- Parse C headers via CppSharp
- Generate Layer 1 extern declarations
- Map basic C types to F# equivalents
- Proof-of-concept with cJSON library

**Required for Demo Day:**
- PQC library bindings (Kyber/Dilithium from pq-crystals or liboqs)
- Could be done manually if Farscape not ready

**Future (Post-Demo):**
- GIR parsing for GTK4 bindings
- Objective-C parsing for AppKit
- WinMD parsing for WinUI

---

### BAREWire (Binary Serialization)

**Status: Well-structured, .NET-dependent**

| Component | State | Notes |
|-----------|-------|-------|
| Schema Definition | ✅ Structured | Type-safe DSL |
| Encoder | ✅ Structured | BARE protocol encoding |
| Decoder | ✅ Structured | BARE protocol decoding |
| Memory Mapping | ✅ Structured | Zero-copy access |
| IPC | ✅ Structured | Shared memory, pipes |
| Network | ✅ Structured | Transport protocols |

**Assessment:**
BAREWire is designed as a .NET library with dependencies on System.Runtime.InteropServices and other BCL types. Porting to Fidelity would require:
- Replacing BCL memory types with FNCS NativeTypes
- Replacing Span<T> with NativeSpan
- Removing FSharp.UMX dependency or porting it

**For Demo Day:**
- BAREWire integration is likely **out of scope** for 6 weeks
- IR protocol could use simpler custom framing
- Consider BAREWire as "future integration" showcase

---

## Demo Day Targets

Based on the Hardware Showcase Roadmap and current state:

### Primary Demo: QuantumCredential (STM32L5)

**Goal:** Unikernel generating post-quantum credentials from hardware entropy

**Required Work:**

| Task | Complexity | Dependencies |
|------|------------|--------------|
| ARM Cortex-M33 LLVM target | Medium | LLVM toolchain setup |
| GPIO/ADC primitives in FNCS | Medium | Hardware docs |
| ADC sampling for zener entropy | Medium | Analog circuit |
| PQC library bindings (manual) | High | pq-crystals or pqm4 |
| Entropy validation | Medium | NIST SP 800-90B |
| Credential generation | Medium | Key format design |
| IR transmission | Medium | Protocol design |

**Risk Assessment:** HIGH - Many unknowns in embedded toolchain

### Secondary Demo: Keystation (Sweet Potato Linux App)

**Goal:** Linux application with touchscreen UI for credential management

**Required Work:**

| Task | Complexity | Dependencies |
|------|------------|--------------|
| AArch64 Linux target | Low | Already close to x86_64 |
| External ADC driver | Medium | I2C/SPI bindings |
| PQC library bindings | High | Same as QuantumCredential |
| LVGL or GTK4 UI | High | Farscape GIR or manual |
| IR reception | Medium | GPIO/protocol |
| Credential display | Medium | UI framework |

**Risk Assessment:** MEDIUM - Linux environment more familiar

### Fallback Demo: Desktop Console Applications

**Goal:** Demonstrate Firefly compilation pipeline working

**Already Working:**
- HelloWorld compiles and runs
- Console I/O functional
- Dead code elimination working

**Could Add:**
- More complex console apps
- Time operations (fix TimeLoop sample)
- User input handling
- Simple crypto operations

**Risk Assessment:** LOW - Foundation already working

---

## Six-Week Sprint Plan

### Week 1: Stabilization & ARM Target Setup

**Firefly:**
- [ ] Fix TimeLoop sample (module-level expression handling)
- [ ] Test samples 03 and 04
- [ ] Document current limitations

**Toolchain:**
- [ ] Set up ARM Cortex-M33 LLVM cross-compilation
- [ ] Create `thumbv8m.main-none-eabi` target configuration
- [ ] Test minimal ARM binary generation

**FNCS:**
- [ ] Review and test Time.fs primitives
- [ ] Ensure all current primitives have Alex bindings

### Week 2: ARM Platform Primitives

**FNCS:**
- [ ] Add GPIO read/write primitives
- [ ] Add ADC sampling primitive
- [ ] Add basic timer primitives

**Alex:**
- [ ] Add ARM syscall/register bindings
- [ ] Create STM32L5-specific memory map
- [ ] Test LED blink on hardware

**Hardware:**
- [ ] Verify STM32L5 Nucleo dev environment
- [ ] Test ADC with simple analog input
- [ ] Prototype zener noise circuit

### Week 3: Entropy & Basic PQC

**Entropy:**
- [ ] Implement ADC sampling loop
- [ ] Port basic health tests (repetition count, adaptive proportion)
- [ ] Implement SHA-3 or SHAKE conditioning
- [ ] Validate entropy quality

**PQC:**
- [ ] Manual bindings for Kyber-512 (or ML-KEM)
- [ ] Manual bindings for Dilithium-2 (or ML-DSA)
- [ ] Test key generation with seeded RNG
- [ ] Integrate hardware entropy with PQC

### Week 4: IR Communication & Sweet Potato

**IR Protocol:**
- [ ] Define simple framing protocol
- [ ] Implement 38kHz carrier generation (timer/PWM)
- [ ] Test basic data transmission
- [ ] Implement receive path

**Sweet Potato:**
- [ ] Set up AArch64 Linux target
- [ ] Port console samples to ARM Linux
- [ ] Test external ADC over I2C/SPI
- [ ] Basic GPIO for IR

### Week 5: Integration & UI

**Credential Flow:**
- [ ] Define credential format
- [ ] Implement key generation pipeline
- [ ] Implement signing operations
- [ ] Test credential exchange over IR

**UI (Stretch):**
- [ ] Evaluate LVGL vs GTK4 feasibility
- [ ] If feasible: basic credential display
- [ ] If not: console-based status output

### Week 6: Polish & Demo Prep

**Testing:**
- [ ] End-to-end credential generation test
- [ ] IR communication reliability test
- [ ] Error handling and edge cases

**Demo:**
- [ ] Prepare demo script
- [ ] Create fallback scenarios
- [ ] Document what works and what's stretch

---

## Critical Path

The **minimum viable demo** requires:

1. **ARM binary generation** - Without this, no hardware demo
2. **GPIO/ADC primitives** - Without this, no entropy
3. **PQC bindings** - Without this, no credentials

If any of these fail, fall back to:
- Desktop demo showing compilation pipeline
- Console app demonstrating Firefly capabilities
- Documentation of hardware architecture (designs without working code)

---

## Resource Allocation

### Highest Priority (Blocks Everything)

1. ARM Cortex-M33 target in LLVM pipeline
2. Basic GPIO/ADC primitives for STM32L5

### High Priority (Core Demo)

3. PQC library bindings (manual is fine)
4. Entropy sampling and validation
5. IR communication basics

### Medium Priority (Enhanced Demo)

6. Sweet Potato Linux port
7. Credential format and exchange
8. Error handling

### Lower Priority (Stretch Goals)

9. UI framework integration
10. BAREWire integration
11. Farscape GIR parser

---

## Realistic Expectations

**What Will Likely Work:**
- Desktop console applications
- Basic ARM binary generation
- GPIO LED blink on hardware
- Entropy sampling proof-of-concept

**What Might Work:**
- Full PQC key generation on hardware
- IR transmission of data
- Basic credential flow

**What Probably Won't Make It:**
- Full GTK4 UI on desktop
- BAREWire integration
- Production-quality entropy validation
- Full bidirectional credential exchange

**Demo Day Narrative:**
"We have built an F# ahead-of-time compiler that produces native binaries without runtime dependencies. Today we demonstrate it compiling F# code to run on an ARM microcontroller, sampling true random noise from a quantum mechanical source, and using post-quantum cryptographic algorithms to generate certificates that will remain secure even against future quantum computers."

Even a partial demo hitting these points would be compelling.

---

## References

- `docs/Hardware_Showcase_Roadmap.md` - Hardware platform details
- `docs/PostQuantum_Credential_Architecture.md` - PQC and entropy design
- `docs/Demo_UI_Stretch_Goal.md` - UI framework options
- `docs/Farscape_GIR_Integration.md` - Future binding generation
