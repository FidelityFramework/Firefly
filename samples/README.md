# Firefly Sample Suite

This directory contains sample projects demonstrating Firefly's capabilities for compiling F# to native code. The samples serve as both documentation and regression tests for the compiler, progressing toward the **WREN Stack Alpha** capstone.

> **WREN** = **W**ebView + **R**eactive + **E**mbedded + **N**ative

## The Sample Progression

The `console/FidelityHelloWorld/` directory contains a carefully designed progression. Each sample builds on the previous, proving specific compiler capabilities. The same output can require dramatically different compilation complexity: Sample 01 and Sample 04 both print "Hello, World!" but Sample 04 requires closure creation, capture analysis, and escape analysis.

### Complete Roadmap (Samples 01-31)

| Phase | # | Sample | Key Features | PRD | Status |
|-------|---|--------|--------------|-----|--------|
| **A: Foundations** | 01 | HelloWorldDirect | Static strings, direct calls | — | ✓ |
| | 02 | HelloWorldSaturated | Arena allocation, byref | — | ✓ |
| | 03 | HelloWorldHalfCurried | Pipe operator, forward refs | — | ✓ |
| | 04 | HelloWorldFullCurried | Full currying, partial application | — | ✓ |
| | 05 | AddNumbers | Discriminated unions, pattern matching | — | ✓ |
| | 06 | AddNumbersInteractive | String parsing, arithmetic | — | ✓ |
| | 07 | BitsTest | Byte order, bit casting intrinsics | — | ✓ |
| | 08 | Option | Option type, Some/None | — | ✓ |
| | 09 | Result | Result type, Ok/Error | — | ✓ |
| | 10 | Records | Record types, copy-update, nesting | — | ✓ |
| **B: Functional** | 11 | Closures | Lambdas, capture analysis, mutable state | [PRD-11](docs/WREN_Stack_PRDs/PRD-11-Closures.md) | ✓ |
| | 12 | HigherOrderFunctions | Functions as values, composition | [PRD-12](docs/WREN_Stack_PRDs/PRD-12-HigherOrderFunctions.md) | ✓ |
| | 13 | Recursion | Tail recursion, mutual recursion | [PRD-13](docs/WREN_Stack_PRDs/PRD-13-Recursion.md) | ✓ |
| **C: Lazy/Seq** | 14 | Lazy | `lazy { }`, `Lazy.force`, flat closures | [PRD-14](docs/WREN_Stack_PRDs/PRD-14-Lazy.md) | ✓ |
| | 15 | SimpleSeq | `seq { }`, `yield`, state machines | [PRD-15](docs/WREN_Stack_PRDs/PRD-15-SimpleSeq.md) | ✓ |
| | 16 | SeqOperations | Seq.map, filter, fold, collect | [PRD-16](docs/WREN_Stack_PRDs/PRD-16-SeqOperations.md) | ✓ |
| **D: Async** | 17 | BasicAsync | `async { return }`, LLVM coroutines | [PRD-17](docs/WREN_Stack_PRDs/PRD-17-BasicAsync.md) | · |
| | 18 | AsyncAwait | `let!`, suspension coeffects | [PRD-18](docs/WREN_Stack_PRDs/PRD-18-AsyncAwait.md) | · |
| | 19 | AsyncParallel | `Async.Parallel` composition | [PRD-19](docs/WREN_Stack_PRDs/PRD-19-AsyncParallel.md) | · |
| **E: Regions** | 20 | BasicRegion | Region alloc/dispose, `NeedsCleanup` | [PRD-20](docs/WREN_Stack_PRDs/PRD-20-BasicRegion.md) | · |
| | 21 | RegionPassing | Region parameters, `BorrowedRegion` | [PRD-21](docs/WREN_Stack_PRDs/PRD-21-RegionPassing.md) | · |
| | 22 | RegionEscape | Escape analysis, `CopyOut` | [PRD-22](docs/WREN_Stack_PRDs/PRD-22-RegionEscape.md) | · |
| **F: Networking** | 23 | SocketBasics | TCP via `Sys.*` intrinsics | [PRD-23](docs/WREN_Stack_PRDs/PRD-23-SocketBasics.md) | · |
| | 24 | WebSocketEcho | WebSocket protocol | [PRD-24](docs/WREN_Stack_PRDs/PRD-24-WebSocketEcho.md) | · |
| **G: Desktop** | 25 | GTKWindow | GTK FFI, `ExternCall` | [PRD-25](docs/WREN_Stack_PRDs/PRD-25-GTKWindow.md) | · |
| | 26 | WebViewBasic | WebKitGTK WebView | [PRD-26](docs/WREN_Stack_PRDs/PRD-26-WebViewBasic.md) | · |
| **H: Threading** | 27 | BasicThread | `Thread.create`/`join` | [PRD-27](docs/WREN_Stack_PRDs/PRD-27-BasicThread.md) | · |
| | 28 | MutexSync | Mutex, `SyncPrimitive` | [PRD-28](docs/WREN_Stack_PRDs/PRD-28-MutexSync.md) | · |
| **I: Capstone** | 29 | BasicActor | `MailboxProcessor.Start` | [PRD-29](docs/WREN_Stack_PRDs/PRD-29-BasicActor.md) | · |
| | 30 | ActorReply | `PostAndReply` pattern | [PRD-30](docs/WREN_Stack_PRDs/PRD-30-ActorReply.md) | · |
| | 31 | ParallelActors | Multi-actor with regions | [PRD-31](docs/WREN_Stack_PRDs/PRD-31-ParallelActors.md) | · |

**Legend**: ✓ = Implemented · = Planned

### The Capstone: MailboxProcessor

Samples 29-31 synthesize all prior capabilities: async (message loop), closures (behavior functions), threading (parallelism), regions (worker memory), and records/DUs (message types). With Sample 31 complete, the WREN Stack Alpha becomes possible.

## Building and Running

### Single Sample

```bash
cd samples/console/FidelityHelloWorld/01_HelloWorldDirect
firefly compile HelloWorld.fidproj
./target/helloworld
```

### With Intermediate Files

```bash
firefly compile HelloWorld.fidproj -k
ls target/intermediates/
# fncs_phase_*.json, alex_coeffects.json, *.mlir, *.ll
```

### Interactive Samples

Samples 02-04 and 06 require input. Each has a `.stdin` file:

```bash
./target/helloworld < HelloWorld.stdin
```

## Regression Test Suite

The test harness lives in `tests/regression/`.

```bash
cd tests/regression

# Full suite
dotnet fsi Runner.fsx

# Parallel execution
dotnet fsi Runner.fsx -- --parallel

# Specific samples
dotnet fsi Runner.fsx -- --sample 11_Closures --sample 14_Lazy
```

Test definitions are in `Manifest.toml`. The runner reports compilation and execution status for each sample.

## Directory Structure

```
samples/
├── console/
│   ├── FidelityHelloWorld/     # Canonical progression (01-31)
│   ├── TimeLoop/               # Platform time operations
│   └── SignalTest/             # Reactive signals
├── embedded/                   # ARM microcontroller targets
│   ├── common/                 # Startup and linker scripts
│   ├── stm32l5-blinky/         # NUCLEO-L552ZE-Q
│   └── stm32l5-uart/           # Serial communication
├── sbc/                        # Single-board computers
│   └── sweet-potato-blinky/    # Libre Sweet Potato (ARM64)
├── templates/                  # Platform configurations
└── samples.json                # Sample catalog
```

## Related Documentation

- [WRENStack_Roadmap.md](docs/WRENStack_Roadmap.md) - Architecture and milestones
- [WREN_Stack_PRDs/](docs/WREN_Stack_PRDs/) - PRD index and all feature specs
- [Learning to Walk](https://speakez.com/blog/learning-to-walk/) - PSG traversal via samples
- [Gaining Closure](https://speakez.com/blog/gaining-closure/) - Flat closure architecture
- [Why Lazy Is Hard](https://speakez.com/blog/why-lazy-is-hard/) - Lazy evaluation
- [Seq'ing Simplicity](https://speakez.com/blog/seqing-simplicity/) - Sequence expressions

## License

MIT License - See [LICENSE](../LICENSE) for details.
