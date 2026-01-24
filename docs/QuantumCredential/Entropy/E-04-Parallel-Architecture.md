# Parallel Entropy Architecture - Tactical Plan

## Strategic Shift

After proving single-channel entropy generation works (but is Python-bottlenecked at 1117ms for 4096 bytes), we're pivoting to:

1. **4 parallel avalanche circuits** using Chris Tacke's simpler design
2. **Native code generation** via fsnative/Fidelity for direct hardware access
3. **MLIR scf.parallel** for concurrent 4-channel ADC sampling

## Expected Performance Gains

| Optimization | Factor | Cumulative |
|--------------|--------|------------|
| Native vs Python | 10-20x | 10-20x |
| 4 parallel channels | 4x | 40-80x |
| DMA/burst reads | 2-5x | 80-400x |

**Conservative target**: 4096 bytes in **10-50ms** (vs 1117ms Python baseline)

## Phase 1: Hardware - 4x Parallel Circuits

### Chris Tacke's Simpler Design
From the video analysis, the key insight is:
- Don't amplify the noise - just read it
- Compare above/below ADC center
- Use Von Neumann debiasing (or LSB extraction)
- Circuit complexity is minimal

### Per-Channel Components
- 1x BZX55C3V3 Zener diode (3.3V)
- 1x Current-limiting resistor (470Ω proven good)
- 1x Decoupling capacitor (optional, for HF noise rejection)
- Direct connection to MCP3004 analog input

### MCP3004 Channel Mapping
| Channel | GPIO/Pin | Entropy Circuit |
|---------|----------|-----------------|
| CH0 | MCP3004 pin 2 | Avalanche #1 |
| CH1 | MCP3004 pin 3 | Avalanche #2 |
| CH2 | MCP3004 pin 4 | Avalanche #3 |
| CH3 | MCP3004 pin 5 | Avalanche #4 |

### Isolation Requirements
- Each circuit should be electrically isolated (no shared current paths)
- Separate bias networks if using any amplification
- Ground plane considerations for noise isolation

## Phase 2: Validation - 4-Channel Quality Check

Before writing native code, validate all 4 channels with Python:

```python
# Quick 4-channel validation script
for channel in range(4):
    samples = read_adc(channel, n=4096)
    print(f"CH{channel}: range={max(s)-min(s)}, mean={avg(s)}")
    # Check LSB balance for bits 0-3
```

**Success criteria** (per channel):
- ADC range > 500 counts
- Mean centered (400-600 for 10-bit ADC)
- LSB bits 0-2 balanced within 0.47-0.53

## Phase 3: fsnative Development

### Target: ARM Native Types for BCM2837

Key types needed in fsnative:

```fsharp
// SPI register access
type SPIRegisters = {
    CS: Ptr<uint32>      // Control/Status
    FIFO: Ptr<uint32>    // TX/RX FIFO
    CLK: Ptr<uint32>     // Clock divider
    DLEN: Ptr<uint32>    // Data length
}

// GPIO register access
type GPIORegisters = {
    FSEL: Ptr<uint32>    // Function select
    SET: Ptr<uint32>     // Output set
    CLR: Ptr<uint32>     // Output clear
    LEV: Ptr<uint32>     // Pin level
}

// MCP3004 ADC abstraction
type ADCChannel = CH0 | CH1 | CH2 | CH3
```

### Platform Binding Convention

Following Fidelity's `Platform.Bindings` pattern:

```fsharp
module Platform.Bindings.SPI =
    let transfer (cs: GPIOPin) (tx: NativeSpan<byte>) (rx: NativeSpan<byte>) : int =
        Unchecked.defaultof<int>  // Alex provides implementation
```

## Phase 4: MLIR scf.parallel Integration

### Target IR Structure

```mlir
// Parallel 4-channel ADC read
%results = scf.parallel (%ch) = (%c0) to (%c4) step (%c1)
           init (%init) -> (tensor<4xi32>) {
    %sample = call @read_adc_channel(%ch) : (index) -> i32
    scf.reduce(%sample : i32) {
        ^bb0(%lhs: i32, %rhs: i32):
            // Accumulate into result tensor
            scf.reduce.return %lhs : i32
    }
}
```

### Alex Binding for Parallel ADC

The binding should:
1. Configure SPI for the target channel
2. Assert chip select
3. Send command byte (0x01, 0x80 | (ch << 4), 0x00)
4. Read result from FIFO
5. Deassert chip select
6. Extract 10-bit value

### Parallelism Considerations

The MCP3004 is a single SPI device - true parallel reads aren't possible. But we can:
1. **Interleave efficiently**: Round-robin channels with minimal switching overhead
2. **Pipeline**: Start next channel's command while processing previous result
3. **Batch**: Group all 4 reads in a tight loop with no Python overhead

The `scf.parallel` expresses the *logical* parallelism. MLIR lowering decides actual execution.

## Phase 5: Entropy Extraction Pipeline

```
4x ADC Channels → 4x Raw Samples → Interleaved LSBs → Entropy Bytes
                                         ↓
                           XOR whitening (optional)
                                         ↓
                           4096 bytes output
```

### Throughput Math

At 1.25MHz SPI (proven stable):
- Single channel: ~3000 samples/sec
- 4 channels interleaved: ~12000 samples/sec total (3000 each)
- 4-bit LSB extraction: 48000 bits/sec = 6000 bytes/sec
- 4096 bytes: **~680ms** (Python overhead eliminated)

With 2MHz SPI and native code:
- Estimated: **200-400ms** for 4096 bytes

With DMA burst reads:
- Potential: **<100ms** for 4096 bytes

## File Locations

| Component | Repository | Path |
|-----------|------------|------|
| fsnative types | ~/repos/fsnative | TBD - ARM native types |
| Platform.Bindings.SPI | Fidelity.Platform | src/Platform.fs |
| scf.parallel codegen | Firefly | src/Alex/Bindings/ |
| Entropy sample | Firefly | samples/embedded/EntropyGenerator/ |

## Success Criteria

1. **Hardware**: All 4 channels produce quality entropy
2. **fsnative**: ARM register types compile through Fidelity pipeline
3. **MLIR**: scf.parallel generates correct ARM64 code
4. **Performance**: 4096 bytes in <500ms (stretch: <100ms)
5. **Integration**: Entropy feeds into QuantumCredential key derivation
