# Quantum Credential Entropy PoC - Results Summary

## Objective
Generate 4096 bytes of quantum-derived entropy in <500ms using avalanche noise circuit connected to YoshiPi.

## Hardware Setup
- **YoshiPi**: Raspberry Pi Zero 2 W (ARM Cortex-A53, quad-core 1GHz)
- **ADC**: MCP3004 (10-bit, 4-channel SPI) on SPI1, GPIO8 as chip select
- **Entropy Circuit**: Single-channel avalanche noise
  - Zener diode: BZX55C3V3 with 470Ω feed resistor
  - Op-amp: LM324 non-inverting configuration
  - Feedback: 1MΩ (unity gain for noise, DC-coupled)
  - Bias: 100kΩ + 330Ω voltage divider (~16.5mV input bias)

## Signal Quality Achieved
```
ADC Range: 162-872 counts (710 count swing)
Mean: ~522 (well-centered in 0-1023 range)
LSB Balance (bits 0-3): All within 0.47-0.53 ✓
```

## Python Performance Ceiling

### Method 1: Von Neumann Debiasing
Compare consecutive samples to center, output bit only when they differ.
- Efficiency: ~30% (70% of sample pairs discarded)
- **Result: 15.2 seconds for 4096 bytes** ❌

### Method 2: LSB Extraction
Extract bottom N bits from each ADC sample directly.
- Quality validated for 1-4 LSBs at appropriate SPI speeds
- **Best result: 1117ms at 1.25MHz SPI, 4-bit extraction**
- Still 2.2x slower than 500ms target ❌

### Configuration Search Results
| SPI Speed | LSB Bits | Time (4096B) | Status |
|-----------|----------|--------------|--------|
| 0.50 MHz | 2 | 3123ms | OK |
| 1.00 MHz | 3 | 1878ms | OK |
| 1.25 MHz | 4 | **1117ms** | OK |
| 1.50 MHz | 3 | 1448ms | OK |
| 2.00 MHz | 3 | 1376ms | OK |
| 2.00 MHz | 4 | --- | FAIL (bit 3 biased) |

## Python Bottlenecks Identified

1. **Interpreted Execution**: Every `read_adc()` call goes through Python bytecode interpreter
2. **Library Overhead**:
   - `spidev.xfer2()` allocates Python objects per transaction
   - `RPi.GPIO` library adds latency to chip-select toggling
3. **No Register Access**: Cannot directly manipulate SPI/GPIO registers
4. **No DMA**: Cannot use DMA controller for burst ADC reads
5. **Dynamic Typing**: Bit manipulation requires type checks at runtime

## Projected Native Performance

A Fidelity-compiled native binary could achieve:

| Optimization | Estimated Gain |
|--------------|----------------|
| Direct SPI register access | 3-5x |
| Inline bit manipulation | 2x |
| No interpreter overhead | 2-3x |
| DMA burst reads | 5-10x |

**Conservative estimate: 50-200ms for 4096 bytes** (5-20x improvement)

This would meet the <500ms human-imperceptible target.

## Why Fidelity is the Right Tool

1. **Bare-metal targeting**: Fidelity can generate freestanding binaries with direct hardware access
2. **Zero runtime overhead**: No GC, no interpreter, no dynamic dispatch
3. **Type-safe register access**: F# types can model hardware registers safely
4. **Platform bindings**: Alex can generate optimal ARM64 code for BCM2837 peripherals

## Files on YoshiPi

Location: `hhh@192.168.68.60:~/bin/`

| Script | Purpose |
|--------|---------|
| `entropy_sampler.py` | Basic ADC sampling with timestamps |
| `entropy_sampler_vonneumann.py` | Von Neumann debiasing implementation |
| `entropy_benchmark.py` | Full 4096-byte speed benchmark |
| `entropy_delay_test.py` | Inter-sample delay optimization |
| `lsb_entropy_test.py` | LSB quality and balance analysis |
| `fast_lsb_entropy.py` | High-speed LSB extraction |
| `optimal_entropy.py` | SPI speed × bit depth search |
| `README_entropy.md` | Summary documentation |

## Next Steps

1. Design F# module for direct SPI register access on BCM2837
2. Implement as Platform.Bindings for ARM64 Linux
3. Create entropy generator using Fidelity compilation
4. Validate sub-500ms performance target
5. Integrate with QuantumCredential key derivation pipeline
