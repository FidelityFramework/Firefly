# PH2-00: Embedded Platform Strategy

## Overview

Phase 2 targets bare-metal embedded platforms for production QuantumCredential devices. The primary target is the **Renesas RA6M5** evaluation kit, with STM32L5 as a secondary option.

---

## Platform Priority

| Priority | Platform | Status | Rationale |
|----------|----------|--------|-----------|
| **Primary** | Renesas RA6M5 | Active development | Complete feature set, TrustZone, Secure Crypto Engine |
| Secondary | STM32L5 | On hold | Good platform, but RA6M5 better suited for security applications |

---

## Why Renesas RA6M5

The [EK-RA6M5 evaluation kit](https://www.renesas.com/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ek-ra6m5-evaluation-kit-ra6m5-mcu-group) provides everything needed for QuantumCredential in a single board:

### Security Features

| Feature | Benefit |
|---------|---------|
| **Hardware Unique Key (HUK)** | Factory-programmed 256-bit key unique to each MCU; enables device-bound credentials |
| **Unique ID** | Immutable 16-byte device identifier; cryptographically verifiable identity |
| **Arm TrustZone** | Hardware isolation between secure and non-secure code |
| **Secure Crypto Engine (SCE9)** | Hardware acceleration; HUK accessible only via private bus |
| **Tamper detection** | Physical security monitoring |
| **Power analysis resistance** | Side-channel attack mitigation |

### The Sovereignty Advantage

The **HUK fundamentally changes the trust model**. Traditional credential systems require trust in external authorities: certificate authorities, identity providers, key escrow services. The RA6M5's factory-provisioned HUK enables a different architecture:

| Traditional Model | HUK-Enabled Model |
|-------------------|-------------------|
| Identity issued by authority | Identity anchored in hardware |
| Keys stored in software/HSM | Keys bound to specific silicon |
| Cloneable with sufficient access | Physically unclonable |
| Trust delegated upward | Trust rooted in device |

With the HUK, the device itself becomes the anchor point for cryptographic identity. Credentials generated and wrapped on a specific RA6M5 are intrinsically bound to that physical device. No external service provisioned this identity; no external service can revoke or clone it. The owner of the device holds the only instance of that cryptographic identity in existence.

This is why the RA6M5 is elevated as the preferred platform: the STM32L5 can implement the same algorithms, but it lacks the factory-provisioned hardware identity that enables this sovereignty model. The HUK is not merely a security feature; it is the foundation for device-anchored self-sovereign credentials.

For a quantum-resistant credential device, these hardware security features are essential; they cannot be replicated in software alone.

### ADC Capabilities

| Specification | Value | Significance |
|---------------|-------|--------------|
| ADC count | 2x 12-bit | Dual independent converters |
| Sample rate | 5 Msps (interleaved) | High-speed entropy sampling |
| Resolution | 12-bit | 4096 levels vs MCP3004's 1024 |
| Channels | Multiple per ADC | Four-channel avalanche support |

The 12-bit resolution and 5 Msps rate exceed the YoshiPi's MCP3004 by significant margins.

### Processing Power

| Specification | Value |
|---------------|-------|
| Core | Arm Cortex-M33 |
| Frequency | 200 MHz |
| Flash | 2 MB |
| SRAM | 512 KB |

The Cortex-M33 includes the DSP extension and single-cycle multiplier, useful for cryptographic operations and entropy conditioning.

### Connectivity

| Interface | Capability |
|-----------|------------|
| USB | Full Speed + High Speed |
| Ethernet | MAC with DMA |
| CAN FD | Automotive-grade |
| Serial | Multiple UART/SPI/I2C |

USB High Speed enables fast credential transfer without the USB Full Speed bottleneck.

---

## Development Environment

### Renesas FSP (Flexible Software Package)

The RA6M5 uses Renesas FSP rather than STM32's HAL. FSP provides:

- Hardware abstraction layer
- Driver modules for all peripherals
- Security driver for TrustZone and Crypto Engine
- Integration with e2 studio IDE

### Zephyr RTOS Support

The RA6M5 has [first-class Zephyr RTOS support](https://docs.zephyrproject.org/latest/boards/renesas/ek_ra6m5/doc/index.html), providing an alternative to bare-metal or NuttX approaches.

### Firefly Integration Path

```
F# Source + FNCS
       ↓
   Firefly/Alex
       ↓
   MLIR (ARM Cortex-M33 target)
       ↓
   LLVM
       ↓
   ELF Binary (linked with FSP drivers)
```

The Platform.Bindings pattern applies identically; only the binding implementations differ.

---

## Phase 2 Document Organization

```
Phase2-Embedded/
├── PH2-00-Embedded-Strategy.md      ← This document
├── PH2-01-RA6M5-Platform.md         ← Primary target details
├── PH2-02-RA6M5-Security.md         ← TrustZone, Crypto Engine
├── PH2-03-RA6M5-ADC-Bindings.md     ← 12-bit ADC integration
└── STM32L5/                         ← Secondary target (on hold)
    ├── PH2-01-Strategy-Overview.md
    ├── PH2-02-Hardware-Platforms.md
    ├── PH2-03-Farscape-Assessment.md
    └── PH2-04-UI-Options.md
```

---

## Migration from YoshiPi

The Phase 1 YoshiPi implementation provides the foundation:

| Component | YoshiPi (Phase 1) | RA6M5 (Phase 2) |
|-----------|-------------------|-----------------|
| Entropy source | 4-channel avalanche | Same circuit, different ADC |
| ADC interface | MCP3004 via SPI/IIO | RA6M5 internal 12-bit ADC |
| Epsilon evaluation | Software (F#) | Same algorithm |
| XOR combination | Software (F#) | Same algorithm |
| Crypto operations | Software (ML-KEM/DSA) | Hardware accelerated (SCE9) |
| Credential storage | File system | Secure Flash (TrustZone) |
| Device binding | None (portable) | **HUK-wrapped (clone-proof)** |
| Device identity | MAC address | **Factory Unique ID** |

The F# application code remains largely unchanged; Platform.Bindings implementations differ.

---

## STM32L5 Status

The STM32L5 documentation in `STM32L5/` remains valid but is on hold:

- **Not abandoned**: The platform analysis and Farscape assessment remain useful
- **Lower priority**: RA6M5's factory-provisioned HUK enables the sovereignty model that STM32L5 cannot match
- **Different use case**: STM32L5 may suit applications where device-bound identity is not required
- **Future option**: May revisit for cost-optimized variants or different product tiers

---

## Next Steps

1. **PH2-01-RA6M5-Platform.md**: Document EK-RA6M5 board specifics
2. **PH2-02-RA6M5-Security.md**: TrustZone partitioning for credential isolation
3. **PH2-03-RA6M5-ADC-Bindings.md**: Platform.Bindings for FSP ADC driver
4. Validate avalanche circuit compatibility with RA6M5 ADC input range

---

## References

- [Renesas RA6M5 Product Page](https://www.renesas.com/en/products/ra6m5)
- [EK-RA6M5 Evaluation Kit](https://www.renesas.com/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ek-ra6m5-evaluation-kit-ra6m5-mcu-group)
- [RA6M5 Security Manual](https://www.renesas.com/en/document/apn/ra6m5-mcu-group-security-manual) - HUK, SCE9, key wrapping details
- [Secure Key Management Tool](https://www.renesas.com/en/document/mat/security-key-management-tool-users-manual)
- [Zephyr RTOS RA6M5 Support](https://docs.zephyrproject.org/latest/boards/renesas/ek_ra6m5/doc/index.html)
- [RA6M5 Datasheet](https://www.renesas.com/en/document/fly/renesas-ra6m5-group)
