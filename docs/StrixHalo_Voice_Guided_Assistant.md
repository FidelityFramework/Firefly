# Strix Halo Voice-Guided Assistant

## Status: Speculative Design

This document describes a local MoE inference system with voice I/O, compiled entirely through the Fidelity/Firefly stack. No llama.cpp, no vLLM, no Python runtime. The entire inference pipeline is F# source compiled to native code via MLIR.

The design principle is complete hardware utilization. Every processor on the die does meaningful inference work: the CPU runs a ternary routing model that makes hardware-aware dispatch decisions in microseconds, the NPU runs quantized experts at 50 TOPS, and the GPU runs dense language models at full FP16 precision. The ternary router is not a decision tree; it is a small, fast model whose job is to make the agentic flow responsive to what each processor does best.

---

## Hardware Target

AMD Strix Halo APU with unified memory:

| Processor | Role | Capability |
|-----------|------|------------|
| **Zen 5 CPU** | MoE routing, TTS synthesis, orchestration | AVX-512, 16 cores, L2-resident router model |
| **XDNA 2 NPU** | ASR inference, ternary expert inference | 50 TOPS INT8, low power |
| **RDNA 3.5 GPU** | Dense LLM inference, attention-heavy computation | FP16/FP32 SIMT parallelism |
| **Unified LPDDR5X** | Shared memory for all processors | Up to 128 GB, zero-copy between all three |

All models reside in unified memory. No copies between processors. BAREWire zero-copy semantics map directly to the hardware topology.

---

## Architecture

### Conversational Pipeline

The pipeline is sequential by nature of conversation, but within the inference phase, the NPU and GPU can work concurrently on different aspects of the response. The router determines which processor handles each expert activation based on model characteristics: ternary-quantized experts dispatch to the NPU, dense language models dispatch to the GPU. This is not a quality fallback; it is hardware-aware scheduling.

```
Microphone
    │
    ▼
┌──────────────────────────┐
│  Mel Spectrogram (CPU)   │  FFT + mel filterbank on PCM audio
└──────────┬───────────────┘
           │ mel features (unified memory)
           ▼
┌──────────────────────────┐
│  ASR Model (NPU)         │  Whisper tiny/small, INT8 quantized
└──────────┬───────────────┘
           │ text tokens (unified memory)
           ▼
┌──────────────────────────┐
│  Ternary Router (CPU)    │  Small MoE routing model, AVX-512
│  L2-cache resident       │  Hardware-aware dispatch: selects expert(s) + processor
└──────────┬───────────────┘
           │ activation signal
           ▼
┌──────────────────────────┐
│  Expert Inference         │  Ternary experts → NPU (INT8)
│  (NPU and/or GPU)        │  Dense LLMs → GPU (FP16)
└──────────┬───────────────┘
           │ response tokens (streaming)
           ▼
┌──────────────────────────┐
│  TTS Model (CPU)         │  VITS/Piper (mel generation)
│  Vocoder (CPU)           │  HiFi-GAN (waveform synthesis)
└──────────┬───────────────┘
           │ PCM audio
           ▼
        Speaker
```

### Why TTS on CPU, Not NPU

During response generation, the NPU and GPU are occupied with expert and LLM inference respectively. Response tokens stream out incrementally; the TTS model converts them to audio as they arrive. Running TTS on CPU allows the inference processors to continue generating tokens while the CPU synthesizes speech from already-generated tokens. This overlap is what makes the response feel immediate.

If inference finishes before TTS completes (short responses), both NPU and GPU are idle. If XDNA 2's driver stack supports concurrent model contexts on separate AI Engine tiles, TTS could migrate to the NPU for these cases. But the CPU-based TTS path is the safe default that requires no assumptions about NPU concurrency.

### Model Inventory

| Model | Task | Parameters | Quantized Size | Processor | Residency |
|-------|------|-----------|---------------|-----------|-----------|
| Whisper small | ASR | 244M | ~244 MB (INT8) | NPU | Always resident |
| Routing model | MoE dispatch | 10-50M | ~5-25 MB (1.58-bit) | CPU | Always resident (L2) |
| Language expert (ternary) | General text | 100-500M | ~50-250 MB (1.58-bit) | NPU | Hot-loadable |
| Code expert (ternary) | Programming tasks | 100-500M | ~50-250 MB (1.58-bit) | NPU | Hot-loadable |
| Reasoning expert (ternary) | Logic, planning | 200-800M | ~100-400 MB (1.58-bit) | NPU | Hot-loadable |
| General LLM (dense) | Open-ended generation | 1-3B | ~2-6 GB (INT8/FP16) | GPU | Hot-loadable |
| Long-context LLM (dense) | Multi-turn, summarization | 1-7B | ~2-14 GB (INT8/FP16) | GPU | Hot-loadable |
| VITS/Piper | TTS (mel gen) | 25-60M | ~25-60 MB | CPU | Always resident |
| HiFi-GAN | Vocoder | 14M | ~14 MB | CPU | Always resident |

Total always-resident: under 400 MB. With 128 GB unified memory, multiple experts and dense LLMs can be loaded simultaneously. The GPU-hosted models are not fallbacks; they are the workhorses for tasks where full-precision attention and generation quality matter. The ternary experts on NPU handle the tasks where speed and power efficiency dominate. The router's job is to know which is which.

---

## Expert Hot-Loading

### Memory-Mapped Weight Files

Expert weights are stored as memory-mapped files on NVMe. The OS virtual memory subsystem handles paging:

- **Resident expert**: weights are in physical memory. Inference starts immediately.
- **Non-resident expert**: weights are memory-mapped but paged out. First access triggers page-in from NVMe.
- **Prefetch on routing**: when the router selects an expert, issue `madvise(MADV_WILLNEED)` on that expert's weight file. NVMe starts reading while the current expert finishes.

### Page-In Latency

| Model Size | NVMe Read (14 GB/s PCIe 5.0) | Acceptable? |
|------------|-------------------------------|-------------|
| 50 MB (ternary expert) | ~4 ms | Yes |
| 250 MB (ternary expert) | ~18 ms | Yes |
| 500 MB (ternary expert) | ~36 ms | Marginal |
| 2 GB (dense LLM, GPU) | ~143 ms | Noticeable; use prefetch or keep resident |
| 7 GB (dense LLM, GPU) | ~500 ms | Keep resident; first-load only |

For ternary experts under 250 MB, page-in latency is imperceptible. The router's prediction accuracy determines the effective cache hit rate. If the router consistently selects the same 2-3 experts for a conversation, those experts stay resident after first use.

### Eviction Policy

Simple LRU by expert. When memory pressure requires eviction, drop the least-recently-used expert's pages. The OS handles this natively for memory-mapped files; the application just tracks access order for prefetch decisions.

---

## Actor Topology

Each pipeline stage is a `MailboxProcessor` actor. Messages are BAREWire-serialized references to unified memory buffers (zero-copy).

```fsharp
// Pipeline actors
let audioCapture = AudioCapture.spawn deviceId
let melExtractor = MelSpectrogram.spawn { sampleRate = 16000; fftSize = 400; hopLength = 160 }
let asrActor     = NPUInference.spawn whisperModel
let routerActor  = CPUInference.spawn routerModel
let expertPool   = ExpertPool.spawn expertConfigs    // manages hot-loading
let ttsActor     = CPUInference.spawn ttsModel
let vocoderActor = CPUInference.spawn vocoderModel
let audioOutput  = AudioOutput.spawn deviceId

// Wiring (types enforce correctness at each boundary)
audioCapture  |> pipeTo melExtractor     // PCMFrame → MelFeatures
melExtractor  |> pipeTo asrActor         // MelFeatures → TokenSequence
asrActor      |> pipeTo routerActor      // TokenSequence → RoutingDecision
routerActor   |> pipeTo expertPool       // RoutingDecision → TokenStream (streaming)
expertPool    |> pipeTo ttsActor         // TokenStream → MelFrames (streaming)
ttsActor      |> pipeTo vocoderActor     // MelFrames → PCMFrame
vocoderActor  |> pipeTo audioOutput      // PCMFrame → speaker
```

### Type Safety Across the Pipeline

Each actor boundary has a distinct message type. The compiler rejects wiring errors:

```fsharp
type PCMFrame      = { samples: float32 array; sampleRate: int }
type MelFeatures   = { frames: float32 array; numMels: int; numFrames: int }
type TokenSequence = { tokens: int array; confidence: float32 }
type RoutingDecision = { expertId: ExpertId; target: ProcessorTarget; query: TokenSequence; priority: Priority }
type TokenStream   = IAsyncEnumerable<int>   // streaming, token-by-token
type MelFrames     = IAsyncEnumerable<float32 array>  // streaming, frame-by-frame
```

Attempting to pipe `MelFeatures` to `expertPool` is a compile-time error. The types document the data flow and enforce it simultaneously.

### Streaming Response

The expert generates response tokens incrementally. Each token flows to TTS immediately:

```fsharp
// Expert inference streams tokens as they are generated
let! responseStream = expertPool.InferStreaming(routingDecision)

// TTS processes tokens as they arrive (not waiting for full response)
responseStream
|> AsyncSeq.bufferByCount 4          // accumulate a few tokens for natural prosody
|> AsyncSeq.map ttsActor.Synthesize  // mel generation per phrase
|> AsyncSeq.iter vocoderActor.Vocalize  // waveform per phrase → speaker
```

The user hears the beginning of the response while the expert is still generating the end. The pipeline latency from first generated token to first audible output is: TTS inference time (~20-40ms for a short phrase on CPU) plus vocoder time (~5-10ms). Under 50ms from token to sound.

---

## What This Replaces

| Conventional Stack | This Architecture |
|-------------------|-------------------|
| whisper.cpp (C++) | Whisper model compiled from F# weight loader + NPU dispatch |
| llama.cpp / vLLM (C++/Python) | Ternary and dense inference compiled from F# via MLIR, hardware-dispatched |
| Python glue scripts | F# actor pipeline, compile-time wired |
| GGML/GGUF format | BAREWire-serialized weight tensors |
| Multiple processes, IPC | Single process, actors on unified memory |
| nvidia-smi / htop monitoring | Per-actor latency signals, Fidelity.UI dashboard |

### What the F# Inference Engine Must Implement

The inference engine is the core engineering deliverable. It replaces the C++ inference runtimes (GGML, vLLM) with F# compiled through MLIR:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Tokenizer (BPE) | Low | String processing; well-understood algorithm |
| Ternary layer forward pass | Low | Add-subtract on packed integers; ~30 lines F# per layer type |
| Attention mechanism | **High** | KV cache management, multi-head attention, softmax |
| Layer normalization | Medium | Element-wise; needs vectorization for throughput |
| Activation functions (GELU, SiLU) | Low | Single-pass element-wise |
| Weight loading (memory-mapped) | Medium | mmap integration via platform binding |
| NPU dispatch | Medium | AMD XDNA driver API; platform binding |
| GPU dispatch (dense LLMs) | **High** | ROCm/HIP integration for RDNA 3.5 |
| KV cache management | **High** | Per-expert cache, context-length-dependent sizing |

Attention and KV cache management are the hardest problems. For ternary experts, attention heads may use reduced precision (INT8 activations), which simplifies the computation but still requires correct implementation. The KV cache for a 4K context window with 32 heads and 128-dim keys is ~32 MB per expert.

---

## Distillation Pipeline

The user distills their own models using a conventional training stack (PyTorch), then exports weights for the F# inference engine.

### Training (External, PyTorch)

1. **Teacher model**: Large pretrained model (e.g., Llama 70B on rented GPU)
2. **Ternary experts**: Distill task-specific experts with ternary quantization-aware training (NPU targets)
3. **Dense models**: Fine-tune or quantize (INT8/FP16) general-purpose LLMs for GPU inference
4. **Router distillation**: Train a small router on the expert selection and hardware dispatch task
5. **Export**: Save weights in a flat binary format (BAREWire-compatible); quantization field identifies the target processor

### Weight Format

```
[header: 64 bytes]
  magic: u32          ("FWGT")
  version: u16
  quantization: u8    (0=FP32, 1=FP16, 2=INT8, 3=Ternary)
  num_layers: u16
  hidden_dim: u16
  num_heads: u16
  vocab_size: u32

[layer_offsets: num_layers * 8 bytes]
  offset: u64         (byte offset from file start)

[weight_data: contiguous]
  layer 0 weights...
  layer 1 weights...
  ...
```

Memory-mapping this file gives direct pointer access to each layer's weights. No deserialization. The mmap region is the weight tensor. BAREWire's zero-copy principle extends to model loading: the bits on disk are the bits in memory are the bits the NPU reads.

---

## Incremental Milestones

### Milestone 1: CPU-Only Pipeline

All inference on CPU via AVX-512. No NPU, no GPU. Proves the actor pipeline, weight loading, and inference engine work end-to-end.

- [ ] BPE tokenizer in F#
- [ ] Ternary weight loader (memory-mapped)
- [ ] Ternary forward pass (add-subtract, AVX-512)
- [ ] Attention mechanism (scalar F#, then vectorized)
- [ ] Whisper inference on CPU (slow but correct)
- [ ] TTS inference on CPU (VITS/Piper)
- [ ] Actor pipeline wiring
- [ ] End-to-end: speak → transcribe → route → infer → speak response

### Milestone 2: NPU Acceleration

Move ASR and ternary expert inference to XDNA 2 NPU.

- [ ] AMD XDNA driver platform binding
- [ ] NPU model loading and context management
- [ ] ASR on NPU (Whisper)
- [ ] Ternary expert inference on NPU
- [ ] Latency profiling and optimization

### Milestone 3: GPU LLM Inference

Dense language models on RDNA 3.5. These are not fallbacks; they are the primary path for open-ended generation, multi-turn conversation, and tasks where attention precision dominates quality.

- [ ] ROCm/HIP platform binding for RDNA 3.5
- [ ] Dense forward pass on GPU (FP16 matmul)
- [ ] GPU attention with KV cache
- [ ] Router dispatch logic: hardware-aware expert selection (NPU for ternary, GPU for dense)
- [ ] Concurrent NPU + GPU inference for multi-expert activations

### Milestone 4: Hot-Loading and Adaptive Routing

Expert pool management and learned routing.

- [ ] Memory-mapped expert pool with LRU eviction
- [ ] Prefetch on routing decision (`madvise`)
- [ ] Usage-based expert residency (keep frequently-used experts in memory)
- [ ] Fidelity.UI dashboard showing per-actor latency, expert residency, processor utilization

---

## Honest Constraints

**AMD's NPU driver stack is immature.** The XDNA 2 hardware is real and the TOPS are real, but the software ecosystem (model compilation toolchain, multi-context scheduling, profiling) is not at the maturity level of CUDA/TensorRT. This is the primary integration risk for Milestones 2 and 3.

**Model quality depends on distillation.** Ternary quantization works well for encoder models and feed-forward-dominant architectures. Autoregressive generation with ternary weights and INT8 activations may produce lower quality output than FP16 inference. This is exactly why the GPU runs dense LLMs as first-class participants: the router dispatches to the processor and model that best serves each query, not as a fallback but as part of the hardware-aware scheduling that makes the system work.

**Attention is the engineering bottleneck.** Everything else in the inference engine (tokenizer, ternary forward pass, layer norm, activation functions) is straightforward. Attention with KV cache management is where llama.cpp has thousands of lines of carefully optimized C++. The F# implementation will be correct before it is fast; MLIR/LLVM optimization narrows the gap over time, but the initial version will not match llama.cpp throughput.

**The training pipeline remains Python.** Distillation requires PyTorch and GPU access. The F# stack handles inference, not training. The boundary between training and inference is the weight file format.

---

## Cross-References

### Firefly Docs
- [Architecture_Canonical.md](./Architecture_Canonical.md): Two-layer model, platform bindings
- [Platform_Binding_Model.md](./Platform_Binding_Model.md): How NPU and GPU bindings would be structured

### SpeakEZ Blog
- [A Unified Vision for Ternary Models](/blog/a-unified-vision-for-ternary-models/): Ternary quantization, MoE routing, Strix Halo deployment model
- [Bringing Posit Arithmetic to F#](/blog/bringing-posit-arithmetic-to-fsharp/): Posit accumulation for precision-critical paths
- [Fidelity.Rx / Signal-Actor Isomorphism](/blog/fidelityrx-native-reactivity-in-fidelity/): Actor pipeline model

### External
- [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna): NPU specifications
- [Whisper](https://github.com/openai/whisper): ASR model family
- [Piper TTS](https://github.com/rhasspy/piper): Fast, local neural TTS
- [BitNet](https://arxiv.org/abs/2310.11453): Ternary quantization-aware training
