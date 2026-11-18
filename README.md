# Claymore

Claymore is an experimental Rust inference runtime built around Hugging Face Candle, targeting efficient execution of DiffusionGemma and Clover-35B-A3B-preview on a single NVIDIA GPU.

## Target development machine

- GPU: NVIDIA RTX 5090 (32 GiB VRAM)
- CPU: AMD Ryzen 9 7950X3D
- System memory: 128 GiB
- Environment: WSL2
- Compute strategy: mixed precision, with BF16 as the preferred baseline where supported

## Architecture

The planned workspace is split into six crates:

- `claymore-core`: shared runtime types and configuration
- `claymore-memory`: memory budgets, placement policies, and expert-cache policy
- `claymore-scheduler`: work scheduling and prefetch policy
- `claymore-model-gemma`: DiffusionGemma integration boundary
- `claymore-model-clover`: Clover model integration boundary
- `claymore-cli`: command-line interface and diagnostics

The optimization roadmap includes a memory orchestrator, adaptive expert cache, pinned host memory, asynchronous H2D/D2H transfers, event-safe eviction, predictive prefetch, telemetry, and specialized CUDA kernels.

## Important implementation status

This repository is an early-stage runtime. The v0.3 memory crate now contains deterministic budget validation, allocation accounting, admission planning, LRU/LFU/adaptive eviction policy hooks, and an event-horizon safety check. These are host-side policies only: real CUDA allocation, pinned memory, asynchronous transfers, Candle model execution, and specialized kernels remain future work.

The initial scaffold has not yet been compiled or tested in this environment. Validate locally with:

```bash
cargo fmt --all
cargo test --workspace
```

## Hardware profile

The initial RTX 5090 profile uses a soft VRAM budget of 26 GiB and a hard ceiling of 28 GiB, leaving safety headroom. Expert-cache targets are 14 GiB with a 16 GiB maximum, included within the model working-set budget rather than added on top.

See `config/rtx5090-7950x3d.json` and the documentation as they are added.
