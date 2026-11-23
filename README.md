# Claymore

Claymore is an experimental Rust inference runtime with its own device, memory, tensor, and model-execution architecture, targeting efficient execution of DiffusionGemma and Clover-35B-A3B-preview on a single NVIDIA GPU.

## Target development machine

- GPU: NVIDIA RTX 5090 (32 GiB VRAM)
- CPU: AMD Ryzen 9 7950X3D
- System memory: 128 GiB
- Environment: WSL2
- Compute strategy: mixed precision, with BF16 as the preferred baseline where supported

## Architecture

**Minimum supported Rust version (MSRV): 1.86.0.** Every workspace crate inherits this version from the root manifest, and CI runs the complete workspace test suite on Rust 1.86 as well as stable. Raise the MSRV only with an explicit compatibility review and synchronized manifest, CI, and documentation updates.

The workspace is split into ten crates:

- `claymore-core`: shared runtime types and configuration
- `claymore-device`: backend-neutral Device/Buffer/Tensor contracts and CPU simulator
- `claymore-memory`: memory budgets, placement policies, and expert-cache policy
- `claymore-scheduler`: work scheduling and prefetch policy
- `claymore-runtime`: model-agnostic request lifecycle, execution context, and cancellation
- `claymore-model-gemma`: DiffusionGemma integration boundary
- `claymore-model-clover`: Clover model integration boundary
- `claymore-cuda`: optional CUDA Driver API backend (feature-gated)
- `claymore-cli`: command-line interface and diagnostics

The optimization roadmap includes a memory orchestrator, adaptive expert cache, pinned host memory, asynchronous H2D/D2H transfers, event-safe eviction, predictive prefetch, telemetry, and specialized CUDA kernels.

## Important implementation status

This repository is an early-stage runtime. The M7 CPU path now connects scheduled requests, queue runners, and backend-neutral memory operations. The memory layer enforces the soft VRAM target and hard ceiling, applies the expert-cache maximum, supports reload-cost-aware adaptive eviction, and exposes CPU-side execution telemetry. `CudaTransferExecutor` now maps scheduled H2D/D2H operations to the dedicated owner-thread `CudaMemoryManager` and waits for real CUDA events before releasing scheduler dependencies. Its real-device behavior and VRAM-pressure guarantees were validated on the target RTX 5090 for the M7.1 code SHA documented in `docs/audits/m7.1-rtx5090-hardware-validation.md`. M8 adds the canonical `claymore-tensor` crate, explicit operation contracts and dispatch, and four native F32 CUDA kernels (add, matmul, stable softmax, RMSNorm) with RTX 5090 differential validation documented in `docs/audits/m8-rtx5090-hardware-validation.md`. M9 adds a shared block-based diffusion orchestration API in `claymore-runtime` (validated plans, scheduler-backed step dependencies, cooperative cancellation, failure propagation, and CPU-side metrics). Model-specific execution, weights, and tensor payloads remain outside this phase.

The CUDA backend is opt-in and links directly to the NVIDIA Driver API. The default workspace build remains hardware-independent. The M4/M5 CUDA path has been validated on the target RTX 5090 for the commits and scenarios documented in `docs/audits/m4-m5-rtx5090-hardware-validation.md`; this does not imply that every CUDA path or model execution path is hardware-validated. Validate locally with:

```bash
cargo fmt --all
cargo test --workspace --all-targets
cargo run -p claymore-cli -- profile
cargo run -p claymore-cli --features cuda-driver -- device
```

## Hardware profile

The initial RTX 5090 profile uses a soft VRAM budget of 26 GiB and a hard ceiling of 28 GiB, leaving safety headroom. Expert-cache targets are 14 GiB with a 16 GiB maximum, included within the model working-set budget rather than added on top.

See [`config/rtx5090-7950x3d.json`](config/rtx5090-7950x3d.json) and the [consolidated project reference](docs/Claymore.md). The profile is parsed and validated by `claymore-runtime`, converted into scheduler and memory budgets, and embedded as the CLI default. Override the CLI profile with `CLAYMORE_HARDWARE_PROFILE=/path/to/profile.json`.
