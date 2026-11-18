# Architecture

## Runtime layers

1. Runtime/API: configuration, lifecycle, diagnostics, and CLI.
2. Model engines: separate DiffusionGemma and Clover adapters.
3. Optimization core: memory orchestrator, expert cache, scheduler, transfer engine, kernel dispatcher, and quantization policy.
4. Backend: Candle with CUDA support.

## Model-specific work

### DiffusionGemma
- Diffusion canvas/state management and denoising loop.
- Bidirectional attention and state/cache lifetime policy.
- Mixed-precision validation for each kernel path.

### Clover-35B-A3B-preview
- GDN and block-diffusion execution paths.
- MoE routing and expert residency.
- Expert cache, attention, and prefetch integration.

## Safety invariants
- Treat configured memory budgets as admission limits, not guaranteed free memory.
- Keep a safety reserve; never plan to consume all reported VRAM.
- Evict only after outstanding device work using an allocation has completed.
- Swap is not a model-weight tier.
