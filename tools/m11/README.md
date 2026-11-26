# M11 CPU differential reference

`gen_cpu_reference.py` generates small deterministic PyTorch CPU outputs from
the Clover formulas pinned in `docs/audits/clover-upstream-manifest.md`.

It covers:

- recurrent GDN commit and read-only bidirectional block preview;
- causal and bidirectional/right-context convolution;
- decay and beta gates;
- zero-centered q/k RMSNorm and GDN RMSNorm + SiLU gate;
- partial RoPE;
- the full-attention path: RMSNorm, partial RoPE, 16Q/2KV GQA, prefix/block
  mask, and output sigmoid gate.

The script only creates CPU tensors. It does not load model weights, import
CUDA, query a GPU, allocate VRAM, or require the checkpoint. Do not add CUDA
calls to this generator.

Run from the repository root in an environment with CPU PyTorch:

```bash
python tools/m11/gen_cpu_reference.py --out /tmp/clover-cpu-reference
```

If the pinned source file is available locally, verify it as part of generation:

```bash
python tools/m11/gen_cpu_reference.py \
  --upstream-file /path/to/modeling_clover_moe.py \
  --out /tmp/clover-cpu-reference
```

The output directory must be fresh or empty. The generator writes
`clover_gdn_upstream_fixtures.rs` and `manifest.json`; the former is the
Rust test fixture module, the latter records source revision/hash, CPU/PyTorch
environment, seed, tensor shapes and output checksums. Copy the generated Rust
fixture module to `crates/claymore-tensor/src/` and update
`docs/audits/m11.2-cpu-reference-manifest.json` only after verifying the
generated hashes.

Golden values are stored as exact F32 bit patterns after widening upstream
BF16/F32 outputs. Rust tests use explicit absolute tolerances for FP32 and
BF16 operations; eager BF16 attention allows at most one representable BF16
ULP to account for CPU reduction-order differences.
