# `tools/m10` — reproducible M10 CPU reference generation

One versioned tool replaces the 2026-11-22/24 throwaway campaign scripts.
It regenerates the per-layer reference fixture set that the CPU bisection
tests (`tests/golden_cpu_layer_bisect.rs`, `tests/vision_layer_bisect.rs`,
`tests/decoder_layer_bisect.rs`) and the chain differential
(`tests/golden_cpu.rs`) validate against, exactly as the
[CPU finalization contract](../../docs/audits/m10-cpu-finalization-contract.md)
requires: versioned, deterministic, fingerprinted, and **never** overwriting
an approved golden set.

## The committed fixture set

`crates/claymore-model-gemma/tests/data/m10_layers/` — 455 FP32 little-endian
`.bin` artifacts plus `manifest.json` + `manifest.sha256`. `tests/m10_reference.rs`
(non-ignored, runs in default CI) pins the manifest identity
(`PINNED_MANIFEST_SHA256 = 660da49e8a12e24a…c74e3`), verifies every artifact's
size, finiteness and SHA-256 against the manifest, and re-checks the recorded
generation gates (85 internal chain-consistency checks, 311 bit-exactness
gates against the historical campaign goldens, zero failures).

Only the checkpoint is external: the M10 tests read it from
`CLAYMORE_M10_CHECKPOINT`.

## Checkpoint

`google/diffusiongemma-26B-A4B-it`, pinned revision
`cd40195554d3cfcb56f34b9adfb66efc790ed5b6`:

```sh
huggingface-cli download google/diffusiongemma-26B-A4B-it \
  --revision cd40195554d3cfcb56f34b9adfb66efc790ed5b6 --local-dir /path/to/checkpoint
```

Pinned file digests (recorded again in every generated manifest):

| file | SHA-256 |
| --- | --- |
| `config.json` | `13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506` |
| `model.safetensors.index.json` | `6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a` |

## Environment (CPU-only, ~1 min + downloads)

`setup_env.sh` builds a venv from `requirements.cpu.lock.txt` and hard-asserts
torch `2.15.0+cpu`, no CUDA, and importability of `diffusion_gemma` + `gemma4`
from the pinned upstream commit `671aa77576266f41f33f9c28e2d824bfd1b1b63c`:

```sh
bash tools/m10/setup_env.sh /tmp/m10-venv
```

The environment fingerprint (torch/transformers/numpy versions, SHA-256 of the
upstream module files actually imported, libc, CPU model, thread counts) is
recorded into every generated `manifest.json`.

## Generation

```sh
/tmp/m10-venv/bin/python tools/m10/gen_reference.py \
  --checkpoint /path/to/checkpoint \
  --out /tmp/m10-reference/run-$(date +%F) \
  --golden /path/to/golden_eager \
  --layers /path/to/golden_layers
```

* `--out` must not exist or must be empty — the tool refuses to overwrite, and
  historical goldens are read-only inputs, never targets.
* `--golden` / `--layers` point at the historical campaign golden sets
  (`/tmp/opencode/m10/golden_eager`, `golden_layers` when working from the
  campaign tree). Every regenerated tensor that overlaps a historical golden
  must be bit-identical; any mismatch fails the run non-zero. Omitting them
  skips only the historical gates — internal checks still gate.
* The run (~36 s on an AMD 7950X3D with a page-cached checkpoint, single
  process, `CUDA_VISIBLE_DEVICES=""`) ends with `DONE_REFERENCE`.

## Validation of a fresh run

```sh
python tools/m10/validate_reference.py \
  --against crates/claymore-model-gemma/tests/data/m10_layers/manifest.json \
  --dir /tmp/m10-reference/run-<date> [--strict-env]
```

Every artifact must match the pinned manifest by SHA-256 (`--strict-env`
also makes environment-fingerprint drift fail instead of warn).

## Regenerating the committed set (procedure)

Changing the committed fixture set is an architect decision (it repins the
reference). Procedure:

1. Full CPU machine with the pinned checkpoint and the historical golden sets.
2. `setup_env.sh`, then `gen_reference.py` into a **fresh** directory; the run
   must end `DONE_REFERENCE` with zero gate failures.
3. `validate_reference.py --against <committed manifest> --dir <new run>` must
   pass every artifact (a drift is an incident: stop and investigate, e.g. a
   CPU/glibc/torch difference).
4. Replace the directory contents **only** in a deliberate commit: copy the
   `.bin` files + `manifest.json` + `manifest.sha256`, update
   `PINNED_MANIFEST_SHA256` in `tests/m10_reference.rs` to the new
   `sha256sum manifest.json`, and record both digests plus the environment
   fingerprint in the audit report.
5. `cargo test -p claymore-model-gemma --test m10_reference` must be green.

## Scenario (pinned, do not vary silently)

Prompt `Sail across the silver sea` (7 tokens); denoising step 0; canvas
`torch.randn` under `torch.manual_seed(42)`, 16 tokens; pixels
`torch.randn(1, 9, 3, 16, 16)` under `torch.manual_seed(1234)`; vision
position ids a 3×3 x-major grid (factor 4 → one 2×2 pooling group); decoder
positions `arange(7, 23)`. Model: `dtype=bfloat16`, CPU, `attn_implementation
="eager"`, `experts_implementation="eager"`, `low_cpu_mem_usage=True` — the
pinned upstream defaults for this model, verified bit-reproducible across
thread counts.
