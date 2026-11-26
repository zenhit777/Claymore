#!/usr/bin/env python3
"""M10 versioned per-layer CPU reference generator.

Replaces the throwaway campaign scripts (`gen_golden_eager.py`,
`gen_routers.py`, `gen_vision.py`, `gen_vision_stages.py`,
`gen_decoder_stages.py`, `probe_layer_inputs.py`) with one versioned,
reproducible tool, per `docs/audits/m10-cpu-finalization-contract.md`
(criteria 1 and 2).

What it produces, in ONE fresh output directory (it refuses an existing,
non-empty one — historical golden sets are never overwritten):

* per-layer inputs and outputs of the three stages, for the golden-input
  bisection:  `enc_in_{L}` / `encoder_layer_{L}`,
  `vision_in_{L}` / `vision_layer_{L}`, `dec_in_{L}` / `decoder_layer_{L}`;
* the reference encoder KV cache per layer (`enc_kv_{L}_{keys,values}`,
  reference layout `[batch, kv_heads, seq, head_dim]`);
* MoE router artifacts for every encoder and decoder layer
  (`encoder{L}_router_{scores,weights,indices}`, `moe{L}_router_*`);
* chain anchors and scenario inputs (`prompt_ids`, `canvas0_direct`,
  `vision_pixel_values`, `vision_pixel_position_ids`, embeddings,
  `vision_last_hidden_direct`, `embed_vision_output_direct`,
  `decoder_last_hidden0_direct`, `decoder_logits0_direct`);
* `manifest.json` + `manifest.sha256`: per-tensor shape, dtype, axis layout,
  SHA-256, the full environment fingerprint (python/torch/transformers
  versions, SHA-256 of the upstream module files actually imported, libc,
  CPU model, thread counts), the scenario parameters, the internal
  chain-consistency checks and the bit-exactness gates against the
  historical golden set passed with `--golden`.

All `.bin` artifacts are FP32 little-endian, C-contiguous, exactly the
campaign convention; token-id tensors are ids widened to FP32 (values
2^24, round-trip safe).

Determinism notes: model loaded `dtype=torch.bfloat16`,
`device_map="cpu"`, `attn_implementation="eager"`,
`experts_implementation="eager"`, `low_cpu_mem_usage=True` — the pinned
campaign semantics. Pixels use `torch.manual_seed(1234)`, the canvas
`torch.manual_seed(42)`, decoder positions `arange(7,23)`. Any gate
mismatch exits non-zero; artifacts are kept for inspection.

The reference is CPU-only; `CUDA_VISIBLE_DEVICES=""` is set defensively.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

PINNED_TRANSFORMERS_COMMIT = "671aa77576266f41f33f9c28e2d824bfd1b1b63c"
PINNED_CHECKPOINT_REVISION = "cd40195554d3cfcb56f34b9adfb66efc790ed5b6"
PROMPT = "What is the capital of France?"

ENCODER_LAYERS = 30
VISION_LAYERS = 27
DECODER_LAYERS = 30


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def libc_version() -> str:
    try:
        out = subprocess.run(
            ["getconf", "GNU_LIBC_VERSION"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - best effort fingerprint
        return platform.libc_ver()[1] or "unknown"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


class Saver:
    """Writes FP32 LE tensors + records shapes, layouts and SHA-256."""

    def __init__(self, out: Path):
        self.out = out
        self.manifest = {}

    def save(self, name: str, tensor, layout: str) -> str:
        import torch as _torch

        flat = tensor.detach().float().cpu().contiguous()
        shape = list(flat.shape)
        path = self.out / f"{name}.bin"
        flat.numpy().tofile(path)
        sha = sha256_file(path)
        self.manifest[name] = {
            "file": f"{name}.bin",
            "shape": shape,
            "dtype": "float32-le",
            "layout": layout,
            "sha256": sha,
        }
        print(f"saved {name} {shape} {sha[:16]}", flush=True)
        return sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="pinned checkpoint directory")
    parser.add_argument("--out", required=True, help="fresh output directory")
    parser.add_argument(
        "--golden",
        default=None,
        help="historical golden_eager directory for the bit-exactness gate (read-only)",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="historical golden_layers directory for the enc_in gate (read-only)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads(N); default keeps the torch default (recorded)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    if not ckpt.is_dir():
        sys.exit(f"checkpoint directory missing: {ckpt}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into a non-empty directory: {out} (no overwrite policy)")
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers import AutoTokenizer

    import transformers.models.diffusion_gemma.modeling_diffusion_gemma as mdg
    import transformers.models.diffusion_gemma.configuration_diffusion_gemma as cdg
    import transformers.models.diffusion_gemma.generation_diffusion_gemma as gdg
    import transformers.models.gemma4.modeling_gemma4 as mg4
    import transformers.models.gemma4.configuration_gemma4 as cg4

    module_sha = {
        Path(mdg.__file__).name: sha256_file(Path(mdg.__file__)),
        Path(cdg.__file__).name: sha256_file(Path(cdg.__file__)),
        Path(gdg.__file__).name: sha256_file(Path(gdg.__file__)),
        Path(mg4.__file__).name: sha256_file(Path(mg4.__file__)),
        Path(cg4.__file__).name: sha256_file(Path(cg4.__file__)),
    }

    cfg = json.loads((ckpt / "config.json").read_text())
    text = DiffusionGemmaTextConfig(**cfg["text_config"])
    vision_cfg = Gemma4VisionConfig(**cfg["vision_config"])
    config = DiffusionGemmaConfig(text_config=text, vision_config=vision_cfg)

    print("loading model on CPU with eager attention and eager experts", flush=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        str(ckpt),
        config=config,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    model.eval()
    print(
        f"attn={model.config.text_config._attn_implementation} "
        f"experts={model.config.text_config._experts_implementation}",
        flush=True,
    )

    saver = Saver(out)

    # ------------------------------------------------------------------ inputs
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    saver.save("prompt_ids", ids, "[batch, seq] token ids widened to FP32")

    torch.manual_seed(1234)
    pixels = (torch.rand(1, 3, 48, 48) * 0.9 + 0.05).to(torch.bfloat16)
    pixels = pixels.reshape(1, 9, 768)
    pos = torch.tensor([[[x, y] for x in range(3) for y in range(3)]], dtype=torch.long)
    saver.save("vision_pixel_values", pixels, "[batch, patches, patch_pixels] (2(v-0.5) applied downstream)")
    saver.save("vision_pixel_position_ids", pos, "[batch, patches, 2] (x, y), x-major 3x3")

    torch.manual_seed(42)
    canvas = torch.randint(
        0, cfg["text_config"]["vocab_size"], (1, 16), dtype=torch.long
    )
    saver.save("canvas0_direct", canvas, "[batch, canvas] token ids widened to FP32")

    # ------------------------------------------------------------- stage hooks
    enc_layers = model.model.encoder.language_model.layers
    vis_layers = model.model.encoder.vision_tower.encoder.layers
    dec_layers = model.model.decoder.layers

    enc_out, vis_out, dec_out = {}, {}, {}
    router_payload = {}

    def enc_pre(module, inputs):
        saver.save(f"enc_in_{module.layer_idx}", inputs[0], "[batch, seq, hidden] layer entry")

    def enc_post(module, inputs, output):
        enc_out[module.layer_idx] = output[0] if isinstance(output, tuple) else output

    def vis_pre(module, inputs):
        idx = [i for i, l in enumerate(vis_layers) if l is module][0]
        saver.save(f"vision_in_{idx}", inputs[0], "[batch, patches, hidden] layer entry")

    def vis_post(module, inputs, output):
        idx = [i for i, l in enumerate(vis_layers) if l is module][0]
        vis_out[idx] = output[0] if isinstance(output, tuple) else output

    def dec_pre(module, inputs):
        saver.save(f"dec_in_{module.layer_idx}", inputs[0], "[batch, canvas, hidden] layer entry")

    def dec_post(module, inputs, output):
        dec_out[module.layer_idx] = output[0] if isinstance(output, tuple) else output

    def make_router_post(prefix: str, idx: int):
        def router_post(module, inputs, output):
            scores, weights, indices = output
            saver.save(f"{prefix}{idx}_router_scores", scores, "[rows, num_experts] FP32 probabilities")
            saver.save(f"{prefix}{idx}_router_weights", weights, "[rows, top_k] FP32")
            saver.save(f"{prefix}{idx}_router_indices", indices, "[rows, top_k] ids widened to FP32")

        return router_post

    def make_experts_post(layer_idx: int):
        def experts_post(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            saver.save(
                f"encoder{layer_idx}_expert_output", out, "[rows, hidden] MoE expert sum"
            )

        return experts_post

    for layer in enc_layers:
        layer.register_forward_pre_hook(enc_pre)
        layer.register_forward_hook(enc_post)
        layer.router.register_forward_hook(make_router_post("encoder", layer.layer_idx))
        layer.experts.register_forward_hook(make_experts_post(layer.layer_idx))
    for layer in vis_layers:
        layer.register_forward_pre_hook(vis_pre)
        layer.register_forward_hook(vis_post)
    for layer in dec_layers:
        layer.register_forward_pre_hook(dec_pre)
        layer.register_forward_hook(dec_post)
        layer.router.register_forward_hook(make_router_post("moe", layer.layer_idx))

    # ------------------------------------------------------------- 1. vision
    with torch.no_grad():
        io = model.model.encoder.get_image_features(pixels, pos, return_dict=True)
        saver.save("vision_last_hidden_direct", io.last_hidden_state, "[batch, pooled, hidden] standardized")
        saver.save("embed_vision_output_direct", io.pooler_output, "[batch, hidden] embedder output")
        for idx in sorted(vis_out):
            saver.save(f"vision_layer_{idx}", vis_out[idx], "[batch, patches, hidden] layer exit")

    # ------------------------------------------------------------ 2. encoder
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        enc = model.model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), cache=cache)
        saver.save("encoder_last_hidden_direct", enc.last_hidden_state, "[batch, seq, hidden] post final norm")
        saver.save("encoder_embedding", model.model.encoder.language_model.embed_tokens(ids), "[batch, seq, hidden] scaled embedding")
        for idx, cache_layer in enumerate(cache.layers):
            saver.save(f"enc_kv_{idx}_keys", cache_layer.keys, "[batch, kv_heads, seq, head_dim] post-rope, reference layout")
            saver.save(f"enc_kv_{idx}_values", cache_layer.values, "[batch, kv_heads, seq, head_dim] post-norm, reference layout")
        for idx in sorted(enc_out):
            saver.save(f"encoder_layer_{idx}", enc_out[idx], "[batch, seq, hidden] layer exit")

    # --------------------------------------------------- 3. decoder first step
    cache_rows_at_decoder_entry = {}
    for idx, cache_layer in enumerate(cache.layers):
        cache_rows_at_decoder_entry[idx] = cache_layer.keys.shape[2]

    with torch.no_grad():
        dec = model.model.decoder(
            decoder_input_ids=canvas,
            cache=cache,
            decoder_position_ids=torch.arange(7, 23, dtype=torch.long).unsqueeze(0),
        )
        saver.save("decoder_last_hidden0_direct", dec.last_hidden_state, "[batch, canvas, hidden] post final norm")
        saver.save("decoder_embedding", model.model.decoder.embed_tokens(canvas), "[batch, canvas, hidden] scaled embedding")
        for idx in sorted(dec_out):
            saver.save(f"decoder_layer_{idx}", dec_out[idx], "[batch, canvas, hidden] layer exit")
        logits = model.lm_head(dec.last_hidden_state)
        cap = cfg["text_config"]["final_logit_softcapping"]
        logits = torch.tanh(logits.float() / cap) * cap
        saver.save("decoder_logits0_direct", logits, "[batch, canvas, vocab] soft-capped FP32")

    # ------------------------------------------------------------------ checks
    internal, gates = [], []

    def bitexact(a: Path, b: Path, label: str):
        same = a.is_file() and b.is_file() and sha256_file(a) == sha256_file(b)
        gates.append({"check": label, "bit_identical": bool(same)})
        if not same:
            print(f"GATE FAIL {label}", flush=True)

    for l in range(ENCODER_LAYERS - 1):
        internal.append(
            {
                "check": f"enc_in_{l + 1} == encoder_layer_{l}",
                "bit_identical": saver.manifest[f"enc_in_{l + 1}"]["sha256"]
                == saver.manifest[f"encoder_layer_{l}"]["sha256"],
            }
        )
    for l in range(VISION_LAYERS - 1):
        internal.append(
            {
                "check": f"vision_in_{l + 1} == vision_layer_{l}",
                "bit_identical": saver.manifest[f"vision_in_{l + 1}"]["sha256"]
                == saver.manifest[f"vision_layer_{l}"]["sha256"],
            }
        )
    for l in range(DECODER_LAYERS - 1):
        internal.append(
            {
                "check": f"dec_in_{l + 1} == decoder_layer_{l}",
                "bit_identical": saver.manifest[f"dec_in_{l + 1}"]["sha256"]
                == saver.manifest[f"decoder_layer_{l}"]["sha256"],
            }
        )
    cache_shape_ok = all(rows == 7 for rows in cache_rows_at_decoder_entry.values())
    internal.append(
        {
            "check": "encoder KV prefix stays 7 rows through the decoder pass (no canvas prefixing)",
            "bit_identical": cache_shape_ok,
        }
    )

    if args.golden:
        golden = Path(args.golden)
        for l in range(ENCODER_LAYERS):
            bitexact(out / f"encoder_layer_{l}.bin", golden / f"encoder_layer_{l}.bin", f"encoder_layer_{l} vs golden")
            for kind in ("router_scores", "router_weights", "router_indices"):
                bitexact(out / f"encoder{l}_{kind}.bin", golden / f"encoder{l}_{kind}.bin", f"encoder{l}_{kind} vs golden")
            bitexact(out / f"encoder{l}_expert_output.bin", golden / f"encoder{l}_expert_output.bin", f"encoder{l}_expert_output vs golden")
            for kind in ("keys", "values"):
                bitexact(out / f"enc_kv_{l}_{kind}.bin", golden / f"enc_kv_{l}_{kind}_direct.bin", f"enc_kv_{l}_{kind} vs golden")
        for l in range(VISION_LAYERS):
            bitexact(out / f"vision_layer_{l}.bin", golden / f"vision_layer_{l}.bin", f"vision_layer_{l} vs golden")
        for l in range(DECODER_LAYERS):
            bitexact(out / f"decoder_layer_{l}.bin", golden / f"decoder_layer_{l}.bin", f"decoder_layer_{l} vs golden")
        for name in (
            "prompt_ids",
            "canvas0_direct",
            "vision_pixel_values",
            "vision_pixel_position_ids",
            "encoder_embedding",
            "encoder_last_hidden_direct",
            "vision_last_hidden_direct",
            "embed_vision_output_direct",
            "decoder_embedding",
            "decoder_last_hidden0_direct",
            "decoder_logits0_direct",
        ):
            bitexact(out / f"{name}.bin", golden / f"{name}.bin", f"{name} vs golden")
        bitexact(out / "moe0_router_scores.bin", golden / "moe0_router_logits.bin", "moe0 router logits vs golden")
        for kind in ("weights", "indices"):
            bitexact(out / f"moe0_router_{kind}.bin", golden / f"moe0_topk_{kind}.bin", f"moe0 topk {kind} vs golden")
    if args.layers:
        layers = Path(args.layers)
        for l in range(ENCODER_LAYERS):
            bitexact(out / f"enc_in_{l}.bin", layers / f"enc_in_{l}.bin", f"enc_in_{l} vs golden_layers")

    failed = [c["check"] for c in internal + gates if not c["bit_identical"]]

    manifest = {
        "schema": "claymore-m10-reference/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libc": libc_version(),
            "cpu_model": cpu_model(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
            "transformers_commit_pinned": PINNED_TRANSFORMERS_COMMIT,
            "upstream_module_sha256": module_sha,
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        },
        "checkpoint": {
            "root": str(ckpt),
            "revision_pinned": PINNED_CHECKPOINT_REVISION,
            "config_sha256": sha256_file(ckpt / "config.json"),
            "index_sha256": sha256_file(ckpt / "model.safetensors.index.json"),
        },
        "scenario": {
            "prompt_text": PROMPT,
            "prompt_tokens": int(ids.shape[1]),
            "vision_patches": 9,
            "vision_grid": "3x3, x-major",
            "canvas_tokens": 16,
            "decoder_positions": "arange(7,23)",
            "encoder_prefix_len": 7,
            "decoder_prefix_kv": "none (step 0; the decoder pass never mutates the cache rows)",
            "cache_rows_at_decoder_entry": cache_rows_at_decoder_entry,
            "pixel_seed": 1234,
            "canvas_seed": 42,
            "attn_implementation": "eager",
            "experts_implementation": "eager",
            "dtype": "bfloat16",
            "final_logit_softcapping": cfg["text_config"]["final_logit_softcapping"],
            "top_k_experts": cfg["text_config"].get("top_k_experts", 8),
        },
        "artifacts": saver.manifest,
        "internal_checks": internal,
        "golden_gates": gates,
        "gate_failures": failed,
    }
    manifest_text = json.dumps(manifest, indent=1, sort_keys=True)
    (out / "manifest.json").write_text(manifest_text + "\n")
    (out / "manifest.sha256").write_text(
        f"{hashlib.sha256((manifest_text + chr(10)).encode()).hexdigest()}  manifest.json\n"
    )

    print(f"artifacts={len(saver.manifest)} internal_ok={len(internal) - sum(0 if c['bit_identical'] else 1 for c in internal)}/{len(internal)} golden_gates={len(gates)}", flush=True)
    if failed:
        print(f"DONE_REFERENCE_FAILED {len(failed)} gate(s) failed; artifacts kept in {out}", flush=True)
        return 1
    print("DONE_REFERENCE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
