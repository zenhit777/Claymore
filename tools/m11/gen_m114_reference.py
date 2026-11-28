#!/usr/bin/env python3
"""Generate the M11.4 upstream differential fixture from the pinned checkpoint.

The reference values are produced by the immutable upstream implementation
(`modeling_clover_moe.py` at the pinned revision) running on CPU with the real
release weights. Nothing here reuses Claymore's Rust arithmetic: the fixture
is an external observation of the upstream model, and the Rust side only
compares against it.

The script never touches CUDA: the model is loaded on CPU in BF16 and the
forward passes run under `torch.inference_mode()`.

Three upstream passes are recorded, mirroring the three Claymore passes of the
differential test:

1. `preview`  : one bidirectional block over the prompt, no clean state.
2. `prefill`  : the causal prefill of the same prompt.
3. `continue` : one causal token appended to the prefill state.

Usage:
    python3 tools/m11/gen_m114_reference.py \
        --checkpoint /path/to/Clover-35B-A3B-Preview-4b0402b \
        --out docs/audits/fixtures/m114-upstream-reference.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import torch

SOURCE_REVISION = "4b0402b65e523d0183a14c431955438714cb16ff"
SOURCE_FILE = "modeling_clover_moe.py"
SOURCE_SHA256 = "2584bd5502d6109c2cea9bee2f2ec98f147bbb8954c43858c0315eb4af9ffb0c"
SAMPLING_FILE = "sampling_clover_moe.py"
SAMPLING_SHA256 = "56768a0b23c4c7c26dc3b6f075741761ffada7d5fc2ff745c6aea5e3cc88011c"

PROMPT_TOKEN_IDS = [151_644, 872, 198, 9_707]
CONTINUATION_TOKEN_ID = 9_707

# Sampling strides keep the fixture reviewable while still covering every
# tensor kind with a deterministic, reproducible subset of elements.
HIDDEN_STRIDE = 64
ROUTER_STRIDE = 16
CONV_STATE_STRIDE = 512
SSM_STATE_STRIDE = 16_384
KV_STRIDE = 64
TOP_LOGITS = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_vector(values: torch.Tensor, stride: int) -> dict:
    """Summarize a 1-D tensor with strided samples and stable aggregates."""
    flat = values.detach().to(torch.float64).reshape(-1)
    indices = list(range(0, flat.numel(), stride))
    # The sample positions are `i * stride`; the stride is recorded once in the
    # fixture header, so the indices are derivable and not duplicated per row.
    return {
        "length": int(flat.numel()),
        "sample_values": [float(flat[index]) for index in indices],
        "absmax": float(flat.abs().max()),
        "sum_squares": float(flat.square().sum()),
    }


def squeeze_batch(values: torch.Tensor) -> torch.Tensor:
    """Drop the batch dimension of a [batch, ...] upstream tensor."""
    detached = values.detach()
    return detached[0] if detached.dim() == 3 else detached


def sample_matrix(values: torch.Tensor, stride: int) -> dict:
    """Summarize a [tokens, features] tensor row by row."""
    return {
        "rows": [sample_vector(row, stride) for row in squeeze_batch(values)]
    }


def top_logits(values: torch.Tensor, count: int = TOP_LOGITS) -> dict:
    row = values.detach().to(torch.float64).reshape(-1)
    order = torch.argsort(row, descending=True, stable=True)[:count]
    return {
        "ids": [int(index) for index in order],
        "values": [float(row[index]) for index in order],
        "argmax": int(torch.argmax(row)),
        "max": float(row.max()),
        "absmax": float(row.abs().max()),
        "sum_squares": float(row.square().sum()),
    }


def routing_from_logits(logits: torch.Tensor, top_k: int, coreset_top_p: float,
                        valid: torch.Tensor, preview: bool) -> dict:
    """Reproduce the upstream routing decision for one layer.

    `CloverMoeBlock.forward` is the source of truth; this mirrors its three
    steps (coreset masking, top-k selection, weight normalization) so the
    fixture records the effective expert set and weights per token.
    """
    from upstream_clover.modeling_clover_moe import coreset_torch

    row_logits = logits.detach().to(torch.float32)
    if row_logits.dim() == 2:
        row_logits = row_logits.unsqueeze(0)
    if preview:
        row_logits = coreset_torch(row_logits, valid, top_k, coreset_top_p)
    probabilities = row_logits.softmax(-1)
    ids = torch.argsort(row_logits, dim=-1, descending=True, stable=True)[..., :top_k]
    weights = probabilities.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return {
        "expert_ids": [[int(value) for value in row] for row in ids[0]],
        "weights": [[float(value) for value in row] for row in weights[0]],
    }


class LayerCapture:
    """Collect per-layer stage outputs of one forward pass."""

    def __init__(self, model, config) -> None:
        self.config = config
        self.reset()
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook(index)))
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
            attention = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            self.handles.append(
                attention.register_forward_hook(self._attention_hook(index))
            )
            self.handles.append(layer.mlp.register_forward_hook(self._moe_hook(index)))
            self.handles.append(
                layer.mlp.gate.register_forward_hook(self._router_hook(index))
            )
        self.handles.append(
            model.model.norm.register_forward_hook(self._final_norm_hook())
        )

    def reset(self) -> None:
        self.inputs: dict[int, torch.Tensor] = {}
        self.layer_outputs: dict[int, torch.Tensor] = {}
        self.attention_outputs: dict[int, torch.Tensor] = {}
        self.moe_outputs: dict[int, torch.Tensor] = {}
        self.router_logits: dict[int, torch.Tensor] = {}
        self.final_norm: dict[int, torch.Tensor] = {}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _pre_hook(self, index):
        def hook(_module, args):
            self.inputs[index] = args[0].detach()

        return hook

    def _layer_hook(self, index):
        def hook(_module, _args, output):
            self.layer_outputs[index] = output[0].detach()

        return hook

    def _attention_hook(self, index):
        def hook(_module, _args, output):
            self.attention_outputs[index] = output[0].detach()

        return hook

    def _moe_hook(self, index):
        def hook(_module, _args, output):
            self.moe_outputs[index] = output.detach()

        return hook

    def _router_hook(self, index):
        def hook(_module, _args, output):
            self.router_logits[index] = output.detach()

        return hook

    def _final_norm_hook(self):
        def hook(_module, _args, output):
            self.final_norm[0] = output.detach()

        return hook

    def layer_fixtures(self, valid: torch.Tensor, preview: bool) -> list[dict]:
        fixtures = []
        for index in sorted(self.layer_outputs):
            attention = self.attention_outputs[index]
            residual = self.inputs[index] + attention
            fixtures.append(
                {
                    "layer": index,
                    "layer_type": self.config.layer_types[index],
                    "attention_output": sample_matrix(attention, HIDDEN_STRIDE),
                    "residual_after_attention": sample_matrix(residual, HIDDEN_STRIDE),
                    "moe_output": sample_matrix(self.moe_outputs[index], HIDDEN_STRIDE),
                    "layer_output": sample_matrix(self.layer_outputs[index], HIDDEN_STRIDE),
                    "router_logits": sample_matrix(self.router_logits[index], ROUTER_STRIDE),
                    "routing": routing_from_logits(
                        self.router_logits[index],
                        self.config.num_experts_per_tok,
                        self.config.moe_block_coreset_top_p,
                        valid,
                        preview,
                    ),
                }
            )
        return fixtures


def state_fixtures(cache, config) -> list[dict]:
    states = []
    for index, layer_type in enumerate(config.layer_types):
        entry = {"layer": index, "layer_type": layer_type}
        state = cache.layers[index]
        if layer_type == "linear_attention":
            entry["conv_state"] = sample_vector(state["conv"].reshape(-1), CONV_STATE_STRIDE)
            entry["ssm_state"] = sample_vector(state["ssm"].reshape(-1), SSM_STATE_STRIDE)
        else:
            # The backend stores `[token, head, head_dim]`; upstream stores
            # `[batch, head, token, head_dim]`, so the head/token axes are
            # transposed and flattened per token before sampling.
            entry["key"] = sample_matrix(
                state["key"][0].transpose(0, 1).reshape(state["key"].shape[2], -1), KV_STRIDE
            )
            entry["value"] = sample_matrix(
                state["value"][0].transpose(0, 1).reshape(state["value"].shape[2], -1), KV_STRIDE
            )
        states.append(entry)
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.checkpoint.resolve()
    if not root.is_dir():
        print(f"checkpoint directory not found: {root}", file=sys.stderr)
        return 2

    staging = Path(tempfile.mkdtemp(prefix="claymore-m114-upstream-"))
    try:
        package = staging / "upstream_clover"
        package.mkdir()
        for name in (
            "configuration_clover_moe.py",
            "modeling_clover_moe.py",
            "sampling_clover_moe.py",
        ):
            shutil.copy2(root / name, package / name)
        (package / "__init__.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(staging))

        from upstream_clover.configuration_clover_moe import CloverMoeConfig
        from upstream_clover.modeling_clover_moe import CloverMoeModelLM

        if sha256_file(root / SOURCE_FILE) != SOURCE_SHA256:
            print("upstream modeling source digest mismatch", file=sys.stderr)
            return 3
        if sha256_file(root / SAMPLING_FILE) != SAMPLING_SHA256:
            print("upstream sampling source digest mismatch", file=sys.stderr)
            return 3

        config = CloverMoeConfig.from_pretrained(str(root))
        # Measured, not declared: the fixture records whether this process ever
        # touched a CUDA context, which is the host-side evidence that the CPU
        # reference genuinely ran without one.
        print(f"loading upstream model on CPU (bfloat16) from {root} "
              f"[cuda_initialized_before_load={torch.cuda.is_initialized()}]", flush=True)
        model = CloverMoeModelLM.from_pretrained(
            str(root),
            config=config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.eval()
        model.to("cpu")
        print("model loaded", flush=True)

        capture = LayerCapture(model, config)
        token_ids = torch.tensor([PROMPT_TOKEN_IDS], dtype=torch.long)
        valid = torch.ones_like(token_ids, dtype=torch.bool)

        with torch.inference_mode():
            preview = model(
                input_ids=token_ids,
                attention_mask=valid,
                past_key_values=None,
                use_cache=True,
                preview=True,
            )
        preview_fixture = {
            "layers": capture.layer_fixtures(valid, preview=True),
            "final_hidden": sample_matrix(capture.final_norm[0], HIDDEN_STRIDE),
            "logits": [top_logits(row) for row in preview.logits[0].detach()],
        }

        capture.reset()
        with torch.inference_mode():
            prefill = model(
                input_ids=token_ids,
                attention_mask=valid,
                past_key_values=None,
                use_cache=True,
                preview=False,
            )
        prefill_fixture = {
            "layers": capture.layer_fixtures(valid, preview=False),
            "states": state_fixtures(prefill.past_key_values, config),
        }

        capture.reset()
        continuation = torch.tensor([[CONTINUATION_TOKEN_ID]], dtype=torch.long)
        with torch.inference_mode():
            continued = model(
                input_ids=continuation,
                attention_mask=torch.ones((1, len(PROMPT_TOKEN_IDS) + 1), dtype=torch.bool),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                preview=False,
            )
        continuation_fixture = {
            "layers": capture.layer_fixtures(
                torch.ones((1, 1), dtype=torch.bool), preview=False
            ),
            "states": state_fixtures(continued.past_key_values, config),
        }

        fixture = {
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "sampling_source_sha256": SAMPLING_SHA256,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_used": torch.cuda.is_initialized(),
            "config_sha256": sha256_file(root / "config.json"),
            "index_sha256": sha256_file(root / "model.safetensors.index.json"),
            "prompt_token_ids": PROMPT_TOKEN_IDS,
            "continuation_token_id": CONTINUATION_TOKEN_ID,
            "strides": {
                "hidden": HIDDEN_STRIDE,
                "router": ROUTER_STRIDE,
                "conv_state": CONV_STATE_STRIDE,
                "ssm_state": SSM_STATE_STRIDE,
                "kv": KV_STRIDE,
            },
            "preview": preview_fixture,
            "prefill": prefill_fixture,
            "continuation": continuation_fixture,
        }

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}", flush=True)
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
