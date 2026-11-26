#!/usr/bin/env python3
"""Generate tiny CPU-only differential fixtures from the pinned Clover formulas.

The reference equations below are transcribed from the immutable upstream
`modeling_clover_moe.py` revision recorded in the manifest. No checkpoint is
loaded and no CUDA API/device is touched. Output is refused if the target
folder is non-empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_REVISION = "4b0402b65e523d0183a14c431955438714cb16ff"
SOURCE_FILE = "modeling_clover_moe.py"
SOURCE_SHA256 = "2584bd5502d6109c2cea9bee2f2ec98f147bbb8954c43858c0315eb4af9ffb0c"
SEED = 20261126


def recurrent_commit_torch(q, k, v, g, beta, state, valid):
    # Pinned upstream lines 452-468.
    h = v.shape[2]
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(h // q.shape[2], 2) / q.shape[-1] ** 0.5
    k = k.repeat_interleave(h // k.shape[2], 2)
    state = state.float()
    out = []
    for t in range(v.shape[1]):
        decayed = state * g[:, t].float().exp()[..., None, None]
        remembered = torch.einsum("bhvk,bhk->bhv", decayed, k[:, t])
        delta = (v[:, t].float() - remembered) * beta[:, t].float()[..., None]
        updated = decayed + delta[..., None] * k[:, t, :, None, :]
        state = torch.where(valid[:, t, None, None, None], updated, state)
        y = torch.einsum("bhvk,bhk->bhv", state, q[:, t])
        out.append(y * valid[:, t, None, None])
    return torch.stack(out, dim=1).to(v.dtype), state


def bd_gdn_conv1d(mixed_qkv, clean_conv_states, conv_weights, block_size, active_valid_mask, conv_right_weight):
    # Pinned upstream lines 253-317, batch size one fixture.
    num_tokens, channels = mixed_qkv.shape
    batch_size = clean_conv_states.shape[0]
    assert num_tokens == batch_size * block_size
    history = clean_conv_states
    valid = active_valid_mask.reshape(batch_size, block_size)
    current = mixed_qkv.reshape(batch_size, block_size, channels)
    current = current * valid[:, :, None].to(current.dtype)
    convolution_input = torch.cat((history, current.transpose(1, 2)), dim=-1)
    weight = conv_weights.reshape(channels, 1, -1)
    output = F.conv1d(convolution_input.float(), weight.float(), groups=channels)
    width = weight.shape[-1]
    for distance in range(1, min(width, block_size)):
        output[:, :, :-distance] += (
            current[:, distance:, :].transpose(1, 2).float()
            * conv_right_weight[:, 0, distance - 1].float()[None, :, None]
        )
    output = F.silu(output)
    output = output.transpose(1, 2) * valid[:, :, None].to(output.dtype)
    return output.reshape(num_tokens, channels).to(mixed_qkv.dtype)


def bd_gdn_local_memory(query, key, value, g, clean_ssm_states, active_valid_mask,
                        scale, softmax_logit_scale, softmax_history_bias, softmax_history_c):
    # Pinned upstream lines 320-421, batch size one fixture.
    num_tokens = query.shape[1]
    batch_size = clean_ssm_states.shape[0]
    block_size = num_tokens // batch_size
    num_value_heads = value.shape[2]
    query = query.repeat_interleave(num_value_heads // query.shape[2], dim=2)
    key = key.repeat_interleave(num_value_heads // key.shape[2], dim=2)
    valid = active_valid_mask.reshape(batch_size, block_size)
    valid_float = valid[:, :, None].float()
    query_float = query.reshape(batch_size, block_size, num_value_heads, query.shape[-1]).float()
    key_float = key.reshape(batch_size, block_size, num_value_heads, key.shape[-1]).float()
    value_float = value.reshape(batch_size, block_size, num_value_heads, value.shape[-1]).float()
    query_float = query_float * torch.rsqrt(query_float.square().sum(dim=-1, keepdim=True) + 1e-6)
    key_float = key_float * torch.rsqrt(key_float.square().sum(dim=-1, keepdim=True) + 1e-6)
    clean_state = clean_ssm_states.float()
    scaled_query = query_float * scale
    clean_output = torch.einsum("blhk,bhvk->blhv", scaled_query, clean_state)
    decay = g.reshape(batch_size, block_size, num_value_heads).float().exp()
    history = clean_output * decay[..., None]
    logit_scale = softmax_logit_scale.reshape(-1).float()
    history_bias = softmax_history_bias.reshape(-1).float()
    history_c = softmax_history_c.reshape(-1).float()
    logits = torch.einsum("blhk,bshk->bhls", query_float, key_float)
    logits = logits * logit_scale.view(1, -1, 1, 1)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    history_logit = history_bias.view(1, -1, 1, 1).expand(batch_size, -1, block_size, 1)
    valid_block_size = valid.sum(dim=-1).clamp_min(1).float()
    block_size_log_ratio = torch.log(valid_block_size / 32.0)
    history_logit = history_logit + history_c.view(1, -1, 1, 1) * block_size_log_ratio.view(batch_size, 1, 1, 1)
    weights = torch.softmax(torch.cat((logits, history_logit), dim=-1), dim=-1)
    local_output = torch.einsum(
        "bhls,bshv->blhv", weights[..., :-1], value_float * valid_float[..., None]
    )
    history_weight = weights[..., -1].permute(0, 2, 1)
    output = history_weight[..., None] * history + local_output
    output = output * valid_float[..., None]
    return output.to(query.dtype).reshape(1, num_tokens, num_value_heads, value.shape[-1])


def causal_conv_invalid_torch(raw, history, weight, valid):
    # Pinned upstream lines 531-542: input-dtype products, FP32 sum, SiLU.
    pieces = []
    next_history = history.clone()
    for token in range(raw.shape[1]):
        window = torch.cat((next_history, raw[:, token, :, None]), -1)
        y = F.silu((window * weight[:, 0]).float().sum(-1)).to(raw.dtype)
        pieces.append(y * valid[:, token, None])
        next_history = torch.where(valid[:, token, None, None], window[..., 1:], next_history)
    return torch.stack(pieces, 1), next_history


def rms_norm(x, weight, eps, zero_centered=True, gate=None):
    # Pinned upstream lines 433-449.
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    effective_weight = 1 + weight.float() if zero_centered else weight.float()
    y = y * effective_weight
    if gate is not None:
        y = y * F.silu(gate.float())
    return y.to(x.dtype)


def rotary_partial(x, positions, theta, partial_factor):
    # Pinned upstream CloverAttention.rotary, lines 597-608.
    rotary_dim = int(x.shape[-1] * partial_factor)
    inv = 1.0 / theta ** (
        torch.arange(0, rotary_dim, 2, device="cpu", dtype=torch.float32) / rotary_dim
    )
    phase = positions.float()[..., None] * inv
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    first, second = x[..., :rotary_dim].chunk(2, -1)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
    return torch.cat((rotated, x[..., rotary_dim:]), -1), cos.squeeze(1), sin.squeeze(1)


def attention_eager(q, k, v, gate, allow):
    # Pinned CloverAttention eager branch, lines 640-650.
    query_heads = q.shape[1]
    kv_heads = k.shape[1]
    k = k.repeat_interleave(query_heads // kv_heads, 1)
    v = v.repeat_interleave(query_heads // kv_heads, 1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    scores = scores.masked_fill(~allow, -float("inf"))
    probs = torch.nan_to_num(scores.softmax(-1)).to(v.dtype)
    attended = probs @ v
    gated = attended.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    gated = gated * gate.reshape(q.shape[0], q.shape[2], -1).sigmoid()
    return attended.transpose(1, 2), gated


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def rust_const(name: str, tensor) -> str:
    values = tensor.detach().to(torch.float32).cpu().contiguous().reshape(-1).tolist()
    entries = [f"f32::from_bits(0x{f32_bits(value):08x})" for value in values]
    return f"pub const {name}: &[f32] = &[\n    " + ",\n    ".join(entries) + ",\n];"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="fresh output directory")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    fixtures = {}

    def add(name, value):
        assert value.device.type == "cpu", f"{name} unexpectedly left CPU"
        fixtures[name] = value.detach().to(torch.float32).cpu().contiguous()

    # GDN recurrent commit (BF16 activations, FP32 persistent state).
    q = torch.tensor([[[[0.3, -0.7]], [[1.1, 0.2]], [[-0.4, 0.9]]]], dtype=torch.bfloat16)
    k = torch.tensor([[[[-0.2, 0.8]], [[0.5, -0.3]], [[1.0, 0.4]]]], dtype=torch.bfloat16)
    v = torch.tensor([[[[0.2, 0.7], [-0.5, 0.3]], [[0.8, -0.1], [0.4, 0.9]], [[-0.6, 0.2], [0.1, -0.8]]]], dtype=torch.bfloat16)
    g = torch.tensor([[[ -0.2, -0.1], [-0.4, -0.3], [-0.1, -0.25]]], dtype=torch.float32)
    beta = torch.tensor([[[0.6, 0.4], [0.7, 0.5], [0.3, 0.8]]], dtype=torch.bfloat16)
    state = torch.tensor([[[[0.1, -0.2], [0.3, 0.05]], [[-0.15, 0.25], [0.2, -0.1]]]], dtype=torch.float32)
    valid = torch.tensor([[True, False, True]], dtype=torch.bool)
    rec_out, rec_state = recurrent_commit_torch(q, k, v, g, beta, state, valid)
    for name, value in {
        "REC_Q": q, "REC_K": k, "REC_V": v, "REC_G": g, "REC_BETA": beta,
        "REC_STATE": state, "REC_VALID": valid.float(), "REC_OUTPUT": rec_out,
        "REC_STATE_OUT": rec_state,
    }.items():
        add(name, value)

    # Bidirectional GDN preview against a read-only FP32 state.
    preview_out = bd_gdn_local_memory(
        q, k, v, g, state, valid,
        2.0 ** -0.5,
        torch.tensor([10.0, 9.0]),
        torch.tensor([5.0, 4.5]),
        torch.tensor([1.0, 0.75]),
    )
    for name, value in {
        "PREVIEW_OUTPUT": preview_out,
        "PREVIEW_LOGIT_SCALE": torch.tensor([10.0, 9.0]),
        "PREVIEW_HISTORY_BIAS": torch.tensor([5.0, 4.5]),
        "PREVIEW_HISTORY_C": torch.tensor([1.0, 0.75]),
    }.items():
        add(name, value)

    # Causal convolution with padding; raw invalid rows are zeroed by the caller.
    raw = torch.tensor([[[0.2, -0.4], [0.0, 0.0], [0.7, 0.1], [-0.3, 0.6]]], dtype=torch.bfloat16)
    conv_history = torch.tensor([[[0.1, -0.2, 0.3], [-0.4, 0.2, 0.05]]], dtype=torch.bfloat16)
    conv_weight = torch.tensor([[[0.2, -0.1, 0.3, 0.4]], [[-0.2, 0.5, 0.1, -0.3]]], dtype=torch.bfloat16)
    conv_valid = torch.tensor([[True, False, True, True]], dtype=torch.bool)
    causal_out, causal_state = causal_conv_invalid_torch(raw, conv_history, conv_weight, conv_valid)
    for name, value in {
        "CAUSAL_RAW": raw, "CAUSAL_HISTORY": conv_history, "CAUSAL_WEIGHT": conv_weight,
        "CAUSAL_VALID": conv_valid.float(), "CAUSAL_OUTPUT": causal_out,
        "CAUSAL_STATE_OUT": causal_state,
    }.items():
        add(name, value)

    # Bidirectional preview convolution with independent right-context weights.
    block_raw = torch.tensor([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]], dtype=torch.bfloat16)
    block_history = torch.tensor([[[0.1, -0.2, 0.3], [-0.4, 0.2, 0.05]]], dtype=torch.bfloat16)
    block_weight = conv_weight
    right_weight = torch.tensor([[[0.15, -0.25, 0.35]], [[-0.1, 0.2, 0.3]]], dtype=torch.bfloat16)
    block_valid = torch.tensor([True, True, False], dtype=torch.bool)
    block_out = bd_gdn_conv1d(block_raw, block_history, block_weight, 3, block_valid, right_weight)
    for name, value in {
        "BLOCK_RAW": block_raw, "BLOCK_HISTORY": block_history, "BLOCK_WEIGHT": block_weight,
        "BLOCK_RIGHT_WEIGHT": right_weight, "BLOCK_VALID": block_valid.float(),
        "BLOCK_OUTPUT": block_out,
    }.items():
        add(name, value)

    # Per-token GDN decay and beta gates.
    gate_a = torch.tensor([[[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]]], dtype=torch.bfloat16)
    gate_b = torch.tensor([[[0.3, -0.8], [0.5, 0.2], [-0.1, 0.9]]], dtype=torch.bfloat16)
    a_log = torch.tensor([0.1, -0.2], dtype=torch.float32)
    dt_bias = torch.tensor([1.0, 0.8], dtype=torch.float32)
    decay = -a_log.exp() * F.softplus(gate_a.float() + dt_bias)
    beta_out = gate_b.sigmoid()
    for name, value in {
        "GATE_A": gate_a, "GATE_B": gate_b, "GATE_A_LOG": a_log,
        "GATE_DT_BIAS": dt_bias, "GATE_DECAY": decay, "GATE_BETA": beta_out,
    }.items():
        add(name, value)

    # Clover RMSNorm zero-centered q/k and non-zero-centered GDN output gate.
    norm_input = torch.tensor([[0.2, -0.7, 1.1, 0.3], [-0.4, 0.6, 0.1, -0.9]], dtype=torch.bfloat16)
    norm_weight = torch.tensor([-0.25, 0.5, 0.0, 0.125], dtype=torch.bfloat16)
    norm_out = rms_norm(norm_input, norm_weight, 1e-6, True)
    norm_gate_input = torch.tensor([[0.2, -0.7, 1.1, 0.3], [-0.4, 0.6, 0.1, -0.9]], dtype=torch.bfloat16)
    norm_gate_weight = torch.tensor([0.5, 1.0, 1.5, 0.25], dtype=torch.bfloat16)
    norm_gate = torch.tensor([[-0.2, 0.4, 0.8, -1.0], [0.3, -0.5, 0.1, 0.7]], dtype=torch.bfloat16)
    norm_gate_out = rms_norm(norm_gate_input, norm_gate_weight, 1e-6, False, norm_gate)
    for name, value in {
        "NORM_INPUT": norm_input, "NORM_WEIGHT": norm_weight, "NORM_ZERO_OUT": norm_out,
        "NORM_GATE_INPUT": norm_gate_input, "NORM_GATE_WEIGHT": norm_gate_weight,
        "NORM_GATE": norm_gate, "NORM_GATE_OUT": norm_gate_out,
    }.items():
        add(name, value)

    # Partial RoPE: 50% of an 8-wide head, using positions 3 and 7.
    rope_input = torch.tensor(
        [[[0.2, -0.7, 1.1, 0.3, -0.4, 0.6, 0.1, -0.9],
          [0.5, 0.2, -0.3, 0.8, 0.1, -0.6, 0.7, -0.2]],
         [[-0.1, 0.4, 0.6, -0.8, 0.3, 0.2, -0.5, 0.9],
          [0.7, -0.2, 0.1, 0.5, -0.8, 0.4, 0.3, -0.6]]],
        dtype=torch.bfloat16,
    )
    rope_positions = torch.tensor([3, 7], dtype=torch.float32)
    rope_out, rope_cos, rope_sin = rotary_partial(rope_input, rope_positions, 10000.0, 0.5)
    for name, value in {
        "ROPE_INPUT": rope_input, "ROPE_COS": rope_cos, "ROPE_SIN": rope_sin,
        "ROPE_OUTPUT": rope_out,
    }.items():
        add(name, value)

    # Full-attention reference: zero-centered q/k RMSNorm, partial RoPE,
    # 16Q/2KV GQA, causal-prefix/block mask and output sigmoid gate.
    attn_q_input = ((torch.arange(128).reshape(2, 16, 4).float() % 17) - 8) / 20
    attn_q_input = attn_q_input.to(torch.bfloat16)
    attn_k_input = ((torch.arange(24).reshape(3, 2, 4).float() % 13) - 6) / 15
    attn_k_input = attn_k_input.to(torch.bfloat16)
    attn_q_weight = torch.tensor([0.0, 0.1, -0.1, 0.2], dtype=torch.bfloat16)
    attn_k_weight = torch.tensor([0.1, -0.1, 0.2, 0.0], dtype=torch.bfloat16)
    q_norm = rms_norm(attn_q_input, attn_q_weight, 1e-6, True)
    k_norm = rms_norm(attn_k_input, attn_k_weight, 1e-6, True)
    q_rot, q_cos, q_sin = rotary_partial(q_norm, torch.tensor([1, 2]), 10000.0, 0.5)
    k_rot, k_cos, k_sin = rotary_partial(k_norm, torch.tensor([0, 1, 2]), 10000.0, 0.5)
    attn_v = (((torch.arange(48).reshape(3, 2, 4).float() % 19) - 9) / 20).to(torch.bfloat16)
    attn_gate = (((torch.arange(128).reshape(2, 16, 4).float() % 11) - 5) / 8).to(torch.bfloat16)
    allow = torch.tensor([[[[True, True, False], [True, True, False]]]], dtype=torch.bool)
    q_heads = q_rot.permute(1, 0, 2).unsqueeze(0)
    k_heads = k_rot.permute(1, 0, 2).unsqueeze(0)
    v_heads = attn_v.permute(1, 0, 2).unsqueeze(0)
    attended, gated = attention_eager(q_heads, k_heads, v_heads, attn_gate.unsqueeze(0), allow)
    for name, value in {
        "ATTN_Q_INPUT": attn_q_input, "ATTN_Q_WEIGHT": attn_q_weight,
        "ATTN_K_INPUT": attn_k_input, "ATTN_K_WEIGHT": attn_k_weight,
        "ATTN_Q_NORM": q_norm, "ATTN_K_NORM": k_norm,
        "ATTN_Q_COS": q_cos, "ATTN_Q_SIN": q_sin,
        "ATTN_K_COS": k_cos, "ATTN_K_SIN": k_sin,
        "ATTN_Q_ROTATED": q_rot, "ATTN_K_ROTATED": k_rot,
        "ATTN_V": attn_v, "ATTN_GATE": attn_gate,
        "ATTN_MASK": allow.reshape(2, 3).float(),
        "ATTN_OUTPUT": attended, "ATTN_GATED_OUTPUT": gated,
    }.items():
        add(name, value)

    source = {
        "repository": "https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview",
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "sha256": SOURCE_SHA256,
        "transcribed_functions": {
            "recurrent_commit_torch": "modeling_clover_moe.py:452-468",
            "bd_gdn_conv1d": "modeling_clover_moe.py:253-317",
            "bd_gdn_local_memory": "modeling_clover_moe.py:320-421",
            "CloverRMSNorm": "modeling_clover_moe.py:433-449",
            "CloverAttention.rotary": "modeling_clover_moe.py:597-608",
            "CloverAttention eager attention": "modeling_clover_moe.py:640-650",
        },
    }
    rust_lines = [
        "// @generated by tools/m11/gen_cpu_reference.py; do not edit by hand.",
        f'pub const SOURCE_REVISION: &str = "{SOURCE_REVISION}";',
        f'pub const SOURCE_SHA256: &str = "{SOURCE_SHA256}";',
        f"pub const TORCH_VERSION: &str = \"{torch.__version__}\";",
    ]
    golden_names = {
        "REC_OUTPUT", "REC_STATE_OUT", "PREVIEW_OUTPUT", "CAUSAL_OUTPUT",
        "CAUSAL_STATE_OUT", "BLOCK_OUTPUT", "GATE_DECAY", "GATE_BETA",
        "NORM_ZERO_OUT", "NORM_GATE_OUT", "ROPE_OUTPUT", "ATTN_OUTPUT",
        "ATTN_GATED_OUTPUT",
    }
    for name, tensor in fixtures.items():
        if name in golden_names:
            rust_lines.append(rust_const(name, tensor))
    rust_path = out / "clover_gdn_upstream_fixtures.rs"
    rust_path.write_text("\n".join(rust_lines) + "\n")

    manifest = {
        "source": source,
        "generator": "tools/m11/gen_cpu_reference.py",
        "seed": SEED,
        "device": "cpu",
        "cuda_used": False,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "fixtures": {
            name: {
                "shape": list(tensor.shape),
                "dtype": "float32-widened-from-upstream-dtype",
                "values": int(tensor.numel()),
            }
            for name, tensor in fixtures.items()
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (rust_path, manifest_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: sha256={digest} bytes={path.stat().st_size}")
    print(f"generated {len(fixtures)} tensors using CPU-only PyTorch {torch.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
