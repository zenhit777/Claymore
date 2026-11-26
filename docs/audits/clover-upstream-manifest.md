# Clover upstream provenance manifest

## Pinned source

- Repository: `https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview`
- Immutable repository revision: `4b0402b65e523d0183a14c431955438714cb16ff`
- Model identifier: `Diffuspace/Clover-35B-A3B-Preview`
- License: Apache-2.0; retain the upstream `LICENSE` and `ATTRIBUTIONS.md` notices.
- Transformers version declared by the pinned config: `5.19.0`
- The pinned revision is the original release snapshot. Do not silently replace it with `main` or another commit.

## Required file-level manifest

Every entry below belongs to the pinned repository revision above. The commit SHA pins the Git tree, but is **not** a substitute for the required SHA-256 of each downloaded file's raw bytes.

| Artifact | Revision | Raw SHA-256 |
|---|---|---|
| `.gitattributes` | `4b0402b65e523d0183a14c431955438714cb16ff` | `a0e948ca693c63e0b5139f66bd709003cbcf8bc764acb1ee687a552885c8fe2c` |
| `ATTRIBUTIONS.md` | `4b0402b65e523d0183a14c431955438714cb16ff` | `39e291dbe16294b7eb69906549d6927c32fa9d2f718adae06b15a9f6744212e5` |
| `LICENSE` | `4b0402b65e523d0183a14c431955438714cb16ff` | `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32` |
| `README.md` | `4b0402b65e523d0183a14c431955438714cb16ff` | `48009c14f7eadbc956b0ffe39ec87866007d7baf4404091a7d63f6007609d7b0` |
| `chat_template.jinja` | `4b0402b65e523d0183a14c431955438714cb16ff` | `01094df7cd8905d1220ea8b31e4f1086dc9bedf0305d3c2fa6578b5fa33b6815` |
| `config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | `e07639edf658af63dbbb52b6eb488d72d1a2f194f15638539b02e3e94b76ecf5` |
| `configuration_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | `f593baeba289a1f54987904f7782f4cfa7b22f9dfc75112768c189108f041d0d` |
| `generation_config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | `4e5bac9359cb9f9ddefefb302e529c8351c919c4b27bb8f139157246cab58335` |
| `model.safetensors.index.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | `6270f95c35739b2c40db0026e8683fb84544c258fa637cd88a3b4887a79fd063`; `total_size=69,322,701,696` bytes; `weight_map` covers 26 model shards plus 2 GDN sidecars |
| `modeling_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | `2584bd5502d6109c2cea9bee2f2ec98f147bbb8954c43858c0315eb4af9ffb0c` |
| `sampling_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | `56768a0b23c4c7c26dc3b6f075741761ffada7d5fc2ff745c6aea5e3cc88011c` |
| `tokenizer.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | `7c62ba7d8bb13291c94459896cd27537858beb58a88a037b09a150e213420988` |
| `tokenizer_config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | `3399c8d380151f9c1bf562257c0063665b74def51735f4d0b01086e63266950e` |

### Raw byte sizes of the authenticated non-Xet artifacts

Measured on the local pinned snapshot, byte-for-byte identical to the sizes published by the Hub at the pinned revision.

| Artifact | Bytes |
|---|---:|
| `.gitattributes` | 101 |
| `ATTRIBUTIONS.md` | 2,113 |
| `LICENSE` | 11,343 |
| `README.md` | 7,480 |
| `chat_template.jinja` | 7,912 |
| `config.json` | 2,575 |
| `configuration_clover_moe.py` | 13,937 |
| `generation_config.json` | 588 |
| `model.safetensors.index.json` | 70,668 |
| `modeling_clover_moe.py` | 42,362 |
| `sampling_clover_moe.py` | 6,329 |
| `tokenizer.json` | 19,989,133 |
| `tokenizer_config.json` | 6,742 |

### Pinned upstream digests for the 28 indexed Safetensors artifacts

These are the SHA-256 values published in Hugging Face Xet pointer metadata at the pinned revision. The local pinned snapshot was subsequently verified end-to-end: all 28 payload files were streamed and their raw SHA-256 values matched this table (28/28), as recorded in `docs/audits/m11.1-clover-checkpoint-audit-2026-11-26.md` and the accompanying logs.

| Artifact | Revision | Raw SHA-256 |
|---|---|---|
| `bd_gdn_conv_right_weight.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `6b27471aa645628c403ce1cba40105528f60aa9bd5b4e873b5ffaa2074ad71d4` |
| `bd_gdn_softmax_scalars.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `a8165aee2433e9b795bcc572a80c53fdfa1585a1b84534af4c20bbdb1332037a` |
| `model-00001-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `c340eef575e465c061812a527d427f25b1cd09e850b453a936083b3d769f1751` |
| `model-00002-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `2a24ff37d42bf33c8cefd58d96d8f502f98a3da0889d086f5eaa1648ced03610` |
| `model-00003-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `f7d0ec384ebe97ad56b6f3468cb15bb4d55281327ddbfdbc7c359c58e2038bcd` |
| `model-00004-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `0b88421141e09e9cec2316914bf3e5e73d62a0c03a2d4486c7c09932649ff6a0` |
| `model-00005-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `a065d9424ab410e2ddc5e8d0d4f417282e3993d5ea612898ea64c04632c30d67` |
| `model-00006-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `e61750b3be8c7934608df1bf4b9a9e15fdb5ae0c624899321ce1d7887323b23e` |
| `model-00007-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `92a452f10d75a41d2e5eaaba46858dcecfec400ba8f853122460e31c1f825085` |
| `model-00008-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `226e7fb60fbdfe85f3671ebf471f503930b721e6ff06b62c50d460ff2f5e5ca2` |
| `model-00009-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `73ff108e69d42fc9d2353067209546b7b25833796b7e3e1d1290a995d98e4807` |
| `model-00010-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `71b8223e3f135d9f0215f07ba2ffa376d8177ac0913d119e531728813f6e2d21` |
| `model-00011-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `e2a6b92e994b28b7b9125f443dce256668b140dcd71f3c65cb03ac1e31552d44` |
| `model-00012-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `221c34e566494b044813aa4b4de82350fa8ee1601323d2d52224a7db73979180` |
| `model-00013-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `752a860ad1593e9d273d215ab7aa194ce95c35cea75f5d60498142d07a725ccf` |
| `model-00014-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `4351cbf05ee65e8c0b8a4316324423e2bb9874fd98b062f9ad02bd1cd34c5319` |
| `model-00015-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `b6a1f0d82bb6145632e369a1de85adde3446cd721b5cced28b4a037f41d34fe6` |
| `model-00016-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `10ada7f1814b1ef6da75e913e946af05b135bef1cc5c8e874c11e03f238f76e7` |
| `model-00017-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `e598e5845299541f4c38fa0bb80e20e2c39c525e9676880e3b2477e9740df939` |
| `model-00018-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `cdaa56291ae23785a489f9e9fbd063fe25e9b1a7e228d787b910a2e3559ba3eb` |
| `model-00019-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `66550744d97ffa0b9f8f0cad1b66a7a1426ed0b99404ae0731e51e622ca705d1` |
| `model-00020-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `4d9a696b1efc80ef80554cbfb09629e9355e8082bf7eaae7053b67a9eb9556d1` |
| `model-00021-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `96cf4f331f51936dac903f9e8726c2d4125b7fda97bebd2bee34eefa319a2ceb` |
| `model-00022-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `d5e11497960fe856c241662ed6e1bb0bda234baf42b8b7f09421c6bfe35a63e3` |
| `model-00023-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `d9cf73995d153079fdec0e87f83c711914457331139f11f06fa312b15f439308` |
| `model-00024-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `adcf9389c87d8e95d76cad82d35100317d164331cce2fe9ddecf3d5c68f372f9` |
| `model-00025-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `52990ce8792a4cda271a9aad4135ff4f5707c9b91c3051afe1d7ffae071be593` |
| `model-00026-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | `7fb786f9db4227edd863e0cc00f777f454bff460671cda3dd63f6faec2f79592` |

The pinned index declares `metadata.total_size = 69,322,701,696` bytes. Its `weight_map` includes both GDN sidecars as well as the 26 model shards, so the total covers all 28 Safetensors files. The Clover inventory parser enforces this topology and total size.

The SHA-256 values for the 26 model shards and two sidecars above are the upstream SHA-256 values published in Hugging Face Xet pointer metadata at the pinned revision. They are not the repository commit SHA or Xet object IDs. **All 28 were recomputed locally from the pinned snapshot bytes and match the pinned table: 28 equal, 0 mismatch** (`docs/audits` artefacts `payload-sha256-actual.txt`, `payload-digest-comparison.txt`, table extracted from `PINNED_ASSET_SHA256`).

The 13 non-Xet digests above were computed locally with `sha256sum` from the downloaded raw bytes on 2026-11-26. `tokenizer.json` is Xet-backed upstream; its local raw-byte digest equals the published Xet fingerprint `7c62ba7d8bb13291c94459896cd27537858beb58a88a037b09a150e213420988`, which authenticates that file end to end. Byte sizes of all 13 files equal the sizes published by the Hub at the pinned revision, and the same 13 digests were reproduced on the owner's pre-existing local snapshot (13/13).

## Configuration contract

The committed fixture `docs/audits/fixtures/clover-config.json` is a readable transcription of the pinned `config.json`. The Rust schema now covers every field in that upstream file, including the `auto_map` entries and `initializer_range`, and rejects unknown fields.

The fixture was re-checked against the downloaded raw file: `CloverConfig::from_json` plus `validate_pinned` accept the pinned bytes, and all 45 top-level JSON keys are value-identical between the raw upstream file and the fixture. `CONFIG_SHA256` and `INDEX_SHA256` in `crates/claymore-model-clover/src/config.rs` pin the raw digests of `config.json` and `model.safetensors.index.json`; `inspect_checkpoint` refuses a checkpoint whose digested bytes differ. The fixture remains documentation, not identity proof: identity comes from the digest check.

## Inventory and checkpoint gate

Completed on the raw pinned bytes:

- SHA-256 and byte size for each of the 13 non-Xet artifacts (`config.json`, `generation_config.json`, tokenizer, tokenizer config, template, the three Python sources, index, README, LICENSE, ATTRIBUTIONS, `.gitattributes`).
- Revision proof for the local snapshot: every downloaded file carries the pinned commit in its Hub download metadata, and `tokenizer.json` matches the published Xet fingerprint byte for byte.
- Structural review of the pinned configuration, model and sampling Python sources against the documented execution semantics, plus raw-byte SHA-256 for those sources.
- Tokenizer and chat-template structural audit, with recorded fixtures (see `docs/audits/fixtures/clover-tokenizer-fixtures.json`).
- Local authentication of both GDN sidecars, including their full 120-tensor name/shape contract (30 × `[8192, 1, 3]` and 90 × `[32]`, BF16) for the 30 `linear_attention` layers declared by the pinned `layer_types`.
- Streaming SHA-256 verification of all 28 indexed Safetensors files by `inspect_checkpoint --verify-sha256`.
- Header-only inspection of all 28 files: `result=PINNED_CONFIG_AND_HEADER_INVENTORY_VALID`, 813 tensors, 34,661,350,848 parameters, `payload_bytes=69322701696` exactly as pinned.

Observed storage contract on the real headers (all 813 tensors BF16): `model.embed_tokens.weight` and
`lm_head.weight` `[248320, 2048]`; routed experts stacked per layer
`mlp.experts.gate_up_proj` `[256, 1024, 2048]` and `mlp.experts.down_proj` `[256, 2048, 512]` (80 tensors over
40 layers); router `mlp.gate.weight` `[256, 2048]` × 40; one shared SwiGLU expert per layer
(`shared_expert.gate_proj`/`up_proj` `[512, 2048]`, `shared_expert.down_proj` `[2048, 512]`) with learned scalar
gate `shared_expert_gate.weight` `[1, 2048]`; GDN `linear_attn.conv1d.weight` `[8192, 1, 4]` distinct from the
sidecar right-convolution `conv_right_weight` `[8192, 1, 3]`; full attention `q_proj` `[8192, 2048]`,
`k_proj`/`v_proj` `[512, 2048]`, `o_proj` `[2048, 4096]`, `q_norm`/`k_norm` `[256]`. No stored tensor is FP32:
FP32 is the runtime SSM/GDN state only.

No payload tensor has been loaded, no forward pass has been run, and no CUDA context has been created.

## Source-level semantic audit (pinned release)

The following contracts were checked against the Python implementation at the pinned repository revision. The reviewed Python sources are the locally downloaded release files whose raw SHA-256 digests are recorded above, so this review is tied to authenticated bytes; the payload-level tensor inventory remains pending.

### GDN preview and commit

- GDN width is derived as 2 × 16 × 128 + 32 × 128 = 8192; Q/K/V are split into 16 key heads of width 128 and 32 value heads of width 128.
- Preview convolution combines the clean causal history with the current block, adds learned independent right-convolution taps, applies SiLU, and does not mutate the clean convolution cache.
- Preview local memory L2-normalizes Q/K in FP32, repeats them to the value-head count, computes intra-block softmax logits with a learned per-value-head logit scale, and adds one clean-history bucket. Its history logit is history_bias + history_c × log(valid_block_size / 32).
- The clean recurrent read is decayed by exp(g) before being combined with local block attention. Preview does not apply the recurrent update coefficient beta and does not commit the preview state.
- Commit mode is a separate causal recurrence: it applies beta = sigmoid(in_proj_b(x)), updates the FP32 SSM state token-by-token, and advances convolution history only for valid tokens. Preview and commit must not be conflated.

### Full attention and MoE

- Full attention projects Q together with an output gate, applies RMSNorm to Q/K, uses partial ordinary RoPE, expands the two KV heads to 16 query heads, and applies a sigmoid output gate after attention.
- During preview, the current block is bidirectional while the committed prefix remains available as context. During commit, the current tokens are additionally constrained by a causal mask.
- The MoE router computes token top-8. In preview with block coreset enabled, it first aggregates selected routing mass over the block, retains the smallest descending expert set reaching cumulative top-p 0.9 (always at least one), then applies token top-8 within that coreset.
- Per-layer routed experts are stored as stacked gate_up_proj and down_proj parameters. Each layer also has one shared SwiGLU expert and a learned scalar gate. This is the basis for the pinned tensor-name/shape validator added to CloverCheckpointInventory.

### LowConfidence generation and penalties

- The sampling helper supports greedy decoding or temperature/top-k/top-p sampling. Confidence is nevertheless computed from the original, unscaled logits at temperature 1.
- Presence/frequency/repetition and length/thinking biases use committed response history; uncommitted masked-block positions do not penalize one another.
- A position is filled when confidence is strictly greater than the configured threshold. If no pending position qualifies, the most confident pending position is filled to guarantee progress.
- A block is committed only after all of its positions are resolved; EOS handling occurs after block resolution, preserving earlier positions that may have been filled after an EOS position.
- The upstream implementation does not expose beam search or speculative decoding through this generation path. These must not be assumed by the Claymore adapter.

### Tokenizer and prompt boundary

- The pinned tokenizer is `Qwen2Tokenizer` (fast backend `tokenizers`): 248,044 BPE vocabulary entries plus 34 `added_tokens` entries, of which 33 introduce a new ID (`<|mask|>` already occupies BPE vocab index 151669), i.e. 248,077 usable IDs; `model_max_length` 131,072; tokenizer EOS `<|im_end|>` (248,046); tokenizer PAD `<|endoftext|>` (248,044); mask token `<|mask|>` at ID 151669. The model `config.json` declares `vocab_size: 248320`, `eos_token_id: 248044`, `pad_token_id: null`, `mask_token_id: 151669`, and `generation_config.json` declares `eos_token_id: [248046, 248044]` with `pad_token_id: 248044`. The three EOS/PAD layers are deliberately different and must not be collapsed: the tokenizer stops on 248,046, the model config falls back to 248,044, and generation admits both.
- Model vocabulary 248,320 exceeds usable tokenizer IDs 248,077 by 243 unused rows. The pinned tensor schema must keep `lm_head.weight` and the embedding table at `[248320, 2048]`; the extra rows are unreachable by the tokenizer and must never be trimmed.
- Upstream attribution states that ID 151669 was remapped to `<|mask|>`, affected BPE merges were removed, and the chat template was reduced to text-only rendering. Confirmed against the raw release bytes: `tokenizer.json` (SHA-256 `7c62ba7d…0988`) registers `<|mask|>` as `special: true` at index 151669 inside `added_tokens`, `tokenizer_config.json` declares `mask_token: "<|mask|>"`, encoding `<|mask|>` yields the single ID `[151669]` (no split), and none of the 247,581 recorded BPE merges reference `<|mask|>` or ID 151669.
- `chat_template.jinja` (SHA-256 `01094df7…6815`) renders text only: any non-text content part raises `This model accepts text content only.`; no fixture injects `<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`, `<|video_pad|>` or `<|audio_start|>`. Thinking switches are `enable_thinking`, `preserve_thinking` and `add_non_thinking_prefix`; the generation prompt ends with `<think>\n` when thinking is enabled and with `<think>\n\n</think>\n\n` when disabled. Recorded fixtures and exact rendered strings and token IDs are in `docs/audits/fixtures/clover-tokenizer-fixtures.json`.

## Local raw-byte authentication run — 2026-11-26

Executed CPU-only on WSL2 Linux 6.18.58.1, `x86_64`, no CUDA toolkit involvement.

- Snapshot: `/home/sjteb/models/Clover-35B-A3B-Preview-4b0402b`, ext4 on `/dev/sde`, 251 GiB free before and after the metadata-only download.
- Download: `hf download Diffuspace/Clover-35B-A3B-Preview --revision 4b0402b65e523d0183a14c431955438714cb16ff --repo-type model --exclude '*.safetensors' --local-dir <snapshot>` with `huggingface_hub` 1.18.0; exit 0.
- Revision proof: each file's `.cache/huggingface/download/<name>.metadata` first line is `4b0402b65e523d0183a14c431955438714cb16ff`; the second line is the Hub blob/Xet digest (`config.json` → `e9dcc9028a21358f8d467184915e3c473f12d1b7`, `tokenizer.json` → the SHA-256 above). The commit list for the revision reports title `Add Clover-35B-A3B-Preview release`, date `2026-10-28T03:07:54Z`. The repository `main` branch is now `d7fc9c15baa685238a559e8823621abd609c2593`, so the pinned release commit and the floating branch differ; nothing was read from `main`.
- Digests: `sha256sum` over the exact raw bytes of the 13 files listed above; sizes compared against `GET /api/models/.../revision/4b0402b…?blobs=true`.
- Rust contract check at PR head `f930fbfee3dd5a7a6f3591116d1234160e598588`: `CloverConfig::from_json` + `validate_pinned` accept the pinned `config.json`; 45/45 top-level keys match the committed fixture.
- Tokenizer audit: `transformers` 5.19.0 (the version declared by the pinned config), `tokenizers` 0.23.3, `safetensors` 0.8.0, Python 3.12.3, `jinja2` 3.1.6; PyTorch is not installed in the audit environment, so no model class can be instantiated and `torch.cuda.is_available()` is unreachable. No weights were opened.
- GDN sidecars authenticated locally (1.5 MiB total, no model payload involved): both digests match the pinned Xet values exactly — `bd_gdn_conv_right_weight.safetensors` `6b27471a…71d4` and `bd_gdn_softmax_scalars.safetensors` `a8165aee…037a`. `CloverCheckpointInventory::inspect_gdn_sidecars` accepts both files.
- Sidecar tensor contract confirmed on the pinned bytes: 30 tensors `model.layers.<i>.linear_attn.conv_right_weight` of shape `[8192, 1, 3]` and 90 tensors of shape `[32]` (`bd_softmax_history_bias`, `bd_softmax_history_c`, `bd_softmax_logit_scale`), 120 tensors in total. The 30 layer indices `{0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38}` equal exactly the `linear_attention` entries of the pinned `config.json` `layer_types`.
- Stored sidecar dtype is **BF16**, not FP32. FP32 applies to the runtime SSM/GDN state, never to these stored tensors; the adapter must not cast them to FP32 on load.
- Remaining payload digests: none outstanding — all 28 indexed Safetensors were recomputed locally and match.
- Owner's pre-existing local snapshot, used for the payload pass: `/mnt/c/Users/sjteb/.cache/huggingface/hub/hub/models--Diffuspace--Clover-35B-A3B-Preview/snapshots/4b0402b65e523d0183a14c431955438714cb16ff` (Windows-side Hub cache, plain files, no symlinks). Revision proof: the cache tree file `trees/4b0402b65e523d0183a14c431955438714cb16ff.json` carries 41 entries whose 41 sizes and 41 `blob_id` values equal the Hub API values for the pinned revision, `refs/main` is `4b0402b65e523d0183a14c431955438714cb16ff`, and the 13 non-Xet raw digests reproduce exactly (13/13) the values computed from the direct download.
- Payload runs: `inspect_checkpoint` on the pinned snapshot returns `result=PINNED_CONFIG_AND_HEADER_INVENTORY_VALID` with `tensor_count=813`, `parameter_count=34661350848`, `payload_bytes=69322701696`, `payloads_read=false`, `cuda_initialized=false`. `inspect_checkpoint --verify-sha256` then streams all 69.3 GB in 54 min 01 s, exit 0, and prints `artifact_sha256=verified` followed by the same `result=PINNED_CONFIG_AND_HEADER_INVENTORY_VALID`, `payloads_read=false`, `cuda_initialized=false`. That run predates the `Sha256Stream` fix `7ee60fff`; the post-corrective replay is recorded in the next section.
- Performance observation, not a validation issue: the historical 69.3 GB pass ran at ~21 MB/s and
  was originally attributed to `MAX_CHUNK_BYTES = 8 MiB` blocks. That attribution was withdrawn
  (`m11.1-audit-corrections-2026-11-26.md` § 1): `sha256_reader` uses a 1 MiB read buffer. The
  post-corrective pass re-measured ~21 MB/s (55 min 08 s wall, 2 939 s user CPU, 18 s system CPU,
  peak RSS 5 212 KiB), which identifies the pass as **CPU-bound software SHA-256**, not I/O-bound.
  A hardware-accelerated or faster hasher remains an M11.2 performance candidate.

No checkpoint compatibility or executable-model claim is permitted until these gates pass.

## Post-corrective CPU-only refresh — 2026-11-26

Replay of the local evidence after the `Sha256Stream` fix (`7ee60fff`), the strict BF16 gate for the
GDN sidecars (`f23fc1b5`), the metadata-only shard/tensor/chunk plan (`18df143`, `d1949fd`, `91365c7`)
and the chat-template parity script fixes (`1916239`, `7d0b327`, `7d5685a`). Branch
`feat/m11-clover-foundation` at HEAD `443b1c4d923c1b4a33952ca4856d1bdff622025d`; full report
`m11.1-cpu-evidence-refresh-2026-11-26.md`, artefacts prefixed `m11.1-postfix-2026-11-26-`.
Nothing historical was rewritten or deleted; no pinned digest, shape, dtype or golden fixture changed.

- Environment: `rustc 1.100.0` / `cargo 1.100.0`, MSRV tested with `rustc 1.86.0`, Python 3.12.3,
  `transformers` 5.19.0, `tokenizers` 0.23.3, `jinja2` 3.1.6, `safetensors` 0.8.0, PyTorch absent.
  `CUDA_VISIBLE_DEVICES=""` on every command; no `nvidia-smi` call; no CUDA context; no VRAM; no
  payload tensor; no forward pass.
- Revision proof of the reused snapshot: `refs/main` equals the pinned revision; the local cache tree
  file reproduces **41/41** sizes and **41/41** `blob_id` values of the pinned Hub API response
  (`m11.1-postfix-2026-11-26-trees-vs-api-comparison.txt`); the 13 non-Xet raw digests reproduce the
  table above **13/13** (`m11.1-postfix-2026-11-26-nonxet-vs-manifest-comparison.txt`).
- Header inventory: 26 model shards + 2 GDN sidecars, 813 tensors, 34,661,350,848 parameters,
  `payload_bytes=69322701696`; per-artifact arithmetic control holds for all 28 files
  (`bytes = 8 + header_bytes + payload_bytes`, 224 B of u64 length prefixes, 93,976 B of headers,
  `payload == index.metadata.total_size`): `m11.1-postfix-2026-11-26-arithmetic-control.txt`.
- Load plan, metadata-only: `load_plan_shards=28`, `load_plan_tensors=813`,
  `load_plan_chunks=8845`, `load_plan_max_chunk_bytes=8388608`; three consecutive runs produce
  byte-identical output; peak RSS 4.4 MiB. Determinism and the 8 MiB bound are also covered by
  `load_plan_is_deterministic_and_splits_large_tensors_into_bounded_chunks`.
- Streaming digest verification with the fixed primitive: **28/28**, exit 0, `artifact_sha256=verified`,
  55 min 08,79 s, ~21 MB/s, peak RSS 5,212 KiB, `payloads_read=false`, `cuda_initialized=false`; an
  independent `sha256sum` pass over the same 28 files reproduces the pinned table 28/28
  (`m11.1-postfix-2026-11-26-digest-comparison.txt`).
- Negative gates T0–T3 replayed with the fixed binary: identical rejection messages and exit codes to
  the original transcript (T0 exit 0; T1 config digest; T2 `TotalSizeMismatch`; T3 index digest).
- Rust gates: 28 tests green on stable and on 1.86, including
  `inspect_rejects_f32_gdn_sidecars_for_the_pinned_release` (an F32 GDN sidecar is refused),
  `streaming_updates_preserve_partial_blocks_across_calls` and
  `streaming_reader_handles_short_reads`; `clippy -D warnings` and `cargo fmt --check` clean.
- Tokenizer/chat template: regenerated audit is field-for-field identical to the published one except
  the local checkpoint path; the regenerated fixture differs only by the added
  `rendered_token_ids` field. Parity is now a real comparison — `apply_chat_template(tokenize=True)`
  versus re-tokenising the rendered text with `add_special_tokens=False` — for **9/9** positive
  fixtures, with **3/3** expected rejections and zero unexpected failures.
  A tooling defect was found and fixed in this replay: the parity script iterated a `BatchEncoding`
  with `list(...)`, which yields dictionary keys, so all 9 positive fixtures failed; digests of every
  script revision are recorded in `m11.1-postfix-2026-11-26-script-digests.txt`.
- Attestation: the historical `m11.1-checkpoint-2026-11-26-logs.sha256` is left untouched (17/18 of
  its entries still verify; the script entry has been stale since `1916239`, before this replay). The
  new attestation is `m11.1-postfix-2026-11-26-logs.sha256`.

## M11.1 gate status — 2026-11-26

Raw authentication of the pinned revision, the complete 28-file digest verification, the tensor-name/shape/dtype
inventory, the metadata-only load plan and the tokenizer/chat-template audit are **complete and evidenced, twice**:
once at `f930fbf`/`cd811df` and once post-corrective at `443b1c4` (section above). What remains for M11.1 is the
architect's review of PR #53 and decision **A2** (the three distinct EOS/PAD layers, draft recorded in
`docs/audits/m11-architect-decisions.md`, still unapproved) — not missing evidence. M11.1 does not claim an
executable Clover model: no forward pass, no device execution and no memory-admission path exist yet; those belong
to M11.2 and later.
