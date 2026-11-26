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
| `README.md` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `LICENSE` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `ATTRIBUTIONS.md` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `generation_config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `tokenizer.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `tokenizer_config.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `chat_template.jinja` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `configuration_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `modeling_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `sampling_clover_moe.py` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `model.safetensors.index.json` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification; `total_size=69,322,701,696` bytes; `weight_map` covers 26 model shards plus 2 GDN sidecars |
| `bd_gdn_conv_right_weight.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `bd_gdn_softmax_scalars.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification |
| `model-00001-of-00026.safetensors` … `model-00026-of-00026.safetensors` | `4b0402b65e523d0183a14c431955438714cb16ff` | pending raw-byte verification for each shard |

The pinned index declares `metadata.total_size = 69,322,701,696` bytes. Its `weight_map` includes both GDN sidecars as well as the 26 model shards, so the total covers all 28 Safetensors files. The Clover inventory parser enforces this topology and total size.

The raw SHA-256 values must be computed from the actual downloaded bytes (including metadata files and each individual shard/sidecar). Do not infer them from the Hugging Face commit SHA, Xet object ID, file size, a rendered HTML page, or a locally reconstructed JSON fixture.

## Configuration contract

The committed fixture `docs/audits/fixtures/clover-config.json` is a readable transcription of the pinned `config.json`. The Rust schema now covers every field in that upstream file, including the `auto_map` entries and `initializer_range`, and rejects unknown fields. It is still a fixture, not proof of raw-file identity; runtime checkpoint opening must compare the raw file's SHA-256 against the completed manifest.

## Inventory and checkpoint gate

Not yet complete:

- SHA-256 for each required raw artifact.
- Strict index parsing and complete tensor-to-shard inventory.
- Validation of all 26 model shards and both standalone GDN parameter files.
- Tensor-name/shape/dtype inventory, including expert and shared-expert coverage.
- Tokenizer and chat-template structural audit.
- Review of `configuration_clover_moe.py`, `modeling_clover_moe.py`, and `sampling_clover_moe.py` at the pinned revision against the documented execution semantics.

No checkpoint compatibility or executable-model claim is permitted until these gates pass.
