# Campagne CPU M10.2 — M10.6 (23 novembre 2026)

Journal horodaté de la campagne CPU de M10, sans GPU. Les sections « Résultats
mesurés » sont remplies à partir des sorties brutes des commandes, horodatées
en UTC.

## Périmètre

- Branche : `feat/m10-diffusiongemma`, PR #40 ; PR de suivi #44 (`feat/m10-campaign-followup`, base `main`, liée à #40 et #42, non fusionnée).
- Checkpoint épinglé : `google/diffusiongemma-26B-A4B-it`, révision `cd40195554d3cfcb56f34b9adfb66efc790ed5b6` (1047 tenseurs, 11 shards, 51 647 562 456 octets).
- Zéro GPU pendant cette campagne : aucune allocation CUDA, aucun lancement kernel. La campagne hardware est réservée au créneau GPU final, après libération de la VRAM (VRAM occupée par l'utilisateur au moment du travail CPU).

## Environnement

| Élément | Valeur |
|---|---|
| `rustc` principal | 1.100.0 (0ef1bc386 2026-11-09) |
| `rustc` MSRV | 1.86.0 (05f9846f8 2025-03-31) |
| `nvcc` | CUDA 13.4 UMD (PTX `sm_120`, régénéré hors build) |
| venv références dorées | `transformers 5.24.0.dev0 @ 671aa77576266f41f33f9c28e2d824bfd1b1b63c`, `torch 2.15.0+cpu`, `accelerate 1.16.0`, CPU-only |
| RAM | 112 GiB (64 GiB disponibles au pic du chargement complet CPU) |

## Empreintes des artefacts de référence

| Artefact | SHA-256 |
|---|---|
| `config.json` (checkpoint) | `13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506` |
| `model.safetensors.index.json` | `6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a` |
| `generation_config.json` | `99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `d82ed5ed23e322912eeb90c2d14aafd9ade84ffe5278bd7af9184cd31ea93856` |

La liste complète des 70 empreintes des tenseurs golden (F32, CPU,
`/tmp/opencode/m10/golden`) est consignée dans `/tmp/opencode/m10/golden_shas.txt`.

## Entrées de la campagne golden

- Prompt fixe : `"What is the capital of France?"` (7 tokens, ids `[2, 1449, 4470, 2927, 9776, 2312, 2]`).
- Image déterministe : seed 1234, 48×48 RGB → 9 patches de 16 px → pooling kernel 3 → 1 token vision ; pixel values en BF16 (`2(x-0.5)` appliqué par le patch embedder épinglé), positions `[1, 9, 2]`.
- Canvas golden : seed 42, 16 positions (la composition du canvas ne dépend pas de la longueur ; 16 rend la référence CPU complète exécutable).
- Hooks capturés (venv épinglé) : `vision_last_hidden_direct [1,9,1152]`, `embed_vision_output_direct [1,1,2816]`, `encoder_last_hidden_direct [1,7,2816]`, `enc_kv_{0..29}_{keys,values}_direct`, `canvas0_direct [1,16]`, `decoder_last_hidden0_direct [1,16,2816]`, `decoder_logits0_direct [1,16,262144]`, `moe0_router_logits [16,128]`, `moe0_topk_weights [16,8]`, `moe0_topk_indices [16,8]`, `moe0_expert_outputs [16,2816]`.

## Commandes de campagne

```bash
# CI CPU
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo check -p claymore-memory --features cuda-driver --all-targets --locked
cargo check -p claymore-model-gemma --features cuda-streaming --all-targets --locked
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.86 check -p claymore-cuda --features cuda-driver --all-targets --locked

# Références dorées et différentiels CPU (profil release, CPU only)
cargo test --release -p claymore-model-gemma --test golden_cpu --locked -- \
  --ignored --test-threads=1 --nocapture
```

## Résultats mesurés

### CI (19:35 — 20:10 UTC)

- `cargo fmt --all -- --check` : ✅
- `cargo test --workspace --all-targets --locked` : ✅ (70 tests `claymore-tensor`, 63 `claymore-model-gemma`, suite workspace complète, 3 tests golden marqués `#[ignore]`)
- `cargo clippy --workspace --all-targets --locked -- -D warnings` : ✅
- checks `cuda-driver` (`claymore-cuda`, `claymore-memory`) et `cuda-streaming` (`claymore-model-gemma`) : ✅ (compilation sans exécution GPU)
- variants `+1.86` (tests, clippy, checks cuda-driver) : ✅

### Kernels M10.2

- 24 entrées PTX (`m10.cu` → `m10.ptx`, `nvcc -arch=compute_120 -code=compute_120 -O3 -ptx`) : ✅ régénération OK.
- Table de capacité CUDA : 32 entrées, test de couverture `cuda_capabilities_declare_only_implemented_f32_and_bf16_vram_ops` : ✅ (exceptions mono-dtype `widen` BF16→F32, `router_top_k` F32).

### Différentiels golden CPU (profil release, 20:4x UTC)

À remplir depuis `/tmp/opencode/m10/golden_cpu_canvas16.log` :

| Stage | max_abs | rms | Tolérance | Verdict |
|---|---|---|---|---|
| `encoder_last_hidden` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,03`, `rms ≤ 2e-3` | _à consigner_ |
| `enc_kv_0_keys` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,05`, `rms ≤ 2e-3` | _à consigner_ |
| `vision_last_hidden` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,06`, `rms ≤ 2e-3` | _à consigner_ |
| `embed_vision` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,1`, `rms ≤ 4e-3` | _à consigner_ |
| `decoder_last_hidden0` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,12`, `rms ≤ 4e-3` | _à consigner_ |
| `decoder_logits0` | _à consigner_ | _à consigner_ | `max_abs ≤ 2,0`, `rms ≤ 0,1` | _à consigner_ |
| `moe0_router_scores` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,02` | _à consigner_ |
| `moe0_topk_weights` | _à consigner_ | _à consigner_ | `max_abs ≤ 0,01` | _à consigner_ |
| `moe0_topk_indices` | exact | exact | égalité exacte | _à consigner_ |

Les tolérances sont **déclarées avant** la campagne, dérivées de la
précision BF16 (pas ajustées après observation).

## Gate GPU (déclaré, non exécuté)

La campagne hardware (streaming complet sur 51,6 Go, admission/éviction
matérielle, primitives CUDA sur données réelles, génération réelle de 256
tokens avec l'EntropyBoundSampler) est **planifiée dans un unique créneau GPU
final**, après confirmation de la libération de la VRAM par l'utilisateur.
Aucune génération réelle n'est revendiquée avant l'exécution effective et la
consignation des journaux de ce créneau.
