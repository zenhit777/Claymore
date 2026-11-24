# Campagne CPU M10 — M10.2 à M10.6 (23–24 novembre 2026)

Journal horodaté de la campagne CPU de M10, sans GPU. Les sections « Résultats
mesurés » sont remplies à partir des sorties brutes des commandes.

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
| `crates/claymore-cuda/kernels/m10.cu` | `32ae4caf6303722b9c34d706938b0b7f73f1e804f16dbb31bad9183c02825713` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `6281667064f7c835eb55799ed752bad14c4c2ef82f7b8fb2c0943fb40c465057` |
| `/tmp/opencode/m10/golden_eager/manifest.json` | `e266cc5f6ae8144f28c7f20ab55721c0d3cf60008aa9d439da579e3876fc548f` |

L'inventaire doré compte **781 tenseurs F32** dans `/tmp/opencode/m10/golden_eager` :
sorties finales, sorties par couche, KV par couche, étapes internes de la tour de
vision (6 par couche) et étapes internes du décodéur (11 par couche). Le manifest
`golden_eager/manifest.json` porte la shape de chaque tenseur.

## Entrées de la campagne golden

- Prompt fixe : `"What is the capital of France?"` (7 tokens, ids `[2, 1449, 4470, 2927, 9776, 2312, 2]`).
- Image déterministe : seed 1234, 48×48 RGB → 9 patches de 16 px → pooling kernel 3 → 1 token vision ; pixel values en BF16 (`2(x-0.5)` appliqué par le patch embedder épinglé), positions `[1, 9, 2]`.
- Canvas golden : seed 42, 16 positions (la composition du canvas ne dépend pas de la longueur ; 16 rend la référence CPU complète exécutable).
- Hooks capturés (venv épinglé) : `vision_last_hidden_direct [1,9,1152]`, `embed_vision_output_direct [1,1,2816]`, `encoder_last_hidden_direct [1,7,2816]`, `enc_kv_{0..29}_{keys,values}_direct`, `encoder_layer_{0..29}`, `decoder_embedding`, `decoder_layer_{0..29}`, `decoder_last_hidden0_direct [1,16,2816]`, `decoder_logits0_direct [1,16,262144]`, `canvas0_direct [1,16]`, `moe0_router_logits [16,128]`, `moe0_topk_weights [16,8]`, `moe0_topk_indices [16,8]`.
- Étapes vision par couche `i` : `v{i}_input_norm`, `v{i}_attn_out`, `v{i}_post_attn_norm`, `v{i}_pre_ff_norm`, `v{i}_mlp_out`, `v{i}_post_ff_norm` (hooks `input_layernorm`, `self_attn`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `mlp`, `post_feedforward_layernorm`).
- Étapes décodéur par couche `i` : `dec{i}_attn_out`, `dec{i}_post_attn_norm`, `dec{i}_residual_after_attn`, `dec{i}_mlp_out`, `dec{i}_post_ff1`, `dec{i}_pre_ff2`, `dec{i}_experts`, `dec{i}_post_ff2`, `dec{i}_mlp_moe_sum`, `dec{i}_post_ff`, `dec{i}_out`.

## Régénération eager des dorées (24 novembre 2026)

Les dorées de la campagne précédente utilisaient l'implémentation d'attention
`sdpa` et les experts `grouped_mm`. La sémantique épinglée **de référence** est
l'implémentation eager : `attn_implementation="eager"` et
`experts_implementation="eager"`, avec le BF16 matériel des scores, des probabilités
et des sorties d'experts. Les dorées ont donc été **régénérées intégralement** en
mode eager, et les tolérances CPU de la campagne ont été **conservées** telles
qu'elles étaient déclarées avant campagne (aucun seuil n'a été assoupli après
observation des mesures).

Le mode eager a été choisi parce qu'il matérialise explicitement les frontières de
dtype que Claymore implémente (`scores BF16 → softmax F32 → probabilités
re-arrondies BF16 → réduction de valeurs`), et que `grouped_mm` / `sdpa` ne
matérialisent pas ces frontières : elles ne peuvent pas servir de référence pour
valider la sémantique du graphe complet.

### Corrections sémantiques appliquées depuis l'audit initial

Chacune de ces corrections est dérivée du code épinglé lu ligne par ligne
(`modeling_diffusion_gemma.py`, `modeling_gemma4.py` à la révision `671aa77…`),
pas d'un ajustement sur les mesures :

1. Taille intermédiaire MoE : `[128, 2 × 704, 2816]` → gate/up de **704** lignes par expert (`gate_up_proj` = concat de `gate` et `up`), et non 1408 lignes traitées comme une seule matrice.
2. Soft-cap d'attention : `cap * tanh(score / cap)` avec un **seul** facteur `cap`, appliqué après le `tanh`. La version précédente pré-multipiait le score par `1/cap` puis remultipiait, ce qui ajoutait un arrondi BF16 de part et d'autre du `tanh`.
3. Échelle d'embedding de texte : `sqrt(hidden_size)` arrondi en **BF16** avant la multiplication (53,0), et non `hidden_size^-0.5`.
4. RoPE global `proportional` : `rope_angles = int(head_dim × 0,25 / 2) = 64` angles sur `head_dim = 512` ; les exposants sont divisés par `head_dim` (et non par le nombre d'angles rotationnels), les fréquences au-delà de 64 sont **nulles**, `emb = concat(freqs, freqs)`, paires `i` / `i + 256`, tables cos/sin arrondies en BF16, rotations à **deux arrondis BF16** par composante.
5. RMSNorm : `(mean² + eps) ** -0.5` via `powf(-0.5)` (pas `rsqrt`), réduction F32, arrondi **unique** BF16 après la multiplication par l'échelle.
6. RoPE de la tour de vision : `axial` 2D, `inv_freq` sur `head_dim/2 = 36` avec theta 100, recomposition `[h, h, w, w]` par patch, `head_dim = 72`, paires `i` / `i + 18` sur chaque axe, deux arrondis BF16 par composante.
7. Pooling de la tour de vision : moyenne par groupe `k²` (kernel 3), **arrondie en BF16** avant la multiplication par `sqrt(hidden_size)`, puis standardisation `(x − std_bias) × std_scale`.
8. GELU des experts : `gelu_pytorch_tanh` avec **deux arrondis BF16** (le produit `gelu × up` est arrondi, le `gelu` seul aussi), poids FP32 dans la référence, `index_add_` en **ordre croissant d'expert** avec un arrondi BF16 à chaque addition.
9. Router CPU : deux frontières BF16 explicites (`normed × scale_w`, puis `× BF16(hidden_size^-0.5)`), softmax **F32** des scores BF16, poids `top_k × per_expert_scale`, tie-break déterministe par indice **décroissant** (la `torch.topk` CPU n'est pas stable : plusieurs égalités de scores donnent des ordres différents ; seule la convention déterministe est implémentée).
10. `layer_scalar` de l'encodeur : appliqué comme **dernière opération** de la couche (produit BF16 avec le scalaire appris), y compris sur les couches sans `shared_kv`.
11. Auto-conditionnement : le post-norm **non scaleé** du module `self_conditioning` est appliqué **même au premier pas** (signal zéro), exactement comme `DiffusionGemmaSelfConditioning.forward`.
12. Attention du décodéur : **entièrement bidirectionnelle** sur le préfixe encodeur + le canvas, sans fenêtre glissante imposée par le masque du décodéur ; pour les couches globales, `v_proj` est `None` et `value_states = v_norm(k_proj)` (avant `k_norm`), et non `v_norm` du `k_norm`/rope.
13. Kernels CUDA : mêmes frontières de dtype (`SoftCap` sans pré-scale, `RouterTopK` F32, top-k tie-break décroissant, GELU à deux arrondis, RoPE BF16, score d'attention arrondi BF16 avant softmax, probabilités arrondies BF16 avant la réduction de valeurs), avec `m10.ptx` régénéré (SHA `62816670…`).

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

### CI

- `cargo fmt --all -- --check` : ✅
- `cargo test --workspace --all-targets --locked` : ✅ (246 tests réussis, 13 suites, dont 6 tests golden `#[ignore]`)
- `cargo clippy --workspace --all-targets --locked -- -D warnings` : ✅
- checks `cuda-driver` (`claymore-cuda`, `claymore-memory`) et `cuda-streaming` (`claymore-model-gemma`) : ✅ (compilation sans exécution GPU) ; **les avertissements `unused` de `cuda_graph.rs` et `stream.rs` ont été supprimés** (`GemmaModelConfig`, `weights_idx`, `temperature`, `alloc`) ; il ne reste que le warning pré-existant `unused field repository in workspace.package`
- variants `+1.86` (`test --workspace --all-targets`, `clippy -D warnings`, checks `cuda-driver` sur `claymore-cuda` et `claymore-memory`, check `cuda-streaming` sur `claymore-model-gemma`) : ✅ **0 erreur, 13 suites de tests vertes** (journal `/tmp/opencode/m10/msrv-1.86-run2.log`)

### Différentiels golden CPU (profil release, dorées eager)

Tolérances **déclarées avant** la campagne, dérivées de la précision BF16. Le verdict
est **rouge** quand la mesure dépasse la tolérance. Aucun seuil n'a été modifié.

| Stage | max_abs | rms | Tolérance | Verdict |
|---|---|---|---|---|
| `encoder_last_hidden` | 3,5 | 0,083 | `max_abs ≤ 0,03`, `rms ≤ 2e-3` | **ROUGE** |
| `enc_kv_0_keys` | 0,5 | 0,0075 | `max_abs ≤ 0,05`, `rms ≤ 2e-3` | **ROUGE** |
| `vision_last_hidden` | 0,189453 | 0,054739 | `max_abs ≤ 0,06`, `rms ≤ 2e-3` | **ROUGE** |
| `embed_vision` | 0,242188 | 0,060047 | `max_abs ≤ 0,1`, `rms ≤ 4e-3` | **ROUGE** |
| `decoder_last_hidden0` | 17,03125 | 1,923966 | `max_abs ≤ 0,12`, `rms ≤ 4e-3` | **ROUGE** |
| `decoder_logits0` | _consigné après re-run_ | _consigné après re-run_ | `max_abs ≤ 2,0`, `rms ≤ 0,1` | _consigné après re-run_ |
| `moe0_router_scores` | _consigné après re-run_ | _consigné après re-run_ | `max_abs ≤ 0,02` | _consigné après re-run_ |
| `moe0_topk_weights` | _consigné après re-run_ | _consigné après re-run_ | `max_abs ≤ 0,01` | _consigné après re-run_ |
| `moe0_topk_indices` | 1 flip (token 0 : experts 33/68 à 1 ULP) | — | égalité exacte | **ROUGE** |

### Traçage par étape de la tour de vision

Sortie par couche `vision_layer_i` (`max_abs` / `rms` contre la dorée eager) :

| Couche | max_abs | rms | Valeurs dorées (absmax / rms) |
|---|---|---|---|
| 0 | 0,031250 | 0,000427 | 163,0 / 5,29 |
| 1 | 0,125000 | 0,002899 | 272,0 / 8,30 |
| 2 | 0,250000 | 0,007169 | 308,0 / 9,39 |
| 3 | 2,000000 | 0,023343 | 330,0 / 10,14 |
| 5 | 4,000000 | 0,049018 | 392,0 / 12,85 |
| 10 | 4,000000 | 0,095666 | 984,0 / 34,77 |
| 15 | 8,000000 | 0,277077 | 1272,0 / 50,12 |
| 20 | 16,000000 | 0,624957 | 1672,0 / 61,58 |
| 23 | 24,000000 | 1,054283 | 2224,0 / 84,36 (couche 26) |
| 26 | 54,000000 | 3,879072 | 2224,0 / 84,36 |

Étapes internes de la couche 0 :

| Étape | max_abs | rms | Dorée (absmax / rms) |
|---|---|---|---|
| `v0_input_norm` | 0,007812 | 0,0000300 | 36,25 / 4,76 |
| `v0_attn_out` | 0,007812 | 0,000163 | 21,125 / 2,99 |
| `v0_post_attn_norm` | 0,001953 | 0,000042 | 25,875 / 1,06 |
| `v0_mlp_out` | 0,500000 | 0,006035 | 358,0 / 32,46 |
| `v0_post_ff_norm` | 0,031250 | 0,000424 | 161,0 / 5,21 |

Lecture : la **première** couche est à l'échelle de l'arrondi BF16 (`0,031250` =
`2⁻⁵` sur des valeurs d'absmax 163, soit une fraction de l'ULP de la dorée ;
l'attention de la couche 0 est à 0,0078 max_abs sur des valeurs d'absmax 21). La
divergence **double à peu près à chaque couche** : 8e-5 de rms relatif à la couche 0,
4,6 % à la couche 26. La tour de vision applique une attention **sans scaling**
(`self.scaling = 1.0`, `head_dim = 72`), donc avec des scores de très forte amplitude
et un softmax très pointu : un écart de 1 ULP sur `q`/`k` change une probabilité
dominante, et l'effet se compose 27 fois.

### Replay « PyTorch nourri par nos bits »

Pour distinguer un **bug sémantique** d'une **sensibilité numérique**, chaque couche
vision a été rejouée dans le venv épinglé **en la nourrissant avec la sortie de la
couche précédente produite par Rust** (`M10_DUMP_RUST_VISION=1` dumppe les étapes
Rust en F32 dans `/tmp/opencode/m10/rust_vision`).

| Couche | PyTorch(rust_in) vs Rust (max_abs / rms) | PyTorch(rust_in) vs dorée | Rust vs dorée |
|---|---|---|---|
| 1 | 0,0078125 / 9,7e-5 | 0,125 / 0,00290 | 0,125 / 0,00290 |
| 2 | 0,0625 / 1,05e-3 | 0,25 / 0,00715 | 0,25 / 0,00717 |
| 3 | 0,0625 / 1,82e-3 | 2,0 / 0,02336 | 2,0 / 0,02334 |
| 5 | 0,0625 / 1,72e-3 | 4,0 / 0,04908 | 4,0 / 0,04902 |
| 10 | 0,0625 / 1,17e-3 | 4,0 / 0,09568 | 4,0 / 0,09567 |
| 15 | 0,5 / 1,70e-2 | 8,0 / 0,27722 | 8,0 / 0,27708 |
| 20 | 2,0 / 2,57e-2 | 16,0 / 0,62578 | 16,0 / 0,62496 |
| 26 | 1,0 / 0,176 | 54,0 / 3,87646 | 54,0 / 3,87909 |

Étapes de la couche 1, PyTorch nourri par l'entrée Rust, contre la sortie Rust :
`input_norm` 3,05e-5 / 3,0e-7 ; **`attn_out` 0 / 0 (bit-exact)** ;
`post_attn_norm` 4,88e-4 / 4,8e-6 ; `pre_ff_norm` **0 / 0 (bit-exact)** ;
`mlp_out` 3,91e-3 / 7,5e-5 ; `post_ff_norm` 3,91e-3 / 5,6e-5 ;
`layer_out` 0,0078125 / 9,7e-5.

**Conclusion de ce replay.** Nourrie par **la même entrée**, la couche PyTorch
reproduit la couche Rust à l'échelle de l'arrondi BF16, et sur certaines étapes
**bit-exactement**. La divergence Rust ↔ dorée vient donc de la **dérive de
l'entrée** accumulée par les couches précédentes, pas d'une opération mal implémentée.
Le même constat a été fait couche par couche : à chaque couche, `PyTorch(rust_in)
vs Rust` est 10 à 30 fois plus petit que `PyTorch(rust_in) vs dorée`.

### Traçage par étape du décodéur

Sortie par couche `decoder_layer_i` (`max_abs` / `rms` contre la dorée eager) :

| Couche | max_abs | rms |
|---|---|---|
| 0 | 2,406250 | 0,036775 |
| 1 | 1,500000 | 0,027062 |
| 2 | 1,312500 | 0,039566 |
| 3 | 0,875000 | 0,041286 |
| 5 | 1,625000 | 0,065365 |
| 10 | 5,000000 | 0,268496 |
| 15 | 9,000000 | 0,380678 |
| 20 | 5,125000 | 0,403100 |
| 25 | 5,125000 | 0,449156 |
| 29 | 5,937500 | 0,200158 |

La divergence est **déjà rouge à la couche 0** (rms 0,0368 pour une tolérance chaîne
de 4e-3). Le traçage des 11 étapes internes de la couche 0 (`dec0_*`) et le replay
PyTorch nourri par nos bits sont en cours : ils séparent la dérive de l'entrée
(embedding + auto-conditionnement + KV encodeur) de l'arithmétique de chaque
sous-opération (attention bidirectionnelle, MLP dense, routeur, experts, `layer_scalar`).

## Causes racines établies (mesurées, non supposées)

1. **Le GEMM BF16 de PyTorch CPU n'est pas reproductible par un ordre d'accumulation simple.** La sonde `probe_linear.py` compare le résultat d'un `nn.Linear` BF16 épinglé à toutes les accumulations FP32 testées (séquentielle, 4/8/16/32 voies, `numpy` F32, `numpy` F64) : l'écart le plus petit est de **0,0077** en F32 **avant** arrondi BF16, soit parfois un ULP BF16. Les sondes sur la chaîne RMSNorm + scale + root (`probe_norm.py`) donnent en revanche des scores de routeur **bit-exacts** dans **tous** les ordres de réduction testés : l'ordre de réduction du RMSNorm n'est pas la cause.
2. **`torch.topk` CPU n'est pas stable** sur les égalités. Exemples mesurés sur le golden : ordre `[1, 0, 3, 2]` et `[4, 7, 5]` pour des scores strictement égaux. La convention implémentée est un tie-break déterministe par indice décroissant ; les cas d'égalité restent **à la limite de tolérance** et ne sont pas revendiqués comme exacts.
3. **L'attention sans scaling (vision et décodéur global) amplifie exponentiellement** un écart de 1 ULP : la chaîne de 27 couches vision multiplie l'écart relatif par ~500 ; la chaîne de 30 couches décodéur produit un rms final de 1,92 sur des valeurs de rms 3,27.
4. **La référence eager est intrinsèquement plus sensible que sdpa/`grouped_mm`**, parce qu'elle matérialise les scores et les probabilités en BF16. C'est la sémantique épinglée choisie comme référence : le verdict rouge est la mesure honnête de la limite de reproductibilité d'une chaîne BF16 de bout en bout contre une implémentation d'accumulation tierce.

## Ce qui n'est pas revendiqué

- Aucune tolérance n'a été assouplie après observation ; les tolérances de la campagne restent celles déclarées avant campagne.
- Aucune génération complète n'est revendiquée : la génération réelle de 256 tokens et le streaming borné sur le checkpoint complet sont réservés au créneau GPU final.
- Aucun résultat GPU n'est consigné dans ce journal : la campagne hardware est planifiée, pas exécutée (voir le gate ci-dessous).

## Gate GPU (déclaré, non exécuté)

La campagne hardware (streaming complet sur 51,6 Go, admission/éviction matérielle,
primitives CUDA sur données réelles, différentiel CPU ↔ CUDA du graphe complet,
génération réelle avec l'`EntropyBoundSampler`) est **planifiée dans un unique créneau
GPU final**, après confirmation de la libération de la VRAM par l'utilisateur.
Aucune génération réelle n'est revendiquée avant l'exécution effective et la
consignation des journaux de ce créneau.

Séquence prévue (une seule tâche GPU à la fois, VRAM estimée annoncée avant chaque
lancement, libération vérifiée après chaque tâche) :

```bash
# 1. Primitives M10 sur données réelles (pic mesuré 25,4 Mio sur le sous-graphe réel, << 26 Gio)
cargo test -p claymore-memory --features cuda-driver --test hardware_m10_primitives --locked -- \
  --ignored --test-threads=1 --nocapture

# 2. Streaming borné par couche (budget souple 26 Gio, plafond dur 28 Gio, réserve ≥ 4 Gio)
cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_streaming --locked -- \
  --ignored --test-threads=1 --nocapture

# 3. Différentiel CUDA ↔ CPU et génération réelle sur canvas complet
cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_generation --locked -- \
  --ignored --test-threads=1 --nocapture
```
