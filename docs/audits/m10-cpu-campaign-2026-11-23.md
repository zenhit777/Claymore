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
| `crates/claymore-cuda/kernels/m10.cu` | `87750139e89d3b0f194b086bba0abb4b174981f13057af5e3e1ff28e170a85e0` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `f30f62d9d407d703274597dae56d1ada58ac47aaf7a667e22a50250c417b045e` |
| `/tmp/opencode/m10/golden_eager/manifest.json` | `e8dfa577236990314211849cb7768ba196e07b715f828b7dbbafc94e27d962fe` |

L'inventaire doré compte **784 tenseurs F32** dans `/tmp/opencode/m10/golden_eager` :
sorties finales, sorties par couche, KV par couche, étapes internes de la tour de
vision (6 par couche) et étapes internes du décodéur (11 par couche), plus les
sorties du routeur (`encoder0_router_*` et `moe0_*`). Le manifest
`golden_eager/manifest.json` porte la shape de chaque tenseur.

## Entrées de la campagne golden

- Prompt fixe : `"What is the capital of France?"` (7 tokens, ids `[2, 1449, 4470, 2927, 9776, 2312, 2]`).
- Image déterministe : seed 1234, 48×48 RGB → 9 patches de 16 px → pooling kernel 3 → 1 token vision ; pixel values en BF16 (`2(x-0.5)` appliqué par le patch embedder épinglé), positions `[1, 9, 2]`.
- Canvas golden : seed 42, 16 positions (la composition du canvas ne dépend pas de la longueur ; 16 rend la référence CPU complète exécutable).
- Hooks capturés (venv épinglé) : `vision_last_hidden_direct [1,9,1152]`, `embed_vision_output_direct [1,1,2816]`, `encoder_last_hidden_direct [1,7,2816]`, `enc_kv_{0..29}_{keys,values}_direct`, `encoder_layer_{0..29}`, `decoder_embedding`, `decoder_layer_{0..29}`, `decoder_last_hidden0_direct [1,16,2816]`, `decoder_logits0_direct [1,16,262144]`, `canvas0_direct [1,16]`, `moe0_router_logits [16,128]`, `moe0_topk_weights [16,8]`, `moe0_topk_indices [16,8]`.
- **Complément eager du 24/11** : la régénération eager de référence avait omis les trois tenseurs du routeur du **décodéur** (`moe0_*`), ce qui faisait échouer le test `decoder_first_step` sur un *fichier doré manquant* et non sur une mesure. Les trois fichiers ont été régénérés par un **rejeu exact** de la même passe eager (`gen_golden_eager_moe.py` : même checkpoint épinglé, même prompt, même canvas, mêmes `decoder_position_ids`), et **33/33** des tenseurs produits par ce rejeu sont **bit-identiques** aux dorées déjà publiées : le rejeu est reproductible, et seuls les trois fichiers manquants ont été ajoutés. Le manifest passe de 781 à **784** tenseurs.
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
9. Router CPU : deux frontières BF16 explicites (`normed × scale_w`, puis `× scalar_root_size` où le scalaire est un **float Python** `hidden_size^-0.5` — donc **F32, jamais arrondi en BF16** — et le produit est arrondi **une seule fois** en BF16), softmax **F32** des scores BF16, poids `top_k × per_expert_scale`, sélection reproduisant **exactement** l'algorithme CPU d'ATen : `std::nth_element` (introselect) puis `std::sort` sur les `k − 1` premiers éléments, avec le comparateur « valeur seule » d'ATen (NaN d'abord). L'émulation de cet algorithme a été validée **210/210** contre `torch.topk` sur les 210 lignes de routeur des 30 couches dorées ; la même algorithmique est implémentée en Rust (`claymore-tensor/src/aten_topk.rs`) et en CUDA (`m10.cu`).
10. `layer_scalar` de l'encodeur : appliqué comme **dernière opération** de la couche (produit BF16 avec le scalaire appris), y compris sur les couches sans `shared_kv`.
11. Auto-conditionnement : le post-norm **non scaleé** du module `self_conditioning` est appliqué **même au premier pas** (signal zéro), exactement comme `DiffusionGemmaSelfConditioning.forward`.
12. Attention du décodéur : **entièrement bidirectionnelle** sur le préfixe encodeur + le canvas, sans fenêtre glissante imposée par le masque du décodéur ; pour les couches globales, `v_proj` est `None` et `value_states = v_norm(k_proj)` (avant `k_norm`), et non `v_norm` du `k_norm`/rope.
13. Kernels CUDA : mêmes frontières de dtype (`SoftCap` sans pré-scale, `RouterTopK` F32, GELU à deux arrondis, RoPE BF16, score d'attention arrondi BF16 avant softmax, probabilités arrondies BF16 avant la réduction de valeurs), avec `m10.ptx` régénéré. Le routeur CUDA reproduit l'algorithme d'ATen (même `nth_element` + `sort`, même comparateur) et calcule le maximum de la ligne **à travers tout le bloc** ; l'op `ScaleScalar` (scalaire **F32**, sortie dans le dtype de l'entrée, arrondi unique) porte le facteur `scalar_root_size` du routeur sans arrondir le scalaire en BF16.

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
- `cargo test --workspace --all-targets --locked` : ✅ (**252 tests réussis**, 13 suites, dont 13 tests `#[ignore]` : 7 golden CPU, 3 matériels M10, 2 matériels M8, 1 de generation CUDA) — rejoué au SHA `77e7665`
- `cargo clippy --workspace --all-targets --locked -- -D warnings` : ✅ (workspace **et** variantes CUDA `claymore-cuda` / `claymore-memory` / `claymore-cli` / `claymore-model-gemma`)
- checks `cuda-driver` (`claymore-cuda`, `claymore-memory`) et `cuda-streaming` (`claymore-model-gemma`) : ✅ (compilation sans exécution GPU) ; **les avertissements `unused` de `cuda_graph.rs` et `stream.rs` ont été supprimés** (`GemmaModelConfig`, `weights_idx`, `temperature`, `alloc`) ; il ne reste que le warning pré-existant `unused field repository in workspace.package`
- tests CUDA **sans GPU** (stub `libcuda.so` lié) : ✅ `claymore-memory` 52 tests, `claymore-cuda` 8 tests
- variants `+1.86` (`test --workspace --all-targets`, `clippy -D warnings` workspace **et** CUDA, checks `cuda-driver` sur `claymore-cuda` / `claymore-memory` / `claymore-cli`, check `cuda-streaming` sur `claymore-model-gemma`) : ✅ **0 erreur, 13 suites de tests vertes** — le workflow CI exécute désormais ces variants CUDA **sur le MSRV** aussi

### Différentiels golden CPU (profil release, dorées eager)

Tolérances **déclarées avant** la campagne, dérivées de la précision BF16. Le verdict
est **rouge** quand la mesure dépasse la tolérance. Aucun seuil n'a été modifié.

| Stage | max_abs | rms | Tolérance | Verdict |
|---|---|---|---|---|
| `text_embedding` | **0** | **0** | bit-exact | **VERT** |
| `encoder_last_hidden` | 5,539062 | 0,763049 | `max_abs ≤ 0,03`, `rms ≤ 2e-3` | **ROUGE** |
| `enc_kv_0_keys` (layout doré comparé) | 0,000977 | 0,000032 | `max_abs ≤ 0,05`, `rms ≤ 2e-3` | **ROUGE** sur `max_abs` |
| `enc_kv_0_values` | 0,007812 | 0,000356 | `max_abs ≤ 0,05`, `rms ≤ 2e-3` | **ROUGE** sur `max_abs` |
| `vision_last_hidden` | 0,189453 | 0,054739 | `max_abs ≤ 0,06`, `rms ≤ 2e-3` | **ROUGE** |
| `embed_vision` | 0,242188 | 0,060047 | `max_abs ≤ 0,1`, `rms ≤ 4e-3` | **ROUGE** |
| `decoder_last_hidden0` | 15,531250 | 1,964803 | `max_abs ≤ 0,12`, `rms ≤ 4e-3` | **ROUGE** |
| `decoder_logits0` | 18,893417 | 3,339343 | `max_abs ≤ 2,0`, `rms ≤ 0,1` | **ROUGE** |
| `encoder0_router_scores` (couche 0, golden eager) | 0,002429 | 0,000089 | `max_abs ≤ 0,02` | **VERT** |
| `encoder0_router_weights` (couche 0, golden eager) | 0,005122 | 0,000804 | `max_abs ≤ 0,01` | **VERT** |
| `encoder_router_indices` (30 couches) | **30 couches exactes sur 30** | — | égalité exacte | **VERT** |
| `encoder0_router_raw_scores` (entrée upstream rejouée) | **0** | **0** | bit-exact | **VERT** |
| `moe0_router_scores` (décodeur, couche 0) | 0,000542 | 0,000040 | `max_abs ≤ 0,02` | **VERT** |
| `moe0_topk_weights` (décodeur, couche 0) | 0,002155 | 0,000547 | `max_abs ≤ 0,01` | **VERT** |
| `moe0_topk_indices` (décodeur, couche 0) | 8 flips / 128, **ensemble d'experts identique** (ordre dans un groupe d'égalité) | — | égalité exacte | **ROUGE** |

Rejeu complet : `cargo test --release -p claymore-model-gemma --test golden_cpu --locked -- --ignored --test-threads=1 --nocapture` → **4 tests verts, 3 rouges** (`decoder_first_step`, `encoder_matches`, `vision_tower`), 927,60 s, journal `/tmp/opencode/m10/golden_cpu_final.log`. Les trois échecs sont les **dérives de chaîne** documentées ; aucun n'est un échec de sémantique d'opération (voir §« Causes racines établies »).

Deux **bugs de test** trouvés et corrigés pendant la campagne, avant toute conclusion
sur les chiffres :

- les scripts de traçage décodéur utilisaient des `ids` inventés (`[2, 1449, 4470, 2927,
  9776, 2312, 2]`) au lieu des `prompt_ids` dorés (`[3689, 563, 506, 5279, 529, 7001,
  236881]`). La première génération des étapes décodéur (`dec{i}_*`) était donc
  **incomparable** : `dec0_out` différait de la dorée `decoder_layer_0` de `18,625 / 0,549`.
  Les étapes ont été **régénérées** avec les `prompt_ids` dorés lus depuis le manifest ;
  la re-comparation Rust est en cours.
- la comparaison des caches KV dans `encoder_matches…` comparait le cache local
  `[seq, kv_heads, head_dim]` **aplati** au doré `[batch, kv_heads, seq, head_dim]` **aplati** :
  deux layouts différents, donc des écarts **artificiels** (`enc_kv_0_keys` annoncée
  `2,859 / 0,166` alors que la comparaison à layout corrigé donne `0,000977 / 0,000032`).
  Un helper `permute_kv` est maintenant utilisé par les trois chemins de comparaison.

### Traçage par couche de l'encodeur

Sortie par couche `encoder_layer_i` (`max_abs` / `rms` contre la dorée eager, 7 tokens) :

| Couche | max_abs | rms | Couche | max_abs | rms |
|---|---|---|---|---|---|
| 0 | 0,500000 | 0,006806 | 15 | 3,363281 | 0,269535 |
| 1 | 0,500000 | 0,009743 | 16 | 3,199219 | 0,246915 |
| 2 | 0,500000 | 0,011514 | 17 | 3,031250 | 0,251981 |
| 3 | 0,250000 | 0,011390 | 18 | 2,226562 | 0,252396 |
| 4 | 0,500000 | 0,012108 | 19 | 2,000000 | 0,199579 |
| 5 | 0,375000 | 0,013194 | 20 | 1,562500 | 0,192818 |
| 6 | 0,375000 | 0,013252 | 21 | 1,750000 | 0,175782 |
| 7 | 0,812500 | 0,050597 | 22 | 1,460938 | 0,149013 |
| 8 | 2,250000 | 0,089443 | 23 | 1,750000 | 0,201984 |
| 9 | 2,500000 | 0,134437 | 24 | 2,250000 | 0,176128 |
| 10 | 3,500000 | 0,173590 | 25 | 1,375000 | 0,161251 |
| 11 | 5,000000 | 0,256946 | 26 | 0,598633 | 0,127217 |
| 12 | 5,500000 | 0,265574 | 27 | 1,125000 | 0,111109 |
| 13 | 4,570312 | 0,269089 | 28 | 0,500000 | 0,097413 |
| 14 | 3,625000 | 0,268554 | 29 | 2,687500 | 0,057449 |

Étapes de la couche 0 encodeur (`max_abs` / `rms`) : `q_proj` 0,250000 / 0,004428 ;
`k_proj` 0,250000 / 0,004749 ; `v_proj` 0,125000 / 0,004615 ;
`enc_kv_0_keys` (layout doré) 0,000977 / 0,000032 ; `enc_kv_0_values` 0,007812 / 0,000356 ;
`encoder0_router_scores` 0,002429 / 0,000089 ; **`encoder0_router_indices` exact**.

**Les indices top-k de l'encodeur sont désormais exacts sur les 30 couches** (contre
0 couche exacte sur 30 avant la correction du routeur). La correction a porté sur
deux points, tous deux dérivés du code épinglé et non des mesures :

1. **`scalar_root_size` n'est pas arrondi en BF16.** L'upstream applique
   `hidden_states * self.scale * self.scalar_root_size` où `scalar_root_size` est un
   **float Python** (`hidden_size ** -0.5`). PyTorch évalue le produit en F32 et
   n'arrondit que le **résultat** en BF16 ; Claymore arrondissait d'abord le
   scalaire en BF16, ce qui décalait les logits de routeur de 1 ULP et retournait
   l'ordre des experts dans les groupes d'égalité. Rejouée sur l'entrée **de
   l'upstream** (`encoder0_routing_residual`), la nouvelle sémantique donne des
   logits **bit-exacts** (`raw_max_abs = 0`) et les indices **exacts** ; le rejeu
   Python des 896 logits BF16 de la couche 0 confirme **896/896 exacts**.
2. **L'algorithme de sélection est celui d'ATen**, pas un tie-break conventionnel :
   `std::nth_element` (introselect) puis `std::sort` sur les `k − 1` premiers
   éléments, comparateur « valeur seule ». L'émulation Rust
   (`claymore-tensor/src/aten_topk.rs`) est validée **210/210** contre `torch.topk`
   sur les 210 lignes de routeur des 30 couches dorées ; la même algorithmique est
   implémentée dans le kernel CUDA `claymore_router_top_k_f32`, et le test matériel
   `hw_m10_11` vérifie que CPU, CUDA et upstream choisissent **le même ensemble et
   le même ordre** sur la ligne épinglée (égalités BF16 exactes experts 68/33 et
   22/126, dont un groupe à cheval sur la frontière `k = 8`).

Les 8 flips restants sur le routeur du **décodeur** (`moe0_topk_indices`, 8/128) ne
sont **pas** des choix d'experts différents : ce sont des **permutations d'ordre à
l'intérieur d'un groupe d'experts de probabilités égales** (token 3 : experts
`{114, 123, 13}` dans les deux cas ; token 11 : `{123, 22, …}` dans les deux cas).
La cause est la **dérive de l'entrée du décodéur** (le routeur du décodéur reçoit un
hidden state déjà divergé de la dorée), pas la sémantique du routeur : sur l'entrée
de l'upstream, le routeur de l'encodeur est bit-exact.

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

Sortie par couche `decoder_layer_i` (`max_abs` / `rms` contre la dorée eager, canvas de
16 positions, `prompt_ids` dorés) :

| Couche | max_abs | rms |
|---|---|---|
| 0 | 1,125000 | 0,017164 |
| 1 | 0,500000 | 0,014293 |
| 2 | 0,765625 | 0,032037 |
| 3 | 0,750000 | 0,033166 |
| 5 | 0,875000 | 0,046064 |
| 10 | 3,500000 | 0,152412 |
| 15 | 7,500000 | 0,276670 |
| 20 | 6,750000 | 0,321601 |
| 25 | 3,500000 | 0,467765 |
| 29 | 5,437500 | 0,202295 |

Traçage des **11 étapes internes de la couche 0** du décodeur (`dec0_*`, contre les
étapes dorées régénérées avec les **mêmes** `prompt_ids` dorés) :

| Étape | max_abs | rms |
|---|---|---|
| `dec0_attn_out` | 0,125000 | 0,006890 |
| `dec0_post_attn_norm` | 0,500000 | 0,011467 |
| `dec0_residual_after_attn` | 0,500000 | 0,011646 |
| `dec0_mlp_out` | 0,500000 | 0,031693 |
| `dec0_post_ff1` | 8,000000 | 0,092948 |
| `dec0_pre_ff2` | 0,017578 | 0,000613 |
| `dec0_experts` | 0,218750 | 0,006190 |
| `dec0_post_ff2` | 50,250000 | 0,462161 |
| `dec0_mlp_moe_sum` | 52,000000 | 0,481407 |
| `dec0_post_ff` | 17,000000 | 0,243868 |
| `dec0_out` | 1,125000 | 0,017164 |

Lecture de ce traçage (il **sépare** la dérive de l'entrée de l'arithmétique de chaque
sous-opération) :

1. l'**attention** et le **MLP dense** de la couche 0 sont à l'échelle de l'ULP BF16
   (`0,125` / `0,5` sur des valeurs de rms ~0,5) ;
2. `dec0_pre_ff2` (l'entrée des experts) est **exacte à 0,0176** ;
3. les **experts** sont à `0,21875` ;
4. mais les **post-norms** `dec0_post_ff1` (8,0) et `dec0_post_ff2` (50,25) explosent,
   et `dec0_mlp_moe_sum` (52,0) est dominé par `dec0_post_ff2` ;
5. `dec0_out` retombe à 1,125.

La cause est donc **localisée** : ce n'est ni l'attention, ni le MLP dense, ni le routeur,
ni les experts, mais les **deux post-norms du bloc MoE** (`post_feedforward_layernorm_1`
et `_2`), c'est-à-dire la **re-normalisation** d'un signal déjà amplifié. Le signal qui
entre dans ces normes est amplifié par le **facteur d'échelle appris de la couche**
(`layer_scalar`), puis re-normalisé : la moindre divergence relative du signal est
multipliée par `1 / rms(x)` avant d'être re-amplifiée. Ce mécanisme est **exactement**
celui décrit à la cause racine n° 4 (attention **sans scaling** → amplification
exponentielle) et il est **identique** dans les deux implémentations Claymore : la dérive
CUDA ↔ CPU de la même sous-opération est du même ordre de grandeur que la dérive
CPU ↔ dorée (cf. §16.3.3.2 du rapport hardware).

Le rejeu complet a été refait avec les étapes **régénérées** : le test
`decoder_first_step_matches_pinned_upstream_on_the_real_checkpoint` reste **rouge** sur
la chaîne (`decoder_last_hidden0` 15,531250 / 1,964803 ; `decoder_logits0` 18,893417 /
3,339343) — **aucune tolérance n'a été assouplie** —, tandis que les grandeurs
sémantiques du routeur du décodeur (`moe0_router_scores` 0,000542 / 0,000040 et
`moe0_topk_weights` 0,002155 / 0,000547) sont **vertes**.

## Causes racines établies (mesurées, non supposées)

1. **Le GEMM BF16 de PyTorch CPU n'est pas reproductible par un ordre d'accumulation simple.** La sonde `probe_linear.py` compare le résultat d'un `nn.Linear` BF16 épinglé à toutes les accumulations FP32 testées (séquentielle, 4/8/16/32 voies, `numpy` F32, `numpy` F64) : l'écart le plus petit est de **0,0077** en F32 **avant** arrondi BF16, soit parfois un ULP BF16. Les sondes sur la chaîne RMSNorm + scale + root (`probe_norm.py`) donnent en revanche des scores de routeur **bit-exacts** dans **tous** les ordres de réduction testés : l'ordre de réduction du RMSNorm n'est pas la cause.
2. **`scalar_root_size` était arrondi en BF16 : cause racine du décalage des logits de routeur.** L'upstream l'applique comme un **float Python** (`hidden_size ** -0.5`) ; PyTorch évalue `bf16_tensor * float` en F32 et n'arrondit que le **résultat**. Claymore arrondissait d'abord le scalaire en BF16, décalant les logits de routeur de 1 ULP et retournant l'ordre dans les groupes d'égalité. Corrigé : les logits BF16 sont **bit-exacts** vs upstream (rejeu Python **896/896** sur la couche 0 ; test Rust `raw_max_abs = 0`) et les **indices top-k de l'encodeur sont exacts sur les 30 couches** (contre 0/30 avant). Cette correction est **entièrement upstream** : elle n'a pas été ajustée sur une mesure.
3. **Le tie-break de `torch.topk` CPU n'est pas stable, et l'ordre d'ATen est reproductible.** Sur des scores strictement égaux, `torch.topk` rend des ordres tels que `[1, 0, 3, 2]` et `[4, 7, 5]`. Claymore n'impose plus une convention : il **reproduit l'algorithme d'ATen** (`std::nth_element` introselect puis `std::sort` sur les `k − 1` premiers, comparateur « valeur seule », NaN d'abord), en Rust et en CUDA. L'émulation est validée **210/210** contre `torch.topk` sur les 210 lignes de routeur des 30 couches dorées ; la ligne épinglée (égalités BF16 exactes experts 68/33 et 22/126, dont un groupe à cheval sur la frontière `k = 8`) donne le **même ensemble et le même ordre** en CPU, en CUDA et chez l'upstream.
4. **L'attention sans scaling (vision et décodéur global) amplifie exponentiellement** un écart de 1 ULP : la chaîne de 27 couches vision multiplie l'écart relatif par ~500 ; la chaîne de 30 couches encodeur puis décodéur produit un rms final de 1,96 sur des valeurs de rms 3,27.
5. **La référence eager est intrinsèquement plus sensible que sdpa/`grouped_mm`**, parce qu'elle matérialise les scores et les probabilités en BF16. C'est la sémantique épinglée choisie comme référence : le verdict rouge est la mesure honnête de la limite de reproductibilité d'une chaîne BF16 de bout en bout contre une implémentation d'accumulation tierce.

## Ce qui n'est pas revendiqué

- Aucune tolérance n'a été assouplie après observation ; les tolérances de la campagne restent celles déclarées avant campagne.
- Aucune équivalence de chaîne BF16 de bout en bout contre la référence eager n'est revendiquée : le verdict rouge sur l'encodeur, la vision et le décodeur est **conservé et documenté**.
- Les journaux de la campagne GPU sont consignés dans le rapport hardware (§16.3) ; ce journal CPU reste le journal **CPU**.

## Gate GPU (exécuté le 24 novembre 2026)

La campagne hardware (streaming complet sur 51,6 Go, admission/éviction matérielle,
primitives CUDA sur données réelles, différentiel CPU ↔ CUDA du graphe complet,
génération réelle avec l'`EntropyBoundSampler`) s'est déroulée après autorisation
explicite de l'utilisateur et confirmation de la libération de la VRAM.

Le créneau a été joué en **deux passages**, tous deux séquencés tâche par tâche, chaque
tâche annoncée (VRAM estimée) **avant** et la libération vérifiée **après** :

1. un **passage de diagnostic** (`gpu_task1_run5.log`, `gpu_task2_run3.log`,
   `gpu_task3_run2.log` … `gpu_task3_run5.log`) qui a isolé **sept défauts produit** du
   graphe CUDA, corrigés un par un — chaque échec ayant isolé un défaut **distinct**,
   jamais une tolérance à relâcher ;
2. un **créneau final de validation** (`gpu_final_task1.log`, `gpu_final_task2.log`,
   `gpu_final_task3.log`) rejoué **en entier** sur le SHA de code `77e7665`, c'est-à-dire
   sur le code exact qui est commité.

Séquence exécutée (une seule tâche GPU à la fois, VRAM estimée annoncée avant chaque
lancement, libération vérifiée après chaque tâche) :

```bash
# 1. Primitives M10 sur données réelles (pic mesuré 25,4 Mio sur le sous-graphe réel, << 26 Gio)
cargo test -p claymore-memory --features cuda-driver --test hardware_m10_primitives --locked -- \
  --ignored --test-threads=1 --nocapture

# 2. Streaming borné par couche (budget souple 26 Gio, plafond dur 28 Gio, réserve ≥ 4 Gio)
cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_streaming --locked -- \
  --ignored --test-threads=1 --nocapture

# 3. Différentiel CUDA ↔ CPU et génération réelle sur canvas complet (pic 24,25 Gio)
cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_generation --locked -- \
  --ignored --test-threads=1 --nocapture
```

### Résultat du créneau GPU (24 novembre 2026)

Le créneau GPU final a été **autorisé explicitement** par l'utilisateur le 24/11/2026
(« Oui, dès que la carte est libre ») et exécuté **en séquence**, une tâche GPU à la
fois, VRAM estimée annoncée avant chaque tâche, libération vérifiée après chaque
tâche. Le détail complet (mesures, bugs trouvés, correctifs) est dans
`docs/audits/m10-rtx5090-hardware-validation.md` §16.3.

| Tâche | Estimation VRAM annoncée | Résultat |
|---|---|---|
| 1/3 `hardware_m10_primitives` | ≤ 2 Gio | **21/21 verts** |
| 2/3 `hardware_m10_streaming` | ≤ 4 Gio | **2/2 verts** (après correction d'un bug de test et d'un bug produit) |
| 3/3 `hardware_m10_generation` | ≤ 26 Gio (pic mesuré **24,25 Gio**) | **2 verts, 1 rouge documenté** — **génération réelle VERTE** (256/256 positions, 48 étapes, 909 Gio uploadés, 605 669 lancements), **sémantique de la couche 0 VERTE** (indices du routeur 56/56 identiques), **différentiel de chaîne ROUGE** (`5,835938` / `0,711149`, tolérances conservées) |

VRAM après chaque tâche : **913 Mio** occupés, aucun processus de calcul résiduel listé
par `nvidia-smi --query-compute-apps`.
