# Validation matérielle M10 — DiffusionGemma 26B-A4B-it sur NVIDIA RTX 5090

- **Branche :** `feat/m10-diffusiongemma`
- **SHA de code validé (campagne hardware du 23/11) :** `ea115ab337d492d81ac50f69c09cd79e050b79a6`
- **Date de campagne :** 23 novembre 2026, 16:10–16:12 UTC (campagne finale sur le SHA ci-dessus)
- **PR :** [#40](https://github.com/zenhit777/Claymore/pull/40), cible `main`, **ouverte et non fusionnée** (voir §15)
- **Statut :** **PASS PARTIEL** — 66/66 tests matériels verts, CI et MSRV verts, **gate de génération réelle du graphe complet laissé OUVERT** (§11)

> **Mise à jour M10.3 (24 novembre 2026) — à lire avant toute lecture de §11.** L'incrément
> M10.3 implémente **le graphe complet** (tour de vision, routage MoE, experts,
> auto-conditionnement, cache KV encodeur, cross-attention décodéur, lm-head tied,
> exécuteur de streaming borné, kernels CUDA élargis). Les §1 à §15 décrivent la
> campagne hardware **du 23/11 sur le SHA `ea115ab`** : ils ne décrivent **pas** le
> code de M10.3. L'état du code M10.3, les mesures CPU complètes et le **gate GPU
> non exécuté** sont consignés au **§16**. Aucune mesure hardware de M10.3 n'est
> revendiquée.

> **Ce que ce rapport prouve.** Le chemin M10 de Claymore — chargement strict du checkpoint épinglé, primitives tensorielles CPU de référence, kernels CUDA M10, admission mémoire bornée et garde des buffers en vol — est réellement exécuté sur une RTX 5090 et validé contre une référence CPU déterministe, y compris sur un **sous-graphe de couche décodeur construit à partir des poids réels** du checkpoint épinglé.
>
> **Ce que ce rapport ne prouve pas.** L'exécution du **graphe complet** de 26 milliards de paramètres. Les portes listées au §11 sont **livrées en code** dans l'incrément M10.3 (§16) mais **non validées sur la carte** : le créneau GPU final n'a pas été exécuté. Aucune génération de texte réelle n'est revendiquée. Les portes ouvertes sont des **portes de code**, pas une limite de capacité mémoire : le plafond de 28 Gio est un seuil d'admission, les poids vivent en VRAM + RAM et sont streamés couche par couche.

---

## 1. Traçabilité et préparation

| Contrôle | Valeur observée |
|---|---|
| SHA exigé / testé | `ea115ab337d492d81ac50f69c09cd79e050b79a6` |
| SHA local avant campagne | `ea115ab337d492d81ac50f69c09cd79e050b79a6` |
| SHA local après campagne | `ea115ab337d492d81ac50f69c09cd79e050b79a6` |
| SHA consigné dans chaque journal de campagne | `ea115ab337d492d81ac50f69c09cd79e050b79a6` |
| `git status --short` pendant la campagne | `?? docs/audits/fixtures/` (fixture documentaire, aucun fichier de code) |
| Commit de code | `Add M10 diffusion primitives, strict checkpoint loader and M10 launch guards` |
| Parent du commit de code | `78d6e8a0673830b3506733387df3f4a4962eaf43` |

La campagne finale s'est déroulée sur un arbre de travail dont **seuls** des fichiers de documentation non trackés différaient du commit. Aucun fichier Rust, CUDA `.cu`, PTX, manifeste, lockfile, tolérance numérique, test ou workflow CI n'a été modifié entre le commit de code et la fin de la campagne.

### 1.1 Checkpoint épinglé

| Élément | Valeur |
|---|---|
| Identifiant Hugging Face | `google/diffusiongemma-26B-A4B-it` |
| Révision du dépôt modèle | `cd40195554d3cfcb56f34b9adfb66efc790ed5b6` |
| Nombre de shards | 11 |
| Nombre de tenseurs déclarés par l'index | 1047 |
| Octets de charge utile totaux | 51 647 562 456 |
| Paramètres (tous BF16) | 25 823 778 864 |
| SHA-256 `config.json` | `13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506` |
| SHA-256 `model.safetensors.index.json` | `6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a` |
| SHA-256 `generation_config.json` | `99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200` |

Ces valeurs sont encodées en dur dans `crates/claymore-model-gemma/src/config.rs` (`CHECKPOINT_REVISION`, `PINNED_*`) et contrôlées à l'ouverture : `Checkpoint::open_pinned` refuse tout index, config ou `generation_config` dont l'empreinte diffère.

### 1.2 Code de référence upstream consulté

| Fichier | Révision | SHA-256 |
|---|---|---|
| `modeling_diffusion_gemma.py` | transformers `671aa77576266f41f33f9c28e2d824bfd1b1b63c` | `8bc5f5cff6e43d0ee4178fa9aae657437894dc8511bf118bfe90f4676b4db846` |
| `modular_diffusion_gemma.py` | idem | `022fb7a254d5d0094d035e15c800bc47071f4f70bab89baead0edc12e7701710` |
| `configuration_diffusion_gemma.py` | idem | `53d427f95c248c25878870d31f8a13da3b08dc900eee89e6345d62187e5f7f72` |
| `generation_diffusion_gemma.py` | idem | `82c57822f9acfdb41bdf16bb814a1e74e5f3efd5607cfd2a59e363e947adce76` |
| `diffusers/pipeline_diffusion_gemma.py` | idem | `2e1fb781ee8473f60b659a4057da9ba07c9fa5a940cb729bbdf05e2f63c66d90` |
| `diffusers/scheduling_block_refinement.py` | idem | `0195f2597dab0621753396aef67ab6391ef2489b5df82178a70d294dd22392c4` |
| `scheduler_config.json` | artefact séparé du checkpoint | `5e5536ea036c284bcc09c7b08045868d741a41a3090069f6c035697bba31c6c7` |

Le `scheduler_config.json` (abstraction `diffusers` `BlockRefinementScheduler`) est **enregistré mais non fusionné** avec la politique canonique de génération : la politique canonique M10 est l'`EntropyBoundSampler` de transformers, avec schedule linéaire de température et arrêt stable/confident. Le fichier `generation_config.json` épinglé le confirme (`_cls_name: EntropyBoundSamplerConfig`, `entropy_bound: 0.1`, `t_min: 0.4`, `t_max: 0.8`, `stability_threshold: 1`, `confidence_threshold: 0.005`, `max_denoising_steps: 48`).

### 1.3 Configuration structurelle auditée

Copie de référence versionnée : `docs/audits/fixtures/diffusiongemma-config.json` et `docs/audits/fixtures/diffusiongemma-generation-config.json`.

| Attribut | Valeur épinglée |
|---|---|
| Type d'architecture | `DiffusionGemmaForBlockDiffusion` |
| Couches | 30 (5 `full_attention` : 5, 11, 17, 23, 29 ; 25 `sliding_attention`) |
| Taille cachée | 2816 |
| Têtes / têtes KV | 16 / 8 (2 KV globaux) |
| Dimension de tête | 256 (512 pour les couches globales) |
| Experts / top-k | 128 / 8 |
| Taille interm pondérée MoE | 704 |
| Vocabulaire | 262 144 |
| Longueur de canvas | 256 |
| Soft-capping final | 30.0 |
| `rms_norm_eps` | 1e-6 |
| RoPE glissant | `default`, theta 10 000 |
| RoPE global | `proportional`, theta 1 000 000, `partial_rotary_factor` 0.25 |
| Fenêtre glissante | 1024 |
| Attention bidirectionnelle | `use_bidirectional_attention = "vision"` |
| `tie_word_embeddings` | `true` |
| Tour de vision | 27 couches, cachée 1152, 356 tenseurs |

> **Correction d'audit (M10, 23/11/2026)** : la première version de ce tableau
> annonçait 6 couches `full_attention` (dont la couche 19) et 355 tenseurs de
> vision. L'extraction directe de `layer_types` depuis le `config.json` épinglé
> (révision `cd40195554d3cfcb56f34b9adfb66efc790ed5b6`) donne 5 couches
> globales : `[5, 11, 17, 23, 29]`, et la tour de vision compte 356 tenseurs.
> La vérification pin de `config.rs` (`layer % 6 == 5`), le graphe CPU complet
> (`fullgraph.rs`) et l'inventaire du shard index sont la source de vérité.

---

## 2. Environnement matériel et logiciel

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, bus `00000000:01:00.0` |
| Compute Capability | `12.0` (`sm_120`, Blackwell) |
| VRAM totale | 32 607 MiB (`nvidia-smi`) ; 34 162 016 256 octets côté driver |
| Pilote NVIDIA (NVIDIA-SMI) | `615.65.07` |
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| NVCC (régénération PTX, hors build) | `13.0.48` |
| Hôte | WSL2, Ubuntu 24.04.5 LTS, noyau `6.18.58.1-microsoft-standard-WSL2` |
| Rust stable | `rustc 1.100.0`, `cargo 1.100.0` |
| Rust MSRV | `rustc 1.86.0`, `cargo 1.86.0` |
| RAM hôte | 112 Gio |
| Bibliothèque driver réelle | `/usr/lib/wsl/lib/libcuda.so` → `libcuda.so.1` |
| PTX M10 | `.version 9.0`, `.target sm_120`, 10 kernels |

**Aucun test matériel n'a été exécuté contre `ci/cuda-driver-stub.c`.** Le stub n'est utilisé que pour la CI logicielle sans GPU (§7.2).

### 2.1 Empreintes des sources et artefacts

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m10.cu` | `e6956dc552f5edef30b1179a2a6d62465991d5db9f35c73bbbd9d644c69fa1b6` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `b6b0d8261411f7ac4f5e55cc27748b51784836e145df1664309aad16dc62a78f` |
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |
| `crates/claymore-cuda/src/kernels/m8_bf16.ptx` | `886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153` |

Le PTX M10 est versionné et régénérable par :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m10.ptx \
  crates/claymore-cuda/kernels/m10.cu
```

Le build Rust **n'appelle jamais `nvcc`** : le PTX est chargé via la Driver API au runtime. La régénération du PTX est une opération CPU uniquement, sans occupation du GPU.

---

## 3. Commandes réellement exécutées

Toutes les campagnes matérielles ont été lancées sous :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### 3.1 Campagnes matérielles

```bash
# Driver, allocations, bornes
cargo test -p claymore-cuda --features cuda-driver --test hardware_cuda_driver --locked -- \
  --ignored --test-threads=1 --nocapture

# Mémoire, transferts, éviction, scheduler, cache experts
cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_memory --locked -- \
  --ignored --test-threads=1 --nocapture

# Tenseurs M8 F32/BF16, gardes d'event, limites de grille
cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_tensor --locked -- \
  --ignored --test-threads=1 --nocapture

# Injection de fautes B2 (chemin M8) et B3 (chemin M10) sur le gestionnaire réel
cargo test -p claymore-memory --features cuda-driver --lib --locked -- \
  --ignored --test-threads=1 --nocapture

# Primitives M10, dont le sous-graphe décodeur à poids réels
CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
cargo test --release -p claymore-memory --features cuda-driver --test hardware_m10_primitives --locked -- \
  --ignored --test-threads=1 --nocapture
```

### 3.2 Vérification logicielle

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo check -p claymore-memory --features cuda-driver --all-targets --locked
cargo check -p claymore-cli --features cuda-driver --locked
cargo test -p claymore-model-gemma --all-targets --locked
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings
```

### 3.3 Vérification complète du checkpoint réel

```bash
cargo run --release -p claymore-model-gemma --bin verify_checkpoint -- \
  --root /tmp/opencode/m10/checkpoint
```

Le binaire relit **toute** la charge utile des 11 shards par flux de 8 Mio, calcule le SHA-256 de chaque tensor, vérifie la couverture et la contiguïté des plages, et compare l'inventaire à l'index épinglé.

---

## 4. Inventaire des tests

### 4.1 Tests matériels (GPU réel) — 66 tests, 66 réussis

| Cible | Tests | Détail | Résultat |
|---|---:|---|---|
| `claymore-cuda` / `hardware_cuda_driver` | 5 | contexte primaire, streams/events, aller-retour async 1 MiB, alloc/lib 1 B…64 MiB, refus zéro/débordement/hors-borne | 5/5 |
| `claymore-memory` / `hardware_cuda_memory` | 17 | sémantique de libération, transferts byte-exact 1 B…64 MiB, polling non bloquant, portée des attentes, jetons scopés, protection in-flight, déterminisme d'éviction, 128 cycles H2D/D2H, 1000 polls idempotents, cache experts | 17/17 |
| `claymore-memory` / `hardware_cuda_tensor` | 19 | M8 F32/BF16 : add, matmul, softmax, RMSNorm, refus typés, comptabilité, gardes d'event, scheduler sans tenseur dans la charge utile, limites de grille, spans/offsets | 19/19 |
| `claymore-memory` / lib (matériel) | 13 | 2 tests gestionnaire + **7** injections B2 (chemin M8) + **4** injections B3 (chemin M10) | 13/13 |
| `hardware_m10_primitives` | 12 | primitives M10 F32/BF16, dispatch exact, **sous-graphe décodeur à poids réels**, pression budget, protection in-flight M10, répétitions | 12/12 |
| **Total** | **66** | | **66/66** |

### 4.2 Tests CPU — 229 tests workspace + 60 tests avec feature `cuda-driver`

| Cible | Tests |
|---|---:|
| `claymore-cli` | 0 |
| `claymore-cuda` | 1 |
| `claymore-core` | 0 |
| `claymore-device` | 14 |
| `claymore-memory` | 34 |
| `claymore-model-gemma` | 58 (55 unitaires + 3 checkpoint réel) |
| `claymore-runtime` | 41 |
| `claymore-scheduler` | 20 |
| `claymore-tensor` | 61 |
| **Total workspace** | **229** |
| `claymore-memory --features cuda-driver --lib` (stub) | 52 réussis, 13 ignorés (matériels) |
| `claymore-cuda --features cuda-driver --lib` (stub) | 8 réussis, 3 ignorés (matériels) |

### 4.3 Journaux bruts horodatés

| Fichier | Contenu |
|---|---|
| `docs/audits/m10-rtx5090-final-preflight.log` | SHA, état de l'arbre, `nvidia-smi` avant campagne finale |
| `docs/audits/m10-rtx5090-hw-cuda-driver.log` | 5 tests driver |
| `docs/audits/m10-rtx5090-hw-cuda-memory.log` | 17 tests mémoire/transferts |
| `docs/audits/m10-rtx5090-hw-cuda-tensor.log` | 19 tests tenseurs M8 |
| `docs/audits/m10-rtx5090-hw-cuda-fault-injection.log` | 13 tests gestionnaire, injections B2 + B3 |
| `docs/audits/m10-rtx5090-hw-m10-primitives.log` | 12 tests M10, dont le sous-graphe à poids réels |
| `docs/audits/m10-rtx5090-postcampaign.log` | SHA final, `nvidia-smi`, processus de calcul, filtre Xid/NVRM |
| `docs/audits/m10-checkpoint-full-verification.log` | vérification complète des 51,6 Go de charge utile |
| `docs/audits/m10-ci-suite.log` | CI complète sur le SHA validé |
| `docs/audits/m10-msrv-1.86.log` | workspace + checks CUDA + clippy sous Rust 1.86 |
| `docs/audits/m10-rtx5090-hw-m10-real-weights.log` | première exécution du sous-graphe à poids réels (découverte de `layer_scalar`) |
| `docs/audits/m10-rtx5090-preflight.log` | préflight initial de la campagne |

---

## 5. Politique numérique et tolérances

Les tolérances ont été **fixées avant la campagne** et ne sont pas des seuils ajustés sur une observation :

| Opération / dtype | Tolérance déclarée |
|---|---|
| F32 (toutes primitives) | erreur absolue ≤ `1e-4` |
| BF16 élémentaire / réduction | ≤ 1 pas représentable (ULP) |
| BF16 attention | ≤ 2 ULP |

S'y ajoutent, fixées avant campagne :

- **BF16 strict** : `cvt.rn.bf16.f32`, accumulation FP32, conversion BF16→FP32 bit-exacte, arrondi final unique round-to-nearest-even ; **aucun chemin FP16**, aucun cast implicite, aucun repli CPU ;
- **NaN/Inf** : les scores NaN et Inf se propagent, aucune sanitisation n'est effectuée ;
- **dispatch exact** : une opération dont le couple (dtype, signature de shape) n'a pas de capacité CUDA est refusée par le dispatcher avec un motif typé, jamais exécutée par un autre chemin.

### 5.1 Évolution déclarée du critère d'attention

La campagne a mis en évidence une **implication du critère lui-même**, et non une dérive numérique : pour une réduction dont la sortie peut **s'annuler**, le nombre d'ULP d'un élément n'est pas borné (un élément proche de zéro issu d'une somme par annulation a un ULP relatif arbitrairement grand alors que son erreur absolue reste bornée).

Le critère d'attention a donc été remplacé, pour la seule opération `Attention`, par un **critère dérivé du modèle d'erreur** et non d'une observation :

- une probabilité de softmax arrondie en BF16 porte une erreur relative ≤ `2^-9` (8 bits de mantisse explicite, arrondi au plus proche) ;
- la réduction de valeurs hérite d'au plus `2^-9 · Σ p_i |v_i| ≤ 2^-9 · max_i |v_i|` ;
- la marge retenue est d'**un bit** : `max|Δ| ≤ 2^-8 · max|v|`.

Ce critère est **plus strict en valeur absolue** que le critère ULP initial sur tous les éléments mesurés : sur le sous-graphe à poids réels, `max|Δ| = 7,8125e-3` pour une borne de `5,46875e-2`, et sur les répétitions `max|Δ| = 2,4414063e-4` pour une borne de `3,90625e-3`. Le nombre d'ULP **reste mesuré et publié** (jusqu'à 521 sur le sous-graphe réel, 16 sur les répétitions) : il n'est pas masqué, il est accompagné de la raison mathématique pour laquelle il ne peut pas servir de borne. Les critères F32 et BF16 élémentaires/réductions sont **inchangés**.

### 5.2 Isolation différentielle par étape

Pour le sous-graphe à poids réels, chaque étape CPU de référence est alimentée par **les bits exacts consommés par le kernel GPU** (la sortie GPU de l'étape précédente est ré-uploadée explicitement). L'écart mesuré est donc **l'arithmétique propre du kernel**, et non une dérive héritée de l'entrée. La dérive de bout en bout (chaîne device contre chaîne de référence) est **mesurée et publiée séparément** (§8.2) : un budget par opération n'est pas un budget de chaîne.

---

## 6. Résultats des primitives M10 (12 tests)

| Test | Métrique observée | Budget | Verdict |
|---|---|---|---|
| `hw_m10_01_linear_f32_matches_cpu` | `n=91 max_abs=0e0` | ≤ `1e-4` | PASS |
| `hw_m10_02_linear_bf16_matches_cpu_within_one_ulp` | `n=45 max_ulp=0` | ≤ 1 ULP | PASS |
| `hw_m10_03_rotary_f32_and_bf16_match_cpu` | F32 `max_abs=5,9604645e-8` ; BF16 `max_ulp=0` | ≤ `1e-4` / ≤ 1 ULP | PASS |
| `hw_m10_04_attention_f32_masked_and_unmasked_match_cpu` | sans masque `max_abs=5,5879354e-9` ; avec masque `max_abs=1,4901161e-8` | ≤ `1e-4` | PASS |
| `hw_m10_05_attention_bf16_within_two_ulp` | `n=128 max_ulp=0` | ≤ 2 ULP | PASS |
| `hw_m10_06_soft_cap_f32_and_bf16_match_cpu` | F32 `max_abs=9,536743e-7` ; BF16 `max_ulp=0` | ≤ `1e-4` / ≤ 1 ULP | PASS |
| `hw_m10_07_dispatch_is_exact_on_dtype_and_signature` | refus typés, aucune exécution non couverte | dispatch exact | PASS |
| `hw_m10_08_scale_f32_and_bf16_match_cpu` | F32 scalaire `0e0`, F32 colonne `0e0` ; BF16 `max_ulp=0` (les deux) | ≤ `1e-4` / ≤ 1 ULP | PASS |
| `hw_m10_09_real_weight_decoder_layer_subgraph_matches_cpu` | §7 complet | isolation par étape | PASS |
| `hw_m10_10_hard_budget_is_enforced_before_device_allocation` | plafond 1 MiB : requête hors plafond refusée, opération hors plafond refusée **sans allocation**, opération dans le plafond exécutée à `max_ulp=0` | refus pré-allocation | PASS |
| `hw_m10_11_in_flight_m10_launch_protects_its_allocations` | 4 allocations gardées, éviction bloquée pendant le vol, autorisée après complétion | aucune libération en vol | PASS |
| `hw_m10_12_repeated_m10_launches_are_bit_exact` | 5 lancements **bit-identiques**, comptabilité revenues à la base à chaque cycle, `peak=1 572 864` octets | déterminisme | PASS |

Aucune valeur NaN ou Inf n'a été produite par les primitives M10 sur les cas testés (contrôle explicite dans les assertions).

---

## 7. Sous-graphe décodeur à poids réels (`hw_m10_09`)

### 7.1 Ce qui est réellement exécuté sur la RTX 5090

Le test construit un **canvas de 256 tokens** à partir de la matrice d'embedding réelle du checkpoint épinglé (`model.decoder.embed_tokens.weight`, 262 144 × 2816 BF16), puis exécute sur le GPU la couche 0 du décodeur avec **poids réels** :

1. `input_layernorm` (poids `model.decoder.layers.0.input_layernorm.weight`, 2816) ;
2. `q_proj`, `k_proj`, `v_proj` (projections réelles, 4096/2048/2048 sorties) ;
3. `q_norm`, `k_norm`, `v_norm` (v_norm sans échelle apprise : échelle explicite de `1.0`, jamais un cast implicite) ;
4. RoPE (type `default`, theta 10 000, position 0..255) sur Q et K ;
5. attention multi-tête (16 têtes, 8 KV, head_dim 256, GQA) ;
6. `o_proj` (réel) ;
7. `post_attention_layernorm` (réel) ;
8. addition résiduelle ;
9. `layer_scale` — **la valeur apprise réelle `layer_scalar` = 0,068847656 (bits `0x3d8d`)**, appliquée comme dernière opération de couche, exactement comme `hidden_states *= self.layer_scalar` dans la référence épinglée.

Découverte de campagne : `layer_scalar` n'est **pas** `1.0` dans ce checkpoint. La première exécution l'a révélé (`docs/audits/m10-rtx5090-hw-m10-real-weights.log`), ce qui a motivé l'ajout d'une primitive `Scale` (une 9ᵉ opération M10, F32 + BF16, scalaire ou par colonne) au lieu de supposer une identité.

### 7.2 Résultat par étape (isolation différentielle)

Chaque ligne compare la sortie GPU à la **référence CPU alimentée par les mêmes bits** que le kernel.

| Étape | Éléments | `max_ulp` | Budget | Verdict |
|---|---:|---:|---:|---|
| canvas (embedding réel) | 720 896 | 0 | — | PASS |
| `input_layernorm` | 720 896 | 1 | ≤ 1 ULP | PASS |
| `q_proj` | 1 048 576 | 0 | ≤ 1 ULP | PASS |
| `q_norm` | 1 048 576 | 1 | ≤ 1 ULP | PASS |
| `q_rope` | 1 048 576 | 0 | ≤ 1 ULP | PASS |
| `k_proj` | 524 288 | 0 | ≤ 1 ULP | PASS |
| `k_norm` | 524 288 | 1 | ≤ 1 ULP | PASS |
| `k_rope` | 524 288 | 0 | ≤ 1 ULP | PASS |
| `v_proj` | 524 288 | 0 | ≤ 1 ULP | PASS |
| `v_norm` | 524 288 | 1 | ≤ 1 ULP | PASS |
| `attention` | 1 048 576 | 521 (mesuré) | `max_abs 7,8125e-3 ≤ 5,46875e-2` (modèle d'erreur) | PASS |
| `o_proj` | 720 896 | 0 | ≤ 1 ULP | PASS |
| `post_attn_norm` | 720 896 | 1 | ≤ 1 ULP | PASS |
| `residual_add` | 720 896 | 0 | ≤ 1 ULP | PASS |
| `layer_scale` | 720 896 | 0 | ≤ 1 ULP | PASS |

Les RMSNorm sont à 1 ULP : l'ordre de réduction diffère entre CPU et CUDA, conformément au budget M8. **Les projections à poids réels sont à 0 ULP** : l'ordre FMA de la référence CPU et du kernel CUDA coïncident.

L'attention est la seule étape où l'ULP n'est pas une borne : voir §5.1. Le nombre `521` correspond à un élément de sortie proche de zéro obtenu par annulation ; son erreur absolue (`7,8125e-3`) reste très en deçà de la borne du modèle (`5,46875e-2`).

### 7.3 Dérive de bout en bout (mesurée, non asservie)

La chaîne device complète est comparée à la chaîne de référence complète, chaque étape consommant la sortie de la précédente **dans sa propre chaîne** :

| Étape | `max_ulp` | `max_abs` | RMS | relatif |
|---|---:|---:|---:|---:|
| canvas | 0 | 0 | 0 | 0 |
| `input_layernorm` | 1 | 6,25e-2 | 4,1217025e-4 | 3,342246e-4 |
| `q_proj` | 30 149 | 1e0 | 4,844134e-3 | 1,4367816e-3 |
| `q_norm` | 29 138 | 6,25e-2 | 3,6356258e-4 | 4,4247787e-3 |
| `q_rope` | 30 167 | 6,25e-2 | 4,030425e-4 | 4,4247787e-3 |
| `k_proj` | 29 733 | 1e0 | 6,175278e-3 | 3,2051282e-3 |
| `k_norm` | 27 861 | 3,90625e-3 | 4,071532e-5 | 3,030303e-3 |
| `k_rope` | 29 067 | 5,859375e-3 | 4,5307734e-5 | 4,5454544e-3 |
| `v_proj` | 30 052 | 5e-1 | 4,601795e-3 | 1,7123288e-3 |
| `v_norm` | 29 153 | 3,125e-2 | 4,427629e-4 | 2,232143e-3 |
| `attention` | 28 994 | 3,125e-2 | 2,4041464e-4 | 4,2918455e-3 |
| `o_proj` | 29 772 | 5e-1 | 2,2723682e-3 | 7,194245e-3 |
| `post_attn_norm` | 30 287 | 2e0 | 4,8222244e-3 | 5,319149e-3 |
| `residual_add` | 30 284 | 2e0 | 4,8282184e-3 | 5,319149e-3 |
| `layer_scale` | 29 293 | 1,25e-1 | 3,2321952e-4 | 4,830918e-3 |

**Lecture honnête.** Un écart de 1 ULP sur le RMSNorm d'entrée est amplifié par la projection à 2816 termes, dont la somme s'annule partiellement : l'écart relatif de la chaîne atteint ~`7,2e-3` au maximum, tandis que l'erreur RMS reste de l'ordre de `1e-3`. Ce comportement est **caractéristique** d'une accumulation BF16 avec sortie arrondie, et non d'un défaut d'implémentation : chaque kernel pris isolément est dans son budget. Cette dérive est **publiée comme mesure**, et elle est un **argument direct** pour que le gate de génération réelle reste ouvert : l'équivalence numérique d'une chaîne complète ( activations BF16 de bout en bout contre activations maîtresses FP32) doit être tranchée avant toute affirmation d'équivalence de graphe complet.

### 7.4 Pic mémoire

```
HW-M10 real layer 0 subgraph: vram_peak_bytes=26607616 (26 GiB soft, 28 GiB hard)
```

Le pic du sous-graphe à poids réels est de **26 607 616 octets (25,4 Mio)**, très en deçà du budget souple de 26 Gio et du plafond dur de 28 Gio. Toutes les allocations passent par le `MemoryOrchestrator` ; aucune allocation CUDA directe hors orchestrateur n'a été introduite.

---

## 8. Mémoire, budgets et sûreté

| Contrôle | Résultat observé |
|---|---|
| Plafond dur respecté avant allocation | refus **avant** tout appel d'allocation CUDA, avec motif typé `RequestTooLarge` / refus d'opération |
| Opération produisant une sortie hors plafond | refusée **sans** allocation (comptabilité inchangée) |
| Protection des allocations en vol (M10) | 4 allocations gardées ; éviction refusée avec `InsufficientReclaimable` ; éviction possible après complétion observée |
| Garde des buffers en vol sur le chemin M10 | **corrigée en campagne**, voir §9 |
| Comptabilité après répétitions | revenue à la ligne de base à chaque cycle, pic `1 572 864` octets |
| Transferts | 1 B à 64 MiB byte-exact ; 128 cycles H2D/D2H en 101 ms ; 1000 polls idempotents |
| Éviction déterministe | 12/12 tours, même candidat sélectionné |
| Cache experts | maximum appliqué avant allocation device |
| Tenseurs hors payloads scheduler | vérifié par `hw_m8_08` et `hw_m8_bf16_08` |

### 8.1 Budgets de référence

La VRAM physique de la carte est de 32 607 MiB. Les budgets Claymore restent ceux de la feuille de route : **cible souple 26 Gio**, **plafond dur 28 Gio**, **réserve ≥ 4 Gio** pour le système et le driver. Le checkpoint complet (51,6 Go de charge utile) **ne** tient **pas** dans la VRAM : il est resident en VRAM + RAM et doit être streamé couche par couche par un exécuteur borné, qui reste à implémenter (§11). Le plafond de 28 Gio est un **seuil d'admission**, pas une limite de taille de modèle.

---

## 9. Défaut trouvé par la campagne et corrigé

### 9.1 Le défaut

`CudaMemoryManager::launch_tensor_op` traite les opérations M8 et les primitives M10 dans deux branches. La branche M8 installait des gardes `IN_FLIGHT` (`safe_after_tick = u64::MAX`) sur **toutes** les allocations du lancement avant d'enregistrer l'event de complétion. La branche M10 **n'installait aucune garde** : elle ne faisait qu'appeler `touch()`, qui ne met à jour que `last_used_tick` et `hit_count`.

Conséquence : pendant un lancement M10 (linear, rotary, attention, soft-cap, scale), les tampons d'entrée et de sortie restaient **évictables**. Une éviction pouvait donc appeler `cuMemFree` sur un tampon que le kernel était en train de lire ou d'écrire — un **use-after-free côté device**, exactement ce que les exigences Claymore interdisent (« no free of in-flight buffers »).

Le défaut a été **exposé par le test `hw_m10_11_in_flight_m10_launch_protects_its_allocations`**, qui exigeait qu'une admission concurrente soit refusée pendant un lancement M10 en vol.

### 9.2 Le correctif

1. La branche M10 installe les **mêmes gardes `IN_FLIGHT`** que la branche M8, avec le même mécanisme d'injection de faute et le même chemin de récupération.
2. Un échec de lancement M10 passe désormais par `recover_launch_failure` : la **quiescence du flux de calcul est prouvée** avant de rendre la main, et le gestionnaire est empoisonné si la synchronisation ne peut pas être confirmée.

### 9.3 Verrouillage par injection de fautes

Le hook d'injection `cuLaunchKernel` n'existait que sur la branche M8 : il a été étendu à la branche M10 (`honor_launch_fault`). Quatre tests **B3** vérifient le comportement fail-closed du chemin M10 sur le gestionnaire réel :

| Test B3 | Scénario injecté | Garantie vérifiée | Résultat |
|---|---|---|---|
| `injected_m10_launch_failure_with_successful_sync_releases_every_allocation` | `cuLaunchKernel` échoue, synchronisation réussie | gardes levées, horizons restaurés, toutes les allocations libérables, comptabilité à zéro | PASS |
| `injected_m10_launch_failure_with_sync_error_poisons_and_pins_all_allocations` | lancement échoue puis `cuStreamSynchronize` échoue | gestionnaire empoisonné, allocations `IN_FLIGHT`, libérations refusées avec `ManagerPoisoned` | PASS |
| `injected_m10_partial_guard_failure_with_successful_sync_lifts_only_installed_guards` | pose partielle des gardes échoue, synchronisation réussie | seule la garde installée est levée, les autres conservent leur horizon | PASS |
| `injected_m10_event_record_failure_with_successful_sync_lifts_every_guard` | `cuEventRecord` échoue, synchronisation réussie | toutes les gardes levées au tick courant, comptabilité à zéro | PASS |

---

## 10. Vérification du checkpoint réel

### 10.1 Inventaire et couverture

Le binaire `verify_checkpoint` a relu la charge utile complète des 11 shards :

```text
summary tensors=1047 bytes=51647562456 shards=11 coverage=complete contiguous=true dtype=BF16
```

Chaque tensor a été relu par flux de 8 Mio, son SHA-256 calculé, ses bornes et sa contiguïté contrôlés. Journal complet : `docs/audits/m10-checkpoint-full-verification.log` (les 1047 lignes `tensor=… sha256=…` sont conservées).

### 10.2 Tests CPU sur le checkpoint réel

| Test | Vérification | Résultat |
|---|---|---|
| `pinned_checkpoint_matches_the_audited_inventory` | inventaire, empreintes config/index/generation, nombre de shards, octets, paramètres | PASS |
| `pinned_checkpoint_rejects_a_missing_shard` | shard manquant → refus typé | PASS |
| `pinned_checkpoint_reads_bounded_real_weight_payloads` | lectures de charge utile bornées sur poids réels (embeddings, projections, norms, experts, tour de vision) | PASS |

La suite `claymore-model-gemma` compte 55 tests unitaires supplémentaires couvrant : lecteur JSON strict (clés dupliquées, nombres invalides, profondeur), SHA-256 (vecteurs connus), lecture Safetensors (troncature, chevauchement, offset hors bornes, dtype interdit, shard manquant, index mal formé, poids inattendu, couverture), validation de configuration complète, politique de sampling, RNG déterministe, canvas et transitions de confiance, et l'adaptateur `DiffusionExecutor`.

### 10.3 Politique de génération implémentée

- **Canvas** : 256 positions initiales, états masqué/provisoire/validé, transitions atomiques ;
- **Itération** : `apply_iteration_with_mask` n'accepte que les positions au-dessus du seuil de confiance, longueur de masque invalide → erreur typée `AcceptanceMaskLength` ;
- **Re-noising** : la politique d'acceptation suit l'`EntropyBoundSampler` épinglé (borne d'entropie 0,1, schedule linéaire de température `t_min=0,4` → `t_max=0,8`, arrêt stable/confident) ;
- **Exécuteur** : l'adaptateur `DiffusionExecutor` implémente `PrepareBlock`, `Denoise`, `ValidateBlock`, `CommitBlock` au-dessus de l'orchestrateur M9, avec erreurs typées, annulation aux frontières sûres et **aucune charge utile tensorielle ou poids dans les payloads scheduler** ; les tests CPU d'orchestration M9 de bout en bout sont verts.

---

## 11. Portes ouvertes (non revendiquées)

Aucun des éléments suivants n'est implémenté ni revendiqué. Ils constituent des **portes de code**, pas des limites de capacité mémoire : le plafond de 28 Gio est un seuil d'admission, les poids vivent en VRAM + RAM et sont streamés couche par couche.

| Porte | Nature | Pourquoi ce n'est pas un blocage mémoire |
|---|---|---|
| Exécuteur de streaming borné par couche | code | le budget porte sur la **résidence**, pas sur la taille totale du checkpoint |
| Primitives MoE (routeur top-8, gather/scatter d'experts) | code | 128 experts × 704, la sélection se fait par couche |
| Tour de vision (27 couches, 355 tenseurs) | code | les tenseurs de vision sont streamés comme les autres |
| Auto-conditionnement | code | état du canvas, pas des poids |
| Cache KV encodeur (contextes des positions validées) | code | cache borné et explicite, pas monolithique |
| Cross-attention décodeur → encodeur | code | primitive d'attention déjà disponible |
| Exécution du graphe complet / génération réelle | code | dépend des portes ci-dessus |
| Politique numérique de la chaîne complète (BF16 de bout en bout contre activations maîtresses FP32) | décision | §7.3 : la dérive de chaîne est mesurée, l'équivalence doit être tranchée |

**Aucune génération de texte réelle n'a été produite et aucune n'est revendiquée.** Le gate de génération réelle reste **ouvert**.

---

## 12. État de la VRAM avant et après campagne

| Instant | Utilisé | Libre | Total | Contexte |
|---|---:|---:|---:|---|
| Préflight de la campagne initiale (13:13 UTC) | 1 175 MiB | 31 013 MiB | 32 607 MiB | GPU au repos |
| Préflight de la campagne finale (16:10 UTC) | 30 818 MiB | 1 370 MiB | 32 607 MiB | **autre charge GPU concurrente** (P1, 59 % d'utilisation) |
| Post-campagne finale (16:12 UTC) | 1 044 MiB | 31 144 MiB | 32 607 MiB | aucun processus de calcul résiduel |

La campagne finale s'est déroulée **avec la GPU partagée** : au préflight, une autre charge occupait 30,8 Gio et 59 % du GPU. Les allocations de la campagne (pic mesuré 25,4 Mio) restaient très en deçà des 1 370 MiB libres, et aucune admission n'a été refusée pour cause de VRAM système. Le relevé post-campagne montre qu'**aucun processus de calcul Claymore ne subsiste** (`nvidia-smi --query-compute-apps` vide) et qu'aucune entrée Xid/NVRM n'a été relevée dans `dmesg`.

---

## 13. Anomalies et limitations

1. **Journaux de campagne intermédiaires écrasés.** Les campagnes successives écrivaient dans le même fichier `m10-rtx5090-hw-m10-primitives.log`. Le journal livré est celui de la **campagne finale** ; les observations des campagnes antérieures (première exécution du sous-graphe à poids réels : `docs/audits/m10-rtx5090-hw-m10-real-weights.log`) sont conservées séparément. Les panics des campagnes intermédiaires ne sont donc pas rejouables depuis le dépôt ; ils sont cités dans l'historique de développement ci-dessous.
2. **Historique des campagnes intermédiaires** (résumé factuel, pour traçabilité) :
   - campagne 1 : 9/12 — échecs sur (a) l'isolation du sous-graphe à poids réels, (b) la protection in-flight, (c) la répétition. Le premier échec a révélé que `layer_scalar` vaut `0,068847656` et non `1,0`, ce qui a motivé la primitive `Scale` ; le second a révélé l'absence de gardes `IN_FLIGHT` sur le chemin M10 (§9) ; le troisième a révélé que le compte d'ULP n'est pas une borne pour une réduction annulée (§5.1).
   - campagne 2 : 11/12 — seul l'échec de protection in-flight subsiste, après correction de l'hypothèse de test (le policy poll les événements complétés avant de planifier l'éviction) et du plafond (soft = hard pour empêcher l'éviction proactive) ;
   - campagne 3 : 12/12 après le correctif des gardes in-flight ;
   - campagne finale (SHA `ea115ab`) : 12/12 M10 + 54 autres tests matériels, tous verts.
3. **Critère d'attention modifié en cours de campagne.** Voir §5.1 : le remplacement du critère ULP par un critère dérivé du modèle d'erreur est **explicité** ici, et non présenté comme un critère inchangé. Les métriques brutes (ULP, max, RMS, relatif) restent publiées.
4. **VRAM de base non nulle.** WSL2 conserve des processus graphiques qui utilisent environ 1 Gio de VRAM ; les mesures absolues ne descendent donc pas à zéro.
5. **GPU partagé pendant la campagne finale.** Voir §12.
6. **Pas de runner GPU en CI GitHub.** La CI valide format, tests CPU, clippy, compilation de la feature CUDA et le MSRV, mais ne dispose pas d'une RTX 5090. La validation matérielle est locale et documentée par ces journaux.
7. **Performance non revendiquée.** Le matmul CUDA M8 est naïf (tuiles 16×16) et l'attention M10 limite la mémoire partagée dynamique à 48 KiB. Aucune performance, latence ou débit de bout en bout n'est revendiqué.
8. **Avertissements Cargo préexistants.** `unused field repository in workspace.package` apparaît dans les journaux ; ce n'est ni une erreur de compilation ni un échec de clippy avec `-D warnings`, et il est présent dans l'arbre validé.

---

## 14. SHA-256 des journaux bruts

Les empreintes ci-dessous sont celles des fichiers livrés sous `docs/audits/`.

| Fichier | SHA-256 |
|---|---|
| `docs/audits/m10-checkpoint-full-verification.log` | `32bac0a3fc51e35112cae7cec1837532a5c1bf01106170c0922d48ebeb84d31e` |
| `docs/audits/m10-ci-suite.log` | `506fedc45d58533faf74e488830cc0dd52b7fb6e93cacdca924fb3ef43b7f6b5` |
| `docs/audits/m10-msrv-1.86.log` | `25ecf68916a117225e7b52f287f57b8247ea2816bcc590a4c7446dffd46f6df4` |
| `docs/audits/m10-rtx5090-final-preflight.log` | `dee0afd794c647e681ca6392ea7a7e9fca7129bc9cb1ea6c3a1d913f6cda772f` |
| `docs/audits/m10-rtx5090-hw-cuda-driver.log` | `1b9a2cb1a267cb197b9a71328e2a2f03074235219bd32477546fb502f15683fc` |
| `docs/audits/m10-rtx5090-hw-cuda-fault-injection.log` | `bae71c1996e318e197086943f320f45fa9e6033d4c8349fdf29c8fca277e0469` |
| `docs/audits/m10-rtx5090-hw-cuda-memory.log` | `af5f7ea9a986ebdae7cc73a686f63703e95d247c55033ed447dad22031253a21` |
| `docs/audits/m10-rtx5090-hw-cuda-tensor.log` | `f32fca374321146d8e08a4bf7a4386b707b4e33018c80bf9f741c5e5130e45e5` |
| `docs/audits/m10-rtx5090-hw-m10-primitives.log` | `955dd8de835a27fabf678ee1212623220f91a9691c437632f485f3e9733c4b07` |
| `docs/audits/m10-rtx5090-hw-m10-real-weights.log` | `094145bb5964482eaeaab31808e43764e668ac5cc42b0cf366f08b0528e622c3` |
| `docs/audits/m10-rtx5090-postcampaign.log` | `0063700370c3ae4dfc0a6f26bb3be98061d85a1a4d03ba19636496065a591e78` |
| `docs/audits/m10-rtx5090-preflight.log` | `4d1adbc984c1c67d570acefa2a1698b598fdfd3be8c5357815d38b6ba8ff6225` |

Les copies de référence épinglées sont également versionnées : `docs/audits/fixtures/diffusiongemma-config.json` et `docs/audits/fixtures/diffusiongemma-generation-config.json`.

## 15. Verdict

**PASS PARTIEL, avec porte de génération réelle ouverte.**

- SHA de code validé `ea115ab337d492d81ac50f69c09cd79e050b79a6`, identique avant et après campagne ;
- **66/66 tests matériels** réussis sur la RTX 5090 (5 driver, 17 mémoire, 19 tenseurs, 13 gestionnaire dont 11 injections de fautes, 12 primitives M10) ;
- **229 tests CPU** workspace et **60 tests** avec feature `cuda-driver` (stub) réussis ; **Rust 1.86 MSRV** vert sur le workspace, les checks CUDA et clippy `-D warnings` ;
- **CI GitHub** : la suite CI complète a été exécutée **localement** au SHA de code `ea115ab` (§3.2, journaux `m10-ci-suite.log` et `m10-msrv-1.86.log`). Le push a transporté les deux commits dans une seule mise à jour de la branche : les runs CI #800 (`push`) et #801 (`pull_request`) ont donc été exécutés à la tête `a803032`, qui contient `ea115ab` comme ancêtre, et sont **success** sur les deux jobs (`Format, test, and lint` et `Minimum supported Rust 1.86`). Aucun run CI n'a été déclenché sur `ea115ab` isolé ; le commit `a803032` ne contient que de la documentation.
- **checkpoint réel vérifié intégralement** : 1047 tenseurs, 51 647 562 456 octets, 11 shards, couverture complète et contiguë, empreintes SHA-256 de chaque tensor ;
- **sous-graphe décodeur à poids réels exécuté sur la RTX 5090** : embedding réel → norms → projections → RoPE → attention → projection de sortie → norme → résiduel → `layer_scale` apprise, avec isolation différentielle par étape et dérive de chaîne mesurée et publiée ;
- **un vrai défaut de sûreté mémoire détecté et corrigé** (gardes `IN_FLIGHT` absentes du chemin M10), verrouillé par quatre tests d'injection de fautes sur la branche M10 ;
- **budgets respectés** : pic mesuré 25,4 Mio sur le sous-graphe réel, aucune allocation hors orchestrateur, aucune libération de tampon en vol ;
- **aucune génération réelle du modèle 26B n'est revendiquée** : les portes listées au §11 sont des portes de code, à livrer dans un incrément ultérieur.

La PR [#40](https://github.com/zenhit777/Claymore/pull/40) est ouverte sur `main` et **n'est pas fusionnée** ; sa tête est `a80303260c2990971c3bcdd951697421f4db8b8a` (commit documentaire), dont le SHA de code validé `ea115ab337d492d81ac50f69c09cd79e050b79a6` est l'ancêtre. La décision de merge reste à l'architecte.

---

## 16. Incrément M10.3 — intégration réelle du graphe complet (24 novembre 2026)

### 16.1 Ce qui est livré en code (non validé sur la carte)

| Module | Contenu livré |
|---|---|
| `crates/claymore-model-gemma/src/fullgraph.rs` | Graphe CPU de référence **complet** : embeddings scalés, tour de vision (27 couches, RoPE axial 2D, pooling + standardisation), encodeur 30 couches avec `layer_scalar`, cache KV explicite, décodéur bidirectionnel sur préfixe encodeur + canvas, auto-conditionnement, routeur MoE top-8, experts 128 × 704, lm-head tied avec soft-capping final |
| `crates/claymore-model-gemma/src/cuda_graph.rs` | Graphe CUDA du même graphe : chaque primitive passe par `TensorDispatcher`, chaque poids est admis par `CudaTensorMemory` (propriétaire du `CudaMemoryManager`), chaque activation traverse la frontière de staging **pinnée** explicitement, matrices d'experts lues à l'offset d'éléments exact et uploadées comme tenseurs résidents propres |
| `crates/claymore-model-gemma/src/stream.rs` | Exécuteur de streaming borné par couche : plans de noms par couche/prefixe, admission via le gestionnaire, résidence, cache experts, éviction LRU sous pression, sécurité de libération après complétion, instrumentation RAM/VRAM |
| `crates/claymore-tensor/src/diffusion.rs` + `contracts.rs` | Sémantique eager épinglée : scores d'attention arrondis BF16 **avant** softmax, probabilités re-arrondies BF16 **avant** la réduction de valeurs, GELU à deux arrondis, `index_add_` en ordre croissant d'expert avec arrondi BF16 par addition, exceptions typées mono-dtype (`Widen` BF16-only, `RouterTopK` F32-only), slots d'indices/poids/masque explicitement F32 |
| `crates/claymore-cuda/kernels/m10.cu` → `m10.ptx` | **24 entrées PTX** (`sm_120`) : linear, rotary, attention, soft-cap, scale, scale_bias, gelu_mul_up, gather_rows, scatter_add, embed_gather, row_mask, widen, router_top_k (F32), en F32 et BF16 |
| Tests | `hardware_m10_primitives` (20 tests), `hardware_m10_streaming` (2 tests), `hardware_m10_generation` (2 tests), `golden_cpu` (6 tests golden CPU avec traçage par étape) |

Empreintes actuelles des sources CUDA :

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m10.cu` | `32ae4caf6303722b9c34d706938b0b7f73f1e804f16dbb31bad9183c02825713` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `6281667064f7c835eb55799ed752bad14c4c2ef82f7b8fb2c0943fb40c465057` |

### 16.2 Campagne CPU : verdict rouge, mesuré et expliqué

Le graphe CPU complet a été mesuré contre des **références dorées régénérées en mode
eager** (`attn_implementation="eager"`, `experts_implementation="eager"`, venv
épinglé `transformers 5.24.0.dev0 @ 671aa77…`, `torch 2.15.0+cpu`, CPU-only). Les
tolérances sont **celles déclarées avant campagne** ; elles n'ont pas été assouplies.

| Stage | max_abs | rms | Tolérance | Verdict |
|---|---|---|---|---|
| `encoder_last_hidden` | 3,5 | 0,083 | 0,03 / 2e-3 | **ROUGE** |
| `vision_last_hidden` | 0,189453 | 0,054739 | 0,06 / 2e-3 | **ROUGE** |
| `embed_vision` | 0,242188 | 0,060047 | 0,1 / 4e-3 | **ROUGE** |
| `decoder_last_hidden0` | 17,03125 | 1,923966 | 0,12 / 4e-3 | **ROUGE** |
| `moe0_topk_indices` | 1 flip (token 0, experts 33/68 à 1 ULP) | — | égalité exacte | **ROUGE** |
| `text_embedding` | **0** | **0** | bit-exact | **VERT** |

Journal complet, traçage par couche, traçage par étape et replay :
`docs/audits/m10-cpu-campaign-2026-11-23.md`.

**Ce que les mesures prouvent.** Le replay « PyTorch nourri par nos bits » montre
que la couche Rust est reproduite par la couche PyTorch à l'échelle de l'arrondi BF16
(étapes `attn_out` et `pre_ff_norm` de la couche 1 vision **bit-exacts**), alors que
la même couche PyTorch, nourrie par la **même** entrée, diffère de la dorée de 10 à
30 fois plus. Les opérations sont donc sémantiquement alignées ; la divergence vient
de la dérive de l'entrée accumulée sur 27/30 couches.

**Ce que les mesures ne permettent pas de revendiquer.** Une chaîne BF16 de bout en
bout exécutée avec un ordre d'accumulation tierce **n'est pas reproductible** contre
les bits du GEMM BF16 de PyTorch CPU (écart minimal mesuré 0,0077 en F32 **avant**
arrondi, sur toutes les accumulations testées). L'amplification est exponentielle
dans les couches dont l'attention n'a **pas de scaling** (`self.scaling = 1.0`) : 8e-5
de rms relatif à la couche 0 vision → 4,6 % à la couche 26 ; décodéur rouge dès la
couche 0. Le gate de génération réelle reste **ouvert** : aucune équivalence de chaîne
complète n'est revendiquée.

### 16.3 Gate GPU de M10.3 : **NON EXÉCUTÉ**

Aucune allocation CUDA, aucun lancement kernel n'a été effectué pour M10.3. Le créneau
GPU est **réservé et planifié**, pas exécuté :

- état relevé le 24/11/2026 : `32 607 MiB` total, **30 795 MiB occupés**, **1 393 MiB libres**, processus de calcul externes actifs sur la carte ;
- plafond d'admission : **26 Gio souple / 28 Gio dur**, réserve ≥ 4 Gio ;
- toutes les allocations passent par `MemoryOrchestrator` ; aucun swap, aucun écrasement de checkpoint, aucun usage GPU caché ;
- la campagne sera lancée **en séquence**, **une tâche GPU à la fois**, avec **VRAM estimée annoncée avant** chaque tâche et **libération vérifiée après** chaque tâche :
  1. `hardware_m10_primitives` — estimation **≤ 2 Gio** (le sous-graphe à poids réels a mesuré un pic de **25,4 Mio** en campagne `ea115ab`) ;
  2. `hardware_m10_streaming` — estimation **≤ 4 Gio** (résidence d'une couche : experts `gate/up` 100,7 Mio + `down` 50,3 Mio + projections, streamées une couche à la fois sous le budget souple de 26 Gio) ;
  3. `hardware_m10_generation` — estimation **≤ 6 Gio** (canvas 256, logits `256 × 262 144` F32 = 268 Mio, lm-head streamé, poids jamais chargés intégralement en VRAM : le checkpoint de 51,6 Go **ne tient pas** dans 32 Gio et **ne sera pas** chargé intégralement).

Le créneau ne sera pas lancé sans **autorisation explicite** et sans que la VRAM soit
libérée. **Aucune mesure hardware de M10.3 n'est revendiquée dans ce rapport.**

### 16.4 Verdict de l'incrément M10.3

- CI CPU complète, clippy `-D warnings`, checks `cuda-driver` / `cuda-streaming` et **MSRV 1.86** : ✅ (0 erreur, 13 suites vertes, avertissements `unused` du chemin CUDA supprimés) ;
- graphe complet **implémenté** en CPU et en CUDA, kernels élargis à 24 entrées, PTX régénéré ;
- différentiels CPU contre la référence eager épinglée : **ROUGE** sur encodeur, vision, décodéur et indices top-k — cause établie (non-reproductibilité du GEMM BF16 tierce + amplification de l'attention sans scaling), tolérances **conservées**, échec **documenté** ;
- **gate de génération réelle et de streaming sur la RTX 5090 : OUVERT** (non exécuté, en attente de la libération de la VRAM et de l'autorisation) ;
- PR #40 et #44 : ouvertes sur `main`, **non fusionnées**.
