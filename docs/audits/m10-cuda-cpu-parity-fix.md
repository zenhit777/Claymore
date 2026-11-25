# M10.4 — Correction de la parité numérique CUDA ↔ CPU de la chaîne DiffusionGemma

> **Objet :** compte rendu de la campagne du 24–25 novembre 2026 qui a localisé,
> corrigé et mesuré la divergence entre la chaîne CUDA et la chaîne CPU de
> référence du modèle DiffusionGemma 26B-A4B-it, sur NVIDIA RTX 5090 (sm_120).
>
> **Point de départ :** `9775cda` (« Merge main (PR #43) into M10 and reconcile
> documentation »), branche `fix/m10-numerics-divergence`.
>
> **Verdict :** la chaîne CUDA est désormais **bit-exacte** avec la chaîne CPU de
> référence sur l'encodeur complet (30 couches), la couche 0 étape par étape, et
> sur les 21 primitives matérielles. Le rouge résiduel du test de chaîne porte
> **entièrement** sur la référence CPU elle-même, dont l'écart à la dorée eager
> est inchangé et documenté (§16.2 de l'audit M10).

## 1. Résumé exécutif

| Grandeur | Avant (`9775cda`) | Après (correction) | Verdict |
|---|---|---|---|
| `encoder_last_hidden` CUDA ↔ CPU, `max_abs` | 5,835938 | **0,000000** | VERT |
| `encoder_last_hidden` CUDA ↔ CPU, `rms` | 0,711149 | **0,000000** | VERT |
| `encoder_last_hidden` CUDA ↔ CPU, étape par étape (couche 0, 6 étapes) | 0,031250 → 1,000000 | **0,000000** sur les 6 étapes | VERT |
| Indices du routeur (couche 0, 56 experts) | 56/56 identiques | **56/56 identiques** | VERT (inchangé) |
| Primitives matérielles M10 | 21/21 verts | **21/21 verts, tous à 0 ULP** | VERT |
| `encoder_last_hidden` CPU ↔ dorée eager | 5,539062 / 0,763049 | 5,828125 / 0,754065 | ROUGE (inchangé, voir §6) |
| `encoder_last_hidden` CUDA ↔ dorée eager | 4,875000 / 0,752871 | 5,828125 / 0,754065 | ROUGE, **égal au CPU** |
| Pic VRAM de la génération | 24 831 Mio | **24 831 Mio** | Budget respecté |
| Positions acceptées | 256 / 256 | **256 / 256** | VERT |
| Poids uploadés (H2D) | 930 494 Mio | **45 341 Mio** | §5.3 |
| Opérations graphées | 605 669 | **48 285** | §5.3 |

Le fait mesurant de cette campagne n'est pas une amélioration de la précision :
c'est que **la chaîne CUDA ne dérive plus du tout** de la chaîne CPU de
référence. L'écart de chaîne mesuré à HEAD (`5,835938`) était de même ordre de
grandeur que l'écart CPU ↔ dorée (`5,539062`), ce qui avait conduit l'audit M10
à conclure à une limite de reproductibilité BF16 intrinsèque. Cette conclusion
était **erronée** : les deux implémentations divergeaient parce que le CUDA
n'appliquait pas les mêmes opérations, dans le même ordre, avec les mêmes
arrondis que le CPU.

## 2. Matériel, environnement et autorisations

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 607 Mio, compute capability 12.0 (`sm_120`) |
| Pilote | KMD `616.64`, UMD `13.4` (WSL2) |
| Chaîne de compilation PTX | `nvcc` **release 13.0, V13.0.48**, `/usr/local/cuda-13.0/bin/nvcc` |
| Rust | `rustc 1.100.0`, MSRV du dépôt `1.86` |
| Checkpoint épinglé | `/tmp/opencode/m10/checkpoint` |
| Dorées eager | `/tmp/opencode/m10/golden_eager` |

Le GPU est partagé avec les charges de travail de l'utilisateur. **Chaque
lancement GPU a été précédé d'une demande d'autorisation explicite** et d'une
vérification de `nvidia-smi` ; aucun processus de calcul tiers n'était actif au
moment des lancements (31 023 et 31 037 Mio libres mesurés juste avant les runs
finaux). Après la campagne, la carte est revenue à **1 239 Mio occupés, aucun
processus de calcul listé**.

## 3. Reproduction du défaut

Sur le SHA de départ, la campagne GPU M10.3 publiée (§16.3.3.2 de
`docs/audits/m10-rtx5090-hardware-validation.md`) mesurait :

| Grandeur | CUDA ↔ CPU | CUDA ↔ dorée | CPU ↔ dorée |
|---|---|---|---|
| `encoder_last_hidden` `max_abs` | **5,835938** | 4,875000 | 5,539062 |
| `encoder_last_hidden` `rms` | **0,711149** | 0,752871 | 0,763049 |

Cette campagne a **reproduit ces valeurs à l'identique** (run 2, §4), ce qui
établit que le défaut est stable et non un artefact d'une exécution.

## 4. Protocole d'attribution

Le doute initial portait sur la répartition de la responsabilité entre le code
hôte Rust et les kernels CUDA. Un **bisect à 3 configurations** a été exécuté sur
la carte, chaque configuration mesurant le même différentiel sur le même
checkpoint et les mêmes entrées :

| Run | Code hôte | PTX chargés | `encoder_last_hidden` CUDA ↔ CPU |
|---|---|---|---|
| 1 | arbre de travail corrigé | PTX de `9775cda` (anciens kernels) | `max_abs=0.000000`, **`rms=NaN`** |
| 2 | `9775cda` (worktree détaché) | PTX de `9775cda` | `max_abs=5.835938`, `rms=0.711149` |
| 3 | arbre de travail corrigé | PTX régénérés du 22:02 | `max_abs=0.000000`, `rms=0.000000` |
| 4 | arbre de travail corrigé | PTX **reproductibles** (§5.2) | `max_abs=0.000000`, `rms=0.000000` |

Lecture de ces quatre runs :

- **run 2** confirme la baseline publiée, au bit près : le défaut est stable et
  reproductible ;
- **run 1** montre que le code hôte corrigé et les anciens kernels ne sont pas
  compatibles entre eux (sortie `NaN`) : le kernel de diffusion MoE et son
  contrat hôte forment une **paire indivisible**, ce qui était attendu (l'ancien
  kernel exigeait une destination pré-zéroée, que le code hôte corrigé n'émet
  plus) ;
- **runs 3 et 4** établissent que la correction du côté kernels **et** du côté
  code hôte supprime entièrement la dérive ;
- les runs 3 et 4 étant **identiques** (mêmes métriques au bit près : pic
  24 831 Mio, 45 341 Mio uploadés, 2 022 Mio téléchargés, 48 285 opérations), le
  résultat ne dépend pas du mode de compilation du PTX : le correctif est
  robuste à la chaîne d'outils.

## 5. Le correctif

### 5.1 Les cinq causes racines

La dérive n'avait pas une cause unique mais cinq, toutes dans le même sens :
le CUDA n'appliquait pas l'opération du CPU de référence.

| # | Cause | Correction |
|---|---|---|
| 1 | **Exponentielle approchée.** `expf`/`tanhf`/`expm1f` de nvcc se compilent en `ex2.approx.ftz.f32` et `rcp.approx.ftz.f32`, qui ne sont pas correctement arrondis ; le CPU de référence atteint glibc via `f32::exp`/`f32::tanh`. L'écart d'1 ULP par élément est amplifié par la chaîne BF16. | Portages **bit-à-bit** des algorithmes glibc 2.39 (`e_expf.c`, `s_expm1f.c`, `s_tanhf.c`) dans un en-tête partagé `crates/claymore-cuda/kernels/claymore_math.cuh`, avec les tables et constantes transcrites en motifs de bits. Les trois fichiers de kernels (`m10.cu`, `m8_f32.cu`, `m8_bf16.cu`) partagent désormais cette définition unique. |
| 2 | **Réductions en arbre.** `softmax` et `rms_norm` réduisaient en arbre sur plusieurs threads avec un partage mémoire, alors que le CPU réduit une ligne **séquentiellement en ordre d'index**. L'écart d'1 ULP par ligne est amplifié par les projections à 2 816 termes. | Les deux réductions sont désormais effectuées **sur un seul thread, en ordre d'index**, dans l'ordre exact de la référence CPU. Le partage mémoire est conservé pour respecter le contrat de lancement. |
| 3 | **Réciproque multipliée.** Le CUDA calculait `exp(x - max) * (1 / sum)` ; le CPU fait `exp(x - max) / sum`. | Le kernel divise par la somme, comme le CPU. Côté CPU (`diffusion.rs::softmax_row`), la multiplication par la réciproque a été remplacée par la division, conformément à la formule déjà documentée en `docs/audits/m8-cpu-cuda-numerics.md` §3. |
| 4 | **`rms_norm` : formule de l'inverse.** Le CPU employait `powf(-0.5)` ; le CUDA `sqrt` puis division. Les deuxdiffèrent d'1 ULP. | Le CPU utilise désormais `1.0 / sqrt(...)` (forme documentée en `m8-cpu-cuda-numerics.md` §3), identique au kernel. |
| 5 | **`scatter_add` non déterministe.** Le kernel MoE accumulait par `atomicAdd` dans une destination pré-zéroée, en FP32 pour la variante F32 et **en BF16 à chaque étape** pour la variante BF16 ; l'ordre d'arrivée des atomiques est non déterministe et l'arrondi BF16 était répété. C'était la seule opération du contrat sans budget d'ULP. | Le kernel est devenu **déterministe** : un bloc par ligne de destination, accumulation **en FP32 dans l'ordre croissant des indices de paire**, **un seul** arrondi en BF16 à la sortie — exactement l'ordre de la référence CPU (`graph.rs::experts_forward`). La destination est **écrite** en entier : le pré-remplissage à zéro n'est plus requis. |

### 5.2 Le contrat de compilation du PTX

Les ports de `claymore_math.cuh` reposent sur des **multiplications et additions
séparées** : toute contraction FMA implicite modifierait le résultat. La
commande de régénération a donc été complétée par `-fmad=false` dans les trois
en-têtes de kernels, et les trois PTX versionnés ont été régénérés avec :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
  -o crates/claymore-cuda/src/kernels/{m10,m8_f32,m8_bf16}.ptx \
  crates/claymore-cuda/kernels/{m10,m8_f32,m8_bf16}.cu
```

Les trois artefacts sont désormais **byte-identiques** à la sortie de cette
commande (vérifié par `cmp` sur les trois fichiers).

Cette exigence est cohérente avec la politique numérique existante : la
référence CPU utilise `mul_add` (`fmaf`) **explicitement** là où elle veut un
FMA (`ops.rs::matmul`, `diffusion.rs`), et `x * x` **séparé** là où elle ne le
veut pas (`ops.rs::rms_norm`). `-fmad=false` ne supprime que la contraction
*implicite* ; les FMA explicites restent émis. C'est exactement la distinction
nécessaire pour que le CPU et le CUDA produisent les mêmes bits.

> **Vérification de provenance.** La commande documentée dans
> `docs/audits/m8-ptx-provenance.md` §3 reproduit **effectivement** les trois
> PTX versionnés à `9775cda` (byte-identiques sous `nvcc 13.0.48`) : aucune
> dérive de provenance n'existait sur le SHA de départ. Ce que le correctif
> introduit, c'est une **nouvelle contrainte de compilation** : les ports
> glibc de `claymore_math.cuh` reposent sur des opérations séparées, donc les
> artefacts versionnés ne sont plus produits par la commande M8 historique. Le
> document de provenance M8 et son manifeste SHA-256 doivent donc être
> réémis avec `-fmad=false` (§9.1).
>
> Les artefacts ne sont pas équivalents entre les deux modes : sur les nouvelles
> sources, la commande historique (sans `-fmad=false`) émet 44 sites
> `fma.rn.f32` dans `m8_f32.ptx` et `m8_bf16.ptx` (23 à `9775cda`) et 324 dans
> `m10.ptx` (291 à `9775cda`), contre respectivement 15 et 214 avec
> `-fmad=false`. Ce qui est **inchangé**, c'est le résultat mesuré : le
> run 3 (mode historique) et le run 4 (mode `-fmad=false`) donnent l'un et
> l'autre `0,000000 / 0,000000` sur la chaîne complète. Le mode `-fmad=false`
> est retenu non parce que l'autre mode a été mesuré faux sur cette charge, mais
> parce qu'il rend la correspondance CPU ↔ CUDA **structurelle** (une opération
> source = une opération PTX, les FMA voulus étant explicites) au lieu de
> dépendre de ce que le compilateur choisit de contracter ce jour-là.

### 5.3 Effet de bord : trafic mémoire et nombre d'opérations

Le correctif du `scatter_add` a un effet structurel sur le graphe MoE. À HEAD,
le kernel accumulait par atomiques dans **une** destination partagée pré-zéroée :
les 128 experts y ajoutaient leurs lignes, et une seule lecture suivait. Le
nouveau kernel **écrit** un tenseur complet par expert, que le graphe additionne
ensuite. Cela change deux métriques de campagne :

| Métrique | À HEAD | Après | Explication |
|---|---|---|---|
| Opérations graphées | 605 669 | 48 285 | Le graphe ne rejoue plus la même chaîne à chaque passe de débruitage. |
| Poids uploadés (H2D) | 930 494 Mio (909 GiB) | 45 341 Mio (44 GiB) | Moins de re-streaming des slices d'experts. |
| Activations téléchargées (D2H) | 24 197 Mio | 2 022 Mio | Idem. |
| Pic VRAM suivi | 24 831 Mio | 24 831 Mio | Inchangé : le budget reste respecté. |

La baisse du trafic est **favorable** et le pic VRAM est **inchangé**, donc le
budget de 26 Gio souple / 28 Gio dur reste tenu. Aucun gain de performance
n'est revendiqué : les kernels restent naïfs, et la durée des runs finaux
(≈ 7,5 min pour la campagne complète, contre ≈ 55 min à HEAD) doit être lue au
regard de la charge de travail réellement exécutée, pas comme un gain de
performance kernel.

## 6. Ce que la campagne ne prouve pas

Le test `cuda_graph_matches_golden_first_step` reste **rouge**, et le test
`golden_cpu::encoder_matches_pinned_upstream_on_the_real_checkpoint` (mêmes
tolérances : `max_abs ≤ 0,03`, `rms ≤ 2e-3`) est **rouge de la même manière**.
C'est attendu et sans rapport avec le correctif :

| Grandeur | À HEAD | Après | Commentaire |
|---|---|---|---|
| CPU ↔ dorée eager, `max_abs` / `rms` | 5,539062 / 0,763049 | 5,828125 / 0,754065 | Rouge **inchangé**, expliqué et publié en §16.2 de l'audit M10 : une chaîne BF16 de bout en bout n'est pas reproductible contre les bits du GEMM BF16 de PyTorch CPU. |
| CUDA ↔ dorée eager, `max_abs` / `rms` | 4,875000 / 0,752871 | 5,828125 / 0,754065 | **Égal au CPU**, à la valeur près. |

Le fait mesurant est que **le CUDA suit désormais le CPU exactement**, y compris
dans son écart à la dorée. Les deux implémentations Claymore s'écartent de la
référence upstream par la **même** cause (ordre d'accumulation d'un GEMM BF16
tierce, amplifié par les couches d'attention sans scaling), et non par deux
causes indépendantes.

Un A/B CPU-only a mesuré l'effet des deux changements de formulation CPU
(§5.1, causes 3 et 4) sur l'écart à la dorée : il déplace la métrique de
`5,539062 / 0,763049` à `5,828125 / 0,754065`, soit **un niveau d'ULP** — les
deux formulations sont à un ULP près l'une de l'autre (23 % des tirages pour
`powf(-0.5)` contre `1/sqrt`, 1 UPL maximal ; 1,86 M d'écarts sur 5,63 M de
comparaisons pour `v * (1/sum)` contre `v / sum`, 1 ULP maximal). Cet écart est
**sans rapport** avec le rouge de chaîne, qui est de trois ordres de grandeur
supérieur.

Aucune tolérance n'a été assouplie. Le test rouge est conservé tel quel, avec
son message d'échec.

## 7. Vérifications exécutées

### 7.1 CPU (aucune occupation de VRAM)

| Vérification | Résultat |
|---|---|
| `cargo test --workspace --all-targets` | **vert** (0 échec) |
| `cargo clippy --workspace --all-targets -- -D warnings` | **aucun avertissement** |
| `cargo test -p claymore-model-gemma --test kernel_order_parity` | **vert** (nouveau test, §7.3) |
| A/B CPU des formulations `rms_norm` / `softmax` contre la dorée | `5,539062 / 0,763049` (HEAD) → `5,828125 / 0,754065` (corrigé) |
| Reproductibilité des PTX | les 3 PTX **byte-identiques** à `nvcc 13.0 -fmad=false` |

### 7.2 GPU (RTX 5090, autorisation explicite préalable)

| Campagne | Résultat |
|---|---|
| `hardware_m10_generation` (3 tests) | **2 verts, 1 rouge documenté** (§6) : `encoder_first_layer_matches_cpu_step_by_step` vert exact, `real_generation_runs_one_canvas_block_on_the_card` vert (256/256 positions, pic 24 831 Mio), `cuda_graph_matches_golden_first_step` rouge sur le contrat de chaîne (CUDA = CPU = 5,828125 contre la dorée) |
| `hardware_m10_primitives` (21 tests) | **21/21 verts**, tous les écarts à `0e0` / `0` ULP, y compris `scatter_add` F32 et BF16 |

### 7.3 Verrouillage en CI sans GPU

Le nouveau test `crates/claymore-model-gemma/tests/kernel_order_parity.rs` porte
une **transcription Rust** de l'arithmétique de chaque kernel CUDA (même arbre
d'expressions, même ordre de réduction, mêmes points d'arrondi) et vérifie
l'égalité bit-à-bit avec les opérations de référence du CPU. Il balaie les ports
`expf`/`expm1f`/`tanhf` contre glibc : chaque créneau d'exposant, chaque
voisinage de frontière de branche, les frontières `k` de la réduction
d'argument, les subnormaux et les cas particuliers. Le contrat de parité est
ainsi appliqué **en CI, sans GPU** : si l'ordre d'un kernel change, la
transcription doit changer avec lui, et l'égalité doit toujours tenir.

## 8. Artefacts

### 8.1 Fichiers modifiés

| Fichier | Nature du changement |
|---|---|
| `crates/claymore-cuda/kernels/claymore_math.cuh` | **nouveau** — ports glibc de `expf`/`expm1f`/`tanhf`, helpers BF16, softmax en ordre d'index |
| `crates/claymore-cuda/kernels/m10.cu` | `scatter_add` déterministe ; `softmax` et `soft_cap` sur les ports partagés ; en-tête `-fmad=false` |
| `crates/claymore-cuda/kernels/m8_f32.cu` | `softmax` et `rms_norm` en ordre d'index ; en-tête `-fmad=false` |
| `crates/claymore-cuda/kernels/m8_bf16.cu` | idem, variante BF16 |
| `crates/claymore-cuda/src/kernels/{m10,m8_f32,m8_bf16}.ptx` | régénérés, reproductibles byte-à-byte |
| `crates/claymore-cuda/src/compute.rs` | `scatter_add` : signature `out_rows`, grille par ligne de destination, plus de pré-rempli à zéro |
| `crates/claymore-memory/src/cuda.rs` | suppression du pré-remplissage à zéro de la destination `scatter_add` |
| `crates/claymore-tensor/src/ops.rs` | `rms_norm` : `1/sqrt` au lieu de `powf(-0.5)` (forme documentée) |
| `crates/claymore-tensor/src/diffusion.rs` | `softmax_row` : division par la somme au lieu de la multiplication par la réciproque |
| `crates/claymore-tensor/src/contracts.rs` | `ScatterAdd` : limite de grille sur `out_rows` |
| `crates/claymore-memory/tests/hardware_m10_primitives.rs` | tolérances `scatter_add` durcies à l'exactitude (0), test renommé |
| `crates/claymore-model-gemma/tests/kernel_order_parity.rs` | **nouveau** — parité d'ordre kernel ↔ CPU, sans GPU |
| `docs/audits/m10-cuda-cpu-parity-fix.md` | **nouveau** — le présent rapport |
| `docs/audits/m8-ptx-provenance.md` | **révisé** — section 6 : commande `-fmad=false` en vigueur, nouvelles empreintes, `claymore_math.cuh` ajouté au manifeste |
| `.gitignore` | `AGENTS.md` ignoré (fichier d'instructions d'agent local, à ne pas pousser) |

### 8.2 Empreintes des artefacts

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m10.cu` | `afde273b9a3426f6ea62b2d3257d0a73db4d8ca4f5dc0e9c755726b91227b9f9` |
| `crates/claymore-cuda/kernels/m8_f32.cu` | `7ea17e1f064849aa1d4f58cda0496f938e305e40b784fdfdab51ee4ae9c476cf` |
| `crates/claymore-cuda/kernels/m8_bf16.cu` | `42f69ab42769ec44848e08b2c803ff4259f5283ed1e4ff35c6bd60d7f6f493a6` |
| `crates/claymore-cuda/kernels/claymore_math.cuh` | `15efcd8a159db1a240355712bee4ee7ade61a1c435360dbde5af9ad40c2471f7` |
| `crates/claymore-cuda/src/kernels/m10.ptx` | `23d308dd9368fa1b21596c246b5ea4da129fd36b197234dc5a4d93d2ab673da1` |
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `f367bf2d26c490697f5d456d6a6f2afe09b50edbb75b5f202911810511d46826` |
| `crates/claymore-cuda/src/kernels/m8_bf16.ptx` | `89938e3d7507c08b77986d14fab8983a706dbcaec7f21676e8a7026398ca55fa` |

### 8.3 Journaux bruts

Les journaux de campagne sont **versionnés** dans `docs/audits/`, selon la
convention `.log.txt` du dépôt (le motif `*.log` de `.gitignore` exclut les
extensions `.log` simples, comme pour les campagnes M7.1 et M8) :

| Fichier versionné | Contenu |
|---|---|
| `m10-parity-00-campagne-avant-9775cda.log.txt` | Campagne M10.3 publiée : baseline `5,835938 / 0,711149`, 930 494 Mio uploadés, 605 669 opérations, 3 295,94 s |
| `m10-parity-01-bisect-run1-hote-neuf-ptx-ancien.log.txt` | Run 1 — hôte corrigé + PTX de `9775cda` : `rms=NaN`, et `ShapeMismatch` sur la génération (état de développement intermédiaire) |
| `m10-parity-02-bisect-run2-worktree-head.log.txt` | Run 2 — worktree détaché sur `9775cda` : baseline reproduite `5,835938 / 0,711149` |
| `m10-parity-03-bisect-run3-arbre-corrige.log.txt` | Run 3 — arbre corrigé, PTX du 22:02 (mode historique, 44 sites FMA) : `0,000000 / 0,000000` |
| `m10-parity-04-run4-ptx-reproductibles.log.txt` | Run 4 — arbre corrigé, PTX reproductibles `-fmad=false` : `0,000000 / 0,000000`, 256 positions, pic 24 831 Mio, 449,74 s |
| `m10-parity-05-primitives-21.log.txt` | Primitives M10 sur carte : **21/21 verts**, tous les écarts à `0e0` / `0` ULP |
| `m10-parity-06-cpu-golden-formulations-head.log.txt` | `golden_cpu` sur formulations CPU de HEAD : `5,539062 / 0,763049` |
| `m10-parity-07-cpu-golden-formulations-corrigees.log.txt` | `golden_cpu` sur formulations corrigées : `5,828125 / 0,754065` |

Les fichiers bruts restent également disponibles hors dépôt dans
`/tmp/opencode/m10/`.

## 9. Suites à faire

1. **Rouvrir la question de la parité avec la dorée upstream.** Le rouge
   résiduel est côté CPU et documenté (§6) ; il n'est pas traité par ce correctif
   et ne doit pas l'être sans campagne dédiée. Deux contrats coexistent
   aujourd'hui dans `hardware_m10_generation.rs` : CUDA ↔ CPU (désormais exact)
   et CUDA ↔ dorée eager (`emax ≤ 0,06`, `hardware_m10_generation.rs:116`),
   inatteignable tant que la référence CPU est à `5,828125` de la dorée. Le
   second contrat doit être soit atteint par un travail sur la chaîne CPU, soit
   **explicitement amendé** — pas relâché en silence.
2. **Ne pas revendiquer de gain de performance.** La baisse du trafic H2D et du
   nombre d'opérations est structurelle (§5.3), mais les kernels restent naïfs.
3. **Réexécuter la campagne GPU après tout changement de `.cu`.** Les PTX sont
   versionnés et jamais régénérés par le build : une modification source sans
   régénération ferait dériver l'artefact chargé de la source relue.

## 10. Reproduction

```bash
# CPU, sans VRAM
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p claymore-model-gemma --test kernel_order_parity

# Reproductibilité des PTX
for k in m10 m8_f32 m8_bf16; do
  /usr/local/cuda-13.0/bin/nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
    -fmad=false -o /tmp/$k.ptx crates/claymore-cuda/kernels/$k.cu
  cmp /tmp/$k.ptx crates/claymore-cuda/src/kernels/$k.ptx
done

# GPU — autorisation explicite de l'utilisateur requise au préalable
CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
cargo test --release -p claymore-model-gemma --features cuda-streaming \
  --test hardware_m10_generation -- --ignored --test-threads=1 --nocapture

RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
cargo test --release -p claymore-memory --features cuda-driver \
  --test hardware_m10_primitives -- --ignored --test-threads=1 --nocapture
```
