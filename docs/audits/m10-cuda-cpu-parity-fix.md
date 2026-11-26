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

#### 7.1.1 Réplique locale des deux jobs de CI

Au moment du correctif, la CI du dépôt (`ci/cuda-driver-stub.c` +
`ubuntu-latest`, sans GPU) ne se déclenchait que sur `push` vers `main` ou
`feat/**`, et sur les `pull_request` visant `main`
(`.github/workflows/rust.yml`, bloc `on:`). Une branche `fix/**` ciblant
`feat/m10-diffusiongemma` ne déclenchait donc **aucun** run : la preuve CI a été
produite localement, en rejouant à l'identique les deux jobs.

Le bloc `on:` a depuis été étendu, pour que ce cas ne se reproduise plus :

| Événement | Branches | Effet |
|---|---|---|
| `push` | `main`, `feat/**` | inchangé |
| `pull_request` | `main`, **`feat/**`** | couvre les PR de travail vers une branche de jalon |
| **`workflow_dispatch`** | branche appelante | validation à la demande d'une branche de travail, sans PR |

Contrainte consignée dans le fichier même : pour un événement `pull_request`,
GitHub lit la définition du workflow **de la branche de base**, pas celle de la
branche source. Une branche de travail ne peut donc pas activer sa propre CI ;
le déclencheur `pull_request: branches: ["feat/**"]` n'a d'effet une fois fondu
sur la branche ciblée. La PR #40, base `main`, n'a jamais dépendu de ce
changement : son événement `pull_request` visait déjà `main`.

| Job `test` (stable) | Résultat |
|---|---|
| `cargo fmt --all -- --check` | **rouge d'abord** sur 3 fichiers touchés par le correctif, corrigé par `cargo fmt --all` ; le reste du dépôt était propre |
| `cargo test --workspace --all-targets --locked` | **vert** — 324 tests passés, 0 échec, 23 ignorés |
| Stub CUDA puis tests `--lib` de `claymore-memory` et `claymore-cuda` (`--features cuda-driver`) | **verts** |
| `cargo check` des features CUDA (`claymore-cuda`, `claymore-memory`, `claymore-cli`, `claymore-model-gemma --features cuda-streaming`) | **verts** |
| `cargo clippy` workspace + crates CUDA, avec `-D warnings` | **aucun avertissement** |

Réplique du job `msrv` (Rust 1.86) : §7.1.2. Journaux bruts versionnés :
`m10-parity-08-ci-local-stable.log.txt` et
`m10-parity-09-ci-local-msrv-1.86.log.txt`.

#### 7.1.2 Job `msrv` (Rust 1.86)

Rejoué avec `rustup run 1.86.0` (cargo 1.86.0), répertoire cible séparé, même
stub CUDA :

| Étape | Résultat |
|---|---|
| `cargo test --workspace --all-targets --locked` | **vert** — 264 tests passés, 0 échec, 7 ignorés ; 23 crates compilées |
| `cargo check` des quatre cibles de feature CUDA | **verts** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` + les deux commandes CUDA | **aucun avertissement** |

`kernel_order_parity` s'exécute aussi sur 1.86 : ses 7 tests passent, le
portage glibc n'utilise donc aucune API postérieure à la MSRV déclarée.

#### 7.1.3 Réplique rejouée sur la branche d'intégration

Le correctif a été intégré à `feat/m10-diffusiongemma` par fast-forward pur
(§11.1), puis les deux jobs ont été rejoués à l'identique sur cet arbre, en
commençant par le `cargo fmt --all -- --check` qui ouvre le job `test` de la CI.
Journal brut versionné : `m10-parity-10-ci-local-branche-integree.log.txt`.

| Job | Résultat |
|---|---|
| `test` (stable) | **PASS** — fmt vert ; `cargo test --workspace --all-targets --locked` : **324 passés, 0 échec, 24 ignorés** sur 18 binaires ; stub CUDA puis les deux suites `--lib` `cuda-driver` ; les quatre `cargo check` de features CUDA ; les trois `cargo clippy` avec `-D warnings` |
| `msrv` (cargo 1.86.0) | **PASS** — `cargo test --workspace --all-targets --locked` : **264 passés, 0 échec, 8 ignorés** sur 16 binaires ; `check` et `clippy` des cibles CUDA sans avertissement |

Le nombre d'ignorés passe de 23 à 24 (stable) et de 7 à 8 (MSRV) : le test de
bisect par couche §11.4 est `#[ignore]`, il ne s'exécute donc pas en CI (il
exige le checkpoint épinglé et les dorées, hors dépôt).

### 7.2 GPU (RTX 5090, autorisation explicite préalable)

**Provenance de l'arbre exécuté** — ces deux lignes viennent de la campagne de
validation du correctif, lancée sur l'arbre de la branche
`fix/m10-numerics-divergence` **avant** le commit `9ccbe83` qui la rapporte. Les
kernels, l'en-tête partagé et les PTX de cet arbre sont byte-identiques à la tête
intégrée `ef1a1cf` ; le commit `c953b19` postérieur ne modifie que la mise en
forme Rust (`compute.rs`, `contracts.rs`, `kernel_order_parity.rs`), sans changement
de sémantique. La tête `ef1a1cf` elle-même n'a **pas** été rejouée sur carte.

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

Cette phrase doit être lue avec la qualification du §7.4 : l'oracle du balayage
des transcendants est une version *donnée* de la libm (glibc 2.39), pas la libm
de la machine qui exécute le test.

### 7.4 État réel de la CI GitHub (25 novembre) et cause de l'échec

**Ce qui a été observé.** La CI de la PR #40 est **rouge**, et cela a été
vérifié deux fois sur GitHub Actions lui-même (API + journaux d'exécution),
consignation brute aux journaux `m10-parity-12` (§8.4) et `m10-parity-13` :

| Tête | Déclencheur | Run | Jobs en échec | Résolu |
|---|---|---|---|---|
| `ef1a1cf` | `push` | `45533723982` (25/11 02:32:50Z) | `138935091310` (test), `138935090884` (MSRV) | `completed/failure` |
| `ef1a1cf` | `pull_request` | `45533731837` (25/11 02:32:54Z) | `138935118950` (test), `138935119267` (MSRV) | `completed/failure` |
| `85f7772` | `push` | `45538595444` (25/11 03:14:33Z) | `138951244933` (test), `138951245523` (MSRV) | `completed/failure` |
| `85f7772` | `pull_request` | `45538600058` (25/11 03:14:35Z) | `138951262162` (test), `138951261791` (MSRV) | `completed/failure` |

Étape en échec dans les huit cas : « Run tests », et « Verify workspace on the
declared MSRV » pour les deux jobs MSRV. Baseline : les runs sur `9775cda`
(24/11 11:15Z) étaient **verts** sur les deux déclencheurs. Or
`tests/kernel_order_parity.rs` est **introduit par le correctif** : la régression
vient du verrou lui-même, pas d'un rouge hérité. L'élargissement des
déclencheurs du §7.1.1 n'a aucun effet sur ces runs (le workflow d'un
`pull_request` est lu sur la branche de base).

**Cause du premier rouge (ef1a1cf).** Deux assertions du même test échouent, sur
les deux jobs, avec les mêmes valeurs :

| Entrée | bits du port | bits de la libm du runner | écart |
|---|---|---|---|
| `tanhf(0x3c000000 = 2⁻⁷)` | `0x3bfffeaa` | `0x3bfffeab` | 1 ULP |
| `expm1f(0x34000000 = 2⁻²³)` | `0x34000000` | `0x34000001` | 1 ULP |

L'image du runner est `ubuntu-26.04` (image `20261116.202.1`) ; l'hôte de
référence de la campagne, du golden et du PTX est Ubuntu 24.04.5,
`ldd (Ubuntu GLIBC 2.39-0ubuntu8.12) 2.39`. Les ports de `claymore_math.cuh`
implémentent l'algorithme de **glibc 2.39** ; le test comparait les ports à la
libm **installée**. L'oracle n'était donc pas épinglé : le test dépendait de
l'image du runner. Sur l'hôte de référence, le même balayage ne présente
**aucune** divergence.

**Réfutation consignée (a).** Le correctif `85f7772` publiait ces deux entrées
dans `PINNED_GLIBC_2_39` et le journal `m10-parity-12` proclamait : « *Sur
l'image ubuntu-26.04, les ports portables de `claymore_math.cuh` diffèrent de la
libm installée sur exactement DEUX entrées de leur balayage* ». C'est **FAUX**,
et la raison est technique : l'`assert_eq!` est à l'intérieur de la boucle du
balayage, donc le test s'arrête à la **première** divergence rencontrée. Deux
tests en échec donnaient deux divergences **observables** — la première de chaque
fonction —, pas deux divergences **existantes**. La phrase d'origine est
conservée dans le journal `m10-parity-12`, avec sa réfutation en regard.

**Cause du second rouge (85f7772).** Le balayage, libéré de ces deux points,
meurt plus loin — deux nouvelles entrées, mêmes jobs, mêmes étapes :

| Entrée | bits du port | bits de la libm du runner | écart |
|---|---|---|---|
| `tanhf(-2⁻⁷)` | `3154116266` | `3154116267` | 1 ULP |
| `expm1f(1.0)` | `1071378600` | `1071378601` | 1 ULP |

Extrait verbatim du job `138951262162` :

```text
assertion `left == right` failed: expm1f(1.0) port=1.7182817 glibc=1.7182819
    left: 1071378600
   right: 1071378601
  at .../tests/kernel_order_parity.rs:512:9
assertion `left == right` failed: tanhf(-0.0078125) port=-0.0078123407 glibc=-0.007812341
    left: 3154116266
   right: 3154116267
  at .../tests/kernel_order_parity.rs:596:9
test result: FAILED. 5 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out
```

La libm de l'image du runner diffère donc de glibc 2.39 sur **plusieurs**
entrées du balayage, et rien n'établit que la liste soit bornée. En déduire que
« comparer le port à la libm installée » est un contrat était l'erreur : ce
contrat est déplacé par l'image.

**Contrat retenu.** Le test n'affirme plus la parité contre la machine, il
l'affirme contre **l'oracle publié**. Deux paliers, dans
`crates/claymore-model-gemma/tests/kernel_order_parity.rs` :

1. **sur toute machine** — le port doit reproduire les résultats glibc 2.39
   committed dans trois tables de référence versionnées
   (`tests/data/glibc_2_39_{expf,expm1f,tanhf}.bin`, 55 898 entrées au total,
   format `CMP1`, SHA-256 au journal `m10-parity-13`) ; les entrées de la table
   sont en outre affirmées identiques à l'ensemble décimé du balayage, donc une
   dérive du générateur fait échouer le test ;
2. **seulement si l'hôte est l'oracle** — la sonde `SENTINELS_GLIBC_2_39` teste
   deux points connus ; si la libm installée y égale glibc 2.39, le port doit
   **aussi** égaler la libm installée sur **toutes** les entrées balayées,
   épinglées ou non. C'est ce qui maintient le pas dense sous contrat là où les
   tables ne vont pas. Hors hôte-oracle, ce palier est ignoré **avec une ligne
   sur stderr** (le saut n'est pas masqué ; un binaire de test capture ce flux,
   donc la ligne n'apparaît qu'avec `--nocapture`) et le diagnostic ignoré
   `report_divergence_against_installed_libm` publie le jeu de divergence complet.

Les tables ne sont régénérables que sur un hôte-oracle :
`regenerate_glibc_2_39_reference_tables` (ignoré) refuse toute machine dont la
libm diffère des sondes, ce qui rend impossible la publication involontaire de
résultats d'une libc non oracle. Une étape du job `test`, **non assertive**
(`continue-on-error`), publie le jeu de divergence de l'image du runner dans le
journal du job : c'est une donnée pour la régénération, pas un critère de
réussite. Aucune tolérance n'est relâchée, aucun test n'est désactivé, aucune
entrée n'est retirée du balayage.

**Correction de couverture découverte au passage.** Les grilles d'exposants du
balayage construisaient leur mantisse par `(m * 0x40000) & 0x800000` : le masque
capturait le bit de signe et les mantisses collisionnaient par période de 32, si
bien que la grille annoncée « 64 échantillons par exposant » n'en couvrait
effectivement que **deux**. Le masque corrigé `& 0x007f_ffff` en couvre 32. Le
nombre d'entrées poussées est inchangé, le nombre d'entrées **distinctes** non
(`expf` 24 632, `expm1f` 532 952, `tanhf` 120 394, total **677 978**) : la
couverture réellement contractuelle est désormais mesurée, pas annoncée.

**Ce que cela change à la lecture du dossier.** Le contrat de parité CUDA ↔ CPU
est **qualifié par la libc de l'hôte**. Toute revendication « bit-exact » doit
nommer l'hôte : ici glibc 2.39, celui du golden PyTorch et de la campagne
RTX 5090 du §7.2. Le second palier rend cette qualification explicite au lieu de
la laisser implicite : sur une image plus récente, la référence CPU et les
kernels divergeraient d'un ULP sur ce genre d'entrée sans que l'ordre des
kernels ait changé — ce n'est pas une régression du correctif, c'est une
propriété de l'oracle.

**Vérifications rejouées après correction** (hôte glibc 2.39) :
`cargo fmt --all -- --check` OK ; `cargo test -p claymore-model-gemma --test
kernel_order_parity` → **7 passés / 0 échec / 2 ignorés**, palier 2 actif (sur
cet hôte la sonde passe, donc aucune ligne de saut n'est émise) ;
`cargo clippy -p claymore-model-gemma
--all-targets -- -D warnings` code de sortie `0` ; réplique locale du job `test`
→ **12 étapes OK, 325 passés / 0 échec / 26 ignorés** ; réplique locale du job
MSRV `1.86.0` → **8 étapes OK, 264 passés / 0 échec / 10 ignorés**. Chiffres
détaillés au journal `m10-parity-13`.

**État du run déclenché par le commit de correction — relevé sur GitHub Actions.**
Le commit `88d3989e3cac1cbc3b7a0d0cbe80e871ea61e5ae` déclenche deux runs, tous deux
**verts** :

| Déclencheur | Run | `test` | `msrv` |
|---|---|---|---|
| `push` | `45539679704` (25/11 03:23:47Z → 03:24:16Z) | `138954848745` **succès** | `138954848291` **succès** |
| `pull_request` | `45539689788` (25/11 03:23:52Z → 03:24:30Z) | `138954881028` **succès** | `138954880571` **succès** |

Preuves dans le journal du job `138954881028` (relevé brut, §5 du journal
`m10-parity-13`) :

- étape « Run tests » : `kernel_order_parity` → `test result: ok. 7 passed; 0
  failed; 2 ignored`, les deux ignorés étant `regenerate_glibc_2_39_reference_tables`
  et `report_divergence_against_installed_libm` ; **aucune** occurrence de la ligne
  de saut, ce qui confirme que le flux stderr du palier 2 est capturé par le
  binaire de test hors `--nocapture` ;
- étape de mesure (`--nocapture`, `continue-on-error`) sur l'image
  `ubuntu-26.04` : `expf: 0 divergence(s) out of 32878 swept inputs (24632
  distinct)`, `expm1f: 3242 divergence(s) out of 549204 swept inputs (532952
  distinct)`, `tanhf: 890 divergence(s) out of 126112 swept inputs (120394
  distinct)`, `total: 4132 divergence(s) out of 708194 swept inputs (677978
  distinct)` ; désaccords avec les tables épinglées : `expf 0 de 9272`,
  `expm1f 211 de 33752`, `tanhf 446 de 12874`.

Ce relevé **clôture la réfutation** : la libc de l'image du runner diffère de
glibc 2.39 sur **4 132** entrées balayées, dont **657** des entrées des tables
épinglées — pas deux. Le premier rouge n'en montrait que deux parce que
l'assertion coupe au premier écart. Le contrat épinglé, lui, est passé sur le
runner sans modification du port : c'est bien la comparaison à l'hôte qui était la
source du rouge, pas la parité.

Ce fichier ne peut pas contenir l'état du run qu'il déclenche **lui-même** : la
boucle est coupée là, volontairement, et l'état du run de la tête
finale est donné dans le commentaire de la PR, pas ici.

**Réfutation consignée (b).** La conclusion antérieure « les runs de la PR #40 ne
sont pas observables : le jeton n'a pas le périmètre Actions/Checks (réponses
404) » est **fausse**. Les 404 venaient d'un chemin de dépôt erroné
(`StephaneXB/Claymore`). Sur le dépôt réel `zenhit777/Claymore`,
`GET /actions/runs`, `GET /commits/<sha>/check-suites` et
`GET /actions/jobs/<id>/logs` répondent **200** avec le jeton de `.env`, et les
journaux complets ont été téléchargés. La ligne « runs non observés » du dossier
`docs/m10-diffusiongemma.md` est corrigée en conséquence.

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
| `crates/claymore-model-gemma/tests/kernel_order_parity.rs` | **nouveau** — parité d'ordre kernel ↔ CPU, sans GPU ; transcendants balayés contre l'oracle glibc 2.39, palier 1 sur les tables de référence versionnées, palier 2 contre la libm installée seulement sur les hôtes-oracles ; diagnostic `report_divergence_against_installed_libm` (§7.4) |
| `docs/audits/m10-cuda-cpu-parity-fix.md` | **nouveau** — le présent rapport |
| `docs/audits/m8-ptx-provenance.md` | **révisé** — section 6 : commande `-fmad=false` en vigueur, nouvelles empreintes, `claymore_math.cuh` ajouté au manifeste |
| `.gitignore` | `AGENTS.md` ignoré (fichier d'instructions d'agent local, à ne pas pousser) |

**Ajouts de la phase de diagnostic M10.5** (25 novembre 2026, même PR) :

| Fichier | Nature du changement |
|---|---|
| `crates/claymore-model-gemma/src/fullgraph.rs` | **additif** — `encoder_forward_traced_from_hidden(...)`, exécute les couches demandées à partir d'une entrée fournie et renvoie l'état avant la normalisation finale ; `encoder_forward_traced_limited` délègue, aucune signature existante modifiée, chemin CUDA inchangé |
| `crates/claymore-model-gemma/tests/golden_cpu_layer_bisect.rs` | **nouveau** — bisect par couche à entrée dorée injectée, `#[ignore]`, égalité d'ensemble des 8 experts par ligne, borne `MAX_ROW_DRIFT_STEPS = 4.0` pas BF16 fixée une fois sur la mesure (§11.4) |
| `.github/workflows/rust.yml` | déclencheurs élargis (`pull_request` vers `feat/**`, `workflow_dispatch`) et règle de la **branche de base** consignée en commentaire (§7.1.1) ; étape **non assertive** `Report the runner libm divergence set against the pinned oracle` (`continue-on-error`), qui publie le jeu de divergence de l'image sans en faire un critère (§7.4) |
| `crates/claymore-model-gemma/tests/kernel_order_parity.rs` | **modifié deux fois** — d'abord oracle épinglé sur deux entrées mesurées (`PINNED_GLIBC_2_39`, état `85f7772`, **remplacé** : la premise « deux entrées » était fausse, §7.4 réfutation (a)) ; puis contrat en deux paliers : palier 1 contre les tables de référence glibc 2.39 versionnées sur toute machine, palier 2 contre la libm installée seulement si la sonde `SENTINELS_GLIBC_2_39` identifie l'hôte comme l'oracle, plus `regenerate_glibc_2_39_reference_tables` (ignoré, refusé hors hôte-oracle) et correction du masque des grilles d'exposants (`0x800000` → `0x007f_ffff`) (§7.4) |
| `crates/claymore-model-gemma/tests/data/glibc_2_39_{expf,expm1f,tanhf}.bin` | **nouveaux** — tables de référence oracle : 9 272 / 33 752 / 12 874 paires (bits d'entrée, bits de résultat), magic `CMP1`, triées ; 403 608 octets au total, SHA-256 au journal `m10-parity-13` |
| `docs/audits/m10-cuda-cpu-parity-fix.md` | §7.1.1 et §7.1.3 (déclencheurs, réplique rejouée), §7.4 (état réel de la CI GitHub et cause de l'échec), §8.4 (journaux), **§11 (diagnostic)** et §12 (décisions attendues) |
| `docs/audits/m10-rtx5090-hardware-validation.md` | bandeau M10.5 et mise à jour du périmètre « prouvé / non prouvé » ; aucune ligne défavorable supprimée |
| `docs/m10-diffusiongemma.md` | bloc de statut M10.4 — M10.5 (validé où, non validé quoi), clôture de la PR #44 |

### 8.2 Empreintes des artefacts

> **Note de révision (25 novembre 2026, campagne groupée).** Les quatre
> empreintes `m8_f32.cu`, `m8_bf16.cu`, `m8_f32.ptx` et `m8_bf16.ptx` ont
> changé depuis cette campagne : le correctif de la course lecture-écriture
> du softmax F32 (garde `threadIdx.x == 0` sur l'appel mono-thread
> `claymore_softmax_row`) et son jumeau BF16, avec PTX régénérés. Les
> empreintes ci-dessous sont **celles de la campagne du 24 novembre** et
> restent valides pour cette campagne ; les empreintes courantes sont celles
> du manifeste `crates/claymore-model-gemma/tests/kernel_provenance.rs`, qui
> est la source de vérité et est vérifié par la CI. Détail du correctif et de
> sa validation matérielle : `m10-rtx5090-grouped-campaign-2026-11-25.md` §3.
> `m10.cu`, `claymore_math.cuh` et `m10.ptx` sont **inchangés**.

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

### 8.4 Journaux ajoutés par la phase de diagnostic (M10.5)

Le motif `*.log` de `.gitignore` exclut l'extension `.log` simple ; ces journaux
bruts de campagne utilisent donc l'extension `.log.txt`, sans renumérotation des
fichiers déjà cités au §8.3 :

| Fichier versionné | Contenu |
|---|---|
| `m10-parity-08-ci-local-stable.log.txt` | Réplique locale du job `test` (stable), §7.1.1 |
| `m10-parity-09-ci-local-msrv-1.86.log.txt` | Réplique locale du job `msrv` (Rust 1.86), §7.1.2 |
| `m10-parity-10-ci-local-branche-integree.log.txt` | Les deux jobs rejoués sur `feat/m10-diffusiongemma` après intégration, fmt inclus : `CI-REPLICA RESULT: PASS`, `MSRV-REPLICA RESULT: PASS`, §7.1.3 |
| `m10-parity-11-bisect-couches-entrees-dorees.log.txt` | bisect couche par couche à entrée dorée injectée, avec la borne `MAX_ROW_DRIFT_STEPS = 4,0` : `lignes fideles=210/210`, pire écart `2,0` pas BF16 (couche 2), table identique au run sans borne (§11.4) |
| `m10-parity-12-ci-github-ef1a1cf-jobs-rouges.log.txt` | **état réel de la CI GitHub** sur `ef1a1cf` : 2 runs et 4 jobs en échec (IDs, URLs, extraits bruts des deux jobs), image du runner `ubuntu-26.04`, les deux entrées à 1 ULP, diagnostic local `0 divergence / 708 194 entrées` (§7.4). Portion « Lecture » **réfutée par addition** en fin de fichier (§7.4, réfutation (a)), puis **close par mesure** sur le runner : `4 132` entrées divergentes, dont `657` des tables épinglées |
| `m10-parity-13-oracle-epingle-glibc239.log.txt` | **second état rouge observé**, sur `85f7772` : 2 runs et 4 jobs en échec (IDs, URLs), deux nouvelles entrées à 1 ULP, inventaire des causes, contrat en deux paliers, tables de référence glibc 2.39 publiées (entrées, octets, SHA-256), correction de couverture du balayage, les deux répliques locales rejouées ; **§5 : premier run vert observé**, sur `88d3989` (2 runs, 4 jobs en succès, résultat de l'étape de mesure sur l'image du runner) (§7.4) |

Les fichiers bruts restent également disponibles hors dépôt dans
`/tmp/opencode/m10/`.

## 9. Suites à faire

1. **Rouvrir la question de la parité avec la dorée upstream.** **Fait, phase
   M10.5** : le diagnostic est consigné au §11, avec ses mesures et ses limites.
   Le rouge résiduel est côté CPU et n'est pas traité par ce correctif ; il ne
   doit pas l'être sans décision d'architecture (§12). Deux contrats coexistent
   aujourd'hui dans `hardware_m10_generation.rs` : CUDA ↔ CPU (désormais exact)
   et CUDA ↔ dorée eager (`emax ≤ 0,06`, `hardware_m10_generation.rs:116`). Le
   second n'est pas atteignable en l'état : la référence CPU vaut `5,828125` et
   §11.6 montre que cette borne impose l'identité bit sur 5,07 % des éléments. Il
   doit donc être **explicitement amendé** — pas relâché en silence.
2. **Ne pas revendiquer de gain de performance.** La baisse du trafic H2D et du
   nombre d'opérations est structurelle (§5.3), mais les kernels restent naïfs.
3. **Réexécuter la campagne GPU après tout changement de `.cu`.** Les PTX sont
   versionnés et jamais régénérés par le build : une modification source sans
   régénération ferait dériver l'artefact chargé de la source relue.
4. **Nommer l'hôte dans toute revendication de parité bit à bit.** La parité
   CUDA ↔ CPU est qualifiée par la libc de l'hôte de référence : glibc 2.39 ici,
   pas « la libm » en général (§7.4). Une revendication du type « bit-exact »
   sans l'hôte est infalsifiable.
5. **Mesurer le jeu de divergence quand l'image du runner bouge.** L'étape
   `Report the runner libm divergence set against the pinned oracle` du job
   `test` le fait à chaque run, sans faire échouer le job : c'est la sortie du
   diagnostic ignoré, publiée dans le journal du job. Un écart du port **contre
   la table** doit faire échouer le test, pas être épinglé : la table n'est
   régénérée que sur l'hôte-oracle, par
   `cargo test -p claymore-model-gemma --test kernel_order_parity -- --ignored
   regenerate_glibc_2_39_reference_tables`, qui refuse toute machine dont la
   libm diffère de `SENTINELS_GLIBC_2_39` (§7.4). Jamais à partir d'une
   exécution sur une image non oracle.

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

---

## 11. Diagnostic de la divergence CPU ↔ dorée (phase M10.5, 25 novembre 2026)

### 11.1 Intégration, environnement et périmètre

| Contrôle | Observation |
|---|---|
| Intégration du correctif M10.4 | `git merge --ff-only ef1a1cf` dans `feat/m10-diffusiongemma`, **fast-forward pur** : `merge-base` = tête de la branche, donc aucun doublon possible ; `git cherry -v` = 4 commits `+`, 0 `-` |
| PR #40 après poussée | tête `ef1a1cfe26ff87f289f1c94a67726797474455dc`, 40 commits, `state=open`, `merged=false` ; les quatre SHA du correctif présents dans `/pulls/40/commits` |
| PR #44 | fermée **sans fusion** : son contenu est celui de #40 |
| Provenance des PTX, revérifiée sur la branche intégrée | `nvcc 13.0 -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false` : **3/3 byte-identiques** aux fichiers versionnés ; `sha256` (16 premiers) `m10` `23d308dd9368fa1b`, `m8_f32` `f367bf2d26c49069`, `m8_bf16` `89938e3d7507c08b` — soit exactement le manifeste §8.2 |
| Occupation GPU | **aucune**. Toute exécution PyTorch de diagnostic a été lancée avec `CUDA_VISIBLE_DEVICES=""` ; `nvidia-smi` montrait un consommateur étranger à WSL (31 314 Mio, 98 %), aucun processus compute Claymore |
| Dorées | `/tmp/opencode/m10/golden_eager` : 785 tenseurs `.bin` en virgule flottante 32 bits, générés par `gen_golden_eager.py` (bf16, `device_map="cpu"`, `attn_implementation="eager"`, `experts_implementation="eager"`, prompt de 7 identifiants, tour de vision sur amorce épinglée, `decoder_position_ids = arange(7,23)`, soft-cap en fp32). **Lecture seule stricte** : aucun golden régénéré, réépinglé ou modifié pendant le diagnostic |

Ce qui suit est un **diagnostic** : il ne change aucune tolérance, ne désactive
aucun test et ne corrige rien. Les scripts de mesure sont jetables
(`probe_torch_repro_floor.py`, `probe_layer_inputs.py`) ; la seule modification
de code de production est l'ajout d'un point d'entrée de service public dans
`fullgraph.rs` (§11.4).

### 11.2 La référence est reproductible au bit près : le bruit n'est pas dans le golden

La première hypothèse à écarter était un « plancher de bruit » de la référence
elle-même (réductions non déterministes d'ATen sur CPU). Mesure directe — la
référence est rejouée deux fois, à 4 puis 16 threads, et comparée au golden
publié :

| Comparaison | `encoder_last_hidden_direct` `max_abs / rms` | couches avec sélection d'experts différente |
|---|---|---|
| 4 threads contre 16 threads | `0,000000 / 0,000000` | 0 / 30 |
| 4 threads contre le golden publié | `0,000000 / 0,000000` | 0 / 30 |
| 16 threads contre le golden publié | `0,000000 / 0,000000` | 0 / 30 |

**L'hypothèse est réfutée.** La référence est déterministe et bit-reproductible
sur ce poste ; les 5 828 125 × 10⁻³ d'écart de Claymore sont imputables à la
propre arithmétique de Claymore, amplifiée par la chaîne.

### 11.3 Où la chaîne dérive : accumulation lisible, pas de formule fautive

**Couche 0, grain par grain.** Le routeur de la couche 0 est **bit-exact**
(`raw_max_abs = 0,000000000`, 56/56 indices d'experts identiques sur les 7
jetons). Tous les états intermédiaires de la couche restent à un ou deux pas
BF16, soit ≤ 0,7 % en relatif :

| État de la couche 0 | `max_abs` | amplitude locale |
|---|---|---|
| `encoder0_input_normed` | 0,031250 | 178 |
| `encoder0_attention` | 0,062500 | — |
| `encoder0_mlp_output` | 1,0 | 149 |
| `encoder0_experts` | 0,015625 | 2,72 |
| `encoder0_router_weights` | 0,005122 | — |
| `encoder0_routing_residual` | 0,5 | 392 |
| `encoder0_pre2` | 0,007812 | 3,08 |

**Croissance régulière, sans discontinuité locale.** Écart maximal `max_abs` par
couche, mesuré sur la chaîne complète (journal `m10-parity-07`) :

| Étage | couche 0 | milieu de chaîne | fin de chaîne |
|---|---|---|---|
| `enc_kv_N_keys` | 0,000977 | 0,078125 (L8) | 0,17 – 0,28 (plateau L9‑29) |
| `enc_kv_N_values` | 0,007812 | 1,468750 (L8) → **4,5 (L9)** | 0,9 – 3,5 (plateau) |
| `dec_N_out` (décodeur) | 1,125 | 15 (L13‑14) | 2,8 – 8,8 ; `rms` 0,0171 → 0,572 |
| `vision_layer_N` | 2,0 (L3) | 32 (L23‑25) | **60 (L26)**, `rms` 3,91 |
| `vision_pooled_standardized` | — | — | 0,195312 / `rms` 0,050843 |

Le saut de l'encodeur se situe **entre les couches 8 et 9** — immédiatement
après la couche 7, première couche où un groupe de probabilités **identiques** à
cheval sur la frontière du top‑8 compte trois membres (§11.5). La tour de
vision, indépendante de l'encodeur, présente la même allure de croissance
monotone : aucune formule ne dérape à un endroit repérable.

### 11.4 Bisect à entrée dorée injectée : aucune couche ne dérive par elle-même

**Question posée.** La dérive vient-elle d'une couche qui calcule faux, ou de
couches qui calculent chacune *un tout petit peu* différemment, le tout amplifié
par une décision de sélection ? Pour trancher, chaque couche de l'encodeur est
exécutée **seule**, alimentée par l'entrée **exacte** de la référence, hors de
toute accumulation amont.

**Deux ajouts de code, tous deux additifs.**

- `crates/claymore-model-gemma/src/fullgraph.rs` : nouveau point de service public
  `encoder_forward_traced_from_hidden(host, ckpt, cfg, start, cache, layer_trace, layers)`,
  qui exécute les couches de `layers` à partir du tenseur `start` fourni et
  renvoie `x` **avant** la normalisation finale. `encoder_forward_traced_limited`
  lui délègue désormais : une seule source de vérité, aucune signature existante
  modifiée, chemin CUDA non touché.
- `crates/claymore-model-gemma/tests/golden_cpu_layer_bisect.rs` (test `#[ignore]`,
  déclaré explicitement hors CI : il s'auto-suspend si le checkpoint épinglé ou
  les dorées sont absents) : pour chaque couche
  0 → 29, compare la sortie de couche, la sortie agrégée des experts et les
  scores du routeur, et classe chaque ligne (jeton) selon l'**égalité d'ensemble**
  des 8 experts sélectionnés — pas selon l'ordre, qui n'a pas de sens sémantique.

**Validation du point d'injection** (avant toute conclusion) : `enc_in_0` est
identique à `encoder_embedding` et `enc_in_{L+1}` identique à `encoder_layer_L`,
`max_abs = 0,0` dans les deux cas. La couche reçoit donc bien l'entrée de la
référence, pas une approximation.

**Mesure** — écart maximal par couche, et écart rapporté au **pas BF16 du pic de
la ligne dorée** (1,0 = un seul pas d'arrondi BF16) :

| couche | sortie de couche `max_abs` | sortie `rms` | sortie d'experts `max_abs` | scores du routeur `max_abs` | lignes fidèles | pire écart (pas BF16) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,5000 | 0,00681 | 0,01562 | 0,002429 | 7/7 | 1,0 |
| 1 | 0,1875 | 0,00457 | 0,00781 | 0,000413 | 7/7 | 0,5 |
| 2 | 0,5000 | 0,00578 | 0,00781 | 0,000452 | 7/7 | 2,0 |
| 3 | 0,2500 | 0,00464 | 0,01562 | 0,000330 | 7/7 | 1,0 |
| 4 | 0,0625 | 0,00184 | 0,00781 | 0,000056 | 7/7 | 0,2 |
| 5 | 0,2500 | 0,00467 | 0,01562 | 0,000268 | 7/7 | 1,0 |
| 6 | 0,1250 | 0,00177 | 0,00781 | 0,000201 | 7/7 | 0,5 |
| 7 | 0,2500 | 0,00575 | 0,00781 | 0,000326 | 7/7 | 1,0 |
| 8 | 0,2500 | 0,00533 | 0,01562 | 0,000397 | 7/7 | 1,0 |
| 9 | 0,5000 | 0,00598 | 0,00781 | 0,000128 | 7/7 | 1,0 |
| 10 | 0,5000 | 0,00522 | 0,00781 | 0,000076 | 7/7 | 1,0 |
| 11 | 0,1875 | 0,00476 | 0,00391 | 0,000668 | 7/7 | 0,4 |
| 12 | 0,0625 | 0,00189 | 0,00098 | 0,000196 | 7/7 | 0,1 |
| 13 | 0,0625 | 0,00182 | 0,00781 | 0,000037 | 7/7 | 0,1 |
| 14 | 0,1250 | 0,00275 | 0,00391 | 0,000217 | 7/7 | 0,2 |
| 15 | 0,1250 | 0,00454 | 0,00391 | 0,000482 | 7/7 | 0,2 |
| 16 | 0,1250 | 0,00553 | 0,00781 | 0,000407 | 7/7 | 0,2 |
| 17 | 0,2500 | 0,00645 | 0,00781 | 0,002675 | 7/7 | 1,0 |
| 18 | 0,0625 | 0,00246 | 0,00781 | 0,000230 | 7/7 | 1,0 |
| 19 | 0,2500 | 0,00278 | 0,00781 | 0,000173 | 7/7 | 1,0 |
| 20 | 0,5000 | 0,00448 | 0,00195 | 0,000541 | 7/7 | 2,0 |
| 21 | 0,2500 | 0,00285 | 0,00098 | 0,000164 | 7/7 | 1,0 |
| 22 | 0,1562 | 0,00340 | 0,00781 | 0,000140 | 7/7 | 0,6 |
| 23 | 0,1250 | 0,00490 | 0,00781 | 0,000268 | 7/7 | 1,0 |
| 24 | 0,1250 | 0,00237 | 0,00781 | 0,000270 | 7/7 | 1,0 |
| 25 | 0,0312 | 0,00191 | 0,00781 | 0,000192 | 7/7 | 0,2 |
| 26 | 0,0625 | 0,00229 | 0,00781 | 0,000458 | 7/7 | 2,0 |
| 27 | 0,0312 | 0,00181 | 0,00781 | 0,000428 | 7/7 | 1,0 |
| 28 | 0,0156 | 0,00174 | 0,00195 | 0,000171 | 7/7 | 1,0 |
| 29 | 0,1250 | 0,00224 | 0,00195 | 0,000172 | 7/7 | 1,0 |

> `lignes fideles=210/210  pire ligne fidele=2.0 ULP (couche 2)  pire ligne globale=2.0 ULP (couche 2)`
> — journal brut versionné `m10-parity-11-bisect-couches-entrees-dorees.log.txt`.

**Lecture.** Isolées de toute accumulation, les 30 couches reproduisent
**l'ensemble exact** des 8 experts sur **210 lignes sur 210**, et aucune ligne ne
s'écarte de plus de **deux pas BF16** du pic de sa ligne dorée. La couche 7 —
celle après laquelle la chaîne complète saute (§11.3) — n'a **aucun**
comportement particulier : 7/7, 1,0 pas.

Le contraste est la preuve demandée. À entrée dorée, la sortie agrégée des
experts ne dépasse jamais `1,56 × 10⁻²` ; dans la chaîne complète, les valeurs du
cache KV portent un écart de `1,47` en couche 8 et `4,5` en couche 9, soit **deux
ordres de grandeur** au-dessus. Une couche ne dérive donc pas par elle-même : la
dérive naît de l'accumulation et des décisions de sélection franchies en chaîne.

**Verrou ajouté.** Le test porte désormais une borne numérique **unique et
documentée**, `MAX_ROW_DRIFT_STEPS = 4.0` (pas BF16), fixée une fois à partir de
la table ci-dessus — le double de la dérive maximale mesurée (2,0, couches 2, 20
et 26) — avec égalité stricte du nombre de lignes fidèles (`faithful_rows ==
rows`). Elle absorbe le bruit d'ordre de réduction d'un pas et fait échouer toute
régression structurelle : formule modifiée, point d'arrondi déplacé, ou sélection
d'experts qui dérive. Run vert après ajout de la borne : `test result: ok. 1
passed; 0 failed` (330,9 s, checkpoint épinglé + dorées, `--test-threads=1`,
chemin **CPU uniquement**, aucune VRAM sollicitée).

**Ce que ce bisect ne dit pas.** Il valide les couches *à entrée dorée*. Il ne
dit rien de la fidélité de la chaîne quand l'entrée vient de Claymore lui-même —
c'est précisément là que se loge le rouge, et c'est l'objet des §11.5 et §11.6.

### 11.5 Le point de bascule est une décision discontinue, pas une formule

La sélection d'experts est prise par `torch.topk` **sur les probabilités
softmax en fp32**, pas sur des logits (`dg/modeling_diffusion_gemma.py:522‑531` :
`softmax(..., dtype=torch.float32)` puis `torch.topk(router_probabilities, k=8)`).
Les dorées `encoder{L}_router_scores` sont donc bien en fp32 non quantifié : les
16 bits bas des motifs binaires sont non nuls sur **896/896** valeurs testées, ce
qui écarte un artefact d'arrondi BF16 de la Dorée.

Or cette frontière est **exacte** dans les scores de la référence elle‑même :

| Mesure sur les 210 couples (couche, jeton) de la phase encodeur | Valeur |
|---|---|
| Couples où 8ᵉ et 9ᵉ probabilités sont **bit-identiques** | **31 / 210** |
| Couches concernées (sur 30) | **20** |
| Couples où le groupe de valeurs identiques à cheval sur la frontière compte **≥ 3** membres | **6** (dont couche 7, jeton 0 : même probabilité `1,4831516e‑2` aux 8ᵉ, 9ᵉ et 10ᵉ rangs) |
| Nombre de sélections top‑8 **distinctes compatibles** avec les probabilités publiées par la référence | **24 461 180 928** |
| Marge 8ᵉ → 9ᵉ minimale hors égalité | 1,35 × 10⁻⁴ (couche 0) |
| Couples non équivoques (écart strictement supérieur à un pas BF16) | 179 / 210 |

Conséquence : pour ces couples, **aucun plan numérique fidèle ne détermine la
sélection** — elle est décidée par l'ordre intérieur du tri d'ATen, pas par les
nombres. Et l'amplitude de la perturbation de Claymore sur ces mêmes
probabilités est mesurée, couche 0 : `max_abs = 2,429 × 10⁻³`,
`rms = 8,9 × 10⁻⁵`, soit jusqu'à **16 %** de la valeur de frontière
(≈ 1,5 × 10⁻²). 30 couches sur 30 présentent au moins un jeton dont la marge est
sous ce bruit.

Le port Claymore du tri n'est **pas** en cause : sur les scores dorés, il rend
les mêmes indices qu'ATen (couche 0 : routeur bit-exact, 56/56 indices
identiques ; l'émulation `claymore_router_top_k_f32` est validée 210/210 contre
`torch.topk`, cf. `docs/m10-diffusiongemma.md`). La discontinuité est franchie
en amont, par le plan numérique.

Enfin, une sélection différente n'est pas anodine : la contribution d'un expert
à la sortie de couche vaut environ 0,17 × 0,385 ≈ **6 × 10⁻²**, soit **deux fois
le budget de la porte** (0,03) — avant toute accumulation sur les 30 couches.

### 11.6 La porte absolue est inférieure au pas BF16 sur la queue

Sur `encoder_last_hidden_direct` (19 712 éléments, amplitude maximale 23,625), le
**pas BF16 local** — plus petit écart representable, donc plus petite différence
non nulle possible :

| Statistique du pas BF16 | Valeur |
|---|---|
| médiane | 0,007812 |
| 95ᵉ percentille | 0,031250 |
| maximum | 0,125 |
| éléments dont le pas BF16 **dépasse** la porte `max_abs ≤ 0,03` | **999 / 19 712 = 5,07 %** |
| éléments dont le pas BF16 dépasse `0,06` (contrat CUDA ↔ dorée) | 91 = 0,46 % |

Autrement dit, `max_abs ≤ 0,03` n'est pas une tolérance : sur 5 % de la queue de
distribution, cette porte **impose l'identité bit**. Elle exige donc de
reproduire l'arithmétique exacte d'ATen (ordre de réduction des GEMM, réductions
bloquées), et non pas de converger statistiquement. C'est la raison pour laquelle
le rouge restant ne se traite pas par un réglage de seuil.

### 11.7 Sources de bruit résiduel, chiffrées

| Source | Mesure | Décision proposée |
|---|---|---|
| Forme de l'inverse de RMS : `ops.rs:271‑272` calcule `1,0f32 / sqrt(ms + eps)` (choix M10.4, §5.1 item 4, pour aligner le CPU sur le kernel) alors que la référence calcule `pow(ms, -0.5)` | sur 35 valeurs réelles de `ms + eps` (résiduels des couches 0, 1, 7, 8, 29) : `pow(-0.5)` ≠ `1/sqrt` sur **4/35** ; version Rust de `1/sqrt` ≠ `pow(-0.5)` de la référence sur **5/35** ; ≠ `1/sqrt` de la référence sur **1/35** | **ne pas modifier** : ~10 à 15 % des appels de normalisation portent 1 ULP fp32 d'écart, mais aligner le CPU sur ATen casserait la parité CUDA ↔ CPU fraîchement rétablie (§8.2, §7.2) sans aucune chance d'atteindre la porte de §11.6. À consigner comme bruit résiduel assumé. |
| Ordre de réduction des GEMM et des sommes : ATen réduit par blocs vectorisés, Claymore en un seul accumulateur séquentiel | non isolable sans instrumentation d'ATen ; c'est la cause structurelle du 1 ULP observé partout en couche 0 (§11.3) | hors périmètre d'un correctif ; relève de la décision §11.8 |
| Autres formules, ordres, constantes et bornes de dtype de la chaîne encodeur | revue de code contre `modeling_diffusion_gemma.py` : **aucune autre déviation sémantique** relevée (mises à l'échelle `scaling = 1,0` des deux côtés, poids de l'encodeur liés à ceux du décodeur, causalité et fenêtre de l'encodeur inertes à 7 jetons) | aucune action ; noté comme revue de lecture, **non** comme preuve numérique |

### 11.8 Ce que le diagnostic établit, ce qu'il laisse ouvert

**Établi, avec les mesures ci‑dessus :**

1. La référence est bit-reproductible sur le poste de mesure (§11.2) ;
2. les opérations de la chaîne sont, grain par grain, à un ou deux pas BF16 de
   la référence en première couche, et le routeur y est bit-exact (§11.3) ;
3. la sélection top‑8 de la référence est **sous-déterminée par ses propres
   scores** sur 31 des 210 couples mesurés, avec jusqu'à 2,4 × 10¹⁰ sélections
   compatibles (§11.5) ;
4. la porte `max_abs` imposée par les tests est, sur 5,07 % des éléments,
   **inférieure au plus petit écart representable** (§11.6).

**Le mécanisme du rouge est donc** : 1 à 2 ULP d'écart par opération
(accumulation), franchissement répété de la frontière de sélection d'experts
(décision discontinue), puis amplification multiplicatrice sur 30 couches
encodeur et 30 couches décodeur. **Ce n'est pas** : une formule fautive, un
golden corrompu, une non-déterministicité de la référence, ni un défaut du port
du tri.

**Non établi / hors périmètre de ce diagnostic :** attribution chiffrée de
chaque source de 1 ULP entre la forme de `rms_norm` et l'ordre des réductions ;
le comportement de la chaîne sur un prompt plus long, hors du petit jeu épinglé ;
toute performance.

**Qualification nouvelle (25/11, §7.4), à lire avec le point 4 et avec le §7.2 :**
le contrat de parité bit à bit est conditionné à la libc de l'hôte. Les ports
reproduisent **glibc 2.39** (l'hôte du golden et de la campagne) ; sur l'image de
runner `ubuntu-26.04`, la libm installée diffère de cet oracle sur **plusieurs**
entrées du balayage, pas deux : les deux premières (`tanhf(2⁻⁷)`, `expm1f(2⁻²³)`)
ont rendu la CI rouge, et une fois épinglées deux autres (`tanhf(-2⁻⁷)`,
`expm1f(1.0)`) ont pris la suite — le premier écart d'une assertion placée dans
la boucle n'est pas l'inventaire. Le verrou de CI n'épingle donc plus un hôte :
il épingle **les résultats de l'oracle** (tables versionnées, valables sur toute
machine) et ne compare à la libm installée que sur les hôtes reconnus comme
l'oracle. Aucune revendication « bit-exact » ne vaut sans l'hôte nommé.

**Décisions attendues de l'architecte** (§12) : la métrique de conformité contre
la référence upstream ne peut pas rester une borne absolue `max_abs` sur une
chaîne de 60 couches BF16. Quatre positions sont possibles — métrique relative
par étage, comparaison arrêtée à un étage amont validable à entrée dorée,
fabrication de dorées auto‑cohérentes, ou bit-exactness ATen — et leur coût est
sans commune mesure. Aucune campagne GPU n'est utile à cette question : le chemin
CUDA est déjà **bit-égal** au chemin CPU mesuré ici (§7.2), donc toute campagne
sur la carte reproduirait exactement le même différentiel.

## 12. Décisions attendues

Le §11 ne laisse qu'une ouverture réelle : **sur quoi mesurer la conformité à la
référence upstream**. Quatre positions, aux coûts non comparables.

| ID | Position | Coût | Ce que ça prouve | Risque principal |
|---|---|---|---|---|
| **D1** | Métrique **relative par étage** : erreur rapportée à l'amplitude locale, exprimée en pas BF16, plus une règle explicite sur la sélection d'experts (égalité d'ensemble exigée quand la marge de la référence est non équivoque, constatée comme sous-déterminée quand elle est nulle) | reprise des trois assertions de `golden_cpu.rs` et du contrat `hardware_m10_generation.rs:116` ; seuils fixés **une fois** sur la table mesurée ; rien à recalculer côté dorées | la chaîne est fidèle à l'arithmétique BF16 près, et l'écart est suivi étage par étage | une métrique relative mal bornée peut masquer une dérive réelle ; nécessite la garde de §11.4 |
| **D2** | **Comparaison arrêtée à un étage amont** validable à entrée dorée : `golden_cpu_layer_bisect` devient contrat, `encoder_last_hidden_direct` passe en journal de suivi | quasi nul — le verrou existe, il est vert (§11.4) ; c'est une décision de périmètre | chaque couche reproduit l'arithmétique **et** la sélection de la référence | ne dit plus rien de la fidélité de la chaîne longue, qui est l'objet de M10 |
| **D3** | **Bit-exactness ATen** : reproduire l'ordre de réduction bloqué de chaque GEMM et de chaque somme d'ATen | réécriture des GEMM et réductions du CPU, puis régénération des PTX pour ne pas casser la parité interne ; dépendance à la version de libtorch, au nombre de threads et à l'ISA de la machine qui a fabriqué le golden | conformité amont au bit près — sur la machine de référence seulement | **exclue sur preuve, pas sur coût** : §11.5 montre que la sélection de la référence est sous-déterminée par ses propres scores, et §11.6 que la porte impose l'identité bit ; un plan numérique aligné n'est donc pas garant de reproduire l'ordre intérieur du tri |
| **D4** | **Dorées auto-cohérentes** : fabriquer les dorées avec Claymore lui-même | régénération des 785 tenseurs et ré-épinglage | la non-régression de Claymore contre Claymore | ne valide plus rien contre upstream ; supprime le seul signal qui reste sur la conformité amont |

**Recommandation — à trancher par l'architecte, ce rapport ne préjuge pas du
choix** : **D1 et D2 combinés**, D3 exclue, D4 refusée.

- D2 donne le verrou structurel immédiat, déjà vert et chiffré (210/210 lignes,
  dérive maximale 2,0 pas BF16) ;
- D1 remplace la borne absolue infaisable (§11.6) par une borne relative assortie
  d'une règle de sélection, avec les mêmes journaux ;
- le contrat `hardware_m10_generation.rs:116` doit être **réécrit** dans ces
  termes dans le même mouvement — pas élargi, pas désactivé ;
- la forme `1/sqrt` de `rms_norm` (§11.7) est **conservée** : la changer casserait
  la parité CUDA ↔ CPU sans effet mesurable sur la porte.

**Effet sur le statut M10.** Tant que D1/D2 ne sont pas actés, **M10 reste non
validé** : trois tests `golden_cpu` et la porte matérielle contre la dorée sont
rouges, et aucun relâchement de tolérance n'a été commis en attendant.

**Effet sur la campagne GPU.** Aucune campagne RTX 5090 n'est utile à cette
décision : le chemin CUDA est **bit-égal** au chemin CPU mesuré ici (§7.2), donc
une campagne reproduirait exactement le même différentiel contre la dorée. Une
campagne n'a de sens qu'**après** l'adoption de D1/D2, pour revalider la parité
interne des kernels et rejouer la génération réelle sur le SHA final.

