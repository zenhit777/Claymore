# Campagne RTX 5090 groupée M10 — 25 novembre 2026

Exécution du protocole groupé (`m10-rtx5090-grouped-protocol.md`) sur la
branche `feat/m10-diffusiongemma`, après autorisation explicite de
l'utilisateur. **Une seule occupation du GPU**, six étapes séquentielles,
arrêt immédiat à la première erreur.

Verdict : les six étapes sont vertes après correction d'une régression
réelle dans les kernels CUDA M8 (§3).

## 1. Environnement

| élément | valeur |
|---|---|
| poste | WSL2, `/home/sjteb/projets/Claymore` |
| GPU | NVIDIA GeForce RTX 5090, 32 607 Mio, pilote 615.65.07, CUDA UMD 13.4 |
| GPU libre avant session | 1 378 Mio utilisés, **aucun processus compute** (`nvidia-smi --query-compute-apps` vide) |
| GPU libre après session | 1 357 Mio utilisés, aucun processus compute résiduel |
| `libcuda` | `/usr/lib/wsl/lib/libcuda.so` (pilote réel exposé par WSL) |
| compilation PTX | `nvcc` 13.0 (`/usr/local/cuda-13.0/bin/nvcc`), `-arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false` |
| checkpoint | `google/diffusiongemma-26B-A4B-it` rev `cd40195554d3cfcb56f34b9adfb66efc790ed5b6` (`/tmp/opencode/m10/checkpoint`) |
| variables | `RUSTFLAGS="-L native=/usr/lib/wsl/lib"`, `LD_LIBRARY_PATH="/usr/lib/wsl/lib"` |

**Piège d'environnement rencontré.** Le protocole CI construit un stub
`libcuda.so` (`ci/cuda-driver-stub.c`) pour lier sans GPU : ce stub retourne
`CUDA_ERROR_NOT_INITIALIZED` à *tous* les points d'entrée et sert aux
injections de fautes de la CI, **pas** aux tests matériel. L'utiliser ici
donne deux échecs distincts et instructifs : `unable to find library -lcuda`
sans le stub, puis `cuInit code: 3` avec le stub. Le poste expose le pilote
réel ; il faut donc lier `/usr/lib/wsl/lib`, comme le fait le guide de
reproduction `docs/m10-diffusiongemma.md`.

## 2. Résultats

| # | étape | attendu | obtenu | verdict |
|---|---|---|---|---|
| 1 | `hardware_cuda_driver` | 5/5 | 5/5 | ✅ |
| 2 | `hardware_cuda_memory` | 17/17 | 17/17 | ✅ |
| 3 | `hardware_cuda_tensor` | 19/19 | 19/19 | ✅ |
| 4 | `claymore-memory --lib --features cuda-driver -- --ignored` | 13/13 | 13/13 | ✅ |
| 5 | `hardware_m10_primitives` (release) | 21/21 | 21/21 | ✅ |
| 6 | `verify_checkpoint` | identique au log du 2026-11-24 | identique | ✅ |

Journaux bruts horodatés : `m10-rtx5090-grouped-<n>-2026-11-25_16-34-38-*.log.txt`
(étape 4 réécrite à 16:40:50, voir §3.3).

Détails marquants :

- Étape 3 : **19/19**, dont `hw_m8_03_softmax_is_stable_for_large_logits` à
  `max_abs_diff=0e0` sur les quatre formes (contre `2,98e-8` avant le
  correctif §3).
- Étape 5 : les 21 primitives M10 sont à `max_abs=0e0` / `max_ulp=0`,
  y compris le sous-graphe décodeur réel (`hw_m10_09`, pic VRAM 26 607 616 o)
  et la répétition bit-identique de 5 lancements d'attention.
- Étape 6 : `summary tensors=1047 bytes=51647562456 shards=11
  coverage=complete contiguous=true dtype=BF16`, et **chaque**
  `payload_sha256` par shard est identique au journal de référence
  `m10-checkpoint-full-verification.log`.

Durée d'occupation du GPU : ~3 min 45 s (16:34:38 → 16:38:23), dont
~2 min 50 s de calcul effectif ; l'étape 6 est du hachage CPU. Pic VRAM
observé : 26 607 616 o (25,4 Mio) pendant `hw_m10_09`.

## 3. Régression corrigée : course lecture-écriture dans le softmax F32

### 3.1 Symptôme

L'étape 3 échouait sur `hw_m8_03_softmax_is_stable_for_large_logits` :

```text
softmax shape=[8, 257]: index 518 expected 0.000029715795,
got 0.000000040995978, |diff|=0.000029674798, tol=0.0000010002972
```

Les trois autres formes passaient ; seule la ligne 2 de `[8, 257]` divergeait,
avec un rapport `expected / got = 724,85`.

### 3.2 Diagnostic

Ce n'est pas une dérive d'ULP : le rapport est quantifié par la somme `S` de
la ligne (`S = 2,9971445`) :

| hypothèse | valeur prédite | observation |
|---|---|---|
| `exp / S` | 2,9715795e-5 | CPU (attendu) |
| `exp / S⁷` | **4,099598e-8** | **GPU (observé)** |

`exp / S⁷` correspond à `expected / got = S⁶ = 724,85` : la probabilité avait
été **divisée sept fois** au lieu d'une seule.

Cause : `claymore_softmax_row` (`claymore_math.cuh`) est une fonction à
**contrat mono-thread** — elle écrit les exponentielles dans `out` puis les
**relit** pour les diviser par la somme. Le kernel autonome
`claymore_softmax_last_dim_f32` (`m8_f32.cu`) l'appelait **sans garde
`threadIdx.x == 0`**, alors qu'il est lancé avec 256 threads
(`ELEMENTWISE_BLOCK = 256`) : les 256 threads de la ligne se disputent `out`,
ce qui produit la division multiple observée.

Les deux appelants jumeaux n'étaient pas exposés :

- `claymore_attention_f32` / `_bf16` (`m10.cu`) appellent la fonction sous
  `if (tid == 0)` — la chaîne M10 n'a jamais été affectée ;
- `claymore_softmax_row_bf16` recalcule l'exponentielle au lieu de relire
  `out`, donc le kernel BF16 était insensible à la course.

Le défaut était donc **dormant pour M10** mais réel pour le backend tensor
CUDA, seul chemin qui atteint le kernel autonome (`claymore-memory/src/cuda.rs`,
dispatch `TensorOp::SoftmaxLastDim`).

Éléments d'élimination : le port mathématique `claymore_expf` est **exact**
face à glibc (écart `0e0` sur les 8 lignes de la ligne fautive, vérifié sur
CPU sans GPU) ; le PTX embarqué était **identique** à une régénération nvcc ;
`m10.ptx` et `claymore_math.cuh` sont inchangés par la correction.

### 3.3 Correctif

```cuda
// m8_f32.cu — claymore_softmax_last_dim_f32
if (threadIdx.x == 0) {
  claymore_softmax_row(input + row * width, out + row * width, width);
}
```

Le même garde a été appliqué au jumeau BF16
(`claymore_softmax_last_dim_bf16`), qui portait le même contrat mono-thread
pour les mêmes raisons de robustesse. Aucun autre appelant n'est modifié :
les deux chemins d'attention gardaient déjà leur garde.

Effets :

- `m8_f32.ptx` et `m8_bf16.ptx` régénérés, **reproductibles bit à bit** par
  deux compilations nvcc successives ;
- manifeste de provenance `tests/kernel_provenance.rs` mis à jour dans le
  même changement (4 SHA-256) ;
- `hw_m8_03_softmax_is_stable_for_large_logits` passe désormais à
  `max_abs_diff=0e0` sur les quatre formes (strictement mieux que le
  `2,98e-8` du 2026-11-22) ;
- la chaîne M10 est **inchangée** : `m10.ptx` et `claymore_math.cuh` n'ont pas
  bougé, et l'étape 5 rejoue 21/21 primitives à `max_abs=0e0`.

### 3.4 Erreur de protocole corrigée

La première exécution de l'étape 4 a rendu `0 passed; 1 ignored; 64 filtered
out` : le filtre `hardware -- --ignored` comportait deux `--`, le second
neutralisant `--ignored` (qui n'était plus interprété comme option du
harnais). La commande correcte est `cargo test -p claymore-memory --features
cuda-driver --lib --locked -- --ignored --test-threads=1 --nocapture`, qui
exécute bien les 13 tests attendus. Le journal de l'étape 4 a été réécrit
avec l'exécution correcte (16:40:50, 13/13).

## 4. Validation CPU après le correctif

| étape | résultat |
|---|---|
| `cargo fmt --all` | rc=0 |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | rc=0 |
| `cargo test --workspace --all-targets --locked` | rc=0 |
| `cargo +1.86 test --workspace --all-targets --locked` | rc=0 |
| `cargo test -p claymore-model-gemma --test kernel_provenance` | 2/2 |
| `cargo test -p claymore-model-gemma --test kernel_order_parity` | 7/7 (2 ignorés) |

## 5. Hors périmètre

Inchangé : génération complète multi-étapes sur matériel, capture de graphes
CUDA complète du modèle, comparaisons hardware-CPU au-delà des primitives et
du sous-graphe décodeur (`hw_m10_09`). Ces extensions nécessitent chacune
une décision d'architecte et une autorisation GPU distincte.
