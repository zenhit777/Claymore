# M8 — Provenance des PTX intégrés (`m8_f32.ptx`, `m8_bf16.ptx`)

> **Objet :** manifeste de provenance des deux images PTX chargées par
> `crates/claymore-cuda/src/compute.rs` via `include_str!` et
> `cuModuleLoadData`. Les empreintes ci-dessous ont été recalculées avec
> `sha256sum` sur l'arbre de travail de la branche `feat/m8-tensor-foundation`,
> au HEAD `95460c9`, le **22 novembre 2026** (relevé `21:59:26 +01:00`).
>
> ⚠ **Révisé le 25 novembre 2026** à la suite de la correction de parité
> numérique CUDA ↔ CPU (voir
> [m10-cuda-cpu-parity-fix.md](m10-cuda-cpu-parity-fix.md)). Les sections 2 à 4
> décrivent l'état **historique** au HEAD `95460c9` du 22 novembre 2026 ; elles
> restent exactes pour ce commit et pour les campagnes M8 qui y sont référencées,
> mais ne s'appliquent plus à l'arbre courant. La **section 6** donne la
> commande et les empreintes en vigueur.

## 1. Chaîne d'outils observée

| Élément | Valeur réellement observée |
|---|---|
| Date du relevé | 2026-11-22 (21:59:26 +01:00) |
| Dépôt / branche / HEAD | `zenhit777/Claymore`, `feat/m8-tensor-foundation`, `95460c9` |
| `nvcc --version` | `nvcc: NVIDIA (R) Cuda compiler driver`, Copyright 2005-2025 NVIDIA Corporation, built Wed Jul 16 07:30:01 PM PDT 2025 |
| Release CUDA | **release 13.0, V13.0.48** |
| Build CUDA | `cuda_13.0.r13.0/compiler.36260728_0` |
| Binaire `nvcc` | `/usr/local/cuda-13.0/bin/nvcc` |
| En-tête des PTX | `.version 9.0`, `.target sm_120`, `.address_size 64`, `Compiler Build ID: CL-36260728`, `Based on NVVM 20.0.0` |
| Rust stable observé | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| MSRV (rapports M8) | `rustc 1.86.0 (05f9846f8 2025-03-31)` |
| GPU / driver (`nvidia-smi`) | NVIDIA GeForce RTX 5090, driver `616.64` ; compute capability `12.0` (`sm_120`) |

Le pilote CUDA WSL2 et les versions UMD/KMD détaillés sont consignés dans
`docs/audits/m8-bf16-rtx5090-hardware-validation.md` (KMD `616.64`, UMD `13.4`) ;
le présent manifeste ne porte que sur la génération PTX.

## 2. Artefacts et empreintes SHA-256

`compute.rs` intègre exactement deux images PTX :

```rust
const EMBEDDED_PTX: &str = include_str!("kernels/m8_f32.ptx");      // ligne 22
const EMBEDDED_PTX_BF16: &str = include_str!("kernels/m8_bf16.ptx"); // ligne 23
```

| Module PTX intégré | Source CUDA de vérité | SHA-256 du `.cu` | SHA-256 du `.ptx` |
|---|---|---|---|
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `crates/claymore-cuda/kernels/m8_f32.cu` | `5d20ae0ed15affb35ef1c63ef33fc391b094490d6e3d1d499f6891eabbfb33fc` | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |
| `crates/claymore-cuda/src/kernels/m8_bf16.ptx` | `crates/claymore-cuda/kernels/m8_bf16.cu` | `a276fe54e34b54903591f18f2a777fdf7cae1f11bd64498b47df54a96273dc41` | `886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153` |

Ces quatre valeurs sont identiques à celles déjà versionnées dans
`docs/audits/m8-rtx5090-hardware-validation.md`,
`docs/audits/m8-bf16-rtx5090-hardware-validation.md` et
`docs/audits/m8-bf16-00-environment.log.txt`. Les fichiers source et PTX étaient
propres (`git status --short` sans entrée pour ces chemins) au moment du relevé.

## 3. Commande de génération

Aucun script, `Makefile` ou automate de régénération n'existe dans le dépôt.
La commande exacte est celle documentée en en-tête de chaque `.cu` et dans
`docs/m8-architecture.md` (§« Stratégie PTX et compatibilité Blackwell ») :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
  crates/claymore-cuda/kernels/m8_f32.cu

nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m8_bf16.ptx \
  crates/claymore-cuda/kernels/m8_bf16.cu
```

`-arch=compute_120` est la forme acceptée par nvcc pour la cible PTX
`sm_120` déclarée dans le fichier généré ; la forme courte équivalente est
`nvcc -arch=sm_120 -ptx -o <sortie> <source>`. Aucun drapeau
`--use_fast_math`, `-ftz=true` ou `-ffast-math` n'est utilisé ; la politique
numérique qui en résulte est détaillée dans
[m8-cpu-cuda-numerics.md](m8-cpu-cuda-numerics.md).

Le build Rust n'invoque jamais `nvcc` : les PTX sont versionnés et chargés au
runtime par la Driver API. La campagne BF16 a régénéré les deux PTX dans
`/tmp` avec la même commande puis vérifié `diff -q` contre les fichiers
versionnés (les deux `diff` sont sortis en code 0 ; voir
`docs/audits/m8-bf16-02-cuda-feature-ci.log.txt`, section
« PTX regeneration coherence »).

## 4. Vérification reproductible

Depuis la racine du dépôt :

```bash
sha256sum -c <<'EOF'
5d20ae0ed15affb35ef1c63ef33fc391b094490d6e3d1d499f6891eabbfb33fc  crates/claymore-cuda/kernels/m8_f32.cu
a276fe54e34b54903591f18f2a777fdf7cae1f11bd64498b47df54a96273dc41  crates/claymore-cuda/kernels/m8_bf16.cu
e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde  crates/claymore-cuda/src/kernels/m8_f32.ptx
886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153  crates/claymore-cuda/src/kernels/m8_bf16.ptx
EOF
```

Les tests unitaires de `compute.rs` complètent ce contrôle : ils exigent
`.target sm_120`, la présence des huit symboles `claymore_*`, et, pour le module
BF16, `cvt.rn.bf16.f32` et `fma.rn.f32` (ainsi que l'absence de `.target sm_90`).

## 5. Limites de ce manifeste

- Les empreintes sont celles du contenu des fichiers à la date relevée ; toute
  modification d'un `.cu` ou régénération du PTX impose leur mise à jour.
- Aucune signature cryptographique, attestation de build ni comparaison avec un
  PTX produit par une autre version de nvcc n'est revendiquée. CUDA 13.0 /
  V13.0.48 est la seule chaîne observée ici.
- Le JIT du driver compile le PTX à l'ouverture du module ; une incompatibilité
  est signalée par `CudaError::KernelModuleLoad`. La compatibilité matérielle
  est attestée par les campagnes RTX 5090 versionnées, pas par ce manifeste.

## 6. Révision du 25 novembre 2026 (en vigueur)

La correction de la divergence numérique CUDA ↔ CPU de la chaîne M10
([m10-cuda-cpu-parity-fix.md](m10-cuda-cpu-parity-fix.md)) a introduit
`crates/claymore-cuda/kernels/claymore_math.cuh`, un en-tête partagé qui porte
les implémentations `expf`, `expm1f` et `tanhf` conformes à glibc 2.39. Ces
implémentations reposent sur des **multiplications et additions séparées** :
toute contraction FMA implicite en modifierait le résultat bit à bit. La
génération des PTX exige donc `-fmad=false`.

### 6.1 Contexte du relevé

| Élément | Valeur |
|---|---|
| Date du relevé | 2026-11-25 (02:06 UTC) |
| Dépôt / branche / HEAD de base | `zenhit777/Claymore`, `fix/m10-numerics-divergence`, base `9775cda` |
| `nvcc` | `/usr/local/cuda-13.0/bin/nvcc`, **release 13.0, V13.0.48**, `cuda_13.0.r13.0/compiler.36260728_0` |
| GPU de validation | NVIDIA GeForce RTX 5090, driver `616.64`, compute capability `12.0` |

La chaîne d'outils est **inchangée** par rapport à la section 1 : seule la
commande change. Les PTX générés portent le même en-tête (`.version 9.0`,
`.target sm_120`, `.address_size 64`, `CL-36260728`).

### 6.2 Commande en vigueur

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
  -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
  crates/claymore-cuda/kernels/m8_f32.cu

nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
  -o crates/claymore-cuda/src/kernels/m8_bf16.ptx \
  crates/claymore-cuda/kernels/m8_bf16.cu
```

La commande est reprise à l'identique en en-tête de chaque `.cu`. Toujours
aucun `--use_fast_math`, `-ftz=true` ni `-ffast-math`.

`-fmad=false` ne supprime que la contraction **implicite** : les FMA explicites
(`fmaf` dans les sources, correspondant aux `mul_add` de la référence CPU)
demeurent émis en `fma.rn.f32`. Le test unitaire de `compute.rs` qui exige la
présence de `fma.rn.f32` dans le module BF16 passe donc toujours ; le fichier
en contient 15 occurrences, contre 23 pour la version du 22 novembre.

### 6.3 Empreintes SHA-256 en vigueur

> **Révisé le 25 novembre 2026** à la suite de la campagne groupée RTX 5090 :
> le correctif de la course lecture-écriture du softmax F32 (garde
> `threadIdx.x == 0` sur l'appel mono-thread `claymore_softmax_row`) et son
> jumeau BF16 ont modifié `m8_f32.cu`, `m8_bf16.cu` et les deux PTX
> correspondants. `claymore_math.cuh` est **inchangé**. Diagnostic et
> validation matérielle :
> [m10-rtx5090-grouped-campaign-2026-11-25.md](m10-rtx5090-grouped-campaign-2026-11-25.md) §3.

```bash
sha256sum -c <<'EOF'
f65bd84b35ecdcdbcf2bb1c38f8e384991ba1d71e53be299d52bcbc773c19eab  crates/claymore-cuda/kernels/m8_f32.cu
2edac588d887e2f3ba091bd18aa8f02d645aa07993743ba9fd5d5babf2eff948  crates/claymore-cuda/kernels/m8_bf16.cu
55e491b6e7217180c43deb4abcb17ae6ee946a0fe60a6e5f4940dfc8429b9e15  crates/claymore-cuda/src/kernels/m8_f32.ptx
186ab2d533b517bd6922f026037054bb8004edd4872dd8d56541f4c2f88a7c29  crates/claymore-cuda/src/kernels/m8_bf16.ptx
15efcd8a159db1a240355712bee4ee7ade61a1c435360dbde5af9ad40c2471f7  crates/claymore-cuda/kernels/claymore_math.cuh
EOF
```

`claymore_math.cuh` entre dans le manifeste : les deux `.cu` l'incluent, son
contenu fait donc partie de la provenance des PTX.

### 6.4 Vérification de reproductibilité

Chaque PTX versionné a été régénéré dans un fichier temporaire avec la commande
de la section 6.2 puis comparé par `cmp` au fichier versionné : les trois
comparaisons (`m8_f32`, `m8_bf16`, et `m10` pour le troisième module) sortent
en code 0, c'est-à-dire **byte-identiques**. Cette vérification a été exécutée
après le dernier changement des sources (corrections de commentaires), qui ne
modifie pas les PTX produits.

### 6.5 Effet mesuré sur le PTX et sur le résultat

| Fichier | Sites `fma.rn.f32` (22/11, sans `-fmad=false`) | Sites (25/11, avec `-fmad=false`) |
|---|---|---|
| `m8_f32.ptx` | 23 | 15 |
| `m8_bf16.ptx` | 23 | 15 |
| `m10.ptx` | 291 | 214 |

Les sites restants sont les FMA explicites des sources. Sur les sources du
25 novembre, la commande historique (sans `-fmad=false`) produirait 44 sites
dans les modules M8 et 324 dans `m10.ptx`. Cette différence de génération n'a
**pas** modifié le résultat mesuré sur la chaîne M10 : les deux modes donnent
un accord CUDA ↔ CPU à `0,000000 / 0,000000` (runs 3 et 4 de la campagne du
25 novembre). Le mode `-fmad=false` est néanmoins retenu parce qu'il rend la
correspondance CPU ↔ CUDA structurelle plutôt que dépendante des choix de
contraction du compilateur.
