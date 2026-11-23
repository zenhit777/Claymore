# M8 — Provenance des PTX intégrés (`m8_f32.ptx`, `m8_bf16.ptx`)

> **Objet :** manifeste de provenance des deux images PTX chargées par
> `crates/claymore-cuda/src/compute.rs` via `include_str!` et
> `cuModuleLoadData`. Les empreintes ci-dessous ont été recalculées avec
> `sha256sum` sur l'arbre de travail de la branche `feat/m8-tensor-foundation`,
> au HEAD `95460c9`, le **22 novembre 2026** (relevé `21:59:26 +01:00`).

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
