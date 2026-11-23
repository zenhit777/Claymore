# Validation matérielle M8 — NVIDIA RTX 5090

- **Date de campagne :** 2026-11-22, 19:19–19:21 (Europe/Paris, UTC+01:00)
- **Dépôt / PR :** `zenhit777/Claymore`, PR #24, branche `feat/m8-tensor-foundation`
- **SHA de code testé :** `f0b034fde0fd8ee11f82e70bdded7d5d8d7171e4`
- **Commit de code :** `feat(m8): native tensor dispatch and CUDA F32 kernels`
- **Rapport et logs :** commits documentaires postérieurs au SHA de code ; aucun fichier source n'a été modifié après le début de la campagne.
- **Périmètre :** dispatch tensoriel, kernels CUDA F32 (add, matmul, softmax stable, RMSNorm), intégration mémoire/scheduler et events réels. Aucun modèle, aucune boucle de génération et aucune mesure de performance modèle ne sont revendiqués.
- **Statut :** **PASS pour le périmètre défini** — 35/35 tests matériels réussis lors de chacune des deux campagnes (70/70 exécutions), 0 échec, 0 test désactivé ou modifié pour la campagne.

> Ce rapport est spécifique au SHA `f0b034fde0fd8ee11f82e70bdded7d5d8d7171e4`. Il ne remplace pas les rapports M4/M5, M7 et M7.1, qui portent sur d'autres SHA.

---

## 1. Traçabilité

Commandes de préparation et d'état :

```bash
git fetch --all --prune
git checkout -B feat/m8-tensor-foundation origin/feat/m8-tensor-foundation
git rev-parse HEAD
git status --short --branch
```

État observé au début de la campagne :

```text
f0b034fde0fd8ee11f82e70bdded7d5d8d7171e4
## feat/m8-tensor-foundation...origin/feat/m8-tensor-foundation [ahead 1]
```

- Le code testé est exactement le SHA ci-dessus ; la branche locale est en avance d'un commit sur `origin` (les commits sont poussés après la campagne).
- L'arbre de travail était propre ; aucun `#[ignore]` n'a été retiré, aucun test filtré ou assoupli.
- Le stub `ci/cuda-driver-stub.c` n'a été utilisé que pour les contrôles logiciels sans GPU décrits en §4, jamais pour la campagne matérielle. La campagne matérielle utilise le vrai driver NVIDIA via `/usr/lib/wsl/lib`.
- La PR #24 n'a pas été fusionnée.

---

## 2. Environnement

### 2.1 GPU et pilote

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell |
| Compute capability | `12.0` |
| VRAM (`nvidia-smi`) | 32 607 MiB |
| VRAM totale rapportée par le driver | 34 162 016 256 octets |
| Mémoire libre rapportée par le driver pendant les tests | 32 429 309 952 octets |
| NVIDIA-SMI | `615.65.07` |
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| Bus-Id | `00000000:01:00.0` |
| MIG / ECC | N/A / N/A |

Sorties brutes : `docs/audits/m8-00-environment.log.txt`.

### 2.2 Hôte et chaîne d'outils

| Élément | Valeur observée |
|---|---|
| Plateforme | WSL2, noyau `6.18.58.1-microsoft-standard-WSL2` |
| Mémoire hôte | 112 GiB (109 GiB disponibles) |
| CUDA toolkit | `Cuda compilation tools, release 13.0, V13.0.48` |
| PTX intégré | `.version 9.0`, `.target sm_120` |
| Rust stable | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Rust MSRV | `rustc 1.86.0 (05f9846f8 2025-03-31)` |
| Cargo stable | `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| Bibliothèque driver WSL2 | `/usr/lib/wsl/lib/libcuda.so`, `libcuda.so.1` |

---

## 3. Commandes exécutées

### 3.1 Environment

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap,driver_version --format=csv
nvcc --version
rustc --version
cargo --version
rustc +1.86 --version
uname -a
free -h
```

### 3.2 Contrôles logiciels obligatoires

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 clippy --workspace --all-targets -- -D warnings
```

Résultat : toutes les commandes sont sorties avec le code 0. 115 tests unitaires workspace réussis sous Rust stable et sous Rust 1.86. Les avertissements Cargo `unused dependency` / `unused field repository` sont préexistants et ne sont pas des erreurs `-D warnings` du compilateur.

### 3.3 Feature CUDA sans matériel requis (CI locale)

```bash
cc -shared -fPIC ci/cuda-driver-stub.c -o /tmp/claymore-cuda-stub/libcuda.so
RUSTFLAGS="-L native=/tmp/claymore-cuda-stub" LD_LIBRARY_PATH=/tmp/claymore-cuda-stub \
  cargo test -p claymore-memory --features cuda-driver --lib --locked
RUSTFLAGS="-L native=/tmp/claymore-cuda-stub" LD_LIBRARY_PATH=/tmp/claymore-cuda-stub \
  cargo test -p claymore-cuda --features cuda-driver --lib --locked
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo check -p claymore-memory --features cuda-driver --all-targets --locked
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo check -p claymore-cli --features cuda-driver --locked
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo clippy -p claymore-cuda -p claymore-memory -p claymore-cli \
    --features cuda-driver --all-targets --locked -- -D warnings
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo +1.86 check -p claymore-cuda --features cuda-driver --all-targets --locked
```

Résultat : 49 tests réussis et 2 ignorés côté `claymore-memory` avec le stub, 7 réussis et 3 ignorés côté `claymore-cuda`, tous les `check`/Clippy/MSRV en code 0. `cuda_tensor::tests::cuda_capabilities_declare_only_implemented_f32_vram_ops` vérifie sans GPU que le backend CUDA ne déclare que F32/VRAM/contigu.

### 3.4 Campagne matérielle RTX 5090 (commande exacte du dépôt)

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --workspace --features cuda-driver --locked -- --ignored --test-threads=1 --nocapture
```

Découverte :

```bash
cargo test --workspace --features cuda-driver --locked -- --ignored --list
```

La commande a été exécutée deux fois de suite, sans modification du code entre les deux.

---

## 4. Résultats des tests matériels

Découverte : **35 tests ignorés** répartis ainsi :

| Cible | Nombre |
|---|---|
| `claymore-cuda` unit tests (driver) | 3 |
| `hardware_cuda_driver.rs` | 5 |
| `claymore-memory` unit tests (manager) | 2 |
| `hardware_cuda_memory.rs` (M4–M7.1) | 17 |
| `hardware_cuda_tensor.rs` (M8) | 8 |
| **Total** | **35** |

### 4.1 Campagne M8 (8 tests, exécutés deux fois)

| Test | Périmètre vérifié | Run 1 | Run 2 |
|---|---|---|---|
| `hw_m8_01_add_matches_cpu_for_multiple_shapes_and_signs` | add F32 ; shapes `[1]`, `[7]`, `[2,3]`, `[64,257]` ; valeurs négatives/positives, amplitudes jusqu'à ±1 000 | PASS | PASS |
| `hw_m8_02_matmul_matches_cpu_for_rank2_shapes` | matmul row-major ; `2x3x2`, `16x64x32`, `129x17x65` | PASS | PASS |
| `hw_m8_03_softmax_is_stable_for_large_logits` | softmax dernier axe ; `[1,3]`, `[8,257]`, `[1,1024]`, `[2,3,64]` ; logits `±10 000` ; somme par ligne ≈ 1 | PASS | PASS |
| `hw_m8_04_rms_norm_matches_cpu_with_explicit_scale` | RMSNorm ; `[4]`, `[6,64]`, `[2,1,129]` ; epsilon `0`, `1e-5`, `1e-3` ; scale explicite | PASS | PASS |
| `hw_m8_05_dispatch_refusals_and_typed_errors` | backend explicite absent, refus CPU sur VRAM, explication `SELECT`/`REFUSE`, BF16 refusé, shape mismatch, layout stridé refusé | PASS | PASS |
| `hw_m8_06_allocation_accounting_lifetime_and_budget_guards` | comptabilité VRAM avant/pendant/après, libération, snapshot driver, budget serré 8 KiB refusant 400 Ko | PASS | PASS |
| `hw_m8_07_kernel_event_guards_and_explicit_wait` | lancement `launch_tensor_op`, event actif, gardes `IN_FLIGHT`, `wait_for_transfer`, aucune libération avant complétion | PASS | PASS |
| `hw_m8_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler` | `Scheduler` + `QueueRunner` sur file `Compute`, dépendances, tenseurs possédés par l'exécuteur, résultat disponible après event | PASS | PASS |

Durée cumulée de la cible `hardware_cuda_tensor` : 1,33 s (run 1) et 1,32 s (run 2).

### 4.2 Campagnes antérieures rejouées (27 tests)

Les 27 tests matériels préexistants ont également été rejoués : 3/3 driver CUDA, 5/5 intégration driver, 2/2 manager, 17/17 mémoire/transfert/éviction. Aucun échec. Durée de la cible `hardware_cuda_memory` : 26,92 s (run 1) et 26,91 s (run 2). Ces tests couvrent notamment la pression mémoire (allocations 64 MiB, 128 cycles H2D/D2H, éviction en vol, alternances transfert/éviction) et le chemin `CudaTransferExecutor` piloté par le scheduler.

**Total : 35/35 par campagne, 70/70 sur les deux campagnes.**

---

## 5. Tolérances et écarts CPU/CUDA observés

Toutes les comparaisons utilisent `|attendu - obtenu| <= atol + rtol * |attendu|`, avec en plus une exigence de finitude pour chaque sortie.

| Opération | atol | rtol | Justification | Écart maximal observé (run 1 = run 2) |
|---|---:|---:|---|---|
| add | 1e-5 | 1e-6 | même opération IEEE-754 F32, aucun FMA ; différence attendue nulle | **0e0** sur toutes les shapes |
| matmul | 1e-3 | 2e-5 | accumulation FMA sur `K` termes dans le même ordre que la référence CPU ; amplitudes jusqu'à `~1e3` | **0e0** sur les trois shapes |
| softmax | 1e-6 | 1e-5 | sorties dans `[0,1]` ; stabilité par soustraction du max puis réduction en arbre | **2,98e-8** (`[8,257]`) ; `4,44e-16` (`[2,3,64]`) ; `0` ailleurs |
| rms_norm | 1e-5 | 2e-5 | réduction en arbre de `sum(x²)` vs réduction séquentielle CPU, puis `sqrt`/division | **4,77e-7** (`[2,1,129]`) ; `3,58e-7` (`[6,64]`) ; `0` (`[4]`) |

Aucune tolérance n'a été élargie après exécution : les seuils sont ceux documentés dans `docs/m8-architecture.md` et codés dans les tests matériels. Les écarts observés sont très inférieurs aux seuils (facteur ≥ 20 pour RMSNorm, ≥ 30 pour softmax).

Stabilité numérique : les cas `±10 000` produisent des probabilités finies et des sommes par ligne à `1e-4` près, sans débordement ni `NaN`.

---

## 6. Mesures mémoire et durée de vie

- VRAM totale driver : 34 162 016 256 octets ; libre : 32 429 309 952 octets au moment des tests.
- `nvidia-smi` : 32 607 MiB au total, 30 856 MiB libres au relevé de 19:21.
- Comptabilité du policy dans `hw_m8_06` : 0 octet avant, 32 768 octets après deux tenseurs de 4 096 F32, 49 152 octets au pic avec la sortie, retour à 0 après libération des trois tenseurs. La libération backend est confirmée avant le retrait comptable.
- Scénario de pression : budget VRAM dur de 8 KiB, demande de 400 000 octets refusée (`MemoryError::RequestTooLarge` via `CudaTensorError::Memory`) avant toute allocation physique.
- Aucune fuite VRAM observable après les campagnes : `nvidia-smi` après exécution affiche 1 332 MiB utilisés, soit le même niveau qu'au début (session Xwayland du poste).
- `hw_m8_07` confirme que les trois allocations d'une opération restent protégées par un event réel jusqu'à `wait_for_transfer` ; le token `69` référence bien les trois identifiants de policy.
- Le backend attend l'event de son propre stream de calcul ; il n'appelle aucun `cuCtxSynchronize` global. Une seule file de calcul et une seule file de transfert sont créées, conformément au profil matériel.

---

## 7. Échecs rencontrés et résolus avant le SHA testé

Les problèmes suivants ont été détectés pendant le développement puis corrigés **avant** le SHA `f0b034f` ; aucun n'est présent dans la campagne finale :

1. `CString::new` appliqué à un nom de noyau déjà terminé par `\\0` provoquait `InvalidLaunchConfiguration("kernel name has NUL byte")` à l'ouverture du module. Résolution : `CStr::from_bytes_with_nul`.
2. Le kernel softmax utilisait l'indice de pointeur `2` pour la sortie alors qu'il n'a qu'une entrée ; panic `index out of bounds`. Résolution : indice de sortie calculé comme `pointers.len() - 1`, identique pour les quatre opérations.
3. Un test d'event utilisait des ticks codés en dur, divergents de l'horloge logique partagée après `download_f32` (`NonMonotonicTick`). Résolution : exposition de `CudaTensorMemory::tick` et utilisation exclusive de cette horloge.
4. Clippy `-D warnings` : `too_many_arguments` sur deux helpers de lancement (autorisation locale documentée) et `unnecessary_get_then_check` dans le manager. Résolution : `contains_key` et `#[allow]` ciblé.
5. Clippy MSRV 1.86 : `needless_lifetimes` préexistant dans `claymore-runtime::ExecutionContext`. Résolution : elision du lifetime (`impl ExecutionContext<'_>`), sans changement fonctionnel.

Aucun échec matériel n'a été observé lors des deux campagnes finales.

---

## 8. Logs bruts et empreintes

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `docs/audits/m8-00-environment.log.txt` | GPU, driver, toolkit, Rust, hôte | `f3f252078b7890896bf682684f107ef2eaaa9fce1915056abfdd309b09ee5ade` |
| `docs/audits/m8-01-software-checks.log.txt` | fmt, tests workspace, Clippy stable et 1.86 | `4c3f64be52c67aae0987733ea9cac8559fb2eaf19549aa396d66c9e55bdd1da6` |
| `docs/audits/m8-02-cuda-feature-ci.log.txt` | stub sans GPU, checks/Clippy feature CUDA, MSRV | `a79e8af789d52ae74acbb9dfa6344ebdb4f30ab64f6d958b811b40737d43c5ea` |
| `docs/audits/m8-03-hardware-run1.log.txt` | découverte + campagne matérielle 1 | `e8ef2311346b6a3f3c50aaa77f03cca3196be74349120e1897c970d848a6a686` |
| `docs/audits/m8-04-hardware-run2.log.txt` | campagne matérielle 2 + `nvidia-smi` final | `919dc4d083a03cb668fc4ba5abc8fb8a5f80c53a7389133a7f61e7002d406960` |

Artefacts de kernel :

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m8_f32.cu` | `5d20ae0ed15affb35ef1c63ef33fc391b094490d6e3d1d499f6891eabbfb33fc` |
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |

Le PTX a été généré sur cette machine avec :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
  crates/claymore-cuda/kernels/m8_f32.cu
```

---

## 9. Limites restantes

- **Précisions :** seules les opérations F32 sont implémentées et validées. BF16 et FP16 restent non supportés malgré leur présence dans `DType` ; aucune référence CPU ni règle numérique n'existe pour eux.
- **Opérations :** jeu limité à add, matmul rank-2, softmax dernier axe et RMSNorm avec scale explicite. Aucune attention, normalisation autre, activation, réduction, transpose ou opération sur vues stridées. L'inspection des graphes DiffusionGemma/Clover reste à faire pour sélectionner la suite.
- **Kernels :** matmul est une implémentation simple (un thread par élément de sortie, boucle sur `K`), sans tiling partagé ni Tensor Core ; softmax et RMSNorm utilisent un bloc par ligne. Aucune revendication de performance ni comparaison avec cuBLAS/cuDNN n'est faite ici (M12).
- **Asynchronisme :** le backend attend l'event CUDA de chaque opération avant de rendre la main. Cela garantit la sûreté des durées de vie mais n'offre pas de recouvrement transfert/calcul ; l'overlap est un objectif ultérieur.
- **Architecture PTX :** cible `sm_120` uniquement, avec JIT driver (PTX 9.0 / CUDA 13+). Aucune compatibilité avec des GPU plus anciens n'est testée ou revendiquée.
- **Injection de fautes :** l'injection de fautes existe pour les transferts M4/M5/M7.1, pas pour l'échec de lancement ou d'enregistrement d'event d'un kernel tensoriel. Le chemin d'erreur est fail-closed par construction (gardes maintenus), mais non exercé par faute injectée.
- **Outils :** aucun Compute Sanitizer, profiler NVIDIA ni benchmark de performance soutenue n'a été exécuté.
- **Contexte :** GPU partagé avec le compositeur graphique WSL2 (Xwayland, ~1,3 GiB résidents) ; les mesures de mémoire libre ne représentent pas un environnement dédié.
- **Runtime :** `ScheduledRuntime` réserve toujours sa file `Compute` aux requêtes modèles ; l'exécuteur tensoriel M8 utilise un scheduler distinct sur le thread propriétaire. L'intégration complète dans le cycle de requête interviendra avec l'orchestration M9.

---

## 10. Conclusion

Les critères matériels de M8 sont remplis pour le périmètre F32 défini : les quatre kernels CUDA sont réellement exécutés sur la RTX 5090 (`sm_120`), comparés à la référence CPU dans les tolérances documentées, avec allocations budgetées, gardes d'éviction, events réels et scheduler sans propriété des tenseurs. Les contrôles logiciels, la feature CUDA sans GPU, Clippy et le MSRV 1.86 sont verts au SHA testé. La PR #24 reste ouverte et n'est pas fusionnée.
