# Validation matérielle M7 — NVIDIA RTX 5090

- **Date de campagne :** 2026-11-22
- **Dépôt :** `zenhit777/Claymore`
- **PR :** #21 — `feat/m7-deterministic-scheduler`
- **Branche :** `feat/m7-deterministic-scheduler`
- **SHA de code testé :** `13f06147deb3da5bb66132bc426ba7216e11eeeb`
- **Commit de code :** `style(scheduler): match rustfmt output`
- **Commit documentaire :** commit distinct et postérieur, contenant uniquement le présent rapport. Son SHA est communiqué dans le compte rendu de livraison et reste visible via `git log -1 -- docs/audits/m7-rtx5090-hardware-validation.md`.
- **Périmètre :** validation matérielle du scheduler déterministe M7, du `CudaTransferExecutor`, du cache d'experts et des contrats mémoire CUDA associés, après intégration des corrections bloquantes B1, B2, B3 et B4. Aucun kernel CUDA, aucun calcul tensoriel et aucune génération de modèle ne sont validés par cette campagne.

> Cette campagne remplace intégralement la campagne précédente. Les résultats associés à l'ancien SHA `2a2e14fe96baa5ec3962212571cbea28d24d6726` ne sont pas réutilisés et ne doivent pas être attribués au SHA `13f06147deb3da5bb66132bc426ba7216e11eeeb`.

---

## 1. Traçabilité et état du dépôt

Commandes de préparation exécutées :

```bash
git fetch origin
git checkout feat/m7-deterministic-scheduler
git pull --ff-only origin feat/m7-deterministic-scheduler
git rev-parse HEAD
git status --short
```

Résultat :

```text
13f06147deb3da5bb66132bc426ba7216e11eeeb
?? Cargo.lock
```

- La branche locale a été fast-forwardée de `c2d4eee` à `13f0614` sans conflit.
- `HEAD` correspond exactement au SHA demandé : `13f06147deb3da5bb66132bc426ba7216e11eeeb`.
- Aucun fichier suivi n'était modifié avant la campagne.
- Un fichier `Cargo.lock` non suivi était déjà présent dans le worktree. Il n'a pas été ajouté au dépôt et n'est pas inclus dans le commit documentaire.
- Aucun fichier source Rust, aucune API publique, aucune configuration projet et aucune dépendance Cargo n'ont été modifiés pendant la campagne.
- Les tests matériels n'ont pas été modifiés : leurs assertions et leur statut `#[ignore]` sont inchangés.
- La PR #21 n'a pas été fusionnée.

Le commit de code testé et le commit documentaire sont distincts :

| Élément | SHA | Contenu |
|---|---|---|
| Code testé | `13f06147deb3da5bb66132bc426ba7216e11eeeb` | Corrections B1/B2/B3 et formatage |
| Rapport | Commit documentaire postérieur | Uniquement `docs/audits/m7-rtx5090-hardware-validation.md` |

---

## 2. Configuration matérielle et logicielle

### 2.1 Matériel

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell |
| UUID | `GPU-7556cae7-89a8-6267-b8b4-4e2edb9981d6` |
| PCI | `00000000:01:00.0` |
| PCI device ID | `0x2B8510DE` |
| Compute capability | `12.0` |
| VRAM (`nvidia-smi`) | 32 607 MiB |
| VRAM (`device_total_bytes`) | 34 162 016 256 octets (31,82 GiB) |
| Mémoire libre au moment de la détection | 32 429 309 952 octets (30,20 GiB) |
| MIG | N/A |
| ECC | N/A |
| Display Active | Enabled (GPU partagé avec le compositeur Windows) |
| Persistence Mode | Enabled |
| PCIe (maximum rapporté) | Gen 5 ×16 |

Sortie de référence :

```text
name, memory.total [MiB], memory.free [MiB], compute_cap, driver_version
NVIDIA GeForce RTX 5090, 32607 MiB, 30850 MiB, 12.0, 616.64
```

### 2.2 Processeur et mémoire hôte

| Élément | Valeur observée |
|---|---|
| CPU | AMD Ryzen 9 7950X3D 16-Core Processor |
| vCPU visibles dans WSL2 | 16 (8 cœurs × 2 threads) |
| RAM installée annoncée | 128 GiB |
| RAM visible dans WSL2 | 112 GiB |
| RAM disponible au début de la campagne | 109 GiB |
| Swap | 16 GiB |

La différence entre 128 GiB installés et 112 GiB visibles est une caractéristique de la configuration WSL2 de cette machine ; elle n'est pas liée à Claymore.

### 2.3 Système, pilote et CUDA

| Élément | Valeur observée |
|---|---|
| Plateforme | WSL2 |
| Noyau | `6.18.58.1-microsoft-standard-WSL2` |
| Distribution | Ubuntu 24.04.5 LTS |
| Accès GPU WSL2 | `/dev/dxg` présent ; pas de `/dev/nvidia*` (attendu sous WSL2) |
| NVIDIA-SMI | `615.65.07` |
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| Bibliothèque Driver | `/usr/lib/wsl/lib/libcuda.so.1` |
| Liaison dynamique | `libcuda.so.1 => /usr/lib/wsl/lib/libcuda.so.1` |
| Toolkit CUDA dans le `PATH` | CUDA 13.0, `V13.0.48` |
| `/usr/local/cuda` | CUDA 13.5.0 |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Cargo | `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| Toolchain | `stable-x86_64-unknown-linux-gnu` |
| MSRV déclarée | Rust 1.86 |
| MSRV installée localement | Non (voir limitations) |
| Profil de test | `dev` (debug), feature `cuda-driver` |

Variables utilisées pour le linkage et l'exécution CUDA :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Aucune configuration système, aucun pilote et aucun toolkit n'ont été modifiés.

---

## 3. Vérification du profil matériel et des budgets RAM

Commande :

```bash
cargo run -p claymore-cli -- profile
```

Sortie observée :

```text
Claymore v0.5.0
profile=rtx5090-7950x3d
gpu=NVIDIA RTX 5090
vram_total_gib=32
vram_soft_budget_bytes=27917287424
vram_hard_ceiling_bytes=30064771072
vram_safety_reserve_gib=4
expert_cache_target_bytes=15032385536
expert_cache_max_bytes=17179869184
host_budget_bytes=85899345920
pinned_host_budget_bytes=8589934592
cpu_workers=10
max_prefetches=2
lookahead=1
cuda_compute_streams=1
transfer_streams=1
scheduler_tick_us=500
cuda_driver_feature=false
status=profile_validated
```

| Contrôle | Valeur attendue | Valeur observée | Statut |
|---|---:|---:|---|
| Budget Claymore total | 88 GiB | 94 489 280 512 octets = 88 GiB | PASS |
| Tier Host | 80 GiB | 85 899 345 920 octets = 80 GiB | PASS |
| Tier PinnedHost | 8 GiB | 8 589 934 592 octets = 8 GiB | PASS |
| Pool weights | 64 GiB | 68 719 476 736 octets = 64 GiB | PASS |
| Staging/state | 16 GiB | 17 179 869 184 octets = 16 GiB | PASS |
| Somme Host + PinnedHost | 88 GiB maximum | 94 489 280 512 octets = 88 GiB | PASS |
| Budget VRAM souple | 26 GiB | 27 917 287 424 octets = 26 GiB | PASS |
| Plafond VRAM | 28 GiB | 30 064 771 072 octets = 28 GiB | PASS |
| Réserve de sécurité VRAM | 4 GiB | 4 GiB | PASS |
| Cache experts — cible | 14 GiB | 15 032 385 536 octets = 14 GiB | PASS |
| Cache experts — maximum | 16 GiB | 17 179 869 184 octets = 16 GiB | PASS |
| Workers CPU | 10 | 10 | PASS |
| Préchargements max | 2 | 2 | PASS |
| Lookahead | 1 | 1 | PASS |
| Streams de calcul | 1 | 1 | PASS |
| Streams de transfert | 1 | 1 | PASS |
| Tick scheduler | 500 µs | 500 µs | PASS |
| Validation du profil | valide | `status=profile_validated` | PASS |

La sortie CLI contient bien les deux valeurs exigées :

```text
host_budget_bytes=85899345920
pinned_host_budget_bytes=8589934592
```

Le tier Host est calculé comme la somme disjointe du pool weights (64 GiB) et de staging/state (16 GiB), soit 80 GiB. Le tier PinnedHost est limité à 8 GiB. La somme des deux enveloppes est exactement de 88 GiB, égale au budget Claymore, et ne le dépasse pas.

### 3.1 Validations des profils invalides

Commande :

```bash
cargo test -p claymore-runtime profile -- --nocapture
```

Résultat : **7 tests PASS, 0 FAIL**.

| Test | Contrat | Résultat |
|---|---|---|
| `reference_rtx5090_profile_loads_and_converts_to_runtime_budgets` | Host = 80 GiB, PinnedHost = 8 GiB | PASS |
| `profile_rejects_pinned_budget_that_overcommits_disjoint_host_envelopes` | PinnedHost = 88 GiB rejeté | PASS |
| `profile_rejects_host_working_set_plus_pinned_over_budget` | Host 80 GiB + PinnedHost 9 GiB = 89 GiB rejeté | PASS |
| `profile_rejects_unsupported_stream_topology` | Topologie de streams invalide rejetée | PASS |
| `profile_rejects_expert_cache_outside_model_working_set` | Cache experts hors working set rejeté | PASS |
| `profile_rejects_working_set_over_soft_vram_budget` | Working set VRAM hors budget souple rejeté | PASS |
| `hardware_profile_supplies_scheduler_configuration` | Configuration scheduler dérivée du profil | PASS |

La validation `HardwareProfile::validate()` vérifie que `weights_pool_target_gib + staging_and_state_gib + pinned_ceiling_gib <= claymore_budget_gib`. Le profil de référence est donc accepté à exactement 88 GiB, tandis qu'un PinnedHost de 9 GiB est rejeté.

---

## 4. Détection du backend CUDA

Commande :

```bash
cargo run -p claymore-cli --features cuda-driver -- device
```

Sortie observée :

```text
device_name=NVIDIA GeForce RTX 5090
device_ordinal=0
device_total_bytes=34162016256
memory_free_bytes=32429309952
memory_total_bytes=34162016256
```

| Contrôle | Résultat | Statut |
|---|---|---|
| `libcuda.so.1` présente et chargeable | `/usr/lib/wsl/lib/libcuda.so.1` | PASS |
| Initialisation du driver CUDA | réussie | PASS |
| Détection du GPU | NVIDIA GeForce RTX 5090, ordinal 0 | PASS |
| Mémoire totale device | 34 162 016 256 octets | PASS |
| Mémoire libre device | 32 429 309 952 octets | PASS |
| Cohérence avec `nvidia-smi` | 32 607 MiB, GPU partagé avec l'affichage | PASS |

Aucune erreur CUDA n'a été observée pendant la détection.

---

## 5. Campagnes de tests matériels M7

### 5.1 Commande exécutée

```bash
cargo test -p claymore-memory \
  --features cuda-driver \
  --test hardware_cuda_memory \
  -- --ignored --test-threads=1 --nocapture
```

Deux campagnes complètes et indépendantes ont été exécutées séquentiellement, sans autre processus de calcul CUDA. Un échantillonnage VRAM `nvidia-smi` à 200 ms a été actif pendant chaque campagne.

Les 17 tests du binaire `hardware_cuda_memory` sont marqués `#[ignore]` dans le dépôt. Ils ont été sélectionnés par `--ignored` ; aucun test n'a été dé-ignoré, modifié ou supprimé. Le flag `--nocapture` ne modifie ni le code ni les assertions ; il affiche les mesures normalement capturées par le harness Rust.

### 5.2 Résultats des deux campagnes

| Campagne | Exécutés | Réussis | Échoués | Ignorés pendant l'exécution | Code retour | Durée murale | Durée harness |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 17 | 0 | 0 | 0 | 27,079 s | 27,05 s |
| 2 | 17 | 17 | 0 | 0 | 0 | 26,942 s | 26,91 s |
| **Total** | **34** | **34** | **0** | **0** | **0** | — | — |

Les 17 tests sont ignorés par défaut ; le champ « ignorés pendant l'exécution » vaut 0 parce que `--ignored` les a tous sélectionnés et exécutés. Aucun test de cette cible n'est resté non exécuté.

### 5.3 Résultat de chaque test

| ID | Test | Contrat vérifié | Résultat | Preuve observée (campagnes 1 / 2) |
|---|---|---|---|---|
| HW-C2-03 | `hw_c2_03_manager_release_semantics` | Libération explicite, comptabilité, erreurs après libération | PASS | `explicit release, accounting and post-release errors verified` |
| HW-C3-01 | `hw_c3_01_transfer_sizes_are_byte_exact` | H2D/D2H byte-exact de 1 B à 64 MiB | PASS | `size=1/37/4096/1048576/67108864 byte_exact=yes` |
| HW-C3-02 | `hw_c3_02_multiple_allocations_and_successive_transfers` | Transferts successifs sur deux paires d'allocations | PASS | `two allocation pairs transferred successively and verified` |
| HW-C4-01 | `hw_c4_01_nonblocking_polling_and_completion` | Polling non bloquant, états pending/completed, attente | PASS | `retired=0/8 before wait; pending and complete states coherent` |
| HW-C4-02 | `hw_c4_02_wait_for_allocation_is_scoped` | `wait_for_allocation` ciblé | PASS | `wait_for_allocation retired only transfers touching its target` |
| HW-C4-03 | `hw_c4_03_tokens_are_manager_scoped` | Tokens liés à leur manager | PASS | `foreign transfer token rejected by another manager` |
| HW-C5-01 | `hw_c5_01_in_flight_allocation_cannot_be_evicted` | Allocation en vol non évincée | PASS | `protected during 0/8 completed polls; evicted after confirmed completion` |
| HW-C5-02 | `hw_c5_02_shared_allocation_stays_protected_by_remaining_transfer` | Protection partagée par plusieurs transferts | PASS | `shared allocation stayed protected until its last event completed` |
| HW-C5-03 | `hw_c5_03_eviction_skips_protected_candidate` | Éviction excluant une allocation protégée | PASS | `eviction skipped the protected allocation and freed the idle candidate` |
| HW-C6-01 | `hw_c6_01_polling_precedes_eviction_planning` | Polling avant planification d'éviction | PASS | `completed event polled before eviction after 139 / 336 queries` |
| HW-C6-02 | `hw_c6_02_equal_priority_eviction_is_deterministic` | Départage déterministe à priorité égale | PASS | `alpha` évincé 12/12 |
| HW-C6-03 | `hw_c6_03_accounting_and_admission_errors` | Comptabilité et erreurs d'admission | PASS | `accounting stayed consistent across release and admission errors` |
| HW-C7-01 | `hw_c7_01_repeated_allocate_transfer_free_cycles` | 128 cycles allocation/transfert/libération | PASS | `128 H2D/D2H cycles verified in 99.117826 ms / 100.658181 ms` |
| HW-C7-02 | `hw_c7_02_tight_polling_is_idempotent` | Polling rapproché et idempotence | PASS | `1000 polls, retired_total=8/8, state stayed coherent` |
| HW-C7-03 | `hw_c7_03_transfer_and_eviction_alternation` | 16 alternances transfert/éviction | PASS | `16 transfer/eviction alternations verified` |
| **M7-TRANSFER** | `hw_m7_cuda_transfer_executor_runs_scheduler_tasks_after_event_completion` | Chaîne de 8 tâches, H2D/D2H réels, attente d'event, libération ordonnée | **PASS** | `H2D=1048576 bytes in 113981 / 129062 ns; D2H=1048576 bytes in 77171 / 77671 ns; event_wait=133662 / 140261 ns` |
| **M7-EXPERTS** | `hw_m7_expert_cache_max_is_enforced_before_device_allocation` | Plafond du cache d'experts et éviction déterministe | **PASS** | `expert cache maximum enforced with deterministic idle eviction` |

**Synthèse : 34/34 exécutions PASS, 0 FAIL, 0 test non exécuté dans cette cible.**

---

## 6. Mesures H2D/D2H, events et compteurs

### 6.1 Mesures des deux campagnes complètes

| Métrique | Campagne 1 | Campagne 2 |
|---|---:|---:|
| Octets H2D | 1 048 576 | 1 048 576 |
| Durée totale H2D | 113 981 ns | 129 062 ns |
| Octets D2H | 1 048 576 | 1 048 576 |
| Durée totale D2H | 77 171 ns | 77 671 ns |
| Attente cumulée des events | 133 662 ns | 140 261 ns |
| Échecs | 0 | 0 |
| Opérations restantes après exécution | 0 | 0 |

### 6.2 Exécutions isolées du test M7

Le test M7 a également été exécuté seul trois fois, avec `--exact --nocapture`, afin de distinguer le démarrage à froid du contexte CUDA :

| Exécution | Durée H2D | Durée D2H | Attente event | Résultat |
|---|---:|---:|---:|---|
| 1 | 937 813 ns | 73 851 ns | 89 891 ns | PASS |
| 2 | 1 079 386 ns | 72 721 ns | 97 572 ns | PASS |
| 3 | 980 969 ns | 74 612 ns | 95 151 ns | PASS |

**Interprétation.** Les durées H2D des exécutions isolées sont plus élevées que celles de la suite complète, car le test isolé est la première utilisation CUDA du processus : l'initialisation du contexte, des pages pinned et du stream est incluse dans le premier transfert. Dans la suite complète, les tests précédents ont déjà initialisé le contexte. Cette différence est une variance de démarrage à froid, pas un échec.

**Important.** `h2d_total_ns` et `d2h_total_ns` sont des durées de bout en bout incluant l'enqueue et l'attente de l'event. `event_wait_ns` est le temps cumulé passé à attendre les events, toutes opérations confondues. Ces valeurs ne doivent pas être converties en débit DMA : la taille transférée est de 1 MiB, trop petite pour une mesure de bande passante, et l'attente d'event n'est pas un débit. Aucune revendication de performance de transfert n'est faite dans ce rapport.

### 6.3 Métriques du `QueueRunner`

Les métriques `QueueRunnerMetrics` ne sont pas imprimées par le test matériel. Une sonde de mesure hors dépôt, dont le binaire a été produit dans `/tmp/opencode/m7-revalidation/metrics-target` et qui utilise exactement la même séquence de huit tâches via l'API publique, a été exécutée trois fois. Elle n'ajoute aucun test au dépôt et ne modifie pas le code.

| Champ `QueueRunnerMetrics` | Exécution 1 | Exécution 2 | Exécution 3 |
|---|---:|---:|---:|
| `kind` | `Transfer` | `Transfer` | `Transfer` |
| `started` | 8 | 8 | 8 |
| `completed` | 8 | 8 | 8 |
| `failed` | 0 | 0 | 0 |
| `idle_polls` | 0 | 0 | 0 |
| `total_execution_ns` | 4 743 508 | 4 591 570 | 4 329 354 |
| `max_execution_ns` | 2 017 659 | 1 877 561 | 1 923 298 |

L'intégrité de l'aller-retour dans la sonde est `true` pour les trois exécutions, `pending_operations = 0`, la VRAM finale est 0 et la mémoire pinned finale est 0.

### 6.4 Métriques `CudaTransferMetrics`

La même sonde a relevé les champs suivants (processus séparé, contexte à froid) :

| Champ `CudaTransferMetrics` | Exécution 1 | Exécution 2 | Exécution 3 |
|---|---:|---:|---:|
| `allocations` | 2 | 2 | 2 |
| `releases` | 2 | 2 | 2 |
| `h2d_transfers` | 1 | 1 | 1 |
| `d2h_transfers` | 1 | 1 | 1 |
| `h2d_bytes` | 1 048 576 | 1 048 576 | 1 048 576 |
| `d2h_bytes` | 1 048 576 | 1 048 576 | 1 048 576 |
| `h2d_total_ns` | 979 989 | 1 035 025 | 875 233 |
| `d2h_total_ns` | 75 861 | 89 181 | 73 341 |
| `pinned_read_bytes` | 1 048 576 | 1 048 576 | 1 048 576 |
| `event_wait_ns` | 96 161 | 104 222 | 93 092 |
| `failed` | 0 | 0 | 0 |
| `total_execution_ns` | 4 735 368 | 4 582 500 | 4 321 984 |
| `max_execution_ns` | 2 015 069 | 1 874 841 | 1 921 018 |

Les valeurs diffèrent de celles de la suite complète pour les raisons de démarrage à froid exposées ci-dessus. Elles ne remplacent pas les mesures officielles de la section 6.1 ; elles complètent la vérification des compteurs.

### 6.5 Métriques `MemoryExecutionMetrics`

`CudaTransferExecutor` expose `CudaTransferMetrics`, pas `MemoryExecutionMetrics`. `MemoryExecutionMetrics` appartient à l'exécuteur mémoire backend-neutre `MemoryTaskExecutor`, qui n'est pas le chemin CUDA utilisé par le test M7.

La vérification de ces métriques a donc été faite par les tests unitaires logiciels :

| Suite | Résultat | Preuve |
|---|---|---|
| `cargo test -p claymore-memory --lib` | **32 PASS, 0 FAIL** | Politiques, admission, éviction, comptabilité |
| `cargo test -p claymore-scheduler` | **17 PASS, 0 FAIL** | Scheduler, dépendances, files, métriques |
| `cargo test -p claymore-runtime profile` | **7 PASS, 0 FAIL** | Profil et budgets Host/PinnedHost |

Les tests ciblés des corrections bloquantes sont également PASS :

| Correction | Test | Résultat |
|---|---|---|
| B1 | `reference_rtx5090_profile_loads_and_converts_to_runtime_budgets` | PASS |
| B1 | `profile_rejects_pinned_budget_that_overcommits_disjoint_host_envelopes` | PASS |
| B1 | `profile_rejects_host_working_set_plus_pinned_over_budget` | PASS |
| B2 | `submission_cancels_tasks_with_already_failed_or_cancelled_dependencies` | PASS |
| B2 | `request_submitted_after_failed_dependency_is_cancelled_and_payload_pruned` | PASS |
| B2 | `request_submitted_after_cancelled_dependency_is_cancelled_and_payload_pruned` | PASS |
| B3 | `adaptive_eviction_prioritizes_coldness_before_reload_cost` | PASS |
| B3 | `adaptive_eviction_treats_unknown_reload_cost_as_zero_only_after_heat_and_recency` | PASS |
| B3 | `adaptive_eviction_prefers_the_lower_reload_cost` | PASS |

### 6.6 Vérification fonctionnelle B1, B2, B3 et scheduler

#### B1 — Budgets Host / PinnedHost

- Le profil de référence expose Host = 80 GiB et PinnedHost = 8 GiB.
- La somme Host + PinnedHost vaut exactement 88 GiB, soit le budget Claymore total.
- `HardwareProfile::validate()` rejette un PinnedHost de 9 GiB, qui porterait la somme à 89 GiB.
- `HardwareProfile::validate()` rejette un PinnedHost de 88 GiB, qui dépasserait largement l'enveloppe.
- Les tests matériels HW-C2-03 et HW-C6-03 vérifient la comptabilité après allocation, libération et erreurs d'admission.
- La sonde M7 montre l'évolution suivante, sans écart de comptabilité :

```text
step=allocate_host    vram_bytes=0        pinned_bytes=1048576
step=allocate_device  vram_bytes=1048576  pinned_bytes=1048576
step=write_host       vram_bytes=1048576  pinned_bytes=1048576
step=h2d              vram_bytes=1048576  pinned_bytes=1048576
step=d2h              vram_bytes=1048576  pinned_bytes=1048576
step=read_host        vram_bytes=1048576  pinned_bytes=1048576
step=release_device   vram_bytes=0        pinned_bytes=1048576
step=release_host     vram_bytes=0        pinned_bytes=0
```

Nuance de conception : `MemoryBudgets` expose deux limites distinctes (`host_bytes`, `pinned_host_bytes`) et ne porte pas de champ « total ». La contrainte de somme est donc validée au niveau du profil matériel, qui construit ces deux limites à partir des enveloppes disjointes weights + staging/state et pinned. Le backend applique ensuite chaque limite séparément.

#### B3 — Éviction Adaptive

Le classement est défini par la clé lexicographique suivante, triée par ordre croissant :

```text
(hit_count, last_used_tick, reload_cost_ns, bytes)
```

- Les candidats sont d'abord filtrés par tier et par `safe_after_tick <= now_tick` ; une allocation en vol n'est jamais candidate.
- La chaleur (`hit_count`) est comparée avant la récence (`last_used_tick`), puis le coût de rechargement, puis la taille.
- Le coût en nanosecondes n'est plus ajouté aux compteurs de hits ni aux ticks logiques.
- Un coût inconnu vaut zéro et n'intervient qu'après la chaleur et la récence, comme départage.
- Les trois tests Adaptive PASS couvrent le coût élevé sur une allocation froide, le coût inconnu et le départage par coût plus faible.
- `confirm_eviction` retire l'allocation et son coût de rechargement ; le test `confirm_eviction_releases_accounted_capacity` vérifie que la capacité comptabilisée revient à zéro.

Limite explicite : aucun chargeur de poids ou d'experts n'alimente automatiquement `record_reload_cost` à ce stade. La politique est vérifiée par l'API et les tests unitaires, mais l'effet d'un coût mesuré en production n'est pas encore exercé de bout en bout.

#### B2 — Propagation des dépendances échouées ou annulées

- Une tâche soumise avec une dépendance déjà `Failed` ou `Cancelled` passe immédiatement à `Cancelled`.
- Un descendant d'une tâche annulée est également annulé.
- Les payloads de requêtes annulées sont éliminés ; `pending_request_count()` revient à 0.
- Les tests scheduler et runtime correspondants sont PASS.

#### Scheduler et transferts

- La chaîne de huit tâches `allocate_host -> allocate_device -> write_host -> h2d -> d2h -> read_host -> release_device -> release_host` est exécutée dans l'ordre.
- `QueueRunner::run_one` n'appelle `scheduler.complete(task_id)` qu'après le retour `Ok(())` de l'exécuteur.
- `CudaTransferExecutor::execute` attend l'event CUDA de chaque H2D/D2H avant de retourner le succès.
- Les mesures `event_wait_ns` sont non nulles dans toutes les campagnes.
- Les allocations en vol restent protégées jusqu'à la confirmation de l'event (HW-C5-01, HW-C5-02).
- Aucune libération prématurée ni fuite de ressource n'a été observée.

---

## 7. Consommation VRAM

La consommation a été échantillonnée toutes les 200 ms avec `nvidia-smi` pendant chaque campagne complète.

| Moment | VRAM utilisée | VRAM libre | Delta vs baseline |
|---|---:|---:|---:|
| Avant campagne 1 | 1 338 MiB | 30 850 MiB | 0 |
| Campagne 1 — premier échantillon | 1 338 MiB | 30 850 MiB | 0 |
| Pic campagne 1 | 2 346 MiB | 29 842 MiB | +1 008 MiB |
| Après campagne 1 (dernier échantillon) | 1 352 MiB | 30 836 MiB | +14 MiB |
| Avant campagne 2 | 1 338 MiB | 30 850 MiB | 0 |
| Campagne 2 — premier échantillon | 1 338 MiB | 30 850 MiB | 0 |
| Pic campagne 2 | 2 346 MiB | 29 842 MiB | +1 008 MiB |
| Après campagne 2 (dernier échantillon) | 1 348 MiB | 30 840 MiB | +10 MiB |
| Après les deux campagnes | 1 339 MiB | 30 849 MiB | +1 MiB |

- Le pic observé est de **2 346 MiB**, soit environ **7,2 %** de la VRAM totale de 32 607 MiB.
- Le delta maximal par rapport à la baseline est d'environ **1 008 MiB**. Il inclut les allocations device des scénarios de pression (512 MiB pour HW-C5-01, 2 × 256 MiB pour HW-C5-03), les buffers de transfert et l'overhead du contexte CUDA.
- La baseline d'environ 1 338 MiB correspond au compositeur Windows/Xwayland, pas à Claymore.
- Aucune saturation volontaire de la VRAM n'a été effectuée. Le pic reste très inférieur au budget souple de 26 GiB et au plafond de 28 GiB.
- L'échantillonnage à 200 ms peut manquer des pics plus courts que l'intervalle ; les valeurs rapportées sont des observations, pas une borne supérieure absolue.
- Après chaque campagne, la VRAM est revenue à la baseline et aucun processus de calcul CUDA n'était présent.

---

## 8. Erreurs et anomalies observées

| Élément | Observation | Statut |
|---|---|---|
| Échec de test | Aucun | PASS |
| Erreur CUDA / pilote | Aucune | PASS |
| Panique ou assertion échouée | Aucune | PASS |
| Fuite d'allocation détectée | Aucune (VRAM/pinned finales à 0 dans les scénarios concernés) | PASS |
| Avertissement Cargo `workspace.package.repository` inutilisé | Préexistant, sans impact | PASS |
| Avertissement Cargo `claymore-cuda` inutilisé dans `claymore-cli` | Préexistant, sans impact | PASS |
| `cuda_driver_feature=false` dans `profile` | Attendu : commande exécutée sans feature CUDA | PASS |
| Variance des durées H2D à froid | Observée et expliquée ; aucun échec | PASS |
| `Cargo.lock` non suivi | Présent avant la campagne ; non ajouté au dépôt | PASS |
| MSRV Rust 1.86 exécutée localement | Toolchain non installée ; le job CI `msrv` reste la référence | NOT RUN |
| `compute-sanitizer` | Invoqué sur HW-C2-03 ; échec d'initialisation WDDM debugger interface / device not supported ; aucune instrumentation mémoire GPU exploitable | NOT RUN (limitation WSL2/WDDM) |
| Kernels CUDA / calcul tensoriel | Hors périmètre M7 ; aucun résultat de calcul GPU n'est revendiqué | NOT RUN |

Aucun défaut produit n'a été découvert pendant cette campagne.

---

## 9. Vérification de la libération des ressources

| Contrôle | Résultat | Statut |
|---|---|---|
| `pending_operations` après le test M7 | 0 | PASS |
| VRAM finale de la sonde M7 | 0 octet | PASS |
| Mémoire pinned finale de la sonde M7 | 0 octet | PASS |
| Comptage final des scénarios de pression | 0 dans HW-C5-03 et HW-C7-03 | PASS |
| Comptage final du cache d'experts | 0 après libération de `expert-new` | PASS |
| VRAM GPU après campagne | Retour à la baseline (1 339 MiB, Xwayland) | PASS |
| Processus CUDA résiduels | Aucun (`nvidia-smi --query-compute-apps` vide) | PASS |

Le `Drop` de `CudaMemoryManager` synchronise le stream de transfert avant de libérer les buffers ; en cas d'échec de synchronisation, il choisit de ne pas libérer les buffers plutôt que de risquer une libération pendant un DMA en vol. Ce comportement a été vérifié par inspection du code sur le SHA testé ; le chemin d'échec de synchronisation n'a pas été injecté sur matériel.

---

## 10. Verdict

### 10.1 Verdict par catégorie

| Catégorie | Verdict | Justification |
|---|---|---|
| **B1 — Budgets RAM** | **PASS** | Host = 80 GiB, PinnedHost = 8 GiB, somme = 88 GiB ; profils invalides rejetés ; comptabilité matérielle cohérente. |
| **B2 — Dépendances** | **PASS** | Propagation immédiate des dépendances `Failed`/`Cancelled`, y compris descendants et payloads. |
| **B3 — Éviction Adaptive** | **PASS** | Clé `(heat, recency, reload_cost, bytes)` ; coût nanoseconde non additif ; coût inconnu traité comme départage. |
| **Scheduler** | **PASS** | Chaîne de 8 tâches exécutée dans l'ordre ; dépendants libérés seulement après retour de l'exécuteur ; `QueueRunnerMetrics` cohérentes (8/8, 0 échec, 0 idle). |
| **Transferts CUDA** | **PASS** | H2D et D2H réels de 1 MiB, intégrité byte-exact, attente d'event non nulle, 0 échec, 0 opération restante. |
| **Sûreté mémoire** | **PASS** | Allocations en vol protégées, transferts partageant une allocation, polling avant éviction, cycles répétés et comptabilité cohérente : tous les tests HW-C4/C5/C6/C7 PASS. |
| **Cache d'experts** | **PASS** | Plafond maximal appliqué avant allocation device, éviction déterministe de l'expert inactif, comptage VRAM cohérent, aucun dépassement. |
| **Pression VRAM** | **PASS** | Pic observé à 2 346 MiB (7,2 % de la VRAM), retour à la baseline, aucune saturation volontaire. |
| **Télémétrie** | **PASS** | `CudaTransferMetrics` et `QueueRunnerMetrics` relevés et cohérents ; `MemoryExecutionMetrics` vérifié séparément sur l'exécuteur backend-neutre (réserve documentée en 10.3). |

### 10.2 Verdict global

**Verdict global : `PASS`.** Validation matérielle M7 réussie sur NVIDIA RTX 5090, avec les limitations documentées ci-dessous.

- 17/17 tests matériels PASS lors de deux campagnes complètes (34/34 exécutions), 0 FAIL.
- Le test M7 principal et le test du cache d'experts sont PASS.
- Les contrats de protection mémoire, d'éviction et de comptabilité sont PASS.
- Les corrections B1, B2 et B3 sont couvertes par des tests ciblés PASS.
- Aucune saturation VRAM, aucune fuite détectée, retour à la baseline après campagne.
- Aucun fichier source, aucune API, aucune configuration et aucune dépendance n'ont été modifiés.
- La PR #21 n'a pas été fusionnée.

### 10.3 Réserves pour la revue de l'architecte

1. **Taille du transfert M7.** Le test principal transfère 1 MiB. Il valide le protocole scheduler/event et l'intégrité, pas le débit DMA à grande échelle. Aucune conclusion de performance ne doit être tirée des durées H2D/D2H.
2. **Cache d'experts à échelle réduite.** Le test utilise des allocations de 1,5 MiB avec une cible de 1 MiB et un maximum de 2 MiB. Il valide le mécanisme d'application du plafond, pas le comportement du cache de 14/16 GiB du profil en charge réelle.
3. **Pas de kernels.** Aucun kernel CUDA, aucun calcul tensoriel et aucune génération de modèle n'ont été exécutés. M7 ne valide pas les jalons suivants.
4. **Pas d'injection de fautes.** Les erreurs réelles de `cuEventQuery`, `cuStreamSynchronize` ou `cuMemFree` n'ont pas été injectées. Les chemins fail-closed restent couverts par inspection et par tests logiciels.
5. **`MemoryExecutionMetrics`.** Cette structure n'est pas émise par `CudaTransferExecutor` ; sa vérification est indirecte via les tests unitaires de `MemoryTaskExecutor`.
6. **Observation de l'event au moment exact de la libération.** Le test M7 ne place pas de sonde entre la confirmation de l'event et `scheduler.complete`. La garantie est structurelle (ordre des appels) et comportementale (chaîne et intégrité), complétée par les tests HW-C4/C5.
7. **MSRV 1.86.** La toolchain 1.86 n'est pas installée localement ; la campagne locale utilise Rust 1.100.0. Le job CI `msrv` avec `dtolnay/rust-toolchain@1.86.0` reste la référence pour la MSRV.
8. **WSL2/WDDM.** GPU partagé avec l'affichage Windows, pas de MIG/ECC. `compute-sanitizer` 2025.3.0.0 a été invoqué sur HW-C2-03, mais n'a pas pu initialiser l'interface debugger WDDM et a déclaré le device non supporté ; aucune instrumentation mémoire GPU n'est donc exploitable. Les résultats sont représentatifs de cette configuration, pas d'un runner Linux natif.
9. **Échantillonnage VRAM.** L'intervalle de 200 ms peut manquer des pics très courts ; les valeurs de pic sont des observations, pas des bornes absolues.
10. **Variance temporelle.** Les durées H2D à froid (environ 0,88–1,08 ms) diffèrent des mesures en contexte chaud (environ 0,11–0,13 ms). Il s'agit d'un effet d'initialisation, pas d'une instabilité du contrat.
11. **Coût de rechargement non alimenté automatiquement.** `record_reload_cost` est une API publique, mais aucun chargeur de poids ou d'experts ne l'appelle encore. La correction B3 est vérifiée par les tests de politique, pas par une mesure de rechargement réelle en production.
12. **Somme des budgets au niveau runtime.** `MemoryBudgets` contient deux limites distinctes et pas de champ total. La contrainte Host + PinnedHost <= 88 GiB est validée dans `HardwareProfile`; le backend applique ensuite chaque limite séparément.

---

## 11. Reproductibilité

```bash
git fetch origin
git checkout feat/m7-deterministic-scheduler
git pull --ff-only origin feat/m7-deterministic-scheduler
git rev-parse HEAD
# doit afficher 13f06147deb3da5bb66132bc426ba7216e11eeeb

export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p claymore-cuda --features cuda-driver
cargo check -p claymore-memory --features cuda-driver
cargo check -p claymore-cli --features cuda-driver

cargo run -p claymore-cli -- profile
cargo run -p claymore-cli --features cuda-driver -- device

cargo test -p claymore-memory \
  --features cuda-driver \
  --test hardware_cuda_memory \
  -- --ignored --test-threads=1 --nocapture

cargo test -p claymore-scheduler
cargo test -p claymore-memory --lib
cargo test -p claymore-runtime profile
```

Les logs complets de la campagne ont été conservés hors dépôt dans `/tmp/opencode/m7-revalidation/` sur la machine d'exécution :

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `fmt.log` | `cargo fmt --check` (sortie vide, code 0) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `test-workspace.log` | `cargo test --workspace --all-targets` | `8567e14bf8d4a62c84e20c3fc397987788c195eddcc15faba9a3a0effc092d40` |
| `clippy.log` | `cargo clippy -D warnings` | `199c361086059db89fcb4705f295ba579a1d63bac2d854491a4e574c359d70bb` |
| `check-cuda.log` | `cargo check -p claymore-cuda` | `666f83f4488545c6b1e0597a982c126f33287a86539af3803295ac19791650c1` |
| `check-memory.log` | `cargo check -p claymore-memory` | `b98eda10c5fd789f53aef885f0977965f7f46080edbe19825507be95ac10c528` |
| `check-cli.log` | `cargo check -p claymore-cli` | `66081d88eda1e571b73d86c571d30d229c0fb3817b4f103445421c7ccb7e1d7a` |
| `profile.log` | `claymore-cli profile` | `d2423814a43bae404cc60d828e2bbc631af3a2ec0a03d5dfb1f3577a8e012996` |
| `device.log` | `claymore-cli device` | `852664a5f7f4481cf694aa191ec1890e0d6100208c40917bff1eaee20def7ea1` |
| `profile-tests.log` | Tests de validation du profil | `9582d17142021020256e256a2b8d950390bfb628828b7939ffcfeac0529e05cf` |
| `b3-adaptive-tests.log` | Tests B3 Adaptive | `3d13ffa1c1b35190adeaf22e8697da94ceee59d56858396422a0c60d14bfe12e` |
| `b2-scheduler-test.log` | Test B2 scheduler | `c4c8087d09033deeb1c5c0f5799cb3e52d4ee0beace8874fd6b2c56461d2a9dd` |
| `b2-runtime-tests.log` | Tests B2 runtime | `e4eb4e09430aeb924c28e593546b09f8517fe2432cdacd5ffd065503adcd8578` |
| `hw-campaign-1.log` | Campagne matérielle 1 | `900955961fdbd17b37ba6021ebb0533706428891ac8379d57dc51b87d25f8ad6` |
| `hw-campaign-2.log` | Campagne matérielle 2 | `6d5bd9e92cde06490a1ce4930d379c8b6d5cf9dc083e967355dc18c5a97c02e1` |
| `vram-campaign-1.csv` | Échantillonnage VRAM campagne 1 (200 ms) | `a38ba555631266b854e4e59edc1673bec4d625b69cea6d10f4bc22ec81d73ea3` |
| `vram-campaign-2.csv` | Échantillonnage VRAM campagne 2 (200 ms) | `f9c9324f58f9d144592dd570f661f09735b2ebffbc3ab0a19e61624248bca398` |
| `metrics-run-1.log` | Sonde métriques M7, exécution 1 | `55bde45357dbb7a64bbc544fb827658ee4760b15aa6f5d29a6c8c63260dd1f1d` |
| `metrics-run-2.log` | Sonde métriques M7, exécution 2 | `d71ca048ab46cd2518a0403677726fef2731bc55e9f06949bc277a7935196421` |
| `metrics-run-3.log` | Sonde métriques M7, exécution 3 | `f12af26de23554b9bd91bcb1d242eb2e2cc53706691089aaf3e77efa2cb876c8` |
| `m7-isolated.log` | Trois exécutions isolées des tests M7 | `6761db8d12e520f54902a5275a93f262b8b0a3169329d836182083fbc62d7020` |
| `compute-sanitizer.log` | Tentative d'instrumentation `compute-sanitizer` sur HW-C2-03 | `87999fee1f68de499c39bd60833e6e59754c68c2daaf5094bf4022ed8c6b6249` |

---

## 12. Conclusion

La campagne a réellement exécuté les 17 tests matériels M7 sur la RTX 5090 avec le pilote `616.64` et CUDA UMD `13.4`, sur le SHA de code `13f06147deb3da5bb66132bc426ba7216e11eeeb`. Les 17 tests ont réussi lors de deux campagnes consécutives, sans échec, sans panique et sans erreur CUDA. Le scheduler, le `CudaTransferExecutor`, le cache d'experts et les contrats de protection mémoire sont validés pour le périmètre M7.

Les corrections bloquantes sont couvertes : B1 par la répartition Host 80 GiB / PinnedHost 8 GiB et le rejet des profils qui dépassent 88 GiB ; B2 par la propagation immédiate des dépendances échouées ou annulées ; B3 par une clé d'éviction Adaptive qui compare chaleur et récence avant le coût de rechargement. Les mesures VRAM, transferts, events, allocations et libérations sont cohérentes avec ces contrats.

Les limites sont explicites : transfert M7 de 1 MiB, cache d'experts à échelle réduite, absence de kernels et d'injection de fautes, coût de rechargement non alimenté automatiquement, MSRV 1.86 non installée localement, `compute-sanitizer` non exploitable sous WSL2/WDDM, et environnement partagé avec l'affichage Windows. Aucun test non exécuté n'est déclaré PASS.
