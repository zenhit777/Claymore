# Validation matérielle M7 — NVIDIA RTX 5090

**Date de campagne :** 2026-11-21  
**Dépôt :** `zenhit777/Claymore`  
**PR :** #21 — `feat/m7-deterministic-scheduler`  
**SHA testé :** `2a2e14fe96baa5ec3962212571cbea28d24d6726`  
**Commit :** `style(m7): apply final rustfmt corrections from CI`  
**Périmètre :** validation matérielle du scheduler déterministe M7, du `CudaTransferExecutor`, du cache d'experts et des contrats mémoire CUDA associés. Aucun kernel CUDA, aucun calcul tensoriel et aucune génération de modèle ne sont validés par cette campagne.

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
2a2e14fe96baa5ec3962212571cbea28d24d6726
?? Cargo.lock
```

- Le worktree était initialement sur un HEAD détaché à `385db116f4c2cce6138ee530f7bec7771f355d95` (base de la campagne M4/M5). Après checkout de la branche M7 et `pull --ff-only`, `HEAD` correspond exactement au SHA demandé.
- Aucun fichier suivi n'était modifié avant la campagne.
- Un fichier `Cargo.lock` non suivi était déjà présent dans le worktree. Il n'a pas été ajouté au dépôt et n'est pas inclus dans le commit de rapport.
- Aucune API publique, aucun fichier source Rust, aucune configuration projet et aucune dépendance Cargo n'ont été modifiés pendant la campagne.
- La PR #21 n'a pas été fusionnée.

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

## 3. Vérification du profil matériel

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
host_budget_bytes=94489280512
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
| Budget VRAM souple | 26 GiB | 27 917 287 424 octets = 26 GiB | PASS |
| Plafond VRAM | 28 GiB | 30 064 771 072 octets = 28 GiB | PASS |
| Réserve de sécurité | 4 GiB | 4 GiB | PASS |
| Cache experts — cible | 14 GiB | 15 032 385 536 octets = 14 GiB | PASS |
| Cache experts — maximum | 16 GiB | 17 179 869 184 octets = 16 GiB | PASS |
| Budget RAM Claymore | 88 GiB | 94 489 280 512 octets = 88 GiB | PASS |
| Mémoire pinned | 8 GiB | 8 589 934 592 octets = 8 GiB | PASS |
| Workers CPU | 10 | 10 | PASS |
| Préchargements max | 2 | 2 | PASS |
| Lookahead | 1 | 1 | PASS |
| Streams de calcul | 1 | 1 | PASS |
| Streams de transfert | 1 | 1 | PASS |
| Tick scheduler | 500 µs | 500 µs | PASS |
| Validation du profil | valide | `status=profile_validated` | PASS |

Le champ `cuda_driver_feature=false` est attendu pour cette commande : le profil est validé sans activer la feature CUDA. La détection CUDA est réalisée séparément à la section 4.

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

## 5. Campagne de tests matériels M7

### 5.1 Commande exécutée

```bash
cargo test -p claymore-memory \
  --features cuda-driver \
  --test hardware_cuda_memory \
  -- --ignored --test-threads=1
```

La campagne a été exécutée deux fois avec `--nocapture` pour conserver les mesures imprimées par les tests. Le flag `--nocapture` ne modifie ni le code ni les assertions ; il ne fait qu'afficher les sorties normalement capturées par le harness Rust.

Les 17 tests du binaire `hardware_cuda_memory` ont été exécutés. Les tests sont sérialisés par un mutex GPU et par `--test-threads=1`.

### 5.2 Résultats de la suite complète

| Campagne | Exécutés | Réussis | Échoués | Ignorés | Code retour | Durée |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 17 | 0 | 0 | 0 | 26,95 s |
| 2 | 17 | 17 | 0 | 0 | 0 | 27,05 s |
| **Total** | **34** | **34** | **0** | **0** | **0** | — |

### 5.3 Résultat de chaque test

| ID | Test | Contrat vérifié | Résultat | Preuve observée |
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
| HW-C6-01 | `hw_c6_01_polling_precedes_eviction_planning` | Polling avant planification d'éviction | PASS | `completed event polled before eviction after 41 queries` |
| HW-C6-02 | `hw_c6_02_equal_priority_eviction_is_deterministic` | Départage déterministe à priorité égale | PASS | `alpha` évincé 12/12 |
| HW-C6-03 | `hw_c6_03_accounting_and_admission_errors` | Comptabilité et erreurs d'admission | PASS | `accounting stayed consistent across release and admission errors` |
| HW-C7-01 | `hw_c7_01_repeated_allocate_transfer_free_cycles` | 128 cycles allocation/transfert/libération | PASS | `128 H2D/D2H cycles verified in 102.539476ms` |
| HW-C7-02 | `hw_c7_02_tight_polling_is_idempotent` | Polling rapproché et idempotence | PASS | `1000 polls, retired_total=8/8, state stayed coherent` |
| HW-C7-03 | `hw_c7_03_transfer_and_eviction_alternation` | 16 alternances transfert/éviction | PASS | `16 transfer/eviction alternations verified` |
| **M7-TRANSFER** | `hw_m7_cuda_transfer_executor_runs_scheduler_tasks_after_event_completion` | Chaîne de 8 tâches, H2D/D2H réels, attente d'event, libération ordonnée | **PASS** | `H2D=1048576 bytes in 118892 ns; D2H=1048576 bytes in 69841 ns; event_wait=122272 ns` |
| **M7-EXPERTS** | `hw_m7_expert_cache_max_is_enforced_before_device_allocation` | Plafond du cache d'experts et éviction déterministe | **PASS** | `expert cache maximum enforced with deterministic idle eviction` |

**Synthèse : 17/17 tests PASS, 0 FAIL, 0 test non exécuté dans cette cible.**

### 5.4 Test M7 principal — vérification de l'ordre scheduler/event

Le test construit la chaîne de dépendances suivante :

```text
allocate_host
  -> allocate_device
    -> write_host
      -> h2d
        -> d2h
          -> read_host
            -> release_device
              -> release_host
```

Les huit appels à `QueueRunner::run_one` retournent `Completed` dans cet ordre. Le résultat lu après le D2H est identique au motif de 1 MiB écrit dans le buffer pinned.

La garantie « aucune tâche dépendante libérée avant confirmation de l'event » repose sur deux éléments observés sur le SHA testé :

1. `QueueRunner::run_one` n'appelle `scheduler.complete(task_id)` qu'après le retour `Ok(())` de `executor.execute(...)`.
2. `CudaTransferExecutor::execute` appelle `wait_for_transfer` pour chaque H2D/D2H avant de retourner `Ok(())` ; `wait_for_transfer` synchronise l'event CUDA associé avant de retirer la protection de l'allocation.

La mesure `event_wait_ns = 122 272 ns` confirme qu'une attente d'event non nulle a réellement eu lieu. Le test matériel ne place pas de point d'observation à l'instant exact où `scheduler.complete` est appelé ; la preuve est structurelle (ordre des appels) et comportementale (chaîne complète et intégrité des données). Les tests HW-C4 et HW-C5 vérifient par ailleurs directement les états pending/completed et la protection des allocations en vol.

---

## 6. Mesures H2D/D2H et attente d'event

### 6.1 Mesures de la suite complète

| Métrique | Campagne 1 | Campagne 2 |
|---|---:|---:|
| Octets H2D | 1 048 576 | 1 048 576 |
| Durée totale H2D | 118 892 ns | 113 842 ns |
| Octets D2H | 1 048 576 | 1 048 576 |
| Durée totale D2H | 69 841 ns | 74 361 ns |
| Attente cumulée des events | 122 272 ns | 132 292 ns |
| Échecs | 0 | 0 |
| Opérations restantes après exécution | 0 | 0 |

### 6.2 Exécutions isolées du test M7

Le test M7 a également été exécuté seul trois fois, avec `--exact --nocapture` :

| Exécution | Durée H2D | Durée D2H | Attente event | Résultat |
|---|---:|---:|---:|---|
| 1 | 1 041 003 ns | 76 531 ns | 92 960 ns | PASS |
| 2 | 971 738 ns | 69 641 ns | 87 472 ns | PASS |
| 3 | 945 167 ns | 123 942 ns | 139 492 ns | PASS |

**Interprétation.** Les durées H2D des exécutions isolées sont plus élevées que celles de la suite complète, car le test isolé est la première utilisation CUDA du processus : l'initialisation du contexte, des pages pinned et du stream est incluse dans le premier transfert. Dans la suite complète, les tests précédents ont déjà initialisé le contexte. Cette différence est une variance de démarrage à froid, pas un échec.

**Important.** `h2d_total_ns` et `d2h_total_ns` sont des durées de bout en bout incluant l'enqueue et l'attente de l'event. `event_wait_ns` est le temps cumulé passé à attendre les events, toutes opérations confondues. Ces valeurs ne doivent pas être converties en débit DMA : la taille transférée est de 1 MiB, trop petite pour une mesure de bande passante, et l'attente d'event n'est pas un débit. Aucune revendication de performance de transfert n'est faite dans ce rapport.

### 6.3 Métriques du `QueueRunner`

Les métriques `QueueRunnerMetrics` ne sont pas imprimées par le test matériel. Une sonde de mesure hors dépôt, compilée dans `/tmp` et utilisant exactement la même séquence de huit tâches via l'API publique, a été exécutée pour les relever. Elle n'ajoute aucun test au dépôt et ne modifie pas le code.

| Champ `QueueRunnerMetrics` | Valeur observée |
|---|---:|
| `kind` | `Transfer` |
| `started` | 8 |
| `completed` | 8 |
| `failed` | 0 |
| `idle_polls` | 0 |
| `total_execution_ns` | 4 806 449 ns |
| `max_execution_ns` | 2 054 765 ns |

L'intégrité de l'aller-retour dans la sonde est `true`, `pending_operations = 0`, la VRAM finale est 0 et la mémoire pinned finale est 0.

### 6.4 Métriques `CudaTransferMetrics`

La même sonde a relevé les champs suivants lors de son exécution (processus séparé, contexte à froid) :

| Champ `CudaTransferMetrics` | Valeur observée |
|---|---:|
| `allocations` | 2 |
| `releases` | 2 |
| `h2d_transfers` | 1 |
| `d2h_transfers` | 1 |
| `h2d_bytes` | 1 048 576 |
| `d2h_bytes` | 1 048 576 |
| `h2d_total_ns` | 972 947 ns |
| `d2h_total_ns` | 87 461 ns |
| `pinned_read_bytes` | 1 048 576 |
| `event_wait_ns` | 100 710 ns |
| `failed` | 0 |
| `total_execution_ns` | 4 794 779 ns |
| `max_execution_ns` | 2 052 235 ns |

Les valeurs diffèrent de celles de la suite complète pour les raisons de démarrage à froid exposées ci-dessus. Elles ne remplacent pas les mesures officielles de la section 6.1 ; elles complètent la vérification des compteurs.

### 6.5 Métriques `MemoryExecutionMetrics`

`CudaTransferExecutor` expose `CudaTransferMetrics`, pas `MemoryExecutionMetrics`. `MemoryExecutionMetrics` appartient à l'exécuteur mémoire backend-neutre `MemoryTaskExecutor`, qui n'est pas le chemin CUDA utilisé par le test M7.

La vérification de ces métriques a donc été faite par les tests unitaires logiciels du crate `claymore-memory` :

| Test | Contrat | Résultat | Preuve |
|---|---|---|---|
| `executor::tests::memory_executor_reports_operation_counts_and_bytes` | Comptage allocations/écritures/octets | PASS | `allocations=1`, `writes=1`, `bytes_written=4`, `failed=0` |
| `executor::tests::failed_prerequisite_discards_cancelled_memory_payloads` | Nettoyage après échec et annulation | PASS | `failed=1`, `pending_operations=0` |

`cargo test -p claymore-memory --lib` : **30 tests PASS, 0 FAIL**.  
`cargo test -p claymore-scheduler` : **16 tests PASS, 0 FAIL**, dont `queue_runner_metrics_are_scoped_to_its_logical_queue` (`started=1`, `completed=1`, `failed=0`, `idle_polls=1`).

---

## 7. Consommation VRAM

La consommation a été échantillonnée toutes les 200 ms avec `nvidia-smi` pendant chaque campagne complète.

| Moment | VRAM utilisée | VRAM libre | Delta vs baseline |
|---|---:|---:|---:|
| Avant campagne 1 | 1 324 MiB | 30 864 MiB | 0 |
| Pic campagne 1 | 2 331 MiB | 29 857 MiB | +1 007 MiB |
| Après campagne 1 | 1 324 MiB | 30 864 MiB | 0 |
| Avant campagne 2 | 1 326 MiB | 30 862 MiB | 0 |
| Pic campagne 2 | 2 333 MiB | 29 855 MiB | +1 007 MiB |
| Après campagne 2 | 1 326 MiB | 30 862 MiB | 0 |

- Le pic représente environ **7,2 %** de la VRAM totale de 32 607 MiB.
- La baseline d'environ 1,3 GiB correspond au compositeur Windows/Xwayland, pas à Claymore.
- Le delta maximal observé est d'environ 1 007 MiB. Il inclut les allocations device des scénarios de pression (512 MiB pour HW-C5-01, 2 × 256 MiB pour HW-C5-03), les buffers de transfert et l'overhead du contexte CUDA.
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
| `compute-sanitizer` | Non exécuté ; sous WSL2/WDDM l'interface debugger est indisponible (limitation connue) | NOT RUN |
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
| VRAM GPU après campagne | Retour à la baseline (1 326 MiB, Xwayland) | PASS |
| Processus CUDA résiduels | Aucun (`nvidia-smi --query-compute-apps` vide) | PASS |

Le `Drop` de `CudaMemoryManager` synchronise le stream de transfert avant de libérer les buffers ; en cas d'échec de synchronisation, il choisit de ne pas libérer les buffers plutôt que de risquer une libération pendant un DMA en vol. Ce comportement a été vérifié par inspection du code sur le SHA testé ; le chemin d'échec de synchronisation n'a pas été injecté sur matériel.

---

## 10. Verdict

### 10.1 Verdict par catégorie

| Catégorie | Verdict | Justification |
|---|---|---|
| **Scheduler** | **PASS** | Chaîne de 8 tâches exécutée dans l'ordre ; dépendants libérés seulement après retour de l'exécuteur ; `QueueRunnerMetrics` cohérentes (8/8, 0 échec, 0 idle) ; 16 tests scheduler PASS. |
| **Transferts CUDA** | **PASS** | H2D et D2H réels de 1 MiB, intégrité byte-exact, attente d'event non nulle, 0 échec, 0 opération restante. |
| **Sûreté mémoire** | **PASS** | Allocations en vol protégées, transferts partageant une allocation, polling avant éviction, cycles répétés et comptabilité cohérente : tous les tests HW-C4/C5/C6/C7 PASS. |
| **Cache d'experts** | **PASS** | Plafond maximal appliqué avant allocation device, éviction déterministe de l'expert inactif, comptage VRAM cohérent, aucun dépassement. |
| **Pression VRAM** | **PASS** | Pic observé à 2 333 MiB (7,2 % de la VRAM), retour à la baseline, aucune saturation volontaire. |
| **Télémétrie** | **PASS** | `CudaTransferMetrics` et `QueueRunnerMetrics` relevés et cohérents ; `MemoryExecutionMetrics` n'est pas exposé par le chemin CUDA et a été vérifié séparément sur l'exécuteur backend-neutre (réserve documentée en 10.3). |

### 10.2 Verdict global

**Verdict global : `PASS`.** Validation matérielle M7 réussie sur NVIDIA RTX 5090, avec les limitations documentées ci-dessous.

- 17/17 tests matériels PASS lors de deux campagnes complètes (34/34 exécutions), 0 FAIL.
- Le test M7 principal et le test du cache d'experts sont PASS.
- Les contrats de protection mémoire, d'éviction et de comptabilité sont PASS.
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
8. **WSL2/WDDM.** GPU partagé avec l'affichage Windows, pas de MIG/ECC, `compute-sanitizer` non exploitable. Les résultats sont représentatifs de cette configuration, pas d'un runner Linux natif.
9. **Échantillonnage VRAM.** L'intervalle de 200 ms peut manquer des pics très courts ; les valeurs de pic sont des observations, pas des bornes absolues.
10. **Variance temporelle.** Les durées H2D à froid (environ 0,95–1,04 ms) diffèrent des mesures en contexte chaud (environ 0,11–0,12 ms). Il s'agit d'un effet d'initialisation, pas d'une instabilité du contrat.

---

## 11. Reproductibilité

```bash
git checkout 2a2e14fe96baa5ec3962212571cbea28d24d6726

export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo run -p claymore-cli -- profile
cargo run -p claymore-cli --features cuda-driver -- device

cargo test -p claymore-memory \
  --features cuda-driver \
  --test hardware_cuda_memory \
  -- --ignored --test-threads=1 --nocapture

cargo test -p claymore-scheduler
cargo test -p claymore-memory --lib
```

Les logs complets de la campagne ont été conservés hors dépôt dans `/tmp/opencode/` sur la machine d'exécution :

- `/tmp/opencode/m7-tests.log` — campagne complète 1
- `/tmp/opencode/m7-tests-2.log` — campagne complète 2
- `/tmp/opencode/m7-vram.csv` — échantillonnage VRAM campagne 1
- `/tmp/opencode/m7-vram-2.csv` — échantillonnage VRAM campagne 2
- `/tmp/opencode/m7-repeat.log` — trois exécutions isolées des deux tests M7

Empreintes SHA-256 :

```text
81b03ede6e005a3fdf8e2298684ef11e8907bdd20afe91900f61eae553f47a3f  m7-tests.log
8f6e9337e1382a65b0c6ed4ab04560259c91ef0af381ec88bc4524cf9c0956f6  m7-tests-2.log
1bdf037c82bb24938808942fd83af7306b4d04a81926ef303bf0db6b766c9564  m7-vram.csv
26ee567416a479c8fe56a9ee3e327e3998d98b6cf35294c0a2e5604ffe8579d0  m7-vram-2.csv
4c70c3a9c010918d4cecc17a31a9532af97f23ea73e99adb9f92279a5a09ad26  m7-repeat.log
```

---

## 12. Conclusion

La campagne a réellement exécuté les 17 tests matériels M7 sur la RTX 5090 avec le pilote `616.64` et CUDA UMD `13.4`. Les 17 tests ont réussi lors de deux campagnes consécutives, sans échec, sans panique et sans erreur CUDA. Le scheduler, le `CudaTransferExecutor`, le cache d'experts et les contrats de protection mémoire sont validés pour le périmètre M7.

Les limites sont explicites : transfert M7 de 1 MiB, cache d'experts à échelle réduite, absence de kernels et d'injection de fautes, MSRV 1.86 non installée localement, et environnement WSL2/WDDM partagé avec l'affichage. Aucun test non exécuté n'est déclaré PASS.
