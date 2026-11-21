# Validation matérielle M4/M5 — NVIDIA RTX 5090

**Date :** 2026-11-20  
**Branche de validation d'origine :** `test/m4-m5-rtx5090-validation` (preuves et résultats conservés dans le présent rapport)  
**Base :** `feat/m5-cuda-streams-eviction-hardening` @ `c465100202370d7b3246bb2cd656a435fbff4fc6`  
**Base M4 (main au moment de l'audit) :** `b576a112725a245c657113786026fedf70ffa283` (`feat(m4): harden Device lifecycle and async completion contracts (#16)`)  
**État de PR #17 au début de la campagne (2026-11-20) :** ouverte, non fusionnée ; branche et PR non modifiées pendant cette campagne. La PR #17 a depuis été fusionnée dans `main` par le commit `29b138177bf8e90c9815d6b7bc03a2f261934bf0`.  
**Périmètre :** contrats CUDA M4/M5 uniquement. Aucun framework tensoriel externe, aucun kernel, aucun benchmark généraliste, aucun travail M6.

---

## 1. Environnement

### 1.1 Matériel

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell |
| UUID | `GPU-7556cae7-89a8-6267-b8b4-4e2edb9981d6` |
| PCI | `00000000:01:00.0` (bus `0x01`, device `0x00`) |
| PCI device ID | `0x2B8510DE` |
| Compute capability | `12.0` (Driver API) |
| VRAM (nvidia-smi) | 32 607 MiB |
| VRAM (`cuDeviceTotalMem_v2` / `cuMemGetInfo`) | 34 162 016 256 octets |
| MIG | N/A |
| ECC | N/A |
| Display Active | Enabled (GPU partagé avec le compositeur Windows) |
| Driver model | WDDM |

### 1.2 Pilote et CUDA

| Élément | Valeur observée |
|---|---|
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| `cuDriverGetVersion()` | `13040` (CUDA 13.4 Driver API) |
| NVIDIA-SMI tool | `615.65.07` |
| `nvcc` dans le `PATH` | CUDA 13.0, `V13.0.48` (`/usr/local/cuda-13.0/bin/nvcc`) |
| `/usr/local/cuda` | pointe vers `/usr/local/cuda-13.5` (CUDA 13.5.0) |
| Toolkits installés | 12.8.2, 13.0.20250728, 13.3.1, 13.5.0 |
| Bibliothèque Driver | `/usr/lib/wsl/lib/libcuda.so.1.1` (WSL2), chargée par `ldd` |
| Runtime CUDA | non requis pour Claymore ; le backend lie uniquement `libcuda` |
| `compute-sanitizer` | 2025.3.0.0 disponible, mais inutilisable sous WSL2/WDDM (voir limitations) |

Le toolkit n'est pas utilisé pour compiler Claymore. Les tests Rust appellent directement la CUDA Driver API via `libcuda`.

### 1.3 Environnement d'exécution

| Élément | Valeur observée |
|---|---|
| Plateforme | WSL2 |
| Noyau | `6.18.58.1-microsoft-standard-WSL2` |
| Distribution | Ubuntu 24.04.5 LTS |
| Accès GPU WSL2 | `/dev/dxg` présent ; pas de `/dev/nvidia*` (attendu sous WSL2) |
| RAM hôte visible | 112 GiB |
| Swap | 16 GiB |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Cargo | `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| Cible | `x86_64-unknown-linux-gnu` |
| Édition / version workspace | 2021 / `0.5.0` |
| MSRV déclarée | Rust 1.86 (non installée localement, voir limitations) |
| Profil de test | `dev` (debug), feature `cuda-driver` |

**Linking WSL2.** Le linker ne trouve pas `libcuda.so` dans ses chemins par défaut. La documentation existante `docs/wsl2.md` est correcte : il faut passer `-L native=/usr/lib/wsl/lib`. Les commandes utilisées sont :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

Aucune modification du pilote, de CUDA, de WSL2 ou de la configuration système n'a été effectuée.

### 1.4 Sonde indépendante CUDA Driver API

Avant les tests Claymore, une sonde C indépendante (hors dépôt) a vérifié l'initialisation réelle du pilote, la création d'un contexte primaire, le flag de stream, le polling d'events et un transfert H2D/D2H de 512 MiB × 4 :

```text
driver_api_version=13040 device_count=1
device=NVIDIA GeForce RTX 5090 pci=0000:01:00.0 cc=12.0
context_retained=0x5f16ea0215b0
stream_flags=0x1 non_blocking=yes
vram_free_before=32429309952 total=34162016256
enqueue_0 event_query=600 (not_ready)
enqueue_1 event_query=600 (not_ready)
enqueue_2 event_query=600 (not_ready)
enqueue_3 event_query=600 (not_ready)
h2d_ms=46.259
d2h_verified_mismatches=0
probe=OK
```

Cela établit que `libcuda` n'est pas seulement présent : le pilote s'initialise, le contexte primaire est créé, `CU_STREAM_NON_BLOCKING` vaut bien `0x1` sur le handle créé, `cuEventQuery` retourne `CUDA_ERROR_NOT_READY` (`600`) pour du travail en file, et l'aller-retour DMA est exact.

---

## 2. Audit du code M5 et des tests existants

### 2.1 Fichiers examinés

- `crates/claymore-cuda/src/driver.rs`
- `crates/claymore-cuda/src/lib.rs`
- `crates/claymore-memory/src/cuda.rs`
- `crates/claymore-memory/src/lib.rs`
- `crates/claymore-memory/src/lifecycle.rs`
- `crates/claymore-memory/src/device.rs`
- `docs/audits/m4-device-lifecycle-contracts.md`
- `docs/audits/m5-cuda-streams-eviction.md`
- `docs/memory-cuda-integration-v0.5.md`
- `docs/wsl2.md`
- `.github/workflows/rust.yml`

### 2.2 Tests matériels déjà présents

Un seul test matériel existait : `cuda::tests::cuda_manager_reports_device_memory` dans `crates/claymore-memory/src/cuda.rs`, marqué `#[ignore]`. Il couvre l'initialisation du manager, le partage de contexte, un petit H2D/D2H et le polling non bloquant. Il a été conservé et exécuté tel quel (HW-C1-06).

Aucun autre test CUDA matériel n'était présent. Les tests unitaires `claymore-cuda` n'étaient pas compilés par la CI avec `cuda-driver` (voir défaut D1).

### 2.3 Défauts découverts

**D1 — Les tests unitaires `claymore-cuda` ne compilaient pas avec `--features cuda-driver`.**

- Contrat concerné : validation logicielle des conversions d'erreurs `cuEventQuery` et `cuStreamSynchronize` (M5).
- Cause : `assert_eq!(event_query_result(...), Ok(true))` exige `PartialEq` sur `CudaError`, qui n'est pas implémenté.
- Impact : les tests M5 de polling/erreurs ne pouvaient pas être exécutés via `cargo test -p claymore-cuda --features cuda-driver`. La CI ne les compilait pas non plus (`cargo check` sans `--all-targets`).
- Correction : assertions réécrites avec `assert!`/`matches!` (aucun changement de contrat public).
- Régression : ajout de `cargo check -p claymore-cuda --features cuda-driver --all-targets` et `cargo check -p claymore-memory --features cuda-driver --all-targets` dans la CI. Ces commandes ne lient pas `libcuda` et restent exécutables sans GPU.

**D2 — `CudaError::SizeOverflow` n'est pas atteignable sur cible 64 bits.**

- `as_size` fait `usize::try_from(u64)` ; sur `x86_64`, toute valeur `u64` tient dans `usize`.
- Une allocation de `u64::MAX` est donc rejetée par `cuMemAlloc_v2`, pas par la garde `SizeOverflow`.
- Ce n'est pas un défaut de comportement : l'erreur backend est bien propagée. Le test a été corrigé pour vérifier l'erreur réelle du pilote.
- Limitation documentée : `SizeOverflow` reste utile sur cible 32 bits uniquement.

**D3 — Le test de départage à priorité égale utilisait des ticks différents.**

- Le premier jet du test matériel allouait `gamma`, `alpha`, `beta` à des ticks différents ; LRU choisissait donc `gamma` (le plus ancien), pas le départage lexicographique.
- Le test a été corrigé pour allouer les trois candidats au même tick. Résultat final : `alpha` évincé 12/12 fois.
- Ce n'était pas un défaut du code M5.

**Aucun défaut de contrat runtime M4/M5 n'a été découvert sur la RTX 5090.**

---

## 3. Campagne de tests matériels

### 3.1 Fichiers ajoutés

- `crates/claymore-cuda/tests/hardware_cuda_driver.rs` — contrats publics du wrapper CUDA.
- `crates/claymore-memory/tests/hardware_cuda_memory.rs` — contrats du `CudaMemoryManager`.
- Trois tests matériels dans `crates/claymore-cuda/src/driver.rs` :
  - `hardware_streams_are_explicitly_non_blocking`
  - `hardware_stream_synchronize_completes_async_copy`
  - `hardware_device_buffer_free_is_explicit_and_idempotent`

Les cibles d'intégration sont déclarées avec `required-features = ["cuda-driver"]`. Les tests sont `#[ignore]` et ne s'exécutent que sur demande. Un mutex sérialise les tests GPU dans chaque binaire ; les exécutions rapportées utilisent en plus `--test-threads=1`.

### 3.2 Commandes d'exécution

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

cargo test -p claymore-cuda --features cuda-driver -- --ignored --nocapture --test-threads=1
cargo test -p claymore-memory --features cuda-driver -- --ignored --nocapture --test-threads=1
```

### 3.3 Résultats — 24 tests exécutés sur RTX 5090

Tous les tests ci-dessous ont réellement été exécutés sur la RTX 5090 avec le pilote NVIDIA. Durées de suite observées : `claymore-cuda` unitaires 0,61 s + intégration 0,70 s ; test M5 existant 0,24 s ; intégration `claymore-memory` 23,56 s (23,43–23,75 s sur trois exécutions).

| ID | Contrat vérifié | Procédure | Résultat | Preuve / sortie | Limitation |
|---|---|---|---|---|---|
| HW-C1-01 | Initialisation CUDA, contexte primaire, inventaire | `CudaDriver::new(0)`, `memory_info`, second handle, ordinal invalide | **PASS** | `device=NVIDIA GeForce RTX 5090 cc=12.0 total=34162016256 free=32429309952` ; handles partagent le contexte ; ordinal 99 rejeté | GPU partagé avec l'affichage |
| HW-C1-02 | Création/destruction streams et events | 64 streams + 64 events, synchronisation de chacun | **PASS** | `created/destroyed 64 streams and 64 events` | Aucune |
| HW-C1-03 | Flag `CU_STREAM_NON_BLOCKING` | `cuStreamGetFlags` sur le stream créé par Claymore + 32 streams supplémentaires | **PASS** | `flags & 0x1 == 0x1` pour tous | Vérifie le flag rapporté par le pilote ; pas de test temporel séparé du legacy default stream |
| HW-C1-04 | Interopérabilité de contexte et transfert asynchrone | Buffers alloués par un handle, stream créé par un second handle, H2D + D2H 1 MiB | **PASS** | `verified 1048576 bytes across two driver handles` | Aucune |
| HW-C1-05 | Synchronisation explicite du stream | H2D 64 MiB, event, `cuStreamSynchronize`, query | **PASS** | Event complété après synchronisation | Aucune |
| HW-C1-06 | Smoke test M5 existant | `cuda_manager_reports_device_memory`, conservé | **PASS** | Test existant vert en 0,24 s | Petit transfert uniquement |
| HW-C2-01 | Libération explicite et double libération | Allocations 1 B, 37 B, 4 KiB, 1 MiB ; `free()` deux fois ; copie après libération | **PASS** | Double free = no-op ; copie après free renvoie `Driver { operation: "cuMemcpyHtoDAsync_v2", code != 0 }` | L'erreur de copie est une erreur d'enqueue, pas une erreur DMA asynchrone |
| HW-C2-02 | Allocation device/pinned et libération | 1 B à 64 MiB, pinned 4 KiB, free explicite | **PASS** | Toutes tailles allouées/libérées sans erreur | Aucune |
| HW-C2-03 | Rejet zéro, dépassement, bornes | `allocate(0)`, `allocate_pinned(0)`, `allocate(u64::MAX)`, copies hors bornes | **PASS** | Zéro rejeté ; `u64::MAX` rejeté par `cuMemAlloc_v2` ; `BufferTooSmall` avant tout appel DMA | `SizeOverflow` non atteignable en 64 bits |
| HW-C2-04 | Libération manager, comptabilité, accès après libération | Allocations, H2D, release, double release, copies vers allocation libérée, doublon, zéro | **PASS** | Comptage VRAM/pinned cohérent ; `UnknownAllocation` après release ; `DuplicateAllocation` | L'échec réel de `cuMemFree` n'est pas injecté (voir HW-C2-05) |
| HW-C3-01 | Transferts H2D/D2H de tailles variées, intégrité octet par octet | 1 B, 37 B, 4 KiB, 1 MiB, 64 MiB ; motif déterministe ; clear puis D2H | **PASS** | `size=1/37/4096/1048576/67108864 byte_exact=yes` | Taille maximale 64 MiB pour ce test |
| HW-C3-02 | Transferts successifs sur plusieurs allocations | 2 paires pinned/device, H2D enchaînés, D2H enchaînés, vérification | **PASS** | Deux motifs distincts vérifiés après D2H | Aucune |
| HW-C4-01 | Token, polling non bloquant, états pending/completed, wait | 8 H2D de 256 MiB enchaînés, query immédiat, poll, wait, re-poll, D2H | **PASS** | `retired=0/8 before wait` ; dernier token NON_READY ; après wait tous complétés ; poll final = 0 | Le nombre d'events retirés par un poll immédiat peut varier ; le test exige `< 8` |
| HW-C4-02 | `wait_for_allocation` ciblé | Petit transfert et transfert 256 MiB en parallèle ; wait sur la petite allocation | **PASS** | Le petit token est complété ; le gros reste pending ; wait ciblé sur le gros le complète | Aucune |
| HW-C4-03 | Tokens liés à leur manager | Token du manager A présenté au manager B | **PASS** | `UnknownTransferToken` pour query et wait | Aucune |
| HW-C5-01 | Protection d'une allocation en vol | 8 H2D de 512 MiB sur une allocation occupant tout le budget ; admission 64 MiB | **PASS** | Admission refusée (`InsufficientReclaimable`), `retired=0/8`, comptage inchangé ; admission réussie après wait | Budget logique contrôlé par le test, pas de saturation GPU globale |
| HW-C5-02 | Protection partagée entre plusieurs transferts | Transfert A + 8 transferts B vers la même allocation ; wait sur A seulement | **PASS** | L'allocation reste protégée par les events B ; éviction possible seulement après wait sur l'allocation | Aucune |
| HW-C5-03 | Exclusion des allocations protégées du plan d'éviction | Deux allocations de 512 KiB, 8 transferts sur `protected`, admission d'une troisième | **PASS** | `candidate` évincée, `protected` survit et est libérable ensuite | Aucune |
| HW-C6-01 | Polling avant planification d'éviction | Victime 768 KiB, petit transfert, attente de complétion par query, admission 512 KiB | **PASS** | Éviction après 41 queries ; comptage 768 KiB → 512 KiB | Le test utilise une boucle de query/yield, pas un sleep arbitraire |
| HW-C6-02 | Départage déterministe à priorité égale | 3 allocations 1 MiB au même tick, admission 1 MiB, 12 rounds | **PASS** | `alpha` évincé 12/12 | Politique LRU testée ; LFU/Adaptive couverts par tests logiciels |
| HW-C6-03 | Comptabilité et erreurs d'admission | Allocations 1 MiB + 512 KiB, doublon, requête 3 MiB, release, release inconnue | **PASS** | Comptage 1536 KiB → 512 KiB → 0 ; `RequestTooLarge`, `DuplicateAllocation`, `UnknownAllocation` | Aucune |
| HW-C7-01 | Répétition allocations/transferts/libérations | 128 cycles H2D + clear + D2H 64 KiB avec vérification | **PASS** | `128 H2D/D2H cycles verified in 100.85 ms` | Aucune |
| HW-C7-02 | Polling rapproché et idempotence | 8 transferts 64 KiB, 1000 polls + queries par token, puis wait | **PASS** | `1000 polls, retired_total=8/8, state stayed coherent` ; poll final = 0 | Aucune |
| HW-C7-03 | Alternance transfert/éviction | 16 rounds : allocation 768 KiB, H2D, wait, allocation 512 KiB forçant l'éviction | **PASS** | 16 alternances, comptage final 0 | Aucune |

**Synthèse : 24/24 tests matériels PASS, 0 FAIL.**

### 3.4 Tests d'injection non exécutables sur matériel réel — BLOCKED

| ID | Contrat visé | Statut | Raison et couverture de substitution |
|---|---|---|---|
| HW-C1-07 | Propagation d'une vraie erreur `cuStreamSynchronize` | **BLOCKED** | Aucun moyen sûr de provoquer une erreur de synchronisation DMA sur le pilote réel. L'injection de handles synthétiques a provoqué un SIGSEGV dans la sonde ; `0x1` est `CU_STREAM_LEGACY`, pas un handle invalide. Les handles d'un contexte privé sont acceptés par ce pilote. La conversion d'erreur est couverte par le test logiciel `stream_synchronization_driver_errors_are_propagated`. |
| HW-C2-05 | Échec réel de `cuMemFree` et conservation de l'allocation | **BLOCKED** | Libérer un buffer avec du travail en vol est une utilisation indéfinie du pilote ; aucun échec fiable et sûr n'a pu être produit. Le contrat générique est couvert par `device::tests::failed_backend_release_keeps_buffer_and_accounting_for_retry` (fake backend), exécuté avec succès. Le chemin CUDA `release()` n'a pas été exercé en échec sur matériel. |
| HW-C4-04 | Erreur `cuEventQuery` et état du manager | **BLOCKED** | Même limitation : pas d'injection sûre d'une erreur d'event sur le pilote réel. La traduction `CUDA_ERROR_NOT_READY`/erreur est couverte par `event_query_distinguishes_complete_pending_and_driver_failure`. |
| HW-C5-04 | Erreur de polling sans corruption de l'état | **BLOCKED** | Dépend de HW-C4-04. Les invariants de non-mutation avant retrait sont couverts par inspection du code et par les tests logiciels de mapping ; aucun test matériel d'erreur n'est possible. |
| HW-C6-04 | `compute-sanitizer` memcheck | **BLOCKED** | Sous WSL2/WDDM : `Failed to initialize WDDM debugger interface` et `Device not supported`. Le test a tourné sans instrumentation ; aucun résultat sanitizer exploitable. |

### 3.5 Validation logicielle (hors matériel)

Ces résultats ne sont **pas** comptés comme validation matérielle CUDA.

| Commande | Résultat |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo test --workspace --all-targets` | PASS — 38 tests, 0 échec |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo clippy -p claymore-memory -p claymore-cli --features cuda-driver --all-targets -- -D warnings` | PASS |
| `cargo check -p claymore-cuda --features cuda-driver --all-targets` | PASS (sans linkage GPU) |
| `cargo check -p claymore-memory --features cuda-driver --all-targets` | PASS (sans linkage GPU) |
| `cargo check -p claymore-memory --features cuda-driver` et `claymore-cli` | PASS |
| `cargo test -p claymore-cuda --features cuda-driver --lib` | PASS — 4 tests, 3 ignorés |
| `cargo test -p claymore-memory --features cuda-driver --lib` | PASS — 22 tests, 1 ignoré |
| `device::tests::failed_backend_release_keeps_buffer_and_accounting_for_retry` | PASS (injection fake) |
| MSRV Rust 1.86 | **NOT RUN** localement (toolchain non installée) ; le job CI `msrv` reste la référence |

---

## 4. Analyse

### 4.1 Contrats confirmés sur RTX 5090

- Initialisation effective de la CUDA Driver API, du périphérique et du contexte primaire partagé entre handles.
- Création/destruction répétées de streams et d'events.
- Flag `CU_STREAM_NON_BLOCKING` effectivement rapporté par le pilote (`0x1`) sur le stream de transfert Claymore.
- Synchronisation explicite d'un stream contenant un transfert asynchrone.
- Allocations device et pinned de 1 B à 512 MiB, libération explicite, double libération sans effet, rejet des allocations après libération.
- Rejet des tailles nulles, des demandes hors bornes et des allocations absurdes par le backend.
- H2D et D2H réels, byte-exact, pour 1 B, 37 B, 4 KiB, 1 MiB, 64 MiB et 256/512 MiB selon scénario.
- Transferts successifs sur plusieurs allocations et plusieurs paires.
- Création de `TransferToken`, interrogation non bloquante, distinction pending/completed, propagation de `CUDA_ERROR_NOT_READY` comme état non complété.
- `wait_for_transfer` et `wait_for_allocation` : complétion effective, portée correcte, cohérence avec le polling, rejet des tokens étrangers.
- Protection d'une allocation référencée par un transfert pending, y compris lorsqu'un autre transfert la partage et qu'un premier event est déjà complété.
- Un polling qui retourne NOT_READY ne retire ni l'event ni la protection ; un polling positif les retire.
- Polling automatique avant planification d'éviction.
- Éviction d'une allocation non protégée alors qu'une allocation protégée est exclue.
- Comptabilité VRAM/pinned cohérente après release, éviction, erreurs d'admission et doublons.
- Départage déterministe par identifiant à priorité égale (12/12).
- Répétitions : 128 cycles H2D/D2H, 1000 polls, 16 alternances transfert/éviction, 3 exécutions complètes de la suite.

### 4.2 Contrats validés uniquement par tests logiciels

- Traduction d'une erreur arbitraire de `cuEventQuery` en `CudaError::Driver`.
- Traduction d'une erreur de `cuStreamSynchronize`.
- Conservation du buffer et de la comptabilité après échec de libération backend (fake CPU, pas CUDA).
- État interne du manager après une erreur de polling (aucun test matériel d'injection).
- Politiques LFU et Adaptive et leurs départages (tests unitaires CPU ; seul LRU a été exercé sur GPU).
- MSRV 1.86.

### 4.3 Défauts

- **Aucun défaut runtime M4/M5** n'a été observé sur la RTX 5090.
- **D1** : défaut de test/CI (tests `claymore-cuda` non compilables avec la feature), corrigé dans cette branche de validation avec garde CI.
- **D2/D3** : hypothèses de tests incorrectes, corrigées ; pas des défauts produit.

### 4.4 Limitations de l'environnement

- WSL2/WDDM : GPU partagé avec l'affichage Windows ; pas de MIG/ECC ; pas de `/dev/nvidia*`.
- `compute-sanitizer` inutilisable (interface debugger WDDM indisponible).
- Le linker exige `-L native=/usr/lib/wsl/lib` ; le runtime trouve `libcuda.so.1` via `LD_LIBRARY_PATH`.
- Aucun kernel CUDA n'est lancé : la campagne couvre les contrats d'allocation, de stream, d'event et de DMA, pas l'exécution de kernels.
- Les erreurs de synchronisation, de query et de free ne peuvent pas être injectées de façon sûre sur le pilote réel ; elles restent couvertes par tests logiciels ou non couvertes pour le chemin CUDA.
- Pas de multi-GPU, pas de D2D, pas de saturation mémoire globale.
- MSRV 1.86 non installée localement.

### 4.5 Risques résiduels

- Le comportement de `CudaMemoryManager::release` lorsqu'un `cuMemFree` réel échoue n'a pas été prouvé sur matériel ; le code conserve le buffer et la comptabilité, mais ce chemin n'a pas été exercé avec le pilote NVIDIA.
- Le comportement du manager après une erreur réelle de `cuEventQuery` ou `cuStreamSynchronize` n'a pas été prouvé sur matériel ; la conception fail-closed est cohérente et testée au niveau des conversions, mais pas en panne injectée.
- `is_transfer_complete` peut marquer un token complété sans retirer l'event ni les protections ; c'est documenté et le contrat exige `wait_for_transfer`/`wait_for_allocation`/`poll_completed_transfers` pour les lever. Un appelant qui confondrait « token complet » et « allocation libérable » prendrait un risque ; les tests vérifient que le manager ne libère pas avant le wait.
- Les tests matériels sont mono-thread par construction (`Rc`, contexte courant) ; aucune garantie multi-thread n'est ajoutée.
- Les tailles testées restent bornées (max 512 MiB par allocation dans les scénarios de pression) pour ne pas saturer le GPU.

### 4.6 Recommandations pour l'architecte

1. Ajouter un point d'injection contrôlé, réservé aux tests, dans le wrapper CUDA (event/stream/free) afin de couvrir matériellement les chemins fail-closed sans handles synthétiques dangereux.
2. Conserver la garde CI ajoutée : `cargo check ... --features cuda-driver --all-targets` empêche la réapparition de D1.
3. Exécuter la campagne sur un runner Linux natif (pas WSL2/WDDM) pour rendre `compute-sanitizer` exploitable et vérifier les chemins de panne.
4. Le job workspace MSRV 1.86 est exécuté par la CI ; la PR #18 ajoute également la compilation de toutes les cibles `cuda-driver` sur Rust 1.86 afin de couvrir les tests conditionnés par cette feature.
5. Documenter dans `docs/wsl2.md` que `LD_LIBRARY_PATH=/usr/lib/wsl/lib` est nécessaire au runtime, en plus de `RUSTFLAGS` au linkage.
6. Ne pas fusionner M5 sur la seule base de cette campagne si les chemins d'erreur backend sont considérés bloquants pour la release ; sinon, les contrats nominaux M4/M5 sont validés sur RTX 5090.

---

## 5. Conclusion

**Verdict : `HARDWARE VALIDATION PASSED WITH LIMITATIONS`**

Justification :

- 24 tests matériels ont été réellement exécutés sur la NVIDIA RTX 5090, avec pilote `616.64`, CUDA UMD `13.4` et Driver API `13040` : **24 PASS, 0 FAIL**.
- Les contrats nominaux M4/M5 — streams non bloquants, cycle de vie des allocations, H2D/D2H, tokens, polling non bloquant, protection en vol, polling avant éviction, déterminisme et comptabilité — sont confirmés par des observations du pilote et des vérifications octet par octet.
- Les limitations sont réelles et explicites : pas d'injection sûre d'erreurs de synchronisation/query/free sur le pilote, `compute-sanitizer` bloqué sous WSL2/WDDM, MSRV non exécutée localement.
- Aucun test `BLOCKED` n'est présenté comme un succès. Aucun test logiciel CPU n'est compté dans les 24 tests matériels.
- Un défaut de test/CI (D1) a été corrigé et une garde CI a été ajoutée ; aucun contrat public n'a été modifié.

---

## 6. Reproductibilité

```bash
git switch test/m4-m5-rtx5090-validation

export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# Tests matériels wrapper CUDA (3 unitaires + 5 intégration)
cargo test -p claymore-cuda --features cuda-driver -- --ignored --nocapture --test-threads=1

# Tests matériels manager CUDA (1 test M5 existant + 15 intégration)
cargo test -p claymore-memory --features cuda-driver -- --ignored --nocapture --test-threads=1

# Validation logicielle / CI locale
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p claymore-memory -p claymore-cli --features cuda-driver --all-targets -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --all-targets
cargo check -p claymore-memory --features cuda-driver --all-targets
```

Les tests sont isolés par un mutex GPU dans chaque binaire et doivent être lancés avec `--test-threads=1` pour garantir l'absence de compétition entre contextes/streams de tests.
