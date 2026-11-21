# M5 — Validation matérielle CUDA sur NVIDIA RTX 5090

**Périmètre :** stream non bloquant, polling des events, finalisation avant admission/éviction et déterminisme des évictions.  
**Branche :** `feat/m5-cuda-streams-eviction-hardening`.  
**Base M5 de la campagne générale :** `c465100202370d7b3246bb2cd656a435fbff4fc6`.  
**HEAD contenant le correctif B2, le test M5-HW-POLL-01 et la couverture des égalités LRU/LFU/Adaptive :** `0b7314d3a9388ef528426cdb8e55e1c4831fc025`.  
**SHA exact exécuté pour la preuve B1 :** `c7d81f4ded02411ae2054d8e07786ae1f468b499` (descendant documentaire de `0b7314d` ; seule la documentation de `poll_completed_transfers` diffère, sans changement de comportement).\
**Date d'exécution :** 2026-11-21.\
**Campagne générale :** 24 tests matériels PASS, 0 FAIL, documentés initialement dans la campagne M4/M5.  
**Statut de la preuve ciblée B1 :** PASS sur RTX 5090 au SHA `c7d81f4ded02411ae2054d8e07786ae1f468b499`.

## 1. Environnement observé

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, Blackwell |
| Compute capability | 12.0 |
| VRAM | 34 162 016 256 octets (32 Gio nominalement) |
| Pilote NVIDIA | 616.64 |
| CUDA UMD / Driver API | 13.4 / 13040 |
| Outil NVIDIA-SMI | 615.65.07 |
| Toolkit nvcc dans PATH | CUDA 13.0, V13.0.48 |
| Système | WSL2, Ubuntu 24.04.5 LTS |
| Noyau | 6.18.58.1-microsoft-standard-WSL2 |
| Rust / Cargo | 1.100.0 |
| Cible | x86_64-unknown-linux-gnu |
| Bibliothèque Driver | /usr/lib/wsl/lib/libcuda.so.1.1 |
| Mode pilote | WDDM |

Le toolkit CUDA n'est pas utilisé pour compiler Claymore : le backend utilise directement CUDA Driver API via `libcuda`.

Initialisation réelle du pilote, contexte primaire, flag de stream et transfert DMA vérifiés par une sonde indépendante :

```text
driver_api_version=13040 device_count=1
device=NVIDIA GeForce RTX 5090 pci=0000:01:00.0 cc=12.0
stream_flags=0x1 non_blocking=yes
enqueue_0 event_query=600 (not_ready)
enqueue_1 event_query=600 (not_ready)
enqueue_2 event_query=600 (not_ready)
enqueue_3 event_query=600 (not_ready)
d2h_verified_mismatches=0
probe=OK
```

## 2. Campagne matérielle générale déjà exécutée

La campagne M4/M5 générale a exécuté **24 tests matériels sur la RTX 5090 : 24 PASS, 0 FAIL**. Les tests couvrent :

| Groupe | Couverture | Résultat |
|---|---|---|
| HW-C1 (6 tests) | Initialisation, contexte primaire, streams/events, flag non bloquant, H2D/D2H, synchronisation | 6 PASS |
| HW-C2 (4 tests) | Allocations, libération explicite, double libération, tailles et bornes | 4 PASS |
| HW-C3 (2 tests) | Transferts de tailles variées et intégrité octet par octet | 2 PASS |
| HW-C4 (3 tests) | Tokens, polling/query, waits ciblés et ownership manager | 3 PASS |
| HW-C5 (3 tests) | Protection en vol, transferts multiples et exclusion d'éviction | 3 PASS |
| HW-C6 (3 tests) | Polling avant admission, déterminisme matériel LRU et comptabilité | 3 PASS |
| HW-C7 (3 tests) | Répétition, polling rapproché et alternance transfert/éviction | 3 PASS |

Les détails des procédures et sorties de la campagne M4/M5 sont conservés dans le [rapport détaillé](m4-m5-rtx5090-hardware-validation.md). Le scénario HW-C6-01 existant attendait la complétion via `is_transfer_complete` avant de demander l'admission : il ne remplaçait donc pas le test ciblé demandé par la revue architecturale.

## 3. Scénario ciblé M5-HW-POLL-01 — preuve B1

Un nouveau test ignoré par défaut est ajouté dans `crates/claymore-memory/src/cuda.rs` :

`hardware_poll_retires_completed_transfer_before_admission_without_wait`

Procédure attendue :

1. Allouer un buffer pinned hôte et une allocation VRAM `victim`.
2. Enqueue un transfert H2D réel.
3. Appeler directement `CudaMemoryManager::poll_completed_transfers` en boucle, sans `is_transfer_complete`, `wait_for_transfer` ni `wait_for_allocation`.
4. Arrêter uniquement lorsque le polling retourne `retired == 1`, avec un délai maximal borné.
5. Demander une allocation `replacement` qui dépasse le budget disponible.
6. Vérifier que l'admission réussit, que `victim` a été évincée et que la comptabilité correspond à `replacement`.

**Résultat : PASS — exécuté le 2026-11-21 sur `c7d81f4ded02411ae2054d8e07786ae1f468b499`.**

Commande ciblée :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

cargo test -p claymore-memory --features cuda-driver --lib \
  hardware_poll_retires_completed_transfer_before_admission_without_wait \
  -- --ignored --nocapture --test-threads=1
```

Sortie observée :

```text
running 1 test
test cuda::tests::hardware_poll_retires_completed_transfer_before_admission_without_wait ... M5-HW-POLL-01 retired=1; admission evicted completed victim without explicit wait
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 23 filtered out; finished in 0.24s
```

Durée : **0,24 s** pour le test (0,26 s de temps mural pour la commande complète).

Critères vérifiés :

| Critère | Preuve observée |
|---|---|
| Transfert H2D réel terminé | `copy_h2d` enqueué sur le stream de transfert, puis event positivement complété par `cuEventQuery` |
| Polling direct | Seul `poll_completed_transfers` est appelé dans la boucle de complétion |
| `retired == 1` avant expiration | La boucle retourne `1` bien avant la borne de 10 s |
| Aucun wait explicite de complétion | Le corps du test n'appelle ni `wait_for_transfer` ni `wait_for_allocation` après l'enqueue H2D. L'admission passe par `make_room` → `poll_completed_transfers` non bloquant → `release` ; l'appel interne de `release` à `wait_for_allocation` ne trouve plus aucun event en vol une fois l'event retiré. |
| Admission sous pression | `allocate_device("replacement", 512 Kio)` réussit |
| Victime effectivement évincée | `release("victim")` retourne `UnknownAllocation`, et l'usage VRAM passe de 768 Kio à 512 Kio |
| Comptabilité VRAM | `usage_bytes(Vram) == 768 Kio` après retirement, puis `512 Kio` après admission |

Précision sur les waits internes : le `write_pinned` de préparation (antérieur au transfert) et le `release` d'éviction traversent `wait_for_allocation` comme garde-fou. Dans les deux cas aucun transfert n'est en vol, donc ces appels ne bloquent pas et ne participent pas à la complétion. Aucune attente n'est utilisée pour forcer la complétion H2D ou l'admission.

Aucun résultat n'est inféré de la campagne précédente. La preuve B1 est désormais attachée au SHA exact ci-dessus.

## 4. Commandes reproductibles

Sous WSL2, commandes exactement exécutées au SHA `c7d81f4ded02411ae2054d8e07786ae1f468b499` :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# Tests unitaires CUDA non ignorés (bibliothèques claymore-cuda et claymore-memory).
cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib -- --test-threads=1

# Tests matériels ignorés, dont M5-HW-POLL-01.
cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib -- --ignored --nocapture --test-threads=1
```

Résultats :

- Tests non ignorés : `claymore-cuda` 5 PASS, 0 FAIL ; `claymore-memory` 22 PASS, 0 FAIL, 2 ignorés. Durée 0,68 s.
- Tests ignorés : 2 PASS, 0 FAIL (`cuda_manager_reports_device_memory` et M5-HW-POLL-01). Durée 0,41 s.
- Aucun échec, aucun test filtré par erreur.

La campagne générale du wrapper peut être rejouée avec :

```bash
cargo test -p claymore-cuda --features cuda-driver -- --ignored --nocapture --test-threads=1
```

## 5. Limitations et risques résiduels

- `compute-sanitizer` 2025.3.0 est installé mais inutilisable dans cet environnement WSL2/WDDM.
- Les erreurs réelles `cuEventQuery`, `cuStreamSynchronize` et `cuMemFree` n'ont pas été injectées sur le pilote ; l'utilisation de handles CUDA synthétiques contre le pilote réel n'est pas sûre.
- Le chemin de conservation de l'ownership et de la comptabilité lors d'un véritable échec de `cuMemFree` n'est donc pas démontré sur matériel.
- La campagne matérielle nominale n'est pas une preuve de comportement multi-thread ; le manager reste confiné au thread propriétaire.
- M5-HW-POLL-01 est une exécution unique avec un transfert H2D de 4 Kio et une borne de 10 s ; il démontre le chemin nominal de retirement par polling, pas un stress de latence sous charge GPU concurrente.
- L'injection déterministe et les tests manager-level avec backend CUDA simulable sont différés dans l'issue [#19](https://github.com/zenhit777/Claymore/issues/19).

## 6. Verdict

**Campagne nominale générale :** `HARDWARE VALIDATION PASSED WITH LIMITATIONS` (24/24).  
**Critère matériel ciblé B1 :** `PASSED` sur RTX 5090 au SHA `c7d81f4ded02411ae2054d8e07786ae1f468b499` (M5-HW-POLL-01, `retired == 1`, admission sous pression sans wait explicite).

La preuve B1 est désormais attachée au HEAD final de la PR #17. Les limitations résiduelles de la section 5 restent applicables.
