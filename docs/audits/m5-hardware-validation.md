# M5 — Validation matérielle CUDA sur NVIDIA RTX 5090

**Périmètre :** stream non bloquant, polling des events, finalisation avant admission/éviction et déterminisme des évictions.  
**Branche :** `feat/m5-cuda-streams-eviction-hardening`.  
**Base M5 de la campagne générale :** `c465100202370d7b3246bb2cd656a435fbff4fc6`.  
**HEAD contenant le correctif B2, le test M5-HW-POLL-01 et la couverture des égalités LRU/LFU/Adaptive :** `0b7314d3a9388ef528426cdb8e55e1c4831fc025`.  
**Campagne générale :** 24 tests matériels PASS, 0 FAIL, documentés initialement dans la campagne M4/M5.  
**Statut de la preuve ciblée B1 :** scénario dédié ajouté dans cette PR ; CI stable et MSRV verte sur le HEAD indiqué ; exécution sur RTX 5090 requise avant merge.

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

Les détails des procédures et sorties de ces 24 tests sont conservés dans le [rapport détaillé de la campagne M4/M5 sur la branche de validation](https://github.com/zenhit777/Claymore/blob/test/m4-m5-rtx5090-validation/docs/audits/m4-m5-rtx5090-hardware-validation.md). Le scénario HW-C6-01 existant attendait la complétion via `is_transfer_complete` avant de demander l'admission : il ne remplaçait donc pas le test ciblé demandé par la revue architecturale.

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

**Résultat : NOT RUN — exécution ciblée sur RTX 5090 requise sur le HEAD corrigé de cette PR (`0b7314d3a9388ef528426cdb8e55e1c4831fc025` ou son descendant documentaire).**

Aucun résultat n'est inféré de la campagne précédente. Après exécution, consigner ici le SHA exact, la commande, la sortie du test et sa durée.

## 4. Commandes reproductibles

Sous WSL2 :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# Compile et exécute les tests unitaires CUDA de la bibliothèque.
cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib

# Exécute les tests matériels ignorés, dont M5-HW-POLL-01.
cargo test -p claymore-memory --features cuda-driver --lib -- --ignored --nocapture --test-threads=1
```

La campagne générale du wrapper peut être rejouée avec :

```bash
cargo test -p claymore-cuda --features cuda-driver -- --ignored --nocapture --test-threads=1
```

## 5. Limitations et risques résiduels

- `compute-sanitizer` 2025.3.0 est installé mais inutilisable dans cet environnement WSL2/WDDM.
- Les erreurs réelles `cuEventQuery`, `cuStreamSynchronize` et `cuMemFree` n'ont pas été injectées sur le pilote ; l'utilisation de handles CUDA synthétiques contre le pilote réel n'est pas sûre.
- Le chemin de conservation de l'ownership et de la comptabilité lors d'un véritable échec de `cuMemFree` n'est donc pas démontré sur matériel.
- La campagne matérielle nominale n'est pas une preuve de comportement multi-thread ; le manager reste confiné au thread propriétaire.
- L'injection déterministe et les tests manager-level avec backend CUDA simulable sont différés dans l'issue [#19](https://github.com/zenhit777/Claymore/issues/19).

## 6. Verdict

**Campagne nominale générale :** `HARDWARE VALIDATION PASSED WITH LIMITATIONS` (24/24).  
**Critère matériel ciblé B1 :** `PENDING` tant que M5-HW-POLL-01 n'a pas été exécuté avec succès sur RTX 5090.

La PR #17 ne doit pas être considérée comme matériellement validée au regard de B1 avant l'ajout de la preuve du test ciblé sur son HEAD final.
