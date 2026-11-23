# Validation matérielle finale M8 — limites de grille CUDA et récupération B2 sur NVIDIA RTX 5090

- **Date de campagne :** 2026-11-23, 02:58–03:02 (Europe/Paris, UTC+01:00).
- **Dépôt / PR :** `zenhit777/Claymore`, [PR #24](https://github.com/zenhit777/Claymore/pull/24), branche `feat/m8-tensor-foundation`.
- **SHA de code validé :** `ea3e3d53a1d976c32bd28e086eb68905648c991a`.
- **Commit de code :** `style(cuda): allow explicit recovery context arguments`.
- **Statut :** **PASS** — **53/53 tests matériels par campagne**, **106/106 exécutions cumulées**, **0 échec**, deux campagnes consécutives sans modification du code.
- **Périmètre :** validation matérielle finale des réserves de lancement CUDA (borne exacte `grid.y = 65 535`, premier dépassement `grid.y = 65 536`, refus typé du dispatcher, contrôle défensif avant `cuLaunchKernel`, absence de soumission hors limites), réexécution complète des tests B2 d'injection de fautes, et revalidation de l'ensemble des 53 tests matériels M8/M7/M4–M5 sur le vrai driver NVIDIA.

> **Traçabilité des SHA.** Le SHA `ea3e3d53a1d976c32bd28e086eb68905648c991a` est le SHA du **code exécuté** pendant les deux campagnes. Le commit documentaire qui ajoute ce rapport et les journaux sous `docs/audits/` est un successeur distinct sur la branche ; il ne contient que des documents et ne participe pas à la validation matérielle. Aucun résultat matériel n'est réattribué à un autre SHA.

---

## 1. Traçabilité et préparation

Le clone local était initialement positionné sur `7fda2a0d9c0831780738d94e8c488550e1d2cd72`, soit cinq commits derrière la branche distante. Après `git fetch origin --prune`, le SHA cible `ea3e3d53a1d976c32bd28e086eb68905648c991a` a été vérifié comme tip exact de `origin/feat/m8-tensor-foundation`. La branche locale a été avancée **uniquement par fast-forward** (`git merge --ff-only origin/feat/m8-tensor-foundation`). La campagne n'a jamais été lancée sur un autre SHA.

| Contrôle | Valeur observée |
|---|---|
| SHA exigé | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA local après récupération | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA distant `origin/feat/m8-tensor-foundation` | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA avant campagne 1 | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA après campagne 1 | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA avant campagne 2 | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| SHA après campagne 2 | `ea3e3d53a1d976c32bd28e086eb68905648c991a` |
| `git status --short` avant/après chaque campagne | vide |
| `git diff --exit-code -- crates .github Cargo.toml Cargo.lock` | code retour `0` |
| `git diff --check` | code retour `0` |
| Empreinte de l'arbre source à `HEAD` | `af1946a3cd00d5afe336ac8a0165f9e5fec475b7155ea678f2d3a764e1b2f273` |

L'empreinte ci-dessus couvre `git ls-tree -r HEAD -- crates Cargo.toml Cargo.lock .github ci config`. Aucun fichier Rust, CUDA `.cu`, PTX, manifeste, lockfile, tolérance numérique, test ou workflow CI n'a été modifié entre les deux campagnes. Après les campagnes, seuls les journaux et ce rapport ont été ajoutés sous `docs/audits/` ; ils n'étaient pas présents pendant l'exécution.

Journaux de traçabilité :

- `docs/audits/m8-reservations-00-environment.log.txt` ;
- `docs/audits/m8-reservations-00b-software-checks.log.txt` ;
- `docs/audits/m8-reservations-00c-test-discovery.log.txt` ;
- `docs/audits/m8-reservations-01-run1.log.txt` ;
- `docs/audits/m8-reservations-02-run2.log.txt` ;
- `docs/audits/m8-reservations-03-post-campaign.log.txt`.

## 2. Environnement matériel et logiciel

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| UUID GPU | `GPU-7556cae7-89a8-6267-b8b4-4e2edb9981d6` |
| Architecture / compute capability | Blackwell / `12.0` (`sm_120`) |
| VRAM totale | 32 607 MiB (`nvidia-smi`) ; 34 162 016 256 octets côté driver pendant les tests |
| Bus-Id | `00000000:01:00.0` |
| NVIDIA-SMI | `615.65.07` |
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| CUDA toolkit | `nvcc` release 13.0, `V13.0.48`, build `cuda_13.0.r13.0/compiler.36260728_0` |
| PTX intégré | `.version 9.0`, `.target sm_120` pour F32 et BF16 |
| SHA-256 PTX F32 | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |
| SHA-256 PTX BF16 | `886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153` |
| Bibliothèque driver réelle | `/usr/lib/wsl/lib/libcuda.so` → `libcuda.so.1` ; aucun stub logiciel |
| SHA-256 `libcuda.so.1.1` | `697a5878b7b6320dcb5cfe1bc652d4b88359147d6864ab54dafacc64493aed9a` |
| Hôte | WSL2, Ubuntu 24.04.5 LTS, noyau `6.18.58.1-microsoft-standard-WSL2` |
| Rust stable | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0` |
| Rust MSRV | `rustc 1.86.0 (05f9846f8 2025-03-31)`, `cargo 1.86.0` |
| RAM hôte | 112 GiB |

Les exports utilisés pour toutes les commandes CUDA sont :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Le vrai driver WSL2 a été utilisé : les tests ont affiché `NVIDIA GeForce RTX 5090`, `cc=12.0`, créé un contexte primaire, des streams, des events et des allocations, puis chargé et lancé les deux modules PTX. Aucun test matériel n'a été exécuté contre `ci/cuda-driver-stub.c`.

## 3. Commandes réellement exécutées

### 3.1 Vérifications logicielles préalables

Toutes les commandes suivantes ont été exécutées sous les exports ci-dessus et sont sorties avec le code `0` (journal `m8-reservations-00b-software-checks.log.txt`) :

| Commande | Code retour |
|---|---:|
| `cargo fmt --all -- --check` | 0 |
| `cargo test --workspace --all-targets --locked` | 0 |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | 0 |
| `cargo +1.86 test --workspace --all-targets --locked` | 0 |
| `cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings` | 0 |
| `cargo check -p claymore-cuda --features cuda-driver --all-targets --locked` | 0 |
| `cargo check -p claymore-memory --features cuda-driver --all-targets --locked` | 0 |
| `cargo check -p claymore-cli --features cuda-driver --locked` | 0 |
| `cargo clippy -p claymore-cuda -p claymore-memory -p claymore-cli --features cuda-driver --all-targets --locked -- -D warnings` | 0 |

Les suites unitaires workspace totalisent **138 tests réussis** sous Rust stable et **138 tests réussis** sous Rust 1.86. Les avertissements Cargo `unused dependency` / `unused field repository` sont préexistants et ne sont pas des erreurs de compilation ou de Clippy avec `-D warnings`.

### 3.2 Découverte des tests matériels

```bash
cargo test --workspace --features cuda-driver --locked -- --ignored --list
```

Résultat : **53 tests ignorés découverts** (journal `m8-reservations-00c-test-discovery.log.txt`).

### 3.3 Campagnes matérielles

La commande exacte, identique pour les deux campagnes, sans filtre ni exclusion supplémentaire :

```bash
cargo test --workspace --features cuda-driver --locked -- \
  --ignored --test-threads=1 --nocapture
```

Campagne 1 : `2026-11-23T01:59:37Z` → `2026-11-23T02:00:10Z`, code retour `0`.
Campagne 2 : `2026-11-23T02:00:23Z` → `2026-11-23T02:00:55Z`, code retour `0`.

Aucun `#[ignore]` n'a été retiré, aucun test n'a été filtré, aucune tolérance numérique n'a été modifiée, et aucune modification de code n'a eu lieu entre les deux campagnes.

### 3.4 Vérifications post-campagne

```bash
git rev-parse HEAD
git status --short
git diff --stat
git diff --check
git diff --exit-code -- crates .github Cargo.toml Cargo.lock
git ls-tree -r HEAD -- crates Cargo.toml Cargo.lock .github ci config | sha256sum
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
dmesg | grep -i -E "xid|nvrm|nvidia.*(error|fail)"
```

Toutes les vérifications de propreté et de SHA sont vertes. `nvidia-smi` ne montre aucun processus de calcul résiduel. Le filtre `dmesg` ne retourne aucune entrée Xid/NVRM.

## 4. Nombre de tests découverts

| Cible | Tests | Détail |
|---|---:|---|
| `claymore-cuda` (lib) | 3 | tests driver matériels : buffer, synchronisation de stream, streams non bloquants |
| `hardware_cuda_driver` | 5 | `hw_c1_01` à `hw_c2_02` : contexte, streams, events, allocations, bornes |
| `claymore-memory` (lib) | 9 | 2 tests matériels du gestionnaire + **7 tests B2 d'injection de fautes** |
| `hardware_cuda_memory` | 17 | 15 tests M4–M7 + 2 tests M7 (exécuteur CUDA et cache d'experts) |
| `hardware_cuda_tensor` | 19 | 9 tests M8 F32 (`hw_m8_01` à `hw_m8_09`) + 2 tests B1 + 8 tests M8 BF16 |
| **Total** | **53** | |

Le total précédent était de **52 tests matériels**. La seule différence est l'ajout du nouveau test `hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary`, soit **52 + 1 = 53**. Aucun autre test n'a été ajouté, retiré ou désactivé.

Présence vérifiée dans la découverte :

- `hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary` ;
- les sept tests B2 : `injected_create_event_failure_leaves_every_allocation_releasable`, `injected_launch_failure_with_successful_sync_releases_every_allocation`, `injected_launch_failure_with_sync_error_poisons_and_pins_all_allocations`, `injected_event_record_failure_with_successful_sync_lifts_every_guard`, `injected_event_record_failure_with_sync_error_preserves_gpu_guards`, `injected_partial_guard_failure_with_successful_sync_lifts_only_installed_guards`, `injected_partial_guard_failure_with_sync_error_poisons_and_pins_all_allocations`.

## 5. Résultats détaillés des deux campagnes

### 5.1 Synthèse par cible

| Cible | Campagne 1 | Campagne 2 |
|---|---:|---:|
| `claymore-cuda` (lib) | 3/3 | 3/3 |
| `hardware_cuda_driver` | 5/5 | 5/5 |
| `claymore-memory` (lib, dont B2) | 9/9 | 9/9 |
| `hardware_cuda_memory` | 17/17 | 17/17 |
| `hardware_cuda_tensor` | 19/19 | 19/19 |
| **Total** | **53/53** | **53/53** |

**Cumul : 106/106 tests réussis, 0 échec, 0 test ignoré supplémentaire, 0 filtre modifié.** Les deux codes retour Cargo sont `0`.

### 5.2 Tests tensoriels M8 (19 tests)

| Test | Campagne 1 | Campagne 2 |
|---|---|---|
| `hw_m8_01_add_matches_cpu_for_multiple_shapes_and_signs` | PASS | PASS |
| `hw_m8_02_matmul_matches_cpu_for_rank2_shapes` | PASS | PASS |
| `hw_m8_03_softmax_is_stable_for_large_logits` | PASS | PASS |
| `hw_m8_04_rms_norm_matches_cpu_with_explicit_scale` | PASS | PASS |
| `hw_m8_05_dispatch_refusals_and_typed_errors` | PASS | PASS |
| `hw_m8_06_allocation_accounting_lifetime_and_budget_guards` | PASS | PASS |
| `hw_m8_07_kernel_event_guards_and_explicit_wait` | PASS | PASS |
| `hw_m8_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler` | PASS | PASS |
| `hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary` | PASS | PASS |
| `hw_m8_b1_input_offsets_are_honored_for_f32_and_bf16` | PASS | PASS |
| `hw_m8_b1_byte_span_and_capacity_guards` | PASS | PASS |
| `hw_m8_bf16_01_add_matches_cpu_for_multiple_shapes_and_special_values` | PASS | PASS |
| `hw_m8_bf16_02_matmul_matches_cpu_for_rank2_shapes` | PASS | PASS |
| `hw_m8_bf16_03_softmax_is_stable_for_large_logits` | PASS | PASS |
| `hw_m8_bf16_04_rms_norm_matches_cpu_with_explicit_bf16_scale` | PASS | PASS |
| `hw_m8_bf16_05_dispatch_refusals_and_typed_errors` | PASS | PASS |
| `hw_m8_bf16_06_allocation_transfer_lifetime_and_budget_guards` | PASS | PASS |
| `hw_m8_bf16_07_kernel_event_guards_and_explicit_wait` | PASS | PASS |
| `hw_m8_bf16_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler` | PASS | PASS |

### 5.3 Autres groupes matériels (34 tests)

Les 34 tests restants — 3 tests driver dans la lib CUDA, 5 tests d'intégration driver, 9 tests de la lib mémoire (2 matériels + 7 B2), et 17 tests mémoire/transfert/éviction/scheduler — sont tous sortis `ok` dans les deux campagnes. Les durées observées des cibles sont cohérentes : `hardware_cuda_memory` 26,97 s / 26,41 s et `hardware_cuda_tensor` 2,71 s / 2,58 s.

### 5.4 Déterminisme numérique

Les 45 lignes de métriques numériques et de comptabilité imprimées par les tests sont **strictement identiques** entre les deux campagnes (`NUMERIC_METRICS_IDENTICAL=True`, vérifié par `diff` dans le journal post-campagne). Aucune tolérance n'a été élargie, réécrite ou contournée.

## 6. Résultat du nouveau test de limites de grille

Test : `hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary`.

Sortie observée dans les deux campagnes :

```text
test hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary ... HW-M8 grid limit: exact boundary accepted; overflow rejected before kernel submission
ok
```

Le test valide les quatre points suivants :

1. **Borne exacte acceptée :** une matrice de forme `1 048 560 × 1` (soit `65 535 × 16` lignes de tuile) produit `grid.y = 65 535` et reste planifiable par le dispatcher (`dispatcher.plan(&boundary).is_ok()`).
2. **Premier dépassement refusé :** une matrice de forme `1 048 561 × 1` (soit `65 536 × 16` lignes) produit `grid.y = 65 536` et est refusée par le dispatcher avec `DispatchError::NoCapableBackend`, contenant le motif typé `UnsupportedReason::LaunchLimit { axis: "y", blocks: 65_536, maximum: 65_535 }`.
3. **Contrôle défensif du chemin de lancement :** l'appel direct à `launch_tensor_op` sur la même opération hors limites retourne une erreur de configuration contenant `65 536`/`65536`, avant toute soumission de kernel.
4. **Aucune fuite de comptabilité :** les allocations d'entrée, de sortie et la seconde entrée restent toutes comptabilisées après le refus ; aucun guard ni event résiduel n'est laissé derrière.

Les tests unitaires de contrat `cuda_launch_grid_limits_are_axis_specific_and_typed` et `dispatch_explains_an_exact_match` passent également dans les vérifications logicielles préalables. La configuration à la limite exacte `grid.y = 65 535` est donc acceptée, tandis que `grid.y = 65 536` est refusée de façon typée par le dispatcher puis défensivement par le chemin CUDA.

## 7. Résultats des tests d'injection de fautes B2

Les sept tests B2 pilotent le **vrai gestionnaire CUDA** (driver, streams, events et allocations réels) et injectent uniquement le résultat d'API en échec. Ils sont tous sortis `ok` dans les deux campagnes.

| Test B2 | Scénario injecté | Garantie vérifiée | Campagne 1 | Campagne 2 |
|---|---|---|---|---|
| `injected_create_event_failure_leaves_every_allocation_releasable` | `cuEventCreate` échoue | gestionnaire non empoisonné, aucune attente pendante, toutes les allocations restent libérables, comptabilité VRAM à zéro | PASS | PASS |
| `injected_launch_failure_with_successful_sync_releases_every_allocation` | `cuLaunchKernel` échoue, synchronisation réussie | gardes levées, horizons restaurés, toutes les allocations libérables, comptabilité à zéro | PASS | PASS |
| `injected_launch_failure_with_sync_error_poisons_and_pins_all_allocations` | lancement échoue puis `cuStreamSynchronize` échoue | gestionnaire empoisonné, toutes les allocations restent `IN_FLIGHT`, libérations refusées avec `ManagerPoisoned` | PASS | PASS |
| `injected_event_record_failure_with_successful_sync_lifts_every_guard` | `cuEventRecord` échoue, synchronisation réussie | gardes levées au tick courant, allocations libérables, comptabilité à zéro | PASS | PASS |
| `injected_event_record_failure_with_sync_error_preserves_gpu_guards` | `cuEventRecord` échoue puis synchronisation échoue | gestionnaire empoisonné, allocations `IN_FLIGHT`, libérations refusées | PASS | PASS |
| `injected_partial_guard_failure_with_successful_sync_lifts_only_installed_guards` | pose partielle des gardes échoue, synchronisation réussie | seule la garde installée est levée ; les gardes non installées conservent leur horizon ; allocations libérables | PASS | PASS |
| `injected_partial_guard_failure_with_sync_error_poisons_and_pins_all_allocations` | pose partielle échoue puis synchronisation échoue | gestionnaire empoisonné, toutes les allocations restent protégées, libérations refusées | PASS | PASS |

Le comportement **fail-closed** est donc conservé : une erreur récupérable ne lève les gardes qu'après confirmation de la synchronisation du stream ; si la synchronisation échoue, le gestionnaire est empoisonné et aucune allocation n'est libérée prématurément. Aucune allocation fuit et la comptabilité VRAM reste cohérente après chaque scénario.

## 8. État des allocations et de la comptabilité VRAM

Les tests de comptabilité M8 rapportent, dans les deux campagnes :

- F32 : `before=0`, `peak=49152` octets, `after=0` ;
- BF16 : `before=0`, `after=0` ;
- les tests de durée de vie, d'éviction et de libération explicite confirment le retour à zéro de la comptabilité interne ;
- les 128 cycles H2D/D2H et les alternances transfert/éviction se terminent sans allocation résiduelle ;
- aucun processus de calcul n'apparaît dans `nvidia-smi --query-compute-apps` après les campagnes.

## 9. État de la VRAM avant et après

| Instant | Utilisé | Libre | Total |
|---|---:|---:|---:|
| Relevé d'environnement initial | 1 471 MiB | 30 717 MiB | 32 607 MiB |
| Avant campagne 1 | 1 468 MiB | 30 720 MiB | 32 607 MiB |
| Après campagne 1 | 1 466 MiB | 30 722 MiB | 32 607 MiB |
| Avant campagne 2 | 1 466 MiB | 30 722 MiB | 32 607 MiB |
| Après campagne 2 | 1 466 MiB | 30 722 MiB | 32 607 MiB |
| Snapshot post-campagne final | 1 466 MiB | 30 722 MiB | 32 607 MiB |

La variation entre le début et la fin de la campagne 2 est de **0 MiB**. La variation de −2 MiB entre le relevé initial et la fin de la campagne 1 correspond à la fluctuation du processus graphique Xwayland de WSL2, pas à une fuite CUDA ; aucun processus de calcul résiduel n'est visible et la comptabilité interne est revenue à zéro. Aucune entrée Xid/NVRM n'a été relevée dans `dmesg`.

## 10. Politique numérique et écarts observés

Les tolérances existantes n'ont pas été modifiées. Les écarts maximaux observés sont identiques dans les deux campagnes :

| Opération | Tolérance existante, inchangée | Écart maximal observé (runs 1 et 2) |
|---|---:|---:|
| add F32 | `1e-5 / 1e-6` | `0e0` sur 4 formes |
| matmul F32 | `1e-3 / 2e-5` | `0e0` sur 3 formes |
| softmax F32 | `1e-6 / 1e-5` | `2,9802322e-8` (`[8,257]`) ; `4,440892e-16` (`[2,3,64]`) |
| rms_norm F32 | `1e-5 / 2e-5` | `3,5762787e-7` (`[6,64]`) ; `4,7683716e-7` (`[2,1,129]`) |
| add BF16 | 0 pas | **0 pas** sur 5 formes et valeurs spéciales |
| matmul BF16 | 1 pas | **0 pas** sur 4 formes |
| softmax BF16 | 1 pas | **0 pas** sur 5 formes |
| rms_norm BF16 | 1 pas | **0 pas** sur 3 formes |
| event / scheduler F32 | `0e0` | `0e0` |
| event / scheduler BF16 | 0 pas | 0 pas |

Les offsets B1 restent exacts : F32 `offset=8` produit `[11, 22, 33, 44]` et BF16 `offset=2` produit `[1.5, 3.5, 5.5, 7.5]`. Les valeurs softmax `±10 000` restent finies. Les 45 lignes de métriques sont bit à bit identiques entre les deux campagnes.

## 11. SHA-256 des journaux bruts

Les empreintes ci-dessous sont celles des fichiers livrés sous `docs/audits/`. Elles sont identiques aux empreintes des journaux bruts conservés dans `/tmp` avant copie.

| Fichier | SHA-256 |
|---|---|
| `docs/audits/m8-reservations-00-environment.log.txt` | `521c96cfb8c270acedf8d4632f57bc0534c2df4369202ff283aa5606b3b1009a` |
| `docs/audits/m8-reservations-00b-software-checks.log.txt` | `1fc0264f6e09262d64841890f53b76ad4c05466cd0bfc810826d5dd5218d4cff` |
| `docs/audits/m8-reservations-00c-test-discovery.log.txt` | `f6acdba54d0d5b2379d586cf41a0cbe62e3a1f84e87dc2ee5602a02ef90e4374` |
| `docs/audits/m8-reservations-01-run1.log.txt` | `8c2476e876ef5cb430a4fa7841e8eb88145ae0921f222c114ee2aacd5c106b6c` |
| `docs/audits/m8-reservations-02-run2.log.txt` | `3aa975e6b8e39a66125aa92bb6b60d5771c5730ef253b21db69e73c83de59c4e` |
| `docs/audits/m8-reservations-03-post-campaign.log.txt` | `f639ff68904ff351254db6343feda8c7cb078b9eb5ca4c2afe5bf5f5962213fc` |

## 12. Anomalies et limitations

1. **Clone local initialement en retard.** Le clone était positionné cinq commits derrière `origin`. Après `git fetch --prune`, le SHA cible a été vérifié comme tip exact de la branche distante et la branche locale a été avancée uniquement par fast-forward. Aucune campagne n'a été lancée sur le SHA antérieur.
2. **Avertissements Cargo préexistants.** Les avertissements `unused dependency` et `unused field repository` apparaissent dans les journaux mais ne sont pas des erreurs de compilation ni de Clippy avec `-D warnings` ; ils sont présents dans l'arbre de code validé et n'ont pas été modifiés pendant la campagne.
3. **VRAM de base non nulle.** WSL2 conserve deux processus Xwayland qui utilisent environ 1,4 GiB de VRAM. Les mesures absolues ne descendent donc pas à zéro ; la comparaison avant/après et l'absence de processus de calcul résiduel sont les indicateurs de non-fuite.
4. **Pas de runner GPU dans GitHub Actions.** La CI GitHub valide le format, les tests CPU, Clippy, la compilation CUDA et le MSRV, mais ne dispose pas de RTX 5090. La validation matérielle est donc locale et documentée par ces journaux.
5. **Commit documentaire séparé.** Le commit qui ajoute ces fichiers est un successeur du SHA de code validé et ne contient que des documents. Il déclenche une nouvelle exécution CI sur la branche ; le résultat de cette exécution documentaire est rapporté dans le commentaire de la PR #24 et ne se substitue pas au SHA de code validé.

## 13. Verdict

**PASS.**

Toutes les conditions de la mission sont remplies :

- SHA exact `ea3e3d53a1d976c32bd28e086eb68905648c991a` avant et après chaque campagne ;
- arbre de travail propre avant et après chaque campagne ;
- 53/53 tests réussis par campagne, soit 106/106 exécutions cumulées, 0 échec ;
- `hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary` réussi dans les deux campagnes ;
- sept tests B2 réussis dans les deux campagnes, avec garanties fail-closed conservées ;
- aucune modification de code, de test, de tolérance ou de configuration entre les campagnes ;
- comptabilité VRAM interne revenue à zéro, aucune fuite observable, aucun processus de calcul résiduel ;
- aucun test matériel supprimé, filtré ou ignoré artificiellement ;
- CI Rust #737 au SHA `ea3e3d5` : succès des deux jobs (`Format, test, and lint` et `Minimum supported Rust 1.86`), soit 4/4 check-runs sur les événements `push` et `pull_request`.

La PR #24 n'a **pas** été fusionnée. La validation définitive et la décision de merge restent à l'architecte.
