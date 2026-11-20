# Validation matérielle M1 — RTX 5090

**Branche :** `feat/m1-hardware-foundation`  
**Pull Request :** [#5](https://github.com/zenhit777/Claymore/pull/5)  
**SHA validé :** `a3ff0fd10eaf4bfc41da7d1714ffbf9ca9eea9ba`  
**Date de validation :** 20 novembre 2026  
**Rapport source :** [commentaire OhMyPi du 19 novembre 2026](https://github.com/zenhit777/Claymore/pull/5#issuecomment-7024930803)

## 1. Objet

Ce document consigne les preuves de la validation matérielle réelle de la fondation CUDA M1 de Claymore sur une NVIDIA GeForce RTX 5090, sous WSL2.

Il distingue explicitement :

- ce qui a été effectivement exécuté et validé ;
- ce qui reste à éprouver ou hors du périmètre M1.

Les preuves sont versionnées dans le dépôt afin de ne pas dépendre uniquement d'un commentaire GitHub.

## 2. Environnement de validation

| Élément | Valeur observée |
|---|---|
| Hôte | Windows `10.0.26300.9974` + WSL2 |
| WSL2 | version `3.0.5.0` |
| Noyau Linux | `6.18.58.1-microsoft-standard-WSL2` |
| GPU | NVIDIA GeForce RTX 5090, bus `0000:01:00.0` |
| Compute Capability | `12.0` (sm_120) |
| VRAM | `32 607 MiB` (nvidia-smi) / `34 162 016 256` octets (CUDA Driver) |
| Pilote NVIDIA | KMD `616.64` |
| CUDA UMD | `13.4` |
| NVCC | `13.0.48` |
| Rust | `rustc 1.100.0` |
| Cargo | `cargo 1.100.0` |
| Bibliothèque CUDA utilisée | `/usr/lib/wsl/lib/libcuda.so.1` |

Le binaire de test a été contrôlé avec `ldd` :

```text
libcuda.so.1 => /usr/lib/wsl/lib/libcuda.so.1
```

Le test a donc été lié et exécuté contre le pilote NVIDIA de la RTX 5090, et non contre une implémentation CPU ou simulée.

## 3. Référence du code validé

Le code validé est le contenu du SHA `a3ff0fd10eaf4bfc41da7d1714ffbf9ca9eea9ba`.

Les neuf fichiers modifiés par la PR ont été comparés octet par octet avec les blobs Git de ce SHA dans l'environnement de validation :

- `.github/workflows/rust.yml` ;
- `crates/claymore-cuda/src/driver.rs` ;
- `crates/claymore-cuda/src/lib.rs` ;
- `crates/claymore-memory/src/cuda.rs` ;
- `docs/Claymore.md` ;
- `docs/audits/m1-hardware-foundation-audit.md` ;
- `docs/memory-cuda-integration-v0.5.md` ;
- `docs/memory-profile.md` ;
- `docs/wsl2.md`.

Les neuf comparaisons ont réussi. La validation porte donc sur le contenu exact du commit, même si le checkout Git direct n'était pas disponible dans l'environnement de validation.

Les corrections documentaires et le présent rapport ont été ajoutés après cette validation ; ils ne modifient ni le code CUDA ni la logique mémoire.

## 4. Résultats matériels

| Propriété vérifiée | Preuve | Résultat |
|---|---|---|
| Détection de la RTX 5090 | nom retourné par le driver : `NVIDIA GeForce RTX 5090`, ordinal 0 | OK |
| Compute Capability | `CudaDeviceInfo::compute_capability = Some((12, 0))` | OK |
| Partage du contexte primaire | deux instances de `CudaDriver` sur le device 0 : `primary_context_shared=true` | OK |
| Allocation et libération pinned | allocation puis `cuMemFreeHost` via `release` | OK |
| Allocation et libération device | allocation puis `cuMemFree_v2` via `release` | OK |
| Transfert H2D | 1 Mio depuis un buffer pinned source vers un buffer device | OK |
| Transfert D2H | 1 Mio du buffer device vers un buffer pinned distinct prérempli à `0xAA` | OK |
| Intégrité du payload | readback identique au motif source sur `1 048 576` octets | OK |
| Retour à zéro de l'utilisation | usage VRAM et pinned après libération : `0` | OK |
| Récupération de la VRAM | VRAM libre avant et après : `32 429 309 952` octets | OK |

Le test matériel officiel `cuda_manager_reports_device_memory` a réussi :

```text
running 1 test
test cuda::tests::cuda_manager_reports_device_memory ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out
```

### Précision sur la preuve H2D/D2H

Le test officiel utilise le même buffer pinned comme source H2D et destination D2H, avec les octets attendus déjà écrits avant les transferts. Il ne permet donc pas, à lui seul, de distinguer un D2H réel d'un D2H no-op.

La preuve indépendante ci-dessus utilise deux buffers pinned distincts, des motifs différents et une charge de 1 Mio. Elle lève cette ambiguïté et valide réellement les deux sens de transfert ainsi que l'intégrité des données.

## 5. Résultats logiciels

| Vérification | Résultat |
|---|---|
| `cargo fmt --all -- --check` | succès |
| `cargo test --workspace --all-targets` | 7 tests réussis, 0 échec |
| `cargo clippy --workspace --all-targets -- -D warnings` | succès |
| `cargo clippy -p claymore-memory -p claymore-cli --features cuda-driver --all-targets -- -D warnings` | succès |
| `cargo check -p claymore-memory --features cuda-driver` | succès |
| `cargo check -p claymore-cli --features cuda-driver` | succès |
| Test matériel `cuda_manager_reports_device_memory` | 1 réussi, 0 échec, 8 filtrés |

### Tests unitaires CUDA non ignorés

Commande exécutée sur la cible matérielle :

```bash
RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib
```

Résultat exact :

```text
running 3 tests
test driver::tests::resources_must_share_the_same_context ... ok
test driver::tests::zero_sized_allocations_are_rejected_before_driver_calls ... ok
test tests::disabled_backend_reports_actionable_error ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

running 9 tests
test cuda::tests::cuda_manager_reports_device_memory ... ignored, requires a CUDA driver and a usable NVIDIA device
test cuda::tests::logical_ticks_must_be_monotonic ... ok
test cuda::tests::release_guard_rejects_tick_before_safe_horizon ... ok
test tests::admission_succeeds_without_eviction_when_under_limit ... ok
test tests::confirm_eviction_releases_accounted_capacity ... ok
test tests::detects_duplicate_allocations ... ok
test tests::in_flight_allocations_are_never_selected ... ok
test tests::lru_plan_selects_oldest_idle_allocation ... ok
test tests::rejects_invalid_budget_relationships ... ok

test result: ok. 8 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
```

Bilan : **11 tests réussis, 0 échec, 1 ignoré**. Le test ignoré est le test matériel `cuda_manager_reports_device_memory`, exécuté séparément avec `--ignored`.

Cette commande couvre notamment :

- `ensure_same_context` via `resources_must_share_the_same_context` ;
- `as_size` via `zero_sized_allocations_are_rejected_before_driver_calls` ;
- `validate_monotonic_tick` via `logical_ticks_must_be_monotonic` ;
- `ensure_release_safe` via `release_guard_rejects_tick_before_safe_horizon`.

### Avertissements

La compilation avec la feature `cuda-driver` produit un warning préexistant :

```text
warning: unused import: `super::*`
  --> crates/claymore-cuda/src/lib.rs:61:9
```

Quelques avertissements Cargo de manifeste subsistent également (dépendances inutilisées, champ `repository`). Aucun de ces avertissements n'est un échec de test ou de lint bloquant. Aucun test n'a été modifié pour obtenir ces résultats.

## 6. Limites explicites

### Compute Sanitizer

Compute Sanitizer ne peut pas s'initialiser sous WSL2 dans cet environnement :

```text
Error: Failed to initialize WDDM debugger interface.
Please run EnableDebuggerInterface.bat as an administrator
Error: Device not supported.
```

Le test lui-même a réussi pendant la tentative, mais l'instrumentation memcheck n'a pas fonctionné. Aucune validation de sûreté mémoire au niveau Compute Sanitizer n'est revendiquée.

### Chemin fail-closed

Le double échec `event.record` puis `synchronize` n'est pas déclenché par les tests matériels. Le comportement fail-closed est couvert par la revue de code, pas par une injection de faute.

### Moteur tensoriel Claymore

Le moteur tensoriel et son raccordement au backend CUDA sont hors périmètre M1. Ils relèvent des jalons ultérieurs et ne sont pas validés par ce document.

### Performances

Aucun benchmark de performance n'a été exécuté. Aucune conclusion de débit, de latence ou d'efficacité mémoire n'est tirée de cette validation.

### Portée des tests

La CI standard ne lie ni n'exécute les tests unitaires qui nécessitent un driver CUDA ; elle vérifie la compilation et Clippy de la feature `cuda-driver` sans GPU. Les tests matériels et les tests unitaires CUDA non ignorés ont été exécutés manuellement sur la cible RTX 5090.

## 7. Synthèse

### Effectivement validé

- détection de la RTX 5090 ;
- compute capability `(12, 0)` ;
- partage du contexte primaire entre deux `CudaDriver` ;
- allocation et libération de buffers pinned et device ;
- transfert H2D et D2H de 1 Mio avec deux buffers pinned distincts ;
- intégrité du payload ;
- retour à zéro de l'utilisation VRAM et pinned ;
- récupération de la VRAM libre ;
- tests unitaires CUDA non ignorés ;
- tests workspace, formatage, Clippy et `cargo check` avec feature CUDA.

### Restant à éprouver ou hors périmètre

- validation memcheck via Compute Sanitizer ;
- injection de faute sur le chemin fail-closed ;
- raccordement du moteur tensoriel Claymore au backend CUDA ;
- benchmarks de performance ;
- exécution de kernels CUDA et de modèles ;
- backend simulé de l'adaptateur mémoire ;
- polling non bloquant des events.

## 8. Reproductibilité

Commandes de validation :

```bash
RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo test -p claymore-memory --features cuda-driver -- --ignored

RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib

cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings

RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo clippy -p claymore-memory -p claymore-cli --features cuda-driver --all-targets -- -D warnings

RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo check -p claymore-memory --features cuda-driver

RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
cargo check -p claymore-cli --features cuda-driver
```

Diagnostics matériels :

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
nvcc --version
```

## 9. Références

- PR #5 : https://github.com/zenhit777/Claymore/pull/5
- Rapport matériel OhMyPi : https://github.com/zenhit777/Claymore/pull/5#issuecomment-7024930803
- Audit M1 : `docs/audits/m1-hardware-foundation-audit.md`
- Référence produit : `docs/Claymore.md`
