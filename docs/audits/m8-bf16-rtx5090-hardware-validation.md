# Validation matérielle M8 — BF16 sur NVIDIA RTX 5090

- **Date de campagne :** 2026-11-22, 20:22–20:28 (Europe/Paris, UTC+01:00).
- **Dépôt / PR :** `zenhit777/Claymore`, PR #24, branche `feat/m8-tensor-foundation`.
- **SHA de code testé :** `a092f306581fddcba2cda32272bd685a660fbcad`
  (`feat(m8): native BF16 tensor path with FP32 accumulation`).
- **Statut :** **PASS** — 43/43 tests matériels par campagne, deux campagnes consécutives (86/86 exécutions), 0 échec, 0 test désactivé ou assoupli pour la campagne. Tous les tenseurs BF16 comparés CPU/CUDA présentent un écart de **0 pas représentable** dans les deux campagnes. Les 35 tests matériels F32 préexistants ont été rejoués et réussis.
- **Périmètre :** primitives tensorielles F32 et BF16 (add, matmul rank-2, softmax dernier axe, RMSNorm avec scale), dispatch, budget/gardes VRAM, events CUDA réels et exécuteur scheduler. Aucun checkpoint, aucun graphe de modèle, aucune boucle de génération et aucune mesure de performance modèle ne sont revendiqués.

> Ce rapport est attaché au SHA `a092f306581fddcba2cda32272bd685a660fbcad`. Les rapports F32 antérieurs (`m8-rtx5090-hardware-validation.md`, SHA `f0b034fd`) et M4/M5/M7/M7.1 restent rattachés à leurs propres SHA.

> **Validation finale de la PR #24 :** la section **§11** est rattachée au SHA
> `8fc3ca882ede2538bdf4106ba6a4ef14c4fbba25`. Les sections 1 à 10 ci-dessous
> sont historiques et restent rattachées au SHA `a092f30…`. Aucun résultat
> matériel n'est réattribué d'un SHA à un autre.

---

## 1. Traçabilité et préparation

Le travail de code a été commuté avant le relevé d'environnement et la campagne. Le journal brut enregistre, pour chaque campagne, la sortie de `git rev-parse HEAD` et `git status --short --branch`. Aucun fichier de code (Rust, CUDA, PTX, manifestes) n'était modifié dans l'arbre de travail ; seul ce rapport, rédigé après les exécutions, pouvait apparaître comme modifié. La branche locale était en avance de deux commits sur `origin` (les commits sont poussés après la campagne). Les logs ne montrent aucun `#[ignore]` retiré, aucun filtre et aucune tolérance modifiée entre les deux exécutions.

Le stub `ci/cuda-driver-stub.c` n'a servi qu'aux contrôles logiciels sans GPU (§4). Les campagnes matérielles ont utilisé le vrai driver NVIDIA via `/usr/lib/wsl/lib` et ont réellement chargé les deux modules PTX, créé des streams/events et lancé les kernels sur le device.

**Cohérence source ↔ PTX vérifiée avant campagne :** régénération des deux PTX avec la chaîne CUDA locale puis `diff -q` contre les artefacts versionnés (les deux `diff` sortent en code 0) ; empreintes dans le log `m8-bf16-02-cuda-feature-ci.log.txt`.

**État de la branche avant campagne :** le clone local utilisé pour la préparation était en retard de deux commits documentaires par rapport à `origin/feat/m8-tensor-foundation` (les commits `f156506` et `99e44b5`, qui livrent l'audit réel des graphes cibles). Une première campagne avait été exécutée sur un SHA antérieur ; le travail a été **rebasé** sur ces commits avant la campagne définitive, puis toutes les vérifications et les deux campagnes ont été rejouées sur le SHA `a092f306` ci-dessus. Le code Rust, les sources CUDA et les PTX sont bit à bit identiques entre les deux SHA ; seuls des documents ont changé. L'audit authentique `docs/audits/m8-model-graph-requirements.md` est donc bien présent dans l'arbre testé.

## 2. Environnement

### 2.1 GPU, driver, toolkit et Rust

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell |
| Compute capability | `12.0` |
| VRAM (`nvidia-smi`) | 32 607 MiB (30 833 MiB libres au relevé) |
| NVIDIA-SMI | `615.65.07` |
| KMD / Driver Version | `616.64` |
| CUDA UMD Version | `13.4` |
| Bus-Id | `00000000:01:00.0` |
| CUDA toolkit | `Cuda compilation tools, release 13.0, V13.0.48` |
| PTX intégré | `.version 9.0`, `.target sm_120` (les deux modules) |
| Rust stable | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Rust MSRV | `rustc 1.86.0 (05f9846f8 2025-03-31)` |
| Cargo stable / MSRV | `1.100.0` / `1.86.0` |
| Bibliothèque driver | `/usr/lib/wsl/lib/libcuda.so`, `libcuda.so.1` |
| Hôte | WSL2, noyau `6.18.58.1-microsoft-standard-WSL2`, 112 GiB RAM |

Log brut : `docs/audits/m8-bf16-00-environment.log.txt`.

## 3. Commandes exécutées

Contrôles logiciels obligatoires (log `m8-bf16-01-software-checks.log.txt`) :

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 clippy --workspace --all-targets -- -D warnings
```

Toutes sortent en code 0. 131 tests unitaires workspace réussis sous Rust stable et sous Rust 1.86. Les avertissements Cargo `unused dependency` / `unused field repository` sont préexistants et ne sont pas des erreurs `-D warnings` du compilateur.

Feature CUDA sans GPU et compilation avec le vrai driver (log `m8-bf16-02-cuda-feature-ci.log.txt`) :

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
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo +1.86 check -p claymore-memory --features cuda-driver --all-targets --locked
RUSTFLAGS="-L native=/usr/lib/wsl/lib" LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  cargo +1.86 check -p claymore-cli --features cuda-driver --locked

for dtype in f32 bf16; do
  nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
    -o /tmp/m8_${dtype}_regen.ptx crates/claymore-cuda/kernels/m8_${dtype}.cu
  diff -q /tmp/m8_${dtype}_regen.ptx crates/claymore-cuda/src/kernels/m8_${dtype}.ptx
done
```

Découverte des tests matériels (43 tests ignorés, log run 1) :

```bash
RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
cargo test --workspace --features cuda-driver --locked -- --ignored --list
```

Campagne matérielle — commande exacte, exécutée deux fois de suite sans modification du code :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --workspace --features cuda-driver --locked -- --ignored --test-threads=1 --nocapture
```

Répartition de la découverte : 3 tests driver CUDA, 5 intégration driver, 2 manager mémoire, 17 mémoire/transfert/éviction M4–M7.1, 8 tenseurs matériels F32 M8, **8 tenseurs matériels BF16 M8** — total 43.

## 4. Résultats

### 4.1 Opérations BF16 (8 tests, exécutés deux fois)

| Test | Périmètre vérifié | Run 1 | Run 2 |
|---|---|---|---|
| `hw_m8_bf16_01_add_matches_cpu_for_multiple_shapes_and_special_values` | add BF16 ; `[1]`, `[7]`, `[2,3]`, `[64,257]`, `[3,5,7]` ; zéro signé, sous-normal, normal, ±inf, NaN | PASS | PASS |
| `hw_m8_bf16_02_matmul_matches_cpu_for_rank2_shapes` | matmul row-major ; `2x3x2`, `16x64x32`, `129x17x65`, `7x9x5` (dimensions impaires, non multiples du tile 16×16) | PASS | PASS |
| `hw_m8_bf16_03_softmax_is_stable_for_large_logits` | softmax dernier axe ; `[1,3]`, `[8,257]`, `[1,1024]`, `[2,3,64]`, `[5,7,3]` ; logits `±10 000` ; somme par ligne ≈ 1 et finitude | PASS | PASS |
| `hw_m8_bf16_04_rms_norm_matches_cpu_with_explicit_bf16_scale` | RMSNorm ; `[4]`, `[6,64]`, `[2,1,129]`, `[3,5,7]` ; epsilon `0`, `1e-5`, `1e-3` ; scale BF16 explicite | PASS | PASS |
| `hw_m8_bf16_05_dispatch_refusals_and_typed_errors` | refus FP16 (allocation et dispatch), dtype mixte, shape mismatch, layout stridé, `SELECT cuda-native` / `REFUSE cpu-reference` | PASS | PASS |
| `hw_m8_bf16_06_allocation_transfer_lifetime_and_budget_guards` | 2 octets/élément, comptabilité avant/pendant/après, libération, snapshot driver, budget dur 8 KiB refusant 200 Ko | PASS | PASS |
| `hw_m8_bf16_07_kernel_event_guards_and_explicit_wait` | lancement BF16, event réel, gardes `IN_FLIGHT`, attente explicite avant lecture | PASS | PASS |
| `hw_m8_bf16_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler` | `Scheduler` + `QueueRunner` sur file `Compute`, dépendances, tenseurs possédés par l'exécuteur | PASS | PASS |

Durée cumulée de la cible `hardware_cuda_tensor` (16 tests F32+BF16) : 2,45 s (run 1) et 2,40 s (run 2).

### 4.2 Campagnes F32 et antérieures rejouées (35 tests)

Les 35 tests matériels préexistants ont été rejoués dans les deux campagnes : 3/3 driver, 5/5 intégration driver, 2/2 manager, 17/17 mémoire/transfert/éviction, 8/8 tenseurs M8 F32. Aucun échec. Durée de la cible `hardware_cuda_memory` : 26,82 s (run 1) et 27,02 s (run 2).

**Total : 43/43 par campagne, 86/86 sur les deux campagnes.**

## 5. Politique numérique et écarts observés

La comparaison BF16 est définie en **pas représentables** : les motifs 16 bits sont convertis en clé monotone (`bits ^ 0x8000` si positif, `!bits` sinon), deux valeurs BF16 adjacentes différant d'exactement 1. Les NaN doivent être présents des deux côtés ; les infinis sont comparés comme valeurs ordonnées. Cette politique est codée dans les tests et documentée dans `docs/m8-architecture.md` ; elle a été fixée **avant** l'exécution.

| Opération | Limite autorisée | Écart maximal observé (run 1 = run 2) |
|---|---:|---|
| add | 0 pas | **0 pas** sur les 6 cas, y compris valeurs spéciales |
| matmul | 1 pas | **0 pas** sur les 4 shapes |
| softmax | 1 pas | **0 pas** sur les 5 shapes |
| rms_norm | 1 pas | **0 pas** sur les 4 shapes |
| event add / scheduler add | 0 pas | **0 pas** |

Aucune tolérance n'a été élargie après exécution. Les résultats sont plus stricts que la limite autorisée : l'élargissement FP32 des entrées BF16 est exact, l'accumulation FP32 suit le même ordre des deux côtés (`mul_add` / `fmaf`) et l'unique arrondi de sortie est RNE (`Bf16::from_f32` / `__float2bfloat16_rn`). Le PTX BF16 généré contient bien `cvt.rn.bf16.f32` et `fma.rn.f32` (contrôle unitaire).

La campagne F32 conserve ses tolérances historiques : add `1e-5/1e-6`, matmul `1e-3/2e-5`, softmax `1e-6/1e-5`, rms_norm `1e-5/2e-5`. Écarts maximaux observés dans les deux campagnes : add et matmul `0e0` ; softmax `2,98e-8` (`[8,257]`) et `4,44e-16` (`[2,3,64]`) ; rms_norm `4,77e-7` (`[2,1,129]`) et `3,58e-7` (`[6,64]`). Les logits `±10 000` produisent des probabilités finies et des sommes par ligne BF16 à `4e-3` près de 1, sans `NaN` ni débordement.

## 6. Mémoire, durée de vie et événements

- VRAM driver : 34 162 016 256 octets au total ; 32 429 309 952 octets libres pendant les tests. `nvidia-smi` : 32 607 MiB au total, 30 833 MiB libres au relevé, 1 355 MiB utilisés.
- Comptabilité BF16 (`hw_m8_bf16_06`) : 0 octet avant, 16 384 octets après deux tenseurs de 4 096 BF16, 24 576 octets au pic avec la sortie, retour à 0 après libération des trois tenseurs. La libération backend est confirmée avant le retrait comptable.
- Scénario de pression : budget VRAM dur de 8 KiB, demande de 200 000 octets (100 000 BF16) refusée avant allocation physique.
- `nvidia-smi` après chacune des deux campagnes : 1 353 MiB utilisés, soit le même niveau que le relevé initial à la fluctuation près (sessions Xwayland), sans fuite VRAM observable.
- `hw_m8_bf16_07` confirme que les trois allocations d'une opération restent protégées par un event réel jusqu'à `wait_for_transfer`. Le backend attend l'event de son propre stream de calcul ; aucun `cuCtxSynchronize` global.
- `hw_m8_bf16_08` confirme que le scheduler ne possède aucun tenseur et que la completion n'est signalée qu'après l'attente de l'event CUDA.

## 7. Problèmes rencontrés

Aucun échec matériel ni logiciel n'a été observé au SHA testé. Les points suivants ont été corrigés **avant** le SHA de code :

1. Le clone local était en retard sur `origin` de deux commits documentaires (audit des graphes). Le travail a été rebasé dessus et l'intégralité des vérifications rejouée ; le rapport n'est rattaché qu'au SHA réellement exécuté après rebase.
2. Les noms de backend étaient dtype-spécifiques (`*-f32`) alors que les capacités couvrent désormais F32 et BF16 ; ils sont devenus `cpu-reference` et `cuda-native`, avec mise à jour des tests existants (comportement inchangé).
3. Les constantes de test de conversion BF16 ont été corrigées : `f32::powi(-133)` sous-échoue en FP32 et devait être remplacé par une construction bit à bit exacte ; un cas de tie sous-normal attendait le mauvais motif (RNE donne le motif pair `0x0002`).

Aucune tolérance matérielle n'a été touchée.

## 8. Logs bruts et empreintes

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `docs/audits/m8-bf16-00-environment.log.txt` | GPU, driver, toolkit, Rust, hôte, hachages noyaux | `9b859c97fcd29b996e03541eacec5b1441e6e725bf2b7c82cf2bb09129bae50f` |
| `docs/audits/m8-bf16-01-software-checks.log.txt` | fmt, tests workspace, Clippy, MSRV | `5dcb3b50f372c1d922e38d0579bf9900cc33f60b49ccbb4b51512ee2708dfd00` |
| `docs/audits/m8-bf16-02-cuda-feature-ci.log.txt` | stub sans GPU, feature CUDA, MSRV, régénération PTX | `b0b31d6084ab439e9e5f60af0291a17983705653e6084d63c7952df825fc2a3f` |
| `docs/audits/m8-bf16-03-hardware-run1.log.txt` | découverte + campagne matérielle 1 | `ff108692c3ba99f1cd077be92ab021e3ced12c83dbb710906cb84a9b16e945be` |
| `docs/audits/m8-bf16-04-hardware-run2.log.txt` | campagne matérielle 2 + `nvidia-smi` final | `ff3e26a7c7f1e33b1b220fcb6f27fa1b2c83d6b8f44b209ea67483482bbf0614` |

Artefacts de kernel :

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m8_f32.cu` | `5d20ae0ed15affb35ef1c63ef33fc391b094490d6e3d1d499f6891eabbfb33fc` |
| `crates/claymore-cuda/kernels/m8_bf16.cu` | `a276fe54e34b54903591f18f2a777fdf7cae1f11bd64498b47df54a96273dc41` |
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |
| `crates/claymore-cuda/src/kernels/m8_bf16.ptx` | `886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153` |

## 9. Limitations restantes

- **Précisions :** F32 et BF16 uniquement. FP16 reste non supporté, sans référence CPU ni kernel ; son type dans `DType` n'est qu'une métadonnée.
- **Opérations :** jeu limité à add, matmul rank-2, softmax dernier axe et RMSNorm avec scale explicite ; pas de broadcasting, de vue stridée en entrée, d'activation, d'attention, de transpose ou d'opération sur place.
- **Perf :** matmul sans tiling partagé ni Tensor Core ; softmax/RMSNorm avec un bloc par ligne ; aucun benchmark ni comparaison cuBLAS/cuDNN (M12).
- **Asynchronisme :** chaque opération attend son event CUDA ; sûreté des durées de vie mais pas de recouvrement transfert/calcul.
- **Compatibilité :** PTX `sm_120` uniquement, JIT driver PTX 9.0 / CUDA 13+ ; aucune architecture plus ancienne n'est testée.
- **Injection de fautes :** pas de faute injectée sur l'échec de lancement/kernel tensoriel ; le chemin d'erreur reste fail-closed par construction (gardes maintenus).
- **Modèles :** aucun checkpoints chargé, aucun graphe DiffusionGemma/Clover exécuté ; les primitives propres aux modèles relèvent de M9/M10/M11.
- **Contexte :** GPU partagé avec le compositeur WSL2 (Xwayland, ~1,3 GiB résidents) ; les mesures de mémoire libre ne représentent pas un environnement dédié.

## 10. Conclusion

Au SHA `a092f306581fddcba2cda32272bd685a660fbcad`, le socle M8 fournit quatre opérations tensorielles en F32 et en BF16, avec représentation 16 bits exacte, conversion FP32 explicite RNE, accumulation FP32 et intégration complète au memory orchestrator et au scheduler. Les huit kernels CUDA (PTX `sm_120`) ont été réellement exécutés sur la RTX 5090 et comparés à la référence CPU lors de deux campagnes consécutives : 43/43 par campagne, 86/86 au total, écart BF16 nul sur toutes les shapes. Les contrôles logiciels, la feature CUDA sans GPU, Clippy et le MSRV 1.86 sont verts au SHA testé. La PR #24 reste ouverte et n'est pas fusionnée.

---

## 11. Validation finale de la PR #24 — SHA `8fc3ca882ede2538bdf4106ba6a4ef14c4fbba25`

### 11.1 Code réellement exécuté et traçabilité

| Élément | Valeur |
|---|---|
| SHA du code exécuté | `8fc3ca882ede2538bdf4106ba6a4ef14c4fbba25` |
| Branche | `feat/m8-tensor-foundation` |
| Arbre de travail | propre avant et après chaque campagne (`git status --short` vide dans les logs) |
| Campagne 1 | 2026-11-22T21:05:16Z → 21:05:49Z, `SHA_BEFORE` = `SHA_AFTER` = `8fc3ca8…`, exit 0 |
| Campagne 2 | 2026-11-22T21:05:49Z → 21:06:21Z, `SHA_BEFORE` = `SHA_AFTER` = `8fc3ca8…`, exit 0 |

Aucune modification de code, de source CUDA, de PTX, de manifeste ou de tolérance n'a eu lieu entre les deux campagnes. Ce rapport, ses logs bruts et la mise à jour de la description de la PR ont été produits **après** les exécutions ; le SHA ci-dessus désigne le code exécuté, pas le commit documentaire final. Les résultats des sections 1 à 10 ne sont pas réattribués à ce SHA.

### 11.2 Environnement observé

| Élément | Valeur observée |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, Blackwell, compute capability `12.0` |
| VRAM | 32 607 MiB au total, 30 809 MiB libres au relevé |
| NVIDIA-SMI / KMD / UMD | `615.65.07` / `616.64` / `13.4` |
| CUDA toolkit | `nvcc` release 13.0, `V13.0.48`, build `cuda_13.0.r13.0/compiler.36260728_0` |
| PTX intégrés | `.version 9.0`, `.target sm_120` (F32 et BF16) |
| Rust stable | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| MSRV | `rustc 1.86.0 (05f9846f8 2025-03-31)` |
| Hôte | WSL2, noyau `6.18.58.1-microsoft-standard-WSL2`, 112 GiB RAM |
| Bibliothèque driver | `/usr/lib/wsl/lib/libcuda.so` (vrai driver, pas le stub) |

Log brut : `m8-final-00-environment.log.txt`. État après campagnes : `m8-final-04-post-campaign.log.txt`.

### 11.3 Commandes intégrales exécutées

Contrôles logiciels obligatoires (log `m8-final-01-software-checks.log.txt`) :

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings

export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo check -p claymore-memory --features cuda-driver --all-targets --locked
cargo check -p claymore-cli --features cuda-driver --locked
cargo clippy -p claymore-cuda -p claymore-memory -p claymore-cli \
  --features cuda-driver --all-targets --locked -- -D warnings
cargo +1.86 check -p claymore-cuda --features cuda-driver --all-targets --locked

cc -shared -fPIC ci/cuda-driver-stub.c -o /tmp/claymore-cuda-stub/libcuda.so
RUSTFLAGS="-L native=/tmp/claymore-cuda-stub" \
LD_LIBRARY_PATH=/tmp/claymore-cuda-stub \
  cargo test -p claymore-memory --features cuda-driver --lib --locked
RUSTFLAGS="-L native=/tmp/claymore-cuda-stub" \
LD_LIBRARY_PATH=/tmp/claymore-cuda-stub \
  cargo test -p claymore-cuda --features cuda-driver --lib --locked
```

Campagne matérielle, commande exacte exécutée deux fois de suite (logs `m8-final-02-*` et `m8-final-03-*`) :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --workspace --features cuda-driver --locked -- \
  --ignored --test-threads=1 --nocapture
```

### 11.4 Résultats des contrôles logiciels

Les **14/14** commandes ci-dessus sont sorties en code 0. Workspace : **137 tests** réussis sous Rust stable et sous Rust 1.86 (dont 39 tests `claymore-tensor`, 4 nouveaux tests de propriété BF16 couvrant les 65 536 motifs). Stub sans GPU : 50 tests `claymore-memory` et 8 tests `claymore-cuda` réussis. `cargo fmt --check` et tous les Clippy `-D warnings` sont verts, y compris avec `cuda-driver`. Les avertissements Cargo `unused dependency` et `unused field repository` sont préexistants et ne sont pas des diagnostics `-D warnings`.

Régression corrigée avant validation : les quatre commits de correction initiaux (`1996ec7`…`95460c9`) ne compilaient pas sous `cuda-driver` (`Vec<str>` inféré dans `launch_tensor_op`), échouaient à `cargo fmt --check` et faisaient échouer un test de capacité devenu obsolète après l'ajout du contrôle d'alignement. Ces trois défauts ont été corrigés dans le commit `8fc3ca8`, sans écraser les corrections existantes.

### 11.5 Campagnes matérielles — 52 tests par campagne

| Cible | Run 1 | Run 2 |
|---|---:|---:|
| `claymore-cuda` driver (3) | 3/3 | 3/3 |
| Intégration driver CUDA (5) | 5/5 | 5/5 |
| `claymore-memory` lib, dont **7 injections de fautes B2** (9) | 9/9 | 9/9 |
| Matériel mémoire/transfert/éviction M4–M7.1 (17) | 17/17 | 17/17 |
| Tenseurs matériels M8, dont **2 tests B1** (18) | 18/18 | 18/18 |
| **Total** | **52/52** | **52/52** |

**104/104 exécutions**, 0 échec, aucun test ignoré supplémentaire retiré, aucun filtre, aucune tolérance modifiée. Durées : 33 s (run 1) et 32 s (run 2), sans recours à un backend simulé : les deux modules PTX sont chargés et les kernels réellement lancés sur la RTX 5090.

### 11.6 Écarts numériques réellement observés

| Opération | Tolérance historique (inchangée) | Écart maximal run 1 = run 2 |
|---|---|---|
| add F32 | `1e-5 / 1e-6` | `0e0` (4 shapes) |
| matmul F32 | `1e-3 / 2e-5` | `0e0` (3 shapes) |
| softmax F32 | `1e-6 / 1e-5` | `2,98e-8` `[8,257]`, `4,44e-16` `[2,3,64]` |
| rms_norm F32 | `1e-5 / 2e-5` | `3,58e-7` `[6,64]`, `4,77e-7` `[2,1,129]` |
| add / matmul / softmax / rms_norm BF16 | 0 pas (add), 1 pas (autres) | **0 pas** sur toutes les shapes et valeurs spéciales |
| event add F32/BF16, scheduler F32/BF16 | 0 / 0 pas | `0e0` / 0 pas |

Les logits `±10 000` restent finis ; les sommes par ligne BF16 restent à `4e-3` près de 1. Aucune tolérance n'a été élargie. La différence résiduelle F32 provient de l'ordre des réductions ( CPU séquentiel vs arbre en mémoire partagée 256 threads) et de `rcp.rn.f32` contre division ; l'`exp` CUDA est `ex2.approx.ftz.f32` sans fast-math global, l'hôte utilise `libm`. Détail dans `m8-cpu-cuda-numerics.md`.

### 11.7 B1 — offsets, alignement et capacité

- `Tensor::with_layout` rejette tout offset non aligné sur la taille d'élément (`MisalignedOffset`) ; l'offset nul, l'offset F32 aligné de 8 octets et l'offset BF16 aligné de 2 octets sont acceptés, y compris dans les tests CPU.
- `launch_tensor_op` applique `offset_bytes` au pointeur de chaque entrée après vérification `offset + span_bytes <= buffer.len()`. Test matériel `hw_m8_b1_input_offsets_are_honored_for_f32_and_bf16` : des sentinelles `0x5A`/`0xDD` précèdent les tenseurs dans l'allocation ; le noyau lit bien les valeurs décalées (`[11, 22, 33, 44]` en F32, `[1,5, 3,5, 5,5, 7,5]` en BF16) et les octets hors span restent intact. Sans prise en compte de l'offset, le noyau aurait lu les sentinelles.
- `hw_m8_b1_byte_span_and_capacity_guards` vérifie le rejet `BufferTooSmall { requested: 16, capacity: 8 }` côté entrée **et** côté sortie, ainsi que le rejet d'un offset 3 pour un layout F32.
- Les sorties sont écrites à la base de leur allocation : `TensorOp` ne porte pas de vue de sortie. Cette limite est explicite dans le contrat de `launch_tensor_op` et dans `docs/Claymore.md` §6.5.
- Cohérence CPU/CUDA : la même validation `Tensor::with_layout` est utilisée par les deux chemins ; les huit opérations F32/BF16 ont été rejouées avec offset nul dans les 16 tests tenseurs matériels.

### 11.8 B2 — injection de fautes dans `launch_tensor_op`

Sept tests au niveau du **gestionnaire réel** (driver, streams, events, allocations réels), avec codes d'erreur injectés uniquement (`cfg(test)`, thread-local, remis à zéro après chaque test) :

| Test | Faute injectée | Résultat observé |
|---|---|---|
| `injected_create_event_failure_leaves_every_allocation_releasable` | `cuEventCreate` échoue | aucune garde posée, aucune allocation bloquée, 3 libérations réussies, usage VRAM 0 |
| `injected_launch_failure_with_successful_sync_releases_every_allocation` | `cuLaunchKernel` échoue, sync OK | horizons inchangés, 0 garde, 3 libérations réussies |
| `injected_launch_failure_with_sync_error_poisons_and_pins_all_allocations` | `cuLaunchKernel` + `cuStreamSynchronize` échouent | `poisoned`, 3 allocations à `IN_FLIGHT`, toute libération refusée par `ManagerPoisoned` |
| `injected_event_record_failure_with_successful_sync_lifts_every_guard` | `cuEventRecord` échoue, sync OK | gardes levées à `tick`, 0 event en attente, 3 libérations réussies |
| `injected_event_record_failure_with_sync_error_preserves_gpu_guards` | `cuEventRecord` + sync échouent | `poisoned`, gardes conservées à `IN_FLIGHT`, comptabilité inchangée |
| `injected_partial_guard_failure_with_successful_sync_lifts_only_installed_guards` | 2e garde échoue, sync OK | seule la 1re garde est levée, les 2 autres gardent leur horizon, 3 libérations réussies |
| `injected_partial_guard_failure_with_sync_error_poisons_and_pins_all_allocations` | 3e garde + sync échouent | `poisoned`, 3 allocations à `IN_FLIGHT`, libérations refusées |

La récupération est désormais centralisée dans `recover_launch_failure` : sync réussie → gardes levées ; sync non confirmée → empoisonnement et maintien de toutes les gardes. Les sept tests passent dans les deux campagnes. Aucune allocation potentiellement utilisée par le GPU n'est libérée dans les scénarios d'échec de synchronisation.

### 11.9 Mémoire, events et scheduler

- Driver : 34 162 016 256 octets VRAM au total, 32 429 309 952 libres pendant les tests.
- Comptabilité F32 (`hw_m8_06`) : 0 avant, 49 152 octets au pic, 0 après libération ; BF16 (`hw_m8_bf16_06`) : 0 avant et 0 après. La libération backend est confirmée avant retrait comptable.
- Échec de budget : 8 Kio refusant 200 000 octets demandés, sans allocation physique.
- Events : les trois allocations d'une opération restent protégées jusqu'à `wait_for_transfer` ; aucun `cuCtxSynchronize` global. Scheduler : dépendances exécutées, tenseurs possédés par l'exécuteur, results disponibles seulement après l'event.
- `nvidia-smi` après campagnes : 1 379 MiB utilisés, niveau comparable au relevé initial (1 379 MiB) ; aucune fuite VRAM observable. Log : `m8-final-04-post-campaign.log.txt`.

### 11.10 Logs bruts versionnés et empreintes

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `docs/audits/m8-final-00-environment.log.txt` | environnement, SHA, versions | `aff1bebbb99b03f930b458f2cf437bd24ffb9c51c24ed8a2d1291ba5163db37e` |
| `docs/audits/m8-final-01-software-checks.log.txt` | 14 contrôles logiciels, sorties intégrales | `8595b28eca5fe81e008bf9ccc179ed3f93a11e9e1cc10a877de8ebd2021b5cd5` |
| `docs/audits/m8-final-02-hardware-run1.log.txt` | campagne 1 intégrale, SHA avant/après | `c2be591cd90f8259d80c53a6c535b9ed816ecd564f5d6cc4f43a8b48a2dc178f` |
| `docs/audits/m8-final-03-hardware-run2.log.txt` | campagne 2 intégrale, SHA avant/après | `d6acc5fb7cb7baefab13292dcc65079a790f29fbfc9f6c7befd6efa1f4f64e37` |
| `docs/audits/m8-final-04-post-campaign.log.txt` | état GPU après campagnes | `6327ebaebc59b948bc7f0c1521ff82afd707d4eddbd911d5623f9ee10ad96b47` |

Artefacts de kernel (inchangés, voir `m8-ptx-provenance.md`) :

| Fichier | SHA-256 |
|---|---|
| `crates/claymore-cuda/kernels/m8_f32.cu` | `5d20ae0ed15affb35ef1c63ef33fc391b094490d6e3d1d499f6891eabbfb33fc` |
| `crates/claymore-cuda/kernels/m8_bf16.cu` | `a276fe54e34b54903591f18f2a777fdf7cae1f11bd64498b47df54a96273dc41` |
| `crates/claymore-cuda/src/kernels/m8_f32.ptx` | `e2cf50f5edf18b0f5b64086a077c9aa53aa405b1adab28249e8b79f9594a6dde` |
| `crates/claymore-cuda/src/kernels/m8_bf16.ptx` | `886ede72b76b6bdbd64e03588cdda3011f2e1e1a08534fd24ee52c576dfa4153` |

### 11.11 Limites restantes après validation

- Les injections de fautes simulent des codes d'erreur Driver API sur le gestionnaire réel ; elles ne simulent pas une panne matérielle ni une corruption GPU.
- Aucun recouvrement transfert/calcul n'est revendiqué ; chaque opération attend son event (M12).
- F32/BF16 uniquement, quatre primitives sur tenseurs contigus, sans broadcasting, vues stridées en entrée, Tensor Core ni benchmark (M12).
- PTX `sm_120` uniquement (JIT PTX 9.0 / CUDA 13+).
- Aucun checkpoint DiffusionGemma ou Clover n'est chargé ni exécuté : la compatibilité complète avec ces modèles n'est pas revendiquée ; les opérations propres aux modèles relèvent de M9/M10/M11.

---

## 12. Campagne de suivi après les correctifs des limites de lancement CUDA — SHA `93522028066be1dda91940d648983e89008f5e7c`

### 12.1 Statut et traçabilité

Cette section constitue une **campagne de suivi distincte**, exécutée après les correctifs des limites de grille CUDA. Elle ne modifie pas et ne réattribue pas les résultats historiques des §1 à 11, qui restent rattachés à leurs propres SHA.

| Élément | Valeur observée |
|---|---|
| Dépôt / PR | `zenhit777/Claymore`, [PR #24](https://github.com/zenhit777/Claymore/pull/24) |
| Branche | `feat/m8-tensor-foundation` |
| SHA de code impératif testé | `93522028066be1dda91940d648983e89008f5e7c` |
| Commit de code | `style(cuda): format grid limit assertions` |
| SHA local avant campagne 1 | `93522028066be1dda91940d648983e89008f5e7c` |
| SHA local après campagne 1 | `93522028066be1dda91940d648983e89008f5e7c` |
| SHA local avant campagne 2 | `93522028066be1dda91940d648983e89008f5e7c` |
| SHA local après campagne 2 | `93522028066be1dda91940d648983e89008f5e7c` |
| Statut | **PASS** — 52/52 tests matériels par campagne, 104/104 exécutions au total, 0 échec |

Le clone local était initialement en retard de cinq commits sur la branche distante. Après `git fetch --prune origin feat/m8-tensor-foundation`, la branche distante a été vérifiée au SHA imposé et la branche locale a été avancée uniquement par fast-forward. L'arbre de travail était propre avant l'environnement et avant la campagne 1. Aucun fichier Rust, CUDA `.cu`, PTX, manifeste, lockfile, tolérance, test ou workflow CI n'a été modifié.

### 12.2 Dates et environnement matériel réel

| Élément | Valeur réellement observée |
|---|---|
| Fenêtre de campagne | 2026-11-22, 22:34:53–22:40:27 (Europe/Paris, UTC+01:00) |
| Campagne 1 | 22:35:30–22:36:04 ; code retour `0` |
| Campagne 2 | 22:36:37–22:37:10 ; code retour `0` |
| Hôte | WSL2, Ubuntu 24.04.5 LTS, noyau `6.18.58.1-microsoft-standard-WSL2` |
| GPU | NVIDIA GeForce RTX 5090, UUID `GPU-7556cae7-89a8-6267-b8b4-4e2edb9981d6` |
| Architecture / Compute Capability | Blackwell / `12.0` (`sm_120`) |
| VRAM totale | 32 607 MiB selon `nvidia-smi` ; 34 162 016 256 octets selon le driver pendant les tests |
| NVIDIA-SMI / version driver / CUDA UMD | `615.65.07` / `616.64` / `13.4` |
| CUDA toolkit | `nvcc` release 13.0, `V13.0.48`, build `cuda_13.0.r13.0/compiler.36260728_0` |
| Bibliothèque driver réelle | `/usr/lib/wsl/lib/libcuda.so` (`libcuda.so.1`) ; aucun stub utilisé |
| PTX intégré | `.version 9.0`, `.target sm_120` pour les modules F32 et BF16 |
| Rust stable | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0` |
| Rust 1.86 / MSRV installé | `rustc 1.86.0 (05f9846f8 2025-03-31)`, `cargo 1.86.0` |

Le pilote NVIDIA `616.64`, son CUDA UMD `13.4` et le toolkit CUDA `13.0.48` satisfont l'exigence CUDA 13 ou supérieur et la compatibilité PTX 9.0. Le vrai driver WSL2 a été utilisé : les tests ont imprimé le nom `NVIDIA GeForce RTX 5090`, la compute capability `12.0`, créé un contexte primaire, des streams, des events et des allocations, puis chargé et lancé les deux modules PTX.

Log brut : `docs/audits/m8-followup-00-environment.log.txt`.

### 12.3 Commande exécutée

Les deux campagnes ont utilisé exactement la même commande, sans filtre, exclusion ou tolérance supplémentaire :

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --workspace --features cuda-driver --locked -- --ignored --test-threads=1 --nocapture
```

`CARGO_TERM_COLOR=never` a été exporté uniquement pour conserver une sortie texte non colorée dans les journaux. Aucun test n'a été supprimé ou désactivé ; `--ignored`, `--test-threads=1` et `--nocapture` sont restés inchangés. Les sorties intégrales, les horodatages, les SHA avant/après, les mesures VRAM et les codes retour figurent dans :

- `docs/audits/m8-followup-01-hardware-run1.log.txt` ;
- `docs/audits/m8-followup-02-hardware-run2.log.txt` ;
- `docs/audits/m8-followup-03-post-campaign.log.txt`.

### 12.4 Résultats matériels

| Cible | Campagne 1 | Campagne 2 |
|---|---:|---:|
| Driver CUDA et contexte primaire | 3/3 | 3/3 |
| Intégration driver CUDA : streams, events, buffers | 5/5 | 5/5 |
| Gestionnaire mémoire CUDA et injections de fautes | 9/9 | 9/9 |
| Transferts, évictions, scheduler et exécuteur CUDA | 17/17 | 17/17 |
| Tenseurs matériels M8 F32/BF16, offsets et capacités | 18/18 | 18/18 |
| **Total** | **52/52** | **52/52** |

Les deux code retour Cargo sont `0`. Le total est **104 tests réussis, 0 échec, 0 test ignoré supplémentaire, 0 filtre modifié**. Les campagnes couivent le driver et le contexte primaire, les streams et events, le gestionnaire mémoire, les transferts et évictions, le scheduler et l'exécuteur CUDA, les opérations F32 et BF16, les offsets, l'alignement, les contrôles de capacité, les durées de vie et les protections d'allocation.

### 12.5 Résultats numériques F32/BF16

Les métriques numériques imprimées par les deux campagnes sont **strictement identiques** ; la comparaison automatisée du journal post-campagne retourne `NUMERIC_METRICS_IDENTICAL=True`.

| Opération | Tolérance existante, inchangée | Écart maximal observé dans les deux campagnes |
|---|---:|---|
| add F32 | `1e-5 / 1e-6` | `0e0` sur 4 shapes |
| matmul F32 | `1e-3 / 2e-5` | `0e0` sur 3 shapes |
| softmax F32 | `1e-6 / 1e-5` | `2,9802322e-8` (`[8,257]`) ; `4,440892e-16` (`[2,3,64]`) |
| rms_norm F32 | `1e-5 / 2e-5` | `3,5762787e-7` (`[6,64]`) ; `4,7683716e-7` (`[2,1,129]`) |
| add BF16 | 0 pas | **0 pas** sur 5 shapes et valeurs spéciales |
| matmul BF16 | 1 pas | **0 pas** sur 4 shapes |
| softmax BF16 | 1 pas | **0 pas** sur 5 shapes |
| rms_norm BF16 | 1 pas | **0 pas** sur 3 shapes |
| event / scheduler F32 | 0 pas / `0e0` | `0e0` dans les deux campagnes |
| event / scheduler BF16 | 0 pas | 0 pas dans les deux campagnes |

Les offsets B1 restent exacts : F32 `offset=8` produit `[11, 22, 33, 44]` et BF16 `offset=2` produit `[1.5, 3.5, 5.5, 7.5]`. La comptabilité F32 est `before=0`, `peak=49152`, `after=0`; la comptabilité BF16 est `before=0`, `after=0`. Les valeurs `±10 000` de softmax restent finies. Aucune tolérance n'a été élargie ni réécrite.

### 12.6 Comptabilité VRAM et état CUDA

Les snapshots `nvidia-smi` sont les suivants :

| Instant | Utilisé | Libre | Total |
|---|---:|---:|---:|
| Avant campagne 1 | 1 387 MiB | 30 801 MiB | 32 607 MiB |
| Après campagne 1 / avant campagne 2 | 1 387 MiB | 30 801 MiB | 32 607 MiB |
| Après campagne 2 | 1 387 MiB | 30 801 MiB | 32 607 MiB |
| Snapshot post-campagne final | 1 374 MiB | 30 814 MiB | 32 607 MiB |

La variation mesurée entre les deux campagnes est donc **0 MiB utilisé**. Le snapshot final est 13 MiB inférieur au niveau initial, ce qui correspond à la fluctuation du processus graphique Xwayland de WSL2 et non à une fuite CUDA ; le driver ne montre aucun processus de calcul résiduel dans `nvidia-smi`. Les tests d'éviction, de libération et de durée de vie confirment également le retour de la comptabilité interne à zéro.

### 12.7 Contrôles de limites de lancement CUDA

L'inspection du code au SHA testé confirme les points suivants :

1. `crates/claymore-tensor/src/contracts.rs` définit `CUDA_GRID_X_MAX = 2_147_483_647` et `CUDA_GRID_Y_MAX = 65_535`.
2. `cuda_grid_limit` calcule les blocs par axe selon le kernel : `ceil(numel/256)` pour elementwise, `ceil(n/16)` et `ceil(m/16)` pour matmul, `rows` pour softmax/RMSNorm.
3. Le dispatcher refuse le backend CUDA avec le motif typé `UnsupportedReason::LaunchLimit { axis, blocks, maximum }`. Le refus est explicite dans `Display` et ne peut pas se déguiser en simple absence de capacité.
4. `crates/claymore-cuda/src/compute.rs` refait le contrôle dans `grid_for(total, block, maximum)` et retourne `CudaError::InvalidLaunchConfiguration` avant chaque `cuLaunchKernel`. Aucun lancement hors limites n'est transmis au driver par les huit kernels F32/BF16.
5. La configuration à la limite `grid.y = 65_535` est acceptée par le test logiciel `grid_rounding_and_axis_limits_are_explicit`; le dispatcher refuse `grid.y = 65_536` et `grid.x = 2_147_483_648` avec un motif typé. Les tests logiciels sont inclus dans `cargo test --workspace --all-targets --locked`, commande exécutée par la CI stable et MSRV. Ils n'ont pas été remplacés par une simulation matérielle.
6. Les configurations ordinaires acceptables continuent de fonctionner : les 18 tests matériels tensoriels F32/BF16 passent dans chaque campagne.
7. Les tests d'injection récupérable (`launch`, `event_record`, `partial_guard` avec synchronisation réussie) lèvent les gardes installées au tick courant ; aucune allocation ne reste `IN_FLIGHT` après une erreur récupérable. Les scénarios où la synchronisation échoue conservent au contraire les gardes et empoisonnent le gestionnaire, conformément au fail-closed.

### 12.8 Empreintes des journaux

| Fichier | SHA-256 |
|---|---|
| `docs/audits/m8-followup-00-environment.log.txt` | `01fadd844fb9e74a355b1eacc9b8a4fd1e185a068fb91a022a33dcd38f03746d` |
| `docs/audits/m8-followup-01-hardware-run1.log.txt` | `ada5451d80c082c291e91411bbe89be4b667936a66185b96f2751b4887112200` |
| `docs/audits/m8-followup-02-hardware-run2.log.txt` | `ec8def5b8f26805ad81d588bcf57b7ab9e20db3190026878f52bf07b3a1fc243` |
| `docs/audits/m8-followup-03-post-campaign.log.txt` | `a034caa5d34c659cb0e732c39bf10d7310135c0479af809829b340b32973dd99` |

Les empreintes des sources CUDA, des PTX et des deux Rust corrigés sont également consignées dans le journal post-campagne. Les quatre SHA-256 des fichiers `.cu` et `.ptx` sont identiques à ceux de la validation §11, confirmant qu'aucun kernel ni PTX n'a été régénéré ou remplacé.

### 12.9 Problèmes rencontrés et conclusion

- Le seul problème de préparation a été le clone local initialement en retard de cinq commits sur `origin/feat/m8-tensor-foundation`. La branche distante a été récupérée, vérifiée au SHA impératif, puis avancée par fast-forward depuis un arbre propre avant toute campagne. Aucun résultat n'a été attribué au SHA initial `d7d1523…`.
- Cargo a affiché les avertissements préexistants `unused field repository` et `unused dependency` ; ils ne constituent ni une erreur de compilation ni un échec de test.
- Aucun test exécuté n'a échoué. Aucun événement, allocation ou contexte CUDA n'a été laissé dans un état incohérent par les campagnes ou par les erreurs récupérables testées.

Au SHA `93522028066be1dda91940d648983e89008f5e7c`, les deux campagnes matérielles consécutives ont exécuté **52/52 tests chacune, 104/104 au total**, sur une véritable NVIDIA RTX 5090 sous WSL2 avec le pilote NVIDIA et PTX 9.0, sans modification de code entre elles. **M8 est matériellement validé à ce SHA.** Cette campagne ne fusionne pas la PR #24 et ne revendique aucune extension au-delà du périmètre M8 décrit dans ce rapport.
