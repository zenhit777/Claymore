# Validation matérielle M8 — BF16 sur NVIDIA RTX 5090

- **Date de campagne :** 2026-11-22, 20:22–20:28 (Europe/Paris, UTC+01:00).
- **Dépôt / PR :** `zenhit777/Claymore`, PR #24, branche `feat/m8-tensor-foundation`.
- **SHA de code testé :** `a092f306581fddcba2cda32272bd685a660fbcad`
  (`feat(m8): native BF16 tensor path with FP32 accumulation`).
- **Statut :** **PASS** — 43/43 tests matériels par campagne, deux campagnes consécutives (86/86 exécutions), 0 échec, 0 test désactivé ou assoupli pour la campagne. Tous les tenseurs BF16 comparés CPU/CUDA présentent un écart de **0 pas représentable** dans les deux campagnes. Les 35 tests matériels F32 préexistants ont été rejoués et réussis.
- **Périmètre :** primitives tensorielles F32 et BF16 (add, matmul rank-2, softmax dernier axe, RMSNorm avec scale), dispatch, budget/gardes VRAM, events CUDA réels et exécuteur scheduler. Aucun checkpoint, aucun graphe de modèle, aucune boucle de génération et aucune mesure de performance modèle ne sont revendiqués.

> Ce rapport est attaché au SHA `a092f306581fddcba2cda32272bd685a660fbcad`. Les rapports F32 antérieurs (`m8-rtx5090-hardware-validation.md`, SHA `f0b034fd`) et M4/M5/M7/M7.1 restent rattachés à leurs propres SHA.

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
