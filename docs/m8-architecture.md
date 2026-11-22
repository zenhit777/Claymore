# M8 — Moteur tensoriel natif Claymore

## Décision d'architecture

Claymore cible initialement deux modèles de diffusion : DiffusionGemma et Clover. Aucune stratégie autoregressive générique n'est dans le périmètre sans décision produit ultérieure.

M8 construit le socle tensoriel natif commun, en dessous des sémantiques des modèles. Il ne construit ni moteur autoregressif, ni moteur de génération diffusion, ni adaptateur modèle. Les jalons suivants sont M9 (orchestration de diffusion modulaire), M10 (adaptateur DiffusionGemma), M11 (adaptateur Clover), puis M12 (optimisation et validation de performance).

Le principe cible est **un seul runtime Claymore, un seul socle tensoriel et une orchestration de diffusion modulaire commune**, avec des points d'extension explicites. Les adaptateurs spécialisés conservent les graphes, états, règles numériques et comportements propres à chaque modèle ; ils ne deviennent pas des runtimes tensoriels concurrents.

## Responsabilités des couches

1. **claymore-core** : types transverses et configuration.
2. **claymore-device** : identité/capacités des devices, buffers opaques, allocations, copies, synchronisation et erreurs backend. Il ne définit pas une seconde représentation de Tensor.
3. **claymore-tensor** : propriétaire canonique des layouts, shape/strides, Tensor, TensorView, vérifications de bornes et contrats d'ownership. Les buffers restent opaques et sont fournis par claymore-device ; DType est partagé avec le backend.
4. **Opérations et kernels** : contrats explicites par opération (shapes, dtype, layouts, aliasing, précision et comportement numérique), implémentations CPU de référence puis kernels CUDA. Aucun cast, broadcasting, fallback CPU ou aller-retour host implicite.
5. **Runtime / mémoire / scheduler** : claymore-runtime, claymore-memory et claymore-scheduler conservent le cycle de vie des requêtes, l'admission mémoire, les dépendances et la protection des ressources en vol. Le scheduler ne possède pas les tenseurs.
6. **Orchestration de diffusion et adaptateurs** : hors M8. L'orchestration commune appartient à M9 ; les différences irréductibles sont portées par les adaptateurs M10/M11.

## Contrat Tensor

Un Tensor expose un TensorLayout validé : shape, strides mesurés en éléments, dtype, nombre d'éléments et byte-span calculés avec vérification d'overflow. Une vue emprunte le Tensor propriétaire, conserve l'offset en octets et ne peut dépasser le byte-span du parent. Les strides négatifs et dimensions nulles sont exclus du contrat initial. Toute extension doit définir explicitement les règles d'ownership, d'aliasing et d'opérations.

Le type Tensor canonique vit dans claymore-tensor. Les anciens types Tensor/TensorView de claymore-device ont été retirés pendant M8 afin d'éviter deux modèles de tenseurs concurrents. DType reste un contrat de représentation scalaire partagé avec le backend device.

## Opérations CPU de référence livrées dans cette tranche

Le crate claymore-tensor fournit un ensemble déterministe pour deux dtypes, F32 et BF16 :
- addition élément par élément, sans broadcasting ;
- multiplication matricielle row-major rank-2 ;
- softmax stable sur la dernière dimension ;
- RMSNorm sur la dernière dimension avec vecteur de scale explicite (du même dtype que l'entrée).

Ces opérations exigent des layouts contigus, un seul device instance et un dtype unique par opération. Elles ne convertissent pas silencieusement les types. FP16 reste **non supporté** : `DType::F16` est une métadonnée sans chemin de calcul. Ce jeu constitue un socle de référence, pas une affirmation qu'il couvre à lui seul les graphes complets de DiffusionGemma ou Clover. Les opérations supplémentaires relèvent des adaptateurs M10/M11 après le socle d'orchestration M9.

### Politique numérique BF16

`claymore-tensor::Bf16` est un newtype sur `u16` : 1 bit de signe, 8 bits d'exposant, 7 bits de mantice explicite. Les règles sont explicites et testées :

- `from_bits` / `to_bits` sont des vues bit à bit exactes ; l'élargissement BF16 → FP32 est exact pour tout motif 16 bits, y compris sous-normaux, zéro signé, infinis et NaN ;
- `from_f32` applique **round-to-nearest-even** : les 16 bits de poids faible sont abandonnés, le bit 15 est le bit d'arrondi et les bits 14..0 forment le bit collant ; une retenue peut incrémenter l'exposant (débordement vers l'infini, transition sous-normal → normal) ;
- tous les NaN sont canonicalisés en `0x7FC0` à l'aller ; au retour, un NaN BF16 signalant est rendu silencieux (`0x00400000` dans la mantice FP32) ;
- le zéro signé est préservé dans les deux sens.

Aucune arithmétique n'est implémentée sur `Bf16` : l'entrée est élargie en FP32, le calcul a lieu en FP32, puis le résultat final est arrondi une seule fois en BF16. Les quatre opérations suivent cette règle, y compris l'accumulation du matmul (`mul_add` côté CPU, `fmaf` côté CUDA, dans le même ordre d'index) et les réductions de softmax/RMSNorm. La sortie BF16 est donc always le résultat d'un seul arrondi RNE depuis FP32.

## État d'implémentation

- [x] Crate claymore-tensor intégré au workspace.
- [x] Layouts contigus/stridés, arithmétique de tailles vérifiée, Tensor propriétaire et TensorView empruntée.
- [x] Retrait des définitions Tensor concurrentes de claymore-device.
- [x] Opérations CPU F32 et BF16 de référence : add, matmul, softmax, RMSNorm.
- [x] Représentation BF16 exacte, conversion FP32 explicite et arrondi RNE documenté.
- [x] Contrats d'opérations formalisés (`contracts.rs` : descripteurs, contraintes dtype/layout/shape, aliasing en lecture seule + sortie fraîche, erreurs typées).
- [x] Dispatch explicite par capacités backend (`TensorDispatcher` : sélection déterministe, explication de chaque refus, aucun fallback ni cast implicite).
- [x] Allocation/sortie intégrée au memory orchestrator et au scheduler (`CudaTensorMemory`, `CudaMemoryManager::launch_tensor_op`, `CudaTensorComputeExecutor`) sans contourner budgets, gardes ni événements.
- [x] Kernels CUDA F32 et BF16 équivalents et tests différentiels CPU/CUDA (PTX `sm_120` intégrés).
- [x] Validation RTX 5090 F32 sur le SHA documenté dans `docs/audits/m8-rtx5090-hardware-validation.md`.
- [x] Validation RTX 5090 BF16 sur le SHA documenté dans `docs/audits/m8-bf16-rtx5090-hardware-validation.md`.
- [ ] Opérations spécifiques aux modèles (M10/M11) et orchestration de diffusion (M9).

## Séquence M8 — état

1. Formalisation des contrats et erreurs des opérations, règles d'aliasing et tolérances numériques : **fait**.
2. Interface de capacités et dispatch déterministe sans fallback implicite : **fait**.
3. Raccordement de l'allocation des sorties et des durées de vie aux services mémoire/scheduler : **fait**.
4. Implémentation des kernels CUDA F32 puis BF16 derrière la feature CUDA existante : **fait**.
5. Tests différentiels, numériques, mémoire et de cycle de vie sur RTX 5090 pour F32 et BF16 : **fait**.
6. Audit des configurations et index Safetensors publiés de DiffusionGemma et Clover : **fait** (`docs/audits/m8-model-graph-requirements.md`). Il confirme que les deux checkpoints sont BF16 et identifie les primitives manquantes (SiLU, RoPE, attention, KV cache, GDN, MoE). Le prérequis BF16 partagé est désormais traité par cette tranche ; les primitives spécifiques et l'orchestration de diffusion appartiennent à M9/M10/M11.

## Réalisation M8 (PR #24)

### Contrats et dispatch

`claymore-tensor/src/contracts.rs` définit `TensorOp`, `OpKind`, `ShapeSignature`, `ValidatedOp`, `BackendCapabilities`, `OpSupport` et les erreurs typées `DispatchError` / `UnsupportedReason`. `TensorDispatcher::explain` évalue tous les backends enregistrés dans l'ordre d'inscription : le premier backend possédant une correspondance **exacte** de dtype, tier, layout, opération et règle d'aliasing est sélectionné ; chaque autre backend reçoit une décision `SELECT` ou `REFUSE` explicitée. `BackendPreference::Explicit` exige le backend nommé et échoue sans fallback si ses capacités ne correspondent pas.

Les quatre opérations exigent F32 **ou** BF16, des layouts contigus, un seul device instance, une sortie fraîchement allouée et des entrées en lecture seule. Aucun casting, broadcasting, matérialisation de vue stridée, transfert Host implicite ou fallback CPU silencieux n'est autorisé. Un dtype différent déclenche `OpError::MixedDType` avant tout lancement. **FP16 n'est pas déclaré supporté** : `DType::F16` reste une métadonnée sans référence CPU, sans kernel et sans chemin de conversion. Les identifiants de backend sont dtype-neutres (`cpu-reference`, `cuda-native`) ; c'est l'entrée `OpSupport` exacte qui porte le dtype autorisé.

### Backend CUDA et intégration mémoire

`claymore-cuda` fournit `CudaComputeEngine` : chargement de **deux** modules PTX (`m8_f32.ptx` et `m8_bf16.ptx`) via `cuModuleLoadData`, résolution des huit symboles via `cuModuleGetFunction`, lancement via `cuLaunchKernel` sur un stream explicite du contexte primaire. Chaque lancement vérifie l'appartenance au même contexte (buffers, stream, modules, events) et convertit les erreurs de compilation/chargement/lancement en `CudaError` typées. Il n'existe pas de deuxième type `Tensor` et aucune bibliothèque tensorielle tierce. `launch_tensor_op` sélectionne le kernel de façon exacte sur le couple `(dtype, ShapeSignature)` : aucun kernel F32 ne peut traiter des données BF16 et inversement.

`claymore-memory` possède le seul chemin d'allocation VRAM tensorielle : `CudaTensorMemory` passe exclusivement par `CudaMemoryManager::allocate_device`, donc par `MemoryOrchestrator` (budget souple 26 Gio, plafond dur 28 Gio, éviction, gardes en vol) et par la vérification de mémoire libre du driver. `launch_tensor_op` marque toutes les allocations touchées `IN_FLIGHT` avant le lancement, exécute le kernel sur l'unique stream de calcul, enregistre un CUDA event réel et publie un `TransferToken` repris par le tracker existant. `wait_for_transfer` synchronise cet event et ne lève les gardes qu'après confirmation ; un échec de lancement ou d'enregistrement laisse les gardes en place. `ManagedCudaBuffer::release` demande une libération backend réussie avant de retirer la comptabilité.

`CudaTensorComputeExecutor` implémente `TaskExecutor` pour la file `Compute` : les tenseurs sont possédés par l'exécuteur, pas par le scheduler, qui ne manipule que des `TaskId`. La completion est signalée au `QueueRunner` uniquement après l'attente de l'event CUDA.

### Stratégie PTX et compatibilité Blackwell

Le code source des kernels est versionné dans `crates/claymore-cuda/kernels/m8_f32.cu` et `m8_bf16.cu` ; les PTX générés sont versionnés dans `crates/claymore-cuda/src/kernels/m8_f32.ptx` et `m8_bf16.ptx`, inclus par `include_str!`. Le build par défaut et la CI ne nécessitent ni `nvcc` ni GPU. Régénération :

```bash
for dtype in f32 bf16; do
  nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
    -o crates/claymore-cuda/src/kernels/m8_${dtype}.ptx \
    crates/claymore-cuda/kernels/m8_${dtype}.cu
done
```

Les kernels BF16 utilisent `__bfloat162float` pour l'élargissement, `fmaf` pour l'accumulation et `__float2bfloat16_rn` pour l'unique arrondi de sortie : le PTX généré doit contenir `cvt.rn.bf16.f32` et `fma.rn.f32`, ce que vérifie un test unitaire. La cible `sm_120` correspond à la compute capability 12.0 de la RTX 5090 (Blackwell). Le pilote doit prendre en charge le PTX 9.0 / CUDA 13+ (vérifié avec le driver 616.64 et CUDA UMD 13.4). Le JIT du driver compile le PTX à l'ouverture du module ; une incompatibilité est signalée par `CudaError::KernelModuleLoad`. Aucune architecture plus ancienne n'est revendiquée.

### Tolérances numériques documentées

| Opération | Tolérance CPU/CUDA | Justification |
|---|---|---|
| add | `atol = 1e-5`, `rtol = 1e-6` | même opération IEEE-754 F32 ; différence observée nulle |
| matmul | `atol = 1e-3`, `rtol = 2e-5` | accumulation FMA sur `K` termes ; ordre des sommes identique, multiplication par des amplitudes jusqu'à `~1e3` |
| softmax | `atol = 1e-6`, `rtol = 1e-5` | sorties dans `[0,1]` ; stabilité par soustraction du maximum et réduction en arbre |
| rms_norm | `atol = 1e-5`, `rtol = 2e-5` | réduction en arbre de `sum(x²)` puis `sqrt`/division, contre réduction séquentielle CPU |

Ces tolérances sont vérifiées par opération dans la campagne matérielle ; les écarts réellement observés sont consignés dans le rapport.

### Tolérances BF16 (politique fixée avant campagne)

La comparaison BF16 ne porte pas sur des flottants mais sur des **pas représentables** : les motifs 16 bits sont convertis en clé monotone (`bits ^ 0x8000` pour les positifs, `!bits` pour les négatifs), de sorte que deux valeurs BF16 adjacentes diffèrent d'exactement 1. Les NaN doivent apparaître des deux côtés ; les infinis sont comparés comme des valeurs ordonnées.

| Opération | Limite autorisée | Justification |
|---|---:|---|
| add | 0 pas | une seule opération IEEE-754 FP32 puis RNE identique des deux côtés |
| matmul | 1 pas | accumulation FP32 dans le même ordre ; au plus un franchissement de frontière d'arrondi |
| softmax | 1 pas | réduction séquentielle CPU contre réduction en arbre CUDA, puis RNE |
| rms_norm | 1 pas | idem pour `sum(x²)`, `sqrt` et la division |

Aucune tolérance n'a été élargie après exécution. Les écarts réellement observés (0 pas sur toutes les shapes des deux campagnes) sont consignés dans le rapport BF16. La somme par ligne de softmax après arrondi BF16 doit rester à `4e-3` près de 1.

## Validation requise

**CPU / CI**
- tests de shape, strides, overflow, offsets, vues et bornes ;
- tests de chaque opération, erreurs de dtype/layout/shape et stabilité numérique ;
- format, tests workspace, Clippy, MSRV Rust 1.86 et compilation de la feature CUDA sans GPU.

**RTX 5090**
- cycle de vie du contexte et des streams ;
- allocations, copies et synchronisation par events ;
- exactitude des kernels CUDA comparée à la référence CPU avec tolérances par opération ;
- précision F32 et BF16 pour les quatre opérations, FP16 restant explicitement refusé ;
- pression mémoire, gardes en vol, libération et intégration scheduler/memory ;
- rapport versionné avec SHA exact, versions driver/runtime, device, commandes, résultats et écarts numériques.

La CI sans GPU ne constitue jamais une validation matérielle. Aucun résultat RTX 5090 n'est revendiqué pour les opérations CUDA tant que la campagne n'a pas été exécutée sur la machine cible.

## Non-objectifs

- stratégie autoregressive générique ;
- boucle de génération commune imposée aux modèles ;
- implémentation de l'attention, du denoising, du GDN ou du routage MoE dans le cœur Tensor ;
- adoption d'un framework tiers comme propriétaire des tenseurs ou de l'orchestration ;
- affirmation de compatibilité complète des modèles sur la seule base des primitives CPU génériques.

## Limites restantes après finalisation BF16

- deux précisions seulement (F32 et BF16) ; FP16 n'est pas implémenté ;
- quatre opérations seulement, sur des tenseurs contigus, avec sortie fraîche ; pas de broadcasting, de vues stridées en entrée d'opération ni d'opérations sur places ;
- matmul simple sans tiling partagé ni Tensor Core ; softmax/RMSNorm avec un bloc par ligne : aucune revendication de performance (M12) ;
- le backend attend l'event CUDA de chaque opération : sûreté des durées de vie, mais pas de recouvrement transfert/calcul ;
- PTX `sm_120` uniquement, avec JIT driver PTX 9.0 / CUDA 13+ ;
- pas de chargement de checkpoints ni d'exécution de graphes de modèles dans M8 ;
- aucune injection de fautes dédiée à l'échec de lancement/kernel (le chemin d'erreur reste fail-closed par construction).
