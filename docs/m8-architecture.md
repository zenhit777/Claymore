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

Le crate claymore-tensor fournit un premier ensemble déterministe F32 :
- addition élément par élément, sans broadcasting ;
- multiplication matricielle row-major rank-2 ;
- softmax stable sur la dernière dimension ;
- RMSNorm sur la dernière dimension avec vecteur de scale explicite.

Ces opérations exigent des layouts contigus et un dtype F32. Elles ne convertissent pas silencieusement les types. Ce premier ensemble constitue un socle de référence, pas une affirmation qu'il couvre à lui seul les graphes complets de DiffusionGemma ou Clover. L'audit des checkpoints a confirmé que BF16 est requis pour les poids publiés ; le support BF16 explicite reste à livrer avant l'exécution de ces modèles.

## État d'implémentation

- [x] Crate claymore-tensor intégré au workspace.
- [x] Layouts contigus/stridés, arithmétique de tailles vérifiée, Tensor propriétaire et TensorView empruntée.
- [x] Retrait des définitions Tensor concurrentes de claymore-device.
- [x] Opérations CPU F32 de référence : add, matmul, softmax, RMSNorm.
- [x] Contrats d'opérations formalisés (`contracts.rs` : descripteurs, contraintes dtype/layout/shape, aliasing en lecture seule + sortie fraîche, erreurs typées).
- [x] Dispatch explicite par capacités backend (`TensorDispatcher` : sélection déterministe, explication de chaque refus, aucun fallback ni cast implicite).
- [x] Allocation/sortie intégrée au memory orchestrator et au scheduler (`CudaTensorMemory`, `CudaMemoryManager::launch_tensor_op`, `CudaTensorComputeExecutor`) sans contourner budgets, gardes ni événements.
- [x] Kernels CUDA F32 équivalents et tests différentiels CPU/CUDA (PTX `sm_120` intégré, 8 tests matériels).
- [x] Validation RTX 5090 sur le SHA de code documenté dans `docs/audits/m8-rtx5090-hardware-validation.md`.
- [x] Audit des configurations, index de poids et exigences des graphes DiffusionGemma/Clover (`docs/audits/m8-model-graph-requirements.md`).
- [ ] Support BF16 explicite identifié comme prérequis partagé avant chargement/exécution des checkpoints ; non livré dans cette tranche F32.

## État de clôture de M8

Les contrats, le dispatch, l'intégration memory/scheduler, les quatre kernels F32 et la campagne RTX 5090 sont livrés et documentés. L'audit des modèles cibles est également réalisé ; ses conclusions et les besoins non couverts sont consignés dans `docs/audits/m8-model-graph-requirements.md`.

Cet audit confirme que les deux checkpoints publiés sont BF16. Le chemin F32 livré ne permet donc pas de les charger directement. Le support BF16 explicite (ou une stratégie de conversion/quantification formellement décidée) reste un prérequis avant les tests d'exécution modèle et doit être planifié avant M10/M11. Cette limite ne doit pas être confondue avec la validation matérielle des kernels F32 de M8.

## Réalisation M8 (PR #24)

### Contrats et dispatch

`claymore-tensor/src/contracts.rs` définit `TensorOp`, `OpKind`, `ShapeSignature`, `ValidatedOp`, `BackendCapabilities`, `OpSupport` et les erreurs typées `DispatchError` / `UnsupportedReason`. `TensorDispatcher::explain` évalue tous les backends enregistrés dans l'ordre d'inscription : le premier backend possédant une correspondance **exacte** de dtype, tier, layout, opération et règle d'aliasing est sélectionné ; chaque autre backend reçoit une décision `SELECT` ou `REFUSE` explicitée. `BackendPreference::Explicit` exige le backend nommé et échoue sans fallback si ses capacités ne correspondent pas.

Les quatre opérations exigent F32, des layouts contigus, un seul device instance, une sortie fraîchement allouée et des entrées en lecture seule. Aucun casting, broadcasting, matérialisation de vue stridée, transfert Host implicite ou fallback CPU silencieux n'est autorisé. BF16 et FP16 ne sont **pas** déclarés supportés : les types existent dans `DType`, mais aucun chemin de calcul ni référence CPU n'est implémenté pour eux.

### Backend CUDA et intégration mémoire

`claymore-cuda` fournit `CudaComputeEngine` : chargement du module PTX via `cuModuleLoadData`, résolution des quatre symboles via `cuModuleGetFunction`, lancement via `cuLaunchKernel` sur un stream explicite du contexte primaire. Chaque lancement vérifie l'appartenance au même contexte (buffers, stream, module, events) et convertit les erreurs de compilation/chargement/lancement en `CudaError` typées. Il n'existe pas de deuxième type `Tensor` et aucune bibliothèque tensorielle tierce.

`claymore-memory` possède le seul chemin d'allocation VRAM tensorielle : `CudaTensorMemory` passe exclusivement par `CudaMemoryManager::allocate_device`, donc par `MemoryOrchestrator` (budget souple 26 Gio, plafond dur 28 Gio, éviction, gardes en vol) et par la vérification de mémoire libre du driver. `launch_tensor_op` marque toutes les allocations touchées `IN_FLIGHT` avant le lancement, exécute le kernel sur l'unique stream de calcul, enregistre un CUDA event réel et publie un `TransferToken` repris par le tracker existant. `wait_for_transfer` synchronise cet event et ne lève les gardes qu'après confirmation ; un échec de lancement ou d'enregistrement laisse les gardes en place. `ManagedCudaBuffer::release` demande une libération backend réussie avant de retirer la comptabilité.

`CudaTensorComputeExecutor` implémente `TaskExecutor` pour la file `Compute` : les tenseurs sont possédés par l'exécuteur, pas par le scheduler, qui ne manipule que des `TaskId`. La completion est signalée au `QueueRunner` uniquement après l'attente de l'event CUDA.

### Stratégie PTX et compatibilité Blackwell

Le code source des kernels est versionné dans `crates/claymore-cuda/kernels/m8_f32.cu` ; le PTX généré est versionné dans `crates/claymore-cuda/src/kernels/m8_f32.ptx` et inclus par `include_str!`. Le build par défaut et la CI ne nécessitent ni `nvcc` ni GPU. Régénération :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
  crates/claymore-cuda/kernels/m8_f32.cu
```

La cible `sm_120` correspond à la compute capability 12.0 de la RTX 5090 (Blackwell). Le pilote doit prendre en charge le PTX 9.0 / CUDA 13+ (vérifié avec le driver 616.64 et CUDA UMD 13.4). Le JIT du driver compile le PTX à l'ouverture du module ; une incompatibilité est signalée par `CudaError::KernelModuleLoad`. Aucune architecture plus ancienne n'est revendiquée.

### Tolérances numériques documentées

| Opération | Tolérance CPU/CUDA | Justification |
|---|---|---|
| add | `atol = 1e-5`, `rtol = 1e-6` | même opération IEEE-754 F32 ; différence observée nulle |
| matmul | `atol = 1e-3`, `rtol = 2e-5` | accumulation FMA sur `K` termes ; ordre des sommes identique, multiplication par des amplitudes jusqu'à `~1e3` |
| softmax | `atol = 1e-6`, `rtol = 1e-5` | sorties dans `[0,1]` ; stabilité par soustraction du maximum et réduction en arbre |
| rms_norm | `atol = 1e-5`, `rtol = 2e-5` | réduction en arbre de `sum(x²)` puis `sqrt`/division, contre réduction séquentielle CPU |

Ces tolérances sont vérifiées par opération dans la campagne matérielle ; les écarts réellement observés sont consignés dans le rapport.

## Validation requise

**CPU / CI**
- tests de shape, strides, overflow, offsets, vues et bornes ;
- tests de chaque opération, erreurs de dtype/layout/shape et stabilité numérique ;
- format, tests workspace, Clippy, MSRV Rust 1.86 et compilation de la feature CUDA sans GPU.

**RTX 5090**
- cycle de vie du contexte et des streams ;
- allocations, copies et synchronisation par events ;
- exactitude des kernels CUDA comparée à la référence CPU avec tolérances par opération ;
- précision F32/BF16/FP16 uniquement lorsque chaque chemin est explicitement implémenté ;
- pression mémoire, gardes en vol, libération et intégration scheduler/memory ;
- rapport versionné avec SHA exact, versions driver/runtime, device, commandes, résultats et écarts numériques.

La CI sans GPU ne constitue jamais une validation matérielle. Aucun résultat RTX 5090 n'est revendiqué pour les opérations CUDA tant que la campagne n'a pas été exécutée sur la machine cible.

## Non-objectifs

- stratégie autoregressive générique ;
- boucle de génération commune imposée aux modèles ;
- implémentation de l'attention, du denoising, du GDN ou du routage MoE dans le cœur Tensor ;
- adoption d'un framework tiers comme propriétaire des tenseurs ou de l'orchestration ;
- affirmation de compatibilité complète des modèles sur la seule base des primitives CPU génériques.
