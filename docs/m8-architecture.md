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

Ces opérations exigent des layouts contigus et un dtype F32. Elles ne convertissent pas silencieusement les types. Ce premier ensemble constitue un socle de référence, pas une affirmation qu'il couvre à lui seul les graphes complets de DiffusionGemma ou Clover. Les primitives BF16/FP16 et les opérations supplémentaires seront sélectionnées après inspection des graphes et exigences numériques.

## État d'implémentation

- [x] Crate claymore-tensor intégré au workspace.
- [x] Layouts contigus/stridés, arithmétique de tailles vérifiée, Tensor propriétaire et TensorView empruntée.
- [x] Retrait des définitions Tensor concurrentes de claymore-device.
- [x] Opérations CPU F32 de référence : add, matmul, softmax, RMSNorm.
- [ ] Contrats d'opérations formalisés (descripteurs, contraintes dtype/layout, aliasing et tolérances).
- [ ] Dispatch explicite par capacités backend.
- [ ] Allocation/sortie intégrée au memory orchestrator et au scheduler sans contourner leurs gardes.
- [ ] Kernels CUDA équivalents et tests différentiels CPU/CUDA.
- [ ] Validation RTX 5090 sur le SHA final.
- [ ] Inspection des graphes/checkpoints DiffusionGemma et Clover pour compléter le jeu d'opérations requis.

## Séquence restante de M8

1. Formaliser les contrats et erreurs des opérations, les règles d'aliasing et les tolérances numériques.
2. Ajouter l'interface de capacités et un dispatch déterministe sans fallback implicite.
3. Raccorder l'allocation des sorties et les durées de vie aux services mémoire/scheduler existants.
4. Implémenter les kernels CUDA des opérations retenues, derrière la feature CUDA existante.
5. Exécuter les tests différentiels, numériques, mémoire et de cycle de vie sur RTX 5090.
6. Auditer les exigences réelles des graphes et checkpoints des deux modèles ; ajouter uniquement les primitives justifiées.

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
