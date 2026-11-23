# Claymore — Référence produit, fonctionnelle et technique

> **Document de référence unique** — cahier des charges, Product Requirements Document (PRD), spécification technique et feuille de route.
>
> **Projet :** Claymore  
> **Dépôt :** https://github.com/zenhit777/Claymore  
> **Version de référence du workspace :** 0.5.0  
> **État du document :** référence vivante, à mettre à jour à chaque jalon  
> **Dernière mise à jour :** 23 novembre 2026

---

## 1. Objet et statut du document

Ce document consolide les objectifs produit, les décisions d'architecture, les contraintes matérielles, les exigences fonctionnelles et non fonctionnelles, ainsi que les phases de développement de Claymore. Il constitue le point d'entrée documentaire du projet.

Il doit permettre à toute personne intervenant sur le dépôt de répondre sans ambiguïté aux questions suivantes :

- Quel problème Claymore cherche-t-il à résoudre ?
- Quels modèles et quelle configuration matérielle sont ciblés ?
- Quelles responsabilités appartiennent à chaque composant ?
- Quelles décisions sont déjà actées ?
- Qu'est-ce qui est réellement implémenté et validé ?
- Quelles fonctionnalités restent à construire et selon quels critères d'acceptation ?

### 1.1 Règles de lecture

Les statuts utilisés dans ce document ont une signification stricte :

- **Livré** : le code correspondant est présent sur `main`.
- **Validé CI** : les contrôles automatisés indiqués ont réussi sur la configuration couverte par le workflow.
- **À valider sur cible** : nécessite une exécution sur la machine WSL2 / RTX 5090.
- **Planifié** : décision ou objectif retenu, mais code non encore livré.
- **Exploratoire** : piste technique à évaluer avant de la transformer en exigence.

Une interface, une politique de gestion ou une primitive bas niveau ne doit jamais être présentée comme une fonctionnalité intégrée de bout en bout si le raccordement correspondant n'existe pas encore.

Les valeurs mémoire indiquées sont des budgets de planification, pas des garanties de disponibilité. Les postes VRAM sont imbriqués. Les enveloppes RAM hôte sont disjointes et leur somme constitue le budget Claymore.

---


### 1.2 Politique de version Rust

La MSRV du workspace est **Rust 1.86.0**. Chaque crate déclare `rust-version.workspace = true` et hérite de `workspace.package.rust-version`. La CI exécute la suite workspace sur cette version minimale, en plus des contrôles sur la toolchain stable.

La MSRV ne doit être augmentée que par une modification explicite du manifeste workspace, accompagnée :
- d'une justification (API standard ou dépendance nécessitant la nouvelle version) ;
- de la mise à jour de tous les manifests, de la CI et de la documentation ;
- d'une validation complète sur la nouvelle version minimale et sur stable.

Les dépendances doivent rester compatibles avec la MSRV déclarée. Une mise à jour de dépendance qui relève silencieusement la version minimale est une régression, même si le build stable réussit.

## 2. Vision produit

Claymore est un runtime expérimental d'inférence écrit en **Rust**, conçu pour exécuter efficacement des modèles de génération avancés sur une machine personnelle équipée d'un GPU NVIDIA à forte capacité mémoire.

Le projet vise à construire progressivement un moteur spécialisé qui combine :

1. une exécution de modèles fondée sur les composants natifs de Claymore ;
2. une orchestration explicite des ressources CPU, RAM, VRAM et mémoire hôte épinglée ;
3. une gestion de résidence et de cache des experts pour les architectures Mixture-of-Experts (MoE) ;
4. un ordonnanceur conscient des transferts et des coûts de chargement ;
5. des transferts asynchrones et une synchronisation sûre avec le travail GPU ;
6. à terme, des kernels et chemins d'exécution spécialisés lorsque les mesures justifieront leur développement.

Les idées d'orchestration mémoire, de scheduling et de kernels spécialisés sont notamment inspirées des approches observées dans FreeToken et Strata (Niko1221). Il s'agit d'une inspiration d'architecture, pas d'une dépendance logicielle ni d'une promesse de reproduire ces projets.

### 2.1 Objectifs

- Exploiter efficacement les 32 Gio de VRAM de la RTX 5090 sans consommer toute la mémoire disponible.
- Pouvoir gérer des working sets dépassant la VRAM grâce à une stratégie contrôlée de résidence et d'offload en RAM.
- Limiter les transferts inutiles et anticiper les besoins de poids ou d'experts.
- Séparer le cœur générique du runtime des particularités de chaque architecture de modèle.
- Maintenir une base Rust testable sans GPU pour les politiques et les composants indépendants du matériel.
- Introduire les fonctionnalités CUDA derrière une feature optionnelle afin de conserver un build par défaut portable.
- Faire évoluer le projet par jalons versionnés, chacun assorti de critères d'acceptation vérifiables.

### 2.2 Hors périmètre initial

Ne sont pas des objectifs des premières versions :

- un service d'inférence multi-utilisateur distribué ;
- un cluster multi-GPU ou multi-machine ;
- une API réseau de production avec authentification et gestion de tenants ;
- l'entraînement ou le fine-tuning des modèles ;
- une compatibilité universelle avec toutes les architectures de modèles ;
- une promesse de faire tenir intégralement en VRAM un modèle dont les poids dépassent le budget disponible ;
- l'usage du swap système comme niveau normal de stockage des poids.

---

## 3. Périmètre modèles

Les deux cibles fonctionnelles initiales sont indépendantes et disposent chacune d'un adaptateur spécialisé, construit au-dessus d'un runtime, d'un socle tensoriel et de services transverses communs. L'orchestration de diffusion est modulaire : les invariants communs sont factorisés, tandis que les stratégies préservent les différences réelles entre modèles.

### 3.1 DiffusionGemma

**Modèle cible exact :** `google/diffusiongemma-26B-A4B-it` (identifiant Hugging Face).

Objectif : construire un chemin d'exécution adapté à ce modèle de diffusion de texte. **Décision de périmètre initial : texte uniquement.** Les entrées image et vidéo, leur prétraitement et les chemins multimodaux associés sont hors périmètre de la première intégration Claymore ; aucune interface ni compatibilité image/vidéo n'est revendiquée. Une extension multimodale nécessitera une décision produit et une spécification dédiées (formats d'entrée, prétraitement, interfaces modèle, ressources et validation) avant développement. Les mentions de « Gemma » seules désignent la famille ou le backbone architectural, jamais une cible modèle interchangeable.

Les travaux spécifiques prévus comprennent :

- gestion du canvas ou de l'état de génération ;
- boucle de denoising et contrôle des étapes ;
- prise en charge des besoins d'attention bidirectionnelle propres à l'architecture ;
- durée de vie des tenseurs intermédiaires et des états de génération ;
- choix de précision par opération et validation numérique ;
- allocation et libération prévisibles des buffers temporaires.

**Point de vigilance :** la gestion d'un KV cache classique ne doit pas être supposée identique à celle d'un modèle autoregressif. La conception devra suivre les besoins effectifs de l'architecture DiffusionGemma et de son implémentation propre.

### 3.2 Clover-35B-A3B-preview

Objectif : préparer l'exécution d'un modèle MoE / architecture hybride, en maîtrisant le coût de résidence des experts et les opérations spécifiques du modèle.

Travaux spécifiques prévus :

- intégration des chemins GDN et block diffusion lorsqu'ils sont requis par l'architecture ;
- routage MoE et identification des experts nécessaires ;
- chargement, résidence, préchargement et éviction des experts ;
- coordination entre expert cache, ordonnanceur et transferts ;
- gestion des états et caches d'attention ;
- validation des formes, layouts et chemins de précision.

Le cache d'experts est une politique de résidence sous contrainte mémoire. Il ne doit pas être confondu avec un cache de résultats d'inférence.

### 3.3 Principe d'isolation

DiffusionGemma et Clover partagent un seul runtime, le socle tensoriel natif, les abstractions de mémoire, le scheduler, le backend CUDA et une orchestration de diffusion modulaire. Les adaptateurs spécialisés conservent les différences fondamentales de graphe, de calcul et de cycle de vie ; ils ne deviennent pas des runtimes ou moteurs tensoriels indépendants.

---

## 4. Configuration matérielle de référence

La première cible de développement et de validation est la machine suivante :

| Élément | Configuration |
|---|---|
| GPU | NVIDIA RTX 5090 |
| VRAM | 32 Gio |
| CPU | AMD Ryzen 9 7950X3D |
| RAM système | 128 Gio |
| Environnement | WSL2 sous Windows |
| Langage | Rust, édition 2021 |
| Backend de tenseurs visé | Moteur tensoriel natif Claymore (socle M8 livré sur `main` par la PR #24 ; extensions modèles M10/M11 et optimisations M12) |
| Politique numérique | précision mixte, BF16 et FP32 supportés dans les primitives M8 ; FP16 non implémenté |

Cette configuration est le profil de référence, pas une limite architecturale du projet. Les paramètres matériels doivent rester configurables.

### 4.1 Politique de précision

Décision initiale :

- **BF16** comme précision préférée lorsque le matériel, le modèle et les kernels la supportent correctement ;
- **FP32** pour les accumulations ou opérations sensibles qui le nécessitent ;
- **FP16** comme solution de repli *envisagée* lorsque BF16 n'est pas disponible ou approprié : le type existe dans `DType`, mais aucun chemin FP16 n'est implémenté ni validé ; aucun repli implicite n'est autorisé ;
- **INT8 / INT4** comme axes ultérieurs d'optimisation, après mesures de qualité et de performance.

Aucune quantification ne doit être activée par défaut sans validation de la qualité de génération, de la stabilité numérique et du gain réel.

---

## 5. Budgets mémoire de référence

### 5.1 VRAM — RTX 5090

| Poste | Budget initial |
|---|---:|
| VRAM physique | 32 Gio |
| Budget souple Claymore | 26 Gio |
| Plafond dur planifié | 28 Gio |
| Réserve de sécurité | 4 Gio |
| Working set modèles / experts | 18 Gio |
| États, KV, canvas et buffers de génération | 4 Gio |
| Workspace CUDA | 2 Gio |
| Transferts | 1 Gio |
| Marge allocator | 1 Gio |
| Cache experts : cible | 14 Gio |
| Cache experts : maximum | 16 Gio |

Le cache d'experts est **inclus** dans le working set modèles / experts de 18 Gio. Il ne s'ajoute pas à ce budget. La cible de 14 Gio et le maximum de 16 Gio sont des limites de politique initiales, à ajuster à partir des mesures réelles.

Le budget souple, le plafond dur et la réserve répondent à des objectifs différents. Le runtime doit conserver une marge pour les allocations CUDA non attribuées directement à ses buffers, les variations du contexte graphique et les besoins temporaires.

### 5.2 RAM hôte

| Poste | Budget initial |
|---|---:|
| RAM physique | 128 Gio |
| Budget Claymore | 88 Gio |
| Pool poids / experts : cible | 64 Gio |
| Mémoire hôte épinglée : plafond | 8 Gio |
| Staging et états | 16 Gio |
| Réserve hors VM (Windows et processus hôte) | environ 13 Gio avec une limite WSL2 de 115 GB sur 128 Gio physiques ; marge indicative, non garantie |

Les enveloppes RAM hôte sont **disjointes** : 64 + 16 + 8 = 88 Gio. Le pool poids/experts et le staging/états utilisent le tier `Host`, dont la limite doit donc être configurée à **80 Gio** ; le tier `PinnedHost` est limité séparément à **8 Gio**. La mémoire épinglée n'est pas comprise dans les 80 Gio du tier `Host`.

Trois référentiels ne doivent pas être confondus : machine physique (128 Gio), limite WSL2 configurée à memory=115GB (environ 112 Gio visibles dans Linux sur la machine de référence, à recontrôler après changement d'environnement) et budget applicatif Claymore (88 Gio). La marge disponible dans la VM au-delà du budget applicatif de 88 Gio est destinée à Linux, au page cache et aux outils de développement ; elle doit être mesurée en exploitation. Avec une limite WSL2 de 115 GB sur 128 Gio physiques, il reste environ 13 Gio hors VM pour Windows et les processus hôte, sous réserve de la conversion GB/Gio et de la consommation réelle. **VRAM : postes imbriqués ; RAM hôte : enveloppes disjointes.**

### 5.3 WSL2

Configuration de départ proposée dans `%UserProfile%\.wslconfig` :

```ini
[wsl2]
memory=115GB
processors=16
swap=16GB
```

Après modification, redémarrer WSL avec `wsl --shutdown`.

Le swap n'est pas un niveau normal de résidence des poids. Il ne doit pas être utilisé comme substitut au pool RAM ni comme mécanisme de paging du working set d'inférence.

### 5.4 Profil scheduler initial

| Paramètre | Valeur de départ |
|---|---:|
| Workers CPU | 10 |
| Minimum workers CPU | 6 |
| Maximum workers CPU | 12 |
| CUDA compute streams | 1 |
| CUDA transfer streams | 1 |
| Préchargements simultanés maximum | 2 |
| Lookahead | 1 |
| Tick scheduler | 500 µs |

Ces valeurs sont des paramètres initiaux à mesurer. Elles ne constituent pas une vérité universelle et devront évoluer à partir de benchmarks reproductibles.

---

## 6. Architecture cible

L'architecture est organisée en couches avec des frontières explicites.

### 6.1 Couches

1. **Runtime et interface**
   - chargement de configuration ;
   - cycle de vie du runtime ;
   - diagnostics et CLI ;
   - orchestration de haut niveau.

2. **Orchestration de diffusion et adaptateurs**
   - orchestration commune, modulaire et extensible ;
   - adaptateur DiffusionGemma et stratégie de denoising propres au modèle ;
   - adaptateur Clover et stratégie block-diffusion/GDN/MoE propres au modèle.

3. **Cœur d'optimisation**
   - memory orchestrator ;
   - gestion des budgets et allocations ;
   - expert cache ;
   - scheduler ;
   - transfer engine ;
   - kernel dispatcher ;
   - politique de précision et quantification ;
   - télémétrie et profiling.

4. **Backend d'exécution**
   - moteur tensoriel et exécution des modèles propres à Claymore ;
   - backend CUDA pour les allocations, transferts, streams et events ;
   - kernels spécialisés lorsque leur nécessité est démontrée.

### 6.2 Crates du workspace

| Crate | Responsabilité |
|---|---|
| `claymore-core` | Types partagés, configuration et contrats transverses |
| `claymore-device` | Contrats Device/Buffer indépendants du backend et simulateur CPU |
| `claymore-tensor` | Modèle Tensor canonique, layouts, vues et opérations natives |
| `claymore-memory` | Budgets, comptabilité, admission et politiques d'éviction |
| `claymore-scheduler` | Graphe de tâches, files compute/transfer/prefetch, priorités et limites de préchargement |
| `claymore-runtime` | Cycle de vie des requêtes, contexte d'exécution et orchestration synchrone indépendante des modèles |
| `claymore-model-gemma` | Frontière d'intégration du modèle DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) |
| `claymore-model-clover` | Frontière d'intégration Clover |
| `claymore-cuda` | Backend optionnel CUDA Driver API |
| `claymore-cli` | Commandes, profil et diagnostics |

Le workspace utilise Cargo, resolver 2, Rust edition 2021. La version de référence actuelle est 0.5.0.

`claymore-scheduler` fournit désormais un ordonnanceur déterministe de tâches avec dépendances, priorités, files logiques Compute/Transfer/Prefetch, fenêtre de préchargement bornée et transitions explicites. Il ne crée pas de workers, ne lance pas les opérations et ne crée aucun stream CUDA : l'exécuteur reste responsable du travail réel. Le scheduler est `!Send + !Sync` afin de préserver le confinement au thread propriétaire. La validation de `SchedulerConfig` accepte de 1 à 12 workers et au plus deux préchargements simultanés.

### 6.3 Principes de conception

- Les politiques métier et mémoire doivent être testables sans matériel CUDA.
- Les contrats Device/Buffer/Tensor sont backend-neutres ; aucun trait n'impose Send/Sync, car un backend peut être lié à un thread ou à un contexte.
- Les opérations Device de M2 sont synchrones : leur retour garantit la fin de l'accès aux buffers ; tout futur chemin asynchrone doit transporter un jeton de complétion explicite.
- Les appels CUDA doivent être encapsulés dans le crate backend et exposer des contrats de durée de vie explicites.
- Les erreurs de backend ne doivent pas être transformées en succès comptables.
- L'état logique du gestionnaire mémoire ne doit refléter une libération qu'après confirmation du backend.
- Les buffers utilisés par un transfert asynchrone doivent rester vivants et non réutilisés jusqu'à la fin effective du travail GPU.
- Les événements CUDA, et non un simple compteur logique, doivent devenir la source d'autorité pour la sûreté des opérations asynchrones.
- Le scheduler ne doit pas déclencher de préchargements qui dépassent les budgets ou qui évinceraient des allocations encore utilisées.
- Les choix d'optimisation doivent être mesurés avant d'être complexifiés.

### 6.4 Politique de dépendances du moteur tensoriel

Le moteur tensoriel, ses contrats publics et son orchestration appartiennent à Claymore. Un framework ou runtime tensoriel tiers ne doit pas devenir le propriétaire des types `Tensor` / `Device` / `Buffer`, de leur cycle de vie, de la politique mémoire, du scheduler ou du chemin d'exécution des modèles.

Sont autorisés après revue de leur pertinence, de leur licence et de leur coût d'intégration :
- les API et bibliothèques CUDA/NVIDIA de bas niveau (CUDA Driver API, CUDA Runtime si nécessaire, cuBLAS/cuBLASLt, CUTLASS, cuDNN) pour exécuter des primitives ou kernels ;
- les bibliothèques de formats et de données (notamment safetensors), les tokenizers et les parseurs nécessaires au chargement des modèles ;
- les crates Rust utilitaires ciblées (par exemple `half` pour les types BF16/FP16, sérialisation, vérification ou instrumentation).

Ces composants peuvent fournir des primitives spécialisées, mais l'API tensorielle, les règles d'ownership, la synchronisation, la gestion des allocations et les choix d'exécution restent définis par Claymore. Toute nouvelle dépendance doit être justifiée et validée ; aucune bibliothèque externe ne devient automatiquement une dépendance structurante du runtime.

### 6.5 M8 — limites de lancement CUDA et contrat de dispatch

`CudaComputeEngine` (`crates/claymore-cuda/src/compute.rs`) charge deux modules PTX (`m8_f32.ptx`, `m8_bf16.ptx`), résout huit symboles et lance chaque opération explicitement sur un stream fourni par l'appelant. Chaque lancement vérifie l'appartenance au même contexte (engine, modules, stream) ; il n'y a ni copie implicite, ni repli CPU, ni `cuCtxSynchronize` global dans cette couche. `CudaMemoryManager::launch_tensor_op` sélectionne le kernel de façon exacte sur le couple `(dtype, ShapeSignature)` : **aucun kernel ne traite les deux dtypes** et aucune combinaison non déclarée ne descend sur le device.

| Opération (F32 et BF16) | Bloc (`cuLaunchKernel`) | Grille | Mémoire partagée dynamique |
|---|---|---|---|
| `add` | `ELEMENTWISE_BLOCK = 256`, `(256, 1, 1)` | `(ceil(numel / 256), 1, 1)`, boucle grid-stride de pas `blockDim.x * gridDim.x` | 0 |
| `matmul` | `MATMUL_TILE = 16`, `(16, 16, 1)` = 256 threads | `(ceil(cols / 16), ceil(rows / 16), 1)` ; un thread par élément de sortie, boucle sur `K` | 0 |
| `softmax_last_dim`, `rms_norm` | `(256, 1, 1)` | `(rows, 1, 1)`, un bloc par ligne | `ELEMENTWISE_BLOCK * 4 = 1024` octets (un `f32` par thread) |

`grid_for` calcule `total.div_ceil(block)` et convertit le résultat en `c_uint` : au-delà de la plage `u32`, le lancement est refusé par `InvalidLaunchConfiguration("grid dimension exceeds CUDA limit")`. Aucune interrogation des limites réelles du device n'est effectuée. Les dimensions nulles sont rejetées en amont (`checked_count`, `checked_matmul`, `checked_last_dim`). Le matmul est une implémentation simple, sans tuilage en mémoire partagée ni Tensor Core ; la performance relève de M12.

Le profil matériel impose exactement **un stream compute et un stream transfer** (`HardwareProfile::validate`) ; `CudaMemoryManager` les crée tous deux avec `CU_STREAM_NON_BLOCKING`. Toutes les opérations tensoriellesss d'un même manager sont sérialisées sur ce stream compute unique, chacune avec son propre event CUDA : les gardes `IN_FLIGHT` sont installés avant le retour de `launch_tensor_op` (la soumission a lieu sous emprunt exclusif de `&mut self`) et levés seulement après `wait_for_transfer`. Aucun recouvrement transfert/calcul ni préchargement concurrent n'est revendiqué ; l'expression explicite des dépendances compute ↔ transfert reste une limite suivie (§7.3). Les écarts numériques CPU/CUDA et la provenance des PTX sont documentés dans [audits/m8-cpu-cuda-numerics.md](audits/m8-cpu-cuda-numerics.md) et [audits/m8-ptx-provenance.md](audits/m8-ptx-provenance.md).


---

## 7. État réel de l'implémentation

Cette section prévaut sur toute formulation plus ambitieuse de la vision cible. La consolidation matérielle M1 est livrée par la PR #5 : contexte primaire CUDA, contrôles d'appartenance des ressources, compute capability et comportement fail-closed du gestionnaire mémoire.

### 7.0 Travaux en cours

#### M3 — Device foundations hardening (PR #15 fusionnée)

La PR #15 a livré les issues #7 à #10 : MSRV Rust 1.86, identité d'instance des devices, distinction des erreurs OOM/overflow et couverture des cas limites CPU. La fusion dans `main` a été effectuée le 20 novembre 2026 (merge commit `e758862937a0dff376b83449aab2c937400be9f4`).

Sémantique retenue : le clonage d'un handle `CpuDevice` conserve l'identité d'instance ; deux appels indépendants à `CpuDevice::new()` créent deux identités différentes même si `DeviceInfo::id` vaut dans les deux cas `cpu:0`. Cette identité est opaque et locale au processus.

#### M4 — Device lifecycle contracts (PR #16 fusionnée)

La PR #16 a livré les issues #11 et #12. Elle introduit une libération explicite et faillible des buffers, conserve l'ownership et la comptabilité en cas d'échec backend, factorise les invariants de cycle de vie des gestionnaires CPU/CUDA, expose des jetons de complétion pour les transferts H2D/D2H et ajoute les vues tensor stridées.

Le contrat `CompletionToken` et la méthode `Device::copy_async` sont définis, mais aucun raccordement de `CudaDeviceBuffer` au trait `Device` n'est effectué. Le backend CUDA reste confiné au thread propriétaire. M4 est intégré dans `main` ; M5 ajoute le polling non bloquant et la finalisation automatique avant admission.

#### M5 — Robustesse CUDA (livré par les PR #17 et #18)

M5 est livré : stream de transfert `CU_STREAM_NON_BLOCKING`, `poll_completed_transfers`, finalisation des événements terminés avant admission/éviction, départage déterministe des évictions et validation matérielle RTX 5090. La revalidation B1 a exécuté 25 tests matériels trois fois (75/75 PASS) sur le SHA de code `385db116f4c2cce6138ee530f7bec7771f355d95`; le rapport complet et ses limites sont versionnés dans `docs/audits/m4-m5-rtx5090-hardware-validation.md`. L'injection déterministe des erreurs event/stream/free reste un chantier séparé dans l'issue #19 et ne constitue pas une validation matérielle réussie.

### 7.1 Livré sur main

#### M7.1 — Durcissement scheduler/runtime et préparation M8 (PR #22)

**État : livré sur `main` par la PR #22.** Le périmètre logiciel M7.1 et la préparation architecturale M8 sont intégrés. La validation matérielle RTX 5090 a été réalisée sur le SHA de code `c16e9254994f24b6c7442c610332baf010352b1e` : 17/17 tests requis lors de chacune de deux campagnes, et 27/27 tests matériels workspace lors de chacune. Le rapport et les logs sont versionnés dans `docs/audits/m7.1-rtx5090-hardware-validation.md` et les fichiers associés. La CI complète a été validée sur le HEAD documentaire final de la PR. Les limites de couverture restantes sont suivies dans l'issue #23.

#### v0.3 — Memory Orchestrator (politiques host-side)

Le crate `claymore-memory` contient :

- validation typée des budgets VRAM, RAM hôte et mémoire hôte épinglée ;
- comptabilisation des allocations par tier ;
- planification d'admission ;
- sélection de candidats à l'éviction ;
- politiques LRU, LFU et adaptive ;
- protection logique des allocations avec `safe_after_tick` ;
- confirmation explicite d'éviction avant mise à jour de la comptabilité ;
- Les enveloppes RAM sont disjointes : Host = weights pool + staging/state (80 Gio sur le profil de référence), PinnedHost = 8 Gio ; leur somme est validée contre le budget Claymore de 88 Gio.
- Le budget souple VRAM de 26 Gio est une cible d'éviction proactive ; le plafond dur de 28 Gio reste la limite d'admission. Le plafond maximal du cache d'experts est appliqué à chaque admission.
- Cette couche de politiques reste backend-neutre ; les allocations réelles, transferts et libérations sont exécutés par les adaptateurs correspondants.

#### v0.4 — CUDA Driver Backend

Le crate `claymore-cuda` fournit un backend optionnel, activé par la feature `cuda-driver`, qui expose les primitives suivantes :

- initialisation du driver et sélection d'un device par ordinal ;
- lecture du nom du device et de sa mémoire totale ;
- interrogation de la mémoire libre et totale ;
- allocation et libération de buffers device ;
- allocation et libération de mémoire hôte épinglée portable ;
- création et synchronisation de streams ;
- création, enregistrement et synchronisation d'events ;
- transferts asynchrones H2D et D2H.

Le build par défaut reste indépendant du matériel. La feature CUDA requiert une bibliothèque driver NVIDIA accessible dans l'environnement Linux/WSL2, généralement `libcuda.so.1` et un lien de développement utilisable par le linker.

#### v0.5 — Adaptateur Memory ↔ CUDA (opt-in, feature `cuda-driver`)

`claymore-memory` expose `CudaMemoryManager` derrière la feature optionnelle `cuda-driver` :

- allocations et libérations réelles de buffers VRAM et pinned host ;
- exécution des plans d'admission et confirmation comptable après libération backend ;
- transferts H2D/D2H associés à des CUDA events ;
- protection des allocations jusqu'à la synchronisation des events ;
- contrôle de la mémoire VRAM libre et accès borné aux buffers pinned ;
- ticks logiques monotones imposés par le manager.

**État :** livré sur `main` derrière la feature `cuda-driver` ; compilation et Clippy de la feature vérifiés en CI sans lien ni GPU ; validation matérielle réelle effectuée sous WSL2 sur RTX 5090 pour le SHA `a3ff0fd` de la branche `feat/m1-hardware-foundation` (compute capability 12.0, contexte primaire partagé, allocations pinned/device, roundtrip H2D/D2H de 1 Mio, intégrité et libération) ; détails et limites dans `docs/audits/m1-hardware-validation.md` ; pas de backend simulé.

#### M2 — Device / Tensor (backend CPU)

**État :** livré sur `main` par la PR #6. Le crate `claymore-device` fournit les contrats `Device`, `DeviceBuffer`, `DeviceCapabilities` et le simulateur `CpuDevice`. Le type `Tensor` canonique vit dans `claymore-tensor` : les anciennes définitions `Tensor` / `TensorView` de `claymore-device` ont été retirées pendant M8 (§6.2 et §10). `claymore-memory` fournit `DeviceMemoryManager<D>` pour appliquer les politiques d'admission et d'éviction sur ces contrats.

Ce jalon livre des abstractions backend-neutres et leur validation CPU ; il ne livre aucun calcul tensoriel ni opérateur. Le backend CUDA existant n'implémente pas encore `Device` et son raccordement reste conditionné à un contrat de libération faillible et de protection des allocations en vol (voir §7.3 et l'audit M2).

#### M7 — Scheduler et orchestration mémoire

**État : livré sur `main` par la PR #21.** Le jalon relie les requêtes Compute, les tâches Transfer/Prefetch, les exécuteurs mémoire backend-neutres et l'exécuteur CUDA event-backed.

Le scheduler et les exécuteurs sont mono-propriétaires et synchrones. `CudaTransferExecutor` attend l'event CUDA après chaque copie H2D/D2H avant de retourner le succès à `QueueRunner`. Le périmètre ne fournit donc ni recouvrement transfert/calcul, ni préchargement concurrent, ni exécution asynchrone des requêtes Compute. La campagne RTX 5090 versionnée rapporte 34/34 exécutions de tests matériels réussies sur le SHA de code indiqué dans le rapport ; ses limites (transfert M7 de 1 MiB, cache expert à petite échelle, absence de kernels et d'injection de fautes) restent applicables.

Le coût de rechargement est exposé par l'API mémoire, mais aucun chargeur de poids ou d'experts ne l'alimente encore automatiquement ; `CudaTransferExecutor` ne propose pas de tâche `RecordReloadCost`.


### M7.1 — Durcissement scheduler/runtime et gestion mémoire (PR #22)

**Statut : livré sur `main` par la PR #22.** Cette phase consolide les invariants de M7 avant l'ouverture de M8 ; elle ne livre pas le moteur tensoriel M8.

Périmètre de livraison :
- indexer les files Ready par priorité/FIFO, propager les transitions via les dépendances directes et compacter les tâches terminales tout en conservant leur snapshot public ;
- rendre la purge des payloads annulés incrémentale ;
- convertir les panics des exécuteurs et du chemin ScheduledRuntime en échecs de tâches, avec propagation aux dépendants ;
- introduire une frontière privée injectable pour query/synchronization des events, synchronisation du stream et libération des buffers ; tester NOT_READY, complétion, erreurs query/sync/free, conservation des gardes et comportement fail-closed sans GPU ;
- suivre Cargo.lock et exécuter les validations avec `--locked` ;
- documenter `tick_us` comme cadence nominale de boucle d'orchestration, distincte des ticks logiques et des délais d'events ; conserver le contrat actuel d'un stream compute et d'un stream transfer réservés ;
- calculer conjointement les contraintes d'admission VRAM et cache experts ;
- afficher la version CLI depuis les métadonnées Cargo et rendre explicite la consommation/destruction des résultats ReadPinned.

La préparation architecturale de M8 fait également partie de M7.1 : [docs/m8-architecture.md](m8-architecture.md) fixe les frontières Tensor / Device / Operations / Kernel dispatch / Runtime / Model adapters, les invariants d'ownership et les validations CPU/CI et RTX 5090. Ce document ne signifie pas que le code M8 est déjà implémenté.

#### M8 — Socle tensoriel natif Claymore (PR #24)

**État : livré sur `main` par la PR #24.** `claymore-tensor` possède les types Tensor/Layout/Vues canoniques, les opérations de référence CPU F32/BF16 et les contrats de dispatch explicites. Les kernels CUDA F32/BF16 et leur intégration au gestionnaire mémoire sont décrits dans [m8-architecture.md](m8-architecture.md) et les audits matériels versionnés. Cette livraison fournit le socle de calcul ; elle ne signifie pas que les graphes complets de DiffusionGemma ou Clover sont exécutables.

#### M9 — Orchestration de diffusion commune (PR #32)

**État : cœur livré sur `main` par la PR #32 ; couverture CPU et documentation de finalisation suivies dans l'issue #31.** Le runtime expose des plans validés par blocs, une chaîne de dépendances par session, une exécution bornée par étape, l'annulation coopérative et la propagation des échecs. Les données de modèle et les tenseurs restent hors des payloads du scheduler. Les métriques M9 sont des compteurs d'orchestration CPU ; aucune télémétrie GPU n'est déduite ou simulée.

Les lots logiciels sont validés par la CI CPU, y compris les tests de fault injection avec la feature CUDA activée mais sans GPU. Le chemin de production `CudaMemoryManager` refactoré a également été revalidé sur la RTX 5090 au SHA de code `c16e9254994f24b6c7442c610332baf010352b1e` : les 17 tests matériels ciblés ont réussi lors de deux campagnes consécutives (34/34), et les 27 tests matériels découverts par la commande workspace ont également réussi lors de chacune des deux campagnes (54/54). Le vrai driver NVIDIA WSL2 a été utilisé, sans stub ; le rapport, les logs et leurs empreintes SHA-256 sont versionnés dans `docs/audits/m7.1-rtx5090-hardware-validation.md` et les fichiers associés. Cette validation ne couvre pas les kernels tensoriels ni les performances M8 ; ces gates restent distincts.

#### Limites CUDA explicitement suivies après la revue M7.1

La frontière injectable de M7.1 couvre le tracker de transferts partagé entre production et tests, le polling/retirement, l'attente d'allocation, la planification d'admission avant éviction, la libération/confirmation et la synchronisation de stream testée par doubles. Elle ne couvre pas encore de bout en bout les branches suivantes :

- `copy_h2d` / `copy_d2h` : séquence d'échec d'enregistrement d'event puis d'échec de synchronisation, et état d'empoisonnement/fail-closed du manager ;
- `make_room_for_expert` : polling, planification et exécution de l'admission expert sur le chemin complet ;
- `Drop` de `CudaMemoryManager` avec un stream factice injecté.

Ces limites sont suivies dans l'[issue #23](https://github.com/zenhit777/Claymore/issues/23). Elles ne sont pas présentées comme couvertes par les tests M7.1 actuels. Les tests matériels ciblés de la PR #22 vérifient le câblage réel du tracker refactoré ; ils ne remplacent pas les tests de fault injection des branches restantes.

### 7.2 Fonctionnalités et extensions non livrées de bout en bout

Les éléments suivants ne doivent pas être considérés comme fonctionnels de bout en bout :

#### M6 — Runtime unifié (première tranche livrée ; extensions restantes)

Le crate `claymore-runtime` fournit un noyau synchrone, mono-exécution et indépendant des modèles : identifiants monotones, contexte isolé par requête, enveloppes opaques, états explicites, rapport d'exécution, stratégie `GenerationStrategy`, erreurs et annulation coopérative. Aucun trait `Send`/`Sync` n'est imposé à la stratégie.

La limite `max_payload_bytes` porte explicitement sur les seuls octets du payload ; les métadonnées ne sont pas comptabilisées. Si une stratégie panique, la panique se propage, l'identifiant a déjà été consommé et aucun rapport n'est produit : l'instance du runtime doit être abandonnée. Si l'annulation est observée après le retour de la stratégie, elle prévaut sur son résultat ; une éventuelle erreur de stratégie simultanée est écartée du rapport.

**Statut : première tranche livrée sur `main` par la PR #20.** La portée historique initiale de M6 incluait aussi l'orchestration de haut niveau et des frontières explicites entre runtime, architecture de modèle, stratégie et mode d'exécution. Ces éléments ne sont pas implémentés dans cette tranche : l'orchestration, l'ownership des ressources et le scheduling sont reportés à M7 ; l'intégration des moteurs tensoriels et les modes d'exécution concrets sont reportés à M8 et aux phases modèles ultérieures. Cette répartition est un amendement explicite du périmètre, pas une déclaration d'achèvement de ces critères.

La tranche M6 n'ouvre aucun contexte ni ressource CUDA, ne planifie pas de tâches, ne charge pas de modèle et n'effectue aucun calcul tensoriel. Elle est validée par des tests CPU.

 - exécution concurrente/asynchrone des requêtes et stratégie de génération concrète (extensions futures du runtime) ;
- pool allocator VRAM, réutilisation de blocs et défragmentation (hors périmètre M7.1) ;
- intégration des tâches Transfer/Prefetch au scheduler M7 : livrée dans la première tranche, avec exécution synchrone sur le thread propriétaire ;
- prefetch prédictif intégré au cache d'experts ;
- télémétrie consolidée CPU/RAM/VRAM/transferts ;
- graphes complets et chemins d'inférence effectifs des modèles ;
- chargement et exécution réels de DiffusionGemma ;
- chargement et exécution réels de Clover-35B-A3B-preview ;
- kernels CUDA spécialisés propres aux graphes complets des modèles et optimisations non encore livrées ;\n- extensions post-M9 : équité round-robin, hooks de stratégie/retry, rétention/purge des sessions et convergence des schedulers (issues #33 à #38) ;
- validation de performance et de qualité sur la RTX 5090 cible.

Les crates modèles et adaptateurs ne constituent pas à eux seuls la preuve que les modèles sont exécutables. La PR #40 a livré une première tranche DiffusionGemma (chargeur strict, primitives et sous-graphe partiel), mais elle déclare explicitement ouverts le streaming complet, les chemins MoE/vision, le cache encodeur/cross-attention et la génération réelle du checkpoint 26B.

### 7.3 Limites connues du backend CUDA actuel

#### Contrats et limites du runtime M6 (première tranche PR #20)

- `RuntimeConfig::max_payload_bytes` borne uniquement `RequestInput::payload`, pas la taille cumulée des métadonnées ; ces dernières restent des données de contrôle non bornées par ce paramètre.
- Un appel direct à `Runtime::execute` ne transforme pas les panics de stratégie en `RuntimeError`. Dans le chemin `ScheduledRuntime` de M7.1, la panique est interceptée à la frontière d'exécution, convertie en échec de tâche et propagée aux dépendants ; l'état interne de la stratégie après unwinding n'est toutefois pas garanti.
- L'annulation observée après le retour de la stratégie prévaut sur le résultat de celle-ci. Si la stratégie renvoie simultanément une erreur, le rapport contient `Cancelled` et l'erreur de stratégie n'est pas conservée.
- L'exécution reste synchrone et mono-requête ; aucune interruption forcée d'un appel backend bloquant n'est garantie. Le jeton est coopératif.
- Le contrat d'erreur de stratégie est encore une `String` ; son évolution vers un type structuré est une décision préalable à l'intégration des moteurs en M8/M9.
- L'ownership des ressources CUDA est défini pour M7 : scheduler et exécuteurs restent mono-propriétaires (`!Send + !Sync`) ; `CudaTransferExecutor` détient son `CudaMemoryManager` et attend chaque event H2D/D2H. Le raccordement du backend CUDA au trait générique `Device` reste un chantier distinct.
- `RequestOutput` représente une sortie complète ; une API de sortie incrémentale (blocs/tokens) reste à concevoir pour les stratégies qui en auront besoin.
- L'épuisement des identifiants est détecté sans wraparound ; l'identifiant `u64::MAX` n'est pas émis car son incrément ne peut pas être représenté.

- compute capability exposée via `CudaDeviceInfo` ;
- sélection du device par ordinal uniquement ;
- pas de pool allocator, de défragmentation, de télémétrie ni d'éviction périodique/prédictive hors du chemin d'admission de `CudaMemoryManager` ;
- les appels bas niveau de copie asynchrone imposent à l'appelant de conserver les buffers vivants jusqu'à la complétion ; dans `CudaMemoryManager`, les allocations sont protégées par les events suivis ;
- les CUDA events sont connectés aux transferts du `CudaMemoryManager` ; `poll_completed_transfers` interroge les events sans bloquer et `make_room` l'exécute avant tout plan d'admission/éviction ; `is_transfer_complete` seul ne retire pas l'event ni les protections, et les opérations qui consomment les données doivent toujours attendre explicitement ;
- le stream de transfert utilise `CU_STREAM_NON_BLOCKING` et ne bénéficie donc pas des synchronisations implicites avec le legacy default stream ; tout futur consommateur compute (kernels, cuBLAS/cuBLASLt) devra exprimer les dépendances compute ↔ transfert explicitement, notamment avec `cuStreamWaitEvent` ;
- `make_room` propage toute erreur de polling et refuse alors l'admission, même si l'événement en erreur ne référence pas la victime envisagée : ce comportement fail-closed est intentionnel ;
- l'injection déterministe des erreurs event/stream/free et les tests manager-level avec backend simulé sont différés dans l'issue #19 ;
- le backend CUDA et le `CudaMemoryManager` sont confinés au thread propriétaire (`Rc`, types non `Send`/`Sync`) ; le scheduler M7 et `CudaTransferExecutor` respectent cette affinité en exécutant les opérations sur le thread propriétaire, de manière synchrone ; toute évolution vers une exécution concurrente devra définir explicitement un modèle d'ownership compatible ;
- le contrat `Device` M2 actuel libère ses buffers par destruction (`Drop`) et ne peut pas exprimer une libération backend faillible ni protéger une allocation en vol ; le backend CUDA ne doit pas implémenter ce trait avant qu'un contrat de libération faillible et de garde de complétion préserve les garanties fail-closed de M1 ;
- la validation runtime sous WSL2 / RTX 5090 est attestée pour le SHA `a3ff0fd` par `docs/audits/m1-hardware-validation.md` ; elle couvre le contexte primaire, la compute capability, les allocations pinned/device, le roundtrip H2D/D2H et la libération, mais pas Compute Sanitizer, l'injection de faute fail-closed ni les performances.

La CI standard valide le build par défaut sans exiger un GPU. Les tests matériels marqués `#[ignore]` ne sont pas exécutés par la CI. Le backend CUDA n'est pas encore relié au contrat `Device` de Claymore. `CudaDriver::new` retient le contexte primaire du device ; les opérations inter-ressources vérifient l'identité du contexte. Le raccordement futur devra préserver explicitement l'identité du contexte, l'ownership et la durée de vie des buffers. Le backend et le manager restent confinés au thread propriétaire (`Rc`) ; M7 respecte cette contrainte avec un scheduler et des exécuteurs synchrones `!Send + !Sync`. Aucune implémentation `Send`/`Sync` unsafe n'est envisagée.

---

## 8. Exigences fonctionnelles

### FR-01 — Configuration du runtime

Le runtime doit charger un profil matériel configurable et valider les budgets avant de lancer une exécution. Une configuration invalide doit produire une erreur explicite.

### FR-02 — Gestion des budgets

Le gestionnaire mémoire doit connaître les budgets par tier, comptabiliser les allocations enregistrées et refuser une admission qui violerait les limites applicables.

### FR-03 — Admission et éviction

Le gestionnaire doit pouvoir proposer un plan d'admission et des candidats à l'éviction selon la politique active. Une proposition n'est pas une libération : la comptabilité ne change qu'après confirmation du backend.

Validation : l'adaptateur CUDA opt-in a été exercé sur RTX 5090 pour le SHA `a3ff0fd` (allocation, libération et confirmation comptable) ; voir `docs/audits/m1-hardware-validation.md`. Les tests avec backend simulé restent à construire.

### FR-04 — Sûreté asynchrone

Aucune allocation ne peut être libérée, réutilisée ou évincée tant qu'un transfert ou un kernel susceptible de l'utiliser n'est pas terminé. L'intégration CUDA doit utiliser des events réels pour établir cette disponibilité.

Validation : le roundtrip H2D/D2H de 1 Mio avec deux buffers pinned distincts, la synchronisation des events et la libération des allocations ont été validés sur RTX 5090 pour le SHA `a3ff0fd` ; voir `docs/audits/m1-hardware-validation.md`. Le chemin fail-closed après double échec `event.record`/`synchronize` n'a pas été exercé par injection de faute.

### FR-05 — Gestion des experts

Le cache d'experts doit respecter sa cible et son maximum, tenir compte du coût de chargement et collaborer avec le gestionnaire mémoire pour éviter les dépassements.

### FR-06 — Scheduling et prefetch

Le scheduler doit prioriser le travail prêt à être exécuté, limiter le nombre de préchargements simultanés et éviter les préchargements qui compromettent les ressources requises par le travail courant.

### FR-07 — Diagnostics

La CLI doit exposer les informations utiles au diagnostic du profil, du backend et des capacités matérielles disponibles. Une fonctionnalité CUDA doit signaler clairement l'absence du driver ou d'un device utilisable.

### FR-08 — Exécution des modèles

Chaque moteur de modèle doit pouvoir préparer, exécuter et libérer ses états en respectant les contrats du runtime, des tenseurs Claymore et du backend. L'exécution ne sera considérée livrée qu'après tests fonctionnels sur les modèles ciblés.

---

## 8 bis. Couverture actuelle des exigences

| Exigence | Statut | Justification |
|---|---|---|
| FR-01 — Configuration du runtime | **Livré** | `HardwareProfile` charge et valide le profil JSON ; la CLI utilise le profil embarqué ou `CLAYMORE_HARDWARE_PROFILE`. |
| FR-02 — Gestion des budgets | **Livré pour les budgets déclarés** | Budgets VRAM souple/dur, Host et PinnedHost validés sans chevauchement ; le soft-budget est une cible d'éviction proactive et le hard-ceiling reste la limite d'admission. |
| FR-03 — Admission et éviction | **Livré pour les chemins CPU et CUDA couverts** | Plans déterministes, confirmation après libération backend et exécution CUDA opt-in ; l'injection de fautes reste séparée dans l'issue #19. |
| FR-04 — Sûreté asynchrone | **Chemin nominal livré et validé** | Les allocations en vol restent protégées jusqu'à la complétion de l'event ; les transferts CUDA M7 attendent l'event avant de libérer les dépendances. L'injection de fautes reste hors périmètre. |
| FR-05 — Gestion des experts | **Plafond livré ; cible de working set** | Le maximum est imposé à l'admission et le coût de rechargement peut être enregistré. Aucun chargeur de poids/expert ne l'alimente automatiquement. |
| FR-06 — Scheduling et prefetch | **Ordonnanceur et exécution des files livrés, synchrones** | Graphe commun Compute/Transfer/Prefetch, priorités, dépendances, propagation des échecs et limites de préchargement. Pas de recouvrement concurrent transfert/calcul. |
| FR-07 — Diagnostics | **Livré pour profil et détection CUDA** | La CLI affiche le profil actif et la commande `device` interroge le backend CUDA lorsque la feature est activée. |
| FR-08 — Exécution des modèles | **Non livré** | Les moteurs modèles et le calcul tensoriel natif restent des jalons ultérieurs. |

---

## 10. Feuille de route et phases de développement

Les versions v0.3, v0.4 et v0.5 sont des jalons historiques déjà livrés (politiques mémoire, backend CUDA et adaptateur Memory ↔ CUDA). La feuille de route ci-dessous repart de l'état réel du dépôt et utilise une numérotation de travail M0 à M12 ; elle ne réécrit pas l'historique des versions.

| Phase | Nom | Statut |
|---|---|---|
| M0 | Scaffold et configuration | Livré ; profil matériel versionné, non chargé automatiquement |
| M1 | Consolidation de la fondation matérielle | Livré par la PR #5 ; contexte primaire, sûreté et contrats CUDA |
| M2 | Abstractions Device / Tensor | Livré par la PR #6 |
| M3 | Durcissement des fondations Device (issues #7 à #10) | Livré par la PR #15 |
| M4 | Contrat Device et sûreté mémoire (issues #11 et #12) | Livré par la PR #16 |
| M5 | Robustesse et validation CUDA (issues #13, PR #17 et #18) | Livré ; chemin nominal validé sur RTX 5090 ; injection de fautes isolée dans l'issue #19 |
| M6 | Runtime unifié | **Première tranche livrée par la PR #20** ; orchestration haut niveau et frontières mode d'exécution réparties entre M7/M8 |
| M7 | Scheduler et orchestration mémoire | **Livré sur `main` par la PR #21** |
| M7.1 | Durcissement scheduler/runtime et préparation M8 | **Livré sur `main` par la PR #22** ; CI verte ; revalidation RTX 5090 réussie au SHA de code `c16e925` (17/17 requis par campagne, deux campagnes ; 27/27 tests workspace par campagne) ; limites restantes suivies dans l'issue #23 |
| M8 | Socle tensoriel natif : opérations, dispatch et backends CPU/CUDA | **Livré sur `main` par la PR #24** — Tensor canonique, contrats, dispatch, kernels CUDA F32/BF16 et intégration mémoire/scheduler ; preuves matérielles versionnées |
| M9 | Orchestration de diffusion modulaire commune | Cœur livré par la PR #32 ; compléments CPU-only et documentation suivis dans l'issue #31 |
| M10 | Adaptateur et intégration DiffusionGemma | En cours dans la PR #40 ; génération complète et gates listés non validés |
| M11 | Adaptateur et intégration Clover | Planifié |
| M12 | Optimisation et validation de performance | Planifié |

### M0 — Scaffold et configuration

**Objectif :** disposer d'un workspace Rust, de crates séparés, d'un profil matériel versionné et d'une CLI de diagnostic.

**État :** livré. Le profil JSON existe mais n'est pas encore chargé automatiquement par le runtime.

### M1 — Consolidation de la fondation matérielle

**Objectif :** fiabiliser les primitives CUDA existantes et définir un contrat de contexte utilisable par les couches d'exécution qui partageront les ressources CUDA.

Travaux :
- utiliser le contexte primaire CUDA du device afin de permettre l'interopérabilité avec les bibliothèques CUDA qui utilisent ce contexte ;
- exposer la compute capability détectée ;
- vérifier que buffers, streams et events impliqués dans une opération appartiennent au même contexte ;
- rejeter explicitement les ordinaux invalides et les allocations de taille nulle ;
- faire échouer de manière sûre le gestionnaire mémoire lorsqu'une synchronisation de transfert ne permet plus de prouver la fin des opérations ;
- conserver le confinement au thread propriétaire tant qu'une architecture d'exécution concurrente sûre n'est pas définie ;
- documenter précisément les validations matérielles et les limites restantes.

Critères d'acceptation :
- build par défaut indépendant de CUDA ;
- compilation et Clippy de la feature `cuda-driver` dans la CI sans GPU ; la CI n'exécute pas les tests unitaires qui nécessitent un driver CUDA ;
- tests unitaires des validations de contexte et des tailles, exécutés sur la cible matérielle avec `cargo test -p claymore-cuda -p claymore-memory --features cuda-driver --lib` ;
- test matériel existant couvrant allocation VRAM, pinned host, H2D, D2H, vérification des octets et libération, exécuté sur RTX 5090 ;
- preuves de validation versionnées dans `docs/audits/m1-hardware-validation.md` ;
- raccordement du backend CUDA aux contrats Device/Tensor Claymore à confirmer par des tests d'intégration sur la cible matérielle ;

**Périmètre de la PR M1 :** backend CUDA, adaptateur mémoire, documentation d'audit et mise à jour de cette feuille de route. Aucun scheduler, modèle ou moteur de génération n'est introduit dans ce jalon.

### M2 — Abstractions Device / Tensor

**Objectif :** définir des interfaces communes et explicites pour les devices, buffers, transferts et tenseurs, sans exposer les détails CUDA aux couches supérieures.

**État :** livré pour le contrat backend-neutre et le backend CPU. `claymore-device` expose `Device`, `DeviceBuffer`, `DeviceCapabilities` et `CpuDevice` ; `claymore-tensor` porte les types `Tensor`, `TensorLayout` et `TensorView`. `claymore-memory` fournit `DeviceMemoryManager<D>` qui applique les politiques d'admission et d'éviction sur ces contrats.

Critères d'acceptation :
- backend simulé CPU pour les contrats Device/Tensor et l'adaptateur mémoire : livré et testé sans GPU ;
- contrats de propriété, synchronisation synchrone et capacités documentés ;
- aucune borne Send/Sync implicite : les backends peuvent rester confinés à leur thread/contexte ;
- contrats Device/Tensor détenus par Claymore ; aucune bibliothèque tierce de tenseurs n'est imposée comme dépendance ou cible d'intégration ;
- raccordement du backend CUDA au trait Device, transferts asynchrones avec jetons de complétion et validation GPU : hors M2, à traiter avant ou pendant M4/M5 selon le contrat d'exécution.

### M3 — Durcissement des fondations Device

**Objectif :** rendre les contrats Device fiables et explicites avant de développer le runtime, le scheduling ou le raccordement du backend CUDA.

Travaux :
- déclarer et tester la MSRV Rust 1.86 ;
- distinguer l'identité d'instance du device de son identifiant logique ;
- séparer les erreurs de capacité/overflow des échecs d'allocation ;
- couvrir les cas limites du backend CPU et de l'adaptateur mémoire.

**État :** livré sur `main` par la PR **#15** (fusionnée le 20 novembre 2026) ; voir §7.0.

Critères d'acceptation : les issues #7 à #10 sont couvertes, les tests CPU sont déterministes, la CI passe sur MSRV et stable, et la revue architecturale valide les contrats.

### M4 — Contrat Device et sûreté mémoire

**Objectif :** finaliser les contrats de libération, de propriété et de complétion avant tout raccordement du backend CUDA au trait Device.

Travaux : libération backend faillible, garde des allocations en vol, contrats H2D/D2H et complétion vérifiable, vues tensor non contiguës et clarification des types de buffer. Les copies D2D multi-device sont différées tant que le projet reste mono-GPU. Conserver les garanties fail-closed de M1.

### M5 — Robustesse et validation CUDA

**Objectif :** renforcer les validations matérielles et les tests de sûreté CUDA, sans contourner les contrats établis en M4.

Travaux : finalisation automatique des transferts terminés avant admission/éviction, évictions déterministes et validation matérielle sur la cible RTX 5090. L'injection de fautes sur les chemins event/stream/free et les tests manager-level avec backend simulé sont explicitement différés dans l'issue #19 ; ils ne font pas partie du critère de clôture M5.

### M6 — Runtime unifié

**Objectif :** introduire le cycle de vie des requêtes, les contextes d'exécution, les états, les erreurs, les annulations et l'orchestration de haut niveau.

**État :** première tranche livrée sur `main` par la PR #20 : crate `claymore-runtime`, contrat `GenerationStrategy`, contexte par requête, identifiants monotones, machine d'états explicite, rapports d'exécution, erreurs, annulation coopérative et validation de taille/configuration. L'implémentation reste synchrone et ne suppose ni `Send` ni `Sync` pour les stratégies.

Critères de la tranche livrée :
- exécution d'une requête simulée de bout en bout sur CPU ;
- frontières nettes entre runtime et stratégie de génération ;
- aucune dépendance à un modèle particulier dans le cœur runtime ;
- tests des chemins succès, échec, annulation avant/après démarrage et rejet d'entrée invalide.

La concurrence, l'exécution asynchrone et l'intégration du propriétaire des ressources CUDA relèvent de M7 et des tranches ultérieures. La PR #20 ne les implémente pas.

### M7 — Scheduler et orchestration mémoire

**Objectif :** coordonner calcul, résidence mémoire, transferts et préchargements sans violer l'affinité du backend CUDA.

**Première tranche implémentée dans la PR #21 :** graphe de tâches acyclique par construction (dépendances vers des tâches déjà soumises), files logiques Compute/Transfer/Prefetch, sélection priorité-descendante/FIFO, états explicites, propagation des échecs vers les descendants et limites séparées de concurrence et de lookahead pour les préchargements. Le scheduler est mono-propriétaire (`!Send + !Sync`) et ne lance pas lui-même les tâches.

**Complément d'exécution CPU et d'orchestration :** 
- `claymore-scheduler` expose `QueueRunner` et `TaskExecutor`. Un runner exécute au plus une tâche de sa file sur le thread appelant, sans worker ni borne `Send`/`Sync`.
- `claymore-runtime` expose `ScheduledRuntime`, qui soumet les requêtes comme tâches Compute, respecte priorité/dépendances, puis traduit les rapports Completed/Failed/Cancelled en transitions du scheduler. `submit_task` et `run_queue_one` permettent aux tâches Transfer/Prefetch de partager exactement le même graphe de dépendances ; une requête Compute peut donc attendre un transfert confirmé.
- `claymore-memory` expose `MemoryTaskExecutor`, qui relie les tâches Transfer/Prefetch à l'adaptateur `DeviceMemoryManager<D>` backend-neutre. Les opérations CPU synchrones (allocation, écriture, copie, libération, synchronisation et enregistrement du coût de rechargement) sont exécutées par `QueueRunner` et couvertes par des tests avec `CpuDevice`.
- `QueueRunnerMetrics` fournit les durées et compteurs par file ; `ScheduledRuntimeMetrics` mesure le calcul des requêtes ; `MemoryExecutionMetrics` fournit les compteurs d'opérations, volumes d'octets et durées du chemin mémoire ; `CudaTransferMetrics` isole les volumes H2D/D2H et le temps d'attente des events.
- Le gestionnaire mémoire applique le budget VRAM souple comme cible d'éviction proactive et le plafond dur comme limite d'admission ; si la cible souple est inatteignable, il n'admet que si le plafond dur reste respecté.
- Le plafond dur du cache d'experts est appliqué à chaque admission, avec éviction déterministe des experts inactifs ; la politique adaptive tient compte du coût de rechargement enregistré.
- Les gardes de ressources exposées par `DeviceMemoryManager` refusent les horizons antérieurs au tick courant et toute régression de complétion.

Ces interfaces restent mono-propriétaires et synchrones. Elles ne prétendent pas que le backend CUDA implémente `Device` : `CudaTransferExecutor` possède son `CudaMemoryManager` dédié et s'exécute sur le thread propriétaire. Les tâches H2D/D2H ne retournent le succès qu'après `wait_for_transfer` sur l'event réel ; les lectures pinned, allocations et libérations sont également exposées comme tâches. La campagne matérielle RTX 5090 est versionnée dans `docs/audits/m7-rtx5090-hardware-validation.md` : 17 tests matériels exécutés deux fois (34/34 PASS) sur le SHA de code documenté. Cette preuve couvre le périmètre testé, pas les kernels, l'inférence, les grosses charges soutenues du cache d'experts ni les scénarios d'injection de fautes.

Limites et travaux ultérieurs :
- les copies H2D/D2H sont synchronisées par attente d'event avant le retour de l'exécuteur ; aucun recouvrement transfert/calcul n'est livré ;
- les queues Prefetch et Transfer sont ordonnancées, mais leurs exécuteurs restent synchrones et mono-propriétaires ; aucun préchargement concurrent n'est revendiqué ;
- le coût de rechargement est exposé par l'API mémoire, mais aucun chargeur ne l'alimente automatiquement à ce stade ;
- la campagne matérielle a été rejouée après les corrections B1/B3 : 34/34 exécutions réussies sur le SHA de code `13f06147deb3da5bb66132bc426ba7216e11eeeb`, identique au code du HEAD documentaire `7b1db2164af4de10d36fc4b010e7b8d6d2f746ca` ; voir le rapport versionné.

Critères de la tranche CPU livrée :
- tests déterministes couvrant priorité/FIFO, isolation des files, dépendances, échec en cascade, annulation et bornes de préchargement ;
- cycle de requête ScheduledRuntime ↔ Scheduler vérifié sur CPU ;
- tâches Transfer exécutables via QueueRunner et DeviceMemoryManager sur CpuDevice ;
- admission expert bornée par le plafond dur et allocations protégées jusqu'à l'horizon déclaré ;
- aucun lancement implicite ni appel CUDA depuis le scheduler.

Critères de clôture M7 — tous validés :
- [x] Budgets effectifs respectés : Host 80 Gio + PinnedHost 8 Gio = 88 Gio ; budget VRAM souple utilisé comme cible d'éviction proactive et plafond dur appliqué à l'admission ; plafond maximal du cache experts appliqué.
- [x] Allocations en vol protégées jusqu'à la complétion de leur event CUDA ; les tests matériels H2D/D2H confirment l'attente d'event et l'absence d'opération résiduelle.
- [x] Aucun buffer n'est évincé avant la complétion de ses dépendances ; les gardes d'horizon et les scénarios matériels de protection mémoire sont validés.
- [x] Exécution des files sur leurs threads propriétaires, sans transfert de propriété ; métriques distinctes pour les files, le calcul, les opérations mémoire et les transferts/attentes CUDA.
- [x] Validation logicielle complète : 96 tests workspace, formatage, Clippy, feature CUDA et MSRV Rust 1.86 validés en CI.
- [x] Validation RTX 5090 : deux campagnes de 17 tests, soit 34/34 exécutions réussies sur le SHA de code `13f06147deb3da5bb66132bc426ba7216e11eeeb`.

Les réserves de périmètre restent explicites : exécution synchrone mono-propriétaire, aucun overlap transfert/calcul, aucun prefetch concurrent, aucun kernel ni calcul tensoriel, tests experts à échelle réduite et absence d'injection de fautes (issue #19).

### M7.1 — Durcissement scheduler/runtime et gestion mémoire

**Objectif :** fermer les réserves de robustesse de M7 et figer les contrats d'architecture nécessaires avant M8.

Travaux :
- index de files Ready, adjacency de dépendances et compaction des états terminaux ;
- pruning incrémental des tâches et payloads annulés ;
- containment des panics aux frontières scheduler/runtime ;
- frontière CUDA injectable et tests manager-level déterministes sans GPU ;
- Cargo.lock suivi et CI reproductible avec `--locked` ;
- clarification de la cadence `tick_us`, des bornes workers et des streams réservés ;
- admission VRAM/cache experts conjointe ;
- version CLI issue de Cargo et cycle de vie explicite des résultats de lecture ;
- conception M8 Tensor / Device / Operations / Kernel dispatch / Runtime / Model adapters dans [docs/m8-architecture.md](m8-architecture.md).

Critères d'acceptation :
- CI workspace, feature CUDA sans matériel, Clippy et MSRV verts ;
- revalidation des 17 tests matériels CUDA sur RTX 5090 au SHA final, avec rapport versionné dans `docs/audits/` ;
- aucun contrat public modifié sans revue architecturale ;
- tests démontrant la conservation des gardes et de la comptabilité en cas d'erreur backend ;
- conception M8 relue, sans commencer l'implémentation du moteur tensoriel dans cette PR.

### M8 — Socle tensoriel natif Claymore

**Objectif :** livrer un seul modèle Tensor natif, les opérations de référence, le dispatch explicite et les backends CPU/CUDA qui serviront à l'orchestration de diffusion et aux deux adaptateurs. M8 ne développe aucune stratégie autoregressive ni boucle de génération de modèle.

**État : livré sur `main` par la PR #24**, y compris la finalisation BF16 et la validation matérielle documentée séparément. Le crate claymore-tensor porte le Tensor canonique, les layouts validés, les vues empruntées, les opérations CPU F32 **et BF16** de référence (add, matmul, softmax, RMSNorm), les contrats typés (`contracts.rs`) et le dispatch déterministe avec explication de chaque refus. Les définitions Tensor concurrentes ont été retirées de claymore-device. Les huit kernels CUDA (quatre F32, quatre BF16) sont intégrés via une feature optionnelle, avec deux PTX `sm_120` versionnés, chargement par la Driver API, event de complétion réel et intégration aux budgets/gardes de claymore-memory et aux files du scheduler. BF16 utilise une représentation 16 bits exacte, une conversion FP32 explicite avec arrondi round-to-nearest-even et une accumulation FP32 pour matmul et réductions. FP16 reste non supporté. Le détail figure dans [m8-architecture.md](m8-architecture.md), dans le [rapport F32](audits/m8-rtx5090-hardware-validation.md), dans le [rapport BF16](audits/m8-bf16-rtx5090-hardware-validation.md), dans l'[audit des exigences de graphe](audits/m8-model-graph-requirements.md), dans l'[audit des différences numériques CPU/CUDA](audits/m8-cpu-cuda-numerics.md) et dans le [manifeste de provenance des PTX](audits/m8-ptx-provenance.md). Les opérations spécifiques aux modèles relèvent des adaptateurs M10/M11.

Travaux :
- finaliser les contrats Tensor, layouts, vues, aliasing et durée de vie ;
- formaliser les opérations, contraintes de shape/dtype/layout, erreurs et tolérances numériques ;
- fournir les kernels CPU de référence F32 et BF16 et un dispatch explicite par capacités ;
- raccorder les allocations et opérations aux services memory/scheduler, sans contourner les gardes ;
- implémenter les kernels CUDA F32 et BF16 derrière la feature optionnelle ;
- traiter le prérequis BF16 identifié pour les checkpoints DiffusionGemma et Clover ; la sélection des primitives supplémentaires appartient aux phases M9/M10/M11.

Critères d'acceptation :
- un seul type Tensor canonique dans claymore-tensor ;
- opérations de référence testées sur CPU, sans cast, broadcasting ou fallback implicite ;
- dispatch déterministe et erreurs explicites pour les combinaisons non supportées ;
- intégration des durées de vie et de l'admission mémoire avec les contrats M7/M7.1 ;
- CI workspace, Clippy, format, MSRV et compilation CUDA sans GPU ;
- tests différentiels RTX 5090 F32 et BF16 documentés au SHA de code réellement exécuté, en deux campagnes consécutives.

### M9 — Orchestration de diffusion modulaire commune

**État :** cœur d'orchestration livré par la PR #32 ; les compléments CPU-only de couverture et de documentation sont livrés dans cette contribution (issue #31). Les follow-ups d'équité, de hooks de stratégie, de rétention et de convergence des schedulers restent suivis séparément (#33 à #38).

Travaux livrés dans le cœur :
- cycle de génération par blocs, plan validé et transitions explicites ;
- dépendances de tâches et exécution d'une étape bornée ;
- annulation coopérative aux frontières sûres et propagation d'échec ;
- métriques CPU de sessions, étapes, blocs et durées d'exécution par phase ;
- tests unitaires et d'intégration synthétiques CPU multi-blocs.

Critères de validation CPU-only :
- tests d'intégration couvrant progression multi-blocs, commit, erreur isolée, annulation, session indépendante et nouvelle session après annulation ;
- compteurs reproductibles de blocs/étapes/sessions ; durées monotones par phase exposées comme diagnostics et sans effet sur l'ordonnancement ;
- documentation du cycle de vie, du diagramme de séquence, de la propriété/synchronisation et des limites de reprise ;
- aucune génération de modèle, compatibilité de checkpoint ou métrique GPU revendiquée.

### M10 — Adaptateur DiffusionGemma

**Objectif :** intégrer google/diffusiongemma-26B-A4B-it au moteur de diffusion commun, en conservant les exigences propres à son architecture.

Travaux : lecture du checkpoint et de sa configuration, représentation canvas/état, progression de denoising, attention bidirectionnelle, choix de précision et validation numérique, chargement des poids et gestion des tenseurs intermédiaires.

Critères d'acceptation : étape réelle et reproductible du graphe du modèle, comparaison numérique à une référence fiable, gestion mémoire et campagne RTX 5090 versionnée.

### M11 — Adaptateur Clover

**Objectif :** intégrer Clover-35B-A3B-preview au même moteur de diffusion et aux services Claymore, sans effacer ses spécificités GDN, block diffusion et MoE.

Travaux : lecture du checkpoint/configuration, stratégies GDN et block diffusion, routage expert, cache/résidence/préchargement, synchronisation des transferts et états spécifiques.

Critères d'acceptation : chemin représentatif réel, cohérence du routage et des états, validation numérique, mesures mémoire et campagne RTX 5090 versionnée.

### M12 — Optimisation et validation de performance

**Objectif :** optimiser les chemins réellement identifiés comme limitants sur les deux modèles.

Travaux : profiling, kernels spécialisés, politiques de précision, quantification après validation qualité, offloading, transferts groupés et tuning scheduler/streams.

Critères d'acceptation : benchmarks reproductibles, gains mesurés, respect des budgets et absence de régression de qualité non acceptée.

---

## 11. Stratégie de validation et de développement

### 11.1 Contrôles CI standards

Pour chaque contribution Rust :

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Le workflow vérifie `cargo fmt --all -- --check` sans modifier ni pousser la branche. Les tests CI standards ne remplacent pas les tests CUDA sur la machine cible.

### 11.2 Contrôles CUDA dans WSL2

Après installation et exposition du driver NVIDIA dans WSL2 :

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
nvcc --version
```

Puis, sous WSL2 où la bibliothèque driver est exposée dans `/usr/lib/wsl/lib` :

```bash
RUSTFLAGS="-L native=/usr/lib/wsl/lib" cargo check -p claymore-cuda --features cuda-driver
RUSTFLAGS="-L native=/usr/lib/wsl/lib" cargo run -p claymore-cli --features cuda-driver -- device
```

La disponibilité de `nvcc` dépend de l'installation du toolkit ; le backend Driver API a surtout besoin de la bibliothèque driver et de l'environnement de linkage approprié. La version de nvcc réellement observée ici (release 13.0, V13.0.48), les commandes de régénération `sm_120` et les empreintes SHA-256 des deux PTX sont consignées dans [audits/m8-ptx-provenance.md](audits/m8-ptx-provenance.md).

### 11.3 Diagnostics système

```bash
free -h
lscpu
rustc --version
cargo --version
```

### 11.4 Benchmarks à introduire

Chaque benchmark devra préciser :
- modèle et révision exacte ;
- format et précision des poids ;
- paramètres de génération ;
- warmup et nombre d'itérations ;
- latence, débit et temps de transfert ;
- pic VRAM et RAM ;
- version Rust, implémentation Claymore, driver et environnement ;
- qualité ou métrique de comparaison pertinente.

---

## 12. Gestion des versions, branches et livraisons

Le développement suit des jalons versionnés et des branches dédiées. Les changements structurants doivent être proposés par pull request, avec une description du périmètre, des tests réalisés et des limites connues.

Historique de référence :
- **v0.2** : scaffold initial, politiques et configuration de départ ;
- **v0.3** : memory orchestrator host-side ;
- **v0.4** : backend CUDA Driver API optionnel.

Les PR #1 (v0.3), #2 (v0.4) et #3 (documentation de référence) ont été fusionnées dans `main`. La PR #4 a livré v0.5.0 et l'adaptateur Memory ↔ CUDA (opt-in) ; la PR #5 a livré M1 (consolidation de la fondation matérielle). La PR #6 a livré M2 (`Device` / `Tensor` et simulateur CPU ; le type `Tensor` a ensuite été recentré dans `claymore-tensor` pendant M8). Les PR #15 et #16 ont livré M3 et M4 ; les PR #17 et #18 ont livré M5, dont le chemin nominal a été validé sur RTX 5090. La PR #20 a livré la première tranche M6 ; les PR #21 et #22 ont livré M7 et M7.1 ; la PR #24 a livré M8 ; la PR #32 a livré le cœur de l'orchestration M9. La PR #40 propose l'adaptateur DiffusionGemma M10, mais reste ouverte et ne revendique pas une génération complète du checkpoint.

Une PR ne doit pas être déclarée prête uniquement parce que le code existe : elle doit préciser si les tests couvrent le build par défaut, la feature CUDA, le matériel réel ou seulement un backend simulé.

---

## 13. Risques et décisions ouvertes

| Sujet | Risque / question | Décision ou traitement |
|---|---|---|
| Contexte CUDA | Les ressources CUDA doivent partager un contexte cohérent au sein de Claymore et rester utilisables par les bibliothèques CUDA/NVIDIA autorisées | M1 retient le contexte primaire du device pour éviter des contextes privés concurrents et permettre aux composants Claymore ainsi qu'aux bibliothèques bas niveau (cuBLAS/cuBLASLt, CUTLASS, cuDNN selon §6.4) de travailler dans le même contexte ; le raccordement des buffers CUDA aux contrats Device/Tensor devra être démontré par des tests d'intégration. Ne pas transmettre de pointeurs entre contextes distincts. |
| Moteur tensoriel natif | Effort d'implémentation, correction numérique BF16/FP32 et performances GEMM/attention sur sm_120 | Construire par incréments mesurés ; comparer les sorties numériques à une référence indépendante et les performances aux bibliothèques NVIDIA de référence avant d'activer un chemin optimisé. Les bibliothèques de calcul bas niveau restent autorisées selon la politique de dépendances §6.4. |
| Driver CUDA dans WSL2 | Bibliothèque ou lien linker absent | Vérifier `libcuda.so.1` et le linkage sur la cible |
| Durée de vie async | Libération ou réutilisation prématurée | Ownership explicite et event de complétion |
| Comptabilité mémoire | Écart entre état logique et mémoire réelle | Confirmer uniquement après succès backend |
| Pression VRAM externe | Mémoire disponible variable | Réserve et interrogation runtime |
| Mémoire épinglée | Pression sur l'OS | Plafond strict et instrumentation |
| API CUDA | Contexte incohérent entre buffers/streams/events | Vérification d'identité de contexte dans les opérations du backend (M1) |
| Modèles cibles | APIs et kernels spécifiques | Implémenter séparément et tester chaque moteur |
| Précision mixte | Dégradation numérique | Comparaison de qualité avant activation |
| Kernels spécialisés | Complexité sans gain démontré | Développer après profiling |
| Performances | Paramètres scheduler non optimaux | Benchmarks et tuning sur cible |
| Contrat d'erreur des stratégies | `String` ne conserve pas les causes ni les types d'erreur | Décider avant M8 si le trait expose un type d'erreur associé ou une chaîne de causes typée |
| Ownership des ressources du runtime | Les ressources CUDA sont thread-affines | M7 conserve un scheduler et des exécuteurs `!Send + !Sync` sur leur thread propriétaire ; `CudaTransferExecutor` détient son `CudaMemoryManager`. Aucun recouvrement concurrent transfert/calcul n'est livré. |
| Sorties de génération | Le rapport M6 contient une sortie complète unique | Concevoir une sortie incrémentale seulement avec les besoins concrets des stratégies M9/M10 |

Les choix non tranchés doivent être consignés dans une PR ou une décision d'architecture avant d'être codés comme comportement permanent.

---

## 14. Définition de « terminé »

Une fonctionnalité est considérée terminée lorsque :

1. son périmètre est décrit et ses limites connues sont documentées ;
2. les invariants de sûreté sont respectés ;
3. les tests unitaires et d'intégration adaptés existent ;
4. les contrôles CI applicables passent ;
5. les tests matériels requis ont été exécutés ou sont explicitement signalés comme restant à faire, et leurs preuves sont versionnées dans le dépôt ;
6. les métriques ou preuves nécessaires sont disponibles et versionnées dans le dépôt, et pas uniquement dans des commentaires GitHub ;
7. la documentation et les exemples reflètent le comportement réel.

Un prototype compilable n'est pas automatiquement une fonctionnalité prête pour la production.

---

## 15. Documents associés

Ce document est la référence consolidée. Les documents spécialisés ci-dessous peuvent conserver les détails d'implémentation propres à un sous-système ; en cas de divergence, cette référence et l'état réel du code doivent être réconciliés dans une mise à jour documentaire.

- `README.md` — présentation rapide et commandes d'entrée.
- `docs/architecture.md` — vue d'architecture.
- `docs/memory-orchestrator-v0.3.md` — détails de la couche de politique mémoire.
- `docs/cuda-backend-v0.4.md` — détails du backend CUDA.
- `docs/memory-profile.md` — profil mémoire RTX 5090.
- `docs/wsl2.md` — configuration WSL2.
- `docs/audits/m4-m5-rtx5090-hardware-validation.md` — campagne et preuves matérielles M4/M5 sur RTX 5090 (incluant la procédure de revalidation B1).
- `docs/audits/m2-device-tensor-design.md` — contrats Device/Tensor et validation CPU.
- `docs/audits/m3-device-foundations.md` — audit et durcissement des fondations Device.
- `docs/audits/m4-device-lifecycle-contracts.md` — audit des contrats de libération, complétion asynchrone et vues tensor.
- `docs/m8-architecture.md` — socle tensoriel natif, politique numérique F32/BF16 et tolérances.
- `docs/audits/m8-model-graph-requirements.md` — exigences de graphe et frontières M8/M9/M10/M11.
- `docs/audits/m8-rtx5090-hardware-validation.md` — campagne matérielle F32.
- `docs/audits/m8-bf16-rtx5090-hardware-validation.md` — campagne matérielle BF16.
- `docs/audits/m8-cpu-cuda-numerics.md` — différences numériques réelles CPU/CUDA (sous-normaux, réductions, `exp`, BF16/RNE).
- `docs/audits/m8-ptx-provenance.md` — provenance, versions d'outils et empreintes SHA-256 des PTX intégrés.
- `config/rtx5090-7950x3d.json` — profil machine versionné.

---

## 16. Journal des décisions

| Date | Décision |
|---|---|
| 2026-11-18 | RTX 5090 32 Gio, Ryzen 9 7950X3D, 128 Gio RAM et WSL2 retenus comme cible de référence. |
| 2026-11-18 | BF16 privilégié, FP32 pour les accumulations sensibles, FP16 en repli ; quantification basse précision différée. |
| 2026-11-18 | Budget VRAM souple 26 Gio, plafond 28 Gio et réserve de sécurité 4 Gio. |
| 2026-11-18 | Cache experts cible 14 Gio, maximum 16 Gio, inclus dans le working set modèles / experts. |
| 2026-11-18 | Le memory orchestrator reste testable sans CUDA ; le backend CUDA est une feature optionnelle. |
| 2026-11-18 | Une éviction planifiée ne devient comptable qu'après confirmation de la libération effective. |
| 2026-11-18 | La sûreté des opérations asynchrones doit être fondée sur les CUDA events réels, pas seulement sur un tick logique. |
| 2026-11-18 | RAM hôte : enveloppes disjointes 64 Gio Host weights + 16 Gio Host staging/state + 8 Gio PinnedHost = 88 Gio ; limites de tiers Host 80 Gio et PinnedHost 8 Gio. |
| 2026-11-19 | Référentiels RAM séparés : machine 128 Gio, plafond WSL2 115 GB, budget applicatif Claymore 88 Gio ; environ 13 Gio restent hors VM et la marge Linux au-dessus du budget applicatif doit être mesurée. |
| 2026-11-19 | Le backend retient le contexte primaire CUDA du device pour éviter des contextes privés concurrents et permettre le partage cohérent des buffers, streams et events entre composants Claymore et bibliothèques CUDA/NVIDIA bas niveau autorisées (§6.4). Le profilage (Nsight Systems/Compute) sert à mesurer les chemins réels ; le raccordement aux contrats Device/Tensor natifs sera validé par des tests d'intégration dédiés. |
| 2026-11-19 | Correspondance de la feuille de route précédente : ancien M2 (backend Driver) → socle livré v0.4/M0 ; ancien M3 (Memory ↔ CUDA) → v0.5/M0 et durcissement M1 ; ancien M4 (scheduler) → nouveau M4 ; ancien M5 (moteurs modèles) → M5/M6/M7 selon le chemin d'exécution ; ancien M6 (optimisation) → M8. Les tests simulés de l'adaptateur sont reportés à M2/M3 ; la preuve de protection d'une allocation en vol jusqu'à l'event relève de M4. |
| 2026-11-19 | Le backend CUDA utilise `Rc` car son contexte n'est pas `Send`/`Sync` : driver, buffers, streams, events et `CudaMemoryManager` sont confinés au thread propriétaire ; M4 doit prévoir un thread CUDA dédié ou une évolution sûre de ce modèle. |
| 2026-11-18 | Cette référence est rédigée en français ; README et documentation technique existante restent en anglais. |
| 2026-11-18 | DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) et Clover disposent d'adaptateurs spécialisés partageant le runtime, le socle tensoriel et les services transverses. |
| 2026-11-20 | Claymore ne fonde pas son moteur tensoriel sur un runtime tensoriel externe : les contrats Device/Buffer/Tensor, l'ownership, la synchronisation, la politique mémoire et l'exécution des modèles doivent rester cohérents et maîtrisés dans le projet, sans adapter les invariants Claymore au cycle de vie d'un framework tiers. Ce choix évite une dépendance structurante et les frictions d'interopérabilité autour des contextes CUDA, des buffers et des transferts ; il augmente en contrepartie fortement l'effort d'implémentation, de maintenance, de correction numérique et d'optimisation. Claymore pourra s'appuyer sur des bibliothèques de calcul bas niveau et des outils CUDA selon §6.4, sans leur déléguer ses abstractions ni son runtime. |
| 2026-11-20 | Candle et les autres projets externes restent des références d'étude uniquement ; Claymore possède ses contrats Device/Buffer/Tensor et son futur moteur tensoriel natif. DiffusionGemma est référencé par son identifiant Hugging Face exact. |
| 2026-11-20 | M4 introduit une libération de buffer explicite et faillible, des tokens de complétion H2D/D2H vérifiables par CUDA event et des vues tensor stridées ; aucun raccordement CUDA au trait Device n'est autorisé avant validation de ces contrats. |
| 2026-11-21 | M6 (PR #20, tranche initiale) : runtime synchrone et mono-exécution, enveloppes opaques, stratégie sans contrainte `Send`/`Sync`, contexte par requête et annulation coopérative. Le périmètre initial est explicitement partiel : orchestration haut niveau et frontières de mode d'exécution reportées à M7/M8 ; panique de stratégie propagée (runtime à abandonner), annulation prioritaire sur le résultat simultané, limite `max_payload_bytes` appliquée au seul payload. Les choix de type d'erreur, ownership Device/Memory et sortie incrémentale restent ouverts avant M8/M9. |
| 2026-11-21 | M5 utilise un stream de transfert `CU_STREAM_NON_BLOCKING` : les consommateurs compute futurs doivent exprimer explicitement leurs dépendances avec les transferts (par exemple `cuStreamWaitEvent`). Une erreur de polling dans `make_room` bloque l'admission en mode fail-closed. L'injection de fautes et le backend CUDA simulable sont suivis séparément dans l'issue #19. |
| 2026-11-22 | RAM hôte M7 : les tiers sont disjoints ; Host = weights pool + staging/state (80 Gio), PinnedHost = 8 Gio, total = budget Claymore 88 Gio. Le profil rejette tout dépassement de cette somme. |
| 2026-11-22 | VRAM M7 : le soft-budget de 26 Gio est une cible d'éviction proactive, le hard-ceiling de 28 Gio est la limite stricte d'admission ; le cache experts (cible 14 Gio, maximum 16 Gio) est inclus dans le working set VRAM. |
| 2026-11-22 | M7 : scheduler, `QueueRunner` et exécuteur CUDA sont synchrones et mono-propriétaires ; H2D/D2H ne complètent une tâche qu'après attente de l'event. Aucun overlap transfert/calcul ou prefetch concurrent n'est revendiqué. |
| 2026-11-22 | Éviction Adaptive : comparer d'abord la chaleur (hit count), puis la récence (tick), et utiliser le coût de rechargement en nanosecondes uniquement comme départage ; un coût inconnu vaut zéro pour ce seul départage. Aucun chargeur ne l'alimente automatiquement à ce stade. |
| 2026-11-22 | Finalisation M8 : BF16 est implémenté en 16 bits exacts avec conversion FP32 explicite et arrondi round-to-nearest-even. Matmul et réductions accumulent en FP32 ; la sortie BF16 est arrondie une seule fois. FP16 reste non supporté et aucun cast/fallback implicite n'est autorisé. |
| 2026-11-22 | Les deux modules PTX (`m8_f32.ptx` et `m8_bf16.ptx`) ciblent `sm_120` ; les kernels BF16 utilisent `cvt.rn.bf16.f32` et `fma.rn.f32`. La tolérance matérielle BF16 est exprimée en pas représentables (0 pour add, 1 pour matmul/softmax/RMSNorm), fixée avant la campagne. |\n| 2026-11-23 | DiffusionGemma est initialement ciblé en génération de texte uniquement ; image et vidéo sont hors périmètre sans décision produit et spécification multimodale ultérieures. Le runtime M9 expose les métriques CPU d'orchestration et des durées observées par phase, sans métriques GPU simulées. |

### Correspondance de la feuille de route historique

| Ancien jalon | Périmètre historique | Nouveau jalon |
|---|---|---|
| M2 | Backend CUDA Driver API | v0.4 livré ; socle historique M0 |
| M3 | Intégration Memory ↔ CUDA | v0.5 livré dans le socle ; durcissement de sûreté M1 |
| M4 | Scheduler, transferts et cache experts | M4 — Scheduler et orchestration mémoire |
| M5 | Moteur tensoriel natif et premier chemin modèle | M5 — moteur tensoriel Claymore, M6 — autoregressif, M7 — diffusion |
| M6 | Optimisation et validation RTX 5090 | M8 — Optimisation et validation de performance |

Les critères de l'ancien M3 qui n'étaient pas couverts par le nouveau découpage sont explicitement réaffectés : backend simulé de l'adaptateur en M2/M3 ; démonstration de protection d'une allocation en vol jusqu'à la complétion de son event en M4.

---

## 17. Références externes

Les références exactes (URL et révision/commit) doivent être consignées avant de figer les choix d'implémentation. Elles ne sont pas reproduites ici tant que leur URL canonique et leur révision n'ont pas été vérifiées :

- FreeToken — URL/révision à compléter après vérification.
- Strata (Niko1221) — URL/révision à compléter après vérification.
- DiffusionGemma — `google/diffusiongemma-26B-A4B-it` ; révision exacte du dépôt à consigner avant l'implémentation du chargeur.
- Clover-35B-A3B-preview — dépôt Hugging Face et révision à compléter après vérification.


**Fin du document de référence Claymore.**
