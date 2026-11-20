# Claymore — Référence produit, fonctionnelle et technique

> **Document de référence unique** — cahier des charges, Product Requirements Document (PRD), spécification technique et feuille de route.
>
> **Projet :** Claymore  
> **Dépôt :** https://github.com/zenhit777/Claymore  
> **Version de référence du workspace :** 0.5.0  
> **État du document :** référence vivante, à mettre à jour à chaque jalon  
> **Dernière mise à jour :** 20 novembre 2026

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

Les deux cibles fonctionnelles initiales sont indépendantes et doivent disposer chacune de leur propre moteur/adaptateur.

### 3.1 DiffusionGemma

**Modèle cible exact :** `google/diffusiongemma-26B-A4B-it` (identifiant Hugging Face).

Objectif : construire un chemin d'exécution adapté à ce modèle de diffusion de texte. Les mentions de « Gemma » seules désignent la famille ou le backbone architectural, jamais une cible modèle interchangeable.

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

Les moteurs DiffusionGemma et Clover peuvent partager le runtime, les abstractions de mémoire, le scheduler, le backend CUDA et les utilitaires de tenseurs. Ils ne doivent pas être forcés dans une abstraction commune qui masquerait des différences fondamentales de calcul ou de cycle de vie.

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
| Backend de tenseurs visé | Moteur tensoriel natif Claymore (à construire) |
| Politique numérique | précision mixte, BF16 privilégié lorsque supporté |

Cette configuration est le profil de référence, pas une limite architecturale du projet. Les paramètres matériels doivent rester configurables.

### 4.1 Politique de précision

Décision initiale :

- **BF16** comme précision préférée lorsque le matériel, le modèle et les kernels la supportent correctement ;
- **FP32** pour les accumulations ou opérations sensibles qui le nécessitent ;
- **FP16** comme solution de repli lorsque BF16 n'est pas disponible ou approprié ;
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

2. **Moteurs de modèles**
   - adaptateur DiffusionGemma ;
   - adaptateur Clover ;
   - préparation des tenseurs, exécution et états propres à chaque modèle.

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
| `claymore-device` | Contrats Device/Buffer/Tensor indépendants du backend et simulateur CPU |
| `claymore-memory` | Budgets, comptabilité, admission et politiques d'éviction |
| `claymore-scheduler` | Planification du travail et politique de préchargement |
| `claymore-model-gemma` | Frontière d'intégration du modèle DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) |
| `claymore-model-clover` | Frontière d'intégration Clover |
| `claymore-cuda` | Backend optionnel CUDA Driver API |
| `claymore-cli` | Commandes, profil et diagnostics |

Le workspace utilise Cargo, resolver 2, Rust edition 2021. La version de référence actuelle est 0.5.0.

`claymore-scheduler` ne fournit actuellement qu'une structure `SchedulerConfig` (`cpu_workers`, `max_prefetches`, `lookahead`) et sa validation ; aucune boucle ni logique de planification n'est implémentée. La validation accepte de 1 à 12 workers. Les valeurs minimales de 6 workers, les nombres de streams et le tick de 500 µs figurant dans le profil JSON sont des paramètres documentaires, pas des contraintes appliquées par le code.

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


---

## 7. État réel de l'implémentation

Cette section prévaut sur toute formulation plus ambitieuse de la vision cible. La consolidation matérielle M1 est livrée par la PR #5 : contexte primaire CUDA, contrôles d'appartenance des ressources, compute capability et comportement fail-closed du gestionnaire mémoire.

### 7.1 Livré sur main

#### v0.3 — Memory Orchestrator (politiques host-side)

Le crate `claymore-memory` contient :

- validation typée des budgets VRAM, RAM hôte et mémoire hôte épinglée ;
- comptabilisation des allocations par tier ;
- planification d'admission ;
- sélection de candidats à l'éviction ;
- politiques LRU, LFU et adaptive ;
- protection logique des allocations avec `safe_after_tick` ;
- confirmation explicite d'éviction avant mise à jour de la comptabilité ;
- getters de configuration et compteur d'usage du cache d'experts ; la cible de 14 Gio et le maximum de 16 Gio ne sont pas appliqués à l'admission.

Le budget souple VRAM de 26 Gio est actuellement validé par rapport au plafond dur, mais n'est pas appliqué comme limite d'admission : `limit_for(Vram)` renvoie le plafond dur de 28 Gio. Le champ working set de 18 Gio n'existe pas dans `MemoryBudgets` ; l'inclusion du cache d'experts dans ce working set reste donc un invariant documentaire non vérifié par le code.

Cette version est une couche de politique host-side. Elle n'alloue pas elle-même la VRAM, ne réserve pas réellement de pages épinglées et ne réalise pas de transfert CUDA.

#### Scheduler et profil CLI — état réel

`claymore-scheduler` ne contient qu'une configuration et une méthode `validate()` ; il ne planifie pas de tâches et ne pilote aucun préchargement. Le fichier `config/rtx5090-7950x3d.json` est versionné mais n'est pas chargé par le runtime. La commande `claymore-cli profile` affiche les valeurs de `SchedulerConfig::default()` et ne reflète donc pas le profil matériel JSON.

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

### 7.2 Non livré sur main / en cours de revue

**M2 — PR #6 :** les contrats `Device` / `DeviceBuffer` / `Tensor`, `CpuDevice` et `DeviceMemoryManager<D>` sont implémentés sur la branche `feat/m2-device-tensor-abstractions`, mais ne sont pas encore livrés sur `main` tant que la PR #6 n'est pas fusionnée. L'implémentation initiale a été validée par la CI sur le SHA `de83b47bbf359a30166afa0130e06edf18869949` ; les corrections demandées par la revue architecturale sont intégrées dans cette branche et leur validation CI est en cours.

Les éléments suivants ne doivent pas être considérés comme fonctionnels de bout en bout :

- chargement du profil matériel JSON par le runtime ou la CLI ;
- application effective du budget souple VRAM de 26 Gio ;
- enforcement de la cible/max du cache d'experts et de son inclusion dans un working set explicite ;
- logique de scheduling, files de tâches et préchargement ;
- polling non bloquant des CUDA events et finalisation automatique des transferts terminés avant admission ;
- adaptateur CUDA spécifique injectable et tests du cycle de vie CUDA avec backend simulé ;
- pool allocator VRAM, réutilisation de blocs et défragmentation ;
- moteur de transfert piloté par le scheduler ;
- prefetch prédictif intégré au cache d'experts ;
- télémétrie consolidée CPU/RAM/VRAM/transferts ;
- moteur tensoriel natif et chemin d'inférence effectif ;
- chargement et exécution réels de DiffusionGemma ;
- chargement et exécution réels de Clover-35B-A3B-preview ;
- kernels CUDA spécialisés et kernel dispatcher opérationnel ;
- validation de performance et de qualité sur la RTX 5090 cible.

Les crates modèles constituent à ce stade des frontières de workspace, pas la preuve que les modèles sont exécutables.

### 7.3 Limites connues du backend CUDA actuel

- compute capability exposée via `CudaDeviceInfo` ;
- sélection du device par ordinal uniquement ;
- pas de pool allocator, de défragmentation, de télémétrie ou d'éviction automatique ;
- les appels de copie asynchrone imposent à l'appelant de conserver les buffers vivants et de synchroniser avant leur destruction ou réutilisation ;
- les CUDA events sont connectés aux transferts du `CudaMemoryManager` ; leur complétion est traitée par attente explicite, sans polling non bloquant ni finalisation automatique lors d'une admission ;
- le backend CUDA et le `CudaMemoryManager` sont confinés au thread propriétaire (`Rc`, types non `Send`/`Sync`) ; le scheduler M4 devra déléguer les opérations CUDA à un thread propriétaire dédié ou faire évoluer explicitement ce modèle de concurrence ;
- le contrat `Device` M2 actuel libère ses buffers par destruction (`Drop`) et ne peut pas exprimer une libération backend faillible ni protéger une allocation en vol ; le backend CUDA ne doit pas implémenter ce trait avant qu'un contrat de libération faillible et de garde de complétion préserve les garanties fail-closed de M1 ;
- la validation runtime sous WSL2 / RTX 5090 est attestée pour le SHA `a3ff0fd` par `docs/audits/m1-hardware-validation.md` ; elle couvre le contexte primaire, la compute capability, les allocations pinned/device, le roundtrip H2D/D2H et la libération, mais pas Compute Sanitizer, l'injection de faute fail-closed ni les performances.

La CI standard valide le build par défaut sans exiger un GPU. Les tests matériels marqués `#[ignore]` ne sont pas exécutés par la CI. Le backend CUDA n'est pas encore relié au contrat `Device` de Claymore. `CudaDriver::new` retient le contexte primaire du device ; les opérations inter-ressources vérifient l'identité du contexte. Le raccordement futur devra préserver explicitement l'identité du contexte, l'ownership et la durée de vie des buffers. Le backend et le manager restent confinés au thread propriétaire (`Rc`) ; aucune implémentation `Send`/`Sync` unsafe n'est envisagée.

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

| Exigence | Statut sur `main` | Justification |
|---|---|---|
| FR-01 — Configuration du runtime | **Non livré** | Le profil JSON existe, mais aucun chargeur ne l'applique ; la CLI utilise `SchedulerConfig::default()`. |
| FR-02 — Gestion des budgets | **Partiel** | Comptabilité host-side et validation des budgets ; le budget souple VRAM n'est pas appliqué à l'admission. |
| FR-03 — Admission et éviction | **Livré (host-side) ; exécution CUDA opt-in validée sur RTX 5090 pour le SHA `a3ff0fd`** | Adaptateur d'allocation/libération CUDA et confirmation après free livrés derrière `cuda-driver` ; validation matérielle disponible dans `docs/audits/m1-hardware-validation.md` ; tests avec backend simulé encore manquants. |
| FR-04 — Sûreté asynchrone | **Partiel, avec validation GPU du chemin nominal** | Events réels reliés aux transferts et allocations protégées jusqu'à synchronisation explicite ; roundtrip H2D/D2H et libération validés sur RTX 5090 (`docs/audits/m1-hardware-validation.md`) ; pas de polling non bloquant, de backend simulé ni d'injection de faute fail-closed. |
| FR-05 — Gestion des experts | **Non livré** | Getters/compteur seulement ; cible et maximum non appliqués à l'admission. |
| FR-06 — Scheduling et prefetch | **Non livré** | Configuration seulement ; aucune logique de scheduling/prefetch. |
| FR-07 — Diagnostics | **Partiel** | CLI de base et diagnostic device optionnel ; le profil JSON n'est pas chargé. |
| FR-08 — Exécution des modèles | **Non livré** | Pas de moteur tensoriel complet ni d'inférence réelle des deux modèles. |

## 9. Exigences non fonctionnelles et invariants

### NFR-01 — Sécurité mémoire

Pas de use-after-free GPU, pas de libération anticipée d'un buffer en vol, pas de confirmation comptable d'une éviction non effectuée.

### NFR-02 — Résilience

L'indisponibilité du backend CUDA ne doit pas empêcher les tests des politiques host-side ni le build par défaut.

### NFR-03 — Observabilité

Les futures versions doivent exposer les allocations par tier, les budgets, les évictions, les transferts, les temps d'attente et les raisons d'échec d'admission.

### NFR-04 — Reproductibilité

Les tests de politiques doivent être déterministes. Les benchmarks doivent documenter modèle, paramètres, précision, warmup, nombre d'itérations, consommation mémoire et environnement.

### NFR-05 — Qualité Rust

Chaque changement doit passer le formatage, les tests et Clippy sur les configurations prises en charge. Les blocs `unsafe` doivent être minimisés, documentés et entourés de contrats vérifiables.

### Invariants impératifs

1. Un budget est une limite de planification, pas une mesure de mémoire libre.
2. La VRAM ne doit jamais être planifiée à 100 %.
3. Le swap n'est pas un tier de poids.
4. Un candidat à l'éviction n'est pas une éviction confirmée.
5. Une allocation en vol n'est pas libérable.
6. Objectif d'architecture : le cache d'experts doit être inclus dans le working set ; cet invariant n'est pas encore représenté ni vérifié par `MemoryBudgets`.
7. Une validation CI sans CUDA ne prouve pas le fonctionnement du backend CUDA.
8. Une compilation réussie ne prouve pas la correction numérique ni la performance d'un modèle.

---

## 10. Feuille de route et phases de développement

Les versions v0.3, v0.4 et v0.5 sont des jalons historiques déjà livrés (politiques mémoire, backend CUDA et adaptateur Memory ↔ CUDA). La feuille de route ci-dessous repart de l'état réel du dépôt et utilise une numérotation de travail M0 à M8 ; elle ne réécrit pas l'historique des versions.

| Phase | Nom | Statut |
|---|---|---|
| M0 | Scaffold et configuration | Livré ; profil matériel versionné, non chargé automatiquement |
| M1 | Consolidation de la fondation matérielle | Livré par la PR #5 ; contexte primaire, sûreté et contrats CUDA |
| M2 | Abstractions Device / Tensor | Implémenté dans la PR #6 ; revue architecturale en cours, non fusionné |
| M3 | Runtime unifié | Planifié |
| M4 | Scheduler et orchestration mémoire | Planifié |
| M5 | Moteur tensoriel Claymore et premier chemin modèle | Planifié |
| M6 | Stratégie autoregressive | Planifié |
| M7 | Stratégie diffusion (DiffusionGemma / Clover) | Planifié |
| M8 | Optimisation et validation de performance | Planifié |

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

**État :** livré pour le contrat backend-neutre et le backend CPU. `claymore-device` expose `Device`, `DeviceBuffer`, `DeviceCapabilities`, `Tensor` et `CpuDevice`. `claymore-memory` fournit `DeviceMemoryManager<D>` qui applique les politiques d'admission et d'éviction sur ces contrats.

Critères d'acceptation :
- backend simulé CPU pour les contrats Device/Tensor et l'adaptateur mémoire : livré et testé sans GPU ;
- contrats de propriété, synchronisation synchrone et capacités documentés ;
- aucune borne Send/Sync implicite : les backends peuvent rester confinés à leur thread/contexte ;
- contrats Device/Tensor détenus par Claymore ; aucune bibliothèque tierce de tenseurs n'est imposée comme dépendance ou cible d'intégration ;
- raccordement du backend CUDA au trait Device, transferts asynchrones avec jetons de complétion et validation GPU : hors M2, à traiter avant ou pendant M4/M5 selon le contrat d'exécution.

### M3 — Runtime unifié

**Objectif :** introduire le cycle de vie des requêtes, les contextes d'exécution, les états, les erreurs, les annulations et l'orchestration de haut niveau.

Critères d'acceptation :
- exécution d'une requête simulée de bout en bout sur CPU ;
- frontières nettes entre runtime, architecture de modèle, stratégie de génération et mode d'exécution ;
- aucune dépendance à un modèle particulier dans le cœur runtime.

### M4 — Scheduler et orchestration mémoire

**Objectif :** coordonner calcul, résidence mémoire, transferts et préchargements.

Travaux :
- formaliser tâches, états, dépendances et priorités ;
- définir le propriétaire des opérations CUDA et leur file d'exécution ;
- gérer les files compute et transfer, les limites de préchargement et le lookahead ;
- intégrer les politiques LRU/LFU/adaptive aux coûts de chargement et aux métriques observées ;
- protéger les ressources jusqu'à la fin de toutes les opérations dépendantes.

Critères d'acceptation :
- tests déterministes avec backend simulé ;
- aucun dépassement de budget ;
- démonstration qu'une allocation en vol reste protégée jusqu'à la complétion de son event (critère de validation reporté de l'ancien M3) ;
- aucun buffer évincé avant complétion de ses dépendances ;
- mesures séparées du temps de transfert, d'attente et de calcul.

### M5 — Moteur tensoriel Claymore et premier chemin modèle

**Objectif :** construire le premier chemin tensoriel natif de Claymore et obtenir un forward pass reproductible sur un modèle de référence.

Travaux :
- définir et stabiliser les primitives de calcul tensoriel natives de Claymore ;
- définir le chargement et la représentation des poids ;
- connecter les opérations tensorielle aux contrats Device / Tensor / Memory et au backend d'exécution Claymore ;
- valider formes, précision, sorties et durée de vie des buffers.

Critères d'acceptation :
- forward pass reproductible sur CPU ou GPU selon le modèle ;
- test d'interopérabilité avec le contexte CUDA partagé ;
- mesure du pic RAM/VRAM et libération propre des ressources.

### M6 — Stratégie autoregressive

**Objectif :** fournir une stratégie de génération autoregressive réutilisant le runtime unifié.

Travaux : prefill, décodage token par token, KV cache, sampling, arrêt et gestion de plusieurs requêtes.

Critères d'acceptation : génération de référence reproductible, tests du cache et mesure de latence/débit.

### M7 — Stratégie diffusion (DiffusionGemma / Clover)

**Objectif :** ajouter la génération itérative par canvas et les besoins propres aux architectures ciblées.

Travaux : états de canvas, denoising, politiques de confiance, commit, replay causal si requis, self-conditioning, GDN et routage MoE selon le modèle.

Critères d'acceptation : cas de génération reproductible, cohérence des états, qualité comparée à une référence et mesures mémoire.

### M8 — Optimisation et validation de performance

**Objectif :** optimiser les chemins réellement identifiés comme limitants.

Travaux : profiling, expert cache, offloading, transferts groupés, kernels spécialisés si justifiés, quantification et tuning des streams/workers.

Critères d'acceptation : benchmarks reproductibles, gains mesurés, absence de régression de qualité non acceptée et respect des budgets mémoire.

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

La disponibilité de `nvcc` dépend de l'installation du toolkit ; le backend Driver API a surtout besoin de la bibliothèque driver et de l'environnement de linkage approprié.

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

Les PR #1 (v0.3), #2 (v0.4) et #3 (documentation de référence) ont été fusionnées dans `main`. La PR #4 a livré v0.5.0 et l'adaptateur Memory ↔ CUDA (opt-in). La PR #6 porte M2 (`Device` / `Tensor` et simulateur CPU) ; son implémentation est en revue et n'est pas encore intégrée à `main`.

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
- `docs/audits/m2-device-tensor-design.md` — contrats Device/Tensor et validation CPU.
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
| 2026-11-18 | DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) et Clover disposent de moteurs distincts partageant les services transverses du runtime. |
| 2026-11-20 | Claymore ne fonde pas son moteur tensoriel sur un runtime tensoriel externe : les contrats Device/Buffer/Tensor, l'ownership, la synchronisation, la politique mémoire et l'exécution des modèles doivent rester cohérents et maîtrisés dans le projet, sans adapter les invariants Claymore au cycle de vie d'un framework tiers. Ce choix évite une dépendance structurante et les frictions d'interopérabilité autour des contextes CUDA, des buffers et des transferts ; il augmente en contrepartie fortement l'effort d'implémentation, de maintenance, de correction numérique et d'optimisation. Claymore pourra s'appuyer sur des bibliothèques de calcul bas niveau et des outils CUDA selon §6.4, sans leur déléguer ses abstractions ni son runtime. |
| 2026-11-20 | Candle et les autres projets externes restent des références d'étude uniquement ; Claymore possède ses contrats Device/Buffer/Tensor et son futur moteur tensoriel natif. DiffusionGemma est référencé par son identifiant Hugging Face exact. |

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
