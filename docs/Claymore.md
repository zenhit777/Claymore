# Claymore — Référence produit, fonctionnelle et technique

> **Document de référence unique** — cahier des charges, Product Requirements Document (PRD), spécification technique et feuille de route.
>
> **Projet :** Claymore  
> **Dépôt :** https://github.com/zenhit777/Claymore  
> **Version de référence du workspace :** 0.5.0 (candidat PR #4)  
> **État du document :** référence vivante, à mettre à jour à chaque jalon  
> **Dernière mise à jour :** 18 novembre 2026

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

1. une exécution de modèles fondée sur **Hugging Face Candle** ;
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
- une compatibilité universelle avec toutes les architectures Candle ;
- une promesse de faire tenir intégralement en VRAM un modèle dont les poids dépassent le budget disponible ;
- l'usage du swap système comme niveau normal de stockage des poids.

---

## 3. Périmètre modèles

Les deux cibles fonctionnelles initiales sont indépendantes et doivent disposer chacune de leur propre moteur/adaptateur.

### 3.1 DiffusionGemma

Objectif : construire un chemin d'exécution adapté à un modèle de type diffusion de texte.

Les travaux spécifiques prévus comprennent :

- gestion du canvas ou de l'état de génération ;
- boucle de denoising et contrôle des étapes ;
- prise en charge des besoins d'attention bidirectionnelle propres à l'architecture ;
- durée de vie des tenseurs intermédiaires et des états de génération ;
- choix de précision par opération et validation numérique ;
- allocation et libération prévisibles des buffers temporaires.

**Point de vigilance :** la gestion d'un KV cache classique ne doit pas être supposée identique à celle d'un modèle autoregressif. La conception devra suivre les besoins effectifs de DiffusionGemma et de son implémentation Candle.

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

Les moteurs Gemma et Clover peuvent partager le runtime, les abstractions de mémoire, le scheduler, le backend CUDA et les utilitaires de tenseurs. Ils ne doivent pas être forcés dans une abstraction commune qui masquerait des différences fondamentales de calcul ou de cycle de vie.

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
| Backend de tenseurs visé | Hugging Face Candle |
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
| Réserve système | 24 Gio |

Les enveloppes RAM hôte sont **disjointes** : 64 + 16 + 8 = 88 Gio. Le pool poids/experts et le staging/états utilisent le tier `Host`, dont la limite doit donc être configurée à **80 Gio** ; le tier `PinnedHost` est limité séparément à **8 Gio**. La mémoire épinglée n'est pas comprise dans les 80 Gio du tier `Host`.

Trois référentiels ne doivent pas être confondus : machine physique (128 Gio), VM WSL2 (96 Gio) et budget applicatif Claymore (88 Gio). La marge théorique restante dans WSL2 est d'environ 8 Gio pour Linux et le page cache ; elle doit être mesurée sur la machine cible. La réserve d'environ 24 Gio concerne la mémoire physique hors VM et reste indicative. **VRAM : postes imbriqués ; RAM hôte : enveloppes disjointes.**

### 5.3 WSL2

Configuration de départ proposée dans `%UserProfile%\.wslconfig` :

```ini
[wsl2]
memory=96GB
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
   - Candle pour les tenseurs et l'exécution des modèles ;
   - backend CUDA pour les allocations, transferts, streams et events ;
   - kernels spécialisés lorsque leur nécessité est démontrée.

### 6.2 Crates du workspace

| Crate | Responsabilité |
|---|---|
| `claymore-core` | Types partagés, configuration et contrats transverses |
| `claymore-memory` | Budgets, comptabilité, admission et politiques d'éviction |
| `claymore-scheduler` | Planification du travail et politique de préchargement |
| `claymore-model-gemma` | Frontière d'intégration DiffusionGemma |
| `claymore-model-clover` | Frontière d'intégration Clover |
| `claymore-cuda` | Backend optionnel CUDA Driver API |
| `claymore-cli` | Commandes, profil et diagnostics |

Le workspace utilise Cargo, resolver 2, Rust edition 2021. La version de référence actuelle est 0.4.0.

`claymore-scheduler` ne fournit actuellement qu'une structure `SchedulerConfig` (`cpu_workers`, `max_prefetches`, `lookahead`) et sa validation ; aucune boucle ni logique de planification n'est implémentée. La validation accepte de 1 à 12 workers. Les valeurs minimales de 6 workers, les nombres de streams et le tick de 500 µs figurant dans le profil JSON sont des paramètres documentaires, pas des contraintes appliquées par le code.

### 6.3 Principes de conception

- Les politiques métier et mémoire doivent être testables sans matériel CUDA.
- Les appels CUDA doivent être encapsulés dans le crate backend et exposer des contrats de durée de vie explicites.
- Les erreurs de backend ne doivent pas être transformées en succès comptables.
- L'état logique du gestionnaire mémoire ne doit refléter une libération qu'après confirmation du backend.
- Les buffers utilisés par un transfert asynchrone doivent rester vivants et non réutilisés jusqu'à la fin effective du travail GPU.
- Les événements CUDA, et non un simple compteur logique, doivent devenir la source d'autorité pour la sûreté des opérations asynchrones.
- Le scheduler ne doit pas déclencher de préchargements qui dépassent les budgets ou qui évinceraient des allocations encore utilisées.
- Les choix d'optimisation doivent être mesurés avant d'être complexifiés.

---

## 7. État réel de l'implémentation

Cette section prévaut sur toute formulation plus ambitieuse de la vision cible.

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

#### v0.5 — Adaptateur Memory ↔ CUDA (PR #4, branche non fusionnée)

La branche de la PR #4 ajoute `CudaMemoryManager` dans `claymore-memory`, derrière la feature optionnelle `cuda-driver` :

- allocations et libérations réelles de buffers VRAM et pinned host ;
- exécution des plans d'admission et confirmation comptable après libération backend ;
- transferts H2D/D2H associés à des CUDA events ;
- protection des allocations jusqu'à la synchronisation des events ;
- contrôle de la mémoire VRAM libre et accès borné aux buffers pinned ;
- ticks logiques monotones imposés par le manager.

**État :** code présent sur la branche de la PR #4, pas encore livré sur `main`. CI de la feature à valider ; aucune validation sur WSL2 / RTX 5090. L'adaptateur ne dispose pas encore d'un backend simulé pour des tests unitaires sans GPU.

### 7.2 Non livré à ce stade

Les éléments suivants ne doivent pas être considérés comme fonctionnels de bout en bout :

- chargement du profil matériel JSON par le runtime ou la CLI ;
- application effective du budget souple VRAM de 26 Gio ;
- enforcement de la cible/max du cache d'experts et de son inclusion dans un working set explicite ;
- logique de scheduling, files de tâches et préchargement ;
- polling non bloquant des CUDA events et finalisation automatique des transferts terminés avant admission ;
- tests unitaires de l'adaptateur Memory ↔ CUDA avec backend simulé ;
- pool allocator VRAM, réutilisation de blocs et défragmentation ;
- moteur de transfert piloté par le scheduler ;
- prefetch prédictif intégré au cache d'experts ;
- télémétrie consolidée CPU/RAM/VRAM/transferts ;
- intégration effective de Candle dans le chemin d'inférence ;
- chargement et exécution réels de DiffusionGemma ;
- chargement et exécution réels de Clover-35B-A3B-preview ;
- kernels CUDA spécialisés et kernel dispatcher opérationnel ;
- validation de performance et de qualité sur la RTX 5090 cible.

Les crates modèles constituent à ce stade des frontières de workspace, pas la preuve que les modèles sont exécutables.

### 7.3 Limites connues du backend CUDA actuel

- pas de détection de compute capability ;
- sélection du device par ordinal uniquement ;
- pas de pool allocator, de défragmentation, de télémétrie ou d'éviction automatique ;
- les appels de copie asynchrone imposent à l'appelant de conserver les buffers vivants et de synchroniser avant leur destruction ou réutilisation ;
- les CUDA events sont connectés aux transferts du `CudaMemoryManager` sur la branche PR #4 ; leur complétion est traitée par attente explicite, sans polling non bloquant ni finalisation automatique lors d'une admission ;
- aucune validation runtime sur WSL2 / RTX 5090 n'est encore attestée par ce dépôt.

La CI standard valide le build par défaut sans exiger un GPU. Le backend CUDA n'est pas encore relié à Candle. `CudaDriver::new` crée actuellement un contexte CUDA propre (`cuCtxCreate_v2`) ; l'interopérabilité avec le contexte primaire potentiellement utilisé par Candle/cudarc n'est pas établie et doit être vérifiée avant l'intégration M3/M5. Elle ne doit pas être interprétée comme une validation de la feature `cuda-driver` ni comme une validation matérielle sur RTX 5090.

---

## 8. Exigences fonctionnelles

### FR-01 — Configuration du runtime

Le runtime doit charger un profil matériel configurable et valider les budgets avant de lancer une exécution. Une configuration invalide doit produire une erreur explicite.

### FR-02 — Gestion des budgets

Le gestionnaire mémoire doit connaître les budgets par tier, comptabiliser les allocations enregistrées et refuser une admission qui violerait les limites applicables.

### FR-03 — Admission et éviction

Le gestionnaire doit pouvoir proposer un plan d'admission et des candidats à l'éviction selon la politique active. Une proposition n'est pas une libération : la comptabilité ne change qu'après confirmation du backend.

### FR-04 — Sûreté asynchrone

Aucune allocation ne peut être libérée, réutilisée ou évincée tant qu'un transfert ou un kernel susceptible de l'utiliser n'est pas terminé. L'intégration CUDA doit utiliser des events réels pour établir cette disponibilité.

### FR-05 — Gestion des experts

Le cache d'experts doit respecter sa cible et son maximum, tenir compte du coût de chargement et collaborer avec le gestionnaire mémoire pour éviter les dépassements.

### FR-06 — Scheduling et prefetch

Le scheduler doit prioriser le travail prêt à être exécuté, limiter le nombre de préchargements simultanés et éviter les préchargements qui compromettent les ressources requises par le travail courant.

### FR-07 — Diagnostics

La CLI doit exposer les informations utiles au diagnostic du profil, du backend et des capacités matérielles disponibles. Une fonctionnalité CUDA doit signaler clairement l'absence du driver ou d'un device utilisable.

### FR-08 — Exécution des modèles

Chaque moteur de modèle doit pouvoir préparer, exécuter et libérer ses états en respectant les contrats du runtime, de Candle et du backend. L'exécution ne sera considérée livrée qu'après tests fonctionnels sur les modèles ciblés.

---

## 8 bis. Couverture actuelle des exigences

| Exigence | Statut sur `main` / PR #4 | Justification |
|---|---|---|
| FR-01 — Configuration du runtime | **Non livré** | Le profil JSON existe, mais aucun chargeur ne l'applique ; la CLI utilise `SchedulerConfig::default()`. |
| FR-02 — Gestion des budgets | **Partiel** | Comptabilité host-side et validation des budgets ; le budget souple VRAM n'est pas appliqué à l'admission. |
| FR-03 — Admission et éviction | **Partiel (PR #4)** | Adaptateur d'allocation/libération CUDA et confirmation après free sur la branche non fusionnée ; validation matérielle et tests simulés manquants. |
| FR-04 — Sûreté asynchrone | **Partiel (PR #4)** | Events réels reliés aux transferts et allocations protégées jusqu'à synchronisation explicite ; pas de polling non bloquant, de backend simulé ni de validation GPU. |
| FR-05 — Gestion des experts | **Non livré** | Getters/compteur seulement ; cible et maximum non appliqués à l'admission. |
| FR-06 — Scheduling et prefetch | **Non livré** | Configuration seulement ; aucune logique de scheduling/prefetch. |
| FR-07 — Diagnostics | **Partiel** | CLI de base et diagnostic device optionnel ; le profil JSON n'est pas chargé. |
| FR-08 — Exécution des modèles | **Non livré** | Pas d'intégration Candle ni d'inférence réelle des deux modèles. |

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

Les jalons ci-dessous définissent l'ordre logique de réalisation. Les numéros de version sont des repères de projet ; ils ne signifient pas que tous les composants d'un jalon sont déjà implémentés.

| Phase | Nom | Statut |
|---|---|---|
| M0 | Fondation et scaffold | Workspace, crates, fichiers de configuration et CLI minimale livrés ; le profil matériel n'est pas chargé par le runtime |
| M1 | Memory Orchestrator v0.3 | Livré sur `main` |
| M2 | CUDA Driver Backend v0.4 | Livré sur `main`, validation matérielle à faire |
| M3 | Intégration Memory ↔ CUDA v0.5 | Partiel — code sur PR #4, non fusionnée ; CI feature à valider, pas de test RTX 5090 |
| M4 | Scheduler, transferts et expert cache | Planifié |
| M5 | Intégration Candle et moteurs modèles | Planifié |
| M6 | Optimisation, kernels et validation cible | Planifié |

### M0 — Fondation et scaffold

**Objectif :** disposer d'un workspace Rust propre, de crates séparés, d'une configuration RTX 5090 et d'une première documentation d'architecture.

Livrables :
- workspace Cargo ;
- séparation des responsabilités par crate ;
- profil de configuration matériel ;
- CLI de base et commandes de diagnostic initiales ;
- documentation des budgets et de WSL2.

Critères d'acceptation :
- workspace cohérent ;
- configuration lisible et versionnable ;
- compilation et tests de base en CI.

### M1 — Memory Orchestrator v0.3

**Objectif :** établir les politiques de gestion mémoire indépendamment de CUDA.

Livré :
- validation des budgets ;
- comptabilité par tier ;
- admission et candidats à l'éviction ;
- LRU, LFU et adaptive ;
- protection logique par horizon de tick ;
- confirmation explicite d'éviction.

Critères d'acceptation :
- tests déterministes des budgets et des transitions comptables ;
- aucune éviction comptabilisée avant confirmation ;
- protection des allocations déclarées en cours d'utilisation.

### M2 — CUDA Driver Backend v0.4

**Objectif :** fournir une couche d'accès CUDA isolée et optionnelle.

Livré :
- initialisation driver/device ;
- requêtes mémoire ;
- buffers device et pinned host ;
- streams et events ;
- copies H2D/D2H asynchrones ;
- feature optionnelle et build par défaut indépendant du matériel.

À valider :
- compilation avec `--features cuda-driver` dans WSL2 avec la bibliothèque driver ;
- exécution des diagnostics sur RTX 5090 ;
- tests de création/destruction et transferts sur le device cible.

### M3 — Intégration Memory ↔ CUDA v0.5

Spécification détaillée : [memory-cuda-integration-v0.5.md](memory-cuda-integration-v0.5.md). L'adaptateur `CudaMemoryManager` est implémenté sur la branche de la PR #4 (non fusionnée). Il est opt-in via `cuda-driver`, et n'a pas été validé sur RTX 5090.

**Objectif :** faire du backend CUDA l'exécutant réel des décisions du memory orchestrator.

Travaux :
- définir un adaptateur entre les IDs d'allocations du gestionnaire et les buffers CUDA ;
- interroger la mémoire libre réelle avant admission ;
- exécuter les plans d'allocation et d'éviction ;
- conserver l'ownership des buffers et de l'event pendant les transferts asynchrones ;
- ne rendre une allocation évictable qu'après synchronisation effective de tous les events qui la concernent ;
- appeler la confirmation d'éviction uniquement après libération effective ;
- valider l'appartenance des streams, events et buffers au même contexte CUDA ;
- gérer explicitement les erreurs, les allocations nulles et les tailles invalides ;
- éviter toute destruction d'un buffer encore utilisé par le GPU.

Critères d'acceptation :
- tests unitaires de l'adaptateur avec backend simulé (non implémentés à ce stade) ;
- tests d'intégration CUDA sur cible (non exécutés à ce stade) ;
- démonstration qu'une allocation en vol est protégée jusqu'à la complétion de son event ;
- comptabilité identique à l'état réel après succès ou échec des opérations ;
- aucun chemin d'erreur ne confirme une libération non réalisée.

### M4 — Scheduler, transfert et expert cache

**Objectif :** coordonner calcul, résidence mémoire et transferts.

Travaux :
- formaliser les états de tâche et les dépendances ;
- connecter le scheduler au moteur de transfert ;
- gérer les files compute et transfer ;
- appliquer les limites de préchargement et le lookahead ;
- construire une politique de résidence d'experts tenant compte de la fréquence, du coût de chargement et de la mémoire ;
- intégrer LRU/LFU/adaptive à des métriques observées ;
- ajouter télémétrie et traces de scheduling.

Critères d'acceptation :
- aucun préchargement ne dépasse les budgets ;
- le scheduler ne bloque pas inutilement le compute en attendant un transfert non requis ;
- les buffers ne sont pas évincés avant la fin des tâches dépendantes ;
- mesures séparées du temps de transfert, du temps d'attente et du temps de calcul.

### M5 — Candle et moteurs modèles

**Objectif :** transformer les frontières modèles en chemins d'inférence fonctionnels.

Travaux :
- intégrer les versions et features Candle retenues ;
- définir le chargement des poids et leur représentation ;
- connecter les tenseurs Candle aux contrats mémoire et CUDA ;
- implémenter un premier chemin fonctionnel DiffusionGemma ;
- implémenter un premier chemin fonctionnel Clover ;
- intégrer les états spécifiques, attention, GDN, block diffusion et routage MoE selon les besoins réels ;
- valider les sorties et la stabilité numérique.

Critères d'acceptation :
- chargement et inférence reproductibles sur au moins un cas de test documenté par modèle ;
- comparaison de qualité et de précision avec une référence ;
- mesure du pic VRAM, RAM, latence et débit ;
- tests de libération propre des états après inférence.

### M6 — Optimisation et validation RTX 5090

**Objectif :** optimiser les chemins réellement identifiés comme limitants.

Travaux :
- profiling détaillé ;
- optimisation du prefetch et de la résidence des experts ;
- kernels spécialisés et kernel dispatcher si un gain mesuré le justifie ;
- exploration INT8/INT4 ;
- tuning des streams et workers ;
- tests de charge et scénarios de pression mémoire ;
- documentation des performances et limites.

Critères d'acceptation :
- benchmarks reproductibles et comparables ;
- aucune régression de qualité non acceptée ;
- respect des plafonds mémoire et des invariants de sûreté ;
- gains mesurés par rapport à une baseline documentée.

---

## 11. Stratégie de validation et de développement

### 11.1 Contrôles CI standards

Pour chaque contribution Rust :

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Le workflow peut appliquer rustfmt automatiquement sur les branches selon sa configuration. Les tests CI standards ne remplacent pas les tests CUDA sur la machine cible.

### 11.2 Contrôles CUDA dans WSL2

Après installation et exposition du driver NVIDIA dans WSL2 :

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
nvcc --version
```

Puis :

```bash
cargo check -p claymore-cuda --features cuda-driver
cargo run -p claymore-cli --features cuda-driver -- device
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
- version Rust, Candle, driver et environnement ;
- qualité ou métrique de comparaison pertinente.

---

## 12. Gestion des versions, branches et livraisons

Le développement suit des jalons versionnés et des branches dédiées. Les changements structurants doivent être proposés par pull request, avec une description du périmètre, des tests réalisés et des limites connues.

Historique de référence :
- **v0.2** : scaffold initial, politiques et configuration de départ ;
- **v0.3** : memory orchestrator host-side ;
- **v0.4** : backend CUDA Driver API optionnel.

Les PR #1 (v0.3), #2 (v0.4) et #3 (documentation de référence) ont été fusionnées dans `main`. La PR #4 propose v0.5.0 et l'adaptateur Memory ↔ CUDA ; elle reste ouverte et non fusionnée à la date de cette révision.

Une PR ne doit pas être déclarée prête uniquement parce que le code existe : elle doit préciser si les tests couvrent le build par défaut, la feature CUDA, le matériel réel ou seulement un backend simulé.

---

## 13. Risques et décisions ouvertes

| Sujet | Risque / question | Décision ou traitement |
|---|---|---|
| Contexte CUDA / Candle | Le backend crée un contexte via `cuCtxCreate_v2`, alors que Candle/cudarc peut utiliser le contexte primaire ; les pointeurs ne sont pas interchangeables entre contextes | **Décision ouverte avant M3/M5** : vérifier le modèle de contexte Candle/cudarc retenu et choisir un contexte primaire partagé ou une stratégie d'interopérabilité explicitement supportée. Ne pas transmettre de pointeurs entre contextes sans preuve. |
| Driver CUDA dans WSL2 | Bibliothèque ou lien linker absent | Vérifier `libcuda.so.1` et le linkage sur la cible |
| Durée de vie async | Libération ou réutilisation prématurée | Ownership explicite et event de complétion |
| Comptabilité mémoire | Écart entre état logique et mémoire réelle | Confirmer uniquement après succès backend |
| Pression VRAM externe | Mémoire disponible variable | Réserve et interrogation runtime |
| Mémoire épinglée | Pression sur l'OS | Plafond strict et instrumentation |
| API CUDA | Contexte incohérent entre buffers/streams/events | Valider les appartenances au contexte |
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
5. les tests matériels requis ont été exécutés ou sont explicitement signalés comme restant à faire ;
6. les métriques ou preuves nécessaires sont disponibles ;
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
| 2026-11-18 | Référentiels RAM séparés : machine 128 Gio, VM WSL2 96 Gio, budget Claymore 88 Gio ; marge WSL2 ~8 Gio à mesurer. |
| 2026-11-18 | Le contexte CUDA du backend et celui utilisé par Candle/cudarc constituent un risque ouvert à résoudre avant M3/M5. |
| 2026-11-18 | Cette référence est rédigée en français ; README et documentation technique existante restent en anglais. |
| 2026-11-18 | DiffusionGemma et Clover disposent de moteurs distincts partageant les services transverses du runtime. |

---

## 17. Références externes

Les références exactes (URL et révision/commit) doivent être consignées avant de figer les choix d'implémentation. Elles ne sont pas reproduites ici tant que leur URL canonique et leur révision n'ont pas été vérifiées :

- FreeToken — URL/révision à compléter après vérification.
- Strata (Niko1221) — URL/révision à compléter après vérification.
- DiffusionGemma — dépôt Hugging Face et révision à compléter après vérification.
- Clover-35B-A3B-preview — dépôt Hugging Face et révision à compléter après vérification.


**Fin du document de référence Claymore.**
