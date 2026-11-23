# Architecture Claymore

## Périmètre produit

Claymore cible initialement deux modèles de diffusion : DiffusionGemma et Clover. Il ne développe pas de stratégie autoregressive générique sans décision produit ultérieure.

## Architecture d'exécution

1. **Runtime commun** : cycle de vie des requêtes, configuration, diagnostics et coordination des services.
2. **Socle tensoriel natif unique** : types Tensor/Layout, opérations, kernels et dispatch backend appartenant à Claymore.
3. **Services transverses** : memory orchestrator, budgets, expert cache, scheduler, transfer engine, télémétrie et profiling.
4. **Orchestration de diffusion modulaire commune** : cycle de génération et interfaces d'extension partagées lorsque les invariants sont communs. Les stratégies sont composables et ne forcent pas un graphe, un état ou une boucle identique à tous les modèles.
5. **Adaptateurs spécialisés** :
   - DiffusionGemma : canvas/état, progression de denoising, attention bidirectionnelle et règles numériques propres au modèle.
   - Clover : block diffusion/GDN, routage MoE, résidence/préchargement des experts et coordination spécifique.

Les adaptateurs ne constituent pas des moteurs tensoriels ou runtimes autonomes : ils s'appuient sur le socle et les services communs. Ils préservent les différences irréductibles de graphe, de cycle de vie et de calcul.

## Frontières de crates

- claymore-core : types et configuration transverses.
- claymore-device : devices, capacités, buffers opaques et opérations de transfert.
- claymore-tensor : modèle Tensor canonique, layouts et vues.
- claymore-memory : admission, budgets, résidence et éviction.
- claymore-scheduler : dépendances et ordonnancement des tâches.
- claymore-runtime : cycle de vie et orchestration des requêtes.
- claymore-model-gemma / claymore-model-clover : adaptateurs spécialisés.
- claymore-cuda : backend CUDA optionnel, sans propriété des sémantiques Tensor ; il charge et lance les kernels F32 M8 via la Driver API dans le contexte primaire.
- Le dispatch tensoriel est déterministe et backend-neutre : chaque refus est typé et expliqué ; aucun fallback CPU, cast ou transfert Host implicite n'est autorisé. Les sorties CUDA passent par le memory orchestrator et des events réels, le scheduler ne possédant que des identifiants de tâches.

## Invariants de sûreté

- Les budgets mémoire configurés sont des limites d'admission, pas une garantie de mémoire libre.
- Conserver une réserve de sécurité ; ne jamais planifier la consommation de toute la VRAM.
- Ne libérer/évincer un buffer qu'après complétion confirmée de tout travail device qui le référence.
- Les transferts asynchrones sont prouvés terminés par leurs events/tokens, jamais par le seul retour d'enqueue.
- Le swap n'est pas un tier de poids de modèle.
- Aucun cast, fallback CPU ou aller-retour host implicite dans le dispatch.
- Une stratégie de modèle ne contourne pas les contrats communs de mémoire, scheduler et durée de vie.

## Jalons

- M8 : socle tensoriel natif, opérations de référence, dispatch et backends.
- M9 : **livré sur `main`** par la PR #32 — orchestration de diffusion modulaire commune (plans par blocs, dépendances scheduler, annulation coopérative et propagation d'échec ; voir [m9-diffusion-orchestration.md](m9-diffusion-orchestration.md)).
- M10 : **en cours** — intégration DiffusionGemma (voir [m10-diffusiongemma.md](m10-diffusiongemma.md)).
- M11 : intégration Clover.
- M12 : optimisation et validation de performance.

Le détail des contrats et gates de M8 est dans [m8-architecture.md](m8-architecture.md).
