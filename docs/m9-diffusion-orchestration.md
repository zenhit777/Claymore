# M9 — Orchestration de diffusion modulaire commune

## Objectif

M9 introduit un cycle d'orchestration commun pour les générations de diffusion par blocs. Le runtime pilote le séquencement et la durée de vie des étapes ; les adaptateurs de modèles exécutent le calcul. Ce jalon ne charge pas de poids, ne construit pas de canvas Gemma et n'implémente pas les opérations propres à Clover.

## API livrée

Le module `claymore_runtime::diffusion` expose :

- `DiffusionPlan` : plan immuable et validé, ordonné par blocs.
- `DiffusionStep` : `PrepareBlock`, `Denoise`, `ValidateBlock`, `CommitBlock`.
- `DiffusionExecutor` : frontière d'exécution implémentée par l'adaptateur de modèle.
- `DiffusionOrchestrator` : admission, dépendances, progression, annulation et propagation des erreurs.
- `DiffusionSessionSnapshot` et `DiffusionMetrics` : état observable sans exposer de tenseurs ni de buffers.

## Invariants

1. Chaque bloc suit strictement préparation → une ou plusieurs itérations de débruitage → validation → commit.
2. Les blocs d'un plan sont traités dans l'ordre croissant de leur identifiant.
3. Les étapes d'une session forment une chaîne de dépendances dans une instance dédiée de `Scheduler`. `ScheduledRuntime` expose une façade pour soumettre, exécuter et annuler ces sessions avec la même configuration validée que le scheduler des requêtes/transferts. À priorité égale, les sessions sont FIFO par admission : une chaîne déjà soumise passe avant la chaîne soumise ensuite ; il n'y a pas d'entrelacement ni de garantie d'équité dans M9.
4. Le scheduler dédié ne transporte que des identifiants et des dépendances. Les tenseurs, buffers, événements CUDA et poids restent possédés par les couches d'exécution et de mémoire.
5. L'annulation est coopérative et observée aux frontières d'étapes. Une étape en cours n'est pas déclarée annulée avant le retour de l'exécuteur.
6. Une erreur d'étape échoue la session et le scheduler annule les étapes dépendantes restantes.
7. Le module ne suppose ni moteur autoregressif token-par-token, ni graphe identique entre Gemma et Clover.

## Exécution

Créer un `DiffusionPlan`, soit construire `DiffusionOrchestrator` avec la configuration du scheduler, soit utiliser `ScheduledRuntime::submit_diffusion` / `run_diffusion_one` / `cancel_diffusion` comme façade. L'adaptateur implémente `DiffusionExecutor::execute_step`. Chaque appel à `run_one` exécute au plus une étape prête ; l'appelant peut ainsi rendre la main entre étapes, inspecter `session`/ `metrics`, ou demander une annulation via `cancel`/ `cancel_diffusion`/ `CancellationToken`.

L'orchestrateur est synchrone et reste confiné au thread propriétaire du scheduler. Il n'enqueue pas de kernels et ne prétend pas que le retour d'une mise en file CUDA constitue une fin de calcul : l'adaptateur doit retourner uniquement lorsque la frontière d'étape est sûre.

## Décision d'architecture — scheduler dédié\n\nM9 utilise provisoirement une seconde instance de `Scheduler` configurée à l'identique. La file Compute du `ScheduledRuntime` existant est réservée aux requêtes opaques (`run_one`) et ne peut pas être partagée sans refonte de son contrat d'admission et de dispatch. La séparation évite qu'une étape de diffusion soit interprétée comme une requête sans payload.\n\nConséquences explicites : pas d'ordonnancement global entre requêtes et diffusion, pas de fenêtre de prefetch partagée, et aucune dépendance `Transfer` ne peut être exprimée dans le graphe de diffusion. La convergence vers un graphe unifié, avec des types de tâches et des payloads explicitement séparés, est suivie comme évolution post-M9.\n\nÀ priorité égale, les sessions de diffusion sont FIFO : toutes les étapes de la première session admise précèdent celles de la suivante. Une politique round-robin/équité est hors périmètre de cette version.\n\n## Validation

Les tests unitaires du module couvrent la validation du plan (cas positifs et négatifs), l'ordre déterministe multi-blocs, le FIFO inter-sessions, l'isolation des échecs, la propagation d'échec vers les dépendants, l'annulation avant admission/en file et la comptabilisation d'une étape réussie lorsque l'annulation survient pendant son exécution. Les tests CPU sont indépendants de CUDA. La validation matérielle RTX 5090 n'est pas un gate M9, car cette phase ajoute uniquement le contrôle de flux et des contrats CPU.
