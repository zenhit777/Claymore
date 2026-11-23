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
3. Les étapes d'une session forment une chaîne de dépendances dans le scheduler partagé. Des sessions distinctes peuvent s'entrelacer selon la priorité et l'ordre FIFO du scheduler.
4. Le scheduler ne transporte que des identifiants et des dépendances. Les tenseurs, buffers, événements CUDA et poids restent possédés par les couches d'exécution et de mémoire.
5. L'annulation est coopérative et observée aux frontières d'étapes. Une étape en cours n'est pas déclarée annulée avant le retour de l'exécuteur.
6. Une erreur d'étape échoue la session et le scheduler annule les étapes dépendantes restantes.
7. Le module ne suppose ni moteur autoregressif token-par-token, ni graphe identique entre Gemma et Clover.

## Exécution

Créer un `DiffusionPlan`, construire `DiffusionOrchestrator` avec la configuration du scheduler, puis appeler `submit`. L'adaptateur implémente `DiffusionExecutor::execute_step`. Chaque appel à `run_one` exécute au plus une étape prête ; l'appelant peut ainsi rendre la main entre étapes, inspecter `session`/ `metrics`, ou demander une annulation via `cancel`/ `CancellationToken`.

L'orchestrateur est synchrone et reste confiné au thread propriétaire du scheduler. Il n'enqueue pas de kernels et ne prétend pas que le retour d'une mise en file CUDA constitue une fin de calcul : l'adaptateur doit retourner uniquement lorsque la frontière d'étape est sûre.

## Validation

Les tests unitaires du module couvrent la validation du plan, l'ordre déterministe multi-blocs, la propagation d'échec et l'annulation à une frontière sûre. Les tests CPU sont indépendants de CUDA. La validation matérielle RTX 5090 n'est pas un gate M9, car cette phase ajoute uniquement le contrôle de flux et des contrats CPU.
