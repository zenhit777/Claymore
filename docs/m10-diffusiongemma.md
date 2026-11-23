# M10 — Adaptateur DiffusionGemma

## Objectif et frontière

Intégrer le checkpoint exact `google/diffusiongemma-26B-A4B-it` au runtime de diffusion M9, sans créer de runtime, scheduler ou moteur tensoriel parallèle. Le modèle audité est un encoder-decoder de block diffusion : l'encodeur causal construit le contexte/KV cache, le décodeur traite un canvas de 256 tokens avec attention bidirectionnelle et cross-attention, puis un raffinement par confiance valide certaines positions et re-bruite les autres.

Référence d'architecture : `docs/audits/m8-model-graph-requirements.md`. La révision exacte du dépôt modèle, de `config.json`, de l'index Safetensors et des fichiers de code upstream doit être épinglée avant tout chargement réel.

## Sous-phases

### M10.1 — Contrats et ingestion (CPU)

- Parsing strictement typé de la configuration, validation de dimensions et de la précision BF16.
- Identité du checkpoint et provenance des révisions.
- Inspection de l'index Safetensors, validation des noms, shapes, dtypes et shards sans charger les poids.
- Contrats d'erreurs typés ; aucune conversion FP16/F32 implicite.
- Canvas 256 positions, états masked/provisional/committed et transitions atomiques de confiance.
- Tests de propriété des invariants, d'entrée malformée et de batch partiellement invalide.

### M10.2 — Primitives du graphe

- Projections linéaires, RoPE/positions, masques, attention multi-tête bidirectionnelle, cross-attention et gestion du KV cache.
- Opérations de sélection/confiance, sampling et re-noising appartenant à l'adaptateur.
- CPU reference déterministe pour les petites shapes ; contrats shape/layout/dtype explicites.
- Aucune primitive n'est déclarée prête sans tests différentiels et tolérances documentées.

### M10.3 — Exécution et mémoire

- Chargement BF16 shard par shard via le chemin mémoire Claymore ; aucun chargement monolithique non borné.
- Durée de vie explicite des poids, KV cache, canvas et activations.
- Exécution des étapes M9 par un exécuteur Gemma ; retour d'une étape uniquement après complétion sûre.
- Propagation des erreurs et annulation aux frontières sûres.

### M10.4 — Validation de cible

- Tests CPU/CI, compilation CUDA optionnelle, puis exécution sur RTX 5090.
- Vérification des budgets : VRAM souple 26 Gio, plafond 28 Gio, réserve 4 Gio ; les poids/offload ne doivent pas contourner MemoryOrchestrator.
- Comparaison numérique à une référence fiable et traçabilité du checkpoint, de la révision du code, du SHA Claymore, driver et versions outils.
- Rapport matériel reproductible sous `docs/audits/`.

## Critères de sortie

- Checkpoint/révisions explicitement épinglés et index validé.
- Chemin encoder → KV cache → decoder bidirectionnel/cross-attention → confidence/refine exercé sur des données de test déterministes.
- Adaptateur branché au DiffusionExecutor M9 sans runtime concurrent.
- Précision BF16 explicite, aucune conversion/fallback silencieux.
- Gestion mémoire bornée et erreurs fail-closed.
- Référence CPU et exécution CUDA différentielles ; campagne RTX 5090 versionnée.
- Guide d'intégration, limites et résultats réels documentés.

## Limites actuelles du dépôt au démarrage M10

Le crate `claymore-model-gemma` ne contient qu'un enum d'état. M8 ne fournit que add, matmul, softmax et RMSNorm ; les primitives RoPE, attention, cross-attention, KV cache, sampling et sélection de confiance restent à construire. Le rapport M8 confirme que les poids sont BF16 et que l'audit de configuration/index n'est pas une exécution du checkpoint.

Le premier incrément M10 ajoute les contrats CPU de configuration/canvas. Il ne prétend pas charger ni exécuter le modèle complet. Le chemin CUDA, les kernels et la validation de génération réelle nécessitent la machine RTX 5090 et sont détaillés dans `docs/agents/m10-rtx5090-agent-prompt.md`.
