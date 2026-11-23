# Agent IA — M10 DiffusionGemma sur RTX 5090

## Mission

Poursuivre M10 dans le dépôt Claymore après le socle CPU préparatoire. Travailler sur la branche `feat/m10-diffusiongemma` (ou une branche de travail issue de celle-ci) et livrer une intégration réelle, reproductible et auditée du checkpoint exact `google/diffusiongemma-26B-A4B-it` sur la RTX 5090 32 Gio sous WSL2.

Ne pas déclarer le modèle compatible sur la seule base d'une compilation, d'un chargement de config ou de tests synthétiques. Ne pas remplacer DiffusionGemma par un modèle autoregressif. Ne pas créer de second runtime ou moteur tensoriel. Les contrats Tensor, Device, mémoire, scheduler et orchestration M9 de Claymore sont propriétaires et doivent rester les seuls chemins d'exécution.

## Pré-vol obligatoire

1. Lire `docs/Claymore.md`, `docs/architecture.md`, `docs/m8-architecture.md`, `docs/m9-diffusion-orchestration.md`, `docs/audits/m8-model-graph-requirements.md` et ce document M10.
2. Vérifier `git status`, branche et SHA. Ne pas écraser de modifications utilisateur.
3. Capturer `nvidia-smi`, modèle/VRAM/compute capability, driver, CUDA toolkit, `rustc -Vv`, `cargo -V`, `nvcc --version`, `free -h`.
4. Épingler les révisions exactes du dépôt Hugging Face, `config.json`, `model.safetensors.index.json` et code d'implémentation de référence. Enregistrer URLs, commit/revision et SHA-256 dans le rapport. Ne jamais suivre silencieusement `main` mobile.

## Architecture exigée

- `claymore-model-gemma` est l'adaptateur ; `claymore-runtime` conserve l'orchestration et le scheduler ; `claymore-tensor` reste propriétaire des opérations et layouts ; `claymore-memory` reste propriétaire de l'admission/résidence/libération.
- L'encodeur causal traite le prompt propre et les blocs déjà validés et construit le KV cache requis.
- Le décodeur applique attention bidirectionnelle sur le canvas et cross-attention vers le contexte encodeur.
- Canvas initial de 256 positions (sauf si la révision épinglée documente explicitement une valeur différente).
- Chaque itération produit des candidats et scores de confiance ; seules les positions atteignant le seuil sont validées. Les autres suivent exactement la politique de re-noising/scheduling de la référence épinglée.
- Les dépendances entre étapes sont exprimées par M9 ; aucune ressource tensorielle ou poids ne doit être placée dans les payloads scheduler.
- BF16 explicite pour les poids ; accumulation FP32 selon les contrats numériques. FP16 non supporté. Aucun cast, fallback CPU, transfert Host ou allocation non budgétée implicite.

## Travail à effectuer

### A. Chargement et audit checkpoint

- Implémenter un lecteur Safetensors/ShardIndex borné et sûr : validation de toutes les entrées, shapes, offsets, dtype BF16, tailles et correspondance avec les paramètres de configuration.
- Charger shard par shard via les contrats Device/Memory ; refuser les offsets chevauchants, noms inattendus, dtypes interdits, dépassements et fichiers incomplets.
- Ajouter tests CPU de corruption, truncation, overflow, index mal formé, shard manquant et poids inattendus.
- Définir clairement les poids résidents, poids offloadés, buffers temporaires et budgets.

### B. Primitives tensorielle manquantes

- Ajouter les opérations minimales réellement nécessaires au graphe : projections/linear, RoPE, masquage, attention multi-tête bidirectionnelle, cross-attention, cache KV (append/read/position), logits et sélection de confiance.
- CPU reference déterministe pour petites dimensions.
- Kernel CUDA F32/BF16 approprié, exact dispatch (dtype + shape/layout/capacités), aucune opération non supportée n'est dispatchée.
- Tester stabilité softmax, masques, séquences longues, dimensions non alignées, cas limites, NaN/Inf et écarts CPU/CUDA.
- Ne pas développer des kernels génériques non nécessaires au graphe vérifié.

### C. Exécuteur et orchestration

- Implémenter l'adaptateur `DiffusionExecutor` en respectant les étapes PrepareBlock, Denoise, ValidateBlock et CommitBlock.
- L'exécution doit être bornée par étape et retourner uniquement après complétion sûre des kernels/transferts (events/tokens réels).
- Implémenter annulation aux frontières sûres, propagation des erreurs typées et cleanup sans libération de buffers en vol.
- Ne pas prétendre que la simple mise en file CUDA signifie que l'étape est terminée.

### D. Campagne RTX 5090

Exécuter au minimum :
1. découverte driver/device et compatibilité PTX/SM 120 ;
2. tests d'allocation/libération et intégration MemoryOrchestrator ;
3. tests des kernels individuels F32/BF16 contre CPU ;
4. tests différentiels du sous-graphe encoder, KV cache, decoder bidirectionnel/cross-attention ;
5. un passage de denoising complet sur un canvas synthétique ;
6. si ressources et licence/accès le permettent, un chargement et une génération avec le checkpoint réel épinglé ;
7. pression mémoire, répétitions, annulation et échec injecté ; vérifier absence de fuite, OOM non maîtrisé et use-after-free.

Ne pas diminuer les tolérances après observation. Fixer les critères avant campagne et consigner écarts max, RMS error, erreurs relatives, cas NaN/Inf et éventuelles différences BF16 en ULP.

## Sécurité mémoire et budgets

- VRAM physique 32 Gio ; soft budget 26 Gio ; hard ceiling 28 Gio ; réserver au moins 4 Gio au système/driver.
- Aucun dépassement de budget par allocation CUDA directe hors MemoryOrchestrator.
- KV cache, activations et canvas doivent avoir une durée de vie explicite et des mesures de pic.
- Si le checkpoint ne tient pas dans les budgets, documenter et implémenter un offload borné validé ; ne jamais compter sur swap système.

## CI et qualité

Exécuter et corriger :
```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --locked
cargo test -p claymore-model-gemma --all-targets --locked
```
Respecter Rust 1.86 MSRV et le lockfile. Les tests CI sans GPU doivent rester indépendants du matériel.

## Livrables obligatoires

- Code compilable et tests CPU/CUDA.
- Rapport `docs/audits/m10-rtx5090-hardware-validation.md` contenant SHA Claymore testé, SHA/revisions checkpoint/code upstream, OS/WSL2, GPU, driver, CUDA, Rust, commandes exactes, sorties, budgets/pics mémoire, tolérances fixées avant test, résultats bruts et limites.
- Logs bruts horodatés dans `docs/audits/`.
- Mise à jour de `docs/m10-diffusiongemma.md`, `docs/Claymore.md` et `docs/architecture.md` sans revendiquer de capacité non testée.
- Résumé des changements, commandes exécutées, tests passés/échoués, risques et SHA de commit.

Créer une PR ciblant `main`. Ne jamais fusionner la PR. Si le checkpoint réel ne peut être chargé (accès, licence, RAM/VRAM ou format), arrêter à un état explicite, fournir les preuves et laisser le gate de génération réelle ouvert plutôt que simuler un succès.
