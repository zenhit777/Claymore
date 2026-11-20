# Audit technique M1 — Fondation matérielle de Claymore

**Dépôt :** https://github.com/zenhit777/Claymore  
**Branche auditée :** main  
**Révision auditée :** 7a0f4fc84cdf50d93a8f2d048bf3283d04ff106f  
**Date :** 19 novembre 2026  
**Objet :** état réel du dépôt, risques bloquants et périmètre de consolidation M1.

---

## 1. Synthèse exécutive

Claymore dispose déjà d'une base Rust organisée en crates, d'une politique mémoire host-side, d'un backend CUDA Driver API optionnel et d'un adaptateur reliant la politique mémoire aux allocations/transferts CUDA.

Le projet n'est pas encore un runtime d'inférence complet : le scheduler ne planifie pas de tâches, les crates Gemma et Clover sont des frontières vides et le profil matériel JSON n'est pas chargé automatiquement. Le moteur tensoriel et les chemins d'exécution natifs restent à construire.

Le risque architectural prioritaire avant de construire les couches Device/Tensor et le moteur d'exécution est le modèle de contexte CUDA. Sur main, le backend crée un contexte privé avec cuCtxCreate_v2. Les composants Claymore doivent partager un contexte cohérent ; les pointeurs CUDA ne sont pas interchangeables entre contextes distincts.

La PR M1 durcit la fondation existante sans introduire prématurément le runtime, le scheduler ou les moteurs de modèles.

## 2. Inventaire de l'état réel sur main

| Élément | Code présent | État réel |
|---|---|---|
| Workspace Cargo | 7 crates | Livré |
| claymore-core | types Precision, MemoryTier, MemoryBudget | Minimal |
| claymore-memory | budgets, comptabilité, admission, LRU/LFU/adaptive | Fonctionnel en politique host-side |
| CudaMemoryManager | allocations device/pinned, H2D/D2H, events, eviction | Implémenté derrière feature ; test matériel présent |
| claymore-cuda | Driver API, buffers, streams, events | Implémenté derrière feature |
| claymore-scheduler | SchedulerConfig et validate | Aucune boucle ni file de tâches |
| claymore-model-gemma | crate et lib minimale | Pas de modèle |
| claymore-model-clover | crate et lib minimale | Pas de modèle |
| claymore-cli | commandes de profil et device | Diagnostics de base |
| Profil RTX 5090 JSON | budgets et paramètres | Versionné, non chargé automatiquement |
| Moteur tensoriel Claymore | primitives de calcul et exécution | Non livré |
| Runtime unifié | aucune crate dédiée | Non livré |
| Tests simulés CUDA/memory adapter | absents | À construire avec les abstractions M2/M3 |

## 3. Risques identifiés

### P0 — Contexte CUDA privé incompatible avec une interopérabilité implicite

Sur main, CudaDriver::new appelle cuCtxCreate_v2. Cette fonction crée un contexte distinct du contexte primaire du device. Un composant Claymore utilisant le contexte primaire ne peut pas utiliser directement les allocations/streams/events de ce contexte privé.

**Traitement M1 :** retenir le contexte primaire avec cuDevicePrimaryCtxRetain, puis équilibrer cette rétention avec cuDevicePrimaryCtxRelease_v2 lorsque le dernier objet possédant le contexte est détruit.

**Limite :** ce choix établit le contrat de contexte CUDA commun aux composants Claymore. Le raccordement du backend CUDA aux abstractions Device/Tensor devra être démontré par des tests d'intégration dédiés.

### P1 — Absence de contrôle d'appartenance au contexte

Les méthodes de copie et CudaEvent::record supposaient que les buffers, streams et events fournis appartenaient au même contexte. L'API publique permet pourtant de créer plusieurs drivers.

**Traitement M1 :** comparer les handles de contexte avant toute opération inter-ressources et retourner CudaError::ContextMismatch plutôt que d'appeler le driver avec des handles incompatibles. Deux wrappers distincts du même contexte primaire restent compatibles car ils exposent le même handle CUDA.

### P1 — Capacité GPU non renseignée

CudaDeviceInfo prévoit compute_capability, mais la valeur restait None.

**Traitement M1 :** renseigner la capability via cuDeviceGetAttribute (COMPUTE_CAPABILITY_MAJOR / MINOR). Cette donnée permettra aux phases suivantes de sélectionner les capacités disponibles sans coder en dur le modèle de GPU.

### P1 — Échec incertain après envoi d'une copie

Dans l'adaptateur mémoire, si l'enregistrement d'un event échoue puis que la synchronisation du stream échoue également, le manager ne dispose plus d'une preuve de fin du transfert. Continuer à traiter les allocations comme évictables serait dangereux.

**Traitement M1 :** empoisonner le manager après cet échec double, marquer les allocations concernées comme en vol et refuser toute opération ultérieure. Le destructeur tente une nouvelle synchronisation ; si la fin du travail ne peut toujours pas être prouvée, les propriétaires CUDA sont volontairement conservés en mémoire plutôt que libérés prématurément.

### P2 — Entrées invalides

Les allocations de taille nulle et les ordinaux non représentables ou hors plage étaient laissés au driver pour produire une erreur générique.

**Traitement M1 :** rejeter ces entrées avant l'appel CUDA avec des erreurs explicites.

### Risque de concurrence — Confinement au thread propriétaire

Le backend utilise Rc, ce qui rend volontairement les ressources non Send/Sync. Il ne faut pas contourner cette propriété avec des implémentations unsafe. M4 devra choisir un thread CUDA propriétaire (actor/executor) ou une autre stratégie démontrée sûre.

## 4. Périmètre livré par la PR M1

- contexte primaire CUDA retenu/libéré avec ownership partagé entre driver et ressources ;
- compute capability renseignée dans CudaDeviceInfo ;
- validation des contextes pour les transferts et l'enregistrement d'events ;
- erreurs explicites pour ordinal invalide, allocation de taille nulle et contextes incompatibles ;
- mode fail-closed du CudaMemoryManager après échec de synchronisation incertain ;
- tests unitaires hardware-independent pour les contrôles de contexte et de taille ;
- mise à jour de docs/Claymore.md et présent audit.

## 5. Éléments volontairement reportés

| Sujet | Phase | Motif |
|---|---|---|
| Raccordement du backend CUDA aux contrats Device/Tensor | M4/M5 | Les contrats et le chemin d'exécution doivent être définis dans Claymore |
| Abstraction Device/Tensor et backend CPU simulé | M2 | Contrats à définir avant le runtime |
| Runtime de requêtes | M3 | Dépend des contrats Device/Tensor |
| Actor/thread CUDA propriétaire | M4 | À concevoir avec le scheduler et le modèle de concurrence |
| Pool allocator, slabs, defragmentation | M8 | Nécessite mesures d'allocation et profils réels |
| Polling non bloquant des events | M4 | Doit s'intégrer au scheduler/transfer engine |
| Télémétrie complète | M4/M8 | Définir les métriques avec les services qui les consomment |
| Kernels spécialisés | M8 | Aucun benchmark ne justifie encore leur coût |
| Chargement réel Gemma/Clover | M5/M7 | Hors fondation matérielle |
| Backend simulé de l'adaptateur mémoire | M2/M3 | Nécessite une frontière backend testable sans CUDA |

## 6. Validation et limites de preuve

### Validations de référence déjà observées avant cette PR

L'utilisateur a exécuté sur WSL2 / RTX 5090 le test matériel cuda_manager_reports_device_memory de claymore-memory. Le test a réussi et comprend allocation pinned, allocation VRAM, H2D, D2H, comparaison des octets et libération.

Le linkage CUDA sous WSL2 a nécessité le chemin de recherche /usr/lib/wsl/lib, fourni temporairement avec RUSTFLAGS="-L native=/usr/lib/wsl/lib".

### Validation attendue de cette PR

CI :
- cargo fmt --all -- --check
- cargo test --workspace --all-targets
- cargo check -p claymore-memory --features cuda-driver
- cargo check -p claymore-cli --features cuda-driver
- Clippy par défaut et avec la feature CUDA.

Matériel :
- relancer le test roundtrip existant après checkout de la branche ;
- vérifier que compute capability est renseignée ;
- valider ensuite le raccordement du backend CUDA aux contrats Device/Tensor de Claymore.

**Aucune compilation locale par l'assistant, aucun résultat CI de cette PR ni aucune validation du moteur tensoriel ne sont revendiqués dans cet audit.** Les résultats seront ceux des checks GitHub et des essais sur la machine cible.

## 7. Questions proposées à l'architecte logiciel

1. Valider le contexte primaire comme contrat de partage entre les composants CUDA de Claymore.
2. Confirmer le maintien du confinement Rc/thread jusqu'à l'introduction de l'executor CUDA propriétaire.
3. Valider le choix fail-closed du manager après une synchronisation incertaine, y compris la stratégie de fuite volontaire en dernier recours.
4. Confirmer que les abstractions Device/Tensor sont le prochain jalon et qu'elles ne doivent pas être ajoutées dans M1.

## 8. Conclusion

M1 doit être considéré comme une consolidation de la fondation CUDA, pas comme une livraison du runtime. Une fois cette PR revue et acceptée, M2 pourra définir les contrats Device/Tensor au-dessus d'une frontière matérielle plus explicite et plus sûre.
