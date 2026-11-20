# M5 — Validation des streams CUDA, complétion et évictions

**Base :** `b576a112725a245c657113786026fedf70ffa283` (merge M4 / PR #16).  
**Issue couverte :** #13.  
**Branche :** `feat/m5-cuda-streams-eviction-hardening`.

## 1. Objectifs

M5 renforce le cycle de vie CUDA sans implémenter le trait générique `Device` pour CUDA et sans ajouter de moteur tensoriel externe.

Le jalon traite :
- le polling non bloquant des événements de transfert ;
- la sémantique des streams ;
- les erreurs d'événement et de synchronisation en mode fail-closed ;
- la finalisation des allocations dont les transferts sont terminés avant un plan d'éviction ;
- le déterminisme des évictions à priorité égale ;
- la validation CI de la feature CUDA sans matériel.

## 2. Constats et décisions

### Streams

Le backend créait le stream de transfert avec `CU_STREAM_DEFAULT` (flags nuls). Ce stream pouvait participer aux synchronisations implicites du legacy default stream.

Le stream dédié est désormais créé avec `CU_STREAM_NON_BLOCKING`. Les transferts restent asynchrones ; les points de synchronisation demeurent explicites :
- `wait_for_transfer` / `wait_for_allocation` lorsqu'un consommateur exige les données ;
- polling des events avant un plan d'admission/éviction ;
- synchronisation du stream dans le chemin de destruction du gestionnaire pour établir la sûreté des buffers.

### Polling et finalisation

`CudaMemoryManager::poll_completed_transfers(tick)` interroge chaque event sans bloquer. Seul un résultat positif de `cuEventQuery` autorise le retrait de l'event et la levée des protections d'allocations qui ne sont plus couvertes par un autre transfert en vol.

`CUDA_ERROR_NOT_READY` conserve l'event et ses allocations protégées. Toute autre erreur est propagée : elle n'est jamais convertie en complétion.

Le chemin d'admission appelle ce polling avant de calculer les victimes d'éviction. Les opérations de lecture/écriture et de libération conservent leurs attentes explicites.

### Déterminisme des évictions

Les clés LRU/LFU/Adaptive ne distinguaient pas deux allocations de priorité identique. Comme les candidats proviennent d'une `HashMap`, l'ordre d'itération pouvait varier.

Le comparateur ajoute l'identifiant d'allocation comme dernier critère lexicographique. Un test répète le même plan 32 fois avec des candidats de priorité identique et vérifie que le même identifiant est choisi.

### Fail-closed

- Une erreur de `cuEventQuery` quitte le polling avant de retirer l'event concerné ; l'event et ses allocations restent suivis.
- Une erreur de `cuEventSynchronize` laisse les protections suivies par `wait_for_allocation`.
- Une erreur de synchronisation du stream pendant `Drop` provoque toujours la fuite volontaire des buffers/events suivis, plutôt qu'une libération incertaine.
- Un échec de synchronisation du stream après échec de `cuEventRecord` empoisonne le gestionnaire et marque les deux allocations comme en vol.

Les tests sans GPU vérifient la traduction des résultats de query et la propagation des erreurs de synchronisation au niveau du wrapper. Le scénario matériel reste séparé.

## 3. Changements

- `CU_STREAM_NON_BLOCKING` utilisé à la création du stream dédié.
- Ajout de `poll_completed_transfers` avec retrait uniquement après complétion confirmée.
- Polling automatique avant admission/éviction.
- Départage déterministe par identifiant en cas d'égalité de priorité.
- Test de non-régression pour les évictions à priorité égale.
- Test de propagation d'erreur de synchronisation de stream.
- Extension du test matériel ignoré par défaut pour exercer le polling et les chemins d'attente H2D/D2H.

## 4. Validation

La CI existante couvre déjà :
- `cargo fmt --all -- --check` ;
- `cargo test --workspace --all-targets` ;
- `cargo check -p claymore-memory --features cuda-driver` ;
- `cargo check -p claymore-cli --features cuda-driver` ;
- Clippy standard et avec `cuda-driver` ;
- tests workspace sur Rust 1.86.

Aucune CI standard ne doit exiger un GPU ni linker le pilote CUDA. Les tests matériels sont explicitement ignorés sans pilote et périphérique NVIDIA.

## 5. Limites

- La validation réelle de `cuEventQuery`, `cuEventSynchronize`, `cuStreamSynchronize` et des transferts DMA sur RTX 5090 doit être exécutée sur la machine cible.
- Les tests d'injection sans GPU valident les conversions d'erreurs et les invariants de décision ; ils ne remplacent pas une panne injectée dans un pilote NVIDIA réel.
- Aucun support multi-GPU/D2D ni implémentation du trait `Device` CUDA n'est ajouté.
- Aucune dépendance Candle ou autre framework tensoriel externe n'est introduite.

## 6. Critères de clôture

- [x] Stream dédié explicitement non bloquant.
- [x] Polling non bloquant des events et finalisation avant admission.
- [x] Les états pending et les erreurs ne sont pas assimilés à une complétion.
- [x] Les allocations restent protégées tant qu'un event associé reste en vol.
- [x] Départage déterministe des évictions de priorité égale.
- [x] Tests unitaires indépendants du GPU pour les résultats d'event et les erreurs de stream.
- [x] CI verte sur le HEAD `1fe6ff5e8cc53f4fd5fd6d4d508e9321a56d47a2` : format, tests workspace, feature CUDA, Clippy standard/CUDA et MSRV Rust 1.86 (workflow [44817813430](https://github.com/zenhit777/Claymore/actions/runs/44817813430)).
- [ ] Revue architecturale et fusion.
- [ ] Validation matérielle RTX 5090.
