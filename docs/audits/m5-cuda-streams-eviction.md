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

Le stream dédié est désormais créé avec `CU_STREAM_NON_BLOCKING`. Il ne participe donc plus aux synchronisations implicites avec le legacy default stream. Tout futur consommateur compute (kernels CUDA, cuBLAS/cuBLASLt) devra exprimer explicitement les dépendances compute ↔ transfert, notamment avec `cuStreamWaitEvent`. Les transferts restent asynchrones ; les points de synchronisation demeurent explicites :
- `wait_for_transfer` / `wait_for_allocation` lorsqu'un consommateur exige les données ;
- polling des events avant un plan d'admission/éviction ;
- synchronisation du stream dans le chemin de destruction du gestionnaire pour établir la sûreté des buffers.

### Polling et finalisation

`CudaMemoryManager::poll_completed_transfers(tick)` interroge chaque event sans bloquer. Seul un résultat positif de `cuEventQuery` autorise le retrait de l'event et la levée des protections d'allocations qui ne sont plus couvertes par un autre transfert en vol.

`CUDA_ERROR_NOT_READY` conserve l'event et ses allocations protégées. Toute autre erreur est propagée : elle n'est jamais convertie en complétion.

Le chemin d'admission appelle ce polling avant de calculer les victimes d'éviction. Toute erreur de polling est propagée par `make_room` et bloque l'admission, même si l'event en erreur ne référence pas la victime envisagée : ce comportement fail-closed est intentionnel. Les opérations de lecture/écriture et de libération conservent leurs attentes explicites.

### Déterminisme des évictions

Les clés LRU/LFU/Adaptive ne distinguaient pas deux allocations de priorité identique. Comme les candidats proviennent d'une `HashMap`, l'ordre d'itération pouvait varier.

Le comparateur ajoute l'identifiant d'allocation comme dernier critère lexicographique. Un test couvre LRU, LFU et Adaptive, répète 32 fois le même plan avec des scores égaux et vérifie l'ordre `alpha`, `beta` lorsque deux victimes sont nécessaires.

### Fail-closed

- Une erreur de `cuEventQuery` quitte le polling avant de retirer l'event concerné ; l'event et ses allocations restent suivis.
- Une erreur de `cuEventSynchronize` laisse les protections suivies par `wait_for_allocation`.
- Une erreur de synchronisation du stream pendant `Drop` provoque toujours la fuite volontaire des buffers/events suivis, plutôt qu'une libération incertaine.
- Un échec de synchronisation du stream après échec de `cuEventRecord` empoisonne le gestionnaire et marque les deux allocations comme en vol.

Les tests unitaires `event_query_status_conversion_distinguishes_complete_pending_and_driver_failure` et `driver_error_conversion_preserves_operation_and_code` vérifient uniquement la conversion de codes de retour en résultats/erreurs. Le second ne teste pas `cuStreamSynchronize` lui-même ; aucun des deux ne teste le comportement manager-level de `poll_completed_transfers`. La CI compile désormais les cibles de tests avec `cargo check ... --all-targets`, mais ne les exécute pas sous `cuda-driver`. Ne pas présenter ces vérifications comme des tests d'injection ou comme une preuve de propagation d'erreur dans le chemin runtime.

## 3. Changements

- `CU_STREAM_NON_BLOCKING` utilisé à la création du stream dédié.
- Ajout de `poll_completed_transfers` avec retrait uniquement après complétion confirmée.
- Polling automatique avant admission/éviction.
- Départage déterministe par identifiant en cas d'égalité de priorité.
- Test de non-régression pour les évictions à priorité égale.
- Tests de conversion des codes d'erreur CUDA au niveau du wrapper (pas d'injection dans le chemin stream).
- Extension du test matériel ignoré par défaut pour exercer le polling et les chemins d'attente H2D/D2H.

## 4. Validation

La CI existante couvre déjà :
- `cargo fmt --all -- --check` ;
- `cargo test --workspace --all-targets` ;
- `cargo check -p claymore-memory --features cuda-driver` ;
- `cargo check -p claymore-cli --features cuda-driver` ;
- Clippy standard et avec `cuda-driver` ;
- tests workspace sur Rust 1.86.

Aucune CI standard ne doit exiger un GPU ni linker le pilote CUDA. Les tests matériels sont explicitement ignorés sans pilote et périphérique NVIDIA. La CI compile les cibles de tests CUDA avec `--all-targets`, sans les exécuter : la compilation des tests et leur exécution sur un GPU sont deux validations distinctes.

## 5. Limites

- La validation réelle de `cuEventQuery`, `cuEventSynchronize`, `cuStreamSynchronize` et des transferts DMA sur RTX 5090 doit être exécutée sur la machine cible.
- L'injection déterministe d'erreurs event/stream/free et les tests manager-level avec backend simulé sont différés dans l'issue #19 ; les conversions de codes seules ne couvrent pas ces chemins.
- Aucun support multi-GPU/D2D ni implémentation du trait `Device` CUDA n'est ajouté.
- Aucune dépendance Candle ou autre framework tensoriel externe n'est introduite.

## 6. Critères de clôture

- [x] Stream dédié explicitement non bloquant.
- [x] Polling non bloquant des events et finalisation avant admission.
- [x] Les états pending et les erreurs ne sont pas assimilés à une complétion.
- [x] Les allocations restent protégées tant qu'un event associé reste en vol.
- [x] Départage déterministe des évictions de priorité égale pour LRU, LFU et Adaptive, y compris un plan nécessitant deux victimes.
- [x] Tests de conversion des codes d'état CUDA au niveau du wrapper ; ils ne démontrent pas l'exécution des chemins event/stream ni le comportement de `poll_completed_transfers` et ne sont pas exécutés par la CI actuelle.
- [ ] Scénario matériel ciblé `M5-HW-POLL-01` : `poll_completed_transfers` retire un event confirmé terminé, puis une admission sous pression évince l'allocation sans `wait_for_*` ; exécution RTX 5090 et preuve à consigner dans `docs/audits/m5-hardware-validation.md` avant merge.
- [x] CI verte sur le HEAD `0b7314d3a9388ef528426cdb8e55e1c4831fc025` : format, tests workspace, Clippy standard/CUDA, compilation des cibles `cuda-driver` avec `--all-targets` et MSRV Rust 1.86 (runs push et pull request [44840317161](https://github.com/zenhit777/Claymore/actions/runs/44840317161) et [44840312918](https://github.com/zenhit777/Claymore/actions/runs/44840312918)). Les tests `cuda-driver` sont compilés par le contrôle `--all-targets`, mais ne sont pas exécutés par la CI.
- [ ] Revue architecturale et fusion.
- [ ] Validation matérielle RTX 5090 ciblée et preuve versionnée.


## 7. Suivi hors M5

L'ajout d'une frontière CUDA simulable et l'injection déterministe des erreurs `cuEventQuery`, `cuStreamSynchronize` et `cuMemFree`, avec tests manager-level des invariants fail-closed, sont suivis dans l'issue [#19](https://github.com/zenhit777/Claymore/issues/19). Cette couverture n'est pas revendiquée comme livrée par M5.
