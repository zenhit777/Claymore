# Intégration Memory ↔ CUDA v0.5

## Objectif

Cette phase relie les décisions de la couche de politique `claymore-memory` aux opérations réelles du backend `claymore-cuda`, sans introduire Candle ni exécuter un modèle. Le chemin reste opt-in via la feature `cuda-driver`. Le build et les tests de politique par défaut restent indépendants d'un GPU.

## Code ajouté dans cette contribution

- `CudaMemoryManager` possède le driver, un stream de transfert, le `MemoryOrchestrator`, les buffers device et pinned host.
- L'admission demande d'abord un plan au gestionnaire. Les candidats sont libérés par le backend avant que `confirm_eviction` ne modifie la comptabilité.
- Les allocations VRAM vérifient aussi la mémoire libre remontée par le driver juste avant l'allocation.
- Les buffers ont une opération `free()` explicite et faillible. Une erreur CUDA de libération ne doit pas être convertie en succès comptable.
- Les copies H2D et D2H enregistrent un CUDA event sur le stream de transfert. Les deux allocations impliquées sont marquées en vol jusqu'à la synchronisation effective.
- Le manager impose une horloge logique monotone : toute opération recevant un tick inférieur au précédent est rejetée.
- `release` vérifie que `safe_after_tick <= tick` avant de libérer le buffer, afin d'éviter une divergence entre mémoire physique et comptabilité.
- Si la synchronisation d'un event échoue, l'event reste suivi et les allocations restent non évictables.
- Le backend CUDA et le manager sont confinés au thread propriétaire : le contexte interne utilise `Rc` et n'est ni `Send` ni `Sync`. Le scheduler M4 devra utiliser un thread CUDA propriétaire dédié ou faire évoluer explicitement cette architecture.
- La feature `claymore-memory/cuda-driver` active le backend optionnel. La feature équivalente du CLI la relaie.

## API de départ

La surface publique est disponible derrière `--features cuda-driver` :

- `CudaMemoryManager::new(ordinal, budgets, eviction)`
- `allocate_device(id, bytes, tick, expert)`
- `allocate_pinned(id, bytes, tick)`
- `write_pinned(id, offset, data, tick)` et `read_pinned(id, offset, len, tick)`
- `copy_h2d(pinned_id, device_id, bytes, tick)`
- `copy_d2h(device_id, pinned_id, bytes, tick)`
- `wait_for_allocation(id, tick)`
- `release(id, tick)`
- `snapshot(host_total_bytes, host_available_bytes)`

`read_pinned` attend automatiquement la fin des transferts associés avant de retourner les octets. Les opérations de copie sont asynchrones ; les allocations restent possédées par le manager. Les ticks fournis à toutes les opérations du manager doivent être monotones et non décroissants.

## Limites explicites

- Cette phase ne raccorde pas encore Candle/cudarc et ne résout pas le risque de contexte primaire versus contexte créé par `cuCtxCreate_v2`.
- Le manager ne prend en charge que les tiers VRAM et PinnedHost pour les allocations physiques. Le tier Host reste géré par la politique hôte.
- Le profil JSON n'est pas chargé automatiquement.
- La politique de budget souple VRAM de 26 Gio, le working set de 18 Gio et l'enveloppe cache experts 14/16 Gio ne sont pas encore appliqués comme contraintes distinctes par le gestionnaire existant.
- Les observations RAM hôte sont fournies par l'appelant ; aucune télémétrie système n'est intégrée ici.
- Les tests CUDA sont ignorés par défaut et exigent un runner avec driver NVIDIA. La CI standard doit rester hardware-independent.
- Aucune exécution sur WSL2 / RTX 5090 n'est attestée par cette PR.

## Validation attendue

CI standard :

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Validation matérielle à réaliser ultérieurement dans WSL2 :

```bash
nvidia-smi
cargo check -p claymore-memory --features cuda-driver
cargo test -p claymore-memory --features cuda-driver -- --ignored
```

Le test CUDA ignoré inclut un aller-retour H2D puis D2H avec vérification des octets. La validation sur cible devra compléter ces commandes par un scénario de pression mémoire provoquant une éviction et une vérification de la comptabilité après chaque succès ou échec.
