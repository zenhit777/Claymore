# M2 — Contrats Device / Tensor et backend CPU

**Statut :** M2 est livré par la PR #6 : contrats `Device` / `DeviceBuffer` / `Tensor`, simulateur CPU et adaptateur mémoire générique. Aucun calcul tensoriel ni raccordement CUDA au trait `Device` n'est inclus dans ce jalon.

## Portée

M2 introduit une frontière backend-neutre pour les périphériques, les buffers et les métadonnées de tenseurs, ainsi qu'un backend CPU déterministe permettant de tester les contrats et l'adaptateur mémoire sans GPU.

Ce jalon ne livre ni opérateurs tensoriels, ni autograd, ni exécution de modèles, ni runtime, ni scheduler, ni intégration d'une bibliothèque tensorielle tierce, ni implémentation du trait Device par le backend CUDA existant.

## Contrats livrés

### Device et capacités

- `Device` expose l'identité, les capacités, l'allocation, l'accès borné, la copie et la synchronisation.
- `DeviceCapabilities` rapporte explicitement mémoire device/host/pinned, transferts asynchrones et calcul tensoriel.
- `DeviceBuffer` possède sa mémoire et expose son device, son tier et sa capacité.
- Aucun trait n'impose `Send` ou `Sync`. Un backend lié à un thread ou à un contexte reste valide ; le runtime devra respecter ce confinement.

### Synchronisation et propriété

Les opérations de M2 (`write`, `read`, `copy`) sont synchrones : leur retour garantit que l'accès aux buffers est terminé. Un futur chemin asynchrone ne devra pas réutiliser ce contrat en prétendant qu'un simple retour signifie complétion ; il devra retourner un jeton de transfert explicite dont l'attente protège les buffers jusqu'à la complétion.

Les buffers sont possédés par leurs wrappers backend et libérés par leur destruction. `DeviceMemoryManager` détruit désormais le buffer détenu avant de confirmer la libération comptable. Cette séquence est sûre pour le simulateur CPU et le contrat actuel, mais la destruction par `Drop` ne peut pas signaler un échec de libération du backend.

### Tensor

Le type `Tensor` de M2 décrit un buffer possédé, un dtype, une shape et des strides contigus row-major. La construction vérifie les dimensions, les débordements et la capacité du buffer.

Il ne fournit aucun calcul, opérateur, autograd, layout arbitraire ou conversion de précision.

## Backend CPU

`CpuDevice` est un simulateur déterministe utilisant des allocations `Vec<u8>`. Il permet de tester les tiers Host et Vram comme espaces de stockage simulés. Il refuse `PinnedHost` : la mémoire ordinaire du simulateur ne doit pas être présentée comme physiquement épinglée.

Il ne fournit ni transferts asynchrones, ni kernels, ni calcul tensoriel.

## Adaptateur mémoire

`DeviceMemoryManager<D>` connecte un backend implémentant `Device` au `MemoryOrchestrator` existant. Il couvre l'allocation, l'accès, la copie synchrone, la synchronisation, la libération et l'application des plans d'éviction.

Les tests CPU couvrent notamment :
- allocation, lecture/écriture et copie entre buffers ;
- libération et mise à jour de la comptabilité ;
- rejet des identifiants dupliqués et des ticks non monotones ;
- éviction déterministe selon la politique mémoire.

Ce manager générique est distinct du `CudaMemoryManager` spécifique M1, qui conserve actuellement son implémentation CUDA dédiée.

## Indépendance du moteur tensoriel

Les types `Device`, `DeviceBuffer` et `Tensor` définis dans ce jalon sont des abstractions propriétaires de Claymore. Ils constituent la base du futur moteur tensoriel et des chemins d'exécution des modèles.

Claymore ne délègue pas ses abstractions, sa gestion mémoire ni son runtime à un framework tensoriel tiers. Ce choix conserve la maîtrise des contrats `Device` / `Buffer` / `Tensor`, de l'ownership, de la synchronisation et des chemins d'exécution spécifiques aux modèles, et évite une dépendance structurante ainsi que les adaptations imposées par le cycle de vie d'un framework externe. Il augmente en contrepartie l'effort de développement et de maintenance du moteur, la responsabilité de la correction numérique et le risque de performances inférieures à des runtimes établis.

La politique de dépendances distingue les frameworks tensoriels des primitives spécialisées. Les API et bibliothèques NVIDIA de bas niveau (CUDA Driver API, CUDA Runtime si nécessaire, cuBLAS/cuBLASLt, CUTLASS, cuDNN) sont autorisées pour les opérations et kernels ; les bibliothèques de formats (dont safetensors), les tokenizers et les crates utilitaires Rust (par exemple `half`) le sont également après revue. Ces dépendances ne doivent pas devenir propriétaires des types tensoriels, du cycle de vie des buffers, de la politique mémoire ou du runtime de Claymore.

Le raccordement du backend CUDA devra respecter les contrats Claymore de contexte, d'ownership, de synchronisation et de durée de vie des buffers. La destruction actuelle via `Drop` ne permet pas d'exprimer une libération backend faillible ni une protection d'allocation en vol ; avant d'implémenter `Device` pour CUDA, le contrat devra prévoir une opération de libération faillible (par exemple `free(self) -> Result`) et une garde explicite jusqu'à la complétion du travail GPU, afin de préserver les garanties fail-closed de M1.

## Limites et suites

- Le backend CUDA n'implémente pas encore `Device` ; il ne devra pas le faire avant l'évolution du contrat de libération pour préserver les garanties fail-closed de M1.
- Les transferts asynchrones et leurs jetons de complétion restent à définir avant toute orchestration concurrente.
- L'injection de fautes et le test de sûreté des buffers en vol relèvent du raccordement CUDA et de M4.
- Le CPU simulator ne prouve ni le comportement matériel CUDA, ni la mémoire pinned réelle, ni la performance.
- Le moteur tensoriel natif reporte sur Claymore la responsabilité de la correction numérique (notamment BF16/FP32), des performances GEMM/attention sur sm_120 et de la comparaison systématique à une référence indépendante.

## Validation CI

SHA final validé par la CI après les corrections de revue : `795bcd9ad8b49e5de4cd0c67a1e27a4f9f037a24`.

Les deux checks sont verts sur ce SHA : [CI push](https://github.com/zenhit777/Claymore/actions/runs/44702130042) et [CI pull request](https://github.com/zenhit777/Claymore/actions/runs/44702134289). Le job CI a exécuté :
- `cargo fmt --all -- --check` : succès ;
- `cargo test --workspace --all-targets` : succès, dont 6 tests `claymore-device` et 9 tests `claymore-memory` ;
- `cargo check -p claymore-memory --features cuda-driver` et `cargo check -p claymore-cli --features cuda-driver` : succès ;
- Clippy workspace et Clippy avec feature CUDA : succès.

La feature CUDA est compilée sans lien vers le driver ni GPU, conformément au workflow. Ces checks ne valident pas l'implémentation future du trait Device par CUDA.
