# M2 — Contrats Device / Tensor et backend CPU

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

Les buffers sont possédés par leurs wrappers backend et libérés par leur destruction. La couche mémoire confirme la libération comptable uniquement après la destruction du buffer détenu.

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

Aucune bibliothèque tensorielle tierce n'est une dépendance requise ni une cible d'intégration de Claymore. Les choix de représentation, d'opérations, de propriété mémoire et d'exécution seront définis et validés dans l'architecture du projet. Les références externes peuvent servir à comparer des approches, sans dicter les contrats ni introduire de dépendance.

Le raccordement du backend CUDA devra respecter les contrats Claymore de contexte, d'ownership, de synchronisation et de durée de vie des buffers.

## Limites et suites

- Le backend CUDA n'implémente pas encore `Device`.
- Les transferts asynchrones et leurs jetons de complétion restent à définir avant toute orchestration concurrente.
- L'injection de fautes et le test de sûreté des buffers en vol relèvent du raccordement CUDA et de M4.
- Le CPU simulator ne prouve ni le comportement matériel CUDA, ni la mémoire pinned réelle, ni la performance.

## Validation CI

SHA validé par la CI : `0b122279af7699d90550fc400045451f42b2958b`.

Sur ce SHA, les deux checks déclenchés (push et pull request) sont verts. Le job CI a exécuté :
- `cargo fmt --all -- --check` : succès ;
- `cargo test --workspace --all-targets` : succès, dont 6 tests `claymore-device` et 9 tests `claymore-memory` ;
- `cargo check -p claymore-memory --features cuda-driver` et `cargo check -p claymore-cli --features cuda-driver` : succès ;
- Clippy workspace et Clippy avec feature CUDA : succès.

La feature CUDA est compilée sans lien vers le driver ni GPU, conformément au workflow. Ces checks ne valident pas l'implémentation future du trait Device par CUDA.
