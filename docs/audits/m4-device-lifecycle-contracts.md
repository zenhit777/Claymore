# M4 — Audit et durcissement des contrats Device / mémoire

**Statut :** implémentation proposée dans la PR M4 ; non livrée sur main avant fusion.  
**Base auditée :** merge M3 `e758862937a0dff376b83449aab2c937400be9f4`.  
**Issues couvertes :** #11 et #12.

## 1. Périmètre

M4 prépare le cycle de vie explicite des allocations et la description des opérations asynchrones avant tout raccordement de `CudaDriver` au trait `Device`.

Le jalon couvre :
- une opération de libération explicite et faillible au niveau du contrat `DeviceBuffer` ;
- la conservation de l'ownership et de la comptabilité après un échec de libération ;
- une source de vérité commune pour les ticks monotones et les horizons de libération ;
- un contrat de jeton de complétion pour les opérations asynchrones ;
- des jetons publics pour les transferts H2D/D2H gérés par `CudaMemoryManager`, avec interrogation réelle des CUDA events ;
- des vues de tenseurs stridées et bornées ;
- la clarification du nom du buffer CUDA.

M4 n'implémente pas `Device` pour CUDA, ne lance aucun kernel et ne construit pas de scheduler.

## 2. Constats de l'audit

### 2.1 Libération et comptabilité

Le contrat M3 ne proposait que la destruction implicite d'un `DeviceBuffer`. Le gestionnaire générique retirait le buffer de sa table puis le détruisait par `Drop`, sans pouvoir connaître un échec backend.

Le backend CUDA possédait déjà des opérations explicites `free()` faillibles, mais leur utilisation était spécifique au gestionnaire CUDA. Le contrat générique ne pouvait pas exprimer cette garantie.

**Décision :** `DeviceBuffer::release(&mut self)` est explicite et faillible. Le gestionnaire générique appelle cette opération avant de retirer le buffer et de confirmer l'éviction. Si la libération échoue, le buffer et sa comptabilité restent présents afin de permettre un diagnostic ou une nouvelle tentative. Les implémentations doivent refuser la libération d'une ressource encore utilisée (`BufferBusy`).

### 2.2 Invariants communs des gestionnaires

Les gestionnaires CPU et CUDA vérifiaient séparément :
- qu'un tick ne recule pas ;
- qu'une allocation n'est libérable qu'après son horizon de complétion.

Cette duplication était une source de divergence.

**Décision :** les fonctions pures communes sont centralisées dans `claymore-memory/src/lifecycle.rs`. Les gestionnaires conservent leurs types d'erreurs et leurs politiques spécifiques ; seuls les invariants identiques sont factorisés. La logique CUDA d'events et de contexte reste spécifique au backend.

### 2.3 Complétion asynchrone

Les transferts H2D/D2H du `CudaMemoryManager` étaient associés à des events réels et les buffers restaient protégés jusqu'à synchronisation. Cependant, les méthodes de copie retournaient `()` : le demandeur ne pouvait pas identifier et interroger la complétion d'une opération donnée.

**Décision :**
- le contrat backend-neutre expose `CompletionToken` avec `is_complete` et `wait` ;
- `Device::copy_async` retourne un token empruntant le device et les deux buffers ; le CPU, qui n'annonce pas de transfert asynchrone, retourne explicitement `UnsupportedOperation` ;
- `CudaMemoryManager::copy_h2d` et `copy_d2h` retournent un `TransferToken` opaque ;
- `is_transfer_complete` interroge le CUDA event via `cuEventQuery` (le code CUDA_ERROR_NOT_READY signifie simplement « pas encore terminé ») ;
- `wait_for_transfer` synchronise l'event. En cas d'erreur, l'event et les protections d'allocations restent suivis.

Le contrat de `CompletionToken` impose à chaque implémentation de synchroniser ou de conserver les leases backend si le token est abandonné avant complétion. Un simple abandon du token ne doit jamais rendre une allocation évictable.

### 2.4 Vues tensor non contiguës

M3 ne décrivait que des tenseurs propriétaires contigus. Cela ne suffisait pas pour représenter une tranche ou une vue transposée sans copie.

**Décision :** ajout de `TensorView`, vue empruntée contenant shape, strides en éléments, dtype et offset en octets. Le calcul de l'étendue mémoire utilise des opérations vérifiées et rejette les rangs incompatibles, les dimensions nulles, les débordements et les vues dépassant le buffer. Aucun kernel ou accès arithmétique aux éléments n'est introduit.

Les strides sont non signés ; les strides négatifs ne sont pas pris en charge. Les vues ne possèdent pas le buffer et ne prolongent pas sa durée de vie au-delà de leur emprunt.

### 2.5 Collision de nommage

Les crates `claymore-device` et `claymore-cuda` exposaient chacune un type nommé `DeviceBuffer`, alors que ces types ont des responsabilités et des contrats différents.

**Décision :** le wrapper CUDA est renommé `CudaDeviceBuffer`. Le nom `DeviceBuffer` est réservé au contrat backend-neutre de Claymore.

### 2.6 Copies inter-devices

Le profil matériel actuel cible un seul GPU NVIDIA. Les chemins H2D et D2H sont déjà présents dans `CudaMemoryManager`. Un chemin D2D multi-device n'est pas requis pour ce jalon et supposerait un contrat de coordination entre deux contextes/device owners.

**Décision :** conserver H2D/D2H et leurs jetons de complétion ; différer D2D jusqu'à une exigence multi-GPU explicite. Aucun faux support inter-device n'est ajouté.

## 3. Changements implémentés

- Ajout de `DeviceBuffer::release` et des erreurs `BufferReleased`, `BufferBusy`, `ReleaseFailed` et `UnsupportedOperation`.
- Le simulateur CPU libère explicitement son stockage et refuse les accès après libération.
- `DeviceMemoryManager` ne retire plus une allocation de la table ni du budget avant le succès de `release`.
- Test fake injectant un échec de libération : buffer, données et comptabilité restent disponibles après l'erreur.
- Factorisation des invariants de tick monotone et d'horizon de libération dans un module partagé.
- Ajout de `CompletionToken` et de `Device::copy_async` avec comportement par défaut fail-closed.
- Ajout de `TransferToken` aux transferts CUDA H2D/D2H, avec interrogation non bloquante de l'event et attente explicite.
- Ajout de `CudaEvent::is_complete` utilisant `cuEventQuery`.
- Renommage du wrapper CUDA en `CudaDeviceBuffer`.
- Ajout de `TensorView` et de tests de strides, offset, rang et bornes.

## 4. Limites et risques assumés

- Aucun raccordement de `CudaDeviceBuffer` au trait `DeviceBuffer` backend-neutre n'est effectué dans M4.
- Le contrat `CompletionToken` est défini, mais aucun backend CUDA générique `Device::copy_async` ne l'implémente encore. Les transferts concrets restent exposés par `CudaMemoryManager`.
- `TransferToken` expose une interrogation d'event et une attente, mais ne fournit pas encore de polling automatique ni de finalisation automatique des allocations ; ces traitements relèvent du jalon M5.
- Les tests unitaires de la logique pure et du backend fake ne remplacent pas un test matériel des nouveaux appels `cuEventQuery` sur RTX 5090.
- Le modèle CUDA demeure confiné au thread propriétaire ; aucune garantie `Send` / `Sync` n'est ajoutée.
- `Drop` ne peut pas remonter une erreur. La voie normale de libération gérée reste donc l'appel explicite et faillible avant retrait de l'ownership et de la comptabilité. Le `Drop` CUDA conserve son rôle de dernier recours best-effort ; en cas d'incertitude sur une opération en vol, le gestionnaire fuit volontairement les ressources plutôt que de risquer un use-after-free.

## 5. Critères de clôture

- [x] Les gestionnaires CPU et CUDA utilisent une source commune pour les invariants identiques de cycle de vie.
- [x] La libération générique est explicite et faillible.
- [x] Un échec de libération conserve le buffer et la comptabilité (test fake).
- [x] Les opérations CUDA H2D/D2H exposent un token de complétion vérifiable par event.
- [x] Les contrats asynchrones définissent les obligations de durée de vie et de fail-closed.
- [x] Les vues tensor stridées valident rang, overflow et étendue mémoire.
- [x] Le wrapper CUDA possède un nom distinct du contrat backend-neutre.
- [ ] CI stable/MSRV et feature CUDA validées sur le HEAD final de la PR.
- [ ] Revue architecturale et fusion.
- [ ] Validation matérielle des nouveaux chemins event query/wait sur RTX 5090 (suivi M5 si non disponible pendant M4).

