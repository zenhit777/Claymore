# M3 — Audit technique et durcissement des fondations Device

**Statut :** implémentation proposée dans la PR M3 ; non livrée sur `main` tant que cette PR n'est pas fusionnée.  
**Base auditée :** merge M2 `7cdd5e8680c8fc74d925b7aab931df7e7ff291a3`.  
**Issues couvertes :** #7, #8, #9 et #10.

## 1. Périmètre et méthode

M3 consolide les contrats introduits par M2 avant d'étendre Claymore vers les transferts asynchrones, le runtime tensoriel ou le raccordement CUDA au trait `Device`.

Le périmètre est volontairement limité à :
- la déclaration et la vérification de la MSRV ;
- l'identité des instances de device et l'ownership des buffers ;
- la classification des erreurs de taille et d'allocation ;
- la couverture de tests déterministes du backend CPU et de l'adaptateur mémoire.

Aucun kernel, calcul tensoriel, transfert asynchrone, nouveau comportement CUDA ou changement du contrat de libération n'est introduit ici. Ces sujets restent dans les jalons ultérieurs et les issues #11 à #13.

## 2. Constats de l'audit sur M2

### 2.1 MSRV implicite

Le workspace n'avait pas de déclaration `rust-version`. Le code utilise toutefois `HashMap::get_disjoint_mut`, API stabilisée avec Rust 1.86. L'absence de MSRV dans les manifests et d'un job CI dédié rendait cette contrainte implicite et sujette à régression.

**Décision M3 :** Rust 1.86.0 est la MSRV workspace. Chaque crate hérite de `workspace.package.rust-version`. La CI conserve le job stable existant et exécute aussi la suite complète sur 1.86. La politique de hausse est documentée.

### 2.2 Identité de device insuffisante

M2 comparait `DeviceBuffer::device_id()` avec `DeviceInfo::id`. Pour `CpuDevice`, cette valeur est `cpu:0` pour toutes les instances ; deux simulateurs indépendants pouvaient donc accepter les buffers l'un de l'autre.

**Décision M3 :** ajouter `DeviceInstanceId`, opaque et unique dans le processus, à `DeviceInfo` et `DeviceBuffer`. Le champ `id` reste un identifiant logique/diagnostique, pas une frontière d'ownership. Les opérations CPU valident l'identité d'instance avant tout accès ou copie.

Sémantique de clonage : `CpuDevice::clone()` représente un autre handle vers la même instance et conserve donc son `DeviceInstanceId`. Deux appels distincts à `CpuDevice::new()` créent deux instances différentes, même si leur identifiant logique est identique. L'identité n'est ni persistante, ni globale entre processus.

### 2.3 Confusion entre overflow et épuisement mémoire

M2 transformait toute erreur de `Vec::try_reserve_exact` en `SizeOverflow`. Or une demande peut être représentable mais échouer faute de capacité mémoire.

**Décision M3 :**
- `SizeOverflow` reste réservé aux conversions et tailles non représentables ;
- `OutOfMemory { requested_bytes, reason }` signale un échec de réservation pour une taille représentable ;
- le texte de l'erreur de réservation est conservé pour le diagnostic sans exposer le type concret de l'allocateur dans l'API publique.

L'injection d'échec dans le helper privé d'allocation rend le chemin OOM déterministe. Le chemin overflow utilise une limite de plateforme simulée dans un helper privé, sans dépendre d'une architecture 32 bits ni d'une saturation réelle.

### 2.4 Couverture de tests incomplète

Les tests M2 ne couvraient pas explicitement la contrainte expert/VRAM, les copies aliasées, les allocations nulles, l'isolation entre deux devices ni la cohérence des capacités annoncées.

M3 ajoute ces scénarios au backend CPU et à l'adaptateur mémoire. Les assertions vérifient notamment que les rejets n'entraînent ni mutation de destination ni comptabilisation d'une allocation refusée.

## 3. Changements implémentés dans la PR

### 3.1 MSRV

- `rust-version = "1.86"` déclaré dans `[workspace.package]`.
- Tous les manifests de crates héritent de `rust-version.workspace = true`.
- Nouveau job CI `msrv` utilisant Rust 1.86.0 et `cargo test --workspace --all-targets`.
- README et référence produit précisent la MSRV et la procédure explicite de mise à niveau.

### 3.2 Identité d'instance

- Nouveau type public `DeviceInstanceId`, opaque, comparable et hashable.
- Identité allouée atomiquement à chaque nouvelle instance CPU.
- Identité conservée lors du clonage du handle CPU.
- Nouveau contrat `DeviceBuffer::device_instance_id()`.
- Les chemins CPU de lecture, écriture et copie vérifient l'identité exacte avant le downcast et l'accès aux octets.
- Les tests utilisent deux instances dont l'identifiant logique est identique et vérifient les rejets croisés ainsi que l'absence de mutation.

### 3.3 Erreurs d'allocation

- Ajout de `DeviceError::OutOfMemory` avec taille demandée et diagnostic de réservation.
- `usize::try_from` et le contrôle de représentabilité continuent de retourner `SizeOverflow`.
- Les erreurs d'allocation ne sont plus classées comme des overflow.
- Tests d'injection indépendants de la pression mémoire de l'hôte.

### 3.4 Tests explicites

Le backend `claymore-device` couvre désormais :
- l'identité distincte de deux devices CPU ;
- la sémantique de clonage d'un handle ;
- le rejet des buffers étrangers en lecture, écriture et copie ;
- l'absence de mutation lors d'un rejet de copie ;
- la cohérence entre capacités annoncées et tiers réellement allouables ;
- le refus d'une allocation nulle ;
- les chemins distincts overflow et OOM.

L'adaptateur `claymore-memory` couvre désormais :
- le rejet d'un expert hors VRAM sans comptabilisation ;
- le rejet d'une copie aliasée sans altération des octets ;
- le refus d'une allocation de taille nulle sans comptabilisation ;
- la cohérence des capacités exposées par le device géré.

## 4. Limites assumées

- `DeviceInstanceId` est unique uniquement dans le processus courant. Il ne doit pas être sérialisé ni utilisé comme identifiant matériel persistant.
- La sémantique de clonage documentée ici concerne `CpuDevice`. Chaque backend futur devra définir si son clone est un alias du même device/context ou une nouvelle instance.
- `OutOfMemory.reason` conserve le diagnostic textuel de l'allocateur ; il ne promet pas une classification OS plus précise que celle fournie par `TryReserveError`.
- Les tests CPU ne valident pas la sûreté du matériel CUDA, les allocations VRAM physiques, la mémoire pinned, les streams ou les events.
- Les contrats de libération faillible et de protection des allocations en vol restent prérequis avant l'implémentation de `Device` par CUDA ; ils sont suivis par #11 et #12.

## 5. Critères de clôture

- [x] MSRV déclarée dans le workspace et héritée par chaque crate.
- [x] CI configurée pour exécuter les tests workspace sur Rust 1.86.
- [x] Identité d'instance distincte de l'identifiant logique.
- [x] Sémantique de clonage documentée et testée.
- [x] Distinction explicite OOM / overflow et tests déterministes.
- [x] Tests CPU pour les cas demandés par #10.
- [ ] CI verte sur le SHA final de la PR.
- [ ] Revue architecturale et fusion de la PR.

Les deux derniers critères ne seront cochés qu'après observation des résultats GitHub sur le commit final et validation de la revue.
