# Campagne RTX 5090 finale M10 — différentiel CUDA ↔ CPU des 30 couches du décodeur et des logits

Exécution de la réserve 1 du registre de l'architecte
(`docs/audits/m10-architect-decisions.md` §2) : la mesure CUDA ↔ CPU des
**30 couches du décodeur** et des **logits** sur RTX 5090, jamais produite
jusqu'ici. Branche `feat/m10-diffusiongemma`, SHA
`460c9ef9c2f7d3ff2d5a15c26462f25102d919ad`, après autorisation explicite de
l'utilisateur et revalidation de l'occupation (`nvidia-smi` en lecture seule
avant chaque étape).

Verdict : **9/9 étapes vertes**. Le différentiel décodeur est **bit-exact**
(`max_abs = 0,000000`, `rms = 0,000000`, **0,0 pas BF16**) sur les
**810 comparaisons d'étape** (30 couches × 27 étapes) et sur les **logits**.
Aucune modification de kernel, de source `src/`, de dorée ni de registre de
décision n'a été faite pour obtenir ce résultat.

## 1. Ce que la campagne mesure, et comment

### 1.1 Périmètre couvert

| objet | mesure |
|---|---|
| 30 couches décodeur (0..29) | 27 étapes comparées par couche (noms, projections, norms QK, RoPE, concat du cache, attention, résiduel, MLP dense, routeur, experts, somme dense, norms post-FF, scalaire de couche, sortie de couche) |
| routeur MoE | scores bruts F32 (après `widen`), poids post-softmax, ensembles d'experts discrets (top-8) |
| tête `lm_head` liée | logits F32 sur le hidden de référence, entrée identique des deux côtés |

### 1.2 Méthode

Chaque couche est exécutée **depuis son propre état d'entrée doré**
(`dec_in_L`) et **son propre cache K/V de référence** (`enc_kv_L_*`), jamais
depuis un état de chaîne Claymore ni un cache vide. Le CPU est le corps de
couche partagé de production (`fullgraph::decoder_forward_traced_from_hidden`).
Le CUDA est composé à partir des primitives publiques de `CudaGraph` dans
l'ordre exact de production, afin que **chaque intermédiaire** soit
téléchargeable et comparable.

Trois **joints bit-à-bit** (assertions fail-fast) valident le harnais avant
toute interprétation des chiffres :

1. **CPU** : la réplique d'attention atteint `stages[0]` (sortie `o_proj`) de
   la production, bit pour bit ;
2. **CPU** : le routeur rejoué sur les mêmes tenseurs reproduit les poids du
   `trace` de production ;
3. **CUDA** : la composition répliquée rend **la même sortie que le
   `text_layer` officiel** (qui libère ses internes en fin de portée), bit
   pour bit — c'est la preuve qu'aucune libération prématurée ne corrompt
   un tenseur encore lu.

Une **seconde passe CUDA** rejoue les 30 couches et exige des sorties
**bit-identiques** (absence de contamination inter-couches de l'arène/LRU).

### 1.3 Bornes fixées avant l'exécution

| métrique | borne appliquée | provenance |
|---|---:|---|
| dérive pire ligne CUDA ↔ CPU (étapes de couche) | ≤ 20,0 pas BF16 | borne D1 du décodeur (mesure CPU↔référence : 10,0 pas) |
| routeur, scores bruts et poids (F32) | `max_abs ≤ 1e-3` | précision d'égalité de frontière top-8/top-9 de la référence |
| ensembles d'experts | identiques, sinon **un** échange de frontière à égalité de score de référence ; **zéro** structurel | contrat du bisect CPU (`m10-cpu-campaign-2026-11-25.md` §7) |
| logits CUDA ↔ CPU | `max_abs ≤ 2,0`, `rms ≤ 0,1` | bornes déjà codées dans `hardware_m10_generation.rs` |
| pic VRAM | ≤ 28 Gio (dur) / 26 Gio (souple) | plafond M10 |
| finitude | toutes valeurs comparées finies | D1/D2 |

Aucune de ces bornes n'a été relâchée après observation ; les valeurs
observées sont **très en deçà** des bornes fixées.

## 2. Environnement et autorisation

| élément | valeur |
|---|---|
| poste | WSL2, `/home/sjteb/projets/Claymore` |
| GPU | NVIDIA GeForce RTX 5090, 32 607 Mio, pilote NVIDIA-SMI 615.65.07, KMD 616.64, CUDA UMD 13.4 |
| GPU libre avant session | 1 299 Mio utilisés, 1–3 % d'utilisation, **aucun processus compute** (`--query-compute-apps` vide) |
| GPU libre après session | 1 300 Mio utilisés, **aucun processus compute résiduel**, aucune entrée Xid/NVRM dans `dmesg` |
| `libcuda` | `/usr/lib/wsl/lib/libcuda.so` (pilote réel, **jamais** le stub `ci/cuda-driver-stub.c`) |
| variables | `RUSTFLAGS="-L native=/usr/lib/wsl/lib"`, `LD_LIBRARY_PATH="/usr/lib/wsl/lib"` |
| checkpoint | `google/diffusiongemma-26B-A4B-it` rev `cd40195554d3cfcb56f34b9adfb66efc790ed5b6` (`/tmp/opencode/m10/checkpoint`) |
| profil | `--release` pour les étapes à poids réels |

**Autorisation.** La RTX 5090 est partagée avec l'utilisateur. Deux
autorisations écrites ont été obtenues : la première après un préflight où
la carte était libre, la seconde après un relevé montrant une nouvelle
occupation externe (31 787 Mio / 99 %, processus Windows invisible depuis
Linux). La session n'a démarré qu'après confirmation écrite de l'utilisateur
et un nouveau relevé à 1 299 Mio. Aucune commande GPU n'a été lancée entre
le relevé d'occupation et la confirmation.

**Occupation pendant la session.** La session s'est déroulée en une seule
occupation de 21:10:45 à 21:53:19 (42 min 34 s), en neuf étapes
séquentielles, une seule tâche GPU à la fois. L'étape 6 est du hachage CPU
(4 min 35 s) ; l'étape 7 est le rejeu de la génération réelle (12 min 43 s) ;
les étapes 8 et 9 sont les deux passes du différentiel (7 min 35 s chacune).

## 3. Résultats par étape

| # | étape | attendu | obtenu | verdict |
|---|---|---|---|---|
| 1 | `hardware_cuda_driver` | 5/5 | 5/5 | ✅ |
| 2 | `hardware_cuda_memory` | 17/17 | 17/17 | ✅ |
| 3 | `hardware_cuda_tensor` | 19/19 | 19/19 | ✅ |
| 4 | `claymore-memory --lib --features cuda-driver -- --ignored` | 13/13 | 13/13 | ✅ |
| 5 | `hardware_m10_primitives` (release) | 21/21 | 21/21 | ✅ |
| 6 | `verify_checkpoint` | inventaire complet | `tensors=1047 bytes=51647562456 shards=11 coverage=complete contiguous=true` | ✅ |
| 7 | `hardware_m10_generation` (release) | 3/3 | 3/3 (pic 24 831 Mio) | ✅ |
| 8 | `hardware_m10_decoder_layers` — passe 1 | 2/2 | 2/2 | ✅ |
| 9 | `hardware_m10_decoder_layers` — passe 2 | 2/2, identique | 2/2, **tables identiques** | ✅ |

L'étape 7 confirme, sur le même SHA, que la chaîne encodeur CUDA reste
**bit-égale** au CPU (`decoder_logits0 CUDA vs CPU: max_abs=0.000000
rms=0.000000`) et que la génération réelle d'un bloc de canvas tient sur la
carte (256 positions acceptées, pic 24 831 Mio).

### 3.1 Table du différentiel décodeur (étapes 8 et 9)

Toutes les lignes de la table ont la forme
`couche | étape | max_abs | rms | pire_ligne_pas`.

| métrique sur les 810 comparaisons d'étape | valeur |
|---|---|
| `max_abs` maximal | **0,000000** |
| `rms` maximal | **0,000000** |
| pire-ligne maximal (pas BF16) | **0,0** |
| étapes non nulles | **0 sur 810** |

Échantillon (couche 0, couche 26 — la plus déviante côté référence — et
couche 29, extrait du journal) :

| couche | étape | max_abs | rms | pire ligne (pas) |
|---|---|---:|---:|---:|
| 0 | `input_norm` … `layer_output` (27 étapes) | 0,000000 | 0,000000 | 0,0 |
| 26 | `layer_output` | 0,000000 | 0,000000 | 0,0 |
| 29 | `layer_output` | 0,000000 | 0,000000 | 0,0 |

Routeur (30 couches × 2 mesures F32) : `max_abs = 0,000000`, `rms =
0,000000` sur les scores bruts **et** sur les poids post-softmax, pour toutes
les couches.

### 3.2 Ensembles d'experts (top-8)

| métrique | valeur | politique |
|---|---:|---|
| ensembles CUDA **identiques** à la référence | 470 / 480 | — |
| échanges de frontière (score de référence à égalité, écart ≤ 1e-3) | 10 / 480 | contrat du bisect CPU |
| divergences **structurelles** | **0** | interdit |
| ensembles CUDA **identiques** au CPU | 480 / 480 | rapporté |

Les 10 échanges de frontière sont **exactement les 10** déjà relevés par le
bisect CPU du 25/11 : le routeur CUDA ne diverge pas de la référence
au-delà de la précision d'égalité de la frontière top-8/top-9.

### 3.3 Logits (tête liée)

| comparaison | `max_abs` | `rms` | pire ligne (pas) |
|---|---:|---:|---:|
| logits CUDA ↔ logits CPU (entrée identique) | **0,000000** | **0,000000** | **0,0** |
| logits CUDA ↔ dorée eager (**suivi**, D2) | 0,094659 | 0,000362 | 0,8 |
| logits CPU ↔ dorée eager (**suivi**, D2) | 0,094659 | 0,000362 | 0,8 |

L'entrée du test est le hidden de référence `decoder_last_hidden0_direct`
re-quantifié en BF16 ; l'erreur de re-quantification est **0,000000** (le
dump est représentable exactement), donc les deux têtes ont reçu le même
vecteur.

Classement (rapporté, D2) : **argmax identiques 16/16 lignes**, **top-8
identiques 16/16 lignes**. La divergence de classement contre la dorée
eager est nulle et ne laisse aucun résidu.

### 3.4 Suivi CUDA ↔ dorée par couche (D2, non contractuel)

Pour mémoire, la dérive de la sortie de couche CUDA contre la dorée eager
(suivi, jamais une porte) :

| couche | `max_abs` | `rms` | pire ligne (pas) |
|---|---:|---:|---:|
| 0 | 0,500000 | 0,005732 | 2,0 |
| 5 | 0,750000 | 0,013932 | 6,0 |
| 17 | 1,375000 | 0,032405 | 5,5 |
| 23 | 0,452148 | 0,024566 | 1,8 |
| **26** | 0,156250 | 0,004456 | **10,0** (pire) |
| 29 | 0,687500 | 0,007825 | 5,5 |

La couche 26 reste la plus déviante côté référence, **cohérent avec le
bisect CPU (pire 10,0 pas sur cette couche)** : la CUDA ne fait qu'epouser
la référence.

### 3.5 Reproductibilité

Les deux passes (étapes 8 et 9) donnent des **tables strictement
identiques** (`diff` des 810 lignes d'étape et des 30 lignes de suivi :
aucune différence). La seconde passe interne du test rejoue en outre les 30
couches et exige des sorties de couche **bit-identiques** à la première
passe : aucune contamination de l'arène entre couches, aucune instabilité
d'ordre d'accumulation.

### 3.6 Mémoire

| mesure | valeur |
|---|---|
| pic VRAM du différentiel (cumul de l'arène) | 11 889 Mio |
| pic VRAM de la tête `lm_head` isolée | 1 448 Mio |
| pic VRAM de la génération réelle (étape 7) | 24 831 Mio |
| plafond contractuel | 26 Gio (souple) / 28 Gio (dur) |
| VRAM résiduelle après session | 1 300 Mio (repos WSL2), aucun processus compute |

## 4. Interprétation

1. **Le chemin CUDA du décodeur est bit-exact vis-à-vis du CPU** sur les 30
   couches et sur la tête liée, pour le scénario de référence (canvas de 16
   positions, cache K/V de référence par couche, poids épinglés). La réserve
   1 du registre est **levée sur le plan de la mesure**.
2. **Les portes fixes avant campagne sont franchies avec une marge
   confortable** : 0,0 pas contre 20,0 autorisés, 0,000000 contre 1e-3 sur
   le routeur, 0,000000/0,000000 contre 2,0/0,1 sur les logits.
3. **Aucune borne n'a été relâchée.** Les bornes candidates pour un
   resserrement (décision d'architecte, non prise ici) sont, si la
   campagne est rejouée à l'identique : pas de couche 0,0 pas, routeur
   0,000000, logits 0,000000. Un resserrement à 0,0 pas rendrait le contrat
   équivalent à une **égalité bit**, ce qui est **plus fort** que ce que D1
   exprime (D1 mesure une dérive en pas, pas une identité) et rendrait le
   test sensible à tout changement d'ordre d'accumulation côté CUDA. Ce
   resserrement est **proposé**, pas appliqué.
4. **Les divergences contre la dorée eager restent un suivi D2**, inchangé :
   elles sont mesurées (pire 10,0 pas sur la couche 26, logits 0,8 pas),
   jamais converties en porte absolue.

## 5. Limites

1. **Scénario unique.** La mesure porte sur le canvas de référence (16
   positions, positions 7..23), le cache de référence par couche et les
   poids épinglés. Elle ne couvre ni un canvas de 256 positions en
   génération longue, ni les chemins multi-blocs, ni une distribution de
   tokens autre que celle des fixtures.
2. **Cache de référence injecté.** Le cache K/V utilisé est celui de la
   référence (`enc_kv_L_*`), pas un cache produit par Claymore : la mesure
   isole la couche, elle ne mesure pas l'erreur d'accumulation du cache sur
   une chaîne longue.
3. **Masque.** Sur ce scénario, les positions 7..23 sont en deçà de la
   fenêtre glissante : le masque produit est `None` dans les deux sémantiques
   (CPU sans fenêtre, CUDA avec fenêtre de 1024). Les deux sémantiques ne
   divergent qu'au-delà de 1024 lignes de cache, **hors de ce scénario** ;
   l'observation est consignée, aucune correction n'est faite sans
   décision.
4. **Pas de performance.** Aucune latence, débit ou occupation de la carte
   n'est revendiqué ; la campagne mesure l'exactitude, pas la vitesse.
5. **Harnais de test.** Le harnais est un **test** (`tests/hardware_m10_decoder_layers.rs`,
   `#[ignore]`) : il n'entre pas dans le chemin de production et n'est pas
   exécuté par la CI standard (feature `cuda-streaming` requise).

## 6. Journaux bruts et empreintes

Journaux horodatés, en-tête `HEAD: 460c9ef9c2f7d3ff2d5a15c26462f25102d919ad`,
`nvidia-smi` avant et après chaque étape :

| fichier | contenu |
|---|---|
| `m10-final-1-2026-11-25_21-10-45-cuda-driver.log.txt` | 5 tests driver |
| `m10-final-2-2026-11-25_21-11-38-cuda-memory.log.txt` | 17 tests mémoire/transferts |
| `m10-final-3-2026-11-25_21-13-00-cuda-tensor.log.txt` | 19 tests tenseurs |
| `m10-final-4-2026-11-25_21-14-18-fault-injection.log.txt` | 13 tests gestionnaire, injections B2/B3 |
| `m10-final-5-2026-11-25_21-15-04-m10-primitives.log.txt` | 21 tests M10, dont `hw_m10_09` |
| `m10-final-6-2026-11-25_21-16-09-verify-checkpoint.log.txt` | vérification des 51,6 Go de charge utile |
| `m10-final-7-2026-11-25_21-22-06-m10-generation.log.txt` | 3 tests de génération matérielle |
| `m10-final-8-2026-11-25_21-36-40-m10-decoder-diff-1.log.txt` | différentiel 30 couches + logits, passe 1 (table complète) |
| `m10-final-9-2026-11-25_21-45-43-m10-decoder-diff-2.log.txt` | différentiel, passe 2 (reproductibilité) |
| `m10-final-2026-11-25-logs.sha256` | empreintes SHA-256 des neuf journaux |

## 7. Vérification logicielle sur le même SHA

| porte | résultat |
|---|---|
| `cargo fmt --all -- --check` | rc=0 |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | rc=0 |
| `cargo clippy -p claymore-model-gemma --features cuda-streaming --all-targets --locked -- -D warnings` | rc=0 |
| `cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings` | rc=0 |
| `cargo +1.86 clippy -p claymore-model-gemma --features cuda-streaming --all-targets --locked -- -D warnings` | rc=0 |
| `cargo test --workspace --all-targets --locked` | rc=0 |
| CI GitHub Actions (format/test/lint + MSRV 1.86) | verte sur le SHA final |

## 8. Ce qui reste à décider

1. **Resserrement des portes CUDA ↔ CPU** : les valeurs mesurées sont à 0,0
   pas ; le resserrement proposé (0,0 pas, routeur et logits à 0,000000) est
   une **décision d'architecte** et n'est pas appliqué ici. D1/D2 ne sont
   pas modifiés.
2. **Réalignement de la documentation générale** (`docs/m10-diffusiongemma.md`,
   `docs/architecture.md`, `docs/Claymore.md`) sur la finalisation CPU et la
   présente campagne : fait dans le même changement que ce rapport.
3. **Merge de la PR #40** : reste une décision distincte. **Aucun merge
   n'est effectué.**
