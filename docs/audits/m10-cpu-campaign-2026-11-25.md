# Campagne CPU de finalisation M10 — 25 novembre 2026

Exécution des travaux CPU-only imposés par le verdict architecte du 25/11
(`m10-architect-decisions.md` §2, repris dans `m10-diffusiongemma.md` §« État
normatif »), sur la branche `feat/m10-diffusiongemma`. **Aucune utilisation du
GPU** : cette campagne est purement CPU, la RTX 5090 et le créneau matériel
restent réservés à l'agent matériel (protocole groupé §`m10-rtx5090-grouped-protocol.md`,
préparé, non exécuté).

Contrat appliqué : `m10-cpu-finalization-contract.md` ; politique de
reproductibilité : `docs/reproducibility.md`.

## 1. Départs et empreintes

- Branche de départ : `feat/m10-diffusiongemma` à `97d22d6` (contient
  `24cb8dd` verdict D1/D2, le contrat `ec09d6f` et l'outillage de génération).
- Checkpoint épinglé : `google/diffusiongemma-26B-A4B-it` révision
  `cd40195554d3cfcb56f34b9adfb66efc790ed5b6` ; `config.json`
  `13b11d2fe87302cc…fc70c506`, `model.safetensors.index.json`
  `6e33e8465d55fe6c…f61dac69a` (revérifiés dans le manifeste généré).
- Environnement de référence (venv épinglé, CPU only) : Python 3.12.3,
  torch 2.15.0+cpu, transformers 5.24.0.dev0 @ `671aa77576266f41f33f9c28e2d824bfd1b1b63c`,
  numpy 2.5.5 ; glibc 2.39 ; AMD Ryzen 9 7950X3D ; `CUDA_VISIBLE_DEVICES=""`.
- Outils : journal des versions dans chaque commande exécutée (section 8).

## 2. Référence par couche versionnée

Génération unique par `tools/m10/gen_reference.py` dans un répertoire neuf
(`/tmp/m10-reference/run-2026-11-25/`, anti-écrasement) : **455 artefacts**
FP32 LE, `manifest.json` + `manifest.sha256`.

- Contrôles internes de cohérence de chaîne : **85/85** bit-identiques
  (`enc_in_{L+1} == encoder_layer_L`, `vision_in_{L+1} == vision_layer_L`,
  `dec_in_{L+1} == decoder_layer_L`, lignes de cache à 7 à chaque entrée
  décodeur).
- Portes de bit-exactité contre les dorées historiques `golden_eager` /
  `golden_layers` : **311/311**, `gate_failures` vide.
- Journal brut : `m10-cpu-2026-11-25-01-reference-generation.log.txt`
  (`DONE_REFERENCE`).

Faute d'outillage trouvée et corrigée par le validateur : le
`manifest.sha256` écrit par le générateur contenait le *repr* Python de
l'objet hash au lieu de l'empreinte hexadécimale. Corrigé (`.hexdigest()`,
commit `de90bef`) ; le sidear a été régénéré à partir du manifeste inchangé,
**aucun artefact ni manifeste modifié**.

## 3. Fixtures engagées et validateur CI

`crates/claymore-model-gemma/tests/data/m10_layers/` (455 `.bin` + manifeste,
41 Mio) — commit `ef329ca`. Le test **non ignoré** `tests/m10_reference.rs`
(exécuté par la CI standard, sans checkpoint) épingle l'identité du jeu
(SHA-256 du manifeste `660da49e8a12e24a…c74e3`), revérifie taille, finitude
et SHA-256 de chaque artefact, les portes enregistrées (455/85/311, zéro
échec) et les empreintes d'environnement et de checkpoint.

## 4. Points d'injection par couche et égalité bit à bit

Refactor de `fullgraph.rs` en **extraction de corps partagé, sans
duplication** (le corps d'une couche est littéralement celui des chaîns) :

- vision : `vision_forward_stages` → prologue patch-embed +
  `vision_forward_inner(layers)` + épilogue pooling ; entrée diagnostic
  `vision_forward_layer_from_hidden` ;
- décodeur : `decoder_forward_traced` → prologue embeddings/self-cond +
  `decoder_forward_range(layers)` + norm finale ; entrée diagnostic
  `decoder_forward_traced_from_hidden` ;
- encodeur : `encoder_forward_traced_from_hidden` (préexistant, inchangé).

**Preuve d'égalité bit à bit** du refactor sur le même arbre : la chaîne
`golden_cpu` (7 tests) exécutée avant (arbre HEAD, sans refactor) et après
(arbre refactorisé), 7/7 les deux fois ; le `diff` des tables par étape et par
couche (`max_abs=`/`rms=`/`exact=`) est **identique**. Journaux
`-02-chain-…-pre-refactor` et `-03-chain-…-post-refactor` ; seul le horodatage
diffère.

## 5. Bisect encodeur à entrées dorées — rejoué sur la tête finale

`tests/golden_cpu_layer_bisect.rs` (chemins par `CLAYMORE_M10_CHECKPOINT` /
`CLAYMORE_M10_LAYERS_DIR`) — journal `-04-bisect-encoder-entrees-dorees` :

**210/210 lignes fidèles (30 couches × 7 lignes), top-2 exact partout,
pire ligne 2,0 ULP à la couche 2** — conforme au journal historique
`m10-parity-11` (210/210, 2,0 ULP, couche 2). Porte D1 : 4,0 = 2 × mesuré.

## 6. Bisect des 27 couches de vision

`tests/vision_layer_bisect.rs` — chaque couche reçoit son entrée de référence
(`vision_in_L`), fidélité d'enchaînement vérifiée bit à bit
(`vision_in_{L+1} == vision_layer_L`) avant toute comparaison. Journal
`-05-bisect-vision-27-couches` :

| couche pire | max_abs | pire ligne (pas BF16 du pic de ligne) |
|---|---:|---:|
| 24 | 2.000000 | **0.2** |
| 25 | 1.000000 | 0.1 |
| 26 | 1.000000 | 0.1 |

Pire dérive sur les 27 couches : **0,2 pas BF16 (couche 7)**. Porte fixée à
la règle uniforme `max(2 × mesuré, 4.0)` = **4,0** — identique à l'encodeur,
le plancher absorbe une différence d'ordre de réduction d'un pas entier sur
un autre hôte, un défaut structurel se mesurant des ordres de grandeur audessus.

## 7. Bisect des 30 couches du décodeur

`tests/decoder_layer_bisect.rs` — entrée de référence (`dec_in_L`) **et cache
K/V de référence** de la couche (`enc_kv_{L}_{keys,values}`, permutation du
layout amont `[batch, têtes, seq, dim]` vers le layout local), positions
`7..23`, jamais de cache vide ni Claymore. Journal `-06-bisect-decodeur-30-couches` :

- sélection d'experts : **470/480 ensembles top-8 identiques à bit exact,
  10 échanges de frontière, 0 structurels** ;
- dérive : pire ligne **10,0 pas BF16** (couche 26) ;
  pires couches : 26 → 10,0 ; 8 → 6,0 ; 5 → 6,0 ; 17/29 → 5,5.

Mesure décisive sur la frontière top-8/top-9 : dans les scores **de la
référence elle-même**, l'écart entre le 8ᵉ expert retenu et le 9ᵉ non retenu
est **0.000000 dans la majorité des lignes** et ne dépasse jamais ~5e-4 — la
frontière est à précision d'égalité. Une égalité stricte d'ensemble n'est
donc reproductible que sur une réduction bit identique ; l'exiger serait une
porte absolue que D1 interdit. La porte retenue (fixée par cette mesure,
article 5 du contrat) : ensemble identique, **ou** un seul échange de
frontière dont les deux experts sont à égalité dans la référence dans la
limite de dérive de scores observée (1e-3, soit 2 × le max `scores_max`
mesuré 4,3e-4) ; deux échanges ou un échange au-delà de l'égalité sont
structurels. Dérive : porte **20,0 = 2 × 10,0**, même règle `max(2×mesuré, 4.0)`.

## 8. Suite logicielle CPU

Réplique complète des deux jobs de la CI (`.github/workflows/rust.yml`), sur
la tête finale, chaque étape avec son code de retour vérifié — journal brut
`-07-ci-replica.log.txt` :

| Étape | Résultat |
|---|---|
| `cargo fmt --all -- --check` | rc=0 |
| `cargo test --workspace --all-targets --locked` | rc=0 — **268 passés / 0 échec / 12 ignorés** (7 `golden_cpu` + 3 bisects + 2 `kernel_order_parity`, tous libérés par le checkpoint épinglé) |
| rapport de divergence libm (`continue-on-error`) | rc=0 (mesure, pas assertion) |
| stub CUDA (`ci/cuda-driver-stub.c`) + `claymore-memory --features cuda-driver --lib` (injections de fautes, pas de GPU) | rc=0 |
| `claymore-cuda --features cuda-driver --lib` (stub) | rc=0 |
| checks `cuda-driver` / `cuda-streaming` (cuda, memory, cli, model-gemma) | rc=0 ×4 |
| clippy `-p claymore-cuda -p claymore-memory -p claymore-cli --features cuda-driver` et `claymore-model-gemma --features cuda-streaming`, `-D warnings` | rc=0 ×2 |
| clippy workspace `-D warnings` | rc=0 |
| **MSRV `+1.86.0`** : `test --workspace --all-targets --locked` | rc=0 — **268 / 0 / 12** |
| MSRV : checks CUDA + clippy workspace et features, `-D warnings` | rc=0 ×7 |

Le jeu de tests ignoretés est exactement celui documenté ; `m10_reference`
(+2) passe dans la suite standard sans checkpoint, comme les deux bisects ne
s'exécutent que sur la machine qui porte le checkpoint épinglé.

## 9. Journaux bruts et empreintes

Sept journaux horodatés dans `docs/audits/`, empreintes SHA-256 consolidées
dans `m10-cpu-2026-11-25-logs.sha256` :

| fichier | contenu |
|---|---|
| `-01-reference-generation` | génération 455 artefacts, 85/85, 311/311 |
| `-02-chain-golden-cpu-pre-refactor` | chaîne `golden_cpu`, arbre HEAD |
| `-03-chain-golden-cpu-post-refactor` | chaîne `golden_cpu`, arbre refactorisé |
| `-04-bisect-encoder-entrees-dorees` | 210/210, 2,0 ULP couche 2 |
| `-05-bisect-vision-27-couches` | 0,2 pas BF16 (couche 7) |
| `-06-bisect-decodeur-30-couches` | 470+10/480, 0 structurel, 10,0 pas (couche 26) |
| `-07-ci-replica` | réplique complète des deux jobs CI, codes de retour en ligne |

## 10. Limites et hors périmètre

- La référence CPU est bit-reproductible entre comptes de threads (vérifié
  en campagne historique) ; la dérive CPU↔référence de la chaîne **complète**
  reste un suivi D2, inchangé par ce travail.
- Rien de GPU n'a été exécuté. Le protocole de campagne groupée RTX 5090 est
  prêt (`m10-rtx5090-grouped-protocol.md`) et **en attente d'autorisation
  utilisateur explicite**.
- PR #40 : toujours ouverte, non fusionnée (décision distincte). Issue #42 :
  mise à jour de traçabilité faite, l'issue reste ouverte jusqu'à validation
  hardware de la tête de finalisation.
