# M10 — Adaptateur DiffusionGemma

## Objectif et frontière

Intégrer le checkpoint exact `google/diffusiongemma-26B-A4B-it` au runtime de diffusion M9, sans créer de runtime, scheduler ou moteur tensoriel parallèle. Le modèle audité est un encoder-decoder de block diffusion : l'encodeur causal construit le contexte/KV cache, le décodeur traite un canvas de 256 tokens avec attention bidirectionnelle et cross-attention, puis un raffinement par confiance valide certaines positions et re-bruite les autres.

Référence d'architecture : `docs/audits/m8-model-graph-requirements.md`. La révision exacte du dépôt modèle, de `config.json`, de l'index Safetensors et des fichiers de code upstream est épinglée (`CHECKPOINT_REVISION = cd40195554d3cfcb56f34b9adfb66efc790ed5b6`, code de référence transformers `671aa77576266f41f33f9c28e2d824bfd1b1b63c`).

## État après la campagne du 23 novembre 2026

Validation matérielle complète et journaux bruts : [m10-rtx5090-hardware-validation.md](audits/m10-rtx5090-hardware-validation.md), au SHA de code `ea115ab337d492d81ac50f69c09cd79e050b79a6`.

### Livré et réellement exercé sur RTX 5090

- **Ingestion stricte et traçable** : lecteur JSON strict, SHA-256, lecteur Safetensors/ShardIndex borné avec validation de clés dupliquées, offsets, shapes, dtypes, couverture, shards ; ouverture `Checkpoint::open_pinned` qui refuse toute empreinte différente de la révision épinglée. Vérifié sur le checkpoint réel : 1047 tenseurs, 51 647 562 456 octets, 11 shards, couverture complète et contiguë.
- **Configuration complète validée** : 30 couches (6 globales), cachée 2816, 16 têtes / 8 KV (2 KV globaux), head_dim 256/512, 128 experts top-8, vocabulaire 262 144, canvas 256, soft-cap 30, RMS eps 1e-6, RoPE glissant theta 10 000 et global `proportional` theta 1e6 avec `partial_rotary_factor` 0.25, `tie_word_embeddings`, `use_bidirectional_attention = "vision"`, tour de vision 27 couches.
- **Primitives M10** (`claymore-tensor::diffusion` + kernels CUDA) : `Linear`, `RotaryEmbed`, `Attention`, `SoftCap`, `Scale` — référence CPU déterministe et 10 kernels F32/BF16 `sm_120`, dispatch exact sur couple (dtype, signature), 18 entrées de capacité.
- **Découverte de campagne** : `layer_scalar` est une valeur apprise par couche (`0,068847656` en couche 0, bits `0x3d8d`), pas `1.0` ; la primitive `Scale` a été ajoutée pour l'appliquer explicitement comme dernière opération de couche, comme dans la référence épinglée.
- **Sous-graphe décodeur à poids réels exécuté sur la carte** : embedding réel → `input_layernorm` → `q_proj`/`k_proj`/`v_proj` → `q_norm`/`k_norm`/`v_norm` → RoPE → attention multi-tête GQA → `o_proj` → `post_attention_layernorm` → résiduel → `layer_scale` apprise, comparé à la référence CPU **étape par étape avec les mêmes bits d'entrée**.
- **Politique de génération** : `EntropyBoundSampler` (borne d'entropie 0,1, schedule linéaire `t_min 0,4` → `t_max 0,8`, arrêt stable/confident), canvas 256 positions, acceptation masquée par seuil de confiance, re-noising des positions non validées.
- **Adaptateur `DiffusionExecutor`** : `PrepareBlock`, `Denoise`, `ValidateBlock`, `CommitBlock` sur l'orchestrateur M9, erreurs typées, annulation aux frontières sûres, aucune ressource tensorielle ni poids dans les payloads scheduler, retour d'étape après complétion réelle.
- **Sûreté mémoire** : admission bornée par plafond dur avec refus pré-allocation, gardes `IN_FLIGHT` sur les lancements M10 (défaut trouvé et corrigé en campagne), quiescence de flux prouvée sur échec de lancement, verrouillé par 4 tests d'injection de fautes sur la branche M10.

### Tolérances fixées avant campagne

| Opération | Tolérance |
|---|---|
| F32 | erreur absolue ≤ `1e-4` |
| BF16 élémentaire / réduction | ≤ 1 pas représentable (ULP) |
| BF16 attention | critère dérivé du modèle d'erreur : `max_abs ≤ 2^-8 · max\|v\|` |

Le critère d'attention est **dérivé**, pas ajusté : une probabilité de softmax arrondie en BF16 porte une erreur relative ≤ `2^-9`, donc la réduction de valeurs hérite d'au plus `2^-9 · max|v|`, avec un bit de marge. Le **compte d'ULP n'est pas une borne** pour une réduction dont la sortie peut s'annuler : un élément proche de zéro peut avoir un ULP arbitrairement grand alors que son erreur absolue reste bornée. L'ULP, le max, le RMS et le relatif sont **mesurés et publiés** (sous-graphe réel : `max_ulp=521` pour `max_abs=7,8125e-3` contre une borne de `5,46875e-2`).

La **dérive de bout en bout** de la chaîne device contre la chaîne de référence est mesurée et publiée séparément (jusqu'à `max_abs 2e0`, RMS `4,8e-3`, relatif `7,2e-3`) : un budget par opération n'est pas un budget de chaîne. Cette dérive est la raison pour laquelle la politique numérique de la chaîne complète (BF16 de bout en bout contre activations maîtresses FP32) reste une **décision ouverte**.

### Gates ouvertes (explicitement non revendiquées)

Ce sont des **portes de code**, pas des limites de capacité mémoire : le plafond dur de 28 Gio est un **seuil d'admission**, les poids vivent en VRAM **et** RAM et sont streamés couche par couche.

1. exécuteur de streaming borné par couche (résidence/expert-cache sur 51,6 Go de charge utile) ;
2. primitives MoE : routeur top-8, gather/scatter d'experts ;
3. tour de vision (27 couches, 355 tenseurs) ;
4. auto-conditionnement ;
5. cache KV encodeur et cross-attention décodeur → encodeur sur données réelles ;
6. politique numérique de la chaîne complète ;
7. **chargement complet et génération réelle du checkpoint 26B : le gate de génération reste OUVERT, aucune génération réelle n'est revendiquée.**

## Sous-phases

### M10.1 — Contrats et ingestion (CPU) — livré

- Parsing strictement typé de la configuration, validation de dimensions et de la précision BF16. ✅
- Identité du checkpoint et provenance des révisions. ✅
- Inspection de l'index Safetensors, validation des noms, shapes, dtypes et shards sans charger les poids. ✅
- Contrats d'erreurs typés ; aucune conversion FP16/F32 implicite. ✅
- Canvas 256 positions, états masked/provisional/committed et transitions atomiques de confiance. ✅
- Tests de propriété des invariants, d'entrée malformée et de batch partiellement invalide. ✅ (corruption, troncature, overflow, index mal formé, shard manquant, poids inattendu)

### M10.2 — Primitives du graphe — livré pour le sous-graphe audité

- Projections linéaires, RoPE/positions, masques, attention multi-tête bidirectionnelle, soft-capping, `Scale` (valeur apprise par couche). ✅
- Cross-attention et gestion du KV cache : **non livrés**.
- Opérations de sélection/confiance, sampling et re-noising appartenant à l'adaptateur. ✅
- CPU reference déterministe pour les petites shapes ; contrats shape/layout/dtype explicites. ✅
- Aucune primitive n'est déclarée prête sans tests différentiels et tolérances documentées. ✅ (12 tests matériels, tolérances fixées avant campagne)

### M10.3 — Exécution et mémoire — partiel

- Chargement BF16 shard par shard via le chemin mémoire Claymore : lectures bornées de charge utile validées ; **chargement résident streamé par couche non livré**.
- Durée de vie explicite des poids, KV cache, canvas et activations : poids et activations du sous-graphe audité ✅ ; **KV cache non livré**.
- Exécution des étapes M9 par un exécuteur Gemma ; retour d'une étape uniquement après complétion sûre. ✅
- Propagation des erreurs et annulation aux frontières sûres. ✅

### M10.4 — Validation de cible — livré, avec limites explicites

- Tests CPU/CI, compilation CUDA optionnelle, puis exécution sur RTX 5090 : ✅ 66 tests matériels verts.
- Vérification des budgets : VRAM souple 26 Gio, plafond 28 Gio, réserve 4 Gio ; tous les poids/tampons passent par MemoryOrchestrator. ✅ (pic mesuré 25,4 Mio sur le sous-graphe réel)
- Comparaison numérique à une référence fiable et traçabilité du checkpoint, de la révision du code, du SHA Claymore, driver et versions outils. ✅
- Rapport matériel reproductible sous `docs/audits/`. ✅

## Critères de sortie

- Checkpoint/révisions explicitement épinglés et index validé. ✅
- Chemin encoder → KV cache → decoder bidirectionnel/cross-attention → confidence/refine exercé sur des données de test déterministes : **partiel** — sous-graphe décodeur réel ✅, encodeur/KV cache/cross-attention ❌.
- Adaptateur branché au DiffusionExecutor M9 sans runtime concurrent. ✅
- Précision BF16 explicite, aucune conversion/fallback silencieux. ✅
- Gestion mémoire bornée et erreurs fail-closed. ✅
- Référence CPU et exécution CUDA différentielles ; campagne RTX 5090 versionnée. ✅
- Guide d'intégration, limites et résultats réels documentés. ✅

## Limites actuelles

Le crate `claymore-model-gemma` contient l'ingestion épinglée, la configuration validée, la politique de génération, l'état canvas et l'adaptateur `DiffusionExecutor` ; il ne contient **pas** le graphe complet du modèle. Les primitives M10 couvrent le sous-graphe audité ; le routage MoE, la tour de vision, l'auto-conditionnement et le chemin KV encodeur restent à construire. **Aucune génération de texte réelle n'est revendiquée.**

Le matmul CUDA M8 est naïf (tuiles 16×16) et le kernel d'attention M10 limite la mémoire partagée dynamique à 48 KiB : **aucune performance ni débit n'est revendiqué**.

## Guide de reproduction

```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CI sans GPU
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo test -p claymore-model-gemma --all-targets --locked

# Campagne matérielle (RTX 5090, driver réel)
cargo test -p claymore-cuda --features cuda-driver --test hardware_cuda_driver --locked -- --ignored --test-threads=1 --nocapture
cargo test -p claymore-memory --features cuda-driver --lib --locked -- --ignored --test-threads=1 --nocapture
cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_memory --locked -- --ignored --test-threads=1 --nocapture
cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_tensor --locked -- --ignored --test-threads=1 --nocapture
CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
  cargo test --release -p claymore-memory --features cuda-driver --test hardware_m10_primitives --locked -- \
  --ignored --test-threads=1 --nocapture

# Vérification complète du checkpoint réel (CPU, flux borné 8 Mio)
cargo run --release -p claymore-model-gemma --bin verify_checkpoint -- --root /tmp/opencode/m10/checkpoint
```

Le PTX M10 est versionné et régénéré hors build (`nvcc` n'est jamais appelé par `cargo`) :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m10.ptx \
  crates/claymore-cuda/kernels/m10.cu
```

Les tests matériels de la suite M10 nécessitent le checkpoint épinglé ; sans `CLAYMORE_M10_CHECKPOINT`, le test à poids réels s'arrête explicitement sur un saut déclaré plutôt que de simuler un succès.
