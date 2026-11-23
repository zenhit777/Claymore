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

1. exécuteur de streaming borné par couche (résidence/expert-cache sur 51,6 Go de charge utile) : **code livré** (`stream.rs`), admissions par `CudaMemoryManager`, instrumenté, non encore exercé sur carte ;
2. primitives MoE : routeur top-8, gather/scatter d'experts : **livrées** (contrats, CPU de référence, 12 kernels CUDA F32/BF16) ;
3. tour de vision (27 couches, 356 tenseurs, RoPE axial 2D) : **livrée côté CPU de référence**, kernels CUDA des primitives shared ;
4. auto-conditionnement : **livré** (embeddings souples → MLP gated → ajout résiduel, norme sans échelle) ;
5. cache KV encodeur et cross-attention décodeur → encodeur sur données réelles : **livré côté CPU de référence** (validation golden en cours), chemin CUDA à exercer sur carte ;
6. politique numérique de la chaîne complète : **décision documentée** (voir « Politique numérique de la chaîne ») ;
7. **chargement complet et génération réelle du checkpoint 26B sur RTX 5090 : le gate de génération reste OUVERT, aucune génération réelle n'est revendiquée.**

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
- Chemin encoder → KV cache → decoder bidirectionnel/cross-attention → confidence/refine exercé sur des données de test déterministes : **CPU de référence complet exercé sur le checkpoint réel** (golden venv épinglé) ; chemin CUDA sur carte : **à exercer dans le créneau GPU final**.
- Adaptateur branché au DiffusionExecutor M9 sans runtime concurrent. ✅
- Précision BF16 explicite, aucune conversion/fallback silencieux. ✅
- Gestion mémoire bornée et erreurs fail-closed. ✅
- Référence CPU et exécution CUDA différentielles ; campagne RTX 5090 versionnée. ✅
- Guide d'intégration, limites et résultats réels documentés. ✅

## Campagne CPU du 23 novembre 2026 (M10.2 — M10.6)

### Contrats étendus (M10.2)

8 nouvelles opérations typées dans `claymore-tensor::contracts` : `Widen`
(BF16 → F32, mono-dtype), `RouterTopK` (F32 uniquement), `GeluMulUp`,
`GatherRows`, `ScatterAdd`, `EmbedGather`, `RowMask`, `ScaleBias`. Les
slots d'indices sont validés par `validate()` (`F32`-encodés), chaque op
déclare sa `output_dtype`, 32 entrées de capacité CPU, référence CPU
déterministe dans `diffusion.rs`, 12 kernels CUDA F32/BF16 (`sm_120`,
PTX régénéré), dispatch complet dans `launch_m10_op` et la table
`CUDA_OPS` (32 entrées, y compris les deux exceptions mono-dtype).

Le masque additif d'attention est un slot **F32 typé** (les clés bloquées
valent `-inf` en FP32, indépendamment du dtype d'activations) ; le kernel
BF16 lit le masque F32 et l'additionne aux scores F32.

Le `ScatterAdd` BF16 utilise `atomicAdd` bf16 avec arrondi par étape, ce
qui reproduit le comportement de `index_add_` bf16 de la référence épinglée.
L'ordre d'accumulation atomique n'est pas déterministe : la tolérance de
différentiel MoE est une **borne dérivée du modèle d'erreur** (bornes
`max|v|` par expert × poids top-k), pas une borne ULP.

### Exécuteur de streaming (M10.3)

`claymore-model-gemma::stream::WeightStream` : admission de chaque tenseur
uniquement via `CudaMemoryManager` (`allocate_pinned`/`allocate_device`/
`copy_h2d`/`release`), staging par chunks de 8 Mio, zéro allocation CUDA
directe, zéro swap système comme tier, éviction LRU avant admission
(experts evincés en dernier), cache de résidence avec `record_reload_cost`,
annulation (`cancel`) qui libère dans l'ordre LRU et laisse les tenseurs
`IN_FLIGHT` sous la garde du manager, et instrumentation RAM/VRAM/transferts
(`StreamMetrics` : pics VRAM/pinned, octets lus disque, chargements,
cache-hit, évictions). Les plans de noms (`decoder_layer_names`,
`vision_layer_names`, `vision_prefix_names`, `text_prefix_names`) couvrent
la couche 5 sans `v_proj` (géométrie globale lue depuis `layer_types`).

### Graphe CPU complet (M10.3/M10.4)

`claymore-model-gemma::graph` + `fullgraph` reproduisent l'upstream épinglé
op par op : encodeur causal (KV append), décodeur canvas (cache read-only,
positions `prefix..prefix+256`, masques glissants/causal), MoE (routeur
RMSNorm sans échelle × scale × root-scale → softmax F32 → top-8 trié
deterministement → poids normalisés × `per_expert_scale` → gather par expert
→ `gelu(gate)*up` → down → scatter-add pondéré), MLP dense, layer_scalar,
tour de vision (patch embedder `2(x-0.5)`, table de positions 2×10240,
RoPE axial 2D recomposé `[h,h,w,w]`, theta 100, pooling k² × sqrt(hidden),
standardisation `std_bias`/`std_scale`), auto-conditionnement, lm head tied
avec softcap final `/cap → tanh → ×cap`.

Le encodeur textuel partage les poids du décodeur (l'index ne contient
`layer_scalar` que sous `model.encoder.language_model`, les 30 couches
textuelles vivent sous `model.decoder.layers` — vérifié par l'extraction
d'index, documenté dans `fullgraph.rs`).

### Boucle de génération (M10.6)

`generation_loop.rs` : EntropyBoundSampler (entropie par position à la
température du schedule linéaire `t 0,4 → 0,8`, acceptation sous la borne
0,1, re-noising RNG des positions non validées, arrêt par confiance
(mean entropy < 0,005 tout accepté) ou stabilité (canvas accepté identique
`stability_threshold` pas), max 48 pas). 6 tests unitaires verts.

### Références dorées CPU (venv épinglé)

Provenance : venv `transformers 5.24.0.dev0 @ 671aa77576266f41f33f9c28e2d824bfd1b1b63c`
+ `torch 2.15.0+cpu` + `accelerate 1.16.0`, CPU-only, checkpoint épinglé
révision `cd40195554d3cfcb56f34b9adfb66efc790ed5b6`.
Prompt fixe `"What is the capital of France?"` (7 tokens), image
déterministe (seed 1234, 48×48 → 9 patches → pooling 3 → 1 token
vision), canvas seed 42. Golden capturés : `vision_last_hidden_direct`,
`embed_vision_output_direct`, `encoder_last_hidden_direct`,
`enc_kv_{0..29}_{keys,values}_direct`, `canvas0_direct`,
`decoder_last_hidden0_direct`, `decoder_logits0_direct`
(+ `moe0_router_logits`, `moe0_topk_weights`, `moe0_topk_indices`,
`moe0_expert_outputs` du run de génération). Le test `golden_cpu` compare
chaque stage Rust aux golden F32 avec tolérances dérivées documentées
(encodeur `max_abs ≤ 0,03`/RMS `2e-3`, vision `0,06`/`2e-3`, embed
`0,1`/`4e-3`, décodeur `0,12`/`4e-3`, logits `2,0`/`0,1`).

### Politique numérique de la chaîne

- Activations maîtresses BF16, accumulation en F32 dans les kernels et la
  référence CPU (softmax, scores d'attention, routeur), comme l'upstream.
- Masques d'attention F32 typés (`-inf`), jamais BF16.
- Scores et top-k du routeur en F32 (`RouterTopK` mono-F32).
- Scatter-add MoE atomique : tolérance dérivée, order non déterministe
  assumé et documenté.
- Softcap final et division par `cap` en F32 (logits élargis par `Widen`).
- `layer_scalar`, `router.scale`, `per_expert_scale` appliqués par
  `ScaleBias`/`Scale` BF16 explicites, jamais par plagiats FP32.
- La dérive de chaîne encodeur → KV → décodeur → logits est mesurée et
  publiée par le test golden : elle n'est pas absorbée par les tolérances
  par opération.


## Limites actuelles

Le crate `claymore-model-gemma` contient l'ingestion épinglée, la configuration
validée, la politique de génération, l'état canvas, l'adaptateur
`DiffusionExecutor`, l'exécuteur de streaming, le graphe CPU complet
(encodeur, KV, décodeur, MoE, vision, auto-conditionnement, lm head) et la
boucle de diffusion. Les primitives M10 couvrent le graphe complet. **Aucune
génération de texte réelle sur carte n'est revendiquée tant que le créneau
GPU final n'a pas été exécuté.**

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

# Références dorées CPU (venv épinglé, profil release, CPU only)
cargo test --release -p claymore-model-gemma --test golden_cpu --locked -- \
  --ignored --test-threads=1 --nocapture
```

Le PTX M10 est versionné et régénéré hors build (`nvcc` n'est jamais appelé par `cargo`) :

```bash
nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
  -o crates/claymore-cuda/src/kernels/m10.ptx \
  crates/claymore-cuda/kernels/m10.cu
```

Les tests matériels de la suite M10 nécessitent le checkpoint épinglé ; sans `CLAYMORE_M10_CHECKPOINT`, le test à poids réels s'arrête explicitement sur un saut déclaré plutôt que de simuler un succès.
