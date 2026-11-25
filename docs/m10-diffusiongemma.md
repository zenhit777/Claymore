# M10 — Adaptateur DiffusionGemma

## Objectif et frontière

Intégrer le checkpoint exact `google/diffusiongemma-26B-A4B-it` au runtime de diffusion M9, sans créer de runtime, scheduler ou moteur tensoriel parallèle. Le modèle audité est un encoder-decoder de block diffusion : l'encodeur causal construit le contexte/KV cache, le décodeur traite un canvas de 256 tokens avec attention bidirectionnelle et cross-attention, puis un raffinement par confiance valide certaines positions et re-bruite les autres.

Référence d'architecture : `docs/audits/m8-model-graph-requirements.md`. La révision exacte du dépôt modèle, de `config.json`, de l'index Safetensors et des fichiers de code upstream est épinglée (`CHECKPOINT_REVISION = cd40195554d3cfcb56f34b9adfb66efc790ed5b6`, code de référence transformers `671aa77576266f41f33f9c28e2d824bfd1b1b63c`).

## État après les campagnes des 23 et 24 novembre 2026

Validation matérielle complète et journaux bruts : [m10-rtx5090-hardware-validation.md](audits/m10-rtx5090-hardware-validation.md).

- Campagne du 23 novembre 2026 (sous-graphe audité) : SHA de code `ea115ab337d492d81ac50f69c09cd79e050b79a6`.
- **Campagne GPU finale du 24 novembre 2026** (créneau GPU final, 3/3 tâches exécutées) : SHA de code **`77e7665`** (« Fix the M10 CUDA graph against the pinned upstream semantics »). Détail : §16.3 du rapport hardware.

PR de la campagne : #40 (`feat/m10-diffusiongemma`, base `main`, **ouverte, non fusionnée**). PR de suivi M10.2 — M10.6 : #44 (`feat/m10-campaign-followup`, base `main`) — **fermée sans fusion le 25 novembre 2026** : son contenu (correctif de parité numérique) a été intégré à #40 par fast-forward, et #44 n'a donc aucun commit qui lui soit propre.

### Statut au 25 novembre 2026 (M10.4 — M10.5)

**M10 n'est pas validé.** Le différentiel de chaîne CPU ↔ référence eager
reste rouge et les tolérances n'ont **pas** été assouplies. Le correctif M10.4
rétablit la parité **interne** CUDA ↔ CPU ; le diagnostic M10.5 (§11 du rapport
de correctif) établit l'origine du rouge restant et ouvre une décision
d'architecture, pas un correctif.

| Validation | Exécutée où | Résultat |
|---|---|---|
| Primitives M10 sur carte, **arbre corrigé** | RTX 5090, 25/11, arbre de `fix/m10-numerics-divergence` — kernels, en-tête et PTX **byte-identiques** à la tête intégrée `ef1a1cf` (le commit suivant ne change que la mise en forme Rust) | **21/21 verts**, tous les écarts à `0e0` / `0` ULP, `scatter_add` inclus — journal `m10-parity-05` |
| Parité CUDA ↔ CPU sur la chaîne complète, **arbre corrigé** | même arbre | **`0,000000 / 0,000000`** (bit-exact) — journal `m10-parity-04` |
| Génération réelle, **arbre corrigé** | même arbre | **verte** — canvas 256, 256/256 positions acceptées, pic VRAM **24 831 Mio**, 449,74 s — journal `m10-parity-04` |
| Campagne M10.3 **avant** correctif | RTX 5090, 24/11, SHA `77e7665` | primitives 21/21 et génération verte avec les tolérances antérieures (`scatter_add` pas encore durci à l'exactitude) ; dérive de chaîne publiée `5,835938 / 0,711149`, 3 295,94 s — journal `m10-parity-00` |
| Reproductibilité des PTX versionnés | local (CPU) | **3/3 byte-identiques** à `nvcc 13.0 -fmad=false`, empreintes §8.2 revérifiées sur la branche intégrée |
| Suite workspace, features CUDA, clippy `-D warnings`, MSRV 1.86 | **local uniquement** — réplique des deux jobs (`m10-cuda-cpu-parity-fix.md` §7.1), rejouée après la réécriture de l'oracle | **PASS** : 325 passés / 0 échec / 26 ignorés (12 étapes) ; MSRV 264 / 0 / 10 (8 étapes). Jamais présenté comme la CI GitHub |
| Runs GitHub Actions de la PR #40 | **GitHub Actions, observés le 25/11** (API + journaux téléchargés) | **ROUGE, deux fois** : `ef1a1cf` (runs `45533723982` push 02:32:50Z, `45533731837` pull_request 02:32:54Z) puis `85f7772` (runs `45538595444` push 03:14:33Z, `45538600058` pull_request 03:14:35Z), 2 jobs à chaque fois (`Format, test, and lint` et `Minimum supported Rust 1.86`). Cause racine : le verrou `kernel_order_parity` comparait les ports à la libm **installée** (image du runner `ubuntu-26.04`) au lieu de glibc 2.39 ; la première correction, qui prétendait n'avoir trouvé que deux entrées divergentes, est **réfutée** — l'assertion s'arrête au premier écart, et deux autres sont apparues. Contrat remplacé : oracle épinglé par tables versionnées + comparaison à la libm installée sur les seuls hôtes-oracles. Baseline `9775cda` verte. Détail au **§7.4** du rapport, journaux `m10-parity-12` et `m10-parity-13`. Puis **VERT sur `88d3989`** : runs `45539679704` (push, 03:23:47Z → 03:24:16Z) et `45539689788` (pull_request, 03:23:52Z → 03:24:30Z), 4 jobs en succès, avec sur le runner `expf 0 / expm1f 3 242 / tanhf 890` divergences de la libm installée contre glibc 2.39 (4 132 sur 708 194 balayées) — la clôture chiffrée de la réfutation |
| Conformité CPU ↔ dorée eager | local | **rouge documenté** : encodeur `5,828125 / 0,754065` pour une porte `0,03 / 2e-3` ; trois tests `golden_cpu` rouges |

Aucune exécution GPU n'a été lancée **pendant** la campagne de validation du
correctif (25 novembre, sur l'arbre rapporté par le commit `9ccbe83`) : les
travaux M10.5 sont purement CPU et ne consomment ni VRAM ni carte. La tête
intégrée `ef1a1cf` **n'a pas** été rejouée sur carte.

La seule exécution GPU postérieure est la **campagne groupée du 25 novembre
2026** (autorisation explicite, 6/6 étapes vertes, §« Campagne GPU groupée »
ci-dessous), sur la tête de finalisation CPU.

### Livré et réellement exercé sur RTX 5090

- **Ingestion stricte et traçable** : lecteur JSON strict, SHA-256, lecteur Safetensors/ShardIndex borné avec validation de clés dupliquées, offsets, shapes, dtypes, couverture, shards ; ouverture `Checkpoint::open_pinned` qui refuse toute empreinte différente de la révision épinglée. Vérifié sur le checkpoint réel : 1047 tenseurs, 51 647 562 456 octets, 11 shards, couverture complète et contiguë.
- **Configuration complète validée** : 30 couches (6 globales), cachée 2816, 16 têtes / 8 KV (2 KV globaux), head_dim 256/512, 128 experts top-8, vocabulaire 262 144, canvas 256, soft-cap 30, RMS eps 1e-6, RoPE glissant theta 10 000 et global `proportional` theta 1e6 avec `partial_rotary_factor` 0.25, `tie_word_embeddings`, `use_bidirectional_attention = "vision"`, tour de vision 27 couches.
- **Primitives M10** (`claymore-tensor::diffusion` + kernels CUDA) : `Linear`, `RotaryEmbed`, `Attention`, `SoftCap`, `Scale`, `ScaleScalar`, `ScaleBias`, `GeluMulUp`, `GatherRows`, `ScatterAdd`, `EmbedGather`, `RowMask`, `Widen`, `RouterTopK` — référence CPU déterministe et **26 entrées PTX `sm_120`** (F32/BF16), dispatch exact sur couple (dtype, signature), **34 entrées** de capacité CUDA.
- **Découverte de campagne** : `layer_scalar` est une valeur apprise par couche (`0,068847656` en couche 0, bits `0x3d8d`), pas `1.0` ; la primitive `Scale` a été ajoutée pour l'appliquer explicitement comme dernière opération de couche, comme dans la référence épinglée.
- **Sous-graphe décodeur à poids réels exécuté sur la carte** : embedding réel → `input_layernorm` → `q_proj`/`k_proj`/`v_proj` → `q_norm`/`k_norm`/`v_norm` → RoPE → attention multi-tête GQA → `o_proj` → `post_attention_layernorm` → résiduel → `layer_scale` apprise, comparé à la référence CPU **étape par étape avec les mêmes bits d'entrée**.
- **Graphe complet en CPU et en CUDA** : encodeur 30 couches avec cache KV, tour de vision 27 couches (RoPE axial 2D, pooling, standardisation), décodeur canvas bidirectionnel, auto-conditionnement, MoE top-8 (routeur compatible ATen, gather, gate/up/gelu/down par expert, scatter-add), tête LM liée avec soft-cap final, **résidence bornée** (arène à emplacements stables, libération par portée, éviction LRU sous la marque haute VRAM).
- **Politique de génération** : `EntropyBoundSampler` (borne d'entropie 0,1, schedule linéaire `t_min 0,4` → `t_max 0,8`, arrêt stable/confident), canvas 256 positions, acceptation masquée par seuil de confiance, re-noising des positions non validées.
- **Adaptateur `DiffusionExecutor`** : `PrepareBlock`, `Denoise`, `ValidateBlock`, `CommitBlock` sur l'orchestrateur M9, erreurs typées, annulation aux frontières sûres, aucune ressource tensorielle ni poids dans les payloads scheduler, retour d'étape après complétion réelle.
- **Sûreté mémoire** : admission bornée par plafond dur avec refus pré-allocation, gardes `IN_FLIGHT` sur les lancements M10 (défaut trouvé et corrigé en campagne), quiescence de flux prouvée sur échec de lancement, verrouillé par 4 tests d'injection de fautes sur la branche M10.
- **Campagne GPU finale (24 novembre 2026)** : `hardware_m10_primitives` **21/21 verts**, `hardware_m10_streaming` **2/2 verts**, `hardware_m10_generation` — **génération réelle verte** (canvas 256, politique épinglée, **48 étapes** de débruitage, **256/256 positions** acceptées, pic VRAM **24 831 Mio**, **909 Gio** de poids uploadés, **605 669** lancements de kernel, 3 296 s) et **sémantique validée par étape** (couche 0 rejouée CUDA ↔ CPU : chaque étape à l'échelle de l'ULP BF16, **56/56 indices d'experts identiques**).

### Tolérances fixées avant campagne

| Opération | Tolérance |
|---|---|
| F32 | erreur absolue ≤ `1e-4` |
| BF16 élémentaire / réduction | ≤ 1 pas représentable (ULP) |
| BF16 attention | critère dérivé du modèle d'erreur : `max_abs ≤ 2^-8 · max\|v\|` |

Le critère d'attention est **dérivé**, pas ajusté : une probabilité de softmax arrondie en BF16 porte une erreur relative ≤ `2^-9`, donc la réduction de valeurs hérite d'au plus `2^-9 · max|v|`, avec un bit de marge. Le **compte d'ULP n'est pas une borne** pour une réduction dont la sortie peut s'annuler : un élément proche de zéro peut avoir un ULP arbitrairement grand alors que son erreur absolue reste bornée. L'ULP, le max, le RMS et le relatif sont **mesurés et publiés** (sous-graphe réel : `max_ulp=521` pour `max_abs=7,8125e-3` contre une borne de `5,46875e-2`).

La **dérive de bout en bout** de la chaîne device contre la chaîne de référence est mesurée et publiée séparément (jusqu'à `max_abs 2e0`, RMS `4,8e-3`, relatif `7,2e-3`) : un budget par opération n'est pas un budget de chaîne. Cette dérive est la raison pour laquelle la politique numérique de la chaîne complète (BF16 de bout en bout contre activations maîtresses FP32) reste une **décision ouverte**.

### Gates : état après le créneau GPU final du 24 novembre 2026

Les portes 1 à 6 sont **fermées** (code livré **et** exercé sur la carte). La porte 7 — **équivalence de chaîne** — reste **ouverte** et documentée.

1. exécuteur de streaming borné par couche (résidence/expert-cache sur 51,6 Go de charge utile) : **livré et exercé sur carte** (`stream.rs`, admissions par `CudaMemoryManager`, instrumenté ; `hardware_m10_streaming` **2/2 verts**) ;
2. primitives MoE : routeur top-8, gather/scatter d'experts : **livrées et exercées** (contrats, CPU de référence, kernels CUDA F32/BF16 ; routeur compatible ATen validé **210/210** contre `torch.topk`) ;
3. tour de vision (27 couches, 356 tenseurs, RoPE axial 2D) : **livrée et exercée** sur CPU (dorées eager) et sur carte (graphe CUDA complet) ;
4. auto-conditionnement : **livré et exercé** (embeddings souples → MLP gated → ajout résiduel, norme sans échelle) ;
5. cache KV encodeur et cross-attention décodeur → encodeur sur données réelles : **livré et exercé** (pré-remplissage du cache KV sur les `prompt_ids` dorés, puis décodage canvas sur ce contexte) ;
6. politique numérique de la chaîne complète : **décision documentée** (voir « Politique numérique de la chaîne ») ; le **différentiel de chaîne** est **rouge et documenté**, tolérances **conservées** ;
7. **génération réelle du checkpoint 26B sur RTX 5090 : EXÉCUTÉE et VERTE** — canvas 256, politique épinglée, **48 étapes** de débruitage, **256/256 positions** acceptées, pic VRAM **24 831 Mio** (< 26 Gio souple), **909 Gio** de poids uploadés, **605 669** lancements de kernel, 3 296 s. Le checkpoint de 51,6 Go n'est **jamais** chargé intégralement : les poids sont streamés couche par couche. Ce qui n'est **pas** revendiqué est l'**équivalence numérique de la chaîne** (point 6), pas la réalité de la génération.

## Sous-phases

### M10.1 — Contrats et ingestion (CPU) — livré

- Parsing strictement typé de la configuration, validation de dimensions et de la précision BF16. ✅
- Identité du checkpoint et provenance des révisions. ✅
- Inspection de l'index Safetensors, validation des noms, shapes, dtypes et shards sans charger les poids. ✅
- Contrats d'erreurs typés ; aucune conversion FP16/F32 implicite. ✅
- Canvas 256 positions, états masked/provisional/committed et transitions atomiques de confiance. ✅
- Tests de propriété des invariants, d'entrée malformée et de batch partiellement invalide. ✅ (corruption, troncature, overflow, index mal formé, shard manquant, poids inattendu)

### M10.2 — Primitives du graphe — livré

- Projections linéaires, RoPE/positions, masques, attention multi-tête bidirectionnelle, soft-capping, `Scale` (valeur apprise par couche), `ScaleScalar` (scalaire F32 du routeur), `ScaleBias`, MoE (`GeluMulUp`, `GatherRows`, `ScatterAdd`, `RouterTopK`), `EmbedGather`, `RowMask`, `Widen`. ✅
- Cross-attention et gestion du KV cache : **livrés**.
- Opérations de sélection/confiance, sampling et re-noising appartenant à l'adaptateur. ✅
- CPU reference déterministe pour les petites shapes ; contrats shape/layout/dtype explicites. ✅
- Aucune primitive n'est déclarée prête sans tests différentiels et tolérances documentées. ✅ (**21 tests matériels**, tolérances fixées avant campagne)

### M10.3 — Exécution et mémoire — livré

- Chargement BF16 shard par shard via le chemin mémoire Claymore : lectures bornées de charge utile validées ; **chargement résident streamé par couche livré et exercé sur carte**. ✅
- Durée de vie explicite des poids, KV cache, canvas et activations : **livrée** — arène à emplacements stables, libération par portée (couche, étape de débruitage), éviction LRU des poids residents sous la marque haute VRAM, cache KV encodeur porté côté hôte. ✅
- Exécution des étapes M9 par un exécuteur Gemma ; retour d'une étape uniquement après complétion sûre. ✅
- Propagation des erreurs et annulation aux frontières sûres. ✅

### M10.4 — Validation de cible — livré, avec limites explicites

- Tests CPU/CI, compilation CUDA optionnelle, puis exécution sur RTX 5090 : ✅ **campagne GPU finale du 24 novembre 2026** — `hardware_m10_primitives` **21/21 verts**, `hardware_m10_streaming` **2/2 verts**, `hardware_m10_generation` **génération réelle verte** + **sémantique de la couche 0 verte**, **différentiel de chaîne rouge et documenté** (tolérances conservées).
- Vérification des budgets : VRAM souple 26 Gio, plafond 28 Gio, réserve 4 Gio ; tous les poids/tampons passent par MemoryOrchestrator. ✅ (pic mesuré **24 831 Mio** pendant la génération complète, 25,4 Mio sur le sous-graphe réel)
- Comparaison numérique à une référence fiable et traçabilité du checkpoint, de la révision du code, du SHA Claymore, driver et versions outils. ✅
- Rapport matériel reproductible sous `docs/audits/`. ✅

## Critères de sortie

- Checkpoint/révisions explicitement épinglés et index validé. ✅
- Chemin encoder → KV cache → decoder bidirectionnel/cross-attention → confidence/refine exercé sur des données de test déterministes : **CPU de référence complet exercé sur le checkpoint réel** (golden venv épinglé) **et chemin CUDA exercé sur la carte** (`hardware_m10_generation` : génération complète de 48 étapes, 256/256 positions acceptées). ✅
- Adaptateur branché au DiffusionExecutor M9 sans runtime concurrent. ✅
- Précision BF16 explicite, aucune conversion/fallback silencieux. ✅
- Gestion mémoire bornée et erreurs fail-closed. ✅
- Référence CPU et exécution CUDA différentielles ; campagne RTX 5090 versionnée. ✅
- Guide d'intégration, limites et résultats réels documentés. ✅

## Campagne CPU du 23 novembre 2026 (M10.2 — M10.6)

### Contrats étendus (M10.2)

9 nouvelles opérations typées dans `claymore-tensor::contracts` : `Widen`
(BF16 → F32, mono-dtype), `RouterTopK` (F32 uniquement), `GeluMulUp`,
`GatherRows`, `ScatterAdd`, `EmbedGather`, `RowMask`, `ScaleBias` et
`ScaleScalar` (scalaire **F32**, un seul arrondi, sortie dans le dtype de
l'entrée). Les
slots d'indices sont validés par `validate()` (`F32`-encodés), chaque op
déclare sa `output_dtype`, **34 entrées** de capacité CPU, référence CPU
déterministe dans `diffusion.rs`, **26 entrées PTX** CUDA F32/BF16 (`sm_120`,
PTX régénéré), dispatch complet dans `launch_m10_op` et la table
`CUDA_OPS` (**34 entrées**, y compris les deux exceptions mono-dtype).

Le routeur reproduit l'algorithme d'ATen (`std::nth_element` introselect puis
`std::sort` sur les `k − 1` premiers, comparateur « valeur seule ») dans
`claymore-tensor/src/aten_topk.rs` **et** dans le kernel CUDA
`claymore_router_top_k_f32` : l'émulation est validée **210/210** contre
`torch.topk` sur les 210 lignes de routeur des 30 couches dorées, et le test
matériel `hw_m10_11` vérifie que CPU, CUDA et upstream choisissent **le même
ensemble et le même ordre** sur la ligne épinglée (égalités BF16 exactes experts
68/33 et 22/126, dont un groupe à cheval sur la frontière `k = 8`).

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
vision), canvas seed 42 de 16 positions (la composition du canvas est
indépendante de la longueur ; 16 rend la référence CPU complète
exécutable dans le créneau de la campagne). Golden capturés : `vision_last_hidden_direct`,
`embed_vision_output_direct`, `encoder_last_hidden_direct`,
`enc_kv_{0..29}_{keys,values}_direct`, `canvas0_direct`,
`decoder_last_hidden0_direct`, `decoder_logits0_direct`
(+ `moe0_router_logits`, `moe0_topk_weights`, `moe0_topk_indices`,
`moe0_expert_outputs` du run de génération). Le test `golden_cpu` compare
chaque stage Rust aux golden F32 avec tolérances dérivées documentées
(encodeur `max_abs ≤ 0,03`/RMS `2e-3`, vision `0,06`/`2e-3`, embed
`0,1`/`4e-3`, décodeur `0,12`/`4e-3`, logits `2,0`/`0,1`).

### Campagne GPU finale (24 novembre 2026)

Créneau GPU final autorisé explicitement, **une tâche à la fois**, VRAM estimée annoncée
avant chaque lancement, libération vérifiée après chaque tâche (relevé : `913 Mio`
occupés, aucun processus de calcul résiduel). Rejoué en entier sur le SHA `77e7665`.

| Tâche | Estimation VRAM | Résultat |
|---|---|---|
| `hardware_m10_primitives` | ≤ 2 Gio | **21/21 verts** |
| `hardware_m10_streaming` | ≤ 4 Gio | **2/2 verts** |
| `hardware_m10_generation` | ≤ 26 Gio (pic **24,25 Gio**) | **2 verts, 1 rouge documenté** |
- **Génération réelle : VERT** — canvas 256, politique épinglée, **48 étapes** de
  débruitage, **256/256 positions** acceptées, pic VRAM **24 831 Mio**, **909 Gio** de
  poids uploadés, **605 669** lancements de kernel, 3 296 s. Le checkpoint de 51,6 Go
  n'est jamais chargé intégralement. Les métriques sont **bit-identiques** au rejeu
  intermédiaire : la génération est déterministe.
- **Sémantique de la couche 0 : VERT exact** — couche 0 rejouée CUDA ↔ CPU : chaque
  étape à l'échelle de l'ULP BF16 et **56/56 indices d'experts identiques**.
- **Différentiel de chaîne : ROUGE, tolérances conservées** — `encoder_last_hidden`
  5,835938 / 0,711149 (CUDA ↔ CPU) contre 5,539062 / 0,763049 (CPU ↔ dorée eager) : la
  dérive CUDA ↔ CPU est du **même ordre** que la dérive CPU ↔ dorée, cause commune
  (ordre d'accumulation du GEMM BF16 tierce) et non défaut d'opération.

Sept défauts produit ont été trouvés par cette campagne et corrigés (offset de shard
relatif utilisé comme absolu, `intermediate_size` confondu avec
`moe_intermediate_size`, disposition de la sortie du routeur, entrée du routeur, table
d'embedding transposée dans l'auto-conditionnement, forme de la table transposée de la
tête LM liée, résidence non bornée). Détail : §16.3 du rapport hardware.

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
(encodeur, KV, décodeur, MoE, vision, auto-conditionnement, lm head), le graphe
CUDA complet et la boucle de diffusion. Les primitives M10 couvrent le graphe
complet. **La génération réelle sur carte a été exécutée le 24 novembre 2026**
(canvas 256, 48 étapes, 256/256 positions acceptées, pic VRAM 24 831 Mio) ; ce qui
n'est **pas** revendiqué est l'**équivalence numérique de la chaîne** contre la
référence eager (différentiel de chaîne rouge, tolérances conservées) et toute
performance ou tout débit.

Le matmul CUDA M8 est naïf (tuiles 16×16) et le kernel d'attention M10 limite la mémoire partagée dynamique à 48 KiB : **aucune performance ni débit n'est revendiqué**. Le re-streaming des slices d'experts à chaque passe de débruitage (909 Gio de trafic H2D pour la génération complète) est la conséquence du budget VRAM, pas une optimisation.

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

# Créneau GPU final : streaming borné, puis génération complète sur canvas 256
CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
  cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_streaming --locked -- \
  --ignored --test-threads=1 --nocapture
CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
  cargo test --release -p claymore-model-gemma --features cuda-streaming --test hardware_m10_generation --locked -- \
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


## État normatif après verdict architecte du 25 novembre 2026

Cette section prévaut sur les formulations historiques contradictoires conservées dans les journaux et rapports de campagnes antérieurs.

- **PR #40** : ouverte, non fusionnée ; verdict **VALIDÉE SOUS RÉSERVES** sur le SHA `24cb8dd8731fb767574204b3723cbaa2ae6030b0`. La CI GitHub Actions sur cette tête est verte (4/4 jobs).
- **Génération réelle** : déjà démontrée sur RTX 5090 le 24 novembre : canvas 256, 48 étapes, 256/256 positions acceptées, pic 24 831 Mio. Ne pas répéter cette revendication comme un travail encore à développer.
- **Contrat numérique** : appliquer D1/D2 de `audits/m10-architect-decisions.md`. Le bisect encodeur à entrée dorée a été rejoué dans la campagne CPU de finalisation (210/210 lignes, top-8 exact, pire 2,0 pas BF16, borne 4,0 ; journal `m10-cpu-2026-11-25-04-bisect-encoder-entrees-dorees.log.txt`). Les bisects vision et décodeur ont également été exécutés ; voir le rapport CPU pour les métriques et bornes. Les comparaisons de chaîne sont des mesures de suivi, pas des gates absolus. Les dorées ne sont ni modifiées ni ré-épinglées.
- **Travaux CPU-only** : terminés le 25/11 et consignés dans `audits/m10-cpu-campaign-2026-11-25.md`. Le générateur et les 455 artefacts de référence, les fixtures versionnées, les bisects vision/décodeur, les bornes D1 fixées après mesure, les tests exécutés sur le checkpoint et les journaux/hash sont livrés. Ne pas redévelopper ces éléments ; les chiffres et limites sont ceux du rapport.
- **Travail réservé à l'agent matériel** : **exécuté le 25/11** — campagne groupée 6/6 étapes vertes, puis campagne finale 9/9 étapes vertes sur le SHA `460c9ef` (driver, mémoire, tenseurs, injections, primitives M10, vérification intégrale du checkpoint, generation matérielle 3/3, et **deux passes du différentiel CUDA↔CPU des 30 couches du décodeur et des logits**, bit-exactes). Journaux bruts, empreintes SHA-256, métriques, budgets et preuve de libération mémoire : `audits/m10-rtx5090-final-decoder-logits-2026-11-25.md` et `audits/m10-final-*.log.txt`. Ne pas refaire la campagne déjà réussie sans motif. Toute modification `.cu`/`.cuh`/PTX impose régénération, provenance mise à jour et nouvelle campagne matérielle complète.
- **Pas de merge automatique** : la fusion de #40 reste une décision distincte.

### Finalisation CPU du 25 novembre 2026 — les cinq travaux CPU sont faits

Cette sous-section met à jour la précédente ; le rapport complet, les journaux
bruts et leurs empreintes sont dans `audits/m10-cpu-campaign-2026-11-25.md`.
Aucune utilisation du GPU.

1. **Procédure reproductible versionnée** : `tools/m10/` (générateur unique
   anti-écrasement avec portes de bit-exactité contre les dorées historiques,
   environment épinglé, manifeste d'empreintes) et `tools/m10/README.md` ;
   politique projet dans `docs/reproducibility.md`. Le générateur a produit
   455 artefacts : 85/85 contrôles internes, 311/311 portes dorées.
2. **Fixtures par couche engagées** : `crates/claymore-model-gemma/tests/data/m10_layers/`
   (41 Mio, engagement `ef329ca`), identité épinglée par SHA-256 du manifeste
   et validée en CI standard par `tests/m10_reference.rs` (sans checkpoint).
   Les tests contractuels ne dépendent plus que du checkpoint épinglé, via
   `CLAYMORE_M10_CHECKPOINT`.
3. **Bisects vision et décodeur ajoutés** avec les points d'injection partagés
   avec les chaînes (`vision_forward_layer_from_hidden`,
   `decoder_forward_traced_from_hidden` — extraction de corps partagé, sans
   duplication ; égalité bit à bit chaîne avant/après refactor :
   `diff` des tables `golden_cpu` identique). Égalité d'entrée vérifiée avant
   chaque comparaison ; le cache du décodeur est le cache K/V de référence de
   la couche, jamais un cache Claymore.
4. **Bornes D1 fixées après mesure complète** : encodeur 4,0 (2 × 2,0,
   inchangé, rejeu conforme `parity-11` : 210/210, 2,0 ULP, couche 2) ;
   vision **0,2 mesuré → borne 4,0** ; décodeur **10,0 mesuré → borne 20,0**
   (règle uniforme `max(2 × mesuré, 4,0)`). Sélection d'experts du décodeur :
   la frontière top-8/top-9 de la référence est à précision d'égalité (écarts
   mesurés : 0.0 en majorité, max ~5e-4) — une égalité stricte d'ensembles
   exigerait une réduction bit identique, ce que D1 interdit ; porte retenue =
   ensemble identique **ou** échange unique de frontière à égalité de scores
   de référence (470/480 + 10, 0 structurel).
5. **Logs, hashes, commandes et rapport publiés** : `audits/m10-cpu-campaign-2026-11-25.md`
   + six journaux bruts `m10-cpu-2026-11-25-*.log.txt` + `…-logs.sha256`.

### Campagne GPU groupée du 25 novembre 2026 — exécutée, six étapes vertes

Après autorisation explicite de l'utilisateur, le créneau groupé a été exécuté
sur `feat/m10-diffusiongemma` : **6/6 étapes vertes**
(`5/5` driver, `17/17` mémoire, `19/19` tenseurs, `13/13` injections de
fautes, `21/21` primitives M10 à `max_abs=0e0`, checkpoint vérifié
**bit-identique** au journal de référence). Détail, diagnostic et correctif :
`audits/m10-rtx5090-grouped-campaign-2026-11-25.md`.

La campagne a révélé et permis de corriger une **régression réelle** des
kernels CUDA M8 : `claymore_softmax_last_dim_f32` appelait la fonction
mono-thread `claymore_softmax_row` sans garde `threadIdx.x == 0`, les 256
threads du bloc se disputant le buffer de sortie (division multiple des
probabilités, `exp/S⁷` au lieu de `exp/S`). Le défaut était dormant pour M10
(le kernel d'attention garde déjà l'appel sous `if (tid == 0)`) mais réel
pour le backend tensor CUDA. Correctif + PTX régénérés et reproductibles,
manifeste de provenance mis à jour dans le même changement ; le test
`hw_m8_03_softmax_is_stable_for_large_logits` passe désormais à
`max_abs_diff=0e0` (contre 2,98e-8 le 22 novembre).

Le protocole groupé reste le document de référence pour une future session :
`audits/m10-rtx5090-grouped-protocol.md`. PR #40 toujours ouverte, non
fusionnée.


### État de clôture actualisé — 25 novembre 2026

La finalisation CPU et la campagne RTX 5090 groupée sont désormais exécutées ; voir les rapports `audits/m10-cpu-campaign-2026-11-25.md` et `audits/m10-rtx5090-grouped-campaign-2026-11-25.md`. Le bisect vision (0,2 pas BF16, borne 4) et le bisect décodeur (10,0 pas, borne 20 ; 470/480 + 10 échanges de frontière, 0 structurel) sont contractuels selon D1. La campagne matérielle est verte sur ses six étapes ; le décodeur réel de couche 0 est bit-exact.

**La réserve de mesure matérielle est désormais levée.** La campagne finale du 25/11 (SHA `460c9ef`, 9/9 étapes vertes) a produit les mesures CUDA↔CPU des **30 couches du décodeur et des logits** : **bit-exact** — `max_abs = 0,000000`, `rms = 0,000000`, **0,0 pas BF16** sur les 810 comparaisons d'étape (30 couches × 27 étapes), sur le routeur (scores bruts et poids, 30 couches) et sur les logits ; ensembles d'experts CUDA identiques à la référence sur 470/480 lignes, avec les **10 échanges de frontière** du bisect CPU et **zéro** divergence structurelle ; deux passes de tables **strictement identiques**. Rapport : `audits/m10-rtx5090-final-decoder-logits-2026-11-25.md` ; journaux bruts `audits/m10-final-<n>-2026-11-25_*.log.txt` ; empreintes `audits/m10-final-2026-11-25-logs.sha256`.

La dérive de chaîne contre la référence eager reste un suivi D2, non une porte de clôture. Les décisions D1/D2, les dorées et leurs empreintes ne doivent pas être modifiées. Le **resserrement** des portes CUDA↔CPU à partir de ces mesures (valeurs candidates : 0,0 pas, routeur et logits à 0,000000) est une **décision d'architecte** : il n'est pas appliqué ici, car une égalité bit stricte serait plus forte que ce que D1 énonce. Le **merge de la PR #40** reste une décision distincte.
