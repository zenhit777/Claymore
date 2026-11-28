# M11 — Adaptateur Clover-35B-A3B-Preview

## Statut

M11 est découpé en cinq incréments (#48 à #52). **M11.1 a été fusionné dans `main` via la PR #53** (merge commit `362be819`, registre A2 mis à jour dans `cb01a77`). La décision A2 est acceptée avec amendements ; A2-B reste volontairement ouverte pour M11.4 (sémantique EOS et traitement des 243 IDs excédentaires), comme consigné dans `docs/audits/m11-architect-decisions.md`. **M11.2 a été fusionné dans `main` via la PR #54** : références CPU GDN, attention, RoPE et contrats numériques dans `claymore-tensor`, sans moteur parallèle. **M11.3 est en revue finale Tech-Lead dans la PR #55** ; l'audit du code est terminé sans défaut bloquant propre à M11.3 identifié, l'approbation architecte reste attendue. **M11.4 est en revue Tech-Lead dans la PR #56 (Draft, branche `feat/m11-4-clover-lowconfidence`, empilée sur #55)**. État vérifié le 28/11 : #55 HEAD `6677cc79d6b12933480463778759c991ab1636a7`, Rust CI #46152385453 verte (stable + MSRV 1.86) ; une revue COMMENT Tech-Lead non approbative publiée. #56 code review baseline `dce13772d024b4ad06e561093fbbe449fbe88638`, Rust CI #46152315572 verte (stable + MSRV 1.86) ; avis Tech-Lead détaillé publié, avec blocages P1/P2/P4/P6/P8/P10 et preuve pinned multi-blocs réelle encore à exécuter. #56 reste Draft. Le dossier [`docs/audits/m11.4-review-readiness-2026-11-28.md`](audits/m11.4-review-readiness-2026-11-28.md) fait l'inventaire exigences #51 → implémentation → tests → preuves, corrections livrées, défauts remontés non corrigés et points d'arbitrage architecte.

- PR #53 : `feat/m11-clover-foundation` → `main`, fusionnée le 2026-11-27.
- PR #54 : `feat/m11-clover-cpu-primitives` → `main`, fusionnée le 2026-11-27.
- PR #55 : `feat/m11-3-clover-moe` → `main`, ouverte ; CI et revue encore obligatoires.
- Issues de suivi M11.1 à M11.3 : #48 à #50.
- Les tests M11.3 utilisent le backend CPU simulé ; aucun contexte CUDA ni aucune VRAM physique n'est initialisé.

## Référence upstream

- Modèle : `Diffuspace/Clover-35B-A3B-Preview`.
- Révision immuable : `4b0402b65e523d0183a14c431955438714cb16ff`.
- Licence : Apache-2.0 ; conserver `LICENSE` et `ATTRIBUTIONS.md`, y compris les attributions Qwen / Hugging Face applicables.
- Rapport technique : [Clover](https://arxiv.org/abs/2611.24636).
- Manifeste : `docs/audits/clover-upstream-manifest.md`.

L'audit réel publié rapporte 13/13 empreintes non-Xet et 28/28 empreintes Safetensors concordantes avec la table Xet. Il relève 813 tenseurs, 34 661 350 848 paramètres et 69 322 701 696 octets de payload. Ces preuves ont été **rejouées en post-correctif** le 2026-11-26 sur le HEAD `443b1c4d`, puis **une troisième fois sur le HEAD de publication `580a8984`** : hachage intégral Claymore des 28 artefacts (`artifact_sha256=verified`, 55 min 08,79 s puis 55 min 11,51 s, ~20,9 Mo/s, `payloads_read=false`), hachage indépendant `sha256sum` 28/28 sur les trois passes, gates négatives T0–T3 et porte BF16 de sidecar (T4) reproduites, plan de chargement exercé (28 shards / 813 tenseurs / 8 845 chunks, 8 MiB maximum, déterministe, byte-identique entre passes postfix et finale), parité réelle de tokenisation directe ↔ rendu retokenisé sur les 9 cas positifs et fixtures byte-identiques entre ces deux passes. Détail et journaux : `docs/audits/m11.1-final-replay-2026-11-26.md`, `docs/audits/m11.1-cpu-evidence-refresh-2026-11-26.md` et l'addendum `docs/audits/m11.1-audit-corrections-2026-11-26.md` § 10–11.

Ne jamais utiliser `main` comme source reproductible pour le checkpoint et ne pas déclarer Clover pris en charge tant que les gates de provenance, de schéma, de tokenisation et d'exécution n'ont pas été satisfaites.

## Architecture publiée à prendre en charge

| Élément | Valeur de référence |
|---|---:|
| Paramètres totaux | environ 35B |
| Paramètres actifs/token | environ 3B |
| Couches | 40 |
| Motif | 3 × GDN puis 1 × full attention, répété |
| Hidden size | 2048 |
| Experts routés | 256 |
| Top-k par token | 8 |
| Expert partagé | 1 par couche |
| Attention complète | 16 têtes Q, 2 têtes KV, dimension 256 |
| GDN | 16 têtes K, 32 têtes V, dimensions 128 |
| Convolution GDN | largeur 4 |
| RoPE | partiel, facteur 0.25, theta 10 000 000 |
| Poids stockés | BF16, sidecars GDN inclus |
| État SSM/GDN à l'exécution | FP32 |
| Contexte maximal | 131 072 |
| Bloc par défaut | 32 |
| Coreset | activé, masse top-p 0.9 |
| Décodage | LowConfidence, seuil 0.95 |

La génération est une diffusion par blocs : le préfixe validé reste causal ; le bloc actif est bidirectionnel. GDN combine mémoire récurrente historique et attention intra-bloc, avec une branche de convolution de contexte droit. Ce calcul ne doit pas être remplacé par une attention standard ni par une boucle autoregressive token-par-token.

Le coreset agrège les scores top-8 sur le bloc et sélectionne un sous-ensemble d'experts couvrant une fraction de masse. Il n'altère ni les poids stockés, ni le top-8 par token, ni le sens du routeur.

Les valeurs EOS/PAD ne sont pas interchangeables entre `tokenizer.json`, `config.json` et `generation_config.json`. La décision A2 proposée est consignée dans `docs/audits/m11-architect-decisions.md` et attend explicitement l'architecte.

## Découpage

- #48 — audit upstream, config typée, inventaire checkpoint et provenance.
- #49 — références CPU et contrats numériques GDN/full attention/RoPE/SiLU.
- #50 — routage MoE, coreset, streaming et résidence d'experts.
- #51 — graphe complet et LowConfidence sur le `DiffusionExecutor` commun.
- #52 — kernels CUDA, différentiel et campagne RTX 5090.

## M11.1 — fondation livrée dans la PR #53

`crates/claymore-model-clover/src/config.rs` introduit le contrat typé de tous les champs du `config.json` de release, y compris `auto_map` et `initializer_range`, avec validation pinned et désérialisation fail-closed.

`claymore-checkpoint` mutualise les primitives indépendantes des modèles : JSON strict, Safetensors et SHA-256 streaming. Les adaptateurs Gemma et Clover réutilisent ces primitives sans dépendre l'un de l'autre et conservent leurs propres inventaires et sémantiques.

Le socle Clover comprend :
- un parseur strict de `model.safetensors.index.json` ;
- l'inspection header-only des 26 shards de modèle et des deux sidecars GDN ;
- le contrôle des noms, formes, offsets, tailles, couverture, dtypes BF16 et du schéma pinned des 813 tenseurs ;
- un rejet explicite des sidecars GDN F32 ;
- `CloverCheckpointInventory::load_plan()`, qui dérive un plan déterministe par shard/tenseur/chunk à partir des offsets validés, avec des chunks de 8 MiB maximum ;
- la commande CPU-only `inspect_checkpoint`, qui expose l'inventaire et les compteurs du plan, avec option `--verify-sha256` ;
- les scripts d'audit tokenizer/template, les fixtures et les démonstrations fail-closed T0–T3.

Le plan de chargement est **metadata-only** : il ne lit pas les payloads, n'alloue aucun tenseur, ne réserve aucune mémoire et n'initialise aucun device. Il n'est pas encore un ordonnanceur de lecture, un transfert asynchrone ou une intégration au MemoryOrchestrator. Ces responsabilités appartiennent aux incréments suivants.

## M11.2 — références CPU et contrats numériques (fusionnée, PR #54)

La PR #54 cible `main` après l'intégration de #53. Elle est prête pour la revue finale ; la CI GitHub stable et MSRV 1.86 est verte sur le HEAD source `c737e02` et sur le HEAD documentaire ultérieur `d5a1de2` (2/2 jobs sur les événements push et pull_request).

Les primitives communes déjà présentes dans M8/M10 sont réutilisées : projections linéaires, RMSNorm, attention GQA, RoPE, masques, softmax, top-k et gather/scatter. Les ajouts Clover restent dans `claymore-tensor` :

- activations SiLU/sigmoid, multiplication élémentaire et variantes RMSNorm zero-centered / RMSNorm-SiLU-gate ;
- références GDN : gates `A_log` / `dt_bias`, transition récurrente FP32, convolution causale, convolution de preview avec right-context indépendant, preview softmax bidirectionnel et bucket historique ;
- masque full-attention causal-prefix / bloc bidirectionnel en preview et causal au commit, avec lignes de requêtes invalides entièrement masquées dans les deux modes ;
- gate de sortie full-attention et voie RoPE partielle qui ne tourne que les 25 % initiaux du head Clover.

Le crate `claymore-model-clover` n'introduit pas de dépendance vers le crate tensor : il consommera ces primitives lorsqu'il assemblera le graphe. Les états de preview restent en lecture seule ; seul le commit causal avance les états persistants.

**Références différentielles ajoutées :** `tools/m11/gen_cpu_reference.py` transcrit les équations des fonctions Clover épinglées et produit 17 tenseurs dorés (1 356 valeurs) à partir de petits tenseurs PyTorch CPU BF16/F32. Les fixtures sont enregistrées dans `clover_gdn_upstream_fixtures.rs`, avec provenance dans `docs/audits/m11.2-cpu-reference-manifest.json` : révision et SHA-256 du source, environnements de génération et de vérification séparés, seed, plateforme et observation `cuda_used=false`. Les tests couvrent GDN commit/preview, convolutions causale/right-context, gates, RMSNorm, RoPE partiel (dont la configuration de production) et pipeline full-attention 16Q/2KV avec output gate. Les tolérances BF16 sont une borne relative 2⁻⁷ (au plus 2 ULP BF16), explicitement documentée.

**État CI :** le premier passage après correction du masque a détecté deux attentes obsolètes : une ligne preview invalide encore visible dans un test, et des sorties dorées non nulles pour une requête invalide. Le générateur a été corrigé pour construire le masque preview avec les validités amont ; les 64 sorties de la requête invalide sont nulles dans les fixtures. Sur le HEAD `c737e02`, les workflows GitHub `Format, test, and lint` et `Minimum supported Rust 1.86` sont **verts** sur les événements push et pull_request (2/2 jobs chacun). Le workflow stable confirme notamment les 120 tests `claymore-tensor` réussis, les tests workspace, les contrôles CUDA sur stub sans GPU et Clippy. Aucun test matériel ni accès VRAM n'a été exécuté.

Source d'implémentation : `modeling_clover_moe.py` à la révision `4b0402b65e523d0183a14c431955438714cb16ff`. Aucun kernel CUDA, benchmark ni accès VRAM n'est inclus.

## M11.3 — routage MoE, coreset et résidence (PR #55 en revue)

La PR #55 implémente l'issue #50 sans créer de runtime, scheduler ou gestionnaire mémoire parallèle.

- `project_router_logits_cpu` reproduit la projection BF16 sans biais, puis élargit les logits BF16 en FP32 pour le softmax et le routage.
- `CloverMoeRouter::route_batched` préserve le périmètre du coreset par élément de batch. Le top-k est stable (égalité départagée par l'index expert), le top-p sélectionne le préfixe minimal couvrant la masse, les tokens invalides ne contribuent pas au coreset et ne sont pas masqués.
- Le coreset est appliqué uniquement en preview. Le top-k du routeur par token reste inchangé ; les experts exclus restent présents dans le top-k avec un poids nul, comme après masquage des logits par `-inf` dans la référence.
- `execute_moe_cpu` fournit la référence BF16 du calcul SwiGLU, de la réduction FP32 des contributions routées et de la combinaison avec l'expert partagé.
- `CloverExpertCache` utilise le `DeviceMemoryManager` commun, donc les budgets, l'admission/éviction et les gardes IN_FLIGHT du `MemoryOrchestrator` existant. Les poids des experts routés **et de l'expert partagé** sont lus à partir du `CloverLoadPlan` par intersections de chunks Safetensors bornés ; aucune allocation de payload complet du modèle n'est créée. Le backend CPU peut matérialiser un seul expert résident en BF16 pour la référence numérique.
- `prefetch_requests_for_routing` déduplique les experts issus du coreset et les routes actives des tokens invalides ; les experts masqués à poids nul ne sont pas préchargés. `CloverExpertPrefetchRequest` ordonne le préchargement par priorité décroissante, puis couche/expert croissants.
- `CloverExpertLoadSchedule` soumet au scheduler commun un identifiant de tâche Prefetch puis une tâche Compute qui en dépend. Les plages/chunks restent dans les descripteurs du transfert, hors du graphe scheduler ; aucun tenseur ou buffer n'est stocké dans les tâches.
- Les compteurs exposent requêtes, hits/misses, résidents, évictions, octets transférés, échecs, annulations, admissions refusées, synchronisations et protections IN_FLIGHT. Les écritures de ce chemin sont synchrones ; il n'ajoute pas d'attente asynchrone artificielle.

Les tests CPU/fakes couvrent les égalités, la projection BF16, le routage par batch, les lignes invalides, NaN, le calcul expert partagé, les plages de chunks, le chargement, l'annulation après transfert partiel, le rollback sur erreur de lecture, la pression mémoire, le refus d'éviction d'un expert en vol et l'ordre déterministe de préchargement.

**État de validation :** les workflows `Format, test, and lint` et `Minimum supported Rust 1.86` sont verts sur le HEAD actuel de la PR #55 `6677cc79d6b12933480463778759c991ab1636a7` (run #46152385453). La PR #55 reste ouverte pour revue et fusion ; aucune validation matérielle n'est revendiquée. Aucun test GPU, kernel CUDA ou benchmark n'est requis pour #50.

## Contraintes d'intégration

- Le seul runtime et orchestrateur de génération restent ceux de Claymore.
- Les tenseurs et buffers restent possédés par les couches d'exécution et de mémoire ; le scheduler ne transporte que des identifiants.
- Le streaming par shards/chunks, l'admission et la résidence/éviction des experts sont implémentés dans la PR #55 ; ils restent soumis à sa validation complète et à sa fusion avant d'être considérés comme livrés.
- Pas de fallback CPU implicite, cast implicite, FP16, quantification, multi-GPU ou benchmark M12 sans décision architecturale.
- Toute opération CUDA sur RTX 5090 requiert un préflight, un budget annoncé et une autorisation explicite ponctuelle de l'utilisateur. Aucun agent ne peut consommer de VRAM sans cette autorisation.

## Validation et gates restantes

La CI Rust stable et Rust 1.86 est **verte sur le HEAD `443b1c4d`** (run [45859382944](https://github.com/zenhit777/Claymore/actions/runs/45859382944)) et **verte sur le commit des preuves finales `9333242d`** (runs [45908958747](https://github.com/zenhit777/Claymore/actions/runs/45908958747) push et [45908963012](https://github.com/zenhit777/Claymore/actions/runs/45908963012) pull_request, 2/2 jobs chacun) ; la réplique locale complète (`cargo fmt --check`, `clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets --locked` sur stable et 1.86.0 : 284 tests, 0 échec, 12 ignorés) est **verte sur le HEAD de publication**.

Items de validation du rafraîchissement CPU-only (`docs/audits/prompts/m11.1-cpu-evidence-refresh-agent.md`), exécutés le 2026-11-26 puis rejoués sur le HEAD de publication :

1. ~~rejouer le hachage complet de 69,3 GB avec la nouvelle primitive SHA-256~~ — **fait deux fois** : 28/28, exit 0, 55 min 08,79 s (postfix) puis 55 min 11,51 s (finale), avec contrôle croisé `sha256sum` 28/28 ;
2. ~~régénérer les fixtures tokenizer/template et vérifier la parité entre tokenisation directe et rendu retokenisé~~ — **fait deux fois**, 9/9 identités d'identifiants, 3/3 rejets aux motifs attendus, fixtures byte-identiques entre les deux passes ; un défaut d'outillage du script (`BatchEncoding` itéré par `list`) avait été trouvé et corrigé au passage postfix ;
3. ~~mettre à jour les journaux/checksums sans modifier les empreintes épinglées~~ — **fait** sous les préfixes `m11.1-postfix-2026-11-26-` et `m11.1-final-2026-11-26-`, attestations historiques laissées intactes et revérifiées (29/29 postfix ; 17/18 checkpoint, écart N4 préexistant et documenté) ;
4. **fait** : la PR #53 a été fusionnée et la décision A2 acceptée avec amendements ; A2-B reste ouverte pour M11.4, conformément au registre architectural.

Aucun forward pass, aucune génération, aucun kernel CUDA ni aucune validation RTX 5090 n'est requis pour M11.1. La campagne RTX appartient à #52, après les incréments CPU. Voir `docs/audits/m11.1-final-replay-2026-11-26.md`, `docs/audits/m11.1-audit-corrections-2026-11-26.md` et `docs/audits/prompts/m11.1-cpu-evidence-refresh-agent.md`.


## M11.4 — graphe et LowConfidence (en cours, branche `feat/m11-4-clover-lowconfidence`)

La première tranche d'intégration ajoute le graphe Clover et l'adaptateur de génération au-dessus de l'API M9 existante, sans créer de runtime ou d'orchestrateur modèle parallèle.

- `CloverGraph` construit l'ordre exact à partir de `config.layer_types`. Pour chaque couche, il exécute la normalisation d'entrée, GDN ou full-attention, le résiduel, la normalisation post-attention, la projection routeur, le top-k/coreset de #50, le MoE incluant l'expert partagé, puis le résiduel. La normalisation finale et la projection LM head same-position sont exécutées pour le preview seulement.
- `CloverGraphBackend` est le contrat de calcul : les poids, tenseurs, caches GDN/KV et la résidence des experts restent détenus par le backend et les services mémoire communs. Un preview doit être strictement read-only sur les états propres ; le préfill et le replay causal sont les seuls chemins qui les avancent.
- `CloverDiffusionExecutor` implémente `DiffusionExecutor` de M9. Il prépare un bloc complet de tokens masque, lance les previews same-position, applique LowConfidence strict (`confidence > threshold`), remplit toutes les positions admissibles ou la plus confiante si aucune ne passe, valide le bloc, puis ne rejoue causalement que le préfixe émis lorsqu'un bloc suivant est nécessaire.
- Les blocs de sortie incomplets sont tout de même débruités à leur taille de preview complète, conformément à l'upstream. EOS est évalué en ordre positionnel après résolution complète du bloc ; les positions antérieures à l'EOS sont conservées.
- La sémantique EOS suit le code pinned : liste EOS de génération si fournie, sinon repli sur `config.eos_token_id`, union avec `stop_token_ids`, déduplication et tri ; PAD suit `generation_config.pad_token_id`, puis `config.pad_token_id`, puis le premier terminal, puis zéro.
- Les pénalités présence/fréquence/répétition et les biais longueur/think-end ne comptent que les tokens de réponse déjà committés. Les tokens non committés du bloc ne se pénalisent pas mutuellement. La confiance reste calculée sur les logits FP32 non divisés par la température, même en mode sampling.
- M9 notifie désormais l'adaptateur à la fin d'une session ; l'annulation d'une session encore en file peut être vidée via `cancel_with_executor`. Le backend peut ainsi libérer les états propres sans placer de buffers dans le scheduler.
- La politique concernant les 243 lignes de tête sans ID tokenizer reste ouverte (A2-B) : `CloverGenerationOptions` exige un choix explicite `Reject` ou `Mask`, sans valeur par défaut.

**Couverture CPU ajoutée :** parcours layer-by-layer avec résidus et logits same-position dorés, motif pinned 30 GDN / 10 full-attention, top-8 sur 256 experts, LowConfidence multi-blocs et bloc final incomplet, fallback strict, EOS/stop/PAD, répétition/présence/fréquence, length/think-end, politiques Reject/Mask explicites, replay causal, annulation avant/pendant/après bloc, échec MoE, limite de contexte et isolation par exécuteurs/orchestrateurs indépendants.

**Première brique de chargement réel (commit `78c6918`) :** `CloverCheckpointInventory::read_bf16_range()` lit une plage d'éléments BF16 d'un tenseur indexé en s'appuyant sur les offsets Safetensors déjà validés. Chaque lecture est limitée à 8 MiB, vérifie le dtype et les bornes en éléments, et n'alloue ni le shard complet ni le tenseur complet. L'API retourne les motifs BF16 stockés sans conversion numérique ; le consommateur peut ainsi construire des lecteurs de matrices et des opérateurs en streaming. Les tests couvrent la lecture d'une plage, une plage hors limites et un nom de tenseur absent. Cette API ne constitue pas encore le backend d'exécution. Le nouveau `CloverCpuWeightStore` s'appuie sur ce lecteur pour effectuer le gather d'embeddings et les projections linéaires BF16 bias-free. Les projections accumulent en FP32, arrondissent en BF16 en sortie et lisent les matrices par tuiles de 8 MiB maximum. Les tests de fixture Safetensors vérifient l'embedding, la projection dense, la projection routeur (sortie BF16 élargie en FP32), RMSNorm zero-centered, les IDs hors vocabulaire ainsi que les erreurs de forme et de tenseur absent.

**Limite actuelle :** ce commit établit la composition du graphe et le contrat du backend, avec une implémentation backend synthétique dans les tests. Il ne prétend pas encore charger le checkpoint réel ni fournir un backend complet de matérialisation/exécution des poids Clover. La génération réelle à partir des 69,3 GB reste bloquée tant que ce backend n'est pas relié aux poids et aux opérations CPU de référence. #51 reste ouverte : aucune validation complète ni clôture d'issue ne sera revendiquée sur la seule base du graphe synthétique. Aucun contexte CUDA ou VRAM n'est initialisé.

### Backend CPU concret et boucle LowConfidence de bout en bout (M11.4)

`CloverCpuBackend` (`crates/claymore-model-clover/src/cpu_backend.rs`) implémente `CloverGraphBackend` en composant strictement les primitives CPU épinglées de `claymore-tensor` et les services M9/M10 existants : aucune arithmétique propre.

- GDN : dérivation des portes (decay FP32, beta arrondi en BF16 comme l'upstream), convolution causale avec historique propre avancé seulement après le dernier read faillible de la couche (`out_proj`), convolution bidirectionnelle de preview avec poids de contexte droit distincts, `recurrent_commit` causale ou `preview` lecture seule, RMSNorm non centré sur la largeur de tête de valeur × `silu(z)` en un seul arrondi, puis `out_proj`. Les tokens invalides sont remis à zéro avant toute projection, comme `CloverGatedDeltaNet.forward`.
- Full-attention : dé-interleassage `[tête | gate]` de `q_proj`, RMSNorm zéro-centrés Q/K, RoPE partiel avec cos/sin arrondis en BF16 avant rotation (production du `theta` et du `partial_rotary_factor` épinglés), masque additif `full_attention_mask(prefix, block, preview)`, scores FP32 à l'échelle `1/sqrt(head_dim)`, GQA, gate de sortie, puis `o_proj`. La clé est stockée après norm+RoPE, la valeur brute ; l'état KV est FP32, arrondi BF16 uniquement aux points d'activation épinglés — documenté dans le code.
- Résidence : ordre de prefetch déterministe identique à `prefetch_ordered` (priorité décroissante, puis couche et expert croissants), puis pour chaque expert routé à poids non nul un `ensure_resident` immédiatement suivi du read — sous budget plus petit que l'ensemble routé, le LRU ne peut pas strandé un expert entre son chargement et son usage. Les experts à poids nul (tokens invalides masqués) ne sont jamais chargés ; l'expert partagé est chargé après les routés.
- Le cache KV n'est avancé qu'après succès complet de l'opération causale ; une erreur en cours de calcul (shard tronqué après construction, test `attention_rollback_on_io_failure_keeps_cache` et son équivalent GDN) laisse tous les états bit à bit identiques. La politique d'annulation est transmise à `ensure_resident_with_cancel` et comptée dans les métriques de résidence.

`tests/m11_4_low_confidence.rs` exécute la boucle complète : `CloverDiffusionExecutor` + `CloverCpuBackend` + `DiffusionOrchestrator`/`Scheduler` sur un checkpoint synthétique à topologie épinglée (générateur `synth_checkpoint`, aucun téléchargement). Vérifié : acceptation LowConfidence du premier bloc puis coupure stricte à l'EOS ; threshold 1.0 → fallback sur toutes les itérations du bloc avec longueur complète ; trajectoire gloutonne reproductible entre deux backends neufs ; préfill commité (compteurs GDN/KV = prompt) et previews postérieurs laissant tout l'état propre bit à bit identique ; nettoyage des états et de la résidence des experts à la fin de session. Le checkpoint réel n'est pas requis par ces tests.

`tests/m11_4_pinned_checkpoint.rs` (ignoré, `CLAYMORE_M11_CHECKPOINT` obligatoire, `CLAYMORE_M11_VERIFY_SHA=1` optionnel pour le re-hash complet des 28 artefacts) couvre l'inspection épinglée (813 tenseurs, 26 shards, 2 sidecars, 34 661 350 848 paramètres, payload 69 322 701 696 octets), le schéma, le plan de chargement (28 shards / 8 845 chunks ≤ 8 MiB), le preview read-only et le préfill causal sur le checkpoint authentique. **Le scénario LowConfidence multi-blocs sur poids réels** (`pinned_checkpoint_runs_multiblock_lowconfidence_generation_on_real_weights`, 3 tokens, bloc 2) a été ajouté le 28/11 après les campagnes de preview ci-dessous : il compile en CI, mais son exécution sur le snapshot local reste à faire. Il vérifie le rejeu causal du premier bloc, les états GDN/KV avant le second bloc et l'adressabilité tokenizer des tokens produits.

**Exécution sur le checkpoint authentique (27–28/11, avant l'ajout du scénario pinned multi-blocs).** Le snapshot complet est présent en local (26 shards, 69,32 GB, sidecars GDN et sources Python) ; rien n'est téléchargé par les tests, `CLAYMORE_M11_CHECKPOINT` pointe sur le snapshot Hugging Face local et l'absence du chemin ignore proprement les tests. Les trois niveaux de preuve ci-dessous ont été exécutés ; ils ne comprennent pas le nouveau test pinned multi-blocs, qui reste à lancer :

- `tests/m11_4_pinned_checkpoint.rs` — inspection épinglée, schéma, plan de chargement, puis `forward_preview` de bout en bout sur les poids réels : **vert en 651,47 s** (campagne initiale, sans le test multi-blocs ajouté ensuite).
- `tests/m11_4_low_confidence.rs` — contrat LowConfidence de bout en bout sur le backend CPU réel (`CloverCpuBackend` + graphe complet 40 couches) : 15 tests, dont le rejet d'une pénalité dont l'empreinte ne correspond pas aux tenseurs nommés (`active_penalties_are_validated_and_executed_end_to_end`), l'exécution effective de la trajectoire pénalisée (le point le plus bas de la pénalité est atteint et modifie les logits), l'annulation pendant le streaming expert, terminale et nettoyée (`cancellation_during_expert_streaming_is_terminal_and_clears_state`), l'absence de tout forward quand la session est annulée avant son premier pas (`a_session_cancelled_before_its_first_step_runs_no_forward`), l'échec d'E/S expert terminal avec nettoyage (`expert_stream_io_failure_is_terminal_and_clears_state`), le refus avant tout forward d'un bloc dépassant le contexte, l'isolation de deux sessions à budgets indépendants et la égalité du rejeu causal incrémental avec un préfill en une passe : **15/15 verts**.
- `tests/m11_4_upstream_differential.rs` + `tools/m11/gen_m114_reference.py` — différentiel de couches contre l'implémentation upstream épinglée, jouée sur CPU par PyTorch 2.12 pour le prompt `[151644, 872, 198, 9707]`, le préfill 4 tokens et la continuation d'un token, avec un échantillon stride-païen par tenseur et par couche et les agrégats de ligne complète (`absmax`, somme des carrés) : **les deux modes sont verts en 1569,14 s**. Mesures, politique de tolérance et divergences consignées dans [`docs/audits/m11.4-upstream-differential-2026-11-27.md`](audits/m11.4-upstream-differential-2026-11-27.md).

**Limites assumées, mesurées plutôt que supposées.** En routage naturel, les jeux de 8 experts parmi 256 divergent de upstream à partir de la couche 3 (`preview`, `continuation`) et de la couche 1 (`prefill`) — 0 flip à la couche 0 — parce que le top-k sur des logits BF16 est discret sur des quasi-égalités ; le mode épinglé, qui injecte les identifiants et poids amont couche par couche, valide la chaîne numérique complète sur les 40 couches et reproduit le classement des logits (top-1 sur les 4 lignes de preview, recouvrement top-10 complet). La passe causale GDN dérive de l'upstream à mesure que l'état récurrent avance : l'écart élémentaire relatif passe de 1,7·10⁻³ à 5,2·10⁻² entre les couches 0 et 39, alors que la passe bidirectionnelle reste à 2,9·10⁻³ sur la même profondeur. La relecture du code épinglé montre que `causal_conv_silu` upstream utilise des produits dans le dtype d'entrée, une somme FP32, une SiLU FP32 et un cast final, ce que reproduit le chemin Rust ; l'explication initiale par un arrondi séparé de la somme était incorrecte. La cause résiduelle est à isoler dans la récurrence FP32 (`l2_normalize`, réductions `einsum`/`dot`, exponentielle et ordre d'accumulation). **La dérive est mesurée, pas corrigée** : elle reste un point de travail M11.2 (#52), et les tolérances du différentiel ne constituent pas une parité bit à bit. Le code est exécuté exclusivement sur CPU ; aucun contexte CUDA et aucune VRAM ne sont initialisés. #51 reste ouverte.

### Corrections de la revue technique (28/11, PR #56)

Cinq défauts trouvés en revue ont été corrigés en TDD strict — test de non-régression écrit et exécuté
en **rouge** sur le code d'avant correction, correction minimale, puis vert. Aucune référence dorée ni
aucune tolérance n'a été modifiée pour faire passer.

1. **Atomicité du commit** (`executor.rs`) : `commit_block` publiait les tokens avant le rejouement
   causal ; un échec de `forward_causal` laissait donc des tokens publiés sans base causale sur une
   session Failed. Le rejouement devient la dernière opération faillible, la publication suit.
   Test : `a_failed_causal_replay_publishes_no_tokens`.
2. **Réutilisation d'un adaptateur** (`executor.rs`) : rejouer un adaptateur dont la session était
   terminée marquait la seconde session `Completed` sans préfill et exposait les tokens de la requête
   précédente. Contrat « un adaptateur = une session » rendu explicite par `on_session_end` qui ferme
   l'adaptateur ; tout pas ultérieur renvoie `CloverExecutorError::SessionClosed`, le résultat de la
   requête terminée restant lisible. Test : `a_closed_adapter_refuses_a_second_plan_instead_of_reusing_request_state`.
3. **Périmètre de la rampe de longueur** (`generation.rs`) : l'amont épinglé construit l'ensemble ciblé
   comme `eos ∪ stop ∪ {pad}` (`modeling_clover_moe.py:899-901`, `sampling_clover_moe.py:79,116-120`) ;
   la rampe ne visait que les terminaux. Le PAD n'y entre que s'il vient d'une source réelle, jamais d'un
   repli inventé. Test : `length_penalty_ramps_the_explicit_pad_token_as_well`. Le cas épinglé masque ce
   défaut (`pad_token_id 248044 ∈ eos [248046, 248044]`), d'où l'absence de signal jusqu'ici.
4. **Borne de `think_end_token_id`** (`generation.rs`) : un id au-dessus du vocabulaire tokenizer était
   accepté, ce qui rend le biais inopérant sous `Mask` (lignes à −inf) et produit une prédiction
   inutilisable sous `Reject`. `validate` le refuse maintenant par `InvalidTokenId`.
   Test : `rejects_a_think_end_token_outside_the_tokenizer_vocabulary`.
5. **Containment de panique dans l'orchestrateur** (`claymore-runtime/src/diffusion.rs`) : `run_one`
   appelait `execute_step` sans `catch_unwind` ; une panique laissait la tâche `Running` indéfiniment,
   la session hors de l'ensemble clos, et `on_session_end` (libération de la résidence et des caches)
   jamais appelé. La panique est convertie dans le chemin d'échec ordinaire, comme le font déjà
   `claymore-scheduler/src/executor.rs:119-121` et `claymore-runtime/src/scheduled.rs:201-213`.
   Test : `a_panicking_adapter_fails_the_session_instead_of_leaking_the_task`.

**Revalidation après correction (28/11, CPU uniquement).** `cargo test --workspace --all-targets --locked`
= 432 passed / 0 failed / 15 ignored (base 427, +5 tests de non-régression), vert aussi sous rustc 1.86.0 ;
`cargo fmt --all -- --check` et `cargo clippy -p claymore-model-clover -p claymore-tensor -p claymore-runtime
--all-targets -- -D warnings` propres ; `m11_4_pinned_checkpoint` rejoué sur le checkpoint épinglé **vert en
664,20 s** ; `m11_4_upstream_differential` rejoué dans les deux modes : **2 tests verts en 1573,95 s,
`routing_mismatches=0 divergences=0**.
Les corrections C3 et C4 sont numériquement inertes sur le cas épinglé (`pad 248044 ∈ eos`) et C1/C2/C5 ne
touchent pas à l'arithmétique — le différentiel le confirme. Détail complet dans
[`docs/audits/m11.4-review-readiness-2026-11-28.md`](audits/m11.4-review-readiness-2026-11-28.md) §8.

Les défauts R1/R2 de résidence des experts relevés en revue ont été corrigés sur la branche de base M11.3 `feat/m11-3-clover-moe` (commit `6d56e63b71d25725f646cdbf8076a68a14196869`, CI #46133255162) : rollback de libération retentable avec erreur racine conservée, et validation complète du shard Safetensors avant streaming sur le handle conservé. Les points M9 mutualisés avec Gemma (liaison session↔exécuteur, `on_session_end` sans canal d'erreur, métriques de blocs, état d'orchestration non borné) restent ouverts et sont listés, avec les tests proposés, dans
[`docs/audits/m11.4-review-readiness-2026-11-28.md`](audits/m11.4-review-readiness-2026-11-28.md), qui
contient aussi la matrice exigences #51 → implémentation → tests → preuves, les statuts M11.3/M11.4, les
dépendances de PR, les limites CPU/CUDA et la liste des points d'arbitrage laissés à l'architecte
(A2-B notamment, toujours ouverte).
