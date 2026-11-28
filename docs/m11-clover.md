# M11 — Adaptateur Clover-35B-A3B-Preview

## Statut

M11 est découpé en cinq incréments (#48 à #52). **M11.1 a été fusionné dans `main` via la PR #53** (merge commit `362be819`, registre A2 mis à jour dans `cb01a77`). La décision A2 est acceptée avec amendements ; A2-B reste volontairement ouverte pour M11.4 (sémantique EOS et traitement des 243 IDs excédentaires), comme consigné dans `docs/audits/m11-architect-decisions.md`. **M11.2 a été fusionné dans `main` via la PR #54** : références CPU GDN, attention, RoPE et contrats numériques dans `claymore-tensor`, sans moteur parallèle. **M11.3 est en cours de revue dans la PR #55** ; elle ajoute la référence CPU MoE, le coreset par élément de batch et le streaming/résidence bornée des experts.

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

**État de validation :** les workflows `Format, test, and lint` et `Minimum supported Rust 1.86` sont verts sur le HEAD `176efad315e7e64e6e3a7ba053200d16eab8a015` (run Rust CI #46152142329, deux jobs `success`). La revue de code Tech-Lead est terminée sans défaut bloquant propre au périmètre M11.3 identifié ; l'approbation finale de l'architecte reste attendue. Aucune validation matérielle n'est revendiquée. Aucun test GPU, kernel CUDA ou benchmark n'est requis pour #50.

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
