# M11 — Adaptateur Clover-35B-A3B-Preview

## Statut

M11 est découpé en cinq incréments (#48 à #52). **M11.1 est implémenté côté code CPU-only dans la PR #53** et ses preuves locales ont été **rafraîchies après correctifs puis rejouées sur le HEAD de publication** le 2026-11-26 (`m11.1-final-replay-2026-11-26.md`) : la validation de cette étape n'est plus conditionnée qu'à la revue architecturale et à la décision A2. Cette étape établit les contrats de configuration, la provenance, l'inventaire Safetensors et un plan de lecture metadata-only. Elle ne constitue pas une compatibilité Clover exécutable et ne lance aucun forward pass.

- PR #53 : `feat/m11-clover-foundation` → `main`, ouverte. L'API GitHub la renvoie **non draft** au 2026-11-26 alors que ce document la décrivait en draft : écart consigné (`m11.1-cpu-evidence-refresh-2026-11-26.md`, N2), à trancher par le titulaire de la PR.
- Issue de suivi M11.1 : #48.
- Aucun payload de tenseur n'est chargé ; aucune VRAM n'est consommée.

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

## Contraintes d'intégration

- Le seul runtime et orchestrateur de génération restent ceux de Claymore.
- Les tenseurs et buffers restent possédés par les couches d'exécution et de mémoire ; le scheduler ne transporte que des identifiants.
- Le streaming effectif des poids par shards/chunks, l'admission via MemoryOrchestrator et la résidence/éviction des experts seront implémentés dans les phases suivantes.
- Pas de fallback CPU implicite, cast implicite, FP16, quantification, multi-GPU ou benchmark M12 sans décision architecturale.
- Toute opération CUDA sur RTX 5090 requiert un préflight, un budget annoncé et une autorisation explicite ponctuelle de l'utilisateur. Aucun agent ne peut consommer de VRAM sans cette autorisation.

## Validation et gates restantes

La CI Rust stable et Rust 1.86 est **verte sur le HEAD `443b1c4d`** (run [45859382944](https://github.com/zenhit777/Claymore/actions/runs/45859382944)) ; la réplique locale complète (`cargo fmt --check`, `clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets --locked` sur stable et 1.86.0 : 284 tests, 0 échec, 12 ignorés) est **verte sur le HEAD de publication** ; la CI GitHub sur ce HEAD est consignée dans le relevé de suivi dès les deux jobs terminés.

Items de validation du rafraîchissement CPU-only (`docs/audits/prompts/m11.1-cpu-evidence-refresh-agent.md`), exécutés le 2026-11-26 puis rejoués sur le HEAD de publication :

1. ~~rejouer le hachage complet de 69,3 GB avec la nouvelle primitive SHA-256~~ — **fait deux fois** : 28/28, exit 0, 55 min 08,79 s (postfix) puis 55 min 11,51 s (finale), avec contrôle croisé `sha256sum` 28/28 ;
2. ~~régénérer les fixtures tokenizer/template et vérifier la parité entre tokenisation directe et rendu retokenisé~~ — **fait deux fois**, 9/9 identités d'identifiants, 3/3 rejets aux motifs attendus, fixtures byte-identiques entre les deux passes ; un défaut d'outillage du script (`BatchEncoding` itéré par `list`) avait été trouvé et corrigé au passage postfix ;
3. ~~mettre à jour les journaux/checksums sans modifier les empreintes épinglées~~ — **fait** sous les préfixes `m11.1-postfix-2026-11-26-` et `m11.1-final-2026-11-26-`, attestations historiques laissées intactes et revérifiées (29/29 postfix ; 17/18 checkpoint, écart N4 préexistant et documenté) ;
4. **restant** : obtenir le verdict explicite de l'architecte sur la PR #53 et la décision A2.

Aucun forward pass, aucune génération, aucun kernel CUDA ni aucune validation RTX 5090 n'est requis pour M11.1. La campagne RTX appartient à #52, après les incréments CPU. Voir `docs/audits/m11.1-final-replay-2026-11-26.md`, `docs/audits/m11.1-audit-corrections-2026-11-26.md` et `docs/audits/prompts/m11.1-cpu-evidence-refresh-agent.md`.
