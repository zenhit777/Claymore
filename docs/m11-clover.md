# M11 — Adaptateur Clover-35B-A3B-Preview

## Statut

M11 est suivi par cinq issues techniques (#48 à #52), qui restent rattachées à un seul objectif d'intégration Clover. **M11.1 est implémenté côté code CPU-only dans la PR #53**, mais sa validation reste conditionnée au rafraîchissement de preuves locales et à la revue architecturale. **M11.2 est en cours dans la PR #54** : les références CPU GDN, attention, RoPE et contrats numériques sont développés dans `claymore-tensor`, sans moteur parallèle. Aucune compatibilité Clover exécutable n'est encore revendiquée.

- PR #53 : `feat/m11-clover-foundation` → `main`, toujours ouverte et en draft jusqu'à la clôture des gates documentées.
- Issue de suivi M11.1 : #48.
- Aucun payload de tenseur n'est chargé ; aucune VRAM n'est consommée.

## Référence upstream

- Modèle : `Diffuspace/Clover-35B-A3B-Preview`.
- Révision immuable : `4b0402b65e523d0183a14c431955438714cb16ff`.
- Licence : Apache-2.0 ; conserver `LICENSE` et `ATTRIBUTIONS.md`, y compris les attributions Qwen / Hugging Face applicables.
- Rapport technique : [Clover](https://arxiv.org/abs/2611.24636).
- Manifeste : `docs/audits/clover-upstream-manifest.md`.

L'audit réel publié rapporte 13/13 empreintes non-Xet et 28/28 empreintes Safetensors concordantes avec la table Xet. Il relève 813 tenseurs, 34 661 350 848 paramètres et 69 322 701 696 octets de payload. Le hachage Claymore complet de 69,3 GB doit être rejoué avec la correction SHA-256 ; la comparaison indépendante `sha256sum` 28/28 reste une preuve distincte. Le script de template a également été corrigé pour comparer la tokenisation directe à la retokenisation du texte rendu ; les fixtures locales doivent être régénérées avec ce script avant de revendiquer la parité.

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

## M11.2 — références CPU et contrats numériques (en cours, PR #54)

La PR #54 est empilée sur la fondation #53 tant que celle-ci n'est pas intégrée. Elle reste en draft ; son code n'est pas considéré comme validé tant que la CI complète et les comparaisons différentielles ne sont pas vertes.

Les primitives communes déjà présentes dans M8/M10 sont réutilisées : projections linéaires, RMSNorm, attention GQA, RoPE, masques, softmax, top-k et gather/scatter. Les ajouts Clover restent dans `claymore-tensor` :

- activations SiLU/sigmoid, multiplication élémentaire et variantes RMSNorm zero-centered / RMSNorm-SiLU-gate ;
- références GDN : gates `A_log` / `dt_bias`, transition récurrente FP32, convolution causale, convolution de preview avec right-context indépendant, preview softmax bidirectionnel et bucket historique ;
- masque full-attention causal-prefix / bloc bidirectionnel en preview et causal au commit ;
- gate de sortie full-attention et voie RoPE partielle qui ne tourne que les 25 % initiaux du head Clover.

Le crate `claymore-model-clover` n'introduit pas de dépendance vers le crate tensor : il consommera ces primitives lorsqu'il assemblera le graphe. Les états de preview restent en lecture seule ; seul le commit causal avance les états persistants.

**État de validation au HEAD documenté :** la première CI a révélé un conflit d'emprunt dans le décalage de l'historique convolutionnel et des écarts rustfmt. Le conflit a été corrigé avec `copy_within` et les corrections de format ont été appliquées. Aucun run CI vert n'est encore confirmé sur le HEAD actuel. Les fixtures différentielles indépendantes issues de l'implémentation upstream épinglée, ainsi que les tests workspace / Clippy / MSRV 1.86, restent obligatoires.

Source d'implémentation : `modeling_clover_moe.py` à la révision `4b0402b65e523d0183a14c431955438714cb16ff`. Aucun kernel CUDA, benchmark ni accès VRAM n'est inclus.

## Contraintes d'intégration

- Le seul runtime et orchestrateur de génération restent ceux de Claymore.
- Les tenseurs et buffers restent possédés par les couches d'exécution et de mémoire ; le scheduler ne transporte que des identifiants.
- Le streaming effectif des poids par shards/chunks, l'admission via MemoryOrchestrator et la résidence/éviction des experts seront implémentés dans les phases suivantes.
- Pas de fallback CPU implicite, cast implicite, FP16, quantification, multi-GPU ou benchmark M12 sans décision architecturale.
- Toute opération CUDA sur RTX 5090 requiert un préflight, un budget annoncé et une autorisation explicite ponctuelle de l'utilisateur. Aucun agent ne peut consommer de VRAM sans cette autorisation.

## Validation et gates restantes

La CI Rust stable et Rust 1.86 a réussi sur le code corrigé, notamment le correctif SHA-256, le plan chunk/shard et le contrôle BF16 des sidecars. La CI du HEAD de documentation/scripts doit rester verte.

Avant de sortir M11.1 du statut draft :
1. rejouer le hachage complet de 69,3 GB avec la nouvelle primitive SHA-256 ;
2. régénérer les fixtures tokenizer/template et vérifier la parité entre tokenisation directe et rendu retokenisé ;
3. mettre à jour les journaux/checksums sans modifier les empreintes épinglées ;
4. obtenir le verdict explicite de l'architecte sur la PR #53 et la décision A2.

Aucun forward pass, aucune génération, aucun kernel CUDA ni aucune validation RTX 5090 n'est requis pour M11.1. La campagne RTX appartient à #52, après les incréments CPU. Voir `docs/audits/m11.1-audit-corrections-2026-11-26.md` et `docs/audits/prompts/m11.1-cpu-evidence-refresh-agent.md`.
