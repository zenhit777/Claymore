# Registre des décisions architecturales M11

## Statut du registre

**BROUILLON — en attente de revue et de décision explicite de l'architecte.**

Ce fichier ne vaut ni approbation architecturale ni autorisation de déclarer Clover exécutable. Les décisions proposées ci-dessous doivent être acceptées, amendées ou rejetées par l'architecte ; ne pas les présenter comme déjà validées.

## A2 — Préserver les trois couches EOS/PAD du checkpoint Clover

**Contexte authentifié**

Sur la révision upstream épinglée `4b0402b65e523d0183a14c431955438714cb16ff`, l'audit CPU-only M11.1 relève trois contrats distincts :

| Source | EOS | PAD |
|---|---|---|
| `tokenizer.json` / `tokenizer_config.json` | `<|im_end|>` = 248046 | `<|endoftext|>` = 248044 |
| `config.json` | `eos_token_id = 248044` | `pad_token_id = null` |
| `generation_config.json` | `eos_token_id = [248046, 248044]` | `pad_token_id = 248044` |

L'audit observe également que le tokenizer rapide émet 248077 identifiants, alors que les matrices d'embeddings et de sortie conservent 248320 lignes. Ces 243 lignes supplémentaires ne doivent pas conduire à tronquer les poids.

**Risque architectural**

Réduire ces trois sources à un EOS scalaire ou remplacer le PAD nul de `config.json` par la valeur du tokenizer/generation config détruirait une distinction réelle du checkpoint et pourrait modifier la terminaison de génération.

**Décision proposée à l'architecte**

1. Conserver les valeurs brutes dans des contrats source-spécifiques : configuration modèle, configuration de génération et configuration/tokenizer.
2. Ne pas normaliser ni réécrire `config.eos_token_id` ou `config.pad_token_id` à partir du tokenizer.
3. Pour la génération, utiliser la liste complète `generation_config.eos_token_id = [248046, 248044]` comme ensemble des identifiants terminaux, sous réserve de validation de la sémantique dans M11.2/M11.4.
4. Pour la tokenisation, respecter les identifiants EOS/PAD du tokenizer, indépendamment du contrat de génération.
5. Conserver les dimensions de poids épinglées `[248320, 2048]` ; ne jamais réduire les matrices à la taille du vocabulaire émis.

**Périmètre**

Cette décision fixe la frontière des contrats de métadonnées. Elle n'implémente pas le décodeur, ne tranche pas les comportements de fin de bloc/EOS en cours de diffusion et ne constitue pas une preuve d'exécution.

**État**

- [ ] Acceptée par l'architecte
- [ ] Amendée par l'architecte
- [ ] Rejetée par l'architecte

**Référence probante**

`docs/audits/m11.1-clover-checkpoint-audit-2026-11-26.md`, section 9 ; `docs/audits/fixtures/clover-tokenizer-fixtures.json`, champ `eos_pad_layers`.
