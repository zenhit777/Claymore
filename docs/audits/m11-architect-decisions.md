# Registre des décisions architecturales M11

## Statut du registre

**A2 — ACCEPTÉE AVEC AMENDEMENTS par l'architecte le 2026-11-27.**

Cette décision fixe les contrats de métadonnées du checkpoint Clover. Elle ne vaut pas validation d'une génération exécutable et ne tranche pas les comportements de terminaison ou d'échantillonnage qui relèvent de A2-B.

## A2 — Préserver les trois couches EOS/PAD du checkpoint Clover

**Contexte authentifié**

Sur la révision upstream épinglée `4b0402b65e523d0183a14c431955438714cb16ff`, l'audit CPU-only M11.1 relève trois contrats distincts :

| Source | EOS | PAD |
|---|---|---|
| `tokenizer.json` / `tokenizer_config.json` | `<|im_end|>` = 248046 | `<|endoftext|>` = 248044 |
| `config.json` | `eos_token_id = 248044` | `pad_token_id = null` |
| `generation_config.json` | `eos_token_id = [248046, 248044]` | `pad_token_id = 248044` |

L'audit observe également que le tokenizer rapide émet 248077 identifiants, alors que les matrices d'embeddings et de sortie conservent 248320 lignes. Ces 243 lignes supplémentaires ne doivent pas conduire à tronquer les poids.

**Décision de l'architecte — ACCEPTÉE AVEC AMENDEMENTS**

1. Conserver les trois sources distinctes et leurs valeurs brutes : configuration modèle, configuration de génération et configuration/tokenizer.
2. Ne pas normaliser ni réécrire `config.eos_token_id` (248044) ou `config.pad_token_id` (`null`) à partir du tokenizer.
3. Pour la tokenisation, respecter les identifiants EOS/PAD du tokenizer, indépendamment du contrat de génération.
4. Conserver les dimensions de poids épinglées `[248320, 2048]`. `tie_word_embeddings = false` : `lm_head` est un tenseur distinct.
5. Les contrats source-spécifiques futurs doivent préserver le type réellement fourni : `generation_config.eos_token_id` accepte un scalaire ou une liste ; `pad_token_id` reste optionnel. Aucun repli implicite, aucune valeur PAD par défaut, aucun `Default` qui remplace un `null`. Les champs inconnus restent rejetés et chaque source doit avoir ses propres tests.

**Exclusions explicites et décisions ouvertes**

- Le point proposé initialement sur l'utilisation de `[248046, 248044]` comme ensemble terminal est retiré de A2. C'est une décision de comportement du moteur, pas un contrat de métadonnées.
- **A2-B — OUVERTE :** la sémantique de terminaison EOS sera décidée dans l'incrément qui implémente la boucle de génération (M11.4), après lecture du code de génération upstream épinglé. D'ici là, aucun ensemble terminal ne doit être codé en dur ; la liste de `generation_config.json` est une donnée d'entrée, pas une règle d'exécution.
- **A2-B — OUVERTE :** le traitement des 243 lignes d'embedding excédentaires (masquage ou rejet à l'échantillonnage) n'est pas tranché. Le moteur ne doit pas supposer que tout identifiant de `[0, 248320)` est un jeton valide.
- Les décisions de fin de bloc/EOS en cours de diffusion restent hors du périmètre de A2.

**Périmètre**

Cette décision fixe la frontière des contrats de métadonnées. Elle n'implémente pas le décodeur et ne constitue pas une preuve d'exécution.

**État**

- [x] Acceptée avec amendements par l'architecte
- [ ] Acceptée sans amendement
- [ ] Rejetée

**Référence probante**

Verdict de l'architecte sur la PR #53, HEAD `f62c31c5396ed6aaadfcc538f2515c1fbaca3edf`, commentaire GitHub du 2026-11-27. Audit : `docs/audits/m11.1-clover-checkpoint-audit-2026-11-26.md`, section 9 ; `docs/audits/fixtures/clover-tokenizer-fixtures.json`, champ `eos_pad_layers`.
