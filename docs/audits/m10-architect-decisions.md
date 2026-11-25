# M10 — Registre des décisions de l'architecte (PR #40, 25 novembre 2026)

Suite du §12 de `m10-cuda-cpu-parity-fix.md`.

Les décisions attendues au §12 sont tranchées. Ce registre est normatif : il remplace
les formulations « à trancher » des §11.8 et §12 du rapport, ainsi que le §8 de
`m10-rtx5090-final-decoder-logits-2026-11-25.md` (« Ce qui reste à décider »), qui
reste inchangé comme enregistrement daté de la campagne.

| ID | Décision | Conséquence dans le code |
|---|---|---|
| **D1** | **Adoptée.** La conformité à la **référence amont** se mesure en pas BF16 relatifs à l'amplitude locale (pic de ligne), jamais par une borne absolue `max_abs`. | `worst_row_drift_steps` dans `golden_cpu.rs` ; dérive journalisée. Seuil de contrat : `MAX_ROW_DRIFT_STEPS = 4,0` du bisect (§11.4), déjà fixé sur la table mesurée. |
| **D2** | **Adoptée.** Le contrat amont est `golden_cpu_layer_bisect` (210/210 lignes, ensemble top-8 exact, ≤ 4,0 pas). Les comparaisons de **chaîne** contre la dorée passent en **suivi** : finitude et taille vérifiées, dérive journalisée, aucune borne numérique. | Trois tests `golden_cpu` renommés `*_chain_drift_is_tracked_against_pinned_upstream` ; porte `hardware_m10_generation.rs` CUDA ↔ dorée remplacée par un suivi. |
| **D3** | **Exclue** (sélection sous-déterminée, §11.5 ; porte imposant l'identité bit, §11.6). | — |
| **D4** | **Refusée** (supprimerait le seul signal amont). Les dorées ne sont ni régénérées ni réépinglées. | — |
| **D5** | **Adoptée le 25/11 (revue finale).** Le contrat **interne** CUDA ↔ CPU est l'**égalité bit** pour les couches du décodeur, le routeur et les logits, comme il l'était déjà pour la chaîne encodeur. Les bornes antérieures (`20,0` pas ; `1e-3` ; `2,0` / `0,1`) sont **resserrées à zéro**. Ce n'est **pas** plus fort que D1 : D1 régit la conformité à la référence **amont** (dorée), D5 compare **deux implémentations Claymore** dont les kernels sont des portes de l'arithmétique CPU par construction (ports glibc 2.39, réductions en ordre d'index, PTX `-fmad=false`), verrouillées par `kernel_order_parity` et `kernel_provenance`. Toute différence non nulle est une régression des kernels, pas du bruit. Le resserrement ne s'applique **jamais** à la comparaison à la référence amont. | `hardware_m10_decoder_layers.rs` : `CUDA_CPU_MAX_ROW_STEPS = 0.0`, `CUDA_CPU_MAX_SCORE_DRIFT = 0.0`, `LOGITS_MAX_ABS = 0.0`, `LOGITS_RMS = 0.0` ; ensembles d'experts CUDA = CPU sur toutes les lignes et argmax / top-8 des logits identiques sur toutes les lignes (auparavant rapportés). `BOUNDARY_TIE_GAP = 1e-3` est **séparé** : il qualifie un échange de frontière contre l'ensemble de **référence** (D1/D2) et ne change pas. |
| **Contrat interne CUDA ↔ CPU** | La chaîne encodeur CUDA est **bit-égale** au CPU (`0,000000 / 0,000000`, `m10-parity-04`). Les 30 couches du décodeur (810 étapes), le routeur et les logits le sont aussi (campagne finale, SHA `460c9ef`, deux passes identiques). Toute différence non nulle est une régression des kernels (D5). | `cuda_graph_chain_is_bit_equal_to_cpu_and_tracks_golden` (encodeur) ; `hardware_m10_decoder_layers` (décodeur, logits). |
| **Contrat de parité glibc** | **Validé** : palier 1 (tables glibc 2.39) sur toute machine ; palier 2 sur hôte-oracle seulement. Le palier 2 est conservé (la couverture dense reste sous contrat chez les développeurs). | inchangé |
| **Provenance des kernels** | Verrou **en CI** : empreintes SHA-256 des `.cu`, `.cuh` et `.ptx` épinglées ; toute modification d'une source sans mise à jour du manifeste, de la procédure et de la campagne GPU échoue. | `tests/kernel_provenance.rs` |
| **Campagne RTX 5090** | Campagne groupée du 25/11 (6/6) et campagne finale du 25/11 (9/9, SHA `460c9ef`) exécutées. Tout changement `.cu`/`.cuh`/PTX impose une nouvelle campagne complète et la mise à jour de la provenance. Aucune nouvelle campagne n'est commandée à ce stade ; un rejeu de `hardware_m10_decoder_layers` est recommandé à la prochaine session GPU autorisée (voir R7). | — |

## 1. Ce que ces décisions ne prouvent pas

- Les comparaisons de chaîne n'étant plus bornées, **elles ne protègent pas contre une dérive de chaîne** : conformément à D2, elles restent un suivi. Les bisects à entrée dorée de l'encodeur, de la vision et du décodeur sont désormais exécutés et contractuels selon leurs bornes mesurées ; voir `m10-cpu-campaign-2026-11-25.md`.
- Les suites CPU dorées/bisects ont été réellement exécutées sur l'environnement avec checkpoint épinglé ; les 12 tests ignorés de la CI standard sont explicitement documentés et couverts par les journaux de campagne appropriés.
- La bit-exactitude CUDA ↔ CPU est **conditionnée** à la libc oracle nommée (glibc 2.39), au PTX `sm_120` compilé en `-fmad=false` et au scénario mesuré (R2). Elle n'est pas une propriété démontrée du PTX lui-même (R3).

## 2. Statut M10

**M10 : VALIDÉ SOUS RÉSERVES NON BLOQUANTES (R1 à R7 ci-dessous).** Les réserves bloquantes du verdict précédent (bisects vision et décodeur, mesure CUDA ↔ CPU du décodeur et des logits, exécution effective des tests CPU, cohérence documentaire) sont **levées**. Il n'y a **pas** de « validé sans réserve » : les réserves résiduelles bornent ce qui peut être revendiqué.

État après la campagne finale du 25/11 :
1. **Mesure CUDA↔CPU des 30 couches du décodeur et des logits : réalisée.** 810/810 étapes à `max_abs=0`, `rms=0`, 0,0 pas BF16 ; routeur brut et post-softmax à 0 ; logits à 0 ; 480/480 ensembles d'experts CUDA = CPU ; 470/480 identiques à la référence et 10 échanges de frontière déjà admis par le contrat CPU, zéro divergence structurelle. Deux passes strictement reproductibles. Pic différentiel 11 889 Mio ; pic génération 24 831 Mio ; aucun processus compute résiduel ni erreur Xid/NVRM. Rapport et neuf journaux horodatés avec manifeste SHA-256 publiés dans `m10-rtx5090-final-decoder-logits-2026-11-25.md`. Le journal brut de l'étape 8 a été relu intégralement lors de la revue : les 810 lignes d'étape, les 60 lignes de routeur et les logits sont à zéro.
2. **Documentation générale : réalignée** sur les campagnes CPU/GPU du 25/11 ; la décision D5 est consignée ici et dans `m10-diffusiongemma.md`.
3. **Bornes CUDA ↔ CPU : décision prise (D5).** Resserrées à l'égalité bit. Voir le tableau.
4. **Décision de merge :** distincte de la validation technique et réservée au propriétaire.

Les décisions D1/D2 ne changent pas. La dérive de chaîne contre la référence eager reste un suivi, pas un gate numérique absolu. Les dorées et leurs empreintes ne doivent être ni régénérées ni réépinglées. Toute modification de `.cu`/`.cuh`/PTX après le gel impose la mise à jour du manifeste et une nouvelle campagne RTX 5090 complète.

## 3. Réserves résiduelles (non bloquantes, bornent les revendications)

| ID | Réserve | Borne de revendication | Levée |
|---|---|---|---|
| **R1** | **Sémantique de la fenêtre glissante divergente entre les deux chemins.** Le décodeur CPU applique `sliding = None` ; le décodeur CUDA (`CudaGraph::decoder_forward`) applique `Some(sliding_window)` (1024) sur les couches locales. Les deux ne coïncident que tant que le cache K/V compte ≤ 1024 lignes, ce qui est le cas de toutes les mesures publiées (7 + 16 lignes). Le rapport final le consigne (§5.3) sans correction. | Le contrat bit-égal CUDA ↔ CPU et toute revendication de génération sont **limités à un cache K/V ≤ 1024 lignes**. | Trancher la sémantique contre le code amont épinglé (`modeling_diffusion_gemma.py`, révision `671aa77…`) ; aligner **un** des deux chemins ; ajouter un test CPU-only à `kv_len > 1024` (sans GPU) ; rejouer la campagne si le chemin CUDA change. |
| **R2** | **Scénario de mesure unique** : canvas de 16 positions, préfixe de 7 jetons, cache de référence injecté par couche. Le canvas de 256 positions et les préfixes longs ne sont couverts que par la génération réelle (déterministe, 256/256 positions), pas par un différentiel CUDA ↔ CPU. | Pas de revendication d'égalité bit sur la génération longue. | Campagne dédiée (autorisation GPU explicite). |
| **R3** | `kernel_order_parity` **transcrit** l'arithmétique des kernels en Rust ; il ne teste pas le PTX. Le verrou de provenance fige les empreintes ; l'équivalence réelle n'est prouvée que par la campagne GPU. | Inchangé. | Toute modification `.cu`/PTX → campagne complète. |
| **R4** | Palier 2 de la parité glibc ignoré sur hôte non-oracle sans trace dans le journal CI standard (flux capturé). Sur le runner, seule la comparaison aux tables épinglées est active. | Choix assumé, validé. | — |
| **R5** | Revendication « bit-exact » conditionnée à glibc 2.39. | Nommer la libc dans toute revendication. | — |
| **R6** | Aucune performance, latence ni débit revendiqué (kernels naïfs, 909 Gio de trafic H2D par canvas). | Inchangé. | Hors périmètre M10. |
| **R7** | Le durcissement D5 du harnais `hardware_m10_decoder_layers.rs` a été commité **après** la campagne : constantes et assertions seulement, compilées et lintées par la CI. Il n'a **pas été rejoué sur la carte**. Les valeurs mesurées (zéros exacts, 810/810 et logits) satisfont les nouvelles portes, mais le test durci n'a pas été vu vert. | Aucune revendication supplémentaire. | Rejouer `hardware_m10_decoder_layers` (≈ 15 min) à la prochaine session GPU autorisée. |
