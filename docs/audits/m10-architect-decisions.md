# M10 — Registre des décisions de l'architecte (PR #40, 25 novembre 2026)

Suite du §12 de `m10-cuda-cpu-parity-fix.md`.

Les décisions attendues au §12 sont tranchées. Ce registre est normatif : il remplace
les formulations « à trancher » des §11.8 et §12 du rapport.

| ID | Décision | Conséquence dans le code |
|---|---|---|
| **D1** | **Adoptée.** La conformité se mesure en pas BF16 relatifs à l'amplitude locale (pic de ligne), jamais par une borne absolue `max_abs`. | `worst_row_drift_steps` dans `golden_cpu.rs` ; dérive journalisée. Seuil de contrat : `MAX_ROW_DRIFT_STEPS = 4,0` du bisect (§11.4), déjà fixé sur la table mesurée. |
| **D2** | **Adoptée.** Le contrat amont est `golden_cpu_layer_bisect` (210/210 lignes, ensemble top-8 exact, ≤ 4,0 pas). Les comparaisons de **chaîne** contre la dorée passent en **suivi** : finitude et taille vérifiées, dérive journalisée, aucune borne numérique. | Trois tests `golden_cpu` renommés `*_chain_drift_is_tracked_against_pinned_upstream` ; porte `hardware_m10_generation.rs` CUDA ↔ dorée remplacée par un suivi. |
| **D3** | **Exclue** (sélection sous-déterminée, §11.5 ; porte imposant l'identité bit, §11.6). | — |
| **D4** | **Refusée** (supprimerait le seul signal amont). Les dorées ne sont ni régénérées ni réépinglées. | — |
| **Contrat interne CUDA ↔ CPU** | La chaîne encodeur CUDA est **bit-égale** au CPU (mesure `0,000000 / 0,000000`, `m10-parity-04`). Toute différence non nulle est une régression des kernels. | `cuda_graph_chain_is_bit_equal_to_cpu_and_tracks_golden` exige `max_abs == 0` et `rms == 0` pour l'encodeur. Décodeur et logits CUDA ↔ CPU : bornes existantes **conservées**, jamais mesurées avant ce correctif (le test s'arrêtait plus tôt) ; à resserrer après la prochaine campagne. |
| **Contrat de parité glibc** | **Validé** : palier 1 (tables glibc 2.39) sur toute machine ; palier 2 sur hôte-oracle seulement. Le palier 2 est conservé (la couverture dense reste sous contrat chez les développeurs). | inchangé |
| **Provenance des kernels** | Nouveau verrou **en CI** : empreintes SHA-256 des `.cu`, `.cuh` et `.ptx` épinglées ; toute modification d'une source sans mise à jour du manifeste, de la procédure et de la campagne GPU échoue. | `tests/kernel_provenance.rs` |
| **Campagne RTX 5090** | **Non commandée.** Rejeu obligatoire après tout changement de `.cu`/PTX (le verrou de provenance le détecte) avec les critères du manifeste de `kernel_provenance.rs`. | — |

## 1. Ce que ces décisions ne prouvent pas

- Les comparaisons de chaîne n'étant plus bornées, **elles ne protègent plus d'une dérive
  de chaîne** : seul le bisect de l'encodeur est contractuel. **Il n'existe pas encore de
  bisect à entrée dorée pour la tour de vision ni pour le décodeur** ; ces deux étages ne
  sont donc couverts que par le suivi. Ce trou borne le statut M10 (§2 ci-dessous).
- Les gates de chaîne ont été remplacés sans nouvelle mesure sur dorées : ces tests
  sont `#[ignore]` et n'ont pas pu être rejoués dans ce cycle (checkpoint absent). Ils ont
  été compilés, formatés et passés à clippy `-D warnings`.

## 2. Statut M10

**M10 : validé sous réserve, pas validé sans réserve.**
Les réserves de clôture, par ordre de priorité :
1. produire les entrées par couche de référence (`golden_layers`) pour la vision et le
   décodeur et ajouter leurs bisects (même métrique D1, bornes fixées sur la mesure) ;
2. rejouer une fois la campagne RTX 5090 sur la tête finale pour mesurer décodeur et logits
   CUDA ↔ CPU, puis resserrer les bornes du tableau ci-dessus ;
3. exécuter les tests `golden_cpu` / bisect sur une machine disposant du checkpoint, avec
   résultat publié.
