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
| **Campagne RTX 5090** | Campagne groupée du 25/11 exécutée : 6/6 étapes vertes, checkpoint vérifié, primitives 21/21 bit-exactes et sous-graphe décodeur couche 0 bit-exact. Réserve restante : mesures CUDA↔CPU des 30 couches du décodeur et des logits. Tout changement `.cu`/PTX impose une nouvelle campagne complète et mise à jour de provenance. | — |

## 1. Ce que ces décisions ne prouvent pas

- Les comparaisons de chaîne n'étant plus bornées, **elles ne protègent pas contre une dérive de chaîne** : conformément à D2, elles restent un suivi. Les bisects à entrée dorée de l'encodeur, de la vision et du décodeur sont désormais exécutés et contractuels selon leurs bornes mesurées ; voir `m10-cpu-campaign-2026-11-25.md`.
- Les suites CPU dorées/bisects ont été réellement exécutées sur l'environnement avec checkpoint épinglé ; les 12 tests ignorés de la CI standard sont explicitement documentés et couverts par les journaux de campagne appropriés.

## 2. Statut M10

**M10 : validé sous réserves, pas validé sans réserve.** La finalisation CPU du 25/11 est achevée : générateur et fixtures de référence, bisects des 30 couches encodeur (210/210, 2,0 pas), des 27 couches vision (0,2 pas) et des 30 couches décodeur (10,0 pas ; 470/480 + 10 échanges de frontière justifiés), tests exécutés sur checkpoint, journaux et empreintes publiés.

La campagne RTX 5090 groupée du 25/11 est également exécutée (6/6 étapes vertes). Elle couvre driver, mémoire, tenseurs, injections de fautes, primitives M10 et vérification complète du checkpoint ; le sous-graphe décodeur réel de couche 0 est bit-exact.

Réserves de clôture restantes :
1. mesurer CUDA↔CPU sur les **30 couches du décodeur et les logits** sur RTX 5090 ; publier métriques, journaux bruts, SHA, budgets et preuve de libération des ressources ; resserrer les bornes CUDA↔CPU existantes uniquement après mesure ;
2. réaligner et vérifier `docs/m10-diffusiongemma.md`, `docs/architecture.md` et `docs/Claymore.md` sur la finalisation CPU et la campagne matérielle du 25/11.

Les décisions D1/D2 ne changent pas. La dérive de chaîne contre la référence eager reste un suivi, pas un gate numérique absolu. Les dorées et leurs empreintes ne doivent être ni régénérées ni réépinglées. Toute modification de `.cu`/`.cuh`/PTX après le gel impose la mise à jour du manifeste et une nouvelle campagne RTX 5090 complète.
