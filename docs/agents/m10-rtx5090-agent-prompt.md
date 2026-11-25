# Agent IA — finalisation matérielle M10 sur RTX 5090

## Mission et état de départ

Tu interviens exclusivement pour lever les dernières réserves matérielles de la PR #40 / issue #42 de Claymore.

Dépôt : https://github.com/zenhit777/Claymore  
Branche de travail : `feat/m10-diffusiongemma`  
PR à ne jamais fusionner : https://github.com/zenhit777/Claymore/pull/40  
Issue de suivi : https://github.com/zenhit777/Claymore/issues/42

**Le travail CPU-only est terminé et ne doit pas être redéveloppé.** Les rapports et fixtures existants sont la référence :
- `docs/audits/m10-cpu-campaign-2026-11-25.md`
- `docs/audits/m10-cpu-2026-11-25-logs.sha256`
- `crates/claymore-model-gemma/tests/data/m10_layers/`
- `tools/m10/`
- `docs/audits/m10-architect-decisions.md`

Les bisects de référence ont déjà été exécutés : encodeur 210/210 (pire 2,0 pas BF16, borne 4,0), vision 0,2 pas (borne 4,0), décodeur 10,0 pas (borne 20,0 ; 470/480 + 10 échanges de frontière justifiés, zéro divergence structurelle). Ne régénère ni ne réépingle aucune dorée et ne change pas D1/D2.

Une campagne RTX 5090 groupée a déjà obtenu 6/6 étapes vertes sur la tête CPU finalisée : driver 5/5, mémoire 17/17, tenseurs 19/19, injections 13/13, primitives M10 21/21 et vérification complète du checkpoint. Le sous-graphe décodeur réel de couche 0 est bit-exact. Rapport : `docs/audits/m10-rtx5090-grouped-campaign-2026-11-25.md`.

**Réserves matérielles non levées :**
1. mesures différentielles CUDA ↔ CPU des 30 couches du décodeur ;
2. mesures différentielles CUDA ↔ CPU des logits ;
3. publication des mesures, limites, journaux et preuve de propreté mémoire sur la tête finale.

La dérive de chaîne contre la référence eager amont est un suivi D2, pas un gate numérique absolu. Ne réintroduis pas l'ancienne exigence de comparaison de chaîne bit-exacte contre la dorée eager.

## Autorisation obligatoire — ne pas lancer avant confirmation

La RTX 5090 est partagée. **Avant toute commande qui initialise CUDA, alloue de la VRAM ou lance un kernel**, vérifie qu'une autorisation explicite et récente du propriétaire pour cette campagne figure dans les instructions de mission. Le présent prompt décrit le travail mais ne constitue pas, à lui seul, une preuve que la carte est libre ou qu'un créneau vient d'être accordé.

Si l'autorisation manque, arrête-toi après les vérifications CPU/non intrusives et demande l'accord. Ne sonde pas la carte de manière répétée et ne lance aucun test matériel.

## Phase 0 — gel et préflight

1. Lire `AGENTS.md`, les instructions du dépôt, les rapports CPU/GPU, le protocole `docs/audits/m10-rtx5090-grouped-protocol.md` et le registre D1/D2.
2. Vérifier `git status`, branche et SHA exact. Refuser de travailler sur un arbre sale ou une branche différente sans instruction explicite. Ne jamais écraser des modifications préexistantes.
3. Vérifier que la tête est celle de la PR #40 et que la CI stable + Rust 1.86 est verte. Consigner le SHA complet qui sera réellement testé.
4. Confirmer que les sources CUDA, PTX et le manifeste de provenance correspondent exactement à la tête gelée. Calculer et consigner leurs SHA-256.
5. Vérifier la disponibilité du checkpoint épinglé via `CLAYMORE_M10_CHECKPOINT`, sans le télécharger à nouveau si l'artefact vérifié est déjà présent. Identité attendue : `google/diffusiongemma-26B-A4B-it`, révision `cd40195554d3cfcb56f34b9adfb66efc790ed5b6`.
6. Après autorisation, faire un seul préflight matériel : `nvidia-smi`, VRAM libre, processus compute, driver, SM 120, CUDA toolkit/nvcc, WSL2, RAM et espace disque. Si la VRAM libre est insuffisante pour la prochaine étape, ne pas démarrer ; documenter et demander un autre créneau.
7. Annoncer le budget VRAM maximal estimé avant chaque étape. Une seule tâche GPU à la fois. Vérifier la libération après chaque tâche.

Liaison WSL2 attendue pour le vrai driver, selon le protocole validé :
```bash
export RUSTFLAGS="-L native=/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib"
```
Ne pas utiliser le stub CUDA de CI pour une campagne matérielle.

## Phase 1 — mesures CUDA ↔ CPU du décodeur complet

Objectif : mesurer le différentiel sur **les 30 couches**, et non seulement sur la couche 0 déjà validée.

1. Réutiliser les entrées, états et caches de référence versionnés dans les fixtures M10. Pour chaque couche, injecter l'entrée dorée exacte et les K/V de référence correspondant à cette couche. Ne jamais substituer un cache Claymore dérivé à un cache K/V de référence.
2. Exécuter le chemin CPU et le chemin CUDA avec les mêmes poids épinglés, mêmes tenseurs d'entrée, même masque, mêmes positions et même état de canvas.
3. Comparer chaque étape pertinente de chaque couche, au minimum : entrée, normalisations, projections Q/K/V, normalisations Q/K, RoPE, attention, projections de sortie, résiduels, MLP/MoE, couche de sortie et états/cache utilisés par la couche suivante.
4. Mesurer séparément les logits finaux : tenseur brut avant sélection, dimensions, valeurs finies, max absolu, RMS, erreur relative, pas BF16 relatifs au pic de ligne, et divergences de classement si une sélection est appliquée.
5. Produire un tableau par couche et par étape, ainsi qu'un résumé global et les pires cas avec index de tenseur/ligne/élément.
6. Répéter les opérations sensibles pour vérifier la reproductibilité. Distinguer les différences de calcul d'un défaut de cache, de layout, de synchronisation ou d'ordre d'accumulation.

**Contrat numérique :**
- Respecter D1 : exprimer la dérive en pas BF16 relatifs au pic de ligne, pas en seuil absolu isolé.
- Respecter D2 : la comparaison de chaîne contre la dorée eager reste un suivi, pas une porte absolue.
- Pour les comparaisons CUDA ↔ CPU, relever les bornes actuellement codées avant l'exécution. Ne jamais les relâcher après observation. Si une borne échoue, isoler et corriger le défaut ; si les mesures justifient une borne plus stricte, proposer la modification avec preuve et test contractuel. Ne pas inventer une tolérance a posteriori.
- Ne pas modifier les décisions architecturales, les dorées, leurs empreintes ou le contrat top-k.

## Phase 2 — campagne matérielle finale

Sur le SHA gelé, rejouer séquentiellement les contrôles matériels du protocole, au minimum :
1. `hardware_cuda_driver` ;
2. `hardware_cuda_memory` ;
3. `hardware_cuda_tensor` ;
4. tests matériels de `claymore-memory` ignorés et injections de fautes ;
5. `hardware_m10_primitives` en release ;
6. vérification intégrale du checkpoint avec `verify_checkpoint` ;
7. différentiel complet des 30 couches du décodeur et des logits ;
8. génération réelle DiffusionGemma avec la politique épinglée, si le protocole de validation finale la requiert et si le budget annoncé le permet.

Ne lance jamais deux tâches simultanément. Pour chaque étape, consigner commande exacte, heure, code de retour, VRAM avant/pic/après et processus compute résiduels. Vérifier qu'aucun buffer n'est libéré en vol, qu'aucun OOM non maîtrisé ne survient et qu'aucun processus de calcul ne subsiste après la campagne.

Budget connu : VRAM physique 32 Gio, budget souple 26 Gio, plafond dur 28 Gio, réserve de sécurité minimale 4 Gio. Le plafond est un seuil d'admission, pas une limite de taille du checkpoint ; les poids sont streamés par couche et le swap n'est pas un tier de poids.

## Modifications de kernels : règle stricte

Le but de cette mission est la mesure sur le code gelé. **Ne modifie aucun `.cu`, `.cuh` ou PTX pendant la campagne.**

Si un défaut de kernel est découvert :
1. arrêter la campagne en cours et préserver tous les journaux ;
2. diagnostiquer le défaut et proposer un correctif séparé ;
3. si le correctif est autorisé, mettre à jour le manifeste de provenance dans le même changement, régénérer les PTX de manière reproductible et vérifier leurs empreintes ;
4. relancer la CI complète ;
5. refaire une campagne matérielle complète sur le nouveau SHA, y compris toutes les étapes précédentes. Une campagne sur un SHA antérieur ne valide pas le nouveau code.

Aucun changement opportuniste de tolérance, aucun test désactivé, aucun fallback CPU, aucun cast implicite, aucune allocation CUDA hors MemoryOrchestrator, aucun swap comme tier de poids.

## Phase 3 — rapport et livrables

Créer un rapport dédié, par exemple `docs/audits/m10-rtx5090-final-decoder-logits-2026-11-25.md`, qui contient :
- SHA complet du code testé et état de la branche ;
- révision du checkpoint et empreintes de ses artefacts ;
- environnement matériel/logiciel complet ;
- autorisation et préflight, budgets annoncés ;
- commandes exactes et journaux bruts horodatés ;
- résultats détaillés des 30 couches et des logits, métriques et bornes ;
- résultats de tous les tests matériels, génération si exécutée ;
- VRAM/RAM/transferts/pics, vérification de libération et absence de processus résiduel ;
- écarts, échecs, limites et éléments non revendiqués ;
- manifeste SHA-256 des journaux.

Mettre à jour les documents M10 uniquement avec des faits prouvés. Les documents généraux `docs/m10-diffusiongemma.md`, `docs/architecture.md`, `docs/Claymore.md` doivent rester cohérents avec le rapport CPU et la campagne matérielle. Ne pas réécrire l'historique comme si les anciennes campagnes n'avaient pas eu lieu.

## Phase 4 — validation logicielle et compte rendu

Après tout changement, exécuter :
```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo check -p claymore-cuda --features cuda-driver --all-targets --locked
cargo check -p claymore-memory --features cuda-driver --all-targets --locked
cargo check -p claymore-model-gemma --features cuda-streaming --all-targets --locked
cargo +1.86 test --workspace --all-targets --locked
cargo +1.86 clippy --workspace --all-targets --locked -- -D warnings
```
Vérifier les deux jobs GitHub Actions sur le commit final : stable et Rust 1.86, événements push et pull_request si disponibles.

Pousser les commits sur `feat/m10-diffusiongemma` et mettre à jour la PR #40. Ne créer aucune PR concurrente et ne fusionner ni #40 ni une autre PR.

À la fin, fournir :
- SHA final testé ;
- liste exacte des réserves levées et non levées ;
- résultats GPU détaillés et liens vers les logs/rapports ;
- CI et MSRV observés ;
- tout défaut restant avec cause et preuve ;
- déclaration explicite indiquant si la validation définitive de l'architecte peut être demandée.

**Ne jamais déclarer M10 définitivement validé et ne jamais fusionner la PR #40 : seul l'architecte rend le verdict final et le propriétaire décide du merge.**
