# Agent IA — Validation matérielle M11 Clover sur NVIDIA RTX 5090

## Statut : prompt préparatoire, ne pas exécuter avant le gate CPU/CI

Ce prompt ne constitue **pas** une autorisation d'utiliser le GPU. L'agent doit attendre une autorisation explicite et actuelle du propriétaire avant toute allocation VRAM, initialisation de contexte, lancement CUDA ou test qui touche la carte.

## Contexte

Projet : https://github.com/zenhit777/Claymore

Jalon : M11 — intégration native de `Diffuspace/Clover-35B-A3B-Preview`.

Branche de travail M11 : `feat/m11-clover-foundation`.
Base M10 : `feat/m10-diffusiongemma`, actuellement clôturée administrativement sans merge dans `main`.

Référence matérielle attendue : NVIDIA RTX 5090 32 Gio, compute capability sm_120. L'agent doit détecter et consigner le système hôte, le CPU, la RAM, le noyau WSL2, le driver et le toolkit réellement présents ; ne jamais supposer des valeurs fixes.

Architecture Clover à valider : 40 couches hybrides GDN/full attention, 256 experts routés, top-8/token, 1 expert partagé, BF16 poids, état GDN FP32, bloc 32 par défaut, LowConfidence. Les poids publiés représentent environ 69 GB : ils ne peuvent pas être chargés intégralement dans les 32 Gio de VRAM. Le chemin doit utiliser le streaming et la résidence partielle contrôlés par MemoryOrchestrator.

## Préconditions obligatoires — arrêt si une seule manque

1. Les issues CPU-only #48, #49, #50 et #51 sont implémentées, revues, documentées et leurs tests passent.
2. Un SHA exact de la branche M11 est communiqué et gelé. Aucun commit, changement de PTX ou modification de fixture pendant la campagne.
3. CI stable **et** Rust 1.86 sont vertes sur ce SHA, y compris les checks/clippy des features CUDA applicables.
4. Les révisions upstream et hashes du config, index, tokenizer, code modèle, génération et artefacts de référence sont épinglés.
5. Les tests CPU de référence et différentiels sont verts ; toutes les tolérances ont été définies avant la campagne.
6. Le checkpoint et les fixtures nécessaires sont déjà disponibles localement ; ne pas lancer de téléchargement massif pendant le créneau GPU.
7. Un préflight est présenté au propriétaire avec commandes exactes, ordre des tâches et budget VRAM maximal par tâche.
8. Le propriétaire répond explicitement qu'il autorise cette session. En l'absence de réponse claire, ne rien lancer.

## Phase A — Préflight sans allocation CUDA

Exécuter uniquement après les préconditions CPU/CI, sans créer de contexte ni allouer de VRAM :

```bash
git status --short
git rev-parse HEAD
git branch --show-current
rustc --version
cargo --version
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap,driver_version --format=csv
```

Vérifier que la carte est bien une RTX 5090, que le compute capability est sm_120 et que la VRAM libre suffit au budget annoncé plus une marge. Vérifier l'absence de processus compute externe concurrent. Si la carte n'est pas libre ou si le budget ne passe pas, arrêter et rendre compte ; ne jamais tuer un processus utilisateur.

## Phase B — Plan de campagne à faire approuver

Préparer une liste séquentielle des tests disponibles sur le SHA gelé, par exemple :
- discovery/initialisation du backend et tests de contrats CUDA ;
- primitives GDN et full attention sur petits cas ;
- tests différentiels BF16/FP32 par opération ;
- routeur MoE, top-8, tie-break, expert partagé et coreset ;
- streaming de poids et admission/éviction des experts ;
- graphe complet sur contexte/bloc représentatif ;
- génération réelle LowConfidence sur un prompt fixe, si ce test est livré.

Pour chaque tâche, annoncer :
- commande exacte ;
- estimation VRAM avant lancement ;
- durée estimée ;
- fichiers/logs produits ;
- condition d'arrêt et procédure de vérification de libération.

**Ne pas inventer de nom de test** : lister les tests réellement présents avec `cargo test -- --list` ou l'inventaire du dépôt. Les commandes de compilation/lint CPU-only ne sont pas des validations GPU.

## Phase C — Exécution autorisée

Après autorisation explicite :
- exécuter une seule tâche à la fois ;
- ne jamais dépasser le budget souple de 26 Gio ni le plafond dur planifié de 28 Gio ;
- conserver 4 Gio de réserve physique, sous réserve de l'usage externe réel ;
- contrôler la VRAM avant/après chaque tâche ;
- vérifier que tous les événements CUDA sont terminés et que les gardes mémoire sont libérées avant de poursuivre ;
- interrompre la campagne au premier échec de sûreté, fuite, erreur driver/Xid/NVRM ou pression mémoire imprévue ;
- ne pas relancer avec des tolérances modifiées et ne pas masquer un échec.

Les poids doivent être chargés shard/chunk par shard/chunk via le MemoryOrchestrator. Aucune allocation ad hoc, aucun chargement intégral du checkpoint en VRAM, aucun fallback CPU silencieux.

## Phase D — Preuves à versionner

Produire un rapport sous `docs/audits/` contenant :
- SHA exact du code et statut Git avant/après ;
- versions OS/WSL2, kernel, Rust, CUDA toolkit/nvcc, driver et GPU ;
- révisions upstream exactes et SHA-256 des artefacts ;
- commandes exactes, paramètres, ordre des tâches et tolérances pré-enregistrées ;
- logs bruts complets et empreintes SHA-256 ;
- sorties de comparaison CPU/CUDA par opération/couche/scénario ;
- pics VRAM/RAM, volumes transférés, temps d'attente et évictions ;
- état VRAM et processus avant/après chaque tâche ;
- erreurs Xid/NVRM, ou preuve qu'aucune n'a été observée ;
- limites et cas non exécutés.

Ne jamais déclarer « génération validée », « bit-exact » ou « compatible » sans résultat direct correspondant et traçable. Ne pas présenter les tests unitaires CPU comme preuve matérielle.

## Interdictions

- Ne pas merger de PR, fermer d'issue, pousser sur `main`, ni modifier le SHA gelé.
- Ne pas modifier les tolérances, dorées, révisions upstream ou portes après observation.
- Ne pas régénérer les PTX pendant la campagne.
- Si un kernel/PTX doit changer, arrêter : provenance, CI et campagne matérielle complète doivent être recommencées sur un nouveau SHA après nouvelle autorisation.
- Ne pas lancer benchmarks ou tuning de performance M12 dans cette campagne M11.
- Ne pas continuer à utiliser la carte après la fin de la session autorisée.

## Livrable de l'agent

Un rapport factuel avec verdict par test (PASS/FAIL/NOT RUN), logs et hashes. En cas d'échec, décrire le premier échec reproductible et s'arrêter ; ne pas improviser de correctif sur le créneau matériel.