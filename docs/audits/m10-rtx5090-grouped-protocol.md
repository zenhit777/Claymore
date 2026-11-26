# Protocole de campagne RTX 5090 groupée — après finalisation CPU M10

Document **préparatoire uniquement** : rien de ce fichier n'autorise une
utilisation du GPU. La RTX 5090 du poste de travail est partagée ; chaque
campagne commence par une autorisation explicite de l'utilisateur (AGENTS.md),
et aucune commande ci-dessous n'est lancée sans cette confirmation.

## 0. Prérequis (CPU, sans GPU)

- Tête de branche fusionnée et propre (`git status` vide), validation CPU
  complète verte (fmt, clippy `-D warnings` stable **et** 1.86,
  `test --workspace --all-targets --locked`, `m10_reference`, les trois bisects
  et la chaîne `golden_cpu` rejoués, rapports `docs/audits/` à jour).
- Checkpoint épinglé présent (SHA-256 de `config.json` et de
  `model.safetensors.index.json` revérifiés, cf. `tools/m10/README.md`).
- Enregistre la tête exacte dans chaque journal (`git rev-parse HEAD` en
  en-tête, comme pour la campagne du 2026-11-24).

## 1. Autorisation et état d'occupation

1. `nvidia-smi` (lecture seule, sans allocation) → mémoire utilisée,
   utilisation, processus compute.
2. Signaler à l'utilisateur la VRAM occupée et demander la VRAM libérable.
3. Ne lancer qu'après confirmation écrite. Minimum observé nécessaire :
   ~24 GiB pic pour `hw_m10_09` (rapport matériel §7.4/§8) ; prévoir 28 GiB
   libres pour éviter toute éviction pendant la session groupée.

## 2. Session groupée (une seule occupation du GPU, ordre fixé)

Chaque commande écrit un journal brut horodaté
`docs/audits/m10-rtx5090-grouped-<n>-*.log.txt`, avec `--nocapture`,
`--locked`, profil `--release` pour les tests à poids réels. Arrêt immédiat de
la session à la première erreur non environnementale (pas de re-tentative
silencieuse ; le journal de l'échec est la preuve).

**Environnement de liaison.** Sur ce poste, le pilote NVIDIA réel est exposé
par WSL : `RUSTFLAGS="-L native=/usr/lib/wsl/lib"` et
`LD_LIBRARY_PATH="/usr/lib/wsl/lib"`. Le stub `ci/cuda-driver-stub.c` de la
CI ne convient **pas** ici : il retourne `CUDA_ERROR_NOT_INITIALIZED` à tous
les points d'entrée et sert aux injections de fautes sans GPU, pas aux tests
matériel. Un stub linké ici produit `cuInit code: 3` et un échec trompeur.
Le stub reste requis pour la réplique CI sans GPU (§0 du rapport CPU).

| # | Commande | Attendu |
| - | -------- | ------- |
| 1 | `cargo test -p claymore-cuda --features cuda-driver --test hardware_cuda_driver --locked -- --ignored --test-threads=1 --nocapture` | 5/5 |
| 2 | `cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_memory --locked -- --ignored --test-threads=1 --nocapture` | 17/17 |
| 3 | `cargo test -p claymore-memory --features cuda-driver --test hardware_cuda_tensor --locked -- --ignored --test-threads=1 --nocapture` | 19/19 |
| 4 | `cargo test -p claymore-memory --features cuda-driver --lib --locked -- --ignored --test-threads=1 --nocapture` | 13/13 (injections B2/B3 comprises) |
| 5 | `CLAYMORE_M10_CHECKPOINT=/chemin/checkpoint cargo test --release -p claymore-memory --features cuda-driver --test hardware_m10_primitives --locked -- --ignored --test-threads=1 --nocapture` | 21/21, dont `hw_m10_09` |
| 6 | `cargo run --release -p claymore-model-gemma --bin verify_checkpoint -- /chemin/checkpoint` | identique à `m10-checkpoint-full-verification.log` |

**Note sur l'étape 4.** Un seul `--` sépare les options du harnais : un
second `--` ferait passer `--ignored` comme argument du filtre de test et la
commande ne exécuterait rien (`0 passed; 1 ignored; 64 filtered out`) tout en
retournant rc=0.

Comptages de référence : rapport matériel §4.1 (66 tests). La source du
sous-graphe décodeur étant inchangée par la finalisation CPU (extraction de
corps partagé sans duplication, chemins CUDA non touchés — preuve d'égalité
bit à bit par la chaîne `golden_cpu`), tout écart de ces résultats par
rapport au 2026-11-24 est un incident à instruire, pas un seuil à ajuster.

## 3. Après la session

- `nvidia-smi` de sortie (aucun processus compute résiduel) ;
- horodater et archiver les six journaux bruts ;
- mettre à jour `docs/audits/m10-rtx5090-hardware-validation.md` §2
  (version d'outils, SHA du code) et §4.3 (pointeurs vers les nouveaux
  journaux), en conservant les résultats du 2026-11-24 en historique ;
- rapporter à l'utilisateur la durée exacte d'occupation du GPU.

## 4. Ce que cette campagne ne couvre pas

Toujours hors périmètre, inchangé : génération complète multi-étapes sur
matériel, capture de graphes CUDA complète du modèle, et toute comparaison
hardware-contre-CPU au-delà des primitives et du sous-graphe décodeur
(`hw_m10_09`). Ces extensions nécessitent chacune une décision d'architecte et
une autorisation GPU distincte.
