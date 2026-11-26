# M11 — Adaptateur Clover-35B-A3B-Preview

## Statut

M11 est le prochain jalon produit après M10. Il est en cours de découpage et d'implémentation. Le premier incrément CPU-only introduit le contrat de configuration typée dans `claymore-model-clover`. Il ne constitue pas une compatibilité checkpoint ni une exécution du modèle.

## Référence upstream

- Modèle : `Diffuspace/Clover-35B-A3B-Preview`
- Révision du `config.json` consultée : `4b0402b65e523d0183a14c431955438714cb16ff`
- La révision du code modèle, du tokenizer, du template de conversation, de l'index Safetensors et des poids doit être auditée et épinglée séparément avant l'ouverture du chargeur de checkpoint.
- Licence publiée : Apache-2.0 ; les attributions Qwen / Hugging Face doivent être conservées pour le code dérivé.
- Référence d'architecture : [rapport technique Clover](https://arxiv.org/abs/2611.24636).

Ne pas utiliser `main` comme source reproductible et ne pas déclarer le modèle pris en charge tant que toutes les révisions et empreintes ne sont pas consignées.

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
| Poids | BF16 |
| État SSM/GDN | FP32 |
| Contexte maximal | 131 072 |
| Bloc par défaut | 32 |
| Coreset | activé, masse top-p 0.9 |
| Décodage | LowConfidence, seuil 0.95 |

La génération est une diffusion par blocs : le préfixe validé reste causal ; le bloc actif est bidirectionnel. GDN combine mémoire récurrente historique et attention intra-bloc, avec une branche de convolution de contexte droit. Ce calcul ne doit pas être remplacé par une attention standard ni par une boucle autoregressive token-par-token.

Le coreset agrège les scores top-8 sur le bloc et sélectionne un sous-ensemble d'experts couvrant une fraction de masse. Il n'altère ni les poids stockés, ni le top-8 par token, ni le sens du routeur.

## Découpage

- #48 — audit upstream, config typée, inventaire checkpoint et provenance.
- #49 — références CPU et contrats numériques GDN/full attention/RoPE/SiLU.
- #50 — routage MoE, coreset, streaming et résidence d'experts.
- #51 — graphe complet et LowConfidence sur le `DiffusionExecutor` commun.
- #52 — kernels CUDA, différentiel et campagne RTX 5090.

## Premier incrément implémenté

`crates/claymore-model-clover/src/config.rs` introduit :
- le parseur typé des champs structurants du config ;
- validation BF16 explicite, dimensions, géométrie GQA, top-k, paramètres flottants, motif des couches et sémantique de débruitage ;
- validation de l'architecture épinglée ;
- fixture JSON et tests négatifs/positifs CPU-only.

Ce premier incrément ne vérifie pas encore le SHA-256 du fichier upstream et ne valide pas l'index Safetensors. Ces points restent dans #48 ; ne pas interpréter le seul parseur comme une preuve de provenance du checkpoint.

## Contraintes d'intégration

- Le seul runtime et orchestrateur de génération restent ceux de Claymore.
- Les tenseurs et buffers restent possédés par les couches d'exécution et de mémoire ; le scheduler ne transporte que des identifiants.
- Les poids BF16 sont streamés par shards/chunks, avec admission via MemoryOrchestrator.
- Le budget VRAM souple de référence est 26 Gio et le plafond dur planifié 28 Gio sur 32 Gio ; la réserve reste indisponible au modèle.
- Pas de fallback CPU implicite, cast implicite, FP16, quantification, multi-GPU ou benchmark M12.
- Toute opération CUDA sur RTX 5090 requiert un préflight, un budget annoncé et une autorisation explicite ponctuelle du propriétaire.

## Validation

Pour chaque incrément :
```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked --no-default-features
```

La CI doit également couvrir Rust 1.86 et les features CUDA sans GPU via les stubs existants. Les tests matériels sont distincts et ne doivent jamais être exécutés automatiquement en CI.
