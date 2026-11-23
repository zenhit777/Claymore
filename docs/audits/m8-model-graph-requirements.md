# M8 — Audit des graphes et exigences des modèles cibles

Date de l'audit : 2026-11-22

## Périmètre et méthode

Cet audit clôt l'inspection architecturale demandée par M8 pour les deux cibles produit de Claymore. Il s'appuie sur les configurations publiées, les index Safetensors, les descriptions d'architecture et les implémentations de référence publiquement accessibles. Il ne prétend pas être une exécution des checkpoints dans Claymore : aucun poids complet n'a été chargé, aucun forward pass n'a été exécuté et aucune compatibilité modèle n'est revendiquée.

Références examinées :

- DiffusionGemma : https://huggingface.co/google/diffusiongemma-26B-A4B-it
- DiffusionGemma config : https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/config.json
- DiffusionGemma weight index : https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/model.safetensors.index.json
- DiffusionGemma architecture / pipeline : https://huggingface.co/docs/diffusers/main/en/api/pipelines/diffusion_gemma
- Clover : https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview
- Clover config : https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/config.json
- Clover weight index : https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/model.safetensors.index.json
- Clover model implementation : https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/modeling_clover_moe.py
- Clover technical report : https://arxiv.org/abs/2611.24636

Les fichiers upstream peuvent évoluer. Les futures intégrations devront épingler les révisions exactes des configurations, index et codes sources qu'elles prennent en charge.

## 1. DiffusionGemma

Checkpoint audité : `google/diffusiongemma-26B-A4B-it`.

Éléments confirmés par la configuration et la documentation :

- architecture `DiffusionGemmaForBlockDiffusion`, modèle encoder-decoder spécialisé block diffusion ;
- poids déclarés BF16 ;
- canvas de débruitage de 256 tokens ;
- l'encodeur causal traite le prompt propre et les blocs déjà validés, puis alimente un KV cache ;
- le décodeur traite le canvas avec attention bidirectionnelle et cross-attention vers le contexte mis en cache ;
- chaque étape de débruitage échantillonne des candidats, valide les positions suffisamment confiantes et re-bruite les autres ;
- la génération alterne une progression externe par canvas et une boucle interne de débruitage. Cette orchestration n'est pas une boucle autoregressive token-par-token générique.

### Primitives tensorielle requises

- projections linéaires et produits matriciels ;
- addition/résidus et opérations élémentaires ;
- RMSNorm avec paramètres appris ;
- attention multi-tête bidirectionnelle et cross-attention ;
- softmax stable et produit attention-probabilities / values ;
- gestion des masques, positions et caches KV ;
- calcul de logits et opérations de sélection/confiance nécessaires au raffinement du canvas.

Les opérations de masque, d'échantillonnage, de sélection des positions et de progression du canvas relèvent de l'orchestration diffusion / adaptateur, et non du propriétaire Tensor générique.

Le checkpoint étant BF16, le chemin F32 livré dans M8 ne permet pas à lui seul de consommer directement les poids publiés. Une prise en charge BF16 explicite (stockage, transfert, contrats numériques et kernels) ou une stratégie de conversion/quantification explicitement décidée est nécessaire avant l'exécution réelle du modèle. Aucun cast implicite ne doit être ajouté.

## 2. Clover-35B-A3B-Preview

Checkpoint audité : `Diffuspace/Clover-35B-A3B-Preview`.

Éléments confirmés par la configuration publiée :

- 40 couches, largeur cachée 2048 ;
- poids déclarés BF16 ;
- architecture hybride : trois couches Gated DeltaNet / linear attention suivies d'une couche full attention, répétées sur les 40 couches ;
- 16 têtes query/key et 32 têtes value pour les blocs GDN, dimensions key/value de 128 ;
- full attention avec 16 têtes query, 2 têtes key/value et dimension de tête 256 ;
- 256 experts routés, top-8 par token et un expert partagé par couche ;
- activation SiLU, RMSNorm, RoPE partiel (25 %) ;
- convolution linéaire de largeur 4 ;
- bloc de diffusion par défaut de 32 tokens, avec résolution LowConfidence ;
- paramètres additionnels de GDN bidirectionnel, incluant branche de convolution de contexte droit et paramètres de softmax.

Le modèle combine mémoire récurrente du préfixe historique et interactions bidirectionnelles dans le bloc actif. Son chemin GDN ne doit pas être assimilé à une attention standard ni remplacé par un décodeur autoregressif générique.

### Primitives tensorielle requises

- projections matricielles et opérations élémentaires ;
- RMSNorm, activation SiLU et normalisations associées ;
- full attention avec Q/K/V, RoPE partiel, softmax et agrégation des valeurs ;
- primitives GDN : projections, convolution locale, opérations de portes, mise à jour récurrente de l'état et interaction bidirectionnelle intra-bloc ;
- MoE : projection du routeur, sélection top-k, dispatch/gather/scatter par expert, projections expertes, expert partagé et combinaison pondérée ;
- opérations de transfert / résidence des experts coordonnées par memory et scheduler.

La sélection des experts, le coreset de bloc et le contrôle de débruitage sont des décisions de l'adaptateur Clover. Le runtime partagé fournit les contrats tensoriels, les allocations, les kernels génériques et les mécanismes d'ordonnancement, sans imposer la politique de routage.

Comme pour DiffusionGemma, BF16 est le type publié des poids. Le chemin F32 M8 ne suffit donc pas à consommer directement le checkpoint. La prise en charge BF16 doit être conçue explicitement avant l'exécution modèle.

## 3. Évaluation du socle M8 livré

Le socle actuel fournit quatre opérations F32 CPU/CUDA : add, matmul rank-2 row-major, softmax stable sur le dernier axe et RMSNorm avec scale explicite. Il fournit également des layouts/strides validés, des contrats de dispatch déterministes, des allocations VRAM intégrées au gestionnaire mémoire, des événements CUDA et un exécuteur scheduler dédié.

Ces éléments constituent une base utile et validée matériellement, mais ne constituent pas un runtime modèle prêt à charger l'un ou l'autre checkpoint.

| Capacité | M8 actuel | Besoin modèle |
|---|---|---|
| F32 add / matmul / softmax / RMSNorm | Implémenté CPU + CUDA | Réutilisable |
| BF16 stockage et calcul | **Livré** : 16 bits exacts, RNE, accumulation FP32, CPU + CUDA `sm_120` | Requis par les deux checkpoints |
| SiLU | Non livré | Requis par Clover |
| RoPE / QK norm / attention complète | Non livré | Requis par les deux, avec variantes |
| Cross-attention et KV cache | Non livré | Requis par DiffusionGemma |
| GDN / convolution de contexte droit / état récurrent | Non livré | Requis par Clover |
| Top-k / dispatch / combinaison MoE | Non livré | Requis par Clover et backbone MoE de DiffusionGemma |
| Canvas / denoising / LowConfidence | Hors M8 | Adaptateurs et orchestration diffusion |

## 4. Décisions de frontière et séquencement

1. **M8 reste un socle tensoriel natif, pas un moteur de modèle.** Les quatre kernels F32 validés restent une première tranche de référence ; les résultats RTX 5090 ne doivent pas être présentés comme validation de DiffusionGemma ou Clover.
2. **BF16 est un prérequis partagé, désormais livré.** Il a été traité comme une capacité explicite du système tensoriel (représentation 16 bits, transferts, opérations, accumulation FP32, arrondi RNE et tolérances), et non comme une conversion silencieuse. Les quatre primitives BF16 ont été validées sur RTX 5090 dans `docs/audits/m8-bf16-rtx5090-hardware-validation.md` ; FP16 reste non supporté.
3. **M9 porte l'orchestration diffusion commune** : état des requêtes/blocs, dépendances, progression de débruitage et points d'extension ; il ne fusionne pas les graphes des deux modèles.
4. **M10 porte l'adaptateur DiffusionGemma** : encoder, cache KV, décodeur bidirectionnel/cross-attention et raffinement du canvas.
5. **M11 porte l'adaptateur Clover** : GDN bidirectionnel, full attention, MoE, coreset d'experts et résidence/transferts.
6. **M12 porte l'optimisation** : kernels fusionnés, Tensor Cores, recouvrement transferts/calcul et mesures de performance.

## 5. Critères de clôture de l'audit M8

- [x] Les deux checkpoints cibles sont identifiés sans ambiguïté.
- [x] Les configurations et index Safetensors publics ont été examinés.
- [x] Les paradigmes de génération et les différences irréductibles sont consignés.
- [x] Les familles de primitives requises et les capacités manquantes sont identifiées.
- [x] Les frontières M8/M9/M10/M11/M12 sont explicitées.
- [x] Le support BF16 explicite est livré dans cette PR : types, conversions, contrats, référence CPU, kernels CUDA, accumulation FP32 et tolérances, avec validation matérielle RTX 5090.

**Conclusion :** l'audit architectural demandé par M8 est réalisé. La PR valide un socle tensoriel natif F32 **et BF16** et ses chemins CUDA sur RTX 5090 ; elle ne valide pas l'exécution de modèles. Les primitives spécifiques listées ci-dessus (SiLU, RoPE, attention, KV cache, GDN, MoE, canvas/denoising) restent à livrer dans M9/M10/M11 avant toute revendication de compatibilité DiffusionGemma ou Clover.
