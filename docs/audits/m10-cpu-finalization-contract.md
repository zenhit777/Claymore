# M10 — Contrat de finalisation CPU-only

Ce document complète le registre D1/D2 de `m10-architect-decisions.md`.
Il ne modifie aucune référence dorée, aucune tolérance et aucun contrat GPU.

## État prouvé

- Le bisect encodeur est déjà implémenté par
  `tests/golden_cpu_layer_bisect.rs` et
  `encoder_forward_traced_from_hidden`.
- Le journal `m10-parity-11-bisect-couches-entrees-dorees.log.txt`
  atteste un vrai passage en profil release : 30 couches × 7 lignes,
  210/210 sélections top-8 fidèles, pire dérive de 2,0 pas BF16.
- Ce résultat est une preuve historique sur l'environnement de campagne.
  Il ne remplace pas un rejeu sur la tête finale de la PR #40.
- Les tests `golden_cpu.rs` de chaîne vision/décodeur ne sont que des
  diagnostics de dérive ; ils ne constituent pas des bisects à entrée dorée.
- Le journal 11 décrit les scripts de préparation `probe_layer_inputs.py`
  et `gen_golden_eager.py` comme scripts de campagne jetables. Ils ne sont
  pas présents dans le dépôt et ne forment donc pas une procédure reproductible
  versionnée.

## Contrat de bisect vision à construire

La tour vision expose 27 couches et `vision_forward_stages` conserve sept
sorties intermédiaires par couche. Le futur point d'injection doit accepter
l'état caché **à l'entrée de la couche**, après patch embedding et addition des
embeddings de position, puis exécuter exactement une couche.

Artefacts attendus dans le répertoire de référence, sans écrasement :

- `vision_in_L.bin` : entrée de référence de la couche L, F32 little-endian,
  forme logique `[1, 9, 1152]` pour le fixture de campagne courant ;
- `vision_layer_L.bin` : sortie de la couche L, même forme ;
- les métadonnées associées doivent enregistrer le commit upstream, la version
  Transformers/PyTorch, les paramètres CPU, la commande, le SHA-256 et la
  forme de chaque tenseur.

Validation préalable obligatoire : `vision_in_0` doit être identique à
l'entrée de la première couche observée dans le rejeu de référence ; pour
chaque L, la sortie de L doit être identique à `vision_in_{L+1}` avant toute
comparaison Rust. Les mesures de conformité seront calculées avec D1, et la
borne ne sera fixée qu'après mesure complète des 27 couches.

## Contrat de bisect décodeur à construire

Un état caché d'entrée seul ne suffit pas : l'attention du décodeur consomme
également le KV cache d'encodeur et le préfixe KV déjà écrit pour le décodeur.
Le futur test doit donc injecter, pour chaque couche, l'état caché d'entrée et
les caches K/V de référence dans la disposition attendue par `KvCache`.

Artefacts minimaux à spécifier et produire :

- `decoder_in_L.bin` : état caché de référence avant la couche L ;
- K et V de référence du préfixe encodeur pour la couche L ;
- si le scénario de référence inclut un préfixe décodeur antérieur, ses K/V
  doivent être fournis séparément et dans le même ordre que l'exécution eager ;
- `decoder_layer_L.bin` : sortie de référence après la couche L ;
- manifeste décrivant dimensions, ordre des axes, dtype source, conversion
  vers F32 little-endian, longueur du préfixe, positions et paramètres
  d'attention.

Le test doit échouer si un cache est absent, si les dimensions ne correspondent
pas ou si le point d'injection n'est pas validé. Un cache vide ou le cache
produit par une chaîne Claymore dérivée ne constitue pas une entrée de référence.
Les bornes D1 ne seront établies qu'après exécution de toutes les couches.

## Procédure et critères de clôture CPU

1. Versionner le générateur de références et son environnement épinglé. Il doit
   refuser d'écrire dans un répertoire de dorées existant ; la génération se fait
   dans un nouveau répertoire, puis la validation compare les empreintes et les
   métadonnées sans remplacer les références approuvées.
2. Versionner un validateur d'artefacts qui contrôle le manifeste, les tailles,
   les formes, l'ordre des axes, les valeurs finies et les SHA-256.
3. Ajouter les points d'injection vision et décodeur comme API additives,
   en gardant les chemins normaux et CUDA inchangés.
4. Ajouter les deux tests de bisect avec assertions de fidélité des entrées,
   métrique D1 et bornes fixées sur les résultats mesurés.
5. Exécuter en profil release sur l'hôte CPU de référence :
   `golden_cpu`, `golden_cpu_layer_bisect`, puis les deux nouveaux bisects.
   Publier commandes complètes, SHA, versions, logs bruts et empreintes.
6. Rejouer format, tests, Clippy, CI stable et MSRV 1.86 sur le SHA final.

Aucun travail RTX 5090 n'est requis pour ces étapes. La campagne GPU reste
bloquée jusqu'à la clôture de ces critères et au gel du SHA CPU final.
