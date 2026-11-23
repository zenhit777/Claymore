# M8 — Différences numériques CPU / CUDA (F32, BF16)

> **Objet :** documenter précisément ce que font réellement les opérations de
> référence CPU et les kernels CUDA M8, sans inventer de tolérance ni affirmer
> une identité bit à bit qui n'est pas garantie.
>
> **Statut :** document d'audit descriptif ; il ne modifie ni le comportement
> ni les critères d'acceptation définis dans [docs/m8-architecture.md](../m8-architecture.md).

## 1. Périmètre et sources inspectées

| Côté | Fichiers réellement lus |
|---|---|
| CPU | `crates/claymore-tensor/src/ops.rs` (`add`, `matmul`, `softmax_last_dim`, `rms_norm`), `crates/claymore-tensor/src/bf16.rs` |
| CUDA | `crates/claymore-cuda/kernels/m8_f32.cu`, `crates/claymore-cuda/kernels/m8_bf16.cu` |
| PTX intégré | `crates/claymore-cuda/src/kernels/m8_f32.ptx`, `crates/claymore-cuda/src/kernels/m8_bf16.ptx` |
| Lancement | `crates/claymore-cuda/src/compute.rs`, `crates/claymore-memory/src/cuda.rs` (`launch_tensor_op`) |

Les huit kernels CUDA (quatre F32, quatre BF16) sont chargés depuis deux modules
PTX via `cuModuleLoadData` ; la provenance et les empreintes de ces artefacts
sont consignées dans [m8-ptx-provenance.md](m8-ptx-provenance.md).

## 2. Politique numérique commune

Les deux backends suivent la même règle déclarée dans `ops.rs` et dans
`m8_bf16.cu` :

1. les valeurs BF16 sont élargies en FP32 de façon **bit-exacte** :
   `Bf16::to_f32` décale le motif 16 bits de 16 positions (avec mise en silence
   des NaN), et le CUDA utilise `__bfloat162float` (PTX `cvt.f32.bf16`) ;
2. **tous** les intermédiaires — y compris l'accumulation du matmul et chaque
   réduction — sont évalués en FP32 ;
3. le résultat final seulement est arrondi en BF16 avec **round-to-nearest-even**
   : `Bf16::from_f32` (RNE explicite sur les bits) côté CPU,
   `__float2bfloat16_rn` (PTX `cvt.rn.bf16.f32`) côté CUDA ;
4. aucun cast implicite, broadcasting, repli CPU ou aller-retour host n'est
   autorisé. Seuls F32 et BF16 sont supportés ; `DType::F16` reste une
   métadonnée sans chemin de calcul ;
5. il n'existe **aucun kernel croisé** : `launch_tensor_op` sélectionne le
   kernel de façon exacte sur le couple `(dtype, ShapeSignature)`. Un kernel
   F32 ne traite jamais de données BF16, et réciproquement.

## 3. Comportement par opération

| Opération | Référence CPU (`ops.rs`) | Kernel CUDA (source `.cu` / PTX) | Conséquence numérique |
|---|---|---|---|
| `add` | `x + y` en FP32 | `a[index] + b[index]` → PTX `add.f32` ; BF16 : `cvt.f32.bf16` → `add.f32` → `cvt.rn.bf16.f32` | Même opération IEEE-754 ; un seul arrondi de sortie en BF16. |
| `matmul` | `acc = a.mul_add(b, acc)` pour `inner = 0..k` (FMA, ordre des indices croissant) | `fmaf(a_row[k], b[k*cols+col], acc)` → PTX `fma.rn.f32`, même ordre `k = 0..inner` ; nvcc déroule par 8 mais conserve la chaîne de dépendance | FMA fusionné des deux côtés, même ordre de sommation ; l'écart observé sur la cible est nul (Cf. §6), sans que cela constitue une garantie bit à bit sur toute plateforme. |
| `softmax_last_dim` | maximum par `fold(f32::NEG_INFINITY, f32::max)`, puis `sum` séquentiel de `(x-max).exp()`, puis `(x-max).exp() / sum` | maximum et somme par partials par thread + réduction en arbre en mémoire partagée (`bar.sync`), `expf(x-max)`, puis `rcp.rn.f32` et `mul.f32` | Ordres de réduction différents (séquentiel contre arbre), division contre réciproque multiplié, et surtout implémentations d'exponentielle différentes : **pas d'identité bit à bit attendue**. |
| `rms_norm` | `sum::<f32>(x*x)` séquentiel, `/ width`, `sqrt().recip()`, puis `x * inv_rms * scale[index]` | `fmaf(x, x, local)` puis réduction en arbre, `div.rn.f32(sum, width)`, `sqrt.rn.f32`, `rcp.rn.f32`, puis deux `mul.f32` | Réduction séquentielle contre arbre, et `x*x` séparé contre FMA fusionné : résultats proches mais non identiques bit à bit. |

## 4. Points de divergence détaillés

### 4.1 Exponentielle : `expf` CUDA, pas de fast-math global, mais approximation matérielle

- Les sources CUDA appellent `expf` (et **non** `__expf` rapide). La commande de
  régénération documentée (Cf. [m8-ptx-provenance.md](m8-ptx-provenance.md)) ne
  contient ni `--use_fast_math`, ni `-ftz=true`, ni `-ffast-math`.
- Le PTX généré montre néanmoins que l'`expf` de nvcc 13.0 est une approximation
  : réduction de domaine par polynôme (`fma.rn.f32`, `fma.rm.f32`,
  `cvt.sat.f32.f32`) puis **`ex2.approx.ftz.f32`**. Cette instruction est
  approximative et son qualificatif `.ftz` vide les dénormaux de son entrée.
- Côté CPU, `f32::exp` de Rust est résolu vers la bibliothèque mathématique de
  l'hôte (libm), dont l'implémentation est différente de la séquence PTX ci-dessus.
- Le CUDA calcule ensuite `inv_sum = 1.0f / sum` (PTX `rcp.rn.f32`) et multiplie
  (`mul.f32`), alors que le CPU calcule directement `exp / sum`. La différence
  d'arrondi s'ajoute à la différence d'approximation de `exp`.

**Conclusion :** aucun résultat de softmax n'est promu bit-identique entre CPU
et CUDA ; les écarts mesurés sont ceux des campagnes matérielles (Cf. §6).

### 4.2 Ordre des réductions

- CPU softmax : `fold` du maximum sur les éléments dans l'ordre du tampon, puis
  `Iterator::sum` des exponentielles dans le même ordre.
- CUDA softmax : un bloc par ligne, 256 threads, chaque thread accumule ses
  propres éléments avec un pas de `blockDim.x`, puis réduction en arbre dans
  `shared[]` (8 niveaux pour 256 threads) avec `__syncthreads()` entre chaque
  niveau ; le même motif est utilisé pour le maximum et pour la somme.
- CPU RMSNorm : `map(|x| x*x).sum::<f32>()` séquentiel.
- CUDA RMSNorm : `fmaf(x, x, local)` par thread, puis réduction en arbre
  identique dans `shared[]`.

L'addition flottante n'étant pas associative, l'ordre change le résultat, même
quand toutes les opérations individuelles sont IEEE-754.

### 4.3 Sous-normaux

- Aucune instruction arithmétique du PTX intégré ne porte de qualificatif `.ftz`
  sur `add.f32`, `fma.rn.f32`, `mul.f32`, `max.f32`, `div.rn.f32`,
  `sqrt.rn.f32`, `rcp.rn.f32` ou `cvt.f32.bf16` / `cvt.rn.bf16.f32` : ces
  opérations suivent donc la sémantique IEEE-754 des sous-normaux sur `sm_120`,
  sans vidage global des dénormaux à la compilation.
- L'exception est l'`ex2.approx.ftz.f32` issu de l'abaissement de `expf` : les
  entrées sous-normales de l'exponentielle (typiquement des `x - max` très
  négatifs) sont traitées comme zéro. Le CPU n'emprunte pas cette voie.
- La conversion BF16 du CPU documente et teste explicitement les sous-normaux
  : `Bf16::from_f32(2^-134) -> 0x0000`, `0x0001 -> 2^-133` (plus petit
  sous-normal), transition sous-normal → normal et débordement vers l'infini
  selon IEEE-754 (voir `crates/claymore-tensor/src/bf16.rs` et
  `docs/m8-architecture.md`). Le code CUDA, lui, n'a pas de branche dédiée aux
  sous-normaux : il s'appuie sur `cvt.f32.bf16` / `cvt.rn.bf16.f32`.
- Aucun test ne revendique l'identité bit à bit pour des entrées sous-normales
  traversant l'exponentielle ; ne pas en déduire une garantie.

### 4.4 NaN, zéro signé, cas spéciaux

- CPU : `Bf16::from_f32` canonicalise **tout** NaN en `0x7FC0` ;
  `Bf16::to_f32` met en silence un NaN BF16 signalant. Le zéro signé est
  préservé dans les deux sens.
- CUDA : les sources `.cu` n'ajoutent aucune canonicalisation explicite des NaN
  ; le résultat dépend de `cvt.rn.bf16.f32` sur le matériel. La méthode de
  comparaison BF16 documentée exige la présence de NaN des deux côtés, sans
  comparer les charges utiles (payloads) de NaN.
- Les tests matériels BF16 couvrent les valeurs spéciales pour `add` et les
  logits `±10 000` pour `softmax` ; ils ne constituent pas une preuve
  d'identité des bits de NaN ou des dénormaux.

## 5. Ce que le code ne fait pas

- pas de Tensor Core ni de matmul tuilé : un thread par élément de sortie et une
  boucle sur `K` (les performances relèvent de M12) ;
- pas de `__expf`, pas de `--use_fast_math`, pas de FTZ global à la compilation ;
- pas de réduction déterministe identique des deux côtés (arbre CUDA contre
  séquentiel CPU) ;
- pas de tolérance nouvelle définie dans ce document, pas de revendication
  d'identité bit à bit globale.

## 6. Preuves et tolérances existantes (non modifiées)

Les seuils sont ceux de `docs/m8-architecture.md`, §« Tolérances numériques
documentées » et §« Tolérances BF16 » ; ils sont rappelés ici pour éviter toute
interprétation nouvelle, et non redéfinis :

- F32 : add `atol 1e-5 / rtol 1e-6`, matmul `1e-3 / 2e-5`, softmax `1e-6 / 1e-5`,
  rms_norm `1e-5 / 2e-5`.
- BF16 : comparaison en pas représentables, limite 0 pas pour `add`, 1 pas pour
  `matmul`, `softmax` et `rms_norm` ; somme par ligne de softmax dans `4e-3` de 1.

Résultats réellement observés lors des deux campagnes RTX 5090 :

- F32 : add et matmul `0e0` ; softmax jusqu'à `2,98e-8` et `4,44e-16` ; rms_norm
  jusqu'à `4,77e-7` et `3,58e-7` (`docs/audits/m8-rtx5090-hardware-validation.md`, §5).
- BF16 : `0 pas` sur toutes les shapes des deux campagnes
  (`docs/audits/m8-bf16-rtx5090-hardware-validation.md`, §5).

Ces résultats sont des mesures sur la cible et la chaîne d'outils documentées,
pas une propriété algébrique des opérations.
