# OT-CFM (1.025) vs NicheFlow CFM vs NicheFlow VFM — metric comparison

Back-to-back metric comparison on the **same target slide (1.025)** with the **same neutral
classifier slide (1.026)**:

- **OT-CFM (1.025)** — `reports/otcfm_1025/metrics.csv`. Unconditional, expression-only baseline;
  flat whole-slide output with **random placeholder coordinates** (aspatial); raw-gene PCA, 18 classes.
- **NicheFlow CFM** — `reports/nicheflow/metrics.csv`. Niche-shaped output; whitened shared PCA,
  20 classes.
- **NicheFlow VFM** — `reports/nicheflow_vfm/metrics.csv`. Same as CFM but the VFM variant, from a
  **mid-training snapshot** (step 55000, epoch 0 — not a final model); whitened shared PCA, 20 classes.

> **Comparability.**
>
> - **NicheFlow CFM vs VFM are directly comparable** — identical target, classifier, feature space
>   (whitened shared PCA), and output shape. (Note: CFM was trained on an *aligned* pair but
>   evaluated under the adapter's *unaligned* preprocessing; VFM was trained unaligned, matching eval,
>   but is a mid-training checkpoint.)
> - **OT-CFM vs the NicheFlow models is not strictly head-to-head**: OT-CFM uses a raw-gene PCA and a
>   flat aspatial output (random coordinates, 18 classes). So absolute `ot_*/x` (different PCA scale),
>   `psd`/`spd` and `*/pos` (random coords), and `ct/acc_real` (different classifier basis/classes)
>   are confounded for OT-CFM — marked ⚠.
>
> Lower is better unless noted.

## Distribution / two-sample (expression + position)

| Metric           | OT-CFM (1.025) | NicheFlow CFM | NicheFlow VFM | Expected                           |
| ---------------- | -------------- | ------------- | ------------- | ---------------------------------- |
| `c2st/acc` ↓     | 0.998          | 0.933         | 0.594         | →0.5 = indistinguishable from real |
| `c2st/auc` ↓     | 1.000          | 0.981         | 0.619         | →0.5                               |
| `c2st/pos_acc` ↓ | 0.634          | 0.694         | 0.581         | →0.5 (position-only)               |
| `mmd2/x` ↓       | 0.369          | 0.033         | 0.003         | →0 = matched expression marginals  |
| `mmd2/pos` ↓     | 0.073          | 0.014         | 0.028         | →0                                 |
| `ot_w1/x` ↓ ⚠    | 29.07          | 7.23          | 5.21          | →0 (PCA-scale dependent)           |
| `ot_w2/x` ↓ ⚠    | 35.94          | 7.32          | 5.38          | →0                                 |
| `ot_w1/pos` ↓    | 0.281          | 0.275         | 0.214         | →0                                 |
| `ot_w2/pos` ↓    | 0.337          | 0.335         | 0.277         | →0                                 |

## Geometry — point-set distances ⚠ (positions; OT-CFM coords are random placeholders)

| Metric       | OT-CFM (1.025) | NicheFlow CFM | NicheFlow VFM | Expected                         |
| ------------ | -------------- | ------------- | ------------- | -------------------------------- |
| `psd/mean` ↓ | 0.050          | 0.023         | 0.020         | generated lands on real manifold |
| `psd/max` ↓  | 0.606          | 0.362         | 0.164         | low worst-case                   |
| `spd/mean` ↓ | 0.013          | 0.062         | 0.014         | real cloud is covered            |
| `spd/max` ↓  | 0.044          | 0.902         | 0.225         | low worst-case gap               |

## Moran's I — spatial autocorrelation

| Metric                       | OT-CFM (1.025) | NicheFlow CFM | NicheFlow VFM | Expected                   |
| ---------------------------- | -------------- | ------------- | ------------- | -------------------------- |
| `moran/real_mean`            | 0.257          | 0.189         | 0.189         | reference (real structure) |
| `moran/gen_mean` → real_mean | −0.000         | 0.082         | 0.175         | match `real_mean`          |
| `moran/corr` ↑               | 0.096          | 0.684         | 0.981         | →1 (per-gene match)        |
| `moran/mae` ↓                | 0.257          | 0.125         | 0.023         | →0                         |

## Cell-type classifier `ct/*` (neutral 1.026 classifier; read gaps/agreement over absolutes)

| Metric          | OT-CFM (1.025) | NicheFlow CFM | NicheFlow VFM | Expected                           |
| --------------- | -------------- | ------------- | ------------- | ---------------------------------- |
| `ct/acc_real` ⚠ | 0.524          | 0.248         | 0.292         | classifier sanity on real niches   |
| `ct/acc_gen` ↑  | 0.340          | 0.300         | 0.308         | high = generated classifiable      |
| `ct/acc_gap` ↓  | 0.184          | 0.053         | 0.016         | →0 = gen as classifiable as real   |
| `ct/acc` ↑      | 0.531          | 0.419         | 0.606         | →1 = labels agree with paired-real |
| `ct/f1` ↑       | 0.368          | 0.380         | 0.571         | →1                                 |
| `ct/prop_kl` ↓  | 7.31           | 0.543         | 0.158         | →0 = composition matches           |
| `ct/prop_tv` ↓  | 0.469          | 0.212         | 0.140         | →0                                 |
| `ct/prop_jsd` ↓ | 0.199          | 0.061         | 0.016         | →0                                 |

## Regression — matched ground truth (niche-shaped models only)

| Metric      | OT-CFM (1.025) | NicheFlow CFM | NicheFlow VFM | Expected                 |
| ----------- | -------------- | ------------- | ------------- | ------------------------ |
| `x/mae` ↓   | — (skipped)    | 1.073         | 1.062         | →0 vs matched real niche |
| `x/mse` ↓   | — (skipped)    | 1.941         | 1.959         | →0                       |
| `pos/mae` ↓ | — (skipped)    | 0.500         | 0.532         | →0                       |
| `pos/mse` ↓ | — (skipped)    | 0.392         | 0.451         | →0                       |

_Sources: `reports/otcfm_1025/metrics.csv`, `reports/nicheflow/metrics.csv`, `reports/nicheflow_vfm/metrics.csv`._
