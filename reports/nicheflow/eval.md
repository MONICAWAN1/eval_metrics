# NicheFlow CFM checkpoint — full evaluation metrics

Full metric suite for the **NicheFlow CFM** checkpoint (`ckpts/NicheFlow_CFM_ABCA.ckpt`), including
the cell-type-classifier groups (`ct/*`). The neutral spatial classifier was trained on a *close but
different* slide (**1.026**) and applied to the **1.025** target, so it never saw the target. Run
through the bundled NicheFlow adapter (`generator=nicheflow`), which generates niche-shaped cells in
the model's native space, so — unlike the OT-CFM baseline — the **regression** group (`x/*`, `pos/*`)
also runs (the niches carry cell-for-cell matched ground truth).

## Setup

| Item                      | Value                                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                     | NicheFlow CFM, `ckpts/NicheFlow_CFM_ABCA.ckpt` — trained on **1.024 → 1.025** (aligned pair, same mouse)                                                           |
| Source slide              | `adata_Zhuang_Zhuang-ABCA-1.024.h5ad`                                                                                                                              |
| Target slide              | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells; 20 cell types over the 1.024∪1.025 vocabulary)                                                                  |
| Generated cells           | `artifacts/nicheflow/generated.h5ad` — **513 niches × 68 points** (niche-shaped, with paired real ground truth)                                                    |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse (1 out-of-vocabulary cell dropped)                                                       |
| Shared feature space      | NicheFlow's shared whitened PCA on source+target, **50 PCs**; per-slide standardized coordinates + standardized `X_pca` (the space the flow trained in)            |
| Classifier                | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `coord_dim=2`, 20 classes — trained in-process on 1.026 via the adapter's `classifier_h5ad` mechanism |

Everything (generated cells, target, and classifier-training slide) lives in NicheFlow's **shared
standardized `X_pca`** space — the model's native representation. The classifier slide (1.026) is
projected through the *same* source+target PCA + label vocabulary, making it neutral and directly
applicable to the target & generated niches.

> **Caveat — alignment.** The checkpoint was trained on a PASTE2-**aligned** 1.024→1.025 pair, but
> the eval adapter's `preprocess_pair` deliberately omits global alignment (per-slide standardized
> coordinates + minibatch-OT pairing fallback). So these numbers reflect the model run under the
> adapter's unaligned preprocessing, which can differ from the original aligned NicheFlow eval.

## Results

All groups ran (none skipped): the niche-shaped output supplies matched ground truth for
`regression`, and the in-process classifier enables `ct/*`.

### Distribution / two-sample (expression + position)

| Metric         | Value  | Notes                                                  |
| -------------- | ------ | ------------------------------------------------------ |
| `c2st/acc`     | 0.9330 | real-vs-generated classifier accuracy (joint expr+pos) |
| `c2st/auc`     | 0.9808 | still separable, but below the OT-CFM baseline's ~1.0  |
| `c2st/pos_acc` | 0.6937 | position-only C2ST                                     |
| `mmd2/x`       | 0.0325 | MMD² on expression                                     |
| `mmd2/pos`     | 0.0139 | MMD² on coordinates                                    |
| `ot_w1/x`      | 7.2282 | Wasserstein-1, expression                              |
| `ot_w2/x`      | 7.3216 | Wasserstein-2, expression                              |
| `ot_w1/pos`    | 0.2753 | Wasserstein-1, coordinates                             |
| `ot_w2/pos`    | 0.3347 | Wasserstein-2, coordinates                             |

### Regression (matched ground truth, niche-shaped)

| Metric    | Value  | Notes                                           |
| --------- | ------ | ----------------------------------------------- |
| `x/mae`   | 1.0725 | per-cell expression error vs matched real niche |
| `x/mse`   | 1.9412 |                                                 |
| `pos/mae` | 0.4997 | per-cell coordinate error                       |
| `pos/mse` | 0.3925 |                                                 |

### Geometry (point-set distances)

| Metric     | Value  |
| ---------- | ------ |
| `psd/mean` | 0.0232 |
| `psd/max`  | 0.3620 |
| `spd/mean` | 0.0619 |
| `spd/max`  | 0.9022 |

### Moran's I (spatial autocorrelation)

| Metric            | Value  | Notes                                                                   |
| ----------------- | ------ | ----------------------------------------------------------------------- |
| `moran/real_mean` | 0.1887 | real slide spatial structure                                            |
| `moran/gen_mean`  | 0.0816 | generated **carries spatial structure** (vs ≈0 for the aspatial OT-CFM) |
| `moran/corr`      | 0.6838 | per-gene Moran correlation real-vs-gen                                  |
| `moran/mae`       | 0.1248 |                                                                         |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric        | Value  | Notes                                                                       |
| ------------- | ------ | --------------------------------------------------------------------------- |
| `ct/acc_real` | 0.2476 | classifier accuracy on **real** 1.025 niches (20-class, neighbourhood-only) |
| `ct/acc_gen`  | 0.3002 | accuracy on **generated** niches (vs the paired real centroid's true label) |
| `ct/acc_gap`  | 0.0526 | \`                                                                          |
| `ct/acc`      | 0.4191 | label agreement between generated and paired-real niches                    |
| `ct/f1`       | 0.3797 | weighted-F1 of that agreement                                               |
| `ct/prop_kl`  | 0.5430 | cell-type composition divergence (KL)                                       |
| `ct/prop_tv`  | 0.2125 | total variation                                                             |
| `ct/prop_jsd` | 0.0614 | Jensen–Shannon                                                              |

**Reading.** NicheFlow CFM generates **niche-shaped cells with genuine spatial structure**
(`moran/gen_mean ≈ 0.08` vs the real `0.19`, Moran corr `0.68`) and low expression-distribution
distance (`mmd2/x ≈ 0.03`). It remains distinguishable from the real slide (`c2st/auc ≈ 0.98`), but
the small classifier accuracy gap (`0.05`) says its niches are about as classifiable as real ones
under a neutral annotator. The classifier's absolute accuracy is modest (`acc_real ≈ 0.25` over 20
masked-centroid classes from a single held-out slide), so read `ct/acc_gap` (real-vs-gen
consistency) rather than the absolute level.

> **Not comparable to the OT-CFM report.** That run used a different target (1.001), a different
> classifier slide (1.002), 16 classes, a raw-gene PCA space, and a flat aspatial output. The two
> reports characterise each model on its own slides; the numbers are not head-to-head.

## Reproduce

NicheFlow's flow model pins `torch==2.5.1`, which conflicts with this repo's venv, and only the
*generation* step imports `nicheflow`. So run the pipeline in **nicheflow_mba's** venv with the
metrics package installed alongside (no code edits to the metrics):

```bash
NF=../nicheflow_mba
DATA=$NF/data
export CUDA_VISIBLE_DEVICES=""   # this box's GPU driver is too old; run on CPU

# one-time: make the metrics package importable in nicheflow's venv (its torch is untouched)
uv pip install --no-deps -e . --python $NF/.venv/bin/python

# generate (1.024 -> 1.025) + evaluate, training the neutral classifier on 1.026, in one shot.
# Outputs route by generator name: artifacts/nicheflow/generated.h5ad + reports/nicheflow/metrics.csv
$NF/.venv/bin/python -m paired_slides_eval.pipeline \
  generator=nicheflow \
  source=$DATA/adata_Zhuang_Zhuang-ABCA-1.024.h5ad \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  checkpoint=$NF/ckpts/NicheFlow_CFM_ABCA.ckpt \
  generator.classifier_h5ad=$DATA/adata_Zhuang_Zhuang-ABCA-1.026.h5ad \
  generator.variant=cfm
```

*(The classifier is trained in-process by the adapter and is not persisted to disk; pass
`generator.classifier_ckpt=<path>` instead to score with a pre-trained one.)*
