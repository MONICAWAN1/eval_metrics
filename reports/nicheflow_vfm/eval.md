# NicheFlow VFM checkpoint — full evaluation metrics

Full metric suite for the **NicheFlow VFM** model, using the **latest checkpoint from the currently
running training job** (`outputs/2026-06-25/00-54-44/checkpoints/last.ckpt`, **step 55000, epoch 0 —
a mid-training snapshot**, not a final model). Run through the bundled NicheFlow adapter
(`generator=nicheflow generator.variant=vfm`), with the neutral spatial classifier trained on the
held-out slide **1.026** (the same one used for the CFM and OT-CFM runs) and applied to the **1.025**
target. As with CFM, the niche-shaped output means the **regression** group (`x/*`, `pos/*`) runs too.

## Setup

| Item                      | Value                                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                     | NicheFlow VFM (`vfm_objective=GLVFM`), `nicheflow_mba/outputs/2026-06-25/00-54-44/checkpoints/last.ckpt` — **mid-training, step 55000**                            |
| Training data             | `abca_12425.pkl` — **1.024 → 1.025**, same mouse, **unaligned** (no PASTE2)                                                                                        |
| Source slide              | `adata_Zhuang_Zhuang-ABCA-1.024.h5ad`                                                                                                                              |
| Target slide              | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells; 20 cell types over the 1.024∪1.025 vocabulary)                                                                  |
| Generated cells           | `artifacts/nicheflow_vfm/generated.h5ad` — **513 niches × 68 points** (niche-shaped, with paired real ground truth)                                                |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse                                                                                          |
| Shared feature space      | NicheFlow's shared whitened PCA on source+target, **50 PCs**; per-slide standardized coordinates + standardized `X_pca` (the space the flow trained in)            |
| Classifier                | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `coord_dim=2`, 20 classes — trained in-process on 1.026 via the adapter's `classifier_h5ad` mechanism |

> **No alignment mismatch (unlike the CFM report).** This VFM checkpoint was trained on the
> **unaligned** 1.024→1.025 pair (`abca_12425.pkl`), which is exactly what the eval adapter's
> `preprocess_pair` produces (per-slide standardized coordinates, no PASTE2). So training-time and
> eval-time preprocessing agree here — the CFM run's alignment caveat does **not** apply.

## Results

All groups ran (none skipped): the niche-shaped output supplies matched ground truth for
`regression`, and the in-process classifier enables `ct/*`.

### Distribution / two-sample (expression + position)

| Metric           | Value  | Notes                                                                                                     |
| ---------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| `c2st/acc`       | 0.5935 | real-vs-generated classifier accuracy — **near chance (0.5)**: generated cells are hard to tell from real |
| `c2st/auc`       | 0.6188 |                                                                                                           |
| `c2st/graph_acc` | 0.6088 | spatially-aware GCN C2ST (expression-only node features over a joint spatial-kNN graph)                   |
| `c2st/graph_auc` | 0.6519 | near the MLP C2ST — generated niches stay hard to separate even under the graph view                      |
| `c2st/pos_acc`   | 0.5813 | position-only C2ST                                                                                        |
| `mmd2/x`         | 0.0030 | MMD² on expression — very small                                                                           |
| `mmd2/pos`       | 0.0279 | MMD² on coordinates                                                                                       |
| `ot_w1/x`        | 5.2066 | Wasserstein-1, expression                                                                                 |
| `ot_w2/x`        | 5.3829 | Wasserstein-2, expression                                                                                 |
| `ot_w1/pos`      | 0.2142 | Wasserstein-1, coordinates                                                                                |
| `ot_w2/pos`      | 0.2771 | Wasserstein-2, coordinates                                                                                |

### Regression (matched ground truth, niche-shaped)

| Metric    | Value  |
| --------- | ------ |
| `x/mae`   | 1.0624 |
| `x/mse`   | 1.9590 |
| `pos/mae` | 0.5322 |
| `pos/mse` | 0.4506 |

### Geometry (point-set distances)

| Metric     | Value  |
| ---------- | ------ |
| `psd/mean` | 0.0196 |
| `psd/max`  | 0.1640 |
| `spd/mean` | 0.0138 |
| `spd/max`  | 0.2250 |

### Moran's I (spatial autocorrelation)

| Metric            | Value  | Notes                                                          |
| ----------------- | ------ | -------------------------------------------------------------- |
| `moran/real_mean` | 0.1887 | real slide spatial structure                                   |
| `moran/gen_mean`  | 0.1752 | generated **closely matches** the real spatial autocorrelation |
| `moran/corr`      | 0.9811 | per-gene Moran correlation real-vs-gen — very high             |
| `moran/mae`       | 0.0230 | small                                                          |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric        | Value  | Notes                                                                       |
| ------------- | ------ | --------------------------------------------------------------------------- |
| `ct/acc_real` | 0.2924 | classifier accuracy on **real** 1.025 niches (20-class, neighbourhood-only) |
| `ct/acc_gen`  | 0.3080 | accuracy on **generated** niches (vs the paired real centroid's true label) |
| `ct/acc_gap`  | 0.0156 | \`                                                                          |
| `ct/acc`      | 0.6062 | label agreement between generated and paired-real niches                    |
| `ct/f1`       | 0.5715 | weighted-F1 of that agreement                                               |
| `ct/prop_kl`  | 0.1584 | cell-type composition divergence (KL)                                       |
| `ct/prop_tv`  | 0.1404 | total variation                                                             |
| `ct/prop_jsd` | 0.0164 | Jensen–Shannon                                                              |

**Reading.** Even mid-training (step 55000), the VFM is strong: generated cells are **near-inseparable
from real** (`c2st/acc ≈ 0.59`, `c2st/auc ≈ 0.62`), expression marginals match tightly
(`mmd2/x ≈ 0.003`), and the **spatial structure is well reproduced** (`moran/gen_mean 0.175` vs real
`0.189`, Moran corr `0.98`). The classifier accuracy gap is negligible (`0.016`) and label agreement
is high (`ct/acc ≈ 0.61`, `ct/f1 ≈ 0.57`). Caveat: this is a non-final checkpoint from a running job;
numbers may shift as training continues.

> **Comparison note.** Shares the **target (1.025)** and **classifier slide (1.026)** with the
> NicheFlow CFM and the OT-CFM/1.025 reports. The CFM report carries an alignment caveat (its curated
> checkpoint was trained aligned, evaluated unaligned); this VFM run does not, so its numbers are a
> cleaner read of the adapter pipeline. Absolute distances still aren't comparable to the OT-CFM
> report, which uses a raw-gene PCA + flat aspatial output rather than the whitened shared-PCA +
> niche-shaped output here.

## Reproduce

```bash
NF=../nicheflow_mba
DATA=$NF/data
export CUDA_VISIBLE_DEVICES=""   # this box's GPU driver is too old for the eval venv's torch; run on CPU

# one-time: make the metrics package importable in nicheflow's venv (its torch is untouched)
uv pip install --no-deps -e . --python $NF/.venv/bin/python

# generate (1.024 -> 1.025) + evaluate, training the neutral classifier on 1.026, in one shot.
# `variant=vfm` + `vfm_objective=GLVFM` (the adapter default) must match the checkpoint's config.
$NF/.venv/bin/python -m paired_slides_eval.pipeline \
  generator=nicheflow generator.variant=vfm \
  source=$DATA/adata_Zhuang_Zhuang-ABCA-1.024.h5ad \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  checkpoint=$NF/outputs/2026-06-25/00-54-44/checkpoints/last.ckpt \
  generator.classifier_h5ad=$DATA/adata_Zhuang_Zhuang-ABCA-1.026.h5ad \
  out=reports/nicheflow_vfm/metrics.csv \
  generated_out=artifacts/nicheflow_vfm/generated.h5ad
```

*(The latest available checkpoint was used because the VFM training job is still running; point
`checkpoint=` at a newer `last.ckpt`/`ckpt` to re-evaluate as training progresses.)*
