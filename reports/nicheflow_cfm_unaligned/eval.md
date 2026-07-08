# NicheFlow CFM (unaligned) — full evaluation metrics

Full metric suite for the **NicheFlow CFM** model trained on the **unaligned** 1.024→1.025 pair
(`abca_12425.pkl`, no PASTE2), evaluated at **step 55000** — the same step as the
`reports/nicheflow_vfm/` run, so the two are directly comparable. Run through the bundled NicheFlow
adapter (`generator=nicheflow generator.variant=cfm`), with the neutral spatial classifier trained on
the held-out slide **1.026** and applied to the **1.025** target. Niche-shaped output, so the
**regression** group (`x/*`, `pos/*`) runs too. Includes the spatially-aware **`c2st_graph`** metric.

## Setup

| Item                      | Value                                                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                     | NicheFlow CFM, `nicheflow_mba/outputs/2026-06-26/16-56-54/checkpoints/last.ckpt` (step 55000)                                                                      |
| Training data             | `abca_12425.pkl` — **1.024 → 1.025**, same mouse, **unaligned** (no PASTE2)                                                                                        |
| Source slide              | `adata_Zhuang_Zhuang-ABCA-1.024.h5ad`                                                                                                                              |
| Target slide              | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells; 20 cell types over the 1.024∪1.025 vocabulary)                                                                  |
| Generated cells           | `artifacts/nicheflow_cfm_unaligned/generated.h5ad` — **513 niches × 68 points** (niche-shaped, with paired real ground truth)                                      |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse                                                                                          |
| Shared feature space      | NicheFlow's shared whitened PCA on source+target, **50 PCs**; per-slide standardized coordinates + standardized `X_pca` (the space the flow trained in)            |
| Classifier                | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `coord_dim=2`, 20 classes — trained in-process on 1.026 via the adapter's `classifier_h5ad` mechanism |

> **No alignment mismatch.** Unlike the original `reports/nicheflow/` CFM report (whose curated
> checkpoint was trained on a PASTE2-**aligned** pair but evaluated unaligned), this checkpoint was
> trained on the **unaligned** 1.024→1.025 pair (`abca_12425.pkl`) — exactly what the eval adapter's
> `preprocess_pair` produces. Training-time and eval-time preprocessing agree, so the earlier
> alignment caveat does **not** apply here. Same setup as the `reports/nicheflow_vfm/` run.

## Results

All groups ran (none skipped): the niche-shaped output supplies matched ground truth for
`regression`, and the in-process classifier enables `ct/*`.

### Distribution / two-sample (expression + position)

| Metric           | Value  | Notes                                                                                   |
| ---------------- | ------ | --------------------------------------------------------------------------------------- |
| `c2st/acc`       | 0.6018 | real-vs-generated classifier accuracy (joint expr+pos) — **near chance**                |
| `c2st/auc`       | 0.6409 |                                                                                         |
| `c2st/graph_acc` | 0.6293 | spatially-aware GCN C2ST (expression-only node features over a joint spatial-kNN graph) |
| `c2st/graph_auc` | 0.6800 | only marginally above the MLP C2ST → expression↔position coupling looks right           |
| `c2st/pos_acc`   | 0.5902 | position-only C2ST                                                                      |
| `mmd2/x`         | 0.0028 | MMD² on expression — very small                                                         |
| `mmd2/pos`       | 0.0080 | MMD² on coordinates                                                                     |
| `ot_w1/x`        | 5.4709 | Wasserstein-1, expression                                                               |
| `ot_w2/x`        | 5.5960 | Wasserstein-2, expression                                                               |
| `ot_w1/pos`      | 0.1921 | Wasserstein-1, coordinates                                                              |
| `ot_w2/pos`      | 0.2279 | Wasserstein-2, coordinates                                                              |

### Regression (matched ground truth, niche-shaped)

| Metric    | Value  |
| --------- | ------ |
| `x/mae`   | 1.0571 |
| `x/mse`   | 1.9171 |
| `pos/mae` | 0.5528 |
| `pos/mse` | 0.5015 |

### Geometry (point-set distances)

| Metric     | Value  |
| ---------- | ------ |
| `psd/mean` | 0.0199 |
| `psd/max`  | 0.1560 |
| `spd/mean` | 0.0178 |
| `spd/max`  | 0.3617 |

### Moran's I (spatial autocorrelation)

| Metric            | Value  | Notes                                         |
| ----------------- | ------ | --------------------------------------------- |
| `moran/real_mean` | 0.1887 | real slide spatial structure                  |
| `moran/gen_mean`  | 0.1118 | generated carries spatial structure           |
| `moran/corr`      | 0.9374 | per-gene Moran correlation real-vs-gen — high |
| `moran/mae`       | 0.0769 |                                               |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric        | Value  | Notes                                                                       |
| ------------- | ------ | --------------------------------------------------------------------------- |
| `ct/acc_real` | 0.3236 | classifier accuracy on **real** 1.025 niches (20-class, neighbourhood-only) |
| `ct/acc_gen`  | 0.3255 | accuracy on **generated** niches (vs the paired real centroid's true label) |
| `ct/acc_gap`  | 0.0019 | \`                                                                          |
| `ct/acc`      | 0.5439 | label agreement between generated and paired-real niches                    |
| `ct/f1`       | 0.4941 | weighted-F1 of that agreement                                               |
| `ct/prop_kl`  | 0.1134 | cell-type composition divergence (KL)                                       |
| `ct/prop_tv`  | 0.1930 | total variation                                                             |
| `ct/prop_jsd` | 0.0246 | Jensen–Shannon                                                              |

**Reading.** Trained directly on the unaligned pair, the CFM is strong: generated cells are
**near-inseparable from real** (`c2st/acc ≈ 0.60`, `c2st/auc ≈ 0.64`, and the graph C2ST only `0.68`
AUC), expression marginals match tightly (`mmd2/x ≈ 0.003`), and the **spatial structure is well
reproduced** (`moran/gen_mean 0.112` vs real `0.189`, Moran corr `0.94`). The classifier accuracy gap
is negligible (`0.0019`) with moderate label agreement (`ct/acc ≈ 0.54`).

> **Comparison note.** Shares the **target (1.025)**, **classifier slide (1.026)**, feature space
> (whitened shared PCA), output shape, and the unaligned `abca_12425.pkl` training pair with the
> `reports/nicheflow_vfm/` run, and both are evaluated at the same step — so the two are directly
> head-to-head. Absolute distances are **not** comparable to the OT-CFM/1.025 reports, which use a
> raw-gene PCA + flat aspatial output rather than the whitened shared-PCA + niche-shaped output here.

## Reproduce

```bash
NF=../nicheflow_mba
DATA=$NF/data

# one-time: make the metrics package importable in nicheflow's venv (its torch is untouched)
uv pip install --no-deps -e . --python $NF/.venv/bin/python

# generate (1.024 -> 1.025) + evaluate, training the neutral classifier on 1.026, in one shot.
$NF/.venv/bin/python -m paired_slides_eval.pipeline \
  generator=nicheflow generator.variant=cfm \
  source=$DATA/adata_Zhuang_Zhuang-ABCA-1.024.h5ad \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  checkpoint=$NF/outputs/2026-06-26/16-56-54/checkpoints/last.ckpt \
  generator.classifier_h5ad=$DATA/adata_Zhuang_Zhuang-ABCA-1.026.h5ad \
  out=reports/nicheflow_cfm_unaligned/metrics.csv \
  generated_out=artifacts/nicheflow_cfm_unaligned/generated.h5ad
```
