# NicheFlow CFM (unaligned) — full evaluation metrics

Full metric suite for the **NicheFlow CFM** model trained on the **unaligned** 1.024→1.025 pair
(`abca_12425.pkl`, no PASTE2), using the **latest checkpoint from the currently running training job**
(`outputs/2026-06-26/16-56-54/checkpoints/last.ckpt`, **global_step 32000, epoch 0 — a mid-training
snapshot**, not a final model). Run through the bundled NicheFlow adapter (`generator=nicheflow
generator.variant=cfm`), with the neutral spatial classifier trained on the held-out slide **1.026**
and applied to the **1.025** target. Niche-shaped output, so the **regression** group (`x/*`,
`pos/*`) runs too. Includes the spatially-aware **`c2st_graph`** metric.

## Setup

| Item | Value |
|---|---|
| Model | NicheFlow CFM, `nicheflow_mba/outputs/2026-06-26/16-56-54/checkpoints/last.ckpt` — **mid-training, step 32000** |
| Training data | `abca_12425.pkl` — **1.024 → 1.025**, same mouse, **unaligned** (no PASTE2) |
| Source slide | `adata_Zhuang_Zhuang-ABCA-1.024.h5ad` |
| Target slide | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells; 20 cell types over the 1.024∪1.025 vocabulary) |
| Generated cells | `artifacts/nicheflow_cfm_unaligned/generated.h5ad` — **513 niches × 68 points** (niche-shaped, with paired real ground truth) |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse |
| Shared feature space | NicheFlow's shared whitened PCA on source+target, **50 PCs**; per-slide standardized coordinates + standardized `X_pca` (the space the flow trained in) |
| Classifier | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `coord_dim=2`, 20 classes — trained in-process on 1.026 via the adapter's `classifier_h5ad` mechanism |

> **No alignment mismatch.** Unlike the original `reports/nicheflow/` CFM report (whose curated
> checkpoint was trained on a PASTE2-**aligned** pair but evaluated unaligned), this checkpoint was
> trained on the **unaligned** 1.024→1.025 pair (`abca_12425.pkl`) — exactly what the eval adapter's
> `preprocess_pair` produces. Training-time and eval-time preprocessing agree, so the earlier
> alignment caveat does **not** apply here. Same setup as the `reports/nicheflow_vfm/` run.

## Results

All groups ran (none skipped): the niche-shaped output supplies matched ground truth for
`regression`, and the in-process classifier enables `ct/*`.

### Distribution / two-sample (expression + position)

| Metric | Value | Notes |
|---|---|---|
| `c2st/acc` | 0.5787 | real-vs-generated classifier accuracy (joint expr+pos) — **near chance** |
| `c2st/auc` | 0.6146 | |
| `c2st/graph_acc` | 0.6100 | spatially-aware GCN C2ST (expression-only node features over a joint spatial-kNN graph) |
| `c2st/graph_auc` | 0.6637 | only marginally above the MLP C2ST → expression↔position coupling looks right |
| `c2st/pos_acc` | 0.5932 | position-only C2ST |
| `mmd2/x` | 0.0024 | MMD² on expression — very small |
| `mmd2/pos` | 0.0119 | MMD² on coordinates |
| `ot_w1/x` | 5.5267 | Wasserstein-1, expression |
| `ot_w2/x` | 5.6487 | Wasserstein-2, expression |
| `ot_w1/pos` | 0.2258 | Wasserstein-1, coordinates |
| `ot_w2/pos` | 0.2822 | Wasserstein-2, coordinates |

### Regression (matched ground truth, niche-shaped)

| Metric | Value |
|---|---|
| `x/mae` | 1.0590 |
| `x/mse` | 1.9194 |
| `pos/mae` | 0.5347 |
| `pos/mse` | 0.4665 |

### Geometry (point-set distances)

| Metric | Value |
|---|---|
| `psd/mean` | 0.0203 |
| `psd/max` | 0.1556 |
| `spd/mean` | 0.0202 |
| `spd/max` | 0.3738 |

### Moran's I (spatial autocorrelation)

| Metric | Value | Notes |
|---|---|---|
| `moran/real_mean` | 0.1887 | real slide spatial structure |
| `moran/gen_mean` | 0.1039 | generated carries spatial structure |
| `moran/corr` | 0.9254 | per-gene Moran correlation real-vs-gen — high |
| `moran/mae` | 0.0848 | |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric | Value | Notes |
|---|---|---|
| `ct/acc_real` | 0.2807 | classifier accuracy on **real** 1.025 niches (20-class, neighbourhood-only) |
| `ct/acc_gen` | 0.3177 | accuracy on **generated** niches (vs the paired real centroid's true label) |
| `ct/acc_gap` | 0.0370 | `|acc_real − acc_gen|` — small gap → generated niches about as classifiable as real |
| `ct/acc` | 0.6296 | label agreement between generated and paired-real niches |
| `ct/f1` | 0.5909 | weighted-F1 of that agreement |
| `ct/prop_kl` | 0.3044 | cell-type composition divergence (KL) |
| `ct/prop_tv` | 0.1696 | total variation |
| `ct/prop_jsd` | 0.0238 | Jensen–Shannon |

**Reading.** Trained directly on the unaligned pair, even at step 32000 the CFM is strong:
generated cells are **near-inseparable from real** (`c2st/acc ≈ 0.58`, `c2st/auc ≈ 0.61`, and the
graph C2ST only `0.66` AUC), expression marginals match tightly (`mmd2/x ≈ 0.002`), and the
**spatial structure is well reproduced** (`moran/gen_mean 0.104` vs real `0.189`, Moran corr `0.93`).
The classifier accuracy gap is small (`0.037`) with high label agreement (`ct/acc ≈ 0.63`). Caveat:
this is a non-final checkpoint from a running job; numbers may shift as training continues.

> **Comparison note.** Shares the **target (1.025)** and **classifier slide (1.026)** with the
> `reports/nicheflow_vfm/` run, and both are trained on the same **unaligned** `abca_12425.pkl`, so
> the two are directly comparable on this axis. **Step caveat:** this CFM snapshot is at
> **global_step 32000**, while the VFM report's frozen generation is at **step 55000** — different
> training jobs at different steps, so head-to-head differences partly reflect that, not just the
> CFM-vs-VFM objective. Re-evaluate both at a matched, later step as training progresses.

## Reproduce

```bash
NF=../nicheflow_mba
DATA=$NF/data
export CUDA_VISIBLE_DEVICES=""   # this box's GPU driver is too old for the eval venv's torch; run on CPU

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

*(The latest available checkpoint was used because the CFM training job is still running; point
`checkpoint=` at a newer `last.ckpt` to re-evaluate as training progresses.)*
