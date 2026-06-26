# Naive-spatial OT-CFM (trained on 1.025) — full evaluation metrics

A **naive spatial baseline**: the same OT-CFM as [`otcfm_1025`](../otcfm_1025/eval.md), but trained
over the **concatenated** `[whitened-PCA expression | standardized 2-D coords]` vector
(`train_cfm_spatial.py --spatial_key spatial`), so it **jointly generates spatial coordinates**
instead of receiving random placeholders. This is the most naive way to make the expression-only
OT-CFM spatial — no OT weighting between the two blocks, no conditioning, just concatenation.

Same target (**1.025**) and the same neutral **1.026**-trained spatial classifier as `otcfm_1025`,
so it reads directly against that aspatial baseline and the NicheFlow CFM report. The classifier is
generator-independent (trained on real 1.026 cells), so it is **reused as-is** — no retraining.

## Setup

| Item | Value |
|---|---|
| Model | naive-spatial OT-CFM (`fm_mnist`), `outputs/cfm_mouse_pca5_1025_spatial/ckpt_final.pt` |
| Training | `scripts/train_cfm_spatial.py`, config identical to `cfm_mouse_pca5_1025` **plus `--spatial_key spatial`**: `n_pcs=5` → model `dim=7` (5 expression + 2 coords), `hidden=512`, `steps=20000`, `batch=256`, `lr=2e-3`, `interpolant=linear`, `coupling=ot`, `ot_method=exact`, `whiten=True`, `test_frac=0.1`, `seed=0`. **GPU** (allocate via SLURM). |
| Generated cells | `artifacts/otcfm_spatial_1025/generated.h5ad` — 9962 flat cells in gene space, **model-generated coordinates** (un-standardized from the coord tail), via `generator=otcfm_spatial` (`coord_mode=generate`) |
| Target slide | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells, 1122 genes, 18 cell types) |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse |
| Shared feature space | PCA fit on the target's raw genes, **50 PCs** (`--n_pcs 50`); generated cells projected into it |
| Classifier checkpoint | `artifacts/otcfm_1025/classifier.ckpt` — **reused** from the `otcfm_1025` report (1.026-trained, 18-class, raw-gene-PCA + raw coords) |

## Results

> **Status: not yet run.** The values below are filled by the `evaluate` step in *Reproduce*
> (`reports/otcfm_spatial_1025/metrics.csv`). Training requires a GPU allocation (left to the user).

`regression` is skipped (flat whole-slide output, no cell-for-cell matched ground truth); the `ct/*`
groups reconstruct paired niches from geometry.

### Distribution / two-sample (expression + position)

| Metric | Value | Notes |
|---|---|---|
| `c2st/acc` | _TBD_ | MLP real-vs-generated classifier accuracy (joint expr+pos) |
| `c2st/auc` | _TBD_ | |
| `c2st/pos_acc` | _TBD_ | position-only C2ST |
| `c2st/graph_acc` | _TBD_ | **spatially-aware C2ST** — GCN over a joint spatial-kNN graph, expression as node features; tests whether the *relative spatial arrangement* matches (the MLP C2ST is blind to this) |
| `c2st/graph_auc` | _TBD_ | |
| `mmd2/x` | _TBD_ | MMD² on expression |
| `mmd2/pos` | _TBD_ | MMD² on coordinates |
| `ot_w1/x` | _TBD_ | Wasserstein-1, expression |
| `ot_w2/x` | _TBD_ | Wasserstein-2, expression |
| `ot_w1/pos` | _TBD_ | Wasserstein-1, coordinates |
| `ot_w2/pos` | _TBD_ | Wasserstein-2, coordinates |

### Geometry (point-set distances)

| Metric | Value |
|---|---|
| `psd/mean` | _TBD_ |
| `psd/max` | _TBD_ |
| `spd/mean` | _TBD_ |
| `spd/max` | _TBD_ |

### Moran's I (spatial autocorrelation)

| Metric | Value | Notes |
|---|---|---|
| `moran/real_mean` | _TBD_ | real slide spatial structure (≈ 0.257 on 1.025) |
| `moran/gen_mean` | _TBD_ | **the key contrast vs `otcfm_1025`** — aspatial baseline was ≈ 0; a coord-generating model should be > 0 if it captured any structure |
| `moran/corr` | _TBD_ | per-gene Moran correlation real-vs-gen |
| `moran/mae` | _TBD_ | |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric | Value | Notes |
|---|---|---|
| `ct/acc_real` | _TBD_ | should match `otcfm_1025` (≈ 0.524) — same classifier + target |
| `ct/acc_gen` | _TBD_ | accuracy on generated niches |
| `ct/acc_gap` | _TBD_ | `|acc_real − acc_gen|` (lower is better) |
| `ct/acc` | _TBD_ | label agreement, generated vs paired-real |
| `ct/f1` | _TBD_ | weighted-F1 of that agreement |
| `ct/prop_kl` / `ct/prop_tv` / `ct/prop_jsd` | _TBD_ | cell-type composition divergence |

**Reading (expectations, to confirm against the numbers).** Expression metrics (`mmd2/x`, `ot_*/x`,
`c2st`) should land near `otcfm_1025` — the expression block of the joint model is essentially the
same fit. The interesting change is spatial: because coordinates are now generated jointly with
expression (rather than random), `moran/gen_mean` and the `pos`/geometry metrics should move off the
"no structure" floor. How *far* they move measures how much a naive concat — with no conditioning of
coords on expression and no OT weighting between the blocks — actually recovers. Compare directly
against `otcfm_1025` (aspatial) and the NicheFlow CFM report (proper spatial model).

`c2st/graph_*` is the sharpest spatial read here: the MLP `c2st` sees absolute coordinates and is
blind to a *wrong expression↔position coupling* (right marginals, wrong arrangement), whereas the
graph-C2ST judges each cell against its spatial neighbourhood. A naive concat baseline that gets the
two marginals roughly right but couples them poorly should show `graph_auc` notably above the MLP
`c2st/pos_acc` floor. (On a matched null both sit near 0.5.)

## Reproduce

```bash
NF=../nicheflow_mba
FM=../fm_mnist
DATA=$NF/data

# 1. Train the naive-spatial OT-CFM on 1.025 in fm_mnist. Concatenates standardized coords after
#    the 5-PC expression block (model dim 7). Needs a GPU — run INSIDE your SLURM allocation; the
#    script auto-selects cuda when visible. (fm_mnist's `fm` + torchcfm; the nicheflow venv has a
#    GPU-capable torch.)
$NF/.venv/bin/python $FM/scripts/train_cfm_spatial.py \
  --data $DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  --n_pcs 5 --hidden 512 --steps 20000 --batch 256 --lr 2e-3 \
  --interpolant linear --coupling ot --ot_method exact --test_frac 0.1 --seed 0 \
  --spatial_key spatial \
  --out $FM/outputs/cfm_mouse_pca5_1025_spatial --save_every 1000 --no_wandb

# 2. Generate the naive-spatial cells on 1.025 (coord_mode=generate reads the model's coord tail).
#    Light/CPU is fine for generation + eval.
python -m paired_slides_eval.generate generator=otcfm_spatial \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  source=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  checkpoint=$FM/outputs/cfm_mouse_pca5_1025_spatial/ckpt_final.pt \
  generator.fm_root=$FM generated_out=artifacts/otcfm_spatial_1025/generated.h5ad

# 3. Evaluate with the REUSED 1.026 classifier (no retraining — generator-independent).
#    `c2st_graph` (the spatially-aware GCN C2ST) is a default group and runs automatically here;
#    no extra flag needed. To run it alone: add `--groups c2st_graph`.
python -m paired_slides_eval.evaluate \
  --target $DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  --generated artifacts/otcfm_spatial_1025/generated.h5ad \
  --classifier artifacts/otcfm_1025/classifier.ckpt \
  --ct_key class --n_pcs 50 --out reports/otcfm_spatial_1025/metrics.csv
```
