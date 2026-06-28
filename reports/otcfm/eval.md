# OT-CFM checkpoint — full evaluation metrics

Full metric suite for the **OT-CFM** (`fm_mnist`) baseline, including the cell-type-classifier
groups (`ct/*`). The neutral spatial classifier was trained on a *close but different* slide
(**1.002**) and applied to the **1.001** target, so it never saw the target.

## Setup

| Item | Value |
|---|---|
| Target slide | `adata_Zhuang_Zhuang-ABCA-1.001.h5ad` (5893 cells, 1122 genes, 16 cell types) |
| Generated cells | `artifacts/otcfm/generated.h5ad` — OT-CFM sampled from `cfm_mouse_pca5/ckpt_last.pt` (trained on 1.001), 5893 flat cells in gene space, **random placeholder coordinates** (the model is aspatial) |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.002.h5ad` — adjacent serial section, same mouse (5503 cells after dropping 2 out-of-vocabulary cells) |
| Shared feature space | PCA fit on the target's raw genes, **50 PCs** (`--n_pcs 50`); generated cells projected into it |
| Classifier | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `n_neighbors=32`, `coord_dim=2`, 16 classes |
| Classifier checkpoint | `outputs/clf_train_otcfm/checkpoints/last.ckpt` (symlinked as `artifacts/otcfm/classifier.ckpt`) |

The classifier is trained in **exactly** the representation `evaluate.py` feeds it on the OT-CFM
path — raw-gene-PCA expression (un-whitened) + **raw** coordinates — so train- and eval-time niches
match. The classifier slide (1.002) is projected through the *same* target PCA (1.001) and the
*same* 16-class label order, making it neutral and directly applicable.

## Results

`regression` is skipped (a flat whole-slide model has no cell-for-cell matched ground truth); the
`ct/*` groups are computed by reconstructing paired niches from geometry (5893 centroids paired to
their nearest real cells).

### Distribution / two-sample (expression + position)

| Metric | Value | Notes |
|---|---|---|
| `c2st/acc` | 0.9975 | real-vs-generated classifier accuracy (joint expr+pos); ~1.0 → trivially separable |
| `c2st/auc` | 1.0000 | |
| `c2st/pos_acc` | 0.6635 | position-only C2ST |
| `mmd2/x` | 0.2558 | MMD² on expression |
| `mmd2/pos` | 0.0795 | MMD² on coordinates |
| `ot_w1/x` | 25.3833 | Wasserstein-1, expression |
| `ot_w2/x` | 30.6471 | Wasserstein-2, expression |
| `ot_w1/pos` | 0.2175 | Wasserstein-1, coordinates |
| `ot_w2/pos` | 0.2610 | Wasserstein-2, coordinates |

### Geometry (point-set distances)

| Metric | Value |
|---|---|
| `psd/mean` | 0.0656 |
| `psd/max` | 0.6653 |
| `spd/mean` | 0.0137 |
| `spd/max` | 0.0541 |

### Moran's I (spatial autocorrelation)

| Metric | Value | Notes |
|---|---|---|
| `moran/real_mean` | 0.2527 | real slide has spatial structure |
| `moran/gen_mean` | -0.0003 | generated ≈ 0 → **no spatial structure** (OT-CFM coords are random placeholders) |
| `moran/corr` | 0.4803 | per-gene Moran correlation real-vs-gen |
| `moran/mae` | 0.2530 | |

### Cell-type classifier (`ct/*`, neutral 1.002-trained classifier)

| Metric | Value | Notes |
|---|---|---|
| `ct/acc_real` | 0.5454 | classifier accuracy on **real** 1.001 niches (16-class, neighbourhood-only) — confirms the neutral classifier transfers |
| `ct/acc_gen` | 0.2749 | accuracy on **generated** niches |
| `ct/acc_gap` | 0.2705 | real − generated accuracy gap (lower is better) |
| `ct/acc` | 0.3886 | label agreement between generated and paired-real niches |
| `ct/f1` | 0.3623 | weighted-F1 of that agreement |
| `ct/prop_kl` | 0.0943 | cell-type composition divergence (KL) |
| `ct/prop_tv` | 0.1719 | total variation |
| `ct/prop_jsd` | 0.0214 | Jensen–Shannon |

**Reading.** The OT-CFM is an unconditional, expression-only baseline: it reproduces the rough
expression marginals (moderate MMD/EMD) but is trivially separable from the real slide (`c2st ≈ 1`)
and carries **no spatial structure** (`moran/gen_mean ≈ 0`, large `ct/acc_gap`). The expression
metrics are the meaningful read for this model; the spatial/classifier metrics quantify the absence
of geometry. The neutral classifier reaching ~0.55 on real niches (vs. ~0.27 on generated) shows the
gap is a property of the generated cells, not a broken classifier.

## Reproduce

> ⚠️ **The table above was produced with the now-retired raw-gene-PCA path**
> (`prepare_classifier_slide` + `evaluate --n_pcs 50`), which is **not comparable** to the NicheFlow
> models. The unified flow below standardises on NicheFlow's recipe (shared PCA + standardised
> coords), so it yields **different, cross-model-comparable** numbers. See
> [`docs/comparability_plan.md`](../../docs/comparability_plan.md).

```bash
DATA=../nicheflow_mba/data
FM=../fm_mnist
SOURCE=$DATA/adata_Zhuang_Zhuang-ABCA-1.000.h5ad   # the source slide that fixes the shared PCA basis
export CUDA_VISIBLE_DEVICES=""   # this box's GPU driver is too old; train/eval on CPU

# 1. Build the shared pair pkl (basis) + classifier-slide pkl, both in the NicheFlow recipe
python -m paired_slides_eval.adapters.prepare_shared_slides \
  --source $SOURCE --target $DATA/adata_Zhuang_Zhuang-ABCA-1.001.h5ad \
  --classifier_slide $DATA/adata_Zhuang_Zhuang-ABCA-1.002.h5ad \
  --ct_key class --n_pcs 50 \
  --out_pair data/abca_pair.pkl --out_classifier data/abca_1.002_clf.pkl

# 2. Train the ONE spatial classifier on the classifier-slide pkl (Hydra trainer)
python -m paired_slides_eval.classifier.train \
  data=ct_abca_spatial model=classifier_spatial data.n_classes=16 \
  data.datamodule.data_fp=$PWD/data/abca_1.002_clf.pkl \
  callbacks.model_checkpoint.monitor=val/f1 callbacks.early_stopping.monitor=val/f1 \
  '~callbacks.lr_monitor' model.plot_callbacks=False \
  trainer=cpu +trainer.max_epochs=20 hydra.run.dir=outputs/clf_train_otcfm

# 3. Generate the OT-CFM cells (gene space + raw coords)
python -m paired_slides_eval.generate generator=otcfm \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.001.h5ad generator.fm_root=$FM \
  checkpoint=$FM/outputs/cfm_mouse_pca5/ckpt_last.pt \
  generated_out=artifacts/otcfm/generated.h5ad

# 4. Evaluate against the SHARED pair pkl: project genes (--shared_pca) + standardise coords; one
#    classifier; fixed (model-independent) ct/acc_real
python -m paired_slides_eval.evaluate \
  --target data/abca_pair.pkl --generated artifacts/otcfm/generated.h5ad \
  --classifier outputs/clf_train_otcfm/checkpoints/last.ckpt --ct_key class \
  --shared_pca --standardize_coords --ct_real_reference fixed \
  --out reports/otcfm/metrics.csv
```

*(`artifacts/otcfm/generated.h5ad` is the OT-CFM output produced by `generate` — which now writes to
`artifacts/<generator>/generated.h5ad` by default:*
`python -m paired_slides_eval.generate generator=otcfm target=$DATA/adata_..-1.001.h5ad
checkpoint=../fm_mnist/outputs/cfm_mouse_pca5/ckpt_last.pt`*.)*
