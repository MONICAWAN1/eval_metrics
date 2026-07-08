# OT-CFM (trained on 1.025) — full evaluation metrics

Full metric suite for an **OT-CFM** retrained on slide **1.025** — the same target the NicheFlow CFM
checkpoint generates — so the two models can be read against a common target. Trained in `fm_mnist`
with the **same config as the original `cfm_mouse_pca5`** checkpoint (only the data slide changed),
on **GPU**. The neutral spatial classifier was trained on a *close but different* slide (**1.026**,
the same held-out slide the NicheFlow run used) and applied to the **1.025** target.

## Setup

| Item                      | Value                                                                                                                                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model                     | OT-CFM (`fm_mnist`), `outputs/cfm_mouse_pca5_1025/ckpt_final.pt` — **newly trained on 1.025**                                                                                                                                                                                                                 |
| Training                  | `scripts/train_cfm_spatial.py`, config identical to `cfm_mouse_pca5`: `n_pcs=5`, `dim=5`, `hidden=512`, `steps=20000`, `batch=256`, `lr=2e-3`, `interpolant=linear`, `coupling=ot`, `ot_method=exact`, `whiten=True`, `test_frac=0.1`, `seed=0`. Ran on **GPU** (RTX A6000), 20k steps in ~5.6 min (~62 it/s) |
| Generated cells           | `artifacts/otcfm_1025/generated.h5ad` — 9962 flat cells in gene space, **random placeholder coordinates** (the model is aspatial)                                                                                                                                                                             |
| Target slide              | `adata_Zhuang_Zhuang-ABCA-1.025.h5ad` (9962 cells, 1122 genes, 18 cell types)                                                                                                                                                                                                                                 |
| Classifier-training slide | `adata_Zhuang_Zhuang-ABCA-1.026.h5ad` — nearby serial section, same mouse (9129 cells after dropping out-of-vocabulary cells)                                                                                                                                                                                 |
| Shared feature space      | PCA fit on the target's raw genes, **50 PCs** (`--n_pcs 50`); generated cells projected into it                                                                                                                                                                                                               |
| Classifier                | `SpatialCTClassifierNet` (Set-Transformer, masked centroid), `n_neighbors=32`, `coord_dim=2`, 18 classes                                                                                                                                                                                                      |
| Classifier checkpoint     | `outputs/clf_train_otcfm_1025/checkpoints/last.ckpt` (symlinked as `artifacts/otcfm_1025/classifier.ckpt`)                                                                                                                                                                                                    |

As on the original OT-CFM report, the classifier is trained in **exactly** the representation
`evaluate.py` feeds it — raw-gene-PCA expression (un-whitened) + **raw** coordinates — with 1.026
projected through the *same* target (1.025) PCA and 18-class label order, so it is neutral and
directly applicable.

> **Why not reuse the NicheFlow run's 1.026 classifier?** That one was trained inside the NicheFlow
> pipeline in NicheFlow's *whitened shared-PCA* space (a different basis + standardisation) and was
> not persisted to disk. The OT-CFM eval fits a plain raw-gene PCA on the target, so it needs a
> classifier in *that* space — hence a fresh one, still trained on the same held-out slide (1.026).

## Results

`regression` is skipped (a flat whole-slide model has no cell-for-cell matched ground truth); the
`ct/*` groups are computed by reconstructing paired niches from geometry (9962 centroids paired to
their nearest real cells).

### Distribution / two-sample (expression + position)

| Metric         | Value   | Notes                                                                              |
| -------------- | ------- | ---------------------------------------------------------------------------------- |
| `c2st/acc`     | 0.9985  | real-vs-generated classifier accuracy (joint expr+pos); ~1.0 → trivially separable |
| `c2st/auc`     | 1.0000  |                                                                                    |
| `c2st/pos_acc` | 0.6338  | position-only C2ST                                                                 |
| `mmd2/x`       | 0.3692  | MMD² on expression                                                                 |
| `mmd2/pos`     | 0.0729  | MMD² on coordinates                                                                |
| `ot_w1/x`      | 29.0703 | Wasserstein-1, expression                                                          |
| `ot_w2/x`      | 35.9401 | Wasserstein-2, expression                                                          |
| `ot_w1/pos`    | 0.2809  | Wasserstein-1, coordinates                                                         |
| `ot_w2/pos`    | 0.3373  | Wasserstein-2, coordinates                                                         |

### Geometry (point-set distances)

| Metric     | Value  |
| ---------- | ------ |
| `psd/mean` | 0.0503 |
| `psd/max`  | 0.6063 |
| `spd/mean` | 0.0129 |
| `spd/max`  | 0.0437 |

### Moran's I (spatial autocorrelation)

| Metric            | Value   | Notes                                                                            |
| ----------------- | ------- | -------------------------------------------------------------------------------- |
| `moran/real_mean` | 0.2569  | real slide spatial structure                                                     |
| `moran/gen_mean`  | -0.0000 | generated ≈ 0 → **no spatial structure** (OT-CFM coords are random placeholders) |
| `moran/corr`      | 0.0963  | per-gene Moran correlation real-vs-gen (near 0)                                  |
| `moran/mae`       | 0.2569  |                                                                                  |

### Cell-type classifier (`ct/*`, neutral 1.026-trained classifier)

| Metric        | Value  | Notes                                                                                                                   |
| ------------- | ------ | ----------------------------------------------------------------------------------------------------------------------- |
| `ct/acc_real` | 0.5242 | classifier accuracy on **real** 1.025 niches (18-class, neighbourhood-only) — confirms the neutral classifier transfers |
| `ct/acc_gen`  | 0.3404 | accuracy on **generated** niches                                                                                        |
| `ct/acc_gap`  | 0.1838 | \`                                                                                                                      |
| `ct/acc`      | 0.5310 | label agreement between generated and paired-real niches                                                                |
| `ct/f1`       | 0.3684 | weighted-F1 of that agreement                                                                                           |
| `ct/prop_kl`  | 7.3117 | cell-type composition divergence (KL) — large; the generated cell-type mix is skewed                                    |
| `ct/prop_tv`  | 0.4690 | total variation                                                                                                         |
| `ct/prop_jsd` | 0.1990 | Jensen–Shannon                                                                                                          |

**Reading.** As expected for an unconditional, expression-only baseline, the retrained OT-CFM
reproduces the coarse expression marginals but is trivially separable from the real slide
(`c2st ≈ 1`) and carries **no spatial structure** (`moran/gen_mean ≈ 0`, `moran/corr ≈ 0.1`). The
neutral classifier reaches ~0.52 on real 1.025 niches vs ~0.34 on generated ones, and the cell-type
composition diverges sharply (`prop_kl ≈ 7.3`) — the model does not match the real type mix.

> **Comparison note.** This run shares the **target (1.025)** and **classifier slide (1.026)** with
> the NicheFlow CFM report, so the two are closer to comparable than the original OT-CFM/1.001 run.
> Still not strictly head-to-head: the OT-CFM eval uses a **raw-gene PCA** and a **flat aspatial**
> output, while NicheFlow uses its **whitened shared-PCA** space and **niche-shaped** cells (which is
> why NicheFlow also reports `regression`). Use the within-report `ct/acc_gap`, `c2st`, and Moran
> contrasts rather than cross-comparing absolute distances across feature spaces.

## Reproduce

> ⚠️ **The table above was produced with the now-retired raw-gene-PCA path**
> (`prepare_classifier_slide` + `evaluate --n_pcs 50`), which is **not comparable** to the NicheFlow
> models. The unified flow below standardises on NicheFlow's recipe (shared PCA + standardised
> coords) and yields **different, cross-model-comparable** numbers. See
> [`docs/comparability_plan.md`](../../docs/comparability_plan.md).

```bash
NF=../nicheflow_mba
FM=../fm_mnist
DATA=$NF/data

# 1. Train OT-CFM on 1.025 in fm_mnist (GPU). Uses fm_mnist's `fm` + torchcfm; the nicheflow venv
#    has a GPU-capable torch (cu124), so run it there. Config = the original cfm_mouse_pca5.
$NF/.venv/bin/python $FM/scripts/train_cfm_spatial.py \
  --data $DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  --n_pcs 5 --hidden 512 --steps 20000 --batch 256 --lr 2e-3 \
  --interpolant linear --coupling ot --ot_method exact --test_frac 0.1 --seed 0 \
  --out $FM/outputs/cfm_mouse_pca5_1025 --save_every 1000 --no_wandb

export CUDA_VISIBLE_DEVICES=""   # the rest runs on CPU (this box's GPU driver is too old for the eval venv's torch)

# 2. Build the shared pair pkl (basis) + classifier-slide pkl, both in the NicheFlow recipe.
#    SOURCE fixes the shared PCA basis; use the NicheFlow source slide for the comparison (1.024 here).
SOURCE=$DATA/adata_Zhuang_Zhuang-ABCA-1.024.h5ad
python -m paired_slides_eval.adapters.prepare_shared_slides \
  --source $SOURCE --target $DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  --classifier_slide $DATA/adata_Zhuang_Zhuang-ABCA-1.026.h5ad \
  --ct_key class --n_pcs 50 \
  --out_pair data/abca_1025_pair.pkl --out_classifier data/abca_1.026_clf.pkl

# 3. Train the ONE spatial classifier on the classifier-slide pkl (1.025 has 18 cell types)
python -m paired_slides_eval.classifier.train \
  data=ct_abca_spatial model=classifier_spatial data.n_classes=18 \
  data.datamodule.data_fp=$PWD/data/abca_1.026_clf.pkl \
  callbacks.model_checkpoint.monitor=val/f1 callbacks.early_stopping.monitor=val/f1 \
  '~callbacks.lr_monitor' model.plot_callbacks=False \
  trainer=cpu +trainer.max_epochs=20 hydra.run.dir=outputs/clf_train_otcfm_1025

# 4. Generate the OT-CFM cells on 1.025 (gene space + raw coords)
python -m paired_slides_eval.generate generator=otcfm \
  target=$DATA/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  checkpoint=$FM/outputs/cfm_mouse_pca5_1025/ckpt_final.pt \
  generator.fm_root=$FM generated_out=artifacts/otcfm_1025/generated.h5ad

# 5. Evaluate against the SHARED pair pkl. Gene-space cells project automatically and coords are
#    auto-reconciled (--coords auto, the default); one classifier; fixed ct/acc_real.
python -m paired_slides_eval.evaluate \
  --target data/abca_1025_pair.pkl --generated artifacts/otcfm_1025/generated.h5ad \
  --classifier outputs/clf_train_otcfm_1025/checkpoints/last.ckpt --ct_key class \
  --ct_real_reference fixed --out reports/otcfm_1025/metrics.csv
```
