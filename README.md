# nicheflow-eval

Standalone evaluation metrics for spatial single-cell generative models (NicheFlow). Give it a
**real target slide** and a set of **generated cells**, and it computes the full metric suite.
It also bundles the **full cell-type-classifier training pipeline** used by the concordance
metric (not just a frozen checkpoint).

## Install

```bash
cd nicheflow-eval
uv venv && uv pip install -e ".[wandb]"   # or: pip install -e .
```

## The input contract

Everything reduces to two objects of plain arrays (no flow model, checkpoints, or Hydra needed):

| Object | Fields | Notes |
|---|---|---|
| `TargetSlide` | `x (N,P)`, `pos (N,2)`, `ct (N,)`, `grid_x/grid_pos`, `n_classes` | the real slide; `ct` only for concordance |
| `GeneratedNiches` | `x (B,N,P)`, `pos (B,N,P)`, optional `gt_x/gt_pos` | `(B, N, D)`, **centroid at point 0**; `gt_*` enables regression |

Both sides must live in the **same PCA space**, and the target slide is the **target slide on its
own** — build `TargetSlide` from the dedicated `target_abca.pkl` (produced by preprocessing in the
shared `X_pca` basis the model generated in). **Do not** use the concatenated source+target aligned
pkl as the target:

```python
from nicheflow_eval import TargetSlide, GeneratedNiches, evaluate
from nicheflow_eval.data import load_h5ad_dataset_dataclass

ds = load_h5ad_dataset_dataclass("data/target_abca.pkl")   # the TARGET slide only
target = TargetSlide.from_dataclass(ds, timepoint=ds.timepoints_ordered[-1])
generated = GeneratedNiches(x=gen_x, pos=gen_pos)   # or .from_trajectory(x_traj, pos_traj)

results = evaluate(target, generated)               # {"test/mmd2/x": ..., "test/c2st/acc": ..., ...}
```

## Train the concordance classifier

Concordance uses a **neutral** classifier: trained on a **held-out same-mouse slide** (neither
source nor target) — `data/abca_aligned_clf.pkl` — then applied to **both** the generated niche
and its paired real target niche. The full Hydra/Lightning pipeline lives under
`nicheflow_eval.classifier` (gene-only MLP + spatial DeepSet / SetTransformer / GCN variants);
configs are in `configs/` and the data config points at `abca_aligned_clf.pkl`.

```bash
python -m nicheflow_eval.classifier.train experiment=classifier/abca          # gene-only MLP
python -m nicheflow_eval.classifier.train experiment=classifier/abca_spatial   # spatial SetTransformer
```

It reads `X_pca`, `coords`, `ct`, and the precomputed neighbour graph from that `.pkl`. See
`docs/classifier_eval_summary.md`.

## Run the evaluation (command line)

`python -m nicheflow_eval.evaluate` runs the whole suite on one `(target, generated)` pair: a
preprocessing `.pkl` for the target slide and an `.npz` of generated cells (arrays `x (B,N,P)`,
`pos (B,N,P)`; optional `gt_x`/`gt_pos`). It prints a `metric value` table and, with `--out`,
writes a CSV.

```bash
# geometry + distribution + C2ST (no trained classifier needed)
python -m nicheflow_eval.evaluate \
  --target PATH/TO/target.pkl \
  --generated PATH/TO/generated.npz \
  --out PATH/TO/results.csv

# full suite incl. cell-type concordance (pass the trained classifier checkpoint)
python -m nicheflow_eval.evaluate \
  --target PATH/TO/target.pkl \
  --generated PATH/TO/generated.npz \
  --classifier PATH/TO/Classifier_Spatial_ABCA_SetTransformer.ckpt \
  --out PATH/TO/results.csv
```

Groups whose inputs are missing are skipped automatically (no `gt_*` → skips regression; no
`--classifier` → skips concordance) and reported in a `skipped:` line. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--classifier` | none | classifier `.ckpt`; enables the concordance group. `n_neighbors` is read from the checkpoint so eval matches training |
| `--groups` | all | subset, e.g. `--groups c2st moran` |
| `--timepoint` | last slide | which slide in the `.pkl` is the target |
| `--n_pcs` | all | truncate expression to the first N PCs |
| `--seed` | `0` | |
| `--hidden_dim` / `--num_heads` / `--coord_dim` / `--no_mask_centroid` | `64` / `4` / `2` / off | classifier net hyperparameters — **must match training** (only used with `--classifier`) |

## Metrics

| Group | Keys | Needs |
|---|---|---|
| Pointwise regression | `x/{mse,mae}`, `pos/{mse,mae}` | matched `gt_*` |
| Point/shape distances | `psd/{mean,max}`, `spd/{mean,max}` | — |
| Distribution | `mmd2/{x,pos}`, `ot_w1/{x,pos}`, `ot_w2/{x,pos}` | `torch`, `pot` |
| C2ST (label-free) | `c2st/{acc,auc,pos_acc,sig_*}` | `sklearn` |
| Moran's I (label-free) | `moran/{mae,corr,real_mean,gen_mean}` | `squidpy` |
| Cell-type concordance | `ct/{f1,acc,prop_kl,prop_tv,prop_jsd}` | neutral classifier + paired real niches `gt_*` |

See `docs/metric_comparison.md` for what each metric means and example results, and
`docs/metrics.md` for the one-line definitions.

## Layout

```
src/nicheflow_eval/
  contract.py          TargetSlide / GeneratedNiches
  evaluate.py          evaluate(target, generated) -> flat dict
  metrics/             kernels + wrappers (c2st, distribution, morans, distances, concordance)
  classifier/          full classifier training (nets, task, dataset, datamodule, train)
  data/                preprocessing .pkl schema + loader
  utils/               logging / hydra instantiation / seeding / plotting
configs/               hydra configs for classifier training
notebooks/evaluation.ipynb   the end-to-end deliverable
tests/                 synthetic-data metric tests (no .pkl needed)
```
