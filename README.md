# nicheflow-eval

Standalone evaluation metrics for spatial single-cell generative models (NicheFlow). Inputs are
**original AnnData (`.h5ad`) files** — raw gene expression + spatial coordinates. Two ways to use it:

- **Standalone** — give it a real **target slide** and a set of **generated cells** (both AnnData),
  and it computes the full metric suite.
- **Full pipeline** — give it the **source** + **target** slides, a slide to **train the
  classifier** on, and a trained flow **checkpoint**; it preprocesses the niches, generates the
  target with the flow (importing NicheFlow as a blackbox), and evaluates — no manual preprocessing.

It also bundles the **full cell-type-classifier training pipeline** used by the classifier metrics.

## Install

```bash
cd nicheflow-eval
uv venv && uv pip install -e ".[wandb]"   # or: pip install -e .
```

The standalone metrics need only the deps above. The **full pipeline** additionally imports
NicheFlow (not on PyPI); install it from the sibling repo:

```bash
pip install -e ../nicheflow_mba       # provides the `nicheflow` package used for generation
```

## Inputs: AnnData (`.h5ad`)

No preprocessed pickle is required. Each slide is a plain AnnData with:

| Field | Where | Notes |
|---|---|---|
| expression | `adata.X` | **raw genes** (default). Or an `obsm`/`layers` key via `expr_key=` if you already reduced. |
| coordinates | `adata.obsm["spatial"]` | configurable via `spatial_key=` |
| cell types | `adata.obs[ct_key]` | optional; needed by the classifier metrics |

Generated cells are stored **flat** (one row per cell) with `obs["niche_id"]` grouping each niche's
points (centroid first) and coords in `obsm["spatial"]`; paired ground-truth niches in
`obsm["gt_x"]`/`obsm["gt_pos"]` and the paired real centroid's true label in `obs["gt_ct"]`.

PCA is **not** assumed. Target and generated must share one feature space — either keep both as raw
genes (same panel), or pass `n_pcs=` to fit one PCA on the target and project the generated cells
through it:

```python
from nicheflow_eval import TargetSlide, GeneratedNiches, evaluate

target = TargetSlide.from_anndata("target.h5ad", ct_key="class")        # raw genes + coords
generated = GeneratedNiches.from_anndata("generated.h5ad")              # (B, N, D), centroid first
results = evaluate(target, generated)                                   # {test/group/metric: ...}

# optional shared-PCA: fit on the target, project the generated cells into the same basis
target = TargetSlide.from_anndata("target.h5ad", ct_key="class", n_pcs=50)
generated = GeneratedNiches.from_anndata("generated.h5ad").project(target.pca)
```

## Full pipeline (checkpoint → generated cells → metrics)

```python
from nicheflow_eval.pipeline import run_pipeline

res = run_pipeline(
    "source.h5ad", "target.h5ad", "flow.ckpt",   # source/target slides + trained flow checkpoint
    classifier_h5ad="classifier_slide.h5ad",      # held-out slide to train the neutral classifier
    n_pcs=50,
)
print(res.metrics)                                # flat {test/group/metric: value} dict
```

Or from the command line:

```bash
python -m nicheflow_eval.pipeline \
  --source SOURCE.h5ad --target TARGET.h5ad \
  --checkpoint FLOW.ckpt \
  --classifier CLASSIFIER_SLIDE.h5ad \
  --generated_out generated.h5ad --out results.csv
```

The pipeline preprocesses the source+target pair into the niche scaffolding (shared PCA on the
concatenated pair, per-slide coordinate standardization, radius graph + grid subsample — the
original NicheFlow preprocessing, ported here; **no global alignment / PASTE2**), generates the
target with `flow.sample`, trains/loads the classifier, and evaluates. Generated cells live in the
preprocessor's standardized `X_pca` space, and the target is read from the same space so the metrics
are comparable.

## Run the standalone evaluation (command line)

`python -m nicheflow_eval.evaluate` runs the suite on one `(target slide, generated cells)` pair: a
target `.h5ad` and the generated cells as `.npz` (arrays `x (B,N,P)`, `pos (B,N,P)`; optional
`gt_x`/`gt_pos`/`gt_ct`) or a flat generated `.h5ad`.

```bash
# geometry + distribution + C2ST + Moran (no trained classifier needed)
python -m nicheflow_eval.evaluate --target TARGET.h5ad --generated generated.npz --out results.csv

# add the classifier groups (concordance + accuracy gap)
python -m nicheflow_eval.evaluate \
  --target TARGET.h5ad --generated generated.npz \
  --classifier Classifier_Spatial.ckpt --out results.csv
```

Groups whose inputs are missing are skipped automatically (no `gt_*` → skips regression; no
`--classifier` → skips the `ct/*` groups) and reported in a `skipped:` line. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--classifier` | none | classifier `.ckpt`; enables the `ct/*` groups. `n_neighbors` is read from the checkpoint |
| `--groups` | all | subset, e.g. `--groups c2st moran` |
| `--n_pcs` | none | fit a PCA on the target to N PCs and project the generated cells into it |
| `--expr_key` / `--spatial_key` / `--ct_key` | `X`/`spatial`/none | where expression/coords/labels live |
| `--seed` | `0` | |

## Metrics

| Group | Keys | Needs |
|---|---|---|
| Pointwise regression | `x/{mse,mae}`, `pos/{mse,mae}` | matched `gt_*` |
| Point/shape distances | `psd/{mean,max}`, `spd/{mean,max}` | — |
| Distribution | `mmd2/{x,pos}`, `ot_w1/{x,pos}`, `ot_w2/{x,pos}` | `torch`, `pot` |
| C2ST (label-free) | `c2st/{acc,auc,pos_acc,sig_*}` | `sklearn` |
| Moran's I (label-free) | `moran/{mae,corr,real_mean,gen_mean}` | `squidpy` — over **all** generated cells vs the full real slide |
| Cell-type concordance | `ct/{f1,acc,prop_kl,prop_tv,prop_jsd}` | classifier + paired real niches `gt_*` |
| Classifier accuracy gap | `ct/{acc_real,acc_gen,acc_gap}` | classifier + paired niches `gt_*` + true labels `gt_ct` |

The **accuracy gap** runs the trained classifier on the real target niches and on the generated
niches, each scored against the true centroid labels; a small `|acc_real - acc_gen|` means the
generated niches are as classifiable as the real ones. See `docs/metric_comparison.md` and
`docs/metrics.md`.

## Layout

```
src/nicheflow_eval/
  contract.py          TargetSlide / GeneratedNiches (from_anndata / from_dataclass)
  data/anndata.py      read raw .h5ad -> arrays; optional shared PCA
  data/dataclass.py    the niche dataclass schema + loader
  evaluate.py          evaluate(target, generated) -> flat dict + standalone CLI
  metrics/             kernels + wrappers (c2st, distribution, morans, distances, concordance, classifier_gap)
  preprocessing/       raw AnnData -> niche dataclass (radius graph + grid subsample); ported from NicheFlow
  pipeline/            checkpoint -> generated cells -> metrics (imports nicheflow as a blackbox)
  classifier/          full classifier training (nets, task, dataset, datamodule, train, train_helper)
configs/               hydra configs for classifier training
notebooks/evaluation.ipynb   the end-to-end deliverable
tests/                 synthetic-data metric tests (no real data needed)
```
