# nicheflow-eval

Evaluation metrics for generative models on spatial transcriptomics. Inputs are
**original AnnData (`.h5ad`) files** with raw gene expression + spatial coordinates. Two ways to use it:

- **Full pipeline** — give it the **source** + **target** slides, a trained **checkpoint**, and a
  **`generator`** (the model blackbox); it generates the target and evaluates. The bundled 
  `nicheflow_generator` is one example of such generator; bring your own by writing
  a callable that returns a `GenerationOutput` (see *Bring your own model* below).
- **Standalone** — if there already exists a precomputed **generated cells** AnnData file along with the real 
**target slide** AnnData file, it can computes the full metric suite directly. 

It also bundles the **full cell-type-classifier training pipeline** used by the classifier metrics. The classifier
needs to be trained first on the classifier-training-slide before running the evaluation metrics.  

## Install

```bash
cd nicheflow-eval
uv venv && uv pip install -e ".[wandb]"   # or: pip install -e .
```

The standalone metrics and the generic pipeline need only the deps above. The bundled **NicheFlow
adapter** (`nicheflow_eval.adapters.nicheflow`) additionally imports NicheFlow (not on PyPI);
install it from the sibling repo only if using nicheflow:

```bash
pip install -e ../nicheflow_mba       # provides the `nicheflow` package used for generation
```

## Inputs: AnnData (`.h5ad`)

Each slide is a plain AnnData with:

| Field | Where | Notes |
|---|---|---|
| expression | `adata.X` | **raw genes** (default). Or an `obsm`/`layers` key via `expr_key=` if you already reduced. |
| coordinates | `adata.obsm["spatial"]` | configurable via `spatial_key=` |
| cell types | `adata.obs[ct_key]` | optional; needed by the classifier metrics |

**Generated cells come in two shapes** — pick by what your model emits:

- **`GeneratedSlide`** — a **flat** whole slide: `X` + `obsm["spatial"]`, one row per cell, exactly
  like the target. For ordinary generative models. The label-free metrics (psd, spd, distribution,
  c2st, moran) run on it; the niche metrics are skipped.
- **`GeneratedNiches`** — **niche-shaped**: flat rows with `obs["niche_id"]` grouping each niche's
  points (centroid first), coords in `obsm["spatial"]`; paired ground-truth in
  `obsm["gt_x"]`/`obsm["gt_pos"]` and the paired real centroid label in `obs["gt_ct"]`. Required by
  the niche metrics (regression, concordance, ct_gap).

Target and generated must share one feature space — either keep both as raw
genes (same panel), or pass `n_pcs=` to fit one PCA on the target and `.project()` the generated
cells through it:

```python
from nicheflow_eval import TargetSlide, GeneratedSlide, GeneratedNiches, evaluate

target = TargetSlide.from_anndata("target.h5ad", ct_key="class")        # raw genes + coords

# whole-slide model -> flat cells (niche metrics auto-skipped):
generated = GeneratedSlide.from_anndata("generated.h5ad")               # (N, D)
# OR niche-shaped cells (enables the niche metrics):
generated = GeneratedNiches.from_anndata("generated.h5ad")             # (B, N, D), centroid first

results = evaluate(target, generated)                                   # {test/group/metric: ...}

# optional shared-PCA: fit on the target, project the generated cells into the same basis
target = TargetSlide.from_anndata("target.h5ad", ct_key="class", n_pcs=50)
generated = GeneratedSlide.from_anndata("generated.h5ad").project(target.pca)
```

## Full pipeline (checkpoint → generated cells → metrics)

The pipeline is **model-agnostic**: `run_pipeline` takes a `generator` (the blackbox). The bundled
NicheFlow adapter is one generator:

```python
from nicheflow_eval.pipeline import run_pipeline
from nicheflow_eval.adapters.nicheflow import nicheflow_generator

res = run_pipeline(
    "source.h5ad", "target.h5ad", "flow.ckpt",   # source/target slides + trained checkpoint
    generator=nicheflow_generator,                # the model blackbox
    classifier_h5ad="classifier_slide.h5ad",      # held-out slide to train the neutral classifier
    n_pcs=50,                                      # forwarded to the adapter
)
print(res.metrics)                                # flat {test/group/metric: value} dict
```

Or from the command line (defaults to the NicheFlow adapter):

```bash
python -m nicheflow_eval.pipeline \
  --source SOURCE.h5ad --target TARGET.h5ad \
  --checkpoint FLOW.ckpt \
  --classifier CLASSIFIER_SLIDE.h5ad \
  --generated_out generated.h5ad --out results.csv
```

The NicheFlow adapter preprocesses the source+target pair into the niche scaffolding (shared PCA on
the concatenated pair, per-slide coordinate standardization, radius graph + grid subsample), generates the
target with `flow.sample`, trains/loads the classifier, and hands a comparable `(target, generated)`
pair (in the standardized `X_pca` space) back to the pipeline.

### Bring your own model

If you are using other generative models, write a `generator` — any callable that turns the
raw slides + a checkpoint into a `GenerationOutput`. If your model writes generated cells to a
`.h5ad` in gene space, `from_generated_anndata` builds the output in one line:

```python
from nicheflow_eval.pipeline import run_pipeline, from_generated_anndata

def my_generator(*, source, target, checkpoint, **kw):
    my_model_generate(source, target, checkpoint, out="gen.h5ad")   # your code; gene-space niches
    return from_generated_anndata("gen.h5ad", target, ct_key="class", n_pcs=50)

res = run_pipeline("source.h5ad", "target.h5ad", "model.ckpt",
                   generator=my_generator, classifier="classifier.ckpt")
```

The generated `.h5ad` follows the `GeneratedNiches.from_anndata` layout: one row per cell,
`obs['niche_id']` grouping each niche (centroid first), coords in `obsm['spatial']`, and optional
paired ground truth in `obsm['gt_x']` / `obsm['gt_pos']` / `obs['gt_ct']`.

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

The label-free groups run on **either** a flat `GeneratedSlide` or `GeneratedNiches`; the niche
groups (regression, concordance, ct_gap) need `GeneratedNiches` and are auto-skipped for a flat slide.

| Group | Keys | Shape | Needs |
|---|---|---|---|
| Pointwise regression | `x/{mse,mae}`, `pos/{mse,mae}` | niche | matched `gt_*` |
| Point/shape distances | `psd/{mean,max}`, `spd/{mean,max}` | flat or niche | — |
| Distribution | `mmd2/{x,pos}`, `ot_w1/{x,pos}`, `ot_w2/{x,pos}` | flat or niche | `torch`, `pot` |
| C2ST (label-free) | `c2st/{acc,auc,pos_acc,sig_*}` | flat or niche | `sklearn` |
| Moran's I (label-free) | `moran/{mae,corr,real_mean,gen_mean}` | flat or niche | `squidpy` — over **all** generated cells vs the full real slide |
| Cell-type concordance | `ct/{f1,acc,prop_kl,prop_tv,prop_jsd}` | niche | classifier + paired real niches `gt_*` |
| Classifier accuracy gap | `ct/{acc_real,acc_gen,acc_gap}` | niche | classifier + paired niches `gt_*` + true labels `gt_ct` |

The **accuracy gap** runs the trained classifier on the real target niches and on the generated
niches, each scored against the true centroid labels; a small `|acc_real - acc_gen|` means the
generated niches are as classifiable as the real ones. See `docs/metric_comparison.md` and
`docs/metrics.md`.

## Layout

```
src/nicheflow_eval/
  contract.py          TargetSlide / GeneratedSlide (flat) / GeneratedNiches — the AnnData contract
  data/anndata.py      read raw .h5ad -> arrays; optional shared PCA          [common, model-agnostic]
  data/dataclass.py    the internal niche pickle schema + loader
  evaluate.py          evaluate(target, generated) -> flat dict + standalone CLI
  metrics/             kernels + each metric's own prep (c2st, distribution, morans, distances,
                       concordance, classifier_gap) — Moran/classifier niches built here, locally
  pipeline/            model-agnostic run_pipeline + Generator protocol (NO nicheflow import)
  adapters/nicheflow/  the NicheFlow generator blackbox: preprocess + graph + generate (imports
                       nicheflow); the only model-specific code
  classifier/          full classifier training (nets, task, dataset, datamodule, train, train_helper)
configs/               hydra configs for classifier training
notebooks/evaluation.ipynb   the end-to-end deliverable
tests/                 synthetic-data metric tests (no real data needed)
```
