# paired-slides-eval

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
cd paired-slides-eval
uv venv && uv pip install -e ".[wandb]"   # or: pip install -e .
```

The standalone metrics and the generic pipeline need only the deps above. The bundled **NicheFlow
adapter** (`paired_slides_eval.adapters.nicheflow`) additionally imports NicheFlow (not on PyPI);
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
from paired_slides_eval import TargetSlide, GeneratedSlide, GeneratedNiches, evaluate

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
from paired_slides_eval.pipeline import run_pipeline
from paired_slides_eval.adapters.nicheflow import nicheflow_generator

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
python -m paired_slides_eval.pipeline \
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

**You never edit this package, and the package never imports your model.** You write a small
*generator function* in **your own code** — that function is the only place your model gets
imported — and hand it to `run_pipeline` as `generator=…`. So:

- **Where do you import your model?** Inside your generator's module (top of the file, or lazily in
  the function) — a file you own, anywhere on your `PYTHONPATH`.
- **Where do you define your generator?** The same file. It's just a function matching the contract:

  ```python
  def my_generator(*, source, target, checkpoint, **kwargs) -> GenerationOutput: ...
  ```

  where `source`/`target` are the raw `.h5ad` slides and `checkpoint` is your model — all passed
  straight through from `run_pipeline`. It returns a `GenerationOutput` (a `target` + `generated`
  pair, and optionally a trained `classifier`).

Your generator owns exactly one job: make the real target and the generated cells comparable. The
easiest path is `from_generated_anndata` — your model writes generated cells to a gene-space
`.h5ad`, and the helper fits one PCA on the target and projects both sides into it (target and
generated **must share a feature space**), auto-detecting niche-shaped vs flat:

```python
# my_project/run_eval.py
from paired_slides_eval.pipeline import run_pipeline, from_generated_anndata

from my_model import load_and_sample          # <-- YOUR model, imported in YOUR module

def my_generator(*, source, target, checkpoint, ct_key="class", n_pcs=50, **_):
    # (a) run your model however it likes; write generated cells to a gene-space .h5ad
    load_and_sample(source=source, target=target, checkpoint=checkpoint, out="gen.h5ad")
    # (b) turn that file into the (target, generated) pair the metrics expect
    return from_generated_anndata("gen.h5ad", target, ct_key=ct_key, n_pcs=n_pcs)

res = run_pipeline(
    "source.h5ad", "target.h5ad", "model.ckpt",  # -> your generator's source/target/checkpoint
    generator=my_generator,                       # your function
    classifier="classifier.ckpt",                 # optional: enables the ct/* metrics (or omit)
    ct_key="class", n_pcs=50,                      # extra kwargs -> forwarded into my_generator(**kwargs)
)
print(res.metrics)                                # flat {test/group/metric: value} dict
```

What `run_pipeline` does with your function: (1) calls `my_generator(source=…, target=…,
checkpoint=…, **extra_kwargs)` — any extra keyword you pass to `run_pipeline` lands in your
generator's `**kwargs`; (2) picks the classifier (your `GenerationOutput.classifier`, else the
`classifier=` you passed); (3) runs `evaluate(...)` and returns a `PipelineResult` with `.metrics`,
`.target`, `.generated`.

The generated `.h5ad` follows the `GeneratedNiches.from_anndata` layout: one row per cell,
`obs['niche_id']` grouping each niche (centroid first), coords in `obsm['spatial']`, and optional
paired ground truth in `obsm['gt_x']` / `obsm['gt_pos']` / `obs['gt_ct']`. A whole-slide model
instead emits flat cells (`X` + `obsm['spatial']`, no `niche_id`); the niche metrics are then
skipped.

Prefer full control? Build the dataclasses yourself and return them, instead of using the helper:

```python
from paired_slides_eval import TargetSlide, GeneratedNiches
from paired_slides_eval.pipeline import GenerationOutput

def my_generator(*, source, target, checkpoint, **_):
    target_slide = TargetSlide.from_anndata(target, ct_key="class", n_pcs=50)
    generated = GeneratedNiches.from_anndata("gen.h5ad").project(target_slide.pca)  # share PCA basis
    return GenerationOutput(target=target_slide, generated=generated, classifier=None)
```

### Alternative: generate once, then evaluate (two steps)

Instead of the one-shot `run_pipeline`, split it: **generate** the cells to a file once, then
**evaluate** that file as many times (and on whatever metric subset) as you like without
regenerating. The model-agnostic generate CLI runs *any* generator — pick one by dotted path
(`module.path:callable`); it defaults to the bundled NicheFlow adapter:

```bash
# bring your own model (a Generator callable), write the cells, then evaluate them
python -m paired_slides_eval.generate \
  --generator mypkg.mymodel:my_generator \
  --source source.h5ad --target target.h5ad --checkpoint model.ckpt \
  --generated_out generated.h5ad \
  --gen-kwarg n_pcs=50 --gen-kwarg radius=0.15      # extra opts forwarded to your generator

python -m paired_slides_eval.evaluate --target target.h5ad --generated generated.h5ad
```

`--generated_out` takes `.h5ad` or `.npz`; the layout round-trips through the evaluator's loader.
From Python the same is `generate_cells(source, target, checkpoint, generator=…, out="generated.h5ad")`.
Prefer the low-level NicheFlow API directly? `preprocess_pair(...)` → `generate(...)` →
`gen.to_anndata().write_h5ad("generated.h5ad")` works too. Either way, evaluate it with the CLI below.

## Evaluate (all metrics, or a selected subset)

`python -m paired_slides_eval.evaluate` runs the suite on one `(target slide, generated cells)` pair: a
target `.h5ad` plus generated cells as a `.h5ad` (niche-shaped with `obs['niche_id']`, or flat
`X`+`obsm['spatial']`) or an `.npz` (`x`/`pos`, 3-D for niches or 2-D for flat; optional `gt_*`).

```bash
# ALL applicable metrics (default). Geometry + distribution + C2ST + Moran need no classifier:
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --ct_key class --n_pcs 50 --out results.csv

# a SELECTED subset only:
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --groups c2st moran

# add the classifier groups (concordance + accuracy gap) — needs a TRAINED classifier (see below):
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --ct_key class --n_pcs 50 \
  --classifier Classifier_Spatial.ckpt --out results.csv
```

Groups whose inputs are missing are skipped automatically (no `gt_*` → skips regression; no
`--classifier` → skips the `ct/*` groups) and reported in a `skipped:` line. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--classifier` | none | trained classifier `.ckpt`; enables the `ct/*` groups. `n_neighbors` is read from the checkpoint |
| `--groups` | all | subset, e.g. `--groups c2st moran` |
| `--n_pcs` | none | fit a PCA on the target to N PCs and project the generated cells into it |
| `--expr_key` / `--spatial_key` / `--ct_key` | `X`/`spatial`/none | where expression/coords/labels live |
| `--seed` | `0` | |

## Train the classifier (required for the `ct/*` metrics)

The two classifier metrics (`concordance`, `ct_gap`) need a **trained spatial cell-type
classifier** — there is no default one. Train it on a held-out slide **before** running those
groups. The classifier must live in the **same PCA basis** as the target you evaluate, so it's
projected into the source+target basis at preprocessing time.

**Easiest — let the pipeline train it inline.** `run_pipeline` / the pipeline CLI with
`--classifier <slide.h5ad>` preprocesses the held-out slide into the source+target basis and trains
the classifier automatically — no checkpoint to manage:

```bash
python -m paired_slides_eval.pipeline \
  --source ../nicheflow_mba/data/adata_Zhuang_Zhuang-ABCA-1.024.h5ad \
  --target ../nicheflow_mba/data/adata_Zhuang_Zhuang-ABCA-1.025.h5ad \
  --checkpoint ../nicheflow_mba/ckpts/NicheFlow_CFM_ABCA.ckpt \
  --classifier ../fm_mnist/data/adata_Zhuang_Zhuang-ABCA-1.001.h5ad \
  --variant cfm
```

**Reusable checkpoint (Hydra).** To train once and reuse the `.ckpt` across standalone evaluations,
use the configured entry point. It reads a preprocessed niche `.pkl` via `data.datamodule.data_fp`:

```bash
python -m paired_slides_eval.classifier.train experiment=classifier/abca_spatial \
  data.datamodule.data_fp=PATH/TO/classifier_niches.pkl
```

The checkpoint lands under `outputs/.../checkpoints/`; pass it to evaluation with `--classifier
<ckpt>`. Build the `.pkl` from a raw slide by projecting it into the **same** source+target basis,
then pickling:

```python
import pickle
from paired_slides_eval.adapters.nicheflow import preprocess_pair, preprocess_classifier_slide

_, pre = preprocess_pair("source.h5ad", "target.h5ad", n_pcs=50, cell_type_column="class")
clf_ds = preprocess_classifier_slide("classifier_slide.h5ad", pre, cell_type_column="class")
pickle.dump(clf_ds, open("classifier_niches.pkl", "wb"))   # niche .pkl for the Hydra trainer
```

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
src/paired_slides_eval/
  contract.py          TargetSlide / GeneratedSlide (flat) / GeneratedNiches — the AnnData contract
  data/anndata.py      read raw .h5ad -> arrays; optional shared PCA          [common, model-agnostic]
  data/dataclass.py    the internal niche pickle schema + loader
  evaluate.py          evaluate(target, generated) -> flat dict + standalone CLI
  generate.py          model-agnostic generate-only entry point + CLI (resolve generator, write cells)
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
