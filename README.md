# paired-slides-eval

A **model-agnostic evaluation library** for generative models on spatial transcriptomics. It scores
a set of **generated cells** against a real **target slide** — each given as plain arrays/AnnData files, an
expression matrix paired with spatial coordinates — and reports the full metric suite. Evaluation is
decoupled from generation: a model's architecture and sampling code stay wherever the model lives,
and this library consumes only the cells it produces. An optional integrated-generation layer is
available for models that ship an adapter (see
[Optional: integrated generation](#optional-integrated-generation)).

## Install

```bash
pip install -e .                 # the metrics — all you need to evaluate
```

Optional extras:

```bash
pip install -e ".[classifier]"   # + the cell-type-classifier training stack (for the ct/* metrics)
pip install -e ".[pipeline]"     # + Hydra, for configuration-driven generation
pip install -e ".[nicheflow]"    # + the bundled NicheFlow generation adapter
pip install -e ".[wandb]"        # + Weights & Biases logging for classifier training
```

Running the bundled NicheFlow adapter needs `[pipeline,nicheflow]` plus the `nicheflow` package
(not on PyPI): `pip install -e ../nicheflow_mba`.

## Quickstart — evaluate two files

Call `evaluate_files` right after your own pipeline writes its generated
cells.

```python
from paired_slides_eval import evaluate_files

metrics = evaluate_files(
    "target.h5ad",        # real slide: raw genes in X + coords in obsm['spatial']
    "generated.h5ad",     # your model's output (flat X+coords, or niche-shaped)
    ct_key="class",       # enables the classifier metrics (optional)
    n_pcs=50,             # shared PCA so target & generated live in one feature space
)
print(metrics)            # {test/group/metric: value, ...}
```

Same thing on the command line:

```bash
python -m paired_slides_eval.evaluate \
  --target target.h5ad --generated generated.h5ad --ct_key class --n_pcs 50 --out results.csv
```

## Inputs

**Target slide** — a plain AnnData (`.h5ad`):

| Field | Where | Notes |
|---|---|---|
| expression | `adata.X` | **raw genes** (default). Or an `obsm`/`layers` key via `expr_key=` if already reduced. |
| coordinates | `adata.obsm["spatial"]` | configurable via `spatial_key=` |
| cell types | `adata.obs[ct_key]` | optional; needed by the classifier metrics |

**Generated cells** — write whatever your model emits to a file, in one of two shapes:

- **`GeneratedSlide` (flat)** — a whole slide: `X` + `obsm["spatial"]`, one row per cell, no
  `niche_id`. For ordinary whole-slide models. The label-free metrics run on it; the classifier
  metrics also run (niches are reconstructed from geometry); only regression is skipped.
- **`GeneratedNiches` (niche-shaped)** — flat rows with `obs["niche_id"]` grouping each niche's
  points (centroid first); optional model-supplied pairing in `obsm["gt_x"]`/`obsm["gt_pos"]` +
  `obs["gt_ct"]`. For microenvironment models; required by regression.

Files are auto-detected by extension and contents: `.h5ad` (niche-shaped if `obs['niche_id']`, else
flat), `.npz` (`x`/`pos`, 3-D for niches or 2-D for flat; optional `gt_*`), or `.pkl` (a dict of
those arrays, or a generator result object).

**One thing to get right — shared feature space.** Target and generated must live in the same
space: either both raw genes (same panel) or both projected through one PCA. Pass `n_pcs=` (or
`--n_pcs`) and the helpers fit the PCA on the target. Projection of the generated cells is then
**auto-detected by feature dimension**: gene-space cells are projected into the target's PCA basis,
while cells a model already produced in PCA space (e.g. a flow that samples latents) are passed
through unchanged rather than double-transformed. For that already-reduced case the target must be
supplied in the *same* basis (a `TargetSlide` with `pca=None`, or an AnnData read via
`expr_key="X_pca"` with `n_pcs=None`).

Build the inputs explicitly if you prefer (instead of `evaluate_files`):

```python
from paired_slides_eval import TargetSlide, GeneratedSlide, evaluate

target = TargetSlide.from_anndata("target.h5ad", ct_key="class", n_pcs=50)
generated = GeneratedSlide.from_anndata("generated.h5ad").project(target.pca)  # share the PCA basis
metrics = evaluate(target, generated)
```

## Evaluate (all metrics, or a subset)

```bash
# ALL applicable metrics (default). Geometry + distribution + C2ST + Moran need no classifier:
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --n_pcs 50 --out results.csv

# a SELECTED subset only:
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --groups c2st moran

# add the classifier groups (concordance + accuracy gap) — needs a TRAINED classifier (see below):
python -m paired_slides_eval.evaluate --target TARGET.h5ad --generated generated.h5ad \
  --ct_key class --n_pcs 50 --classifier Classifier_Spatial.ckpt --out results.csv
```

Groups whose inputs are missing are skipped automatically (a flat slide → skips regression; no
`--classifier` → skips the `ct/*` groups) and reported on a `skipped:` line; anything reconstructed
on the fly (e.g. classifier niches auto-built from a flat slide) is reported on a `notes:` line.

| Flag | Default | Purpose |
|---|---|---|
| `--classifier` | none | trained classifier `.ckpt`; enables the `ct/*` groups. `n_neighbors` read from the checkpoint |
| `--groups` | all | subset, e.g. `--groups c2st moran` |
| `--n_pcs` | none | fit a PCA on the target to N PCs and project the generated cells into it |
| `--expr_key` / `--spatial_key` / `--ct_key` | `X`/`spatial`/none | where expression/coords/labels live |
| `--seed` | `0` | |

## Metrics

The **label-free** groups run on **either** a flat `GeneratedSlide` or `GeneratedNiches` and need no
classifier — this is the default suite any model can use. The **classifier** groups (`ct/*`) are
**advanced**: they need a trained cell-type classifier (see below). Only regression needs
cell-for-cell matched ground truth and is therefore niche-only.

| Group | Keys | Shape | Needs |
|---|---|---|---|
| Point/shape distances | `psd/{mean,max}`, `spd/{mean,max}` | flat or niche | — |
| Distribution | `mmd2/{x,pos}`, `ot_w1/{x,pos}`, `ot_w2/{x,pos}` | flat or niche | `torch`, `pot` |
| C2ST (label-free) | `c2st/{acc,auc,pos_acc,sig_*}` | flat or niche | `sklearn` |
| Moran's I (label-free) | `moran/{mae,corr,real_mean,gen_mean}` | flat or niche | `squidpy` — over **all** generated cells vs the full real slide |
| Pointwise regression | `x/{mse,mae}`, `pos/{mse,mae}` | niche only | matched `gt_*` |
| Cell-type concordance *(advanced)* | `ct/{f1,acc,prop_kl,prop_tv,prop_jsd}` | flat or niche | trained classifier (+ paired niches `gt_*`, auto-built from a flat slide) |
| Classifier accuracy gap *(advanced)* | `ct/{acc_real,acc_gen,acc_gap}` | flat or niche | trained classifier (+ `gt_*` and `gt_ct`; from `ct_key` for a flat slide) |

See `docs/metric_comparison.md` and `docs/metrics.md` for details.

### Advanced: the classifier (`ct/*`) metrics

`concordance` and `ct_gap` compare a **trained spatial cell-type classifier**'s labels on the
generated vs. the real microenvironments. There is no default classifier — train one (extra
`[classifier]`) on a held-out slide, in the **same PCA basis** as the target you evaluate, then pass
it with `--classifier <ckpt>`. Omit it and these two groups are simply skipped; the rest of the
suite runs. The full training pipeline lives under `paired_slides_eval.classifier` (Hydra entry
point `python -m paired_slides_eval.classifier.train`).

## Optional: integrated generation

Given a trained checkpoint and the model's code, the package can generate cells and evaluate them in
one step. Generation is configuration-driven (Hydra): a generation **adapter** wraps a model behind
the `BaseGenerator` contract, and a config selects and constructs it via `_target_`. This path
requires the `[pipeline]` extra.

Run generation, then evaluate:

```bash
python -m paired_slides_eval.generate \
  generator=nicheflow \
  source=source.h5ad target=target.h5ad checkpoint=model.ckpt generated_out=generated.h5ad
python -m paired_slides_eval.evaluate --target target.h5ad --generated generated.h5ad
```

Or generate and evaluate in one command:

```bash
python -m paired_slides_eval.pipeline \
  generator=nicheflow \
  source=source.h5ad target=target.h5ad checkpoint=model.ckpt \
  classifier=classifier.ckpt out=results.csv
```

Per-run parameter overrides use Hydra syntax, e.g. `generator.n_pcs=50 generator.radius=0.2`.

### Adding a model

1. Add an adapter under `src/paired_slides_eval/adapters/<name>/`: a `BaseGenerator` subclass whose
   constructor takes the model's parameters and whose call returns a `GenerationOutput`.

   ```python
   from paired_slides_eval.adapters.base import BaseGenerator
   from paired_slides_eval.pipeline import from_generated_arrays
   from my_model import sample                      # the model package, imported here

   class MyGenerator(BaseGenerator):
       def __init__(self, n_pcs=50):
           self.n_pcs = n_pcs

       def __call__(self, *, source, target, checkpoint, **_):
           x, pos = sample(source, target, checkpoint)
           return from_generated_arrays(x, pos, target, n_pcs=self.n_pcs)
   ```

   `from_generated_arrays` (and `from_generated_anndata`) reconcile the feature space automatically:
   gene-space cells are projected through the target PCA; cells already in PCA space are passed
   through unchanged (supply a `TargetSlide` already in that basis).

2. Add `configs/generator/<name>.yaml` pointing `_target_` at the class:

   ```yaml
   _target_: paired_slides_eval.adapters.<name>.MyGenerator
   n_pcs: 50
   ```

3. Select it with `generator=<name>`.

For library use, a `BaseGenerator` instance (or any matching callable) is accepted directly:
`run_pipeline("source.h5ad", "target.h5ad", "model.ckpt", generator=MyGenerator())`.

## Layout

```
src/paired_slides_eval/
  evaluate.py          evaluate() / evaluate_files() + the standalone CLI   [general — the headline]
  contract.py          TargetSlide / GeneratedSlide / GeneratedNiches       [general]
  data/anndata.py      read raw .h5ad -> arrays; optional shared PCA        [general]
  metrics/             the metric kernels (distances, distribution, c2st, morans, concordance,
                       classifier_gap) — Moran/classifier niches built locally
  pipeline/            generation orchestration: run_pipeline, the Generator/BaseGenerator contract,
                       and generated-cell I/O                               [extra: pipeline]
  generate.py          generate entry point (Hydra)                          [extra: pipeline]
  adapters/base.py     BaseGenerator contract
  adapters/nicheflow/  the NicheFlow adapter + its .pkl I/O                  [extra: nicheflow]
  classifier/          cell-type-classifier training for the ct/* metrics   [extra: classifier]
configs/               hydra configs (generate / pipeline / generator / classifier training)
tests/                 synthetic-data metric tests (no real data needed)
```
