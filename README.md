# Evaluation Metrics for Generative Models on Spatial Transcriptomics

This repository is a **model-agnostic evaluation library** for generative models on spatial transcriptomics. It scores
a set of **generated cells** against a real **target slide** — an expression matrix paired with
spatial coordinates, given as plain arrays/AnnData files — and reports the full metric suite.
Evaluation is decoupled from generation: a model's architecture and sampling code stay wherever the
model lives, and this library consumes only the cells it produces. An optional integrated-generation
layer is available for models that ship an adapter (see [Models & adapters](#models--adapters)).

## Package: paired_slides_eval

Install the metric suite with [uv](https://docs.astral.sh/uv/) (each subproject has its own
per-project `.venv` and lock file):

```bash
uv sync                          # the metrics — all you need to evaluate
uv sync --group dev              # development tools: tests, formatters, pre-commit
```

The package is organized as:

- **`evaluate` / `contract`** — the headline API (`evaluate_files`, `evaluate`) and the
  `TargetSlide` / `GeneratedSlide` / `GeneratedNiches` data contract. The `evaluate` orchestrator is
  supported by focused modules: `loaders` (read generated files), `reconcile` (map coordinates into
  the target frame), `probes` (load the classifier/regressor + build paired niches), and `cli` (the
  command line; `python -m paired_slides_eval.evaluate` still works).
- **`metrics/`** — the metric kernels (distances, distribution, c2st, morans, concordance,
  classifier_gap); Moran's I and classifier niches are assembled locally.
- **`data/`** — raw `.h5ad` → arrays, plus the shared-space `Basis` (one PCA + coordinate frame) that
  keeps target and generated in one feature space.
- **`adapters/`** — generation adapters (NicheFlow, OT-CFM) behind the `BaseGenerator` contract
  *(extra: `pipeline`)*.
- **`classifier/`** — cell-type-classifier training for the `ct/*` metrics *(extra: `classifier`)*.
- **`pipeline/` / `generate`** — Hydra-driven generation orchestration *(extra: `pipeline`)*.

### Optional Dependencies

The core install is the metric suite only. Everything heavier is an opt-in extra:

```bash
uv sync --extra classifier       # + the cell-type-classifier training stack (for the ct/* metrics)
uv sync --extra pipeline         # + Hydra, for configuration-driven generation
uv sync --extra nicheflow        # + the bundled NicheFlow generation adapter
uv sync --extra wandb            # + Weights & Biases logging for classifier training
```

Extras compose (`uv sync --extra pipeline --extra nicheflow`). Running the bundled NicheFlow adapter
additionally needs the `nicheflow` package, which is not on PyPI:
`uv pip install -e ../nicheflow_mba`.

## Development Workflow

For development, use `uv` from the repository root:

```bash
uv sync --group dev
```

Install the formatting/standardization hooks once per clone:

```bash
uv run --group dev pre-commit install
```

Run the same checks manually before pushing:

```bash
uv run --group dev pre-commit run --all-files
```

The pre-commit workflow standardizes YAML, Python imports/formatting, docstrings, Markdown, trailing
commas, trailing whitespace, and final newlines. Focused tests use the same environment:

```bash
uv run --group dev pytest
```

## Quickstart

Call `evaluate_files` right after your own pipeline writes its generated cells.

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
uv run python -m paired_slides_eval.evaluate \
  --target target.h5ad --generated generated.h5ad --ct_key class --n_pcs 50 --out results.csv
```

### As a results table

`paired_slides_eval.me` (the metrics namespace) wraps the same call and returns a tidy
`pandas.DataFrame` — the numbers are identical, just shaped like a benchmarking table. Concatenate
several models into one comparison table with `compare`:

```python
import paired_slides_eval as pse

# one model -> metrics as rows, one value column
df = pse.me.metrics_files("target.h5ad", "otcfm.h5ad", name="otcfm", ct_key="class", n_pcs=50)

# several models -> a wide metrics x models table
table = pse.me.compare(
    {"otcfm": ("shared_pair.pkl", "otcfm.h5ad"),
     "nicheflow": ("shared_pair.pkl", "nicheflow.h5ad")},
    from_files=True, ct_key="class",
)
```

`pse.me` is the metrics namespace and `pse.pp` the preprocessing namespace. The shared space is
defined once as `pp.Basis` (`normalize → log1p → PCA → whiten` for expression, `(pos − mean) / std`
for coordinates): `pp.fit_basis(source, target)` fits it, `basis.apply(genes, coords)` places any
slide into it, and `basis.to_fm_npz(path)` exports it so a model can be trained in the same space.

## Inputs

**Target slide** — a plain AnnData (`.h5ad`):

| Field       | Where                   | Notes                                                                                  |
| ----------- | ----------------------- | -------------------------------------------------------------------------------------- |
| expression  | `adata.X`               | **raw genes** (default). Or an `obsm`/`layers` key via `expr_key=` if already reduced. |
| coordinates | `adata.obsm["spatial"]` | configurable via `spatial_key=`                                                        |
| cell types  | `adata.obs[ct_key]`     | optional; needed by the classifier metrics                                             |

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

## Metrics

The **label-free** groups run on **either** a flat `GeneratedSlide` or `GeneratedNiches` and need no
classifier — this is the default suite any model can use. The **classifier** groups (`ct/*`) are
**advanced**: they need a trained cell-type classifier (see below). Only regression needs
cell-for-cell matched ground truth and is therefore niche-only. Metrics split conceptually along two
axes — **expression / distribution fidelity** (the `x` variants: `mmd2/x`, `ot_w*/x`, gene-C2ST) and
**spatial & joint fidelity** (Moran's I, joint C2ST, the `pos` distribution variants, and the `ct/*`
concordance groups).

| Group                                | Keys                                             | Shape         | Needs                                                                     |
| ------------------------------------ | ------------------------------------------------ | ------------- | ------------------------------------------------------------------------- |
| Distribution                         | `mmd2/{x,pos}`, `ot_w1/{x,pos}`, `ot_w2/{x,pos}` | flat or niche | `torch`, `pot`                                                            |
| C2ST (label-free)                    | `c2st/{acc,auc,gene_acc,gene_auc}`               | flat or niche | `sklearn`                                                                 |
| C2ST nearest-neighbor (label-free)   | `c2st/{nn,nn_std,nn_real_ref}`                   | flat or niche | `scipy`                                                                   |
| Moran's I (label-free)               | `moran/{mae,corr,real_mean,gen_mean}`            | flat or niche | `squidpy` — over **all** generated cells vs the full real slide           |
| Pointwise regression                 | `x/{mse,mae}`, `pos/{mse,mae}`                   | niche only    | matched `gt_*`                                                            |
| Expression reconstruction            | `recon/{mse_gen,mse_real,mse_gap}`               | flat or niche | trained regressor (`--regressor`)                                         |
| Cell-type concordance *(advanced)*   | `ct/{f1,acc,prop_kl,prop_tv,prop_jsd}`           | flat or niche | trained classifier (+ paired niches `gt_*`, auto-built from a flat slide) |
| Classifier accuracy gap *(advanced)* | `ct/{acc_real,acc_gen,acc_gap}`                  | flat or niche | trained classifier (+ `gt_*` and `gt_ct`; from `ct_key` for a flat slide) |

Example metric tables and per-model write-ups live under `reports/` (e.g.
`reports/model_comparison_shared50.csv`, `reports/otcfm1025_vs_nicheflow.md`).

### Advanced: the classifier (`ct/*`) metrics

`concordance` and `ct_gap` compare a **trained spatial cell-type classifier**'s labels on the
generated vs. the real microenvironments. There is no default classifier — train one (extra
`classifier`) on a held-out slide, in the **same PCA basis** as the target you evaluate, then pass it
with `--classifier <ckpt>`. Omit it and these two groups are simply skipped; the rest of the suite
runs. The full training pipeline lives under `paired_slides_eval.classifier` (Hydra entry point
`uv run python -m paired_slides_eval.classifier.train`).

## Models & adapters

Any model can be evaluated with no adapter at all: write the cells it emits to a flat or
niche-shaped file (see [Inputs](#inputs)) and point `evaluate_files` at it. Models that ship an
adapter can additionally be *driven* by this library through the integrated-generation layer:

- **NicheFlow** — microenvironment flow-matching model; CFM (regresses the conditional velocity) and
  VFM (regresses the endpoint) variants. Adapter in `adapters/nicheflow/` *(extras: `pipeline`,
  `nicheflow`, plus `../nicheflow_mba`)*.
- **OT-CFM** — OT conditional flow matching (Tong et al.); a flat baseline (`otcfm`) and a
  coordinate-generating spatial baseline (`otcfm_spatial`) that also produces positions.
- **Bring your own** — any other model: emit a file in the expected shape and evaluate it directly.

### Integrated generation

Given a trained checkpoint and the model's code, the package can generate cells and evaluate them in
one step. Generation is configuration-driven (Hydra): a generation **adapter** wraps a model behind
the `BaseGenerator` contract, and a config selects and constructs it via `_target_`. This path
requires the `pipeline` extra.

Run generation, then evaluate:

```bash
uv run python -m paired_slides_eval.generate \
  generator=nicheflow \
  source=source.h5ad target=target.h5ad checkpoint=model.ckpt generated_out=generated.h5ad
uv run python -m paired_slides_eval.evaluate --target target.h5ad --generated generated.h5ad
```

Or generate and evaluate in one command:

```bash
uv run python -m paired_slides_eval.pipeline \
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

1. Add `configs/generator/<name>.yaml` pointing `_target_` at the class:

   ```yaml
   _target_: paired_slides_eval.adapters.<name>.MyGenerator
   n_pcs: 50
   ```

1. Select it with `generator=<name>`.

For library use, a `BaseGenerator` instance (or any matching callable) is accepted directly:
`run_pipeline("source.h5ad", "target.h5ad", "model.ckpt", generator=MyGenerator())`.

## Resources

- **`reports/`** — example metric tables and per-model evaluation write-ups
  (`model_comparison_shared50.csv`, `model_comparison_1025.md`, `otcfm1025_vs_nicheflow.md`, …).
- **`notebooks/evaluation.ipynb`** — an end-to-end generate → evaluate → visualize walkthrough.
- **`configs/generator/`** — the shipped generator configs (`nicheflow`, `otcfm`, `otcfm_spatial`).
- **Sibling projects** — `../nicheflow_mba` (the NicheFlow fork this adapter wraps) and `../fm_mnist`
  (origin of the OT-CFM baseline and the shared `mmd2_rbf` / `ot_distance` metric kernels).

## References

The metric kernels and bundled adapters build on:

- **NicheFlow** — Sakalyan, Palma, Guerranti, Theis. *Modeling Microenvironment Trajectories on
  Spatial Transcriptomics with NicheFlow* (2025).
- **OT-CFM** — Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch
  Optimal Transport* (2024).
- The `mmd2_rbf` (mixture-RBF MMD) and `ot_distance` (exact-EMD W1/W2) kernels originated in
  `../fm_mnist` and are shared, in spirit, across the workspace.

## Development

- **Environment.** `uv sync --group dev --extra classifier --extra pipeline` (or the subset you need); each
  subproject keeps its own `.venv` and lock file — don't `pip install` into a base environment.
- **Lint & format.** `uv run --group dev pre-commit run --all-files`; for a quick Python-only check,
  `uv run --group dev ruff check .`.
- **Tests.** `uv run --group dev pytest` — synthetic data only, no real data needed. The suite skips
  cleanly when optional heavy deps (`torch`, `squidpy`, `lightning`, …) are absent; keep it that way.

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
