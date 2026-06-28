"""One-shot generate + evaluate entry point (Hydra): ``python -m paired_slides_eval.pipeline``.

Selects a generation adapter from the ``generator`` config group, generates cells from a checkpoint,
and runs the metric suite on them. For library use, call
:func:`paired_slides_eval.pipeline.run.run_pipeline` directly.

Usage::

    python -m paired_slides_eval.pipeline \\
        generator=nicheflow \\
        source=source.h5ad target=target.h5ad checkpoint=model.ckpt \\
        classifier=classifier.ckpt out=results.csv

Requires the ``[pipeline]`` extra (Hydra).
"""

from __future__ import annotations

import csv
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from paired_slides_eval.evaluate import ALL_GROUPS
from paired_slides_eval.pipeline.io import write_generated
from paired_slides_eval.pipeline.run import run_pipeline


@hydra.main(config_path="../../../configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    generator = instantiate(cfg.generator)
    groups = tuple(cfg.groups) if cfg.get("groups") else ALL_GROUPS

    evaluate_kwargs = {"ct_real_reference": cfg.get("ct_real_reference", "paired")}
    if cfg.get("ct_real_n") is not None:
        evaluate_kwargs["ct_real_n"] = cfg.get("ct_real_n")

    res = run_pipeline(
        cfg.source,
        cfg.target,
        cfg.checkpoint,
        generator=generator,
        classifier=cfg.get("classifier"),
        groups=groups,
        seed=cfg.get("seed", 0),
        evaluate_kwargs=evaluate_kwargs,
    )

    if cfg.get("generated_out"):
        write_generated(res.generated, cfg.generated_out)

    metrics = dict(res.metrics)
    skipped = metrics.pop("_skipped", [])
    notes = metrics.pop("_notes", [])
    rows = sorted(metrics.items())
    print(f"target: {res.target.x.shape[0]} cells, {res.target.x.shape[1]} features")
    if res.generated.x.ndim == 3:
        print(f"generated: {res.generated.x.shape[0]} niches x {res.generated.x.shape[1]} points")
    else:
        g = res.generated
        print(f"generated: {g.x.shape[0]} cells, {g.x.shape[1]} feats (flat slide)")
    for k, v in rows:
        print(f"{k:24s} {v:.4f}")
    if notes:
        print("notes:", "; ".join(notes))
    if skipped:
        print("skipped:", "; ".join(skipped))

    out = cfg.get("out")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="") as fh:
            wcsv = csv.writer(fh)
            wcsv.writerow(["metric", "value"])
            wcsv.writerows(rows)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
