"""Generation entry point (Hydra).

Selects a generation adapter from configuration and runs it for one (source, target, checkpoint),
writing the generated cells to disk. The adapter class is chosen by the ``generator`` config group
(``configs/generator/<name>.yaml``) and constructed from its ``_target_`` and parameters; switching
models is a single config override.

Usage::

    python -m paired_slides_eval.generate \\
        generator=nicheflow \\
        source=source.h5ad target=target.h5ad checkpoint=model.ckpt generated_out=generated.h5ad

The written file uses the layout the evaluator reads, so the generated cells can be scored with
``python -m paired_slides_eval.evaluate``. This module requires the ``[pipeline]`` extra (Hydra).
"""

from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from paired_slides_eval.contract import GeneratedNiches
from paired_slides_eval.pipeline.io import write_generated


@hydra.main(config_path="../../configs", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    generator = instantiate(cfg.generator)
    output = generator(source=cfg.source, target=cfg.target, checkpoint=cfg.checkpoint)

    path = write_generated(output.generated, cfg.generated_out)
    g = output.generated
    if isinstance(g, GeneratedNiches):
        shape = f"{g.x.shape[0]} niches x {g.x.shape[1]} points"
    else:
        shape = f"{g.x.shape[0]} cells, {g.x.shape[1]} feats (flat slide)"
    print(f"generated: {shape} -> {path}")


if __name__ == "__main__":
    main()
