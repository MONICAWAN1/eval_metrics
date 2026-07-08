"""Generation orchestration: generator instances, generate_cells, write_generated round-trips."""

import numpy as np
import pytest

from paired_slides_eval import generate_cells, write_generated
from paired_slides_eval.adapters.base import BaseGenerator
from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide
from paired_slides_eval.loaders import _load_generated
from paired_slides_eval.pipeline import run_pipeline
from paired_slides_eval.pipeline.run import GenerationOutput


def _niche_generator(*, source, target, checkpoint, **kw):
    rng = np.random.default_rng(0)
    g = GeneratedNiches(
        x=rng.random((4, 6, 5)), pos=rng.random((4, 6, 2)),
        gt_x=rng.random((4, 6, 5)), gt_pos=rng.random((4, 6, 2)), gt_ct=np.arange(4),
    )
    return GenerationOutput(target=None, generated=g)


def _slide_generator(*, source, target, checkpoint, **kw):
    rng = np.random.default_rng(1)
    return GenerationOutput(
        target=None, generated=GeneratedSlide(x=rng.random((10, 5)), pos=rng.random((10, 2)))
    )


class _SlideGenerator(BaseGenerator):
    """A BaseGenerator subclass (the shape a Hydra `_target_` instantiates)."""

    def __init__(self, n_cells=10, n_feat=5):
        self.n_cells = n_cells
        self.n_feat = n_feat

    def __call__(self, *, source, target, checkpoint, **kw):
        rng = np.random.default_rng(2)
        return GenerationOutput(
            target=None,
            generated=GeneratedSlide(
                x=rng.random((self.n_cells, self.n_feat)), pos=rng.random((self.n_cells, 2))
            ),
        )


# --- write_generated round-trips through the eval loader ---------------------

@pytest.mark.parametrize("ext", [".h5ad", ".npz"])
def test_niche_round_trip(tmp_path, ext):
    out = generate_cells("s", "t", "ckpt", generator=_niche_generator, out=str(tmp_path / f"g{ext}"))
    back = _load_generated(str(tmp_path / f"g{ext}"))
    assert isinstance(back, GeneratedNiches)
    assert back.x.shape == (4, 6, 5) and back.gt_x.shape == (4, 6, 5)
    assert np.array_equal(back.gt_ct, np.arange(4))
    assert np.allclose(back.x, out.generated.x, atol=1e-5)


@pytest.mark.parametrize("ext", [".h5ad", ".npz"])
def test_flat_round_trip(tmp_path, ext):
    generate_cells("s", "t", "ckpt", generator=_slide_generator, out=str(tmp_path / f"g{ext}"))
    back = _load_generated(str(tmp_path / f"g{ext}"))
    assert isinstance(back, GeneratedSlide) and back.x.shape == (10, 5)


def test_unsupported_extension_raises(tmp_path):
    g = _slide_generator(source="s", target="t", checkpoint="c").generated
    with pytest.raises(ValueError, match="\\.h5ad.*\\.npz"):
        write_generated(g, str(tmp_path / "g.txt"))


# --- generate_cells / run_pipeline accept a callable or a BaseGenerator instance ---

def test_generate_cells_out_none_writes_nothing(tmp_path):
    res = generate_cells("s", "t", "ckpt", generator=_slide_generator, out=None)
    assert isinstance(res.generated, GeneratedSlide)
    assert not list(tmp_path.iterdir())  # nothing written


def test_generate_cells_forwards_kwargs():
    seen = {}

    def _echo(*, source, target, checkpoint, **kw):
        seen.update(kw)
        return _slide_generator(source=source, target=target, checkpoint=checkpoint)

    generate_cells("s", "t", "ckpt", generator=_echo, n_pcs=50, radius=0.15)
    assert seen == {"n_pcs": 50, "radius": 0.15}


def test_base_generator_instance_works(tmp_path):
    gen = _SlideGenerator(n_cells=8, n_feat=5)  # what instantiate(cfg.generator) yields
    out = generate_cells("s", "t", "ckpt", generator=gen, out=str(tmp_path / "g.h5ad"))
    assert out.generated.x.shape == (8, 5)
    assert isinstance(_load_generated(str(tmp_path / "g.h5ad")), GeneratedSlide)


def test_run_pipeline_accepts_generator_instance(real_slide):
    from paired_slides_eval import TargetSlide

    class _G(BaseGenerator):
        def __call__(self, *, source, target, checkpoint, **kw):
            rng = np.random.default_rng(3)
            tgt = TargetSlide(x=real_slide["x"], pos=real_slide["pos"],
                              ct=real_slide["ct"], n_classes=real_slide["n_classes"])
            gen = GeneratedSlide(x=rng.random((40, real_slide["x"].shape[1])),
                                 pos=rng.uniform(0, 10, size=(40, 2)))
            return GenerationOutput(target=tgt, generated=gen)

    res = run_pipeline("s", "t", "ckpt", generator=_G(), groups=("regression",))
    assert any("regression" in item for item in res.metrics["_skipped"])
