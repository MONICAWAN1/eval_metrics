"""The model-agnostic generate entry point: generator resolution + write/round-trip."""

import numpy as np
import pytest

from paired_slides_eval import generate_cells, write_generated
from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide
from paired_slides_eval.evaluate import _load_generated
from paired_slides_eval.generate import _coerce, resolve_generator
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


# --- resolve_generator -------------------------------------------------------

def test_resolve_generator_passthrough_callable():
    assert resolve_generator(_slide_generator) is _slide_generator


def test_resolve_generator_dotted_path():
    # any importable callable resolves; write_generated is a convenient real target
    assert resolve_generator("paired_slides_eval.generate:write_generated") is write_generated


@pytest.mark.parametrize("spec", ["nocolon", "paired_slides_eval.generate:does_not_exist"])
def test_resolve_generator_bad_spec_raises(spec):
    with pytest.raises((ValueError, AttributeError)):
        resolve_generator(spec)


def test_resolve_generator_non_callable_raises():
    with pytest.raises(TypeError):
        resolve_generator("paired_slides_eval.generate:DEFAULT_GENERATOR")  # a str, not callable


# --- _coerce -----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [("none", None), ("null", None), ("true", True), ("false", False),
     ("50", 50), ("0.15", 0.15), ("euler", "euler")],
)
def test_coerce(text, expected):
    assert _coerce(text) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        assert isinstance(_coerce(text), int)


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


# --- generate_cells orchestration -------------------------------------------

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
