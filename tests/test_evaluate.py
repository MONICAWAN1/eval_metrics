import numpy as np

from paired_slides_eval import GeneratedNiches, TargetSlide, evaluate


def _build(real_slide, generated_niches):
    target = TargetSlide(
        x=real_slide["x"],
        pos=real_slide["pos"],
        ct=real_slide["ct"],
        n_classes=real_slide["n_classes"],
    )
    generated = GeneratedNiches(x=generated_niches["x"], pos=generated_niches["pos"])
    return target, generated


def test_evaluate_geometric_groups(real_slide, generated_niches):
    target, generated = _build(real_slide, generated_niches)
    out = evaluate(target, generated, groups=("psd", "spd"))
    assert set(out) - {"_skipped", "_notes"} == {
        "test/psd/mean",
        "test/psd/max",
        "test/spd/mean",
        "test/spd/max",
    }


def test_evaluate_skips_unavailable_groups(real_slide, generated_niches):
    target, generated = _build(real_slide, generated_niches)
    out = evaluate(target, generated, groups=("regression", "concordance", "psd"))
    # regression needs matched GT; concordance needs a classifier -> both skipped, psd runs.
    assert "test/psd/mean" in out
    assert len(out["_skipped"]) == 2


def test_generated_niches_validates_shape(real_slide):
    import pytest

    with pytest.raises(ValueError):
        GeneratedNiches(x=np.zeros((10, 5)), pos=np.zeros((10, 2)))  # 2-D, not (B, N, D)
