"""Loading the NicheFlow processed `.pkl` format: target slide + generated cells."""

import pickle

import numpy as np
import pytest

from paired_slides_eval import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.data.dataclass import H5ADDatasetDataclass
from paired_slides_eval.evaluate import _load_generated


@pytest.fixture
def niche_dataclass(rng):
    """A minimal `H5ADDatasetDataclass` (NicheFlow preprocessed pickle) with two slides A, B."""
    na, nb = 30, 40
    x = np.vstack([rng.random((na, 5)), rng.random((nb, 5))]).astype(np.float32)
    coords = np.vstack([rng.random((na, 2)), rng.random((nb, 2))]).astype(np.float32)
    ct = np.array((["T", "B", "T"] * 100)[: na + nb], dtype=object)
    return H5ADDatasetDataclass(
        X_pca=x, coords=coords, ct=ct, PCs=np.zeros((5, 5)),
        timepoints_ordered=["A", "B"], timepoint_column="slide",
        timepoint_to_int={"A": 0, "B": 1},
        timepoint_indices={"A": np.arange(na), "B": np.arange(na, na + nb)},
        ct_column="class", ct_ordered=["T", "B"], ct_to_int={"T": 0, "B": 1},
        timepoint_neighboring_indices={}, timepoint_num_neighbors={},
        subsampled_timepoint_idx={}, standardize_coordinates=True,
        radius=0.15, dx=0.15, dy=0.2, stats={}, test_microenvs=0,
    )


def test_target_from_dataclass_object(niche_dataclass):
    t = TargetSlide.from_dataclass(niche_dataclass)  # default: last timepoint (B)
    assert t.x.shape == (40, 5) and t.pos.shape == (40, 2)
    assert t.n_classes == 2 and t.ct.dtype == np.int64
    assert set(t.ct.tolist()) <= {0, 1}
    assert t.pca is None  # X_pca already reduced
    # explicit timepoint selects the other slide
    assert TargetSlide.from_dataclass(niche_dataclass, timepoint="A").x.shape == (30, 5)


def test_target_from_dataclass_path(niche_dataclass, tmp_path):
    p = tmp_path / "slide.pkl"
    p.write_bytes(pickle.dumps(niche_dataclass))
    assert TargetSlide.from_dataclass(str(p)).x.shape == (40, 5)


def test_target_from_dataclass_integer_labels(niche_dataclass):
    niche_dataclass.ct = np.array(([0, 1, 0] * 100)[:70], dtype=np.int64)
    t = TargetSlide.from_dataclass(niche_dataclass)
    assert t.ct.dtype == np.int64 and set(t.ct.tolist()) <= {0, 1}


def test_load_generated_pkl_flat_dict(rng, tmp_path):
    p = tmp_path / "g.pkl"
    p.write_bytes(pickle.dumps({"x": rng.random((10, 5)), "pos": rng.random((10, 2))}))
    g = _load_generated(str(p))
    assert isinstance(g, GeneratedSlide) and g.x.shape == (10, 5)


def test_load_generated_pkl_niche_dict(rng, tmp_path):
    p = tmp_path / "g.pkl"
    p.write_bytes(pickle.dumps({
        "x": rng.random((4, 6, 5)), "pos": rng.random((4, 6, 2)),
        "gt_x": rng.random((4, 6, 5)), "gt_pos": rng.random((4, 6, 2)), "gt_ct": np.arange(4),
    }))
    g = _load_generated(str(p))
    assert isinstance(g, GeneratedNiches) and g.gt_ct.shape == (4,)


class _FakeGenerationResult:
    """Module-level (so it pickles) stand-in for the NicheFlow adapter's GenerationResult."""

    def to_generated_niches(self):
        return GeneratedNiches(x=np.zeros((2, 3, 5)), pos=np.zeros((2, 3, 2)))


def test_load_generated_pkl_generationresult_ducktype(tmp_path):
    p = tmp_path / "g.pkl"
    p.write_bytes(pickle.dumps(_FakeGenerationResult()))
    assert isinstance(_load_generated(str(p)), GeneratedNiches)


def test_load_generated_pkl_rejects_slide_dataclass(niche_dataclass, tmp_path):
    p = tmp_path / "g.pkl"
    p.write_bytes(pickle.dumps(niche_dataclass))  # a real slide, not generated cells
    with pytest.raises(ValueError, match="TargetSlide.from_dataclass"):
        _load_generated(str(p))
