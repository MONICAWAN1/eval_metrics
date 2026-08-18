import json

import numpy as np
import pytest

from paired_slides_eval.adapters.dkd_nicheflow import (
    held_out_condition_macro,
    load_dkd_generated,
)


def test_load_dkd_generated_requires_audited_sidecar(tmp_path):
    path = tmp_path / "generated.npz"
    np.savez(path, x=np.zeros((2, 3, 4)), pos=np.zeros((2, 3, 2)))
    path.with_suffix(".json").write_text(
        json.dumps({"seed": 0, "checkpoint": "x.pt", "condition": "DKD", "held_out_group": "g1"})
    )
    generated, metadata = load_dkd_generated(path)
    assert generated.x.shape == (2, 3, 4)
    assert metadata["held_out_group"] == "g1"


def test_load_dkd_generated_rejects_missing_group(tmp_path):
    path = tmp_path / "generated.npz"
    np.savez(path, x=np.zeros((2, 3, 4)), pos=np.zeros((2, 3, 2)))
    path.with_suffix(".json").write_text(
        json.dumps({"seed": 0, "checkpoint": "x.pt", "condition": "DKD"})
    )
    with pytest.raises(ValueError, match="held_out_group"):
        load_dkd_generated(path)


def test_load_dkd_generated_accepts_audited_real_control(tmp_path):
    path = tmp_path / "real_control.npz"
    np.savez(path, x=np.zeros((2, 3, 4)), pos=np.zeros((2, 3, 2)))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "artifact_kind": "held_out_real_niches",
                "seed": 0,
                "condition": "Control",
                "held_out_group": "control_1",
            }
        )
    )
    generated, metadata = load_dkd_generated(path)
    assert generated.x.shape == (2, 3, 4)
    assert metadata["artifact_kind"] == "held_out_real_niches"


def test_condition_macro_weights_conditions_equally():
    rows = [
        {
            "condition": "a",
            "held_out_group": "a1",
            "seed": 0,
            "comparison": "m",
            "metric": "auc",
            "value": 0.6,
        },
        {
            "condition": "a",
            "held_out_group": "a2",
            "seed": 0,
            "comparison": "m",
            "metric": "auc",
            "value": 0.8,
        },
        {
            "condition": "b",
            "held_out_group": "b1",
            "seed": 0,
            "comparison": "m",
            "metric": "auc",
            "value": 0.4,
        },
    ]
    assert held_out_condition_macro(rows)[0]["value"] == pytest.approx(0.55)
