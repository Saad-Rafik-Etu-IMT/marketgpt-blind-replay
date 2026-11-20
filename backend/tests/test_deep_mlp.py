"""Regression tests for the deep MLP learning module (training audit fixes)."""

from pathlib import Path

import numpy as np
import pytest

LEARNING = Path(__file__).resolve().parents[1] / "learning"


@pytest.fixture
def deep(monkeypatch):
    monkeypatch.syspath_prepend(str(LEARNING / "mlp"))
    monkeypatch.syspath_prepend(str(LEARNING / "analysis"))
    import deep_mlp

    return deep_mlp


def test_oracle_never_reads_past_the_sets_last_label_bar(deep):
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.01, 200)
    indices = tuple(range(70, 120))  # last label bar = 120
    before = deep.compute_best_decisions(values, indices, horizon=3)
    tampered = values.copy()
    tampered[121:] = 0.5  # huge moves in the gap after the set
    after = deep.compute_best_decisions(tampered, indices, horizon=3)
    np.testing.assert_array_equal(before["labels"], after["labels"])
    np.testing.assert_allclose(before["weights"], after["weights"])
    assert [r["forward3dReturn"] for r in before["records"]] == [
        r["forward3dReturn"] for r in after["records"]
    ]


def test_retained_epoch_is_flagged_once(deep):
    rng = np.random.default_rng(1)
    xt, xv = rng.normal(0, 0.01, (120, 64)), rng.normal(0, 0.01, (40, 64))
    yt, yv = rng.integers(0, 2, 120), rng.integers(0, 2, 40)
    _, history, _ = deep.train_deep_mlp(xt, yt, xv, yv, seed=0, epochs=3, batch_size=32)
    assert sum(h["selected"] for h in history) == 1


def test_deep_mlp_backward_matches_finite_differences(deep):
    import gradcheck

    for dropout in (0.0, 0.25):
        errors = gradcheck.gradient_check(dropout=dropout)
        assert max(errors.values()) < 1e-4, errors


def test_overview_reports_the_retained_epoch_loss():
    from src.services.study_artifacts import overview_entry

    report = dict(
        model="MLP",
        variant="reference",
        seed=0,
        macroF1=0.5,
        curves=[
            dict(step=1, trainingLoss=0.7, validationLoss=0.70, selected=True),
            dict(step=2, trainingLoss=0.6, validationLoss=0.65, selected=False),
        ],
        cases=[dict(p=0.6, next=0.01)],
    )
    assert overview_entry("MLP-reference-0", report)["loss"] == 0.70
    for curve in report["curves"]:
        curve.pop("selected")
    assert overview_entry("MLP-reference-0", report)["loss"] == 0.65
