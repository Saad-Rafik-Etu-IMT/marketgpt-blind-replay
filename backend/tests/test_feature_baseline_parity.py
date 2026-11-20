"""The registered baseline equals the inputs the existing players already use.

- F3 ``build_observation``: what a replay player sees at a decision bar
  (bit-identical: same ``log(p1 / p0)`` formula).
- W1 ``make_xy``: the training windows of the reference MLP.
- ``deep_mlp.extract_blind_features``: its first 64 columns are the raw window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from feature_fixtures import synthetic_bars

from src.data.anonymizer import build_observation
from src.features import BASELINE_FEATURES, compute
from src.schemas.market import OHLCVBar

LEARNING = Path(__file__).resolve().parents[1] / "learning"
BARS = synthetic_bars(260, seed=3)
POSITIONS = (64, 130, 259)


def _ohlcv(frame):
    return [
        OHLCVBar(date=day.date(), volume=int(row.pop("volume")), **row)
        for day, row in frame.to_dict("index").items()
    ]


@pytest.fixture(scope="module")
def baseline():
    return compute(BARS, BASELINE_FEATURES).values.to_numpy()


def test_matches_blind_observation(baseline):
    bars = _ohlcv(BARS)
    for t in POSITIONS:
        observation = build_observation(bars[: t + 1], 64)
        np.testing.assert_allclose(
            baseline[t], observation.log_returns, rtol=0, atol=1e-15
        )


@pytest.fixture
def learning_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(LEARNING / "mlp"))
    monkeypatch.syspath_prepend(str(LEARNING / "analysis"))
    import deep_mlp
    import step2_windows

    return step2_windows, deep_mlp


def test_matches_w1_training_windows_and_deep_features(baseline, learning_modules):
    step2_windows, deep_mlp = learning_modules
    # W1 indexing: values[i] is the return ending at bar i + 1, so the window
    # of example t is the feature row of bar t + 1.
    values = np.diff(np.log(BARS["adjusted_close"].to_numpy()))
    examples = tuple(t - 1 for t in POSITIONS[:-1])
    x, _ = step2_windows.make_xy(values, examples, 64)
    # W1 computes diff(log(p)); F3 and the engine compute log(p1 / p0). Equal in
    # exact arithmetic, they differ by floating-point rounding (~1e-15).
    np.testing.assert_allclose(x, baseline[list(POSITIONS[:-1])], rtol=0, atol=1e-14)
    deep = deep_mlp.extract_blind_features(x)
    np.testing.assert_array_equal(deep[:, :64], x)
