"""Optional reference-model checks exercised explicitly in CPU CI."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "variant", ["reference", "no-position", "no-dropout", "no-decay", "fixed-lr"]
)
def test_reference_transformer_training_and_prediction(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "learning/mlp")
    )
    fit_transformer = import_module("transformer_reference").fit_transformer

    rng = np.random.default_rng(42)
    xt, xv = rng.normal(0, 0.01, (16, 64)), rng.normal(0, 0.01, (8, 64))
    yt, yv = np.arange(16) % 2, np.arange(8) % 2
    checkpoint = tmp_path / "reference.pt"
    predict, history, parameters = fit_transformer(
        xt, yt, xv, yv, 7, variant, 2, checkpoint
    )
    assert len(history) == 2
    assert all(np.isfinite(e["val_loss"]) for e in history)
    assert parameters > 0
    probabilities = predict(xv)
    assert probabilities.shape == (8,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_array_equal(probabilities, predict(xv))
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["variant"] == variant
    assert len(saved["edges"]) > 0
