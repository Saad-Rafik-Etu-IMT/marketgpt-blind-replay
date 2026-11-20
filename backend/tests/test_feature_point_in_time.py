"""Generic anti-lookahead tests applied to EVERY registered feature.

A feature added in a later tranche is covered automatically: these tests
iterate over the default catalogue rather than naming features.
"""

from __future__ import annotations

import numpy as np
import pytest
from feature_fixtures import perturb_after, synthetic_bars

from src.features import DEFAULT_REGISTRY, compute, vector_at

ALL_FEATURES = DEFAULT_REGISTRY.ids()
BARS = synthetic_bars(320, seed=7)
CUTS = (140, 200, 318)


@pytest.mark.parametrize("feature_id", ALL_FEATURES)
def test_changing_future_bars_never_changes_past_values(feature_id):
    reference = compute(BARS, [feature_id]).values
    for cut in CUTS:
        altered = compute(perturb_after(BARS, cut), [feature_id]).values
        np.testing.assert_array_equal(
            altered.iloc[: cut + 1].to_numpy(),
            reference.iloc[: cut + 1].to_numpy(),
            err_msg=f"{feature_id}: rows <= {cut} changed when only later bars changed",
        )


@pytest.mark.parametrize("feature_id", ALL_FEATURES)
def test_replay_vector_equals_batch_row(feature_id):
    batch = compute(BARS, [feature_id]).values
    for position in (DEFAULT_REGISTRY.get(feature_id).lookback, *CUTS):
        replay = vector_at(BARS, [feature_id], position)
        np.testing.assert_allclose(
            replay.to_numpy(), batch.iloc[position].to_numpy(), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("feature_id", ALL_FEATURES)
def test_warm_up_is_exactly_the_declared_lookback(feature_id):
    spec = DEFAULT_REGISTRY.get(feature_id)
    values = compute(BARS, [feature_id]).values.to_numpy()
    assert np.isnan(values[: spec.lookback]).all()
    assert np.isfinite(values[spec.lookback :]).all()


def test_changing_the_past_does_change_the_present():
    """Sanity check that the mutation tests are not vacuous."""
    altered = BARS.copy()
    altered.iloc[100, altered.columns.get_loc("adjusted_close")] *= 1.01
    altered.iloc[100, altered.columns.get_loc("close")] *= 1.01
    altered.iloc[100, altered.columns.get_loc("high")] *= 1.01
    before = compute(BARS, ["returns_window_64"]).values.iloc[150]
    after = compute(altered, ["returns_window_64"]).values.iloc[150]
    assert not np.allclose(before, after)
