"""F3 blind-observation tests: synthetic bars only, no network or market data."""

from __future__ import annotations

import dataclasses
import math
from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from src.data.anonymizer import (
    BlindObservation,
    InsufficientHistoryError,
    ObservationError,
    build_observation,
)
from src.database import Base, build_engine, build_session_factory
from src.models.market import OHLCVBarORM
from src.schemas.market import OHLCVBar
from src.services.replay_engine import create_session, get_visible_bars, step_session

LOOKBACK = 64


def trading_days(n: int, start: date = date(2021, 1, 4)) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def adjusted(i: int) -> float:
    """Deterministic, non-monotonic positive series."""
    return 100.0 * (1.0 + 0.01 * math.sin(i)) * (1.0005**i)


def make_bars(n: int, *, scale: float = 1.0, start: date = date(2021, 1, 4)) -> list[OHLCVBar]:
    bars = []
    for i, day in enumerate(trading_days(n, start)):
        close = 400.0 * scale + i
        bars.append(
            OHLCVBar(
                date=day,
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1_000 + i,
                adjusted_close=adjusted(i) * scale,
            )
        )
    return bars


# --- Shape and indexing -------------------------------------------------------


def test_observation_holds_lookback_returns_oldest_first() -> None:
    bars = make_bars(LOOKBACK + 1)
    obs = build_observation(bars, LOOKBACK)

    assert len(obs.log_returns) == LOOKBACK
    assert obs.bar_offsets == tuple(range(-63, 1))
    assert obs.log_returns[0] == pytest.approx(math.log(adjusted(1) / adjusted(0)))
    assert obs.log_returns[-1] == pytest.approx(math.log(adjusted(64) / adjusted(63)))


def test_off_by_one_first_decision_needs_lookback_plus_one_closes() -> None:
    """Regression: 64 returns need 65 closes (64 warm-up bars + the decision bar)."""
    bars = make_bars(LOOKBACK + 1)
    build_observation(bars, LOOKBACK)
    with pytest.raises(InsufficientHistoryError, match="65 visible closes; got 64"):
        build_observation(bars[:-1], LOOKBACK)

    # The earliest warm-up close (index 0) feeds the first return...
    changed = list(bars)
    changed[0] = bars[0].model_copy(update={"adjusted_close": 50.0})
    assert build_observation(changed, LOOKBACK).log_returns[0] != (
        build_observation(bars, LOOKBACK).log_returns[0]
    )
    # ...and one bar later it has left the window entirely.
    longer = make_bars(LOOKBACK + 2)
    changed_longer = [longer[0].model_copy(update={"adjusted_close": 50.0}), *longer[1:]]
    assert build_observation(changed_longer, LOOKBACK) == build_observation(longer, LOOKBACK)


@pytest.mark.parametrize("n", [0, 1, LOOKBACK])
def test_missing_history_is_rejected(n: int) -> None:
    with pytest.raises(InsufficientHistoryError):
        build_observation(make_bars(n) if n else [], LOOKBACK)


def test_non_positive_lookback_is_rejected() -> None:
    with pytest.raises(ObservationError, match="lookback"):
        build_observation(make_bars(3), 0)


# --- Field and identity checks -------------------------------------------------


def test_observation_exposes_only_returns_and_offsets() -> None:
    obs = build_observation(make_bars(LOOKBACK + 1), LOOKBACK)
    assert [f.name for f in dataclasses.fields(obs)] == ["log_returns", "bar_offsets"]
    assert all(type(v) is float for v in obs.log_returns)
    assert all(type(v) is int for v in obs.bar_offsets)
    assert not hasattr(obs, "__dict__")  # no ad-hoc attributes can be attached


def test_observation_is_invariant_to_dates_and_price_level() -> None:
    base = build_observation(make_bars(LOOKBACK + 1), LOOKBACK)
    shifted = build_observation(make_bars(LOOKBACK + 1, start=date(2009, 3, 2)), LOOKBACK)
    rescaled = build_observation(make_bars(LOOKBACK + 1, scale=37.5), LOOKBACK)

    assert shifted == base
    assert rescaled.bar_offsets == base.bar_offsets
    assert rescaled.log_returns == pytest.approx(base.log_returns, abs=1e-12)


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, math.nan, math.inf])
def test_invalid_adjusted_close_is_rejected(bad: float | None) -> None:
    bars = make_bars(LOOKBACK + 1)
    bars[10] = bars[10].model_copy(update={"adjusted_close": bad})
    with pytest.raises(ObservationError, match="adjusted_close at bar offset -54"):
        build_observation(bars, LOOKBACK)


def test_invalid_adjusted_close_outside_the_window_is_ignored() -> None:
    bars = make_bars(LOOKBACK + 2)
    bars[0] = bars[0].model_copy(update={"adjusted_close": None})
    assert len(build_observation(bars, LOOKBACK).log_returns) == LOOKBACK


def test_unordered_bars_are_rejected() -> None:
    bars = make_bars(LOOKBACK + 1)
    bars[-1], bars[-2] = bars[-2], bars[-1]
    with pytest.raises(ObservationError, match="increasing date order"):
        build_observation(bars, LOOKBACK)


@pytest.mark.parametrize(
    ("returns", "offsets", "message"),
    [
        ((), (), "at least one"),
        ((0.1,), (-1, 0), "exactly one"),
        ((0.1, 0.2), (0, 1), "consecutively"),
        ((0.1, math.nan), (-1, 0), "finite"),
    ],
)
def test_observation_constructor_validates(
    returns: tuple[float, ...], offsets: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ObservationError, match=message):
        BlindObservation(log_returns=returns, bar_offsets=offsets)


# --- Replay-engine boundary: future perturbation -----------------------------


def _replay_db(bars: list[OHLCVBar]) -> Session:
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = build_session_factory(engine)()
    for bar in bars:
        db.add(OHLCVBarORM(instrument="AAA", **bar.model_dump()))
    db.commit()
    return db


def _observation_at(db: Session, index: int) -> BlindObservation:
    session = create_session(db, "AAA", date(2000, 1, 1), date(2100, 1, 1))
    step_session(db, session.session_id, index)
    # The bounded visible-bars read path is the only source of player inputs.
    return build_observation(get_visible_bars(db, session.session_id), LOOKBACK)


def test_observation_ignores_perturbed_future_bars() -> None:
    bars = make_bars(LOOKBACK + 20)
    future_changed = [
        b if i <= LOOKBACK else b.model_copy(update={"adjusted_close": b.adjusted_close * 3})
        for i, b in enumerate(bars)
    ]
    truncated = bars[: LOOKBACK + 1]
    with _replay_db(bars) as a, _replay_db(future_changed) as b, _replay_db(truncated) as c:
        reference = _observation_at(a, LOOKBACK)
        assert _observation_at(b, LOOKBACK) == reference
        assert _observation_at(c, LOOKBACK) == reference


def test_first_scored_decision_index_through_replay_engine() -> None:
    """Session index 64 (first scoring bar) is the first index with a full observation."""
    with _replay_db(make_bars(LOOKBACK + 5)) as db:
        session = create_session(db, "AAA", date(2000, 1, 1), date(2100, 1, 1))
        step_session(db, session.session_id, LOOKBACK - 1)
        with pytest.raises(InsufficientHistoryError):
            build_observation(get_visible_bars(db, session.session_id), LOOKBACK)
        step_session(db, session.session_id, 1)
        obs = build_observation(get_visible_bars(db, session.session_id), LOOKBACK)
        assert obs.log_returns[0] == pytest.approx(math.log(adjusted(1) / adjusted(0)))
