"""MO1 rule-based baselines: synthetic observations and bars only, offline."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from src.ai.baselines import (
    BaselineConfigError,
    BuyAndHoldPlayer,
    InsufficientObservationError,
    MovingAverageCrossoverPlayer,
    relative_price_path,
    simple_moving_average,
)
from src.ai.player import Player
from src.data.anonymizer import BlindObservation
from src.eval.execution import (
    DecisionWindow,
    decide_and_submit,
    load_benchmark_profile,
    resolve_outcome,
    start_benchmark_session,
)
from src.models.market import OHLCVBarORM
from src.services.replay_engine import step_session


def observation_from_closes(closes: list[float]) -> BlindObservation:
    returns = tuple(math.log(b / a) for a, b in zip(closes, closes[1:], strict=False))
    return BlindObservation(log_returns=returns, bar_offsets=tuple(range(1 - len(returns), 1)))


# --- Helpers ---------------------------------------------------------------------


def test_relative_price_path_starts_at_one_and_tracks_ratios() -> None:
    closes = [50.0, 55.0, 44.0, 66.0]
    path = relative_price_path(observation_from_closes(closes).log_returns)
    assert path[0] == 1.0 and len(path) == len(closes)
    assert path == pytest.approx([c / closes[0] for c in closes])


def test_simple_moving_average_uses_the_tail() -> None:
    assert simple_moving_average([1.0, 2.0, 3.0, 4.0], 2) == 3.5
    with pytest.raises(InsufficientObservationError):
        simple_moving_average([1.0, 2.0], 3)
    with pytest.raises(InsufficientObservationError):
        simple_moving_average([1.0, 2.0], 0)


# --- Buy-and-hold ----------------------------------------------------------------


@pytest.mark.parametrize("closes", [[100.0, 101.0, 102.0], [100.0, 90.0, 80.0]])
def test_buy_and_hold_is_always_long_without_probabilities(closes: list[float]) -> None:
    decision = BuyAndHoldPlayer().decide(observation_from_closes(closes))
    assert decision.target == "long" and decision.probabilities is None


# --- MA crossover ----------------------------------------------------------------


def test_default_windows_and_name() -> None:
    player = MovingAverageCrossoverPlayer()
    assert (player.short_window, player.long_window) == (20, 50)
    assert player.name == "ma_crossover_20_50" and player.min_returns == 49


@pytest.mark.parametrize(
    ("closes", "target"),
    [
        ([100.0 + i for i in range(65)], "long"),   # steady rise: short MA above long
        ([200.0 - i for i in range(65)], "flat"),   # steady fall: short MA below long
        ([100.0] * 65, "flat"),                     # equal MAs: tie is flat
    ],
)
def test_crossover_on_default_lookback(closes: list[float], target: str) -> None:
    decision = MovingAverageCrossoverPlayer().decide(observation_from_closes(closes))
    assert decision.target == target and decision.probabilities is None


def test_signal_equals_moving_averages_on_real_closes() -> None:
    """The relative path gives the same decision as MAs on the hidden prices."""
    closes = [100 * math.exp(0.02 * math.sin(i / 3) + 0.001 * i) for i in range(65)]
    for end in range(50, 66):
        window = closes[:end]
        expected = "long" if (
            sum(window[-20:]) / 20 > sum(window[-50:]) / 50
        ) else "flat"
        decision = MovingAverageCrossoverPlayer().decide(observation_from_closes(window))
        assert decision.target == expected, end


def test_signal_ignores_price_level() -> None:
    closes = [100.0 + 3 * math.sin(i / 4) for i in range(65)]
    player = MovingAverageCrossoverPlayer(short_window=5, long_window=12)
    scaled = [c * 37.5 for c in closes]
    assert player.decide(observation_from_closes(closes)) == player.decide(
        observation_from_closes(scaled)
    )


def test_crossover_needs_long_window_closes() -> None:
    player = MovingAverageCrossoverPlayer(short_window=2, long_window=4)
    player.decide(observation_from_closes([1.0, 2.0, 3.0, 4.0]))  # 3 returns = 4 closes
    with pytest.raises(InsufficientObservationError, match="at least 3 returns"):
        player.decide(observation_from_closes([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    ("short", "long", "message"),
    [
        (50, 20, "short_window must be < long_window"),
        (20, 20, "short_window must be < long_window"),
        (0, 50, "positive integer"),
        (20, 50.0, "positive integer"),
        (True, 50, "positive integer"),
    ],
)
def test_invalid_windows_are_rejected(short: int, long: int, message: str) -> None:
    with pytest.raises(BaselineConfigError, match=message):
        MovingAverageCrossoverPlayer(short_window=short, long_window=long)


def test_players_satisfy_the_protocol() -> None:
    players: list[Player] = [BuyAndHoldPlayer(), MovingAverageCrossoverPlayer()]
    assert [p.name for p in players] == ["buy_and_hold", "ma_crossover_20_50"]


# --- Through the execution adapter ------------------------------------------------

LOOKBACK = 4
WINDOW = DecisionWindow(first_decision_index=LOOKBACK, last_decision_index=18)


def run(db: Session, player: Player, closes: list[float]) -> list[str]:
    """Seed bars, decide at every bar of the window, step; return the actions."""
    # Each open equals the previous close: no overnight gap beyond the reserve.
    opens = closes[:1] + closes[:-1]
    for i, (o, c) in enumerate(zip(opens, closes, strict=True)):
        db.add(OHLCVBarORM(
            instrument="AAA", date=date(2022, 1, 3) + timedelta(days=i), open=o,
            high=max(o, c) + 1, low=min(o, c) - 1, close=c, volume=1_000, adjusted_close=c,
        ))
    db.commit()
    profile = load_benchmark_profile()
    sid = start_benchmark_session(
        db, "AAA", date(2022, 1, 1), date(2022, 12, 31), profile
    ).session_id
    step_session(db, sid, LOOKBACK)
    actions = []
    for _ in range(WINDOW.last_decision_index - LOOKBACK + 1):
        record = decide_and_submit(db, sid, player, WINDOW, LOOKBACK, profile)
        assert record.player_name == player.name and record.probabilities is None
        actions.append(record.action)
        step_session(db, sid, 1)
        outcome = resolve_outcome(db, record)
        assert outcome is None or outcome.status == "filled"
    return actions


def test_buy_and_hold_enters_once_and_never_trades_again(db_session: Session) -> None:
    actions = run(db_session, BuyAndHoldPlayer(), [100.0 + i for i in range(20)])
    assert actions[0] == "submit_buy"
    assert set(actions[1:]) == {"none_already_long"}


def test_crossover_enters_on_rise_and_exits_on_fall(db_session: Session) -> None:
    rise = [100.0 + i for i in range(10)]
    closes = rise + [rise[-1] - 3 * (i + 1) for i in range(10)]
    actions = run(db_session, MovingAverageCrossoverPlayer(2, 4), closes)
    assert actions[0] == "submit_buy"
    assert "submit_sell" in actions
    sell = actions.index("submit_sell")
    assert set(actions[1:sell]) == {"none_already_long"}
    assert set(actions[sell + 1 :]) == {"none_already_flat"}
