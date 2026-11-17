"""F5 player interface and execution adapter: synthetic bars only, offline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.ai.player import BinaryProbabilities, PlayerDecision, PlayerDecisionError
from src.core.fill_logic import EXECUTION_POLICY_VERSION
from src.data.anonymizer import BlindObservation
from src.eval import execution
from src.eval.execution import (
    PROFILES_DIR,
    DecisionWindow,
    DecisionWindowError,
    ExecutionAdapterError,
    ProfileMismatchError,
    decide_and_submit,
    load_benchmark_profile,
    long_quantity,
    resolve_outcome,
    start_benchmark_session,
    verify_profile,
)
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.schemas.risk import RiskLimitsUpdate
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import step_session
from src.services.risk_service import update_limits

LOOKBACK = 4
WINDOW = DecisionWindow(first_decision_index=LOOKBACK, last_decision_index=18)


@dataclass
class Scripted:
    """Test player: replays fixed targets and remembers what it was shown."""

    targets: list[str]
    probabilities: BinaryProbabilities | None = None
    name: str = "scripted"
    seen: list[BlindObservation] = field(default_factory=list)

    def decide(self, observation: BlindObservation) -> PlayerDecision:
        self.seen.append(observation)
        return PlayerDecision(self.targets.pop(0), self.probabilities)  # type: ignore[arg-type]


def seed(db: Session, opens: list[float], closes: list[float] | None = None) -> None:
    closes = closes or opens
    for i, (o, c) in enumerate(zip(opens, closes, strict=True)):
        db.add(OHLCVBarORM(
            instrument="AAA", date=date(2022, 1, 3) + timedelta(days=i), open=o,
            high=max(o, c) + 1, low=min(o, c) - 1, close=c, volume=1_000, adjusted_close=c,
        ))
    db.commit()


@pytest.fixture
def profile():  # type: ignore[no-untyped-def]
    return load_benchmark_profile()


def start(db: Session, profile, opens: list[float] | None = None, closes=None) -> str:  # type: ignore[no-untyped-def]
    if opens is None:
        # Each next open equals the previous close: no overnight gap.
        opens, closes = [100.0 + i for i in range(20)], [101.0 + i for i in range(20)]
    seed(db, opens, closes)
    sid = start_benchmark_session(db, "AAA", date(2022, 1, 1), date(2022, 12, 31), profile).session_id
    step_session(db, sid, LOOKBACK)
    return sid


def order_count(db: Session, sid: str) -> int:
    return db.scalar(
        select(func.count()).where(OrderORM.session_id == sid, OrderORM.parent_order_id.is_(None))
    ) or 0


# --- Player decisions ------------------------------------------------------------


@pytest.mark.parametrize(
    ("long", "flat", "message"),
    [
        (0.7, 0.4, "sum to 1"),
        (1.2, -0.2, r"\[0, 1\]"),
        (math.nan, 0.5, "finite"),
        (True, 0.0, "number"),
    ],
)
def test_invalid_probabilities_are_rejected(long: float, flat: float, message: str) -> None:
    with pytest.raises(PlayerDecisionError, match=message):
        BinaryProbabilities(long=long, flat=flat)


def test_probability_sum_tolerance_and_targets() -> None:
    BinaryProbabilities(long=0.6, flat=0.4 + 5e-7)
    with pytest.raises(PlayerDecisionError, match="target"):
        PlayerDecision("short")  # type: ignore[arg-type]
    with pytest.raises(PlayerDecisionError, match="BinaryProbabilities"):
        PlayerDecision("long", {"long": 1.0, "flat": 0.0})  # type: ignore[arg-type]
    assert PlayerDecision("flat").probabilities is None  # rule-based: none invented


# --- Profile -------------------------------------------------------------------


def test_repository_profiles_v1_unchanged_and_v2_default(profile) -> None:  # type: ignore[no-untyped-def]
    v1 = load_benchmark_profile(PROFILES_DIR / "benchmark-v1.json")
    assert v1.profile_id == "benchmark-v1" and v1.cash_reserve_fraction is None
    # Identity recorded during Stage 2; adding v2 must not reinterpret v1.
    assert v1.sha256 == "a96bad1842b59e6d3183bfbe9edacf0e938d83130f4ea104fabe412e68e8bcf5"
    assert v1.sizing_reserve == 0.0

    assert profile.profile_id == "benchmark-v2" and profile.status == "provisional"
    assert profile.cash_reserve_fraction == 0.005 and profile.sha256 != v1.sha256
    assert (profile.initial_cash, profile.fee_per_unit, profile.spread_bps) == (100_000, 0, 1)
    assert profile.risk_limits == v1.risk_limits
    assert profile.risk_limits.model_dump() == {
        "max_drawdown_pct": 100, "max_risk_per_trade_pct": 100,
        "max_open_positions": 1, "max_exposure_pct": 100,
    }


def test_profile_is_applied_and_mismatch_is_refused(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    sid = start(db_session, profile)
    verify_profile(db_session, sid, profile)
    update_limits(db_session, sid, RiskLimitsUpdate(max_risk_per_trade_pct=2))
    with pytest.raises(ProfileMismatchError, match="benchmark-v2"):
        decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    assert order_count(db_session, sid) == 0


def test_invalid_profile_file(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"profile_id": "x"}), encoding="utf-8")
    with pytest.raises(ExecutionAdapterError, match="invalid benchmark profile"):
        load_benchmark_profile(path)


# --- Sizing ------------------------------------------------------------------------


def test_long_quantity_matches_pre_trade_reservations() -> None:
    assert long_quantity(100_000, 100.0, 1.0, 1.0, 0.0) == 999  # 1000 * 1.0001 > cash
    assert long_quantity(1000, 100.0, 1.0, 0.0, 0.0) == 10
    assert long_quantity(1000, 98.0, 1.0, 0.0, 1.0) == 10  # 10 * (98 + 2 fees) = 1000
    assert long_quantity(1000, 50.0, 2.0, 0.0, 0.0) == 10  # multiplier included
    for bad in ((0, 100, 1, 0, 0), (1000, 0, 1, 0, 0), (1000, math.nan, 1, 0, 0), (50, 100, 1, 0, 0)):
        assert long_quantity(*bad) == 0


def test_zero_quantity_submits_nothing(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    sid = start(db_session, profile, opens=[200_000.0] * 20)
    record = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    assert record.action == "none_zero_quantity" and record.order_id is None
    assert order_count(db_session, sid) == 0 and resolve_outcome(db_session, record) is None


# --- Target policy ------------------------------------------------------------------


def test_repeated_long_never_rebalances_and_flat_exits(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    sid = start(db_session, profile)
    player = Scripted(["long", "long", "long", "long", "flat", "flat"])
    actions = []
    for _ in range(6):
        record = decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
        actions.append(record.action)
        step_session(db_session, sid, 1)
        assert current_snapshot(db_session, sid).cash >= 0

    assert actions == [
        "submit_buy", "none_already_long", "none_already_long", "none_already_long",
        "submit_sell", "none_already_flat",
    ]
    assert order_count(db_session, sid) == 2
    assert all(p.quantity == 0 for p in current_snapshot(db_session, sid).positions)


def test_pending_order_blocks_a_second_order(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    sid = start(db_session, profile)
    player = Scripted(["long", "long"])
    first = decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
    second = decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
    assert second.action == "none_pending_order"
    assert second.execution_context.pending_order_id == first.order_id
    assert order_count(db_session, sid) == 1


def test_pre_trade_rejection_is_recorded(
    db_session: Session, profile, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    sid = start(db_session, profile)
    # Bypass profile verification so the stricter limit reaches the order service.
    monkeypatch.setattr(execution, "verify_profile", lambda *args: None)
    update_limits(db_session, sid, RiskLimitsUpdate(max_risk_per_trade_pct=2))
    record = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    assert record.action == "pre_trade_rejected" and record.order_id is None
    assert record.pre_trade_rejection and "risk limits breached" in record.pre_trade_rejection


# --- Decision window and timing -----------------------------------------------------


def test_decisions_refused_outside_window(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    seed(db_session, [100.0 + i for i in range(20)])
    sid = start_benchmark_session(db_session, "AAA", date(2022, 1, 1), date(2022, 12, 31), profile).session_id
    step_session(db_session, sid, LOOKBACK - 1)  # last warm-up bar
    player = Scripted(["long"])
    with pytest.raises(DecisionWindowError, match="outside decision bars"):
        decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
    assert player.seen == [] and order_count(db_session, sid) == 0

    step_session(db_session, sid, 2)  # bar LOOKBACK + 1: past a one-bar window
    with pytest.raises(DecisionWindowError, match="outside decision bars"):
        decide_and_submit(
            db_session, sid, player, DecisionWindow(LOOKBACK, LOOKBACK), LOOKBACK, profile
        )

    step_session(db_session, sid, 19)  # last bar: fills/mark only, session completed
    with pytest.raises(DecisionWindowError, match="completed"):
        decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
    assert player.seen == [] and order_count(db_session, sid) == 0


# --- Records ------------------------------------------------------------------------


def test_records_separate_decision_from_later_outcome(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    sid = start(db_session, profile)
    probabilities = BinaryProbabilities(long=0.62, flat=0.38)
    player = Scripted(["long"], probabilities, name="model-x")
    record = decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile)
    before = record.model_dump()

    assert record.decision_bar_index == LOOKBACK
    assert record.observation.log_returns == player.seen[0].log_returns
    assert record.probabilities is not None and record.probabilities.long == 0.62
    assert record.execution_context.reference_close == 105.0  # close of bar N
    assert record.order_quantity == long_quantity(100_000, 105.0, 1, 1, 0, 0.005)
    assert record.cash_reserve_fraction == 0.005
    assert record.execution_policy_version == EXECUTION_POLICY_VERSION
    assert (record.risk_profile_id, record.risk_profile_sha256) == ("benchmark-v2", profile.sha256)

    pending = resolve_outcome(db_session, record)
    assert pending is not None and pending.status == "pending" and pending.fill_bar is None
    step_session(db_session, sid, 1)
    outcome = resolve_outcome(db_session, record)
    assert outcome is not None and outcome.status == "filled"
    assert (outcome.fill_bar, outcome.observed_at_bar) == (LOOKBACK + 1, LOOKBACK + 1)
    assert outcome.fill_price is not None and outcome.fill_price > 105.0  # next open + half spread
    assert record.model_dump() == before  # the decision record never changes
    assert json.loads(record.model_dump_json())["target"] == "long"
    assert json.loads(outcome.model_dump_json())["status"] == "filled"


def test_gap_up_rejection_reaches_the_outcome_record(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    opens = [100.0] * 5 + [130.0] * 15
    sid = start(db_session, profile, opens=opens)
    record = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    step_session(db_session, sid, 1)

    outcome = resolve_outcome(db_session, record)
    assert outcome is not None
    assert (outcome.status, outcome.rejection_reason, outcome.fill_bar) == (
        "rejected", "INSUFFICIENT_CASH", None
    )
    assert current_snapshot(db_session, sid).cash == profile.initial_cash
    # After a rejection the player may decide again; nothing is pending.
    retry = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    assert retry.action == "submit_buy"


# --- benchmark-v2 cash reserve ------------------------------------------------------


def gap_session(db: Session, profile, next_open: float) -> str:  # type: ignore[no-untyped-def]
    """Close N (bar 4) = 100; every later bar opens and closes at ``next_open``."""
    prices = [100.0] * 5 + [next_open] * 15
    return start(db, profile, opens=prices, closes=prices)


@pytest.mark.parametrize(
    ("profile_file", "next_open", "status"),
    [
        ("benchmark-v2.json", 100.3, "filled"),  # 0.3% gap fits the 0.5% reserve
        ("benchmark-v1.json", 100.3, "rejected"),  # same gap, no reserve
        ("benchmark-v2.json", 101.0, "rejected"),  # 1% gap exceeds the reserve
    ],
)
def test_reserve_absorbs_a_modest_gap_and_larger_gaps_still_reject(
    db_session: Session, profile_file: str, next_open: float, status: str
) -> None:
    profile = load_benchmark_profile(PROFILES_DIR / profile_file)
    sid = gap_session(db_session, profile, next_open)
    record = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, profile)
    step_session(db_session, sid, 1)

    outcome = resolve_outcome(db_session, record)
    snapshot = current_snapshot(db_session, sid)
    assert outcome is not None and outcome.status == status
    assert snapshot.cash >= 0
    if status == "filled":
        assert outcome.fill_quantity == record.order_quantity  # never resized
    else:
        assert outcome.rejection_reason == "INSUFFICIENT_CASH" and outcome.fill_quantity is None
        assert snapshot.cash == profile.initial_cash
        assert all(p.quantity == 0 for p in snapshot.positions)


def _previous_policy(cash: float, close: float, mult: float, spread: float, fee: float) -> int:
    """Stage 2 sizing before the reserve existed, kept as the reference."""
    unit = close * mult * (1 + spread / 10000) + 2 * fee
    quantity = math.floor(cash / unit)
    while quantity > 0 and quantity * unit > cash:
        quantity -= 1
    return quantity


def test_zero_reserve_reproduces_previous_sizing(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    grid = [
        (100_000, 100.0, 1.0, 1.0, 0.0), (1000, 98.0, 1.0, 0.0, 1.0),
        (1000, 50.0, 2.0, 0.0, 0.0), (12_345.67, 451.23, 1.0, 1.0, 0.01),
    ]
    for args in grid:
        assert long_quantity(*args, 0.0) == long_quantity(*args) == _previous_policy(*args)
    assert long_quantity(100_000, 100.0, 1.0, 1.0, 0.0, 0.005) == 994  # 99,500 / 100.01

    v1 = load_benchmark_profile(PROFILES_DIR / "benchmark-v1.json")
    zero = profile.model_copy(update={"cash_reserve_fraction": 0.0})
    seed(db_session, [100.0 + i for i in range(20)], [101.0 + i for i in range(20)])
    quantities = []
    for p in (v1, zero):
        sid = start_benchmark_session(
            db_session, "AAA", date(2022, 1, 1), date(2022, 12, 31), p
        ).session_id
        step_session(db_session, sid, LOOKBACK)
        record = decide_and_submit(db_session, sid, Scripted(["long"]), WINDOW, LOOKBACK, p)
        quantities.append((record.order_quantity, record.cash_reserve_fraction))
    assert quantities == [(_previous_policy(100_000, 105.0, 1.0, 1.0, 0.0), 0.0)] * 2


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, math.nan, math.inf])
def test_invalid_reserve_values_are_rejected(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="cash_reserve_fraction"):
        long_quantity(1000, 100.0, 1.0, 0.0, 0.0, bad)
    data = json.loads((PROFILES_DIR / "benchmark-v2.json").read_text(encoding="utf-8"))
    data["cash_reserve_fraction"] = bad
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExecutionAdapterError, match="cash_reserve_fraction"):
        load_benchmark_profile(path)


def test_boolean_reserve_is_rejected() -> None:
    with pytest.raises(ValueError, match="cash_reserve_fraction"):
        long_quantity(1000, 100.0, 1.0, 0.0, 0.0, True)


def test_every_player_receives_the_same_sizing_policy(db_session: Session, profile) -> None:  # type: ignore[no-untyped-def]
    seed(db_session, [100.0 + i for i in range(20)], [101.0 + i for i in range(20)])
    players = [
        Scripted(["long"], name="rule-based"),
        Scripted(["long"], BinaryProbabilities(long=0.9, flat=0.1), name="model"),
    ]
    records = []
    for player in players:
        sid = start_benchmark_session(
            db_session, "AAA", date(2022, 1, 1), date(2022, 12, 31), profile
        ).session_id
        step_session(db_session, sid, LOOKBACK)
        records.append(decide_and_submit(db_session, sid, player, WINDOW, LOOKBACK, profile))

    keys = ("order_quantity", "cash_reserve_fraction", "risk_profile_id", "risk_profile_sha256")
    assert [tuple(getattr(r, k) for k in keys) for r in records] == [
        (long_quantity(100_000, 105.0, 1, 1, 0, 0.005), 0.005, "benchmark-v2", profile.sha256)
    ] * 2
