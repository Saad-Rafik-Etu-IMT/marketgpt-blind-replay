"""F5: trusted execution adapter between blind players and the replay engine.

Players see only a ``BlindObservation``. This adapter, which players never
see, turns a binary target position into at most one market order through the
existing order service, so pre-trade risk checks, N+1 execution and the
broker's fill-time affordability check all apply unchanged.

Target policy (benchmark MVP, long/flat only):

| target | position | pending top-level order | action |
| --- | --- | --- | --- |
| any | any | yes | none: the pending order resolves first |
| long | flat | no | market buy, sized by ``long_quantity`` |
| long | long | no | none: no rebalancing |
| flat | long | no | market sell of the whole position |
| flat | flat | no | none |

Records: a frozen ``DecisionRecord`` is written at decision time and never
updated; the order's later status, rejection reason and fill are read
separately into an ``ExecutionOutcome``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ai.player import Player, PlayerDecision, Target
from src.config import BACKEND_DIR
from src.core.fill_logic import EXECUTION_POLICY_VERSION
from src.data.anonymizer import BlindObservation, build_observation
from src.data.scenarios import Scenario, canonical_sha256
from src.models.fill import FillORM
from src.models.order import OrderORM
from src.schemas.order import CreateOrderRequest, RejectionReason
from src.schemas.replay import ReplaySessionRead
from src.schemas.risk import RiskLimitsRead, RiskLimitsUpdate
from src.services.order_service import OrderRejectedError, submit_order
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import create_session, get_session, get_visible_bars
from src.services.risk_service import get_limits, update_limits

PROFILES_DIR: Path = BACKEND_DIR / "config" / "benchmark_profiles"
DEFAULT_BENCHMARK_PROFILE_PATH: Path = PROFILES_DIR / "benchmark-v2.json"
RECORD_SCHEMA_VERSION = 2


class ExecutionAdapterError(RuntimeError):
    """The adapter refused to act (wrong profile, window or state)."""


class ProfileMismatchError(ExecutionAdapterError):
    """A session's costs or risk limits differ from the benchmark profile."""


class DecisionWindowError(ExecutionAdapterError):
    """A decision was requested outside the scenario's decision bars."""


# --- Benchmark profile --------------------------------------------------------


class BenchmarkProfile(BaseModel):
    """Experimental settings shared by every player; not realistic estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    status: Literal["provisional", "approved"]
    note: str = ""
    initial_cash: float = Field(gt=0, allow_inf_nan=False)
    fee_per_unit: float = Field(ge=0, allow_inf_nan=False)
    # Full synthetic bid-ask spread; the broker charges half on each fill.
    spread_bps: float = Field(ge=0, allow_inf_nan=False)
    # Fraction of available cash kept back before sizing (benchmark-v2 onward).
    # Absent (None) in benchmark-v1, whose sizing used no reserve; leaving it out
    # of the hash keeps v1's recorded identity unchanged.
    cash_reserve_fraction: float | None = Field(default=None, ge=0, lt=1, allow_inf_nan=False)
    risk_limits: RiskLimitsRead

    @property
    def sizing_reserve(self) -> float:
        return self.cash_reserve_fraction or 0.0

    @property
    def sha256(self) -> str:
        exclude = {"cash_reserve_fraction"} if self.cash_reserve_fraction is None else None
        return canonical_sha256(self.model_dump(mode="json", exclude=exclude))


def load_benchmark_profile(path: Path = DEFAULT_BENCHMARK_PROFILE_PATH) -> BenchmarkProfile:
    try:
        return BenchmarkProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExecutionAdapterError(f"invalid benchmark profile {path}: {exc}") from exc


def start_benchmark_session(
    db: Session, instrument: str, start_date: date, end_date: date, profile: BenchmarkProfile
) -> ReplaySessionRead:
    """Create a session with the profile's costs, then apply and verify its limits."""
    session = create_session(
        db,
        instrument,
        start_date,
        end_date,
        initial_cash=profile.initial_cash,
        fee_per_unit=profile.fee_per_unit,
        spread_bps=profile.spread_bps,
    )
    update_limits(
        db, session.session_id, RiskLimitsUpdate(**profile.risk_limits.model_dump())
    )
    verify_profile(db, session.session_id, profile)
    return session


def verify_profile(db: Session, session_id: str, profile: BenchmarkProfile) -> None:
    session = get_session(db, session_id)
    actual = {
        "initial_cash": session.initial_cash,
        "fee_per_unit": session.fee_per_unit,
        "spread_bps": session.spread_bps,
        "risk_limits": get_limits(db, session_id).model_dump(),
    }
    expected = {
        "initial_cash": profile.initial_cash,
        "fee_per_unit": profile.fee_per_unit,
        "spread_bps": profile.spread_bps,
        "risk_limits": profile.risk_limits.model_dump(),
    }
    if actual != expected:
        raise ProfileMismatchError(
            f"session {session_id} does not match profile {profile.profile_id}: "
            f"expected {expected}, got {actual}"
        )


# --- Sizing -------------------------------------------------------------------


def long_quantity(
    cash: float,
    reference_close: float,
    multiplier: float,
    spread_bps: float,
    fee_per_unit: float,
    cash_reserve_fraction: float = 0.0,
) -> int:
    """Whole units a flat account may buy, estimated from bar-N information only.

    First ``cash_reserve_fraction`` of cash is kept back (headroom for a next
    open above ``reference_close``). Per unit the rest reserves the full spread
    and two commissions, which keeps the existing pre-trade checks (risk =
    notional x (1 + full spread) + round-trip fees; cash outlay = notional x
    (1 + half spread) + fees) within cash. It is an estimate, not an execution
    guarantee: the broker's fill-time check rejects an order the account cannot
    pay for at the actual next open.
    """
    if (
        isinstance(cash_reserve_fraction, bool)
        or not math.isfinite(cash_reserve_fraction)
        or not 0.0 <= cash_reserve_fraction < 1.0
    ):
        raise ValueError(f"cash_reserve_fraction must be in [0, 1) (got {cash_reserve_fraction!r})")
    values = (cash, reference_close, multiplier, spread_bps, fee_per_unit)
    if not all(math.isfinite(v) for v in values) or cash <= 0 or reference_close <= 0:
        return 0
    budget = cash * (1.0 - cash_reserve_fraction)
    unit = reference_close * multiplier * (1 + spread_bps / 10000) + 2 * fee_per_unit
    if unit <= 0:
        return 0
    quantity = math.floor(budget / unit)
    while quantity > 0 and quantity * unit > budget:  # guard float rounding
        quantity -= 1
    return quantity


# --- Records ------------------------------------------------------------------


class ObservationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    log_returns: tuple[float, ...]
    bar_offsets: tuple[int, ...]


class ProbabilityRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    long: float
    flat: float


class ExecutionContext(BaseModel):
    """Account state the adapter used; never shown to the player."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cash: float
    position_side: Literal["long", "flat"]
    position_quantity: int
    reference_close: float
    multiplier: float
    pending_order_id: str | None


Action = Literal[
    "submit_buy",
    "submit_sell",
    "none_already_long",
    "none_already_flat",
    "none_pending_order",
    "none_zero_quantity",
    "pre_trade_rejected",
]


class DecisionRecord(BaseModel):
    """Written once at decision time; later fills never modify it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_schema_version: Literal[2] = 2
    session_id: str
    player_name: str
    decision_bar_index: int
    observation: ObservationRecord
    target: Target
    probabilities: ProbabilityRecord | None
    execution_context: ExecutionContext
    action: Action
    order_id: str | None
    order_side: Literal["buy", "sell"] | None
    order_quantity: int | None
    pre_trade_rejection: str | None
    execution_policy_version: str
    risk_profile_id: str
    risk_profile_sha256: str
    # Reserve actually applied by sizing (0.0 for profiles without one).
    cash_reserve_fraction: float


class ExecutionOutcome(BaseModel):
    """What later happened to a decision's order, read after replay steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_schema_version: Literal[1] = 1
    order_id: str
    decision_bar_index: int
    observed_at_bar: int
    status: Literal["pending", "filled", "partially_filled", "cancelled", "rejected"]
    rejection_reason: RejectionReason | None
    fill_bar: int | None
    fill_price: float | None
    fill_quantity: int | None
    fee: float | None


# --- Decisions ----------------------------------------------------------------


@dataclass(frozen=True)
class DecisionWindow:
    first_decision_index: int
    last_decision_index: int

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> DecisionWindow:
        return cls(scenario.first_decision_index, scenario.last_decision_index)


def observe(db: Session, session_id: str, lookback: int) -> BlindObservation:
    """The player's only input: built from the bounded visible-bars read path."""
    return build_observation(get_visible_bars(db, session_id), lookback)


def _pending_top_level_order(db: Session, session_id: str) -> str | None:
    return db.scalar(
        select(OrderORM.order_id).where(
            OrderORM.session_id == session_id,
            OrderORM.status == "pending",
            OrderORM.parent_order_id.is_(None),
        )
    )


def decide_and_submit(
    db: Session,
    session_id: str,
    player: Player,
    window: DecisionWindow,
    lookback: int,
    profile: BenchmarkProfile,
) -> DecisionRecord:
    """Ask ``player`` for a target at the current bar and act on it."""
    session = get_session(db, session_id)
    index = session.current_bar_index
    if session.status == "completed":
        raise DecisionWindowError(f"session {session_id} is completed")
    if not window.first_decision_index <= index <= window.last_decision_index:
        raise DecisionWindowError(
            f"bar {index} is outside decision bars "
            f"{window.first_decision_index}..{window.last_decision_index}"
        )
    verify_profile(db, session_id, profile)

    observation = observe(db, session_id, lookback)
    decision = player.decide(observation)
    if not isinstance(decision, PlayerDecision):
        raise ExecutionAdapterError(f"player {player.name!r} returned {type(decision).__name__}")

    snapshot = current_snapshot(db, session_id)
    held = [p for p in snapshot.positions if p.side != "flat" and p.quantity > 0]
    if any(p.side == "short" for p in held):
        raise ExecutionAdapterError("short positions are outside the long/flat benchmark")
    quantity_held = sum(p.quantity for p in held)
    reference_close = get_visible_bars(db, session_id)[-1].close
    pending = _pending_top_level_order(db, session_id)
    context = ExecutionContext(
        cash=snapshot.cash,
        position_side="long" if quantity_held else "flat",
        position_quantity=quantity_held,
        reference_close=reference_close,
        multiplier=session.multiplier,
        pending_order_id=pending,
    )

    action: Action
    side: Literal["buy", "sell"] | None = None
    quantity: int | None = None
    if pending is not None:
        action = "none_pending_order"
    elif decision.target == "long" and quantity_held:
        action = "none_already_long"
    elif decision.target == "flat" and not quantity_held:
        action = "none_already_flat"
    elif decision.target == "long":
        quantity = long_quantity(
            snapshot.cash, reference_close, session.multiplier,
            session.spread_bps, session.fee_per_unit, profile.sizing_reserve,
        )
        action, side = ("submit_buy", "buy") if quantity else ("none_zero_quantity", None)
    else:
        action, side, quantity = "submit_sell", "sell", quantity_held

    order_id: str | None = None
    rejection: str | None = None
    if side is not None and quantity:
        try:
            order = submit_order(
                db,
                CreateOrderRequest(
                    session_id=session_id,
                    side=side,
                    order_type="market",
                    quantity=quantity,
                    rationale=f"benchmark player {player.name}: target {decision.target}",
                ),
            )
            order_id = order.order_id
        except OrderRejectedError as exc:
            action, rejection = "pre_trade_rejected", str(exc)

    probabilities = decision.probabilities
    return DecisionRecord(
        session_id=session_id,
        player_name=player.name,
        decision_bar_index=index,
        observation=ObservationRecord(
            log_returns=observation.log_returns, bar_offsets=observation.bar_offsets
        ),
        target=decision.target,
        probabilities=(
            None
            if probabilities is None
            else ProbabilityRecord(long=probabilities.long, flat=probabilities.flat)
        ),
        execution_context=context,
        action=action,
        order_id=order_id,
        order_side=side,
        order_quantity=quantity,
        pre_trade_rejection=rejection,
        execution_policy_version=EXECUTION_POLICY_VERSION,
        risk_profile_id=profile.profile_id,
        risk_profile_sha256=profile.sha256,
        cash_reserve_fraction=profile.sizing_reserve,
    )


def resolve_outcome(db: Session, record: DecisionRecord) -> ExecutionOutcome | None:
    """Current state of the record's order; ``None`` when no order was submitted."""
    if record.order_id is None:
        return None
    order = db.get(OrderORM, record.order_id)
    if order is None:
        raise ExecutionAdapterError(f"order {record.order_id} not found")
    fill = db.scalar(
        select(FillORM)
        .where(FillORM.order_id == order.order_id)
        .order_by(FillORM.fill_bar.desc())
        .limit(1)
    )
    return ExecutionOutcome(
        order_id=order.order_id,
        decision_bar_index=record.decision_bar_index,
        observed_at_bar=get_session(db, record.session_id).current_bar_index,
        status=order.status,  # type: ignore[arg-type]
        rejection_reason=order.rejection_reason,  # type: ignore[arg-type]
        fill_bar=None if fill is None else fill.fill_bar,
        fill_price=None if fill is None else fill.fill_price,
        fill_quantity=None if fill is None else fill.fill_quantity,
        fee=None if fill is None else fill.fee,
    )
