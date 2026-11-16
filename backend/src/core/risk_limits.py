"""Pure risk-limit evaluation.

Used by the order service before persisting a new order. Every breach
is reported (no short-circuit) so the UI can show the operator the
full picture, not just the first rule that fired.

Inputs are computed by the caller; the limits are a snapshot of the
session's active risk configuration (defaults are §4.5 values).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    """Active risk thresholds for a session (per §4.5)."""

    max_drawdown_pct: float = 10.0
    max_risk_per_trade_pct: float = 2.0
    max_open_positions: int = 3
    max_exposure_pct: float = 100.0


@dataclass(frozen=True)
class RiskCheckInputs:
    """Pre-trade snapshot supplied by the caller."""

    equity: float  # account equity at the current bar
    current_drawdown_pct: float  # 0..100 — current DD from peak
    open_position_count: int  # positions currently open
    current_exposure: float  # Σ |position notional| in dollars
    proposed_notional: float  # dollar size of the new trade (qty * price)
    proposed_risk: float  # dollar risk of the new trade


@dataclass(frozen=True)
class Breach:
    """One rule failure with the offending and limit values."""

    code: str
    detail: str
    actual: float
    limit: float


@dataclass(frozen=True)
class RiskCheckResult:
    """Composite outcome — ``allowed`` is True iff ``breaches`` is empty."""

    allowed: bool
    breaches: list[Breach] = field(default_factory=list)


def check_risk_limits(inputs: RiskCheckInputs, limits: RiskLimits) -> RiskCheckResult:
    """Evaluate ``inputs`` against ``limits`` and return all breaches.

    Convention: a breach fires only when the **actual** value is
    strictly worse than the **limit**. Hitting a limit exactly is
    permitted (e.g. 2.0% risk with a 2.0% cap passes).
    """
    breaches: list[Breach] = []

    # Equity gate runs first — every other check depends on positive
    # equity and we mustn't divide by zero (or compute meaningless
    # percentages from a negative number).
    if inputs.equity <= 0:
        breaches.append(
            Breach(
                code="ZERO_EQUITY",
                detail="equity is zero or negative — no further trades permitted",
                actual=inputs.equity,
                limit=0.0,
            )
        )
        # Stop here — the other thresholds are not meaningful.
        return RiskCheckResult(allowed=False, breaches=breaches)

    _check_drawdown(inputs, limits, breaches)
    _check_per_trade_risk(inputs, limits, breaches)
    _check_open_positions(inputs, limits, breaches)
    _check_exposure(inputs, limits, breaches)

    return RiskCheckResult(allowed=not breaches, breaches=breaches)


# --- Individual gates ------------------------------------------------------


def _check_drawdown(
    inputs: RiskCheckInputs, limits: RiskLimits, breaches: list[Breach]
) -> None:
    if inputs.current_drawdown_pct >= limits.max_drawdown_pct:
        breaches.append(
            Breach(
                code="MAX_DRAWDOWN",
                detail=(
                    f"current drawdown {inputs.current_drawdown_pct:.2f}% "
                    f">= limit {limits.max_drawdown_pct:.2f}%"
                ),
                actual=inputs.current_drawdown_pct,
                limit=limits.max_drawdown_pct,
            )
        )


def _check_per_trade_risk(
    inputs: RiskCheckInputs, limits: RiskLimits, breaches: list[Breach]
) -> None:
    risk_pct = abs(inputs.proposed_risk) / inputs.equity * 100.0
    if risk_pct > limits.max_risk_per_trade_pct:
        breaches.append(
            Breach(
                code="MAX_RISK_PER_TRADE",
                detail=(
                    f"trade risk {risk_pct:.2f}% of equity > "
                    f"limit {limits.max_risk_per_trade_pct:.2f}%"
                ),
                actual=risk_pct,
                limit=limits.max_risk_per_trade_pct,
            )
        )


def _check_open_positions(
    inputs: RiskCheckInputs, limits: RiskLimits, breaches: list[Breach]
) -> None:
    # Even at the cap, opening a new position would breach by one.
    if inputs.open_position_count >= limits.max_open_positions:
        breaches.append(
            Breach(
                code="MAX_OPEN_POSITIONS",
                detail=(
                    f"already at {inputs.open_position_count} open positions "
                    f"(limit {limits.max_open_positions})"
                ),
                actual=float(inputs.open_position_count),
                limit=float(limits.max_open_positions),
            )
        )


def _check_exposure(
    inputs: RiskCheckInputs, limits: RiskLimits, breaches: list[Breach]
) -> None:
    total_exposure = inputs.current_exposure + inputs.proposed_notional
    exposure_pct = total_exposure / inputs.equity * 100.0
    if exposure_pct > limits.max_exposure_pct:
        breaches.append(
            Breach(
                code="MAX_EXPOSURE",
                detail=(
                    f"total exposure {exposure_pct:.2f}% of equity > "
                    f"limit {limits.max_exposure_pct:.2f}%"
                ),
                actual=exposure_pct,
                limit=limits.max_exposure_pct,
            )
        )
