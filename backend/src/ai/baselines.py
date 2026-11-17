"""F6 / MO1: rule-based baseline players (no learning, no probabilities).

Both players see only a ``BlindObservation`` and return ``probabilities=None``:
a rule has no class probabilities, and none is invented for it.

- ``BuyAndHoldPlayer`` always targets ``long``. Through the execution adapter
  this means one entry at the first decision and no later rebalancing.
- ``MovingAverageCrossoverPlayer`` (default 20/50) rebuilds a *relative* price
  path from the observed log returns (first visible close = 1.0) and targets
  ``long`` iff the short simple moving average is strictly above the long one,
  otherwise ``flat`` (ties are flat). Moving averages scale linearly with the
  price level, so the comparison equals the one on real adjusted closes while
  the observation still hides price levels. It is a state rule (sign of
  MA_short - MA_long at each bar), not an event rule on the crossing bar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.ai.player import PlayerDecision
from src.data.anonymizer import BlindObservation


class BaselineConfigError(ValueError):
    """A baseline player was configured with invalid parameters."""


class InsufficientObservationError(ValueError):
    """The observation is too short for the player's rule."""


def relative_price_path(log_returns: Sequence[float]) -> tuple[float, ...]:
    """Closes relative to the first visible close: ``len(log_returns) + 1`` values."""
    path = [1.0]
    cumulative = 0.0
    for r in log_returns:
        cumulative += r
        path.append(math.exp(cumulative))
    return tuple(path)


def simple_moving_average(values: Sequence[float], window: int) -> float:
    """Mean of the last ``window`` values."""
    if window < 1 or window > len(values):
        raise InsufficientObservationError(
            f"window {window} needs between 1 and {len(values)} values"
        )
    return math.fsum(values[-window:]) / window


class BuyAndHoldPlayer:
    """Always long; the adapter never rebalances an existing long position."""

    name = "buy_and_hold"

    def decide(self, observation: BlindObservation) -> PlayerDecision:
        return PlayerDecision(target="long", probabilities=None)


class MovingAverageCrossoverPlayer:
    """Long while SMA(short) > SMA(long) on the relative price path, else flat."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        for label, value in (("short_window", short_window), ("long_window", long_window)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BaselineConfigError(f"{label} must be a positive integer (got {value!r})")
        if short_window >= long_window:
            raise BaselineConfigError(
                f"short_window must be < long_window (got {short_window}/{long_window})"
            )
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"ma_crossover_{short_window}_{long_window}"

    @property
    def min_returns(self) -> int:
        """Smallest observation length (returns) the rule accepts."""
        return self.long_window - 1

    def decide(self, observation: BlindObservation) -> PlayerDecision:
        if len(observation.log_returns) < self.min_returns:
            raise InsufficientObservationError(
                f"{self.name} needs at least {self.min_returns} returns "
                f"({self.long_window} closes); got {len(observation.log_returns)}"
            )
        path = relative_price_path(observation.log_returns)
        short = simple_moving_average(path, self.short_window)
        long = simple_moving_average(path, self.long_window)
        return PlayerDecision(target="long" if short > long else "flat", probabilities=None)
