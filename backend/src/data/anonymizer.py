"""F3: blind player-facing observations built from visible replay bars.

The observation holds only past-derived inputs: the ``lookback`` most recent
daily log returns of ``adjusted_close``, oldest first, with their bar offsets
relative to the decision bar (``-(lookback - 1) .. 0``). ``lookback`` returns
need ``lookback + 1`` closes: the decision bar and the ``lookback`` bars before it.

Ticker, dates, scenario identity, raw price levels and accounting metadata
(multiplier, contract kind, cash, position) never enter the observation; any
code that needs them for execution must read them outside the player.

Limits: this reduces identity cues; it does not guarantee that a historical
period cannot be recognized from its return pattern (e.g. a famous crash).
Adjusted returns also do not reconcile corporate actions between learning
data and raw-price execution: fills and accounting keep using raw OHLC.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.schemas.market import OHLCVBar

FEATURE = "log_return_adjusted_close"
UNIT = "natural log return, close to close"


class ObservationError(ValueError):
    """The visible bars cannot produce a valid blind observation."""


class InsufficientHistoryError(ObservationError):
    """Fewer than ``lookback + 1`` visible bars are available."""


@dataclass(frozen=True, slots=True)
class BlindObservation:
    """Everything a player may see at one decision; nothing else."""

    log_returns: tuple[float, ...]
    bar_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.log_returns:
            raise ObservationError("an observation needs at least one return")
        if len(self.log_returns) != len(self.bar_offsets):
            raise ObservationError("each return needs exactly one bar offset")
        if self.bar_offsets != tuple(range(1 - len(self.log_returns), 1)):
            raise ObservationError("bar offsets must run consecutively up to 0")
        if not all(isinstance(r, float) and math.isfinite(r) for r in self.log_returns):
            raise ObservationError("log returns must be finite floats")


def build_observation(visible_bars: Sequence[OHLCVBar], lookback: int) -> BlindObservation:
    """Observation at the last visible bar (the decision bar).

    ``visible_bars`` must come from the replay engine's bounded visible-bars
    read path, so it already ends at the current bar; only its tail is used.
    """
    if lookback < 1:
        raise ObservationError(f"lookback must be >= 1 (got {lookback})")
    if len(visible_bars) < lookback + 1:
        raise InsufficientHistoryError(
            f"{lookback} returns need {lookback + 1} visible closes; got {len(visible_bars)}"
        )
    window = visible_bars[-(lookback + 1) :]
    closes: list[float] = []
    for offset, bar in zip(range(-lookback, 1), window, strict=True):
        value = bar.adjusted_close
        # bool is an int subclass; reject it explicitly as not a price.
        if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ObservationError(
                f"adjusted_close at bar offset {offset} must be positive and finite (got {value!r})"
            )
        closes.append(float(value))
    for earlier, later in zip(window, window[1:], strict=False):
        if later.date <= earlier.date:
            raise ObservationError("visible bars must be in strictly increasing date order")
    return BlindObservation(
        log_returns=tuple(math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)),
        bar_offsets=tuple(range(1 - lookback, 1)),
    )
