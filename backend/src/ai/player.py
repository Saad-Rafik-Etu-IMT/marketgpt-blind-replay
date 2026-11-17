"""F5: player interface for blind replay decisions.

A player receives only a ``BlindObservation`` and returns a binary target
position (``long`` or ``flat``), optionally with model probabilities. Rule-based
players return ``probabilities=None``; no probability is invented for them.
The target is a policy output, distinct from any classification label.
Short positions are a later extension and are not representable here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from src.data.anonymizer import BlindObservation

Target = Literal["long", "flat"]
TARGETS: tuple[Target, ...] = ("long", "flat")
# Same tolerance as the research cockpit's probability-sum check.
PROBABILITY_SUM_TOLERANCE = 1e-6


class PlayerDecisionError(ValueError):
    """A player returned an invalid decision."""


@dataclass(frozen=True, slots=True)
class BinaryProbabilities:
    """Model probabilities of the long and flat classes (not of profit)."""

    long: float
    flat: float

    def __post_init__(self) -> None:
        for name in ("long", "flat"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PlayerDecisionError(f"probability {name} must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PlayerDecisionError(f"probability {name} must be finite and in [0, 1]")
        if abs(self.long + self.flat - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise PlayerDecisionError(
                f"probabilities must sum to 1 within {PROBABILITY_SUM_TOLERANCE}"
            )


@dataclass(frozen=True, slots=True)
class PlayerDecision:
    target: Target
    probabilities: BinaryProbabilities | None = None

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise PlayerDecisionError(f"target must be one of {TARGETS} (got {self.target!r})")
        if self.probabilities is not None and not isinstance(
            self.probabilities, BinaryProbabilities
        ):
            raise PlayerDecisionError("probabilities must be BinaryProbabilities or None")


class Player(Protocol):
    """Anything that maps a blind observation to a decision."""

    name: str

    def decide(self, observation: BlindObservation) -> PlayerDecision: ...
