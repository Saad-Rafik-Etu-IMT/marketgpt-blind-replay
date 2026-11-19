"""MO3: quantile-based tokenizer for financial log returns.

The tokenizer is fitted only on the training split. Its learned bin edges are
then reused unchanged for validation, test, and replay observations.

Input convention: one sample contains exactly 64 oldest-first adjusted-close
log returns, matching ``BlindObservation.log_returns`` and IC2.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TOKENIZER_FORMAT_VERSION = 1
DEFAULT_LOOKBACK = 64
DEFAULT_NUM_BINS = 32


class TokenizerError(ValueError):
    """Raised when return data or a serialized tokenizer is invalid."""


class NumericReturnTokenizer:
    """Map continuous log returns to integer quantile-bin token IDs.

    ``fit`` must receive training observations only. Quantile boundaries are
    calculated over all return values in those observations. Duplicate
    boundaries are removed, so constant or heavily repeated training data may
    produce fewer than ``num_bins`` effective bins. IDs always remain in
    ``[0, vocab_size)``.

    Values outside the fitted training range are clipped naturally into the
    first or last bin; no new boundary is learned during ``encode``.
    """

    def __init__(
        self,
        num_bins: int = DEFAULT_NUM_BINS,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> None:
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise TokenizerError("num_bins must be an integer >= 2")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise TokenizerError("lookback must be an integer >= 1")
        self.num_bins = num_bins
        self.lookback = lookback
        self._bin_edges: tuple[float, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether quantile boundaries have been learned."""
        return self._bin_edges is not None

    @property
    def bin_edges(self) -> tuple[float, ...]:
        """Internal quantile boundaries, in strictly increasing order."""
        self._require_fitted()
        assert self._bin_edges is not None
        return self._bin_edges

    @property
    def vocab_size(self) -> int:
        """Number of effective token IDs (may be below the requested bin count)."""
        return len(self.bin_edges) + 1

    def fit(self, train_returns: Any) -> NumericReturnTokenizer:
        """Learn quantile boundaries from a 2D training matrix ``(n, lookback)``.

        This method must never be called with validation or test observations.
        The caller is responsible for passing only examples belonging to the
        chronological training split.
        """
        self._bin_edges = None
        matrix, _ = _as_matrix(train_returns, self.lookback, allow_single=False)
        values = sorted(value for row in matrix for value in row)
        if not values:
            raise TokenizerError("train_returns must contain at least one observation")

        # Match the common linear quantile definition: interpolate at
        # (n - 1) * q. Only internal quantiles become cut points.
        candidates = [
            _linear_quantile(values, index / self.num_bins)
            for index in range(1, self.num_bins)
        ]
        # Ties are expected with discrete/repeated data. Removing duplicate
        # edges prevents empty bins and keeps token IDs contiguous.
        minimum, maximum = values[0], values[-1]
        self._bin_edges = tuple(
            edge
            for i, edge in enumerate(candidates)
            if minimum < edge < maximum
            and (i == 0 or edge > candidates[i - 1])
        )
        return self

    def encode(self, returns: Any) -> list[int] | list[list[int]]:
        """Encode one 64-return sample or a batch into token IDs.

        A one-dimensional input returns ``list[int]``. A two-dimensional input
        returns ``list[list[int]]`` and preserves the batch shape.
        """
        self._require_fitted()
        matrix, was_single = _as_matrix(returns, self.lookback, allow_single=True)
        edges = self.bin_edges
        encoded = [[bisect.bisect_right(edges, value) for value in row] for row in matrix]
        return encoded[0] if was_single else encoded

    def to_dict(self) -> dict[str, Any]:
        """Return the fitted preprocessing state for checkpoint metadata."""
        return {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "num_bins": self.num_bins,
            "lookback": self.lookback,
            "bin_edges": list(self.bin_edges),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NumericReturnTokenizer:
        """Restore a tokenizer from its serialized state."""
        expected = {"format_version", "num_bins", "lookback", "bin_edges"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise TokenizerError("tokenizer state has missing or unexpected fields")
        if (
            isinstance(payload["format_version"], bool)
            or payload["format_version"] != TOKENIZER_FORMAT_VERSION
        ):
            raise TokenizerError("unsupported tokenizer format_version")
        tokenizer = cls(num_bins=payload["num_bins"], lookback=payload["lookback"])
        raw_edges = payload["bin_edges"]
        if not isinstance(raw_edges, list):
            raise TokenizerError("bin_edges must be a list")
        edges = tuple(_finite_number(edge, "bin edge") for edge in raw_edges)
        if len(edges) >= tokenizer.num_bins:
            raise TokenizerError("too many bin edges for num_bins")
        if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
            raise TokenizerError("bin_edges must be strictly increasing")
        tokenizer._bin_edges = edges
        return tokenizer

    def save(self, path: str | Path) -> None:
        """Save fitted bins as JSON; the file should accompany the model checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NumericReturnTokenizer:
        """Load fitted bins from JSON."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot load tokenizer state from {source}: {exc}") from exc
        return cls.from_dict(payload)

    def _require_fitted(self) -> None:
        if self._bin_edges is None:
            raise TokenizerError("tokenizer is not fitted; call fit(train_returns) first")


def _linear_quantile(sorted_values: list[float], quantile: float) -> float:
    """Calculate a linearly interpolated quantile from sorted values."""
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _as_matrix(
    values: Any, lookback: int, *, allow_single: bool
) -> tuple[list[list[float]], bool]:
    """Convert ordinary sequences, NumPy arrays, or CPU tensors to a matrix."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TokenizerError("returns must be a 1D sample or a 2D batch of samples")
    if not values:
        raise TokenizerError("returns must not be empty")

    first = values[0]
    is_single = not isinstance(first, Sequence) or isinstance(first, (str, bytes))
    if is_single:
        if not allow_single:
            raise TokenizerError("fit expects a 2D training matrix with shape (n, lookback)")
        rows = [values]
    else:
        rows = list(values)

    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TokenizerError(f"sample {row_index} must be a sequence of returns")
        if len(row) != lookback:
            raise TokenizerError(
                f"sample {row_index} must contain exactly {lookback} returns; got {len(row)}"
            )
        matrix.append(
            [_finite_number(value, f"return at sample {row_index}, offset {column}")
             for column, value in enumerate(row)]
        )
    return matrix, is_single


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TokenizerError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TokenizerError(f"{name} must be a finite number")
    return result

