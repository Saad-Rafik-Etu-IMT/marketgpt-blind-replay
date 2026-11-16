"""F1 price-source abstraction.

Everything downstream (validation, provenance, F2 split, scenarios, models)
depends only on ``PriceSource`` and the standard frame it returns, never on a
specific provider. Provider libraries are imported lazily inside ``fetch`` so
importing this module, and running the offline tests, never needs them.

Stooq is the documented fallback; it is intentionally not implemented until
the fallback is actually needed.
"""

from __future__ import annotations

import importlib
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

# Standard frame returned by every source: one row per bar, ascending dates.
# open/high/low/close/volume are the execution prices used by replay fills and
# accounting; adjusted_close is a separate return series for research/training.
STANDARD_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
)


class SourceError(RuntimeError):
    """Raised when a provider returns unusable data."""


class PriceSource(Protocol):
    name: str
    adjustment_convention: str

    def version(self) -> str: ...

    def fetch(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        """Return ``STANDARD_COLUMNS`` for ``start..end`` inclusive (``None`` = latest)."""
        ...


class YFinanceSource:
    name = "yfinance"
    adjustment_convention = (
        "open/high/low/close/volume exactly as served by Yahoo via yfinance with "
        "auto_adjust=False: split-adjusted by Yahoo, NOT dividend-adjusted. "
        "adjusted_close is Yahoo 'Adj Close': split- and dividend-adjusted. "
        "Replay fills/accounting use open/high/low/close only; adjusted_close "
        "is for return series (research/training) only."
    )

    _COLUMN_MAP = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adjusted_close",
    }

    def _module(self) -> Any:
        return importlib.import_module("yfinance")

    def version(self) -> str:
        return str(getattr(self._module(), "__version__", "unknown"))

    def fetch(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        yf = self._module()
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            # yfinance treats ``end`` as exclusive.
            end=(end + timedelta(days=1)).isoformat() if end is not None else None,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        frame = pd.DataFrame(raw)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        missing = [c for c in self._COLUMN_MAP if c not in frame.columns]
        if missing:
            raise SourceError(f"{ticker}: yfinance response lacks column(s) {missing}")

        result = frame[list(self._COLUMN_MAP)].rename(columns=self._COLUMN_MAP)
        result.insert(0, "date", pd.DatetimeIndex(frame.index).date)
        return result.reset_index(drop=True)[list(STANDARD_COLUMNS)]


SOURCES: dict[str, type[YFinanceSource]] = {"yfinance": YFinanceSource}
