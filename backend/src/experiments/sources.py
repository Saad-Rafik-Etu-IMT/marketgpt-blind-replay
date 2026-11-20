"""Local data sources for experiments: health report, loading, synthetic demo.

Only files under ``<data_dir>/raw/yfinance/`` (written by the F1 pipeline) are
sources; names are validated against that folder, never used as paths. The
``synthetic`` source is a deterministic two-asset random walk covering the
default split, so experiments run without any downloaded data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTHETIC = "synthetic"
SOURCE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class SourceError(ValueError):
    """Unknown or unreadable data source."""


def raw_dir(data_dir: Path) -> Path:
    return data_dir / "raw" / "yfinance"


def available(data_dir: Path) -> dict[str, Path]:
    folder = raw_dir(data_dir)
    if not folder.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.glob("*.csv"))
        if p.is_file() and SOURCE_NAME.match(p.stem)
    }


def synthetic_frames() -> dict[str, pd.DataFrame]:
    """Two seeded random-walk assets, 2005-2019 business days (demonstration data)."""
    out = {}
    for name, seed, drift, vol in (
        ("SYN_A", 11, 0.0003, 0.011),
        ("SYN_B", 12, 0.0001, 0.006),
    ):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2005-01-03", "2019-12-31", name="date")
        n = len(dates)
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        opens = np.r_[close[0], close[:-1] * np.exp(rng.normal(0, vol / 3, n - 1))]
        spread = np.abs(rng.normal(0, vol / 2, n))
        out[name] = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, close) * (1 + spread),
                "low": np.minimum(opens, close) * (1 - spread),
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
                "adjusted_close": close,
            },
            index=dates,
        )
    return out


def load_frames(sources: tuple[str, ...], data_dir: Path) -> dict[str, pd.DataFrame]:
    if sources == (SYNTHETIC,):
        return synthetic_frames()
    if SYNTHETIC in sources:
        raise SourceError("the synthetic source cannot be mixed with real sources")
    files = available(data_dir)
    unknown = [s for s in sources if s not in files]
    if unknown:
        raise SourceError(f"unknown local source(s): {unknown}")
    frames = {}
    for name in sources:
        try:
            frames[name] = pd.read_csv(
                files[name], parse_dates=["date"], index_col="date"
            )
        except (OSError, ValueError) as exc:
            raise SourceError(f"{name}: unreadable CSV ({exc})") from exc
    return frames


def health(data_dir: Path) -> list[dict[str, Any]]:
    """Per-source rows, date range, file hash and recorded provenance hash."""
    provenance: dict[str, Any] = {}
    prov_path = raw_dir(data_dir) / "provenance.json"
    if prov_path.is_file():
        try:
            provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
    tickers = provenance.get("tickers", []) if isinstance(provenance, dict) else []
    recorded = (
        {t.get("ticker"): t for t in tickers if isinstance(t, dict)}
        if isinstance(tickers, list)
        else {}
    )
    out = []
    for name, path in available(data_dir).items():
        entry: dict[str, Any] = {"source": name, "kind": "local csv"}
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
            entry.update(
                rows=len(frame),
                first=frame["date"].min().date().isoformat(),
                last=frame["date"].max().date().isoformat(),
                sha256=digest,
                status="ok",
            )
            expected = recorded.get(name, {}).get("sha256")
            entry["provenance_match"] = None if expected is None else expected == digest
        except (OSError, ValueError, KeyError) as exc:
            entry.update(status="unreadable", error=str(exc))
        out.append(entry)
    out.append(
        {
            "source": SYNTHETIC,
            "kind": "generated demonstration data (2 assets, 2005-2019)",
            "status": "ok",
        }
    )
    return out
