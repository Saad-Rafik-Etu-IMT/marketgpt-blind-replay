"""F1: licence-gated, all-or-nothing local download pipeline with provenance.

For each ticker: fetch through a ``PriceSource``, drop (and record) incomplete
rows, check split/dividend adjustment consistency, validate the CSV with the
existing upload validator, and check F2 gaps on the asset's real bars. Files
are written to a temporary directory and only moved into place when every
ticker passes, so a failure never leaves partial data behind.

Raw OHLCV is preserved exactly as returned by the source; ``adjusted_close``
is stored as a separate column and never replaces raw prices.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from src.config import BACKEND_DIR
from src.data.sources import STANDARD_COLUMNS, PriceSource
from src.data.split import Split, SplitConfig, SplitError, assign_splits, validate_gaps
from src.services.data_validator import validate_csv_content

DEFAULT_DATA_SOURCES_PATH: Path = BACKEND_DIR / "config" / "data_sources.json"
PROVENANCE_FILE = "provenance.json"

# Relative tolerance for float noise in provider-rounded adjusted prices.
ADJUSTMENT_TOLERANCE = 5e-4
# A single-day jump of the adjustment factor above this is not a plausible
# dividend and suggests an inconsistent split adjustment.
MAX_DIVIDEND_STEP = 0.25

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataPipelineError(RuntimeError):
    """Raised when the download is refused or any ticker fails its checks."""


class DataSourcesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["yfinance"]
    terms_status: Literal["pending", "approved", "rejected"]
    terms_note: str = ""
    start: date
    end: date | None = None
    tickers: tuple[str, ...] = Field(min_length=1)

    @field_validator("tickers")
    @classmethod
    def _unique_upper(cls, tickers: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(tickers)) != len(tickers):
            raise ValueError("tickers must be unique")
        bad = [t for t in tickers if not t or t != t.strip().upper() or "/" in t or "\\" in t]
        if bad:
            raise ValueError(f"tickers must be upper-case symbols without path separators: {bad}")
        return tickers

    @model_validator(mode="after")
    def _ordered_period(self) -> Self:
        if self.end is not None and self.end < self.start:
            raise ValueError("end must be on or after start")
        return self


def load_data_sources_config(path: Path = DEFAULT_DATA_SOURCES_PATH) -> DataSourcesConfig:
    try:
        return DataSourcesConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DataPipelineError(f"invalid data sources config {path}: {exc}") from exc


class DroppedRow(BaseModel):
    date: str
    reason: str


class TickerProvenance(BaseModel):
    ticker: str
    file: str
    sha256: str
    rows: int
    first_date: date
    last_date: date
    dropped_rows: list[DroppedRow]
    split_bars: dict[str, int]


class Provenance(BaseModel):
    schema_version: int = 1
    source: str
    source_version: str
    terms_status: str
    terms_note: str
    adjustment_convention: str
    columns: dict[str, str]
    requested_start: date
    requested_end: date | None
    retrieved_at: datetime
    split_config: SplitConfig
    tickers: list[TickerProvenance]


COLUMN_ROLES: dict[str, str] = {
    "open, high, low, close, volume": "raw execution prices as downloaded; used for replay fills and accounting",
    "adjusted_close": "split- and dividend-adjusted close; return series for research/training only, never fills",
}


def drop_incomplete_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[DroppedRow]]:
    missing = [c for c in STANDARD_COLUMNS if c not in frame.columns]
    if missing:
        raise DataPipelineError(f"source frame lacks column(s) {missing}")
    incomplete = frame[list(STANDARD_COLUMNS)].isna().any(axis=1)
    dropped = [
        DroppedRow(
            date=str(row["date"]),
            reason="missing " + ", ".join(c for c in STANDARD_COLUMNS if pd.isna(row[c])),
        )
        for _, row in frame[incomplete].iterrows()
    ]
    return frame[~incomplete].reset_index(drop=True), dropped


def check_adjustment(frame: pd.DataFrame) -> list[str]:
    """Consistency of ``adjusted_close`` against raw ``close``.

    With dividend adjustment the factor ``adjusted_close / close`` lies in
    ``(0, 1]``, never decreases over time, and steps up only by plausible
    dividend amounts on ex-dividend dates.
    """
    errors: list[str] = []
    previous_factor: float | None = None
    for _, row in frame.iterrows():
        close, adjusted = float(row["close"]), float(row["adjusted_close"])
        day = row["date"]
        if not (math.isfinite(close) and math.isfinite(adjusted)) or close <= 0 or adjusted <= 0:
            errors.append(f"{day}: close and adjusted_close must be positive and finite")
            previous_factor = None
            continue
        factor = adjusted / close
        if factor > 1 + ADJUSTMENT_TOLERANCE:
            errors.append(f"{day}: adjusted_close {adjusted} exceeds close {close}")
        if previous_factor is not None:
            step = factor / previous_factor
            if step < 1 - ADJUSTMENT_TOLERANCE:
                errors.append(f"{day}: adjustment factor decreased ({previous_factor:.6f} -> {factor:.6f})")
            elif step > 1 + MAX_DIVIDEND_STEP:
                errors.append(
                    f"{day}: adjustment factor jumped by {step - 1:.1%}; "
                    "inconsistent split adjustment suspected"
                )
        previous_factor = factor
    return errors


def to_csv(frame: pd.DataFrame) -> str:
    out = frame[list(STANDARD_COLUMNS)].copy()
    out["date"] = [pd.Timestamp(d).date().isoformat() for d in out["date"]]
    out["volume"] = out["volume"].astype("int64")
    # Default float formatting round-trips exactly: raw prices are not rounded.
    return str(out.to_csv(index=False, lineterminator="\n"))


def _split_counts(dates: list[date], split_config: SplitConfig) -> dict[str, int]:
    counts = {s.value: 0 for s in Split} | {"unassigned": 0}
    for label in assign_splits(dates, split_config):
        counts[label.value if label is not None else "unassigned"] += 1
    return counts


def process_ticker(
    frame: pd.DataFrame, ticker: str, split_config: SplitConfig
) -> tuple[str, TickerProvenance]:
    """Return the CSV text and provenance for one ticker, or raise with every error."""
    if frame.empty:
        raise DataPipelineError("source returned no rows")
    complete, dropped = drop_incomplete_rows(frame)
    if complete.empty:
        raise DataPipelineError("no complete rows")

    errors = check_adjustment(complete)
    csv_text = to_csv(complete)
    report = validate_csv_content(csv_text)
    errors += [f"line {e.row_number} {e.field or ''}: {e.message}" for e in report.errors]
    if errors:
        raise DataPipelineError("; ".join(errors))

    dates = [bar.date for bar in report.bars]
    try:
        validate_gaps(dates, split_config)
    except SplitError as exc:
        raise DataPipelineError(str(exc)) from exc

    record = TickerProvenance(
        ticker=ticker,
        file=f"{ticker}.csv",
        sha256=hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        rows=len(dates),
        first_date=dates[0],
        last_date=dates[-1],
        dropped_rows=dropped,
        split_bars=_split_counts(dates, split_config),
    )
    return csv_text, record


def run_download(
    config: DataSourcesConfig,
    source: PriceSource,
    out_dir: Path,
    split_config: SplitConfig,
    *,
    now: datetime | None = None,
) -> Provenance:
    if config.terms_status != "approved":
        raise DataPipelineError(
            f"download refused: terms_status is '{config.terms_status}'. "
            "Record the teacher-confirmed terms as 'approved' in the data sources config first."
        )
    if source.name != config.source:
        raise DataPipelineError(f"source '{source.name}' does not match config '{config.source}'")

    outputs: dict[str, str] = {}
    records: list[TickerProvenance] = []
    errors: list[str] = []
    for ticker in config.tickers:
        try:
            csv_text, record = process_ticker(
                source.fetch(ticker, config.start, config.end), ticker, split_config
            )
        except Exception as exc:  # report every ticker's failure (including network) together
            errors.append(f"{ticker}: {exc}")
            continue
        outputs[record.file] = csv_text
        records.append(record)
    if errors:
        raise DataPipelineError("download failed; nothing was written:\n" + "\n".join(errors))

    provenance = Provenance(
        source=source.name,
        source_version=source.version(),
        terms_status=config.terms_status,
        terms_note=config.terms_note,
        adjustment_convention=source.adjustment_convention,
        columns=COLUMN_ROLES,
        requested_start=config.start,
        requested_end=config.end,
        retrieved_at=now or datetime.now(timezone.utc),
        split_config=split_config,
        tickers=records,
    )
    _write_atomically(out_dir, outputs, provenance)
    return provenance


def _write_atomically(out_dir: Path, outputs: dict[str, str], provenance: Provenance) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for name, text in outputs.items():
            (staging / name).write_text(text, encoding="utf-8", newline="")
        (staging / PROVENANCE_FILE).write_text(
            provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        backup = out_dir.with_name(f".{out_dir.name}-previous")
        if backup.exists():
            shutil.rmtree(backup)
        if out_dir.exists():
            out_dir.rename(backup)
        try:
            staging.rename(out_dir)
        except OSError:
            if backup.exists():
                backup.rename(out_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
