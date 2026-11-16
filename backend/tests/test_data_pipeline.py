"""F1 pipeline tests: fake sources and synthetic bars only, no network."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data.download import main
from src.data.pipeline import (
    PROVENANCE_FILE,
    DataPipelineError,
    DataSourcesConfig,
    Provenance,
    check_adjustment,
    load_data_sources_config,
    run_download,
)
from src.data.sources import STANDARD_COLUMNS, SourceError, YFinanceSource
from src.data.split import SplitConfig, load_split_config
from src.services.data_validator import validate_csv_file

BACKEND_DIR = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def weekdays(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


def synthetic_frame(
    start: date = date(2015, 1, 1),
    end: date = date(2021, 6, 30),
    dividend_every: int = 63,
    dividend: float = 0.5,
) -> pd.DataFrame:
    """Deterministic bars; adjusted_close applies cash dividends backwards in time."""
    dates = weekdays(start, end)
    close = [100 + 10 * math.sin(i / 20) + i * 0.0137 for i in range(len(dates))]
    open_ = [c - 0.3 * math.cos(i) for i, c in enumerate(close)]
    adjusted = [0.0] * len(dates)
    factor = 1.0
    for i in reversed(range(len(dates))):
        adjusted[i] = close[i] * factor
        if i > 0 and i % dividend_every == 0:
            factor *= 1 - dividend / close[i - 1]
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": [max(o, c) + 1.01 for o, c in zip(open_, close, strict=True)],
            "low": [min(o, c) - 0.99 for o, c in zip(open_, close, strict=True)],
            "close": close,
            "volume": [1_000_000 + i for i in range(len(dates))],
            "adjusted_close": adjusted,
        }
    )


class FakeSource:
    name = "yfinance"
    adjustment_convention = "fake: raw OHLCV, separate dividend-adjusted close"

    def __init__(self, frames: dict[str, pd.DataFrame | Exception]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, date, date | None]] = []

    def version(self) -> str:
        return "fake-1.0"

    def fetch(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        result = self.frames[ticker]
        if isinstance(result, Exception):
            raise result
        return result.copy()


def make_config(tickers: tuple[str, ...] = ("SPY", "TLT"), **overrides: Any) -> DataSourcesConfig:
    values: dict[str, Any] = {
        "source": "yfinance",
        "terms_status": "approved",
        "terms_note": "test",
        "start": "2015-01-01",
        "end": None,
        "tickers": list(tickers),
    }
    return DataSourcesConfig.model_validate({**values, **overrides})


@pytest.fixture
def split_config() -> SplitConfig:
    return load_split_config()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


# --- Config and licence gate -----------------------------------------------


def test_repository_data_sources_config_lists_the_agreed_tickers() -> None:
    config = load_data_sources_config()
    assert config.source == "yfinance"
    assert len(config.tickers) == 15
    assert {"SPY", "IEF", "GLD"} <= set(config.tickers)


@pytest.mark.parametrize("status", ["pending", "rejected"])
def test_download_refused_until_terms_approved(
    tmp_path: Path, split_config: SplitConfig, status: str
) -> None:
    source = FakeSource({"SPY": synthetic_frame(), "TLT": synthetic_frame()})
    with pytest.raises(DataPipelineError, match="refused"):
        run_download(make_config(terms_status=status), source, tmp_path / "out", split_config)
    assert source.calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"tickers": ["SPY", "SPY"]},
        {"tickers": ["spy"]},
        {"tickers": ["../SPY"]},
        {"tickers": []},
        {"end": "2014-12-31"},
        {"source": "stooq"},
        {"terms_status": "maybe"},
        {"extra": 1},
    ],
)
def test_data_sources_config_rejects_invalid_values(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_config(**override)


def test_load_data_sources_config_reports_bad_files(tmp_path: Path) -> None:
    with pytest.raises(DataPipelineError):
        load_data_sources_config(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(DataPipelineError):
        load_data_sources_config(broken)


# --- Happy path and provenance ---------------------------------------------


def test_download_writes_raw_and_adjusted_prices_with_provenance(
    tmp_path: Path, split_config: SplitConfig
) -> None:
    frames = {"SPY": synthetic_frame(), "TLT": synthetic_frame(dividend=0.3)}
    out = tmp_path / "raw" / "yfinance"
    provenance = run_download(make_config(), FakeSource(frames), out, split_config, now=NOW)

    assert sorted(p.name for p in out.iterdir()) == ["SPY.csv", "TLT.csv", PROVENANCE_FILE]
    assert [p.name for p in out.parent.iterdir()] == ["yfinance"]  # no staging leftovers
    for ticker, expected in frames.items():
        path = out / f"{ticker}.csv"
        written = read_csv(path)
        assert tuple(written.columns) == STANDARD_COLUMNS
        # Raw OHLCV and adjusted_close are preserved exactly, in separate columns.
        for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
            assert written[column].tolist() == expected[column].tolist()
        assert validate_csv_file(path).is_valid

        record = next(t for t in provenance.tickers if t.ticker == ticker)
        assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record.rows == len(expected)
        assert (record.first_date, record.last_date) == (expected["date"].iloc[0], expected["date"].iloc[-1])
        assert record.dropped_rows == []
        assert sum(record.split_bars.values()) == record.rows
        assert min(record.split_bars[s] for s in ("train", "gap", "validation", "test")) > 0

    stored = Provenance.model_validate_json((out / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert stored == provenance
    assert stored.source == "yfinance" and stored.source_version == "fake-1.0"
    assert stored.terms_status == "approved"
    assert stored.retrieved_at == NOW
    assert (stored.requested_start, stored.requested_end) == (date(2015, 1, 1), None)
    assert stored.adjustment_convention == FakeSource.adjustment_convention
    assert "fills" in stored.columns["adjusted_close"]
    assert stored.split_config == split_config


def test_incomplete_rows_are_dropped_and_recorded(tmp_path: Path, split_config: SplitConfig) -> None:
    frame = synthetic_frame()
    frame.loc[5, "adjusted_close"] = float("nan")
    frame.loc[9, ["open", "volume"]] = float("nan")
    provenance = run_download(
        make_config(("SPY",)), FakeSource({"SPY": frame}), tmp_path / "out", split_config
    )
    record = provenance.tickers[0]
    assert record.rows == len(frame) - 2
    assert [(d.date, d.reason) for d in record.dropped_rows] == [
        (str(frame.loc[5, "date"]), "missing adjusted_close"),
        (str(frame.loc[9, "date"]), "missing open, volume"),
    ]


def test_rerun_replaces_previous_download(tmp_path: Path, split_config: SplitConfig) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "OLD.csv").write_text("stale", encoding="utf-8")
    run_download(make_config(("SPY",)), FakeSource({"SPY": synthetic_frame()}), out, split_config)
    assert sorted(p.name for p in out.iterdir()) == ["SPY.csv", PROVENANCE_FILE]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


# --- All-or-nothing failures -----------------------------------------------


def _broken_ohlc() -> pd.DataFrame:
    frame = synthetic_frame()
    frame.loc[10, "high"] = frame.loc[10, "close"] - 5
    return frame


def _gap_too_short() -> pd.DataFrame:
    frame = synthetic_frame()
    january_2020 = frame["date"].map(lambda d: date(2020, 1, 1) <= d <= date(2020, 1, 31))
    keep = ~january_2020 | (frame.index.isin(frame[january_2020].index[:3]))
    return frame[keep].reset_index(drop=True)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (_broken_ohlc(), "high"),
        (_gap_too_short(), "validation/test gap"),
        (synthetic_frame().iloc[0:0], "no rows"),
        (synthetic_frame().drop(columns=["adjusted_close"]), "lacks column"),
        (pd.concat([synthetic_frame().iloc[:3], synthetic_frame().iloc[2:]]), "date"),
        (ConnectionError("network down"), "network down"),
    ],
)
def test_any_failing_ticker_aborts_the_whole_download(
    tmp_path: Path, split_config: SplitConfig, bad: pd.DataFrame | Exception, message: str
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "SPY.csv").write_text("previous good data", encoding="utf-8")
    source = FakeSource({"SPY": synthetic_frame(), "TLT": bad})

    with pytest.raises(DataPipelineError, match=message) as excinfo:
        run_download(make_config(), source, out, split_config)

    assert "TLT" in str(excinfo.value)
    assert [c[0] for c in source.calls] == ["SPY", "TLT"]
    # Previous data untouched, nothing new written, no staging leftovers.
    assert sorted(p.name for p in out.iterdir()) == ["SPY.csv"]
    assert (out / "SPY.csv").read_text(encoding="utf-8") == "previous good data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_source_must_match_config(tmp_path: Path, split_config: SplitConfig) -> None:
    source = FakeSource({"SPY": synthetic_frame()})
    source.name = "other"
    with pytest.raises(DataPipelineError, match="does not match"):
        run_download(make_config(("SPY",)), source, tmp_path / "out", split_config)


# --- Split/dividend adjustment consistency ---------------------------------


def test_consistent_dividend_adjustment_passes() -> None:
    assert check_adjustment(synthetic_frame()) == []
    assert check_adjustment(synthetic_frame(dividend=0.0)) == []


def _with(frame: pd.DataFrame, index: int, **values: float) -> pd.DataFrame:
    for column, value in values.items():
        frame.loc[index, column] = value
    return frame


def _unadjusted_split() -> pd.DataFrame:
    # Raw prices halve after a 2:1 split that the adjusted series accounts for
    # but whose earlier raw history was not split-adjusted.
    frame = synthetic_frame(dividend=0.0)
    later = frame.index >= 200
    for column in ("open", "high", "low", "close"):
        frame.loc[later, column] = frame.loc[later, column] / 2
    frame.loc[~later, "adjusted_close"] = frame.loc[~later, "close"] / 2
    frame.loc[later, "adjusted_close"] = frame.loc[later, "close"]
    return frame


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (_with(synthetic_frame(), 20, adjusted_close=200.0), "exceeds close"),
        (_with(synthetic_frame(), 20, adjusted_close=-1.0), "positive"),
        (_with(synthetic_frame(), 20, close=0.0, low=0.0, open=0.0), "positive"),
        (_with(synthetic_frame(dividend=0.0), 20, adjusted_close=50.0), "decreased"),
        (_unadjusted_split(), "split"),
    ],
)
def test_inconsistent_adjustment_is_detected(frame: pd.DataFrame, message: str) -> None:
    errors = check_adjustment(frame)
    assert errors
    assert any(message in e for e in errors)


def test_adjustment_errors_abort_download(tmp_path: Path, split_config: SplitConfig) -> None:
    source = FakeSource({"SPY": _unadjusted_split()})
    with pytest.raises(DataPipelineError, match="split"):
        run_download(make_config(("SPY",)), source, tmp_path / "out", split_config)
    assert not (tmp_path / "out").exists()


# --- yfinance adapter (fake module, no network) ----------------------------


def _yahoo_frame(multi_index: bool) -> pd.DataFrame:
    frame = synthetic_frame(date(2020, 1, 1), date(2020, 1, 10))
    yahoo = pd.DataFrame(
        {
            "Adj Close": frame["adjusted_close"].to_numpy(),
            "Close": frame["close"].to_numpy(),
            "High": frame["high"].to_numpy(),
            "Low": frame["low"].to_numpy(),
            "Open": frame["open"].to_numpy(),
            "Volume": frame["volume"].to_numpy(),
        },
        index=pd.DatetimeIndex(frame["date"], name="Date"),
    )
    if multi_index:
        yahoo.columns = pd.MultiIndex.from_product([yahoo.columns, ["SPY"]])
    return yahoo


@pytest.mark.parametrize("multi_index", [False, True])
def test_yfinance_source_requests_unadjusted_prices_and_maps_columns(
    monkeypatch: pytest.MonkeyPatch, multi_index: bool
) -> None:
    calls: list[dict[str, Any]] = []

    def download(ticker: str, **kwargs: Any) -> pd.DataFrame:
        calls.append({"ticker": ticker, **kwargs})
        return _yahoo_frame(multi_index)

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download, __version__="9.9"))
    source = YFinanceSource()
    frame = source.fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert source.version() == "9.9"
    assert calls[0]["ticker"] == "SPY"
    assert calls[0]["auto_adjust"] is False
    assert calls[0]["actions"] is False
    assert (calls[0]["start"], calls[0]["end"]) == ("2020-01-01", "2020-01-11")
    assert tuple(frame.columns) == STANDARD_COLUMNS
    expected = synthetic_frame(date(2020, 1, 1), date(2020, 1, 10))
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)

    source.fetch("SPY", date(2020, 1, 1), None)
    assert calls[1]["end"] is None


def test_yfinance_source_rejects_incomplete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    incomplete = _yahoo_frame(False).drop(columns=["Adj Close"])
    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: incomplete)
    )
    with pytest.raises(SourceError, match="Adj Close"):
        YFinanceSource().fetch("SPY", date(2020, 1, 1), None)


def test_importing_the_pipeline_does_not_import_yfinance() -> None:
    code = "import sys, src.data.download, src.data.pipeline; assert 'yfinance' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR, check=True)


# --- Command line ------------------------------------------------------------


def _write_config(path: Path, **overrides: Any) -> Path:
    config = make_config(("SPY",), **overrides)
    path.write_text(config.model_dump_json(), encoding="utf-8")
    return path


def test_cli_downloads_and_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path / "sources.json")
    source = FakeSource({"SPY": synthetic_frame()})
    code = main(["--config", str(config), "--out", str(tmp_path / "out")], lambda name: source)
    assert code == 0
    output = capsys.readouterr().out
    assert "SPY" in output and "wrote 1 files" in output
    assert json.loads((tmp_path / "out" / PROVENANCE_FILE).read_text(encoding="utf-8"))["source"] == "yfinance"


def test_cli_refuses_pending_terms(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path / "sources.json", terms_status="pending")
    source = FakeSource({"SPY": synthetic_frame()})
    code = main(["--config", str(config), "--out", str(tmp_path / "out")], lambda name: source)
    assert code == 1
    assert "refused" in capsys.readouterr().err
    assert source.calls == []
    assert not (tmp_path / "out").exists()
