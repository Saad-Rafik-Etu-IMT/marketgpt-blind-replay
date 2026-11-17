"""F4 scenario tests: synthetic weekday calendars and temporary files only."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.data.pipeline import PROVENANCE_FILE, Provenance, TickerProvenance
from src.data.scenarios import (
    DEFAULT_MANIFEST_DIR,
    Scenario,
    ScenarioConfig,
    ScenarioError,
    ScenarioManifest,
    SealedScenarioError,
    build_manifest,
    canonical_sha256,
    generate_manifest,
    half_year_windows,
    load_manifest,
    load_scenario_config,
    main,
    manifest_path,
    read_trading_dates,
    write_manifest,
)
from src.data.split import SplitConfig, load_split_config


def weekdays(start: date, end: date) -> list[date]:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


SPLIT: dict[str, Any] = {
    "train_start": None,
    "train_end": "2019-12-31",
    "validation_start": "2020-03-02",
    "validation_end": "2021-12-31",
    "test_start": "2022-02-01",
    "test_end": None,
    "lookback": 5,
    "label_horizon": 1,
    "embargo": 3,
}
CONFIG: dict[str, Any] = {
    "version": 1,
    "case_study": "synthetic",
    "instruments": ["AAA", "BBB"],
    "window": "calendar_half_year",
    "min_scoring_bars": 100,
    "max_calendar_gap_days": 7,
    "sets": {
        "development": {"split": "validation", "warmup_policy": "inside_split"},
        "test": {"split": "test", "warmup_policy": "pre_split_context_allowed"},
    },
}
DATES = weekdays(date(2019, 6, 3), date(2023, 3, 31))


def split_config(**overrides: Any) -> SplitConfig:
    return SplitConfig.model_validate(SPLIT | overrides)


def scenario_config(**overrides: Any) -> ScenarioConfig:
    return ScenarioConfig.model_validate(CONFIG | overrides)


def manifest(
    scenario_set: str = "development",
    *,
    split: SplitConfig | None = None,
    config: ScenarioConfig | None = None,
    dates: dict[str, list[date]] | None = None,
) -> ScenarioManifest:
    return build_manifest(
        scenario_set,  # type: ignore[arg-type]
        config or scenario_config(),
        split or split_config(),
        dates or {"AAA": DATES, "BBB": DATES},
        {"AAA": "a" * 64, "BBB": "b" * 64},
        "f" * 64,
    )


def reasons(m: ScenarioManifest) -> dict[tuple[str, str], str]:
    return {(e.calendar_group, e.instrument): e.reason for e in m.exclusions}


# --- Windows and selection --------------------------------------------------


def test_half_year_windows_are_fixed_calendar_halves() -> None:
    windows = half_year_windows(date(2017, 4, 1), date(2019, 12, 31))
    assert [w[0] for w in windows] == ["2017H1", "2017H2", "2018H1", "2018H2", "2019H1", "2019H2"]
    assert windows[1][1:] == (date(2017, 7, 1), date(2017, 12, 31))
    assert half_year_windows(date(2020, 8, 1), date(2020, 6, 30)) == []


def test_development_scenarios_use_complete_validation_half_years() -> None:
    m = manifest("development")
    assert [s.scenario_id for s in m.scenarios] == [
        f"dev-{g}-{i}" for g in ("2020H2", "2021H1", "2021H2") for i in ("AAA", "BBB")
    ]
    assert not m.sealed and m.split == "validation"
    assert "not fully inside the validation split" in reasons(m)[("2020H1", "AAA")]
    for s in m.scenarios:
        assert s.warmup_start >= date(2020, 3, 2) and s.scoring_end <= date(2021, 12, 31)
        assert not s.warmup_crosses_split_boundary


def test_test_scenarios_are_sealed_and_need_full_coverage() -> None:
    m = manifest("test")
    assert m.sealed and [s.scenario_id for s in m.scenarios] == ["test-2022H2-AAA", "test-2022H2-BBB"]
    assert "not fully inside the test split" in reasons(m)[("2022H1", "AAA")]
    assert "dataset ends 2023-03-31" in reasons(m)[("2023H1", "BBB")]


def test_off_by_one_warmup_and_decision_indices() -> None:
    """Index 0 = warm-up start; first decision = index lookback = scoring start."""
    s = manifest("development").scenarios[0]
    i = DATES.index(s.scoring_start)

    assert s.scoring_start == date(2020, 7, 1)
    assert DATES[i - 5] == s.warmup_start and DATES[i - 1] == s.warmup_end
    assert s.warmup_bars == s.first_decision_index == 5
    # The first observation's lookback returns need lookback + 1 closes.
    assert len(DATES[DATES.index(s.warmup_start) : i + 1]) == s.warmup_bars + 1
    assert DATES[i + s.scoring_bars - 1] == s.scoring_end
    assert s.total_bars == 5 + s.scoring_bars
    assert (s.last_decision_index, s.last_bar_index) == (s.total_bars - 2, s.total_bars - 1)


@pytest.mark.parametrize(("validation_start", "included"), [("2020-06-24", True), ("2020-06-25", False)])
def test_development_warmup_boundary_inside_validation(validation_start: str, included: bool) -> None:
    split = split_config(
        train_end="2020-06-15", validation_start=validation_start, validation_end="2020-12-31",
        test_start="2021-02-01",
    )
    m = manifest("development", split=split)
    assert ("dev-2020H2-AAA" in {s.scenario_id for s in m.scenarios}) is included
    if not included:
        assert reasons(m)[("2020H2", "AAA")] == (
            "only 4 warm-up bars inside the validation split; 5 required"
        )


def test_test_warmup_may_use_pre_split_context_per_f2() -> None:
    split = split_config(validation_end="2020-06-19", test_start="2020-07-01")
    late = [d for d in DATES if d >= date(2020, 6, 29)]
    m = manifest("test", split=split, dates={"AAA": DATES, "BBB": late})

    aaa = next(s for s in m.scenarios if s.scenario_id == "test-2020H2-AAA")
    assert aaa.warmup_crosses_split_boundary and aaa.warmup_start < date(2020, 7, 1)
    assert reasons(m)[("2020H2", "BBB")] == "only 2 bars before the window; 5 warm-up bars required"


def test_calendar_gap_and_short_windows_are_excluded() -> None:
    gappy = [d for d in DATES if not date(2022, 8, 1) <= d <= date(2022, 8, 19)]
    m = manifest("test", dates={"AAA": gappy, "BBB": DATES})
    assert reasons(m)[("2022H2", "AAA")].startswith("calendar gap of 24 days after 2022-07-29")
    assert [s.instrument for s in m.scenarios] == ["BBB"]

    strict = manifest("development", config=scenario_config(min_scoring_bars=200))
    assert not strict.scenarios
    assert "scoring bars; at least 200 required" in reasons(strict)[("2021H1", "AAA")]


def test_invalid_selection_inputs_are_rejected() -> None:
    with pytest.raises(ScenarioError, match="no dates"):
        manifest(dates={"AAA": DATES, "BBB": []})
    with pytest.raises(ScenarioError, match="strictly increasing"):
        manifest(dates={"AAA": DATES, "BBB": [DATES[1], DATES[0]]})
    only_dev = scenario_config(sets={"development": CONFIG["sets"]["development"]})
    with pytest.raises(ScenarioError, match="not configured"):
        manifest("test", config=only_dev)


def test_scenario_records_reject_inconsistent_indices() -> None:
    record = manifest().scenarios[0].model_dump()
    with pytest.raises(ValidationError, match="total_bars"):
        Scenario.model_validate(record | {"total_bars": record["total_bars"] + 1})
    with pytest.raises(ValidationError, match="leave one bar"):
        Scenario.model_validate(record | {"last_decision_index": record["last_bar_index"]})


# --- Config ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"sets": {"development": {"split": "test", "warmup_policy": "inside_split"}}}, "validation split"),
        (
            {"sets": {"development": {"split": "validation", "warmup_policy": "pre_split_context_allowed"}}},
            "inside the validation split",
        ),
        ({"instruments": ["AAA", "AAA"]}, "unique"),
    ],
)
def test_config_rejects_unsafe_rules(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        scenario_config(**override)


def test_repository_config_declares_two_asset_case_study(tmp_path: Path) -> None:
    config = load_scenario_config()
    assert config.instruments == ("SPY", "IEF")
    assert config.sets["test"].warmup_policy == "pre_split_context_allowed"
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ScenarioError, match="invalid scenario config"):
        load_scenario_config(bad)


# --- Frozen dataset I/O ------------------------------------------------------


def write_dataset(root: Path, *, price_scale: float = 1.0, split: SplitConfig | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for ticker in ("AAA", "BBB"):
        lines = ["date,open,high,low,close,volume,adjusted_close"]
        for i, day in enumerate(DATES):
            price = price_scale * (50 + (i * 7919 % 113))
            lines.append(f"{day},{price},{price + 1},{price - 1},{price},1000,{price * 0.9}")
        content = ("\n".join(lines) + "\n").encode("utf-8")
        (root / f"{ticker}.csv").write_bytes(content)
        records.append(
            TickerProvenance(
                ticker=ticker, file=f"{ticker}.csv", sha256=hashlib.sha256(content).hexdigest(),
                rows=len(DATES), first_date=DATES[0], last_date=DATES[-1], dropped_rows=[],
                split_bars={},
            )
        )
    provenance = Provenance(
        source="synthetic", source_version="0", terms_status="approved", terms_note="",
        adjustment_convention="synthetic", columns={}, requested_start=DATES[0], requested_end=None,
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc), split_config=split or split_config(),
        tickers=records,
    )
    (root / PROVENANCE_FILE).write_text(provenance.model_dump_json(), encoding="utf-8")
    return root


def test_selection_ignores_prices(tmp_path: Path) -> None:
    a = generate_manifest("test", write_dataset(tmp_path / "a"), scenario_config(), split_config())
    b = generate_manifest(
        "test", write_dataset(tmp_path / "b", price_scale=3.7), scenario_config(), split_config()
    )
    hash_fields = {"instrument_file_sha256", "dataset_fingerprint"}
    assert [s.model_dump(exclude=hash_fields) for s in a.scenarios] == [
        s.model_dump(exclude=hash_fields) for s in b.scenarios
    ]
    assert a.exclusions == b.exclusions
    assert a.dataset_fingerprint != b.dataset_fingerprint


def test_generation_is_deterministic_and_records_versions(tmp_path: Path) -> None:
    data = write_dataset(tmp_path)
    first = generate_manifest("development", data, scenario_config(), split_config())
    assert first == generate_manifest("development", data, scenario_config(), split_config())
    s = first.scenarios[0]
    assert s.split_config_sha256 == canonical_sha256(split_config().model_dump(mode="json"))
    assert s.scenario_config_sha256 == canonical_sha256(scenario_config().model_dump(mode="json"))
    assert s.dataset_fingerprint == first.dataset_fingerprint


def test_generation_rejects_modified_or_mismatched_data(tmp_path: Path) -> None:
    data = write_dataset(tmp_path / "ok")
    with (data / "AAA.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ScenarioError, match="provenance hash"):
        generate_manifest("test", data, scenario_config(), split_config())

    other_split = write_dataset(tmp_path / "split", split=split_config(embargo=4))
    with pytest.raises(ScenarioError, match="different split config"):
        generate_manifest("test", other_split, scenario_config(), split_config())

    unknown = scenario_config(instruments=["AAA", "ZZZ"])
    with pytest.raises(ScenarioError, match="ZZZ is not in the frozen dataset"):
        generate_manifest("test", write_dataset(tmp_path / "z"), unknown, split_config())

    with pytest.raises(ScenarioError, match="cannot read"):
        generate_manifest("test", tmp_path / "missing", scenario_config(), split_config())

    data_ok = write_dataset(tmp_path / "gone")
    (data_ok / "BBB.csv").unlink()
    with pytest.raises(ScenarioError, match="BBB: cannot read"):
        generate_manifest("test", data_ok, scenario_config(), split_config())


@pytest.mark.parametrize(("content", "message"), [(b"day,close\n", "no 'date'"), (b"date\n2020-13-01\n", "invalid date")])
def test_read_trading_dates_rejects_bad_csv(content: bytes, message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        read_trading_dates(content)


# --- Freezing and sealing ------------------------------------------------------


def test_manifest_is_written_once_and_frozen(tmp_path: Path) -> None:
    path = tmp_path / "development.json"
    original, other = manifest(), manifest(config=scenario_config(min_scoring_bars=120))
    assert write_manifest(original, path) == "created"
    assert write_manifest(original, path) == "unchanged"
    with pytest.raises(ScenarioError, match="frozen"):
        write_manifest(other, path)
    assert write_manifest(other, path, replace=True) == "replaced"
    assert load_manifest(path) == other


def test_test_manifest_is_sealed_unless_explicitly_allowed(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    write_manifest(manifest("test"), path)
    with pytest.raises(SealedScenarioError, match="used once, at the end"):
        load_manifest(path)
    assert load_manifest(path, allow_sealed_test=True).sealed
    (tmp_path / "broken.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ScenarioError, match="invalid scenario manifest"):
        load_manifest(tmp_path / "broken.json")


def test_command_writes_counts_without_prices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "scenarios.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(SPLIT), encoding="utf-8")
    args = [
        "--config", str(config_path), "--split-config", str(split_path),
        "--data-dir", str(write_dataset(tmp_path / "data")), "--out-dir", str(tmp_path / "out"),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "development: 6 scenarios" in out and "test: 2 scenarios" in out
    assert "excluded 2023H1 AAA: dataset ends" in out
    assert manifest_path("test", tmp_path / "out").exists()
    assert main([*args, "--set", "development"]) == 0
    assert "(unchanged)" in capsys.readouterr().out
    assert main([*args[:-4], "--data-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o2")]) == 1
    assert "error:" in capsys.readouterr().err


# --- Committed manifests ------------------------------------------------------


def test_repository_manifests_match_current_configs() -> None:
    """Frozen manifests must still describe the current split and scenario rules."""
    split = load_split_config()
    config = load_scenario_config()
    dev = load_manifest(manifest_path("development"))
    test = load_manifest(manifest_path("test"), allow_sealed_test=True)
    assert manifest_path("development").parent == DEFAULT_MANIFEST_DIR

    for m, lower, upper in (
        (dev, split.validation_start, split.validation_end),
        (test, split.test_start, split.test_end or date.max),
    ):
        assert m.split_config_sha256 == canonical_sha256(split.model_dump(mode="json"))
        assert m.scenario_config_sha256 == canonical_sha256(config.model_dump(mode="json"))
        assert m.lookback == split.lookback and m.instruments == config.instruments
        ids = [s.scenario_id for s in m.scenarios]
        assert len(ids) == len(set(ids))
        for s in m.scenarios:
            assert lower <= s.window_start and s.window_end <= upper
            assert s.warmup_bars == split.lookback
        for instrument in config.instruments:
            scoring = sorted((s.scoring_start, s.scoring_end) for s in m.scenarios if s.instrument == instrument)
            assert scoring, instrument
            # Non-overlapping windows per instrument.
            assert all(a[1] < b[0] for a, b in zip(scoring, scoring[1:], strict=False))
    assert all(s.warmup_start >= split.validation_start for s in dev.scenarios)
    assert test.sealed and not dev.sealed
