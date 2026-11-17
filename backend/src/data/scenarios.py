"""F4: fixed calendar half-year benchmark scenarios and frozen manifests.

Selection rule (declared in ``config/scenarios.json``, never tuned):

- candidate windows are complete calendar half-years (Jan-Jun, Jul-Dec) shared
  by every instrument, inside the source split (validation for the
  development set, test for the sealed test set);
- a window is kept for an instrument only if the frozen dataset covers it
  (a later bar exists, enough scoring bars, no abnormal calendar gap) and
  ``lookback`` preceding bars exist as read-only warm-up context, under the
  set's F2 warm-up policy.

Only the ``date`` column of each CSV is parsed; file bytes are hashed against
``provenance.json``. Prices and returns are never read here, so selection
cannot depend on performance, and generation prints counts, not results.

Indexing, with index 0 at ``warmup_start`` (the replay session's first bar):

- warm-up bars are indices ``0 .. lookback - 1``; no decisions or orders;
- ``first_decision_index == lookback`` is ``scoring_start``. Its observation
  holds ``lookback`` log returns from the closes of indices ``0 .. lookback``
  (``lookback + 1`` closes);
- decisions run through ``last_decision_index == total_bars - 2``; the last
  bar (``scoring_end``) only receives fills and the final mark, so no
  execution or outcome extends beyond the scenario.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from src.config import BACKEND_DIR, get_settings
from src.data.pipeline import PROVENANCE_FILE, Provenance
from src.data.split import (
    DEFAULT_SPLIT_CONFIG_PATH,
    SplitConfig,
    SplitError,
    load_split_config,
)

DEFAULT_SCENARIO_CONFIG_PATH: Path = BACKEND_DIR / "config" / "scenarios.json"
DEFAULT_MANIFEST_DIR: Path = BACKEND_DIR / "config" / "scenarios"

ScenarioSet = Literal["development", "test"]
WarmupPolicy = Literal["inside_split", "pre_split_context_allowed"]
SCENARIO_SETS: tuple[ScenarioSet, ...] = ("development", "test")
# The development set must never draw on test data, and vice versa.
SET_SPLITS: dict[str, str] = {"development": "validation", "test": "test"}
ID_PREFIX: dict[str, str] = {"development": "dev", "test": "test"}


class ScenarioError(ValueError):
    """Invalid scenario config, dataset or manifest."""


class SealedScenarioError(PermissionError):
    """A sealed test manifest was requested without explicit authorisation."""


class SetRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Literal["validation", "test"]
    warmup_policy: WarmupPolicy


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    case_study: str = ""
    instruments: tuple[str, ...] = Field(min_length=1)
    window: Literal["calendar_half_year"]
    min_scoring_bars: int = Field(ge=1)
    max_calendar_gap_days: int = Field(ge=1)
    sets: dict[ScenarioSet, SetRule]

    @field_validator("instruments")
    @classmethod
    def _unique(cls, instruments: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(instruments)) != len(instruments):
            raise ValueError("instruments must be unique")
        return instruments

    @model_validator(mode="after")
    def _sets_use_their_own_split(self) -> Self:
        for name, rule in self.sets.items():
            if rule.split != SET_SPLITS[name]:
                raise ValueError(f"{name} scenarios must use the {SET_SPLITS[name]} split")
        if self.sets.get("development", None) is not None and (
            self.sets["development"].warmup_policy != "inside_split"
        ):
            # F2: a validation example keeps its whole input window inside validation.
            raise ValueError("development warm-up must stay inside the validation split")
        return self


def load_scenario_config(path: Path = DEFAULT_SCENARIO_CONFIG_PATH) -> ScenarioConfig:
    try:
        return ScenarioConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ScenarioError(f"invalid scenario config {path}: {exc}") from exc


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    scenario_set: ScenarioSet
    calendar_group: str
    instrument: str
    window_start: date
    window_end: date
    warmup_start: date
    warmup_end: date
    scoring_start: date
    scoring_end: date
    warmup_bars: int = Field(ge=1)
    scoring_bars: int = Field(ge=2)
    total_bars: int
    first_decision_index: int
    last_decision_index: int
    last_bar_index: int
    warmup_crosses_split_boundary: bool
    instrument_file_sha256: str
    dataset_fingerprint: str
    split_config_sha256: str
    scenario_config_sha256: str

    @model_validator(mode="after")
    def _consistent_indices(self) -> Self:
        if self.total_bars != self.warmup_bars + self.scoring_bars:
            raise ValueError("total_bars must equal warmup_bars + scoring_bars")
        if self.first_decision_index != self.warmup_bars:
            raise ValueError("the first decision is the first scoring bar")
        if self.last_bar_index != self.total_bars - 1:
            raise ValueError("last_bar_index must be total_bars - 1")
        if self.last_decision_index != self.last_bar_index - 1:
            raise ValueError("the last decision must leave one bar for execution")
        if not (
            self.warmup_start <= self.warmup_end < self.scoring_start <= self.scoring_end
        ):
            raise ValueError("warm-up must precede the scoring period")
        if not (self.window_start <= self.scoring_start and self.scoring_end <= self.window_end):
            raise ValueError("scoring dates must lie inside the calendar window")
        return self


class Exclusion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    calendar_group: str
    instrument: str
    window_start: date
    window_end: date
    reason: str


class ScenarioManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    scenario_set: ScenarioSet
    split: str
    sealed: bool
    warmup_policy: WarmupPolicy
    scenario_config_version: int
    case_study: str
    lookback: int
    instruments: tuple[str, ...]
    dataset_fingerprint: str
    split_config_sha256: str
    scenario_config_sha256: str
    scenarios: tuple[Scenario, ...]
    exclusions: tuple[Exclusion, ...]


# --- Fingerprints -----------------------------------------------------------


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_fingerprint(provenance: Provenance) -> str:
    """Identity of the whole frozen download (all tickers, not only the case study)."""
    return canonical_sha256(
        {
            "source": provenance.source,
            "source_version": provenance.source_version,
            "retrieved_at": provenance.retrieved_at.isoformat(),
            "tickers": sorted(
                (t.ticker, t.sha256, t.rows, t.first_date.isoformat(), t.last_date.isoformat())
                for t in provenance.tickers
            ),
        }
    )


# --- Pure selection ---------------------------------------------------------


def half_year_windows(first: date, last: date) -> list[tuple[str, date, date]]:
    """Calendar half-years intersecting ``first .. last``, in order."""
    windows: list[tuple[str, date, date]] = []
    year, half = first.year, 1 if first.month <= 6 else 2
    while True:
        start = date(year, 1, 1) if half == 1 else date(year, 7, 1)
        if start > last:
            return windows
        end = date(year, 6, 30) if half == 1 else date(year, 12, 31)
        windows.append((f"{year}H{half}", start, end))
        year, half = (year, 2) if half == 1 else (year + 1, 1)


def _split_range(split_config: SplitConfig, split: str) -> tuple[date, date | None]:
    if split == "validation":
        return split_config.validation_start, split_config.validation_end
    return split_config.test_start, split_config.test_end


def _check_increasing(instrument: str, dates: Sequence[date]) -> None:
    for previous, current in zip(dates, dates[1:], strict=False):
        if current <= previous:
            raise ScenarioError(f"{instrument}: dates must be strictly increasing ({previous}, {current})")


def build_manifest(
    scenario_set: ScenarioSet,
    config: ScenarioConfig,
    split_config: SplitConfig,
    dates_by_instrument: Mapping[str, Sequence[date]],
    file_hashes: Mapping[str, str],
    fingerprint: str,
) -> ScenarioManifest:
    """Select scenarios from trading dates only; every rejection is recorded."""
    if scenario_set not in config.sets:
        raise ScenarioError(f"scenario set {scenario_set!r} is not configured")
    rule = config.sets[scenario_set]
    lookback = split_config.lookback
    set_start, set_end = _split_range(split_config, rule.split)
    for instrument in config.instruments:
        if instrument not in dates_by_instrument or not dates_by_instrument[instrument]:
            raise ScenarioError(f"{instrument}: no dates in the frozen dataset")
        _check_increasing(instrument, dates_by_instrument[instrument])

    split_hash = canonical_sha256(split_config.model_dump(mode="json"))
    config_hash = canonical_sha256(config.model_dump(mode="json"))
    upper = set_end or max(d[-1] for d in dates_by_instrument.values())
    scenarios: list[Scenario] = []
    exclusions: list[Exclusion] = []

    for group, window_start, window_end in half_year_windows(set_start, upper):
        for instrument in config.instruments:
            dates = list(dates_by_instrument[instrument])

            def exclude(reason: str, group: str = group, instrument: str = instrument,
                        ws: date = window_start, we: date = window_end) -> None:
                exclusions.append(Exclusion(calendar_group=group, instrument=instrument,
                                            window_start=ws, window_end=we, reason=reason))

            if window_start < set_start or (set_end is not None and window_end > set_end):
                exclude(
                    f"calendar window is not fully inside the {rule.split} split "
                    f"({set_start} .. {set_end or 'open'})"
                )
                continue
            if dates[-1] <= window_end:
                exclude(f"dataset ends {dates[-1]}: no bar after the window, coverage incomplete")
                continue
            first = bisect.bisect_left(dates, window_start)
            last = bisect.bisect_right(dates, window_end) - 1
            scoring_bars = last - first + 1
            if scoring_bars < config.min_scoring_bars:
                exclude(f"{scoring_bars} scoring bars; at least {config.min_scoring_bars} required")
                continue
            if first < lookback:
                exclude(f"only {first} bars before the window; {lookback} warm-up bars required")
                continue
            warm = first - lookback
            crosses = dates[warm] < set_start
            if crosses and rule.warmup_policy == "inside_split":
                available = first - bisect.bisect_left(dates, set_start)
                exclude(
                    f"only {available} warm-up bars inside the {rule.split} split; "
                    f"{lookback} required"
                )
                continue
            span = dates[warm : last + 1]
            gap_days, gap_after = max(
                ((b - a).days, a) for a, b in zip(span, span[1:], strict=False)
            )
            if gap_days > config.max_calendar_gap_days:
                exclude(
                    f"calendar gap of {gap_days} days after {gap_after} exceeds "
                    f"{config.max_calendar_gap_days}"
                )
                continue
            total = lookback + scoring_bars
            scenarios.append(
                Scenario(
                    scenario_id=f"{ID_PREFIX[scenario_set]}-{group}-{instrument}",
                    scenario_set=scenario_set,
                    calendar_group=group,
                    instrument=instrument,
                    window_start=window_start,
                    window_end=window_end,
                    warmup_start=dates[warm],
                    warmup_end=dates[first - 1],
                    scoring_start=dates[first],
                    scoring_end=dates[last],
                    warmup_bars=lookback,
                    scoring_bars=scoring_bars,
                    total_bars=total,
                    first_decision_index=lookback,
                    last_decision_index=total - 2,
                    last_bar_index=total - 1,
                    warmup_crosses_split_boundary=crosses,
                    instrument_file_sha256=file_hashes[instrument],
                    dataset_fingerprint=fingerprint,
                    split_config_sha256=split_hash,
                    scenario_config_sha256=config_hash,
                )
            )

    return ScenarioManifest(
        scenario_set=scenario_set,
        split=rule.split,
        sealed=scenario_set == "test",
        warmup_policy=rule.warmup_policy,
        scenario_config_version=config.version,
        case_study=config.case_study,
        lookback=lookback,
        instruments=config.instruments,
        dataset_fingerprint=fingerprint,
        split_config_sha256=split_hash,
        scenario_config_sha256=config_hash,
        scenarios=tuple(scenarios),
        exclusions=tuple(exclusions),
    )


# --- Frozen dataset I/O -----------------------------------------------------


def read_trading_dates(csv_bytes: bytes) -> list[date]:
    """Parse only the ``date`` column; price columns are never interpreted."""
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    if reader.fieldnames is None or "date" not in reader.fieldnames:
        raise ScenarioError("CSV has no 'date' column")
    try:
        return [date.fromisoformat(row["date"]) for row in reader]
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"invalid date in CSV: {exc}") from exc


def generate_manifest(
    scenario_set: ScenarioSet,
    data_dir: Path,
    config: ScenarioConfig,
    split_config: SplitConfig,
) -> ScenarioManifest:
    """Build a manifest from a frozen F1 download, verifying it is unmodified."""
    try:
        provenance = Provenance.model_validate_json(
            (data_dir / PROVENANCE_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ScenarioError(f"cannot read {data_dir / PROVENANCE_FILE}: {exc}") from exc
    if provenance.split_config != split_config:
        raise ScenarioError(
            "the dataset was gap-checked with a different split config; re-run the F1 download"
        )
    records = {t.ticker: t for t in provenance.tickers}
    dates: dict[str, list[date]] = {}
    hashes: dict[str, str] = {}
    for instrument in config.instruments:
        record = records.get(instrument)
        if record is None:
            raise ScenarioError(f"{instrument} is not in the frozen dataset provenance")
        try:
            content = (data_dir / record.file).read_bytes()
        except OSError as exc:
            raise ScenarioError(f"{instrument}: cannot read {record.file}: {exc}") from exc
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ScenarioError(f"{instrument}: {record.file} does not match its provenance hash")
        dates[instrument] = read_trading_dates(content)
        hashes[instrument] = record.sha256
    return build_manifest(
        scenario_set, config, split_config, dates, hashes, dataset_fingerprint(provenance)
    )


def manifest_path(scenario_set: ScenarioSet, manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> Path:
    return manifest_dir / f"{scenario_set}.json"


def write_manifest(
    manifest: ScenarioManifest, path: Path, *, replace: bool = False
) -> Literal["created", "unchanged", "replaced"]:
    """Write once; an existing, different manifest is frozen unless ``replace``."""
    text = manifest.model_dump_json(indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return "unchanged"
        if not replace:
            raise ScenarioError(
                f"{path} is frozen and differs from the regenerated manifest; "
                "replace it only before any model evaluation, with --replace"
            )
        path.write_text(text, encoding="utf-8", newline="\n")
        return "replaced"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return "created"


def load_manifest(path: Path, *, allow_sealed_test: bool = False) -> ScenarioManifest:
    """Load a manifest; sealed (test) manifests require explicit authorisation."""
    try:
        manifest = ScenarioManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScenarioError(f"invalid scenario manifest {path}: {exc}") from exc
    if manifest.sealed and not allow_sealed_test:
        raise SealedScenarioError(
            f"{path} holds sealed test scenarios; they are used once, at the end"
        )
    return manifest


# --- Command ----------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.data.scenarios`` (run from ``backend``): write frozen manifests."""
    parser = argparse.ArgumentParser(description="Generate frozen F4 scenario manifests.")
    parser.add_argument("--set", choices=[*SCENARIO_SETS, "all"], default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_SCENARIO_CONFIG_PATH)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: data/raw/yfinance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--replace", action="store_true", help="overwrite a differing manifest")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or get_settings().data_dir / "raw" / "yfinance"
    sets: list[ScenarioSet] = list(SCENARIO_SETS) if args.set == "all" else [args.set]
    try:
        config = load_scenario_config(args.config)
        split_config = load_split_config(args.split_config)
        for scenario_set in sets:
            manifest = generate_manifest(scenario_set, data_dir, config, split_config)
            path = manifest_path(scenario_set, args.out_dir)
            status = write_manifest(manifest, path, replace=args.replace)
            # Counts, dates and reasons only: never prices, returns or results.
            print(
                f"{scenario_set}: {len(manifest.scenarios)} scenarios, "
                f"{len(manifest.exclusions)} exclusions -> {path} ({status})"
            )
            for exclusion in manifest.exclusions:
                print(f"  excluded {exclusion.calendar_group} {exclusion.instrument}: {exclusion.reason}")
    except (ScenarioError, SplitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
