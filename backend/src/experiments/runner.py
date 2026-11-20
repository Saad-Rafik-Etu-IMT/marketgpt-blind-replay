"""Run every (feature profile, preset, seed) of an experiment folder.

Folder layout (``<data_dir>/experiments/<uuid>/``)::

    spec.json          validated ExperimentSpec + fingerprint (written by the API)
    status.json        state, progress, failures (atomic writes)
    cancel             present = cancellation requested
    datasets/<fp>.json dataset manifests (one per feature profile and target)
    runs/<key>.json    run report
    checkpoints/<key>.{npz,json}  model weights and model manifest

Rules:

- datasets are built once per feature profile; every profile and target of an
  experiment shares the same rows (common ``min_lookback`` and ``min_horizon``);
- each run trains on inner_train, selects its epoch on inner_validation and
  reports every head's metrics there;
- the validation part is evaluated only when the spec reads validation, and
  each read is appended to ``<data_dir>/experiments/validation_reads.json``;
- confirmation runs also train fresh models on walk-forward folds inside the
  training rows (scaler refitted per fold, fixed epoch count = the run's best
  epoch, no early stopping) and report each fold;
- a run that fails is recorded and the experiment continues; completed runs
  with the same spec fingerprint are skipped on resume.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from src.ai.mlp import MLP, TrainingData, TrainingDivergedError, save, train
from src.data.split import SplitConfig, load_split_config
from src.datasets import (
    Dataset,
    DatasetSpec,
    build_dataset,
    fit_scaler,
    walk_forward_cutoffs,
    write_manifest,
)
from src.datasets.folds import inner_cutoff
from src.datasets.manifest import frame_fingerprint, read_manifest
from src.experiments.models import register_transformer, train_reference
from src.experiments.sources import load_frames
from src.experiments.spec import TRANSFORMERS, ExperimentSpec, Run
from src.features import DEFAULT_REGISTRY, get_profile
from src.features.canonical import to_canonical
from src.services.study_artifacts import read_json, write_json
from src.targets import get_target

SIGMA_LOOKBACK = 20


class CancelledError(RuntimeError):
    """Cancellation requested through the ``cancel`` file."""


class TimeBudgetError(RuntimeError):
    """``max_minutes`` exceeded."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> Any:
    """JSON-safe copy: NaN and infinities become None, recursively."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def common_lookback(spec: ExperimentSpec) -> int:
    lookback = max(
        DEFAULT_REGISTRY.lookback(get_profile(fp).features)
        for fp in spec.feature_profiles
    )
    if any(get_target(t).needs_sigma for t in spec.targets):
        lookback = max(lookback, SIGMA_LOOKBACK)
    return lookback


def common_horizon(spec: ExperimentSpec) -> int:
    return max(get_target(t).horizon_bars for t in spec.targets)


def chronological_masks(index, start, end, horizon, embargo, lookback, context):
    """Purged date boundaries shared across assets, even if an asset has no evaluation rows."""
    fit, evaluate = np.zeros(len(index), bool), np.zeros(len(index), bool)
    for _, rows in index.groupby("asset"):
        bars = rows["bar"].to_numpy()
        dates = rows["date"].to_numpy()
        in_block = dates >= start
        first = bars[in_block].min() if in_block.any() else np.inf
        after = dates >= end if end is not None else np.zeros(len(rows), bool)
        stop = bars[after].min() if after.any() else np.inf
        positions = rows.index.to_numpy()
        fit[positions] = (dates < start) & (bars + horizon < first - embargo)
        begins = bars - lookback if context.value == "own_partition" else bars
        evaluate[positions] = in_block & (begins >= first) & (bars + horizon < stop)
    return fit, evaluate


class Runner:
    def __init__(
        self, folder: Path, data_dir: Path, split: SplitConfig | None = None
    ) -> None:
        self.folder, self.data_dir = folder, data_dir
        frozen = folder / "data_lock.json"
        self.split = (
            split
            if split is not None
            else SplitConfig.model_validate(read_json(frozen)["split"])
            if frozen.is_file()
            else load_split_config()
        )
        stored = read_json(folder / "spec.json")
        self.spec = ExperimentSpec.model_validate(stored["spec"])
        self.fingerprint = stored["fingerprint"]
        if self.spec.fingerprint() != self.fingerprint:
            raise ValueError(
                "spec.json does not match its fingerprint (locked protocol)"
            )
        self.started = time.monotonic()
        self.status: dict[str, Any] = {}

    # ------------------------------------------------------------- status
    def _write_status(self, **changes: Any) -> None:
        self.status.update(changes, updated_at=_now())
        write_json(self.folder / "status.json", _finite(self.status))

    def _check(self) -> None:
        if (self.folder / "cancel").exists():
            raise CancelledError("cancelled by request")
        limit = self.spec.max_minutes
        if limit is not None and time.monotonic() - self.started > 60 * limit:
            raise TimeBudgetError(f"time budget of {limit} minutes exceeded")

    # ------------------------------------------------------------ datasets
    def verify_data(self, frames) -> None:
        """Lock development data and split across retries; only development rows are hashed."""
        lock_path = self.folder / "data_lock.json"
        data = {}
        for name, frame in sorted(frames.items()):
            dates = pd.to_datetime(
                frame["date"] if "date" in frame.columns else frame.index
            )
            visible = frame.loc[
                np.asarray(dates <= pd.Timestamp(self.split.validation_end))
            ]
            data[name] = frame_fingerprint(to_canonical(visible))
        record = {"split": self.split.model_dump(mode="json"), "data": data}
        if lock_path.is_file():
            if read_json(lock_path) != record:
                raise ValueError(
                    "Development data or split changed; create a new experiment"
                )
        else:
            write_json(lock_path, record)

    def _datasets(self, frames, feature_profile: str) -> list[Dataset]:
        common = dict(
            profile=feature_profile,
            context=self.spec.context,
            scaler=self.spec.scaler,
            min_lookback=common_lookback(self.spec),
            min_horizon=common_horizon(self.spec),
            primary_asset=self.spec.primary_asset,
        )
        datasets = [
            build_dataset(frames, DatasetSpec(target=t, **common), self.split)
            for t in self.spec.targets
        ]
        for ds in datasets:
            path = self.folder / "datasets" / f"{feature_profile}__{ds.target.id}.json"
            if path.is_file():
                if read_manifest(path)["dataset_fingerprint"] != ds.fingerprint:
                    raise ValueError(
                        "Dataset changed since the original run; create a new experiment"
                    )
            else:
                write_manifest(ds.manifest, path)
        return datasets

    # ------------------------------------------------------------- one run
    def _evaluate(
        self, model: MLP, datasets: list[Dataset], part: str
    ) -> dict[str, Any]:
        outputs = model.predict(datasets[0].parts[part].x)
        report = {}
        for ds in datasets:
            p = ds.parts[part]
            report[ds.target.id] = ds.target.evaluate(
                outputs[ds.target.id], p.y, p.targets, p.sigma, ds.target_state or {}
            )
        return report

    def _walk_forward(self, run: Run, datasets: list[Dataset]) -> list[dict[str, Any]]:
        folds = self.spec.experiment_profile.walk_forward_folds
        train_part = datasets[0].parts["train"]
        index = train_part.index
        cutoffs = walk_forward_cutoffs(list(index["date"]), folds, 0.4)
        horizon, embargo = common_horizon(self.spec), self.split.embargo
        # The globally retained epoch depends on future training dates.
        # Every outer fold must select its own epoch on a nested earlier split.
        config = self.spec.train_config(run.preset, run.seed)
        reports = []
        for k in range(folds):
            start, end = cutoffs[k], (None if k == folds - 1 else cutoffs[k + 1])
            lookback = datasets[0].lookback
            outer_train, eval_rows = chronological_masks(
                index, start, end, horizon, embargo, lookback, self.spec.context
            )
            nested_cutoff = inner_cutoff(list(index.loc[outer_train, "date"]))
            nested_fit, nested_select = chronological_masks(
                index,
                nested_cutoff,
                start,
                horizon,
                embargo,
                lookback,
                self.spec.context,
            )
            fit_rows, select_rows = (
                outer_train & nested_fit,
                outer_train & nested_select,
            )
            if not fit_rows.any() or not select_rows.any() or not eval_rows.any():
                raise ValueError(
                    "Insufficient rows for a purged nested walk-forward fold"
                )
            scaler = fit_scaler(
                self.spec.scaler, train_part.x_raw[fit_rows], datasets[0].columns
            )
            states, encoded = {}, {}
            for ds in datasets:
                p = ds.parts["train"]
                t = ds.target
                states[t.id] = t.fit(p.targets[fit_rows], p.sigma[fit_rows])
                encoded[t.id] = t.encode(p.targets, p.sigma, states[t.id])
            data = TrainingData(
                x_train=scaler.transform(train_part.x_raw[fit_rows]),
                y_train={key: y[fit_rows] for key, y in encoded.items()},
                x_select=scaler.transform(train_part.x_raw[select_rows]),
                y_select={key: y[select_rows] for key, y in encoded.items()},
                target_states=states,
            )
            if run.preset in TRANSFORMERS:
                with TemporaryDirectory() as folder:
                    result = train_reference(
                        data,
                        config,
                        run.preset,
                        Path(folder) / "fold.pt",
                        self.spec.device,
                        lambda _record: self._check(),
                    )
                    outputs = result.model.predict(
                        scaler.transform(train_part.x_raw[eval_rows])
                    )
            else:
                model = MLP(
                    data.x_train.shape[1],
                    self.spec.architecture(run.preset),
                    seed=run.seed,
                )
                result = train(
                    model, data, config, on_epoch=lambda _record: self._check()
                )
                outputs = model.predict(scaler.transform(train_part.x_raw[eval_rows]))
            metrics = {}
            for ds in datasets:
                p = ds.parts["train"]
                metrics[ds.target.id] = ds.target.evaluate(
                    outputs[ds.target.id],
                    encoded[ds.target.id][eval_rows],
                    p.targets[eval_rows],
                    p.sigma[eval_rows],
                    states[ds.target.id],
                )
            reports.append(
                {
                    "fold": k,
                    "start": start.isoformat(),
                    "end": None if end is None else end.isoformat(),
                    "train_rows": int(fit_rows.sum()),
                    "selection_rows": int(select_rows.sum()),
                    "selection_start": nested_cutoff.isoformat(),
                    "best_epoch": result.best_epoch,
                    "evaluated_rows": int(eval_rows.sum()),
                    "metrics": metrics,
                }
            )
        return reports

    def _run(self, run: Run, datasets: list[Dataset]) -> dict[str, Any]:
        began = time.monotonic()
        config = self.spec.train_config(run.preset, run.seed)
        data = TrainingData.from_datasets(datasets)

        def progress(record: dict[str, Any]) -> None:
            self._check()
            self._write_status(
                current={
                    "run": run.key,
                    "epoch": record["epoch"],
                    "epochs": config.epochs,
                    "train_loss": record["train_loss"],
                    "select_loss": record["select_loss"],
                }
            )
            trace = self.folder / "progress" / f"{run.key}.json"
            trace.parent.mkdir(exist_ok=True)
            previous = (
                read_json(trace)["history"]
                if trace.is_file() and record["epoch"] > 1
                else []
            )
            write_json(trace, {"history": [*previous, _finite(record)]})

        checkpoint = self.folder / "checkpoints" / run.key
        if run.preset in TRANSFORMERS:
            result = train_reference(
                data,
                config,
                run.preset,
                checkpoint.with_suffix(".pt"),
                self.spec.device,
                progress,
            )
            model = result.model
            architecture = {
                "encoder": "reference",
                "variant": TRANSFORMERS[run.preset],
                "embedding": 16,
                "heads": 2,
            }
            device = result.device
        else:
            mlp_architecture = self.spec.architecture(run.preset)
            model = MLP(data.x_train.shape[1], mlp_architecture, seed=run.seed)
            result = train(model, data, config, on_epoch=progress)
            architecture = mlp_architecture.model_dump(mode="json")
            device = "cpu"
        report: dict[str, Any] = {
            "key": run.key,
            "state": "complete",
            "spec_fingerprint": self.fingerprint,
            "feature_profile": run.feature_profile,
            "preset": run.preset,
            "seed": run.seed,
            "targets": list(self.spec.targets),
            "inputs": model.n_in,
            "parameter_count": model.parameter_count,
            "architecture": architecture,
            "training": config.model_dump(mode="json"),
            "device": device,
            "best_epoch": result.best_epoch,
            "epochs_run": len(result.history),
            "stopped_early": result.stopped_early,
            "history": result.history,
            "rows": {name: len(part) for name, part in datasets[0].parts.items()},
            "dataset_fingerprints": {ds.target.id: ds.fingerprint for ds in datasets},
            "inner_validation": self._evaluate(model, datasets, "inner_validation"),
        }
        if self.spec.reads_validation:
            report["validation"] = self._evaluate(model, datasets, "validation")
            self._record_validation_read(run, datasets)
        if self.spec.experiment_profile.walk_forward_folds:
            report["walk_forward"] = self._walk_forward(run, datasets)
        # Separate folder: ``checkpoints/<key>.json`` must never collide with ``runs/<key>.json``.
        if run.preset in TRANSFORMERS:
            register_transformer(checkpoint, result, config, datasets, run.preset)
        else:
            save(
                checkpoint,
                model,
                config,
                dataset=datasets[0],
                target_states={ds.target.id: ds.target_state or {} for ds in datasets},
                history=result.history,
                best_epoch=result.best_epoch,
            )
        report["checkpoint"] = f"checkpoints/{run.key}"
        report["seconds"] = round(time.monotonic() - began, 2)
        return report

    def _record_validation_read(self, run: Run, datasets: list[Dataset]) -> None:
        self.record_validation_read(run.key, datasets[0].fingerprint)

    def record_validation_read(self, label: str, dataset: str | None) -> None:
        """Append one validation read to ``<data_dir>/experiments/validation_reads.json``."""
        ledger = self.data_dir / "experiments" / "validation_reads.json"
        entries = read_json(ledger)["reads"] if ledger.is_file() else []
        entries.append(
            {
                "experiment": self.folder.name,
                "run": label,
                "at": _now(),
                "dataset": dataset,
                "spec_fingerprint": self.fingerprint,
            }
        )
        write_json(ledger, {"reads": entries})

    # ------------------------------------------------------------ campaign
    def _completed(self, run: Run) -> bool:
        path = self.folder / "runs" / f"{run.key}.json"
        if not path.is_file():
            return False
        try:
            report = read_json(path)
        except (OSError, ValueError):
            return False
        return (
            report.get("state") == "complete"
            and report.get("spec_fingerprint") == self.fingerprint
        )

    def run(self) -> dict[str, Any]:
        runs = self.spec.runs()
        (self.folder / "runs").mkdir(exist_ok=True)
        self.status = {
            "state": "running",
            "message": "Loading data",
            "done": sum(self._completed(r) for r in runs),
            "total": len(runs),
            "failed": [],
            "started_at": _now(),
            "spec_fingerprint": self.fingerprint,
        }
        self._write_status()
        try:
            frames = load_frames(self.spec.sources, self.data_dir)
            self.verify_data(frames)
            cache: dict[str, list[Dataset]] = {}
            for run in runs:
                self._check()
                if self._completed(run):
                    continue
                if run.feature_profile not in cache:
                    self._write_status(
                        message=f"Building datasets for {run.feature_profile}"
                    )
                    cache[run.feature_profile] = self._datasets(
                        frames, run.feature_profile
                    )
                self._write_status(
                    message=f"Training {run.key}", current={"run": run.key}
                )
                try:
                    report = self._run(run, cache[run.feature_profile])
                except (CancelledError, TimeBudgetError):
                    raise
                except (
                    TrainingDivergedError,
                    ValueError,
                    ArithmeticError,
                    ImportError,
                    RuntimeError,
                ) as exc:
                    report = {
                        "key": run.key,
                        "state": "failed",
                        "spec_fingerprint": self.fingerprint,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.status["failed"].append(run.key)
                write_json(self.folder / "runs" / f"{run.key}.json", _finite(report))
                self.status["done"] += 1
            state = "complete_with_failures" if self.status["failed"] else "complete"
            self._write_status(state=state, message="Ready", current=None)
        except CancelledError as exc:
            self._write_status(state="cancelled", message=str(exc), current=None)
        except TimeBudgetError as exc:
            self._write_status(state="stopped", message=str(exc), current=None)
        except Exception as exc:  # noqa: BLE001 - the worker must publish any failure
            self._write_status(
                state="failed", message=f"{type(exc).__name__}: {exc}", current=None
            )
            (self.folder / "worker_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        return self.status


def run_folder(
    folder: Path, data_dir: Path, split: SplitConfig | None = None
) -> dict[str, Any]:
    return Runner(folder, data_dir, split).run()


def write_spec(folder: Path, spec: ExperimentSpec) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "spec.json",
        {"spec": spec.model_dump(mode="json"), "fingerprint": spec.fingerprint()},
    )
    write_json(
        folder / "status.json",
        {
            "state": "queued",
            "message": "Starting",
            "done": 0,
            "total": len(spec.runs()),
            "failed": [],
            "updated_at": _now(),
        },
    )


def read_spec(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "spec.json").read_text(encoding="utf-8"))
