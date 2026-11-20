"""Filesystem contract shared by the study worker and HTTP readers."""

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete artifact, never a partially written JSON document.

    Each study artifact has one writer (the worker, or the API under its lock).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def run_keys(config: dict[str, Any]) -> list[str]:
    return [
        f"{model}-{variant}-{seed}"
        for model in config["models"]
        for variant in (config["variants"] if model == "Transformer" else ["reference"])
        for seed in config["seeds"]
    ]


def overview_entry(key: str, report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    return dict(
        key=key,
        model=report["model"],
        variant=report["variant"],
        seed=report["seed"],
        f1=report["macroF1"],
        loss=min(x["validationLoss"] for x in report["curves"]),
        curves=report["curves"],
        hasTrace="trace" in report,
        accuracy=sum((c["p"] >= 0.5) == (c["next"] > 0) for c in cases) / len(cases),
    )


def overview_runs(folder: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read small immutable summaries; older or interrupted writes fall back."""
    runs = []
    for key in sorted(run_keys(config)):
        report = folder / f"{key}.json"
        if not report.is_file():
            continue
        summary = folder / f"{key}.overview"
        if (
            summary.is_file()
            and summary.stat().st_mtime_ns >= report.stat().st_mtime_ns
        ):
            try:
                runs.append(read_json(summary))
                continue
            except (OSError, ValueError):
                # Derived metadata must not make a valid report unavailable.
                pass
        runs.append(overview_entry(key, read_json(report)))
    return runs
