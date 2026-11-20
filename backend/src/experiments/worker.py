"""Subprocess entry point: ``python -m src.experiments.worker <experiment folder>``.

Run from ``backend/``. The folder must already hold ``spec.json`` (written by
the API or by ``runner.write_spec``). Exit code 0 unless the folder is invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import get_settings
from src.experiments.runner import Runner
from src.services.study_artifacts import read_json, write_json


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m src.experiments.worker <experiment folder>",
            file=sys.stderr,
        )
        return 2
    folder = Path(argv[0]).resolve()
    if not (folder / "spec.json").is_file():
        print(f"no spec.json in {folder}", file=sys.stderr)
        return 2
    runner = Runner(folder, get_settings().data_dir)
    status = runner.run()
    if runner.spec.auto_evaluate and status["state"] in {
        "complete",
        "complete_with_failures",
    }:
        from src.evaluation import evaluate_experiment

        path = folder / "status.json"
        write_json(path, {**read_json(path), "evaluation_state": "running"})
        try:
            evaluate_experiment(folder, runner.data_dir, "inner_validation")
            write_json(path, {**read_json(path), "evaluation_state": "complete"})
        except Exception as exc:
            write_json(
                path,
                {
                    **read_json(path),
                    "evaluation_state": "failed",
                    "evaluation_error": str(exc),
                },
            )
    print(f"{folder.name}: {status['state']} ({status['done']}/{status['total']})")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the API subprocess
    raise SystemExit(main(sys.argv[1:]))
