"""Isolated worker for reproducible local validation studies."""

import json
import sys
from pathlib import Path

import pandas as pd
from export_comparison import dataset_timeline, export, synthetic_frame

from src.data.split import load_split_config
from src.eval.execution import load_benchmark_profile
from src.services.study_artifacts import overview_entry, run_keys
from src.services.study_artifacts import write_json as write


def work(root):
    spec = json.loads((root / "config.json").read_text())
    source = spec["source"]
    frame = (
        synthetic_frame()
        if source == "synthetic"
        else pd.read_csv(spec["csv"], parse_dates=["date"], index_col="date")
    )
    total = len(run_keys(spec))
    completed = 0
    config = load_split_config()
    profile = load_benchmark_profile()
    timeline = dataset_timeline(frame, config)
    for model in spec["models"]:
        for variant in spec["variants"] if model == "Transformer" else ["reference"]:
            for seed in spec["seeds"]:
                key = f"{model}-{variant}-{seed}"

                def progress(message, key=key, completed=completed):
                    write(
                        root / "status.json",
                        dict(
                            state="running",
                            message=f"{key} · {message}",
                            done=completed,
                            total=total,
                        ),
                    )

                checkpoint = root / (
                    key + (".pt" if model == "Transformer" else ".npz")
                )
                result = export(
                    frame,
                    "synthetic" if source == "synthetic" else "experimental",
                    seed,
                    checkpoint,
                    model=model,
                    variant=variant,
                    epochs=spec["epochs"],
                    progress=progress,
                    research=True,
                    config=config,
                    profile=profile,
                )
                if result["dataset"] != timeline["dataset"]:
                    raise ValueError("Timeline and run use different data")
                if completed == 0:
                    write(root / "timeline.json", timeline)
                if "bestDecisions" in result and not (root / "best_decisions.json").exists():
                    write(root / "best_decisions.json", result["bestDecisions"])
                write(root / (key + ".json"), result)
                write(root / (key + ".overview"), overview_entry(key, result))
                write(
                    root / ("summary-" + key + ".meta"),
                    dict(
                        id=root.name,
                        key=key,
                        source=source,
                        model=model,
                        variant=variant,
                        seed=seed,
                        f1=result["macroF1"],
                        dataset=result["dataset"],
                        epochs=spec["epochs"],
                    ),
                )
                completed += 1
                write(
                    root / "status.json",
                    dict(
                        state="running",
                        message=f"{key} · Saved",
                        done=completed,
                        total=total,
                    ),
                )
    write(
        root / "status.json",
        dict(state="complete", message="Ready", done=completed, total=total),
    )


def run(folder):
    try:
        work(folder)
    except Exception as exc:
        status_path = folder / "status.json"
        previous = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else dict(done=0, total=0)
        )
        write(
            folder / "status.json",
            {**previous, "state": "failed", "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    run(Path(sys.argv[1]))
