"""Isolated worker for reproducible local validation studies."""

import json
import sys
from pathlib import Path

import pandas as pd
from export_comparison import export, synthetic_frame


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def work(root):
    spec = json.loads((root / "config.json").read_text())
    source = spec["source"]
    frame = (
        synthetic_frame()
        if source == "synthetic"
        else pd.read_csv(spec["csv"], parse_dates=["date"], index_col="date")
    )
    total = len(spec["seeds"]) * sum(
        len(spec["variants"]) if m == "Transformer" else 1 for m in spec["models"]
    )
    completed = 0
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
                )
                write(root / (key + ".json"), result)
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
        dict(state="complete", message="Ready", done=completed, total=total),
    )


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    try:
        work(folder)
    except Exception as exc:
        write(
            folder / "status.json",
            dict(state="failed", message=str(exc), done=0, total=0),
        )
        raise
