"""Exports use saved evidence only; generating a report never trains a model."""

import csv
import io
import json
from pathlib import Path

from src.services.study_artifacts import read_json


def export_report(folder: Path, kind: str) -> tuple[str, str]:
    spec = read_json(folder / "spec.json")
    status = read_json(folder / "status.json")
    runs = [read_json(p) for p in sorted((folder / "runs").glob("*.json"))]
    evaluations = {
        p.stem: read_json(p) for p in sorted((folder / "evaluation").glob("*.json"))
    }
    if kind == "json":
        return json.dumps(
            {"spec": spec, "status": status, "runs": runs, "evaluations": evaluations},
            indent=2,
        ), "application/json"
    if kind == "csv":
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["run", "state", "partition", "target", "metric", "value"])
        for run in runs:
            for part in ("inner_validation", "validation"):
                for target, metrics in run.get(part, {}).items():
                    for name, value in metrics.items():
                        if isinstance(value, (float, int)) and not isinstance(
                            value, bool
                        ):
                            writer.writerow(
                                [run["key"], run["state"], part, target, name, value]
                            )
        return out.getvalue(), "text/csv"
    if kind == "md":
        lines = [
            f"# {spec['spec']['name']}",
            "",
            f"State: {status['state']} · {status['done']}/{status['total']} runs.",
            "",
            f"Spec fingerprint: `{spec['fingerprint']}`",
            "",
            "## Protocol",
            "",
            "```json",
            json.dumps(spec["spec"], indent=2),
            "```",
            "",
            "## Run evidence",
            "",
            "| Run | State | Selected epoch | Parameters |",
            "|---|---|---:|---:|",
        ]
        for run in runs:
            lines.append(
                f"| {run['key']} | {run['state']} | {run.get('best_epoch', '—')} | {run.get('parameter_count', '—')} |"
            )
        for part, evaluation in evaluations.items():
            lines += [
                "",
                f"## {part}",
                "",
                evaluation["ablation"]["summary"],
                "",
                "| Configuration | Target | Rows | Ensemble loss | Strongest baseline | Verdict |",
                "|---|---|---:|---:|---|---|",
            ]
            for entry in evaluation["results"]:
                baseline = entry["strongest_baseline"]
                comparison = entry["baselines"][baseline]["comparison"]
                lines.append(
                    f"| {entry['feature_profile']} / {entry['preset']} | {entry['target']} | {entry['rows']} | {entry['ensemble']['loss']:.6g} | {baseline} | {comparison['verdict']} |"
                )
            lines.extend(["", *[f"- {note}" for note in evaluation["notes"]]])
        lines += [
            "",
            "## Limitations",
            "",
            "Development evidence, not final holdout results. "
            "Seed variance and market-date uncertainty are distinct. No live-trading or profitability claim.",
        ]
        return "\n".join(lines) + "\n", "text/markdown"
    raise ValueError("Unknown export format")
