"""Prospective final-evaluation protocol. No HTTP endpoint can reveal scores.

Freeze a candidate before its prospective period starts. Explicit local CLI
authorization is required after the period ends. A consumed marker is written
before opening test data, so a failed attempt cannot silently become a rerun.
This is local procedural protection, not security against a filesystem owner.
"""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.datasets import DatasetSpec, build_dataset
from src.datasets.manifest import code_revision, read_manifest
from src.experiments.models import load_registered
from src.experiments.runner import Runner, _finite
from src.experiments.sources import SYNTHETIC, SYNTHETIC_SIGNAL, load_frames
from src.services.study_artifacts import read_json, write_json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(folder: Path, data_dir: Path, key: str, start: date, end: date):
    runner = Runner(folder, data_dir)
    if start <= date.today() or end <= start:
        raise ValueError("Final holdout must be prospective, with end after start")
    if {SYNTHETIC, SYNTHETIC_SIGNAL} & set(runner.spec.sources):
        raise ValueError("Synthetic data is a control, not a final historical holdout")
    if key not in {r.key for r in runner.spec.runs()}:
        raise ValueError("Unknown candidate")
    report = read_json(folder / "runs" / f"{key}.json")
    if report.get("state") != "complete":
        raise ValueError("Candidate not completed")
    model = load_registered(folder / report["checkpoint"])
    if not (folder / "data_lock.json").is_file():
        raise ValueError("Candidate predates the development-data lock; rerun first")
    manifest = {
        "version": 1,
        "candidate": key,
        "experiment": folder.name,
        "spec_fingerprint": runner.fingerprint,
        "checkpoint_manifest": digest(model.manifest),
        "dataset_fingerprints": report["dataset_fingerprints"],
        "development": read_json(folder / "data_lock.json"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision(),
        "metrics": "registered target metrics and development-fitted baselines",
        "policy": "predictive final evaluation only; no selection or retraining",
    }
    plan = {**manifest, "authorization_id": digest(manifest)}
    path = folder / "holdout_plan.json"
    # The project-wide record prevents independently locking multiple
    # candidates against one supposedly final period.
    global_path = data_dir / "experiments" / "final_holdout.json"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    with global_path.open("x", encoding="utf-8") as out:
        json.dump(plan, out, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as out:
        json.dump(plan, out, indent=2, allow_nan=False)
    return plan


def evaluate(folder: Path, data_dir: Path, authorization_id: str):
    plan = read_json(folder / "holdout_plan.json")
    if read_json(data_dir / "experiments" / "final_holdout.json") != plan:
        raise ValueError("Plan differs from the project-wide holdout lock")
    body = {key: value for key, value in plan.items() if key != "authorization_id"}
    if digest(body) != authorization_id or plan["authorization_id"] != authorization_id:
        raise ValueError("Authorization must match the immutable holdout plan")
    if date.today() <= date.fromisoformat(plan["end"]):
        raise ValueError("Prospective evaluation period has not ended")
    if plan["code_revision"] != code_revision():
        raise ValueError("Code revision differs from the frozen protocol")
    runner = Runner(folder, data_dir)
    if runner.fingerprint != plan["spec_fingerprint"]:
        raise ValueError("Spec differs from the frozen protocol")
    key = plan["candidate"]
    report = read_json(folder / "runs" / f"{key}.json")
    model = load_registered(folder / report["checkpoint"])
    if digest(model.manifest) != plan["checkpoint_manifest"]:
        raise ValueError("Checkpoint differs from frozen candidate")
    # Record authorization before opening any protected OHLCV. No automatic retry.
    with (folder / "holdout_consumed.json").open("x", encoding="utf-8") as consumed:
        json.dump(
            {
                "authorization_id": authorization_id,
                "at": datetime.now(timezone.utc).isoformat(),
            },
            consumed,
        )
    frames = load_frames(runner.spec.sources, data_dir)
    runner.verify_data(frames)
    if read_json(folder / "data_lock.json") != plan["development"]:
        raise ValueError("Development identity differs")
    end = date.fromisoformat(plan["end"])
    for frame in frames.values():
        if pd.Timestamp(end) - frame.index.max() > pd.Timedelta(days=7):
            raise ValueError("Holdout data does not cover the frozen end")
    # Limit data to the planned end even if the source has later observations.
    frames = {
        asset: frame.loc[frame.index <= pd.Timestamp(end)]
        for asset, frame in frames.items()
    }
    split = runner.split.model_copy(
        update={"test_start": date.fromisoformat(plan["start"]), "test_end": end}
    )
    metrics = {}
    for target_id in runner.spec.targets:
        original = read_manifest(
            folder / "datasets" / f"{report['feature_profile']}__{target_id}.json"
        )
        if original["dataset_fingerprint"] != plan["dataset_fingerprints"][target_id]:
            raise ValueError("Training manifest differs")
        spec = DatasetSpec.model_validate(original["spec"]).model_copy(
            update={"include_test": True}
        )
        ds = build_dataset(frames, spec, split)
        part = ds.parts["test"]
        if (
            not len(part)
            or ds.columns != model.columns
            or ds.target_state != model.target_states[target_id]
        ):
            raise ValueError("Empty holdout or preprocessing mismatch")
        if ds.scaler.to_dict() != model.scaler.to_dict():
            raise ValueError("Training scaler changed")
        output = model.predict_raw_features(part.x_raw)[target_id]
        from src.evaluation.baselines import baseline_outputs

        baselines, _ = baseline_outputs(
            ds.target,
            ds.target_state or {},
            ds.parts["inner_train"],
            ds.parts["inner_validation"],
            part,
        )
        metrics[target_id] = {
            "rows": len(part),
            "model": ds.target.evaluate(
                output, part.y, part.targets, part.sigma, ds.target_state or {}
            ),
            "baselines": {
                name: ds.target.evaluate(
                    z, part.y, part.targets, part.sigma, ds.target_state or {}
                )
                for name, z in baselines.items()
                if np.isfinite(z).all()
            },
        }
    result = _finite({"plan": plan, "metrics": metrics, "state": "complete"})
    write_json(folder / "holdout_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--authorize-id", required=True)
    args = parser.parse_args()
    evaluate(args.folder, get_settings().data_dir, args.authorize_id)
    print("Final evaluation recorded in holdout_result.json")


if __name__ == "__main__":
    main()
