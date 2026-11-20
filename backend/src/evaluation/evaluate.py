"""Scientific evaluation of a finished experiment (tranche T7).

``evaluate_experiment(folder, data_dir, part)`` rebuilds the experiment's
datasets (their fingerprints must match the run reports), reloads every
checkpoint (column order checked) and writes ``evaluation/<part>.json``:

- per (feature profile, preset, target): metrics of the seed ensemble and of
  each seed, every baseline, paired date-block comparisons with verdicts,
  calibration (classification), regime slices, block-preserving feature
  importance (primary target) and, for ``direction_binary_v1``, a simplified
  vectorized trading simulation;
- per feature profile: input and label drift from inner_train to ``part``;
- a feature-group ablation against ``returns_only_v1`` (or the first profile).

``part`` is ``inner_validation`` (default) or ``validation``; every validation
evaluation is appended to the validation-read ledger. The test part is never
built.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ai.mlp import load
from src.data.split import SplitConfig
from src.datasets import Dataset
from src.eval.execution import load_benchmark_profile
from src.evaluation.baselines import _quantile_outputs, baseline_outputs
from src.evaluation.stats import drift, loss_name, paired_difference, row_losses
from src.experiments.runner import Runner, _finite
from src.experiments.sources import load_frames
from src.features import DEFAULT_REGISTRY, compute, get_profile
from src.services.study_artifacts import read_json, write_json
from src.targets import Target, TargetKind, get_target, softmax
from src.targets import metrics as m

PARTS = ("inner_validation", "validation")
REFERENCE_PROFILE = "returns_only_v1"
FINISHED = {"complete", "complete_with_failures"}


class EvaluationError(ValueError):
    """The experiment cannot be evaluated (unfinished, changed data, bad part)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ ensembles
def ensemble(
    target: Target, outputs: list[np.ndarray], state: dict[str, Any]
) -> np.ndarray:
    """Seed ensemble in the target's output space (probabilities, values or quantiles averaged)."""
    if target.kind is TargetKind.CLASSIFICATION:
        p = np.mean([softmax(z) for z in outputs], axis=0)
        return np.log(np.clip(p, 1e-12, 1.0))
    if target.kind is TargetKind.QUANTILE:
        decoded = [target.decode(z, state) for z in outputs]
        mean = {
            q: np.mean([d[q] for d in decoded], axis=0) for q in ("q10", "q50", "q90")
        }
        return _quantile_outputs(mean["q10"], mean["q50"], mean["q90"], state)
    return np.mean(outputs, axis=0)


# --------------------------------------------------------------------- pieces
def regime_labels(
    datasets: list[Dataset],
    part: str,
    frames: dict[str, pd.DataFrame],
    split: SplitConfig,
) -> dict[str, np.ndarray]:
    """Volatility tercile (cut-offs fitted on training sigma) and 60-session trend sign at t."""
    rows = datasets[0].parts[part]
    train_sigma = datasets[0].parts["train"].sigma
    low, high = np.quantile(train_sigma, [1 / 3, 2 / 3])
    sigma = rows.sigma
    volatility = np.where(
        sigma <= low, "calm", np.where(sigma <= high, "normal", "stressed")
    )
    trend = np.empty(len(rows), dtype=object)
    for asset, group in rows.index.groupby("asset").groups.items():
        frame = frames[asset]
        frame = frame.loc[frame.index <= pd.Timestamp(split.validation_end)]
        values = compute(frame, ["cum_return_60"]).values["cum_return_60"].to_numpy()
        positions = np.asarray(list(group))
        bars = rows.index.loc[positions, "bar"].to_numpy()
        trend[positions] = np.where(values[bars] >= 0, "up_60d", "down_60d")
    return {"volatility_tercile": volatility, "trend_60d": trend.astype(str)}


def trading_simulation(
    p_up: np.ndarray, part, positions_from: dict[str, np.ndarray], cost: float
) -> dict[str, Any]:
    """Vectorized long/flat simulation on executed returns; NOT the replay engine."""
    index = part.index
    returns = np.asarray(part.targets, float)[:, 0]
    out = {}
    for name, pos in {
        "model_ensemble": p_up >= 0.5,
        **positions_from,
        "always_long": np.ones(len(p_up), bool),
    }.items():
        pos = np.asarray(pos, float)
        net = np.zeros(len(pos))
        turnover = 0.0
        for _asset, rows in index.groupby("asset").groups.items():
            order = index.loc[list(rows)].sort_values("date").index.to_numpy()
            p = pos[order]
            changes = np.abs(np.diff(np.r_[0.0, p]))
            net[order] = p * returns[order] - cost * changes
            turnover += changes.sum()
        per_date = pd.Series(net).groupby(index["date"].to_numpy()).mean().sort_index()
        curve = per_date.cumsum().to_numpy()
        drawdown = (
            float(np.min(curve - np.maximum.accumulate(np.r_[0.0, curve])[1:]))
            if len(curve)
            else 0.0
        )
        out[name] = {
            "total_log_return_equal_weight": float(per_date.sum()),
            "mean_log_return_per_decision": float(net.mean()),
            "exposure": float(pos.mean()),
            "turnover_per_decision": float(turnover / len(pos)),
            "max_drawdown_log": drawdown,
        }
    out["note"] = (
        "Simplified vectorized long/flat simulation on executed returns with a half-spread cost per "
        "position change; not the replay engine, not used for model selection."
    )
    return out


def importance(
    target: Target,
    models,
    dataset: Dataset,
    part: str,
    state: dict[str, Any],
    features: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Loss increase when one feature's columns are circularly shifted within each asset.

    The shift keeps each column's own time structure (autocorrelation) and only
    breaks its alignment with the label. Predictive association, not causation.
    """
    p = dataset.parts[part]
    base = row_losses(
        target,
        ensemble(target, [mdl.predict(p.x)[target.id] for mdl in models], state),
        p.y,
        p.targets,
        state,
    ).mean()
    groups = p.index.groupby("asset").groups
    out = []
    column_index = {c: i for i, c in enumerate(dataset.columns)}
    for feature in features:
        cols = [column_index[c] for c in DEFAULT_REGISTRY.get(feature).columns]
        x = p.x.copy()
        for rows in groups.values():
            order = p.index.loc[list(rows)].sort_values("date").index.to_numpy()
            x[np.ix_(order, cols)] = np.roll(
                p.x[np.ix_(order, cols)], len(order) // 2, axis=0
            )
        loss = row_losses(
            target,
            ensemble(target, [mdl.predict(x)[target.id] for mdl in models], state),
            p.y,
            p.targets,
            state,
        ).mean()
        out.append(
            {
                "feature": feature,
                "columns": len(cols),
                "loss_increase": float(loss - base),
            }
        )
    out.sort(key=lambda d: -d["loss_increase"])
    return out


# ---------------------------------------------------------------- evaluation
def evaluate_experiment(
    folder: Path,
    data_dir: Path,
    part: str = "inner_validation",
    split: SplitConfig | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    if part not in PARTS:
        raise EvaluationError(f"part must be one of {PARTS}")
    runner = Runner(folder, data_dir, split)
    spec = runner.spec
    status = read_json(folder / "status.json")
    if status.get("state") not in FINISHED:
        raise EvaluationError("the experiment is not finished")
    reports = {
        r["key"]: r
        for r in (read_json(p) for p in sorted((folder / "runs").glob("*.json")))
        if r.get("state") == "complete"
    }
    if not reports:
        raise EvaluationError("no completed run to evaluate")
    frames = load_frames(spec.sources, data_dir)
    cost = load_benchmark_profile().spread_bps / 2 / 10000
    result: dict[str, Any] = {
        "experiment": folder.name,
        "spec": spec.model_dump(mode="json"),
        "spec_fingerprint": runner.fingerprint,
        "part": part,
        "generated_at": _now(),
        "loss_definitions": {t: loss_name(get_target(t)) for t in spec.targets},
        "results": [],
        "drift": {},
        "notes": [
            "Seed ensemble = average of the seeds' probabilities, values or quantiles.",
            "Intervals: 95% paired date-block bootstrap; 'no measurable difference' is not equivalence.",
            "Importance: circular shift within each asset; predictive association, not causation.",
        ],
    }
    if part == "validation":
        runner.record_validation_read("evaluation", None)
    losses: dict[tuple[str, str, str], dict[str, Any]] = {}
    regimes = None
    for fp in spec.feature_profiles:
        datasets = runner._datasets(frames, fp)  # noqa: SLF001
        by_target = {ds.target.id: ds for ds in datasets}
        if regimes is None:
            regimes = regime_labels(datasets, part, frames, runner.split)
        first = datasets[0]
        result["drift"][fp] = {
            "inputs": drift(
                first.parts["inner_train"].x_raw, first.parts[part].x_raw, first.columns
            ),
            "labels": {
                t: {
                    "inner_train_mean": float(
                        np.mean(by_target[t].parts["inner_train"].targets[:, 0])
                    ),
                    f"{part}_mean": float(
                        np.mean(by_target[t].parts[part].targets[:, 0])
                    ),
                }
                for t in spec.targets
            },
            "dropped_undefined": {
                t: by_target[t].manifest["parts"] for t in spec.targets
            },
        }
        baselines = {}
        for t, ds in by_target.items():
            state = ds.target_state or {}
            baselines[t] = baseline_outputs(
                get_target(t),
                state,
                ds.parts["inner_train"],
                ds.parts["inner_validation"],
                ds.parts[part],
            )
        for ps in spec.presets:
            runs = [
                r
                for r in reports.values()
                if r["feature_profile"] == fp and r["preset"] == ps
            ]
            if not runs:
                continue
            for r in runs:
                for t, ds in by_target.items():
                    if r["dataset_fingerprints"][t] != ds.fingerprint:
                        raise EvaluationError(
                            f"{r['key']}: dataset changed since training"
                        )
            models = [
                load(folder / r["checkpoint"], expected_columns=first.columns).model
                for r in runs
            ]
            seeds = [r["seed"] for r in runs]
            gap = float(np.mean([_gap(r) for r in runs]))
            for t, ds in by_target.items():
                target, state, p = get_target(t), ds.target_state or {}, ds.parts[part]
                per_seed = [mdl.predict(p.x)[t] for mdl in models]
                z = ensemble(target, per_seed, state)
                model_rows = row_losses(target, z, p.y, p.targets, state)
                seed_rows = [
                    row_losses(target, zs, p.y, p.targets, state) for zs in per_seed
                ]
                block = max(10, target.horizon_bars)
                dates = p.index["date"].to_numpy()
                base_out, base_info = baselines[t]
                base_rows = {
                    name: row_losses(target, zb, p.y, p.targets, state)
                    for name, zb in base_out.items()
                }
                strongest = min(base_rows, key=lambda name: base_rows[name].mean())
                entry: dict[str, Any] = {
                    "feature_profile": fp,
                    "preset": ps,
                    "target": t,
                    "seeds": seeds,
                    "rows": len(p),
                    "loss": loss_name(target),
                    "ensemble": {
                        "loss": float(model_rows.mean()),
                        "metrics": target.evaluate(z, p.y, p.targets, p.sigma, state),
                    },
                    "per_seed_loss": {
                        str(s): float(rows.mean())
                        for s, rows in zip(seeds, seed_rows, strict=True)
                    },
                    "seed_loss_spread": {
                        "mean": float(np.mean([rows.mean() for rows in seed_rows])),
                        "std": (
                            float(np.std([rows.mean() for rows in seed_rows], ddof=1))
                            if len(seed_rows) > 1
                            else None
                        ),
                    },
                    "train_minus_selection_loss_at_best_epoch": gap,
                    "baselines": {
                        name: {
                            "loss": float(rows.mean()),
                            "metrics": target.evaluate(
                                base_out[name], p.y, p.targets, p.sigma, state
                            ),
                            "comparison": paired_difference(
                                model_rows, rows, dates, block=block, n_boot=n_boot
                            ),
                            "seeds_better": int(
                                sum(sr.mean() < rows.mean() for sr in seed_rows)
                            ),
                        }
                        for name, rows in base_rows.items()
                    },
                    "baseline_settings": base_info,
                    "strongest_baseline": strongest,
                    "regimes": {
                        kind: {
                            str(label): {
                                "rows": int((labels == label).sum()),
                                "model_loss": float(model_rows[labels == label].mean()),
                                "strongest_baseline_loss": float(
                                    base_rows[strongest][labels == label].mean()
                                ),
                            }
                            for label in sorted(set(labels))
                        }
                        for kind, labels in (regimes or {}).items()
                    },
                }
                if target.kind is TargetKind.CLASSIFICATION:
                    probs = softmax(z)
                    entry["calibration"] = {
                        "ece": m.expected_calibration_error(
                            probs.max(axis=1), probs.argmax(axis=1) == p.y
                        ),
                        "reliability_up": m.reliability_bins(
                            probs[:, -1], p.y == target.output_dim - 1
                        ),
                    }
                if t == spec.targets[0]:
                    entry["importance"] = importance(
                        target, models, ds, part, state, get_profile(fp).features
                    )
                if t == "direction_binary_v1":
                    logistic = softmax(base_out["logistic_regression"])[:, 1] >= 0.5
                    entry["trading"] = trading_simulation(
                        softmax(z)[:, 1], p, {"logistic_regression": logistic}, cost
                    )
                losses[(fp, ps, t)] = {
                    "rows": model_rows,
                    "seeds": dict(zip(seeds, seed_rows, strict=True)),
                    "gap": gap,
                    "dates": dates,
                    "block": block,
                    "index": p.index,
                }
                result["results"].append(entry)
    result["ablation"] = ablation(spec, losses, n_boot)
    out = _finite(result)
    target_path = folder / "evaluation" / f"{part}.json"
    target_path.parent.mkdir(exist_ok=True)
    write_json(target_path, out)
    return out


def _gap(report: dict[str, Any]) -> float:
    best = next(h for h in report["history"] if h.get("selected"))
    return float(best["train_loss"] - best["select_loss"])


def ablation(
    spec, losses: dict[tuple[str, str, str], dict[str, Any]], n_boot: int
) -> dict[str, Any]:
    """Each feature profile against the reference profile, same preset and target, same rows."""
    reference = (
        REFERENCE_PROFILE
        if REFERENCE_PROFILE in spec.feature_profiles
        else spec.feature_profiles[0]
    )
    findings = []
    for (fp, ps, t), current in sorted(losses.items()):
        if fp == reference or (reference, ps, t) not in losses:
            continue
        ref = losses[(reference, ps, t)]
        if not current["index"].equals(ref["index"]):
            raise EvaluationError(
                "ablation needs identical rows; rebuild the experiment"
            )
        comparison = paired_difference(
            current["rows"],
            ref["rows"],
            current["dates"],
            block=current["block"],
            n_boot=n_boot,
        )
        common = sorted(set(current["seeds"]) & set(ref["seeds"]))
        seed_diffs = [
            float(current["seeds"][s].mean() - ref["seeds"][s].mean()) for s in common
        ]
        better = sum(d < 0 for d in seed_diffs)
        if comparison["verdict"] == "better":
            category = (
                "improves"
                if better == len(seed_diffs)
                else "improves but unstable across seeds"
            )
        elif comparison["verdict"] == "worse":
            category = "worse than reference"
        else:
            category = "no measurable improvement"
        flags = []
        if 0 < better < len(seed_diffs):
            flags.append("unstable: seeds disagree on the sign")
        if current["gap"] < ref["gap"] - 0.005:
            flags.append(
                "overfitting sign: larger train/selection gap than the reference"
            )
        findings.append(
            {
                "feature_profile": fp,
                "preset": ps,
                "target": t,
                "reference": reference,
                "comparison": comparison,
                "per_seed_difference": dict(
                    zip([str(s) for s in common], seed_diffs, strict=True)
                ),
                "category": category,
                "flags": flags,
                "train_minus_selection_gap": {
                    "profile": current["gap"],
                    "reference": ref["gap"],
                },
            }
        )
    improving = sorted(
        {
            f"{f['feature_profile']} ({f['target']}, {f['preset']})"
            for f in findings
            if f["category"] == "improves"
        }
    )
    return {
        "reference": reference,
        "findings": findings,
        "summary": (
            f"Measurable improvement over {reference} for: {'; '.join(improving)}. "
            "Other profile/target/preset combinations show none."
            if improving
            else f"No feature profile shows a measurable improvement over {reference}."
        ),
    }


def read_evaluation(folder: Path, part: str) -> dict[str, Any]:
    if part not in PARTS:
        raise EvaluationError(f"part must be one of {PARTS}")
    path = folder / "evaluation" / f"{part}.json"
    if not path.is_file():
        raise FileNotFoundError(part)
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["PARTS", "EvaluationError", "evaluate_experiment", "read_evaluation"]
