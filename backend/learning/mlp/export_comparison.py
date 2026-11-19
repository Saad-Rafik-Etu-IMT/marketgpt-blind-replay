"""Export the learning MLP and two baselines on a shared validation replay."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from step8_train import fit_normalizer, make_xy, mlp_forward, normalize, train

from src.ai.baselines import BuyAndHoldPlayer, MovingAverageCrossoverPlayer
from src.ai.player import BinaryProbabilities, PlayerDecision
from src.ai.tokenizer import NumericReturnTokenizer
from src.core.risk_metrics import max_drawdown
from src.data.split import build_examples, load_split_config
from src.database import Base, build_engine, build_session_factory
from src.eval.execution import (
    DecisionWindow,
    decide_and_submit,
    load_benchmark_profile,
    resolve_outcome,
    start_benchmark_session,
)
from src.models.market import OHLCVBarORM
from src.schemas.market import OHLCVBar
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import step_session


class LearningPlayer:
    name = "MLP"

    def __init__(self, params, mean, std):
        self.params, self.mean, self.std = params, mean, std

    def decide(self, observation):
        x = normalize(np.array([observation.log_returns]), self.mean, self.std)
        p = float(mlp_forward(x, self.params)[0][0, 1])
        return PlayerDecision("long" if p >= 0.5 else "flat", BinaryProbabilities(p, 1 - p))


def synthetic_frame():
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2015-01-01", "2017-12-31")
    returns = rng.normal(0, 0.008, len(dates)) + 0.002 * np.sin(np.arange(len(dates)) / 10)
    closes = 100 * np.exp(np.cumsum(returns))
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(dict(open=opens, close=closes, adjusted_close=closes, volume=1000,
                             high=np.maximum(opens, closes), low=np.minimum(opens, closes)), index=dates)


def export(frame, origin, seed, checkpoint):
    config = load_split_config()
    frame = frame.loc[frame.index <= pd.Timestamp(config.validation_end)].copy()
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Expected unique dates in chronological order")
    bars = [OHLCVBar(date=d.date(), **row) for d, row in frame.to_dict("index").items()]
    closes = frame.adjusted_close.to_numpy()
    if not np.isfinite(closes).all() or min(closes) <= 0:
        raise ValueError("Adjusted closes must be positive and finite")
    values = np.diff(np.log(closes))
    splits = build_examples(list(frame.index[1:].date), config)
    # A 64-return window needs 65 closes inside the partition.
    val = tuple(t for t in splits.validation if frame.index[t - 63].date() >= config.validation_start)
    if not splits.train or not val:
        raise ValueError("Insufficient train or validation history")
    xt, yt = make_xy(values, splits.train, 64)
    xv, yv = make_xy(values, val, 64)
    mean, std = fit_normalizer(xt)
    params, history = train(normalize(xt, mean, std), yt, normalize(xv, mean, std), yv, seed=seed)
    np.savez(checkpoint, **params, mean=mean, std=std)
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    profile = load_benchmark_profile()
    segment = bars[val[0] - 63:val[-1] + 3]
    window = DecisionWindow(64, len(segment) - 2)
    strategies = []
    cases = []
    with build_session_factory(engine)() as db:
        db.add_all(OHLCVBarORM(instrument="ETF", **bar.model_dump()) for bar in segment)
        db.commit()
        for player in (LearningPlayer(params, mean, std), MovingAverageCrossoverPlayer(), BuyAndHoldPlayer()):
            sid = start_benchmark_session(db, "ETF", segment[0].date, segment[-1].date, profile).session_id
            step_session(db, sid, 64)
            equity = [profile.initial_cash]
            for i, t in enumerate(val):
                decision = decide_and_submit(db, sid, player, window, 64, profile)
                step_session(db, sid)
                result = resolve_outcome(db, decision)
                equity.append(current_snapshot(db, sid).equity)
                if len(cases) == i:
                    cases.append(dict(p=decision.probabilities.long, next=float(values[t + 1]), actions=[]))
                cases[i]["actions"].append(dict(target=decision.target, action=decision.action,
                    status=result.status if result else decision.action,
                    price=result.fill_price if result else None,
                    reason=(result.rejection_reason if result else decision.pre_trade_rejection)))
            strategies.append(dict(name=player.name, gain=equity[-1] / equity[0] - 1, dd=max_drawdown(equity)))
    engine.dispose()
    probs = np.array([case["p"] for case in cases])
    predicted = probs >= 0.5
    diff = (predicted == yv).astype(float) - (yv == 1).astype(float)
    # Paired circular moving-block bootstrap over chronological validation decisions.
    rng = np.random.default_rng(seed)
    starts = rng.integers(len(diff), size=(1000, (len(diff) + 9) // 10))
    indices = ((starts[..., None] + np.arange(10)) % len(diff)).reshape(1000, -1)
    interval = np.quantile(diff[indices[:, :len(diff)]].mean(axis=1), [0.025, 0.975])
    f1 = []
    for cls in (0, 1):
        denominator = int((predicted == cls).sum() + (yv == cls).sum())
        f1.append(2 * ((predicted == cls) & (yv == cls)).sum() / denominator if denominator else 0)
    calibration = []
    for i in range(5):
        mask = np.minimum((probs * 5).astype(int), 4) == i
        if mask.any():
            calibration.append(dict(count=int(mask.sum()), predicted=float(probs[mask].mean()),
                                    observed=float(yv[mask].mean())))
    bins = NumericReturnTokenizer().fit(xt).bin_edges
    return dict(bins=list(bins), version=2, macroF1=float(np.mean(f1)), interval=interval.tolist(), calibration=calibration, split="validation", origin=origin, seed=seed,
                trainEnd=str(config.train_end), validationStart=str(segment[0].date),
                validationEnd=str(segment[-1].date), trainCount=len(xt),
                profile=profile.profile_id, dataset=hashlib.sha256(frame.to_csv().encode()).hexdigest(),
                checkpoint=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
                curves=[dict(step=h["epoch"], trainingLoss=h["train_loss"], validationLoss=h["val_loss"]) for h in history],
                returns=values[val[0]-63:val[-1]+1].tolist(), cases=cases, strategies=strategies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.suffix.lower() != ".json" or (args.csv and args.csv.resolve() == args.out.resolve()):
        parser.error("Choose a separate .json output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = synthetic_frame() if args.synthetic else pd.read_csv(args.csv, index_col="date", parse_dates=["date"])
    report = export(frame, "synthetic" if args.synthetic else "experimental", args.seed, args.out.with_suffix(".npz"))
    args.out.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
