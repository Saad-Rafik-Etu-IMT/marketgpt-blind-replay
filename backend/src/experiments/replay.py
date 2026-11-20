"""Replay a registered binary model through the existing broker and ledger."""

from pathlib import Path

import pandas as pd

from src.ai.baselines import BuyAndHoldPlayer, MovingAverageCrossoverPlayer
from src.ai.player import BinaryProbabilities, PlayerDecision
from src.core.risk_metrics import max_drawdown
from src.database import Base, build_engine, build_session_factory
from src.eval.execution import (
    DecisionWindow,
    decide_and_submit,
    load_benchmark_profile,
    resolve_outcome,
    start_benchmark_session,
)
from src.experiments.models import load_registered
from src.experiments.runner import Runner, _finite
from src.experiments.sources import load_frames
from src.features.canonical import to_canonical
from src.models.market import OHLCVBarORM
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import step_session
from src.services.study_artifacts import read_json, write_json
from src.targets import softmax

TARGET = "direction_binary_v1"


class RegisteredPlayer:
    name = "Registered MLP"

    def __init__(self, loaded):
        self.loaded = loaded
        self.name = (
            "Registered Transformer"
            if loaded.manifest.get("compatibility", "").startswith("transformer")
            else "Registered MLP"
        )
        self.row = None

    def decide(self, observation):
        # The adapter supplies exactly the current causal feature row, never
        # the next row, dates, asset names or future outcomes.
        probability = float(
            softmax(self.loaded.predict_raw_features(self.row)[TARGET])[0, 1]
        )
        return PlayerDecision(
            "long" if probability >= 0.5 else "flat",
            BinaryProbabilities(probability, 1 - probability),
        )


def replay_run(
    folder: Path, data_dir: Path, key: str, part: str, asset: str | None = None
):
    if part not in {"inner_validation", "validation"}:
        raise ValueError("Only development partitions can be replayed")
    runner = Runner(folder, data_dir)
    if key not in {r.key for r in runner.spec.runs()}:
        raise ValueError("Unknown run")
    report = read_json(folder / "runs" / f"{key}.json")
    if report.get("state") != "complete" or TARGET not in report["targets"]:
        raise ValueError("Replay needs a completed binary-direction model")
    frames = load_frames(runner.spec.sources, data_dir)
    runner.verify_data(frames)
    datasets = runner._datasets(frames, report["feature_profile"])  # noqa: SLF001
    ds = next(d for d in datasets if d.target.id == TARGET)
    if report["dataset_fingerprints"][TARGET] != ds.fingerprint:
        raise ValueError("Dataset changed since training")
    model = load_registered(folder / report["checkpoint"], expected_columns=ds.columns)
    asset = asset or runner.spec.primary_asset or next(iter(frames))
    if asset not in frames:
        raise ValueError("Selected asset is not in this campaign")
    p = ds.parts[part]
    rows = p.index[p.index["asset"] == asset].sort_values("bar")
    if rows.empty:
        raise ValueError("No decisions for selected asset and partition")
    frame = frames[asset]
    frame = to_canonical(
        frame.loc[frame.index <= pd.Timestamp(runner.split.validation_end)]
    )
    first, last = int(rows["bar"].iloc[0]), int(rows["bar"].iloc[-1])
    start = first - max(ds.lookback, 64)
    start = max(0, start)
    if first - start < 64 or last + 1 >= len(frame):
        raise ValueError("Insufficient warm-up or execution history")
    segment = frame.iloc[start : last + 2]
    anchor = float(segment.iloc[0]["close"] / segment.iloc[0]["adjusted_close"])
    market = [
        dict(
            date=d.date(),
            open=float(a.adj_open * anchor),
            high=float(a.adj_high * anchor),
            low=float(a.adj_low * anchor),
            close=float(a.adj_close * anchor),
            adjusted_close=float(a.adj_close * anchor),
            volume=int(a.volume),
        )
        for d, a in segment.iterrows()
    ]
    profile = load_benchmark_profile()
    window = DecisionWindow(first - start, last - start)
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    players = []
    try:
        with build_session_factory(engine)() as db:
            db.add_all(OHLCVBarORM(instrument="ETF", **bar) for bar in market)
            db.commit()
            for player in (
                RegisteredPlayer(model),
                MovingAverageCrossoverPlayer(),
                BuyAndHoldPlayer(),
            ):
                session = start_benchmark_session(
                    db, "ETF", market[0]["date"], market[-1]["date"], profile
                )
                sid = session.session_id
                session = step_session(db, sid, first - start)
                initial = current_snapshot(db, sid)
                events = []
                equity = [initial.equity]
                for ix, row in rows.iterrows():
                    decision_bar = int(row["bar"]) - start
                    if session.current_bar_index < decision_bar:
                        session = step_session(
                            db, sid, decision_bar - session.current_bar_index
                        )
                    if (
                        session.current_bar_index != decision_bar
                        or session.status != "active"
                    ):
                        break
                    if isinstance(player, RegisteredPlayer):
                        player.row = p.x_raw[[ix]]
                    decision = decide_and_submit(db, sid, player, window, 64, profile)
                    session = step_session(db, sid)
                    execution = resolve_outcome(db, decision)
                    snapshot = current_snapshot(db, sid)
                    events.append(
                        dict(
                            decision=decision.model_dump(mode="json"),
                            execution=execution.model_dump(mode="json")
                            if execution
                            else None,
                            portfolio=snapshot.model_dump(mode="json"),
                        )
                    )
                    equity.append(snapshot.equity)
                players.append(
                    dict(
                        name=player.name,
                        initial=initial.model_dump(mode="json"),
                        events=events,
                        equity=equity,
                        gain=equity[-1] / equity[0] - 1,
                        max_drawdown=max_drawdown(equity),
                        decisions=len(events),
                        state=session.status,
                    )
                )
    finally:
        engine.dispose()
    if part == "validation":
        runner.record_validation_read(f"replay:{key}:{asset}", ds.fingerprint)
    result = _finite(
        dict(
            run=key,
            asset=asset,
            part=part,
            profile=profile.model_dump(mode="json"),
            profile_hash=profile.sha256,
            dataset=ds.fingerprint,
            accounting="synthetic total-return index units",
            policy="fixed P(up) >= 0.5, long/flat; no fitting on evaluated rows",
            label="next-open to following-open; execution snapshot is next close",
            market=[{**bar, "date": bar["date"].isoformat()} for bar in market],
            first_decision=window.first_decision_index,
            players=players,
        )
    )
    dest = folder / "replays"
    dest.mkdir(exist_ok=True)
    write_json(dest / f"{key}__{part}__{asset}.json", result)
    return result
