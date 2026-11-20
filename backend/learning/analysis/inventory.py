"""Phase 1 of the training audit: inventory every saved study and test its signal.

Usage (from the repository root):
    .venv/Scripts/python.exe backend/learning/analysis/inventory.py \
        [--studies backend/data/studies] [--out audit_inventory.json]

Reads only the saved validation artifacts (``<Model>-<variant>-<seed>.json``).
The sealed test period is never loaded.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import (  # noqa: E402
    binom_sf,
    binom_two_sided,
    bootstrap_ci,
    brier,
    ece,
    histogram,
    log_loss,
    long_flat_pnl,
    macro_f1,
    permutation_null_f1,
    reliability,
)


def curve_verdict(curves: list[dict]) -> dict:
    tr = np.array([c["trainingLoss"] for c in curves])
    va = np.array([c["validationLoss"] for c in curves])
    best = int(np.argmin(va))
    ln2 = float(np.log(2))
    verdict = []
    if va.min() > ln2 - 0.002:
        verdict.append("val loss never clearly below ln2 (0.6931) -> no generalisable signal")
    if tr[-1] < tr[0] - 0.01 and va[-1] >= va[best] + 0.002:
        verdict.append("train keeps falling while val rises after best epoch -> overfitting")
    if abs(va[-1] - va[0]) < 0.005:
        verdict.append("val loss flat")
    return dict(
        min_val_epoch=best + 1,
        min_val_loss=float(va.min()),
        first_val=float(va[0]),
        last_val=float(va[-1]),
        first_train=float(tr[0]),
        last_train=float(tr[-1]),
        gap_last=float(va[-1] - tr[-1]),
        verdict="; ".join(verdict) or "no clear pattern",
    )


def analyse_run(path: Path, study: str, cfg: dict) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    p = np.array([c["p"] for c in d["cases"]], dtype=float)
    nxt = np.array([c["next"] for c in d["cases"]], dtype=float)
    y = (nxt > 0).astype(int)
    pred = (p >= 0.5).astype(int)
    n = len(y)
    k = int((pred == y).sum())
    up = float(y.mean())
    always_up_k = int(y.sum())

    f1 = macro_f1(pred, y)
    f1_ci = bootstrap_ci(macro_f1, pred, y)
    acc_ci = bootstrap_ci(lambda a, b: float(np.mean(a == b)), pred, y)
    f1_null95, f1_perm_p = permutation_null_f1(pred, y)
    # paired accuracy difference vs always-up
    diff = (pred == y).astype(float) - y.astype(float)
    diff_ci = bootstrap_ci(np.mean, diff)

    strategies = {s["name"]: s for s in d["strategies"]}
    traces = {}
    for pl in d.get("trace", {}).get("players", []):
        acts = collections.Counter(e["decision"]["action"] for e in pl["events"])
        stat = collections.Counter(
            (e["execution"] or {}).get("status") for e in pl["events"] if e["execution"]
        )
        traces[pl["name"]] = dict(
            orders=acts["submit_buy"] + acts["submit_sell"],
            rejected=stat.get("rejected", 0),
        )

    bd = d.get("bestDecisions", {})
    oracle = np.array(
        [1 if r["bestDecision"] == "long" else 0 for r in bd.get("records", [])]
    )
    # Proxy P&L (close-to-close) of different decision rules on the same period
    # The model's actual targets (newer runs use a default-long hold policy,
    # so they can differ from p >= 0.5).
    targets = np.array(
        [c["actions"][0]["target"] == "long" if c.get("actions") else p_ >= 0.5
         for c, p_ in zip(d["cases"], p, strict=True)],
        dtype=float,
    )
    proxy = dict(
        model=long_flat_pnl(targets, nxt),
        always_long=long_flat_pnl(np.ones(n), nxt),
        oracle=long_flat_pnl(oracle, nxt) if len(oracle) == n else None,
    )

    return dict(
        study=study,
        source=cfg.get("source"),
        model=d["model"],
        variant=d["variant"],
        seed=d["seed"],
        epochs=d["epochs"],
        n_train=d.get("trainCount"),
        n_val=n,
        val_period=f'{d["validationStart"]}..{d["validationEnd"]}',
        train_up=d["trainUp"],
        val_up=up,
        accuracy=k / n,
        accuracy_ci=acc_ci,
        always_up_accuracy=always_up_k / n,
        acc_minus_always_up_ci=diff_ci,
        p_binom_vs_half=binom_two_sided(k, n, 0.5),
        p_binom_better_than_base=binom_sf(k, n, max(up, 1 - up)),
        macro_f1=f1,
        macro_f1_ci=f1_ci,
        macro_f1_null95=f1_null95,
        macro_f1_perm_p=f1_perm_p,
        always_up_macro_f1=macro_f1(np.ones(n, int), y),
        predicted_long_share=float(pred.mean()),
        log_loss=log_loss(p, y),
        log_loss_constant=log_loss(np.full(n, up), y),
        brier=brier(p, y),
        brier_constant=brier(np.full(n, up), y),
        ece10=ece(p, y),
        p_mean=float(p.mean()),
        p_std=float(p.std()),
        p_within_0_05=float(np.mean(np.abs(p - 0.5) < 0.05)),
        histogram=histogram(p),
        reliability=reliability(p, y),
        overview_val_loss=min(c["validationLoss"] for c in d["curves"]),
        curves=curve_verdict(d["curves"]),
        strategies={
            k2: dict(gain=v["gain"], dd=v["dd"], **traces.get(k2, {}))
            for k2, v in strategies.items()
        },
        proxy_pnl=proxy,
        oracle_long_share=bd.get("validationLongShare"),
        oracle_agreement_with_label=bd.get("validationAgreementRate"),
        model_oracle_agreement=bd.get("modelOracleAgreement"),
        model_oracle_macro_f1=bd.get("modelOracleMacroF1"),
        oracle_cum_return=bd.get("oracleCumulativeReturn"),
        accounting=d.get("accounting"),
        protocol=d.get("protocol"),
        pooled=d.get("pooled"),
        target_long_share=float(targets.mean()),
        commit=d.get("commit"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--studies", type=Path, default=Path("backend/data/studies"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for study in sorted(p for p in args.studies.iterdir() if p.is_dir()):
        cfg_path = study / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        for run in sorted(study.glob("*-*-*.json")):
            if run.name in ("best_decisions.json",):
                continue
            rows.append(analyse_run(run, study.name, cfg))
    text = json.dumps(rows, indent=2, default=float)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
