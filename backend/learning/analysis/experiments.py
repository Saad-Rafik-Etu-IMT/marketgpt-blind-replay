"""Phase 2 of the training audit: controlled ablations of the deep-MLP protocol.

Headless and CPU-only (NumPy). Runs on a CSV or on a saved study folder
(``config.json`` names the source), and never reads past ``validation_end``
of ``backend/config/split.json``: the sealed test period is never loaded.

Usage (from the repository root):
    .venv/Scripts/python.exe backend/learning/analysis/experiments.py \
        --study backend/data/studies/<uuid> --out <dir> [--seeds 0,1,2,3,4] \
        [--epochs 8] [--sections all] [--device cpu]

Sections: equivalence, ablation, labels, overnight, synthetic, nulls.

The ``--device`` flag exists so the same command line works on the GPU server;
every model in this script is NumPy, so only ``cpu`` is accepted and ``cuda``
fails fast with a clear message rather than silently running on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mlp"))
sys.path.insert(0, str(BACKEND))

from deep_mlp import (  # noqa: E402
    _macro_f1,
    _weighted_cross_entropy,
    compute_best_decisions,
    deep_mlp_backward,
    deep_mlp_forward,
    extract_blind_features,
    fit_robust_scaler,
    init_deep_mlp,
    transform_robust,
)
from stats_utils import (  # noqa: E402
    block_bootstrap_indices,
    log_loss,
    long_flat_pnl,
    macro_f1,
)

from src.data.split import build_examples, load_split_config  # noqa: E402

LOOKBACK = 64


# --------------------------------------------------------------------------- data
def load_frame(csv: Path | None, synthetic: str | None, config) -> pd.DataFrame:
    if synthetic == "course":
        from export_comparison import synthetic_frame

        frame = synthetic_frame()
    elif synthetic == "long":
        frame = long_synthetic_frame()
    else:
        frame = pd.read_csv(csv, parse_dates=["date"], index_col="date")
    # Hard cut: nothing after validation_end is ever loaded into memory below.
    return frame.loc[frame.index <= pd.Timestamp(config.validation_end)].copy()


def long_synthetic_frame(seed: int = 41) -> pd.DataFrame:
    """Same generator as ``synthetic_frame`` but 1995-2019, so val is 2017-2019."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1995-01-02", "2019-12-31")
    returns = rng.normal(0, 0.008, len(dates)) + 0.002 * np.sin(np.arange(len(dates)) / 10)
    closes = 100 * np.exp(np.cumsum(returns))
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(
        dict(open=opens, close=closes, adjusted_close=closes, volume=1000,
             high=np.maximum(opens, closes), low=np.minimum(opens, closes)),
        index=dates,
    )


def prepare(frame: pd.DataFrame, config, label_horizon: int = 1):
    """Replicates export_comparison.export's indexing (values[t] = return into bar t+1)."""
    cfg = config.model_copy(update=dict(label_horizon=label_horizon))
    closes = frame.adjusted_close.to_numpy()
    values = np.diff(np.log(closes))
    splits = build_examples(list(frame.index[1:].date), cfg)
    val = tuple(t for t in splits.validation if frame.index[t - 63].date() >= cfg.validation_start)
    return values, tuple(splits.train), val


def windows(values: np.ndarray, idx) -> np.ndarray:
    return np.stack([values[t - LOOKBACK + 1 : t + 1] for t in idx])


def label_next(values, idx, h=1):
    return np.array([values[t + 1 : t + 1 + h].sum() for t in idx])


def inner_split(train_idx, frac=0.2, purge=LOOKBACK + 6):
    """Chronological inner split of TRAIN: last ``frac`` is inner-val, with a purge gap."""
    n = len(train_idx)
    cut = int(n * (1 - frac))
    return tuple(train_idx[: cut - purge]), tuple(train_idx[cut:])


# --------------------------------------------------------------------------- metrics
def auc(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(p, y, nxt, threshold=0.5, boot=False, block=10):
    pred = (np.asarray(p) >= threshold).astype(int)
    out = dict(
        accuracy=float(np.mean(pred == y)),
        macro_f1=macro_f1(pred, y),
        auc=auc(p, y),
        log_loss=log_loss(p, y),
        log_loss_const=log_loss(np.full(len(y), y.mean()), y),
        long_share=float(pred.mean()),
        p_std=float(np.std(p)),
        pnl=long_flat_pnl(pred, nxt),
    )
    if boot:
        idx = block_bootstrap_indices(len(y), 1000, block, 0)
        aucs = [auc(np.asarray(p)[i], y[i]) for i in idx]
        out["auc_ci"] = [float(v) for v in np.nanquantile(aucs, [0.025, 0.975])]
    return out


# --------------------------------------------------------------------------- training
def calibrate_like_original(params, x_sel, y_sel, sel_oracle):
    """Exact copy of the post-training threshold/temperature step of train_deep_mlp."""
    _, cache = deep_mlp_forward(x_sel, params, training=False)
    diff = cache["logits"][:, 1] - cache["logits"][:, 0]
    oracle_labels = sel_oracle["labels"] if sel_oracle is not None else y_sel
    deltas = np.unique(np.r_[0.0, np.quantile(diff, np.linspace(0.15, 0.65, 51))])
    nxt = (
        np.array([r["nextReturn"] for r in sel_oracle["records"]])
        if sel_oracle is not None
        else (2.0 * y_sel - 1.0) * 0.005
    )
    bh, sd = float(np.sum(nxt)), float(np.std(nxt)) + 1e-8
    best, best_delta = -np.inf, 0.0
    for d in deltas:
        pb = (diff - d >= 0).astype(np.int64)
        if not (0.25 <= pb.mean() <= 0.78):
            continue
        sw = float(np.sum(np.abs(np.diff(np.r_[0, pb]))))
        edge = (float(np.sum(pb * nxt)) - sw * 0.0001 - bh) / (sd * math.sqrt(len(y_sel)))
        s = (0.5 * _macro_f1(pb, y_sel) + 0.25 * _macro_f1(pb, oracle_labels)
             + 0.15 * float(np.mean(pb == y_sel)) + 0.10 * np.clip(edge, -0.5, 0.5))
        if s > best:
            best, best_delta = s, float(d)
    sdiff = float(np.std(diff))
    temp = float(np.clip(sdiff / 0.55, 0.5, 3.0)) if sdiff > 1e-6 else 1.0
    out = {k: v.copy() for k, v in params.items()}
    out["W4"] = out["W4"] / temp
    out["b4"] = out["b4"] / temp
    out["b4"][1] -= 0.5 * (best_delta / temp)
    out["b4"][0] += 0.5 * (best_delta / temp)
    return out, dict(delta=best_delta, temperature=temp, logit_std=sdiff)


def train_variant(
    xt_raw, yt, xs_raw, ys, *, tr_oracle=None, sel_oracle=None, seed=0, epochs=8,
    select="composite", calibrate=True, class_balance=True, lr=1.5e-3,
    weight_decay=2e-3, dropout=0.25, batch_size=64, soft_targets=None,
):
    """Mirror of train_deep_mlp with switches. ``xs``/``ys`` is the SELECTION set.

    With default switches and xs = validation it is train_deep_mlp exactly
    (checked by the ``equivalence`` section).
    ``soft_targets`` optionally overrides the training targets (n, 2).
    """
    ft, fs = extract_blind_features(xt_raw), extract_blind_features(xs_raw)
    med, sc = fit_robust_scaler(ft)
    x_tr, x_s = transform_robust(ft, med, sc), transform_robust(fs, med, sc)
    n, n_in = x_tr.shape
    params = init_deep_mlp(n_in, seed=seed)
    rng = np.random.default_rng(seed)
    c0, c1 = max(1, int(np.sum(yt == 0))), max(1, int(np.sum(yt == 1)))
    w_cls = np.array([n / (2.0 * c0), n / (2.0 * c1)]) if class_balance else np.ones(2)
    y1h = np.eye(2)[yt]
    if soft_targets is not None:
        soft, base = soft_targets, w_cls[yt]
    elif tr_oracle is not None:
        soft = 0.72 * y1h + 0.28 * np.eye(2)[tr_oracle["labels"]]
        base = tr_oracle["weights"] * w_cls[yt]
    else:
        soft, base = y1h, w_cls[yt]
    base = base / np.mean(base)
    ys1h = np.eye(2)[ys]
    soft_s = 0.75 * ys1h + 0.25 * np.eye(2)[sel_oracle["labels"]] if sel_oracle is not None else ys1h

    keys = [k for k in params if not k.startswith(("rm", "rv"))]
    m = {k: np.zeros_like(params[k]) for k in keys}
    v = {k: np.zeros_like(params[k]) for k in keys}
    step = 0
    hist, best_score, best_vl, best_p, best_epoch = [], -np.inf, np.inf, None, 0
    for epoch in range(1, epochs + 1):
        lr_t = lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs))))
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = order[s : s + batch_size]
            if len(b) < 2:
                continue
            step += 1
            _, cache = deep_mlp_forward(x_tr[b], params, training=True, dropout=dropout, rng=rng)
            g = deep_mlp_backward(cache, soft[b], base[b], params)
            bc1, bc2 = 1 - 0.9**step, 1 - 0.999**step
            for k in keys:
                gk = np.clip(g[k], -2.0, 2.0)
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk * gk
                if k.startswith("W"):
                    params[k] *= 1.0 - lr_t * weight_decay
                params[k] -= lr_t * (m[k] / bc1) / (np.sqrt(v[k] / bc2) + 1e-8)
        tp, _ = deep_mlp_forward(x_tr, params, training=False)
        sp, _ = deep_mlp_forward(x_s, params, training=False)
        tl, sl = _weighted_cross_entropy(tp, y1h), _weighted_cross_entropy(sp, ys1h)
        sol = _weighted_cross_entropy(sp, soft_s)
        f1 = _macro_f1(sp.argmax(1), ys)
        hist.append(dict(epoch=epoch, train_loss=tl, sel_loss=sl, sel_f1=f1))
        if select == "composite":
            score = f1 - 0.35 * sol
            take = score > best_score or (best_p is None and sl < best_vl)
            if take:
                best_score = score
                best_vl = min(best_vl, sl)
        elif select == "loss":
            take = sl < best_vl
            if take:
                best_vl = sl
        else:  # "last"
            take = True
        if take:
            best_p, best_epoch = {k: val.copy() for k, val in params.items()}, epoch
    info = dict(selected_epoch=best_epoch, history=hist)
    if calibrate:
        best_p, cal = calibrate_like_original(best_p, x_s, ys, sel_oracle)
        info.update(cal)
    best_p["scaler_median"], best_p["scaler_scale"] = med, sc
    return best_p, info


def predict(params, x_raw):
    f = transform_robust(extract_blind_features(x_raw), params["scaler_median"], params["scaler_scale"])
    return deep_mlp_forward(f, params, training=False)[0][:, 1]


def logistic(xt_raw, yt, xs_raw, ys, l2_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0), iters=400):
    """L2 logistic regression on the same 145 scaled features; L2 picked on the selection set."""
    ft, fs = extract_blind_features(xt_raw), extract_blind_features(xs_raw)
    med, sc = fit_robust_scaler(ft)
    xt, xs = transform_robust(ft, med, sc), transform_robust(fs, med, sc)
    best = None
    for l2 in l2_grid:
        w, b = np.zeros(xt.shape[1]), 0.0
        mw, vw, mb, vb = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
        for it in range(1, iters + 1):
            p = 1 / (1 + np.exp(-(xt @ w + b)))
            gw = xt.T @ (p - yt) / len(yt) + l2 * w / len(yt)
            gb = float(np.mean(p - yt))
            mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw * gw
            mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
            w -= 0.01 * (mw / (1 - 0.9**it)) / (np.sqrt(vw / (1 - 0.999**it)) + 1e-8)
            b -= 0.01 * (mb / (1 - 0.9**it)) / (math.sqrt(vb / (1 - 0.999**it)) + 1e-8)
        ps = 1 / (1 + np.exp(-(xs @ w + b)))
        ll = log_loss(ps, ys)
        if best is None or ll < best[0]:
            best = (ll, l2, w.copy(), b)
    _, l2, w, b = best

    def f(x_raw):
        z = transform_robust(extract_blind_features(x_raw), med, sc)
        return 1 / (1 + np.exp(-(z @ w + b)))

    return f, l2


# --------------------------------------------------------------------------- sections
def section_equivalence(values, tr, val, seed, epochs):
    """Reconstruct the LEGACY pipeline's hidden selection info.

    ``train_variant`` with default switches mirrors the pre-audit
    ``train_deep_mlp`` (checked bit-exact, max |dp| ~ 1e-12, before that
    function was replaced on 2026-11-20).
    """
    xt, xv = windows(values, tr), windows(values, val)
    yt, yv = (label_next(values, tr) > 0).astype(int), (label_next(values, val) > 0).astype(int)
    tro = compute_best_decisions(values, tr, horizon=3)
    vo = compute_best_decisions(values, val, horizon=3)
    mine, info = train_variant(xt, yt, xv, yv, tr_oracle=tro, sel_oracle=vo, seed=seed, epochs=epochs)
    b = predict(mine, xv)
    return dict(
        seed=seed,
        selected_epoch=info["selected_epoch"],
        min_val_loss_epoch=int(np.argmin([h["sel_loss"] for h in info["history"]]) + 1),
        delta=info["delta"],
        temperature=info["temperature"],
        logit_std_before=info["logit_std"],
        val_loss_selected_epoch_uncalibrated=info["history"][info["selected_epoch"] - 1]["sel_loss"],
        val_log_loss_reported_model=log_loss(b, yv),
    )


def section_ablation(values, tr, val, seeds, epochs):
    xt, xv = windows(values, tr), windows(values, val)
    nxt_v = label_next(values, val)
    yt, yv = (label_next(values, tr) > 0).astype(int), (nxt_v > 0).astype(int)
    itr, ival = inner_split(tr)
    xit, xiv = windows(values, itr), windows(values, ival)
    yit, yiv = (label_next(values, itr) > 0).astype(int), (label_next(values, ival) > 0).astype(int)
    tro, vo = compute_best_decisions(values, tr, horizon=3), compute_best_decisions(values, val, horizon=3)
    itro, ivo = compute_best_decisions(values, itr, horizon=3), compute_best_decisions(values, ival, horizon=3)

    configs = {
        # name: (uses val for selection?, kwargs)
        "A_original(val-select+val-calib+oracle)": ("val", dict(select="composite", calibrate=True, oracle=True)),
        "B_original_no_oracle": ("val", dict(select="composite", calibrate=True, oracle=False)),
        "C_original_no_calibration": ("val", dict(select="composite", calibrate=False, oracle=True)),
        "D_val_loss_ES_no_cal_no_oracle": ("val", dict(select="loss", calibrate=False, oracle=False)),
        "E_inner_ES_no_cal_oracle": ("inner", dict(select="loss", calibrate=False, oracle=True)),
        "F_inner_ES_no_cal_no_oracle": ("inner", dict(select="loss", calibrate=False, oracle=False)),
        "G_inner_ES_no_cal_no_oracle_no_balance": ("inner", dict(select="loss", calibrate=False, oracle=False, class_balance=False)),
        "H_inner_last_epoch30_no_oracle": ("inner", dict(select="last", calibrate=False, oracle=False, epochs=30)),
    }
    results = {}
    for name, (sel, kw) in configs.items():
        kw = dict(kw)
        use_oracle = kw.pop("oracle")
        ep = kw.pop("epochs", epochs)
        rows = []
        for seed in seeds:
            t0 = time.time()
            if sel == "val":
                params, info = train_variant(xt, yt, xv, yv, tr_oracle=tro if use_oracle else None,
                                             sel_oracle=vo if use_oracle else None, seed=seed, epochs=ep, **kw)
            else:
                params, info = train_variant(xit, yit, xiv, yiv, tr_oracle=itro if use_oracle else None,
                                             sel_oracle=ivo if use_oracle else None, seed=seed, epochs=ep, **kw)
            p = predict(params, xv)
            r = evaluate(p, yv, nxt_v)
            r.update(seed=seed, selected_epoch=info["selected_epoch"], secs=round(time.time() - t0, 1),
                     curve=[(h["train_loss"], h["sel_loss"]) for h in info["history"]])
            rows.append(r)
        results[name] = dict(runs=rows, summary=summarise(rows))
        print(name, json.dumps(results[name]["summary"]), flush=True)

    # Non-neural references on the same validation decisions
    lr_f, l2 = logistic(xit, yit, xiv, yiv)
    results["L_logistic_inner_selected"] = dict(l2=l2, **evaluate(lr_f(xv), yv, nxt_v, boot=True))
    results["Z_always_long"] = evaluate(np.full(len(yv), 0.99), yv, nxt_v)
    results["Z_constant_base_rate_from_train"] = dict(p=float(yt.mean()), log_loss=log_loss(np.full(len(yv), yt.mean()), yv))
    mom = (xv[:, -20:].sum(1) > 0).astype(float)
    results["Z_momentum20_sign"] = evaluate(0.25 + 0.5 * mom, yv, nxt_v)
    print("logistic", json.dumps({k: v for k, v in results["L_logistic_inner_selected"].items() if k != "pnl"}))
    return results


def summarise(rows):
    keys = ["accuracy", "macro_f1", "auc", "log_loss", "long_share", "p_std", "selected_epoch"]
    s = {k: [round(float(np.mean([r[k] for r in rows])), 4), round(float(np.std([r[k] for r in rows])), 4)] for k in keys}
    s["pnl_gain"] = [round(float(np.mean([r["pnl"]["gain"] for r in rows])), 4), round(float(np.std([r["pnl"]["gain"] for r in rows])), 4)]
    s["switches"] = float(np.mean([r["pnl"]["switches"] for r in rows]))
    s["log_loss_const"] = rows[0]["log_loss_const"]
    return s


def section_labels(values, config, frame, seeds, epochs):
    """Is any label definition more learnable than next-day sign? Inner-val protocol only."""
    out = {}
    specs = [("next_day_sign", 1, None), ("5d_sign", 5, None), ("20d_sign", 20, None),
             ("5d_vol_adj_0.5sigma(drop_small)", 5, 0.5)]
    for name, h, k_sigma in specs:
        vals, tr, val = prepare(frame, config, label_horizon=h)
        fwd_t, fwd_v = label_next(vals, tr, h), label_next(vals, val, h)
        keep_t = np.ones(len(tr), bool)
        if k_sigma is not None:
            sig_t = np.array([vals[t - 19 : t + 1].std() for t in tr]) * math.sqrt(h)
            keep_t = np.abs(fwd_t) > k_sigma * sig_t
        tr_k = tuple(np.asarray(tr)[keep_t])
        itr, ival = inner_split(tr_k, purge=LOOKBACK + 5 + h)
        xit, xiv, xv = windows(vals, itr), windows(vals, ival), windows(vals, val)
        yit = (label_next(vals, itr, h) > 0).astype(int)
        yiv = (label_next(vals, ival, h) > 0).astype(int)
        yv = (fwd_v > 0).astype(int)
        nxt1 = label_next(vals, val, 1)
        lr_f, l2 = logistic(xit, yit, xiv, yiv)
        res = dict(n_train=len(tr_k), n_val=len(val), val_up=float(yv.mean()), train_up=float(yit.mean()),
                   logistic=evaluate(lr_f(xv), yv, nxt1, boot=True, block=max(10, 2 * h)))
        mlp = []
        for seed in seeds:
            params, info = train_variant(xit, yit, xiv, yiv, seed=seed, epochs=epochs, select="loss", calibrate=False)
            mlp.append(evaluate(predict(params, xv), yv, nxt1) | dict(selected_epoch=info["selected_epoch"]))
        res["mlp"] = summarise(mlp)
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "logistic"}),
              "logistic auc", res["logistic"]["auc"], res["logistic"].get("auc_ci"), flush=True)
    return out


def section_overnight(frame, values, val):
    """Does the close-to-close label match what the replay actually earns?

    Replay rule: decision at close of bar d executes at the open of d+1.
    values[t] is the return into bar t+1, so the decision for example t is at
    bar t+1 and the label is values[t+1] = log(C[t+2]/C[t+1]).
    The position from that decision is held from O[t+2] to the next switch;
    for a one-bar hold the realised return is log(O[t+3]/O[t+2]).
    Uses total-return-adjusted OHLC exactly like the research replay.
    """
    adj = frame.adjusted_close.to_numpy() / frame.close.to_numpy()
    o, c = frame.open.to_numpy() * adj, frame.close.to_numpy() * adj
    t = np.array([i for i in val if i + 3 < len(o)])
    label = np.log(c[t + 2] / c[t + 1])
    exec_ret = np.log(o[t + 3] / o[t + 2])
    overnight = np.log(o[t + 2] / c[t + 1])
    intraday = np.log(c[t + 2] / o[t + 2])
    return dict(
        n=len(t),
        sign_agreement_label_vs_executed=float(np.mean((label > 0) == (exec_ret > 0))),
        corr_label_vs_executed=float(np.corrcoef(label, exec_ret)[0, 1]),
        overnight_share_of_variance=float(np.var(overnight) / np.var(label)),
        mean_overnight_bp=float(np.mean(overnight) * 1e4),
        mean_intraday_bp=float(np.mean(intraday) * 1e4),
        overnight_share_of_total_drift=float(np.sum(overnight) / np.sum(label)),
    )


def section_synthetic(config, seeds, epochs):
    """Theoretical ceilings for the sine+noise generator and what the MLP reaches."""
    out = {}
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    for kind in ("course", "long"):
        frame = load_frame(None, kind, config)
        values, tr, val = prepare(frame, config)
        n_all = len(values)
        drift = 0.002 * np.sin(np.arange(n_all + 1) / 10)
        yv = (label_next(values, val) > 0).astype(int)
        # Bayes: knows the deterministic drift of the NEXT return (index t+1)
        bayes_pred = (drift[np.asarray(val) + 1] > 0).astype(int)
        bayes_expected = float(np.mean([max(phi(d / 0.008), 1 - phi(d / 0.008)) for d in drift[np.asarray(val) + 1]]))
        # Causal matched filter: least-squares sin/cos fit on the 64-return window, extrapolated one step
        k = np.arange(LOOKBACK)
        basis = np.stack([np.sin(k / 10), np.cos(k / 10), np.ones(LOOKBACK)], 1)
        nxt_basis = np.array([np.sin(LOOKBACK / 10), np.cos(LOOKBACK / 10), 1.0])
        xv = windows(values, val)
        coef, *_ = np.linalg.lstsq(basis, xv.T, rcond=None)
        mf_pred = ((nxt_basis @ coef) > 0).astype(int)
        xt = windows(values, tr)
        yt = (label_next(values, tr) > 0).astype(int)
        itr, ival = inner_split(tr)
        mlp = []
        for seed in seeds:
            params, info = train_variant(windows(values, itr), (label_next(values, itr) > 0).astype(int),
                                         windows(values, ival), (label_next(values, ival) > 0).astype(int),
                                         seed=seed, epochs=epochs, select="loss", calibrate=False)
            p = predict(params, xv)
            mlp.append(evaluate(p, yv, label_next(values, val)) | dict(selected_epoch=info["selected_epoch"]))
        n = len(yv)
        out[kind] = dict(
            n_train=len(tr), n_val=n,
            accuracy_se_at_ceiling=float(math.sqrt(bayes_expected * (1 - bayes_expected) / n)),
            bayes_expected_accuracy=bayes_expected,
            bayes_realised_accuracy=float(np.mean(bayes_pred == yv)),
            matched_filter_accuracy=float(np.mean(mf_pred == yv)),
            matched_filter_agreement_with_bayes=float(np.mean(mf_pred == bayes_pred)),
            mlp_inner_protocol=summarise(mlp),
            original_pipeline_seed0=evaluate(predict(train_variant(
                xt, yt, xv, yv, tr_oracle=compute_best_decisions(values, tr), sel_oracle=compute_best_decisions(values, val),
                seed=0, epochs=epochs)[0], xv), yv, label_next(values, val)),
        )
        print(kind, json.dumps({k: v for k, v in out[kind].items() if k not in ("mlp_inner_protocol", "original_pipeline_seed0")}),
              "mlp", json.dumps(out[kind]["mlp_inner_protocol"]), flush=True)
    return out


def section_nulls(studies: Path):
    """Is the saved runs' P&L better than random timing with the same exposure?

    Null: circularly shift the model's own position series (keeps exposure and
    switching structure, breaks alignment with returns).
    """
    out = {}
    for run in sorted(studies.glob("*/MLP-*.json")):
        d = json.loads(run.read_text(encoding="utf-8"))
        p = np.array([c["p"] for c in d["cases"]])
        nxt = np.array([c["next"] for c in d["cases"]])
        pos = (p >= 0.5).astype(float)
        obs = long_flat_pnl(pos, nxt)["gain"]
        null = np.array([long_flat_pnl(np.roll(pos, s), nxt)["gain"] for s in range(1, len(pos))])
        out[run.parent.name[:8] + "/" + run.stem] = dict(
            observed_proxy_gain=obs, null_mean=float(null.mean()), null_95=float(np.quantile(null, 0.95)),
            p_value=float((np.sum(null >= obs) + 1) / (len(null) + 1)), exposure=float(pos.mean()),
        )
    print("nulls", json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--study", type=Path, help="study folder; its config.json names the CSV")
    src.add_argument("--csv", type=Path)
    ap.add_argument("--studies", type=Path, default=BACKEND / "data" / "studies")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sections", default="all")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    if a.device != "cpu":
        ap.error("all models in this script are NumPy (CPU-only); use --device cpu")
    seeds = [int(s) for s in a.seeds.split(",")]
    config = load_split_config()
    csv = a.csv
    if a.study is not None:
        spec = json.loads((a.study / "config.json").read_text())
        csv = Path(spec["csv"]) if spec.get("source") != "synthetic" else None
    sections = set(a.sections.split(",")) if a.sections != "all" else {
        "equivalence", "ablation", "labels", "overnight", "synthetic", "nulls"}
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    if csv is not None and sections & {"equivalence", "ablation", "labels", "overnight"}:
        frame = load_frame(csv, None, config)
        values, tr, val = prepare(frame, config)
        print(f"data: {csv.name} train={len(tr)} val={len(val)} last_date={frame.index[-1].date()}", flush=True)
        if "equivalence" in sections:
            results["equivalence"] = [section_equivalence(values, tr, val, s, a.epochs) for s in seeds]
            print("equivalence", json.dumps(results["equivalence"], indent=1), flush=True)
        if "overnight" in sections:
            results["overnight"] = section_overnight(frame, values, val)
            print("overnight", json.dumps(results["overnight"]), flush=True)
        if "ablation" in sections:
            results["ablation"] = section_ablation(values, tr, val, seeds, a.epochs)
        if "labels" in sections:
            results["labels"] = section_labels(values, config, frame, seeds, a.epochs)
    if "synthetic" in sections:
        results["synthetic"] = section_synthetic(config, seeds, a.epochs)
    if "nulls" in sections:
        results["nulls"] = section_nulls(a.studies)
    (a.out / "experiments.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
