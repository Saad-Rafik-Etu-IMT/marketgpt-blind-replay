# Training and evaluation audit — 2026-11-20

Scope: the deep MLP study pipeline (`learning/mlp/deep_mlp.py`, `export_comparison.py`,
`run_study.py`), the reference Transformer (static review only), the three saved
studies in `backend/data/studies/`, and `Projet_LLM.ipynb`. Every number below
comes from validation data (2017-04-03 → 2019-12-31) or earlier. **No code in
this audit loads the sealed test period** (frames are cut at `validation_end`
before any computation).

Reproduce (repository root, CPU, ~4 min total):

```bash
.venv/Scripts/python.exe backend/learning/analysis/inventory.py --out inv.json
.venv/Scripts/python.exe backend/learning/analysis/gradcheck.py
.venv/Scripts/python.exe backend/learning/analysis/experiments.py \
    --study backend/data/studies/dc5e5b9a-7f44-4b68-a96d-1cb073511670 \
    --out audit_out --seeds 0,1,2,3,4 --epochs 8 --device cpu
```

Raw outputs: [`backend/learning/analysis/results/`](../backend/learning/analysis/results).

## TL;DR

1. **There is no detectable next-day signal on SPY, and the pipeline is
   working.** Across 8 training protocols × 5 seeds plus logistic regression,
   the validation AUC is 0.50–0.52 and every model's log-loss is *worse* than
   predicting a constant base rate. The same code does learn when signal exists:
   on a longer synthetic sine series the MLP reaches AUC 0.55 and accuracy
   54.4%, against a Bayes ceiling of 57.4%. The gradient check passes.
2. **The P&L gap to buy-and-hold is explained by exposure and churn, not
   prediction.** Macro-F1 selection, balanced class weights and a val-tuned
   threshold push the model to be long 50–73% of the time in a period that
   was up 57% of days. The model then trades about 220 times over 627 days.
   Under the replay's next-open execution it also misses the overnight return,
   and overnight carried 109% of SPY's validation drift.
3. **The reported validation metrics are optimistic and partly mislabelled.**
   The checkpoint epoch, the decision threshold and the logit scale are all
   tuned on the same validation set that is then reported. The "temperature
   calibration" *sharpens* the logits (T ≈ 0.52–0.58), which raises log-loss.
   The UI marked the wrong epoch as retained; this is fixed below.
4. **Single-seed comparisons are noise.** The seed-to-seed standard deviation
   of the original pipeline's P&L is ±10.9 pp. With 627 decisions, the smallest
   accuracy edge detectable at 80% power is about 5.6 pp.
5. **The synthetic study cannot validate anything.** It has 130 validation
   samples, and on that set even the Bayes-optimal predictor scores 51.5%.

---

## Phase 1 — Inventory

Three saved studies. All are MLP, `reference` variant, 8 epochs, one seed each,
benchmark-v2. No Transformer run is saved: torch is not installed locally.

### Results table

| | 2af84096 | 4562eeff | dc5e5b9a |
|---|---|---|---|
| Source / seed / epochs | synthetic / 0 / 8 | SPY / 10 / 8 | SPY / 1 / 8 |
| Train / val samples | 457 / **130** | 5,961 / 627 | 5,961 / 627 |
| Label up-share train → val | 48.6% → 53.8% | 53.6% → 57.3% | 53.6% → 57.3% |
| Macro-F1 [95% block-bootstrap CI] | 0.535 [0.465, 0.597] | 0.509 [0.470, 0.548] | 0.512 [0.467, 0.555] |
| Macro-F1 permutation null 95th pct (same long share) / p | 0.567 / 0.19 | 0.529 / 0.30 | 0.516 / 0.10 |
| Accuracy [CI] | 55.4% [49.2, 61.5] | 51.2% [47.4, 55.0] | 55.7% [51.8, 59.5] |
| Always-up accuracy | 53.8% | 57.3% | 57.3% |
| Accuracy − always-up, paired CI | [−6.2, +10.0] pp | **[−11.6, −0.5] pp** | [−6.1, +2.7] pp |
| Binomial p vs 0.5 / vs base rate (one-sided) | 0.25 / 0.40 | 0.58 / 0.999 | 0.005 / 0.80 |
| Val loss shown in UI (min of curve) | 0.752 (ep 1) | 0.693 (ep 4) | 0.697 (ep 6) |
| **Epoch actually retained** | 1 | **6** | **1** |
| Log-loss of the evaluated model / constant | 0.710 / 0.690 | 0.721 / 0.683 | 0.708 / 0.683 |
| Brier / constant | 0.258 / 0.249 | 0.262 / 0.245 | 0.255 / 0.245 |
| ECE (10 bins) | 0.084 | 0.116 | 0.075 |
| Predicted long share | 66% | 50% | 73% |
| MLP gain / max DD | +8.3% / 4.9% | +30.0% / 16.7% | +30.7% / 15.5% |
| MA 20/50 gain / max DD | +5.6% / 5.1% | +30.8% / 10.1% | +30.8% / 10.1% |
| Buy & hold gain / max DD | +6.0% / 6.1% | **+39.1%** / 19.2% | **+39.1%** / 19.2% |
| MLP orders / fill rejections | n/a (no trace) | 230 / 5 | 217 / 9 |
| MA orders / rejections | — | 12 / 1 | 12 / 1 |
| Model–oracle agreement (macro-F1) | 50.0% (0.49) | 49.6% (0.49) | 56.9% (0.51) |
| Oracle vs next-day label agreement | 83.8% | 82.5% | 82.5% |

Source: `inventory.py` → `results/inventory.json`. The retained epoch comes from a
bit-exact re-run (`experiments.py --sections equivalence`; max |Δp| = 1e-12
against the saved predictions' pipeline).

### Training curves

| Run | Train loss (ep 1 → 8) | Val loss (ep 1 → 8) | Verdict |
|---|---|---|---|
| synthetic s0 | 0.653 → 0.555 | 0.752 → 0.796, rising from epoch 1 | **Overfits from the first epoch.** 457 samples for 34,818 parameters. The best val loss is above ln 2. |
| SPY s10 | 0.683 → 0.659 | 0.700 → 0.698, min 0.6929 at ep 4 | **Flat / memorising.** Train falls 0.024; val never goes clearly below ln 2 = 0.6931. Epoch 6 was retained, not epoch 4. |
| SPY s1 | 0.683 → 0.656 | 0.699 → 0.698, min 0.6968 at ep 6 | **Flat / memorising.** Val never below ln 2. **Epoch 1 was retained** by the composite score. |

With 30 epochs and no early stopping (ablation H), val log-loss reaches 0.746:
the model fits noise. None of the SPY runs underfit in the usual sense. There is
just nothing to fit beyond the base rate.

---

## Phase 2 — Audit findings

### 1. Signal vs noise — **confirmed: no signal**

- Validation n = 627 (SPY) and 130 (synthetic). For n = 627, the accuracy SE is
  about 2.0 pp and the minimum detectable edge at 80% power is about 5.6 pp.
  Daily index direction does not plausibly offer an edge that large.
- No run beats the always-up predictor. Seed 10 is **significantly worse** than
  always-up (paired CI [−11.6, −0.5] pp). Seed 1's "p = 0.005 vs 0.5" only shows
  that it is long-biased in an up market: against the base rate, p = 0.80.
- The macro-F1 permutation test (same predictions, shuffled labels) is not
  significant for any run (p = 0.10–0.30).
- In the controlled ablation (5 seeds each, table in §5), AUC is 0.505–0.512
  for every MLP protocol, and logistic regression reaches 0.516
  [0.474, 0.562]. Every model's log-loss (0.689–0.746) is worse than the
  validation base rate (0.683) and the train base rate (0.685).
- P&L timing null: circularly shifting each run's own position series, which
  keeps exposure and switching but breaks timing, gives p = 0.15, 0.15 and 0.20.
  The timing is not better than random.

### 2. Label definition — **confirmed misaligned with the replay objective**

`experiments.py --sections overnight` (SPY validation, total-return OHLC exactly
as the research replay builds it):

| Quantity | Value |
|---|---|
| Sign agreement: close-to-close label vs the return a decision actually earns (next open → following open) | **70.6%** |
| Correlation of the two | 0.67 |
| Overnight share of label variance | 30% |
| Mean overnight return / mean intraday return | **+5.7 bp / −0.5 bp** |
| Overnight share of total validation drift | **109%** |

A decision at close *d* executes at the open of *d*+1 (`docs/anti_lookahead.md`).
The label is `r(d→d+1)` close-to-close, so a perfect label predictor still gets
the sign of its realised trade wrong 29% of the time. Every flat→long switch also
forfeits the overnight gap, which is where all the 2017–2019 drift was. This is
why a high-churn model structurally lags buy-and-hold even at zero skill. It also
explains the 6.7 pp gap between seed 1's close-to-close proxy P&L (+37.4%) and
its broker replay (+30.7%). For buy-and-hold the proxy and replay agree:
39.0% vs 39.1%.

Further mismatch:

- The labels are binary and symmetric, while the P&L of long/flat is asymmetric.
  Being wrongly flat on an up day costs the drift.
- Training uses balanced class weights and selection uses macro-F1. Both reward
  predicting "down" about half the time, while the P&L-optimal no-information
  action in a 57%-up period is "always long".

Alternative labels were tested (inner-val protocol, logistic + MLP × 5 seeds):

| Label | Val up-share | Logistic AUC [CI] | MLP AUC (mean ± sd) |
|---|---|---|---|
| next-day sign | 57.3% | 0.516 [0.474, 0.562] | 0.511 ± 0.014 |
| 5-day sign | 65.3% | 0.452 [0.380, 0.533] | 0.455 ± 0.009 |
| 20-day sign | 70.4% | 0.448 [0.337, 0.576] | 0.458 ± 0.011 |
| 5-day, drop \|fwd\| < 0.5σ in train | 65.3% | 0.471 [0.400, 0.553] | 0.468 ± 0.009 |

Longer horizons are not more learnable from 64 returns. Their AUC sits *below*
0.5 (CIs include 0.5): what the model learned on 1993–2016 (mostly
mean-reversion after drops) inverts in the 2017–2019 low-volatility bull market.
That is regime shift, not a coding error.

### 3. Leakage and selection bias

| Check | Verdict | Evidence |
|---|---|---|
| Val used for early stopping / checkpoint selection, then reported | **Confirmed** | `deep_mlp.py:684-690`: composite `val_f1 − 0.35·val_oracle_loss` on val. Transformer: `transformer_reference.py:67-72`, min val loss. |
| Val used to tune the decision threshold and logit scale, then reported | **Confirmed (main optimism source)** | `deep_mlp.py:694-755`: 52 thresholds scored on val macro-F1, val *oracle* macro-F1, val accuracy and **val P&L vs buy-and-hold**; the long share is forced into [0.25, 0.78]. Ablation A vs C: this step alone moves P&L from +19.0% to +29.6% and long share from 54% to 66% *on the same data it was tuned on*. |
| Hindsight `val_oracle` (validation future returns, global Viterbi path) used for selection/calibration | **Confirmed** | Used in the composite score and in the threshold score. It is not a test leak, but it is in-sample tuning on val labels. |
| Scaler / normalizer / tokenizer fit on train only | **Ruled out (clean)** | `fit_robust_scaler(feat_train)` `deep_mlp.py:574`; `fit_normalizer(xt)` `export_comparison.py:132`; `NumericReturnTokenizer().fit(xt)` `transformer_reference.py:14`. |
| Features peek past *t* | **Ruled out** | `extract_blind_features` reads only the 64-return row. Oracle `local_vol` reads `values[t-13:t+1]`. |
| 64-window / split boundaries | **Ruled out** | `values[t]` is the return into bar *t*+1. `build_examples` gets `frame.index[1:]`, and the extra filter `frame.index[t-63] ≥ validation_start` keeps all 65 closes inside validation (`export_comparison.py:123-127`). The replay segment starts at bar `val[0]−63`, and the first decision is on bar `val[0]+1`. |
| Gap bars | **Minor bug, fixed** | Oracle `horizon=3` exceeds `label_horizon=1`, so the last 2 train oracle labels read 2017-01 gap bars. That is not validation, but it breaks the documented rule. Clamped (see Changes). |
| Test period | **Ruled out in this pipeline** | `export()` and `dataset_timeline()` cut the frame at `validation_end`. |
| **Notebook touched the sealed test period** | **Confirmed (process)** | `Projet_LLM.ipynb` cell 5: an 80/10/10 split of SPY 1993 → 2026, so its "test" set is about 2023–2026, inside the project's sealed test (≥ 2020-02-01). It was evaluated (cell 9: 3-class accuracy 38.6% vs 44% majority). The current design was ported from a notebook whose evaluation saw sealed-period data. Record this and do not use that notebook's numbers for any decision. |

### 4. Synthetic data — **confirmed: unusable as a sanity check as configured**

`synthetic_frame()`: `r_t = N(0, 0.008) + 0.002·sin(t/10)`, so the per-bar SNR is
0.25 and the period is about 63 bars.

| | course synthetic (2015–2017) | same generator, 1995–2019 |
|---|---|---|
| Train / val samples | 457 / 130 | 5,675 / 652 |
| Bayes ceiling (knows the phase), expected / realised | 56.3% / **51.5%** | 56.3% / 57.4% |
| Causal matched filter on the 64-return window | 50.8% | 53.4% |
| Accuracy SE at the ceiling | 4.4 pp | 1.9 pp |
| MLP (inner-val protocol, 5 seeds) accuracy / AUC | 49.4% / 0.497 | **54.4% ± 1.4 / 0.553** |

With 130 validation points, the *Bayes-optimal* rule cannot be distinguished
from a coin. On the longer series, the MLP learns the sine and slightly beats a
hand-built matched filter. So the optimiser, backprop and features are fine; the
SPY problem is the data.

### 5. Oracle targets and the full ablation (SPY, 5 seeds, 8 epochs, mean ± sd)

| Config | Selection set | AUC | Log-loss | Long share | Proxy P&L | Switches |
|---|---|---|---|---|---|---|
| A original (val select + val calibrate + oracle) | val | 0.505 ± .018 | 0.714 | 66% | +29.6% ± 10.9 | 240 |
| B A without oracle | val | 0.505 ± .010 | 0.719 | 59% | +18.5% ± 5.2 | 282 |
| C A without calibration | val | 0.505 ± .018 | 0.699 | 54% | +19.0% ± 10.3 | 247 |
| D plain early stopping on val loss | val | 0.507 ± .012 | 0.699 | 52% | +9.8% ± 4.7 | 281 |
| E inner-val ES, oracle | inner | 0.507 ± .009 | 0.695 | 63% | +21.5% ± 8.9 | 233 |
| F inner-val ES, no oracle | inner | 0.511 ± .014 | 0.695 | 62% | +18.7% ± 10.5 | 238 |
| G F without class balancing | inner | 0.512 ± .013 | **0.692** | 72% | +20.5% ± 5.0 | 205 |
| H no ES, 30 epochs | inner | 0.507 ± .018 | 0.746 | 56% | +15.1% ± 9.4 | 282 |
| Logistic regression (L2 picked on inner-val) | inner | 0.516 | 0.689 | 71% | — | — |
| Always long | — | — | — | 100% | **+39.0%** | 1 |

(Proxy P&L is close-to-close with 0.5 bp per switch; the broker replay is lower
for high-churn models, see §2.)

- **Oracle soft targets and weights: no effect on discrimination** (A vs B, E vs
  F, all within noise). The 0.72/0.28 blend changes only the 17–18% of samples
  where the oracle disagrees with the label, which amounts to label smoothing
  plus up-weighting large-move days. It raises P&L only by raising exposure.
  The oracle itself is not a meaningful target: +463% with 0.7% max DD over
  2017–2019, because the Viterbi path depends on the whole future. Verdict:
  **neutral for skill. Remove it, or keep it only as a reporting reference.**
- Across all configs, P&L tracks long share, not AUC. That is the signature of a
  zero-skill model in a trending market.

### 6. Decision rule, calibration and probability collapse — **confirmed**

- **Pre-calibration probabilities collapse to about 0.5:** p std is about 0.06
  (configs C–G). The logit std before the "temperature" step is 0.29–0.32.
- **The "temperature calibration" is a bug.** `T = clip(std(logit diff)/0.55)`
  (`deep_mlp.py:748-750`) forces the logit spread to 0.55 whatever the model
  knows. On SPY it gives T = 0.52–0.58, i.e. it *multiplies* logits by about 1.8.
  The result is p ranging up to 0.84 when the observed frequencies are 0.46–0.57
  (reliability, run dc5e5b9a: the 0.8–0.9 bin predicts 0.83 and observes 0.46,
  the 0.3–0.4 bin predicts 0.37 and observes 0.68). Log-loss rises from 0.699
  (C) to 0.714 (A). Real temperature scaling fits T by minimising NLL on a
  held-out set.
- The fixed 0.5 threshold after a val-tuned bias shift ignores costs. With 1 bp
  spread the switching cost is small in the proxy, but in the replay each switch
  also changes overnight exposure (§2) and risks a fill-time rejection (5–9 per
  MLP run vs 1 for MA; each rejection is a missed entry).

### 7. Capacity and optimisation

| Item | Verdict | Evidence |
|---|---|---|
| `deep_mlp_backward` correctness | **Ruled out: correct** | `gradcheck.py`: worst relative error ≤ 1.1e-6 on every tensor, with and without dropout. `b1/b2/b3` have an exact gradient of 0 (pre-BatchNorm biases are redundant; harmless). Now a regression test. |
| 145 features / 34,818 params vs 5,961 samples | **Contributes to overfitting, not the root cause** | Train loss falls while val is flat; logistic regression with 146 params does as well or better (AUC 0.516, LL 0.689). Capacity is unused because there is no signal. |
| Default 8 epochs | **Fine** (the inner-val ES picks epochs 3–4; 30 epochs overfits). |
| Single seed | **Confirmed problem** | Seed sd of P&L ±5–11 pp, of accuracy ±1–2 pp; seeds 1 and 10 differ by 4.5 pp accuracy on identical data. |
| Composite selection picks epoch 1 | **Confirmed** | Seed 1 retained epoch 1 of 8 (min val loss was epoch 6): selection noise. |
| Transformer (static) | **Unclear, not run** | d = 16, 1 layer, 2 heads, ff 32, lr 2e-3, AdamW wd 0.01, cosine to 0, no warmup, no grad clipping, mean pooling, early stopping on the val loss it reports. Sensible for the size; it will meet the same no-signal wall. No `--device` option. |

### 8. Evaluation fairness

- **Fair on the mechanics** (confirmed): the three players run on the same
  `segment`, the same benchmark-v2 profile, the same decision indices and
  independent accounts (`export_comparison.py:175-215`). Buy-and-hold replay vs
  proxy agree to 0.1 pp.
- **Not fair on information use:** the MLP's threshold and checkpoint are tuned
  on the validation period, and its P&L vs buy-and-hold enters the tuning score
  directly. The baselines get no such tuning. The MLP still loses by about 8.5 pp.
- **Over-interpretation risk** (confirmed): one asset, one 2.7-year bull regime,
  one seed per study. The UI's paired accuracy interval is conditional on that
  seed. SPY results also share the corporate-action blocker (`docs/benchmark.md`);
  the research replay uses a synthetic total-return unit.

### 9. Other findings

- **UI showed the wrong retained checkpoint** (bug, fixed). `ResearchFigures.tsx`
  marked argmin(val loss) as "Retained checkpoint", and the overview `loss` was
  the curve minimum. For the deep MLP, the retained epoch differs in 2 of 3 runs.
  Also, the curve shows the *uncalibrated* model's val loss, while the evaluated
  model's log-loss is 0.708–0.721.
- **The comparison page description was stale** (fixed): it said "NumPy MLP
  (32 hidden units). Lowest validation loss selects the checkpoint."
- **Train/val base-rate shift:** up-share 53.6% → 57.3% (1-day), 62% → 70%
  (20-day). Balanced class weights fight the drift.
- **Fill-time rejections:** benchmark-v2's 0.5% reserve is exceeded by gap-ups,
  so a churny player is penalised more than buy-and-hold (5–9 vs 0 rejections).
- `step2_windows.py` still prints the test-set up-share (an earlier audit noted this).

---

## Phase 3 — Ranked issues

| # | Issue | Type | Expected impact on results |
|---|---|---|---|
| 1 | No exploitable next-day signal in 64 SPY returns over 2017–2019; val set too small to detect realistic edges (MDE ≈ 5.6 pp) | Methodology / data | **Dominant.** No training tweak will beat always-long on this setup. |
| 2 | Objective misalignment: balanced weights + macro-F1 selection/threshold vs long/flat P&L in a drifting market | Methodology | High: explains most of the −8 to −9 pp vs buy-and-hold. |
| 3 | Label ≠ executed return (close-to-close vs next-open; overnight is 109% of drift) | Methodology | High for any churning player; 29% sign mismatch even at perfect skill. |
| 4 | Threshold, epoch and "temperature" tuned on val (with val P&L and a hindsight oracle) and reported on val | Methodology (selection bias) | Medium: inflates P&L by about 10 pp on average (A vs C) and hides the lack of skill. |
| 5 | "Temperature calibration" sharpens logits (T < 1) to a fixed spread | **Bug** | Medium: overconfident probabilities, log-loss worse than constant, misleading reliability and explanations. |
| 6 | Single-seed studies; no CIs across seeds | Methodology | Medium: ±11 pp P&L seed noise makes study-to-study comparisons meaningless. |
| 7 | Retained epoch mislabelled in the UI/overview | **Bug (fixed)** | Low on results, high on interpretability. |
| 8 | Synthetic study too short (130 val) to verify learning | Methodology | Low on results, but it removes the only positive control. |
| 9 | Notebook evaluated sealed-period data | Process | Low technically; important for the integrity claim. |
| 10 | Oracle forward path read 2 gap bars past the train set | **Bug (fixed)** | Negligible (2 labels). |
| 11 | Oracle soft targets/weights | Methodology | ~0 on skill; adds complexity and hindsight coupling. |

---

## Experiment plan (P0, P1 items 6–7, P2 item 9 and the GPU flag applied 2026-11-20, see below)

All runs: inner-val protocol (below), ≥ 5 seeds, report mean ± sd across seeds
**and** a block-bootstrap CI over decisions. The sealed test is untouched until
one pre-registered final run.

**P0 — fix the protocol (no new modelling)**

1. **Train / inner-val / val.** The last 20% of train (about 2012–2016) is
   inner-val, with a purge gap of `lookback + embargo + horizon` examples.
   Epoch selection, threshold, temperature and every hyperparameter use inner-val
   only. Validation is evaluated **once per configuration** and reported as-is.
   (`experiments.py` already implements this as configs E–G.)
2. **Replace the temperature step** with NLL-fitted temperature on inner-val, or
   none. Drop the val threshold search; if a threshold is wanted, choose it on
   inner-val to maximise inner-val *replay-proxy* P&L, not macro-F1.
3. **Remove class balancing and macro-F1 as selection metrics.** Select on
   inner-val log-loss. Report macro-F1, but always next to always-up and the
   permutation null.
4. **Multi-seed by default:** study default seeds `0,1,2,3,4`. The UI shows mean
   ± sd and flags "not distinguishable from always-up" when the paired CI
   includes 0.
5. **Make the positive control real:** extend `synthetic_frame()` to 1995–2019
   (same generator), so the synthetic study has about 5.7k train / 650 val and a
   known ceiling of about 57%. A pipeline change that loses the sine is broken.

**P1 — align the target with what the replay pays**

6. **P&L-aligned label:** `y = sign(log O[d+2] − log O[d+1])` (the return
   actually earned by a decision at close *d*), or a 1-day-hold executed return
   net of 1 bp. Optionally train a **position-utility loss**: maximise
   `p·r_exec − λ·|p_t − p_{t−1}|`, i.e. a differentiable long/flat P&L with a
   turnover penalty, instead of cross-entropy.
7. **Default-long decision rule:** go flat only when `P(down)` exceeds a
   threshold set on inner-val, plus hysteresis (min holding period or a band)
   to cut churn and overnight-gap losses.
8. **Vol-adjusted / multi-horizon labels:** keep the 5/20-day and ±kσ variants
   in the harness. They are not more learnable on SPY (§2), but they need
   checking on the pooled universe.

**P2 — more data, so effects become detectable**

9. **Pool the configured 15-ETF universe** for training and validation (same
   calendar split, per-asset windows, no ticker input). Validation goes from
   627 to about 9k decisions; the bootstrap must resample **calendar blocks
   across assets jointly** (`docs/benchmark.md` already requires this). Report
   per-asset and pooled.
10. **Walk-forward evaluation inside the development period** (e.g. expanding
    train, 1-year folds 2008–2016) gives more independent validation regimes
    than the single 2017–2019 bull market.

**P3 — model and hyperparameter search (only after P0–P2)**

11. Small random search on inner-val, ≥ 3 seeds per point: features {64 raw
    returns, 145 engineered, 81 engineered only}, width {32, 64, 128}, dropout,
    weight decay, lr, epochs ≤ 15; logistic regression always included as the
    reference. Transformer: d {16, 32, 64}, layers {1, 2}, lr {3e-4, 1e-3},
    warmup, last-token or CLS pooling, gradient clipping. The 4 DEC-6 ablations
    run only at the best reference config.
12. **Ablations** (5 seeds each): oracle on/off, class balance on/off, threshold
    tuning on/off, features raw vs engineered, label variants. Report AUC,
    log-loss vs constant, replay P&L vs always-long, turnover.

**GPU server.** The NumPy MLP and logistic models do not need a GPU. The
Transformer search (11) is the only GPU-worthy job. Proposed change: add
`--device {cpu,cuda}` to `transformer_reference.fit_transformer` and expose a
headless entry point `python learning/mlp/run_study.py <study_dir> --device cuda`.
That is not implemented yet because it changes the study worker's interface.
`analysis/experiments.py` already runs headless on a study folder (`--study
<dir> --out <dir> --device cpu`) and is NumPy-only (it refuses `cuda`
explicitly). The M60/M40 cards (Maxwell, sm_52) need a PyTorch build that still
ships sm_52 kernels, so check `torch.cuda.get_arch_list()` on `srv-fisati`
before planning around it.

---

## Changes made in this audit

| Change | File | Test |
|---|---|---|
| Oracle forward paths are clamped at the set's last label bar (no gap-bar reads) | `backend/learning/mlp/deep_mlp.py` (`compute_best_decisions`) | `test_oracle_never_reads_past_the_sets_last_label_bar` |
| Retained epoch flagged in `history` (`selected`) and exported in `curves` | `deep_mlp.py` (`train_deep_mlp`), `export_comparison.py` | `test_retained_epoch_is_flagged_once` |
| Overview `loss` reports the retained epoch's val loss (falls back to min for older runs) | `backend/src/services/study_artifacts.py` | `test_overview_reports_the_retained_epoch_loss` |
| Figure marks the retained epoch, not argmin; optional `selected` field in curve types | `frontend/components/research/ResearchFigures.tsx`, `app/studies/page.tsx`, `lib/research/comparison.ts` | `npm run typecheck`, `npm test` (35 passed) |
| Comparison page text describes the actual model and selection | `frontend/app/comparison/page.tsx` | — |
| Gradient check kept as a regression test | `backend/learning/analysis/gradcheck.py` | `test_deep_mlp_backward_matches_finite_differences` |
| Analysis scripts (read-only) | `backend/learning/analysis/{stats_utils,inventory,gradcheck,experiments}.py`, `results/` | — |

Backend suite: `python -m pytest backend/tests -o addopts=""`: **451 passed,
5 skipped**, with the pre-existing `test_fill_logic.py::test_invariant_holds_under_python_dash_o`
(PYTHONPATH) deselected. `ruff check learning` passes.

Saved studies are **not** regenerated. They keep their original
(unflagged) curves, so the UI falls back to argmin for them. Re-running a study
records the true retained epoch. The two oracle labels affected by the clamp
change the training targets negligibly.

## Applied protocol changes (second pass, 2026-11-20)

Applied after approval. The study documentation describes the protocol for
users: [research-studies.md](research-studies.md#training-protocol).

| Change | Where | Test |
|---|---|---|
| Executed-return label `log(O[d+2]/O[d+1])` for a decision at close *d* (`label_horizon = 2` for example selection only; `split.json` unchanged) | `learning/mlp/protocol.py`, `export_comparison.py` | `test_executed_label_is_next_open_to_following_open` |
| Inner-validation: last 20% of training decisions by calendar date (shared by all assets), whole-window and embargo rules | `protocol.inner_split`, `export_comparison.export` | `test_inner_split_keeps_windows_and_embargo_on_their_side` |
| Epoch = lowest inner-validation cross-entropy; temperature fitted by NLL on inner-validation; val threshold search removed; no class balancing; oracle no longer a training target (still reported) | `deep_mlp.train_deep_mlp`, `fit_temperature` | `test_training_retains_lowest_selection_loss_epoch`, `test_temperature_softens_overconfident_noise` |
| Default-long rule: flat only below a threshold that beats always-long on inner-validation with t ≥ 2; minimum holding period 5 decisions. Classification metrics stay at P(up) ≥ 0.5 | `protocol.HoldPolicy`, `fit_policy` | `test_hold_policy_respects_minimum_holding_period`, `test_policy_stays_long_without_signal_and_goes_flat_with_it` |
| Optional pooled training on every local ETF CSV; `pooled` report with calendar-block bootstrap across assets and per-asset results; replay on the selected ETF | `export_comparison`, `run_study`, `routes_studies` (`pooled`), study form checkbox | `test_pooled_export_end_to_end`, `test_study_defaults_and_pooled_validation` |
| Five seeds by default (API and form) | `routes_studies.StudyConfig`, `frontend/app/studies/page.tsx` | `test_study_defaults_and_pooled_validation` |
| Synthetic series 1995–2019 (about 4.5k train / 650 val) | `export_comparison.synthetic_frame` | end-to-end study below |
| `--device cpu|cuda[:N]` for the Transformer, worker CLI and export CLI; fail-fast check that the card's `sm_XY` is in `torch.cuda.get_arch_list()` | `transformer_reference.resolve_device`, `run_study.py`, `export_comparison.py` | `test_cuda_request_fails_fast_without_a_usable_gpu` (skipped without torch) |
| Case `next` = executed label; `nextClose` keeps the close-to-close return for close-change explanations | `export_comparison`, `routes_studies.close_change` | existing study tests |

The first version of `fit_policy` picked the best threshold by in-sample P&L
and found a spurious threshold in pure noise (caught by a test). The t ≥ 2
requirement is what makes the rule default-long in practice.

### Results with the new protocol (5 seeds, 8 epochs, CPU)

| Study | Val decisions | Accuracy (mean ± sd) | Always-up | Log-loss / constant | Trading rule chosen | Model replay gain / DD | Buy & hold | MA 20/50 |
|---|---|---|---|---|---|---|---|---|
| synthetic `82221745` | 651 | 52.6% ± 0.6 | 49.8% | 0.692 / 0.693 | threshold adopted in 5/5 seeds (t = 3.2–4.3) | −0.7% to +8.3% | −5.2% | −23.2% |
| SPY `285e9073` | 626 | 55.8% ± 1.5 | 58.5% | 0.688 / 0.679 | always long in 5/5 (best t ≤ −0.3) | +38.8% / 19.2% | +38.8% | +30.5% |
| SPY pooled `b18d1473` | 626 (SPY), **9,390 pooled** | 57.7% ± 0.5 (SPY) | 58.5% | 0.686 / 0.679 | always long in 5/5 | +38.8% / 19.2% | +38.8% | +30.5% |

Pooled validation (15 ETFs, 626 dates): accuracy 53.4–54.6% vs always-up
54.9% (paired CI of the difference includes or sits below 0 in every seed),
AUC 0.507–0.514 with 95% CIs from about 0.48 to 0.53. With 15× more decisions
the conclusion of §1 holds: no detectable next-day edge from 64 returns, on any
of the 15 ETFs (per-asset AUC 0.45–0.54).

What changed in practice:

- On SPY the model no longer churns. The rule correctly declines to trade on
  an edge it cannot show on inner-validation, so it matches buy-and-hold
  instead of trailing it by about 9 pp. Orders: 1 instead of about 220; no fill
  rejections.
- Probabilities are honest: fitted temperatures are 1.3–2.2 (softening), log-loss
  is 0.685–0.691 instead of 0.708–0.721, and close to the constant predictor.
- The positive control works end to end: on the synthetic sine the rule adopts
  a threshold and beats buy-and-hold in every seed.
- Nothing in these numbers was tuned on validation, so they can be compared
  between studies.

Not done: the position-utility loss (P1 item 6), walk-forward folds (item 10)
and the hyperparameter search (P3). Transformer runs and the GPU path were not
exercised here because torch is not installed locally.
