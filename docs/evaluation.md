# Evaluation

`backend/src/evaluation/` judges a finished [experiment](experiments.md). It answers one question per model: is it better than a simple baseline on the same rows, by more than chance? It also answers it per feature profile against `returns_only_v1`.

It never builds the test part.

## Running it

```
POST /api/experiments/{id}/evaluate?part=inner_validation
GET  /api/experiments/{id}/evaluation/{part}
```

`part` is `inner_validation` (default) or `validation`. Any other value is refused (422 on POST, 404 on GET). The evaluation is written to `evaluation/<part>.json` in the experiment folder.

Every `validation` evaluation is appended to `experiments/validation_reads.json`, the same ledger as the runs. Use `inner_validation` while iterating.

Refusals (409):

- the experiment is not finished, or it is the one currently running;
- a rebuilt dataset no longer matches the fingerprint stored in the run report (data or code changed since training);
- no run completed.

## What is computed

For every combination of feature profile, preset and target:

| Block | Content |
| --- | --- |
| Model | Seed ensemble (probabilities, values or quantiles averaged), its metrics, the loss of each seed and their spread |
| Baselines | Every baseline of the target family, fitted on inner_train only, settings chosen on inner_validation |
| Comparisons | Paired difference model minus baseline, 95% date-block interval, verdict |
| Calibration | Reliability bins and ECE (classification) |
| Regimes | Metrics per volatility tercile and per 60-session trend sign |
| Importance | Loss increase when a feature's columns are shifted, primary target only |
| Trading | Long/flat simulation, `direction_binary_v1` only |
| Overfitting | Train minus selection loss at the selected epoch |

Per feature profile: input drift (PSI and standardized mean difference per column) and label drift from inner_train to the evaluated part.

## Baselines

| Target family | Baselines |
| --- | --- |
| Classification | Training class frequency; logistic regression (MLP without hidden layer, weight decay chosen in 0, 1e-3, 1e-2, 1e-1) |
| Regression | Train mean, zero return, ridge (closed form, alpha chosen on a grid) |
| Volatility | Naive `realized_vol_20` at t, train mean |
| Quantiles | Train empirical quantiles, normal quantiles scaled by `realized_vol_20`, linear quantile regression |

Gradient-boosted trees are listed under `unavailable` in the report. No such library is installed and none is added.

## Paired date-block bootstrap

Rows are paired: the model and the baseline are scored on exactly the same rows. The per-row loss difference is averaged by date, then dates are resampled in circular blocks of consecutive dates.

Block length is `max(10, horizon)`. A 5-day label spans 5 sessions; resampling single rows would treat overlapping labels as independent and give intervals that are too narrow. 1000 resamples by default.

A synthetic check: on 200 white-noise series, the interval covers the true mean of 0 in 88% to 99% of cases.

## Verdicts

| Interval of (model minus baseline) | Verdict |
| --- | --- |
| entirely below 0 | `better` |
| entirely above 0 | `worse` |
| contains 0 | `no measurable difference` |

Lower is better for every loss: log loss, MAE, QLIKE, mean pinball. "No measurable difference" means the evaluated rows cannot separate the two. It does not prove they are equal.

## Ablation

Each feature profile is compared with `returns_only_v1` (or the first profile if absent), same preset and target. The rows must be identical; the experiment's common lookback and horizon ensure it.

| Category | Rule |
| --- | --- |
| `improves` | Verdict `better` and every seed improves |
| `improves but unstable across seeds` | Verdict `better`, some seeds do not |
| `worse than reference` | Verdict `worse` |
| `no measurable improvement` | Otherwise |

Flags are added when seeds disagree on the sign, and when the train/selection gap is more than 0.005 larger than the reference (overfitting sign). The summary names the profiles that improve, with their target and preset, or says that none does.

## Feature importance

Each feature's columns are circularly shifted by half the series within each asset. The column keeps its own autocorrelation; only its alignment with the label is broken. Shuffling rows would destroy the time structure and overstate importance.

This measures predictive association on the evaluated rows. It is not a causal effect, and correlated features share importance.

## Trading simulation

Long when the ensemble gives p_up ≥ 0.5, flat otherwise. Return of a decision: the executed return `log(adj_open[t+2] / adj_open[t+1])`. Each position change costs half the benchmark spread. Assets are equally weighted per date.

Reported: total log return, mean per decision, exposure, turnover, maximum drawdown, for the model, the logistic baseline and always long.

It is vectorized and simplified. It is not the replay engine (T10) and is never used to select a model.

## First result on the 6 ETFs

Experiment "research demo six ETF": SPY, IEF, QQQ, XLE, GLD, EEM, preset `small`, `direction_binary_v1`, 5 seeds. Inner validation, 5280 rows, 880 dates. The validation part was not read.

| Profile | Ensemble log loss | AUC | Seed std | vs class frequency (0.6918) |
| --- | ---: | ---: | ---: | --- |
| `returns_only_v1` | 0.6932 | 0.507 | 0.0019 | no measurable difference [-0.0007, +0.0037] |
| `returns_plus_volatility` | 0.6932 | 0.512 | 0.0026 | no measurable difference [-0.0010, +0.0040] |
| `core_features` | 0.6945 | 0.504 | 0.0012 | no measurable difference [-0.0001, +0.0056] |

Ablation: no feature profile shows a measurable improvement over `returns_only_v1`. For both other profiles, seeds disagree on the sign.

Trading (total equal-weight log return over the period): always long 0.17, model 0.10 to 0.19, logistic baseline 0.23 to 0.25. Not a selection criterion; the next-day direction of these ETFs is not predictable at this scale with these inputs.

Evaluation time: about 5 s.

## Code

| File | Role |
| --- | --- |
| `stats.py` | Per-row losses, block bootstrap, verdicts, paired difference, PSI, drift |
| `baselines.py` | Baselines per target family |
| `evaluate.py` | Ensemble, regimes, importance, trading simulation, ablation, `evaluate_experiment`, `read_evaluation` |

Tests: `backend/tests/test_evaluation.py`. Bootstrap coverage, verdicts, losses equal to the target metrics, baselines with known values, an ablation with known categories (signal, noise, unstable), a trading simulation checked by hand, an end-to-end evaluation on synthetic data, the validation ledger, refusals and routes.
