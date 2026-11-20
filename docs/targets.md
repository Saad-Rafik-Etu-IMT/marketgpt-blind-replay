# Prediction targets

`backend/src/targets/` defines what a model predicts, independently of the model. Each target turns raw future quantities into labels and provides, for a raw model output `z` of shape `[N, output_dim]`:

| Method | Returns |
| --- | --- |
| `fit(raw, sigma)` | fitted state (scales, class shares, regime cut-offs), training rows only, JSON-ready |
| `encode(raw, sigma, state)` | training labels `y` |
| `loss(z, y, state)` | mean loss and its gradient `d loss / d z` in NumPy |
| `decode(z, state)` | predictions in economic units |
| `evaluate(z, y, raw, sigma, state)` | target-specific metrics |

The gradient lets any model, including the T5 NumPy MLP, back-propagate the loss. A finite-difference test checks it for every target.

## Use with the dataset builder

```python
spec = DatasetSpec(profile="core_features", target="direction_3class_v1", primary_asset="SPY")
ds = build_dataset(frames, spec)
part = ds.parts["validation"]
part.y          # encoded labels (int classes or standardized values)
part.targets    # raw future quantities [N, k]
part.sigma      # realized_vol_20 at the decision bar (point-in-time)
ds.target_state # fitted on fit_on rows (inner_train by default), also in the manifest
```

`label` (T3) and `target` (T4) are exclusive. `target="direction_binary_v1"` gives the same rows and labels as the T3 default (tested). Without a target, a T3 dataset keeps its T3 fingerprint.

## Raw quantities

Example `t` is decided at the close of bar `t`.

| Quantity | Definition | Bars read after `t` |
| --- | --- | ---: |
| Executed return over h sessions | `log(adj_open[t+1+h] / adj_open[t+1])`: filled at the next open, marked h opens later | h + 1 |
| Close-to-close return | `log(adj_close[t+h] / adj_close[t])` | h |
| Future realized volatility | std (ddof=1) of the daily close-to-close log returns of bars `t+1 .. t+h`, per session, not annualized; NaN if prices are constant | h |

## Targets

| Id | Kind | Outputs | Parameterization | Loss |
| --- | --- | --- | --- | --- |
| `direction_binary_v1` | classification | P(down or zero), P(up) of the next executed return | softmax | cross-entropy |
| `direction_binary_close_v1` | classification | same on close-to-close (W1 label) | softmax | cross-entropy |
| `direction_3class_v1` | classification | P(down), P(neutral), P(up) | softmax | cross-entropy |
| `expected_return_h1/h5/h20` | regression | E[executed return over h] | linear, standardized | Huber, delta = 1 training std |
| `expected_return_h1_mse/h5_mse` | regression | same | linear, standardized | squared error / 2 |
| `volatility_h5/h20` | volatility | future realized volatility | standardized log-volatility; `exp` keeps it positive | squared error on the log / 2 |
| `quantiles_h1` | quantile | Q10, Q50, Q90 of the next executed return | `q50 = z0`, `q10 = z0 - softplus(z1)`, `q90 = z0 + softplus(z2)` | mean pinball |
| `multi_horizon_v1` | regression | executed returns over 1, 5 and 20 sessions | linear, standardized per horizon | Huber per horizon, averaged |

Continuous targets are standardized with training mean and std. Models then work in unit scale, and losses are comparable across targets, which matters for multi-task models in T5. `decode` converts back to returns or volatility.

Quantiles cannot cross: the lower and upper gaps are softplus values, so they are always positive. The three quantiles describe an 80% interval and a median, not a full distribution.

No target is a Buy, Sell or Hold action. Turning forecasts into orders stays outside the model.

### Neutral band (three classes)

Decision of 2026-11-20: `neutral` if `|r| <= 0.25 x realized_vol_20[t] x sqrt(h)`. `realized_vol_20[t]` is known at the decision, so nothing is fitted. The band adapts to the asset: IEF moves about three times less than SPY. The builder raises the effective lookback to at least 20 bars for targets that need this sigma.

### Horizons and the sealed test set

A label reading `b` bars needs every gap it crosses to hold `b + embargo` bars.

| Gap with `config/split.json` (SPY) | Bars | 1 session (2 bars) | 5 sessions (6) | 20 sessions (21) |
| --- | ---: | :---: | :---: | :---: |
| train / validation | 62 | yes | yes | yes |
| validation / test | 21 | yes | yes | no (26 needed) |

`expected_return_h20`, `volatility_h20` and `multi_horizon_v1` are therefore **development only**: train and validation work, but the builder refuses them with `include_test=True` and explains why. Scoring them on the test set would need a later `test_start`, which is a team decision.

### Overlapping labels

With h = 5 or 20, neighbouring examples share most of their label. The T3 purge uses the last label bar, so inner folds remain clean. Bootstrap intervals must use blocks of at least `horizon_bars` sessions; `describe()["min_bootstrap_block"]` gives the minimum.

## Metrics

| Kind | Metrics |
| --- | --- |
| Classification | accuracy, balanced accuracy, macro-F1, log loss, multi-class Brier, top-label ECE, confusion matrix; binary also AUC and reliability bins; baselines: training class frequencies (log loss) and majority class (accuracy) |
| Regression | MAE and RMSE in basis points, R² out of sample against the training mean and against zero, Spearman rank correlation, directional accuracy (rows with a non-zero return) |
| Volatility | MAE and RMSE in volatility units, RMSE of the log, QLIKE; the same for the naive forecast `realized_vol_20[t]` and the MAE ratio |
| Quantiles | pinball per quantile, coverage of [Q10, Q90] (80% expected), share below Q10 and above Q90, mean width, crossing rate, coverage by training-fitted volatility tercile |

`macro_f1`, `auc` and `log_loss` match `learning/mlp/protocol.py` (tested). Undefined metrics, such as AUC with one class, are `null`, never 0.

## Reference values on validation

Six ETFs pooled (SPY, IEF, QQQ, XLE, GLD, EEM), profile `returns_only_v1`, validation 2017-04 to 2019-12, `own_partition`. The trivial predictor is the training class frequencies for classification and `z = 0` (the training mean) otherwise. These are the numbers a model must beat.

| Target | Rows | Trivial predictor |
| --- | ---: | --- |
| `direction_binary_v1` | 3,756 | up share 54.5%; accuracy 54.5%; log loss 0.689 |
| `direction_3class_v1` | 3,756 | shares down / neutral / up 35.5% / 21.8% / 42.7%; log loss 1.064 |
| `expected_return_h1` | 3,756 | MAE 63.6 bp |
| `expected_return_h5` | 3,732 | MAE 146.6 bp |
| `expected_return_h20` | 3,642 | MAE 292.7 bp |
| `volatility_h5` | 3,738 | constant MAE 0.0049; naive `realized_vol_20` MAE 0.0032, QLIKE 0.58 |
| `volatility_h20` | 3,648 | constant MAE 0.0047; naive MAE 0.0027, QLIKE 0.49 |
| `quantiles_h1` | 3,756 | training quantiles: coverage 82.4%, width 222 bp |

The naive volatility forecast already beats a constant by about 35% in MAE: volatility persists. A volatility model must beat `realized_vol_20`, not the constant.

## Tests

`backend/tests/test_targets.py`:

- quantities computed by hand;
- for every target, the gradient matches finite differences and invalid output shapes and non-finite outputs are refused;
- probabilities sum to 1, quantiles never cross, decoded volatility is positive;
- the three-class band uses the sigma of the decision bar;
- metric parity with `protocol.py`, plus edge cases;
- `direction_binary_v1` reproduces the T3 label;
- the fitted state ignores validation prices;
- 20-session targets fit wide gaps and are refused across a 21-bar test gap;
- every target builds and evaluates.
