# Feature catalogue and point-in-time engine

`backend/src/features/` holds a versioned feature registry and the engine that computes it. The package is pure: no database, network or file access. Tranche T1 delivered the registry, the engine and Group A (returns). Tranche T2 adds Groups B to G (70 features, 306 columns), redundancy declarations, versioned profiles and training-only diagnostics. Groups H to J arrive later and register in `catalog.py`; the engine and the tests do not change.

## Timing rule

A value at row `t` is known at the close of bar `t`. It uses bars `t - lookback .. t` only.

`lookback` counts earlier bars. A one-day return needs 1, the 64-return window needs 64 (65 closes). Rows `t < lookback` are warm-up: the engine sets them to NaN. Nothing is filled forward or interpolated.

Past context may come from an earlier partition. A validation decision on its first day uses training-period bars as look-back, because they were public at that date. Fitted state is different: scalers, thresholds and regime cut-offs fit on training rows only (tranche T3).

## Price scales

Raw OHLC drives replay fills; features never use it directly. `canonical.py` moves open, high and low onto the adjusted-close scale with the same-day factor `adjusted_close / close`.

| Field | Definition |
| --- | --- |
| `adj_factor` | `adjusted_close / close` |
| `adj_open`, `adj_high`, `adj_low` | raw value × `adj_factor` |
| `adj_close` | `adjusted_close` |

Intraday ratios do not change. Overnight returns no longer jump on ex-dividend dates. `learning/mlp/protocol.py::adjusted_opens` uses the same formula.

`to_canonical` refuses missing columns, unordered or duplicate dates, non-positive or non-finite prices, negative volume and broken OHLC order. It never repairs them.

## Feature contract

Every `FeatureSpec` declares:

| Field | Purpose |
| --- | --- |
| `id`, `version` | Stable identifier, semantic version. A version bump changes every fingerprint that includes it. |
| `group` | A to J (returns, trend, volatility, price structure, volume, statistics, regime, external, calendar, advanced) |
| `description`, `definition` | Text and formula |
| `required_fields` | Canonical fields the function receives. Other fields are not passed. |
| `dependencies` | Other features, resolved first. A cycle is refused. |
| `lookback` | Warm-up bars. Must be at least each dependency's lookback. |
| `columns` | Output names, in a fixed order |
| `visibility` | What the feature reveals (next section) |
| `missing_policy` | `warmup_nan` (finite after warm-up, else error) or `allow_nan` (undefined rows dropped later) |
| `units`, `expected_range`, `hard_bounds` | Documentation; `hard_bounds` is enforced with a 1e-9 rounding tolerance |
| `redundant_with`, `determined_by`, `redundancy_reason` | Redundancy declarations (section "Redundancy") |
| `frequency`, `availability`, `adjustment` | `1d`, `close of bar t`, adjusted scale by default |

## Visibility and blind mode

The blind replay hides tickers, dates and price levels. Each feature declares what it needs:

| Visibility | Meaning | Blind mode |
| --- | --- | --- |
| `returns_only` | Computable from the adjusted log returns of the F3 observation | Allowed |
| `price_relative` | Other adjusted OHLC fields, scale-free ratios only | Allowed |
| `volume_relative` | Volume relative to its own history | Allowed |
| `dated` | Reveals the calendar (Group I) | Refused |
| `external` | Another series aligned by date (Group H) | Refused until reviewed |

`compute(..., blind=True)` refuses a request if any requested feature or dependency is not blind-safe.

The current F3 `BlindObservation` exposes returns only. `price_relative` and `volume_relative` features need an extended observation before a replay player can use them. That change belongs to the replay-integration tranche (T10).

## Group A: returns

| Id | Lookback | Visibility | Definition |
| --- | ---: | --- | --- |
| `log_return_1d` | 1 | returns_only | `log(adj_close[t] / adj_close[t-1])` |
| `returns_window_16/32/64/128` | N | returns_only | `log_return_1d[t-N+1 .. t]`, oldest first; columns `returns_window_N.lag{N-1}` to `.lag0` |
| `cum_return_5/20/60` | k | returns_only | `log(adj_close[t] / adj_close[t-k])` |
| `mean_return_5/20` | k | returns_only | mean of the last k daily log returns |
| `overnight_return` | 1 | price_relative | `log(adj_open[t] / adj_close[t-1])` |
| `intraday_return` | 0 | price_relative | `log(adj_close[t] / adj_open[t])` |

`overnight_return + intraday_return = log_return_1d`, checked to 1e-12.

## Groups B to G

All are scale-free ratios on the adjusted scale. Volatilities are per-session standard deviations of daily log returns, never annualized.

| Group | Features | Visibility |
| --- | --- | --- |
| B trend | `dist_sma_10/20/50/200`, `dist_ema_12/26`, `macd_line`, `macd_signal`, `macd_hist`, `rsi_14`, `roc_5/10/20/60`, `stoch_k_14`, `stoch_d_3`, `adx_14`, `momentum_accel_10` | returns_only, except stochastic and ADX (price_relative) |
| C volatility | `realized_vol_5/10/20/60`, `vol_ratio_5_20`, `vol_ratio_20_60`, `vol_percentile_252`, `atr_14_pct`, `bb_width_20`, `downside_semivol_20`, `max_drawdown_60`, `drawdown_252` | returns_only, except ATR (price_relative) |
| D price structure | `hl_range_pct`, `clv`, `upper_wick_ratio`, `lower_wick_ratio`, `body_ratio`, `dist_high_20/60`, `dist_low_20/60` | price_relative |
| E volume | `rel_volume_20`, `volume_z_20`, `volume_change_5`, `obv_flow_20`, `price_volume_corr_20`, `volume_trend_20`, `dollar_volume_z_60` | volume_relative |
| F statistics | `autocorr_1_60`, `skew_60`, `kurtosis_60`, `return_to_vol_20`, `up_fraction_20`, `efficiency_ratio_20` | returns_only |
| G regimes | `regime_volatility`, `regime_trend`, `regime_momentum`, `regime_drawdown`, `regime_vol_expansion`, `trend_vol_interaction` | returns_only |

Exact definitions are in each spec's `definition` field (`GET` of the catalogue in T6).

### Finite-memory choices

Classic EMA, Wilder RSI, Wilder ADX and On-Balance Volume accumulate from the first bar of the file. Their value at a given date then depends on the download start, and their lookback is unbounded. T2 uses finite versions:

| Indicator | T2 version | Lookback |
| --- | --- | ---: |
| EMA (span s) | Normalized EMA weights over 4s bars; omits 0.03% of the weight for s = 26 | 4s - 1 |
| RSI 14 | Cutler: simple 14-bar averages of gains and losses; 50 if no change | 14 |
| ADX 14 | Simple 14-bar sums of TR and DM, then a 14-bar mean of DX, scaled to [0, 1] | 27 |
| OBV | `obv_flow_20 = sum_20(sign(r) V) / sum_20(V)` | 20 |

Every rolling statistic uses an explicit window (`groups/_window.py`), never a running sum.

### Degenerate windows

A constant series gives a floating-point standard deviation near 1e-17, not 0. Dividing by it produces large random values. `_window.snap_spread` sets any spread below 1e-10 times the window's magnitude to exactly 0. Ratios over a zero spread are then NaN (undefined), and the feature declares `allow_nan`. Zero-range bars (high = low) get wick and body ratios of 0, CLV 0 and stochastic 0.5, by definition.

### Not included

- `open_close_return` and `overnight_gap`: identical to `intraday_return` and `overnight_return`.
- Downside deviation with target 0: identical to `downside_semivol_20`.
- Raw ATR, absolute volume, dollar volume and classic OBV: they expose levels that identify the asset.
- Rolling beta and correlation to a benchmark: they need a second series (Group H, T8).
- A fitted regime classifier: regimes use fixed thresholds declared in `groups/regime.py`.

## Redundancy

Two declarations, each with a written `redundancy_reason`:

- `redundant_with`: this feature carries the same information as **any one** of the listed features (a copy, a monotone transform, a coarser version or a subset).
- `determined_by`: **all** listed features together give this one through an exact linear identity.

Nonlinear combinations such as `vol_ratio_5_20 = realized_vol_5 / realized_vol_20` are not declared. An MLP approximates a ratio poorly, so the explicit ratio can add usable information.

| Kind | Declarations |
| --- | --- |
| Exact identity | `mean_return_k = cum_return_k / k`; `roc_k = exp(cum_return_k) - 1` (k = 5, 20, 60); `intraday = log_return - overnight`; `macd_hist = line - signal`; `body = 1 - upper wick - lower wick` |
| Subset | `returns_window_16/32/64` are the tails of every larger window |
| Coarse version | each `regime_*` label of its continuous source |
| Empirical | `dist_ema_12` ~ `dist_sma_10` (Spearman 0.961), `dist_ema_26` ~ `dist_sma_20` (0.952) |

Empirical rule: Spearman |rho| >= 0.95 on the training period (bars up to 2016-12-31) of **all six** reference ETFs: SPY, IEF, QQQ, XLE, GLD, EEM. Near misses, kept but watched in ablation:

| Pair | Minimum over the six ETFs | Maximum |
| --- | ---: | ---: |
| `macd_line` / `dist_sma_50` | 0.949 | 0.958 |
| `return_to_vol_20` / `cum_return_20` | 0.948 | 0.977 |
| `trend_vol_interaction` / `dist_sma_50` | 0.946 | 0.976 |
| `rel_volume_20` / `volume_z_20` | 0.935 | 0.987 |
| `atr_14_pct` / `realized_vol_20` | 0.913 | 0.948 |

`compute()` accepts a redundant request and lists the findings in `FeatureMatrix.warnings`. Profiles must contain none.

## Profiles

`profiles.py` defines versioned feature sets. Each available profile starts with `returns_window_64`, is blind-safe and has no declared redundancy (`validate_profiles`).

| Profile | Added to the baseline | Columns | Lookback |
| --- | --- | ---: | ---: |
| `returns_only_v1` | nothing | 64 | 64 |
| `returns_plus_trend` | 8 trend features | 72 | 199 |
| `returns_plus_volatility` | 8 volatility features | 72 | 271 |
| `returns_plus_price_structure` | 8 structure features | 72 | 64 |
| `returns_plus_volume` | 5 volume features | 69 | 64 |
| `returns_plus_statistics` | 5 statistics features | 69 | 64 |
| `returns_plus_market_context` | unavailable until T8 | | |
| `core_features` | 12 cross-group representatives | 76 | 251 |
| `extended_features` | every group profile plus `cum_return_5/20/60` and `overnight_return` | 102 | 271 |

Inside a group, a member of a pair in the 0.90 to 0.95 band is dropped when both measure the same concept (for example `atr_14_pct` versus `realized_vol_20`, `rel_volume_20` versus `volume_z_20`).

Longer lookbacks remove early rows. An ablation must compare profiles on the same decision dates: the T3 dataset builder will use the largest lookback of the compared profiles for all of them.

Measured on SPY training rows with `redundancy_report(threshold=0.9)`:

| Profile | Scalar features | Pairs with \|rho\| >= 0.90 | Components for 90 / 95 / 99% variance |
| --- | ---: | --- | --- |
| `core_features` | 12 | none | 8 / 9 / 11 |
| `extended_features` | 38 | `realized_vol_20`~`downside_semivol_20` 0.935, `obv_flow_20`~`up_fraction_20` 0.907, `cum_return_5`~`dist_sma_10` 0.900 | 17 / 21 / 29 |

These SPY-only pairs fall below 0.85 on at least one of the six ETFs, so they are not declared.

## Diagnostics

```python
from src.data.split import load_split_config
from src.features import compute, get_profile, redundancy_report, training_rows

features = get_profile("extended_features").features
rows = training_rows(compute(frame, features).values, load_split_config())
report = redundancy_report(rows, threshold=0.95)   # Spearman by default
report.pairs, report.clusters, report.components_for_variance
```

`training_rows` keeps only rows whose bar is in the F2 training period, so the diagnostic never reads validation or test data. Correlation measures shared information on that period; it does not prove a feature is useless.

## Baseline

`BASELINE_FEATURES = ("returns_window_64",)`. Parity tests check:

- equality with `build_observation(bars, 64).log_returns` to 1e-15, the same `log(p1 / p0)` formula;
- equality with W1 `make_xy` windows to 1e-14. W1 computes `diff(log(p))`, which differs by floating-point rounding only;
- equality of the first 64 columns of `deep_mlp.extract_blind_features` with the W1 window.

## Usage

```python
from src.features import compute, vector_at, DEFAULT_REGISTRY

matrix = compute(frame, ["returns_window_64", "cum_return_20"])
matrix.values            # DataFrame, one row per bar, columns in request order
matrix.ready             # rows where every column is defined
matrix.fingerprint()     # SHA-256 of values, NaN mask, dates, columns
matrix.catalogue_fingerprint  # SHA-256 of ids, versions, columns, dependencies

row = vector_at(frame, ["returns_window_64"], position=500)  # bars 0..500 only
```

`frame` is one asset's bars: the F1 CSV read with `index_col="date"`, or a frame with a `date` column.

## Tests

| File | Checks |
| --- | --- |
| `test_feature_registry.py` | Invalid specs, duplicates, cycles, lookback order, fingerprints, canonical frame refusals, engine contract (declared fields only, warm-up mask, non-finite, bounds, blind mode) |
| `test_feature_returns.py` | Hand-computed values, ex-date overnight, identities, short history |
| `test_feature_point_in_time.py` | For **every** registered feature: future-bar mutation leaves rows `<= t` unchanged; `vector_at(t)` equals batch row `t`; warm-up equals declared lookback |
| `test_feature_baseline_parity.py` | Baseline equals F3, W1 and deep-MLP inputs |
| `test_feature_groups.py` | Groups B to G: hand-computed values, identities, flat and constant-growth markets, zero-range bars, constant volume |
| `test_feature_profiles.py` | Redundancy declarations and reports, profile rules, bound tolerance, training-only diagnostics, detector sanity check |

`test_feature_point_in_time.py` also checks, for every feature, that dropping the first 30 bars changes no value once the declared lookback is covered. This catches an under-declared lookback and any running statistic.

A new feature is covered by the point-in-time tests as soon as it is registered.

## Limits

- The mutation tests cover the synthetic cut points 140, 200 and 318 of a 320-bar series. They detect reads of later rows, not every possible leak (for example, a hard-coded constant fitted on the full history).
- Computing all 306 columns on SPY (8,507 bars) takes about 0.25 s on CPU. `vector_at` recomputes from the start; replay-time caching is deferred to T10.
- No feature is used by a model yet. Whether any group improves validation metrics is a T7 question; "no group helps" is an acceptable result.
