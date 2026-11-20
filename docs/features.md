# Feature catalogue and point-in-time engine

`backend/src/features/` holds a versioned feature registry and the engine that computes it. The package is pure: no database, network or file access. Tranche T1 delivers the registry, the engine and Group A (returns). Groups B to J arrive in later tranches and register in `catalog.py`; the engine and the tests do not change.

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
| `units`, `expected_range`, `hard_bounds` | Documentation; `hard_bounds` is enforced |
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

A new feature is covered by the point-in-time tests as soon as it is registered.

## Limits

- The mutation tests cover the synthetic cut points 140, 200 and 318 of a 320-bar series. They detect reads of later rows, not every possible leak (for example, a hard-coded constant fitted on the full history).
- Computing all 248 Group A columns on 8,000 bars takes about 0.04 s on CPU. `vector_at` recomputes from the start; replay-time caching is deferred to T10.
