# Dataset construction

`backend/src/datasets/` builds model inputs from the [feature catalogue](features.md). It reproduces the team's study pipeline (`learning/mlp/export_comparison.py` and `protocol.py`) and generalizes it to any feature profile, both input shapes, several scalers and a configurable input-context rule. No existing code calls it yet; the study worker adopts it in tranche T6.

## One call

```python
import pandas as pd
from src.datasets import DatasetSpec, build_dataset, write_manifest

frames = {t: pd.read_csv(f"data/raw/yfinance/{t}.csv", parse_dates=["date"], index_col="date")
          for t in ("SPY", "IEF")}
spec = DatasetSpec(profile="core_features", primary_asset="SPY")
ds = build_dataset(frames, spec)          # split from config/split.json

ds.parts["inner_train"].x                 # scaled inputs [N, F]
ds.parts["inner_train"].y                 # 1 = executed return > 0
ds.parts["validation"].index              # asset, date, bar of every row
write_manifest(ds.manifest, Path("data/datasets/core.json"))
```

## Steps

1. **Cut.** Each frame is cut at `validation_end`. The sealed test period is not read at all unless `include_test=True`; a test checks that changing test-period prices leaves the dataset fingerprint unchanged.
2. **Features.** The point-in-time engine computes the profile's features per asset, in blind mode by default.
3. **Decision bars.** `select_examples` picks bars per partition with one effective lookback (rules below).
4. **Inner validation.** The last 20% of the primary asset's training decisions, by calendar, form the inner-validation set; the cut-off date is shared by every asset. Inner-train examples whose last label bar is within `embargo` bars of the cut-off are purged.
5. **Undefined rows.** A row with an undefined feature (an `allow_nan` feature on a degenerate window) is dropped and counted in the manifest. Nothing is filled.
6. **Scaler.** Fitted on `fit_on` rows only (`inner_train` by default), then applied to every part.
7. **Manifest.** Data hashes, feature versions, label, split, selection rules, counts and scaler, with a dataset fingerprint.

Parts: `train`, `inner_train`, `inner_validation`, `validation`, and `test` only with `include_test=True`. Tuning (epoch, threshold, feature group) uses `inner_train` and `inner_validation`; `validation` is read once.

## Indexing and labels

Example `t` is decided at the close of bar `t`. Inputs use bars `t - lookback .. t`; the label uses bars `t + 1 .. t + horizon`.

| Label | Horizon | Definition |
| --- | ---: | --- |
| `executed_return_v1` (default) | 2 | `log(adj_open[t+2] / adj_open[t+1])`: filled at the next open, marked at the following open, as in the replay |
| `close_to_close_v1` | 1 | `log(adj_close[t+1] / adj_close[t])`: the W1 label |

`y = 1` when the value is strictly positive. The full target registry (three classes, regression, quantiles, volatility, multi-horizon) is tranche T4.

The team's code indexes returns, so its example `t'` is decided at the close of bar `t' + 1`. The parity test converts between the two.

## Input-context rule

Team decision of 2026-11-20 (option 3): the rule is configurable and the README invariant stays the default.

| `context` | Train and validation inputs | Test inputs | Labels |
| --- | --- | --- | --- |
| `own_partition` (default) | whole input window inside the example's own partition | may use earlier bars (warm-up) | inside the own partition |
| `available_history` | may use any earlier bar | may use any earlier bar | inside the own partition |

Both rules keep decision and label bars inside the example's partition and keep the `horizon + embargo` gaps, so no label crosses a boundary. `available_history` uses prices that were public at the decision date; it changes the README invariant and must be chosen explicitly. The manifest records the rule, and the rule changes the fingerprint.

Effect on the six reference ETFs pooled (SPY, IEF, QQQ, XLE, GLD, EEM), all profiles aligned with `min_lookback=271`:

| `context` | train | inner_train | inner_validation | validation |
| --- | ---: | ---: | ---: | ---: |
| `own_partition` | 23,547 | 16,599 | 5,280 | 2,514 |
| `available_history` | 23,547 | 16,599 | 6,906 | 4,140 |

Validation up-share: 53.9% and 54.5% respectively.

## Comparing profiles on the same dates

Profiles have different warm-ups (64 to 271 bars), so each would otherwise keep different decision dates and a different inner cut-off. For an ablation, set `min_lookback` to the largest lookback of the compared profiles. A test checks that two profiles built this way have identical row indexes in every part.

## Input shapes

| `representation` | X | Rows |
| --- | --- | --- |
| `tabular` | `[N, F]` | the feature row of bar `t` |
| `sequence` | `[N, T, F]`, `T = sequence_length` | feature rows `t - T + 1 .. t`; effective lookback grows by `T - 1` |

`F` is the number of feature columns (64 for the baseline), `T` the number of time steps. For a sequence, `X[i, -1]` equals the tabular row of the same date (tested).

## Scalers

| `scaler` | Statistics (training rows only) | Use |
| --- | --- | --- |
| `none` | identity | trees, inspection |
| `global_standard` | one mean and one std over all values | W1 baseline reproduction |
| `standard` (default) | per-column mean and population std; zero std becomes 1 | MLP |
| `robust` | per-column median and IQR (std, then 1 as fallbacks), clipped to ±5 | `deep_mlp` style |

For sequences, statistics are per feature, pooled over examples and time steps. Scalers serialize to JSON; `Scaler.from_dict` refuses a different column order.

## Folds inside training

`folds.walk_forward_cutoffs` and `folds.walk_forward` build expanding-window folds over the training examples only:

- fold `k` evaluates decisions in `[cut[k], cut[k+1])` with their labels inside that block;
- its training side keeps examples whose last label bar is more than `embargo` bars before the block (purge and embargo);
- the evaluated blocks are disjoint and chronological.

The study worker will use them for hyperparameter and feature-group selection (T6, T7).

## Manifest

```json
{
  "schema_version": 1,
  "spec": {...},
  "data": {"SPY": {"fingerprint": "...", "rows": 6788, "first": "1993-01-29", "last": "2019-12-31"}},
  "features": {"requested": [...], "versions": {...}, "columns": [...], "catalogue_fingerprint": "...", "redundancy_warnings": []},
  "label": {...}, "split": {...},
  "selection": {"context": "own_partition", "effective_lookback": 271, "inner_cutoff": "...", ...},
  "parts": {"validation": {"rows": 2514, "up_share": 0.539, "per_asset": {...}, "dropped_undefined": 0, "arrays": "..."}},
  "scaler": {...},
  "dataset_fingerprint": "...", "code_revision": "...", "created_at": "..."
}
```

`dataset_fingerprint` covers everything except the creation time and the code revision, including a hash of every part's arrays. `read_manifest` refuses another schema version or a manifest whose content no longer matches its fingerprint.

## Tests

`backend/tests/test_datasets.py`:

- **Parity**: with `returns_only_v1`, `executed_return_v1` and `global_standard`, the builder reproduces `export_comparison` exactly. Same bars in every part and the same inner cut-off; same windows to 1e-14; same labels to 1e-12 relative; same W1 normalizer.
- **Leakage**: every label stays in its partition under both rules. Changing validation and test prices leaves the scaler and the training arrays unchanged. Changing test prices leaves the fingerprint unchanged.
- **Folds**: inner split and walk-forward purge, embargo, disjointness and chronology.
- **Shapes**: the sequence's last step equals the tabular row. Profiles aligned with `min_lookback` share their dates.
- **Robustness**: undefined rows are dropped and counted. Invalid specs, missing assets, short histories, corrupted or tampered manifests and invalid scalers are refused.

## Limits

- No caching: building the extended profile on six ETFs takes about 1.5 s.
- The ticker is never an input, but each row's asset stays in `index` for per-asset reports.
- Labels are binary up/not-up of a continuous return. Other targets wait for T4.
